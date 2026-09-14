# Confluent Interview Questions

---

## Contents

**Coding**
1. [Windowed Key-Value Map (Time-Windowed Averaging & Expiration)](#1-windowed-key-value-map-time-windowed-averaging--expiration)
2. [Implement tail -n on a Streamed File API](#2-implement-tail--n-on-a-streamed-file-api)
3. [LRU Cache with a Concurrency Follow-Up](#3-lru-cache-with-a-concurrency-follow-up)
4. [Match a Variadic Function Signature](#4-match-a-variadic-function-signature)
5. [Detect a Silent Sensor](#5-detect-a-silent-sensor)
6. [Wildcard Pattern Matching with Multiple Stars](#6-wildcard-pattern-matching-with-multiple-stars)
7. [Reach Target Balance (Subset Sum + Reconstruction)](#7-reach-target-balance-subset-sum--reconstruction)
8. [Word & Phrase Search Across Documents](#8-word--phrase-search-across-documents)
9. [Sudoku Validator and Solver as Separate Contracts](#9-sudoku-validator-and-solver-as-separate-contracts)
10. [Defenders Against Every Hostile Monster (DAG Reachability)](#10-defenders-against-every-hostile-monster-dag-reachability)

**Concurrency / Low-Level Design**
11. [Thread-Safe Delayed Task Runner](#11-thread-safe-delayed-task-runner)
12. [Random-Access FIFO Queue](#12-random-access-fifo-queue)

**System Design**
13. [Leader-Based Distributed Key-Value Store](#13-system-design--leader-based-distributed-key-value-store)
14. [Kubernetes-Managed Kafka Service](#14-system-design--kubernetes-managed-kafka-service)
15. [Idempotent URL Shortening Service](#15-system-design--idempotent-url-shortening-service)
16. [Centralized Log Ingestion and Search Platform](#16-system-design--centralized-log-ingestion-and-search-platform)

**Behavioral**
17. [Behavioral Themes](#17-behavioral-themes)

---

## 1. Windowed Key-Value Map (Time-Windowed Averaging & Expiration)

**Problem Statement:**
Confluent's recurring onsite "build a HashMap-flavored data structure" prompt. Design a `WindowedMap`
where every value written for a key automatically expires `window` time units after it was written —
there is no explicit TTL parameter per call, just one fixed window for the whole map (candidates report
`window = 5 minutes` as the example value). Support:

- `put(key, value, ts)` — record `value` for `key` as of time `ts`.
- `get(key, ts)` — return the most recently written, still-unexpired value for `key` as of `ts` (or
  `None`).
- `delete(key, ts)` — remove the most recent still-live entry for `key` as of `ts`.
- `average(key, ts)` — return the average of all still-unexpired values written for `key` as of `ts`.

Timestamps arrive non-decreasing per key but the interviewer explicitly asks what breaks if they don't.
A same-thread follow-up in every report: make it safe under concurrent `put`/`get` from multiple threads.

**Example:**
```
window = 5
put("cpu", 10, ts=0)
put("cpu", 20, ts=5)
put("cpu", 30, ts=8)   # ts=0 entry is now expired (8 - 0 >= 5); ts=5 entry is still live (8 - 5 < 5)
get("cpu", ts=8)       # -> 30
average("cpu", ts=8)   # -> average of the still-live entries: (20 + 30) / 2 = 25.0
```

**Test Cases:**

| Scenario | Expectation |
|---|---|
| `get` on a key with only expired entries | `None` |
| `put` twice at the same `ts` | Both entries live; `get` returns the later of the two |
| `delete` when nothing is live | No-op, doesn't raise |
| `average` with no live entries | `0.0` (interviewer accepts either `0.0` or `None` — state the choice) |
| Two threads `put` to the same key concurrently | No lost updates, no corrupted deque |

**Key Insights:**
1. Evaluate expiration **lazily**, against the timestamp of the current call, rather than running a
   background sweep — there's no wall clock to trust in a deterministic test harness, and eviction this
   way is still amortized O(1): each entry is popped at most once, ever.
2. Store each key's history as a `deque` of `(ts, value)` ordered by insertion time, so eviction is a
   pop from the front and `average`/`get` never have to scan expired entries.
3. The concurrency follow-up is the real signal: a single `RLock` around the whole map is correct but
   serializes unrelated keys. The natural fix is a lock *per key* (e.g., a `defaultdict` of locks), which
   the interviewer wants you to name the tradeoff for — more locks means more memory and a subtle new
   bug class (locks for deleted keys never get cleaned up) in exchange for eliminating cross-key
   contention.

**Python Solution:**
```python
import threading
from collections import defaultdict, deque


class WindowedMap:
    """
    Time:  O(1) amortized for put/get/delete/average (each entry evicted once, ever).
    Space: O(n) for n live entries across all keys.
    """

    def __init__(self, window: float):
        self.window = window
        self._entries: dict[str, deque[tuple[float, float]]] = defaultdict(deque)
        self._locks: dict[str, threading.Lock] = defaultdict(threading.Lock)

    def _evict(self, key: str, ts: float) -> deque:
        dq = self._entries[key]
        while dq and ts - dq[0][0] >= self.window:
            dq.popleft()
        return dq

    def put(self, key: str, value: float, ts: float) -> None:
        with self._locks[key]:
            self._evict(key, ts).append((ts, value))

    def get(self, key: str, ts: float):
        with self._locks[key]:
            dq = self._evict(key, ts)
            return dq[-1][1] if dq else None

    def delete(self, key: str, ts: float) -> None:
        with self._locks[key]:
            dq = self._evict(key, ts)
            if dq:
                dq.pop()

    def average(self, key: str, ts: float) -> float:
        with self._locks[key]:
            dq = self._evict(key, ts)
            return sum(v for _, v in dq) / len(dq) if dq else 0.0
```

---

## 2. Implement tail -n on a Streamed File API

**Problem Statement:**
Five separate onsite reports (spring 2026) converge on the same task: you're given a `FileAPI` for a
file that may be arbitrarily large —

```python
class FileAPI:
    def size(self) -> int: ...                 # total bytes
    def read(self, offset: int, length: int) -> bytes: ...  # bytes[offset, offset+length)
```

— and must stream the **last `n` lines** to stdout. You are explicitly told the file may be too large to
load into memory, so reading forward from byte 0 and keeping only the tail is disallowed by the
interviewer's framing; the discussion centers on reading backward from EOF in bounded chunks.

**Test Cases:**

| Scenario | Expectation |
|---|---|
| `n` ≥ total lines in file | Every line, in order |
| File doesn't end in a trailing newline | Last line still printed, no spurious empty line |
| File ends in a trailing newline | No spurious empty line from the final `\n` |
| `n = 0` | Nothing printed |
| A single line longer than one chunk | Still assembled correctly (multi-chunk backward read) |

**Key Insights:**
1. Read fixed-size chunks working backward from `size()`, prepending each chunk to a buffer, and stop as
   soon as the buffer contains `n` newlines (or you hit offset 0). Cost is bounded by *output size*, not
   file size — the whole point of the exercise.
2. Off-by-one the interviewer watches for: a trailing `\n` in the file produces an empty string as the
   last element of `buffer.split(b"\n")` — drop it before taking the last `n` lines, or you'll under-count
   by one real line.
3. Stated in every report as the deciding factor between candidates: explicitly reasoning about *why*
   forward-from-0 is wrong (I/O proportional to file size regardless of `n`) before writing code.

**Python Solution:**
```python
def tail(file: "FileAPI", n: int, chunk_size: int = 4096) -> None:
    """
    Time:  O(k) where k = bytes actually read backward from EOF — bounded by
           the distance to the n-th newline from the end, not file size.
    Space: O(k) for the trailing buffer.
    """
    if n <= 0:
        return

    pos = file.size()
    newline_count = 0
    buffer = b""

    while pos > 0 and newline_count <= n:
        read_size = min(chunk_size, pos)
        pos -= read_size
        chunk = file.read(pos, read_size)
        buffer = chunk + buffer
        newline_count += chunk.count(b"\n")

    lines = buffer.split(b"\n")
    if lines and lines[-1] == b"":
        lines.pop()  # trailing '\n' in the file, not a real blank last line
    for line in lines[-n:]:
        print(line.decode())
```

---

## 3. LRU Cache with a Concurrency Follow-Up

**Problem Statement:**
The baseline ask is the standard LRU cache: `get(key) -> value` and `put(key, value)` in O(1), evicting
the least-recently-used entry at capacity. What distinguishes Confluent's version, per every report, is
the immediate follow-up: **make it safe for concurrent access**, and explain why a plain
`ReadWriteLock` doesn't work — `get()` is logically a read but it mutates the recency list, so two
concurrent "readers" can corrupt the linked list against each other.

**Test Cases:**

| Scenario | Expectation |
|---|---|
| `put` beyond capacity | Evicts the true least-recently-used key, not insertion order |
| `get` on an existing key | Value returned *and* key becomes most-recently-used |
| `put` on an existing key | Value updated *and* key becomes most-recently-used |
| Two threads call `get` on different keys concurrently | No corruption of the linked list |
| `get` on a missing key | Returns `None`/sentinel, doesn't affect recency of other keys |

**Key Insights:**
1. Hash map (`key -> node`) + doubly linked list (MRU/LRU sentinels) gives true O(1) get/put; don't reach
   for `OrderedDict.move_to_end` as the *answer* to the concurrency follow-up — the interviewer wants you
   to reason about the underlying data structure.
2. The concurrency trap: `get()` reorders the list, so it needs the **same exclusive lock** as `put()`,
   not a shared read lock — a `ReadWriteLock` gives you nothing here because there's no true read-only
   path.
3. If contention on one global lock is a bottleneck under high QPS, the honest answer is **sharding**:
   split into N independently-locked LRU sub-caches keyed by `hash(key) % N`. This trades strict global
   recency ordering for throughput — each shard's own LRU order is exact, but "least recently used
   globally" becomes approximate. Name this tradeoff explicitly; it's the signal the follow-up is
   fishing for.

**Python Solution:**
```python
import threading


class _Node:
    __slots__ = ("key", "value", "prev", "next")

    def __init__(self, key=None, value=None):
        self.key, self.value = key, value
        self.prev = self.next = None


class LRUCache:
    """
    Time:  O(1) get/put.
    Space: O(capacity).
    """

    def __init__(self, capacity: int):
        self.capacity = capacity
        self._map: dict = {}
        self._head, self._tail = _Node(), _Node()  # MRU / LRU sentinels
        self._head.next, self._tail.prev = self._tail, self._head
        self._lock = threading.RLock()  # get() mutates, so no read/write split

    def _unlink(self, node: _Node) -> None:
        node.prev.next, node.next.prev = node.next, node.prev

    def _push_front(self, node: _Node) -> None:
        node.next, node.prev = self._head.next, self._head
        self._head.next.prev = node
        self._head.next = node

    def get(self, key):
        with self._lock:
            node = self._map.get(key)
            if node is None:
                return None
            self._unlink(node)
            self._push_front(node)
            return node.value

    def put(self, key, value) -> None:
        with self._lock:
            if key in self._map:
                node = self._map[key]
                node.value = value
                self._unlink(node)
                self._push_front(node)
                return
            if len(self._map) >= self.capacity:
                lru = self._tail.prev
                self._unlink(lru)
                del self._map[lru.key]
            node = _Node(key, value)
            self._map[key] = node
            self._push_front(node)
```

---

## 4. Match a Variadic Function Signature

**Problem Statement:**
A phone-screen prompt reproduced near-verbatim across independent reports. You're building the matcher
behind a `FunctionLibrary`: given a registry of function signatures, find every registered function whose
signature matches a call's argument types.

**Inputs:**
- `function_names: list[str]` — unique function identifiers.
- `function_argument_types: list[list[str]]` — parallel array, each entry the declared parameter types.
- `is_variadic: list[bool]` — whether the *last* declared type repeats.
- `call_argument_types: list[str]` — the actual argument types of one call site.

**Matching rules:**
- **Non-variadic** function: matches iff its declared types are exactly equal, position by position, to
  the call's argument types (same length required).
- **Variadic** function with declared types `[T0, ..., Tk]`: the prefix `[T0, ..., Tk-1]` must match the
  call positionally, and `Tk` (the final declared type) must then match **one or more** of the remaining
  call arguments — i.e. `[String, Integer]` (variadic on `Integer`) matches `[String, Integer]` and
  `[String, Integer, Integer]`, but not `[String]` (zero repeats isn't "one or more").

Return matching function names **in their original `functionNames` order**.

**Constraints:** up to 20,000 functions; total declared tokens across all signatures ≤ 200,000; call
argument list ≤ 200,000 — the size hints (200K tokens, up to 20K functions × a long call) are the tell
that a straightforward "rescan the whole call for every variadic function" solution is a trap.

**Test Cases:**

| Call | Registered (name: types, variadic?) | Result |
|---|---|---|
| `[String, Integer]` | `A: [String, Integer], False` | `["A"]` |
| `[String, Integer, Integer]` | `B: [String, Integer], True` | `["B"]` |
| `[String]` | `B: [String, Integer], True` | `[]` (zero repeats doesn't satisfy "one or more") |
| `[Int, Bool, Bool, Bool]` | `C: [Int, Bool], True` | `["C"]` |
| `[Int, Bool, Str]` | `C: [Int, Bool], True` | `[]` (tail isn't uniformly `Bool`) |

**Key Insights:**
1. Naively checking, for every variadic function, whether the whole call tail equals the repeated type is
   O(F × C) — with F=20,000 and C up to 200,000 that's 4 billion comparisons, well past budget.
2. Precompute **once** over the call: the start index of its maximal constant suffix run (the longest
   run of identical types ending at the last argument). A variadic function's tail condition then reduces
   to two O(1) checks — does its repeated type equal the call's last argument type, and does its prefix
   length fall at or before that run's start index — instead of rescanning the tail per function.
3. Prefix-equality cost is still paid per function, but it's bounded by that function's declared length,
   so summed across all functions it's bounded by total declared tokens (≤200,000), which is in budget.

**Python Solution:**
```python
def match_signatures(function_names, function_argument_types, is_variadic, call_argument_types):
    """
    Time:  O(T + C) — T = total tokens across all declared signatures,
           C = len(call_argument_types). One pass over the call precomputes
           the constant-suffix run once, instead of rescanning it per
           variadic function.
    Space: O(T) for the output in the worst case; O(1) extra.
    """
    call_len = len(call_argument_types)

    run_start = call_len
    if call_len > 0:
        run_start = call_len - 1
        last = call_argument_types[-1]
        while run_start > 0 and call_argument_types[run_start - 1] == last:
            run_start -= 1

    matches = []
    for name, arg_types, variadic in zip(function_names, function_argument_types, is_variadic):
        m = len(arg_types)
        if not variadic:
            if arg_types == call_argument_types:
                matches.append(name)
            continue

        if call_len < m:
            continue
        if call_argument_types[: m - 1] != arg_types[:-1]:
            continue
        if call_argument_types[-1] != arg_types[-1]:
            continue
        if m - 1 < run_start:
            continue
        matches.append(name)

    return matches
```

---

## 5. Detect a Silent Sensor

**Problem Statement:**
Two independent legacy reports describe the same health-check family: sensors emit `(sensor_id,
timestamp)` pings on an expected `interval`. Build a monitor that can answer, for any point in time,
whether a sensor "was alive" and, given a current time, which sensors have gone **silent** — defined as
three or more consecutive missed expected pings leading up to now.

**API:**
- `record(sensor_id, timestamp)` — log a ping.
- `was_alive(sensor_id, timestamp) -> bool` — did the sensor ping within one `interval` before or at
  `timestamp`?
- `silent_sensors(now) -> list[sensor_id]` — every known sensor with no ping in `(now - 3*interval, now]`.

**Test Cases:**

| Scenario | Expectation |
|---|---|
| Sensor pinged 1 interval ago, `was_alive(now)` | `True` |
| Sensor pinged 4 intervals ago, `was_alive(now)` | `False` |
| Sensor with last ping exactly `3*interval` before `now` | Silent (boundary is inclusive of the gap) |
| Sensor with pings scattered but one inside the last 3 intervals | Not silent |
| Unknown `sensor_id` | Not silent / not alive, doesn't raise |

**Key Insights:**
1. Store each sensor's pings as a **sorted list**; `bisect` gives O(log k) point and range queries
   instead of scanning — the fastprep summary of the accepted approach is literally "hashmap plus binary
   search," which is the whole solution shape.
2. `was_alive` is a nearest-ping-at-or-before-`timestamp` query — `bisect_right` then check the gap to the
   preceding ping, not an exact-match lookup.
3. `silent_sensors` doesn't need to re-derive "3 consecutive missed slots" from raw gaps between
   individual pings — it collapses to one range check per sensor: is there *any* ping in the last three
   intervals? No ping in that window is equivalent to 3+ consecutive misses by definition of the interval.

**Python Solution:**
```python
import bisect
from collections import defaultdict


class SensorMonitor:
    """
    Time:  O(log k) per record/was_alive query for a sensor with k pings;
           O(n log k) for silent_sensors across n sensors.
    Space: O(total pings).
    """

    def __init__(self, interval: int):
        self.interval = interval
        self._pings: dict[str, list[int]] = defaultdict(list)

    def record(self, sensor_id: str, timestamp: int) -> None:
        bisect.insort(self._pings[sensor_id], timestamp)

    def was_alive(self, sensor_id: str, timestamp: int) -> bool:
        pings = self._pings.get(sensor_id)
        if not pings:
            return False
        i = bisect.bisect_right(pings, timestamp)
        return i > 0 and pings[i - 1] > timestamp - self.interval

    def silent_sensors(self, now: int) -> list[str]:
        cutoff = now - 3 * self.interval
        silent = []
        for sensor_id, pings in self._pings.items():
            i = bisect.bisect_right(pings, now)
            if i == 0 or pings[i - 1] <= cutoff:
                silent.append(sensor_id)
        return sorted(silent)
```

---

## 6. Wildcard Pattern Matching with Multiple Stars

**Problem Statement:**
A phone-screen prompt that's explicitly staged: it starts with a pattern containing **exactly one** `*`
(plus literal characters and `?`), then the interviewer generalizes to **multiple** `*`s mid-interview.
Some reports note candidates who used a prefix/suffix shortcut for the single-star version got stuck when
the generalization landed, because that shortcut doesn't extend — full wildcard matching (`?` = any one
character, `*` = any sequence, including empty) needs real DP or two-pointer backtracking.

**Test Cases:**

| `s` | `p` | Match? |
|---|---|---|
| `"aa"` | `"a"` | `False` |
| `"aa"` | `"*"` | `True` |
| `"cb"` | `"?a"` | `False` |
| `"adceb"` | `"*a*b*"` | `True` |
| `"acdcb"` | `"a*c?b"` | `False` |
| `""` | `"***"` | `True` (zero or more, stacked stars still match empty) |

**Key Insights:**
1. **Stage 1 shortcut** (exactly one `*`): split `p` on `*` into `prefix, suffix`; match iff
   `s.startswith(prefix) and s.endswith(suffix) and len(s) >= len(prefix) + len(suffix)` — O(len(s)),
   and worth stating explicitly as the "obviously correct for this restricted case" answer before the
   generalization lands.
2. **Stage 2** (multiple `*`, plus `?`): that shortcut has no natural extension — you need `dp[i][j]` =
   does `s[:i]` match `p[:j]`. `*` either matches zero characters (`dp[i][j-1]`) or consumes one more
   character of `s` while staying on the same `*` (`dp[i-1][j]`).
3. Rolling the DP to a single array (`prev`/`curr`) drops space from O(len(s)·len(p)) to O(len(p)),
   worth mentioning unprompted since Confluent's phone screens reward reasoning about tightening a
   working solution over silently moving on.

**Python Solution:**
```python
def is_match(s: str, p: str) -> bool:
    """
    Time:  O(len(s) * len(p)).
    Space: O(len(p)) via a rolling DP row.
    """
    m, n = len(s), len(p)
    prev = [False] * (n + 1)
    prev[0] = True
    for j in range(1, n + 1):
        prev[j] = prev[j - 1] and p[j - 1] == "*"

    for i in range(1, m + 1):
        curr = [False] * (n + 1)
        for j in range(1, n + 1):
            if p[j - 1] == "*":
                curr[j] = curr[j - 1] or prev[j]
            elif p[j - 1] == "?" or p[j - 1] == s[i - 1]:
                curr[j] = prev[j - 1]
        prev = curr
    return prev[n]
```

---

## 7. Reach Target Balance (Subset Sum + Reconstruction)

**Problem Statement:**
A HackerRank OA question (also reused onsite as "Positive Subset Sum" per one report, itself flagged as
recycled from a Google bank): given a list of transaction amounts on an account, determine whether some
subset sums exactly to a target balance. Return `1` if such a subset exists, `0` otherwise. The onsite
version adds a follow-up: **reconstruct** one such subset, not just decide feasibility.

**Test Cases:**

| `transactions` | `target` | Result |
|---|---|---|
| `[3, 7, 5, 2]` | `10` | `1` (e.g. `3+7` or `5+3+2`) |
| `[1, 1, 1]` | `5` | `0` |
| `[]` | `0` | `1` (empty subset) |
| `[4]` | `4` | `1` |
| `[4]` | `3` | `0` |

**Key Insights:**
1. Standard 0/1 knapsack reachability: `reachable[t]` becomes `True` once some prefix of transactions can
   sum to `t`; iterate the target dimension **downward** per item so each transaction is used at most
   once.
2. Reconstruction needs one extra piece of state beyond the boolean table: for each newly-reachable sum
   `t`, record *which item first made it reachable*. Because the inner loop still walks `t` downward
   within a single item's pass, `dp[t - amount]` read during that pass always reflects state from
   **earlier** items only — never double-counts the current item — so walking `dp[target] -> dp[target -
   amount] -> ...` back to `0` recovers a valid witnessing subset.
3. Both the decision and reconstruction are the same O(n · target) DP; reconstruction just adds an
   O(target) parent array and an O(n)-length backtrace at the end, not a new algorithm.

**Python Solution:**
```python
def reach_target_balance(transactions: list[int], target: int) -> int:
    """Time: O(n * target). Space: O(target)."""
    if target < 0:
        return 0
    reachable = [False] * (target + 1)
    reachable[0] = True
    for amount in transactions:
        for t in range(target, amount - 1, -1):
            if reachable[t - amount]:
                reachable[t] = True
    return 1 if reachable[target] else 0


def reach_target_balance_subset(transactions: list[int], target: int) -> list[int] | None:
    """Time: O(n * target). Space: O(target). Returns a witnessing subset, or None."""
    if target < 0:
        return None
    made_reachable_by = [None] * (target + 1)  # index of the item that first reached t
    made_reachable_by[0] = -1
    for i, amount in enumerate(transactions):
        for t in range(target, amount - 1, -1):
            if made_reachable_by[t] is None and made_reachable_by[t - amount] is not None:
                made_reachable_by[t] = i

    if made_reachable_by[target] is None:
        return None

    subset, t = [], target
    while t != 0:
        i = made_reachable_by[t]
        subset.append(transactions[i])
        t -= transactions[i]
    return subset
```

---

## 8. Word & Phrase Search Across Documents

**Problem Statement:**
Staged across two related onsite reports. First: given a collection of documents (each a list of words),
find every document containing a given word. Then extended: find every document containing a given
**phrase** — an ordered sequence of words appearing **consecutively**. Named follow-ups: deduplicate
repeated matches, and discuss "layered compression" of the index (i.e. delta-encoding position lists).

**Example:**
```
docs = {
  "d1": ["the", "quick", "brown", "fox"],
  "d2": ["the", "slow", "brown", "dog"],
}
search_word("brown")           # -> ["d1", "d2"]
search_phrase(["the", "quick"])  # -> ["d1"]
search_phrase(["brown", "fox"])  # -> ["d1"]
search_phrase(["brown", "dog"])  # -> ["d2"]
```

**Test Cases:**

| Scenario | Expectation |
|---|---|
| Word absent from every document | `[]` |
| Phrase whose words all appear, but never consecutively in that order | `[]` |
| Phrase of length 1 | Same as `search_word` |
| Same phrase appears twice in one document | Document listed once, not twice |
| Empty phrase | `[]` |

**Key Insights:**
1. Build a **positional inverted index**: `word -> doc_id -> sorted positions`. Word search is a direct
   lookup; the interesting part is phrase search.
2. Phrase search reduces to: intersect the candidate document sets for all phrase words (cheap early
   exit if any word is document-less), then for each candidate document check whether some starting
   position `p` of the first word has `p+1, p+2, ...` present in the corresponding position sets for the
   later words — an O(1) set-membership check per offset rather than a substring scan.
3. "Layered compression": once positions are sorted per (word, doc), storing **deltas** between
   consecutive positions instead of raw offsets shrinks the index substantially for common words
   (small deltas encode in fewer bits) — a standard inverted-index technique, worth naming even if not
   implementing it live.

**Python Solution:**
```python
from collections import defaultdict


class DocumentIndex:
    """
    Time:  O(sum of document lengths) to build.
           O(occurrences of the phrase's first word) per phrase query.
    Space: O(sum of document lengths) for the positional index.
    """

    def __init__(self, documents: dict[str, list[str]]):
        self._index: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
        for doc_id, words in documents.items():
            for pos, word in enumerate(words):
                self._index[word][doc_id].append(pos)

    def search_word(self, word: str) -> list[str]:
        return sorted(self._index.get(word, {}).keys())

    def search_phrase(self, phrase: list[str]) -> list[str]:
        if not phrase:
            return []

        candidates = None
        for word in phrase:
            docs = set(self._index.get(word, {}).keys())
            candidates = docs if candidates is None else candidates & docs
            if not candidates:
                return []

        matches = []
        for doc_id in candidates:
            starts = self._index[phrase[0]][doc_id]
            if any(
                all(start + offset in self._index[phrase[offset]][doc_id] for offset in range(1, len(phrase)))
                for start in starts
            ):
                matches.append(doc_id)
        return sorted(matches)
```

---

## 9. Sudoku Validator and Solver as Separate Contracts

**Problem Statement:**
Two related onsite prompts, deliberately kept as two independent APIs by the interviewer: `is_valid_board`
checks whether a partially-filled 9×9 board has no duplicate digit 1–9 in any row, column, or 3×3 box
(empty cells are `0`), and `solve_sudoku` fills in the remaining cells to produce a fully valid board. The
explicit constraint across reports: **the solver must call the validator**, not re-implement its own
incremental legality tracking — the interview is testing contract separation and composition as much as
backtracking itself.

**Test Cases:**

| Scenario | Expectation |
|---|---|
| Board with a duplicate `5` in one row | `is_valid_board` → `False` |
| Board with duplicates only across different boxes (no row/col conflict) | `False` |
| Fully empty board | `is_valid_board` → `True`; `solve_sudoku` → any valid completion |
| Board with exactly one empty cell and one legal digit | Solver fills it, terminates immediately |
| Unsolvable board (no legal digit fits some cell under any assignment) | Solver returns `False`, leaves board unmodified from that branch's perspective (backtracked) |

**Key Insights:**
1. Keeping the two APIs separate is a deliberate tradeoff, not an oversight: `solve_sudoku` re-validates
   the *entire* board (O(81)) on every candidate placement instead of maintaining its own O(1)
   incremental row/col/box sets — much slower per check, but the solver's correctness is now trivially
   implied by the validator's correctness, and the two can be tested, reasoned about, and changed
   independently.
2. Name the alternative unprompted: an incremental-tracking solver (row/col/box `set`s updated in O(1) on
   each placement) is the standard "fast" backtracking Sudoku solver, but it duplicates the validator's
   logic inside the solver — exactly the coupling the interviewer's contract is designed to avoid.
3. Backtracking still needs the usual shape: find the first empty cell, try digits 1–9, recurse, undo on
   failure — the only twist is that legality is delegated.

**Python Solution:**
```python
def is_valid_board(board: list[list[int]]) -> bool:
    """Time: O(81) — fixed 9x9 board. Space: O(1)."""
    rows, cols, boxes = [set() for _ in range(9)], [set() for _ in range(9)], [set() for _ in range(9)]
    for r in range(9):
        for c in range(9):
            v = board[r][c]
            if v == 0:
                continue
            b = (r // 3) * 3 + c // 3
            if v in rows[r] or v in cols[c] or v in boxes[b]:
                return False
            rows[r].add(v)
            cols[c].add(v)
            boxes[b].add(v)
    return True


def solve_sudoku(board: list[list[int]]) -> bool:
    """
    Backtracking solver that delegates every legality check to
    is_valid_board (the interviewer's explicit contract), trading O(1)
    incremental checks for O(81) full-board re-validation per candidate
    placement.
    Time:  O(9^m) branching over m empty cells, each check O(81).
    Space: O(m) recursion depth.
    """
    for r in range(9):
        for c in range(9):
            if board[r][c] != 0:
                continue
            for v in range(1, 10):
                board[r][c] = v
                if is_valid_board(board) and solve_sudoku(board):
                    return True
                board[r][c] = 0
            return False
    return True
```

---

## 10. Defenders Against Every Hostile Monster (DAG Reachability)

**Problem Statement:**
A directed graph of monsters and a "defeats" relation: edge `u -> v` means `u` can defeat `v` (defeat is
**not** assumed transitive by the raw edges — reachability along chains of edges is what counts). Given a
set of `hostile` monsters, return every monster from a `candidates` set that can defeat **all** of them,
directly or via a chain. A related phone-screen prompt ("Least-Strong Common Defeater") adds a per-monster
`strength` value and asks for the single qualifying monster with **minimum** strength.

**Test Cases:**

| Graph edges | `hostile` | `candidates` | Result |
|---|---|---|---|
| `A→B, B→C` | `{C}` | `{A, B}` | `[A, B]` (both reach `C`, `A` transitively) |
| `A→B, B→C` | `{B, C}` | `{A, B}` | `[A]` (`B` doesn't defeat itself) |
| `A→B` | `{C}` | `{A}` | `[]` (`C` unreachable) |
| same as above, with `strength = {A: 5, B: 2}` | `{C}` (via `A→B→C` and separately `B→C`) | least-strong among qualifiers | `min` by strength, tie-break by id |

**Key Insights:**
1. This is single-source reachability run from **every** node — BFS/DFS from each candidate, checking
   whether the reached set is a superset of `hostile`. Precompute all nodes' reachable sets once with BFS
   rather than a fresh traversal per query if the graph is queried repeatedly (a signal to ask "will there
   be multiple queries against the same graph?" before committing to a per-query DFS).
2. "Defeats all hostile monsters" is exactly `hostile ⊆ reach[monster]` — a set-subset check, not a
   per-hostile-monster loop with early exit dressed up as something fancier.
3. For dense graphs where all-pairs BFS (`O(V·(V+E))`) gets expensive, bitset reachability (propagate
   reachability as integer bitmasks in reverse topological order, combining a node's bitmask with the OR
   of its successors') compresses the same computation to `O(V·(V+E)/64)` — worth naming as the next lever
   if asked to scale up, without implementing it unless pushed.

**Python Solution:**
```python
from collections import defaultdict, deque


def _reachability(edges: list[tuple[str, str]]) -> dict[str, set[str]]:
    """Time: O(V * (V + E)). Space: O(V^2) worst case."""
    graph = defaultdict(list)
    nodes = set()
    for u, v in edges:
        graph[u].append(v)
        nodes.update((u, v))

    reach: dict[str, set[str]] = {}
    for start in nodes:
        seen = {start}
        q = deque([start])
        while q:
            u = q.popleft()
            for v in graph[u]:
                if v not in seen:
                    seen.add(v)
                    q.append(v)
        reach[start] = seen - {start}
    return reach


def defenders_against_all(edges, hostile: set[str], candidates: set[str]) -> list[str]:
    reach = _reachability(edges)
    return sorted(m for m in candidates if hostile <= reach.get(m, set()))


def least_strong_common_defeater(edges, hostile: set[str], strength: dict[str, int]):
    reach = _reachability(edges)
    qualifying = [m for m in strength if hostile <= reach.get(m, set())]
    return min(qualifying, key=lambda m: (strength[m], m)) if qualifying else None
```

---

## 11. Thread-Safe Delayed Task Runner

**Problem Statement:**
Design a `schedule(task, delay)` API that runs `task` after `delay` elapses. Staged across the interview:
start with a single consumer thread, then generalize to multiple worker threads. Explicit requirements
named in the report: workers must **block, never spin**, waiting for the next deadline; a task scheduled
with an **earlier** deadline than whatever a worker is currently waiting on must correctly preempt that
wait; support cancellation and clean shutdown; use a monotonic clock (not wall-clock time, which can jump).

**Test Cases:**

| Scenario | Expectation |
|---|---|
| Schedule one task 50ms out | Runs once, ~50ms later, not before |
| Schedule a task 10ms out *after* one already waiting on a 100ms task | The 10ms task runs first; the waiting worker wakes early to notice it |
| Two workers, two tasks with the same deadline | Both run, no double-execution, no deadlock |
| `shutdown()` called with tasks still pending | Workers exit cleanly (accepted behavior: drain pending tasks first, or drop them — state the choice) |
| CPU usage while idle with nothing scheduled | ~0% — no polling loop |

**Key Insights:**
1. Min-heap on deadline, guarded by a single `Condition`. A worker computes `wait_time = deadline - now()`
   and calls `cv.wait(timeout=wait_time)` rather than sleeping and re-checking in a loop — this is what
   "block, don't spin" means concretely.
2. The preemption requirement is the crux: `schedule()` must `notify_all()` after pushing, even though no
   task's deadline has arrived yet — a worker sleeping on a *later* deadline needs to wake up, re-read the
   heap's new top, and possibly re-arm its wait with a shorter timeout.
3. A monotonic clock (`time.monotonic()`, not `time.time()`) avoids deadline math breaking under NTP
   adjustments or manual clock changes — a detail worth stating even if the test harness wouldn't catch
   it.

**Python Solution:**
```python
import heapq
import itertools
import threading
import time
from typing import Callable


class DelayedTaskRunner:
    """
    Time:  O(log n) per schedule/pop.
    Space: O(n) for n pending tasks.
    """

    def __init__(self, num_workers: int = 1):
        self._heap: list[tuple[float, int, Callable]] = []
        self._counter = itertools.count()  # heap tie-break for equal deadlines
        self._cv = threading.Condition()
        self._shutdown = False
        self._workers = [threading.Thread(target=self._run, daemon=True) for _ in range(num_workers)]
        for w in self._workers:
            w.start()

    def schedule(self, task: Callable, delay_seconds: float) -> None:
        deadline = time.monotonic() + delay_seconds
        with self._cv:
            heapq.heappush(self._heap, (deadline, next(self._counter), task))
            self._cv.notify_all()  # a new, earlier deadline may need to preempt a waiter

    def _run(self) -> None:
        with self._cv:
            while True:
                while not self._shutdown and not self._heap:
                    self._cv.wait()
                if self._shutdown and not self._heap:
                    return
                deadline, _, task = self._heap[0]
                remaining = deadline - time.monotonic()
                if remaining > 0:
                    self._cv.wait(timeout=remaining)
                    continue  # heap may have changed while waiting; re-check the top
                heapq.heappop(self._heap)
                self._cv.release()
                try:
                    task()
                finally:
                    self._cv.acquire()

    def shutdown(self) -> None:
        with self._cv:
            self._shutdown = True
            self._cv.notify_all()
        for w in self._workers:
            w.join()
```

---

## 12. Random-Access FIFO Queue

**Problem Statement:**
Design a queue supporting `add` (enqueue), `poll` (dequeue in FIFO order), and `get_random` (return a
uniformly random *currently present* element) all in O(1). Named follow-ups: define equality between two
such queues; run-length-encode runs of duplicate values to save space; make it thread-safe.

**Test Cases:**

| Scenario | Expectation |
|---|---|
| `add(1); add(2); poll()` | Returns `1` (FIFO, not LIFO) |
| `get_random()` immediately after several `add`s, no `poll` | Uniform over all added elements |
| `add`/`poll` interleaved many times | `get_random` only ever returns currently-live elements |
| `poll()` on empty queue | Raises / sentinel (state the choice) |
| Two queues with the same live elements in the same order | Equal under the defined equality |

**Key Insights:**
1. Unlike `RandomizedSet` (LC 380), there's no need to delete from the *middle* — `poll` always removes
   the oldest element, so a hashmap-of-indices isn't needed for correctness, only a way to get O(1)
   indexed access for `get_random`. That rules out a doubly linked list (O(1) add/poll but no O(1) random
   index) in favor of an array.
2. Backing store: a plain list with a lazy `head` index. `poll` just advances `head` (and drops the
   reference so it's collectible) instead of doing an O(n) shift; periodically compact
   (`buf = buf[head:]`) once dead space exceeds half the array, to keep memory bounded — this is exactly
   the amortized-O(1) argument, structured differently from a true ring buffer since random access needs
   contiguous indices rather than modular ones.
3. Run-length encoding follow-up: store `(value, count)` pairs instead of individual elements when
   consecutive `add`s repeat a value; `get_random` then needs to pick a *weighted* random pair (weight =
   count) rather than a uniform index — typically via a cumulative-count array and binary search, not a
   flat uniform pick.

**Python Solution:**
```python
import random


class RandomAccessFIFOQueue:
    """
    Time:  O(1) amortized add/poll/get_random.
    Space: O(n).
    """

    def __init__(self):
        self._buf: list = []
        self._head = 0

    def add(self, value) -> None:
        self._buf.append(value)

    def poll(self):
        if self._head >= len(self._buf):
            raise IndexError("poll from empty queue")
        value = self._buf[self._head]
        self._buf[self._head] = None  # drop reference
        self._head += 1
        if self._head > len(self._buf) // 2:  # amortized compaction
            self._buf = self._buf[self._head :]
            self._head = 0
        return value

    def get_random(self):
        if self._head >= len(self._buf):
            raise IndexError("get_random from empty queue")
        return self._buf[random.randrange(self._head, len(self._buf))]

    def __eq__(self, other) -> bool:
        return isinstance(other, RandomAccessFIFOQueue) and self._buf[self._head :] == other._buf[other._head :]

    def __len__(self) -> int:
        return len(self._buf) - self._head
```

---

## 13. System Design — Leader-Based Distributed Key-Value Store

**Problem Statement:**
Design a horizontally-scaled key-value store with `get`/`put`/`delete`. Reports describe two closely
related versions of this prompt: a single-node version focused on **crash recovery** (WAL, memtables,
SSTables, compaction — the storage-engine internals of something like a Kafka log segment or RocksDB), and
a distributed version layering **leader-based replication** on top for availability and scale.

**Functional Requirements:**
- `put(key, value)`, `get(key)`, `delete(key)`.
- Configurable read mode per client: **strong** (always current, routes to the partition leader) or
  **eventual** (may read slightly stale data from a follower, lower latency).
- Horizontal scale via partitioning; adding partitions/nodes shouldn't require a full remap of existing
  keys.

**Non-Functional Requirements:**
- Survive a leader crash without losing acknowledged writes.
- p99 read latency in the low milliseconds for the eventual-read path.
- No split-brain: a deposed leader must never be able to accept writes after a new leader is elected.

**High-Level Design:**
1. **Partitioning**: consistent hashing (with virtual nodes) maps each key to one of `N` partitions, so
   adding/removing a node moves roughly `1/N` of the keyspace instead of a full reshuffle.
2. **Per-partition replication group**: one leader, several followers, replicated via an append-only log —
   structurally identical to a Kafka partition's own ISR (in-sync replica) model, which is the framing
   every report explicitly leans on given the company.
3. **Write path**: client resolves `key -> partition -> current leader` via a metadata/controller service
   (a small strongly-consistent component, e.g. Raft-backed, analogous to Kafka's controller); write is
   appended to the leader's log and acknowledged once replicated to a quorum of followers.
4. **Read path**: strong reads go to the leader; eventual reads go to the nearest in-sync follower, whose
   replication lag is bounded and exposed so clients can decide their own staleness tolerance.
5. **Local storage engine per replica**: WAL for durability of the most recent writes, an in-memory
   memtable flushed periodically to immutable on-disk SSTables, background compaction to bound read
   amplification, tombstones for deletes (reclaimed during compaction).

**Data Model (sketch):**
```
partition_metadata(partition_id, leader_node, isr[], epoch)   # controller-owned
replica_log(partition_id, offset, key, value, op, epoch)      # append-only per replica
```

**Scaling & Reliability:**
- **Leader failure**: missed heartbeats trigger the controller (or the replication group itself, Raft-
  style) to elect a new leader from the most caught-up follower; the new leader's term/epoch is bumped so
  a stale former leader's writes are rejected by followers on epoch mismatch (fencing) — this is the
  concrete mechanism that prevents split-brain, not just "elect a new leader and move on."
- Clients cache the leader location and retry through the metadata service on a `NotLeader` error rather
  than hardcoding node addresses.
- Rebalancing on scale-out moves only the partitions assigned to virtual nodes on the new node; in-flight
  requests to a partition being moved are either queued briefly or served by the old leader until handoff
  completes, to avoid dropped writes during migration.

**Follow-Up Questions:**
1. How do you bound replication lag reporting for eventual reads? → each follower tracks and exposes its
   last-applied offset relative to the leader's; a client (or a load balancer) can refuse a follower whose
   lag exceeds its staleness budget.
2. What happens to reads mid-leader-election? → they either block briefly until a new leader is elected
   and announced, or the client falls back to an eventual read from a follower if it can tolerate the
   staleness — a real availability/consistency tradeoff to state explicitly.
3. How is read/write amplification managed as the store grows? → tiered/leveled compaction strategy
   choice trades write amplification (frequent merges) against read amplification (more SSTable levels to
   check) — there's no free lunch, and the interviewer wants that tradeoff named, not "just compact more."

---

## 14. System Design — Kubernetes-Managed Kafka Service

**Problem Statement:**
The signature Confluent system-design prompt, per essentially every onsite report: design Kafka itself, as
a managed service running on Kubernetes. Cover both the distributed-log fundamentals (topics, partitions,
replication, consumer groups) and the operational layer of running brokers as Kubernetes-managed pods.

**Functional Requirements:**
- Topics split into partitions; each partition is an ordered, append-only log.
- Producers write to a partition's leader; consumers read via consumer groups, with each partition
  consumed by exactly one member of a group at a time, tracking committed offsets.
- Configurable per-topic replication factor for durability.

**Non-Functional Requirements:**
- No data loss on a single broker (pod) failure, given `replication factor ≥ 2` and appropriate `acks`.
- Rolling upgrades and node maintenance must not drop below the minimum in-sync replica count for any
  partition.
- Multi-tenant: one noisy topic/tenant shouldn't starve others of broker I/O or Kubernetes cluster
  resources.

**High-Level Design:**
1. **Broker pods as a StatefulSet**: each broker gets a stable network identity (`broker-0`, `broker-1`,
   ...) and its own PersistentVolumeClaim, so a rescheduled pod reattaches to *its* existing data volume
   instead of starting empty — critical, since a broker's on-disk log segments are its only copy of
   recently-written, possibly still under-replicated data.
2. **Partition placement & replication**: each partition has a leader and an ISR; a control-plane
   component (Kafka's own controller, or an operator reconciling against it) assigns partition replicas
   across broker pods, ideally spread across Kubernetes nodes/availability zones so a single node failure
   doesn't take out every replica of a partition.
3. **Producer/consumer path**: producers discover the current partition leader via the cluster's metadata
   API and write directly to it; consumer groups coordinate partition assignment and offset commits
   through the broker-hosted group coordinator — no separate service needed for this, it's part of the
   broker protocol itself.
4. **Kubernetes-specific operational layer**: an operator (or StatefulSet rolling-update strategy) drives
   upgrades **one broker at a time**, explicitly waiting for that broker's partitions to rejoin every ISR
   before proceeding to the next — a naive rolling restart that doesn't wait can push a partition below
   its minimum ISR and block writes (or worse, lose data if it also loses the leader mid-restart).
5. **Scaling out**: adding a broker pod means rebalancing some partitions onto it — throttled data
   movement (rate-limited replica reassignment) so a rebalance doesn't saturate the network and degrade
   live traffic.

**Data Model (sketch):**
```
topics(name, partition_count, replication_factor, retention_ms)
partitions(topic, partition_id, leader_broker, isr[])
consumer_group_offsets(group_id, topic, partition_id, committed_offset)
```

**Scaling & Reliability:**
- Tenant isolation via per-topic/per-client-id quotas (produce/consume byte-rate limits) enforced at the
  broker, so one tenant's burst doesn't starve others sharing the cluster.
- Pod eviction (node drain, OOM) recovery: the StatefulSet reschedules the pod, it reattaches its PVC, and
  catches up any missed writes from the current leader before rejoining the ISR — it's *not* immediately
  back in the ISR the moment it starts.
- Cross-AZ replica placement is a deliberate anti-affinity rule, not a default Kubernetes scheduler
  behavior — call this out explicitly, since "just run it on k8s" glosses over exactly the placement logic
  that makes it durable.

**Follow-Up Questions:**
1. How do you avoid a rolling upgrade taking a partition's availability to zero? → never restart a
   partition's current leader and enough of its ISR simultaneously that quorum is lost; trigger a leader
   handoff before restarting a broker that currently leads partitions, so writes keep flowing through the
   new leader during the restart.
2. How does a consumer group survive a rebalance without reprocessing huge swaths of data? → offsets are
   committed (not just consumed) frequently enough that a partition reassignment mid-processing only
   replays a bounded, small window of records, not from the beginning of the partition.
3. What's the blast radius of losing an entire Kubernetes node? → bounded to whatever partitions had a
   replica scheduled there, and specifically not "all replicas of a partition" if anti-affinity rules
   spread them — this is where the design has to point back to placement, not just replication factor.

---

## 15. System Design — Idempotent URL Shortening Service

**Problem Statement:**
Asked across levels from new-grad to staff, with a specific Confluent-flavored twist stated explicitly in
the prompt: it's not enough for short codes to be unique — a **normalized long URL must converge on the
same short code** even under client retries and concurrent requests for the same URL. A naive
counter/Snowflake-ID-based shortener gives you uniqueness for free but not this idempotency: two
concurrent requests for the same long URL would mint two different codes unless something explicitly
prevents it.

**Functional Requirements:**
- `POST /shorten {long_url} -> {short_code}`, idempotent per *normalized* long URL (trailing slashes,
  query-param ordering, scheme case, etc. normalized before dedup/lookup).
- `GET /{short_code} -> 302 to long_url`.
- Optional custom aliases and expiration.

**Non-Functional Requirements:**
- Two concurrent `POST`s for the same long URL must return the **same** short code, not two different
  valid-but-distinct ones.
- Redirect path latency dominated by a cache hit, not a database round-trip, at read-heavy scale (reads
  vastly outnumber writes for a shortener).

**High-Level Design:**
1. **Normalization** happens before anything else touches storage — same long URL, same normalized form,
   every time, or idempotency is unenforceable no matter what comes next.
2. **Idempotent write, two viable approaches**:
   - **Content-addressed code**: derive the short code deterministically from a hash of the normalized
     URL (e.g., a prefix of a base62-encoded hash). Two concurrent requests for the same URL compute the
     *same* code independently — no coordination needed, at the cost of occasional hash-collision handling
     and less control over code shape/length.
   - **Reserve-then-confirm via a uniqueness constraint**: a DB unique index on `normalized_url`, written
     with an upsert (`INSERT ... ON CONFLICT (normalized_url) DO NOTHING RETURNING short_code`, or
     equivalent). Concurrent requests race to insert; the database's own constraint enforcement — not
     application-level locking — resolves the race, and the loser's query returns the winner's row instead
     of erroring.
   - Explicitly reject the tempting-but-wrong middle ground: "look up the URL, and if absent, insert a
     new random code" as two separate steps has a race between the lookup and the insert unless the insert
     itself is atomic against the uniqueness constraint.
3. **Read path**: cache `short_code -> long_url` (high hit rate, since a shortener's traffic is
   read-dominated); cache miss falls through to the database, which is indexed by `short_code` (the
   primary key) for O(1) lookup regardless of which write strategy above was used.
4. **Storage sizing**: a 7-character base62 code space is ~3.5 trillion — plenty of headroom; the
   `normalized_url` unique index (approach 2) is the thing that actually needs to scale with write volume,
   since every distinct URL ever shortened lives in it.

**Data Model (sketch):**
```
urls(short_code PK, normalized_url UNIQUE, long_url, created_at, expires_at)
```

**Follow-Up Questions:**
1. Why not just use a random code and a "does it already exist" pre-check? → that's the race described
   above; a pre-check-then-insert is never atomic without the constraint doing the real work, so the
   pre-check is redundant at best and misleading at worst.
2. How do you handle a hash collision under the content-addressed approach? → append a short disambiguator
   and re-hash, or fall back to the reserve-then-confirm path for that specific collision — state which,
   and why (collision probability at realistic scale is the deciding factor).
3. How does this interact with a CDN in front of the redirect endpoint? → `GET /{code}` responses can be
   cached at the edge with a reasonable TTL for immutable mappings, cutting origin load further — but only
   once expiration/deletion semantics are settled, since a cached redirect for a since-deleted code is a
   correctness bug, not just a staleness one.

---

## 16. System Design — Centralized Log Ingestion and Search Platform

**Problem Statement:**
Design the logging backbone every other service in the company writes to: ingest structured/unstructured
log lines from thousands of services, make them searchable by service, time range, and free text, and
enforce a retention policy — an ELK/Splunk-shaped system, which doubles at Confluent as an implicit "how
would you use Kafka as a component of a larger system" prompt.

**Functional Requirements:**
- Services ship logs continuously; ingestion must not backpressure the emitting service under normal
  load.
- `search(service?, time_range, free_text_query) -> matching log lines`, ranked/ordered by time.
- Configurable retention per log stream (e.g., 7 days hot, 30 days cold, then deleted).

**Non-Functional Requirements:**
- Ingestion durability: an accepted log line shouldn't silently vanish before indexing, even if a
  downstream indexer is temporarily down.
- Search latency acceptable for interactive debugging (low seconds), not analytical-batch speed.
- Ingestion volume dwarfs query volume by orders of magnitude — the system is write-optimized first.

**High-Level Design:**
1. **Ingestion buffer**: services write to a durable, partitioned log (a Kafka topic per log category is
   the natural fit here) rather than directly to a search index — this decouples "log accepted" from
   "log searchable," so an indexer outage doesn't cause data loss, only indexing lag.
2. **Indexing pipeline**: consumers read the ingestion log and build an inverted index (tokenized free
   text) plus structured field indexes (service, timestamp, log level), writing into a search-optimized
   store (e.g., an Elasticsearch/Lucene-family index) sharded by time window.
3. **Tiered storage**: recent, frequently-queried log windows live on fast local disk ("hot" tier);
   older windows roll to cheaper object storage ("cold" tier) with a slower, on-demand-rehydration search
   path — most debugging queries target the last few hours to days, so this tiering captures the actual
   access pattern rather than treating all retained data uniformly.
4. **Retention enforcement**: time-based index/segment deletion, not per-record deletes — drop entire
   time-partitioned shards/indices once they age out, which is O(1) per shard instead of scanning for
   expired individual records.
5. **Query path**: fan out a search query to the relevant time-sharded indices in parallel, merge and
   rank results by timestamp; a query scoped to a narrow time range and service should only touch a small
   number of shards, not the whole corpus.

**Data Model (sketch):**
```
ingestion_topic(service, timestamp, level, message, structured_fields)  # partitioned by service/hash
log_index_shard(time_window, service, tokenized_terms -> doc_ids, structured_field_indexes)
```

**Scaling & Reliability:**
- Indexer consumer-group lag is a first-class metric — under indexer outage, logs keep landing durably in
  the ingestion layer and search results just get stale by the lag amount, which is a graceful, visible
  degradation rather than data loss.
- Ingestion partitioning by service (or a hash of it) keeps one noisy service's log volume from starving
  another's indexing throughput within a shared consumer group, similar to per-tenant quota concerns
  elsewhere in the platform.
- Cold-tier rehydration on a query hitting old data is explicitly slower — surface that in the API/UI
  rather than making every query pay worst-case latency for the rare old-data lookup.

**Follow-Up Questions:**
1. How do you avoid the free-text index becoming the bottleneck at ingestion volume? → index
   asynchronously off the durable ingestion log (already the design), and shard the index itself by time
   window so indexing throughput scales horizontally with more shard-writers, not a single growing index.
2. A service starts emitting 100x its normal log volume during an incident — what happens? → per-service
   ingestion quotas/backpressure at the edge (reject or sample beyond a budget) so one incident's log
   storm doesn't degrade ingestion or indexing for every other service sharing the platform.
3. How would you support "search logs across a 6-month window" cheaply? → cold-tier query fans out to
   object storage with coarser, on-demand index rehydration, explicitly slower and possibly sampled/
   capped — state the latency tradeoff rather than implying cold storage is free.

---

## 17. Behavioral Themes

Confluent's behavioral round is a standard STAR-format interview — see
[`behavioral_interview.md`](./behavioral_interview.md) for general STAR-method prep. Themes and process
details specific to Confluent's loop:

- **Values-alignment round**: reports from 2026 onsites (post-IBM acquisition) describe a distinct
  "values alignment" interview layered on top of the usual behavioral round — expect direct questions
  about how you'd operate under organizational change and ambiguity, not just the standard
  ownership/conflict prompts. Have a story about staying productive through a reorg, leadership change,
  or shifting priorities that didn't originate with you.
- **Distributed-systems incident ownership**: given the product surface (Kafka, streaming
  infrastructure), expect a question probing a time you debugged or owned recovery from a production
  incident involving replication, data loss risk, or a distributed consensus/failover edge case — not
  just "a bug you fixed."
- **Cross-team technical communication**: Confluent's platform sits underneath many other teams' systems;
  be ready to discuss a time you explained a technical tradeoff (consistency vs. latency, a breaking API
  change, a migration timeline) to a non-expert or cross-functional stakeholder and got alignment.
- **Working through a contract you didn't design**: several onsite reports emphasize interviewers testing
  whether you respect an interface contract (e.g., "the solver must call the validator") rather than
  optimizing around it — the behavioral analog is a story about honoring a team convention or API
  boundary you personally would have designed differently, and why.
- **Response-time and process patience**: multiple reports independently note slower-than-typical
  response times and scheduling delays attributed to the IBM acquisition — not an interview *question*,
  but useful context: don't read a slow post-onsite response as a rejection signal by itself.

---

## References

Sources used for compiling these questions:
- [Confluent Interview Questions — 1point3acres](https://www.1point3acres.com/interview/company/confluent)
- [The Ultimate Collection of Confluent Interview Questions: 33 DSA, System Design & OOD (2018–2026) — FastPrep](https://www.fastprep.io/experience/the-ultimate-collection-of-confluent-interview-questions-33-dsa-system-design-oo-7me53gmzw5)
- [Match a Variadic Function Signature (Confluent Phone Screen) — FastPrep](https://www.fastprep.io/problems/confluent-variadic-function-signature-match)
- [Tail N Lines (Confluent Online Assessment) — FastPrep](https://www.fastprep.io/problems/confluent-tail-n-lines)
- [Implement tail -n (Confluent Onsite Interview) — FastPrep](https://www.fastprep.io/problems/confluent-implement-tail-n)
- [Detect a Silent Sensor (Confluent Onsite Interview) — FastPrep](https://www.fastprep.io/problems/confluent-silent-sensor-detector)
- [Reach Target Balance (Confluent Online Assessment) — FastPrep](https://www.fastprep.io/problems/confluent-reach-target-balance)
- [Sudoku Board Validation (Confluent Onsite Interview) — FastPrep](https://www.fastprep.io/problems/confluent-sudoku-board-validation)
- [Confluent Kafka Team Full Interview Experience and Feedback — 1point3acres](https://www.1point3acres.com/interview/thread/1099176)
- [Confluent Interview Guide: Insights on System Design and Coding Questions — 1point3acres](https://www.1point3acres.com/interview/thread/1138571)
- [Confluent Online Assessment: Two Easy to Medium Coding Questions — 1point3acres](https://www.1point3acres.com/interview/thread/1153721)
- [Confluent Software Engineer Online Assessment Experience and Problem Overview — 1point3acres](https://www.1point3acres.com/interview/thread/1151535)
- [Confluent Senior SDE1 Technical Phone Screen and HackerRank OA Review — 1point3acres](https://www.1point3acres.com/interview/thread/1163147)
- [Confluent Senior Software Engineer Interview Experience February 2026 — 1point3acres](https://www.1point3acres.com/interview/thread/1165654)
- [Confluent Senior SDE Interview Experience and Response Times — 1point3acres](https://www.1point3acres.com/interview/thread/1175376)

Note: 1point3acres' individual interview-report threads require forum membership to view full question
text/discussion; the problems above were reconstructed and expanded from publicly visible question titles
and summaries (and, where available, FastPrep's fuller public problem writeups sourced from the same
firsthand reports) into complete, solvable problem statements with original test cases and solutions.
