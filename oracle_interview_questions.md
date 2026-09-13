# Oracle Interview Questions

---

## Contents

**Coding**
1. [Set Matrix Zeroes (In Place)](#1-set-matrix-zeroes-in-place)
2. [Accounts Merge (Union-Find)](#2-accounts-merge-union-find)
3. [Time-Based Key-Value Store](#3-time-based-key-value-store)
4. [Design a Token Authentication Manager](#4-design-a-token-authentication-manager)
5. [Deploy Packages in Dependency Order](#5-deploy-packages-in-dependency-order)
6. [Dropped Requests Rate Limiter](#6-dropped-requests-rate-limiter)
7. [Maximum Subarray Sum with Length at Most K](#7-maximum-subarray-sum-with-length-at-most-k)
8. [Validate Parentheses String (with Wildcards)](#8-validate-parentheses-string-with-wildcards)
9. [Bottom View of a Binary Tree](#9-bottom-view-of-a-binary-tree)

**System Design**
10. [Online Presence System](#10-system-design--online-presence-system)

**Behavioral**
11. [Behavioral Themes](#11-behavioral-themes)

---

## 1. Set Matrix Zeroes (In Place)

**Problem Statement:**
Given an `m x n` integer matrix, if an element is `0`, set its entire row and column to `0`. Do it
in place, using `O(1)` extra space (i.e., don't build a separate set of rows/columns to zero out).

This is one of the few fully publicly-visible problems from Oracle's "featured" interview bank (most of
that bank is paywalled behind forum membership beyond title/tags: array, grid, hashmap, medium
difficulty, ~30 minutes).

**Example:**
```
Input:
[[1,1,1],
 [1,0,1],
 [1,1,1]]

Output:
[[1,0,1],
 [0,0,0],
 [1,0,1]]
```

**Test Cases:**

| Input | Output |
|---|---|
| `[[1,1,1],[1,0,1],[1,1,1]]` | `[[1,0,1],[0,0,0],[1,0,1]]` |
| `[[0,1,2,0],[3,4,5,2],[1,3,1,5]]` | `[[0,0,0,0],[0,4,5,0],[0,3,1,0]]` |
| `[[1]]` | `[[1]]` |
| `[[0]]` | `[[0]]` |
| No zeroes present | matrix unchanged |

**Key Insights:**
1. The naive solution uses `O(m + n)` extra space (a set of rows and a set of columns to zero out
   afterward) — that's an easy warm-up; the interview bar is the `O(1)`-extra-space version.
2. Use the matrix's own first row and first column as the marker arrays: if `matrix[r][c] == 0`, mark
   `matrix[r][0] = 0` and `matrix[0][c] = 0` instead of writing to a separate structure.
3. Because the first row and first column are now doing double duty (both data and markers), record
   *before* mutating anything whether the first row and first column originally contained a zero —
   otherwise you can't tell at the end whether to zero them out too.
4. Process the rest of the matrix (rows/cols `1..end`) using the markers first, then handle row 0 and
   column 0 last, using the two booleans captured in step 3.

**Python Solution:**
```python
def set_zeroes(matrix: list[list[int]]) -> None:
    """
    Mutates matrix in place.
    Time:  O(m * n)
    Space: O(1) extra (beyond the input matrix itself)
    """
    if not matrix or not matrix[0]:
        return

    m, n = len(matrix), len(matrix[0])
    first_row_has_zero = any(matrix[0][c] == 0 for c in range(n))
    first_col_has_zero = any(matrix[r][0] == 0 for r in range(m))

    for r in range(1, m):
        for c in range(1, n):
            if matrix[r][c] == 0:
                matrix[r][0] = 0
                matrix[0][c] = 0

    for r in range(1, m):
        for c in range(1, n):
            if matrix[r][0] == 0 or matrix[0][c] == 0:
                matrix[r][c] = 0

    if first_row_has_zero:
        for c in range(n):
            matrix[0][c] = 0
    if first_col_has_zero:
        for r in range(m):
            matrix[r][0] = 0
```

**Follow-Up Questions:**
1. What if the input is ragged (rows of different lengths)? → guard every inner loop with
   `c < len(matrix[r])` instead of assuming a fixed `n`; the first-row/first-column marker trick still
   works as long as row 0 itself is treated as the longest row (or you fall back to explicit row/column
   sets for a ragged matrix, since a single shared `n` no longer makes sense).
2. Generalize to a matrix of characters where a designated "zero" value isn't literally `0` → parameterize
   the sentinel value the function checks for; the algorithm is otherwise unchanged.
3. Can you do it in a single pass instead of three? → not while keeping `O(1)` extra space: you need the
   first pass to detect zeroes before their positions get overwritten by the marker-writing pass.

---

## 2. Accounts Merge (Union-Find)

**Problem Statement:**
Given a list of accounts, where `accounts[i] = [name, email_1, email_2, ...]`, merge any accounts that
belong to the same person — two accounts belong to the same person if they share at least one email
(the shared email may transitively chain multiple accounts together, even across different `name`
entries with the same underlying person). Return the merged accounts, each as
`[name, sorted_unique_emails...]`, in any order.

This is a visible title from Oracle's public OJ practice list (full description not given by the source
page; this is the standard, well-known version of the problem, matching the listed graph/union-find tags
elsewhere on Oracle's bank).

**Example:**
```
Input: [
  ["John", "johnsmith@mail.com", "john_newyork@mail.com"],
  ["John", "johnsmith@mail.com", "john00@mail.com"],
  ["Mary", "mary@mail.com"],
  ["John", "johnnybravo@mail.com"],
]
Output: [
  ["John", "john00@mail.com", "john_newyork@mail.com", "johnsmith@mail.com"],
  ["Mary", "mary@mail.com"],
  ["John", "johnnybravo@mail.com"],
]
```

**Test Cases:**

| Input | Output (group emails, any order) |
|---|---|
| Two accounts sharing one email | merged into one, emails deduped and sorted |
| Account with no shared emails | stays its own group |
| Three accounts chained A-B-C via shared emails | all three merge into one group |
| Same email appearing twice within one account's email list | deduped in output |

**Key Insights:**
1. Model each **account index** (not each email) as a node in a Union-Find structure — the thing being
   merged is accounts, and two accounts union whenever they share an email.
2. Track `email -> first account index seen with that email` in a hashmap; when a later account lists an
   already-seen email, union the two account indices.
3. After processing, group all emails by their account's *root* index (via `find`), then attach the name
   from any account in that root's group (they're guaranteed to be the same person, but not necessarily
   the same literal name string in weird inputs — take the root account's own name for determinism).

**Python Solution:**
```python
from collections import defaultdict


class DSU:
    """Union-Find with path compression (find) and simple union by pointing to the other root."""

    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]  # path compression
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb


def accounts_merge(accounts: list[list[str]]) -> list[list[str]]:
    """
    Time:  O(N * K * alpha(N)) for N accounts with up to K emails each (near-linear DSU ops),
           plus O(E log E) to sort emails within each merged group.
    Space: O(N + E) for the DSU array and the email-to-account map.
    """
    n = len(accounts)
    dsu = DSU(n)
    email_to_idx: dict[str, int] = {}

    for i, account in enumerate(accounts):
        for email in account[1:]:
            if email in email_to_idx:
                dsu.union(i, email_to_idx[email])
            else:
                email_to_idx[email] = i

    groups: dict[int, set[str]] = defaultdict(set)
    for email, i in email_to_idx.items():
        groups[dsu.find(i)].add(email)

    return [[accounts[root][0]] + sorted(emails) for root, emails in groups.items()]
```

**Follow-Up Questions:**
1. What if emails should be treated case-insensitively? → normalize (`email.lower()`) before using as the
   union-find key, but preserve/report original casing in the output if the spec requires it (track a
   canonical→original mapping separately).
2. Millions of accounts, updates streaming in over time → keep the DSU and `email_to_idx` map as
   persistent state rather than rebuilding from scratch on each new account; unioning a new account is
   still near-O(1) amortized.

---

## 3. Time-Based Key-Value Store

**Problem Statement:**
Design a time-based key-value store that supports:
- `set(key, value, timestamp)` — stores `value` for `key` at the given `timestamp`. Timestamps for a
  given key are guaranteed to be strictly increasing across successive `set` calls.
- `get(key, timestamp)` — returns the value associated with `key` at the largest previously-set
  timestamp that is `<= timestamp`. If no such timestamp exists (nothing was set at or before it, or the
  key doesn't exist), return `""`.

A visible title from Oracle's public OJ list (also listed as "Simplified Time-Based Key-Value Store"
elsewhere on the page, suggesting a warm-up variant of the same idea).

**Example:**
```
set("foo", "bar", 1)
get("foo", 1)  -> "bar"
get("foo", 3)  -> "bar"      # nothing set after ts=1, so the ts=1 value still applies
set("foo", "bar2", 4)
get("foo", 4)  -> "bar2"
get("foo", 5)  -> "bar2"
```

**Test Cases:**

| Operations | Result |
|---|---|
| `get("foo", 1)` with nothing ever set | `""` |
| `set("foo","bar",1)`; `get("foo",0)` | `""` (queried before the first set) |
| `set("foo","bar",1)`; `get("foo",1)` | `"bar"` |
| `set("foo","bar",1)`; `get("foo",3)` | `"bar"` |
| `set("foo","bar",1)`; `set("foo","bar2",4)`; `get("foo",4)` | `"bar2"` |

**Key Insights:**
1. Because timestamps are guaranteed strictly increasing per key, each key's `set` history is already
   sorted by construction — no need to sort or use a balanced tree, just append.
2. "Largest timestamp `<= query`" over a sorted list is a textbook binary search
   (`bisect.bisect_right(timestamps, query) - 1`), not a linear scan.
3. Keep timestamps and values as two parallel lists (or a list of tuples) per key — `bisect` needs a
   plain sorted sequence to search over.

**Python Solution:**
```python
import bisect
from collections import defaultdict


class TimeMap:
    """
    set(): O(1) amortized (list append)
    get(): O(log n) for n values stored under that key
    Space: O(total values stored across all keys)
    """

    def __init__(self):
        self._timestamps: dict[str, list[int]] = defaultdict(list)
        self._values: dict[str, list[str]] = defaultdict(list)

    def set(self, key: str, value: str, timestamp: int) -> None:
        self._timestamps[key].append(timestamp)
        self._values[key].append(value)

    def get(self, key: str, timestamp: int) -> str:
        timestamps = self._timestamps.get(key)
        if not timestamps:
            return ""
        idx = bisect.bisect_right(timestamps, timestamp) - 1
        return self._values[key][idx] if idx >= 0 else ""
```

**Follow-Up Questions:**
1. What if `set` calls for the same key could arrive out of timestamp order? → drop the strictly-
   increasing assumption and switch to `bisect.insort` on insert (O(n) per set) or a balanced structure
   (e.g. a skip list / `sortedcontainers.SortedList`) to keep O(log n) set as well as get.
2. How would you persist this across restarts? → append-only log per key (timestamp, value pairs are
   naturally an append-only log already) replayed on startup, same durability pattern as a WAL.

---

## 4. Design a Token Authentication Manager

**Problem Statement:**
Design an authentication token manager with a fixed `time_to_live` (TTL), supporting:
- `generate(token_id, current_time)` — generates a new token that expires at `current_time + ttl`.
- `renew(token_id, current_time)` — if the token exists and has **not yet expired** at `current_time`,
  extends its expiry to `current_time + ttl`; otherwise does nothing.
- `count_unexpired_tokens(current_time)` — returns how many tokens are **not yet expired** at
  `current_time`. A token is considered expired once `current_time >= expiry_time` (i.e., expiry at
  exactly `current_time` counts as already expired).

**Example:**
```
ttl = 5
generate("t1", 1)                    # t1 expires at 6
generate("t2", 2)                    # t2 expires at 7
count_unexpired_tokens(4) -> 2       # both still valid (6 > 4, 7 > 4)
renew("t1", 6)                        # no-op: t1 expires at 6, 6 >= 6 means already expired
renew("t2", 6)                        # ok: t2 expired at 7, 6 < 7, so renew -> new expiry 11
count_unexpired_tokens(6) -> 1       # t1 gone, t2 (expiry 11) remains
```

**Test Cases:**

| Operations | Result |
|---|---|
| `generate("a",1)`; `count_unexpired_tokens(0)` | `1` |
| `generate("a",1)` (ttl=5); `count_unexpired_tokens(6)` | `0` (expires exactly at 6, counts as expired) |
| `generate("a",1)`; `renew("a",10)` (ttl=5, already expired at 10) | no-op, still expires at 6 |
| `generate("a",1)`; `renew("a",3)` (ttl=5, not yet expired) | expiry updated to `3+5=8` |

**Key Insights:**
1. This is a plain hashmap of `token_id -> expiry_time` — the subtlety is entirely in the boundary
   condition (`>=` for "expired", not `>`), which is exactly what a careless implementation gets wrong.
2. `renew` must check the *current* expiry against `current_time` before overwriting it — renewing an
   already-expired token is explicitly a no-op, not a fresh `generate`.
3. A naive `count_unexpired_tokens` scans every token — fine unless the interviewer pushes on scale (see
   follow-ups) for millions of tokens with frequent count queries.

**Python Solution:**
```python
class AuthenticationManager:
    """
    generate/renew:            O(1)
    count_unexpired_tokens():  O(k) where k = number of currently-tracked (not yet lazily cleaned) tokens
    Space: O(number of live tokens)
    """

    def __init__(self, time_to_live: int):
        self.ttl = time_to_live
        self.expiry: dict[str, int] = {}

    def generate(self, token_id: str, current_time: int) -> None:
        self.expiry[token_id] = current_time + self.ttl

    def renew(self, token_id: str, current_time: int) -> None:
        if token_id in self.expiry and self.expiry[token_id] > current_time:
            self.expiry[token_id] = current_time + self.ttl

    def count_unexpired_tokens(self, current_time: int) -> int:
        expired_ids = [tid for tid, exp in self.expiry.items() if exp <= current_time]
        for tid in expired_ids:
            del self.expiry[tid]
        return len(self.expiry)
```

**Follow-Up Questions:**
1. `count_unexpired_tokens` is called far more often than `generate`/`renew`, on a huge token set → keep
   expiries in a structure ordered by expiry time (e.g. a sorted list / heap of `(expiry, token_id)`) so
   counting unexpired tokens is a binary search rather than a full scan — at the cost of `renew` now
   needing to reposition an entry instead of a plain dict update.
2. Multiple manager instances behind a load balancer (no shared state) → back `expiry` with a shared,
   TTL-capable store (e.g. Redis with native key expiry) instead of an in-process dict.

---

## 5. Deploy Packages in Dependency Order

**Problem Statement:**
Given a list of packages and a list of dependency pairs `(package, depends_on)` meaning `package` must
be deployed strictly after `depends_on`, return a valid deployment order for all packages. If the
dependency graph contains a cycle, no valid order exists — report that explicitly rather than returning a
partial/incorrect order.

A visible title from Oracle's public OJ list; this is the classic "Course Schedule II" shape applied to a
deploy pipeline, matching the topological-sort/graph tags elsewhere on Oracle's bank.

**Example:**
```
packages = ["A", "B", "C", "D"]
dependencies = [("B","A"), ("C","A"), ("D","B"), ("D","C")]
# B and C both need A first; D needs both B and C first.
Valid order: A, B, C, D   (or A, C, B, D — either is acceptable)
```

**Test Cases:**

| packages | dependencies | Result |
|---|---|---|
| `[A,B,C,D]` | `[(B,A),(C,A),(D,B),(D,C)]` | a valid order starting with A, ending with D |
| `[A,B]` | `[(A,B),(B,A)]` | raises — cycle detected |
| `[A,B,C]` | `[]` | any order of A,B,C (no constraints) |
| `[A]` | `[]` | `[A]` |

**Key Insights:**
1. This is Kahn's algorithm: build an adjacency list plus an in-degree count per package, and repeatedly
   deploy (dequeue) any package whose in-degree has dropped to zero.
2. If, at the end, fewer packages were emitted than exist in total, some packages never reached in-degree
   zero — that's exactly a cycle, detected without any separate DFS-based cycle check.
3. Ties (multiple packages simultaneously at in-degree zero) can be deployed in any order relative to
   each other — this is also exactly the set of packages that could be deployed **in parallel** in a real
   pipeline (see follow-up).

**Python Solution:**
```python
from collections import defaultdict, deque


def deploy_order(packages: list[str], dependencies: list[tuple[str, str]]) -> list[str]:
    """
    Time:  O(V + E)
    Space: O(V + E)
    """
    graph = defaultdict(list)
    indegree = {p: 0 for p in packages}
    for pkg, dep in dependencies:
        graph[dep].append(pkg)
        indegree[pkg] += 1

    queue = deque(p for p in packages if indegree[p] == 0)
    order = []

    while queue:
        node = queue.popleft()
        order.append(node)
        for nxt in graph[node]:
            indegree[nxt] -= 1
            if indegree[nxt] == 0:
                queue.append(nxt)

    if len(order) != len(packages):
        raise ValueError("cyclic dependency detected — no valid deploy order")
    return order
```

**Follow-Up Questions:**
1. Which packages can deploy **in parallel**? → group by BFS "layer": everything dequeued in one pass of
   the current queue (before any of that pass's newly-freed packages are added) can deploy concurrently,
   since none of them depends on another package in the same layer.
2. Return *all* valid deploy orders, not just one → becomes exponential backtracking (try every package
   currently at in-degree zero, recurse, restore in-degrees on backtrack) — call out explicitly that this
   only makes sense for small graphs.
3. A deploy fails partway through → packages already deployed stay deployed; everything transitively
   depending on the failed package should be held back — same shape as the failure-propagation logic in a
   DAG-based workflow engine.

---

## 6. Dropped Requests Rate Limiter

**Problem Statement:**
Implement a per-client rate limiter: `allow(client_id, timestamp)` returns `True` if the request should
be let through, or `False` if it should be dropped, given a limit of at most `max_requests` per rolling
`window_seconds` window for that client. Track how many requests have been dropped in total.

A visible title from Oracle's public OJ list, aligned with the "rate-limiting / throttling" tags reported
elsewhere on Oracle's system-design bank.

**Example:**
```
limiter = RateLimiter(max_requests=2, window_seconds=10)
allow("c1", 0)   -> True    # 1st request in window
allow("c1", 1)   -> True    # 2nd request in window
allow("c1", 2)   -> False   # 3rd request within the same 10s window -> dropped
allow("c1", 11)  -> True    # window has fully rolled past timestamp 0 and 1
```

**Test Cases:**

| Calls (client, ts) | Result |
|---|---|
| `("c1",0)` then `("c1",1)`, limit 2/10s | both `True` |
| `("c1",0)`, `("c1",1)`, `("c1",2)`, limit 2/10s | `True, True, False` |
| `("c1",0)`, `("c1",1)`, `("c1",2)`, `("c1",11)`, limit 2/10s | `True, True, False, True` |
| Two different clients each sending 2 requests, limit 2/10s | all 4 allowed (limits are per-client) |

**Key Insights:**
1. This is the sliding-window-log algorithm: keep each client's recent request timestamps in a deque,
   evict everything older than `window_seconds` from the front before checking/recording the new request.
2. Sliding-window-log is exact (unlike a fixed-window counter, which can let through up to `2x` the limit
   right at a window boundary) at the cost of `O(window size)` memory per client — worth naming that
   trade-off explicitly if asked to compare rate-limiting algorithms.
3. Eviction is amortized O(1) per call: each timestamp is pushed once and popped at most once across the
   whole run.

**Python Solution:**
```python
from collections import deque, defaultdict


class RateLimiter:
    """
    allow(): O(1) amortized per call
    Space:   O(max_requests) per active client
    """

    def __init__(self, max_requests: int, window_seconds: float):
        self.max_requests = max_requests
        self.window = window_seconds
        self._log: dict[str, deque] = defaultdict(deque)
        self.dropped_count = 0

    def allow(self, client_id: str, timestamp: float) -> bool:
        log = self._log[client_id]
        while log and timestamp - log[0] >= self.window:
            log.popleft()

        if len(log) >= self.max_requests:
            self.dropped_count += 1
            return False

        log.append(timestamp)
        return True
```

**Follow-Up Questions:**
1. Memory grows with request volume per client under sustained high traffic → switch to a sliding-window
   *counter* (two fixed buckets, weighted-average the previous bucket's count by how much of the window it
   still overlaps) for O(1) memory per client at the cost of approximate accuracy.
2. Shared across multiple rate-limiter processes (not single-process in-memory) → move the per-client log
   or counters into Redis, using `INCR`+`EXPIRE` or a Lua script for atomicity, so all processes see the
   same limiter state.

---

## 7. Maximum Subarray Sum with Length at Most K

**Problem Statement:**
Given an integer array `nums` and an integer `k`, return the maximum sum of any contiguous subarray whose
length is **at most** `k` (not exactly `k` — any length from `1` to `k` is allowed).

**Example:**
```
nums = [-1, 2, 3, -4, 5], k = 2
Subarrays of length <= 2 and their sums: [-1]=-1, [2]=2, [3]=3, [-4]=-4, [5]=5,
[-1,2]=1, [2,3]=5, [3,-4]=-1, [-4,5]=1
Output: 5   (from either [3] alone or [2,3])
```

**Test Cases:**

| nums | k | Result |
|---|---|---|
| `[-1,2,3,-4,5]` | `2` | `5` |
| `[1,1,1,1]` | `1` | `1` |
| `[5,-2,3]` | `3` | `6` (the whole array) |
| `[-5,-1,-3]` | `2` | `-1` (best is the single element `-1`) |

**Key Insights:**
1. Fixed-length-`k` window problems reduce to a simple sliding sum, but "at most `k`" means the optimal
   window's length varies — you can't just slide a single fixed-size window.
2. Reduce to prefix sums: the sum of `nums[l+1..r]` is `prefix[r] - prefix[l]`, and the length constraint
   `r - l <= k` becomes `l >= r - k`. For each `r`, you want the **minimum** `prefix[l]` over
   `l` in `[max(0, r-k), r-1]`.
3. That's a sliding-window-minimum over the prefix array — maintain a monotonically increasing deque of
   prefix-sum indices, popping stale indices off the front (out of the `l` range) and popping larger
   values off the back before pushing (so the front is always the current window's minimum).

**Python Solution:**
```python
from collections import deque


def max_subarray_sum_at_most_k(nums: list[int], k: int) -> int:
    """
    Time:  O(n) — each prefix index enters and leaves the deque at most once
    Space: O(k) for the deque, O(n) for the prefix array
    """
    n = len(nums)
    prefix = [0] * (n + 1)
    for i, x in enumerate(nums):
        prefix[i + 1] = prefix[i] + x

    window = deque([0])  # indices into `prefix`, values kept increasing
    best = float("-inf")

    for r in range(1, n + 1):
        while window and window[0] < r - k:
            window.popleft()

        if window:
            best = max(best, prefix[r] - prefix[window[0]])

        while window and prefix[window[-1]] >= prefix[r]:
            window.pop()
        window.append(r)

    return best
```

**Follow-Up Questions:**
1. What if the subarray length must be **exactly** `k`? → drop the monotonic deque entirely; a plain
   fixed-size sliding sum (`O(n)`, `O(1)` extra space) suffices since there's only one valid window length.
2. What if `k` can be as large as `n` (i.e., effectively "any length")? → this degenerates to the classic
   unconstrained Maximum Subarray (Kadane's algorithm), which the deque approach still handles correctly
   but is needlessly complex for — mention you'd special-case `k >= n` to Kadane's for simplicity/clarity.

---

## 8. Validate Parentheses String (with Wildcards)

**Problem Statement:**
Given a string `s` containing only `'('`, `')'`, and `'*'`, determine whether it can be considered a
valid parentheses string, where `'*'` can be treated as `'('`, `')'`, or an empty string.

**Example:**
```
"()"    -> True
"(*)"   -> True   # '*' as empty
"(*))"  -> True   # '*' as '('
")("    -> False
```

**Test Cases:**

| Input | Output |
|---|---|
| `"()"` | `True` |
| `"(*)"` | `True` |
| `"(*))"` | `True` |
| `")("` | `False` |
| `""` | `True` |
| `"***"` | `True` (all three can be empty) |

**Key Insights:**
1. Track a **range** of possible counts of unmatched `'('` instead of one exact count, since `'*'` is
   ambiguous: `low` = the minimum possible unmatched-open count (treating every usable `'*'` as `')'` or
   empty), `high` = the maximum possible (treating every `'*'` as `'('`).
2. `'('` increments both `low` and `high`; `')'` decrements both; `'*'` decrements `low` and increments
   `high` (it could resolve either way).
3. If `high` ever drops below `0`, no interpretation of the `'*'`s so far can keep the string valid —
   fail immediately. Clamp `low` at `0` (it can't mean "we're in debt" — that just means the more
   pessimistic reading isn't required at this point).
4. At the end, the string is valid iff `low == 0` is achievable, i.e. `0` is within `[low, high]` — since
   `low` is clamped at `0`, that's equivalent to checking `low == 0`.

**Python Solution:**
```python
def check_valid_string(s: str) -> bool:
    """
    Time:  O(n)
    Space: O(1)
    """
    low = high = 0

    for ch in s:
        if ch == "(":
            low += 1
            high += 1
        elif ch == ")":
            low -= 1
            high -= 1
        else:  # '*'
            low -= 1
            high += 1

        if high < 0:
            return False
        low = max(low, 0)

    return low == 0
```

**Follow-Up Questions:**
1. Return one concrete valid assignment of each `'*'` (not just yes/no) → this needs backtracking or a
   DP table over `(index, open_count)` reachability, reconstructing a path — the greedy `low`/`high` trick
   only answers feasibility, not a witness.
2. Support additional bracket types (`{}`/`[]`) alongside wildcarded `()`  → the low/high trick doesn't
   generalize cleanly to multiple bracket types with cross-type nesting; that variant typically needs a
   DP or stack-of-possibilities approach instead.

---

## 9. Bottom View of a Binary Tree

**Problem Statement:**
Given a binary tree, return its **bottom view**: for each horizontal distance (column) from the root,
the value of the node that would be seen when looking at the tree from directly below — i.e., the
bottom-most node at that column; if multiple nodes land in the same column at the same depth, the one
encountered later in a left-to-right traversal wins.

**Example:**
```
        20
       /  \
      8    22
     / \     \
    5   3    25
       / \
      10  14

Horizontal distances: 5=-2, 8=-1, 10=0, 20=0, 3=0, 14=1, 22=1, 25=2
Bottom view (by column, left to right): 5, 8, 10, 14, 25
# column 0 has 20 (depth 0), 3 (depth 2), and 10 (depth 2) all mapping to hd=0 in this example's
# left-skew — the deepest / last-visited one at that column wins, giving 10.
```

**Test Cases:**

| Tree | Bottom view |
|---|---|
| Single node `[5]` | `[5]` |
| Perfectly balanced 3-node tree `[1,[2],[3]]` | `[2, 1, 3]` |
| Left-skewed chain `1 -> 2 -> 3` (all left children) | `[3, 2, 1]` |
| Two nodes landing in the same column, different depths | the deeper one wins |

**Key Insights:**
1. Assign each node a horizontal distance (`hd`): root is `0`, left child is `parent_hd - 1`, right child
   is `parent_hd + 1`.
2. Traverse in BFS (level) order, and simply overwrite `column_map[hd] = node.value` on every visit —
   because BFS visits shallower nodes before deeper ones, and left-to-right within a level, the *last*
   write to a given `hd` is guaranteed to be the bottom-most (and, on a level-and-column tie, the
   rightmost) node — exactly the definition of bottom view.
3. The final answer is just the column map read out in increasing key (`hd`) order.

**Python Solution:**
```python
from collections import deque


def bottom_view(root) -> list:
    """
    root: a binary tree node with .val, .left, .right (or None for an empty tree)
    Time:  O(n)
    Space: O(n)
    """
    if root is None:
        return []

    column_map: dict[int, object] = {}
    queue = deque([(root, 0)])

    while queue:
        node, hd = queue.popleft()
        column_map[hd] = node.val  # later visits (deeper, or same-level rightward) overwrite earlier ones

        if node.left:
            queue.append((node.left, hd - 1))
        if node.right:
            queue.append((node.right, hd + 1))

    return [column_map[hd] for hd in sorted(column_map)]
```

**Follow-Up Questions:**
1. Return the **top** view instead → same traversal, but only write `column_map[hd]` if the column
   hasn't been seen yet (first write per column wins, instead of last).
2. Extremely wide/skewed trees causing many distinct horizontal distances → the algorithm's complexity is
   unaffected (`O(n)` regardless of shape), but note that `hd` can range over `O(n)` distinct values for a
   maximally-skewed tree, which is fine for a hashmap-based `column_map` but would matter if you tried to
   use a fixed-size array instead.

---

## 10. System Design — Online Presence System

**Problem Statement:**
Design an online-presence system, like a chat app's "online / away / offline" indicator for each user,
that scales to hundreds of millions of users with near-real-time updates.

**Functional Requirements:**
- Track each user's current presence state (`online`, `away`, `offline`).
- Let a client query the presence of its contacts/subscribed users.
- Propagate a presence change to interested subscribers promptly.
- Presence should reflect ungraceful disconnects (app killed, network drop) without requiring an
  explicit "going offline" signal from the client.

**Non-Functional Requirements:**
- Near-real-time propagation (seconds, not minutes) to subscribers of a presence change.
- Scale to hundreds of millions of concurrent users without a single bottleneck component.
- Tolerate a spike of simultaneous disconnects (e.g., a mobile network outage) without cascading load.

**High-Level Design:**
1. **Heartbeat + TTL**: each connected client periodically pings a lightweight heartbeat endpoint (or the
   heartbeat rides on an existing persistent connection, e.g. a WebSocket). Presence state is written to
   a fast key-value store (e.g., Redis) with a TTL slightly longer than the heartbeat interval.
2. **Implicit offline detection**: because the TTL entry naturally expires if heartbeats stop, an
   ungraceful disconnect resolves to "offline" for free — there's no need for the server to detect the
   disconnect explicitly or run a separate reaper process for the common case.
3. **Fan-out on change**: when a user's presence changes (a new heartbeat after being offline, or a TTL
   expiry), publish a presence-changed event on a per-user pub/sub channel; only that user's actual
   subscribers (contacts, or viewers of their profile) receive it — never a global broadcast.
4. **Read path**: a client fetches a contact's presence either by subscribing to that pub/sub channel
   (for a live-updating UI) or via a direct point read of the KV store (for a one-off check, e.g.
   rendering a contact list on app open).
5. **High-fanout accounts**: a small number of accounts (e.g., a public/celebrity account) can have
   millions of subscribers watching one presence value — cache the current value aggressively and
   consider coalescing/delaying updates for very high-fanout accounts rather than pushing every flicker
   in real time to every subscriber.

**Data Model (sketch):**
```
presence(user_id, status, last_heartbeat_ts)     # KV store, TTL ~= 2x heartbeat interval
subscriptions(user_id -> set of subscriber_ids)  # or derived from an existing contacts/social graph
```

**Scaling & Reliability:**
- Shard the KV store by `user_id` (consistent hashing) so heartbeat writes spread evenly across nodes.
- Heartbeat traffic is the dominant write volume at this scale — keep the heartbeat payload minimal and
  the write path a single fast key update, not a multi-step transaction.
- Pub/sub fan-out is per-subscriber-count, not global, so a spike in disconnects (e.g. a network outage
  recovering) causes a burst of presence-changed events bounded by how many users are actually affected,
  not the whole user base.
- For very high-fanout accounts, decouple "true" presence (as tracked internally) from "displayed"
  presence (which can lag slightly or be sampled) to cap fan-out cost.

**Follow-Up Questions:**
1. How do you distinguish "away" from "offline" if both look like "no recent activity"? → "away" is
   typically a client-reported state (app backgrounded but still holding a connection) sent explicitly,
   distinct from the TTL-driven "offline" which requires no client cooperation at all.
2. A user has a flaky connection and flaps between online/offline every few seconds — how do you avoid
   spamming subscribers? → debounce presence-changed events (only publish if the state has been stable
   for some minimum duration) rather than publishing on every raw heartbeat gap.
3. How would you test TTL-based expiry behavior deterministically? → inject a fake clock into the
   heartbeat/TTL logic in tests rather than relying on wall-clock sleeps.

---

## 11. Behavioral Themes

See [`behavioral_interview.md`](./behavioral_interview.md) for general STAR-method prep. Themes
specific to Oracle's loop:

- **Ownership under ambiguity**: standard enterprise-software behavioral theme — can you drive a project
  to completion with unclear requirements, and take ownership of outcomes (including failures) rather
  than deflecting.
- **Proactive requirement clarification**: a story showing you clarified ambiguous requirements up front
  (rather than building the wrong thing and finding out later) tends to land well here.
- **Leadership without authority**: influencing a technical direction or unblocking a cross-team
  dependency without having formal reporting authority over the people involved.

---

## References

Sources used for compiling these questions:
- [Oracle Interview Questions - 1point3acres](https://www.1point3acres.com/interview/problems/company/oracle)

Note: the source page's "featured" bank requires forum membership to view full question text for nearly
every entry — "Set Matrix Zeroes" was the one fully publicly-visible coding problem there, and the page's
system-design and behavioral sections show no individual titles at all (only a generic "log in to unlock"
placeholder), so problem #10 and the behavioral themes above remain reconstructed from general Oracle
interview-loop patterns rather than page-specific titles. Problems #2–9, however, are real, publicly
visible titles from the page's separate "OJ practice problems" list (shown with titles but no
descriptions); each was implemented here as the standard, well-known version of that named problem.
