# LinkedIn Interview Questions

Source: https://www.1point3acres.com/interview/problems/company/linkedin

> **Note:** read on 2026-11-06 (49 questions: 36 coding, 9 system design, 2 behavioral, 2 other; 114 related
> practice problems). The page shows only a summary card per question and keeps the full text for members, so
> each problem below is restated from its card — **clarify requirements with the interviewer**; signatures and
> follow-ups are my reconstruction. Every Python block runs and passes its own asserts.

**What the page says about the loop** (summarised, not verbatim): Staff candidates sit a two-round phone screen
(coding plus a project retro) before a five-round onsite that includes an **AI-coding session** in CoderPad's
assistant pane. Behavioral depth is the main downlevel axis — the same coding bar that passes at Senior can cap a
Staff loop with a downlevel offer. Several recent coding rounds are delivered as the AI-coding round, where the
interviewer watches whether you keep the data-structure decision under your own control.

---

## Table of Contents

**Coding**
1. [Phone Keypad: Letter Combinations and Word Filter](#1-phone-keypad-letter-combinations-and-word-filter-high)
2. [Weighted Sampling from a Distribution](#2-weighted-sampling-from-a-distribution-medium)
3. [Max Stack](#3-max-stack-medium)
4. [LRU Cache with a Thread-Safe Follow-up](#4-lru-cache-with-a-thread-safe-follow-up-medium)
5. [Minimum Sum-of-Distances Meeting Point](#5-minimum-sum-of-distances-meeting-point-medium)
6. [Sort a Quadratic-Transformed Sorted Array](#6-sort-a-quadratic-transformed-sorted-array-single-report)
7. [Count Distinct Values in a Massive Sorted Array](#7-count-distinct-values-in-a-massive-sorted-array-single-report)
8. [Biased Coin to Uniform Range](#8-biased-coin-to-uniform-range-low)
9. [LFU Cache](#9-lfu-cache-low)
10. [Function Inclusive / Exclusive Time](#10-function-inclusive--exclusive-time-low)
11. [Alert Monitor: Rolling Window and Spike Detection](#11-alert-monitor-rolling-window-and-spike-detection-low)
12. [Meeting Scheduler: Earliest Available Slot](#12-meeting-scheduler-earliest-available-slot-low)

**System Design**
13. [Metrics and Monitoring Platform](#13-system-design--metrics-and-monitoring-platform-medium)
14. [Distributed Job Scheduler / Calendar Service](#14-system-design--distributed-job-scheduler--calendar-service-medium)
15. [Type-Ahead / Autocomplete](#15-system-design--type-ahead--autocomplete-low)

**Behavioral**
16. [Behavioral Pack and Project Deep-Dive](#16-behavioral-pack-and-project-deep-dive-very-high)

---

## 1. Phone Keypad: Letter Combinations and Word Filter (high)

Card: the classic letter combinations of a digit string shows up in nearly every recent loop as the warm-up. A
common variant: given a digit string and a dictionary, return the words that can be typed with those digits —
the twist that turns it into a hash/trie problem instead of pure backtracking.

```python
KEYPAD = {"2": "abc", "3": "def", "4": "ghi", "5": "jkl",
          "6": "mno", "7": "pqrs", "8": "tuv", "9": "wxyz"}
LETTER_TO_DIGIT = {ch: d for d, letters in KEYPAD.items() for ch in letters}


def letter_combinations(digits):
    if not digits:
        return []
    out = [""]
    for d in digits:
        out = [prefix + ch for prefix in out for ch in KEYPAD[d]]
    return out


def words_for_digits(digits, dictionary):
    # Map each word to its digit string once; the query is then a dict lookup.
    index = {}
    for word in dictionary:
        key = "".join(LETTER_TO_DIGIT[ch] for ch in word.lower())
        index.setdefault(key, []).append(word)
    return sorted(index.get(digits, []))


assert letter_combinations("") == []
assert letter_combinations("2") == ["a", "b", "c"]
assert sorted(letter_combinations("23")) == sorted(
    ["ad", "ae", "af", "bd", "be", "bf", "cd", "ce", "cf"])
assert len(letter_combinations("7979")) == 4 ** 4

words = ["home", "good", "gone", "hood", "hoof", "hello", "Hind"]
assert words_for_digits("4663", words) == ["gone", "good", "home", "hood", "hoof"]
assert words_for_digits("43556", words) == ["hello"]
assert words_for_digits("999", words) == []
```

Complexity: combinations are O(4^n · n) output-bound; the dictionary variant is O(total letters) to build and
O(1) per query. If the interviewer says "many queries, huge dictionary", build the index once (that is the
answer), and if the input is a *prefix* of digits, use a trie keyed by digit.

## 2. Weighted Sampling from a Distribution (medium)

Card: draw from `N` categories in O(log N) — inverse-CDF with a prefix array and binary search; follow-ups on
weights that do not sum to 1, very peaked distributions, and rejection-sampling cost. Last asked 2026-10-27.

```python
import bisect
import random
from itertools import accumulate


class WeightedSampler:
    def __init__(self, weights, rng=None):
        assert weights and all(w >= 0 for w in weights) and sum(weights) > 0
        self.cdf = list(accumulate(weights))            # unnormalised; no need to sum to 1
        self.total = self.cdf[-1]
        self.rng = rng or random.Random()

    def sample(self):
        u = self.rng.random() * self.total               # uniform in [0, total)
        return bisect.bisect_right(self.cdf, u)          # first bucket whose cdf exceeds u


sampler = WeightedSampler([1, 0, 3, 6], random.Random(42))
counts = [0, 0, 0, 0]
for _ in range(40000):
    counts[sampler.sample()] += 1
assert counts[1] == 0                                    # zero weight is never drawn
for got, want in zip(counts, [0.1, 0.0, 0.3, 0.6]):
    assert abs(got / 40000 - want) < 0.02
assert WeightedSampler([5], random.Random(1)).sample() == 0
# Deterministic boundary check with a stub RNG: u = 0.0999 -> bucket 0, u = 0.1 -> skips to bucket 2.
class Stub:
    def __init__(self, value):
        self.value = value

    def random(self):
        return self.value


assert WeightedSampler([1, 0, 3, 6], Stub(0.0999)).sample() == 0
assert WeightedSampler([1, 0, 3, 6], Stub(0.1)).sample() == 2
```

Follow-ups: for *many* samples from a fixed distribution use the **alias method** (O(N) build, O(1) per draw);
for streaming weights use a Fenwick tree (O(log N) update and sample); to avoid float issues with huge weights,
sample in log space (Gumbel-max) or keep integer weights.

## 3. Max Stack (medium)

Card: LeetCode 716 — `push`, `pop`, `top`, `peekMax`, `popMax`; the optimal answer is a doubly linked list plus a
sorted structure keyed on value (O(log N) `popMax`, O(1) the rest).

Python has no TreeMap, so use a max-heap with lazy deletion keyed by a unique node id — same big-O in practice
and much shorter to write.

```python
import heapq
import itertools


class MaxStack:
    def __init__(self):
        self.stack = []                  # list of (id, value); lazily pruned of removed ids
        self.heap = []                   # (-value, -id) so ties pop the most recent push first
        self.removed = set()
        self.ids = itertools.count()

    def _prune_stack(self):
        while self.stack and self.stack[-1][0] in self.removed:
            self.stack.pop()

    def _prune_heap(self):
        while self.heap and -self.heap[0][1] in self.removed:
            heapq.heappop(self.heap)

    def push(self, x):
        i = next(self.ids)
        self.stack.append((i, x))
        heapq.heappush(self.heap, (-x, -i))

    def pop(self):
        self._prune_stack()
        i, x = self.stack.pop()
        self.removed.add(i)
        return x

    def top(self):
        self._prune_stack()
        return self.stack[-1][1]

    def peek_max(self):
        self._prune_heap()
        return -self.heap[0][0]

    def pop_max(self):
        self._prune_heap()
        neg_x, neg_i = heapq.heappop(self.heap)
        self.removed.add(-neg_i)
        return -neg_x


s = MaxStack()
for x in (5, 1, 5):
    s.push(x)
assert s.top() == 5 and s.peek_max() == 5
assert s.pop_max() == 5                     # removes the *top-most* 5
assert s.top() == 1 and s.peek_max() == 5
assert s.pop() == 1
assert s.top() == 5 and s.pop_max() == 5
s.push(2)
s.push(9)
s.push(3)
assert s.pop_max() == 9 and s.top() == 3 and s.peek_max() == 3
assert s.pop() == 3 and s.pop() == 2
```

Each element is added and removed once, so everything is amortised O(log N). If the interviewer wants the
"textbook" version, say you would use a doubly linked list plus a sorted map of value → list of nodes, and the
lazy heap is the Python-friendly equivalent; ties must pop the **top-most** copy.

## 4. LRU Cache with a Thread-Safe Follow-up (medium)

Card: LRU cache, then thread-safe. In post-reform loops this is delivered as the AI-coding round: outline the
design, let the assistant generate the skeleton, then dry-run tests and reason about lock granularity.

```python
import threading
from collections import OrderedDict


class LRUCache:
    def __init__(self, capacity):
        assert capacity > 0
        self.capacity = capacity
        self.data = OrderedDict()
        self.lock = threading.Lock()

    def get(self, key):
        with self.lock:                          # get mutates recency, so it needs the lock too
            if key not in self.data:
                return None
            self.data.move_to_end(key)
            return self.data[key]

    def put(self, key, value):
        with self.lock:
            self.data[key] = value
            self.data.move_to_end(key)
            if len(self.data) > self.capacity:
                self.data.popitem(last=False)    # evict least recently used


c = LRUCache(2)
c.put("a", 1)
c.put("b", 2)
assert c.get("a") == 1                           # a is now most recent
c.put("c", 3)                                    # evicts b
assert c.get("b") is None and c.get("a") == 1 and c.get("c") == 3

shared = LRUCache(50)


def hammer(offset):
    for i in range(2000):
        shared.put((offset, i % 80), i)
        shared.get((offset, (i * 7) % 80))


threads = [threading.Thread(target=hammer, args=(t,)) for t in range(8)]
for t in threads:
    t.start()
for t in threads:
    t.join()
assert len(shared.data) <= 50                    # capacity held under contention
```

Say the design out loud even if you use `OrderedDict`: a hash map to nodes plus a doubly linked list so every
operation is O(1); `OrderedDict` is that pair. Lock-granularity discussion: one lock is correct but serialises
reads; options are striping (shard the cache by key hash, each shard with its own lock and capacity) or a
read-mostly approximation (record accesses in a buffer and apply them in batches, as Caffeine does).

## 5. Minimum Sum-of-Distances Meeting Point (medium)

Card: given points on a line, find the point minimising total absolute distance — the median; the signal is the
proof. Follow-ups: L2 distance with k positions (much harder) and streaming median.

**Proof in two lines:** moving the meeting point right by `ε` past a point set changes the cost by
`ε·(#points to its left − #points to its right)`; it improves while more points are on the right, so the optimum
is where the two counts balance — the median (any point between the two middle values when `n` is even).

```python
import heapq
import random


def best_meeting_point(points):
    xs = sorted(points)
    median = xs[len(xs) // 2]
    return median, sum(abs(x - median) for x in xs)


def quickselect(values, k):
    values = list(values)
    lo, hi = 0, len(values) - 1
    while True:
        pivot = values[random.randint(lo, hi)]
        less = [v for v in values[lo:hi + 1] if v < pivot]
        equal = [v for v in values[lo:hi + 1] if v == pivot]
        if k < lo + len(less):
            values[lo:lo + len(less)] = less
            hi = lo + len(less) - 1
        elif k < lo + len(less) + len(equal):
            return pivot
        else:
            greater = [v for v in values[lo:hi + 1] if v > pivot]
            lo += len(less) + len(equal)
            values[lo:hi + 1] = greater


class RunningMedian:
    def __init__(self):
        self.low, self.high = [], []            # max-heap (negated) and min-heap

    def add(self, x):
        heapq.heappush(self.low, -x)
        heapq.heappush(self.high, -heapq.heappop(self.low))
        if len(self.high) > len(self.low):
            heapq.heappush(self.low, -heapq.heappop(self.high))

    def median(self):
        return -self.low[0]                     # lower median


point, cost = best_meeting_point([1, 2, 9])
assert (point, cost) == (2, 8)
assert best_meeting_point([1, 2, 3, 10])[1] == 10               # any point in [2, 3] gives 10
assert best_meeting_point([5])[1] == 0
brute = min(sum(abs(x - c) for x in [1, 4, 6, 20]) for c in range(0, 25))
assert best_meeting_point([1, 4, 6, 20])[1] == brute
for _ in range(200):
    data = [random.randint(-50, 50) for _ in range(random.randint(1, 15))]
    assert quickselect(data, len(data) // 2) == sorted(data)[len(data) // 2]
rm = RunningMedian()
for x in [5, 1, 9, 3]:
    rm.add(x)
assert rm.median() == 3
```

Complexity: sort is O(n log n); quickselect gets the median in expected O(n). For L2 with `k` positions the
problem becomes 1-D k-means (optimal via DP in O(k·n²), or O(kn) with the monotone-matrix trick) — mention it
and do not attempt it live.

## 6. Sort a Quadratic-Transformed Sorted Array (single report)

Card: sorted integers and coefficients `a, b, c`; apply `f(x) = ax² + bx + c` and return the results sorted in
O(n). Asked alongside an ML-fundamentals oral.

The parabola opens up when `a > 0` (values smallest at the vertex, largest at the ends), down when `a < 0`; the
transformed array is therefore two monotone runs merged from the ends toward the middle.

```python
def sort_transformed(nums, a, b, c):
    f = lambda x: a * x * x + b * x + c
    n = len(nums)
    out = [0] * n
    lo, hi = 0, n - 1
    index = n - 1 if a >= 0 else 0
    step = -1 if a >= 0 else 1
    while lo <= hi:
        left, right = f(nums[lo]), f(nums[hi])
        if a >= 0:                             # fill from the largest end
            if left > right:
                out[index] = left
                lo += 1
            else:
                out[index] = right
                hi -= 1
        else:                                   # fill from the smallest end
            if left < right:
                out[index] = left
                lo += 1
            else:
                out[index] = right
                hi -= 1
        index += step
    return out


assert sort_transformed([-4, -2, 2, 4], 1, 3, 5) == [3, 9, 15, 33]
assert sort_transformed([-4, -2, 2, 4], -1, 3, 5) == [-23, -5, 1, 7]
assert sort_transformed([1, 2, 3], 0, 2, 1) == [3, 5, 7]            # a == 0: linear, increasing
assert sort_transformed([1, 2, 3], 0, -2, 1) == [-5, -3, -1]        # a == 0: linear, decreasing
assert sort_transformed([], 1, 1, 1) == []
import random
for _ in range(300):
    data = sorted(random.sample(range(-30, 30), random.randint(0, 12)))
    a, b, c = (random.randint(-3, 3) for _ in range(3))
    assert sort_transformed(data, a, b, c) == sorted(a * x * x + b * x + c for x in data)
```

The `a == 0` case is the usual miss — with `a ≥ 0` handled as "fill from the largest end" it works for both
increasing and decreasing lines because the extremes are still at the two ends.

## 7. Count Distinct Values in a Massive Sorted Array (single report)

Card: `n` may be billions, distinct count `k` only hundreds. Skip each equal-value run with a boundary search for
O(k log n); analyse a divide-and-conquer alternative that can degrade to O(n).

```python
import bisect


def count_distinct(arr):
    n, i, distinct = len(arr), 0, 0
    while i < n:
        distinct += 1
        i = bisect.bisect_right(arr, arr[i], i, n)      # jump past the whole run of arr[i]
    return distinct


def count_distinct_dc(arr, lo=0, hi=None):
    """Divide and conquer: if both ends match, the whole slice is one value."""
    hi = len(arr) - 1 if hi is None else hi
    if lo > hi:
        return 0
    if arr[lo] == arr[hi]:
        return 1
    mid = (lo + hi) // 2
    left = count_distinct_dc(arr, lo, mid)
    right = count_distinct_dc(arr, mid + 1, hi)
    return left + right - (1 if arr[mid] == arr[mid + 1] else 0)


assert count_distinct([]) == 0
assert count_distinct([7]) == 1
assert count_distinct([1, 1, 1, 2, 2, 5, 5, 5, 5]) == 3
big = [0] * 1_000_000 + [1] * 1_000_000 + [2] * 3
assert count_distinct(big) == 3
assert count_distinct_dc(big) == 3
assert count_distinct_dc([1, 2, 3, 4, 5]) == 5              # all distinct: O(n) work
```

Why the first wins: it does `k` binary searches (`O(k log n)`), so with `k` in the hundreds and `n` in the
billions that is a few thousand steps. The divide-and-conquer version prunes only when a slice is constant, so
when `k ≈ n` it visits everything — O(n). Mention that you would use exponential (galloping) search from `i` if
runs are short, to get O(k log(n/k)).

## 8. Biased Coin to Uniform Range (low)

Card: given a biased Bernoulli, build a uniform sampler over `[0, 6]` — Von Neumann to get a fair bit, bit-pack
three fair bits, reject values outside the range.

```python
import random


def fair_bit(biased):
    while True:
        a, b = biased(), biased()
        if a != b:
            return a


def uniform_0_6(biased):
    while True:
        value = (fair_bit(biased) << 2) | (fair_bit(biased) << 1) | fair_bit(biased)
        if value <= 6:                           # reject 7 so the other eight-to-seven odds stay equal
            return value


rng = random.Random(11)
coin = lambda: 1 if rng.random() < 0.7 else 0
draws = [uniform_0_6(coin) for _ in range(35000)]
assert set(draws) == set(range(7))
for v in range(7):
    assert abs(draws.count(v) / 35000 - 1 / 7) < 0.012
```

Cost: each fair bit needs about `1/(p(1−p))` flips, and each output needs `3 × 8/7` fair bits on average. Name the
shape of the generalisation: `ceil(log2 n)` bits with rejection for any `n`.

## 9. LFU Cache (low)

Card: LeetCode 460 in O(1): key→node, frequency→ordered list, and a `min_freq` cursor.

```python
from collections import defaultdict, OrderedDict


class LFUCache:
    def __init__(self, capacity):
        self.capacity = capacity
        self.values = {}                          # key -> value
        self.freq = {}                            # key -> access count
        self.buckets = defaultdict(OrderedDict)   # count -> keys in LRU order (oldest first)
        self.min_freq = 0

    def _touch(self, key):
        count = self.freq[key]
        del self.buckets[count][key]
        if not self.buckets[count]:
            del self.buckets[count]
            if self.min_freq == count:
                self.min_freq += 1
        self.freq[key] = count + 1
        self.buckets[count + 1][key] = None

    def get(self, key):
        if key not in self.values:
            return -1
        self._touch(key)
        return self.values[key]

    def put(self, key, value):
        if self.capacity == 0:
            return
        if key in self.values:
            self.values[key] = value
            self._touch(key)
            return
        if len(self.values) >= self.capacity:
            victim, _ = self.buckets[self.min_freq].popitem(last=False)   # LFU, ties -> LRU
            if not self.buckets[self.min_freq]:
                del self.buckets[self.min_freq]
            del self.values[victim]
            del self.freq[victim]
        self.values[key] = value
        self.freq[key] = 1
        self.buckets[1][key] = None
        self.min_freq = 1


c = LFUCache(2)
c.put(1, 1)
c.put(2, 2)
assert c.get(1) == 1
c.put(3, 3)                                       # evicts key 2 (lowest frequency)
assert c.get(2) == -1 and c.get(3) == 3
c.put(4, 4)                                       # 1 and 3 both have freq 2; 1 is older, so 1 goes
assert c.get(1) == -1 and c.get(3) == 3 and c.get(4) == 4
z = LFUCache(0)
z.put(1, 1)
assert z.get(1) == -1
```

Tie-breaking is the usual source of bugs: among equal frequencies evict the **least recently used**, which is why
each frequency bucket is an ordered list rather than a set. Resetting `min_freq = 1` on every insert is correct
because a new key always has count 1.

## 10. Function Inclusive / Exclusive Time (low)

Card: LeetCode 636 variant — `(function_id, start|end, timestamp)` logs; compute inclusive and exclusive time for
a target function using a stack that attributes elapsed time to the current top.

```python
def function_times(logs):
    """logs: list of (func, 'start'|'end', timestamp); 'end' is inclusive of that timestamp tick."""
    inclusive, exclusive = {}, {}
    stack = []                                   # [func, start_time, child_time]
    for func, kind, ts in logs:
        if kind == "start":
            stack.append([func, ts, 0])
        else:
            f, started, child = stack.pop()
            assert f == func, "mismatched end"
            duration = ts - started + 1
            exclusive[f] = exclusive.get(f, 0) + duration - child
            if not any(frame[0] == f for frame in stack):   # outermost call only: avoid double counting recursion
                inclusive[f] = inclusive.get(f, 0) + duration
            if stack:
                stack[-1][2] += duration
    return inclusive, exclusive


logs = [("a", "start", 0), ("b", "start", 2), ("b", "end", 5), ("a", "end", 6)]
inc, exc = function_times(logs)
assert inc == {"a": 7, "b": 4}
assert exc == {"a": 3, "b": 4}

recursive = [("f", "start", 0), ("f", "start", 1), ("f", "end", 2), ("f", "end", 3)]
inc, exc = function_times(recursive)
assert inc == {"f": 4}                            # not 4 + 2: the inner call is already inside the outer
assert exc == {"f": 4}                            # exclusive counts every tick exactly once
```

Pin down the convention before coding (is `end` inclusive of its timestamp? — here yes, as in LC 636). The
recursion case is what separates a careful answer: inclusive time must not double count nested calls of the
same function, while exclusive time naturally does not.

## 11. Alert Monitor: Rolling Window and Spike Detection (low)

Card: an `AlertMonitor` over an append-only stream with three queries: alerts in the trailing 15 minutes, the
severity histogram for the current hour-block, and the next-greater-volume index per minute (monotonic stack).

```python
from collections import deque, Counter


class AlertMonitor:
    WINDOW = 15 * 60
    HOUR = 3600

    def __init__(self):
        self.recent = deque()                  # timestamps inside the trailing window
        self.by_hour = {}                      # hour index -> Counter of severities
        self.per_minute = Counter()            # minute index -> count

    def record(self, ts, severity):
        self.recent.append(ts)
        self.by_hour.setdefault(ts // self.HOUR, Counter())[severity] += 1
        self.per_minute[ts // 60] += 1

    def count_last_15m(self, now):
        while self.recent and self.recent[0] <= now - self.WINDOW:
            self.recent.popleft()              # amortised O(1): each timestamp leaves once
        return len(self.recent)

    def severity_histogram(self, now):
        return dict(self.by_hour.get(now // self.HOUR, {}))

    def next_greater_minute(self):
        """For each minute with data, the next minute with strictly more alerts (or -1)."""
        minutes = sorted(self.per_minute)
        answer = {m: -1 for m in minutes}
        stack = []                             # minutes whose answer is still unknown, counts decreasing
        for m in minutes:
            while stack and self.per_minute[stack[-1]] < self.per_minute[m]:
                answer[stack.pop()] = m
            stack.append(m)
        return answer


mon = AlertMonitor()
for ts, sev in [(10, "warn"), (70, "crit"), (75, "warn"), (130, "warn")]:
    mon.record(ts, sev)
assert mon.count_last_15m(now=130) == 4
assert mon.severity_histogram(now=100) == {"warn": 3, "crit": 1}
mon.record(4000, "crit")
assert mon.count_last_15m(now=4000) == 1               # old entries aged out
assert mon.severity_histogram(now=4000) == {"crit": 1}
assert mon.next_greater_minute() == {0: 1, 1: -1, 2: -1, 66: -1}   # minute 1 (two alerts) is the peak

spiky = AlertMonitor()
for ts in [0, 60, 60, 120, 180, 180, 180]:             # minute counts: 1, 2, 1, 3
    spiky.record(ts, "warn")
assert spiky.next_greater_minute() == {0: 1, 1: 3, 2: 3, 3: -1}
```

The point of the round: each query gets the structure that fits it — a deque for the sliding window, a
counter keyed by hour for the histogram, a monotonic stack for "next greater" — instead of forcing one
structure to answer all three. If events can arrive out of order, replace the deque with a heap or bucketed
counts per minute.

## 12. Meeting Scheduler: Earliest Available Slot (low)

Card: given existing meetings and a new meeting's earliest start and duration, return the earliest start that
fits. Delivered as the AI-coding round: use the assistant for boilerplate, keep the data-structure choice
(sorted-by-start + binary search vs segment tree) under your control.

Keep meetings as non-overlapping sorted intervals; binary-search to the first meeting that could conflict and
walk forward through the gaps.

```python
import bisect


class Calendar:
    def __init__(self):
        self.starts, self.ends = [], []          # parallel, sorted, non-overlapping

    def book(self, start, end):
        i = bisect.bisect_left(self.starts, start)
        assert (i == 0 or self.ends[i - 1] <= start) and (i == len(self.starts) or end <= self.starts[i]), \
            "overlap"
        self.starts.insert(i, start)
        self.ends.insert(i, end)

    def earliest_slot(self, not_before, duration):
        t = not_before
        i = bisect.bisect_right(self.ends, t)    # first meeting ending after t could block us
        while i < len(self.starts):
            if t + duration <= self.starts[i]:
                return t                         # fits in the gap before meeting i
            t = max(t, self.ends[i])
            i += 1
        return t                                 # after the last meeting (or empty calendar)


cal = Calendar()
cal.book(9, 10)
cal.book(10, 11)
cal.book(13, 14)
assert cal.earliest_slot(not_before=9, duration=1) == 11     # back-to-back block is skipped
assert cal.earliest_slot(not_before=9, duration=3) == 14     # the 11-13 gap is only 2 long
assert cal.earliest_slot(not_before=11, duration=2) == 11    # fits exactly in 11-13
assert cal.earliest_slot(not_before=8, duration=1) == 8      # before everything
assert Calendar().earliest_slot(5, 4) == 5
cal.book(11, 12)
assert cal.earliest_slot(not_before=9, duration=1) == 12
try:
    cal.book(9, 10)
    raise AssertionError("expected overlap")
except AssertionError as error:
    assert str(error) == "overlap"
```

Complexity: `bisect` is O(log n), the walk is O(meetings it skips), `list.insert` is O(n) — mention a balanced tree
or segment tree if bookings are heavy. Concurrency follow-up: two requests may pick the same slot, so make
`book` a conditional write (check-and-insert under a lock, or a unique constraint on the slot in a database).

---

## 13. System Design — Metrics and Monitoring Platform (medium)

Card: ingest production telemetry, support queries at arbitrary granularity, power a visualisation layer;
recurring twists are top-k exceptions by error class, severity histograms over time, and an explicit
**no-alerting** scope.

**Shape of the answer** [General]:

1. **Scope first:** say aloud what is out — alerting, paging, anomaly detection — then list the queries: time
   series by label set, top-k error classes in a window, histograms by severity.
2. **Ingest path:** agents/SDKs batch and push (or a collector scrapes) to a stateless gateway → a log
   (Kafka) partitioned by series id → stream aggregators that pre-aggregate to 10s/1m buckets → storage.
   Backpressure and sampling for cardinality blowups.
3. **Storage:** a time-series store with columnar blocks, delta-of-delta timestamp and XOR value compression
   (Gorilla-style); **tiered retention** (raw 7 days, 1m rollups 30 days, 1h rollups a year) so old data is
   cheap and coarse. Index labels with an inverted index; enforce a cardinality limit per metric.
4. **Query path:** a query service that picks the coarsest rollup meeting the requested granularity, fans out
   by shard and time range, merges partial results, and caches common dashboards. Top-k uses per-shard top-k
   with an error bound (Space-Saving / Count-Min sketch) and a merge step — exact top-k needs all counts.
5. **Histograms:** store fixed buckets or mergeable sketches (t-digest / DDSketch) so quantiles aggregate
   correctly across hosts; never average percentiles.
6. **Reliability and cost:** multi-AZ replication, late-data handling (watermark then accept corrections),
   and per-tenant quotas.

Trade-off to name: pre-aggregation is cheap to query but loses the ability to slice by a label you did not
roll up; keep raw samples for a short window to cover that.

## 14. System Design — Distributed Job Scheduler / Calendar Service (medium)

Card: millions of scheduled tasks, range queries for upcoming tasks, real-time dashboard updates; variants
include timezones and recurring events.

**Shape of the answer** [General]:

- **Model:** `job(job_id, owner, run_at_utc, payload, status, attempt, lease_until)`; recurring events store the
  recurrence rule (RRULE + timezone id) and **materialise** instances for a rolling horizon (e.g. next 90 days)
  so range queries are plain index scans. Store times in UTC plus the original zone — DST is resolved when
  expanding the rule, not when storing.
- **Storage and sharding:** partition by time bucket and/or `job_id` hash; the hot index is `(bucket,
  run_at)`. A calendar view is a range scan on `(owner, start_time)`, a different index from the dispatcher's.
- **Dispatch:** a timing layer (hierarchical timing wheel, or per-shard "next due" poll with `SELECT … FOR
  UPDATE SKIP LOCKED`) moves due jobs onto a queue; workers take a **lease** and heartbeat. A lost lease
  returns the job to the queue (at-least-once), so handlers must be idempotent.
- **Reliability:** retries with backoff and a dead-letter state; leader election (or consistent hashing of
  shards to schedulers) so one shard has one active dispatcher; catch-up policy after downtime
  (run-missed vs skip) per job type.
- **Dashboards:** change events published to a stream, fanned out over WebSocket/SSE to subscribed viewers;
  clients reconcile with a version number and refetch on gaps.
- **Scale numbers:** state the dispatch rate you design for, the lookahead window, and the hot-minute spike
  (many jobs at 09:00 sharp) — spread with jitter where semantics allow.

## 15. System Design — Type-Ahead / Autocomplete (low)

Card: top-K completions for a prefix in under 100 ms; follow-ups on personalisation, multi-region freshness and
tail latency when the trie spans hundreds of millions of phrases.

**Shape of the answer** [General]:

- **Data structure:** a trie (or FST / sorted array with prefix ranges) where each node stores its precomputed
  top-K by score, so a query is O(|prefix|) with no subtree scan. A tested miniature of this is in the Anthropic
  file's hashtag autocomplete problem.
- **Build pipeline:** offline job aggregates query logs → score (frequency × recency decay × quality filters,
  remove blocked terms) → build the trie → ship an immutable snapshot to serving nodes; deploy by swapping a
  pointer. Hot updates (trending queries) go into a small mutable overlay merged at query time.
- **Serving:** shard by first 1–2 characters or by hash of the prefix; replicate for availability; cache the top
  prefixes at the edge (most traffic is a few thousand short prefixes). Return as the user types with request
  debouncing and cancelation on the client.
- **Personalisation:** merge global candidates with a per-user/session source (recent searches, connections,
  company graph for LinkedIn) and rerank with a light model; keep it under the latency budget by capping
  candidates.
- **Tail latency and freshness:** hedged requests, per-shard timeouts with partial results, and multi-region
  snapshot distribution so staleness is bounded by the build cadence.

## 16. Behavioral Pack and Project Deep-Dive (very high)

Card: the behavioral pack (24 reported discussions, last asked 2026-10-30) is explicit about **leadership
narrative depth** — interviewers flag when a Staff story reads like Senior-IC work. The project deep-dive is
graded on four axes: **Leadership, Craftsmanship, Execution, Impact**, with an architecture diagram you lead the
interviewer through.

Preparation, drawn from the cards [General]:

- Bring two or three **multi-quarter, cross-functional** stories with quantified business impact for each of
  the common themes (conflict, ambiguity, influence without authority, a failure and what changed).
- For the deep-dive, pre-draw the architecture, explain the decision points you owned (what you chose and what
  you rejected), the numbers before and after, and what you would do differently.
- For Staff level, show scope: how you raised the bar for other teams, not only what you built.
- See [`behavioral_interview.md`](./behavioral_interview.md) for the STAR scaffolding.

---

*Total: 16 questions (12 coding, 3 system design, 1 behavioral)*
*Last updated: 2026-11-06*
