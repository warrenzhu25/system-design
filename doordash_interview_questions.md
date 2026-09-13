# DoorDash Interview Questions

---

## Contents

**Coding**
1. [Dasher Pay Calculation (Event Stream, Peak Hours, In-Store Wait)](#1-dasher-pay-calculation-event-stream-peak-hours-in-store-wait)
2. [Random Dasher Picker (Debug + O(1) Weighted Pick)](#2-random-dasher-picker-debug--o1-weighted-pick)
3. [Minimum Deletions to Make a Parentheses String Valid](#3-minimum-deletions-to-make-a-parentheses-string-valid)
4. [First Unique Restaurant ID in an Order Stream](#4-first-unique-restaurant-id-in-an-order-stream)
5. [Minimum Rider Processing Speed](#5-minimum-rider-processing-speed)
6. [Filter Open Restaurants Within a City Range](#6-filter-open-restaurants-within-a-city-range)
7. [Bootstrap API: Aggregate User, Payment, and Address Services with Retries](#7-bootstrap-api-aggregate-user-payment-and-address-services-with-retries)
8. [Refund Workflow Engine on a DAG](#8-refund-workflow-engine-on-a-dag)
9. [Refund Decision Tree Evaluation](#9-refund-decision-tree-evaluation)
10. [Nearest DashMart: Multi-Source BFS on a Grid](#10-nearest-dashmart-multi-source-bfs-on-a-grid)
11. [Order Assignment: Round Robin & Consistent Hashing](#11-order-assignment-round-robin--consistent-hashing)
12. [Order Batching](#12-order-batching)
13. [Validate Shopping Cart](#13-validate-shopping-cart)
14. [Maximize Total Profit by Assigning Chefs to Dishes](#14-maximize-total-profit-by-assigning-chefs-to-dishes)

**System Design**
15. [Restaurant & Dish Ranking / Search](#15-system-design--restaurant--dish-ranking--search)
16. [Idempotent Payment & Refund Processing](#16-system-design--idempotent-payment--refund-processing)
17. [Real-Time Order Status Notifications (Fan-Out)](#17-system-design--real-time-order-status-notifications-fan-out)

**Behavioral**
18. [Behavioral Themes](#18-behavioral-themes)

---

## 1. Dasher Pay Calculation (Event Stream, Peak Hours, In-Store Wait)

**Problem Statement:**
This is DoorDash's signature "Code Craft" prompt — it recurs across nearly every onsite loop as a base
problem with incremental follow-ups added live. You're given a chronological stream of events for a
single dasher's shift. Each event has an `order_id`, an event `type`, and a `timestamp`:

- `ACCEPTED` — dasher accepted the order
- `ARRIVED` — dasher arrived at the restaurant
- `PICKED_UP` — dasher picked up the food (ends the in-store wait, starts the drive-to-customer leg)
- `DELIVERED` (sometimes `FULFILLED`) — order delivered to the customer
- `CANCELED` — order canceled at any point after `ACCEPTED`

Compute the dasher's total pay for the shift.

**Requirements (built up across follow-ups, in the order interviewers typically ask):**
1. **Base pay**: pay `base_rate` per minute of "active time" per order, where active time runs from
   `ACCEPTED` to `DELIVERED`/`CANCELED`.
2. **Peak-hour multiplier**: any portion of active time that falls inside a given peak window is paid
   at `base_rate * peak_multiplier` instead of `base_rate`. Split a single interval proportionally if it
   straddles a peak window boundary.
3. **In-store wait pay** (`ARRIVED` → `PICKED_UP` is a mutually exclusive state): this leg is paid at a
   separate `wait_rate` (usually higher, since the dasher can't multi-app while waiting inside a
   restaurant). A second `ARRIVED` for the same order before a `PICKED_UP` is a data error — reject it.
4. **Cancellations**: if an order is `CANCELED` before delivery, pay for whatever active time elapsed
   up to the cancellation (from `ACCEPTED`, or from `PICKED_UP`/`ARRIVED` if further along).
5. **Double-pay windows (multi-apping)**: if a dasher is juggling two orders at once, the overlapping
   wall-clock time should be paid **once per order**, not deduplicated across orders — i.e., don't merge
   intervals belonging to different `order_id`s.

**Example:**
```
09:58 ACCEPTED   order=A1
10:00 ARRIVED    order=A1
10:05 PICKED_UP  order=A1
10:20 DELIVERED  order=A1
```
With `base_rate=0.50/min`, `wait_rate=0.75/min`, peak window `10:00–10:15` at `2x`:
- `09:58→10:00` (2 min, drive, off-peak): `2 * 0.50 = 1.00`
- `10:00→10:05` (5 min, wait, all peak): `5 * 0.75 * 2 = 7.50`
- `10:05→10:20` (15 min, drive, 10 min peak / 5 min off-peak): `10*0.50*2 + 5*0.50 = 10.00 + 2.50 = 12.50`
- **Total: 21.00**

**Test Cases:**

| Scenario | Expectation |
|---|---|
| Simple accept→deliver, no peak, no wait | `pay = minutes(accepted, delivered) * base_rate` |
| Order canceled 3 min after `ACCEPTED` (never arrived) | `pay = 3 * base_rate` |
| Interval spans a peak window boundary | Split proportionally; only the overlapping minutes get the multiplier |
| Two orders overlap 10:00–10:10 (multi-apping) | Both orders' intervals are paid in full for that window (not halved) |
| Second `ARRIVED` for an order with no intervening `PICKED_UP` | Raise a validation error |
| `PICKED_UP` with no prior `ARRIVED` | Raise a validation error |

**Key Insights:**
1. Model pay as a list of `(start, end, kind)` intervals — one per (order, leg) — then price each interval
   independently. This is what makes overlapping orders "just work" for double-pay: you never merge
   intervals across orders.
2. `ARRIVED`/`PICKED_UP` exclusivity is state tracked per open order (`arrived_at` is set on `ARRIVED` and
   cleared on `PICKED_UP`); a second `ARRIVED` while it's already set is the bug DoorDash is testing for.
3. Peak-hour splitting is just interval overlap: `overlap = max(0, min(end, peak_end) - max(start, peak_start))`.
   No need for a sweep-line — the number of peak windows in a shift is small.
4. Keep `_price_interval` as a pure function of `(start, end, kind)` — every later follow-up (peak hours,
   cancellations, double-pay) is naturally satisfied by *not* adding special cases into it.

**Python Solution:**
```python
from dataclasses import dataclass
from datetime import datetime


@dataclass
class Event:
    order_id: str
    type: str  # ACCEPTED, ARRIVED, PICKED_UP, DELIVERED/FULFILLED, CANCELED
    timestamp: datetime


class DasherPayCalculator:
    """
    Time:  O(n log n) to sort events, O(n) to process, O(w) per interval to price
           against w peak windows.
    Space: O(k) for k orders open at once.
    """

    def __init__(self, base_rate=0.50, wait_rate=0.75,
                 peak_windows=None, peak_multiplier=2.0):
        self.base_rate = base_rate
        self.wait_rate = wait_rate
        self.peak_windows = peak_windows or []   # list[(datetime, datetime)]
        self.peak_multiplier = peak_multiplier

    def compute(self, events: list[Event]) -> float:
        events = sorted(events, key=lambda e: e.timestamp)
        open_orders: dict[str, dict] = {}
        intervals: list[tuple[datetime, datetime, str]] = []

        for e in events:
            order = open_orders.setdefault(e.order_id, {})

            if e.type == "ACCEPTED":
                order["accepted"] = e.timestamp

            elif e.type == "ARRIVED":
                if order.get("arrived_at") is not None:
                    raise ValueError(f"{e.order_id}: duplicate ARRIVED before PICKED_UP")
                order["arrived_at"] = e.timestamp
                intervals.append((order["accepted"], e.timestamp, "drive"))

            elif e.type == "PICKED_UP":
                if order.get("arrived_at") is None:
                    raise ValueError(f"{e.order_id}: PICKED_UP without ARRIVED")
                intervals.append((order["arrived_at"], e.timestamp, "wait"))
                order["picked_up_at"] = e.timestamp
                order["arrived_at"] = None  # exclusivity: leg consumed

            elif e.type in ("DELIVERED", "FULFILLED"):
                start = order.get("picked_up_at") or order.get("arrived_at") or order["accepted"]
                intervals.append((start, e.timestamp, "drive"))
                del open_orders[e.order_id]

            elif e.type == "CANCELED":
                start = order.get("picked_up_at") or order.get("arrived_at") or order["accepted"]
                intervals.append((start, e.timestamp, "drive"))
                del open_orders[e.order_id]

        return round(sum(self._price_interval(*iv) for iv in intervals), 2)

    def _price_interval(self, start: datetime, end: datetime, kind: str) -> float:
        rate = self.wait_rate if kind == "wait" else self.base_rate
        total_minutes = (end - start).total_seconds() / 60

        peak_minutes = 0.0
        for p_start, p_end in self.peak_windows:
            overlap = (min(end, p_end) - max(start, p_start)).total_seconds() / 60
            peak_minutes += max(0.0, overlap)
        peak_minutes = min(peak_minutes, total_minutes)

        normal_minutes = total_minutes - peak_minutes
        return normal_minutes * rate + peak_minutes * rate * self.peak_multiplier
```

**Follow-Up Questions:**
1. What if events for different orders can arrive out of order (clock skew across services)? →
   sort defensively but flag/clip negative-duration intervals rather than trusting the stream blindly.
2. How would you make this incremental (pay running total as events stream in, not batch at end of day)? →
   keep `open_orders` as durable state; price and emit each interval as soon as it closes.
3. How do you unit test the peak-window splitting in isolation from the event state machine? →
   expose `_price_interval` as pure/testable, feed it interval fixtures directly.

---

## 2. Random Dasher Picker (Debug + O(1) Weighted Pick)

**Problem Statement:**
Part A ("Debug Random Dasher Allocation" / "Debug a Random Dasher Picker"): you're handed a function
that assigns an incoming order to a random available dasher, and told "orders keep going to the same
handful of dashers, and it occasionally crashes." Find and fix the bugs.

```python
import random

def assign_dasher(order_id, available_dashers):
    random.seed(order_id % 100)                       # BUG 1
    idx = random.randint(0, len(available_dashers))    # BUG 2
    return available_dashers[idx]
```

- **Bug 1**: reseeding the global RNG from `order_id % 100` collapses entropy to 100 buckets — every
  order whose id shares the same `% 100` residue draws the *identical* pseudo-random sequence, so it
  always lands on the same relative index regardless of who's actually in the pool that moment. This is
  what "always goes to the same dashers" is describing — it isn't random at all.
- **Bug 2**: `random.randint(a, b)` is inclusive on both ends, so `randint(0, len(available_dashers))`
  can return `len(available_dashers)`, which is out of bounds → `IndexError`. This is the crash.
- Also missing: no guard for an empty `available_dashers` list.

**Fix:**
```python
import random

def assign_dasher(order_id, available_dashers):
    if not available_dashers:
        raise ValueError("no dashers available")
    idx = random.randrange(len(available_dashers))   # or random.choice(available_dashers)
    return available_dashers[idx]
```
Never reseed a shared global RNG per call on caller-controlled input — seed once (or not at all) at
process start, or use a local `random.Random()` instance if determinism-for-testing is actually needed.

Part B ("Implement and Debug a Random Dasher Picker"): the follow-up asks for a data structure that
supports dashers going online/offline in **O(1)**, with a **uniform** random pick in **O(1)** — a naive
`list.remove()` is O(n) and shifts indices, breaking uniformity if done carelessly.

**Key Insights:**
1. To delete by value from a list in O(1), swap the target with the last element, then `pop()` — this
   avoids the O(n) shift, but requires a value→index map to find the target first.
2. Keep the index map in sync on every swap, not just on insert/remove.

**Python Solution:**
```python
import random


class DasherPool:
    """
    add/remove/pick_random all O(1) expected.
    Time:  O(1) per operation
    Space: O(n) for n dashers currently in the pool
    """

    def __init__(self):
        self._dashers: list[str] = []
        self._index: dict[str, int] = {}

    def add_dasher(self, dasher_id: str) -> None:
        if dasher_id in self._index:
            return
        self._index[dasher_id] = len(self._dashers)
        self._dashers.append(dasher_id)

    def remove_dasher(self, dasher_id: str) -> None:
        if dasher_id not in self._index:
            return
        idx = self._index[dasher_id]
        last_id = self._dashers[-1]

        self._dashers[idx] = last_id
        self._index[last_id] = idx

        self._dashers.pop()
        del self._index[dasher_id]

    def pick_random(self) -> str:
        if not self._dashers:
            raise ValueError("no dashers available")
        return random.choice(self._dashers)
```

**Follow-Up Questions:**
1. How would you make `pick_random` *weighted* (e.g., by acceptance rate or proximity)? →
   Fenwick tree keyed by cumulative weight, binary-search a random draw in `O(log n)`; add/remove become
   `O(log n)` point updates instead of O(1).
2. How do you unit test randomness without flaky assertions? → seed a local `random.Random` in tests,
   or assert statistical properties (distribution over many trials) rather than exact output.

---

## 3. Minimum Deletions to Make a Parentheses String Valid

**Problem Statement:**
Given a string `s` of `'('`, `')'`, and other characters, return the minimum number of parentheses to
remove so the remaining parentheses are balanced (every `'('` has a matching `')'` and vice versa; other
characters are untouched and don't count).

**Example:**
```
Input:  "a)b(c)d)"
Output: 2   # remove the leading ")" and one of the unmatched "("/")" — e.g. "ab(c)d"
```

**Test Cases:**

| Input | Output |
|-------|--------|
| `"lee(t(c)o)de)"` | `1` |
| `"a)b(c)d"` | `1` |
| `"))(("` | `4` |
| `"(a(b(c)d)"` | `1` |
| `""` | `0` |

**Key Insights:**
1. Single left-to-right pass with a counter for unmatched `'('`; any `')'` seen with counter at 0 is
   immediately unmatched (increment deletions).
2. Whatever `'('` remain unmatched at the end (counter value) are also deletions.
3. No stack of characters needed — just a counter, since we only need the *count*, not the positions
   (positions are trivial to recover the same way if the follow-up asks for the actual resulting string).

**Python Solution:**
```python
def min_deletions_to_valid(s: str) -> int:
    """
    Time:  O(n)
    Space: O(1)
    """
    deletions = 0
    open_count = 0

    for ch in s:
        if ch == "(":
            open_count += 1
        elif ch == ")":
            if open_count > 0:
                open_count -= 1
            else:
                deletions += 1  # unmatched ')'

    return deletions + open_count  # + unmatched '(' left over
```

**Follow-Up Questions:**
1. Return the resulting valid string, not just the count → re-run the same pass, marking indices to
   drop (a `set` of unmatched `')'` indices found greedily, plus the last `open_count` unmatched `'('`
   indices), then filter.
2. What if brackets can also be `{}`/`[]` and must nest correctly (not just parens)? → this becomes
   full bracket-matching with a type-aware stack (LeetCode 921/1249 combined with 20).

---

## 4. First Unique Restaurant ID in an Order Stream

**Problem Statement:**
Orders arrive one at a time as a stream of `restaurant_id`s. Support two operations, both O(1)
amortized:
- `add(restaurant_id)` — record that an order came in for this restaurant.
- `first_unique()` — return the restaurant_id that has appeared **exactly once** so far, among the
  earliest such id (i.e., the oldest one still unique); return `None` if there isn't one.

This is the streaming/"online" variant of "first unique character in a string" (LeetCode 387), matching
LeetCode 1429's structure but keyed on restaurant IDs instead of array values.

**Example:**
```
add("r1"); add("r2"); add("r1")
first_unique() -> "r2"     # r1 now has count 2, r2 still unique
add("r2")
first_unique() -> None     # both now have count 2
```

**Test Cases:**

| Operations | `first_unique()` result |
|---|---|
| `add(r1)` | `r1` |
| `add(r1); add(r1)` | `None` |
| `add(r1); add(r2); add(r1)` | `r2` |
| `add(r1); add(r2); add(r1); add(r3); add(r2)` | `r3` |

**Key Insights:**
1. Maintain a `count` map plus a doubly-linked list (Python: `OrderedDict`) of currently-unique ids in
   insertion order.
2. On `add`: if the id was unique (count was 1), remove it from the ordered structure; increment count;
   if new (count becomes 1), append to the ordered structure.
3. `first_unique()` is then just "peek the first key of the ordered dict" — O(1) — since we prune
   non-unique ids eagerly rather than scanning on query.

**Python Solution:**
```python
from collections import OrderedDict, defaultdict


class FirstUniqueRestaurant:
    """
    add():          O(1) amortized
    first_unique(): O(1)
    Space: O(k) distinct restaurant ids seen
    """

    def __init__(self):
        self._counts: dict[str, int] = defaultdict(int)
        self._unique_order: "OrderedDict[str, None]" = OrderedDict()

    def add(self, restaurant_id: str) -> None:
        self._counts[restaurant_id] += 1
        if self._counts[restaurant_id] == 1:
            self._unique_order[restaurant_id] = None
        elif restaurant_id in self._unique_order:
            del self._unique_order[restaurant_id]

    def first_unique(self) -> str | None:
        if not self._unique_order:
            return None
        return next(iter(self._unique_order))
```

**Follow-Up Questions:**
1. Support `remove(restaurant_id)` (an order gets canceled/refunded and shouldn't count) → decrement
   count; if it drops to 1, re-insert into `_unique_order` at the *end* (approximation — true recency
   ordering on decrement needs an actual doubly linked list, since `OrderedDict.move_to_end` changes
   position semantics you'd need to think through with the interviewer).
2. Scale to a distributed stream (multiple ingestion hosts) → shard by `restaurant_id` hash so all
   events for one id land on one shard; each shard keeps local state; unique-ness is then a per-shard
   property, no cross-shard coordination needed.

---

## 5. Minimum Rider Processing Speed

**Problem Statement:**
A rider (or kitchen) has a list of `prep_times[i]` (minutes required to prepare pile `i` of orders) and
must clear all piles within `h` hours. At an integer processing `speed` (orders' worth of prep-minutes
handled per hour... concretely: at speed `k`, pile `i` takes `ceil(prep_times[i] / k)` hours, and a pile
must be finished before starting the next — same shape as LeetCode 875 "Koko Eating Bananas," reframed
around dasher/kitchen throughput). Return the minimum integer `speed` such that all piles finish within
`h` hours.

**Example:**
```
prep_times = [3, 6, 7, 11], h = 8
Output: 4
```

**Test Cases:**

| prep_times | h | speed |
|---|---|---|
| `[3,6,7,11]` | `8` | `4` |
| `[30,11,23,4,20]` | `5` | `30` |
| `[30,11,23,4,20]` | `6` | `23` |
| `[1,1,1]` | `3` | `1` |

**Key Insights:**
1. Feasibility is monotonic in `speed`: if `speed` works, any faster speed also works → binary search
   the answer instead of trying every speed linearly.
2. Search space is `[1, max(prep_times)]` — any speed ≥ `max(prep_times)` finishes every pile in 1 hour.
3. `feasible(speed) = sum(ceil(t / speed) for t in prep_times) <= h`.

**Python Solution:**
```python
import math


def min_processing_speed(prep_times: list[int], h: int) -> int:
    """
    Time:  O(n log(max(prep_times)))
    Space: O(1)
    """
    def hours_needed(speed: int) -> int:
        return sum(math.ceil(t / speed) for t in prep_times)

    lo, hi = 1, max(prep_times)
    while lo < hi:
        mid = (lo + hi) // 2
        if hours_needed(mid) <= h:
            hi = mid       # mid works, look for something smaller (or equal)
        else:
            lo = mid + 1   # too slow, need more speed
    return lo
```

**Follow-Up Questions:**
1. What if `h` can be fractional (partial hours allowed mid-pile)? → the ceiling drops out; feasibility
   becomes `sum(t / speed) <= h`, solvable directly without search, or with binary search on a real-valued
   speed to a tolerance.
2. Multiple riders working in parallel → this becomes a bin-packing/scheduling variant; binary search on
   speed still applies, but `feasible()` becomes "can these piles be partitioned across `r` riders such
   that each rider's own piles fit in `h` hours at this speed," itself a greedy/DP subproblem.

---

## 6. Filter Open Restaurants Within a City Range

**Problem Statement:**
Given a list of restaurants, each with an `x` coordinate along a single delivery corridor and an
`(open_time, close_time)` daily window, answer queries of the form: "at time `t`, which restaurants
with `x` in `[x_min, x_max]` are open?" Design for many queries against a fixed restaurant list.

**Example:**
```
restaurants = [
  ("A", x=2,  open=9,  close=21),
  ("B", x=5,  open=11, close=15),
  ("C", x=9,  open=0,  close=24),
]
query(t=12, x_min=0, x_max=6) -> ["A", "B"]
query(t=16, x_min=0, x_max=6) -> ["A"]         # B already closed
```

**Test Cases:**

| Query | Result |
|---|---|
| `t=12, [0,6]` | `["A","B"]` |
| `t=16, [0,6]` | `["A"]` |
| `t=12, [8,10]` | `["C"]` |
| `t=23, [0,10]` | `["C"]` |
| `t=8, [0,10]` (before A opens) | `[]` |

**Key Insights:**
1. Preprocess once: sort restaurants by `x`. A range query on `x` then becomes two `bisect` calls →
   `O(log n)` to find the candidate slice instead of scanning all restaurants per query.
2. Within the candidate slice, checking `open_time <= t < close_time` is an `O(k)` scan where `k` is the
   slice size — acceptable since `k` is typically small for a tight geographic range; if it isn't, add a
   secondary interval index (e.g., bucket by hour) on top.
3. Handle overnight windows (`close_time < open_time`, e.g., open 20:00–02:00) as a special case:
   `t >= open_time or t < close_time`.

**Python Solution:**
```python
import bisect
from dataclasses import dataclass


@dataclass
class Restaurant:
    id: str
    x: float
    open_time: float
    close_time: float


class RestaurantFinder:
    """
    Build:  O(n log n)
    Query:  O(log n + k) for k restaurants in the x-range
    Space:  O(n)
    """

    def __init__(self, restaurants: list[Restaurant]):
        self._by_x = sorted(restaurants, key=lambda r: r.x)
        self._xs = [r.x for r in self._by_x]

    def query(self, t: float, x_min: float, x_max: float) -> list[str]:
        lo = bisect.bisect_left(self._xs, x_min)
        hi = bisect.bisect_right(self._xs, x_max)

        return [
            r.id for r in self._by_x[lo:hi]
            if self._is_open(r, t)
        ]

    @staticmethod
    def _is_open(r: Restaurant, t: float) -> bool:
        if r.open_time <= r.close_time:
            return r.open_time <= t < r.close_time
        return t >= r.open_time or t < r.close_time  # overnight window
```

**Follow-Up Questions:**
1. 2D location instead of 1D `x` → replace the sorted-array + bisect with a k-d tree or geohash grid for
   the spatial filter; the open/closed check is unchanged.
2. Very high query volume, restaurant set changes rarely → precompute/cache per-hour bitsets of open
   restaurant ids, intersect with the spatial candidate set at query time.

---

## 7. Bootstrap API: Aggregate User, Payment, and Address Services with Retries

**Problem Statement:**
Implement an endpoint that assembles a "checkout bootstrap" payload by calling three independent
downstream services — User, Payment, Address — and merging their responses into one JSON object. Each
call can fail transiently and should be retried with backoff; a permanently-failing service shouldn't
block the other two from returning (partial results with per-field error markers are acceptable).

**Requirements:**
- Call the three services (mocked as functions that can raise/timeout) concurrently, not sequentially.
- Retry each failing call up to `max_retries` times with exponential backoff.
- If a service still fails after retries, include an explicit error marker for that key rather than
  failing the whole request.
- Must be unit-testable — the retry/backoff logic and the aggregation logic should be separable.

**Test Cases:**

| Scenario | Expectation |
|---|---|
| All three succeed | payload has all three keys populated |
| Payment fails twice then succeeds (within `max_retries`) | payload has `payment` populated, no error |
| Address fails every attempt | payload has `address: {"error": ...}`, `user`/`payment` still populated |
| All three fail | payload has all three keys as errors, call still returns (no exception) |

**Key Insights:**
1. Run the three calls concurrently (`ThreadPoolExecutor` for I/O-bound mocked calls, or `asyncio.gather`
   in an async codebase) — sequential calls would triple the latency for no reason.
2. Wrap retry logic in a small reusable decorator/helper so it's independent of *which* service is being
   called — this is what the interviewer is checking for (don't hardcode three near-identical try/except
   blocks).
3. Isolate failures per-service: one service's exhausted retries must not raise out of the whole
   aggregation — catch at the per-call boundary, not around the whole `gather`.

**Python Solution:**
```python
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable


def call_with_retry(fn: Callable[[], dict], max_retries: int = 3,
                     base_delay: float = 0.1) -> dict:
    last_exc = None
    for attempt in range(max_retries):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 - deliberately broad at the retry boundary
            last_exc = exc
            time.sleep(base_delay * (2 ** attempt))
    return {"error": str(last_exc)}


def bootstrap(user_id: str, fetch_user: Callable, fetch_payment: Callable,
              fetch_address: Callable) -> dict:
    """
    Time:  O(max service latency) instead of O(sum of service latencies)
    Space: O(1) beyond the three payloads
    """
    calls = {
        "user": lambda: fetch_user(user_id),
        "payment": lambda: fetch_payment(user_id),
        "address": lambda: fetch_address(user_id),
    }

    result: dict = {}
    with ThreadPoolExecutor(max_workers=len(calls)) as pool:
        futures = {
            pool.submit(call_with_retry, fn): key
            for key, fn in calls.items()
        }
        for future in as_completed(futures):
            key = futures[future]
            result[key] = future.result()

    return result
```

**Follow-Up Questions:**
1. Add per-call timeouts (a hung downstream shouldn't hang the endpoint) → wrap each call in
   `future.result(timeout=...)`, catching `TimeoutError` and treating it like an exhausted retry.
2. Add a circuit breaker so a service that's down doesn't get hammered with retries from every request →
   track failure rate per service; short-circuit to an immediate error response once open, with a
   half-open probe after a cooldown.
3. Write tests for the retry/backoff logic without real sleeps → inject a fake clock/sleep function, or
   assert on call counts with `base_delay=0`.

---

## 8. Refund Workflow Engine on a DAG

**Problem Statement:**
Build a small local engine that runs refund-processing steps as a DAG. Each step declares which other
steps it depends on; a step only runs once all its dependencies have completed. If a step fails,
everything transitively depending on it is skipped, but unrelated branches still run to completion.

**Example:**
```
validate -> check_fraud -> issue_credit -> notify_customer
validate -> notify_merchant
```
If `check_fraud` fails: `issue_credit` and `notify_customer` are skipped; `notify_merchant` still runs
(it only depends on `validate`, which succeeded).

**Test Cases:**

| Scenario | Expectation |
|---|---|
| All steps succeed | all run exactly once, in a valid topological order |
| Cyclic dependency declared | engine raises at build time, not mid-run |
| Middle step fails | its transitive dependents are marked `SKIPPED`; independent branches still `SUCCEEDED` |
| Independent step raises | doesn't affect sibling branches |

**Key Insights:**
1. This is Kahn's algorithm (BFS topological sort via in-degree counting) plus execution — build the
   graph, validate it's acyclic up front, then process ready nodes (in-degree 0) in waves.
2. Failure propagation is itself a graph traversal from the failed node forward — mark everything
   reachable from a failure as `SKIPPED` before it's ever scheduled.
3. Independent branches can run concurrently (each "wave" of ready nodes with no dependency on each
   other) — worth mentioning even if the base implementation runs them sequentially for simplicity.

**Python Solution:**
```python
from collections import defaultdict, deque
from enum import Enum
from typing import Callable


class Status(Enum):
    PENDING = "PENDING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"


class RefundWorkflowEngine:
    """
    Build:  O(V + E) cycle check
    Run:    O(V + E)
    Space:  O(V + E)
    """

    def __init__(self):
        self._steps: dict[str, Callable[[], None]] = {}
        self._deps: dict[str, list[str]] = defaultdict(list)     # step -> its dependencies
        self._dependents: dict[str, list[str]] = defaultdict(list)  # step -> steps that need it

    def add_step(self, name: str, fn: Callable[[], None], depends_on: list[str] | None = None):
        self._steps[name] = fn
        for dep in depends_on or []:
            self._deps[name].append(dep)
            self._dependents[dep].append(name)

    def run(self) -> dict[str, Status]:
        in_degree = {name: len(self._deps[name]) for name in self._steps}
        status = {name: Status.PENDING for name in self._steps}
        ready = deque([n for n, d in in_degree.items() if d == 0])
        processed = 0

        while ready:
            name = ready.popleft()
            processed += 1
            try:
                self._steps[name]()
                status[name] = Status.SUCCEEDED
            except Exception:
                status[name] = Status.FAILED
                self._skip_descendants(name, status)

            for dependent in self._dependents[name]:
                if status[dependent] == Status.SKIPPED:
                    continue
                in_degree[dependent] -= 1
                if in_degree[dependent] == 0:
                    ready.append(dependent)

        if processed < len(self._steps):
            raise ValueError("cycle detected in refund workflow graph")

        return status

    def _skip_descendants(self, failed: str, status: dict[str, Status]):
        stack = list(self._dependents[failed])
        while stack:
            node = stack.pop()
            if status[node] == Status.SKIPPED:
                continue
            status[node] = Status.SKIPPED
            stack.extend(self._dependents[node])
```

**Follow-Up Questions:**
1. Run independent steps concurrently → schedule each "ready" wave onto a thread/task pool instead of a
   `deque.popleft()` loop; join before advancing in-degree updates for that wave.
2. Add retries per step → wrap `self._steps[name]()` with the same `call_with_retry` helper as Q7.
3. Persist workflow state so a crashed engine can resume → checkpoint `status`/`in_degree` after each
   step completes, keyed by a `workflow_run_id`.

---

## 9. Refund Decision Tree Evaluation

**Problem Statement:**
Given an order and a decision tree of refund policies, evaluate which leaf policy applies. Each internal
node has a `condition(order) -> bool` and two children (`if_true`, `if_false`); each leaf carries a
`policy` (e.g., `{"refund_pct": 100, "reason": "never_arrived"}`).

**Example tree:**
```
root: is order.status == "never_delivered"?
  true  -> leaf: full refund
  false -> is order.reported_within_hours <= 24?
             true  -> is order.item_missing?
                        true  -> leaf: full refund
                        false -> leaf: 50% refund
             false -> leaf: no refund
```

**Test Cases:**

| Order | Resulting policy |
|---|---|
| `status=never_delivered` | full refund |
| `status=delivered, reported=2h, item_missing=True` | full refund |
| `status=delivered, reported=2h, item_missing=False` | 50% refund |
| `status=delivered, reported=48h` | no refund |

**Key Insights:**
1. This is plain binary-tree evaluation, not search — no backtracking, `O(depth)` per order.
2. Keep `condition` as an injected `Callable[[Order], bool]` per node rather than hardcoding field
   comparisons, so the tree is data (can be built from a config/JSON) instead of code.
3. The interesting part interviewers probe: what happens on a malformed tree (leaf missing a policy,
   internal node missing a child) — validate at construction time, not at evaluation time.

**Python Solution:**
```python
from dataclasses import dataclass
from typing import Callable, Optional


@dataclass
class DecisionNode:
    condition: Optional[Callable[[dict], bool]] = None
    if_true: Optional["DecisionNode"] = None
    if_false: Optional["DecisionNode"] = None
    policy: Optional[dict] = None  # set only on leaves

    def is_leaf(self) -> bool:
        return self.policy is not None


def evaluate_refund(order: dict, node: DecisionNode) -> dict:
    """
    Time:  O(depth of tree)
    Space: O(depth) recursion stack (O(1) if converted to a loop)
    """
    while not node.is_leaf():
        node = node.if_true if node.condition(order) else node.if_false
    return node.policy
```

**Follow-Up Questions:**
1. Multiple applicable policies with a priority/most-specific-wins rule → this stops being a simple
   binary tree and becomes rule evaluation with precedence — evaluate all matching leaves, pick by
   explicit priority rather than tree order.
2. Explainability requirement ("why did this order get a 50% refund?") → have `evaluate_refund` also
   return the path of conditions it took, not just the final leaf.

---

## 10. Nearest DashMart: Multi-Source BFS on a Grid

**Problem Statement:**
Given an `m x n` grid where `0` = road, `1` = obstacle/building, `2` = DashMart store, compute for every
road cell the shortest grid distance (4-directional) to the nearest DashMart. Unreachable cells (blocked
by obstacles) get `-1`.

**Example:**
```
grid = [
 [0, 0, 2],
 [1, 1, 0],
 [0, 0, 0],
]
distances = [
 [2, 1, 0],
 [-1, -1, 1],
 [4, 3, 2],
]
```

**Test Cases:**

| Grid | Notes |
|---|---|
| No `2` in grid | every cell is `-1` |
| Single `2`, no obstacles | plain single-source BFS / Manhattan-ish grid distances |
| Multiple `2`s | distance is to the *nearest* one — multi-source BFS handles this without comparing sources pairwise |
| Cell fully walled off by `1`s | `-1` |

**Key Insights:**
1. Don't run BFS from every store separately and take the min (`O(k * m * n)`) — push **all** store
   cells into the queue at distance 0 simultaneously ("multi-source BFS"); the first time a cell is
   visited, it's necessarily from its nearest store, so a single `O(m*n)` pass suffices.
2. Standard BFS grid traversal: visited/distance array doubles as the "seen" check.

**Python Solution:**
```python
from collections import deque


def nearest_dashmart(grid: list[list[int]]) -> list[list[int]]:
    """
    Time:  O(m * n)
    Space: O(m * n)
    """
    if not grid or not grid[0]:
        return grid

    rows, cols = len(grid), len(grid[0])
    dist = [[-1] * cols for _ in range(rows)]
    queue = deque()

    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 2:
                dist[r][c] = 0
                queue.append((r, c))

    while queue:
        r, c = queue.popleft()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if (0 <= nr < rows and 0 <= nc < cols
                    and grid[nr][nc] != 1 and dist[nr][nc] == -1):
                dist[nr][nc] = dist[r][c] + 1
                queue.append((nr, nc))

    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 2:
                dist[r][c] = 0  # stores themselves stay 0, not overwritten by traversal

    return dist
```

**Follow-Up Questions:**
1. Weighted terrain (e.g., some roads slower than others) → multi-source BFS becomes multi-source
   Dijkstra (priority queue keyed by accumulated cost instead of a plain FIFO queue).
2. Grid updates live (a DashMart opens/closes) → full recompute is `O(m*n)` per change, which is fine at
   modest grid sizes; at scale, incrementalizing this requires more machinery than an interview expects
   — worth naming as the tradeoff rather than over-engineering live.

---

## 11. Order Assignment: Round Robin & Consistent Hashing

**Problem Statement:**
Part A: debug a round-robin order dispatcher that's supposed to spread incoming orders evenly across a
pool of matching servers.

```python
class RoundRobinBalancer:
    def __init__(self, servers):
        self.servers = servers
        self.index = 0

    def next_server(self):
        server = self.servers[self.index]
        self.index += 1                 # BUG: never wraps, will IndexError
        return server
```
**Bug**: `self.index` is never taken modulo `len(self.servers)`, so it eventually runs off the end of
the list. It also isn't safe if `servers` is mutated (a server added/removed) between calls — `index`
can end up pointing at the wrong logical position.

**Fix:**
```python
class RoundRobinBalancer:
    def __init__(self, servers):
        self.servers = servers
        self.index = 0

    def next_server(self):
        if not self.servers:
            raise ValueError("no servers configured")
        server = self.servers[self.index % len(self.servers)]
        self.index += 1
        return server
```

Part B: plain round robin reassigns *most* orders to different servers whenever the pool resizes (every
key's `index % len(servers)` shifts). Implement consistent hashing so pool resizes only remap `~1/N` of
keys.

**Key Insights:**
1. Consistent hashing places both servers and keys on a hash ring; a key goes to the first server
   clockwise from its hash position — adding/removing one server only affects the keys between it and
   its neighbor on the ring.
2. Use virtual nodes (multiple ring positions per physical server) to smooth out load when the server
   count is small — otherwise one server can end up owning a disproportionate arc of the ring.
3. `bisect` on a sorted list of ring positions gives `O(log n)` lookup for "first server at or after this
   hash."

**Python Solution:**
```python
import bisect
import hashlib


class ConsistentHashRing:
    """
    add_server/remove_server: O(v log(n*v)) for v virtual nodes, n servers
    get_server:               O(log(n*v))
    """

    def __init__(self, virtual_nodes: int = 100):
        self.virtual_nodes = virtual_nodes
        self._ring: list[int] = []              # sorted hash positions
        self._ring_map: dict[int, str] = {}      # hash position -> server id

    def _hash(self, key: str) -> int:
        return int(hashlib.md5(key.encode()).hexdigest(), 16)

    def add_server(self, server_id: str) -> None:
        for i in range(self.virtual_nodes):
            h = self._hash(f"{server_id}#{i}")
            bisect.insort(self._ring, h)
            self._ring_map[h] = server_id

    def remove_server(self, server_id: str) -> None:
        for i in range(self.virtual_nodes):
            h = self._hash(f"{server_id}#{i}")
            idx = bisect.bisect_left(self._ring, h)
            if idx < len(self._ring) and self._ring[idx] == h:
                self._ring.pop(idx)
                del self._ring_map[h]

    def get_server(self, order_id: str) -> str:
        if not self._ring:
            raise ValueError("no servers in ring")
        h = self._hash(order_id)
        idx = bisect.bisect_right(self._ring, h) % len(self._ring)
        return self._ring_map[self._ring[idx]]
```

**Follow-Up Questions:**
1. How do you verify load is actually balanced after adding virtual nodes? → hash a large sample of
   synthetic order ids, histogram the resulting server assignment, check the spread.
2. Weighted servers (some handle more capacity) → give higher-capacity servers proportionally more
   virtual nodes on the ring.

---

## 12. Order Batching

**Problem Statement:**
Given a list of ready orders, each with a `ready_time` and a `deadline` (must be delivered by), and a
dasher who can carry at most `capacity` orders per trip, greedily group orders into batches (trips) such
that every order in a batch can feasibly be picked up and delivered within its own window, minimizing
the number of trips.

**Simplifying model**: a batch of orders is feasible if there's a common pickup instant that is
`>= max(ready_time)` and `<= min(deadline)` across the batch (i.e., the intersection of all
`[ready_time, deadline]` windows in the batch is non-empty), and the batch has at most `capacity` orders.

**Example:**
```
orders = [(id=1, ready=0, deadline=10),
          (id=2, ready=2, deadline=6),
          (id=3, ready=5, deadline=8),
          (id=4, ready=9, deadline=20)]
capacity = 2

Batches: [1, 2] (intersection [2,6] ok), [3] alone (adding 3 to batch1 -> intersection [5,6] still fits
capacity but batch1 already full at 2), [4] alone
-> 3 trips
```

**Test Cases:**

| Orders (ready, deadline) | capacity | Trips |
|---|---|---|
| `[(0,10),(2,6),(5,8),(9,20)]` | `2` | `3` |
| `[(0,5),(1,4),(2,3)]` | `3` | `1` (intersection `[2,3]` non-empty) |
| `[(0,2),(3,5)]` | `2` | `2` (windows disjoint, can't batch) |
| `[(0,100)] * 5` | `2` | `3` |

**Key Insights:**
1. Sort by `deadline` — classic greedy interval scheduling: always try to extend the *current* batch
   with the next order (by deadline) rather than looking ahead, because committing to the earliest
   deadline first never loses optimality for "minimize number of groups under a window-intersection +
   capacity constraint."
2. Maintain the current batch's running window `[cur_lo, cur_hi]` (intersection so far). An order joins
   if the batch isn't full **and** `max(cur_lo, ready) <= min(cur_hi, deadline)`; otherwise close the
   current batch and start a new one with this order.
3. This is the same family as "minimum number of platforms"/interval partitioning, adapted with a
   capacity cap on group size.

**Python Solution:**
```python
from dataclasses import dataclass


@dataclass
class Order:
    id: str
    ready_time: float
    deadline: float


def batch_orders(orders: list[Order], capacity: int) -> list[list[str]]:
    """
    Time:  O(n log n) for the sort, O(n) to batch
    Space: O(n)
    """
    orders = sorted(orders, key=lambda o: o.deadline)
    batches: list[list[str]] = []

    cur_ids: list[str] = []
    cur_lo, cur_hi = None, None

    for o in orders:
        if cur_ids and len(cur_ids) < capacity:
            new_lo = max(cur_lo, o.ready_time)
            new_hi = min(cur_hi, o.deadline)
            if new_lo <= new_hi:
                cur_ids.append(o.id)
                cur_lo, cur_hi = new_lo, new_hi
                continue

        if cur_ids:
            batches.append(cur_ids)
        cur_ids = [o.id]
        cur_lo, cur_hi = o.ready_time, o.deadline

    if cur_ids:
        batches.append(cur_ids)

    return batches
```

**Follow-Up Questions:**
1. Orders also have distinct restaurant locations — batching should also require the restaurants to be
   within some walking/driving distance of each other → add a spatial feasibility check alongside the
   time-window check before joining a batch.
2. Minimize total lateness instead of number of trips → this changes the objective from a greedy
   interval-partitioning problem to a scheduling optimization that likely needs DP or a different greedy
   rule (e.g., earliest-deadline-first with lateness tracking, à la scheduling theory).

---

## 13. Validate Shopping Cart

**Problem Statement:**
Given a shopping cart (list of `{item_id, quantity, unit_price}`), a `catalog` (valid item ids and
current prices), and an optional applied `coupon`, validate the cart and return a list of validation
errors (empty list = valid).

**Rules:**
- Every `item_id` in the cart must exist in the catalog.
- `quantity` must be a positive integer.
- `unit_price` in the cart must match the catalog's current price (stale client-side price).
- If a coupon is applied: it must not be expired, the cart subtotal must meet its `min_order_value`, and
  it can only be applied once (not stacked, even if the client sends duplicate coupon lines).
- The cart's stated `total` must equal `subtotal - discount`, within floating point tolerance.

**Test Cases:**

| Cart | Result |
|---|---|
| Valid items, correct prices, no coupon, correct total | `[]` |
| Item id not in catalog | `["unknown item: X"]` |
| `quantity = 0` | `["invalid quantity for item X"]` |
| Client price differs from catalog price | `["stale price for item X"]` |
| Coupon subtotal below `min_order_value` | `["coupon min order value not met"]` |
| Coupon applied twice | `["coupon already applied"]` |
| Stated total doesn't match computed total | `["total mismatch"]` |

**Key Insights:**
1. This is a pure validation function — collect *all* applicable errors in one pass rather than
   short-circuiting on the first one, since the caller (a checkout UI) wants to show every problem at once.
2. Compute the subtotal/discount/total from catalog truth, not from client-supplied values, then compare
   against what the client sent — never trust client-side money math.
3. Keep each rule as an independent check function so new rules (e.g., item availability, max quantity)
   can be added without touching existing ones.

**Python Solution:**
```python
from dataclasses import dataclass


@dataclass
class CartItem:
    item_id: str
    quantity: int
    unit_price: float


@dataclass
class Coupon:
    code: str
    discount_pct: float
    min_order_value: float
    expired: bool


def validate_cart(items: list[CartItem], catalog: dict[str, float],
                   stated_total: float, coupon: Coupon | None = None) -> list[str]:
    """
    Time:  O(n) for n cart items
    Space: O(n)
    """
    errors: list[str] = []
    subtotal = 0.0

    for item in items:
        if item.item_id not in catalog:
            errors.append(f"unknown item: {item.item_id}")
            continue
        if item.quantity <= 0:
            errors.append(f"invalid quantity for item {item.item_id}")
            continue
        catalog_price = catalog[item.item_id]
        if abs(item.unit_price - catalog_price) > 1e-6:
            errors.append(f"stale price for item {item.item_id}")
        subtotal += catalog_price * item.quantity

    discount = 0.0
    if coupon is not None:
        if coupon.expired:
            errors.append("coupon expired")
        elif subtotal < coupon.min_order_value:
            errors.append("coupon min order value not met")
        else:
            discount = subtotal * coupon.discount_pct

    computed_total = subtotal - discount
    if abs(computed_total - stated_total) > 1e-6:
        errors.append("total mismatch")

    return errors
```

**Follow-Up Questions:**
1. Support stacking multiple non-exclusive coupons → validate each independently against the *running*
   subtotal-after-previous-discounts, and disallow stacking two coupons flagged as mutually exclusive.
2. Inventory check (item in stock) requires a live lookup with latency → separate "structural" validation
   (this function) from "availability" validation (an async/batched call), since one is pure and one has
   I/O — don't conflate them in one function/test.

---

## 14. Maximize Total Profit by Assigning Chefs to Dishes

**Problem Statement:**
Given `chefs[i]` = skill level of chef `i`, and dishes described by parallel arrays `requirement[j]`
(minimum skill needed) and `profit[j]` (profit if cooked), assign each chef to at most one dish (a chef
can only cook a dish if `chef_skill >= requirement`) to maximize total profit. A chef who can't meet any
dish's requirement earns nothing; dishes can be reused across chefs (unlimited supply of each dish type).
This is exactly LeetCode 826 ("Most Profit Assigning Work") reframed with chefs/dishes.

**Example:**
```
requirement = [2, 4, 6, 8, 10]
profit      = [10, 20, 30, 40, 50]
chefs       = [4, 5, 6, 7]
Output: 100   # chef skills 4,5 -> best profit 20 (req<=4/5); 6,7 -> best profit 30/40
              # 20 + 20 + 30 + 40 = 110... (worked example — see solution for the exact DP)
```

**Test Cases:**

| requirement | profit | chefs | total profit |
|---|---|---|---|
| `[2,4,6,8,10]` | `[10,20,30,40,50]` | `[4,5,6,7]` | `100` |
| `[1,1,1]` | `[1,2,3]` | `[0]` | `0` |
| `[1]` | `[100]` | `[100,100]` | `200` |

**Key Insights:**
1. A chef with skill `s` should take whichever dish has the **best profit among all dishes with
   `requirement <= s`** — not necessarily the dish with the highest individual requirement (a lower-bar
   dish can pay more).
2. Precompute, for each distinct requirement threshold (sorted), the best profit achievable at or below
   that threshold — a running max over dishes sorted by `requirement`. This collapses "best profit
   reachable at skill `s`" into a binary-searchable step function.
3. Sort chefs too, then sweep both sorted arrays with two pointers (`O(n log n + m log m)`) instead of
   binary-searching per chef (`O(m log n)`) — either is acceptable, but the two-pointer sweep is the
   canonical LC 826 answer.

**Python Solution:**
```python
def max_profit_assignment(requirement: list[int], profit: list[int],
                           chefs: list[int]) -> int:
    """
    Time:  O(n log n + m log m) for n dishes, m chefs
    Space: O(n)
    """
    dishes = sorted(zip(requirement, profit))
    chefs = sorted(chefs)

    total = 0
    best_so_far = 0
    i = 0

    for skill in chefs:
        while i < len(dishes) and dishes[i][0] <= skill:
            best_so_far = max(best_so_far, dishes[i][1])
            i += 1
        total += best_so_far

    return total
```

**Follow-Up Questions:**
1. Each dish has limited supply (can only be cooked `k` times total) → this becomes a bipartite
   assignment/flow problem instead of a greedy sweep — greedy no longer guarantees optimality once
   dishes are a scarce resource.
2. One chef, multiple dishes per shift, time-boxed → this turns into a knapsack variant (maximize profit
   subject to a total time budget), not a per-chef independent assignment.

---

## 15. System Design — Restaurant & Dish Ranking / Search

**Problem Statement:**
The recurring system-design favorite on the DoorDash loop. Design the service that ranks and returns
restaurants (and dishes within them) for a consumer's home-feed and search query, given their location,
time of day, and past order history.

**Functional Requirements:**
- Given `(lat, lon, query?)`, return a ranked list of restaurants currently open and deliverable to that
  location.
- Rank by a blend of relevance (query match), predicted conversion/engagement, delivery ETA, and
  business constraints (promoted placements, diversity of cuisine).
- Personalize using the consumer's order history and affinities.
- Freshness: a restaurant that just closed or went out-of-stock on its top items should drop out quickly.

**Non-Functional Requirements:**
- p99 latency budget in the tens of milliseconds for the ranking call itself (feed load time matters a
  lot for conversion).
- Recall/candidate generation must be geographically scoped — never score every restaurant in the country
  per request.
- Ranking model updates shouldn't require redeploying the serving path.

**High-Level Design:**
1. **Candidate generation (retrieval)**: geo-index (e.g., geohash/H3 cells or an R-tree) narrows to
   restaurants deliverable to the consumer's location; a lightweight filter drops closed/out-of-stock
   ones. This stage optimizes for recall over precision — get a few hundred to a few thousand candidates
   cheaply.
2. **Feature fetch**: pull precomputed features in parallel — restaurant embeddings, consumer embeddings,
   real-time signals (current queue/ETA, promo status) — from a low-latency feature store (Redis/key-value)
   rather than computing on the fly.
3. **Ranking (scoring)**: a two-tower model (consumer tower + restaurant tower, dot-product or learned
   similarity) gives a first-pass relevance score cheaply at scale; a heavier re-ranker (GBDT or small
   neural net) re-scores the top-K candidates using richer cross-features (query-restaurant interaction,
   ETA, recency).
4. **Business logic layer**: apply promoted-slot insertion, diversity constraints (don't show 10 pizza
   places in a row), and any experiment/holdout logic, on top of the model-ranked list.
5. **Response**: return the final ordered list; log the full ranking (candidates, scores, final order)
   for offline model training and A/B analysis.

**Data Model (sketch):**
```
restaurants(id, geo_cell, cuisine_tags, is_open, delivery_zones, embedding_id)
restaurant_features(id, avg_rating, avg_eta, conversion_rate_7d, ...)   # feature store, low-latency
consumer_features(id, embedding_id, cuisine_affinities, order_history_summary)
ranking_logs(request_id, consumer_id, candidates[], scores[], final_order[], ts)
```

**Scaling & Reliability:**
- Retrieval and feature-fetch stages are embarrassingly parallel across candidates — fan out, gather.
- Feature store is refreshed asynchronously (streaming/batch jobs update embeddings and aggregates);
  the serving path only ever reads, never computes features inline.
- Fall back to a simpler heuristic ranking (e.g., distance + rating) if the ML ranking service times out
  or errors — never block the feed on the model.
- Cache ranked results per (geo-cell, time-bucket) for anonymous/low-personalization traffic to cut load.

**Follow-Up Questions:**
1. How do you A/B test a new ranking model safely? → shadow traffic first (score but don't serve),
   then a small traffic-split experiment with guardrail metrics (conversion, cancellation rate).
2. A restaurant runs out of a promoted dish mid-session — how fast can the feed reflect that? → depends
   on how "real-time" the out-of-stock signal is piped into the feature store (streaming vs. batch);
   discuss the latency/cost tradeoff explicitly.
3. Cold-start for a brand-new restaurant with no history → fall back to content-based features
   (cuisine, price tier, location) until enough interaction data accumulates for the learned embedding.

---

## 16. System Design — Idempotent Payment & Refund Processing

**Problem Statement:**
Design the payment and refund pipeline for an order: charge the consumer on order placement, and issue
refunds for cancellations/quality issues, such that retries (from client timeouts, network blips,
duplicate webhook deliveries) never double-charge or double-refund.

**Functional Requirements:**
- Charge a consumer's payment method for an order total.
- Issue full or partial refunds against a prior charge.
- Support async payment processor callbacks (webhooks) that may arrive out of order or duplicated.

**Non-Functional Requirements:**
- Exactly-once *effect* despite at-least-once delivery/retries anywhere in the pipeline.
- Auditable: every state transition on a payment/refund must be reconstructable after the fact.
- Payment processor calls are the slow, unreliable part — the system must tolerate their latency/failure
  without blocking order placement indefinitely.

**High-Level Design:**
1. **Idempotency key**: every charge/refund request carries a client-generated idempotency key
   (`order_id` + `attempt_type`, e.g., `charge:order_123`, `refund:order_123:reason_x`). The payment
   service persists `(idempotency_key -> result)` and short-circuits a retry with the exact prior result
   instead of re-executing the side effect.
2. **State machine per payment**: `INITIATED -> AUTHORIZED -> CAPTURED -> REFUND_PENDING -> REFUNDED`
   (plus `FAILED` branches). Transitions are conditional updates (`UPDATE ... WHERE status = expected`)
   so a duplicate/racing request that already sees the wrong state is a no-op, not a double-transition.
3. **Outbox pattern for calling the processor**: write the intended charge/refund and its idempotency key
   to a local "outbox" table in the same transaction as the order state change; a separate worker polls
   the outbox and calls the payment processor's API (which itself accepts an idempotency key), marking
   the outbox row done on success. This avoids the classic "DB commit succeeded, but the network call to
   the processor never happened / happened twice" failure mode.
4. **Webhook handling**: processor callbacks are deduplicated by the processor's own event id
   (`processed_events(event_id)` uniqueness constraint) before being applied to the state machine.
5. **Reconciliation job**: periodically diff the processor's transaction log against internal state to
   catch anything that fell through (crashed worker mid-call, etc.) — treat this as the safety net, not
   the primary correctness mechanism.

**Data Model (sketch):**
```
payments(payment_id PK, order_id, status, amount, idempotency_key UNIQUE, processor_ref, updated_at)
payment_outbox(id, payment_id, action, payload, status, created_at)
processed_events(event_id PK, payment_id, applied_at)   # webhook dedup
refunds(refund_id PK, payment_id, amount, reason, idempotency_key UNIQUE, status)
```

**Scaling & Reliability:**
- Outbox worker(s) scale horizontally with row-level locking (`SELECT ... FOR UPDATE SKIP LOCKED`) to
  avoid double-processing the same outbox row across workers.
- All money fields as integer minor units (cents), never floats, to avoid rounding drift across retries.
- Partial refunds accumulate against the original payment; enforce `sum(refunds.amount) <= payments.amount`
  at the application layer with a transactional check, since it can't be a simple column constraint.

**Follow-Up Questions:**
1. The payment processor call succeeds, but the response is lost before your worker sees it (ack never
   arrives) → this is exactly what the idempotency key protects: the retried call to the processor
   returns the same result rather than charging again.
2. How do you test this without hitting a real payment processor? → a fake processor client with
   injectable failure modes (timeout, duplicate webhook, out-of-order webhook) driven by contract tests.
3. Refund initiated twice from two different code paths (support tooling + automated fraud reversal) at
   the same time → the conditional state transition (`WHERE status = 'CAPTURED'`) plus the refund's own
   idempotency key ensures only one actually executes; the second observes the already-updated state.

---

## 17. System Design — Real-Time Order Status Notifications (Fan-Out)

**Problem Statement:**
Design the system that pushes real-time order status updates (`order placed`, `restaurant confirmed`,
`dasher assigned`, `picked up`, `delivered`) to the consumer's app, the merchant's tablet, and the
dasher's app simultaneously, as soon as each state change happens.

**Functional Requirements:**
- A single state-change event fans out to up to 3 distinct recipients (consumer, merchant, dasher), each
  possibly on a different channel (push notification, in-app websocket, SMS fallback).
- Recipients who are offline at delivery time still see the correct latest state when they reconnect.
- Ordering matters within a single order's timeline (a client must never see "delivered" before "picked
  up" due to out-of-order delivery).

**Non-Functional Requirements:**
- Low fan-out latency (sub-second from state change to push) for the live-tracking experience.
- Must tolerate spiky load (dinner rush) without dropping or reordering events for a given order.
- Multi-channel: connected clients get websocket pushes; disconnected/backgrounded clients get a mobile
  push notification instead.

**High-Level Design:**
1. **Event source**: the order-state service publishes a `state_changed` event
   (`order_id, new_status, ts`) to a pub/sub topic (e.g., Kafka) partitioned by `order_id` — this is what
   guarantees per-order ordering, since all events for one order land on the same partition and are
   consumed in order.
2. **Fan-out worker**: consumes the topic, looks up the three recipients (consumer, merchant, dasher) and
   their current connection state (connected-via-websocket vs. not) from a connection registry, and
   routes the event accordingly.
3. **Connection registry**: a low-latency key-value store (`user_id -> gateway_node`) maintained by the
   websocket gateway tier; a client connects to a gateway node, which registers/heartbeats itself into
   this store and deregisters on disconnect.
4. **Delivery paths**:
   - Connected: fan-out worker publishes to the specific gateway node the client is attached to (via an
     internal pub/sub or direct RPC), which pushes over the open websocket.
   - Disconnected: enqueue a mobile push notification (APNs/FCM) instead; also persist the latest status
     so the client fetches current truth via a REST call on reconnect/app-foreground (never rely on the
     push notification alone as the source of truth).
5. **Idempotent client rendering**: since a client might get both a push notification and, on
   reconnect, a fresh fetch of current state, the client renders off `(order_id, status, ts)` and ignores
   anything older than what it's already showing — this absorbs any duplicate/out-of-order delivery at
   the edge without needing perfect exactly-once delivery end-to-end.

**Data Model (sketch):**
```
order_status_events(order_id, status, ts)         # Kafka topic, partitioned by order_id
connections(user_id, gateway_node_id, connected_at)  # ephemeral, TTL'd
order_latest_status(order_id, status, ts)          # for reconnect/REST fetch
```

**Scaling & Reliability:**
- Partitioning by `order_id` bounds ordering guarantees to what's actually needed (per-order), letting
  overall throughput scale by adding partitions/consumers.
- Gateway tier scales horizontally; the connection registry is the only shared state, kept small and
  ephemeral (TTL-based cleanup handles ungraceful disconnects).
- Push-notification and websocket paths are independent — a push provider outage degrades to
  "connected clients still get live updates," not a total outage.

**Follow-Up Questions:**
1. A dasher's app reconnects after being offline for 10 minutes during a delivery — how does it catch up
   without replaying every intermediate event? → it doesn't need to; it just fetches
   `order_latest_status` on reconnect and resumes live-tailing from there.
2. How would you extend this to power a live map (continuous location pings, not just discrete status
   changes)? → same fan-out shape but much higher event rate and no strict ordering requirement per
   event — throttle/sample at the gateway rather than delivering every ping.
3. What breaks first under 10x dinner-rush load, and how do you know? → most likely the connection
   registry or gateway fan-out RPC path; discuss load-testing the fan-out worker and gateway tier
   independently, and horizontal scaling levers for each.

---

## 18. Behavioral Themes

DoorDash's behavioral round is a standard STAR-format interview but consistently probes a few company
values — see [`behavioral_interview.md`](./behavioral_interview.md) for general STAR-method prep. Themes
specific to DoorDash's loop:

- **Bias for action / urgency under ambiguity**: a time you shipped something with incomplete
  information because waiting would've cost more than a wrong-but-correctable decision.
- **Ownership beyond your ticket**: a time you fixed or flagged something outside your immediate scope
  (e.g., a flaky dependency, a gap in another team's service) because it was blocking the outcome.
- **Getting 1% better / continuous improvement**: a concrete process or code change you made that
  compounded, not just a one-off fix.
- **Working with operational/marketplace tradeoffs**: DoorDash is a three-sided marketplace (consumers,
  merchants, dashers) — be ready to discuss a time a decision helped one side at another's expense, and
  how you reasoned about the tradeoff.

---

## References

Sources used for compiling these questions:
- [DoorDash Interview Questions - 1point3acres](https://www.1point3acres.com/interview/problems/company/doordash)

Note: the source page requires forum membership to view full question text/discussion threads; the
problems above were reconstructed and expanded from the publicly visible question titles into complete,
solvable problem statements with original test cases and solutions.
