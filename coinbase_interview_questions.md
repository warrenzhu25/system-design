# Coinbase Interview Questions

---

## Contents

**Coding**
1. [Banking System: Accounts, Leaderboard, Scheduled Payments, and Merges](#1-banking-system-accounts-leaderboard-scheduled-payments-and-merges)
2. [Mutable Leaderboard with K-th Highest Score](#2-mutable-leaderboard-with-k-th-highest-score)
3. [Moving Average over a Sliding Window](#3-moving-average-over-a-sliding-window)
4. [Minimum Days to Execute Ordered Tasks with Cooldown](#4-minimum-days-to-execute-ordered-tasks-with-cooldown)
5. [Crypto Trading Order Management System (Limit Order Book)](#5-crypto-trading-order-management-system-limit-order-book)
6. [Block Mining with Dependencies: Maximize Fee Under a Block Size Constraint](#6-block-mining-with-dependencies-maximize-fee-under-a-block-size-constraint)
7. [NFT Registry: Core Operations and Weighted Attribute Combination Generation](#7-nft-registry-core-operations-and-weighted-attribute-combination-generation)
8. [In-Memory Key-Value Store with Per-Key User Locks and Top-N Operation Tracking](#8-in-memory-key-value-store-with-per-key-user-locks-and-top-n-operation-tracking)
9. [Drone Delivery Route Simulation with Charging Stations](#9-drone-delivery-route-simulation-with-charging-stations)
10. [Flappy-Bird Physics: Minimum Jumps to Stay Within Bounds](#10-flappy-bird-physics-minimum-jumps-to-stay-within-bounds)
11. [Parse and Group Log Lines by Thread ID](#11-parse-and-group-log-lines-by-thread-id)
12. [Cursor-Based API Pagination with Filtering](#12-cursor-based-api-pagination-with-filtering)

**System Design**
13. [Ledger & Balance Storage with Sharding](#13-system-design--ledger--balance-storage-with-sharding)
14. [Idempotent Trade Settlement Pipeline](#14-system-design--idempotent-trade-settlement-pipeline)
15. [Real-Time Price Ticker Fan-Out (WebSocket)](#15-system-design--real-time-price-ticker-fan-out-websocket)

**Behavioral**
16. [Behavioral Themes](#16-behavioral-themes)

---

## 1. Banking System: Accounts, Leaderboard, Scheduled Payments, and Merges

**Problem Statement:**
This is a multi-level online assessment — the interviewer typically reveals one level at a time, so
don't over-engineer level 1 for requirements you don't have yet. Implement a small bank backend that
grows across four levels:

1. **Basic account operations**: create an account, deposit, and transfer between accounts (rejecting a
   transfer that would overdraw the sender).
2. **Spending leaderboard**: return the top-k accounts by total amount sent (outgoing transfers).
3. **Scheduled / cancellable payments**: schedule a transfer to execute at a future timestamp, allow
   cancelling a scheduled payment before it executes, and process all payments that are now due.
4. **Account merges**: merge two accounts into one (combined balance, combined spending total), including
   any payments still scheduled against the account being absorbed.

**Example:**
```
create_account("A"); create_account("B")
deposit("A", 100)
transfer("A", "B", 30)              # A=70, B=30
top_spenders(1)                     # [("A", 30)]

pid = schedule_payment(execute_ts=100, "A", "B", 20)
run_due_payments(50)                # not due yet, no-op
run_due_payments(100)                # executes -> A=50, B=50

pid2 = schedule_payment(execute_ts=200, "A", "B", 10)
cancel_payment(pid2)
run_due_payments(200)                # cancelled, no-op

create_account("C"); deposit("C", 5)
merge_accounts("A", "C")             # A=55, C removed
```

**Test Cases:**

| Operations | Expected result |
|---|---|
| `create_account(A); deposit(A,100); transfer(A,B,30)` | `balances = {A:70, B:30}` |
| `transfer(A, B, 1000)` (insufficient funds) | raises an error, no state change |
| `schedule_payment(ts=100,...); run_due_payments(50)` | payment still pending, balances unchanged |
| `schedule_payment(ts=100,...); run_due_payments(100)` | payment executes |
| `schedule_payment(ts=200,...); cancel_payment(pid); run_due_payments(200)` | payment does **not** execute |
| `merge_accounts(primary, secondary)` | `secondary`'s balance and spend total roll into `primary`; any payment still scheduled with `secondary` as sender/receiver now references `primary` |

**Key Insights:**
1. **Lazy deletion for cancellation**: a cancelled scheduled payment is marked inactive in place rather
   than removed from the heap — removing an arbitrary element from a heap is `O(n)`; flip a flag and skip
   it on pop instead (`O(1)` cancel, cost paid later at pop time).
2. **Incremental aggregate, not recomputed**: the leaderboard's per-account total is updated on every
   transfer rather than recomputed from a transaction log on each `top_spenders` call — a write-cost vs.
   read-cost tradeoff that's worth naming explicitly.
3. **Merge must touch scheduled state, not just balances**: forgetting to remap `from_id`/`to_id` on
   still-pending scheduled payments is the easiest way to silently lose or misdirect money after a merge —
   this is usually the detail that separates a complete level-4 answer from a partial one.
4. Heap entries are `[execute_ts, payment_id, from_id, to_id, amount, active]` — `payment_id` is strictly
   increasing and unique, so tuple/list comparison never needs to fall through to comparing `from_id`/`to_id`
   (which would break on ties between different string types in other languages).

**Python Solution:**
```python
import heapq
from collections import defaultdict


class BankingSystem:
    """
    create_account/deposit/transfer: O(1)
    top_spenders(k):                 O(n log k) for n accounts with outgoing transfers
    schedule_payment/cancel_payment: O(log n) / O(1) (lazy delete)
    run_due_payments:                O(d log n) for d payments becoming due
    merge_accounts:                  O(s) to remap s currently-scheduled payments
    """

    def __init__(self):
        self.balances: dict[str, float] = {}
        self.total_outgoing: dict[str, float] = defaultdict(float)
        self._scheduled: list[list] = []  # heap of [execute_ts, payment_id, from_id, to_id, amount, active]
        self._scheduled_by_id: dict[int, list] = {}
        self._next_payment_id = 1

    # --- Level 1: basic account operations ---
    def create_account(self, acct_id: str) -> None:
        if acct_id in self.balances:
            raise ValueError(f"account {acct_id} already exists")
        self.balances[acct_id] = 0.0

    def deposit(self, acct_id: str, amount: float) -> None:
        self.balances[acct_id] += amount

    def transfer(self, from_id: str, to_id: str, amount: float) -> None:
        if self.balances.get(from_id, 0) < amount:
            raise ValueError("insufficient funds")
        self.balances[from_id] -= amount
        self.balances[to_id] += amount
        self.total_outgoing[from_id] += amount

    # --- Level 2: spending leaderboard ---
    def top_spenders(self, k: int) -> list[tuple[str, float]]:
        return heapq.nlargest(k, self.total_outgoing.items(), key=lambda kv: kv[1])

    # --- Level 3: scheduled / cancellable payments ---
    def schedule_payment(self, execute_ts: int, from_id: str, to_id: str, amount: float) -> int:
        pid = self._next_payment_id
        self._next_payment_id += 1
        entry = [execute_ts, pid, from_id, to_id, amount, True]  # last field: active
        heapq.heappush(self._scheduled, entry)
        self._scheduled_by_id[pid] = entry
        return pid

    def cancel_payment(self, payment_id: int) -> None:
        entry = self._scheduled_by_id.get(payment_id)
        if entry is not None:
            entry[5] = False  # lazy delete: never search/remove from the heap directly

    def run_due_payments(self, now_ts: int) -> None:
        while self._scheduled and self._scheduled[0][0] <= now_ts:
            _, _, from_id, to_id, amount, active = heapq.heappop(self._scheduled)
            if active:
                self.transfer(from_id, to_id, amount)

    # --- Level 4: account merges ---
    def merge_accounts(self, primary_id: str, secondary_id: str) -> None:
        self.balances[primary_id] = self.balances.get(primary_id, 0) + self.balances.pop(secondary_id, 0)
        self.total_outgoing[primary_id] += self.total_outgoing.pop(secondary_id, 0)
        for entry in self._scheduled:
            if entry[2] == secondary_id:
                entry[2] = primary_id
            if entry[3] == secondary_id:
                entry[3] = primary_id
```

**Follow-Up Questions:**
1. Tie-breaking in `top_spenders` when two accounts have equal totals → decide and state a rule (e.g.,
   earlier account id first) and pass it as a secondary sort key rather than leaving it to whatever
   `heapq.nlargest` happens to do with the input order.
2. What if `run_due_payments` needs to run continuously in the background rather than being polled? →
   a scheduler thread sleeping until the next heap-top `execute_ts` (recomputed after every schedule/cancel)
   instead of a caller-driven sweep.
3. Concurrent transfers into/out of the same account from multiple threads → the balance mutations need a
   per-account lock (or a single lock around the whole `BankingSystem` for simplicity first, then discuss
   sharding locks by account for throughput).
4. **Multi-currency support** (a reported follow-up level): balances become `dict[(account, currency), float]`
   instead of `dict[account, float]`; a transfer between different currencies needs an explicit
   exchange-rate lookup and should record the rate used at transfer time (rates fluctuate — the ledger
   entry, not a live re-lookup, is the source of truth for "what actually happened").
5. **History replay** (a reported follow-up level): rather than mutating `balances` directly, append every
   operation to an immutable event log (`(ts, op, args)`); `balances` becomes a materialized view rebuilt
   by replaying the log, and "account state as of time T" is just "replay the log up to T" — this is the
   same event-sourcing pattern used by the ledger design in system design #13 below.

---

## 2. Mutable Leaderboard with K-th Highest Score

**Problem Statement:**
Support a leaderboard where a player's score can be set or updated at any time, and queries ask for the
k-th highest score currently on the board.

- `update(player_id, new_score)`: set (or overwrite) a player's score.
- `kth_highest(k)`: return the k-th highest score currently on the board (1-indexed; `k=1` is the top score).

**Example:**
```
update("a", 10)
update("b", 20)
update("c", 15)
kth_highest(1)   # 20
kth_highest(2)   # 15
kth_highest(3)   # 10
update("a", 25)  # a's score changes from 10 to 25
kth_highest(1)   # 25
```

**Test Cases:**

| Operations | Result |
|---|---|
| `update(a,10); update(b,20); update(c,15); kth_highest(2)` | `15` |
| `update(a,10); update(a,25); kth_highest(1)` | `25` (old score of 10 must not linger) |
| `update(a,10); update(b,10); kth_highest(2)` | `10` (duplicate scores both count toward ranking) |
| `kth_highest(k)` with `k` larger than the number of players | raises an error |

**Key Insights:**
1. **This is a multiset, not a set**: duplicate scores are common (ties), so the underlying structure must
   count every player's current score, not deduplicate scores — a plain `set` of scores is the wrong
   primitive.
2. **Update must remove the old score before adding the new one**: `update` on an existing player is a
   move within the multiset, not a pure insert — forgetting to remove the stale score is the most common
   bug (it leaves a "ghost" score influencing rankings).
3. A sorted-container (`sortedcontainers.SortedList`) gives `O(log n)` insert/remove and `O(log n)` (really
   `O(sqrt(n))` amortized in practice) indexed access for the k-th-highest query — the pragmatic interview
   answer. A from-scratch alternative for bounded/discrete score ranges is a Fenwick tree over
   coordinate-compressed scores, supporting `O(log n)` "find the k-th order statistic" via binary search on
   prefix counts.

**Python Solution:**
```python
from sortedcontainers import SortedList


class Leaderboard:
    """
    update:      O(log n) (remove old score, insert new)
    kth_highest: O(log n)
    """

    def __init__(self):
        self.scores = SortedList()               # ascending, duplicates allowed
        self.player_score: dict[str, int] = {}

    def update(self, player_id: str, new_score: int) -> None:
        old_score = self.player_score.get(player_id)
        if old_score is not None:
            self.scores.remove(old_score)
        self.player_score[player_id] = new_score
        self.scores.add(new_score)

    def kth_highest(self, k: int) -> int:
        if k < 1 or k > len(self.scores):
            raise ValueError("k out of range")
        return self.scores[-k]
```

**Follow-Up Questions:**
1. Break ties deterministically (two players with the same score) → track `(score, update_seq)` per
   player and sort by that composite key, so ties resolve consistently (e.g., earlier update ranks higher)
   instead of depending on multiset insertion order.
2. Scale to hundreds of millions of players → a single in-process `SortedList` won't fit in memory;
   shard players across nodes and maintain a smaller "top-K candidates per shard" structure, merging only
   the candidate sets when a global top-K query comes in (exact arbitrary-k-th-highest across shards is
   much harder — call this out explicitly as a real tradeoff, not something to hand-wave).
3. Score updates arrive far more often than `kth_highest` queries → same "don't pay maintenance cost on
   every write if reads are rare" tradeoff seen elsewhere in these docs (e.g., Top-K Frequent Items) — an
   unsorted score map plus `heapq.nlargest` on query is simpler and can beat a continuously-sorted
   structure if `kth_highest` truly is rare relative to `update`.

---

## 3. Moving Average over a Sliding Window

**Problem Statement:**
Implement `MovingAverage(size)` that supports `next(val)`, which appends a new value to a data stream and
returns the average of the last `size` values (or fewer, if fewer than `size` values have arrived yet).

**Example:**
```
m = MovingAverage(3)
m.next(1)   # 1.0        -> window [1]
m.next(10)  # 5.5        -> window [1, 10]
m.next(3)   # 4.666...   -> window [1, 10, 3]
m.next(5)   # 6.0        -> window [10, 3, 5] (1 evicted)
```

**Test Cases:**

| Calls (size=3) | Returned averages |
|---|---|
| `next(1), next(10), next(3), next(5)` | `1.0, 5.5, 4.667, 6.0` |
| `MovingAverage(1)`, `next(4), next(9)` | `4.0, 9.0` (window of 1 = always the latest value) |
| `MovingAverage(5)`, `next(2), next(4)` | `2.0, 3.0` (fewer than `size` values so far — average over what exists) |

**Key Insights:**
1. Maintain a running `total` alongside the window `deque`, updating it by `+= new_val` and `-= evicted_val`
   instead of re-summing the window on every call — this is what makes `next` O(1) amortized instead of
   O(size).
2. The window only evicts once it *exceeds* `size` (`>`, not `>=`), so the first `size - 1` calls
   correctly average over fewer than `size` values rather than over-evicting early.

**Python Solution:**
```python
from collections import deque


class MovingAverage:
    """
    next(): O(1) amortized
    Space:  O(size)
    """

    def __init__(self, size: int):
        self.size = size
        self.window: deque[float] = deque()
        self.total = 0.0

    def next(self, val: float) -> float:
        self.window.append(val)
        self.total += val
        if len(self.window) > self.size:
            self.total -= self.window.popleft()
        return self.total / len(self.window)
```

**Follow-Up Questions:**
1. Window defined by elapsed time, not a fixed count (e.g., "average over the last 60 seconds") → store
   `(value, timestamp)` pairs and evict from the left while `timestamp < now - window_seconds`, still
   O(1) amortized per call since each value is pushed and popped at most once.
2. Support many independent moving averages sharing one process (one per trading symbol, say) → factor out
   the class as-is and instantiate one per key; if the key space is huge and mostly idle, lazily create
   instances on first use and evict idle ones on an LRU basis to bound memory.

---

## 4. Minimum Days to Execute Ordered Tasks with Cooldown

**Problem Statement:**
You're given a fixed, ordered sequence of tasks (identified by type, e.g., `["A", "A", "B", "A"]`) that
must execute in exactly that order — you cannot reorder them. The same task type cannot run again until
`cooldown` full days have passed since its last run (idle days may need to be inserted to satisfy this).
Return the total number of days needed to execute every task in the sequence.

**Example:**
```
tasks = ["A", "A", "B"], cooldown = 2
Day 1: A
Day 2: (idle — A on cooldown until day 4)
Day 3: (idle)
Day 4: A
Day 5: B
Output: 5
```

**Test Cases:**

| tasks | cooldown | days |
|---|---|---|
| `["A","A","B"]` | `2` | `5` |
| `["A","B","A"]` | `2` | `4` (B on day 2 buys A one day of the cooldown "for free") |
| `["A","A","A"]` | `0` | `3` (no cooldown, run back-to-back) |
| `["A"]` | `5` | `1` |

**Key Insights:**
1. Because the order is fixed (unlike the classic "Task Scheduler" problem, which asks you to *choose* an
   order to minimize total time), this is a direct greedy simulation, not a scheduling-optimization
   problem — no need for a frequency-max-heap.
2. Track only `last_run[task_type]`; the earliest a task can run is
   `max(current_day + 1, last_run[task_type] + cooldown + 1)` — the `current_day + 1` term enforces the
   fixed order (can't run before the previous task in sequence finishes), and the second term enforces the
   per-type cooldown.

**Python Solution:**
```python
def min_days_to_execute(tasks: list[str], cooldown: int) -> int:
    """
    Time:  O(n)
    Space: O(distinct task types)
    """
    last_run: dict[str, int] = {}
    current_day = 0

    for task in tasks:
        earliest = current_day + 1
        if task in last_run:
            earliest = max(earliest, last_run[task] + cooldown + 1)
        current_day = earliest
        last_run[task] = current_day

    return current_day
```

**Follow-Up Questions:**
1. Tasks *can* be reordered to minimize total days (the classic variant) → this becomes the frequency-based
   greedy "Task Scheduler" problem (LeetCode 621): the answer is driven by the most frequent task type and
   the gaps its cooldown forces, not a direct simulation.
2. Multiple tasks can execute in parallel (e.g., `W` workers) → per-day capacity becomes `W` instead of 1;
   the simulation now needs to track, per day, which tasks are eligible (order-ready and off cooldown) and
   greedily fill up to `W` slots, which is a meaningfully harder scheduling problem than the single-worker
   case above.

---

## 5. Crypto Trading Order Management System (Limit Order Book)

**Problem Statement:**
Implement a simplified limit order book for a single trading pair:
- `place_order(order_id, side, price, qty)`: `side` is `"BUY"` or `"SELL"`. If the order crosses the
  book (a buy at or above the best ask, or a sell at or below the best bid), match immediately — in full
  or in part — against resting orders in **price-time priority** (best price first, then earliest arrival
  among equal prices). Any unfilled remainder rests in the book.
- `cancel_order(order_id)`: removes an order (fully or whatever quantity remains) from the book if it
  hasn't already fully filled.

**Example:**
```
place_order("s1", "SELL", 100, 5)   # rests, no cross
place_order("b1", "BUY", 101, 3)    # crosses -> trades 3 @ 100 (resting order's price) against s1
                                     # s1 now has 2 remaining; b1 fully filled
place_order("b2", "BUY", 99, 5)     # doesn't cross (best ask is still 100) -> rests
cancel_order("s1")                  # remaining 2 qty removed; b2 still resting, unaffected
```

**Test Cases:**

| Sequence | Resulting trades |
|---|---|
| `SELL s1 @100 x5`, then `BUY b1 @101 x3` | one trade: `(b1, s1, price=100, qty=3)`; `s1` has 2 left resting |
| `BUY b1 @99 x5` (no resting sells) | no trade, `b1` rests at 99 |
| `SELL s1 @100 x5`, `SELL s2 @100 x5`, `BUY b1 @100 x7` | `b1` matches `s1` first (arrived earlier) for qty 5, then `s2` for qty 2 (price-time priority) |
| `SELL s1 @100 x5`, `cancel_order(s1)`, `BUY b1 @101 x3` | no trade — `s1` was cancelled before it could match |

**Key Insights:**
1. **Two heaps, opposite orientation**: a max-heap for buys (simulate with negated prices in Python's
   min-heap) and a min-heap for sells, so `heap[0]` is always each side's best price.
2. **Price-time priority needs a tiebreaker in the heap key**: a strictly increasing sequence number per
   order, included in the heap tuple, makes earlier-arrived orders pop first among equal prices.
3. **Lazy deletion for cancel**: exactly like the Banking System's scheduled payments — mark the order
   inactive and skip it when it's popped, rather than searching the heap for it (`O(n)`).
4. **Execution price is the resting (maker) order's price**: whichever side has the smaller sequence
   number was already in the book when the crossing order arrived, so that side's price is what a real
   exchange would print as the trade price.

**Python Solution:**
```python
import heapq
import itertools


class OrderBook:
    """
    place_order:  O(log n) per heap operation; O(k log n) if a taker order
                  matches against k resting orders
    cancel_order: O(1) (lazy delete)
    """

    def __init__(self):
        self._buy_heap: list[tuple[float, int, str]] = []   # (-price, seq, order_id)
        self._sell_heap: list[tuple[float, int, str]] = []  # (price, seq, order_id)
        self._orders: dict[str, dict] = {}
        self._seq = itertools.count()
        self.trades: list[tuple[str, str, float, int]] = []  # (buy_id, sell_id, price, qty)

    def place_order(self, order_id: str, side: str, price: float, qty: int) -> None:
        seq = next(self._seq)
        self._orders[order_id] = {"side": side, "price": price, "qty": qty, "active": True, "seq": seq}
        book = self._sell_heap if side == "SELL" else self._buy_heap
        heap_price = price if side == "SELL" else -price
        heapq.heappush(book, (heap_price, seq, order_id))
        self._match()

    def cancel_order(self, order_id: str) -> None:
        order = self._orders.get(order_id)
        if order is not None:
            order["active"] = False

    def _best(self, heap: list) -> tuple[str, dict] | tuple[None, None]:
        while heap:
            _, _, oid = heap[0]
            order = self._orders[oid]
            if not order["active"] or order["qty"] == 0:
                heapq.heappop(heap)
                continue
            return oid, order
        return None, None

    def _match(self) -> None:
        while True:
            buy_id, buy = self._best(self._buy_heap)
            sell_id, sell = self._best(self._sell_heap)
            if buy is None or sell is None or buy["price"] < sell["price"]:
                break

            trade_qty = min(buy["qty"], sell["qty"])
            trade_price = sell["price"] if sell["seq"] < buy["seq"] else buy["price"]
            self.trades.append((buy_id, sell_id, trade_price, trade_qty))

            buy["qty"] -= trade_qty
            sell["qty"] -= trade_qty
            if buy["qty"] == 0:
                buy["active"] = False
            if sell["qty"] == 0:
                sell["active"] = False
```

**Follow-Up Questions:**
1. Support order modification ("cancel-replace") → implement as `cancel_order` followed by a fresh
   `place_order` with a new sequence number — this correctly costs the order its time priority, matching
   real exchange behavior (a modified order goes to the back of the queue at its price level).
2. Scale to millions of orders/sec on one symbol → shard by symbol, with each symbol's book owned
   exclusively by one thread/process to avoid locking the heaps; cross-symbol operations (e.g., a
   portfolio-wide risk check) happen in a separate layer that reads a consistent snapshot rather than
   locking multiple books.
3. Add market orders (no limit price, match immediately against the best available price(s) until filled
   or the book is exhausted) → same `_match` loop, but the incoming order's "price" is treated as
   always-crossing (`+inf` for a market buy, `-inf` for a market sell) instead of comparing an explicit limit.

---

## 6. Block Mining with Dependencies: Maximize Fee Under a Block Size Constraint

**Problem Statement:**
You're building the next block from a pool of pending transactions. Each transaction has an `id`, a
`size` (bytes), a `fee`, and an optional `depends_on` (another transaction id that must also be included
in the block for this one to be valid — e.g., it spends an output created by that transaction).
**Assume dependencies form disjoint simple chains**: each transaction depends on at most one other, and is
depended on by at most one other (a linear chain per "thread" of transactions, not a branching tree) — a
reasonable simplification to state explicitly with the interviewer before coding, since general dependency
DAGs make this NP-hard. Choose a subset of transactions, respecting dependencies and a total block size
limit, that maximizes total fee.

**Example:**
```
transactions:
  t1: size=2, fee=5,  depends_on=None
  t2: size=3, fee=4,  depends_on=t1        # chain: t1 -> t2
  t3: size=4, fee=10, depends_on=None      # standalone chain of length 1

block_size = 5
Best choice: {t1, t2} (size 5, fee 9) vs {t3} (size 4, fee 10) vs {t1} (size 2, fee 5)
Output: 10   (take t3 alone — including t1+t2 together uses the whole budget for less fee)
```

**Test Cases:**

| Transactions | block_size | Max fee |
|---|---|---|
| t1(2,5)→t2(3,4); t3(4,10) standalone | `5` | `10` |
| t1(2,5)→t2(3,4); t3(4,10) standalone | `6` | `15` (best is `{t1, t3}`: size 2+4=6, fee 5+10=15 — beats `{t1,t2}` at size 5 fee 9, or `{t3}` alone at fee 10) |
| single chain t1(1,3)→t2(1,3)→t3(1,3) | `2` | `6` (must take prefix — {t1,t2}, can't skip t1 to take t2 alone) |
| no transactions fit (`block_size` smaller than any single transaction) | any | `0` |

**Key Insights:**
1. **A chain forces "all-or-a-prefix" selection**: to include the k-th transaction in a chain you must
   include every transaction before it (its full ancestor chain) — so the only valid selections *within
   one chain* are "take the first `k` transactions" for some `k` (including `k=0`), not an arbitrary subset.
2. **This reduces to grouped knapsack**: precompute each chain's prefix `(cumulative_size, cumulative_fee)`
   pairs as that chain's "menu of options" (including the option of taking none), then solve 0/1 knapsack
   where you pick **at most one option per chain (group)** — the standard "knapsack with grouped/mutually
   exclusive choices" pattern.
3. Iterate the size dimension **downward** within a group's inner loop (as in ordinary 0/1 knapsack) so
   each chain's own prefix options don't get combined with each other in the same pass — but base every
   candidate on the *previous* group's `dp` (before this group), not the in-progress `new_dp`, which is
   what actually enforces "at most one option from this group."

**Python Solution:**
```python
def max_fee_block(transactions: dict[str, tuple[int, int, str | None]], block_size: int) -> int:
    """
    transactions: id -> (size, fee, depends_on_id_or_None)
    Assumes dependencies form disjoint simple chains.
    Time:  O(n + block_size * total_chain_prefixes)
    Space: O(block_size)
    """
    child_of = {tx_id: None for tx_id in transactions}
    roots = []
    for tx_id, (_, _, dep) in transactions.items():
        if dep is None:
            roots.append(tx_id)
        else:
            child_of[dep] = tx_id

    groups: list[list[tuple[int, int]]] = []
    for root in roots:
        prefixes = []
        cum_size = cum_fee = 0
        node = root
        while node is not None:
            size, fee, _ = transactions[node]
            cum_size += size
            cum_fee += fee
            prefixes.append((cum_size, cum_fee))
            node = child_of[node]
        groups.append(prefixes)

    dp = [0] * (block_size + 1)
    for prefixes in groups:
        new_dp = dp[:]
        for cum_size, cum_fee in prefixes:
            for cap in range(block_size, cum_size - 1, -1):
                candidate = dp[cap - cum_size] + cum_fee
                if candidate > new_dp[cap]:
                    new_dp[cap] = candidate
        dp = new_dp

    return max(dp)
```

**Follow-Up Questions:**
1. Dependencies form a general tree (a transaction can have multiple children spending different outputs
   of it) → still solvable in polynomial time via a tree-knapsack DP (merge children's `dp` arrays
   bottom-up at each node), but meaningfully more complex than the chain reduction above.
2. Dependencies form a general DAG (multiple parents per transaction) → this is NP-hard in general; real
   miners (e.g., Bitcoin Core) don't solve it exactly at mempool scale — they use a greedy heuristic
   (highest fee-rate — fee/size — first, in a valid topological order), trading optimality for speed.
3. Transactions can also depend on transactions *not* in the candidate pool (already confirmed on-chain)
   → treat those as already-satisfied dependencies (drop them from the chain-building step entirely, they
   don't consume block space or need to be "chosen").

---

## 7. NFT Registry: Core Operations and Weighted Attribute Combination Generation

**Problem Statement:**
Two related asks, typically given as one exercise with the second as a follow-up extension:

**Part A — Core registry:** implement `mint(nft_id, owner, attributes)`, `transfer(nft_id, from_owner,
new_owner)` (must reject if `from_owner` doesn't actually hold the token), `owner_of(nft_id)`, and
`tokens_of(owner)`. Include tests exercising the ownership-validation behavior.

**Part B — Collection generation:** given a set of attribute categories, each with possible values and a
rarity weight (e.g., `background: [(red, 0.5), (blue, 0.3), (gold, 0.2)]`), generate `count` **unique**
attribute combinations for a new collection by weighted-random sampling per category, rejecting and
re-sampling on collisions.

**Example:**
```
registry.mint("nft1", "alice", {"background": "red"})
registry.transfer("nft1", "alice", "bob")   # ok
registry.transfer("nft1", "alice", "carol") # raises — alice no longer owns nft1
registry.owner_of("nft1")                    # "bob"

generate_unique_combinations({"bg": [("red",1),("blue",1)], "eyes": [("normal",1),("laser",1)]}, count=4)
# -> all 4 combinations of {bg} x {eyes}, each appearing exactly once
```

**Test Cases:**

| Call | Result |
|---|---|
| `mint("n1","alice",{}); transfer("n1","alice","bob")` | `owner_of("n1") == "bob"` |
| `transfer("n1","alice","carol")` after the above | raises (alice no longer owns `n1`) |
| `mint("n1", ...)` twice with same id | raises (no re-minting an existing id) |
| `generate_unique_combinations(options, count=total_combo_space)` | returns exactly that many, all distinct |
| `generate_unique_combinations(options, count > total_combo_space, max_attempts=N)` | returns fewer than requested rather than looping forever |

**Key Insights:**
1. **`transfer` must validate the claimed sender, not just move the token**: without checking
   `from_owner == current_owner`, a stale or malicious call could "transfer" a token the caller doesn't
   actually hold — this check is the whole point of the exercise, not an edge case.
2. **De-duplicate on the full combination tuple, not per-category**: sampling each category independently
   guarantees variety *within* a category but says nothing about the joint tuple being new — the seen-set
   must key on the complete `(cat1_value, cat2_value, ...)` tuple.
3. **Rejection sampling degrades as the requested count approaches the full combination space** (a
   birthday-paradox effect: collisions get more likely the fuller the set gets) — always cap attempts
   rather than looping unconditionally.

**Python Solution:**
```python
import random


class NFTRegistry:
    """
    mint/transfer/owner_of: O(1)
    tokens_of:              O(k) for k tokens held by that owner
    """

    def __init__(self):
        self._owner: dict[str, str] = {}
        self._attributes: dict[str, dict] = {}
        self._by_owner: dict[str, set[str]] = {}

    def mint(self, nft_id: str, owner: str, attributes: dict) -> None:
        if nft_id in self._owner:
            raise ValueError(f"{nft_id} already minted")
        self._owner[nft_id] = owner
        self._attributes[nft_id] = attributes
        self._by_owner.setdefault(owner, set()).add(nft_id)

    def transfer(self, nft_id: str, from_owner: str, new_owner: str) -> None:
        if self._owner.get(nft_id) != from_owner:
            raise ValueError("transfer sender does not own this token")
        self._by_owner[from_owner].discard(nft_id)
        self._owner[nft_id] = new_owner
        self._by_owner.setdefault(new_owner, set()).add(nft_id)

    def owner_of(self, nft_id: str) -> str:
        return self._owner[nft_id]

    def tokens_of(self, owner: str) -> set[str]:
        return set(self._by_owner.get(owner, set()))


def generate_unique_combinations(attribute_options: dict[str, list[tuple[str, float]]],
                                  count: int, max_attempts: int = 10_000) -> list[dict]:
    """
    attribute_options: category -> list of (value, weight)
    Time: O(count) expected when count is well below the total combination
          space; degrades as it approaches the full space (see Key Insights).
    """
    categories = list(attribute_options.keys())
    seen: set[tuple] = set()
    results: list[dict] = []
    attempts = 0

    while len(results) < count and attempts < max_attempts:
        attempts += 1
        combo = {}
        key_parts = []
        for cat in categories:
            values, weights = zip(*attribute_options[cat])
            choice = random.choices(values, weights=weights, k=1)[0]
            combo[cat] = choice
            key_parts.append(choice)

        key = tuple(key_parts)
        if key in seen:
            continue
        seen.add(key)
        results.append(combo)

    return results
```

**Follow-Up Questions:**
1. `count` equals the entire combination space (rejection sampling gets slow near the end) → once few
   unused combinations remain, switch to explicitly enumerating the remaining unused ones (e.g., via
   `itertools.product` minus `seen`) and sampling uniformly among just those, instead of pure rejection.
2. How would this differ as an actual on-chain smart contract? → the same core invariants (owner mapping,
   sender-must-match-owner on transfer) are enforced by contract logic instead of in-process Python, with
   the added constraints of gas cost per operation and every state change being a durable, public
   transaction rather than an in-memory mutation.

---

## 8. In-Memory Key-Value Store with Per-Key User Locks and Top-N Operation Tracking

**Problem Statement:**
Implement an in-memory key-value store with:
- `lock(user, key)` / `unlock(user, key)`: exclusive per-key lock held by a specific user; a different
  user's write is rejected while a key is locked by someone else (the lock holder can still operate on it
  freely).
- `set_by_user(user, key, value)` / `delete_by_user(user, key)`: writes, subject to the lock check above.
- `get(key)`: a read (not subject to locking — reads are always allowed).
- `top_n_by_operations(n)`: return the `n` keys with the most total operations (reads + writes) recorded
  against them.

**Example:**
```
kv.lock("alice", "k1")
kv.set_by_user("bob", "k1", 1)     # raises — locked by alice
kv.set_by_user("alice", "k1", 1)   # ok — alice holds the lock
kv.unlock("alice", "k1")
kv.set_by_user("bob", "k1", 2)     # ok now
```

**Test Cases:**

| Sequence | Result |
|---|---|
| `lock(alice,k1); set_by_user(bob,k1,1)` | raises `PermissionError` |
| `lock(alice,k1); set_by_user(alice,k1,1)` | succeeds |
| `lock(alice,k1); unlock(alice,k1); set_by_user(bob,k1,2)` | succeeds |
| `set_by_user(a,k1,1); get(k1); get(k1); top_n_by_operations(1)` | `[("k1", 3)]` |

**Key Insights:**
1. **The lock is keyed by holder identity, not a boolean**: a plain "is this key locked" flag would block
   the very user who holds the lock from continuing to operate on it — the check must be "locked by
   someone *other than* the caller."
2. **Operation counting must include reads**, not just writes, or `top_n_by_operations` can't actually
   answer "which keys are hot" — a key that's read constantly but rarely written is still a hot key.
3. Same "compute top-N on demand, don't maintain a live-sorted structure" tradeoff as the other top-k
   problems in this set: `heapq.nlargest` on a plain counts dict is the right default unless
   `top_n_by_operations` is called about as often as `get`/`set` themselves.

**Python Solution:**
```python
from collections import defaultdict
import heapq


class LockingKeyValueStore:
    """
    lock/unlock/set_by_user/delete_by_user/get: O(1)
    top_n_by_operations(n):                     O(k log n) for k distinct keys touched
    """

    def __init__(self):
        self._data: dict[str, object] = {}
        self._locked_by: dict[str, str] = {}
        self._op_count: dict[str, int] = defaultdict(int)

    def lock(self, user: str, key: str) -> None:
        holder = self._locked_by.get(key)
        if holder is not None and holder != user:
            raise PermissionError(f"{key} is locked by another user")
        self._locked_by[key] = user

    def unlock(self, user: str, key: str) -> None:
        if self._locked_by.get(key) == user:
            del self._locked_by[key]

    def _check_access(self, user: str, key: str) -> None:
        holder = self._locked_by.get(key)
        if holder is not None and holder != user:
            raise PermissionError(f"{key} is locked by another user")

    def set_by_user(self, user: str, key: str, value: object) -> None:
        self._check_access(user, key)
        self._data[key] = value
        self._op_count[key] += 1

    def delete_by_user(self, user: str, key: str) -> None:
        self._check_access(user, key)
        self._data.pop(key, None)
        self._op_count[key] += 1

    def get(self, key: str) -> object:
        self._op_count[key] += 1
        return self._data.get(key)

    def top_n_by_operations(self, n: int) -> list[tuple[str, int]]:
        return heapq.nlargest(n, self._op_count.items(), key=lambda kv: kv[1])
```

**Follow-Up Questions:**
1. A client crashes while holding a lock, leaking it forever → add an expiry timestamp to each lock and
   treat an expired lock as absent on the next access attempt, rather than requiring an explicit unlock.
2. Lock multiple keys atomically for a multi-key transaction → acquire locks in a globally consistent
   order (e.g., sorted key order) across all callers to prevent circular-wait deadlock between two
   transactions locking the same two keys in opposite orders.

---

## 9. Drone Delivery Route Simulation with Charging Stations

**Problem Statement:**
A delivery drone travels in a straight line from `start` to `destination`. It has a maximum range
`range_limit` per charge and recharges to full range at any charging station it stops at. Simulate the
delivery by always advancing to the **nearest** charging station ahead that's within current range (a
greedy, reactive strategy — not necessarily the strategy that minimizes number of stops). Determine
whether the destination is reachable, and if so, which stations were used.

**Example:**
```
start=0, destination=100, range_limit=30, stations=[25, 55, 80]
Route: 0 -> 25 -> 55 -> 80 -> 100 (each hop <= 30)
Output: (distance=100, stops_used=[25, 55, 80])
```

**Test Cases:**

| start, destination, range | stations | Result |
|---|---|---|
| `0, 100, 30` | `[25, 55, 80]` | `(100, [25, 55, 80])` |
| `0, 50, 30` | `[]` | `None` (no stations, destination out of range) |
| `0, 40, 30` | `[]` | `(40, [])` (destination itself within initial range, no stops needed) |
| `0, 100, 20` | `[25, 55, 80]` | `None` (gap from 0 to 25 exceeds range 20 before any station is reachable) |

**Key Insights:**
1. **Total distance traveled is always exactly `destination - start`** in this 1D, forward-only setup —
   there's no backtracking, so "distance" alone is a trivial output. The actual question worth answering
   is *reachability* and *which/how many stops* are required — say this explicitly rather than presenting
   distance as if it were the interesting result.
2. Because the strategy is specified as "always hop to the nearest reachable station" (not "hop as far as
   possible," which is the optimal-stop-count strategy from the classic "Minimum Refueling Stops"
   problem), this is a direct greedy simulation with no need for a max-heap of passed-but-unused stations.

**Python Solution:**
```python
def simulate_drone_delivery(start: float, destination: float, range_limit: float,
                              stations: list[float]) -> tuple[float, list[float]] | None:
    """
    Time:  O(n log n) to sort stations, O(n) to simulate
    Space: O(n)
    Returns (total_distance, stops_used) if reachable, else None.
    """
    stations = sorted(s for s in stations if start < s <= destination)
    position = start
    stops_used: list[float] = []
    idx = 0

    while destination - position > range_limit:
        while idx < len(stations) and stations[idx] <= position:
            idx += 1
        if idx >= len(stations) or stations[idx] - position > range_limit:
            return None
        position = stations[idx]
        stops_used.append(position)
        idx += 1

    return destination - start, stops_used
```

**Follow-Up Questions:**
1. Minimize the number of stops instead of always taking the nearest one → switch to the classic greedy:
   extend as far as possible each leg, keeping a max-heap of all stations already passed but not yet used,
   and pop from it (i.e., "retroactively" use the best passed station) only when a shortfall is detected.
2. Generalize to 2D routing (not a straight line) → this becomes a graph/shortest-path problem: build an
   edge between any two stations (or start/destination) within `range_limit` of each other, then run
   Dijkstra/BFS for reachability and shortest total travel distance.

---

## 10. Flappy-Bird Physics: Minimum Jumps to Stay Within Bounds

**Problem Statement:**
A bird moves under simplified discrete physics: each tick, if it doesn't jump, `velocity += gravity` then
`position += velocity`; if it jumps, `velocity` is instead set to a fixed `jump_velocity` before the
position update. Given `gravity` (negative), `jump_velocity` (positive), a starting position, a floor
`min_safe_y`, a ceiling `max_safe_y`, and a number of `ticks` to survive, compute the **minimum number of
jumps** needed to keep the bird's position within `[min_safe_y, max_safe_y]` for every tick — or determine
it's impossible.

**Example:**
```
gravity=-1, jump_velocity=5, start_position=0, min_safe_y=-3, max_safe_y=10, ticks=5
Tick-by-tick (no jump unless forced): positions -1, -3, 2 (jump here), 6, 9
Output: 1 jump
```

**Test Cases:**

| gravity, jump_velocity, start, floor, ceiling, ticks | Min jumps |
|---|---|---|
| `-1, 5, 0, -3, 10, 5` | `1` |
| `-1, 5, 0, -100, 100, 3` (floor never threatened) | `0` |
| `-1, 2, -10, 0, 100, 3` (already 10 below the floor, and `jump_velocity=2` can't close that gap in one tick) | `None` (impossible — jumping still lands below the floor) |
| `-1, 5, 0, -3, 4, 5` (tight ceiling) | `None` once a forced jump would exceed the ceiling |

**Key Insights:**
1. **Greedy "jump only when forced" is provably optimal here**: because a jump resets velocity to the same
   fixed `jump_velocity` regardless of when it's used, jumping earlier than necessary only means more
   accumulated free-fall descent was skipped for no benefit — an exchange argument shows any jump can be
   delayed to the last tick before it would violate the floor without making anything worse (and this also
   minimizes ceiling-overshoot risk, since delaying keeps the bird lower before each jump).
2. Check the ceiling **after** applying the tick's actual velocity (jumped or not) — a solution that only
   checks the floor will silently accept ceiling-violating jump sequences.
3. A jump isn't automatically a fix — it's only a valid response to an impending floor violation if the
   jump itself actually lands at or above the floor. If the bird is already far enough below the floor (or
   `jump_velocity` is too weak) that even jumping can't recover, that's a genuine impossibility, not
   something to paper over by jumping anyway.

**Python Solution:**
```python
def min_jumps_to_survive(gravity: float, jump_velocity: float, start_position: float,
                          min_safe_y: float, max_safe_y: float, ticks: int) -> int | None:
    """
    Time:  O(ticks)
    Space: O(1)
    """
    position = start_position
    velocity = 0.0
    jumps = 0

    for _ in range(ticks):
        next_velocity_no_jump = velocity + gravity
        next_position_no_jump = position + next_velocity_no_jump

        if next_position_no_jump < min_safe_y:
            next_position_if_jump = position + jump_velocity
            if next_position_if_jump < min_safe_y:
                return None  # even jumping can't clear the floor
            velocity = jump_velocity
            jumps += 1
        else:
            velocity = next_velocity_no_jump

        position += velocity
        if position > max_safe_y:
            return None

    return jumps
```

**Follow-Up Questions:**
1. **Autopilot with a limited "coin" budget** (a reported extension): given only `K` available jumps,
   determine if survival is possible without exceeding the budget → since the greedy above computes the
   true minimum jump count (by the exchange argument), survival is possible iff `min_jumps_to_survive(...)
   <= K` — no separate algorithm is needed.
2. Gravity or jump strength varies over time (e.g., wind gusts per tick) → the same simulation loop works
   unchanged; just index `gravity`/`jump_velocity` by tick instead of treating them as constants.

---

## 11. Parse and Group Log Lines by Thread ID

**Problem Statement:**
Given raw log lines of the form `"[<timestamp>] <thread_id>: <message>"`, parse each line and group
messages by `thread_id`, with each thread's messages sorted by timestamp ascending. Skip malformed lines
rather than failing the whole batch.

**Example:**
```
lines = [
  "[100] t1: started",
  "[50] t2: waiting",
  "[75] t1: progress",
  "garbage line",
]
Output:
{
  "t1": [(75, "progress"), (100, "started")],
  "t2": [(50, "waiting")],
}
```

**Test Cases:**

| Input lines | Output |
|---|---|
| `["[100] t1: a", "[50] t1: b"]` | `{"t1": [(50,"b"), (100,"a")]}` |
| `["not a log line"]` | `{}` |
| `["[1] t1: a", "[2] t2: b"]` | `{"t1": [(1,"a")], "t2": [(2,"b")]}` |
| `[]` | `{}` |

**Key Insights:**
1. Real log streams always contain some malformed lines (truncated writes, mixed formats) — skip lines
   that don't match rather than raising, so one bad line doesn't lose the whole batch.
2. Group first, then sort each (typically much smaller) per-thread list, rather than sorting the entire
   input up front — cheaper when there are many threads each with relatively few lines.

**Python Solution:**
```python
import re
from collections import defaultdict


def group_logs_by_thread(lines: list[str]) -> dict[str, list[tuple[int, str]]]:
    """
    Time:  O(n log m) where m is the largest single thread's line count
           (dominates the O(n) parse pass)
    Space: O(n)
    """
    pattern = re.compile(r"^\[(\d+)\]\s+(\S+):\s*(.*)$")
    grouped: dict[str, list[tuple[int, str]]] = defaultdict(list)

    for line in lines:
        match = pattern.match(line)
        if not match:
            continue
        ts, thread_id, message = match.groups()
        grouped[thread_id].append((int(ts), message))

    for thread_id in grouped:
        grouped[thread_id].sort(key=lambda entry: entry[0])

    return dict(grouped)
```

**Follow-Up Questions:**
1. Logs arrive as a continuous stream rather than a fixed batch, and per-thread output must stay sorted at
   all times → if each thread's own lines are already roughly time-ordered (typical for real logs), simple
   append-only lists stay sorted for free; if not, maintain a small heap per thread.
2. Timestamps come from different machines with clock skew → "sort by timestamp" is only as trustworthy as
   the clock source — note this explicitly, and mention logical/vector clocks or a centrally-assigned
   sequence number as the fix for a genuinely distributed system.

---

## 12. Cursor-Based API Pagination with Filtering

**Problem Statement:**
Implement a paginated listing endpoint over an in-memory dataset: `list_records(filters, cursor,
page_size)` returns up to `page_size` records matching exact-match `filters`, plus an opaque
`next_cursor` for the following page (or `None` if this is the last page). Pagination must remain correct
(no skipped or duplicated records) even if records are inserted or deleted between page requests —
ruling out simple offset-based pagination.

**Example:**
```
store = PaginatedStore([{"id":1,"type":"a"},{"id":2,"type":"b"},{"id":3,"type":"a"}])
page1, cursor = store.list_records({"type":"a"}, cursor=None, page_size=1)
# page1 = [{"id":1,"type":"a"}], cursor = <opaque token encoding "after id 1">
page2, cursor2 = store.list_records({"type":"a"}, cursor=cursor, page_size=1)
# page2 = [{"id":3,"type":"a"}], cursor2 = None (last page)
```

**Test Cases:**

| Call sequence | Result |
|---|---|
| Page through all matching records, `page_size=1` at a time | every matching record returned exactly once, in id order |
| A record is deleted between two page fetches (not the cursor record) | remaining pages still correct — no skip/duplicate |
| Last page returned exactly `page_size` records that happen to be the last ones | `next_cursor` is `None`, not an unnecessary extra empty page |
| `filters={}` | all records returned, unfiltered |

**Key Insights:**
1. **The cursor encodes "last seen id," not an offset** — an offset shifts meaning under concurrent
   inserts/deletes (page 2 can skip or repeat records); a cursor keyed to a specific record's position in a
   stable sort order does not.
2. The sort key backing pagination must be **unique and monotonic** (an auto-incrementing id works;
   a mutable or non-unique field does not) or pages can silently skip or repeat records.
3. Encode the cursor opaquely (base64 of a small JSON payload) so it's an implementation detail the client
   passes back verbatim, never something it constructs or interprets itself — this leaves room to change
   the underlying pagination strategy later without breaking clients.

**Python Solution:**
```python
import base64
import json


class PaginatedStore:
    """
    list_records: O(n) filter/sort scan per call in this in-memory reference
                  version; a real service pushes the filter+sort into an
                  indexed database query instead.
    """

    def __init__(self, records: list[dict], id_field: str = "id"):
        self.records = records
        self.id_field = id_field

    def _matches(self, record: dict, filters: dict) -> bool:
        return all(record.get(k) == v for k, v in filters.items())

    def _encode_cursor(self, last_id) -> str:
        return base64.urlsafe_b64encode(json.dumps({"after": last_id}).encode()).decode()

    def _decode_cursor(self, cursor: str):
        return json.loads(base64.urlsafe_b64decode(cursor.encode()).decode())["after"]

    def list_records(self, filters: dict, cursor: str | None, page_size: int):
        matching = sorted(
            (r for r in self.records if self._matches(r, filters)),
            key=lambda r: r[self.id_field],
        )

        start_idx = 0
        if cursor is not None:
            after_id = self._decode_cursor(cursor)
            start_idx = next(
                (i + 1 for i, r in enumerate(matching) if r[self.id_field] == after_id),
                len(matching),
            )

        page = matching[start_idx:start_idx + page_size]
        has_more = start_idx + page_size < len(matching)
        next_cursor = self._encode_cursor(page[-1][self.id_field]) if page and has_more else None
        return page, next_cursor
```

**Follow-Up Questions:**
1. Filtering needs ranges or full-text search, not just exact match → in a real system this pushes down to
   a proper index (a range index, or a search engine) rather than a linear scan; the cursor mechanics stay
   the same regardless of how filtering is implemented underneath.
2. The dataset is too large to re-sort on every call → maintain the sort order persistently (e.g., a
   B-tree index on the id field) instead of recomputing a full sort per request.

---

## 13. System Design — Ledger & Balance Storage with Sharding

**Problem Statement:**
Design the storage layer for a crypto exchange's account balances: every balance-affecting event (trade,
deposit, withdrawal, fee) must be durably recorded, current balances must be queryable with low latency,
and historical "balance as of time T" queries must be supportable for audits.

**Functional Requirements:**
- Record every balance-affecting event as part of an auditable history, per user per asset.
- Serve low-latency current-balance reads (needed before allowing a trade or withdrawal).
- Support "what was this user's balance at time T" queries for compliance/audit.

**Non-Functional Requirements:**
- Correctness is paramount: a balance must never go negative, and must never be lost or duplicated, even
  under concurrent updates or partial failures — this outweighs raw throughput as a priority.
- Scale to many millions of users across many asset types.
- A small number of extremely active accounts (market makers) can be far hotter than typical users.

**High-Level Design:**
1. **Immutable ledger as the source of truth**: every balance-affecting event is appended as an immutable,
   double-entry ledger row (a debit and a credit); ledger rows are never mutated or deleted.
2. **Materialized balances table**: the current balance per `(user_id, asset)` is a fast-read cache/view,
   updated in the *same* transaction as the ledger append it derives from, so the two can never drift.
3. **Sharding by user_id**: both the ledger and the balances table are partitioned by `user_id` so a given
   user's full history and current balance live on one shard, keeping single-user operations (deposits,
   withdrawals, fees) single-shard transactions.
4. **Cross-shard trade settlement**: a trade between two users on different shards can't be a single-shard
   transaction — it needs a saga or two-phase-commit pattern (with per-leg idempotency keys) to move
   balance out of one shard and into another atomically, without either side succeeding alone.
5. **Historical queries via snapshots**: periodic per-user balance snapshots bound how far back a
   "balance as of time T" query has to replay ledger entries, instead of replaying a user's entire history
   every time.

**Data Model (sketch):**
```
ledger_entries(entry_id, user_id, asset, delta, balance_after, ref_trade_id, ts)  # append-only
balances(user_id, asset, current_balance, last_entry_id)      # materialized, updated with each ledger insert
balance_snapshots(user_id, asset, balance, as_of_ts)           # periodic, bounds replay length
```

**Scaling & Reliability:**
- Shard by `user_id`; single-user events (deposit, withdrawal, fee) stay single-shard and single-transaction.
- Cross-shard trades use a saga/2PC with per-leg idempotency keys so a retried settlement never
  double-applies a balance change.
- Periodic balance snapshots bound audit-query replay cost regardless of account age/activity.

**Follow-Up Questions:**
1. How do you guarantee a balance never goes negative under concurrent trades? → a conditional update
   (`UPDATE balances SET balance = balance - X WHERE balance >= X`) inside the same transaction as the
   ledger append, enforced at commit time — never a separate read-then-write.
2. How would you catch a bug that let the materialized `balances` table drift from the ledger? → a
   scheduled reconciliation job that replays the ledger per user (or per shard) and diffs the recomputed
   balance against the materialized value, alerting on any mismatch.
3. A single hyperactive account becomes a hot shard → shard further by `(user_id, asset)` if the
   bottleneck is one user's own volume, or move very high-volume accounts onto dedicated capacity.

---

## 14. System Design — Idempotent Trade Settlement Pipeline

**Problem Statement:**
Design the pipeline that takes a matched trade (from a matching engine) and durably applies its effects —
updating both parties' balances, recording the fill, and notifying both users — exactly once, even though
the matching engine, settlement service, and notification layer communicate over an at-least-once
messaging system.

**Functional Requirements:**
- On a `TradeMatched` event, debit the seller's asset and credit the buyer's asset (and vice versa for the
  quote currency), record the fill, and notify both parties.
- The matching engine itself must not block waiting on settlement or notification to complete.

**Non-Functional Requirements:**
- Exactly-once *effect* on balances despite at-least-once message delivery (Kafka-style).
- Low settlement latency (users expect a near-immediate balance update after a fill), without coupling
  that latency to the matching engine's own throughput.
- Full auditability of every state transition, for compliance.

**High-Level Design:**
1. The matching engine emits a `TradeMatched` event to a topic partitioned by a key that keeps one trade's
   events ordered (e.g., `trade_id`).
2. The settlement service consumes the topic; before applying any balance change, it checks a
   `processed_trades(trade_id)` table — if already present, it skips the event (safe re-delivery).
3. Settlement runs as a state machine per trade — `MATCHED -> SETTLING -> SETTLED` (or `FAILED`, with a
   compensating-reversal path) — where each transition is a conditional update guarded by the expected
   prior state, so a duplicate or racing consumer sees the wrong expected state and no-ops instead of
   double-applying.
4. The balance update (system design #13) and the `processed_trades` insert and the `SETTLED` transition
   all happen in **one transaction** — this atomicity is what prevents "moved the money but forgot to mark
   it processed" (or the reverse) under a crash mid-way.
5. Notification is a separate downstream consumer of the `SETTLED` transition (outbox pattern), decoupled
   from the settlement transaction — a slow or failing notification channel never blocks or risks
   re-running the actual money movement.

**Data Model (sketch):**
```
trade_events(trade_id, buyer_id, seller_id, price, qty, status)  # MATCHED / SETTLING / SETTLED / FAILED
processed_trades(trade_id PK, processed_at)                        # idempotency guard
notification_outbox(id, trade_id, payload, sent)
```

**Scaling & Reliability:**
- Kafka partitioning by `trade_id` bounds ordering guarantees to what's actually needed, without
  serializing unrelated trades through one partition.
- Settlement workers scale horizontally by claiming work with `SELECT ... FOR UPDATE SKIP LOCKED`-style
  semantics, so no two workers double-process the same trade.
- A periodic reconciliation job compares `SETTLED` trades against the ledger to catch anything a bug let
  slip through.

**Follow-Up Questions:**
1. The service crashes after debiting the buyer but before crediting the seller — how do you avoid a stuck
   half-settled trade? → both legs must be one atomic transaction (invoking the cross-shard saga from
   system design #13 if the two accounts are on different shards), never two independent operations the
   settlement service performs in sequence.
2. How does this relate to the in-process `OrderBook` coding problem (#5)? → that's the single-process
   matching logic deciding *who* trades with *whom* at *what* price; this is the separate, durable,
   distributed pipeline that takes an already-matched trade and makes its balance effects exactly-once
   across services.
3. How would you support reversing a trade (e.g., a fraud clawback)? → model it as a new ledger entry (a
   reversal referencing the original `trade_id`), never as a mutation of the original settled record — this
   preserves the immutable-ledger audit property from system design #13.

---

## 15. System Design — Real-Time Price Ticker Fan-Out (WebSocket)

**Problem Statement:**
Design the system that pushes real-time price updates for many trading symbols to a large number of
concurrently connected clients (e.g., live crypto prices on a trading UI).

**Functional Requirements:**
- A market-data feed publishes price ticks per symbol; every client subscribed to a symbol receives each
  tick with low latency.
- A newly-connecting client immediately sees the current price for symbols it subscribes to, not just the
  next tick.
- Clients can detect a missed/dropped tick and resynchronize, rather than silently displaying a stale price.

**Non-Functional Requirements:**
- Sub-second fan-out latency from tick ingestion to client delivery.
- Correctness matters more than in a typical notification feed: a stale or silently-dropped price during
  active trading is a real financial-harm concern, not just a UX blemish.
- Extreme subscriber skew across symbols (a small number of symbols — e.g., BTC/USD — can have orders of
  magnitude more subscribers than a typical altcoin).

**High-Level Design:**
1. **Ingestion**: the market-data feed publishes each tick (`symbol, price, sequence_number, ts`) to a
   pub/sub topic (Kafka or Redis pub/sub), partitioned/keyed by `symbol` so per-symbol ordering is
   preserved.
2. **Gateway tier**: stateless WebSocket-holding servers; each subscribes only to the symbols its
   currently-connected clients care about, and fans ticks out over the open sockets. This is the same
   "stateless connection-holder subscribes via pub/sub, fans out to its own sockets" shape used for chat
   fan-out and presence systems — the pattern generalizes across all three.
3. **Sequence numbers**: each tick carries a monotonically increasing `sequence_number` per symbol; a
   client that observes a gap (received sequence `N` then `N+2`) knows it missed a tick and requests a
   fresh snapshot instead of continuing to trust a possibly-incomplete price history.
4. **Snapshot cache**: the last known price (and sequence number) per symbol is cached (e.g., Redis) so a
   newly-connecting or resyncing client gets an immediate correct value instead of waiting for the next
   tick.
5. **Hot-symbol handling**: a popular symbol's subscriber list is sharded across multiple gateway-facing
   pub/sub channels (or partitions) so fanning it out isn't bottlenecked on one process/thread.

**Data Model (sketch):**
```
ticks(symbol, price, sequence_number, ts)            # pub/sub topic, keyed by symbol
latest_price(symbol, price, sequence_number, ts)     # cache, one row per symbol
subscriptions(gateway_node_id, symbol) -> client_ids  # in-memory on each gateway node
```

**Scaling & Reliability:**
- Gateway tier scales horizontally; each node only needs pub/sub bandwidth for the symbols its own
  clients actually want, not the full universe.
- The snapshot cache is the only shared read-heavy state and can be replicated/sharded by symbol like any
  other hot-key cache.
- A gateway node crash drops only the clients connected to it; reconnecting clients get a fresh snapshot
  plus current sequence number and resume from there — no replay of historical ticks is needed.

**Follow-Up Questions:**
1. How do you avoid one extremely hot symbol overwhelming a single gateway process? → shard that symbol's
   subscriber base across multiple internal channels/partitions rather than treating "one symbol = one
   fan-out unit."
2. A client's network stalls for 30 seconds during high volatility — what does it see on reconnect? → a
   fresh snapshot at the current sequence number, not a replay of every missed tick; the UI shows the
   latest true price immediately.
3. How would this differ if it needed exactly-once delivery guarantees for billing/settlement, not just
   display? → display-only tolerates "detect gap, resync"; a billing-relevant feed would need durable,
   ordered, acknowledged delivery (e.g., a persistent per-client cursor into the Kafka log) instead of
   best-effort WebSocket push.

---

## 16. Behavioral Themes

See [`behavioral_interview.md`](./behavioral_interview.md) for general STAR-method prep. Themes specific
to Coinbase's loop:

- **Technical leadership deep-dive**: be ready to walk through a project you drove end-to-end — problem,
  your specific contribution, one hard technical tradeoff, measurable outcome, and what you'd change.
- **AI-collaboration in daily workflow**: have a concrete, specific answer for how you actually use AI
  coding tools — real examples of when you accepted a suggestion and when you caught and rejected a wrong
  one read as far more credible than a generic "it speeds up boilerplate."
- **Ownership under ambiguity**: financial-infrastructure work often starts underspecified — a story
  about clarifying requirements or scope proactively, rather than building the wrong thing first, fits
  this loop well.

---

## References

Sources used for compiling these questions:
- [Coinbase Interview Questions - 1point3acres](https://www.1point3acres.com/interview/problems/company/coinbase)

Note: the source page requires forum membership to view full question text/discussion threads for its
featured question bank; several of the coding problems above (Mutable Leaderboard, Moving Average, Task
Cooldown, Crypto Trading Order Management, Block Mining, NFT Registry, In-Memory KV Store with Locks,
Drone Delivery, Flappy-Bird Physics, Log Parsing, Pagination) were reconstructed and expanded from
publicly-visible practice-problem titles on the same page into complete, solvable problem statements with
original test cases and solutions.
