# Coinbase Interview Questions

---

## Contents

**Coding**
1. [Banking System: Accounts, Leaderboard, Scheduled Payments, and Merges](#1-banking-system-accounts-leaderboard-scheduled-payments-and-merges)

**System Design**
2. [Real-Time Price Ticker Fan-Out (WebSocket)](#2-system-design--real-time-price-ticker-fan-out-websocket)

**Behavioral**
3. [Behavioral Themes](#3-behavioral-themes)

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

---

## 2. System Design — Real-Time Price Ticker Fan-Out (WebSocket)

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

## 3. Behavioral Themes

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

Note: the source page requires forum membership to view full question text/discussion threads; the
problems above were reconstructed and expanded from the publicly visible question titles/tags into
complete, solvable problem statements with original test cases and solutions.
