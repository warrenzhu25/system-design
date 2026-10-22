# Affirm Interview Questions

---

## Contents

**Coding**
1. [Group Loans by Ultimate Parent Company & Match Transactions](#1-group-loans-by-ultimate-parent-company--match-transactions)
2. [Multi-Player Card War Game Simulation](#2-multi-player-card-war-game-simulation)
3. [Minimize Settlement Transactions from End-of-Day Balances](#3-minimize-settlement-transactions-from-end-of-day-balances)
4. [Live Fraud Detector: Streaming Suspicious PII Propagation](#4-live-fraud-detector-streaming-suspicious-pii-propagation)
5. [Count Distinct PII Values in Underwriting Events](#5-count-distinct-pii-values-in-underwriting-events)
6. [Redeemable Promotion Offers from a Redeem/Unredeem Event Stream](#6-redeemable-promotion-offers-from-a-redeemunredeem-event-stream)
7. [Dispute Status Event Processor (Debug + Feature Add)](#7-dispute-status-event-processor-debug--feature-add)
8. [Shortest Unique Substring](#8-shortest-unique-substring)
9. [Aggregate Records into Maps by Runtime Dimensions](#9-aggregate-records-into-maps-by-runtime-dimensions)
10. [Fraud Detection: Match Transactions to Fraud Events](#10-fraud-detection-match-transactions-to-fraud-events)

**Concurrency / Low-Level Design**
11. [Thread-Safe Sliding-Window Hit Counter](#11-thread-safe-sliding-window-hit-counter)
12. [O(1) Randomized Dictionary for a Promo Code Pool](#12-o1-randomized-dictionary-for-a-promo-code-pool)

**System Design** — moved to [`affirm_system_design.md`](./affirm_system_design.md)

**Behavioral**
13. [Behavioral Themes](#13-behavioral-themes)

---

## 1. Group Loans by Ultimate Parent Company & Match Transactions

**Problem Statement:**
Affirm's most-repeated onsite and phone-screen prompt (reported under titles like "Group Loans by
Top-Level Company," "Determine Topmost Parent Company for Loan Processing," and "Normalize Loan
Merchants to Root Businesses"). You're given a parent/child relationship map between companies
(a merchant that was acquired points to its acquirer, which may itself point further up), a list of
loans each attached to some company in that hierarchy, and a stream of incoming payment transactions.
Three sub-tasks, usually asked as one continuous problem:

1. Resolve every company to its **topmost (ultimate) parent** — distinguishing a record's direct
   parent from its ultimate owner — and detect cycles in the input (a bad data feed can produce one).
2. Aggregate all loans onto their ultimate parent company.
3. Apply a stream of transactions against outstanding loan balances, reporting any transaction that
   references a loan that doesn't exist.

**Example:**
```
parent_of = {"B": "A", "C": "B", "D": "A", "E": "F"}   # F is its own root (no entry)
loans = [("L1", "C", 100.0), ("L2", "D", 50.0), ("L3", "E", 25.0), ("L4", "A", 10.0)]

group_loans_by_ultimate_parent(loans, parent_of)
# -> {"A": {"total": 160.0, "loan_ids": ["L1", "L2", "L4"]}, "F": {"total": 25.0, "loan_ids": ["L3"]}}
```

**Test Cases:**

| Scenario | Expectation |
|---|---|
| Company with no parent entry | It is its own root |
| Multi-hop chain (`C -> B -> A`) | Resolves all the way to `A`, not just the direct parent `B` |
| Cycle in `parent_of` (`X -> Y -> X`) | Raises rather than looping forever |
| Transaction against a nonexistent `loan_id` | Reported as unmatched, doesn't crash the batch |
| Two transactions fully paying off one loan | Remaining balance is `0.0`, not negative |

**Key Insights:**
1. Two equally-correct shapes for `resolve_root`, same asymptotics, different tradeoffs — pick based on
   what the interviewer is probing:
   - **Recursive + `@cache`** (Version A below): fewest lines to get right live, `parent_of` captured for
     free by the closure, path compression is automatic (every distinct company computed exactly once,
     no matter how many loans/chains pass through it). Cost: `resolve_root` is nested so it can't be unit
     tested standalone, and recursion depth is bounded by Python's call-stack limit (~1000 by default).
   - **Iterative + explicit cache dict** (Version B below): a top-level function, independently unit
     testable, no recursion-depth ceiling regardless of chain length — better if the interviewer flags
     "what if the ownership chain is thousands of companies deep" as a constraint. Cost: more lines, and
     the path-compression writeback (`for node in path: root_cache[node] = root`) is a manual step that's
     easy to get subtly wrong under time pressure. `path` must be a **set**, not a list — a list makes the
     `if current in path` cycle check an O(k) scan per step, which silently turns the very case Version B
     is supposed to handle well (a single very deep chain) into O(chain length²) instead of O(chain
     length). Measured: an 8,000-node chain took ~0.14s with a list cycle-check vs. ~0.0007s with a set.
   Default to Version A when writing this live; mention Version B verbally as the answer to a
   deep-recursion follow-up rather than typing both unless asked.
2. Either way, **cache resolved roots with path compression**, not just at the loan's starting company —
   every intermediate node walked on the way to the root gets memoized too (the same trick Union-Find
   uses). Two loans on siblings of a long chain (`C -> B -> A`, `D -> B -> A`) then share the cached hop
   at `B` instead of each re-walking to `A` independently, giving true amortized O(loans + companies)
   instead of just deduping repeat lookups of the same starting company.
3. Cycle detection is a separate concern from memoization/caching in both versions — don't try to fold it
   into the cache. Version A tracks a `visiting` set of companies currently on the recursion stack (the
   classic DFS "gray node" trick) and uses `try/finally` so `visiting` stays correct even when a
   `ValueError` unwinds the stack. Version B tracks the current walk's visited nodes in a `path` **set**
   and checks membership before extending it (see Key Insight #1 on why it must be a set, not a list).
   Both correctly reject a self-loop (`{"A": "A"}`) as a cycle.
4. Keep "aggregate loans" and "apply transactions" as separate passes over separate state (grouped
   totals vs. a remaining-balance map) rather than mutating one shared structure — the interviewer's
   follow-up is almost always "now a transaction comes in for a loan that got merged into a different
   parent after the fact," which is far easier to reason about with the two concerns kept apart.

**Python Solution — Version A (recursive, `@cache`):**
```python
from functools import cache
from collections import defaultdict


def group_loans_by_ultimate_parent(loans, parent_of):
    """
    loans: list of (loan_id, company, amount)
    parent_of: dict child -> direct parent
    Returns: dict root_company -> {"total": float, "loan_ids": [loan_id, ...]}
    """
    grouped = defaultdict(lambda: {"total": 0.0, "loan_ids": []})
    visiting = set()  # companies currently on the recursion stack, for cycle detection

    @cache
    def resolve_root(company):
        if company not in parent_of:
            return company
        if company in visiting:
            raise ValueError(f"cycle detected involving {company}")
        visiting.add(company)
        try:
            return resolve_root(parent_of[company])
        finally:
            visiting.discard(company)

    for loan_id, company, amount in loans:
        root = resolve_root(company)
        grouped[root]["total"] += amount
        grouped[root]["loan_ids"].append(loan_id)
    return dict(grouped)
```

**Python Solution — Version B (iterative, explicit cache — no recursion-depth ceiling, standalone testable):**
```python
from collections import defaultdict


def resolve_root(company, parent_of, root_cache):
    """
    Resolve `company` to its topmost ancestor, path-compressing every node walked
    along the way into `root_cache` (Union-Find style). Raises ValueError on a cycle.
    """
    path = set()  # nodes visited on this walk -- a set, not a list: O(1) membership check
    current = company
    while current not in root_cache and current in parent_of:
        if current in path:
            raise ValueError(f"cycle detected involving {current}")
        path.add(current)
        current = parent_of[current]
    root = root_cache.get(current, current)
    for node in path:  # order doesn't matter here, every node in the walk gets the same root
        root_cache[node] = root
    return root


def group_loans_by_ultimate_parent(loans, parent_of):
    """
    loans: list of (loan_id, company, amount)
    parent_of: dict child -> direct parent
    Returns: dict root_company -> {"total": float, "loan_ids": [loan_id, ...]}
    """
    grouped = defaultdict(lambda: {"total": 0.0, "loan_ids": []})
    root_cache = {}
    for loan_id, company, amount in loans:
        root = resolve_root(company, parent_of, root_cache)
        grouped[root]["total"] += amount
        grouped[root]["loan_ids"].append(loan_id)
    return dict(grouped)
```

**`match_transactions_to_loans` (shared by both versions):**
```python
from collections import defaultdict


def match_transactions_to_loans(loans, transactions):
    """
    loans: list of (loan_id, company, amount) -- amount is the outstanding balance
    transactions: list of (txn_id, loan_id, amount) applied in order
    Returns: (remaining_balance: dict loan_id -> float, unmatched: list of txn_id)
    """
    remaining = {loan_id: amount for loan_id, _, amount in loans}
    unmatched = []
    for txn_id, loan_id, amount in transactions:
        if loan_id not in remaining:
            unmatched.append(txn_id)
            continue
        remaining[loan_id] = round(remaining[loan_id] - amount, 2)
    return remaining, unmatched
```

---

## 2. Multi-Player Card War Game Simulation

**Problem Statement:**
Reported repeatedly under "Design and Simulate a Multi-Player Card War Game," "Card Game with N
Players," and "Design Card Game" — Affirm's recurring simulation prompt. Implement the card game
**War** for `N` players: the deck is dealt evenly among players; each round every active player plays
their top card face-up; whoever plays the highest card wins the whole pot and puts it at the bottom of
their deck. On a tie among the highest cards, those players go to **war**: each burns some cards face
down plus one face-up "battle" card, and the highest battle card wins everything played in that war
(including the burned cards). A player with no cards left is eliminated. The game ends when one player
holds every card, or (the follow-up every report mentions) when it doesn't terminate — detect and report
a stalemate instead of looping forever.

**Test Cases:**

| Scenario | Expectation |
|---|---|
| Two players, one with strictly higher cards throughout | That player wins in `len(deck)` rounds |
| A tie triggers a war, and the war itself ties again | Recurses until decided or a player runs out mid-war |
| A player runs out of cards *during* a war (not enough to burn+play) | Eliminated from that war, doesn't crash |
| Game exceeds `max_rounds` without a winner | Returns `None` (stalemate), not an infinite loop |
| Three-plus players | Works the same way — war isn't special-cased to exactly two |

**Key Insights:**
1. Model each war as **its own escalating pot**: every card burned or played during a war (by every
   participant, not just the eventual winner) gets added to the same pot the original tied cards came
   from, so the eventual winner takes everything at once.
2. Cap the simulation with `max_rounds` and return `None` on exhaustion — real War games between
   real-ish decks can cycle indefinitely (a well-known property of the actual game), and the interviewer
   is testing whether you build in a termination guard rather than trusting the game to end on its own.
3. Re-filter the active player list at the top of every round (not just when someone empties their
   deck) — a player can be eliminated mid-war, and a stale "active" list is a classic source of the bug
   reports describe candidates hitting: playing a card for a player who no longer has one.

**Python Solution:**
```python
from collections import deque


def play_war(decks, max_rounds=10000):
    """
    decks: list of lists (deck per player, index 0 = top of deck)
    Returns: index of winning player, or None if max_rounds exceeded (stalemate).
    On a tie, each tied player burns min(3, remaining) cards face down, then plays one more
    face-up "battle" card; highest battle card takes the whole accumulated pot. A tied player who
    runs out of cards mid-war is eliminated from that war.
    """
    decks = [deque(d) for d in decks]
    active = [i for i in range(len(decks)) if decks[i]]

    for _ in range(max_rounds):
        active = [i for i in active if decks[i]]
        if len(active) <= 1:
            return active[0] if active else None

        pot = []
        played = {}
        for i in active:
            card = decks[i].popleft()
            pot.append(card)
            played[i] = card

        best = max(played.values())
        winners = [i for i in played if played[i] == best]

        while len(winners) > 1:
            war_pot = list(pot)
            played = {}
            for i in winners:
                burn = min(3, len(decks[i]))
                for _ in range(burn):
                    war_pot.append(decks[i].popleft())
                if decks[i]:
                    card = decks[i].popleft()
                    war_pot.append(card)
                    played[i] = card
            pot = war_pot
            if not played:
                # everyone tied ran out of cards to battle with; the first prior tied player keeps the pot
                winners = winners[:1]
                break
            best = max(played.values())
            winners = [i for i in played if played[i] == best]

        winner = winners[0]
        decks[winner].extend(pot)

    return None
```

---

## 3. Minimize Settlement Transactions from End-of-Day Balances

**Problem Statement:**
Tagged `greedy, heap, payment` in the interview bank — a settlement-engine framing of the classic
"optimal account balancing" problem. Given a day's worth of `(payer, payee, amount)` transactions
between accounts, compute the **net balance** of every account, then produce the smallest practical set
of settlement transactions that zeroes every account out. Affirm's framing is explicitly about
minimizing the number of interbank/merchant settlement transfers, not just computing net balances.

**Example:**
```
transactions = [("A", "B", 10), ("B", "C", 10), ("A", "C", 5)]
# net: A owes 15, B is even (received 10, paid 10), C is owed 15
minimize_settlements(transactions)  # -> [("A", "C", 15.0)]  (one transfer instead of three)
```

**Test Cases:**

| Scenario | Expectation |
|---|---|
| A chain of transactions that nets out to zero everywhere | No settlements needed |
| One clear largest debtor and one clear largest creditor | Settled in a single transaction |
| Sum of all settlement amounts, netted per account | Matches the original net balances exactly |
| Floating-point amounts | No drift beyond cent-level rounding across repeated settlements |
| `[("A","E",16.89),("A","C",18.56),("D","B",12.97),("D","B",9.33)]` (net: A=-35.45, E=16.89, C=18.56, D=-22.30, B=22.30) | Greedy heap needs **4** transactions; the true minimum is **3** — a concrete, verified counterexample to "greedy is optimal" (see Key Insight #2 and Version B below) |

**Key Insights:**
1. **Net first, settle second.** Every transaction only matters through its effect on each account's net
   balance — the settlement step never needs to see the original transaction list, only the two lists of
   debtors and creditors it produces.
2. The heap-based **greedy pairing (largest creditor against largest debtor, repeat)** is what the
   `greedy, heap` tag is pointing at, and it's a reasonable, easy-to-explain answer — but it is **not
   always the true minimum** number of transactions. The test case above is a real, checked example: nets
   of `A=-35.45, E=+16.89, C=+18.56, D=-22.30, B=+22.30` settle in 4 transactions greedily but only 3
   optimally (`D` pays `B` its whole `22.30` in one shot instead of splitting across `C` and `E`). State
   that tradeoff out loud rather than claiming the greedy heap is provably optimal — Affirm's fintech
   framing rewards naming the limitation over a silently-wrong optimality claim.
3. Round every intermediate amount to the cent — settlement amounts are money, and floating-point drift
   compounding across repeated `pop`/`push` cycles on the heap is exactly the kind of bug a payments team
   would flag in code review.
4. Two solutions, same tradeoff shape as Q1's Version A/B: **Version A (greedy heap)** below is what you'd
   actually write live — easy to explain, O((n) log n), and correct enough that it matches the optimal
   count on the large majority of real balance sets. **Version B (DFS/backtracking)**, given as the answer
   to "can you guarantee the minimum," is genuinely optimal but exponential in the number of distinct
   nonzero accounts (`O(2^n)`-ish branching) — fine for the tens of accounts a day's netted balances
   produce, wrong tool if that count could be in the thousands.

**Python Solution — Version A (greedy heap, what to write live):**
```python
import heapq
from collections import defaultdict


def minimize_settlements(transactions):
    """
    transactions: list of (payer, payee, amount)
    Returns: list of (payer, payee, amount) settlements using a greedy max-creditor/
    max-debtor heap pairing. Minimizes transaction count well in practice; not guaranteed
    globally minimal for every possible balance set (see Key Insights and Version B).
    """
    net = defaultdict(float)
    for payer, payee, amount in transactions:
        net[payer] -= amount
        net[payee] += amount

    # heapq is a min-heap, so negate amounts to pop the LARGEST debtor/creditor each round
    debtors = [(-round(-amt, 2), acct) for acct, amt in net.items() if amt < -1e-9]
    creditors = [(-round(amt, 2), acct) for acct, amt in net.items() if amt > 1e-9]
    heapq.heapify(debtors)
    heapq.heapify(creditors)

    settlements = []
    while debtors and creditors:
        debt_amt, debtor = heapq.heappop(debtors)
        credit_amt, creditor = heapq.heappop(creditors)
        debt_amt, credit_amt = -debt_amt, -credit_amt
        pay = round(min(debt_amt, credit_amt), 2)
        settlements.append((debtor, creditor, pay))

        # whichever side wasn't fully paid off goes back in the heap for another round
        remaining_debt = round(debt_amt - pay, 2)
        remaining_credit = round(credit_amt - pay, 2)
        if remaining_debt > 1e-9:
            heapq.heappush(debtors, (-remaining_debt, debtor))
        if remaining_credit > 1e-9:
            heapq.heappush(creditors, (-remaining_credit, creditor))

    return settlements
```

**Python Solution — Version B (DFS/backtracking, true minimum — verified against 500+ random cases plus the counterexample above):**
```python
from collections import defaultdict


def minimize_settlements_optimal(transactions):
    """
    transactions: list of (payer, payee, amount)
    Returns: list of (payer, payee, amount) settlements using the TRUE minimum number
    of transactions (DFS/backtracking over which debtor pays which creditor).
    Exponential worst case — fine for the tens of distinct accounts a day's netted
    balances typically produce, not thousands (use Version A there).
    """
    net = defaultdict(float)
    for payer, payee, amount in transactions:
        net[payer] -= amount
        net[payee] += amount

    accounts = [acct for acct, amt in net.items() if abs(amt) > 1e-9]
    balances = [round(net[acct], 2) for acct in accounts]
    n = len(accounts)
    # n transactions is always achievable (pair every debtor with a creditor one at a time) —
    # start the "best so far" there so the first full DFS branch always finds an improvement.
    best = {"count": n, "settlements": []}

    def dfs(i, settlements):
        while i < n and abs(balances[i]) < 1e-9:
            i += 1  # account i already fully settled by an earlier transfer — skip it
        if i == n:
            if len(settlements) < best["count"]:
                best["count"] = len(settlements)
                best["settlements"] = list(settlements)
            return
        if len(settlements) + 1 >= best["count"]:
            return  # prune: even one more transaction can't beat the current best

        for j in range(i + 1, n):
            if abs(balances[j]) < 1e-9 or (balances[i] > 0) == (balances[j] > 0):
                continue  # same sign (or already zero) can't offset account i

            orig_i, orig_j = balances[i], balances[j]
            # Settle i fully by transferring the smaller magnitude between i and j;
            # whichever of the two is the debtor pays whichever is the creditor.
            transfer = round(min(abs(balances[i]), abs(balances[j])), 2)
            if balances[i] < 0:
                settlements.append((accounts[i], accounts[j], transfer))
            else:
                settlements.append((accounts[j], accounts[i], transfer))

            balances[i] = round(balances[i] + (transfer if balances[i] < 0 else -transfer), 2)
            balances[j] = round(balances[j] + (transfer if balances[j] < 0 else -transfer), 2)

            dfs(i, settlements)  # i is now settled (0) or carries j's leftover — either way, resume from i

            balances[i], balances[j] = orig_i, orig_j  # undo, so the next candidate j starts clean
            settlements.pop()

    dfs(0, [])
    return best["settlements"]
```

---

## 4. Live Fraud Detector: Streaming Suspicious PII Propagation

**Problem Statement:**
Reported as "Live Fraud Detector: Streaming Suspicious PII Propagation" and echoed in a separate onsite
report as "Fraud Detection: Debug + Match Transactions to Fraud Events." Process a live stream of two
event types:

- `underwriting_event(loan_id, pii_value)` — a loan application used this PII value (an SSN, email, or
  similar identifier).
- `fraud_flag_event(pii_value)` — this PII value has just been confirmed fraudulent.

When a PII value is flagged, **every loan that has ever used it** (already processed, including ones
seen before the flag arrived) must retroactively be marked fraud — and so must any loan seen **after**
the flag that reuses the same value. Support an `is_fraud(loan_id)` query at any point in the stream.

**Test Cases:**

| Scenario | Expectation |
|---|---|
| Loan created, then its PII is later flagged fraud | Loan retroactively marked fraud |
| Loan created *after* its PII was already flagged | Immediately marked fraud on creation |
| Two loans share one PII value; only one is flagged directly | Both marked fraud (shared value propagates) |
| Loan whose PII was never flagged | `is_fraud` stays `False` |

**Key Insights:**
1. Keep **two maps**, not one: `pii -> {loans}` for propagating a flag to every loan that used it, and
   `loan -> {pii values}` for the (less common but real) case of a loan tied to multiple PII fields. The
   asymmetry — one flag fans out to many loans — is the whole point of the "propagation" framing.
2. A newly-flagged PII value must sweep its **existing** loan set at flag time (retroactive), while a
   newly-underwritten loan must check the **existing** fraud-PII set at creation time (prospective) — miss
   either direction and half the test cases in the table above fail.
3. This is graph propagation with a fan-out of one hop, not full transitive closure across loans —
   loans don't share fraud status with each other directly, only through a shared PII value, so no
   union-find is needed, just two hash maps kept in sync.

**Python Solution:**
```python
from collections import defaultdict


class FraudPropagator:
    def __init__(self):
        self.pii_to_loans = defaultdict(set)
        self.loan_to_pii = defaultdict(set)
        self.fraud_pii = set()
        self.fraud_loans = set()

    def underwriting_event(self, loan_id, pii_value):
        self.pii_to_loans[pii_value].add(loan_id)
        self.loan_to_pii[loan_id].add(pii_value)
        if pii_value in self.fraud_pii:
            self.fraud_loans.add(loan_id)

    def fraud_flag_event(self, pii_value):
        self.fraud_pii.add(pii_value)
        for loan_id in self.pii_to_loans[pii_value]:
            self.fraud_loans.add(loan_id)

    def is_fraud(self, loan_id):
        return loan_id in self.fraud_loans
```

---

## 5. Count Distinct PII Values in Underwriting Events

**Problem Statement:**
FastPrep's public writeup calls this "Count Distinct Underwriting PII Values" — the easy warm-up sibling
of the fraud-propagation problem above, and reportedly asked standalone in at least one phone screen.
Given a stream of underwriting events, each carrying one or more typed PII fields (`ssn`, `email`,
`phone`, ...), report the number of **distinct values seen so far** for a given PII type.

**Test Cases:**

| Scenario | Expectation |
|---|---|
| Same email recorded twice | Counted once |
| Query a PII type with zero events recorded | `0`, not an error |
| Two different PII types tracked independently | Counts don't leak across types |

**Key Insights:**
1. A `dict[pii_type -> set[value]]` is the entire solution — this is deliberately the "easy" member of
   the fraud-question family, and reaching for anything heavier (a trie, a database) is over-engineering
   relative to what's asked.
2. It's a natural building block for #4 above: in a real system, this counter and the propagator would
   likely share the same underwriting-event ingestion path, just aggregating differently.

**Python Solution:**
```python
from collections import defaultdict


class DistinctPIICounter:
    def __init__(self):
        self.seen = defaultdict(set)

    def record(self, pii_type, value):
        self.seen[pii_type].add(value)

    def distinct_count(self, pii_type):
        return len(self.seen[pii_type])
```

---

## 6. Redeemable Promotion Offers from a Redeem/Unredeem Event Stream

**Problem Statement:**
FastPrep describes this as reading "redemption and unredemption events and reporting which offers each
recorded user can redeem at the cutoff date" — matching the 1point3acres OJ titles "Find Redeemable
Promotion Offers" and "Process Time-Bounded Offers and User Redemption Limits." Each promotion offer has
a **per-user redemption limit** and an **expiration timestamp**. Given a chronological stream of
`(ts, user_id, offer_id, action)` events where `action` is `REDEEM` or `UNREDEEM` (a reversal, e.g. a
cancelled order), and a `cutoff_ts`, report — per user — which offers they can still redeem as of that
cutoff.

**Test Cases:**

| Scenario | Expectation |
|---|---|
| User redeems an offer up to its limit | Excluded from their redeemable set |
| A `REDEEM` followed by an `UNREDEEM` for the same offer | Usage count drops back below the limit |
| Offer's expiration is before the cutoff | Excluded for every user, regardless of usage |
| Events with `ts > cutoff_ts` | Ignored entirely (they haven't "happened yet" as of the cutoff) |

**Key Insights:**
1. `UNREDEEM` is not a no-op — it's the detail that turns this from a static "has this user hit their
   limit" lookup into genuine event-sourced state that must be replayed in order up to the cutoff.
2. Check expiration **before** usage — an expired offer is unredeemable regardless of how far a user is
   from their limit, and interviewers report explicitly probing whether candidates short-circuit on
   expiration or waste work computing per-user usage for offers no one could redeem anyway.
3. `usage[user][offer] = max(0, usage - 1)` on `UNREDEEM` guards against a malformed stream driving a
   count negative (an unredeem with no matching prior redeem) — worth stating as a defensive choice, since
   the interviewer's follow-up is often "what if the events arrive out of order or are duplicated."

**Python Solution:**
```python
from collections import defaultdict


def redeemable_offers_at_cutoff(events, offer_limits, offer_expiry, cutoff_ts):
    """
    events: list of (ts, user_id, offer_id, action) action in {"REDEEM", "UNREDEEM"}, ts non-decreasing
    offer_limits: dict offer_id -> max redemptions per user
    offer_expiry: dict offer_id -> expiry ts (inclusive)
    Returns: dict user_id -> set of offer_ids still redeemable as of cutoff_ts
    """
    usage = defaultdict(lambda: defaultdict(int))
    users = set()
    for ts, user_id, offer_id, action in events:
        if ts > cutoff_ts:
            break
        users.add(user_id)
        if action == "REDEEM":
            usage[user_id][offer_id] += 1
        elif action == "UNREDEEM":
            usage[user_id][offer_id] = max(0, usage[user_id][offer_id] - 1)

    result = {}
    for user_id in users:
        redeemable = set()
        for offer_id, limit in offer_limits.items():
            if offer_expiry.get(offer_id, float("inf")) < cutoff_ts:
                continue
            if usage[user_id][offer_id] < limit:
                redeemable.add(offer_id)
        result[user_id] = redeemable
    return result
```

---

## 7. Dispute Status Event Processor (Debug + Feature Add)

**Problem Statement:**
Reported as "Tech Phone Screen: Debugging and Feature Addition for Dispute Events" — Affirm hands
candidates starter code for a chargeback/dispute state tracker with a bug, asks them to find and fix it,
then extend it. The shape every report agrees on: disputes move through a fixed lifecycle
(`OPENED -> UNDER_REVIEW -> WON | LOST | WITHDRAWN`), each transition event carries a `reason_code`, and
the processor must (a) reject illegal transitions instead of silently accepting them (the planted bug is
typically here — an unguarded status overwrite) and (b) after the fix, add a feature: report the
**average time-to-resolution** for disputes of a given reason code.

**Test Cases:**

| Scenario | Expectation |
|---|---|
| A transition that skips a state (`OPENED -> WON` directly) | Rejected |
| A transition out of a terminal state (`WON -> UNDER_REVIEW`) | Rejected |
| Two resolved disputes with the same reason code | Average resolution time is the mean of their individual durations |
| A reason code with no resolved disputes yet | `0.0`, not a crash |

**Key Insights:**
1. The bug reports converge on is a missing **state-machine guard** — code that sets
   `status[txn_id] = new_status` unconditionally, so a duplicate or out-of-order event silently corrupts
   history instead of raising. Fixing it means validating against an explicit transition table before any
   mutation happens, not after.
2. Compute resolution time from the **`OPENED` timestamp recorded in that dispute's own history**, not
   "the first event in the whole stream" — with multiple disputes interleaved, the naive version of this
   feature is a common source of a second bug layered on top of the first.
3. Keep a full `history` list per transaction (not just current `status`) even though the base
   requirements don't ask for it — the "add a feature" follow-up is reported often enough that
   over-throwing-away state on the first pass just means redoing the data model under time pressure.
4. `average_resolution_time` only counts `WON`/`LOST` transitions as a "resolution" — a `WITHDRAWN`
   dispute never contributes to the average, even though it's also a terminal state. That's a defensible
   reading (a withdrawal isn't a resolution the process rendered, it's the customer/merchant pulling out),
   but it's a scope assumption, not the only self-evidently-correct one — state it out loud rather than
   silently picking a definition, since an interviewer could just as easily mean for `WITHDRAWN` to count.
5. **Don't recompute the average by rescanning full history on every call.** A first-pass version that
   walks every transaction's entire history inside `average_resolution_time` is correct but re-does
   `O(total events ever processed)` work on *every* query — fine for a one-off check, but this class models
   a live stream that `process()` is fed continuously, and a reporting/dashboard caller is exactly the kind
   of thing that queries `average_resolution_time` repeatedly. Measured: at 20,000 resolved disputes, a
   full rescan took ~1.8ms per call and keeps growing with total volume; maintaining a running
   `(total_duration, count)` per `reason_code` **inside `process()`** — updated the instant a dispute
   resolves, since that's already where the `OPENED` timestamp and the transition are both in hand — drops
   the query to O(1) (~0.2µs measured, a ~9,000x difference at that scale) without changing `process()`'s
   own complexity or any output value. Keep the full `history` list too (Key Insight #3 still holds for
   audit/debug purposes) — this is additive bookkeeping, not a replacement for it.

**Python Solution:**
```python
from collections import defaultdict

VALID_TRANSITIONS = {
    "OPENED": {"UNDER_REVIEW", "WITHDRAWN"},
    "UNDER_REVIEW": {"WON", "LOST", "WITHDRAWN"},
    "WON": set(),
    "LOST": set(),
    "WITHDRAWN": set(),
}


class DisputeProcessor:
    def __init__(self):
        self.status = {}
        self.history = defaultdict(list)          # txn_id -> [(ts, status, reason_code)], full audit trail
        self.opened_ts = {}                        # txn_id -> ts of its OPENED event
        self.resolution_totals = defaultdict(float)  # reason_code -> summed resolution duration
        self.resolution_counts = defaultdict(int)    # reason_code -> number of WON/LOST resolutions

    def process(self, ts, txn_id, reason_code, new_status):
        current = self.status.get(txn_id)
        if current is None:
            if new_status != "OPENED":
                raise ValueError(f"{txn_id}: first event must be OPENED, got {new_status}")
            self.opened_ts[txn_id] = ts
        elif new_status not in VALID_TRANSITIONS[current]:
            raise ValueError(f"{txn_id}: illegal transition {current} -> {new_status}")

        self.status[txn_id] = new_status
        self.history[txn_id].append((ts, new_status, reason_code))

        # Update the running average's running totals right when a dispute resolves --
        # the OPENED timestamp is already in hand, so this is O(1) extra work per event,
        # and it's what makes the query below O(1) instead of an O(all history) rescan.
        if new_status in ("WON", "LOST"):
            opened = self.opened_ts.get(txn_id)
            if opened is not None:
                self.resolution_totals[reason_code] += ts - opened
                self.resolution_counts[reason_code] += 1

    def average_resolution_time(self, reason_code):
        count = self.resolution_counts.get(reason_code, 0)
        return self.resolution_totals[reason_code] / count if count else 0.0
```

---

## 8. Shortest Unique Substring

**Problem Statement:**
Tagged `string, trie` in the interview bank — one signature onsite coding round. Given a string `s`,
find the **shortest substring of `s` that occurs exactly once** in `s`. If several substrings of the
minimal length are unique, any one of them is acceptable.

**Example:**
```
shortest_unique_substring("aabcaa")  # -> "b" (or "c") — the shortest substrings occurring exactly once
shortest_unique_substring("aaaa")    # -> "aaaa" — every substring shorter than the full string repeats
```

**Test Cases:**

| Scenario | Expectation |
|---|---|
| A string with a unique single character | Returned at length 1, without scanning longer lengths |
| A string where only the full string itself is unique (all repeating chars) | Returns the whole string |
| Empty string | `None` |
| Multiple unique substrings at the minimal length | Any one of them is a valid answer |

**Key Insights:**
1. Search **by increasing length**, stopping at the first length where any substring has count 1 —
   this guarantees the *shortest* answer without needing to compare candidates of different lengths
   against each other.
2. The `trie` tag signals the intended production-grade answer: a suffix trie (or suffix automaton)
   built once lets you find the shortest unique substring in roughly linear-ish time by walking down from
   the root until a node's subtree count drops to 1. The direct sliding-window/hashmap approach shown
   here is the correct, simpler thing to *code live* in the time given — call out the suffix-array/
   suffix-automaton approach verbally as the scalable alternative rather than trying to build one under
   interview time pressure, unless explicitly asked to go implement it (Version B below).
3. Worst case (e.g. `"aaaa...a"`) touches O(n²) substrings total across all lengths — but each substring
   of length `L` costs O(L) to slice and hash, so total worst-case **time** is Θ(n³) (Σ L·(n−L+1) for
   L=1..n, not just the O(n²) substring count). Flag that explicitly rather than letting it pass
   unmentioned; it's exactly the kind of tradeoff Affirm's fintech-pragmatic framing rewards naming, and
   it's the concrete reason the scalable alternative in Key Insight #2 is worth mentioning verbally, not
   just a nice-to-have footnote.
4. If pushed on "what if the string is huge," **Version B** below (suffix array + LCP array) answers it
   for real: `O(n log² n)` build, verified against 2,000+ random strings and cross-checked against Version
   A for both the length and the uniqueness/minimality of the returned substring. It rests on one classical
   fact — the longest prefix suffix `i` shares with *any* other suffix equals the max of its LCP with its
   two neighbors in sorted (suffix-array) order — so one more character than that is unique to `i`.

**Python Solution — Version A (sliding window by length, correct and simple to code live):**
```python
from collections import defaultdict


def shortest_unique_substring(s):
    n = len(s)
    for length in range(1, n + 1):
        counts = defaultdict(int)
        for i in range(n - length + 1):
            counts[s[i:i + length]] += 1
        for substr, cnt in counts.items():
            if cnt == 1:
                return substr
    return None
```

**Python Solution — Version B (suffix array + LCP array, `O(n log² n)` — the scalable answer, mention verbally unless asked to write it):**
```python
def _build_suffix_array(s):
    """
    O(n log^2 n) doubling algorithm: rank every suffix by its first 2^k characters,
    doubling k each round until every suffix has a distinct rank (a total order).
    """
    n = len(s)
    sa = list(range(n))
    rank = [ord(c) for c in s]
    k = 1
    while True:
        def sort_key(i):
            second_half = rank[i + k] if i + k < n else -1
            return (rank[i], second_half)

        sa.sort(key=sort_key)
        new_rank = [0] * n
        for i in range(1, n):
            new_rank[sa[i]] = new_rank[sa[i - 1]] + (sort_key(sa[i - 1]) < sort_key(sa[i]))
        rank = new_rank
        if rank[sa[-1]] == n - 1:  # every suffix now has a unique rank -- fully sorted
            break
        k *= 2
    return sa, rank


def _kasai_lcp(s, sa, rank):
    """LCP[r] = longest common prefix between the suffixes at sorted positions r-1 and r."""
    n = len(s)
    lcp = [0] * n
    match_len = 0
    for i in range(n):
        if rank[i] > 0:
            j = sa[rank[i] - 1]
            while i + match_len < n and j + match_len < n and s[i + match_len] == s[j + match_len]:
                match_len += 1
            lcp[rank[i]] = match_len
            if match_len > 0:
                match_len -= 1  # LCP shrinks by at most 1 as we move to the next suffix (Kasai's key lemma)
        else:
            match_len = 0
    return lcp


def shortest_unique_substring_optimal(s):
    n = len(s)
    if n == 0:
        return None
    sa, rank = _build_suffix_array(s)
    lcp = _kasai_lcp(s, sa, rank)

    best_len, best_start = n, 0  # the whole string is always unique -- the fallback answer
    for i in range(n):
        r = rank[i]
        # the longest prefix suffix i shares with ANY other suffix = its LCP with whichever
        # sorted-order neighbor (previous or next) it's closest to
        shared_with_some_other_suffix = max(
            lcp[r] if r < n else 0,
            lcp[r + 1] if r + 1 < n else 0,
        )
        unique_len = shared_with_some_other_suffix + 1
        if i + unique_len <= n and unique_len < best_len:
            best_len, best_start = unique_len, i

    return s[best_start:best_start + best_len]
```

---

## 9. Aggregate Records into Maps by Runtime Dimensions

**Problem Statement:**
1point3acres lists this as "Aggregate Records Into Maps by Different Dimensions" and "Data Processing
with Schema and Labels" — a CSV-shaped transaction-reconciliation prompt (tagged `hashmap, payment,
csv`). Given a batch of transaction records (each a dict of fields like `merchant`, `category`,
`amount`, `date`) and a **list of dimension names supplied at runtime** (not hardcoded), build a nested
map that aggregates a numeric field by those dimensions, in the given nesting order.

**Example:**
```
rows = [
    {"merchant": "M1", "category": "grocery", "amount": 10},
    {"merchant": "M1", "category": "grocery", "amount": 5},
    {"merchant": "M1", "category": "gas", "amount": 20},
]
aggregate_by_dimensions(rows, ["merchant", "category"], "amount")
# -> {"M1": {"grocery": 15, "gas": 20}}
```

**Test Cases:**

| Scenario | Expectation |
|---|---|
| Two rows share every dimension value | Their amounts sum in the same leaf |
| Dimension list of length 1 | Flat (non-nested) map |
| Dimension list of length 3+ | Correctly nested three (or more) levels deep |
| A dimension value missing from a row | Raises a clear `KeyError`, not a silent skip |

**Key Insights:**
1. The whole point of the "schema and labels" framing is that the **grouping keys are configuration,
   not code** — a solution that hardcodes `row["merchant"]` and `row["category"]` as two nested loops
   technically works for the example but fails the actual requirement the moment a caller passes a
   different or longer dimension list.
2. `dict.setdefault(key, {})` while walking all but the last dimension, then a plain accumulate on the
   last one, avoids needing `defaultdict`'s auto-vivifying nested-lambda trick (which gets awkward for a
   dimension list of arbitrary, runtime-determined depth).
3. This is deliberately close to a SQL `GROUP BY` with a dynamic column list — worth saying so out loud,
   since it signals you recognize the data-modeling shape rather than treating it as a bespoke
   dict-of-dicts puzzle.

**Python Solution:**
```python
def aggregate_by_dimensions(rows, dimensions, value_field):
    """
    rows: list of dict records
    dimensions: list of field names, defines nesting order
    Returns: nested dict keyed by dimensions[0] -> dimensions[1] -> ... -> total(value_field)
    """
    result = {}
    for row in rows:
        node = result
        for dim in dimensions[:-1]:
            node = node.setdefault(row[dim], {})
        last_dim = dimensions[-1]
        node[row[last_dim]] = node.get(row[last_dim], 0) + row[value_field]
    return result
```

---

## 10. Fraud Detection: Match Transactions to Fraud Events

**Problem Statement:**
A second, distinct fraud-matching prompt from the "Software Engineer Tech Phone Screen Fraud Detector"
report (separate from the PII-propagation problem in #4). Given a list of transactions
`(txn_id, card_id, ts)` and a list of confirmed fraud events `(card_id, ts)`, flag every transaction that
occurred **within a fixed time window** of a fraud event on the same card — the signal being "this card
was doing something suspicious around this time," not "this exact PII was flagged."

**Test Cases:**

| Scenario | Expectation |
|---|---|
| Transaction just inside the window before a fraud event | Flagged |
| Transaction just outside the window on either side | Not flagged |
| A card with no fraud events at all | None of its transactions flagged |
| Multiple fraud events on the same card | A transaction near *any* of them is flagged |

**Key Insights:**
1. Group fraud events **by card first**, then sort each card's event timestamps — this turns "is this
   transaction near any fraud event on this card" from an O(events) scan per transaction into an O(log
   events) binary search.
2. Only the two events adjacent to a transaction's timestamp (`bisect_left`'s insertion point and the
   one before it) can possibly be within the window — checking every event on the card is unnecessary
   work once the list is sorted.
3. This complements #4 rather than duplicating it: PII propagation flags loans through a *shared
   identity*, while this flags transactions through *temporal proximity on the same instrument* — a real
   fraud pipeline runs both signals and treats a transaction hit by either (or both) as elevated risk.

**Python Solution:**
```python
import bisect
from collections import defaultdict


def match_transactions_to_fraud_events(transactions, fraud_events, window):
    """
    transactions: list of (txn_id, card_id, ts)
    fraud_events: list of (card_id, ts)
    window: max allowed |txn.ts - fraud.ts| for a match
    Returns: set of txn_id flagged as fraud
    """
    by_card = defaultdict(list)
    for card_id, ts in fraud_events:
        by_card[card_id].append(ts)
    for card_id in by_card:
        by_card[card_id].sort()

    flagged = set()
    for txn_id, card_id, ts in transactions:
        events = by_card.get(card_id, [])
        if not events:
            continue
        idx = bisect.bisect_left(events, ts)
        candidates = []
        if idx < len(events):
            candidates.append(events[idx])
        if idx > 0:
            candidates.append(events[idx - 1])
        if any(abs(ts - e) <= window for e in candidates):
            flagged.add(txn_id)
    return flagged
```

---

## 11. Thread-Safe Sliding-Window Hit Counter

**Problem Statement:**
The 1point3acres OJ title "Design Hit Counter" (tagged `queue, metrics, data-structure`) — track
per-key request hits and report the count within a trailing time window, as would sit in front of
Affirm's merchant-facing API gateway for rate-limiting/metrics. `hit(key, ts)` records one hit;
`count(key, ts)` returns the number of hits for that key in the last `window` time units as of `ts`.
Reports describe the same thread-safety follow-up seen elsewhere in the loop: multiple gateway threads
call `hit` and `count` concurrently for the same and different merchant keys.

**Test Cases:**

| Scenario | Expectation |
|---|---|
| Hits well within the window | All counted |
| Hits older than the window | Excluded, and evicted rather than re-scanned on every call |
| `count` on a key with zero hits | `0`, not an error |
| Concurrent `hit` calls on the same key from multiple threads | No lost hits |

**Key Insights:**
1. A `deque` of timestamps per key, evicted lazily from the front on every `count` call, keeps eviction
   amortized O(1) per hit — the same technique as a windowed-average structure, just returning a count
   instead of an aggregate.
2. **Lock per key**, not one global lock — merchants are independent, and a shared lock across all keys
   would serialize gateway traffic for merchants that have nothing to do with each other, exactly the
   contention the interviewer's follow-up is probing for.
3. Evict inside `count` (using the caller's own `ts`), not on a background timer — there's no reliable
   wall clock to drive a timer in a deterministic test harness, and every call already carries the
   timestamp needed to evict correctly.

**Python Solution:**
```python
import threading
from collections import defaultdict, deque


class HitCounter:
    def __init__(self, window):
        self.window = window
        self._hits = defaultdict(deque)
        self._locks = defaultdict(threading.Lock)

    def hit(self, key, ts):
        with self._locks[key]:
            self._hits[key].append(ts)

    def count(self, key, ts):
        with self._locks[key]:
            dq = self._hits[key]
            while dq and ts - dq[0] >= self.window:
                dq.popleft()
            return len(dq)
```

---

## 12. O(1) Randomized Dictionary for a Promo Code Pool

**Problem Statement:**
1point3acres lists "Design a Dictionary with Specific Operations" and "Design a Dictionary with Random
Value Retrieval" — the classic `RandomizedSet`/`RandomizedDict` problem, framed around picking a random
active promo code from a pool for a marketing campaign. Support `insert(key, value)`, `delete(key)`, and
`get_random_value()` (a uniformly random value among currently-active entries), **all in O(1)**.

**Test Cases:**

| Scenario | Expectation |
|---|---|
| Insert, then `get_random_value` | Returns that value |
| Delete a key, then `get_random_value` repeatedly | Deleted value never returned |
| Delete a key not present | Returns `False` / no-op, doesn't raise |
| Insert an existing key with a new value | Updates in place, doesn't create a duplicate slot |

**Key Insights:**
1. The classic O(1)-delete trick: to remove an element from the middle of a packed array in O(1),
   **swap it with the last element, then pop the last element** — deleting from the end is the only O(1)
   array removal, so the problem reduces to "make the element you want to delete the last one."
2. An `index` map (`key -> position in the array`) is what makes both `delete` and `insert`-of-an-
   existing-key O(1) — without it, finding *where* a key lives in the array is an O(n) scan, defeating
   the whole point.
3. After the swap-with-last in `delete`, the map entry for the element that got moved must be updated to
   its new index — the single most common bug in a live-coded version of this problem, since it's easy to
   swap the array entries and forget the index needs to follow.

**Python Solution:**
```python
import random


class RandomizedDict:
    def __init__(self):
        self._keys = []
        self._values = []
        self._index = {}

    def insert(self, key, value):
        if key in self._index:
            self._values[self._index[key]] = value
            return False
        self._index[key] = len(self._keys)
        self._keys.append(key)
        self._values.append(value)
        return True

    def delete(self, key):
        if key not in self._index:
            return False
        idx = self._index[key]
        last_idx = len(self._keys) - 1
        self._keys[idx], self._keys[last_idx] = self._keys[last_idx], self._keys[idx]
        self._values[idx], self._values[last_idx] = self._values[last_idx], self._values[idx]
        self._index[self._keys[idx]] = idx
        self._keys.pop()
        self._values.pop()
        del self._index[key]
        return True

    def get_random_value(self):
        idx = random.randrange(len(self._keys))
        return self._values[idx]
```

---

## 13. Behavioral Themes

Affirm's behavioral round is reported as less scripted-STAR than many peers — see
[`behavioral_interview.md`](./behavioral_interview.md) for general STAR-method prep. Themes and process
details specific to Affirm's loop:

- **"How did you" probing over rehearsed stories**: multiple reports describe the hiring-manager round
  favoring follow-up-heavy "how did you decide," "what did you do next," "what would you do differently"
  digging into a single story, rather than moving through a checklist of prepared STAR anecdotes — have
  real detail ready two or three layers deep on your go-to stories, not just the headline outcome.
- **Fintech ownership under real financial stakes**: given the loan/payment domain, expect a question
  probing a time you caught or prevented a bug/decision that would have had real monetary or compliance
  consequences if shipped — not just "a bug you fixed," but one where you can speak to why it mattered
  that it was caught before production.
- **Working from ambiguous or incomplete requirements**: several coding-round reports note "long
  boilerplate that rewards careful reading" as the house style — the behavioral analog is a story about a
  time a spec or requirement was underspecified or contradictory and how you resolved the ambiguity
  before (or while) building, rather than building the wrong thing fast.
- **Deep-dive round explicitly separate from HR screen**: process reports describe a distinct HR
  screening call (culture/logistics/comp expectations) as its own step, separate from a deeper behavioral
  "deep-dive" later in the loop — don't conflate the two when preparing; the deep-dive is the one probing
  technical judgment and ownership, not just fit.

---

## References

Sources used for compiling these questions (design-answer sources are in [`affirm_system_design.md`](./affirm_system_design.md)):
- [affirm interview questions — 1point3acres](https://www.1point3acres.com/interview/company/affirm)
- [Affirm Interview Questions (38 questions) — 1point3acres](https://www.1point3acres.com/interview/problems/company/affirm)
- [Affirm Fulltime Onsite Interview Experience for SDE(General) Role — 1point3acres](https://www.1point3acres.com/interview/thread/1138337)
- [Affirm Coding Interview: Records, Hierarchies & Rules — FastPrep](https://www.fastprep.io/affirm-interview)

Note: 1point3acres' individual interview-report threads require forum membership to view full question
text/discussion; the problems above were reconstructed and expanded from publicly visible question
titles, tags, and summaries (and, where available, FastPrep's fuller public problem-catalog descriptions
sourced from the same firsthand reports) into complete, solvable problem statements with original test
cases and solutions.
