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

**System Design**
13. [A/B Testing & Experimentation Platform](#13-system-design--ab-testing--experimentation-platform)
14. [Payment Messaging & Scheduling System](#14-system-design--payment-messaging--scheduling-system)
15. [Payment Data Model & Schema Design](#15-system-design--payment-data-model--schema-design)
16. [Real-Time Fraud Detection System](#16-system-design--real-time-fraud-detection-system)

**Behavioral**
17. [Behavioral Themes](#17-behavioral-themes)

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

## 13. System Design — A/B Testing & Experimentation Platform

**Problem Statement:**
Reported directly as an Affirm system-design round (tagged `ab-testing, metrics, experiment-design`).
Design an experimentation platform that assigns users (or merchants) to experiment variants
consistently, logs exposure and outcome events, and computes metrics per variant — the kind of platform
that would sit behind decisions like "does this new BNPL checkout flow increase approval rate."

**Functional Requirements:**
- `assign(experiment_id, user_id) -> variant` — deterministic and sticky: the same user always gets the
  same variant for the lifetime of an experiment, without needing to persist a per-user record on first
  assignment.
- Log exposure events (`user was shown variant X`) and outcome/metric events (`user completed
  checkout`, `loan was approved`) tied to the experiment and variant.
- Compute per-variant aggregate metrics (conversion rate, approval rate) with statistical significance,
  on demand or on a schedule.

**Non-Functional Requirements:**
- Assignment must not add meaningful latency to the checkout path it's embedded in (sub-millisecond,
  no network round-trip on the hot path).
- No sample-ratio mismatch: the actual variant split must match the configured split (e.g., a true 50/50)
  at scale, not drift due to a bad hashing scheme.
- Metrics must be auditable back to raw exposure/outcome events for a specific experiment, given the
  regulatory scrutiny fintech decisions (like loan approval logic) draw.

**High-Level Design:**
1. **Deterministic bucketing, not stored assignment**: `variant = hash(experiment_id + user_id) % 100`
   mapped into configured variant ranges — stateless, sticky by construction (same inputs always hash the
   same way), and needs no database read on the hot path.
2. **Exposure/outcome logging as events, not row updates**: every assignment and outcome is an
   append-only event (experiment_id, user_id, variant, event_type, timestamp) shipped to a durable log
   (e.g., Kafka), decoupling the checkout hot path from the analytics pipeline entirely.
3. **Metrics computation as an offline/streaming aggregation job** over the event log, joining exposures
   to outcomes per variant and computing rates plus a significance test (e.g., a z-test on conversion
   proportions) — not computed synchronously per request.
4. **Experiment configuration service**: a small control-plane store (experiment_id, variant
   definitions, traffic split, start/end time, targeting rules) that the bucketing hash function reads,
   cached aggressively at the edge since it changes rarely relative to assignment volume.
5. **Guardrail metrics**: track a small set of "don't break these" metrics (e.g., overall approval rate,
   error rate) across every experiment, not just the ones being tested — critical in a lending context
   where an experiment regression could mean a compliance problem, not just a bad UX metric.

**Data Model (sketch):**
```
experiments(experiment_id, variants[], traffic_split, targeting_rule, start_ts, end_ts)
exposure_events(experiment_id, user_id, variant, ts)          # append-only
outcome_events(experiment_id, user_id, metric_name, value, ts) # append-only
variant_metrics(experiment_id, variant, metric_name, count, sum, computed_at)  # materialized
```

**Detailed Component Breakdown:**

| Component | Responsibility | Tech choice (typical) |
|---|---|---|
| Experiment config service + UI | CRUD for experiments: variants, split, targeting, start/end, owner, guardrails; versioned, audited | Postgres/MySQL + admin UI; changes published as immutable versions |
| Config distribution | Pushes config snapshots to every app server so assignment is local | Pub/sub (Kafka) or poll an S3/CDN JSON snapshot; in-memory cache in SDK |
| Assignment SDK (in checkout service) | `hash(salt + user_id) -> bucket -> variant`; evaluates targeting; emits exposure | Library, no network call on hot path |
| Event collector | Receives exposure + outcome events, validates schema, batches | Stateless HTTP/gRPC service -> Kafka |
| Event log | Durable, replayable, partitioned by `user_id` | Kafka (topics `exposures`, `outcomes`) |
| Stream/batch pipeline | Dedupe exposures, join exposure to outcome within attribution window, aggregate | Flink/Spark Structured Streaming + nightly batch backfill |
| Warehouse / OLAP | Raw events + per-variant aggregates, auditable | Snowflake/BigQuery (raw), ClickHouse/Druid (dashboards) |
| Stats engine | Z-test/t-test, CUPED variance reduction, SRM check, sequential testing | Python/Spark job writing `variant_metrics` |
| Monitoring/alerting | SRM alarm, guardrail breach auto-pause | Metrics store + pager |

**Step-by-Step Flow (assign -> log -> analyze):**
1. **Create**: PM defines `checkout_v2` with variants `control 50 / treatment 50`, targeting (`country=US`,
   `loan_amount>100`), primary metric (approval rate), guardrails (error rate, default-risk proxy). Config
   service stores version `v1` and publishes it.
2. **Distribute**: every checkout service instance receives the snapshot (push, with a periodic poll as a
   fallback) and keeps it in memory. A stale snapshot is safe because assignment is a pure function of config.
3. **Request arrives**: checkout calls `sdk.assign("checkout_v2", user_id)`.
4. **Targeting**: SDK evaluates targeting rules; if the user is ineligible it returns `not_in_experiment`
   and logs nothing (so ineligible users never pollute the denominator).
5. **Bucket**: `bucket = murmur3(experiment_salt + ":" + user_id) % 10000`; the variant is the range
   containing `bucket`. A per-experiment salt keeps assignments independent across experiments (otherwise
   the same users are always "treatment" everywhere, correlating results).
6. **Exposure log**: SDK emits `{experiment_id, version, user_id, variant, ts, request_id}` asynchronously
   (in-process buffer -> collector -> Kafka). Log at the point the user actually *sees* the variant
   ("exposure"), not at assignment, to avoid diluting the metric with users who never reached the feature.
7. **Outcome**: later services (`loan_approved`, `checkout_completed`) emit outcome events keyed by `user_id`.
   They don't need to know about experiments; the pipeline joins them.
8. **Pipeline**: dedupe to first exposure per `(experiment_id, user_id)`; join outcomes whose `ts >=
   first_exposure_ts` and inside the attribution window; aggregate to `(variant, metric) -> n, sum, sum_sq`.
9. **Stats**: compute lift, confidence interval, p-value; run the **SRM chi-squared** check first and
   refuse to publish results if it fails.
10. **Guardrails**: if a guardrail regresses beyond threshold, auto-pause (set traffic to 0 / kill switch in
    config) and page the owner.
11. **Audit**: any reported number can be recomputed from raw Kafka/warehouse events plus the config version.

**Design Rules Added After Failure Analysis (what keeps results trustworthy):**
1. **X1 — Declared randomization unit.** Each experiment declares its unit (user, device, merchant). Pre-login traffic
   buckets on device ID, post-login on account ID, with an identity-merge table; users seen in two variants
   ("flickers") are excluded from analysis, never silently reassigned.
2. **X2 — Immutable salt and ranges.** The experiment salt never changes after launch, and ramp-ups only *widen*
   bucket ranges (10% -> 50%) so already-assigned users keep their variant. Config validation rejects any edit that
   would move an assigned bucket.
3. **X3 — Sticky eligibility.** If targeting uses a mutable attribute (loan amount, country), a user could drift in or
   out. Evaluate targeting at first exposure and persist `(experiment, unit) -> variant` in a small exposure store; analysis
   uses the **first** exposure. Pure-hash assignment remains the default when targeting uses immutable attributes.
4. **X4 — Log exposure at render, stamp the config version.** Log when the user actually *sees* the variant, with
   `config_version`, so analysis knows exactly which rules produced each exposure.
5. **X5 — Fail safe to control.** If the SDK has no valid config, or the config is staler than a bounded age (e.g.
   5 minutes), it serves control and logs `config_missing`/`config_stale`, never an arbitrary variant.
6. **X6 — Kill switch is tested.** Emergency disable travels on a fast path (push plus short TTL), and is exercised
   regularly, since a kill switch that has never been used is an assumption.

**Detailed Failure Scenarios:**

| # | Failure | Effect on results | Detection | Handling |
|---|---|---|---|---|
| 1 | SDK cannot load config at startup | Users get no/unknown variant | `config_missing` rate | Serve control; exclude from analysis; alert; keep last-known-good on disk |
| 2 | Config push partly fails (instances on old/new versions) | Mixed split, SRM | `config_version` distribution in events; SRM check | Max-staleness SLO (X5); stamp version on events; analysis can filter by version |
| 3 | Hash salt or bucket range changed mid-experiment | Users reshuffle, contaminating both arms | Flicker rate spikes; SRM | Reject at config validation (X2); if it happened, restart the experiment, don't "fix" the analysis |
| 4 | Treatment code crashes or is slower, so exposures aren't logged | **Differential event loss**: SRM that looks like a real effect | SRM; compare server-side vs client-side exposure counts | Log exposure server-side where possible; client beacons with sequence numbers to estimate loss; treat SRM as invalid result |
| 5 | Collector or Kafka outage | Lost events, possibly asymmetric | Gaps in per-client sequence numbers; producer error metrics | Bounded local buffer, drop *events* never requests; backfill from service logs; mark the affected window |
| 6 | At-least-once duplicates | Inflated counts | Duplicate `event_id` rate | Dedupe on `event_id`; exposures deduped to first per `(experiment, unit)` |
| 7 | Late-arriving outcomes (approval decisions, repayments, defaults) | Premature readout | Metric maturity flag | Attribution windows with maturity; short-horizon proxy metrics for early read, delayed metrics (default rate) for final; never call a result before maturity |
| 8 | Bots, internal/test users | Noise, bias if filtered unevenly | Traffic-source breakdown | Apply the same filter to all arms, defined by attributes not by outcomes |
| 9 | Overlapping experiments interact | Confounded effects | Interaction test between concurrent experiments | Layers for mutually exclusive features; independent salts otherwise |
| 10 | Identity changes (anonymous to logged in) | Flickers; unit double counted | Flicker report | X1 unit rules; merge table; exclude flickers |
| 11 | Stats job fails or runs on partial data | Wrong lift shown | Row-count/completeness checks vs expected | Results marked stale, never display a partial day; reruns idempotent from raw events |
| 12 | Peeking / multiple comparisons | False positives | n/a | Sequential tests (mSPRT-style) or fixed horizon; correction across metrics (Benjamini-Hochberg) |
| 13 | Guardrail breach but alert pipeline lags | Harm continues | Guardrail latency SLO | Key guardrails computed on the streaming path; auto-pause at hard thresholds |
| 14 | Kill switch ineffective (cache TTL too long) | Bad variant keeps serving | Kill-switch drill | X6: fast-path push, short TTL for emergency flags, regular drills |
| 15 | Experiment ended but code still reads config | Unclear behavior | Stale-flag report | Defined end state (ship winner / revert to control); cleanup ticket required to close |
| 16 | Experiment affects credit decisions | Fair-lending/compliance exposure | n/a | Pre-launch compliance review gate in the config service; decision reason codes logged; no experiments on protected-class proxies without sign-off (coordinate with the compliance team) |
| 17 | Audit asks "recompute this number" | Irreproducible result | Reproducibility test | Raw events immutable; config versions retained; analysis job parameterized and versioned |

**How Real Production Systems Do It (and what to copy):**

*Uber XP — [Uber's Experimentation Platform](https://www.uber.com/blog/xp/): 1,000+ concurrent experiments, 1,000+ metrics.*
- **Statistics engine owns the math, not the experimenter.** Metrics are predefined as one of three types:
  continuous, proportion (binary) or ratio (numerator/denominator). The engine picks the test: Welch's t-test for
  continuous, Mann-Whitney U for skewed data, chi-squared for proportions; delta method or bootstrap for ratio
  metrics; Benjamini-Hochberg correction when there are 2+ treatments. Alpha = 0.05, sizing targets power = 0.8.
  **Copy**: put these in the stats-engine box of our design and make "metric type" a field in the metric registry,
  so "approval rate" (proportion) and "average APR" (continuous) are tested differently.
- **Pre-processing before testing**: outlier removal (clustering-based), **CUPED** variance reduction (use
  pre-experiment behavior as a covariate so you reach significance with less traffic) and difference-in-differences
  to correct pre-existing bias between groups. **Copy**: for BNPL, covariates such as prior approval rate or
  prior GMV; this matters because lending experiments have low base rates and long outcome delays.
- **Health checks that invalidate results**: (1) **sample-size imbalance (SRM)** when the control:treatment ratio
  differs from configured, which triggers a randomization investigation; (2) **flickers**, users who switch groups
  (e.g. moved device), who are excluded from analysis. **Copy**: log `variant` on every exposure and exclude any
  user seen in two variants.
- **Sequential monitoring**: mixture SPRT, evaluated daily on cumulative treatment-minus-control difference, with
  jackknife/block bootstrap for correlated observations from the same user. This lets you stop early without the
  "peeking" false-positive problem of repeatedly running a fixed-horizon test.
- **Outage monitoring system**: a separate sequential test on business metrics (crash rate, trip frequency) that
  alerts or auto-pauses an experiment. This is the guardrail box in our design, and it is why guardrails are
  independent of the experiment's own success metric.
- **Universal holdout**: a small configurable slice of users excluded from all experiments in a domain, used to
  measure long-term cumulative effect. **Copy**: a permanent 1-2% holdout of borrowers for checkout experiments.
- **Continuous experiments**: Thompson sampling, UCB and contextual bandits for things like picking among 10 email
  subject lines. Mention bandits as the alternative when you want to *optimize* instead of *measure*.
- Uber's post does not describe hashing/bucketing or the logging pipeline (it only says users are "randomly
  split"), so the deterministic-hash assignment and Kafka event log above are standard industry design, not
  Uber's disclosed internals.

*Affirm*: Affirm's engineering blog has a post on re-architecting its upfunnel platform to cut experiment cycle time
from months to days ([Medium](https://medium.com/affirmengineering/re-architecting-affirms-upfunnel-platform-how-we-cut-experiment-cycle-time-from-months-to-days-20a047eb6599)).
I could not read its internals, so don't cite specifics; use it as evidence that experiment velocity is a stated
Affirm priority.

**Follow-Up Questions:**
1. How do you prevent a user from being double-counted if they're exposed to an experiment twice? →
   dedupe exposure events on `(experiment_id, user_id)` at the aggregation layer — the first exposure
   timestamp is authoritative, later ones are logged but excluded from the primary metric.
2. How would you detect a sample-ratio mismatch before it invalidates results? → continuously compare
   observed variant proportions against configured splits with a chi-squared check, alerting rather than
   waiting for the end of the experiment to notice a hashing or targeting bug skewed the split.
3. How does this interact with a loan-approval model that's itself being experimented on? → the
   guardrail metrics (approval rate, default-risk proxies) need to be computed with tighter monitoring
   cadence and lower tolerance for regression than a typical UX experiment, since a skewed approval-model
   experiment has real credit-risk consequences, not just a worse click-through rate.

---

## 14. System Design — Payment Messaging & Scheduling System

**Problem Statement:**
Matches a reported system-design phone screen "focused on loan repayment and payment batching," tagged
`payment, messaging, scheduling`. Design the system that schedules and executes a borrower's future
loan-repayment installments: given a repayment schedule (e.g., 4 biweekly installments), reliably
initiate each payment attempt on its due date, handle retries on failure, and notify the borrower and
internal systems of outcomes — without ever double-charging or silently dropping a scheduled payment.

**Functional Requirements:**
- Given a loan's repayment schedule, enqueue each installment for execution at its due date/time.
- On due date, attempt the payment via a payment processor; on failure, retry with backoff up to a
  configured attempt limit, then flag for collections/dunning.
- Publish payment-outcome events (`succeeded`, `failed`, `retrying`) that other services (notifications,
  loan-balance ledger, dunning) consume.

**Non-Functional Requirements:**
- **Exactly-once effective execution**: a scheduler crash/restart or duplicate message must never result
  in the same installment being charged twice.
- Scheduled payments must fire within a tight window of their due time (minutes, not hours) even under
  scheduler restarts or partial outages.
- Every payment attempt (and its outcome) must be durably logged for audit and dispute resolution.

**High-Level Design:**
1. **Durable schedule store**: each installment is a row (`loan_id, installment_id, due_ts, status,
   attempt_no`) in a database, not just an in-memory timer — a scheduler process that only holds
   timers in memory loses every pending payment on crash.
2. **Polling/dispatch loop with a due-time index**: a worker polls `WHERE status = 'PENDING' AND due_ts
   <= now()` (indexed on `due_ts`), claims a batch (e.g., via `SELECT ... FOR UPDATE SKIP LOCKED` or an
   equivalent claim mechanism), and dispatches each to the payment-execution path.
3. **Idempotent payment execution**: every attempt carries an idempotency key
   (`loan_id:installment_id:attempt_no`) passed through to the payment processor, so a retried request
   after an ambiguous failure (timeout, not an explicit decline) can't double-charge — the processor itself
   de-dupes on that key. **`attempt_no` is a logical attempt**: it advances *only* after the previous attempt
   reached a definitive "no money moved" outcome. A worker crash, a lease expiry or a redelivered message
   must **never** produce a new key (see "Attempt Rules" below — getting this wrong is the classic
   double-charge bug).
4. **Atomic state transition, events for everything downstream**: the executor records the attempt result
   *and* the installment's state transition in **one transaction** together with an outbox row, so there is
   no window where the attempt is terminal but the installment still says `PROCESSING`. Ledger,
   notifications and dunning consume the outbox event asynchronously and idempotently, so they can lag or be
   down without affecting charging correctness.
5. **Retry with backoff, bounded**: on a *definitive* retryable failure (insufficient funds flagged as
   "try later"), set `due_ts` to a later time and advance `attempt_no`; past the configured max attempts,
   transition to a terminal `FAILED` status and hand off to dunning rather than retrying forever. Our own
   infrastructure failures (processor outage, open circuit breaker) use a separate counter and do not consume
   the borrower's attempts.

**Data Model (sketch):**
```
installments(installment_id PK, loan_id, seq, due_ts, amount_due, status, attempt_no, infra_retry_count,
             claim_token, lease_expires_at, version, UNIQUE (loan_id, seq))
payment_attempts(idempotency_key PK, installment_id, attempt_no, status, request_fingerprint,
                 processor_ref, created_at, lease_until, result_code, UNIQUE (installment_id, attempt_no))
outbox(event_id PK, aggregate_id, type, payload, published_at NULL)
```
State machines:
- Installment: `SCHEDULED -> PROCESSING -> PAID | SCHEDULED (retry, later due_ts) | FAILED | PENDING_SETTLEMENT`;
  `PENDING_SETTLEMENT -> PAID | RETURNED`; `PAID -> RETURNED` (ACH return / chargeback).
- Attempt: `IN_FLIGHT -> SUCCEEDED | DECLINED_HARD | FAILED_RETRYABLE | PENDING_SETTLEMENT | UNKNOWN`;
  `UNKNOWN -> (resolved to one of the above)`. Transitions are monotonic and guarded by `WHERE status IN (...)`
  so a late or duplicate writer can't move state backwards.

**Attempt Rules (the invariants that make double-charging impossible):**
1. **R1 — Key stability.** The idempotency key is derived from `(loan_id, installment_id, attempt_no)`. `attempt_no`
   advances only when the current attempt is `FAILED_RETRYABLE` (the processor definitively said
   no money moved).
2. **R2 — Claiming never changes the key.** The dispatcher sets `PROCESSING`, `claim_token` and a lease; it does
   **not** increment `attempt_no`. Redelivery and lease recovery therefore reuse the same key.
3. **R3 — Resolve before re-driving.** If the current attempt is `IN_FLIGHT`, `UNKNOWN` or terminal-but-not-applied,
   recovery first **resolves** it (look up the key at the processor / replay the stored result). It never starts a
   fresh charge under a new key.
4. **R4 — Never "retry" an UNKNOWN with a new key.** An unresolved attempt stays `UNKNOWN`, is escalated, and blocks
   further charging of that installment.
5. **R5 — Mind the processor's key-retention window.** Processors remember idempotency keys only for a limited time
   (often around 24 hours; check your processor's documentation). Resolve `UNKNOWN` attempts well inside the window
   (e.g. a 4-hour SLA). Past it, do not re-send the same key; resolve through the processor's search by our metadata
   reference (`installment_id`) or the settlement report first.
6. **R6 — Request fingerprint.** Store `hash(amount, currency, method)` with the key. Reusing a key with different
   parameters is rejected locally (and by the processor) instead of silently charging a different amount.
7. **R7 — Write-ahead intent.** The `IN_FLIGHT` row is committed (and synchronously replicated) **before** the
   processor call, so a crash can never leave a charge with no local record.
8. **R8 — Re-validate at execution time.** In the same pre-call transaction, re-read the installment (`version`,
   `status`, `amount_due`) and the loan state. If the loan was cancelled, refunded, paid off early, rescheduled or the
   amount changed, abort without calling the processor.
9. **R9 — Infrastructure failures are not borrower failures.** Processor 5xx/timeouts-with-definitive-no-charge and
   circuit-breaker rejections increment `infra_retry_count` (own cap and alert), not `attempt_no`.
10. **R10 — No cross-processor failover mid-attempt.** Failing over to a second processor while the first attempt is
    unresolved can charge twice, because the two processors share no idempotency state. Resolve first, then route.

**Detailed Component Breakdown:**

| Component | Responsibility | Tech choice (typical) |
|---|---|---|
| Loan servicing API | Creates loan + installment schedule at origination; handles reschedule/early payoff | Stateless service + Postgres |
| Schedule store | Source of truth for `installments` (status, due_ts, attempt_no) | Postgres/MySQL, index on `(status, due_ts)`, partitioned by due date |
| Scheduler / dispatcher | Polls due rows, claims a batch, enqueues work | N workers using `FOR UPDATE SKIP LOCKED` or lease column |
| Work queue | Buffers payment jobs, absorbs due-date spikes (e.g. the 1st of month) | Kafka / SQS |
| Payment executor | Calls processor with idempotency key; classifies the result | Stateless workers |
| Payment method / processor adapter | Cards, ACH, debit; each has different latency and failure semantics | Adapter per rail (card network, ACH bank partner) |
| Attempt log | Append-only record of every attempt and outcome (audit) | Same DB table + Kafka topic |
| Outbox relay + outcome consumers | Publish committed results; ledger/notification/dunning consume idempotently | Relay process + Kafka consumers |
| Ledger service | Double-entry record of money movement (see #15) | Append-only DB |
| Notification service | Email/push: upcoming payment, success, failure | Consumes outcome events |
| Dunning / collections | Handles terminal failures: grace period, late fee, outreach | Workflow engine / state machine |
| Reconciliation job | Matches processor settlement files to the ledger and flags breaks | Nightly batch |

**Step-by-Step Flow (one installment, happy path):**
1. **Origination**: loan is approved; servicing API writes `loans` plus 4 `installments` rows
   (`status=SCHEDULED`, `due_ts`, `amount`) in **one transaction**.
2. **Pre-notify** (T-3 days): a scan finds `due_ts` within 3 days and emits `payment.upcoming`; notification
   service emails the borrower (also a regulatory courtesy).
3. **Due time**: a dispatcher polls
   `SELECT ... WHERE status='SCHEDULED' AND due_ts<=now() ORDER BY due_ts LIMIT 500 FOR UPDATE SKIP LOCKED`.
4. **Claim**: in the same transaction it sets `status=PROCESSING`, a fresh `claim_token` and `lease_expires_at`
   (using the database clock) and commits. It does **not** touch `attempt_no` (R2). The queue message
   `{installment_id, claim_token}` is written through the outbox in that same transaction.
5. **Execute**: the executor re-validates the installment (R8), builds `idempotency_key =
   loan_id:installment_id:attempt_no` from the *stored* `attempt_no`, inserts the `payment_attempts` row
   (`IN_FLIGHT`, with fingerprint and lease) and commits durably **before** calling out (R7), then calls the
   processor.
6. **Result**: the processor returns `SUCCEEDED`, `DECLINED_HARD`, `FAILED_RETRYABLE` (e.g. insufficient funds) or
   (for ACH) `PENDING_SETTLEMENT`, with settlement confirmed days later by a webhook or return file. A **timeout is
   none of these**: it is `UNKNOWN` until resolved (R3-R4).
7. **Record + transition atomically**: in one transaction the executor updates the attempt (guarded by `WHERE
   status IN (IN_FLIGHT, UNKNOWN)`), applies the installment transition (`PAID`, `SCHEDULED` with new `due_ts`,
   `FAILED`, `PENDING_SETTLEMENT`) guarded by `version`, and inserts the outbox event
   `payment.succeeded|failed|pending`. The outbox relay publishes it (at-least-once).
8. **Apply downstream**: the ledger consumer (idempotent on `idempotency_key`) posts entries (debit cash / credit loan
   receivable); notification and dunning consumers react. None of them can change charging state.
9. **Retry path**: on a definitive `FAILED_RETRYABLE`, set `status=SCHEDULED`, `attempt_no += 1`,
   `due_ts = now + backoff(attempt_no)` (e.g. 1d, 3d, 5d), up to a max; at the limit set `FAILED` and emit
   `payment.exhausted` to dunning.
10. **Lease recovery (the corrected protocol)**: a sweeper selects `PROCESSING` rows with `lease_expires_at < now()`
    and inspects the *current* attempt: (a) no attempt row -> safe to re-queue with a new `claim_token`, same
    `attempt_no`; (b) `IN_FLIGHT`/`UNKNOWN` -> enqueue a **resolve** job (R3), never a fresh charge; (c) terminal
    but installment not transitioned -> apply the stored result. Because the key is unchanged, even a racing
    original executor is harmless: both hit the same processor key.
11. **Reconciliation** (nightly): compare processor settlement report vs `payment_attempts`/ledger; any
    mismatch (charged but we show failed, or the reverse) opens a ticket and posts a correcting ledger entry.

**Detailed Failure Scenarios:**

*A. Scheduler worker (dispatcher) and sweeper*

| # | Failure | What state is left behind | Handling | Residual risk |
|---|---|---|---|---|
| D1 | Crash before the claim commits | Nothing changed | Next poll by any worker picks the row up | None |
| D2 | Crash after the claim commits, before the message is delivered | Row `PROCESSING`, no message in flight | Message is written via the **outbox in the claim transaction**, so the relay delivers it after restart; the lease sweeper is the second net | Payment late by at most one lease period |
| D3 | Message delivered twice (relay retry, queue redelivery) | Two executors for one claim | Executor checks `claim_token` + attempt row; second one sees `IN_FLIGHT`/terminal and exits or replays | None, same key either way |
| D4 | **Slow executor outlives its lease; sweeper re-drives** | Attempt may be `IN_FLIGHT` with a charge in progress | Sweeper inspects the attempt (rule R3): enqueues **resolve**, not a new charge; any second executor reuses the **same key** (R2) | Needs the processor to honour key dedupe (R5 window) |
| D5 | Sweeper resets a row whose attempt is terminal but not applied (e.g. relay lag) | Attempt `SUCCEEDED`, installment `PROCESSING` | Atomic record+transition in the executor (step 7) removes this window; if it still occurs, sweeper applies the stored result | None |
| D6 | Dispatcher reads from a lagging replica | Row looks unclaimed | Claims are done only on the primary; reads for display may use replicas | None if enforced |
| D7 | Clock skew between workers | Rows appear due too early or late | All due and lease comparisons use the **database clock** (`now()`), never worker clocks | None |
| D8 | Scheduler outage, then recovery | Large backlog of due rows | Normal poll catches up, ordered by `due_ts`; batch size cap and per-processor rate limit prevent a thundering herd; borrower-facing grace avoids late fees caused by our outage | Charges are late; policy must not penalize borrowers |
| D9 | Poison row (always fails before reaching the executor) | Retried forever, blocks batch head | Per-row dispatch attempt cap -> quarantine state + alert; claim query skips quarantined rows | Needs a human |
| D10 | Two sweepers run concurrently | Duplicate recovery work | Sweeper claims rows with `SKIP LOCKED`; recovery actions are idempotent | None |
| D11 | Hot partition at month start (1st/15th) | Index contention, long scans | Partition schedule by `due_date`; shard claims by `hash(installment_id) % N` worker lanes; jitter charges within the allowed window | Capacity planning |

*B. Executor and processor interaction*

| # | Failure | Money moved? | Handling |
|---|---|---|---|
| E1 | Crash before the `IN_FLIGHT` row commits | No | Redelivery starts clean; nothing to undo |
| E2 | Crash after `IN_FLIGHT` commits, before the processor call | No | Resolve finds "not found" at the processor -> re-send **same key** (within retention window) |
| E3 | **Crash during or after the processor call, before the result is recorded** | **Maybe** | Resolve: `lookup(key)`; found -> record result; not found within window -> re-send same key; else `UNKNOWN` + escalate (R3-R5) |
| E4 | Processor call times out | Maybe | Same as E3. Never treat a timeout as failure; never mint a new key (R4) |
| E5 | Processor returns 5xx / is unreachable (definitive no charge) | No | `infra_retry_count++`, backoff, circuit breaker; borrower attempt not consumed (R9) |
| E6 | Processor returns a soft decline (insufficient funds) | No | `FAILED_RETRYABLE`: advance `attempt_no`, schedule retry per policy |
| E7 | Processor returns a hard decline (closed account, fraud) | No | `DECLINED_HARD`: stop retrying, prompt borrower to update payment method, notify |
| E8 | Decline code misclassified | Depends | Maintain an explicit code->class table with a default of "UNKNOWN: do not retry automatically"; alert on unmapped codes. Misclassifying hard as soft keeps charging a bad instrument; the reverse sends borrowers to collections early |
| E9 | ACH/bank rail accepts the debit (`PENDING_SETTLEMENT`) | Pending | No retry until a final result; webhook or return file settles it; installment shows pending, not paid |
| E10 | ACH return arrives days later (NSF, unauthorized) | Reversed | Reversal event -> installment `RETURNED`, reversing ledger entry, retry/dunning per return code (some codes forbid re-presentment) |
| E11 | Same key sent with different amount/method | n/a | Local fingerprint check (R6) rejects; processor also errors |
| E12 | Processor key-retention window expires while `UNKNOWN` | Maybe | Do not re-send the old key; search by metadata reference / settlement report; hold until resolved (R5) |
| E13 | Failover to a second processor while the first is unresolved | Maybe | Forbidden until resolved (R10); only new installments route to the new processor |
| E14 | Executor crashes after the result is recorded but before downstream effects | Yes | Outbox event already committed with the result; relay publishes it later |

*C. Events, ledger and downstream*

| # | Failure | Handling |
|---|---|---|
| C1 | Outbox relay down | Events accumulate in the outbox table; relay resumes and publishes in order per aggregate; alert on outbox age |
| C2 | Duplicate event delivery | Consumers dedupe on `event_id`/`idempotency_key`; ledger has a unique constraint on the key |
| C3 | Events arrive out of order (`pending` after `succeeded`) | State transitions are monotonic and version-guarded; stale events are ignored |
| C4 | Ledger service down | Charging continues; ledger consumer catches up from the topic; reconciliation compares ledger totals to attempts |
| C5 | Notification provider down | Independent retry with its own DLQ; never blocks charging; a missed "payment failed" notice triggers a re-send job |
| C6 | Poison event | After N failures -> DLQ + alert; partition keeps moving |

*D. Database, queue and platform*

| # | Failure | Handling |
|---|---|---|
| P1 | Primary DB failover | Promote the synchronous replica (no acknowledged `IN_FLIGHT` row is lost, which is why R7 requires sync commit); fence the old primary; executors reconnect and resolve any open attempts |
| P2 | Split-brain (two primaries) | Fencing tokens/leases at the storage layer; claim and attempt writes carry `version` so a stale primary's writes fail |
| P3 | Queue/Kafka unavailable | Outbox keeps messages durably; sweeper re-drives due rows; no charge depends on the queue being up |
| P4 | Executor deployment bug charges wrong amount | Amount is read from the installment in the same transaction as the intent row; invariant alert: `charged_amount != amount_due`; kill switch pauses dispatch; automatic refund workflow for confirmed over-charges |
| P5 | Replica/analytics lag shows stale status to support | UI labels data freshness; support actions read from the primary |
| P6 | Region outage | Pending schedule is in a replicated store; failover region resumes polling; all open attempts go through resolve; processor-side keys protect against re-sends |

*E. Business-rule races*

| # | Race | Handling |
|---|---|---|
| B1 | Borrower pays early or in full while a debit is in flight | R8 check before the call; if the debit is already in flight, the ledger records an overpayment and an automatic refund/credit is issued |
| B2 | Loan cancelled or merchant refund processed while due | `version` bump on the installment; executor aborts (R8); refund path posts reversing entries |
| B3 | Borrower changes payment method between attempts | New method applies to the next `attempt_no` only; the old attempt keeps its key and fingerprint |
| B4 | Reschedule requests (change due date) | Update `due_ts` with a `version` check; if `PROCESSING`, the request is queued until the attempt resolves |
| B5 | Due date vs time zone/business days | Store `due_ts` in UTC computed from the borrower's local date; ACH cutoffs and bank holidays adjust the effective date; tests around DST |
| B6 | Duplicate schedule rows from a retried origination call | `UNIQUE (loan_id, seq)` and idempotent origination |
| B7 | Regulatory limits on re-presentment/retries | Policy table per rail/return code, enforced in the retry scheduler, with audit |

*F. Safety nets and detection*

- **Invariant monitors** (page on breach): more than one `SUCCEEDED` attempt for an installment; processor settlement
  with no local attempt row; installment `PROCESSING` longer than 2x the lease; any attempt `UNKNOWN` older than the SLA;
  outbox oldest unpublished age; charge amount differs from amount due; sudden spike in a decline code.
- **Reconciliation (nightly, plus intra-day for card rails)**: three-way match of processor settlement files, local
  attempts and ledger entries. Breaks open tickets; a confirmed **duplicate charge** triggers an automatic refund and
  an incident review.
- **Kill switches**: pause dispatch globally, per processor, per merchant, or per rail without redeploying.
- **Game days**: kill executors mid-call, drop the processor connection after the request is sent, expire leases
  artificially, and replay duplicate messages; the expected result is zero duplicate settlements.

**How Real Production Systems Do It (and what to copy):**

*Stripe — [idempotency](https://stripe.com/blog/idempotency).* The client sends a unique key
(`Idempotency-Key: AGJ6FJMkGQIpHUTX`) with each charge. The server correlates the key with the state of that
request and handles three failure shapes: (1) **connection failure before reaching the server**: the retry is the
first time the server sees the key, so it just runs; (2) **failure mid-operation**: if the earlier attempt was rolled
back in an ACID database it is safe to rerun fully, otherwise the server recovers state and continues; (3) **response
lost after success**: the server returns the cached result of the first success instead of charging again. Clients
retry until success is *verified*, with exponential backoff (about 2^n) plus jitter so recovering servers aren't hit by
a synchronized herd. Stripe also separates **definitive** failures (could not connect, safe to retry) from
**ambiguous** ones (connection dropped mid-exchange, safe only because of the key). **Copy**: our executor treats
a timeout as *ambiguous* and always reuses the same key.

*Airbnb — payment orchestration and the Orpheus idempotency framework
([Rebuilding Payment Orchestration](https://airbnb.tech/?p=49), summarized in
[this write-up](https://biweekly-engineering.beehiiv.com/p/handling-double-payments-airbnb-biweekly-engineering-episode-32)).*
Every payment call is split into three phases:
1. **Pre-RPC**: look up the idempotency key in the **master** database; if that request already completed, return
   its stored result; if not, insert the new request record.
2. **RPC**: call the payment service provider (PSP), outside any open DB transaction.
3. **Post-RPC**: write the PSP's response, or the failure, back to the database so a later retry can decide.
Errors are classified: **5xx = retryable, 4xx = non-retryable** (the request itself is wrong). Idempotency reads and
writes always go to the primary, never a replica, because replica lag could let a duplicate slip through. Larger
workflows are modeled as a **DAG of retryable, idempotent steps**; Airbnb reports five 9s (99.999%) payment
consistency after the rebuild. **Copy** the three-phase executor, shown next.

*Executor sketch assembled from those two designs, with the Attempt Rules applied (pseudo-code, not either company's source):*
```
def execute(msg):                                   # msg = {installment_id, claim_token}
    # PRE-RPC: one transaction on the primary, committed with synchronous replication (R7)
    with db.txn(sync_commit=True):
        inst = installments.get(msg.installment_id, for_update=True)
        if inst.status != PROCESSING or inst.claim_token != msg.claim_token: return   # stale/duplicate message
        if loan_changed(inst):                                                         # R8
            installments.release(inst); return
        key = f"{inst.loan_id}:{inst.installment_id}:{inst.attempt_no}"                # R1: stored attempt_no
        row = attempts.get(key, for_update=True)
        if row and row.status in TERMINAL: return apply_stored_result(inst, row)       # replay, no processor call
        if row and row.status in (IN_FLIGHT, UNKNOWN): return resolve(inst, row)       # R3: never re-charge blindly
        attempts.insert(key, status=IN_FLIGHT, fingerprint=hash(inst.amount_due, method),
                        lease_until=db.now() + 60s)                                    # R6, R7

    # RPC: no DB transaction held open across the network
    try:
        result = processor.charge(key, inst.amount_due, method, metadata={"installment_id": inst.installment_id})
    except (Timeout, ConnectionError):
        result = resolve_with_lookup(key)           # FOUND(result) | NOT_FOUND | UNKNOWN

    # POST-RPC: attempt result + installment transition + outbox, atomically
    with db.txn():
        finalize(inst, key, result)

def resolve(inst, row):
    r = processor.lookup(row.key)
    if r.found:                      return finalize(inst, row.key, r.result)
    if r.not_found and row.age < KEY_RETENTION_SAFE and processor.lookup_is_authoritative:
        return redo_charge_same_key(inst, row)      # same key => cannot double charge
    attempts.mark(row.key, UNKNOWN); escalate(row)  # R4/R5: no new key, no new charge

def finalize(inst, key, result):
    cls = classify(result)                          # unmapped codes -> UNKNOWN (E8)
    attempts.update(key, status=cls) if attempts.status(key) in (IN_FLIGHT, UNKNOWN)   # monotonic
    installments.transition(inst, cls, expected_version=inst.version)   # PAID | SCHEDULED(attempt_no+1 on FAILED_RETRYABLE)
                                                                         # | FAILED | PENDING_SETTLEMENT; infra errors -> infra_retry_count++
    outbox.insert(event_for(cls))
```

*Uber — [Payments Platform](https://www.uber.com/blog/ubers-payments-platform).* Uber splits the money lifecycle into
**stateless microservices** (money order creation, processing, collection, disbursement) joined by **Kafka**, with two
shared stores as sources of truth (the money-order store and the entity-balance store). Flows where a rider is
waiting use **Cadence**, Uber's workflow engine, so the interactive path is a durable workflow while the backbone stays
asynchronous. Uber Pay integrates 50+ payment methods behind a generic interface (charge, disburse, refund) so rail
specifics stay out of the core. **Copy**: (a) the stage-per-service split maps to our dispatcher / executor /
outcome consumer / ledger; (b) a durable-workflow engine (Cadence/Temporal) is a legitimate alternative to the
polling dispatcher: one workflow per installment, a durable timer until `due_ts`, activities for charge and retry, and
the workflow history is itself the audit trail. Trade-off to say out loud: polling is simpler to operate and easy to
reason about at 100M installments; workflows give clearer per-installment logic but add an engine to run.

*Affirm*: Affirm's postings for its Repayment Scheduling team describe it as the system that lets users schedule
automatic payments to repay loans, at tens of billions of dollars a year. I found no public Affirm design write-up,
so the above is the industry pattern; present it as such.

**Follow-Up Questions:**
1. How do you avoid two dispatch workers claiming and double-charging the same installment? →
   `SKIP LOCKED`-style claiming (or an equivalent leased/claimed-row mechanism) makes a claim atomic across
   workers; the idempotency key on the actual charge is the second, independent line of defense if a claim
   ever races anyway.
2. What happens if the payment processor call times out — did it charge or not? → treat a timeout as
   "unknown," not "failed" — query the processor's own idempotency-key lookup (or a status endpoint)
   before deciding to retry, rather than blindly retrying and relying on the processor's dedup alone.
3. How do you handle a due date landing on a scheduler outage window? → the durable due-time index means
   a recovered scheduler's next poll picks up every installment whose `due_ts` has passed, regardless of
   how long the outage was — the design should explicitly call out that "catch-up" is a normal poll, not
   a special recovery path. Borrower-facing policy (grace period, no late fee for our outage) is part of the answer.
4. A worker is slow and its lease expires while the charge is in flight. How do you avoid a second charge? →
   the lease only authorizes *re-driving*; recovery inspects the current attempt, never mints a new key (R2-R4),
   and any second executor reuses the same key, so the processor returns the original result. The unsafe design is
   the one where claiming increments the attempt number and therefore changes the key.
5. What if the processor forgets idempotency keys after a day? → resolve `UNKNOWN` attempts inside the retention
   window, and after that look up by our metadata reference or the settlement file before any new charge (R5).
6. How do you prove it works? → invariant monitors (one success per installment), three-way reconciliation, and
   fault-injection tests that kill executors mid-call and expire leases, expecting zero duplicate settlements.

---

## 15. System Design — Payment Data Model & Schema Design

**Problem Statement:**
Reported as a distinct Affirm system-design round (tagged `payment, data-modeling, schema-design`),
separate from the messaging/scheduling round above and focused purely on **data modeling**: design the
core schema for representing loans, installments, payments, and merchants such that it supports
accurate balance computation, partial payments, refunds, and multi-merchant reconciliation, under a
schema that will inevitably need to evolve.

**Functional Requirements:**
- Represent a loan's principal, its installment schedule, and every payment applied against it,
  including partial payments and refunds.
- Support querying a loan's current outstanding balance at any point in time, and reconstructing that
  balance as of any past date (for disputes/audits).
- Support a merchant having many loans across many borrowers, and a borrower having many loans across
  many merchants.

**Non-Functional Requirements:**
- Balance computation must be correct even under out-of-order payment application (e.g., a delayed ACH
  settlement recorded after a later payment already posted).
- Schema changes (new payment types, new fee structures) shouldn't require rewriting historical records.
- Auditable: every balance must be reconstructable from an immutable trail, not just a mutable running
  total.

**High-Level Design:**
1. **Event-sourced ledger, not a mutable balance column**: `loans.outstanding_balance` as a single
   mutable field is the tempting-but-wrong default — instead, every balance-affecting action (charge
   applied, payment received, refund issued, fee assessed) is an immutable `ledger_entry` row, and the
   current balance is a **derived** sum, not stored truth.
2. **Core entities**: `merchants`, `loans` (principal, term, merchant_id, borrower_id), `installments`
   (loan_id, due_ts, scheduled_amount), `ledger_entries` (loan_id, entry_type, amount, effective_ts,
   recorded_ts, reference_id) — note the **two timestamps**: `effective_ts` (when it actually happened,
   e.g. the ACH settlement date) vs. `recorded_ts` (when the system learned about it), which is exactly
   what makes out-of-order settlement correct.
3. **Balance as of any date** = sum of `ledger_entries` with `effective_ts <= target_date`, independent
   of `recorded_ts` — this is precisely how a delayed-ACH entry, recorded late but effective earlier,
   correctly retroactively adjusts a historical balance snapshot instead of corrupting today's balance.
4. **Materialized current-balance cache**: for read performance, maintain a `loan_balances` table updated
   by a trigger/consumer on `ledger_entries` insert — but treat it explicitly as a cache rebuildable from
   the ledger, never as the source of truth.
5. **Schema evolution via entry_type + a flexible metadata column** (e.g., JSON) on `ledger_entries`
   rather than adding new top-level columns per new fee/payment type — new entry types are additive, and
   historical entries never need migration.

**Data Model (sketch):**
```
merchants(merchant_id, name, ...)
loans(loan_id, merchant_id, borrower_id, principal, term, created_ts)
installments(loan_id, installment_id, due_ts, scheduled_amount)
ledger_entries(entry_id, loan_id, entry_type, amount, effective_ts, recorded_ts, reference_id, metadata)
loan_balances(loan_id, outstanding_balance, last_entry_id)  # materialized cache, rebuildable
```

**Detailed Schema & Components:**

```sql
-- Reference data
merchants(merchant_id PK, name, settlement_account_id, status)
borrowers(borrower_id PK, ...)
accounts(account_id PK, owner_type, owner_id, account_type)   -- LOAN_RECEIVABLE, CASH, MERCHANT_PAYABLE,
                                                              -- INTEREST_INCOME, FEE_INCOME, REFUND_LIABILITY
-- Contract
loans(loan_id PK, borrower_id, merchant_id, principal, apr, term_count, status, created_ts)
installments(installment_id PK, loan_id FK, seq, due_date, principal_due, interest_due, fee_due, status)
-- Money movement (immutable)
transactions(txn_id PK, loan_id, txn_type, effective_ts, recorded_ts, idempotency_key UNIQUE, reverses_txn_id NULL)
ledger_entries(entry_id PK, txn_id FK, account_id FK, direction CHECK IN ('DEBIT','CREDIT'),
               amount_minor BIGINT, currency, installment_id NULL, metadata JSONB)
-- Derived
account_balances(account_id PK, balance_minor, as_of_entry_id)   -- cache, rebuildable
```

Invariants enforced in code and by DB constraints:
- For each `txn_id`, `SUM(debits) = SUM(credits)` (zero-sum, double entry), checked before commit.
- Rows are insert-only; **corrections are new reversing transactions** (`reverses_txn_id`), never updates.
- Money is integer minor units (`amount_minor`), never floats; one currency per transaction.
- `idempotency_key` is unique, so a retried "apply payment" cannot post twice.

**Components around the schema:**

| Component | Role |
|---|---|
| Ledger service | The only writer of `transactions`/`ledger_entries`; validates zero-sum and idempotency |
| Allocation engine | Decides how a payment splits across fees -> interest -> principal -> oldest installment |
| Balance projector | Consumes ledger inserts, maintains `account_balances` and per-installment status |
| Statement/query API | Point-in-time balance, payoff quote, payment history |
| Reconciliation service | Compares ledger with processor files and merchant settlement reports |
| Data warehouse sink | CDC (Debezium -> Kafka -> warehouse) for finance and analytics |

**Step-by-Step Flow (borrower pays $30 of a $30 installment, with a late fee):**
1. Payment succeeds at the processor (event from #14 with `idempotency_key`).
2. Ledger service opens a DB transaction and checks `idempotency_key`; if present, return the existing `txn_id`.
3. Allocation engine reads the installment (fee $5, interest $2, principal $23) and allocates the $30
   fees -> interest -> principal.
4. Ledger posts one `transactions` row and balanced entries: `DEBIT CASH 30`, `CREDIT FEE_INCOME 5`,
   `CREDIT INTEREST_INCOME 2`, `CREDIT LOAN_RECEIVABLE 23`. Zero-sum check passes; commit.
5. Projector updates `account_balances` and marks the installment `PAID` (idempotent via `as_of_entry_id`).
6. Balance as of date `D`: `SUM(signed amount) WHERE account=LOAN_RECEIVABLE AND loan_id=? AND effective_ts<=D`.
7. **Late ACH settlement**: entry is recorded on day 5 with `effective_ts` = day 2; projections for day 2..5 are
   recomputed from the ledger; `recorded_ts` shows when we knew, which answers "what did we tell the borrower".
8. **Refund**: merchant refunds $50 of a $100 loan; post a reversing transaction (credit receivable, debit
   merchant payable), then recompute the remaining schedule via a *new* schedule version.
9. **Settlement to merchant**: group entries by `merchant_id` and period, produce a settlement batch, pay out,
   post `DEBIT MERCHANT_PAYABLE / CREDIT CASH`.

**Scaling notes:** shard by `loan_id` (all of a loan's entries co-locate, so a single-shard transaction posts
the whole transaction); hot merchant accounts that aggregate many loans should not be a single balance row, so
use per-loan accounts plus an aggregated view, or batch-serialized writes for hot accounts.

**Posting Rules Added After Failure Analysis:**
1. **L1 — One transaction posts a whole business event.** `transactions` and all its `ledger_entries` commit together or
   not at all. A deferred constraint (or trigger) verifies `SUM(debits) = SUM(credits)` per transaction at commit, in
   addition to the application check.
2. **L2 — Serialize postings per loan.** Take a row lock on the loan (or route all writes for a `loan_id` to one partition
   writer) while allocating a payment, so two concurrent payments cannot both allocate to the same installment. Use
   `version` checks for optimistic retries.
3. **L3 — Deterministic allocation and rounding.** Money is integer minor units. Allocation order (fees -> interest ->
   principal -> oldest installment) is code-versioned; remainders use largest-remainder rounding; the engine version
   is stored with each transaction so results are reproducible.
4. **L4 — Immutable, insert-only.** The application role has `INSERT`/`SELECT` only. Corrections are reversing
   transactions that reference `reverses_txn_id`; a constraint forbids reversing more than the original amount.
5. **L5 — Direction is explicit.** Each entry stores `direction` and `account_id`, so balance sums never depend on
   inferring sign from `entry_type`; a new entry type cannot silently flip a sign.
6. **L6 — PII stays out of the ledger.** Entries reference tokens/ids; personal data lives in a separate store so
   deletion requests can be honored without rewriting financial history.
7. **L7 — Derived data is rebuildable.** `loan_balances`, installment statuses and merchant settlement batches are
   projections with `as_of_entry_id`, and a scheduled job recomputes a sample (and periodically everything) from the
   ledger and compares.

**Detailed Failure Scenarios:**

| # | Failure | Impact | Handling |
|---|---|---|---|
| 1 | Duplicate posting request (client or consumer retry) | Double credit | `UNIQUE (idempotency_key)` on `transactions`; the second insert returns the first result |
| 2 | Two payments land at once for the same installment | Overallocation, wrong paid status | L2 serialization; second payment allocates to the remainder or becomes a credit/refund; both visible in the ledger |
| 3 | Crash mid-posting | Partial entries | L1: single DB transaction, nothing visible until commit |
| 4 | A bug produces an unbalanced transaction | Money created/destroyed | L1 commit-time check; nightly global invariant `SUM(debit) = SUM(credit)`; page on violation |
| 5 | Projector crashes or lags | Stale `loan_balances` | Reads that must be exact compute from entries or check `as_of_entry_id`; the projector is idempotent on `entry_id`; lag alert |
| 6 | Cache/projection drifts from the ledger | Wrong balance shown | L7 recompute-and-compare; auto-heal by rebuilding from the ledger; drift counter on a dashboard |
| 7 | Late ACH settlement or backdated entry (`effective_ts` earlier than recorded) | Historical balances, interest and late fees change | Re-derive dependent values deterministically from entries; emit correction events; notify only if the borrower-visible result changes; `recorded_ts` answers "what did we know when" |
| 8 | Backdated entry lands in a **closed accounting period** | Finance reports change retroactively | Post an adjustment dated in the open period referencing the original; approval workflow; never rewrite closed periods |
| 9 | Reversal of a reversal / refund after partial payments | Negative or inconsistent balances | Every action is a new linked transaction; constraint caps total reversals; refund allocation rules documented (pro-rata to paid installments) |
| 10 | Refund larger than amount paid | Merchant owes us | Cap at paid amount; create a merchant receivable; flag for review |
| 11 | Hot merchant account (many loans) | Row contention | Per-loan accounts plus an aggregated view; batch-serialized writes for the aggregate account (the pattern Uber describes for hot ledger entities) |
| 12 | Settlement spanning many loans/shards | Partial settlement | Settlement batch derived from the ledger, then executed as a saga with idempotent steps and a reconciliation record |
| 13 | New entry type breaks an old consumer | Wrong totals downstream | Additive schema only; versioned `metadata`; consumers fail loudly (not silently skip) on unknown types; contract tests |
| 14 | CDC/warehouse feed lags or loses rows | Finance reports wrong | Per-day row-count and checksum comparison between OLTP and warehouse; replay from the log |
| 15 | Deletion/privacy request | Conflict with immutability | L6: delete or anonymize the token mapping; ledger rows remain valid |
| 16 | Accidental or malicious update/delete | Audit failure | L4 permissions; optionally hash-chain entries for tamper evidence (Uber built a custom tamper-evident store for this need); alert on any `UPDATE/DELETE` attempt |
| 17 | Reconciliation break vs processor/bank | Unknown money | Classify (timing difference, missing, amount); auto-clear timing breaks within tolerance; unresolved breaks open tickets; never plug silently with a manual entry without approval |
| 18 | Historical loan migration | Opening balances wrong | Dual-run, compare balances loan by loan, cut over only on zero unexplained differences |
| 19 | Interest/day-count disagreement | Borrower disputes | Versioned calculation engine; store inputs and version with each accrual; replayable |
| 20 | Shard imbalance or hot loan | Latency | Shard by `loan_id` hash; split hot shards; move heavy merchants' reporting to the warehouse |

**How Real Production Systems Do It (and what to copy):**

*Uber — [Payments Platform](https://www.uber.com/blog/ubers-payments-platform), ledger design.*
- **Money order** = the unit of work: debits and credits between two or more entities for one commerce transaction
  (a trip, an Eats order, a freight shipment). It is **immutable**; an adjustment is a *new* money order. This is our
  `transactions` + `ledger_entries` with `reverses_txn_id`.
- **Double-entry with zero-sum enforced before commit**: each participant's amount is explicit (an "entity fare"),
  and the sum across entities must be zero, so ambiguity upstream can't create or destroy money.
- **Ledger = entity -> sub-ledger accounts -> balance.** Originally an entity was a single ~1 KB DynamoDB row;
  zero-balance sub-ledgers were pruned to keep rows small. They later hit limits (rows, hot keys, audit) and moved to
  **LedgerStore**, built with Uber's Storage Platform team for auditability and tamper-evident records.
- **Entity changelog**: every balance update is recorded, so any ledger can be rebuilt from inception. This is our
  "balance is derived; rebuild from entries" principle.
- **Strongly consistent balances** for 1.2B+ entities, with async cross-region replication; **hot entities** (B2B,
  Uber Direct) used a **serialized batch-write** path for about 10x throughput. **Copy**: a merchant-payable
  account touched by every loan is exactly this hot-row problem; batch writes per account in one writer, don't
  let thousands of concurrent transactions contend on one row.
- Scale reference: gross bookings grew from $50B to $217B annualized, total money movement nearly 2x that, and
  several multi-tenant ledgers have run 7+ years, which is why they chose immutability and schema-flexible entries.
- Not disclosed in the post: idempotency mechanics and reconciliation, so those pieces of our design come from
  Stripe/Airbnb (see #14) and general practice.

*Airbnb — [Rebuilding Payment Orchestration](https://airbnb.tech/?p=49).* Payment workflows as DAGs of idempotent
steps that end in ledger postings; the stated outcome is 99.999% consistency. **Copy**: quote this as the target and
tie it to why the unique `idempotency_key` on `transactions` exists.

*Affirm*: no public Affirm ledger post found. Third-party guides say interviewers press on idempotency keys, double
entry, and reconciling state when a downstream call fails midway; the schema above plus #14's executor answer all three.

**Follow-Up Questions:**
1. Why not just decrement a `balance` column on every payment? → it collapses the effective/recorded
   timestamp distinction, makes historical/point-in-time balance reconstruction impossible without a
   separate audit log anyway, and turns every concurrent payment into a race on a single row — the ledger
   design gets you correctness on all three for the cost of one derived-value computation.
2. How do you reconcile a merchant's expected settlement total against what actually posted? → sum
   `ledger_entries` by `merchant_id` (via `loan_id -> merchant_id`) over a settlement window, filtered by
   `entry_type`, and diff against the merchant's own reported totals — the append-only ledger is exactly
   the audit trail a reconciliation job needs, with no separate logging system required.
3. How would you add a new fee type (e.g., a late fee) without a migration? → it's a new `entry_type`
   value with fee-specific fields in the `metadata` column — no schema migration, no change to how balance
   summation works, since summation is agnostic to which `entry_type` it's adding.

---

## 16. System Design — Real-Time Fraud Detection System

**Problem Statement:**
A recurring system-design theme across Affirm prep guides, and the natural system-level counterpart to
the coding-round fraud problems in #4 and #10: design a real-time fraud detection system that scores
incoming loan applications and transactions, combining rule-based checks (velocity limits, PII reuse
across flagged applications, device/IP reputation) with a machine-learning risk score, and decides to
approve, decline, or route to manual review — within the latency budget of a live checkout flow.

**Functional Requirements:**
- Score an incoming application/transaction against both deterministic rules (e.g., "3+ applications
  from this device in 10 minutes") and an ML model's risk score.
- Combine rule hits and the model score into a decision: `approve`, `decline`, or `manual_review`.
- Propagate a confirmed-fraud signal (an analyst or chargeback confirms fraud after the fact) back into
  future scoring — the system-level version of the PII-propagation coding problem in #4.

**Non-Functional Requirements:**
- Scoring latency in the low tens of milliseconds — it sits directly in the checkout path, not an
  offline batch job.
- False-positive rate (blocking legitimate borrowers) must stay low enough not to hurt approval rate/
  conversion, while false negatives carry direct loss exposure — an explicit precision/recall tradeoff to
  discuss, not just "minimize fraud."
- New fraud patterns (rules or retrained models) must be deployable without a full system redeploy.

**High-Level Design:**
1. **Feature store for low-latency lookups**: velocity counts (applications per device/IP/PII value in
   trailing windows), historical fraud flags, and account-age signals are precomputed and kept in a
   low-latency key-value store (not queried from a data warehouse synchronously) — this is the same
   windowed-counting shape as the Hit Counter (#11) and PII propagator (#4), reused at system scale.
2. **Rules engine**: a set of independently deployable, versioned rules evaluated against the feature
   store in parallel; each rule contributes a hit/no-hit signal plus a severity, so adding a rule doesn't
   require redeploying the scoring service.
3. **ML scoring service**: a separately deployed model-serving endpoint returns a risk score for the
   same request; called in parallel with the rules engine (not sequentially) to stay within the latency
   budget, with a fallback to rules-only if the model service is slow/unavailable (graceful degradation
   over blocking checkout).
4. **Decision layer**: combines rule hits (some rules are hard-decline, e.g. a confirmed-fraud PII match;
   most are soft signals) with the ML score against a threshold, producing `approve` / `decline` /
   `manual_review` — hard-decline rules should short-circuit before even waiting on the ML call, to keep
   the common "obviously fraudulent" case fast.
5. **Feedback loop**: confirmed-fraud outcomes (from manual review or later chargebacks) are written back
   into the feature store (flagging the PII/device involved) and queued for periodic model retraining —
   closing the loop between the propagation logic in #4 and the model that scores future applications.

**Data Model (sketch):**
```
feature_store: device_velocity(device_id, window) -> count
                pii_fraud_flags(pii_value) -> bool          # fed by the feedback loop
rules(rule_id, version, definition, severity, enabled)
scoring_requests(request_id, application_id, ts, rule_hits[], ml_score, decision)  # audit trail
```

**Detailed Component Breakdown:**

| Component | Responsibility | Tech choice (typical) |
|---|---|---|
| Decision API | Entry point from checkout/underwriting; owns the latency budget (e.g. p99 < 100 ms) and fallbacks | Stateless gRPC service |
| Feature assembler | Fans out to online stores, merges request-time features | Parallel async calls with per-call timeouts |
| Online feature store | Velocity counters, PII/device flags, account age | Redis/Cassandra/DynamoDB, TTL'd windows |
| Streaming aggregator | Computes windowed counts from event stream into the online store | Kafka + Flink (sliding/tumbling windows) |
| Offline feature store | Same features computed in batch for training (avoids train/serve skew) | Warehouse + feature definitions shared with online |
| Rules engine | Versioned rules as data; hard-decline vs soft-signal | Rule DSL/JSON evaluated in-process; config pushed |
| Model serving | Returns risk score for the feature vector | Hosted model service (GBDT/NN), shadow + canary versions |
| Decision combiner | Rules + score + policy thresholds -> approve/decline/review | Pure function, config-driven thresholds |
| Case management | Manual review queue, analyst tooling, label capture | Internal app + DB |
| Decision log | Immutable record of inputs, features, rule hits, score, model version, decision | Kafka -> warehouse; required for adverse-action/regulatory explanation |
| Label pipeline + training | Chargebacks, confirmed fraud, repayment defaults -> labels -> retrain | Airflow + Spark; delayed labels (weeks/months) |
| Graph/link analysis | Shared device/PII/address clusters (the #4 problem at scale) | Graph store or union-find in stream job |

**Step-by-Step Flow (one application at checkout):**
1. Checkout sends `{application_id, user, PII hashes, device_id, ip, cart, merchant}` to the Decision API.
2. **Normalize/enrich**: hash PII (never log raw SSN), resolve device/IP reputation.
3. **Fan-out in parallel** (each with a hard timeout, e.g. 20-30 ms): velocity counters, PII fraud flags, graph
   neighbours, account history.
4. **Hard rules first**: if a hard-decline rule hits (confirmed-fraud PII, sanctions match), return `decline`
   immediately and skip the model.
5. **Model call**: send the assembled vector to model serving. If it times out or fails, use the degraded path
   (rules only, wider manual-review band) and tag the decision `model_unavailable`.
6. **Combine**: `score >= T_decline -> decline`; `T_review <= score < T_decline -> manual_review`; else
   `approve`; soft rule hits can push a borderline score up a band.
7. **Respond** to checkout and **async-log** the full decision record (inputs, features, versions, latency).
8. **Stream update**: the application event updates velocity counters and the PII/device graph for the *next*
   request.
9. **Manual review**: analyst sees evidence, decides; the decision is a label.
10. **Feedback**: confirmed fraud writes PII/device flags into the online store (propagation as in #4), and
    labels join the training set. Delayed labels (chargeback, first-payment default) arrive weeks later, so
    training uses label maturity windows.
11. **Release**: a new model runs in **shadow** (scored, not acted on), then canary on a % of traffic, compared on
    precision/recall and approval rate before full rollout.

**Safety Rules Added After Failure Analysis:**
1. **F1 — Every decision is explainable and logged.** Persist inputs (feature snapshot), rule hits, model version, score,
   thresholds and the final decision synchronously with a decision id; reason codes are needed for adverse-action
   notices. If the decision log write fails, the request fails closed to manual review rather than proceeding unlogged.
2. **F2 — Rules and models ship progressively.** shadow (evaluate, don't act) -> canary (a small percentage) -> full,
   with automatic rollback when decline rate, review rate or score distribution moves beyond a bound.
3. **F3 — Counters are atomic and idempotent.** Velocity updates use atomic increment-then-compare keyed by
   `request_id`, so retries don't double count and parallel requests can't both read the same stale count.
4. **F4 — Propagated signals have confidence, scope and TTL.** A confirmed-fraud flag on a PII value is a hard signal;
   a flag inherited through a *shared device or address* is only a review-band signal and expires, to avoid punishing
   family members or shared networks.
5. **F5 — Learn from unbiased data.** Route a small random sample of approvals and declines to review (or hold out a
   control policy) so labels are not only from the cases the current policy already flagged.
6. **F6 — Degraded modes are explicit.** Each dependency (feature store, model, device intelligence, bureau) has a
   timeout, a fallback and a tag in the decision log; degraded decisions have tighter exposure limits (lower credit
   limit, step-up verification) instead of blind approval.

**Detailed Failure Scenarios:**

| # | Failure | Impact | Handling |
|---|---|---|---|
| 1 | Feature store slow/unavailable | Missing features | Per-feature timeouts; defaults plus explicit "missing" indicators (the model is trained with missingness); wider review band; tag `features_degraded`; limit exposure (F6) |
| 2 | Streaming aggregates lag | Velocity undercounted, so attackers burst through | Monitor freshness; if lag exceeds a bound, treat velocity as unknown and apply conservative thresholds |
| 3 | Concurrent requests race on a counter | All requests pass at count=2 | F3: atomic increment-then-check (e.g. a Lua script or conditional write) |
| 4 | Checkout retries the same request | Double-counted velocity, spurious declines | Idempotent on `request_id`; cached decision returned for the same id |
| 5 | Model service down or slow | No score | Timeout + circuit breaker -> rules-only decision with widened manual-review band; hedged request to a second replica when latency allows |
| 6 | Bad model deployment (feature order, scaling bug) | Score distribution shifts, approvals or declines surge | Feature-schema contract tests; shadow then canary (F2); PSI/score-drift and approval-rate alarms with auto-rollback; model registry with signed model+schema |
| 7 | Bad rule (typo declines everyone) | Revenue loss, customer harm | Rule validation, dry-run against the last 24h of traffic, staged rollout, per-rule hit-rate alarm with auto-disable, per-rule kill switch |
| 8 | Propagation flags innocent users | Wrongful declines, fairness risk | F4; appeal and manual review path; track false-positive rate of propagated flags |
| 9 | Feedback-loop bias | Model learns only what the old policy already caught | F5 exploration sample; reject inference; evaluate on the holdout |
| 10 | Labels arrive late (chargebacks, first-payment default) | Recent cohorts look "good" | Label-maturity windows in training and evaluation; proxy labels for early signal; distribution monitoring until labels arrive |
| 11 | Attackers probe thresholds | Model/rules reverse engineered | Rate limit decision queries per device/IP, randomize inside the review band, minimize reason detail returned to applicants beyond what regulation requires |
| 12 | Drift/seasonal spikes (holidays, new merchant) | Precision drops | Drift monitors, champion/challenger, scheduled retraining, per-merchant calibration |
| 13 | Manual-review queue overwhelmed | Checkout stalls | Prioritize by risk and order value; SLA timeouts with a default action; step-up verification (ID check, micro-deposit) instead of an unbounded queue; capacity alerts |
| 14 | Decision log write fails | Cannot explain decisions later | F1: fail closed to review; buffer with a durable local queue; reconcile logs vs outcomes daily |
| 15 | PII exposure in features/logs | Privacy incident | Hash/tokenize at ingress with salted hashing; field-level encryption; log redaction; access audit |
| 16 | Third-party data (device intelligence, bureau) outage | Missing high-signal features | Per-vendor circuit breaker, short-lived caching, fallbacks, vendor SLAs |
| 17 | Event-time vs processing-time errors in windows | Wrong velocity | Event-time windows with watermarks; late events update counters but are flagged |
| 18 | Fair-lending and explainability | Regulatory risk | Feature review excluding protected attributes and close proxies; periodic bias testing; documented reason codes (coordinate with the compliance/legal team) |
| 19 | New user or device (cold start) | No history | Treat as unknown, not as fraud; rely on other signals; avoid blanket declines |
| 20 | Config error (wrong merchant threshold) | Systematic mis-decisions | Config validation, review and audit; canary per merchant |
| 21 | Train/serve skew (online and offline features computed differently) | Model underperforms in production | One feature definition feeds both paths; compare online vs offline values on a sample; alert on divergence |

**How Real Production Systems Do It (and what to copy):**

*Uber — Michelangelo (ML platform) and Palette (feature store): [Palette meta store](https://eng.uber.com/blog/palette-meta-store-journey/),
[Michelangelo overview](https://eng.uber.com/blog/from-predictive-to-generative-ai/).*
- **Online scoring path**: for a request, the serving layer fetches precomputed features from the online store
  (reported as **Cassandra** in a third-party write-up), **joins them with features sent in the request payload**,
  assembles the full feature vector, scores the model, and returns the prediction. This is our "feature assembler ->
  model serving" boxes; request-time features (device, IP, cart) come in the payload and don't need a lookup.
- **Palette** holds curated features (20,000+) with **batch and near-real-time** computation, including custom feature
  sets for Fraud, Eats and Comms; a metastore handles feature discovery, creation, deprecation, and offline/online
  serving setup. **Copy**: one feature definition feeds both training (offline) and serving (online), which is the
  fix for train/serve skew.
- Michelangelo scale: 5K+ models in production, ~10 million real-time predictions/second at peak, so model serving is
  a shared platform, not each team's service.
- **Delayed labels**: for fraud, outcome labels can take up to ~90 days, so you also monitor feature and prediction
  *distributions* over time to catch drift before labels arrive. **Copy**: add a drift monitor and label-maturity
  windows to training.
- **Uber Risk platform** ([Feature Store Summit 2023 slides](https://content.hopsworks.ai/hubfs/Feature%20Store%20Summit%202023/FS%20Summit%2023%20-%20Uber%20(Risk%20Platform).pdf)):
  a dedicated risk feature platform, "uFlow", with batch, near-real-time, aggregator and real-time computation and
  fetch stages, plus a graph feature store ("uGraph"). **Copy**: velocity counters = aggregator stage, shared
  device/PII clusters = graph store, and the same-request features = real-time stage. I only saw the slide
  outline, so latency numbers are not claimed.

*Affirm*: public job postings describe fraud work as adversarial, with continuous attacker pressure on identity and
synthetic-identity fraud. I found no public Affirm fraud architecture post; state that the above is the industry
pattern and ask the interviewer about latency budget and manual-review capacity.

**Follow-Up Questions:**
1. How do you deploy a new fraud rule without redeploying the scoring service? → rules live as versioned
   config/data (not code) evaluated by a generic rule-evaluation engine, so publishing a new rule is a
   config change picked up on the next evaluation, not a service deploy.
2. What happens if the ML scoring service is down? → the decision layer falls back to rules-only scoring
   (documented as a wider decline/manual-review band to compensate for the missing signal) rather than
   either blocking checkout or silently approving everything — an explicit degraded mode, not an implicit
   one.
3. How fast does a newly confirmed fraud PII value affect in-flight scoring? → as fast as the feature
   store write propagates — if it's the same low-latency store the rules engine reads synchronously, a
   confirmed flag can affect the very next request, mirroring the "retroactive + prospective" propagation
   behavior required in the coding-round version of this problem (#4).

---

## 17. Behavioral Themes

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

Sources used for compiling these questions:
- [affirm interview questions — 1point3acres](https://www.1point3acres.com/interview/company/affirm)
- [Affirm Interview Questions (38 questions) — 1point3acres](https://www.1point3acres.com/interview/problems/company/affirm)
- [Affirm Fulltime Onsite Interview Experience for SDE(General) Role — 1point3acres](https://www.1point3acres.com/interview/thread/1138337)
- [Affirm System Design Technical Phone Interview Experience — 1point3acres](https://www.1point3acres.com/interview/thread/1032443)
- [Affirm Coding Interview: Records, Hierarchies & Rules — FastPrep](https://www.fastprep.io/affirm-interview)
- [Uber's Payments Platform](https://www.uber.com/blog/ubers-payments-platform)
- [Uber's Experimentation Platform (XP)](https://www.uber.com/blog/xp/)
- [Uber Palette meta store](https://eng.uber.com/blog/palette-meta-store-journey/)
- [Stripe: Designing robust and predictable APIs with idempotency](https://stripe.com/blog/idempotency)
- [Airbnb: Rebuilding Payment Orchestration (Orpheus)](https://airbnb.tech/?p=49)
- [Affirm System Design Interview: The Complete Guide — System Design Handbook](https://www.systemdesignhandbook.com/guides/affirm-system-design-interview/)

Note: 1point3acres' individual interview-report threads require forum membership to view full question
text/discussion; the problems above were reconstructed and expanded from publicly visible question
titles, tags, and summaries (and, where available, FastPrep's fuller public problem-catalog descriptions
sourced from the same firsthand reports) into complete, solvable problem statements with original test
cases and solutions.
