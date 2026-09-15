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
1. Resolve roots with an explicit **visited list**, not just "follow parent pointers until none" — that's
   exactly what turns a malformed cyclic feed into an infinite loop instead of a caught error, and the
   interviewer explicitly asks what happens on bad data.
2. **Cache resolved roots per company** the first time they're seen — loans commonly cluster under a
   handful of parents, so memoizing turns what would be O(loans × chain depth) into amortized O(loans +
   companies).
3. Keep "aggregate loans" and "apply transactions" as separate passes over separate state (grouped
   totals vs. a remaining-balance map) rather than mutating one shared structure — the interviewer's
   follow-up is almost always "now a transaction comes in for a loan that got merged into a different
   parent after the fact," which is far easier to reason about with the two concerns kept apart.

**Python Solution:**
```python
from collections import defaultdict


def resolve_root(company, parent_of):
    """Resolve `company` to its topmost ancestor. Raises ValueError on a cycle."""
    seen = []
    current = company
    while current in parent_of:
        if current in seen:
            raise ValueError(f"cycle detected involving {current}")
        seen.append(current)
        current = parent_of[current]
    return current


def group_loans_by_ultimate_parent(loans, parent_of):
    """
    loans: list of (loan_id, company, amount)
    parent_of: dict child -> direct parent
    Returns: dict root_company -> {"total": float, "loan_ids": [loan_id, ...]}
    """
    grouped = defaultdict(lambda: {"total": 0.0, "loan_ids": []})
    root_cache = {}
    for loan_id, company, amount in loans:
        if company not in root_cache:
            root_cache[company] = resolve_root(company, parent_of)
        root = root_cache[company]
        grouped[root]["total"] += amount
        grouped[root]["loan_ids"].append(loan_id)
    return dict(grouped)


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
            still_in = []
            for i in winners:
                burn = min(3, len(decks[i]))
                for _ in range(burn):
                    war_pot.append(decks[i].popleft())
                if decks[i]:
                    card = decks[i].popleft()
                    war_pot.append(card)
                    played[i] = card
                    still_in.append(i)
            pot = war_pot
            if not played:
                winners = still_in or winners[:1]
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

**Key Insights:**
1. **Net first, settle second.** Every transaction only matters through its effect on each account's net
   balance — the settlement step never needs to see the original transaction list, only the two lists of
   debtors and creditors it produces.
2. The heap-based **greedy pairing (largest creditor against largest debtor, repeat)** is what the
   `greedy, heap` tag is pointing at, and it's a reasonable, easy-to-explain answer — but it is **not
   always the true minimum** number of transactions (a known property of this problem class: some balance
   sets need genuine subset-partitioning, solvable optimally only via DFS/backtracking, to hit the
   absolute minimum). State that tradeoff out loud rather than claiming the greedy heap is provably
   optimal — Affirm's fintech framing rewards naming the limitation over a silently-wrong optimality claim.
3. Round every intermediate amount to the cent — settlement amounts are money, and floating-point drift
   compounding across repeated `pop`/`push` cycles on the heap is exactly the kind of bug a payments team
   would flag in code review.

**Python Solution:**
```python
import heapq
from collections import defaultdict


def minimize_settlements(transactions):
    """
    transactions: list of (payer, payee, amount)
    Returns: list of (payer, payee, amount) settlements using a greedy max-creditor/
    max-debtor heap pairing. Minimizes transaction count well in practice; not guaranteed
    globally minimal for every possible balance set (see Key Insights).
    """
    net = defaultdict(float)
    for payer, payee, amount in transactions:
        net[payer] -= amount
        net[payee] += amount

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

        remaining_debt = round(debt_amt - pay, 2)
        remaining_credit = round(credit_amt - pay, 2)
        if remaining_debt > 1e-9:
            heapq.heappush(debtors, (-remaining_debt, debtor))
        if remaining_credit > 1e-9:
            heapq.heappush(creditors, (-remaining_credit, creditor))

    return settlements
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
        self.history = defaultdict(list)  # txn_id -> [(ts, status, reason_code)]

    def process(self, ts, txn_id, reason_code, new_status):
        current = self.status.get(txn_id)
        if current is None:
            if new_status != "OPENED":
                raise ValueError(f"{txn_id}: first event must be OPENED, got {new_status}")
        elif new_status not in VALID_TRANSITIONS[current]:
            raise ValueError(f"{txn_id}: illegal transition {current} -> {new_status}")

        self.status[txn_id] = new_status
        self.history[txn_id].append((ts, new_status, reason_code))

    def average_resolution_time(self, reason_code):
        totals, count = 0, 0
        for txn_id, events in self.history.items():
            opened_ts = None
            for ts, status, rc in events:
                if status == "OPENED":
                    opened_ts = ts
                elif status in ("WON", "LOST") and rc == reason_code and opened_ts is not None:
                    totals += ts - opened_ts
                    count += 1
        return totals / count if count else 0.0
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
   here is the correct, simpler thing to *code live* in the time given — call out the trie/suffix-automaton
   approach verbally as the scalable alternative rather than trying to implement a suffix trie under
   interview time pressure.
3. Worst case (e.g. `"aaaa...a"`) is O(n²) substrings to hash — flag that complexity explicitly rather
   than letting it pass unmentioned; it's exactly the kind of tradeoff Affirm's fintech-pragmatic framing
   rewards naming.

**Python Solution:**
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
   attempt_count`) in a database, not just an in-memory timer — a scheduler process that only holds
   timers in memory loses every pending payment on crash.
2. **Polling/dispatch loop with a due-time index**: a worker polls `WHERE status = 'PENDING' AND due_ts
   <= now()` (indexed on `due_ts`), claims a batch (e.g., via `SELECT ... FOR UPDATE SKIP LOCKED` or an
   equivalent claim mechanism), and dispatches each to the payment-execution path.
3. **Idempotent payment execution**: every attempt carries an idempotency key
   (`loan_id:installment_id:attempt_count`) passed through to the payment processor, so a retried request
   after an ambiguous failure (timeout, not an explicit decline) can't double-charge — the processor itself
   de-dupes on that key.
4. **Outcome as an event, state transition as a side effect**: a successful/failed charge publishes an
   outcome event to a durable log; the schedule store's `status` is updated by a consumer of that event
   (not synchronously in the dispatch path), so downstream systems and the schedule store stay
   consistent even if one of them is temporarily down.
5. **Retry with backoff, bounded**: on a retryable failure (processor timeout, insufficient funds
   flagged as "try later"), re-enqueue with an incremented `attempt_count` and a pushed-out `due_ts`; past
   the configured max attempts, transition to a terminal `FAILED` status and hand off to dunning rather
   than retrying forever.

**Data Model (sketch):**
```
installments(loan_id, installment_id, due_ts, amount, status, attempt_count, idempotency_key)
payment_attempts(idempotency_key, installment_id, ts, outcome, processor_ref)  # append-only, audit trail
```

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
   a special recovery path.

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
- [Affirm System Design Interview: The Complete Guide — System Design Handbook](https://www.systemdesignhandbook.com/guides/affirm-system-design-interview/)

Note: 1point3acres' individual interview-report threads require forum membership to view full question
text/discussion; the problems above were reconstructed and expanded from publicly visible question
titles, tags, and summaries (and, where available, FastPrep's fuller public problem-catalog descriptions
sourced from the same firsthand reports) into complete, solvable problem statements with original test
cases and solutions.
