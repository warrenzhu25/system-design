# VLL: A Lock Manager Redesign for Main Memory Database Systems

> VLDB Journal 24(5), 2015 (Special Issue Paper) — structured reading notes

Full paper content: [Markdown conversion](../original/vll-vldb-journal-2015.md)

## Paper information

- **Authors:** Kun Ren (Yale University), Alexander Thomson (Google), Daniel J. Abadi (Yale
  University)
- **Venue:** The VLDB Journal, Volume 24, Number 5, 2015, pages 681–705
- **DOI:** [10.1007/s00778-014-0377-7](https://doi.org/10.1007/s00778-014-0377-7)
- **Keywords:** lightweight locking, main memory, lock manager, deterministic, contention,
  scalability

## One-sentence summary

VLL replaces the classic lock manager — a hash table of per-record linked lists of lock requests —
with **two integers stored next to each record** plus **one global queue ordering transactions by
when they requested locks**, cutting locking overhead from ~22% to ~1.5%; the information lost by
this compression is reconstructed **only when the CPU would otherwise sit idle**, by an
optimization called **selective contention analysis (SCA)**.

## Problem

As main memory grows cheap, OLTP datasets live in RAM and disk I/O stops being the bottleneck.
**As a rule, when one bottleneck is removed, others appear** — and for main memory databases with
pessimistic concurrency control, the lock manager is the next one.

Prior measurements: **16–25% of transaction time is spent interacting with the lock manager** in a
main memory DBMS — and that study ran on a **single core with no physical contention for the lock
data structures**. Other studies show substantially larger overheads once transactions on multiple
cores compete for lock manager access. As cores per machine grow, this only gets worse.

### What a traditional lock manager actually does

The near-universal design (from System R onward) is:

```text
hash table:  primary key ──► lock head ──► request₁ ──► request₂ ──► request₃ ...
                             (mutex +          (linked list of lock requests)
                              lock state)
```

Every lock acquisition and release requires:

- a **hash table lookup**,
- a **mutex acquisition** on the lock head, since adding or removing list elements must happen in a
  critical section,
- and on every release, **a traversal of the linked list** to decide which request inherits the
  lock.

On disk-based systems these are negligible next to an I/O. In main memory, **"the additional memory
accesses, cache misses, CPU cycles, and critical sections invoked by lock manager operations can
approach or exceed the costs of executing the actual transaction logic."** Worse, as concurrency
rises the **per-lock request lists grow longer**, so the traversal cost on each release grows too —
exactly when you can least afford it.

### Why this matters *now* (the Calvin connection)

In partitioned distributed systems, the **distributed commit protocol (2PC) is normally the primary
bottleneck**, not the lock manager. But recent deterministic systems such as **Calvin** eliminate
2PC for distributed transactions, improving throughput by up to an order of magnitude — and
**thereby reintroducing the lock manager as the dominant bottleneck**.

Conveniently, Calvin **locks all data for a transaction at the very start of execution**, which
happens to be exactly VLL's precondition. The fit is not accidental — it is why the paper's
strongest results come from VLL inside Calvin.

## The two design changes

1. **Move lock information out of a central structure and colocate it with the data.** A tuple gets
   hidden attributes holding its row-level lock state, so **a single memory access retrieves both
   the data and its lock information in one cache line**.
2. **Remove all information about *which* transactions hold or want each lock.** Instead of a linked
   list of requests, keep **a pair of semaphores counting outstanding requests** — one for shared,
   one for exclusive.

Change 2 creates the central difficulty: **with no request list, how do you know which transaction
should inherit a lock when it is released?** The paper's key contribution is the answer: **force
every transaction to request all its locks at once, and order transactions by when they requested
them.** That global order tells you what to unblock.

## The VLL algorithm

### State

- Per record: an integer pair **`(Cₓ, Cₛ)`** stored **immediately preceding the record's value**,
  counting transactions requesting exclusive and shared locks. When nothing is accessing the record,
  both are 0.
- Per partition: a global **`TxnQueue`** tracking all active transactions **in the order they
  requested their locks**.

### Requesting locks

A transaction arriving at a partition attempts to lock **every record at that partition it will
ever access**. Each request simply **increments `Cₓ` or `Cₛ`**. Grant rules:

| Lock type | Granted if, after incrementing… | Because |
| --- | --- | --- |
| **Exclusive** | `Cₓ = 1` and `Cₛ = 0` | No other shared or exclusive lock is held |
| **Shared** | `Cₓ = 0` | No exclusive lock is held |

Requesting the locks **and** adding the transaction to the `TxnQueue` happen inside **the same
critical section**, so only one transaction per partition passes through this step at a time. To
keep that critical section short, **the transaction determines its full read and write set before
entering it** — which is not always trivial (see below).

### Free vs. blocked

On leaving the critical section, VLL classifies the transaction two ways: **local vs. distributed**
(does its read/write set span partitions?), and **free vs. blocked**:

- **Free** — acquired all locks immediately. **Executes right away.** On completion it decrements
  every counter it incremented and removes itself from the `TxnQueue`. A *distributed* free
  transaction may still have to wait for remote reads.
- **Blocked** — failed to acquire at least one lock. Tagged blocked in the `TxnQueue`, **not allowed
  to begin executing** until VLL explicitly unblocks it.

Note the queue is not strictly a queue: **a transaction need not be at the front when it is
removed.**

### The unblocking theorem — and why VLL is deadlock-free

The obvious approach — have a background thread scan blocked transactions and check whether their
counters have dropped to grantable values — **has a fatal flaw**: if another transaction entered the
queue and also incremented `Cₓ` for the same record, **both are blocked forever, because `Cₓ` will
always be at least 2.**

The resolution is a single observation:

> **A blocked transaction that reaches the front of the `TxnQueue` can always be unblocked and
> executed — no matter how large `Cₓ` and `Cₛ` are for the records it accesses.**

Why: because every transaction requests all its locks *and* enters the queue in one critical
section, **a transaction at the front implies every transaction that requested locks before it has
completed.** And every transaction that requested locks after it will be blocked if their sets
conflict.

Two consequences:

1. **Every transaction in the queue eventually becomes unblockable** — the front can always run, so
   the queue always drains.
2. **There is no deadlock within a partition, ever**, regardless of workload. This is a structural
   property of the lock acquisition order, not something detected and repaired.

A blocked transaction therefore has **two paths to unblocking**: reach the front of the queue, or
become the only remaining transaction in the queue that requested locks on each of its keys.

### Bounding the queue

As the `TxnQueue` grows, **the probability that a new transaction acquires all its locks
immediately falls**, since it must avoid conflicting with *every* transaction in the entire queue.
So VLL caps queue occupancy: past a threshold the system **stops accepting new transactions and
redirects processing resources to finding transactions it can unblock**.

The threshold needs to be tuned by contention level — high-contention workloads need a smaller
queue, low-contention workloads tolerate a longer one. The elegant fix: **set the threshold by the
number of *blocked* transactions rather than the total queue size**, since high-contention workloads
reach that count sooner automatically. The parameter tunes itself.

## Three implementation variants

### Colocated vs. arrayed VLL

**Colocated** (the default) stores `Cₓ` and `Cₛ` inside the record. Because they are **simple
integers rather than linked lists, they are easy to embed**, and one memory request brings both
record and lock state into cache.

The disadvantage: **it spreads lock information across the entire dataset.** For code that touches
*only* lock information, that is exactly wrong. The motivating case is **Calvin, which sometimes
runs its lock manager in a dedicated thread that never touches raw data** — that core's cache should
hold lock data, not be polluted with records.

**Arrayed VLL** therefore stores all `Cₓ` values in one vector and all `Cₛ` values in another, with
element *i* corresponding to record *i*. Slightly more overhead in the general case (two separate
requests for data and lock state), but **preferable when the lock manager runs in its own thread**,
especially when the record count is small or access is skewed.

### Single-threaded VLL

For H-Store-style deployments where **data is partitioned across cores with one thread per
partition**, running the general algorithm with one thread would degrade into **serial execution** —
and worse, the thread would sit **asleep waiting for remote reads** during distributed transactions,
making no progress at all.

VLL adds a **third state: "waiting"** — a transaction that began executing but cannot finish without
an outstanding remote read result. On entering it, the thread **sets the transaction aside and looks
for another one to execute**; when hunting for work it considers the front of the `TxnQueue`, new
requests, **and any waiting transaction whose remote results have arrived**.

So one thread now works on multiple transactions at once, switching instead of sleeping — while
**retaining H-Store's advantage of needing no latches or critical sections** around lock
acquisition.

## Two impediments to acquiring all locks at once

VLL's deadlock-freedom depends on acquiring all locks together in a critical section. Two things
make that hard:

**1. The read/write set may be unknown before running the transaction** — e.g. a transaction that
updates a tuple found through a secondary index lookup.

*Solution:* before entering the critical section, let the transaction **perform whatever reads it
needs at no isolation** to discover what it will access (do the index lookups). Then enter the
critical section and request the locks it expects to need. If during execution it discovers it
lacks a lock it needs — say the secondary index changed right after the exploratory read — **the
transaction aborts, releases its locks, and resubmits itself as a completely new transaction.**

**2. Different partitions may order transactions differently**, since each has its own `TxnQueue`
and its own local critical section. This permits **distributed deadlock**: one partition grants all
locks and activates a transaction while that same transaction sits blocked in another partition's
queue.

Two candidate solutions, both implemented and measured:

| Approach | Verdict |
| --- | --- |
| **Allow distributed deadlock, detect and abort** | **Problematic under high contention** — "the overhead of handling and detecting distributed deadlock completely negates the VLL advantage of reducing the overhead of lock management" |
| **Coordinate across partitions so multi-partition transactions enter every `TxnQueue` in the same order** | Adds nontrivial coordination overhead, but **still yields improved performance** |

For low-contention workloads either works. The paper uses the second, and observes that
**deterministic systems like Calvin already establish a global transaction order *before* execution
begins** — so the coordination cost is already paid. Hence "the integration of VLL and deterministic
database systems seems to be a particularly good match."

## The trade-off: what VLL gives up

VLL "compresses a standard lock manager's linked list of lock requests into two integers." The
price is **lost concurrency information**. A traditional manager inspects request queues to decide
whether a lock can be granted; VLL can only test two far weaker predicates:

- (a) is this the **only** lock in the queue, or
- (b) is it **so old** that no other transaction could possibly precede it in any lock queue?

So transactions frequently **cannot run even though they "should" be able to**. The paper's worked
example:

| Transaction | Write set |
| --- | --- |
| A | x |
| B | y |
| C | x, z |
| D | z |

With A and B executing, C conflicts with A on `x` and D conflicts with C on `z`, so both are queued
blocked:

```text
     VLL                            Standard
 key  Cx  Cs                    key  request queue
  x    2   0                     x   A, C
  y    1   0                     y   B
  z    2   0                     z   C, D
 TxnQueue: A, B, C, D
```

Now **A completes and releases its locks**:

```text
     VLL                            Standard
 key  Cx  Cs                    key  request queue
  x    1   0                     x   C
  y    1   0                     y   B
  z    2   0                     z   C, D
 TxnQueue: B, C, D
```

**A standard lock manager sees C at the head of all its request queues and knows C can run. VLL
cannot tell.** At low contention this costs little; under high contention — and especially with
distributed transactions — **VLL's CPU utilization suffers badly**.

## Selective contention analysis (SCA)

SCA **simulates the standard lock manager's ability to detect which transactions should inherit
released locks** — but spends the work **only when CPUs would otherwise be idle** (the queue is full
and no obviously unblockable transaction exists). So **VLL selectively increases its lock management
overhead when, and only when, it is beneficial.**

The insight that makes it cheap: any blocked transaction conflicted with something ahead of it *at
the time it was queued* — but those transactions may since have completed. **The i-th transaction in
the queue can now conflict with at most (i−1) prior transactions**, whereas when it was queued it
had to contend with up to `TxnQueueSizeLimit` of them. So **transactions near the front are much
less likely to be *actually* blocked.**

The algorithm scans from the front, maintaining two bit arrays **`Dₓ` and `Dₛ`, each 100 kB — chosen
so both fit inside a 256 kB L2 cache** — initialized to zero, with the invariant after scanning the
first *i* transactions:

- `Dₓ[j] = 1` iff some scanned transaction's **write** set hashes to *j*
- `Dₛ[k] = 1` iff some scanned transaction's **read** set hashes to *k*

Then the next transaction `T_next` can safely run if:

- `Dₓ[hash(key)] = 0` for all keys in its **read** set,
- `Dₓ[hash(key)] = 0` for all keys in its **write** set,
- `Dₛ[hash(key)] = 0` for all keys in its **write** set.

**Hashing into a 100 kB bitstring can produce false negatives** — a genuinely runnable transaction
still seen as blocked — **but never false positives**, so correctness holds.

SCA is "selective" in **two distinct senses**:

1. It **only activates when needed**, unlike a traditional lock manager which always pays to track
   contention even when the information is never used.
2. It **avoids all-to-all conflict analysis**, limiting itself to the transactions **most likely to
   be runnable and cheapest to check**.

**Implementation optimization:** re-hashing every key on every pass is expensive, so **hash results
are cached in the transaction's state** the first time SCA encounters it; later passes reuse the
saved offsets.

## VLLR: locking ranges

Range locks matter for workloads that read, write, or delete many consecutive rows in one
transaction — they **avoid phantoms**, and handle the common case of deleting an entity whose rows
share a primary key prefix. **Spanner uses range locks exclusively** in place of point locks.

**VLLR locks bitstring prefixes.** A key range is expressed as a range `R` of lexicographically
sorted bitstrings, then converted into a **prefix set `P`** such that every key in `R` has some
element of `P` as a prefix. The simplest construction takes the **longest common prefix of the
minimum and maximum** of `R`.

That construction is conservative, sometimes badly so. The paper's example: in an 8-bit key space,
locking `R = [00111100, 01000010]` yields the prefix `0xxxxxxx` — **half the key space instead of
the necessary 7/256.** Finer decompositions exist (`001111xx`, `0100000x`, `01000010` locks exactly
`R`), and the right choice is workload-dependent: **coarse prefixes are cheaper to lock but risk
expensive false contention.**

Mechanically VLLR resembles **hierarchical locking**, where intention locks are acquired coarse to
fine before the target lock. It keeps **four counters per key: `Cₓ`, `Cₛ`, `Iₓ`, `Iₛ`.** Requesting a
lock on prefix `p` increments `Cₓ[p]` or `Cₛ[p]`, and **for each nonempty strict prefix `pⱼ` of `p`
increments `Iₓ[pⱼ]` or `Iₛ[pⱼ]`**; each incremented counter is checked against its conflicting
counters to determine whether the lock is granted.

Two properties worth noting:

- **Overhead is bounded to one increment/decrement per bit** across the union of the transaction's
  prefix sets.
- **Overlapping lock ranges cost nothing extra** — unlike traditional range locking, where
  overlapping ranges must be **split**.

SCA extends to VLLR by **adding two more bitmaps** for `Iₓ` and `Iₛ` (though it must set more bits
per transaction, given the extra prefixes).

## Evaluation

### Setup

- **Nine systems implemented** in C++ across three families: single-machine, distributed
  partition-per-machine, and distributed partition-per-core.
- **Hardware:** Amazon EC2 `m3.2xlarge` (30 GB memory, eight virtual cores), a shared-nothing
  cluster of **eight instances** unless noted.
- **Core allocation:** three of eight cores per machine are devoted to components independent of the
  locking scheme (load generation, monitoring, intra-process communication, input logging), leaving
  **five cores for worker and lock management threads**. Worker pool sizes were hand-tuned per
  technique.
- For **Calvin-based deadlock-free schemes, one core is dedicated entirely to the lock manager
  thread**, leaving four for workers — a detail that matters in the TPC-C results below.
- **Deadlock detection:** the authors tried timeouts (used in an earlier version of this work) but
  found **waits-for graphs perform better in practice**, and additionally tuned a blocked-transaction
  threshold to limit deadlock. This **substantially improved the baselines** relative to their prior
  paper.

**Benchmarks:**

| Workload | Description |
| --- | --- |
| **"Short" microbenchmark** | Each transaction reads 10 records and updates a value at each. One record from a small **hot** set, nine from a large **cold** set. **Contention index** = probability any two transactions conflict, tuned by hot-set size (1,000 hot records → 0.001; one hot record → 1). ~50 ms per transaction; **a high fraction of time is spent acquiring locks** |
| **"Long" microbenchmark** | Same, plus **10 ms of CPU work per data item**; ~150 ms per transaction, comparable to TPC-C New Order. Closer to real-world workloads |
| **TPC-C** | Full benchmark: New Order 45%, Payment 43%, Order Status 4%, Stock Level 4%, Delivery 4%. Complex logic and high contention, so most similar to "long" under high contention |

A **"no locking" baseline** (all locking removed, isolation forgone) makes the pure overhead of each
scheme visible.

### Single-server, multi-core

Compared against standard 2PL (with deadlock detection) and a **deadlock-free 2PL** variant that
also places all lock requests in one atomic step — isolating the effect of the *data structure* from
the effect of the *protocol*.

**Locking overhead at low contention** (difference from the no-locking baseline):

| Scheme | "Long" transactions | "Short" transactions |
| --- | ---: | ---: |
| **Standard 2PL** | **22%** | **43%** |
| **VLL** | **1.5%** | **10.2%** |

The 22% figure is consistent with prior published measurements of main-memory locking overhead. The
short-transaction numbers are higher for both because a greater share of transaction time is lock
acquisition.

Other findings:

- **VLL's remaining short-transaction overhead comes from the critical section**, which the
  multi-threaded version still needs around lock acquisition.
- **Deadlock-free 2PL performed extremely poorly on short transactions** — consistently bad and
  *unaffected by contention index* — because its critical section does the same job using **the
  much heavier traditional hash-based lock manager**, making it the sole bottleneck.
- **SCA improves VLL by up to 41% on "long" transactions**, but only modestly on "short" ones,
  because short transactions execute so fast that removing a transaction from the queue slightly
  earlier than it would have reached the front buys little.
- **SCA's benefit is a bubble.** At low contention there is nothing to unblock; at *extremely* high
  contention nearly every transaction conflicts with every other, so **SCA cannot find anything to
  unblock either**. Gains appear at medium-to-high contention.
- As contention rises, VLL and 2PL converge, since the extra information 2PL maintains becomes
  increasingly useful. **But VLL+SCA stays comparable even there**, "since SCA can quickly construct
  the relevant part of transactional data dependencies on the fly."
- **Deadlock changes the picture in VLL's favor.** With only one hot item per transaction deadlock
  is essentially impossible, which *hides* a real 2PL disadvantage. Raising the number of contested
  records per transaction degrades the 2PL implementations at high contention while **VLL is
  completely unaffected — it is deadlock-free by construction.**

### Distributed: partition-per-core

40 partitions across 8 machines. Low contention = 10,000 hot of 1,000,000 records per partition
(index 0.0001); high contention = 100 hot (index 0.01). The percentage of multi-partition
transactions is swept from 0 to 100%.

- **SCA becomes essential**, not merely helpful: **up to ~100% improvement on "long" transactions
  and ~130% on "short"** under high contention. The reason is structural — in the single-machine
  case the head of the `TxnQueue` **can always run**, so progress is always available. In the
  distributed case **the head can be stalled waiting for a remote message**, and without SCA *the
  entire queue waits behind it*. SCA finds other transactions to run meanwhile.
- **Under low contention with >60% multi-partition transactions, SCA slightly hurts.** Three
  compounding reasons: fewer blocked transactions to unblock; the queue is longer (more
  multi-partition transactions waiting on remote reads) so **each SCA pass costs more**; and blocked
  transactions are more likely stuck behind a multi-partition transaction, **which SCA cannot
  accelerate**. The penalty stays small because **SCA only runs when the CPU would otherwise be
  idle** — it costs something only if the CPU would have woken before the pass finished.
- **Colocated vs. arrayed:** colocated wins when transactions are mostly local — arrayed's extra
  memory access is **~10% of a "short" transaction and ~3–4% of a "long" one**. As distributed
  transactions are added, cross-partition coordination dominates and the difference **becomes much
  less visible**.
- **VLL vs. a per-core traditional lock manager:** **10–30% better with few distributed
  transactions**, narrowing as more are added. Since both allow a thread to work on other
  transactions while awaiting remote messages, **the only difference is locking overhead** — which is
  a smaller share of longer or distributed transactions.
- **H-Store (serial execution) degrades severely** as multi-partition percentage rises, because a
  partition has no intra-partition concurrency and **must sit idle awaiting remote reads**. The
  telling result is at the left edge: **even at 0% multi-partition transactions, H-Store cannot
  significantly outperform VLL — despite acquiring no locks at all.** That is the strongest single
  statement of how cheap VLL is.
- **At very high contention the per-core lock manager eventually slightly beats VLL+SCA.** This
  inflection point is where **fully tracking contention at all times finally outweighs the cost of
  maintaining lock queues** — past it, the lock manager's information unblocks transactions faster
  than VLL can reconstruct it.
- **H-Store is unaffected by contention entirely**, since it processes transactions serially.

### Distributed: partition-per-machine

Four systems: **2PL + 2PC** (System-R* design with distributed deadlock detection),
**nondeterministic VLL+SCA + 2PC**, **Calvin**, and **Calvin + VLL**.

The headline: **Calvin is significantly outperformed by traditional 2PL+2PC because of its lock
manager bottleneck — and VLL completely removes that bottleneck, enabling Calvin to outperform the
nondeterministic system at almost every data point.** And separately: **VLL improves the
nondeterministic design too**, and even at high contention is not beaten by hash-based lock
management, thanks to SCA.

### TPC-C

Partition-per-core: 40 × 10-warehouse partitions, contention index ≈ 0.02. Partition-per-machine:
8 × 20-warehouse partitions, ≈ 0.01. Distributed transaction percentage swept 0–100% (**the actual
TPC-C spec produces under 10%**).

- **SCA improves over plain VLL by 40–145%** when many transactions are multi-partition — again
  because it finds work while the queue head awaits a remote message.
- **VLL+SCA and the traditional per-core lock manager are close on TPC-C**, because TPC-C
  transactions are long and, at high contention with many distributed transactions, **the lock
  manager is not the bottleneck** — so replacing it changes little (and *without* SCA, hurts a lot).
- **But at the real TPC-C distributed-transaction rate (<10%), VLL+SCA beats the hash-based scheme
  by 8%.** The authors are explicit that one should not read the high-distributed portion of the
  graph as a verdict on TPC-C.
- **Throughput drops less with distributed percentage than in the microbenchmark**, because
  microbenchmark distributed transactions touch **one hot key per partition** while TPC-C's touch
  **one hot key total** — so **the contention index actually decreases** as distribution rises.
- **At 0% distributed transactions, 2PL+2PC beats the deterministic systems.** Reason: TPC-C
  transactions are long, so the lock request rate is low, so **dedicating an entire core to lock
  acquisition wastes it** — while 2PL+2PC uses all CPU resources. As distributed transactions
  increase, 2PC and distributed deadlock costs take over and the deterministic systems win.

### Scalability

2 → 48 machines, 20% multi-partition, "long" transactions.

- **VLL scales as linearly as Calvin**, maintaining and extending its advantage at scale.
- **Neither achieves perfect linear scaling under high contention**, due to **execution progress
  skew**: machines occasionally fall briefly behind from workload variation or RPC latency
  fluctuation, slowing others; **the more machines, the more likely at least one is lagging at any
  moment**, and higher contention plus more distributed transactions increases sensitivity to it.
- **2PL+2PC scales comparably under low contention but degrades much more steeply under high
  contention** — it suffers execution progress skew *plus* **an increase in distributed deadlocks**,
  which further increase contention.
- **H-Store scales poorly**, since all distributed transactions execute serially.

### VLLR (range locking)

Compared against two baselines in a single-machine, single-threaded harness with **artificial delays
simulating remote reads** (100 µs and 500 µs), each transaction locking one range:

- **Standard Range Lock Manager** — explicitly maps key ranges to request queues, **fragmenting
  ranges when new ones partially overlap**, backed by `std::map` (a red-black tree) since ranges must
  stay sorted.
- **Hierarchical Lock Manager** — a hash-table lock manager doing the **same bitwise-prefix
  hierarchical locking as VLLR**, acquiring intention locks for every nonempty prefix.

Contention was varied by choosing ranges whose endpoints share prefixes of ~15 bits (low) or ~6 bits
(high). Because VLLR and the hierarchical manager lock **conservative prefixes rather than exact
ranges**, they observe **higher contention (0.0002 / 0.0178) than the standard manager (0.0001 /
0.0125)** — but require **significantly fewer intention locks under high contention**, so lower CPU
overhead.

Findings:

- **SCA is as critical to VLLR as to VLL** under high contention with distributed delays: it costs
  little and throughput drops significantly without it. With short or infrequent stalls it adds
  little.
- **The Standard Range Lock Manager pays higher CPU** (red-black tree operations plus range
  splitting) **but degrades more gracefully** under simulated delays, precisely because it locks
  exact ranges and therefore experiences less contention.
- **The Hierarchical Lock Manager is crippled by CPU overhead** from enqueuing and dequeuing vast
  numbers of intention lock requests — **it remains the sole throughput bottleneck in nearly every
  configuration, even when 100% of transactions incur delays.** Only under *both* high contention and
  frequent long delays does contention rather than CPU become the limit.

The comparison isolates the paper's real claim: **the prefix scheme and the counter scheme are
separable, and it is the counters — not the prefixes — that buy the performance.** VLLR and the
hierarchical manager use the same prefix strategy; only the data structure differs.

## Positioning against related work

- **System R's lock manager** is the design almost all databases adopted. Prior work reduced the
  *number* of lock calls, which **"does not address the root cause of high lock manager overhead —
  the size and complexity of the data structure used to store lock requests."**
- **Lightweight Intent Lock (LIL)** also maintains lightweight counters, but **in a global lock
  table rather than colocated with data**, and **a transaction that cannot acquire all its locks
  blocks waiting for a message from another transaction's thread**. VLL instead uses the **global
  transaction order** to decide what to unblock.
- **Colocating lock state with records was proposed ~two decades earlier** (Gottemukkala & Lehman,
  1992), but with a **linked list of "Lock Request Blocks" per record**, which complicates the record
  structures. **VLL's contribution is the compression to two integers**, not the colocation itself.
- **Shore-MT, Horikawa, Jung et al.** improve multicore scalability by carefully optimizing the lock
  manager and removing latches — **but keep the basic two-phase locking design**. VLL instead changes
  *what* lock information is tracked and *where*. **DORA** partitions the lock manager across cores.
- **Serial execution without concurrency control** (H-Store and kin) buys throughput but **only works
  when the workload partitions cleanly with few multi-partition transactions**. VLL gets much of the
  low-overhead benefit **across a far wider range of workloads.**
- **OCC and MVCC** (HANA's MVCC, Hekaton's optimistic MVCC, Google F1's OCC) eliminate locking
  overhead but introduce their own: **optimistic schemes pay for aborts when the optimistic
  assumption fails, plus data access tracking; multi-version schemes pay expensive memory for
  multiple copies.**
- **Key range locking** was pioneered by Lomet; not all systems implement it, but **Spanner uses it
  exclusively.**

## Limitations and questions

- **The whole design rests on knowing the read/write set up front.** The exploratory-read workaround
  runs **at no isolation** and can force a full abort-and-restart if the set turns out to be wrong —
  a cost that grows with how dynamic the workload is.
- **Distributed deadlock is not eliminated, only avoided by coordination.** The alternative (detect
  and abort) was measured to **negate VLL's entire advantage** under high contention. So VLL's
  benefits in a distributed setting are partly contingent on adopting a deterministic ordering layer
  like Calvin.
- **VLL loses to a traditional lock manager past an inflection point** at very high contention, where
  always-on contention tracking pays for itself.
- **SCA's bit arrays admit false negatives** from hashing, and SCA's cost scales with queue length —
  hurting exactly the low-contention/high-distribution corner.
- **The multi-threaded variant still needs a critical section** around lock acquisition, which is
  visible as ~10% overhead on short transactions.
- **VLLR's prefix construction can lock dramatically more than requested** (half the key space in the
  paper's own example), and the finer decompositions that fix this are not automated.
- **The paper targets in-place update systems only** — multi-versioned VLL and integrated
  hierarchical locking are named as future work.
- **Baselines are self-implemented**, though the authors went to unusual lengths to strengthen them
  (waits-for graphs over timeouts, tuned blocked-transaction thresholds, deadlock-free 2PL as a
  separate control).

## Practical design checklist

VLL's approach fits when:

- the database is **main memory resident** and lock manager overhead is measurable;
- transactions are **short and their read/write sets are determinable up front** (or cheaply
  discoverable);
- the system already establishes a **global transaction order** — deterministic systems get VLL
  nearly for free;
- workloads are **not cleanly partitionable**, ruling out H-Store-style serial execution;
- **contention is low to moderate**, or SCA is enabled to cover the high-contention range.

Look elsewhere when:

- read/write sets genuinely cannot be predicted without executing the transaction;
- contention is extreme and sustained, where full contention tracking wins;
- you need multi-versioning or long-running read-only transactions;
- the workload partitions perfectly, where serial per-partition execution avoids locking entirely.

## Takeaways

1. **When you remove one bottleneck, profile again.** Eliminating disk I/O promoted the lock manager
   from negligible to 22% of transaction time — the same pattern the paper opens with and the
   evaluation keeps confirming.
2. **The data structure is the overhead, not the protocol.** Deadlock-free 2PL performed terribly
   despite using VLL's acquisition discipline, because it kept the hash table and linked lists.
3. **Ordering is information.** Replacing per-lock request queues with one global order is a
   compression: you lose per-record detail but gain a total order that answers "who runs next" — and
   **makes deadlock structurally impossible** rather than something to detect.
4. **Colocate state with the data it describes.** Two integers next to the record turn a lock check
   into part of a cache line you were already fetching.
5. **Pay for information only when it would change your decision.** SCA is the paper's most
   transferable idea: reconstruct expensive bookkeeping **lazily, and only when the CPU is otherwise
   idle** — inverting the traditional design that always pays whether or not the data is used.
6. **Let the tuning parameter tune itself.** Thresholding on *blocked* transactions rather than total
   queue size makes the limit contention-adaptive with no knob.
7. **Know where your optimization stops winning.** The paper is unusually forthright about the
   inflection point where a traditional lock manager wins, and about SCA's low-contention penalty.
8. **Deterministic execution and lightweight locking compose.** Calvin needed a global order anyway;
   VLL needed one to exist. Together they remove both 2PC and the lock manager — the two bottlenecks
   that each system alone leaves standing.

## Citation

```bibtex
@article{ren2015vll,
  author = {Kun Ren and Alexander Thomson and Daniel J. Abadi},
  title = {{VLL}: a lock manager redesign for main memory database systems},
  journal = {The VLDB Journal},
  volume = {24},
  number = {5},
  pages = {681--705},
  year = {2015},
  doi = {10.1007/s00778-014-0377-7}
}
```
