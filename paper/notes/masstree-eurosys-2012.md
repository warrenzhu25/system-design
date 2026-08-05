# Masstree: Cache Craftiness for Fast Multicore Key-Value Storage

> EuroSys 2012 — structured reading notes

Full paper content: [Markdown conversion](../original/masstree-eurosys-2012.md)

## Paper information

- **Authors:** Yandong Mao, Eddie Kohler (Harvard University), Robert Morris
- **Affiliations:** MIT CSAIL; Harvard University
- **Venue:** EuroSys '12, April 10–13, 2012, Bern, Switzerland
- **Keywords:** multicore; in-memory; key-value; persistent

## One-sentence summary

Masstree is an in-memory key-value store whose central structure is a **trie of B⁺-trees**, each
indexed by a fixed 8-byte slice of the key — giving efficient handling of arbitrary-length binary
keys with long shared prefixes — combined with **lock-free optimistic reads, node-local write
locks, and a node layout and fanout tuned so a whole node arrives in one DRAM latency**, reaching
**over six million queries per second on 16 cores with logging and networking enabled**.

## Problem and goals

Single-server storage performance matters even in large deployments: faster servers reduce cost
and reduce **load imbalance caused by partitioning data among servers**, and intermediate-sized
deployments may avoid multi-server complexity entirely.

Masstree targets key-value data that **fits in memory but must persist across restarts**, with a
deliberately flexible storage model:

- **Arbitrary variable-length keys**, including binary strings.
- **Range queries** — clients can traverse subsets or the whole database in sorted key order.
- **Good performance on keys with long shared prefixes.** The motivating example is Bigtable-style
  permuted URL keys such as `edu.harvard.seas.www/news-events`, which group a domain's pages
  together for interesting range queries but share long prefixes.
- **Efficiency with small values**, where disk and network throughput are not the limit.

The combination "could free performance-sensitive users to use richer data models than is common
for stores like memcached today."

Three design challenges shaped everything:

1. Efficiently support **many key distributions**, including variable-length binary keys with long
   common prefixes.
2. Allow **fine-grained concurrent access**, and **get operations must never dirty shared cache
   lines** by writing shared data structures.
3. The layout must **support prefetching and collocate important information on few cache lines**.

Properties 2 and 3 together are what the paper calls **cache craftiness**.

### System interface

Four operations, where `c` is an optional list of column numbers letting clients read or write
subsets of a value:

| Operation | Meaning |
| --- | --- |
| `get_c(k)` | Read (selected columns of) the value for key `k` |
| `put_c(k, v)` | Write (selected columns of) a value |
| `remove(k)` | Delete a key |
| `getrange_c(k, n)` | "Scan": return up to `n` key-value pairs starting at or after `k`, in lexicographic key order. **Not atomic with respect to inserts and updates** |

A single client message can carry many queries.

## The data structure

**A Masstree is a trie with fanout 2⁶⁴ where each trie node is a B⁺-tree.** The trie structure
handles long keys with shared prefixes; the B⁺-trees handle short keys, fine-grained concurrency,
and effective use of cache lines through medium fanout.

Equivalently, a Masstree is **one or more layers of B⁺-trees, each indexed by a different 8-byte
slice of the key**:

| Layer | Indexed by key bytes | Holds |
| --- | --- | --- |
| 0 (root tree) | 0–7 | all keys up to 8 bytes long |
| 1 | 8–15 | |
| 2 | 16–23 | |
| … | … | |

Each tree has at least one **border node** and zero or more **interior nodes**. Border nodes
resemble B⁺-tree leaves, but **can also store pointers to deeper trie layers**.

### Placement invariants

Keys are stored as close to the root as possible, subject to:

1. Keys shorter than `8h + 8` bytes are stored at layer ≤ *h*.
2. Any keys in the same layer-*h* tree share the same `8h`-byte prefix.
3. **When two keys share a prefix, they are stored at least as deep as the shared prefix** — if two
   keys longer than `8h` bytes share an `8h`-byte prefix, they are stored at layer ≥ *h*.

Layers are created lazily; insertion prefers existing trees and creates a new tree only when an
invariant would otherwise be violated. **Removal deletes completely empty trees but does not
otherwise rearrange keys.**

Worked example on an initially empty tree `t`:

1. `t.put("01234567AB")` stores the key in the root layer — slice `"01234567"` stored separately
   from the 2-byte suffix `"AB"`. A `get` searches for the slice, then compares the suffix.
2. `t.put("01234567XY")` shares an 8-byte prefix, so a new layer is created: both values go into a
   freshly allocated border node under slices `"AB"` and `"XY"`, and that node **replaces** the
   `"01234567AB"` entry in the root layer. **Concurrent gets observe either the old state or the
   new layer**, so `"01234567AB"` remains visible throughout.
3. `t.remove("01234567XY")` descends to the layer-1 tree and deletes `"XY"`; `"AB"` remains there.

### Balance and complexity

A Masstree's shape depends on the key distribution — 1000 keys sharing a 64-byte prefix generate
at least 8 layers, where without the prefix they would fit in one. Nevertheless:

| Structure | Cost |
| --- | --- |
| B-tree, *n* keys of max length *ℓ* | O(log n) node examinations, O(log n) key comparisons, each O(ℓ) → **O(ℓ log n)** total |
| Masstree | O(log n) comparisons in each of O(ℓ) layers, but each compares a **fixed-size slice** → **O(ℓ log n)** total — the same |
| Masstree, long common prefixes | **O(ℓ + log n)** — ℓ for the prefix plus log n for the suffix |

The trade-off: **Masstree's range queries have higher worst-case complexity than a B⁺-tree's**,
since they must traverse multiple layers.

Compared with **partial-key B-trees** (which avoid some key comparisons while preserving true
balance), Masstree **bounds the non-node memory references needed to find a key to at most one per
lookup**, and its 8-byte-slice comparisons are easy to code efficiently. Masstree can use more
memory on some distributions because its nodes are wide, but **outperformed the authors' pkB-tree
implementation by 20% or more** on several benchmarks.

## Node layout

```text
struct interior_node:            struct border_node:
  uint32_t version;                uint32_t version;
  uint8_t  nkeys;                  uint8_t  nremoved;
  uint64_t keyslice[15];           uint8_t  keylen[15];
  node*    child[16];              uint64_t permutation;
  interior_node* parent;           uint64_t keyslice[15];
                                   link_or_value lv[15];
union link_or_value:               border_node* next;
  node*   next_layer;              border_node* prev;
  [opaque] value;                  interior_node* parent;
                                   keysuffix_t keysuffixes;
```

At heart these are internal and leaf nodes of a **B⁺-tree of width 15**. Border nodes are
**doubly linked** to support `remove` and `getrange`.

Key details:

- **`keyslice` stores 8-byte slices as 64-bit integers, byte-swapped if necessary so native
  less-than comparison matches lexicographic string comparison.** This was "the most valuable of
  our coding tricks, improving performance by 13–19%." Short slices are zero-padded.
- **Key lengths** distinguish different keys with the same slice — necessary because null
  characters are valid in binary keys, so the 8-byte key `"ABCDEFG\0"` must be distinguished from
  the 7-byte `"ABCDEFG"`.
- **At most 10 keys can share a slice** in one tree: lengths 0 through 8, plus either one key of
  length > 8 **or** a link to a deeper layer. (Only one key longer than 8 bytes is possible,
  because a second would create the deeper layer.)
- **All keys with the same slice live in the same border node.** This slims interior nodes (they
  need no key lengths) and simplifies concurrency invariants, at the cost of extra checking during
  splits. Masstree is in this sense a restricted **prefix B-tree**.
- **Key suffixes** live in `keysuffixes` structures placed either inline or in separate memory
  blocks; **Masstree adaptively decides how much per-node suffix memory to allocate and whether to
  inline it**. Versus the simple approach of reserving fixed space for 15 suffixes per node, this
  **cuts memory by up to 16% for short-key workloads and improves performance by 3%**.
- **Values live in `link_or_value` unions**, distinguished from next-layer pointers by the
  `keylen` field. Users control all bits in `value` slots.

**Fanout choice.** Performance is dominated by DRAM latency for node fetches. Masstree
**prefetches all of a node's cache lines in parallel** before using it, so the whole node becomes
usable after a single DRAM latency. Up to a point, **larger nodes cost the same as smaller ones**
while giving wider fanout and lower tree height. On the paper's hardware, **four cache lines (256
bytes, fanout 15)** gave the highest total performance.

## Non-concurrent modification

Standard B⁺-tree algorithms form the baseline. Inserting into a full border node **splits** it: a
new node is allocated, old plus new keys distributed, and the new node inserted into the parent —
recursively splitting up the tree, terminating at a node with room or at the root, where a new
interior node is created.

Removal simply deletes from the border node; empty border nodes are freed and removed from their
parents, continuing up the tree. **Masstree does not redistribute keys on removal** — removal
without rebalancing has theoretical and practical advantages.

Two supporting mechanisms:

- **A per-tree doubly linked list among border nodes** speeds range queries in both directions.
  A singly linked list would suffice for forward-only queries, but **backlinks are required by
  concurrent remove anyway**.
- **Sequential-insert optimization:** sequential insertions are easy to detect (the item goes at
  the end of a node with no `next` sibling). If a sequential insert needs a split, **the old node's
  keys stay in place and the new item goes into an empty node**, improving memory utilization and
  performance for sequential workloads.

## Concurrency

**Fine-grained locking for writers, optimistic concurrency control for readers.** Readers acquire
**no locks whatsoever** and **never write to globally accessible shared memory** — because writes
to shared memory limit performance both by causing contention (e.g. readers contending for a
node's read lock) and by **wasting DRAM bandwidth on writebacks**.

The consequence: readers may observe intermediate states such as partially inserted keys. The
communication channel is a **per-node `version` counter**, which writers mark **dirty** before
creating intermediate states and **increment** when done. Readers snapshot `version` before
accessing a node and compare afterwards; **if it differs or is dirty, the reader must retry**.

**Correctness condition: no lost keys.** A `get(k)` must return a correct value for `k` regardless
of concurrent writers — when `get(k)` and `put(k, v)` run concurrently, either the old or the new
value is acceptable. **The biggest challenge is concurrent splits and removes, which can shift
responsibility for a key away from a subtree even as a reader traverses that subtree.**

### Version number layout

| Field | Purpose |
| --- | --- |
| `locked` | Claimed by update or insert |
| `inserting` | "Dirty" bit set during inserts |
| `splitting` | "Dirty" bit set during splits |
| `vinsert` | Counter incremented after each insert |
| `vsplit` | Counter incremented after each split |
| `isroot` | Whether this node is the root of some B⁺-tree |
| `isborder` | Interior or border |
| `unused` | Allows more efficient operations on the version word |

Separating **insert** and **split** counters is what lets readers **retry locally for inserts but
from the root only for splits**.

### Writer–writer coordination

Per-node **spinlocks**, stored as one bit in the version counter. Any modification of a node's keys
or values requires its lock, but **some data is protected by other nodes' locks**: a node's
`parent` pointer by its parent's lock, and a border node's `prev` pointer by its previous sibling's
lock. This **minimizes simultaneous locks during splits** — an interior node splitting can assign
its children's parent pointers **without locking them**.

Splits and deletions need multiple simultaneous locks: splitting node *n* requires holding *n*'s
lock, its new sibling's lock, **and its parent's lock** — preventing a concurrent split from moving
*n* (and hence its sibling) to a different parent before the new sibling is inserted. **Lock
ordering prevents deadlock: locks are always acquired up the tree.**

The authors evaluated alternatives including **lock-free algorithms based on compare-and-swap**,
and the locking protocol performed as well or better, because **on cache-coherent multicore
machines the major cost of locking — the cache coherence protocol — is also incurred by lock-free
CAS**, and Masstree never holds a lock long.

### Writer–reader coordination

The naïve correct algorithm — snapshot *every* node's version, track every examined node, re-check
all of them before returning — "would clearly perform terribly." Efficiency comes from
**eliminating unnecessary version changes, restricting which snapshots readers must track, and
limiting the scope over which readers retry.**

**Updates (changing an existing key's value).** Handled by **atomically updating values with
aligned write instructions**, which on modern machines have atomic effect — a concurrent reader
sees either the old or the new value. Therefore updates **need not increment the version and do not
force readers to retry**. Writers must not free old values until concurrent readers finish, solved
by **epoch-based reclamation**; all reader-accessible data is freed the same way.

**Border inserts and the permutation field.** A conventional B-tree leaf insert rearranges keys
into sorted order, creating invalid intermediate states. Masstree instead makes each insert visible
in **one atomic step**, eliminating the invalid state entirely.

The 64-bit `permutation` is 16 four-bit subfields: the lowest 4 bits are `nkeys` (0–15), the rest
form `keyindex[15]`, a permutation of 0–15. Entries `keyindex[0..nkeys-1]` hold the indexes of live
keys **in increasing key order**; the rest list unused slots. To insert, a writer locks the node,
loads the permutation, **rearranges it to shift an unused slot into the correct position and
increment `nkeys`**, writes the key and value into that previously unused slot, then **writes back
the new permutation and unlocks**. The key becomes visible only at that last write, so **readers see
either the old order without the key or the new order with the key in its proper place — no key
rearrangement and no version increment**.

A compiler fence, and on some architectures a machine fence, is required between writing the
key/value and writing the permutation.

**New layers.** When inserting `k1` into a border node holding conflicting key `k2`, Masstree
allocates a new empty border node `n'`, inserts `k2`'s value under the appropriate slice, and
replaces `k2`'s value in `n` with the `next_layer` pointer. Since only one key is affected, **no
version or permutation update is needed** — but readers must reliably distinguish values from
layer pointers, and the pointer and the marker are stored separately. The write sequence is:
**mark the key `UNSTABLE`** (readers seeing this retry), **write the `next_layer` pointer**, then
**mark the key `LAYER`**.

**Splits.** Unlike ordinary inserts, splits **remove active keys from a visible node and insert them
elsewhere**, so a concurrent `get` might report a shifting key as lost. Versions must therefore be
updated, and the hard part is doing so such that no change is missed.

The protocol is **hand-over-hand locking and marking in the writer, and hand-over-hand validation
in the opposite direction in the reader**:

```text
split(node n, key k):                 // precondition: n locked
  n' ← new border node
  n.version.splitting ← 1
  n'.version ← n.version              // n' is initially locked
  split keys among n and n', inserting k
ascend:
  p ← lockedparent(n)                 // hand-over-hand locking
  if p = NIL:                         // n was old root
      create new interior node p with children n, n'
      unlock(n); unlock(n'); return
  else if p is not full:
      p.version.inserting ← 1
      insert n' into p
      unlock(n); unlock(n'); unlock(p); return
  else:
      p.version.splitting ← 1
      unlock(n)
      p' ← new interior node; p'.version ← p.version
      split keys among p and p', inserting n'
      unlock(n'); n ← p; n' ← p'; goto ascend
```

```text
findborder(node root, key k):
retry:  n ← root; v ← stableversion(n)
        if v.isroot is false: root ← root.parent; goto retry
descend: if n is a border node: return ⟨n, v⟩
        n' ← child of n containing k
        v' ← stableversion(n')
        if n.version ⊕ v ≤ "locked":  // hand-over-hand validation
            n ← n'; v ← v'; goto descend
        v'' ← stableversion(n)
        if v''.vsplit ≠ v.vsplit: goto retry   // split → retry from root
        v ← v''; goto descend                  // otherwise retry from n
```

**Why this is correct.** Consider interior node B splitting into B′ with parent A, where child X
moves to B′. The split proceeds: (1) mark B and B′ `splitting`; (2) shift children including X to
B′; (3) lock A and mark it `inserting`; (4) insert B′ into A; (5) unlock all three, incrementing
A's `vinsert` and B/B′'s `vsplit`.

Now take a concurrent `findborder(X)` starting at A:

- If it traverses to **B′**, it finds X — because X moved in step 2 **before** the pointer to B′ was
  published in step 4.
- If it traverses to **B**, then because findborder loads the child's version **before**
  re-checking the parent's, it must have loaded B's version **before** A was marked `inserting`
  (step 3), hence **before step 1** (which would have made `stableversion` retry). Then either it
  completes before step 1 and finds X, or it is delayed past step 1 and **always detects the split
  and retries from the root** — the `B.version ⊕ v` check fails on the `splitting` flag, the
  following `stableversion(B)` blocks until the flag clears at step 5, and by then B's `vsplit` has
  changed.

**Measured rarity:** in an 8-thread insert test, **fewer than 1 insert in 10⁶ had to retry from the
root due to a concurrent split**, while **concurrent inserts were observed 15× more often** —
which is exactly why the two counters are separate and inserts retry only locally. Alternative
schemes such as backing up the tree step by step "were more complex to code but performed no
better."

**Border-node splits use links instead.** The key invariant is that **nodes split "to the right"**:
a splitting border node's *higher* keys move to its new sibling. Plus:

- The initial node of a B⁺-tree is a border node, **not deleted until the tree is completely empty,
  and always the leftmost node**.
- Every border node *n* is responsible for `[lowkey(n), highkey(n))`. Splits and deletes can change
  `highkey(n)`, but **`lowkey(n)` is constant over the node's lifetime**.

So `get` reliably finds the right border node by comparing the key against the next border node's
`lowkey`. Stale roots caused by concurrent splits are handled at the start of `findborder`: **the
layer-0 global root is updated immediately, but other roots (stored in border nodes' `next_layer`
pointers) are updated lazily** during later operations.

```text
get(node root, key k):
retry:   ⟨n, v⟩ ← findborder(root, k)
forward: if v.deleted: goto retry
         ⟨t, lv⟩ ← extract link_or_value for k in n
         if n.version ⊕ v > "locked":
             v ← stableversion(n); next ← n.next
             while !v.deleted and next ≠ NIL and k ≥ lowkey(next):
                 n ← next; v ← stableversion(n); next ← n.next
             goto forward
         else if t = NOTFOUND: return NOTFOUND
         else if t = VALUE:    return lv.value
         else if t = LAYER:    root ← lv.next_layer; advance k to next slice; goto retry
         else:                 goto forward     // t = UNSTABLE
```

### Removes — the subtle case

Masstree includes a **full implementation of concurrent remove**, unlike some prior work. Several
non-obvious consequences:

**Removes combined with inserts must sometimes force readers to retry.** Consider:

```text
get(n, k1):    locate k1 at position i
remove(n, k1):     remove k1 from position i
put(n, k2, v2):        insert k2, v2 at position j
get (cont.):   lv ← n.lv[i]; check n.version; return lv.value
```

The `get` may legitimately return `k1`'s removed value, since the operations overlapped — so
**`remove` must not clear the memory for the key or value; it only changes the permutation**. But
if the `put` happens to pick `j = i`, the `get` would return `v2`, which is **not** a valid value
for `k1`. Therefore **Masstree must increment `vinsert` when removed slots are reused.**

Other consequences:

- When a border node becomes empty it is removed, along with any resulting empty ancestors —
  requiring the **doubly** linked border list. A naïve implementation would break the list under
  concurrent splits and removes, so **compare-and-swap operations (some with flag bits) are needed
  in both split and remove**, slightly slowing split.
- Removed nodes are **marked `deleted` and reclaimed later**; **any operation encountering a
  `deleted` node retries from the root**.
- Interior-node manipulation resembles split, using hand-over-hand locking to find the key to
  remove; once removed the node becomes **completely unreferenced**.
- **Removes can empty whole layer-*h* trees (h ≥ 1)**, which are not cleaned up immediately —
  normal operations lock at most one layer at a time, but removing a full tree requires locking
  both the empty layer-*h* tree **and** the layer-(*h*−1) border node pointing to it. **Epoch-based
  reclamation tasks are scheduled to clean up empty and pathologically shaped layer trees.**

## Values

A value is a version number plus an array of variable-length strings called **columns**, addressed
by integer index. **Multi-column puts are atomic** — a concurrent `get` sees all or none of the
modifications.

The evaluated implementation (best for small values) allocates each value as **a single memory
block** and **never modifies in place**, since that would expose intermediate states: `put` creates
a new value object, copying unmodified columns. This uses cache well for small values but would
cause excessive copying for large ones, for which Masstree offers a design storing each column in a
separately allocated block.

## Discussion of micro-choices

- **More than 30% of lookup cost is computation, not DRAM waits** — mostly key search within nodes.
  **Linear search has worse complexity than binary search but better locality**, and the winner is
  architecture-dependent: on Intel, linear search was **up to 5% faster**; on AMD they tied.
- **PALM's parallel lookup** (overlapping DRAM fetches by looking up a batch of keys together) did
  **not** help on the 48-core AMD machine but raised throughput **up to 34% on a 24-core Intel
  machine**. The authors planned to restructure the network stack to exploit it.

## Networking and persistence

- **Per-core receive and transmit queues** reduce contention when short query packets arrive from
  many clients; per-core UDP ports can be bound to a single core's receive queue for short
  connections. The benchmarks instead use **long-lived TCP connections from few clients (or client
  aggregators)**, equally effective at avoiding network overhead.
- **Logging is per-core:** each query thread has its own log file and in-memory buffer, with a
  logging thread on the same core writing it out in the background — **logging proceeds in parallel
  on each core**. A `put` appends to the buffer and **responds to the client without forcing the
  buffer to storage**; logging threads batch for sequential throughput but **force to storage at
  least every 200 ms**. Different logs may live on different disks/SSDs.
- **Recovery** uses value version numbers and log record timestamps. Sequential updates to a value
  get distinct increasing version numbers, written into the log with the operation, and each record
  is timestamped. Masstree sorts logs by timestamp and computes the **recovery cutoff**
  `τ = min over logs of (max timestamp in that log)`, then **replays updates in parallel**, applying
  each value's updates in increasing version order and **dropping updates with timestamp ≥ τ**.
- **Checkpoints** contain all keys and values, speeding recovery and allowing log reclamation.
  Recovery loads the latest valid checkpoint completed before τ, then replays logs from the
  timestamp at which the checkpoint began.

**Measured (not deeply evaluated, included to show persistence need not limit performance):**
checkpointing **140 million key-value pairs (9.1 GB) takes 58 seconds**; **recovery from it takes 38
seconds**. The bottleneck for both is **imbalance in parallelization across cores**. Checkpoints run
concurrently with request processing; a put-only workload achieves **72% of ordinary throughput**
during a checkpoint, due to disk contention.

## Evaluation

### Setup

- 48-core server (eight 2.4 GHz six-core AMD Opteron 8431), Linux 3.1.5. Per core: 64 KB L1
  instruction and data, 512 KB L2; **6 MB L3 shared per six-core chip**; 64-byte cache lines; 8 GB
  DRAM per chip. **Tests use up to 16 cores on up to three chips**, with only those chips' DRAM, to
  mimic a machine more like those easily purchasable.
- Four SSDs (90–160 MB/s sequential write), all used for logs and checkpoints. 10 Gb NIC, 25 client
  machines over TCP, interrupts distributed across all cores. Results averaged over three runs.
- Keys mostly ≤ 10 bytes, values 1–10 bytes, uniformly distributed. **The key space is not
  partitioned**: a border node generally holds keys from different clients. The common distribution
  is **"1-to-10-byte decimal"** — decimal strings of random numbers in [0, 2³¹), of which **80% are
  9 or 10 bytes long, forcing layer-1 trees**.
- Get experiments start with a full store (80–140M keys), run 20 seconds. Put experiments start
  empty and run 140M puts (**~10% become updates** since clients occasionally collide). **Puts run
  ~30% slower than gets.**

### Factor analysis: from a binary tree to Masstree

140M-key 1-to-10-byte-decimal workloads, 16 cores, each server thread generating its own load (no
network or logging):

| Step | What changed | Effect |
| --- | --- | --- |
| **Binary** | Fast concurrent lock-free binary tree, 40-byte nodes (full key, value pointer, two child pointers), jemalloc | baseline |
| **+Flow** | Switch to Flow, their Streamflow implementation — memory allocation often bottlenecks multicore performance | — |
| **+Superpage** | 2 MB x86 superpages | **+27–37%** (fewer TLB misses, lower kernel allocation overhead) |
| **+IntCmp** | Integer key-slice comparison | **+15–24%** |
| **4-tree** | Fanout 4; **nearly halves depth**; two cache lines per node but **usually only the first must be fetched**, containing all four child pointers and the first 8 bytes of each key. All internal nodes full; lockless reads that never retry; lock-free CAS inserts | **+41–44%** |
| **B-tree** | Concurrent B⁺-tree, fanout 15, space for the first 16 bytes of each key, using the paper's concurrency scheme | **−12% on puts**, little get change — conventional inserts must rearrange keys (4-tree never does), and 5 cache lines for average fanout 11 is a worse ratio than 4-tree's |
| **+Prefetch** | Prefetch the wide nodes to overlap DRAM latency | **+9–31% over 4-tree** |
| **+Permuter** | Leaf-node permutations | **+4% on puts** |
| **Masstree** | Full trie-of-B⁺-trees | **+4–8%** |

The last result surprised the authors: with these keys, **33% of keys end up in layer-1 nodes but
the average layer-1 node holds just 2.3 keys** — worse node utilization than a true B-tree. Masstree
still wins, apparently because of efficiencies like **storing 8 bytes per key per interior node
rather than 16**.

### System relevance

With logging on and load over the network, Masstree provides **1.90× (gets) and 1.53× (puts)** the
throughput of "+IntCmp," the fastest binary tree — showing tree design matters in a full system.
Absolute figures: **8.03 Mreq/s gets (77% of the no-network value) and 5.78 Mreq/s puts (63%)**.

### What flexibility costs

| Feature | Comparison | Cost |
| --- | --- | --- |
| **Variable-length keys** | vs. a fixed-8-byte-key B-tree, 16 cores, 80M keys | Masstree 9.84 Mreq/s vs 9.93 — **0.8%**, essentially free, because the trie-of-trees "effectively has fixed-size keys in most tree nodes" |
| **Keys with common prefixes** | vs. "+Permuter," 80M decimal keys where only the final 8 bytes vary | Masstree gives **3.4× throughput for long keys** (+Permuter takes a cache miss for every key suffix compared) and **1.4× even at 16-byte keys** stored fully inline — because Masstree examines the first 8 bytes **once** instead of O(log₂ n) times |
| **Concurrency** | vs. a single-core Masstree with locking, versions, and interlocked instructions removed | Single-core version wins by only **13%** |
| **Range queries** | vs. a concurrent hash table in the same framework, 16 cores, 80M 8-byte keys | **The hash table has 2.5× the throughput** |

**Conclusion: of these features, only range queries appear inherently expensive.**

### Scalability

At 16 cores, Masstree reaches **12.7× (gets) and 12.5× (puts)** its one-core throughput. The limit
for gets is **increasing DRAM fetch cost**: computation stays at ~1000 cycles per operation
regardless of core count, while **average per-operation DRAM stall grows from 2050 cycles at one
core to 2800 at sixteen** — matching the observed throughput drop and consistent with contention for
DRAM or interconnect bandwidth.

### Partitioning versus sharing under skew

Comparison against **hard-partitioned Masstree** — 16 single-core instances each owning a static
equal-sized partition, allocating from local DRAM, with clients routing by key. Skewness δ means
15 partitions get equal request counts while the last gets δ× more (at δ = 9, one partition handles
40% of requests).

| Workload | Result |
| --- | --- |
| **Uniform (δ = 0)** | **Hard-partitioned wins by 1.5×**, mostly by avoiding remote DRAM access and interlocked instructions |
| **Skewed (δ = 9)** | **Masstree wins by 3.5×.** Hard-partitioned throughput falls with skew: the core serving the hot partition saturates at δ ≥ 1 and throttles the whole system, since other clients must wait to preserve skewness — **at δ = 9, 80% of total CPU time is idle** |

Masstree's throughput is **constant across skew**. The authors note the uniform-case disadvantage
may diminish on single-chip machines where all DRAM is local.

### System comparison

Against MongoDB 2.0, VoltDB 2.0, memcached 1.4.8, and Redis 2.4.5, each configured for its best
16-core performance (8 MongoDB processes + config server; 4 VoltDB processes × 4 sites; 16 Redis
processes; 16 memcached processes; Masstree 16 threads). The authors state plainly that **"the
comparisons in this section are not entirely fair"** — the other systems support features Masstree
does not, disabled where possible. Configuration notes: VoltDB replication off; MongoDB on an
in-memory filesystem with 300 MB chunks; Redis with per-process logs on all four SSDs and
checkpointing/log rewriting disabled (log rewriting costs >50% throughput). **In all cases Masstree
includes logging and network I/O.**

Databases initialized with 20M pairs. **MYCSB** is YCSB with a Zipfian key distribution, 10 columns
of 4 bytes, columns identified by number, and YCSB-E modified to return one column per key so the
network is not the bottleneck.

Throughput in millions of requests/second (and as % of Masstree):

| Workload | Masstree | MongoDB | VoltDB | Redis | memcached |
| --- | ---: | ---: | ---: | ---: | ---: |
| **get** (uniform, 1–10 B keys, one 8 B column) | 9.10 | 0.04 (0.5%) | 0.22 (2.4%) | 5.97 (65.6%) | **9.78 (107.4%)** |
| **put** | **5.84** | 0.04 (0.7%) | 0.22 (3.7%) | 2.97 (50.9%) | 1.21 (20.7%) |
| **1-core get** | **0.91** | 0.01 (1.1%) | 0.02 (2.6%) | 0.54 (59.4%) | 0.77 (84.3%) |
| **1-core put** | **0.60** | 0.04 (6.8%) | 0.02 (3.6%) | 0.28 (47.2%) | 0.11 (17.7%) |
| **MYCSB-A** (50% get / 50% put) | **6.05** | 0.05 (0.9%) | 0.20 (3.4%) | 2.13 (35.2%) | N/A |
| **MYCSB-B** (95% get / 5% put) | **8.90** | 0.04 (0.5%) | 0.20 (2.3%) | 2.69 (30.2%) | N/A |
| **MYCSB-C** (all get) | **9.86** | 0.05 (0.5%) | 0.21 (2.1%) | 2.70 (27.4%) | 5.28 (53.6%) |
| **MYCSB-E** (95% getrange / 5% put) | **0.91** | 0.00 (0.1%) | 0.00 (0.1%) | N/A | N/A |

Systems are omitted where unsupported: the hash-table stores cannot run MYCSB-E (range queries),
and memcached cannot run MYCSB-A/B (individual-column update).

Conclusions the authors draw:

- **The one loss is memcached's 7.4% edge on uniform 16-core gets**, explained by partitioning
  avoiding remote DRAM access — **on a single core Masstree slightly exceeds memcached**.
- **Batched query support is vital** on these benchmarks; memcached's update performance is far
  worse than its get performance because its client library cannot batch puts.
- **VoltDB's range query support lags its pure-get support.**
- **Partitioned stores do better on uniform than skewed workloads** — compare Redis and memcached on
  the uniform get workload versus Zipfian MYCSB-C.

## Positioning against related work

- **OLFIT** is a B^link-tree with optimistic concurrency control using per-node version numbers.
  Masstree adopts the idea but, like **Bronson et al.**, **splits the version into two parts**
  (insert and split counters) plus other improvements, leading to **less frequent retries**.
- **PALM** is a lock-free concurrent B⁺-tree with twice OLFIT's throughput, using SIMD and
  **batched, sorted, partitioned lookups** — clever for cache use, but it **requires fixed-length
  keys and its batching raises latency**. Many of its techniques are complementary.
- **Bohannon et al.** and **AlphaSort** store partial keys in nodes to cut DRAM fetches; Masstree
  achieves the same goal with a **trie**.
- **Rao et al.** store children contiguously (CSB⁺-trees) for cache efficiency, wasting memory on
  nonexistent nodes; **Cha et al. report a fast B⁺-tree outperforms a CSB⁺-tree**, and Masstree
  uses more local techniques.
- **H-Store/VoltDB** partition data among cores to avoid concurrency and locking costs; **Masstree
  shares data among all cores** to avoid partitioned load imbalance, using lock-free lookups and
  locally locked inserts.
- **Shore-MT** identified lock contention as the multicore bottleneck and removed locks
  incrementally; **Masstree provides high concurrency from the start**.

## Limitations and questions

- **Range queries are the expensive feature** — a hash table gives 2.5× the throughput, and
  Masstree's range queries have worse worst-case complexity than a plain B⁺-tree because they cross
  layers.
- **`getrange` is not atomic** with respect to concurrent inserts and updates.
- **Shape depends on key distribution.** Long shared prefixes force many layers, and the measured
  layer-1 nodes averaged only 2.3 keys — memory efficiency suffers even when speed does not.
- **Partitioning still wins on uniform workloads** (1.5×), because sharing pays for remote DRAM
  access and interlocked instructions; sharing only wins under skew.
- **Scaling is DRAM-bound, not lock-bound** — per-operation stall grew 37% from 1 to 16 cores, so
  the ceiling is memory bandwidth, not the concurrency design.
- **The version counter could wrap** if a reader blocked mid-computation for 2²² inserts; a 64-bit
  counter would never overflow in practice.
- **Checkpointing is acknowledged as not deeply evaluated**, is bottlenecked by cross-core
  imbalance, and costs 28% of put throughput while running.
- **The system comparison is explicitly not apples-to-apples** — competitors were configured for
  best performance with features disabled, but still carry capabilities Masstree lacks
  (transactions in VoltDB, secondary indexes in MongoDB).
- **No cluster story.** The design targets multicore, not distribution, "though in principle one
  could operate a cluster of Masstree servers."

## Practical design checklist

Masstree's techniques transfer whenever an in-memory ordered index must serve many cores:

- **Make readers write nothing.** Optimistic version validation avoids both lock contention and
  wasted DRAM write bandwidth — and read locks are writes.
- **Split your version counter by event type** so common events (inserts) retry locally and rare
  ones (splits) retry globally.
- **Publish state changes with a single aligned write.** The permutation field converts a
  multi-step key rearrangement into one atomic publication, eliminating an entire class of
  intermediate states.
- **Choose fanout from DRAM latency, not from theory.** If you prefetch a whole node in parallel, a
  wider node costs the same as a narrow one — measure to find the knee (here, 4 cache lines).
- **Compare fixed-size slices as integers.** Byte-swapping 8-byte slices so integer comparison
  matches lexicographic order was worth 13–19% by itself.
- **Order lock acquisition consistently** (here, up the tree) and **protect a field with its
  neighbor's lock** when that reduces the number of simultaneously held locks.
- **Reclaim by epoch, not immediately** — anything a reader might still be looking at.
- **Share the index when load is skewed; partition it when load is uniform.** Neither is universally
  right, and the skew crossover is sharp.

## Takeaways

1. **The bottleneck is DRAM, so design for fetch count and fetch overlap.** Fanout, prefetching,
   cache-line layout, and slice comparison are all in service of "one DRAM latency per node."
2. **A trie of B⁺-trees gets both properties you want:** prefix sharing handled structurally by the
   trie, short keys and concurrency handled well by the B⁺-trees, and **every comparison inside a
   tree is a fixed-size integer compare**.
3. **Optimistic reads are cheap only if you engineer away the retries.** Separate counters, atomic
   value writes, permutation-based inserts, and layer-local retries all exist to make the common
   case never re-run.
4. **Concurrency is nearly free; range queries are not.** 13% for the whole concurrency apparatus
   versus 2.5× for supporting ordered scans.
5. **Sharing beats partitioning under skew by a wide margin** — 3.5× at δ = 9, where the partitioned
   design left 80% of CPU idle. Partitioning converts skew into hard throughput loss.
6. **Remove is the operation that breaks the invariants.** Slot reuse forcing version increments,
   doubly linked lists requiring CAS, deleted-node retries, and layer cleanup are all consequences
   of implementing concurrent delete properly.
7. **Persistence does not have to cost performance.** Per-core logs, batched background flushes with
   a 200 ms bound, and version-ordered parallel replay keep >6M queries/second with logging on.

## Citation

```bibtex
@inproceedings{mao2012masstree,
  author = {Yandong Mao and Eddie Kohler and Robert Morris},
  title = {Cache Craftiness for Fast Multicore Key-Value Storage},
  booktitle = {Proceedings of the 7th ACM European Conference on Computer Systems (EuroSys '12)},
  pages = {183--196},
  year = {2012},
  publisher = {ACM},
  doi = {10.1145/2168836.2168855}
}
```
