# MICA: A Holistic Approach to Fast In-Memory Key-Value Storage

> NSDI 2014 — structured reading notes

Full paper content: [Markdown conversion](../original/mica-nsdi-2014.md)

## Paper information

- **Authors:** Hyeontaek Lim (Carnegie Mellon University), Dongsu Han (KAIST), David G. Andersen
  (Carnegie Mellon University), Michael Kaminsky (Intel Labs)
- **Venue:** 11th USENIX Symposium on Networked Systems Design and Implementation (NSDI '14),
  April 2–4, 2014, Seattle, WA, pages 429–444
- **Paper page:** https://www.usenix.org/conference/nsdi14/technical-sessions/presentation/lim
- **Name:** MICA = **M**emory-store with **I**ntelligent **C**oncurrent **A**ccess

## One-sentence summary

MICA reaches **65.6–76.9 million key-value operations per second on one commodity server —
4–13.5× the next fastest system** — by making an unconventional choice in each of three layers at
once: **exclusive-access partitioned data**, a **kernel-bypass network stack where clients encode
the target core into the UDP port so the NIC delivers each request to the right core**, and
**circular logs plus lossy hash indexes** that make writes as cheap as reads.

## Goals and non-goals

**Non-goals** (stated first, deliberately):

- **No change to cluster architecture** — sharding, load balancing, replication, and failure
  recovery across nodes are unaffected.
- **No large items spanning multiple packets.** Most items fit in one packet; a client can store a
  large item elsewhere and put a pointer in MICA, since **one extra round trip costs less than
  transferring a multi-packet item**.
- **No durability.** All data is in DRAM; log-based mechanisms like RAMCloud's would be needed to
  survive power failure.

**Goals:**

| Goal | Reasoning |
| --- | --- |
| **High single-node throughput** | Sites like Facebook replicate nodes purely to handle load. Fewer, faster nodes reduce cost, replication and invalidation overhead, and handle spikes and hot spots better. Critically, **a single user request can create more than 500 key-value requests**; fewer nodes reduce fan-out and thus tail-driven job completion time |
| **Low end-to-end latency** | Matters most when a client sends dependent back-to-back requests. Minimize both local processing latency and round-trip count |
| **Consistent performance across workloads** | Real workloads are Zipf-distributed, and modern uses demand **write-intensive** performance too |
| **Small, variable-length items** | Most items are small; the target is packet-processing speed — **40–80 Gbps, like software routers**. Variable length requires careful memory management to limit fragmentation |
| **Standard KV interface and semantics** | `GET`/`PUT`/`DELETE`. **Cache mode** may evict at its discretion (e.g. LRU); **store mode** must never remove an item without client permission while keeping good memory utilization |
| **Commodity hardware** | Cheaper to develop, buy, and operate; today's servers match specialized FPGA/RDMA hardware for high-speed I/O |

**Why prior work falls short:** some systems support only small fixed-length keys; many rely on
**client-side request batching** to amortize network I/O, which is less effective in large
installations because it is hard to accumulate multiple requests for the same server; some use
specialized hardware (FPGAs, RDMA NICs), often with multiple round-trips or no item eviction; many
lack evaluation on skewed workloads, and some are **slower for writes than reads**. Software
routers set the bar for how fast a system *might* go, but do not teach how to apply their
techniques to higher-level key-value processing.

## The three unconventional choices

### 1. Parallel data access: partition and access exclusively

| Model | How | Problem |
| --- | --- | --- |
| **Concurrent access** (most KV systems) | Multiple cores share data, protected by mutexes, optimistic locking, or lock-free structures | **Concurrent writes scale poorly** — frequent cache line transfers, since only one core can hold a line for writing at a time |
| **Exclusive access** (rarer) | Partition ("shard") data so each core owns its partition, with no inter-core communication | Prior work found partitioning gives the best throughput **but loses badly under imbalanced load**, i.e. skewed key popularity. It also requires **request direction** to get each request to the right core |

**MICA's choice:** partition and use **mainly exclusive access**, then attack the skew weakness
directly — exploiting CPU caches and packet burst I/O to **disproportionately speed up the more
loaded partitions**. MICA can fall back to **concurrent reads** under extreme skew but **never uses
concurrent writes**, which are always slower than exclusive writes.

### 2. Network stack: bypass the kernel, and let clients steer packets

**Network I/O.** TCP processing alone can consume **70% of CPU time** on a many-core optimized
key-value store. Socket I/O is portable but has high per-`read()` overhead, which is why systems
batch requests into larger packets. **Direct NIC access** (as in software routers) bypasses the
kernel, delivers packets in bursts to use CPU cycles and the PCIe bus efficiently — but gives up
TCP retransmission, flow control, and congestion control. **MICA takes direct NIC access**,
arguing that small items need fewer transport features and **clients are responsible for
retransmission**.

**Request direction.**

| Mechanism | How | Limitation |
| --- | --- | --- |
| **Flow-level core affinity** — RSS (hash the 5-tuple) or Flow Director (flexible header fields + a user table) | Reduces **transport-layer** contention | Does **not** reduce application-level contention, because one flow can contain requests for any object; even for datagrams the benefit is small due to lack of locality across datagrams |
| **Object-level core affinity** | Requests for the same key all go to the core owning that key's partition | **Commodity NICs cannot parse application-level semantics**, and software redirection (message passing) reintroduces the inter-core communication exclusive access exists to avoid |

**MICA's choice:** use **Flow Director**, but have **clients encode object-level affinity
information in a form Flow Director understands** — with servers telling clients the
object-to-partition mapping.

### 3. Data structures: exploit cache semantics

**Memory allocator.**

- **Dynamic object allocator** (Memcached's slab approach): size classes with segregated pools and
  a global manager rebalancing large blocks between them. The problem is **fragmentation** — some
  classes have no free blocks while others' blocks sit partly empty after deletions. Defragmenting
  requires **expensive memory copies**, made worse if rebalancing runs concurrently with readers
  and writers.
- **Append-only log**: new items go at the tail; updates append an overriding item. Sequential
  access means **fewer cache and TLB misses**. Common in flash stores, rare in in-memory ones. The
  problem is **garbage collection** — reclaiming space by copying live items to a new log is costly,
  trading memory efficiency against request processing speed.

**MICA's choice:** separate allocators per mode. **Cache mode** uses a log structure with
**inexpensive garbage collection and in-place update support**, exploiting cache semantics to
**eliminate log GC entirely** and drastically simplify defragmentation. **Store mode** uses
**segregated fits over a unified memory space**, avoiding size-class rebalancing.

**Indexing.**

- **Read-oriented** (hash tables, trees) are much slower for writes: hash tables examine many slots
  to find space; trees need multiple operations to maintain invariants.
- **Write-friendly chaining** inserts cheaply but faces a time-space tradeoff — long chains mean
  multiple **random dependent memory accesses** per lookup, short chains mean high pointer overhead.
- **Lossy structures** are unusual in software key-value stores but are **the standard design in
  hardware indexes such as CPU caches**.

**MICA's choice:** in cache mode, a **lossy index that evicts an old item on hash collision**
instead of spending resources resolving it — making insert speed comparable to lookup speed. In
store mode, **bulk chaining** allocates a cache-line-aligned spare bucket per chain segment,
keeping chains short and space efficiency high.

## Design in detail

### Keyhash-based partitioning

A **keyhash** is the 64-bit hash of the key, **computed by the client** and used throughout
processing. Its **high-order bits select the partition**.

Why hashing tames skew: in a Zipf(0.99) population of 192 Mi keys (the YCSB distribution), **the
most popular key is 9.3 × 10⁶ times more frequently accessed than average — but after partitioning
into 16 partitions, the most popular partition receives only 53% more requests than average.**

MICA then handles the residual partition-level skew by processing hot partitions **more
efficiently**, for two reasons:

1. **A partition is popular *because* it holds hot items**, so it has high data locality and fewer
   CPU cache misses.
2. **Skew makes packet I/O more efficient for popular partitions** — a busier core does I/O less
   frequently, so its bursts are larger and its **per-packet I/O cost falls**.

Result: **throughput on the Zipf workload is 86% of the uniform workload.**

### Operation modes

| Mode | Reads | Writes | Purpose |
| --- | --- | --- | --- |
| **EREW** (Exclusive Read Exclusive Write) | One core per partition | Same core | No synchronization or inter-core communication → **linear scaling with cores** |
| **CREW** (Concurrent Read Exclusive Write) | Any core | One core | All cores serve reads under heavy skew, while exclusive writes still avoid cache line transfer; read-write conflicts handled by **efficient optimistic locking** |
| **CRCW** (Concurrent Read Concurrent Write) | Any core | Any core | **Provided only as a baseline** modelling non-partitioned systems |

**The CREW cache-semantics problem:** a `GET` may need to update cache management state (e.g. LRU
bookkeeping), causing conflicts and cache-line bouncing — defeating the point of exclusive writes.
**MICA's approximate answer: count reads only from the exclusive-write core.** Since clients
round-robin CREW reads across cores in a NUMA domain, this is effectively **sampling-based
approximate LRU**.

### Network stack

Requests and responses use **UDP**, with **sequence numbers in packets** and reliance on **the
idempotency of GET and PUT** for stateless, application-driven loss recovery — justified because
some queries are useless past a deadline, and well-provisioned networks make retransmission rare
and congestion control less crucial.

- **Direct NIC access via Intel DPDK.** NUMA-aware allocation ensures **each CPU and NIC touches
  only packet buffers in its own NUMA domain**. NIC multi-queue support gives **each core a
  dedicated RX and TX queue**, accessed without synchronization, mirroring EREW.
- **Burst packet I/O** (up to 32 packets per transfer) reduces the per-packet cost of queue access
  while adding only trivial delay. **Crucially it is what makes skew benign:** a core on a popular
  partition spends longer processing, does I/O less often, therefore gets **larger bursts and lower
  per-packet cost**, freeing more CPU for key-value work — while an unpopular partition's core pays
  a higher per-packet cost but handles fewer requests.
- **Zero-copy processing.** MTU-sized RX buffers regardless of request size; on receipt, MICA
  **reuses the request packet to build the response** — flipping source/destination addresses and
  ports and updating only the differing payload bytes. No allocation, no copy.

**Client-assisted hardware request direction.** The client caches a server directory describing
operation mode, core count, NUMA domains, NICs, and partition count. Then:

- For **exclusive** access (EREW read/write, CREW write) the client computes the **partition index
  from the keyhash**.
- For a request any core can serve (a **CREW read**) it picks a **core index round-robin across
  requests, within the same NUMA domain**.
- It **encodes that partition or core index as the UDP destination port** (different port ranges
  distinguish the two).
- The server programs **Flow Director with a "perfect match filter" on the UDP destination port,
  without hashing**, indexing a table that maps ports to RX queues.

The client-side cost is small: with a fast string hash (CityHash), **one dual-6-core client machine
generates over 40 M requests/second including key hashing**. Clients also **include the keyhash in
the request**, so servers reuse the offloaded computation.

### Circular log (cache mode)

Items are appended at the **tail**; **in-place update** is allowed as long as the new key+value
does not exceed the size at first insertion. The log size is **fixed**, so adding to a full log
**evicts the oldest items at the head**. Each entry holds key and value lengths, key, value, the
**initial item size** (to locate the next item and support resizing), the **keyhash** (for fast
lookup), and a client-set **expire time**.

**Garbage collection and defragmentation disappear.** Deleted items are automatically collected
when new items enter the log, and **almost all free space stays contiguous between tail and head**.

**Eviction of live items is a feature.** Items evicted at the head are **not reinserted even if
unexpired** — valid under cache semantics. This gives eviction policies for free:

- **Natural policy: FIFO.**
- **True LRU:** reinsert any requested item at the tail, since only the least recently used items
  are evicted at the head.
- **Approximate LRU:** reinsert **selectively**, ignoring items already close to the tail — same
  effect without frequent reinserts, because recently accessed items stay far from the head.

**Low-level memory management:** **hugepages** (2 MiB) cut TLB misses substantially, and allocation
is NUMA-local. A neat trick avoids range checking: MICA **maps the virtual addresses right after
the end of the log to the same physical pages as the log's start**, so the log appears locally
contiguous and an entry near the end can be read without an invalid access or segfault.

### Lossy concurrent hash index (cache mode)

A **set-associative cache, like a CPU cache**: multiple buckets (count configurable per workload),
each with a fixed number of index entries (**15 in the prototype, occupying exactly two cache
lines**). Part of the keyhash selects the bucket; the item may occupy any entry in it.

Each entry holds a **tag** (another portion of the keyhash, used to filter non-matching lookups
without touching the log) and the **item offset in the log**. **A zero tag value is avoided** —
zero marks an empty entry — and deletion writes zeros, after which the log entry is garbage
collected automatically. The keyhash bits used for partition index, bucket number, and tag **do not
overlap**; 64-bit keyhashes provide enough bits.

**Lossiness:** inserting into a full bucket **evicts an entry chosen by age** — the entry whose item
offset is furthest behind the log tail, i.e. the oldest (or least recently used, if the log uses
LRU). This **avoids the expensive hash collision resolution that lossless indexes need**, making
**insert speed comparable to lookup speed**.

**Dangling pointers.** When the log evicts an item, MICA does *not* delete the index entry —
storing back pointers would require a random memory write and locking. Instead it uses **wider
offsets than needed**: 48-bit offsets for a 16 GiB (34-bit) log. A pointer is detected as dangling
if `(Tail − ItemOffset + 2⁴⁸) mod 2⁴⁸ > LogSize`. Since the tail eventually wraps the 48-bit space
and could make a stale pointer look valid, MICA **scans the index incrementally**, needing only to
complete a cycle before the tail wraps. The overhead is negligible: **at 2³⁰ bytes/second of writes
the tail wraps every 2¹⁸ seconds, so with 2²⁴ buckets MICA scans just 2⁶ buckets per second.**

**Concurrent access.** Each bucket has a **32-bit version number**. Readers proceed
**optimistically, generating no memory writes**: check the version is even before, re-read after
fetching from index and log, and retry on mismatch. Writers **increment before and after**. In
CRCW mode a writer additionally **spins with compare-and-swap until the version is even**. The
prototype optimizes further: **CREW uses plain instructions exploiting x86 memory ordering** (one
writer only), and **EREW ignores version numbers entirely** — which is why the prototype cannot
switch modes at runtime.

**Multi-stage prefetching.** Request parsing, index lookup, and log retrieval all cause random
memory access that stalls cores on cache and TLB misses. MICA **interleaves computation and memory
access in a software pipeline**: on a burst of 8 packets it fetches packets 0–1 while prefetching
2–3; decodes 0–1 and prefetches the index buckets they will touch while prefetching payloads 4–5;
then prefetches the log entries for 0–1 while prefetching index buckets for 2–3 and payloads 6–7 —
continuing until all requests are processed.

### Store mode

**Segregated fits** over a unified space: size classes incrementing by 8 bytes, each with a
freelist. Insert picks the smallest class large enough with free space, stores the item, and
**returns the unused remainder to the matching freelist**; delete **coalesces adjacent free regions
using boundary tags**.

The contrast with Memcached's SLAB allocator is the point: **Memcached's simple segregated storage
dynamically assigns blocks to size classes, effectively partitioning memory by class and requiring
rebalancing.** MICA's unified space needs no rebalancing, and since **MICA has already partitioned
by keyhash**, adding a second partitioning by size class would waste memory by allocating whole
blocks for few items.

**Bulk chaining** converts the lossy index into a lossless one: the lossy index becomes the **main
buckets**, with a smaller number of **spare buckets** allocated separately. On overflow, an unused
spare bucket is chained onto the full bucket; **if no spare buckets remain, MICA rejects the item
with an out-of-space error**.

Two properties make this work:

- **Main buckets store about 95% of items**, so index lookups need **close to 1 random memory
  read** — versus **1.5 expected** for the cuckoo hashing used in improved Memcached systems.
- **Spare buckets need only be 10% of the main buckets** to hold the entire 192 Mi-item dataset.

## Evaluation

### Setup

- **Server:** dual 8-core Intel Xeon E5-2680 @ 2.70 GHz (Hyper-Threading disabled), 20 MiB L3 per
  CPU, 64 GiB RAM (32 GiB per NUMA domain over quad-channel DDR3-1600), **eight 10 GbE ports (four
  Intel X520-T2)**, two NICs directly attached to each socket via PCIe gen2, QuickPath between
  sockets. Half the memory reserved for hugepages. **16 partitions**, one per core; cache mode uses
  approximate LRU.
- **Clients:** two machines, dual 6-core Xeon L5640, two X520-T2 each, directly connected to the
  server (no switch), each connected to NICs in both NUMA domains.
- **MICA:** 12K lines of C on x86-64 Linux, DPDK 1.4.1.
- **Compared systems:** Memcached, MemC3, Masstree, RAMCloud — **all modified to use MICA's
  lightweight network stack**, so the comparison isolates data structures and parallelism rather
  than socket overhead. **No client-side request batching in any experiment.** RAMCloud's statistics
  collection was disabled after it was found to cause lock contention.
- **Datasets:**

  | Dataset | Key size | Value size | Count |
  | --- | ---: | ---: | ---: |
  | Tiny | 8 B | 8 B | 192 Mi |
  | Small | 16 B | 64 B | 128 Mi |
  | Large | 128 B | 1024 B | 8 Mi |

- **Workloads:** uniform vs. **Zipf skewness 0.99** (YCSB's); **50% GET (write-intensive, YCSB-A)**
  and **95% GET (read-intensive, YCSB-B)**.
- **Methodology note:** the request rate is tuned to allow only **marginal packet loss (<1% at any
  NIC port)** rather than flooding for best-effort processing — which could boost measured
  throughput by more than 10% but "real deployments would not tolerate excessive packet losses, and
  such flooding can distort the intended skew by causing biased packet losses at different cores."

### Throughput

**Tiny items:**

| Configuration | MICA |
| --- | --- |
| Uniform | **75.5–76.9 Mops** |
| Skewed | **65.6–70.5 Mops** — at most a **14% penalty** from skew |

At that rate MICA uses **54.9–66.4 Gbps**, very close to the **66.6 Gbps its network stack can
handle doing packet I/O alone**. The next best system is **Masstree at 16.5 Mops**; the others are
below 6.1 Mops. **All systems except MICA suffer noticeably under write-intensive 50% GET.**

**Small items** show similar results, but the gap narrows because **MICA becomes network
bottlenecked** while the others never saturate the network.

**Large items** exacerbate the network bottleneck: MICA achieves **12.6–14.6 Mops at 50% GET and
8.6–9.4 Mops at 95% GET**. Note the inversion — **MICA is faster at lower GET ratios** because
responses can omit key and value, using less bandwidth, whereas **every other system is faster at
95% GET because they are locally bottlenecked, not network bottlenecked.**

Cache and store modes differ only marginally.

**Skew resistance, quantified.** Under skew, several cores process *more* requests than under
uniform load, because:

- the most loaded core's **RX burst size grows from 10.2 to 17.3 packets per I/O**, cutting its
  per-packet I/O cost, and
- **average cache hit ratio across all cores rises from 67.8% to 77.8%.**

A local benchmark without networking confirms skewed workloads give better local key-value
throughput purely from data locality.

### Latency

Against original Memcached with the kernel network stack, uniform 50% GET on tiny items:
**MICA's end-to-end latency is 24–52 µs**, varying with offered throughput, while Memcached's is
nearly flat up to its ceiling. **At a comparable ~40 µs latency, MICA delivers 69 Mops — more than
two orders of magnitude more than Memcached.** Because MICA uses **a single round trip per
request**, unlike RDMA-based systems, the authors claim best-in-class low-latency key-value
operations.

### Scalability

- **CPU cores** (skewed tiny-item workloads, hardest case for partitioned stores): **only MICA and
  Masstree improve with more cores**. Memcached, MemC3, and RAMCloud **peak at 2 cores** for 50%
  GET. Interesting note: MemC3 reaches **5.7 Mops at 4 cores** here versus 4.4 Mops at 16 cores in
  its own paper — because the faster network stack **exposes a different bottleneck** and changes
  the optimal core count.
- **NIC ports:** MICA scales with available bandwidth because **it can use almost all of it for
  request processing**, and the GET ratio barely matters. Masstree matches MICA at 2 ports under 95%
  GET but **neither it nor the others scale with more ports**.

### Why the holistic approach is necessary

**Parallel data access modes** (end-to-end, tiny items): EREW is consistently good; **CREW is
slightly better at high GET ratios on skewed workloads**, since despite version-management overhead
it can use multiple cores for popular items without excessive inter-core communication. **CRCW
offers no benefit over either** — though it still beats every other system — "this suggests that we
should avoid CRCW."

**Network stack.** Simply moving Masstree onto MICA's stack raised it from **8.9 Mops with request
batching (its own paper) to 16.5 Mops without batching**. But hardware request direction matters
independently: a software-only alternative — clients round-robin to any core, server cores forward
via DPDK inter-core queues — achieves **only 40.0–44.1% of MICA's throughput** due to inter-core
communication overhead. **"MICA's request direction is crucial for realizing the benefit of
exclusive access."**

**Data structures.** Partitioning an existing structure is not enough: a **"partitioned Masstree"**
(one instance per core, concurrency disabled, same partitioning and request direction as MICA)
achieves **only 8.2–27.3% of MICA's performance under skew** — and its **50% GET throughput is even
lower than non-partitioned Masstree**:

| System | 50% GET | 95% GET |
| --- | ---: | ---: |
| Partitioned Masstree | 5.8 Mops | 17.9 Mops |
| **MICA** | **70.4 Mops** | **65.6 Mops** |

**Conclusion: "the holistic approach is essential; any missing component significantly degrades
performance."**

## Positioning against related work

- **Non-partitioned DRAM stores** — Memcached, RAMCloud, MemC3, Masstree, Silo — use one partition
  per node. Masstree and Silo show partitioning can be efficient on some workloads but slow under
  skew and cross-partition transactions. **MICA can partition successfully because the simple
  key-value requests it targets never cross partitions**, and because burst I/O and locality make
  loaded partitions run faster.
- **Partitioned systems** — Memcached on Tilera, CPHash, Chronos — exclusively access partitioned
  hash tables like MICA's EREW, but **lack anything like CREW** for read-intensive skewed workloads.
- **H-Store and VoltDB** use single-threaded per-partition execution engines and therefore need
  careful data partitioning (even using machine learning) and dynamic load balancing. **MICA gets
  similar throughput on uniform and skewed workloads without that effort**, because keyhash
  partitioning mitigates skew and burst I/O plus cache-friendly access absorbs the rest.
- **Low-latency systems:** RAMCloud achieves 4.9–15.3 µs and Chronos ~10 µs average / 30 µs p99 on
  **InfiniBand and Myrinet**; Pilaf serves reads with one-sided RDMA. MICA's 10 GbE has a much
  higher base latency, and evaluation on a low-latency network was left as future work.
- **Affinity-Accept** also uses Flow Director on commodity NICs, but to load balance TCP
  connections. **Chronos directs requests using client-supplied information like MICA, but classifies
  in software**, which is far slower for small requests.
- **Eviction cost:** MemC3 replaced Memcached's LRU with a CLOCK approximation to avoid list-
  management contention. **MICA's lossy log and index support common eviction schemes at low cost by
  construction**, and bulk chaining extends the same index to lossless operation.

## Limitations and questions

- **No durability at all.** Everything is in DRAM; persistence is explicitly out of scope.
- **No multi-packet items**, no range queries, no multi-key transactions. The design's key
  assumption — **requests never cross partitions** — is what makes exclusive access viable, and it
  fails for richer semantics. The authors flag applying MICA's techniques to durable, range-query,
  or transactional systems as future work whose difficulty is exactly this.
- **UDP with client-side recovery.** No retransmission, flow control, or congestion control;
  correctness relies on GET/PUT idempotency and a well-provisioned network.
- **Client-server co-design required.** Clients must hash keys, cache a server directory, and
  encode routing in the UDP port — MICA is not a drop-in server for existing clients.
- **Mode switching is not supported at runtime** because EREW/CREW synchronization is hard-coded
  for speed.
- **Cache mode silently drops data.** The log evicts unexpired live items and the index evicts
  entries on collision — correct under cache semantics, unacceptable under store semantics without
  the segregated-fits/bulk-chaining variant.
- **Store mode can reject writes** with an out-of-space error when spare buckets are exhausted.
- **Approximate LRU in CREW** counts reads only from the exclusive-write core — a sampling
  approximation, not true LRU.
- **Comparisons carry caveats the authors state:** Masstree supports range queries and RAMCloud
  targets InfiniBand, while neither supports automatic item eviction; the evaluation deliberately
  compares only common single-key operations.
- **The client-side workload generator samples responses** to work around a PCIe RX issue, so not
  every response is received (the server does full packet RX).

## Practical design checklist

MICA's approach fits when:

- requests are **independent single-key operations on small items**;
- the deployment can **co-design clients**, or clients are a library you control;
- you can **bypass the kernel** and program the NIC's flow steering;
- durability and range queries are provided elsewhere (or not needed);
- **skewed key popularity** is expected — MICA's answer is better than either naïve partitioning or
  full sharing.

Look elsewhere when:

- operations cross partitions (transactions, scans, secondary indexes);
- you need durability, TCP semantics, or standard client compatibility;
- items are large enough that network bandwidth, not CPU, is the binding constraint — MICA's
  advantage shrinks exactly there.

## Takeaways

1. **Optimizing one layer just moves the bottleneck.** Every component was necessary: software
   request direction cost 56–60%, partitioning Masstree without MICA's data structures cost 73–92%,
   and even the best data structures need the kernel-bypass stack to be visible.
2. **Hash the key, then partition — skew mostly evaporates.** A 9.3-million-fold key skew becomes a
   53% partition skew, which is a tractable engineering problem rather than a fundamental one.
3. **Make the hot partition faster, not the load flatter.** Larger RX bursts (10.2 → 17.3 packets)
   and better cache locality (67.8% → 77.8% hit rate) mean a busier core is a **more efficient**
   core — turning partitioning's classic weakness into a mild 14% penalty.
4. **Let the client do the routing.** Encoding the target partition in the UDP destination port
   turns application-level affinity into something a commodity NIC's exact-match filter can
   execute — no software redirection, no inter-core messaging.
5. **A bounded circular log makes garbage collection disappear.** Eviction at the head *is* the
   collector, free space stays contiguous, and FIFO/LRU/approximate-LRU all fall out of reinsertion
   policy.
6. **Losing data on collision is a legitimate design choice** — it is what CPU caches do. It makes
   insert as fast as lookup, and bulk chaining converts the same structure to lossless when store
   semantics are required.
7. **Wide pointers substitute for back pointers.** 48-bit offsets over a 34-bit log make dangling
   references detectable by arithmetic, with a background scan whose cost is 2⁶ buckets per second.
8. **Pipeline the memory accesses, not just the packets.** Multi-stage prefetching across parse →
   index → log for a burst of packets is what keeps cores from stalling on the three unavoidable
   random accesses per request.

## Citation

```bibtex
@inproceedings{lim2014mica,
  author = {Hyeontaek Lim and Dongsu Han and David G. Andersen and Michael Kaminsky},
  title = {{MICA}: A Holistic Approach to Fast In-Memory Key-Value Storage},
  booktitle = {11th USENIX Symposium on Networked Systems Design and Implementation (NSDI 14)},
  pages = {429--444},
  year = {2014},
  publisher = {USENIX Association}
}
```
