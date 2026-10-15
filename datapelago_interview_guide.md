# DataPelago (Nucleus) Founder Interview Guide

Role: **engine / systems engineer on Nucleus**. Format: mixed founder conversation (background, vision fit,
technical depth, why join). Interviewer: **Rajan Goyal**, founder and CEO.

How to read this guide. Every claim is tagged:
- **[Public]**: stated on DataPelago/AWS/NetApp pages or press coverage I read.
- **[Vendor]**: a DataPelago performance claim that no independent source verified. Quote it as "they report".
- **[General]**: standard engineering knowledge, not specific to DataPelago.
- **[Ask]**: not publicly documented; ask the founder instead of asserting.

## Contents
1. [Company and Founder Snapshot](#1-company-and-founder-snapshot)
2. [The NetApp Acquisition: What It Changes](#2-the-netapp-acquisition-what-it-changes)
3. [Nucleus Architecture, Step by Step](#3-nucleus-architecture-step-by-step)
4. [Competitive Landscape](#4-competitive-landscape)
5. [Technical Depth Q&A](#5-technical-depth-qa)
6. [Design Exercise: Safe GPU Acceleration for Spark SQL](#6-design-exercise-safe-gpu-acceleration-for-spark-sql)
7. [Founder Conversation Prep](#7-founder-conversation-prep)
8. [Behavioral Stories to Prepare](#8-behavioral-stories-to-prepare)
9. [30-60-90 Pitch and Cheat Sheet](#9-30-60-90-pitch-and-cheat-sheet)
10. [References](#references)

---

## 1. Company and Founder Snapshot

| Item | Detail |
|---|---|
| Company | DataPelago, Mountain View, CA. Note the spelling: it is **DataPelago**, not "Dataplego" |
| Founded / funding | Founded 2021 by Rajan Goyal (CEO) and Anand Iyer (CPO); out of stealth Oct 2024; raised $75M+, latest round $47M in 2024 [Public] |
| Product | **Nucleus**, a "Universal Data Processing Engine"; packaged for Spark as **DPA-S** (DataPelago Accelerator for Spark) [Public] |
| Pitch | Make existing engines (Spark, Trino, Ray, Dask) run on CPUs/GPUs (company also lists FPGAs) with no application changes, for better price/performance [Public] |
| Founder background | Goyal led hardware-software co-design at **Cavium** (processors for internet-scale services) and later helped invent **DPUs** (data processing units) to cut data-movement latency in cloud infrastructure; 150+ patents [Public] |

**Founder thesis (from his "Why I started DataPelago" post) [Public]:**
1. The bottleneck has moved from I/O to **compute**; CPU hardware and software can't keep up with data growth.
2. Scaling went clock speed -> cores -> more servers; the next step is **domain-specific hardware plus refactored software**.
3. Unstructured data is about 90% of new data and growing double digits a year, mostly "stored and ignored" (under 1%
   analyzed); preparing data for LLMs is often the most expensive step of AI projects.
4. Open-source engines (Spark, Trino/Presto) can be accelerated without code changes, avoiding vendor lock-in.

**Why this matters for you:** Goyal is a hardware/systems person. Expect him to value (a) first-principles reasoning
about memory bandwidth, data movement and hardware utilization, and (b) co-design thinking (what should the hardware
do vs the software). Frame your answers around where the bytes live and how many times they move.

---

## 2. The NetApp Acquisition: What It Changes

- **Announced July 2026**: NetApp is acquiring DataPelago, which will operate as a wholly owned subsidiary. Price and
  close date were not disclosed in coverage I read [Public].
- NetApp's framing: GPU-accelerated data processing **at the storage layer**, "zero-copy activation" of enterprise
  data for AI, processing data where it lives and supporting lakehouse formats such as Iceberg and Delta Lake [Public].
- What to infer (label as your inference): the roadmap likely shifts toward **data-local execution** (compute near
  ONTAP-style storage), enterprise/on-prem deployments, and integration with NetApp's AI data stack, in addition to
  cloud Spark acceleration.

**Smart things to ask (see section 7):** which team does Nucleus sit in after close; does the roadmap prioritize
storage-resident execution or cloud Spark; hiring plan and where the team is based; how independent the engine's
architecture stays.

**Risk to be aware of:** a pending acquisition means uncertainty about compensation structure (equity vs NetApp),
reporting lines and timing. Ask directly and politely.

---

## 3. Nucleus Architecture, Step by Step

### Component table [Public, from DataPelago docs/press unless tagged]

| Layer | What it does |
|---|---|
| Host engine (Spark/Trino/Ray) | Parses SQL/dataframe, runs its own optimizer, produces a **physical plan**. Unchanged. |
| **DPA-S plugin** | Spark plugin JAR installed in driver and executors; intercepts the physical plan; no infrastructure changes |
| **DataApp / unified query engine** | Converts plans into standard form via **Apache Gluten / Substrait**; supports SQL and Python |
| **Intelligent Execution Optimizer** | Builds execution pipelines and **chooses hardware per operator** using real-time cost/performance analysis |
| **DataVM** | A virtual machine with a **domain-specific instruction set** for data operations; targets CPU and GPU via LLVM, CUDA, ROCm |
| **DataOS** | Schedules/places operations on hardware at runtime (described by secondary coverage; its internals are [Ask]) |
| Engine implementation | Written in **Rust**, vectorized and **columnar** end to end |

### End-to-end flow of a Spark query

1. User runs a Spark SQL query; Catalyst parses, analyzes and optimizes it as usual and produces a physical plan.
2. The DPA-S hook intercepts the plan **before execution** (adds a pre-execution step at the executor level).
3. Eligible operators (filters, projections, joins, aggregations, sorts, string processing) are converted to a
   Substrait-style intermediate representation through Gluten.
4. The optimizer decides per operator/stage: GPU, or vectorized CPU. Ineligible stages stay on standard Spark.
5. The plan becomes a **data flow graph** that DataVM/DataOS executes, managing concurrent operators and resources.
6. Data is read from the existing store (S3: Parquet, Delta, ORC) with no migration; results return to Spark.
7. If a stage can't be accelerated, execution **falls back to standard Spark automatically**, per query/stage, instead
   of failing the job (stated in the AWS partner post) [Public].

### What is *not* public ([Ask] list)
Memory management and host-device transfer strategy, spill behavior, how DataOS places work, how fallback boundaries
convert data formats between Nucleus and Spark's row/columnar representations, what the cost model uses, FPGA status
(the company lists FPGAs; NetApp coverage mentions CPUs and GPUs).

### Vendor performance claims (quote as "they report") [Vendor]
- Up to 10x faster Spark and up to 80% lower cost; customers report 2-10x faster and 50-80% cheaper; micro-benchmarks
  up to 60x on memory/compute-heavy operations (AWS partner post).
- Versus NVIDIA cuDF on one H100 (Nucleus v3.0 vs cuDF 25.04): 3.7-10.5x on projection/filter, 3.3-4.5x on
  aggregation, up to 4.5x on memory-intensive hash join/aggregation, up to 38.6x on variable-length string hash
  joins. Benchmarks used a P5.48xlarge (8x H100); the post names no standard suite such as TPC-DS.

---

## 4. Competitive Landscape

| System | Approach | How Nucleus differs (as positioned) |
|---|---|---|
| Spark RAPIDS / cuDF (NVIDIA) | GPU-only operators via CUDA libraries; plugin to Spark | Nucleus claims hardware-agnostic VM, per-operator placement across CPU/GPU, better memory-heavy ops |
| Databricks Photon | Proprietary vectorized C++ engine, Databricks only | Nucleus accelerates open-source Spark in your own environment |
| Velox + Gluten (Meta/Intel/others) | Open-source vectorized C++ execution library under Spark/Presto | Nucleus builds on Gluten/Substrait for plan handoff; its own engine is Rust and accelerator-first |
| Apache DataFusion / Comet | Rust-based engine; Comet accelerates Spark with DataFusion | Closest in language; Comet is CPU-first, open source |
| Trino/Presto native (Prestissimo) | Velox-based C++ workers | Same family as Velox |

**Say-it-in-one-sentence positioning:** "Most acceleration stacks pick one backend. Nucleus puts a hardware
abstraction (DataVM) under the operators and an optimizer on top, so the same plan can run wherever it's cheapest."
Then be ready to probe whether the abstraction costs performance versus hand-tuned CUDA kernels [General question to ask].

---

## 5. Technical Depth Q&A

Each answer is written to be said out loud (about 1-2 minutes), followed by likely follow-ups with short answers.
Answers are [General] engineering knowledge unless tagged; numbers are round, typical figures, not DataPelago
measurements. Where a question touches an undocumented Nucleus detail, the answer ends with a question for Goyal
instead of a guess.

### Q1. Why is GPU analytics often limited by memory transfer, not compute?

**Spoken answer:**
"A GPU is extremely fast at the math but it sits across a bus from the data. On a typical server, PCIe Gen4 x16 moves
roughly 25-30 GB/s in each direction, Gen5 about double that, while the GPU's own HBM runs at around 3 TB/s. So the
memory on the card is about a hundred times faster than the road into it. Take a 100 GB scan: shipping it over PCIe
takes about four seconds, but the GPU could chew through it in a few tens of milliseconds. If I copy raw data in,
run one cheap filter and copy results out, I've built a very expensive way to wait on PCIe.

"So the rule is: do enough work per byte moved. Practically that means four things. First, reduce bytes before they
cross the bus: push filters and projections down so I only ship needed columns, and ship data compressed or encoded
then decode on the GPU. Second, keep intermediate results on the device across operators, so a filter, join and
aggregate run as one on-device pipeline rather than round-tripping to the host. Third, overlap copy and compute with
streams and double buffering, so while batch N is computing, batch N+1 is transferring. Fourth, where the hardware
allows, skip the host: GPUDirect Storage reads from NVMe or network straight into GPU memory.

"The way I think about it is arithmetic intensity: operations per byte transferred. Hash joins, aggregations over many
groups, sorts and string work are high intensity, so the GPU wins. A selective filter that returns 1% of rows, or a
simple projection, is low intensity and can be as fast on a vectorized CPU. That's why a per-operator placement
decision, which is what I understand Nucleus's optimizer to do, has to price the transfer between operators and not just
each operator in isolation."

**Follow-ups:**
- *When is the CPU faster?* Small inputs where launch and transfer overhead dominates (a kernel launch is
  microseconds, a copy has fixed latency), highly selective scans, trivial expressions, and branch-heavy
  row-at-a-time logic like some UDFs.
- *How would you measure it?* Profile with Nsight Systems: look at the timeline for gaps between kernels, the
  fraction of time in H2D/D2H copies, and achieved memory bandwidth versus the card's peak.
- *Does NVLink help?* Between GPUs yes (hundreds of GB/s), and on some platforms between CPU and GPU, but the
  general design rule still holds; it narrows the gap rather than removing it.
- *Question for Goyal:* "How does your optimizer account for host-to-device transfer when it picks hardware per
  operator?" [Ask]

### Q2. Design a GPU hash join.

**Spoken answer:**
"I'll describe an inner equi-join of a large fact table with a smaller dimension table, then the hard cases.

"Build phase: I take the smaller side and build a hash table in device memory. I'd use open addressing with linear
probing because it's cache and coalescing friendly, sized with a load factor around 0.5-0.7 so probe chains stay
short, so memory is about 1.5-2x the build side's keys plus payload row ids. Each thread hashes a key and inserts
with an atomic compare-and-swap into the first empty slot. For payload I store a row index, not the whole row, and
gather columns later.

"Probe phase: I stream the large side in batches. Each thread hashes a probe key, walks the probe sequence until it
hits a match or an empty slot, and emits matches. The output size is data-dependent, so I use two passes: first count
matches per thread block, run a prefix sum to get write offsets, then a second pass writes results. Or I allocate
an upper bound and compact. That avoids contention on a single output counter.

"The hard cases: First, the build side may not fit in GPU memory. Then I partition both sides by hash into
partitions that do fit, a radix or grace-style join, and process partitions one at a time, spilling the others to
host memory. Second, skew: if one key has millions of matches, one thread block does all the work, so I detect heavy
hitters and handle them specially, for example splitting their probe work across blocks. Third, join types: outer
joins need a match flag per build row, semi and anti joins only need existence checks. Fourth, variable-length string
keys: I store strings as an offsets array plus a character buffer, compute and store a fixed-width hash for each,
compare hashes first, and only compare bytes on a hash match, so most probes never touch the string data.

"If the plan says the build side is too large or the data is too skewed, I'd rather fall back to a CPU join for that
stage than fail the query."

**Pseudocode (build + probe sketch):**
```
build(keys, rowids):                       # one thread per build row
    h = hash(key) % capacity
    while atomicCAS(table[h].key, EMPTY, key) != EMPTY and table[h].key != key_dup_policy:
        h = (h + 1) % capacity             # linear probe
    table[h].row = rowid

probe(batch):                              # pass 1: count, pass 2: write
    for each probe row (parallel):
        h = hash(key) % capacity
        while table[h].key != EMPTY:
            if table[h].key == key: emit(probe_row, table[h].row)
            h = (h + 1) % capacity
```

**Follow-ups:**
- *Duplicate keys on the build side?* Either chain row ids (a payload list per key) or store all duplicates as
  separate slots; the count pass must then count all matches.
- *Why not sort-merge join?* Sort-merge is good when inputs are already sorted or memory is tight; hash is typically
  faster when one side fits in memory. A good optimizer chooses based on size and ordering.
- *Why store row ids instead of full rows?* Keeps the table small and the access pattern regular; columns are
  gathered after matching, which also lets you gather only the columns the query needs.
- *Question for Goyal:* "Nucleus reports a large lead on variable-length string hash joins; what's the technique:
  prefix hashing, dictionary encoding, something in the DataVM?" [Ask]

### Q3. Columnar vs row execution, and why vectorized?

**Spoken answer:**
"Analytic queries touch a few columns across many rows. In a columnar layout each column is contiguous, so a filter on
`price` reads only price bytes, wastes no bandwidth on other columns, compresses well (similar values together:
dictionary, run-length, bit-packing), and maps cleanly onto SIMD lanes on a CPU and coalesced loads on a GPU, where
adjacent threads read adjacent addresses.

"Vectorized execution means each operator processes a batch of, say, a thousand to a few thousand values at once, not
one row at a time. That amortizes interpretation overhead: instead of paying a virtual call and branch per row, I pay
once per batch and run a tight loop the compiler can SIMD. Batches are sized to fit in L1/L2 cache. Filters produce a
selection vector or bitmap, so I don't copy surviving rows, and later operators just read through the selection.

"Rows still win in a few places: point lookups of one full record, workloads that produce wide rows for output, and
row-at-a-time user code. So engines like Spark have to convert at the boundaries. Conversion cost matters: a plan that
flips between Spark's representation and a native columnar one repeatedly can lose all its gains, which is why you
want to offload contiguous segments of the plan, not isolated operators."

**Follow-ups:**
- *Why not compile queries to machine code instead of vectorizing?* Code generation (Spark's whole-stage codegen,
  HyPer-style) removes interpretation overhead too; vectorization is simpler to build, debug, and profile and works well
  with SIMD and GPUs. Mature engines mix both.
- *What's a selection vector vs a bitmap?* A selection vector is a list of surviving indexes; a bitmap is one bit per
  row. Bitmaps are cheaper when selectivity is high; vectors when it's low.
- *How does a nested or variable-length column look?* Offsets array plus values buffer, plus a validity bitmap for nulls
  (the Arrow-style layout).

### Q4. How do you offload an operator safely, with fallback?

**Spoken answer:**
"I'd treat it as a three-layer safety net: decide conservatively, convert carefully, recover at runtime.

"Decide: for every plan node and every expression inside it, check a capability registry. Is this data type
supported, is this function implemented with Spark-identical semantics, are the null and overflow rules covered? If
anything in a node is unsupported, the node stays on Spark. Then I group supported nodes into the largest contiguous
segments so there are as few engine boundaries as possible. Say the plan is Scan, Filter, Project, HashAggregate, Sort,
then a Python UDF. I'd offload everything up to the UDF as one segment and let the UDF stay on Spark; I would not
offload the Filter alone if it forces conversions on both sides.

"Convert: at each boundary I convert between Spark's representation and the native columnar batches, and I count that
conversion as a cost in the placement decision. If offloading a cheap operator adds two conversions, it's a loss and I
keep it on the CPU.

"Recover: at runtime, things happen the planner couldn't know: GPU out-of-memory, a batch with a value the kernel can't
handle, a driver error. For these I rerun just that stage on the CPU path with the same inputs. For that to be safe
the stage must be deterministic and idempotent. Spark already retries tasks, so I'd lean on that: fail the task with a
fallback marker so the retry takes the CPU path. And I'd log every fallback with a reason, so we learn what to
implement next and can see fallback rate per query in production."

**Follow-ups:**
- *How do you avoid ping-ponging?* Penalize each boundary in the cost model and prefer maximal segments; a minimum
  benefit threshold before offloading a segment.
- *Is fallback per operator, stage, or query?* Finer is better for performance but harder to engineer; the AWS partner
  post says Nucleus reverts individual queries and stages to Spark rather than failing the job [Public]. Ask whether
  it goes finer than a stage. [Ask]
- *How do you roll this out safely?* Per-operator feature flags, shadow mode (run both engines and compare results on
  a sample), canary clusters, and a kill switch that disables acceleration instantly.
- *Question for Goyal:* "How do you model boundary-conversion cost, and what's the typical fallback rate on customer
  workloads?" [Ask]

### Q5. Correctness: how do you match Spark's semantics?

**Spoken answer:**
"The promise is 'no application changes', and that's only true if results are identical to Spark, not 'close'. So
I'd list where engines usually disagree and build tests around each.

"Nulls: `NULL = NULL` is unknown, not true; `NOT IN` with a null in the list returns no rows; aggregates skip nulls
except `COUNT(*)`; null-safe equality is its own operator. Decimals: Spark has precise rules for the result precision
and scale of add, multiply and divide, and for what happens when a value overflows. In ANSI mode overflow raises an
error, in legacy mode it can return null, so the native engine must follow the session's mode, including the
casts. Floating point: Spark treats `NaN = NaN` as true and orders NaN above every other value, and normalizes `-0.0`
to `0.0` in grouping and joins; most hardware libraries don't do either by default. Also floating-point addition isn't
associative, so a parallel sum in a different order can produce different low-order bits. Spark itself isn't
deterministic here across partitions, but I'd still want to bound the difference and document it. Time: timestamps
depend on the session time zone and daylight-saving rules. Strings: comparison is UTF-8 byte order by default, and
functions like `substring` count characters, not bytes, so non-ASCII text exposes bugs.

"How I'd test: differential testing, running the same query on Spark and on the native engine over randomly generated
data that deliberately includes edge cases (nulls, empty strings, max and min values, NaN, negative zero,
non-ASCII, empty tables), compare results exactly or within a defined tolerance; plus per-expression conformance tests
and TPC-DS/TPC-H result checks in CI. And in production, a shadow mode that compares sampled query results."

**Follow-ups:**
- *What tolerance is acceptable for float sums?* For deterministic equality, order the reduction deterministically;
  otherwise accept a relative tolerance and make it a documented, configurable behavior. Money should use decimals, which
  are exact.
- *What if a semantic can't be matched on the GPU?* Don't offload that expression; fallback is the right answer.
- *How do you catch regressions across Spark versions?* Run the conformance suite against each supported Spark release
  and pin behavior per version.

### Q6. How would you benchmark this honestly?

**Spoken answer:**
"I'd start by naming the question: what does a customer actually save? That's end-to-end query time and cost, not
operator throughput.

"Setup: fix the hardware, Spark version and configuration; use a standard suite, TPC-DS and TPC-H, at several scale
factors (for example 1 TB and 10 TB), so people can reproduce it; run baseline Spark on a comparable CPU cluster and
the accelerated run. Measure: total query time including scan, shuffle and write, not just the accelerated stages;
cold run versus warm run, since caches and file system state change results; multiple repetitions with p50 and p95,
not the best run; and the breakdown by stage so we know what accelerated and what didn't.

"Then normalize by cost. GPU instances cost more per hour, so the speedup has to beat the price ratio. As an
illustration: if a GPU node costs 3x as much as the CPU node, a 3x speedup is only break-even, and a 10x speedup is
about a 70% cost reduction. Always report cost per query, not only seconds.

"Pitfalls I'd call out: micro-benchmarks on a single operator with in-memory input, which overstate real gains;
comparing against an untuned baseline; ignoring shuffle and I/O; data that fits entirely in cache; and cherry-picking
the queries that accelerated well. DataPelago's published numbers are strong, up to 10x speed and 80% lower cost,
and 3.7x to 38.6x over cuDF on operator benchmarks, but those are vendor-reported and mostly operator-level; I'd
want to see them on full TPC-DS with cost normalization, and I'd be glad to help build that."

**Follow-ups:**
- *Why report p95?* Production cares about tail latency, and GPU memory pressure and fallbacks show up in the tail.
- *How do you handle noisy cloud neighbors?* Repeat runs, dedicated or isolated instances, report variance.
- *What would convince you the speedup is real?* Same result checksums, same cluster cost accounting, publicly
  reproducible scripts.

### Q7. Shuffle and distributed execution: what happens when compute gets faster?

**Spoken answer:**
"Accelerating compute doesn't make the whole query faster by the same amount; it moves the bottleneck. The
formula is Amdahl's law: speedup = 1 / ((1 - p) + p / s), where p is the fraction of time in the accelerated part and
s is how much faster it gets. Suppose 60% of a query's time is compute I can accelerate and 40% is shuffle, network
and storage reads I can't. If I make the compute 10x faster, the total time becomes 0.4 + 0.06 = 0.46, a speedup of about
2.2x. Even infinitely fast compute caps out at 1 / 0.4 = 2.5x. So the first question for any workload is: how much of
the time is actually compute?

"That's why a serious accelerator also attacks the data-movement side. For shuffle: reduce the bytes (project early,
compress, use better serialization), keep the data columnar through the shuffle so I don't pay to convert, and on
GPU-rich clusters consider RDMA or GPUDirect to move data between GPU memory over the network without CPU copies. For
storage reads: push filters into the scan, use column pruning and row-group skipping, cache hot data locally, and
decode on the device. And at the plan level, avoid shuffles the optimizer can eliminate, such as broadcast joins for
small tables.

"What I'd want to know about the product is what fraction of customer workloads are compute-bound; that tells you where
the 2x to 10x range comes from. And it relates directly to NetApp's angle: processing data at the storage layer is a
way to attack the I/O and movement side of this equation."

**Follow-ups:**
- *Can Spark's shuffle be left alone?* Yes for a first version; it's the lowest-risk integration, and the plugin
  keeps Spark's shuffle service, though the conversion cost at the shuffle boundary has to be handled.
- *What about spill?* Faster compute means more concurrent data in memory; spill to host memory or disk needs a plan
  or stages fall back.
- *How do you find the bottleneck?* Spark UI stage timeline: task time split into read, compute, shuffle read/write,
  GC; compare against GPU timeline.

### Q8. Parquet/Iceberg decode on the GPU: why and how?

**Spoken answer:**
"Parquet files store data as compressed, encoded pages: dictionary encoding, run-length and bit-packing, delta
encoding, then a general compression codec like Snappy or Zstd. Decoding those on the CPU can itself be a major cost, and if
I decode on the CPU, I then ship fully expanded data over PCIe, which is bigger. If I decode on the GPU, I ship the
compressed pages, which is smaller, and decode in parallel: pages and column chunks are independent, so thousands
of threads can work at once; bit-unpacking and dictionary lookups map nicely to GPU threads.

"The flow: first, on the host, read the file footer and use its statistics to prune row groups and columns, so I only
fetch what the query needs. That planning happens before any GPU work. Then fetch the needed column chunks, ideally
straight into GPU memory with GPUDirect Storage, then decompress, decode, and build columnar batches on the device,
ready for the next operator without a round trip.

"For Iceberg or Delta tables there's a metadata layer on top: snapshot and manifest planning, partition pruning and
file-level statistics happen on the host and produce a list of files and row ranges. Then there's delete handling:
position deletes and equality deletes must be applied to the scanned rows correctly or results are wrong; I'd apply
them as a filter on device when possible. And schema evolution, with columns added, dropped or renamed, has to be
mapped by column id, not name, when the format uses ids."

**Follow-ups:**
- *Why not decode everything on CPU in parallel?* You can, and for I/O-light workloads it's fine; GPU decode wins when
  PCIe or decode CPU time is the bottleneck.
- *Which encodings are hard on GPUs?* Variable-length (strings, nested) and codecs with sequential dependencies;
  bulk-parallel decompression exists for common codecs but with trade-offs.
- *What's the NetApp angle?* Reading at the storage layer, if compute runs next to the data, removes the network hop
  entirely. That is my inference from NetApp's "zero-copy" framing, so I'd ask how it changes the I/O path. [Ask]

### Q9. Multi-tenancy and GPU sharing: how do you share a GPU across Spark tasks?

**Spoken answer:**
"The scarce resource is GPU memory, not compute. A Spark executor runs many tasks concurrently, say 16 or 32 cores'
worth, but a GPU can't safely run that many full workloads with their working sets at once. So I need to bridge the two
models.

"Options: the simplest is one executor per GPU, with a limit on how many tasks can use the GPU at the same time
(a semaphore), and the rest of the task work, like I/O and shuffle read, overlapping on CPU. A more flexible option on
big cards is to partition the GPU with MIG into isolated slices, each with its own memory and compute, so different
jobs get hard isolation. Time-slicing is the weakest because there's no memory isolation.

"I'd enforce a memory budget: a pool allocator per executor so there's no fragmentation from many small allocations,
tracking of every allocation, and a rule that when a task can't get memory it either waits, spills lower-priority
buffers to host memory, or falls back to the CPU. An out-of-memory crash on a shared GPU would take down every task on
it, so I'd put admission control in front: estimate each stage's memory need from statistics and refuse to run more
than fits.

"For fairness across jobs in a shared cluster, I'd rely on Spark's scheduler pools plus per-tenant GPU quotas, and
export metrics: GPU utilization, memory high-water mark, queue wait, spill bytes."

**Follow-ups:**
- *Why a pool allocator?* Raw `cudaMalloc`/`cudaFree` are slow and fragment memory; a pool reuses large blocks.
- *MIG vs MPS?* MIG gives hardware isolation with fixed slices; MPS lets multiple processes share a GPU with better
  utilization but weaker fault isolation.
- *What if estimates are wrong?* Spill and fallback paths are the safety net; log the misses to improve estimates.

### Q10. Rust for a data engine: pros and cons?

**Spoken answer:**
"Nucleus is written in Rust [Public], and I think it's a defensible choice for a long-lived engine. The main benefit is
memory safety without a garbage collector: the compiler rules out use-after-free, data races and many buffer errors,
which is exactly the class of bug that causes silent data corruption or crashes in an engine that shares buffers
across threads. That matters a lot in a database where a rare bug becomes a wrong answer in production. Rust also gives
predictable performance, with no GC pauses, zero-cost abstractions, good SIMD support, and a strong ecosystem for
columnar data (Arrow, Parquet and DataFusion are Rust).

"The costs: compile times are long; GPU programming still goes through CUDA or LLVM-based paths, so there's an FFI
boundary to C/C++ libraries, and the code at the device boundary has to use `unsafe`, so the safety guarantees are
weaker exactly where the hardest bugs live; and the pool of engineers with deep Rust and GPU experience is smaller than
for C++. Also the Spark side is JVM, so there's a JNI boundary to manage carefully: copying data across it kills
performance, so batches should be shared by memory address (Arrow C data interface), not copied.

"Personally I'd be comfortable ramping up on it: I've [fill in your C++/Rust/systems experience here], and the
concepts of ownership and lifetimes map to how I already reason about buffer lifetimes."

**Follow-ups:**
- *How do you handle JVM-native data exchange?* Pass off-heap buffers via the Arrow C data interface so no copy
  happens; the native side owns or borrows memory with clear lifetime rules.
- *How do you keep `unsafe` contained?* Small wrappers with documented invariants, fuzzing and sanitizers, and tests
  at the boundary.
- *Why not C++ like Velox?* Velox benefits from a huge existing C++ ecosystem; Rust trades that for safety
  guarantees. Neither is wrong.

### Q11. Why a custom VM/instruction set instead of calling CUDA libraries directly?

**Spoken answer:**
"I can see strong reasons and real costs, and I'd like to hear how DataPelago weighs them. The reasons for a VM with a
data-specific instruction set: first, portability. The same operator program can target NVIDIA through CUDA, AMD through
ROCm, and CPUs through LLVM, so the planner doesn't depend on a single vendor and customers aren't locked to one GPU
supplier. Second, fusion: if the engine has its own intermediate representation, it can combine a filter, a projection
and an expression into one generated kernel, so intermediate results stay in registers or cache instead of being
written to memory and read back. That removes memory traffic, which is the real bottleneck. Third, optimization
freedom: the cost model and scheduler can place work on different hardware without rewriting operator logic, and the
backend can improve without touching the planner.

"The costs: abstraction overhead if the generated code isn't as good as hand-tuned kernels; you now own a compiler and
runtime, which is a big engineering and maintenance burden; and debugging is harder because bugs can be in the
planner, the IR, the code generator, or the hardware backend. Libraries like cuDF come tuned by NVIDIA for their own
chips.

"So the question I'd ask is how much the abstraction costs in measured terms, and where the team bypasses it with
hand-written kernels for the hottest operators. DataPelago reports beating cuDF on a set of operators, which suggests
the overhead is manageable, but I'd want to understand the details."

**Follow-ups:**
- *What does LLVM give you?* A mature optimizer and CPU/GPU backends (NVPTX, AMDGPU), so you generate IR once and
  compile per target; JIT compile time then becomes a cost to amortize (cache compiled kernels).
- *How do you handle JIT latency for short queries?* Cache compiled kernels by expression shape, precompile common
  ones, and only JIT when the data size justifies it.
- *How would you debug a wrong result?* Reproduce with the same batch on the CPU backend and the GPU backend, diff
  the outputs, then bisect through fused operators.
- *Question for Goyal:* "Where do you bypass DataVM with hand-written kernels, and what is the measured overhead?" [Ask]

### Q12. Resume deep-dive prompt.
Be ready to describe your hardest performance project: baseline, profiling tool and finding (perf, flame graphs,
Nsight, cache misses, memory bandwidth), the change, measured result, and what you'd do differently. Founders with a
hardware background probe the **measurement**, not the headline.

---

## 6. Design Exercise: Safe GPU Acceleration for Spark SQL

*Prompt you might get:* "Design a plugin that accelerates Spark SQL on a mixed CPU/GPU cluster without changing
user code, and never produces wrong results."

**Components:** plugin (driver + executor), plan rewriter, capability registry, cost model, native engine (Rust/C++),
GPU memory manager, fallback bridge, metrics/telemetry, test harness.

**Step-by-step flow:**
1. **Intercept** the physical plan after Catalyst (a Spark plugin/extension rule).
2. **Check capability** per operator and expression: supported types, functions, null/overflow semantics; mark
   unsupported nodes.
3. **Segment** the plan into maximal supported subtrees to minimize engine boundary crossings.
4. **Cost** each segment on CPU-vectorized vs GPU: scan bytes, estimated rows/selectivity, transfer cost, operator
   throughput, GPU memory need. Use statistics where available; otherwise conservative defaults.
5. **Rewrite** the plan: replace supported segments with native-engine operators; insert explicit columnar
   conversion at boundaries.
6. **Execute** per task: read batches, transfer, run fused kernels, keep intermediates on device within a segment,
   spill when over budget.
7. **Fallback**: on unsupported runtime condition (GPU OOM, data type surprise) rerun the segment on CPU; record why.
8. **Observe**: emit per-stage metrics (device time, transfer time, fallback reason) to the Spark UI/metrics sink.
9. **Verify**: shadow mode on real queries (run both, compare results) before enabling by default; feature flags per
   operator.

**Trade-offs to state:** more offload = fewer boundaries but riskier correctness; conservative cost model avoids
regressions; per-operator kill switches for safe production rollout; accept that some queries won't speed up and
report it honestly.

### Worked Example: a TPC-DS-style query, end to end

All sizes, costs and timings below are **illustrative numbers I chose to show the reasoning**, not measurements of
Nucleus or any real cluster. The query shape is modeled on TPC-DS "sales by brand" queries.

**Query:**
```sql
SELECT d.d_year, i.i_brand_id, i.i_brand, SUM(ss.ss_ext_sales_price) AS revenue
FROM store_sales ss
JOIN date_dim d ON ss.ss_sold_date_sk = d.d_date_sk
JOIN item i     ON ss.ss_item_sk = i.i_item_sk
WHERE i.i_manufact_id = 128 AND d.d_moy = 11
GROUP BY d.d_year, i.i_brand_id, i.i_brand
ORDER BY d.d_year, revenue DESC
LIMIT 100;
```

**Assumed data (hypothetical, ~1 TB scale):** `store_sales` 2.9B rows (about 250 GB Parquet, 3 columns needed),
`date_dim` 73K rows, `item` 300K rows.

**Step 1. Spark's physical plan (what the plugin intercepts):**
```
TakeOrderedAndProject(limit=100, order by d_year, revenue desc)
 +- HashAggregate(final)  keys=[d_year, brand_id, brand]  sum(ext_sales_price)
    +- Exchange hashpartitioning(d_year, brand_id, brand)            <- shuffle
       +- HashAggregate(partial)
          +- Project
             +- BroadcastHashJoin ss_item_sk = i_item_sk             <- join 2
                :- BroadcastHashJoin ss_sold_date_sk = d_date_sk     <- join 1
                :  :- Scan parquet store_sales (ss_sold_date_sk, ss_item_sk, ss_ext_sales_price)
                :  +- BroadcastExchange <- Filter(d_moy=11) <- Scan date_dim
                +- BroadcastExchange <- Filter(i_manufact_id=128) <- Scan item
```

**Step 2. Capability check (per node, per expression):**

| Node | Expressions / types | Supported? | Note |
|---|---|---|---|
| Scan store_sales | int keys, decimal(7,2) price | Yes | Column pruning keeps 3 of ~23 columns |
| Scan + Filter dims | `d_moy = 11`, `i_manufact_id = 128` | Yes | Tiny tables |
| BroadcastHashJoin x2 | inner, int keys | Yes | Build sides are small |
| Project | pass-through | Yes | |
| HashAggregate (partial/final) | `SUM(decimal(7,2))`, group by int + string | Yes | Decimal sum widens to decimal(17,2); check overflow rules |
| Exchange | shuffle | Stays on Spark | Spark's shuffle service is kept |
| TakeOrderedAndProject | sort + limit 100 | Yes (or CPU) | Only 100 rows survive, so cheap either way |

**Step 3. Segmentation:** there are two maximal supported segments, split by the Exchange.
- **Segment A (map side):** Scan -> Filter joins -> Project -> partial HashAggregate. This is where about all the
  bytes are.
- **Segment B (reduce side):** final HashAggregate -> TakeOrdered. Small input after the partial aggregation.

**Step 4. Cost each segment (illustrative):**

| Segment | CPU vectorized | GPU path | Decision |
|---|---|---|---|
| A: scan 250 GB (pruned to ~30 GB of needed columns), 2 broadcast joins, partial agg | 120 s cluster-wide | Transfer ~30 GB compressed (decode on device) + compute: 35 s | **GPU** (3.4x) |
| B: final agg over ~2 M partial rows | 1.0 s | 0.6 s compute + 0.5 s transfer and launch | **CPU** (offloading loses) |

Why B stays on the CPU: it has too little data, so launch and transfer overhead dominate (Q1). Placing it on the GPU
would also add a boundary conversion for no gain.

**Step 5. Rewritten plan:**
```
TakeOrderedAndProject / HashAggregate(final)   [Spark / CPU vectorized]
 +- Exchange (Spark shuffle, columnar->row boundary conversion here)
    +- NativeStage[GPU]: Scan -> BroadcastJoin(date) -> BroadcastJoin(item) -> Project -> PartialAgg
         (dimension tables broadcast once, kept resident in GPU memory as hash tables)
```

**Step 6. Execution inside one GPU task (per input split of ~128-256 MB):**
1. Dimension filters run on the host or device once; the filtered `date_dim` (about 30 days of rows) and `item`
   (rows for manufacturer 128) become tiny hash tables, built once per executor and reused by every task (Q2).
2. Read the three needed column chunks, with row-group pruning from Parquet statistics first (Q8).
3. Decode on device, probe both hash tables in a **fused** pass (the filters mean about 1-2% of rows survive), then
   partial aggregation in a device hash table keyed by `(d_year, brand_id, brand)`.
4. Copy only the small partial aggregate (thousands of rows) back and hand it to Spark's shuffle.

The reason this wins: it moves about 30 GB over the bus and returns kilobytes, so the high-arithmetic-intensity
work (join probe and aggregate) happens next to fast memory.

**Step 7. Failure and fallback scenarios:**

| Event | What happens |
|---|---|
| A partition has a huge skewed `ss_item_sk` | Probe hits one heavy key; the hot-key handling splits the work; if the output buffer would exceed the memory budget, that task retries on the CPU path |
| GPU out of memory during the partial agg (too many groups) | Spill or repartition; if it still fails, the task reruns on CPU with the same input split; a `fallback_reason=gpu_oom` metric is recorded |
| `ss_ext_sales_price` aggregation overflows decimal(17,2) | Behavior must follow the session's ANSI setting (error vs null); if the native kernel can't match it, the capability registry marks `SUM(decimal)` unsupported and Segment A stays on CPU for that cluster |
| The query adds a Python UDF in the Project | The UDF node stays on Spark; Segment A splits into A1 (scan + joins on GPU) and A2 (UDF and the rest on CPU), and cost is re-evaluated: offload only if A1 still wins after the extra boundary conversion |

**Step 8. What you'd show in the Spark UI / metrics (illustrative output):**
```
stage 3 (NativeStage[GPU])  tasks=2048  fallback_tasks=3 (0.15%, reason=gpu_oom)
  scan+decode: 9.2s   h2d_transfer: 6.1s   join+agg kernels: 13.4s   d2h: 0.2s   idle_wait: 4.9s
  rows_in=2.9B  rows_out=2.1M   gpu_mem_peak=61% of 80GB
stage 4 (CPU vectorized)    tasks=200   final agg + limit: 0.9s
total: 38s   baseline (CPU Spark): 131s   speedup: 3.4x   estimated cost ratio: 0.9x
```
This kind of per-stage breakdown is what lets you honestly answer "where did the time go" and see that transfer
(h2d) and idle wait are the next things to optimize, not the kernels.

**Step 9. Rollout plan for this feature:**
1. Run in **shadow mode** on sampled production queries: execute both engines, compare results (exact for ints and
   decimals, tolerance for floats), record mismatches. Zero mismatches over N thousand queries before enabling.
2. Enable per operator via flags (scan first, then joins, then aggregates) so a bug is isolated and can be turned off.
3. Canary on one cluster, alert on fallback rate, p95 latency and result-mismatch counters; automatic disable on
   threshold breach.
4. Publish a "not accelerated because..." report per query so customers see why some queries didn't speed up.

**How to use this in the interview:** walk the interviewer through steps 1-5 first, since the segmentation and the
"keep the tiny final aggregation on the CPU" decision show you reason about cost, not just offload everything. Then go deep on
whatever they probe: Step 6 for kernels, Step 7 for safety, Step 9 for operations. If asked about real Nucleus behavior,
say which parts are your design and which you'd want to compare with theirs.

---

## 7. Founder Conversation Prep

### "Tell me about yourself" (60-90 seconds)
Connect your background to **performance work on data systems** and to hardware-aware thinking. Structure: what you
build today -> one concrete performance win (numbers) -> why accelerated data processing is the next frontier -> why
DataPelago specifically.

### Why DataPelago? (draft, tailor it)
- "Compute, not I/O, is now the bottleneck for analytics and AI data prep, and your approach of accelerating existing
  engines without rewriting applications is how this reaches real users."
- "I want to work on the execution engine itself, where hardware-software co-design decides the result."
- Be concrete: mention Gluten/Substrait handoff, per-operator placement, or the cuDF comparison to show you read it.

### Respectful pushback questions (shows depth)
1. Which workloads are compute-bound enough that GPU acceleration beats the price premium end to end?
2. How do you stay ahead of NVIDIA's own RAPIDS/cuDF improvements and of CPU engines (Photon, Velox, Comet)?
3. How much overhead does the DataVM abstraction add versus hand-written kernels, and where do you bypass it?
4. How does fallback work at the granularity of operators, and how do you prevent boundary-conversion regressions?
5. What share of customer value comes from Spark vs Trino vs Python/Ray workloads?
6. Spark's shuffle and storage reads: what do you do when they dominate?

### Questions about the company and acquisition
1. Post-NetApp, what is Nucleus's primary deployment target: cloud Spark, on-prem storage-adjacent, or both?
2. How will the engineering team be organized, and what stays independent?
3. How does the team measure success: customer cost savings, benchmark leadership, or revenue?
4. What does the first six months look like for this role, and what would make me successful?
5. Which part of the engine most needs an owner right now?

### Red flags to avoid
Asserting unpublished internals; repeating vendor benchmark numbers as fact; dismissing GPUs as hype; ignoring the
acquisition; talking only about code, not measurement or customers.

---

## 8. Behavioral Stories to Prepare

Prepare 4-5 stories with numbers, each two or three layers deep (founders probe "how did you decide"):
1. **Performance win**: a bottleneck you found by profiling, the fix, and the measured improvement.
2. **Correctness incident**: a bug in data processing (wrong results) that you caught or fixed; how you added tests.
3. **Cross-boundary work**: you worked across layers (query planner <-> storage <-> hardware/kernel) or with
   customers' constraints.
4. **Ambiguity / startup pace**: a time you shipped with incomplete requirements or through organizational change
   (relevant to an acquisition period).
5. **Disagreement**: you challenged a design with data and either changed minds or accepted the decision.

---

## 9. 30-60-90 Pitch and Cheat Sheet

**30 days:** run the engine locally; reproduce a TPC-H/DS subset against Spark baseline; read the plan interception,
operator capability registry and fallback paths; fix a small conformance bug.
**60 days:** own one operator or expression family end to end (implementation, semantics tests, benchmark);
improve a profiling workflow that shows device vs transfer vs CPU time.
**90 days:** ship an optimization with a measured end-to-end improvement and a rollout plan (flag, shadow mode,
telemetry); propose the next one backed by data.

### Cheat sheet
| Term | One-liner |
|---|---|
| Nucleus | DataPelago's universal data processing engine (Rust, vectorized, columnar) |
| DPA-S | Spark plugin that runs on Nucleus |
| DataApp | Layer converting host plans (via Gluten/Substrait) |
| Intelligent Execution Optimizer | Chooses hardware per operator using cost/performance |
| DataVM | VM with data-specific ISA over CPU/GPU (LLVM/CUDA/ROCm) |
| DataOS | Runtime placement/scheduling layer (details [Ask]) |
| Gluten | Open-source bridge from Spark to native engines |
| Substrait | Cross-engine plan IR |
| Velox | Open-source vectorized execution library |
| cuDF / RAPIDS | NVIDIA GPU dataframe/SQL libraries |
| Zero-copy activation | NetApp's phrase for processing data in place at storage |

---

## References

Sources used for compiling this guide:
- [DataPelago home](https://www.datapelago.ai/)
- [Why I started DataPelago: Rajan Goyal, Founder and CEO](https://www.datapelago.ai/resources/why-i-founded-datapelago)
- [DataPelago architecture docs](https://datapelago-docs.readthedocs-hosted.com/en/latest/architecture.html)
- [Run Apache Spark up to 10x faster with DataPelago on Amazon EKS (AWS)](https://aws.amazon.com/blogs/apn/run-apache-spark-up-to-10x-faster-with-datapelago-on-amazon-eks/)
- [DataPelago Nucleus vs NVIDIA cuDF](https://www.datapelago.ai/resources/DataPelago-Nucleus-Vs-Nvidia-cuDF)
- [DataPelago launch press release (Oct 2024)](https://www.datapelago.io/press-release/2024-10-01-launch)
- [NetApp buys DataPelago: Blocks & Files](https://www.blocksandfiles.com/ai-ml/2026/07/17/netapp-buys-datapelago-to-become-full-stack-ai-data-infrastructure-provider/5274250)
- [NetApp acquires DataPelago: Techzine](https://www.techzine.eu/news/infrastructure/142971/netapp-acquires-datapelago-for-gpu-accelerated-data-processing/)
- [NetApp blog: GPU-powered analytics with DataPelago](https://www.netapp.com/blog/datapelago-gpu-accelerated-data-processing/)

Note: I could not read the cuDF comparison page directly (it redirected); its numbers come from the AWS partner post.
Performance numbers are vendor-reported. Sections tagged [General] reflect standard engineering practice, not
confirmed DataPelago internals. Confirm your role's actual scope and the post-acquisition plans with the recruiter.
