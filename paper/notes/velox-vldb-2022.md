# Velox: Meta's Unified Execution Engine

> VLDB 2022 — structured reading notes

Full paper content: [Markdown conversion](../original/velox-vldb-2022.md)

## Paper information

- **Authors:** Pedro Pedreira, Orri Erling, Masha Basmanova, Kevin Wilfong, Laith Sakka, Krishna
  Pai, Wei He, Biswapesh Chattopadhyay
- **Affiliation:** Meta Platforms Inc.
- **Venue:** Proceedings of the VLDB Endowment, Volume 15, Number 12, 2022, pages 3372–3384
- **DOI:** [10.14778/3554821.3554829](https://doi.org/10.14778/3554821.3554829)
- **Code:** https://github.com/facebookincubator/velox

## One-sentence summary

Velox is a C++ library of reusable, dialect-agnostic **data-plane** components — type system,
Arrow-extended vectors, vectorized expression evaluation, operators, I/O, serializers, and
resource management — that Meta has integrated into **more than a dozen systems** (Presto, Spark,
PyTorch, stream processing, message bus, ingestion, feature engineering) to replace duplicated
execution engines, delivering **~6–7× average speedup on real Presto workloads and the same
throughput on 3× fewer servers**.

## Problem: a siloed data landscape

Workload diversity plus dataset growth produced a proliferation of specialized engines — batch and
interactive analytics, ETL and bulk movement, stream processing, log and timeseries monitoring,
and now AI/ML preprocessing and feature engineering. The result:

- Engines built with **different frameworks, languages, and teams**, sharing nothing.
- **Evolving them per-engine is cost prohibitive** — extending each one for new hardware
  (cache-coherent accelerators, NVRAM), new features (tensor types for ML), or new research results
  is impractical, and **invariably yields engines with disparate optimizations**.
- **Users pay the price**, since finishing one task usually means interacting with several engines
  whose types, functions, aggregates, null handling, and casting behave differently.

The paper's memorable data point: **an informal survey at Meta found at least 12 different
implementations of the simple string function `substr()`** — differing in parameter semantics
(0- vs. 1-based indices), null handling, and exception behavior. A second survey found **about 14
libraries used for ML data preprocessing**.

**The key observation licensing the whole project:** specialized engines genuinely differ in the
**language frontend** (SQL, dataframes, DSLs), the **optimizer**, the **runtime** (how tasks are
distributed), and the **I/O layer** — but **the execution engines at their core are all rather
similar.** Every engine needs a type system, an in-memory (usually columnar) representation, an
expression evaluator, operators (join, aggregation, sort), plus storage and network serialization,
encoding formats, and resource management.

**Velox's three-part value proposition:**

- **Efficiency** — democratize optimizations previously implemented in individual engines: full
  SIMD, lazy evaluation, adaptive predicate reordering and pushdown, common subexpression
  elimination, execution over encoded data, code generation.
- **Consistency** — engines sharing the library expose the exact same data types and
  scalar/aggregate function packages.
- **Engineering efficiency** — every feature and optimization is built and maintained **once**.

**Scope boundary, stated firmly:** Velox takes a **fully optimized query plan** as input and
executes it with local-node resources. It provides **no SQL parser, no dataframe layer, no DSL, and
no global query optimizer**, and is **not meant to be used directly by data users**. In other
words, **Velox is the data plane; individual engines own the control plane.**

## Library components

| Component | What it provides |
| --- | --- |
| **Type** | Generic type system for scalar, complex, and nested types — structs, maps, arrays, tensors — plus an **opaque type** wrapping arbitrary C++ structures. **Extensible** so engines add their own types (Presto's HyperLogLog, `timestamp with timezone`) without modifying the library |
| **Vector** | **Arrow-compatible** columnar memory layout supporting Flat, Dictionary, Constant, Sequence/RLE, and Bias (frame-of-reference) encodings, plus **lazy materialization** and **out-of-order result buffer population** |
| **Expression Eval** | Fully vectorized evaluation over Vector-encoded data, using common subexpression elimination, constant folding, efficient null propagation, **encoding-aware evaluation**, and **dictionary memoization** |
| **Functions** | APIs for custom scalar functions in both **simple (row-by-row)** and **vectorized (batch-by-batch)** style, plus aggregate function APIs. Dialect-compatible packages ship for **Presto and Spark** |
| **Operators** | TableScan, Project, Filter, Aggregation, Exchange/Merge, OrderBy, HashJoin, MergeJoin, Unnest, and more |
| **I/O** | Generic connector interface for pluggable file format codecs and storage adapters; **ORC, Parquet, S3, HDFS** included |
| **Serializers** | Wire protocol interface; **PrestoPage** and **Spark UnsafeRow** supported |
| **Resource Management** | Memory arenas and buffer management, tasks, drivers, CPU thread pools, **spilling, and caching** |

Engines pick only what they need: a system with simple data representation might use Type, Vector,
and Serializer alone, while a full SQL engine uses everything.

**The inclusion rule for plugins is genericity:** if used by multiple engines (Parquet/ORC codecs,
Aggregate, OrderBy, HashJoin) it lives in the main library; otherwise it lives in the client
engine's codebase (ML-specific functions, stream processing operators).

## Use cases at Meta

### Presto → Prestissimo

Presto has a **coordinator** (query receipt, SQL parsing, metadata, global optimization, resource
management) and **workers** (executing plan fragments), both Java, communicating over HTTP REST.
Because all data processing and shuffling happens in or between workers, there is typically a
**100–1000 : 1 ratio of workers to coordinators**, so **the vast majority of CPU time is on
workers**.

**Prestissimo** ("from music theory: the fastest possible tempo; faster than Presto") replaces Java
workers with a C++ process built on Velox — implementing Presto's HTTP REST interface including
worker-to-worker exchange protocol, coordinator orchestration, and status endpoints, so it is a
**drop-in replacement**. The flow: receive a Presto plan fragment from the Java coordinator,
translate to a Velox plan, execute. **No Java, no JVM, and no garbage collection on worker nodes —
which "used to be a source of operational issues."**

Prestissimo uses the entire library, and because it was the first end-to-end implementation, **many
components built for it — the Presto wire protocol, the Presto function and aggregate packages —
became core Velox and are now reused by the realtime infrastructure, stream processing, and ML
platforms.**

### Spark → Spruce

Spark is used at Meta for **batch and ETL SQL** because of its fault-tolerance for long-running
applications. **Spruce** offloads execution by reusing Spark's existing **script transform**
interface (which runs arbitrary binaries): the executor serializes a plan fragment and forwards it
to an external **SparkCpp** process, which deserializes it, converts it to a Velox plan, and
executes.

SparkCpp uses Velox's extensibility APIs to add operators and functions **making the C++ code fully
compatible with the existing Scala engine**, and adds an **UnsafeRow serializer** (Spark's shuffle
and client-return format). Note the honest observation: **Velox is customized differently for
Presto and Spark to preserve backwards compatibility**, but a shared engine **paves the way to
semantic equivalence later** while delivering efficiency now.

### Realtime data infrastructure

- **XStream (stream processing).** Applications read continuously from Scribe, apply business
  logic, and write to Scribe or other sinks. Although the abstraction is one row at a time,
  **reads and writes are batched for I/O** — production batches are **up to 500 kB buffered over at
  most 20 seconds** — so they benefit from vectorized execution. Most operations map directly to
  Velox operators (projections, filters, and lookup joins in progress), and XStream **exposes the
  same function package as Presto**. **Temporal-window aggregations (tumbling, hopping, session)
  are implemented as an XStream extension**, with plans to move them into core Velox and expose
  them to Presto and Spark users — "only possible without substantial duplication of effort due to
  the unified execution engine."
- **Scribe (message bus).** Data is written row-by-row and was traditionally read the same way. The
  Scribe Read Service now uses Velox's wire serialization formats — **more efficient due to
  column-oriented encoding**, and directly deserializable to Vectors. Critically, consumers can
  **push projections and filters down close to storage**, reducing data read and **in many cases
  reducing cross-datacenter traffic** — with **the same semantics as in other engines**.
- **FBETL (ingestion).** For **warehouse ingestion** (Scribe → ORC/DWRF files), Velox lets users
  specify **transformations, UDFs, and filtering at ingestion time**, freeing them from building a
  whole stream processing application — which would cost writing to a new Scribe pipe and reading it
  again. Users can **reuse any Presto function**. For **database ingestion** (scraping operational
  DB logs into warehouse snapshots), Velox aids snapshotting: read the previous snapshot, apply
  modifications from redo logs (**an operation similar to a merge-join**), write a new partition.

### Machine learning

Data preprocessing sits between analytics (large joins, aggregations, filters) and neural networks
(tensor operations), and is usually **row-wise transformations** — normalization, embedding lookups,
image cropping — expressible with expression evaluation and UDFs. Despite the similarity, **Data
Analytics and ML infrastructure evolved independently at Meta**, producing ~14 preprocessing
libraries with incomplete type support, incompatible memory representations, and inconsistent
functions. And this is not a small tail: **preprocessing can consume up to 50% of the resources
used for ML workloads.**

- **TorchArrow** provides a Pandas-like Python dataframe layer inside PyTorch that **translates to
  a Velox plan and delegates execution** — consolidating execution engine code between analytics
  and ML ("DI for AI") and giving ML users consistent behavior across the several engines they
  already touch.
- **F3 (feature engineering)** lets users define features in a DSL (e.g. birthdate → age), from
  which F3 generates **offline data via Spark, realtime data via XStream, and online serving values
  during inference** — using one definition for consistency between training and serving. Since
  Spark and XStream already use Velox, offline and realtime pipelines run natively. **The online
  serving path is the interesting challenge:** very high QPS, low latency, **often a single record
  per call**, where **Velox's vectorized engine is not the optimal fit due to interpretation
  overhead** — motivating the codegen work, since the DAG itself is mostly static.

## Deep dive

### Type system

Primitive types (integers and floats of various precision, varchar and varbinary strings, dates,
timestamps, **functions/lambdas**) plus complex types (**arrays, fixed-size arrays used to
implement ML tensors, maps, rows/structs**), arbitrarily nestable with serialization methods, plus
an **opaque type** for wrapping arbitrary C++ structures. Extensible per engine.

### Vectors

The base layout extends Apache Arrow: a **size**, a **type**, and an **optional nullability
bitmap**, plus methods to copy, resize, hash, compare, and print.

- Vectors hold fixed-size or variable-size elements, nest arbitrarily, and carry an encoding — but
  **the component generating a Vector chooses the encoding**.
- Data lives in **Buffers** — contiguous memory from a memory pool, subclassable for different
  ownership modes. **Vectors and Buffers are reference counted, one Buffer may back several
  Vectors, only singly-referenced data is mutable, and anything can be made writable via
  copy-on-write.**
- **Lazy Vectors** populate only on first use. This matters for cardinality-reducing operations
  (joins, conditionals in projections): depending on selectivity you can **avoid materialization
  entirely or scope it to surviving rows**, and when reading from remote storage it can **optimize
  away entire I/O operations for sparsely accessed columns**. They also support **running a callback
  over loaded data**, allowing computation pushdown (e.g. aggregations) with no intermediate Vector.
- **Decoded Vectors** solve the developer-burden problem: a function or operator generally cannot
  control how its input was encoded. Encoding awareness is an optimization opportunity (evaluate
  only over distinct values of a dictionary) but a cognitive burden. A Decoded Vector transforms an
  arbitrarily encoded Vector into **a flat vector plus indices**, exposing a consistent API —
  **zero-copy for flat, constant, and single-level dictionary inputs** (the common cases), only
  materializing new indices for **nested dictionaries/RLE**.

#### Three deliberate divergences from Apache Arrow

**1. Strings — StringView instead of offsets/lengths.**

```cpp
struct StringView {
    uint32_t size_;
    char prefix_[4];
    union {
        char inlined[8];
        const char* data;
    } value_;
}
```

String vectors have a **metadata buffer of 16 bytes per element** plus a data buffer. Three
payoffs: **a 4-byte prefix is always inline, short-circuiting failed comparisons** to speed up
filtering and ordering; **strings up to 12 bytes are fully inlined**, needing no secondary buffer
access; and operations like `trim()` and `substr()` become **zero-copy, updating only metadata
pointers**.

**2. Out-of-order write support**, to execute conditionals efficiently. For IF/SWITCH, the
condition is evaluated first to produce a **bitmask of which branch each row takes**, then each
branch is processed in a vectorized manner writing into the single output vector. Primitive types
can always be written out of order (constant element size), and **strings can too because
StringView metadata is a constant 16 bytes**. For remaining variable-size types (arrays, maps),
Velox **maintains both lengths and offsets buffers**. Beyond conditionals, this gives the engine
**flexibility to slice and rearrange elements without copying**, since each array's length and
offset update independently — and even **permits arrays/maps with overlapping elements**.

**3. More encodings** — **RLE** and **constant** (all values identical, used for literals and
partition keys), both common in warehouse workloads.

A **conversion API** preserves Arrow interoperability, zero-copy where possible. These extensions
were proposed back to the Arrow community, which was receptive, though incorporation was still
under discussion at publication.

### Expression evaluation

Used in three places: **the FilterProject operator**, **TableScan/IO connectors for consistent
predicate pushdown**, and **standalone** for engines needing only expressions (realtime
infrastructure, most ML preprocessing).

Input is an **expression tree** whose nodes are: a column reference, a constant, a function call
(also used for **AND/OR conjunctions, IF/SWITCH conditionals, and try expressions**), a CAST, or a
lambda. Nodes carry metadata on **determinism** and **null propagation** — the two properties that
gate most optimizations.

**Compilation** produces an executable expression, applying:

- **Common subexpression elimination.** In
  `strpos(upper(a),'FOO') > 0 OR strpos(upper(a),'BAR') > 0`, `upper(a)` is computed once.
  **FilterProject builds a single compiled expression for all filter and project expressions**, so
  subexpressions are shared between them.
- **Constant folding** — `upper(a) = upper('Foo')` becomes `upper(a) = 'FOO'`.
- **Adaptive conjunct reordering.** The engine **tracks the runtime performance of individual
  conjuncts** and evaluates the most effective first, scoring by **`time / (1 + n_in − n_out)`** —
  lowest score wins, i.e. drop the most values in the least time. To maximize the effect,
  compilation **flattens adjacent AND/OR nodes**: `AND(AND(AND(a,b),c),AND(d,e))` becomes
  `AND(a,b,c,d,e)`.

**Evaluation** is a recursive descent passing down a **row mask** of active (non-null, not masked
out) elements. Work is skipped when the node is an already-computed common subexpression, or when
**the expression propagates nulls and any input is null** — implemented by **combining input
nullability bitmasks with SIMD** and updating the active-rows mask.

- **Peeling.** For dictionary-encoded inputs, deterministic expressions need only consider distinct
  values: verify all inputs share the same dictionary wrapping, **peel it off, evaluate on the inner
  vectors, and re-wrap the results with the original indices**. Example: a 1000-row `color` column
  dictionary-encoded over `[red, green, blue]` — `upper(color)` runs on **3 values, not 1000**.
- **Memoization.** Across batches read from TableScan, it is common for many batches to be
  dictionary-encoded over **the same base vector with different indices**. The engine **remembers
  the results computed over the inner vector and re-wraps them for subsequent batches**.

The authors are candid about where this pays: these techniques "might not present considerable
improvements for simple arithmetic operations over primitive types, but they do provide substantial
speed up for complex expressions such as string operations, regular expressions, array/map
manipulation, and other operations over nested data types" — **a conscious decision driven by
empirical data showing those operations are the top CPU consumers**, while keeping fast paths for
base cases.

**Code generation (experimental).** The whole expression tree is rewritten as C++ source, compiled
with gcc/clang into a shared library, dynamically linked, and used instead of the interpreted path.
**Compilation takes up to 10 seconds**, so it is unsuitable for short or interactive queries —
targeting instead **large ETL queries running hours to days** and **fixed expression trees** like
the F3 feature engineering case. Open questions the authors name: when codegen's benefit outweighs
compilation delay, **decreased developer productivity, and debuggability**; codegen versus
JIT/LLVM; and runtime adaptivity between the two paths.

### Functions

**Vectorized scalar functions** receive Vectors, nullability buffers, and an active-rows bitmap.
Some become **constant time** by exploiting the columnar layout: `is_null()` returns the internal
nullability buffer; `cardinality()` uses the internal lengths buffer; `map_keys()`/`map_values()`
return the MapVector's internal buffers.

But for everything else, requiring developers to iterate rows manually, handle nullability, cope
with arbitrary input and output encodings and nested types, and manage output buffers **"turned out
to be too cumbersome (and error-prone)"** — particularly as the number of contributors grows and
because **scalar functions quickly became the largest portion of Velox's codebase** (advanced string
and JSON processing, date/time conversion, array/map/struct manipulation, regular expressions,
mathematical functions for data science).

**The simple function API** hides the engine and data layout while keeping vectorized performance:

```cpp
class MultiplyFunction {
    void call(int64_t& result, const int64_t& a, const int64_t& b) {
        result = a * b;
    }
};
registerFunction<MultiplyFunction, int64_t, int64_t, int64_t>({"multiply"});
```

- The first parameter is the **return value by reference**; inputs are const references. Returning
  `bool` denotes nullability (true = not null); returning `void` signals the function never produces
  nulls.
- **Default null behavior** is assumed — any null input yields null output **without calling the
  function**. Functions needing otherwise provide `callNullable()` taking pointers instead.
- The framework uses **DecodedVectors** to hide encoding and **C++ template metaprogramming to
  apply the method across batches without per-row dispatch cost**, with compiler hints ensuring the
  loop body is **inlined** — so much so that **clang and gcc automatically generate SIMD for
  arithmetic functions from the definition above**.
- Non-primitive types use **proxy objects** (e.g. `ArrayReader`/`ArrayWriter` mirroring
  `std::vector`) that **operate directly on Vector data with no extra allocation or copies**,
  avoiding materialization into `std::string`/`std::vector`.

**The surprising measured result:** comparing three functions written both ways, `plus()` on
primitives showed the simple API costs **nothing** despite the productivity gain — but for the
complex-type functions the **simple implementation was actually *faster***. The cause was **missed
optimization opportunities in the hand-written vectorized versions** — flat-encoding and null-free
fast paths that the simple framework applies automatically. The gap is fixable by hand, but **the
framework takes the burden off developers**.

Functions declare **determinism** and **null behavior** to enable or disable evaluation
optimizations. **The vast majority qualify for the full set**, with few exceptions such as `rand()`
and `shuffle()`.

**Advanced string processing.** Most string functions must handle UTF-8, imposing unnecessary
overhead on ASCII-only input — and **the vast majority of strings in Meta's warehouse tables are
ASCII**. So a function may provide a specialized **`callAscii()`** automatically invoked for
ASCII-only inputs, and may declare its **ASCII behavior** — whether ASCII-only inputs guarantee
ASCII-only outputs — letting the engine **skip ASCII detection on data it produced**. Separately,
functions like `substr()`, `trim()`, and tokenizers can produce **zero-copy results referencing the
input string buffers** by setting a flag, and the same applies to functions producing arrays of
strings such as `split()`. Micro-benchmarks compare `substr()` with no optimizations, ASCII-only,
and ASCII-only plus buffer reuse.

**Aggregate functions** are computed in up to four steps: **partial** (raw input → intermediate),
**final** (intermediate → result), **single** (data already partitioned on grouping keys, so no
shuffle or intermediates), and **intermediate** (combining partials, e.g. from parallel threads, to
reduce data sent to the final stage).

Accumulators are **fixed-size** (`count`, `sum`, `avg`, `min`, `max`) or **variable-size**
(`distinct`, `pct`, and approximate counterparts). Since aggregation stores one row per group,
**fixed-size accumulators are stored inline in the row; variable-size ones live in a separate
buffer with a pointer in the row.**

### Operators and the execution model

Plan nodes convert to Operators nearly one-to-one, with exceptions: **Filter followed by Project
merges into one FilterProject operator**, and nodes with multiple children become multiple
operators (**HashJoin → HashProbe + HashBuild**).

```text
Task  = unit of function shipping in distributed execution
        = a query plan fragment + its Operator tree
        starts at a TableScan or Exchange, ends in an Exchange
  └── Pipelines = linear sub-trees of the Operator tree
                  (HashProbe and HashBuild are one Pipeline each)
        └── Drivers = threads of execution, each with its own state
                      may be on-thread or off-thread
```

A Driver goes **off-thread** when its consumer has not consumed data, its source exchange has not
produced data, or a scan is waiting on files. The paper's justification is architectural: **this
model is more convenient for going on and off thread than the traditional Volcano iterator tree,
because state is resumable without constructing control flow on the stack.** Tasks can also be
**cancelled or paused by other Velox actors at any time** — useful for enforcing priorities,
checkpointing, **forcing another Task to spill**, or other coordination.

All operators share a base API: add a batch of vectors as input, get a batch as output, check
readiness for more input, and **signal no-more-data** — the last used to tell a blocking sort or
aggregation to flush and start producing.

**Table scans, filter, project.** Scans are **column-by-column with filter pushdown**: columns
carrying filters are processed first, producing hit row numbers plus optionally the value. **Filters
are adaptively ordered at runtime by the same score as conjunct reordering:
`time / (1 + values_in − values_out)`.**

- Simple filters evaluate **multiple values at a time with SIMD**, processing **roughly one integer
  hit per CPU clock using AVX2**.
- **Filter results for dictionary-encoded data are cached**, and SIMD checks cache hits via
  **gather + compare + mask lookup + permute** to write out passing rows — **more than one hit per
  CPU clock on average**.
- An efficient **large IN filter** implementation (used for hash join pushdown) **triggers 4 cache
  misses at a time**.
- FilterProject uses **one expression evaluation context for filters and projections**, evaluates
  the filter on all rows first, runs projections **only on survivors**, and **skips projections
  entirely if nothing passed**.

**Aggregations and hash joins** share **one carefully designed hash table**, which both promotes
reuse and **unifies the adaptivity across both cases**. Keys are processed columnar through a
**VectorHasher**, which recognizes key ranges and cardinality and, where applicable, **translates
keys into a smaller integer domain**:

- if all keys map to a handful of integers → **direct mapping to a flat array**;
- multiple keys → **combined into a single 64-bit normalized key** if possible, then used to index a
  flat array or as a single hash key depending on range;
- **an inefficient multipart hash key is used only when none of the above applies**.

The layout is **decided adaptively and may change as new batches arrive**. And because VectorHashers
are effectively **a digest of distinct values per key, they can be pushed down to TableScans as
efficient IN filters** when scan and join are colocated.

The table layout resembles Meta's **F14**: **memory accesses between lookups of different keys are
interleaved** to maximize in-flight cache misses and shorten pipeline stalls from data dependency,
and **values are stored row-wise** because joins and aggregations typically access all dependent
data together.

### Memory management, spilling, and caching

**Memory pools** track Task memory. Small objects (query plans, expression trees, control
structures) come from the C++ heap, but **large objects — data cache entries, hash tables, assorted
buffers — use a custom allocator offering zero fragmentation via `mmap`/`madvise`**. All pool
allocations are **tracked hierarchically and subject to limit enforcement**, and consumers can
**reserve memory to guarantee budget for a specific operation** such as processing a batch of
group-by keys.

**Recovery on allocation failure:** consumers may be **asynchronously paused** — acknowledging by
going off-thread and **returning a continuation future** so execution can resume later. While
paused a task may be **instructed to spill** or **cancelled to make room**, per prioritization
policy.

The default action on exceeding a limit is to invoke a **process-wide memory arbiter** with
visibility into all running tasks, their usage, and their **reclaimable memory** (how much they
could release by spilling). **The policy deciding who spills or dies is pluggable**, so engines
implement their own behavior.

Spilling requires operators to implement an interface reporting **how much could be released** and
**how to spill**. Without it, an operator facing a failed allocation can only **continue without the
allocation or fail**. Operators may also **monitor overall memory pressure and react** — e.g.
Exchange reducing its buffer size when memory is scarce.

**Caching** (for disaggregated storage) exists at **memory and SSD** levels. Memory cache is a
special consumer allowed to use **all memory not otherwise allocated**; **all I/O buffers are
allocated directly from it with arbitrary sizes** matching the columnar layout — **unlike operating
system caches allocated in fixed-size pages** — mixing sizes without fragmentation via
`mmap`/`madvise`.

- Cached columns are read from S3/HDFS, held in RAM on first use, and **eventually persisted to
  local SSD**.
- **Nearby column reads are coalesced** when the gap is small enough — **about 20 KB for SSD and
  500 KB for disaggregated storage** — serving neighboring reads in as few I/O operations as
  possible. This exploits temporal locality so **correlated columns end up cached together on SSD**.
- Since all remote columnar formats share an access pattern (read file metadata for buffer
  boundaries, then read parts of those buffers), **reads can be prefetched to interleave I/O stalls
  with CPU work**. Velox **tracks per-query column access frequencies and adaptively prefetches hot
  columns**.
- The combined effect: **many interactive analytical workloads over small-to-mid tables are served
  effectively from memory, taking I/O stalls off the critical path** so they do not contribute to
  query latency.

**Measured storage hierarchy throughput** (read latency plus decoding and decompression; simple
filter or aggregation on scalar columns; 26-core server, 64 GB RAM, 2×2 TB SSD):

| | RAM | SSD | Disaggregated |
| --- | ---: | ---: | ---: |
| **Read rate** | **8 GB/s** | 2–3 GB/s | 700 MB/s |

That is **~3× RAM over local SSD and ~4× local SSD over remote storage.**

## Evaluation

### TPC-H: Prestissimo vs. Presto Java

Cluster of **80 nodes, 64 GB RAM, 2×2 TB SSD**, both systems with local caching **from a warm
cache**. Dataset is **3 TB TPC-H in ORC with no zstd compression**, with `lineitem` and `orders`
co-partitioned. **Query formulations are hand-written** to give the right join tree shape with
selective joins on the build side; joins are hash joins.

| Query | Wall C++ | Wall Java | Wall speedup | CPU C++ | CPU Java | CPU speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **Q1** (CPU-bound) | 5 s | 42 s | **8.4×** | 2211 s | 14435 s | 6.5× |
| **Q6** (CPU-bound) | 1 s | 9 s | **9×** | 538 s | 2018 s | 3.7× |
| **Q13** (shuffle/IO) | 15 s | 31 s | 2× | 5647 s | 12322 s | 2.1× |
| **Q19** (shuffle/IO) | 6 s | 13 s | 2.1× | 1362 s | 3483 s | 2.5× |

**The bottleneck moved.** For CPU-bound Q1 and Q6, Prestissimo approaches an order of magnitude and
is **now bottlenecked on the coordinator's speed to dispatch work**. For shuffling Q13 and Q19,
**the new bottleneck is shuffle latency**. Named remedies: better coordinator metadata handling,
better shuffle timing and message sizes, and **lightweight encoding to cut shuffle volume**.

### Real production workloads

Acknowledging that **"TPC-H does not provide a comprehensive representation of modern workloads,"**
the authors replayed **production traffic from a variety of interactive analytical tools** against
two identically specified clusters, one Prestissimo and one Presto Java. The result distribution
shows an **average speedup of about 6–7×, with many queries exceeding an order of magnitude**.

### Capacity impact

The question that matters at hyperscale is servers, hence datacenter power. Two clusters shadowed
the same production workloads while the Prestissimo cluster was progressively shrunk: it sustained
the workload with **equal or better user-perceived performance using 3× fewer servers (20 vs. 60)**.

## Positioning against related work

- **DuckDB** shares many design decisions (embeddable C++ library, vectorized engine) but **focuses
  on a full-stack RDBMS with SQL as its main user-facing API**, deeply integrated with Python and R.
  **Velox instead provides modular, language-agnostic building blocks** for integration into
  existing large-scale engines including stream processing, realtime infrastructure, and ML.
- **Arrow Compute** offers scalar, vectorized, and aggregate kernels over Arrow data with an API
  similar in principle to Velox's function APIs, but has **considerably narrower scope — no SQL
  operators, no resource management**. **Gandiva** adds an LLVM-based execution environment for
  kernels; despite the interpreted-vs-JIT difference, both are **restricted to function/kernel
  execution**.
- **Photon** (Databricks) is a C++ vectorized engine transparently integrated into Spark — the
  runtime takes a second pass over the optimized plan to decide what Photon can run, loading the
  library into the JVM and using JNI with off-heap pointers. It shares design decisions with Velox
  but is **Spark-only and proprietary**.
- **Optimized Analytics Package (OAP) / Gazelle** (Intel) similarly targets Spark with
  SIMD-optimized kernels and an LLVM expression engine over Arrow via JNI — again **Spark-focused,
  not engine-agnostic**.

## Future directions the paper argues for

Two trends the authors see as disruptive as cloud/storage disaggregation was: **(a) the rise of AI
as the principal consumer of data management**, and **(b) componentization and specialization of
compute — GPUs, FPGAs, tensor accelerators, and cache-coherent interconnects like CXL**.

The vision: instead of monolithic engines with their own frontend, execution, and storage,
**specialized processing kernels plug into a "data management function bus" such as Velox**, which
executes plans from multiple frontends and fully exploits the hardware. The trend is already
visible in **Apache Arrow** for memory format and **Substrait** for interoperable plan
representation.

Also named: continued convergence of AI and data management stacks; integration with monitoring,
observability, graph, and **operational workloads** — the latter still posing **substantial open
challenges around vectorization, small batch sizes, and low latency**; and **less reliance on an
omniscient query optimizer in favor of local intelligence and adaptivity at every level**, with
systems that are **autonomous, auto-configurable, and self-driven**.

## Limitations and questions

- **Velox is only the data plane.** Every integration must still supply parsing, optimization,
  distribution, and fault tolerance — the paper documents the wins but not the integration cost
  (Prestissimo had to reimplement Presto's entire REST and exchange protocol in C++).
- **Consistency is a goal, not yet an achievement.** Velox is **deliberately customized differently
  for Presto and Spark** to preserve backwards compatibility; semantic equivalence is described as
  future work that a shared engine merely "paves the way" for.
- **Vectorization is a poor fit for single-row, high-QPS serving** — explicitly acknowledged in the
  F3 online path — and the proposed answer, codegen, is **experimental with up to 10-second
  compilation times and unresolved productivity and debuggability costs**.
- **Benchmarks are vendor-authored and favorably configured:** TPC-H queries are **hand-written for
  the right join tree shape**, co-partitioned, warm-cache, uncompressed. The production replay is
  more convincing but is a distribution with no absolute numbers or workload characterization.
- **The comparison baseline is Java Presto**, so a large share of the speedup is attributable to
  removing the JVM rather than to Velox's specific techniques — the paper does not separate these.
- **Arrow divergence has a cost.** StringView, dual lengths+offsets buffers, and extra encodings
  mean **format conversion is only zero-copy "when possible"**, and adoption upstream was
  unresolved.
- **Spilling and memory recovery depend on operator cooperation** — operators that do not implement
  the interface can only proceed without the allocation or fail.
- **No fault tolerance, transactions, or durability** — those remain the embedding engine's problem,
  which is why Spark is still used at Meta for long-running jobs.

## Practical design checklist

Velox fits when:

- you are **building or maintaining an execution engine**, not using one;
- multiple systems in your organization **duplicate type systems, function libraries, and
  operators** with inconsistent semantics;
- workloads are **batch-shaped enough to vectorize** (even stream processing qualifies once reads
  and writes are batched);
- data lives in **disaggregated storage** where caching and prefetching pay;
- **complex types — strings, arrays, maps, structs, tensors — dominate CPU time**, which is where
  Velox's optimizations concentrate.

Look elsewhere when:

- you need a complete database (parser, optimizer, storage, transactions) — DuckDB is the closer
  fit;
- your path is **single-record, latency-critical serving**, where interpretation overhead dominates;
- you cannot afford the integration work of adopting a data plane into an existing engine.

## Takeaways

1. **The parts engines differ in are not the parts they spend CPU on.** Frontend, optimizer,
   runtime, and I/O genuinely differ; type systems, vectors, expressions, and operators do not — and
   that asymmetry is the entire thesis.
2. **Twelve `substr()` implementations is a correctness problem, not just a cost problem.** Sharing
   an execution library is the only mechanism that makes null handling and indexing semantics agree
   across engines.
3. **Encoding-aware evaluation is where the wins are.** Peeling dictionaries to evaluate over
   distinct values, memoizing across batches sharing a base vector, and caching filter results turn
   1000-row work into 3-row work.
4. **Adaptivity replaces the optimizer at the data plane.** Conjunct reordering, filter reordering,
   adaptive hash layouts, and adaptive prefetching all use the same idea: measure at runtime, since
   the plan-time optimizer already did its job and left.
5. **A good API is a performance feature.** The simple function framework was *faster* than
   hand-written vectorized code because it applies the fast paths developers forget — the productivity
   argument and the performance argument turned out to be the same argument.
6. **Extend the standard where the standard costs you.** StringView's inline prefix and inline short
   strings, and dual lengths+offsets buffers for out-of-order writes, are targeted deviations from
   Arrow with a stated purpose and a conversion path back.
7. **Drivers that can go off-thread beat a Volcano stack.** Resumable state without stack-based
   control flow is what makes pausing, spilling, and priority enforcement possible at all.
8. **Report the number that matters.** "6–7× on real traffic" and "3× fewer servers" say more than
   any TPC-H table — and both bottleneck disclosures (coordinator dispatch, shuffle latency) are more
   informative than the speedups themselves.

## Citation

```bibtex
@article{pedreira2022velox,
  author = {Pedro Pedreira and Orri Erling and Masha Basmanova and Kevin Wilfong and
            Laith Sakka and Krishna Pai and Wei He and Biswapesh Chattopadhyay},
  title = {Velox: Meta's Unified Execution Engine},
  journal = {Proceedings of the VLDB Endowment},
  volume = {15},
  number = {12},
  pages = {3372--3384},
  year = {2022},
  doi = {10.14778/3554821.3554829}
}
```
