# DataPelago Interview Guide: Gluten Shuffle and Memory, From the Source

Companion to [datapelago_interview_guide.md](datapelago_interview_guide.md). That guide covers the company and
Nucleus. This one covers one narrow, high-signal area: **how Apache Gluten does shuffle and memory management, and
what broke and got fixed recently.**

**Why this is worth studying.** The company guide records that Nucleus hands plans off through Gluten/Substrait
[Public, per that guide]. The Spark-facing half of Gluten (shuffle manager, memory accounting, spill coordination) is
the part any Gluten-based accelerator has to live with, and the native half (Velox) is the part an accelerator
replaces. Whether Nucleus reuses Gluten's shuffle and memory code is **[Ask]**; do not assert it.

**Basis.** Local checkout `~/github/gluten` (remote `apache/gluten`), branch `main` at commit `258bf8ec8`
(2026-10-26). All paths below are relative to that repo. `main` moves, so run `git -C ~/github/gluten pull` and
re-check anything you plan to quote.

Tags used in this guide:
- **[Source]**: I read it in the code at the commit above.
- **[Commit]**: from `git log` / `git show` of the named commit. The commit message is often empty, so "why" is
  stated only where the message or a code comment says it.
- **[Inference]**: my reading of a diff or code, not stated by the authors. Say "I read it as".
- **[General]**: standard engineering knowledge.
- **[Ask]**: not knowable from the source; ask the interviewer.

## Contents
1. [Code Map](#1-code-map)
2. [Memory: How a Native Allocation Becomes a Spark Reservation](#2-memory-how-a-native-allocation-becomes-a-spark-reservation)
3. [Memory: Issues and Fixes](#3-memory-issues-and-fixes)
4. [Shuffle: Architecture](#4-shuffle-architecture)
5. [Shuffle: Issues and Fixes](#5-shuffle-issues-and-fixes)
6. [Interview Q&A](#6-interview-qa)
7. [Debugging Playbook](#7-debugging-playbook)
8. [Design Exercises](#8-design-exercises)
9. [Questions to Ask DataPelago](#9-questions-to-ask-datapelago)
10. [Cheat Sheet](#10-cheat-sheet)
11. [References](#11-references)

---

## 1. Code Map

| Concern | Where | What it does |
|---|---|---|
| Spark memory consumer tree | `gluten-core/.../memory/memtarget/spark/TreeMemoryConsumer.java`, `TreeMemoryConsumers.java` | Registers one Spark `MemoryConsumer` per task; hangs a tree of named children under it |
| Memory target decorators | `.../memtarget/ThrowOnOomMemoryTarget.java`, `OverAcquire.java`, `RetryOnOomMemoryTarget.java`, `DynamicOffHeapSizingMemoryTarget.java` | Stack of wrappers that add OOM retry, probe-spill, and on-heap/off-heap sharing |
| Spill orchestration | `.../memtarget/TreeMemoryTargets.java` (`spillTree`), `Spillers.java` | Walks the tree, SHRINK then SPILL, biggest child first |
| Listener glue | `gluten-arrow/.../memory/listener/ReservationListeners.java`, `ManagedReservationListener.java`, `gluten-arrow/.../memory/NativeMemoryManager.scala` | Builds the target stack; JNI entry point for native reservations |
| Native block rounding | `cpp/core/memory/AllocationListener.h` (`BlockAllocationListener`) | Rounds reservations to 8 MB blocks before calling Java |
| Velox arbitrator | `cpp/velox/memory/VeloxMemoryManager.cc` (`ListenableArbitrator`) | Plugs Velox pool growth into the Java reservation |
| Operator spill entry | `cpp/velox/compute/WholeStageResultIterator.cc` (`spillFixedSize`), `gluten-arrow/.../vectorized/NativePlanEvaluator.java` | Registers the "spill operators" spiller |
| Shuffle manager | `gluten-substrait/.../shuffle/sort/ColumnarShuffleManager.scala` | Extends Spark's `SortShuffleManager`; routes columnar vs row handles |
| Shuffle writer (JVM) | `backends-velox/.../shuffle/ColumnarShuffleWriter.scala` | Drives the native writer over JNI; commits data and index files |
| Writer type choice | `backends-velox/.../backendsapi/velox/VeloxSparkPlanExecApi.scala` (`getShuffleWriterType`) | Hash vs sort vs RSS-sort |
| Native hash writer | `cpp/velox/shuffle/VeloxHashShuffleWriter.cc` | Per-partition column buffers, split, spill |
| Native sort writer | `cpp/velox/shuffle/VeloxSortShuffleWriter.cc` | Row pages plus a sorted array of 64-bit row ids |
| Local partition writer | `cpp/core/shuffle/LocalPartitionWriter.cc` | Merges small payloads, caches, spills, writes the final file |
| Shuffle reader | `cpp/velox/shuffle/VeloxShuffleReader.cc`, `backends-velox/.../shuffle/ColumnarShuffleReader.scala` | Deserializes fetched streams into batches, merges small ones |
| Config definitions | `gluten-core/.../config/GlutenCoreConfig.scala`, `gluten-substrait/.../config/GlutenConfig.scala`, `backends-velox/.../config/VeloxConfig.scala` | All keys and defaults quoted below |

---

## 2. Memory: How a Native Allocation Becomes a Spark Reservation

### The problem Gluten solves

Spark's unified memory manager governs **JVM-visible** execution memory, per task, with spill callbacks. Native code
allocates outside the JVM. If native allocations are invisible to Spark, Spark cannot ask anyone to spill and the
executor container gets OOM-killed by the OS or cluster manager instead of failing a task cleanly. Gluten's answer:
**every native allocation is reported to Spark's `TaskMemoryManager` as off-heap execution memory, and Spark's spill
requests are routed back into native code.** [Source]

### The reservation path

```
Velox operator / Arrow buffer allocates
        |
        v
Velox memory pool -> ListenableArbitrator.growCapacity      (cpp/velox/memory/VeloxMemoryManager.cc)
        |  asks listener for bytes, rounded up to the transfer capacity (= reservationBlockSize)
        v
BlockAllocationListener   (rounds to 8 MB blocks; only calls Java when the block count changes)
        |  JNI
        v
ManagedReservationListener.reserve       synchronized(sharedLock = TaskMemoryManager)
        |
        v
ThrowOnOomMemoryTarget
   -> OverAcquire                          (probe: borrow ratio*used, immediately repay)
      -> [DynamicOffHeapSizingMemoryTarget, only if enabled]
         -> RetryOnOomMemoryTarget         (legacy mode with more than one task slot)
            -> TreeMemoryConsumer.Node     (this runtime's node, may have a capacity cap)
               -> TreeMemoryConsumer       (the Spark MemoryConsumer, one per task)
                  -> acquireMemory -> Spark TaskMemoryManager / ExecutionMemoryPool
```

Built in `ReservationListeners.create0` and `MemoryTargets.newConsumer`. [Source]

Key points to be able to say out loud:

1. **Two-level rounding.** `BlockAllocationListener` only calls Java when the number of 8 MB blocks changes
   (`reserve()` computes `ceil(used / blockSize)`), so a stream of small allocations does not hammer a lock that is
   shared with Spark. The Velox arbitrator additionally rounds growth up to `memoryPoolTransferCapacity_`, which is
   set to the same block size in `getExtraArbitratorConfigs`. [Source]
2. **One Velox root pool per task runtime.** `ListenableArbitrator` checks `candidates_.size() == 1` in
   `growCapacity` and `shrinkCapacity`, so arbitration never has to choose between competing pools; competition is
   delegated to Spark. [Source]
3. **`growCapacityInternal` reclaims before it asks.** It first tries to satisfy the request from the pool's own free
   capacity, then calls `shrinkPool(pool, 0)` to take back unused reservation, and only then calls
   `listener_->allocationChanged(neededBytes)`. If that throws, it hands the reclaimed bytes back before rethrowing.
   [Source]
4. **The tree.** The root is a real Spark `MemoryConsumer`, so Spark can ask it to spill. Children are named nodes
   (one per native runtime: plan evaluator, shuffle writer, and so on) with optional capacity caps. Usage stats roll
   up from children, which is what the OOM message prints. [Source]

### Who spills, and in what order

`TreeMemoryTargets.spillTree(node, bytes)`: [Source]

- Runs **two phases in order: `SHRINK`, then `SPILL`**. Stops as soon as enough bytes are freed.
- Within a phase, sorts children by `usedBytes` **descending** and recurses into the biggest first. If still short, it
  asks the node's own spiller.
- A spiller sees `(self, phase, size)`. `Spillers.withMinSpillSize` bumps small requests up to the reservation block
  size, because releases smaller than a block never reach the Java-side listener (comment in `Spillers.java`).

Three spillers register on a task: [Source]

| Spiller | Registered in | Responds to | Effect |
|---|---|---|---|
| Native memory manager | `NativeMemoryManager.scala` | `SHRINK` only | `shrink(handle, size)` returns unused Velox pool capacity. Cheap, loses nothing |
| Plan evaluator | `NativePlanEvaluator.java` | `SPILL` only | `WholeStageResultIterator::spillFixedSize`: shrink first, then if `spillStrategy == "auto"` call the arbitrator's `shrinkCapacity(remaining)`, which makes Velox operators (hash agg, join, sort, window) spill to disk |
| Shuffle writer | `ColumnarShuffleWriter.scala` | `SPILL` only | `shuffleWriterJniWrapper.reclaim(...)`, see section 4 |

So under pressure the order is: free reserved-but-unused native capacity, then spill the largest consumer (operators
or shuffle buffers, whichever holds more), then the next one.

### How Spark and Gluten deadlock-proof each other

- `ManagedReservationListener` takes **one lock shared by every listener of the task** (the `TaskMemoryManager`
  object). The comment cites issue #6622: separate locks produced ABBA deadlocks when several listeners existed under
  one TMM. [Source]
- `AppendableSpillerList` is copy-on-write because shuffle writers append spillers lazily (first batch) while another
  thread may be walking the list inside a JNI spill. [Source: comment in `Spillers.java`]

### Retry, OOM, and the "multi-slot" problem

- **Legacy mode** (`spark.gluten.memory.isolation=false`, default): each task may take as much as Spark grants. Spark's
  per-task cap is dynamic (between `1/2N` and `1/N` of the pool), so a request can fail simply because another task
  started. `RetryOnOomMemoryTarget` handles this: if `granted < size`, spill the **whole tree from the root**, then
  borrow the remainder once more. Added for issue #8128 / PR #8132; skipped when there is one task slot.
  [Source: comment in `MemoryTargets.java`]
- **Isolation mode**: each task gets a fixed cap (`conservativeTaskOffHeapMemorySize` = executor off-heap / task
  slots). Recommended by the docs when not all native memory is spillable, to prevent one task from starving others.
  Origin: issue #3030. [Source: `TreeMemoryConsumers.isolatedRoot` Javadoc, `GlutenCoreConfig`]
- **`ThrowOnOomMemoryTarget`**: on failure, calls `System.gc()`, then retries up to 9 times with sleeps of
  1, 2, 4, ... 256 ms (cap 1 s total). Reason in the comment: some off-heap holders release memory in `finalize`, so
  their lifetime depends on a JVM GC. After the budget, it prints Spark's memory usage, the relevant config values, and
  the tree dump, then throws `OutOfMemoryException`. [Source]
- **`OverAcquire`** (`spark.gluten.memory.overAcquiredMemoryRatio`, default **0.3**): after each successful borrow it
  borrows `ratio * used` from a second consumer **and immediately repays it**. The borrow forces Spark to spill *other*
  consumers if the pool is tight; the repay returns the space. Net effect: pressure is applied early, before a
  non-spillable allocation fails. It originally held the extra bytes as a reserve; PR #8247 changed it to this probe,
  and it holds nothing now. [Source: class Javadoc]

### Dynamic off-heap sizing (experimental)

`spark.gluten.memory.dynamic.offHeap.sizing.enabled` (default false, static): ignore `spark.memory.offHeap.size` and
let on-heap and native share `spark.executor.memory`. Native reservations are charged against the **on-heap** pools,
a global counter enforces a quota of on-heap committed plus native used, and the code calls `System.gc()` and adjusts
`MinHeapFreeRatio` / `MaxHeapFreeRatio` through `HotSpotDiagnosticMXBean` so the JVM gives pages back. The docs
themselves warn it can oversubscribe memory. [Source: `GlutenCoreConfig`, `DynamicOffHeapSizingMemoryTarget`]

### Config you should be able to name

| Key | Default | Meaning |
|---|---|---|
| `spark.memory.offHeap.enabled` / `.size` | docs example: true / 30G | Gluten requires off-heap sized for native use |
| `spark.gluten.memory.isolation` | false | Per-task cap = off-heap / slots |
| `spark.gluten.memory.overAcquiredMemoryRatio` | 0.3 | Probe-spill ratio |
| `spark.gluten.memory.reservationBlockSize` | 8MB | JNI reservation granularity |
| `spark.gluten.memory.dynamic.offHeap.sizing.enabled` | false | Experimental shared on-heap/off-heap quota |
| `spark.gluten.memory.untracked` | false | Hides native allocations from Spark; test only |
| `spark.gluten.sql.columnar.backend.velox.spillStrategy` | auto | Whether operator spill-to-disk is allowed |

---

## 3. Memory: Issues and Fixes

Grouped by theme. Hashes are from `git log` on `main`. Each is a good "tell me about a bug you'd expect here"
answer, because the failure mode generalizes.

### Accounting must match reality

| Commit | Issue | Fix |
|---|---|---|
| `08d401926` (2026-07, GLUTEN-12424) | `DynamicOffHeapSizingMemoryTarget` booked the **requested** size before the wrapped target answered. If the target granted less, phantom bytes stayed in a static counter, and later quota checks rejected allocations that were within budget | Book what `borrow` actually granted and what `repay` actually freed [Commit] |
| `f350a440d` (2026-04) | `StdMemoryAllocator::allocateZeroFilled` added `size` instead of `nmemb * size` to its byte counter (and had no overflow check) | Count `nmemb * size`, check overflow [Commit] |
| `4876f4e55` (2026-10, GLUTEN-13093) | With dynamic sizing, reservations are charged to the **on-heap** pools, but "available memory" was read from the **off-heap** pools, so the shuffle writer's per-task budget came from the wrong pool. Separately, a failed `MaxHeapFreeRatio` revert skipped the `MinHeapFreeRatio` revert, leaving a permanent JVM flag change | Read the same pool family as the reservation path, and revert each flag independently [Commit] |
| `44c604825` (2026-08, GLUTEN-12676) | `spark.executor.memory` is declared in MiB (bare number = MiB) but was read with accessors that assume bytes or `getLong`, so units like `384m` failed to parse | Use matching accessors (`getSizeAsMb`), guard sign and overflow in one `mibToBytes` helper [Commit] |
| `1b6197dec` (2026-10, GLUTEN-13359) | Shrinking an aligned block used `realloc`, which guarantees only fundamental alignment, so a moved block could come back misaligned for SIMD use | Allocate an aligned block first and copy if `realloc` returns a misaligned pointer [Commit] |

Generalizable lesson: **a memory ledger is only as good as its weakest accounting path.** Partial grants, frees that
do less than requested, units, and which pool family you read are where it drifts.

### Concurrency around the tree

| Commit | Issue | Fix |
|---|---|---|
| (older) issue #6622 | ABBA deadlock between listeners under one `TaskMemoryManager` | One shared lock for all listeners of a task [Source: comment] |
| `a03f6bf7c` (2026-02, GLUTEN-11509) | `TreeMemoryConsumer` kept children in a `HashMap` while spill threads and task threads touched it. Its `stats()` also asserted `stats.current == getUsed()`, which can legitimately be false while another thread is mid-allocation | `ConcurrentHashMap` with `putIfAbsent`; dropped the racy assertion [Commit] |
| `413901f9d` (2026-08, GLUTEN-12716) | `AppendableSpillerList` was an `ArrayList`. A shuffle writer appends its spiller on the first batch while another thread iterates during a JNI spill | `CopyOnWriteArrayList`; the comment notes a lock would not cover a spiller that appends during its own spill [Commit] |

### Failure reporting

| Commit | Issue | Fix |
|---|---|---|
| `e06ce2994` (2026-09, #13147) | When a task was killed while `ThrowOnOomMemoryTarget` was sleeping between retries, the `InterruptedException` handler did `break`, fell through to the "OOM" path, and reported a cancelled task as an off-heap OOM | Re-set the interrupt flag and throw "Interrupted while waiting..." instead [Commit] |
| `de3b4c40d` (2026-08) | Calling `hold()` on a native memory manager after `release()` dereferenced a freed handle and SIGSEGV'd the executor | Check an `AtomicBoolean` and throw a `GlutenException`; a double release also throws [Commit, Source] |
| `0209ec8e1` (2026-07) | With `-Xms == -Xmx` the JVM heap cannot shrink, so `System.gc()` cannot return pages and dynamic sizing silently cannot work | Log an error at class init, like the existing checks for `-XX:+DisableExplicitGC` [Commit] |

Operational lesson: a wrong error message sends on-call to the wrong team. A cancelled task reported as OOM would
trigger memory-tuning work for a non-problem.

### Native lifetime

| Commit | Issue | Fix |
|---|---|---|
| `933315982` (2026-10, GLUTEN-13225) | Use-after-free of a memory pool in the micro-benchmark input dumper | Fixed ownership in `VeloxWholeStageDumper`; added a C++ test [Commit] |
| `810810ce9` then `f04968b10` (2026-09) | An async data-cache load that was marked cancelled but never scheduled before the timeout kept holding the memory pool pointer. The fix added a wait barrier | **Reverted 4 days later** (#13029). The reverted-by commit gives no reason, so the motivation is **[Ask]** if you want to discuss it |

Source-visible lifetime design: `VeloxMemoryManager::tryDestructSafe` refuses to destroy the manager while any held
pool still has reserved bytes or while extra pools exist, and the destructor logs an error rather than crashing.
`TreeMemoryConsumers.Factory.release` warns if the Spark consumer still has bytes at task end ("may cause memory
leak"). [Source]

---

## 4. Shuffle: Architecture

### Where it plugs in

- `ColumnarShuffleManager` **extends Spark's `SortShuffleManager`**, deliberately. Its class comment explains that
  `spark.shuffle.manager` is process-wide so it also serves row-based exchanges, and Spark skips a defensive per-row
  `UnsafeRow.copy()` only when the manager `isInstanceOf[SortShuffleManager]`. It warns that `registerShuffle`,
  `getWriter` and `unregisterShuffle` are **copied** from Spark and must be re-checked for every Spark version.
  [Source]
- `registerShuffle` returns a `ColumnarShuffleHandle` if the dependency is a `ColumnarShuffleDependency`; otherwise it
  mirrors vanilla Spark's bypass, serialized and base handles. [Source]
- The output is the **standard Spark layout**: one data file per map task plus an index, committed with
  `IndexShuffleBlockResolver.writeMetadataFileAndCommit`. So the external shuffle service, AQE statistics and
  block-fetch machinery all keep working. [Source]
- On the read side `ColumnarShuffleReader` uses a shimmed `ShuffleBlockFetcherIterator` and hands the streams to the
  native deserializer in one batch. If the dependency's serializer is not Gluten's (for example Celeborn fell back to
  the local manager), it dispatches on the **serializer instance type**, not the dependency type, to avoid a
  `ClassCastException` (comment in `ColumnarShuffleReader.scala`). [Source]

### Which writer: hash, sort, or RSS-sort

`VeloxSparkPlanExecApi.getShuffleWriterType`: [Source]

- Not Celeborn: **sort** writer if partitioning is not single and (`numPartitions >= 4000` **or** `numColumns >=
  100000`); otherwise **hash**. The two thresholds are `spark.gluten.sql.columnar.shuffle.sort.partitions.threshold`
  (4000) and `...sort.columns.threshold` (100000).
- GPU stage mode: always hash.
- Celeborn: depends on the configured writer type; with RSS sort it uses `RssSortShuffleWriterType`.

Why two writers: the hash writer keeps **one set of column buffers per partition**, so memory grows with
`partitions x columns`. At thousands of partitions that is too much, so the sort writer instead buffers whole rows in
pages and sorts by partition id. [General, consistent with the 4000-partition default]

### Hash writer internals

`VeloxHashShuffleWriter` (1,697 lines). [Source]

**Per input batch** (`write` -> `doSplit`):
1. Compute partition id per row; count rows per partition (`partition2RowCount_`).
2. State `kPreAlloc`: `calculatePartitionBufferSize` picks a row capacity per partition buffer, then
   `preAllocPartitionBuffers` allocates, resizes, or flushes buffers to fit.
3. State `kSplit`: copy column values into each partition's buffers.
4. If `partitionBufferEvictThreshold_ > 0`, evict any partition whose buffers exceed it, to free memory before the next
   batch. State back to `kInit`.

**Buffer sizing** (`calculatePartitionBufferSize`):
- `bytesPerRow` = fixed-width bytes + running average bytes per row of each binary/string column + average complex
  bytes.
- `memLimit` = the JVM-supplied per-task available off-heap (`availableOffHeapPerTask()` = current available /
  task slots), **plus the cached payload size** (because cached payloads can be spilled), floored at **128 MB**
  (`kMinMemLimit`).
- `preAllocRowCnt = memLimit / bytesPerRow / numPartitions >> 2`, capped by `splitBufferSize_`
  (`spark.gluten.shuffleWriter.bufferSize`, defaults to `maxBatchSize` 4096).
- So the buffer is deliberately sized to use about **a quarter** of the available budget, and bigger partitions
  counts mean smaller buffers, so smaller (and more) output batches.

**Spill** (`reclaimFixedSize`), called when Spark asks the writer to give memory back:
1. If the writer is already inside an evict (`EvictState::kUnevictable`), return 0. Re-entrancy guard, because
   allocating inside an evict can itself trigger a spill request.
2. **Evict cached payloads** (already-merged data in the `PayloadCache`) to a spill file.
3. If still short and the state is `kSplit`, `kInit` or `kStopEvict`: **shrink partition buffers** (give back unused
   capacity; buffers remain usable).
4. If still short and the state is `kInit`: **evict partition buffers** entirely (last resort).

The code comments explain the gating: an OOM in `kSplit` comes from growing a binary column's buffer; in `kInit` from
another operator; in `kStopEvict` from `assembleBuffers`. Order is cache first because it is already a finished unit.
Binary-column resize carries extra guards (`binaryArrayResizeState_`) so a spill triggered *by* resizing a buffer does
not resize that same buffer again.

### Local partition writer (shared by both writers)

`cpp/core/shuffle/LocalPartitionWriter.cc`: [Source]

- `hashEvict(..., kCache)` goes to a **`PayloadMerger`**, which concatenates small payloads for the same partition up
  to `mergeBufferSize` (a payload below `mergeBufferSize * mergeThreshold` is held for merging; threshold default
  0.25), then into the **`PayloadCache`**. Compression only applies above `compressionThreshold` rows (default 100).
- `hashEvict(..., kSpill)` serializes straight to a **spill file** through `LocalSpiller`.
- `reclaimFixedSize`: spill the payload cache first, then flush the merger's partial payloads (uncompressed).
- `stop()`: finish the spill, finish the merger, open the final data file, then **for each partition in order**:
  `mergeSpills(pid)` (read that partition's slice from every spill file) then `writeCachedPayloads(pid)`. Records
  `partitionLengths` for the index.
- **Spill after stop.** `requestSpill(isFinal=true)` lets a spill that arrives *after* `stop()` write directly into
  the final data file (`useSpillFileAsDataFile_`), because by then the data file is being produced and memory must
  still be released.

### Sort writer internals

`VeloxSortShuffleWriter`: [Source]

- Rows are converted from columnar to a compact row format and written into **pages**. Page size is normally
  `min(max(memLimit/4, ...), 64 MB)` but is raised to `minSizeRequired` when a batch needs more.
- An **8-byte "compact row id"** per row packs `partitionId (24 bits) | pageNumber (13 bits) | offsetInPage (27
  bits)`. The array of ids is sorted (radix sort optional) so rows of one partition are adjacent; then
  `evictAllPartitions` walks the sorted ids and emits runs per partition through `diskWriteBufferSize` buffers.
- Hard limits implied by the layout: 16M partitions, 8,192 pages, 128 MiB addressable per page.
- It spills by evicting everything it has (`evictAllPartitions`) and resetting the array to a minimal size. It also
  evicts proactively when the row array would overflow 32 bits or the page count nears 8,191.

### Reader side

`VeloxHashShuffleReaderDeserializer::next` (`VeloxShuffleReader.cc`): [Source]
- Reads blocks from fetched streams; if a block type is "plain payload" it is **merged** with following payloads until
  `batchSize_` rows (merge skipped for dictionary or complex payloads, via `shouldSkipMerge`).
- Small batches are an efficiency problem downstream, so there are three knobs: reader-side raw merge
  `hashShuffle.reader.streamMerge.enabled` (default **false**), `VeloxResizeBatchesExec` on shuffle **input** (default
  true) and **output** (default false), each with a min size of `0.25 * maxBatchSize`.
- `readerBufferSize` and `sort.deserializerBufferSize` default 1 MB.

### Other config

| Key | Default |
|---|---|
| `spark.gluten.sql.columnar.shuffle.codec` / `.codecBackend` | unset (uses Spark shuffle compression codec; backend `qat` possible) |
| `spark.gluten.sql.columnar.shuffle.compression.threshold` | 100 rows |
| `spark.gluten.sql.columnar.shuffle.merge.threshold` | 0.25 |
| `spark.gluten.sql.columnar.shuffle.realloc.threshold` | 0.25 |
| `spark.gluten.sql.columnar.shuffle.partitionBufferEvictThreshold` | -1 (disabled) |
| `spark.gluten.sql.columnar.shuffle.dictionary.enabled` | false |
| `spark.gluten.sql.columnar.shuffle.typeAwareCompress.enabled` | false (not compatible with dictionary) |
| `spark.gluten.shuffleWriter.bufferSize` | unset (falls back to `maxBatchSize` = 4096) |

---

## 5. Shuffle: Issues and Fixes

### Silent data corruption in the sort writer

**`03440927e` (2026-09-30, #13121): "Sort shuffle row ID truncates offsets above 128 MiB."** [Commit]

- *Layout:* the compact row id has 27 bits for the offset within a page, which addresses 128 MiB.
- *Trigger [Inference]:* page capacity is normally capped at 64 MB, but `acquireNewBuffer` takes
  `max(that, minSizeRequired)`, so a single large batch can produce a page larger than 128 MiB. Offsets beyond 2^27
  then wrap when ORed into the id, and rows are read back from the wrong place.
- *Why it was dangerous:* the only guard was `VELOX_DCHECK_LE(pageCursor_, currenPageSize_)`, a **debug-only** check.
  In release builds it would produce wrong shuffle data, not a crash.
- *Fix (from the diff):*
  - `maxRowsToInsert` now limits rows so that **no row starts at an offset above `2^27 - 1`**; a row may *extend* past
    the boundary but the next row starts on a fresh page.
  - Page count is checked against the 13-bit limit; at the limit it evicts everything first.
  - `insertRows` returns `arrow::Status` and validates page, offset, and record size instead of `void` plus a
    `DCHECK`.
  - `evictPartition` re-validates every row id it reads: unknown page, offset outside the page, row length larger
    than the page all return `Invalid` instead of reading out of bounds. Page sizes are snapshotted before spilling.
  - Added a test in `VeloxSortShuffleWriterTest.cc`.
- *Lesson:* **bit-packed ids need explicit range checks in release builds, and decoding must treat its input as
  untrusted even if the encoder is yours.**

### Hash writer spill selection

**`a69348d37` (2026-06-10, GLUTEN-12235): "Fix hash shuffle spill ordering when evicting partition buffers."**
The commit message is empty. What the diff shows [Commit]:
- Ranking by **estimated bytes** (`estimatePartitionBufferBytes`) rather than by buffer row capacity.
- Selects the set of partitions that reaches the requested size, then evicts them in **ascending partition id**.
- Builds each payload with the **rows actually filled** (`partitionBufferBase_[pid]`) instead of the buffer's
  capacity (the old code passed `item.second`, the buffer size, as the row count).
- Returns the real pool-bytes delta.
- I read it as: the old loop could mis-rank partitions and mislabel row counts; the exact production symptom is
  **[Ask]/[Inference]**.

### Memory held by large partition buffers

**`0b1e5115e` (2026-05-19, #12089) and follow-up `4df6e9cd1`:** add
`spark.gluten.sql.columnar.shuffle.partitionBufferEvictThreshold`. After each split, any partition buffer larger than
the threshold is evicted so the next batch has headroom. Off by default (-1). The current code runs the eviction under
a `PartitionScopeGuard`. [Commit, Source]

### Small batches after shuffle

**`ad03c2d99` (2026-05-19, #12097) "Restore hash shuffle reader payload merging"**, plus `71302c639`
(copyRanges fast path for batch resizing). Reader-side merging of plain payloads cuts per-batch overhead before Velox
vectors are even built; config is documented in `VeloxConfig.scala` and still defaults to off. [Commit, Source]

### RSS-sort reader hangs and misreads

Two related fixes for GLUTEN-12985: [Commit]
- **`48b5fca64` (2026-09-17):** a page header could declare more bytes than the stream held (truncated partition
  data). The reader's `next()` ignored the `throwIfPastEnd` argument and returned silently at EOS, so a `for(;;)` loop
  never exited: **an infinite loop instead of an error**. Fix: honor the argument and throw via
  `GLUTEN_ASSIGN_OR_THROW`. A 211-line reader test was added.
- **`cf7650496` (2026-09-21):** pages that span more than one read window. The stream used a single-window refill
  buffer, so a deserializer that seeks backward could read data already overwritten. Fix (per the code comments):
  fast path deserializes in place when the whole page fits in the window; slow path reassembles the page into a
  contiguous buffer; `seekp()` into an earlier window now fails fast. The custom `GlutenByteStream.h` was removed.

### Disk leak on decommissioned executors

**`8b8586340` (2026-09-23, #13072):** `ColumnarShuffleManager` constructed its `IndexShuffleBlockResolver` with the
single-argument constructor, so the resolver created **its own** `taskIdMapsForShuffle`, while the manager kept a
different one. Blocks migrated in during executor decommissioning were recorded in the resolver's map, but
`unregisterShuffle` read the manager's map, so those files were never deleted. Fix: share one map, declare it before
the resolver (Scala `val`s initialize in declaration order, otherwise the resolver captures `null`), iterate it under
`mapTaskIds.synchronized` as Spark does. A suite test fails if the maps are ever unshared. The commit message states
all of this directly. [Commit]

*Lesson:* when you copy a framework class's wiring (as the class comment admits), every upstream constructor change is
a latent divergence. The comment on the constructor call lists the signature differences across Spark 3.4, 3.5, 4.0
and 4.1.

---

## 6. Interview Q&A

Answers are written to be spoken. Numbers and names are real [Source]; use "I read it as" for [Inference] items.

### Q1. A native hash join wants 2 GB. Walk me through what happens inside an executor.

**Spoken answer:**
"The Velox operator's memory pool asks its arbitrator to grow. Gluten plugs in a custom arbitrator,
`ListenableArbitrator`. It first uses free capacity the pool already holds, then shrinks any unused reservation, and
only then asks a listener for the remaining bytes, rounded up to an 8 MB block. A block-rounding listener in front of
the JNI call means we only cross into Java when the block count actually changes.

"On the Java side the request lands in `ManagedReservationListener`, which takes a lock shared by everything in the
task, then goes down a stack of decorators: an OOM wrapper that retries with a GC, an over-acquire probe, and finally a
node in a tree rooted at a real Spark `MemoryConsumer`. That calls `acquireMemory` on Spark's `TaskMemoryManager`.
If the pool is full, Spark asks other consumers to spill, and that comes back into the tree and into native code. If
Spark grants less than requested, the arbitrator throws, the Java side retries, and eventually the task fails with a
message listing every consumer's usage."

**Follow-ups:**
- *Why 8 MB blocks?* Fewer JNI crossings and less contention on a lock shared with Spark; cost is up to one block of
  slack per runtime and a minimum useful spill size (`withMinSpillSize`).
- *What if the arbitrator throws halfway?* It returns the reclaimed bytes via `allocationChanged(-reclaimedFreeBytes)`
  before rethrowing, so the ledger stays consistent.

### Q2. Memory is tight. Who spills first, and why that order?

**Spoken answer:**
"`spillTree` works in two phases. First SHRINK: give back reservation the native pool holds but is not using, which is
free. Only if that is not enough does it go to SPILL. Within a phase it visits children largest-first, so the biggest
holder gives up memory first, which maximizes bytes freed per spill. The consumers that can spill are the plan
evaluator, which triggers Velox operator spill to disk, and the shuffle writer, which can flush cached payloads and
shrink or evict partition buffers. Requests are rounded up to the reservation block size because anything smaller
would not be visible to the Java-side ledger."

**Follow-ups:**
- *Is largest-first always right?* No. A big consumer may be unspillable or expensive to spill (a hash table build),
  so spilling a small cheap one could be better. The tree has no cost model, only size. That is a fair design critique
  [General].
- *What if spilling frees nothing?* `Node.ensureFreeCapacity` loops while `spilledBytes > 0`; zero means OOM. The code
  has `// FIXME should we add retry limit?` on that loop.

### Q3. Explain `OverAcquire`. Does it hold extra memory?

**Spoken answer:**
"Not any more. The idea is to trigger other consumers' spills *before* a non-spillable allocation fails. After each
successful borrow it borrows `ratio` times current usage (default 0.3) from a sibling consumer and immediately repays
it. The borrow makes Spark spill others if the pool is tight; the repay gives the space back. So later allocations
start from a pool that has already been squeezed. It used to keep the extra as a reserve; PR #8247 turned it into a
probe and added the `granted >= size` gate so it only runs after a full grant."

**Follow-ups:**
- *Cost?* Extra reserve/release on every borrow, and extra spills that might have been unnecessary [General].
- *Why a separate consumer for the probe?* So the probe's usage is distinguishable and `overTarget.usedBytes()` can be
  asserted to be 0 before and after (the class has those `Preconditions`).

### Q4. You see "Not enough spark off-heap execution memory". How do you triage?

**Spoken answer:**
"That message is built in `ThrowOnOomMemoryTarget`. It tells you the acquired and granted bytes, prints the key
config values (off-heap enabled and size, task and conservative task sizes, dynamic sizing), and dumps the tree with
current and peak per consumer. I read three things: who holds the memory (operator pool versus shuffle writer versus
Arrow pool), whether the total is near the per-task cap, and whether this is isolation or legacy mode. In legacy mode
with several slots it may just be that another task grew; in isolation mode the cap is off-heap divided by slots.
Then I look at whether spill is happening at all: operator spill needs `spillStrategy=auto` and a spill dir, and the
shuffle writer logs `Trying to spill`. If spill is not helping, the holder is probably unspillable, such as a large
broadcast build side."

**Follow-ups:**
- *Is it always a real OOM?* Until `e06ce2994` a killed task could surface as an OOM. Check for cancellation.
- *What do you change first?* More off-heap per executor or fewer slots per executor, then isolation, before touching
  shuffle buffer settings.

### Q5. Hash versus sort shuffle writer: when and why?

**Spoken answer:**
"The hash writer keeps column buffers for every partition, which is fast and keeps data columnar, but memory scales
with partitions times columns. Gluten switches to the sort writer at 4000 partitions (or an extreme column count),
which stores rows in pages with a 64-bit id per row packing partition, page and offset, sorts the ids, and writes runs
per partition. It trades CPU (columnar to row, sort, back to columnar on read) for bounded memory. Single-partition
shuffles use the hash writer in single mode, which does not keep partition buffers at all."

**Follow-ups:**
- *What limits does the sort writer's id impose?* 24 bits of partition, 13 bits of page, 27 bits of offset. The 128 MiB
  offset limit is exactly what `03440927e` fixed.
- *Why not always sort?* Reader must rebuild columns from rows; costs CPU and the data loses columnar locality
  [General].

### Q6. Walk me through the hash writer's spill logic.

**Spoken answer:**
"Reclaim is a three-step ladder, guarded by a re-entrancy flag because allocating during an evict can request another
spill. Step one: flush cached payloads, the already-merged data, to a spill file. That is the cheapest because it is
finished work. Step two: shrink partition buffers, releasing their unused tail while keeping them usable. Step three,
only when the writer is between batches: evict partition buffers wholesale. Which steps are allowed depends on the
writer's state, because in the middle of a split the buffers are being written. And the buffers themselves are sized
to roughly a quarter of the per-task budget divided by partitions and row width, so the writer is conservative by
construction."

**Follow-ups:**
- *What happens if a spill is requested after the writer has stopped?* `LocalPartitionWriter::requestSpill(true)`
  writes into the final data file.
- *Why sort selected partitions by id before evicting?* I read it as: spill files are laid out per partition and
  ordered writes simplify the merge in `stop()` (`a69348d37`). That is an inference; the commit has no message.

### Q7. Tell me about a real shuffle bug and how it was fixed.

Pick the sort-writer overflow (`03440927e`):

**Spoken answer:**
"The sort-based writer packs each row's location into 64 bits: 24 for partition, 13 for page, 27 for offset in the
page. Pages are normally at most 64 MB, but a large batch can force a bigger page. Offsets past 128 MiB wrapped and
rows were read back from the wrong place. The only guard was a debug check, so a release build would corrupt data
silently rather than crash. The fix bounds how many rows can be placed so no row *starts* at an unencodable offset,
rotates to a new page, evicts at the page-count limit, and validates every decoded id on the spill path, returning an
error rather than reading out of bounds. They added a regression test. The takeaway for me is that any bit-packed
encoding needs release-mode range checks on both encode and decode."

**Follow-ups:**
- *How would you have caught it earlier?* A fuzz or property test with oversized rows and tiny `memLimit`; a
  compile-time assertion relating max page size to the id width [General].
- *Detection in production?* Checksums. Note `4634c80a2` added row-based checksums in `ColumnarShuffleWriter`
  (`GlutenMapStatusUtil.isRowBasedChecksumEnabled`) [Source].

### Q8. Why does `ColumnarShuffleManager` extend `SortShuffleManager`, and what is the risk?

**Spoken answer:**
"Spark decides whether it needs a defensive per-row copy before shuffle by checking `isInstanceOf[SortShuffleManager]`.
Row-based exchanges still go through the same manager, so extending it avoids a pure-overhead copy. The cost is that
several methods are copied from Spark rather than inherited, so every Spark upgrade can silently diverge. We have a
concrete example: the resolver was built without the shared `taskIdMapsForShuffle`, so blocks migrated during executor
decommissioning were never cleaned up. The fix is one shared map, declared before the resolver because Scala
initializes vals in order."

**Follow-ups:**
- *How would you prevent the next one?* A test per supported Spark version comparing behavior to
  `SortShuffleManager`, which is partly what the added suite case does.

### Q9. Shuffle reads produce many tiny batches. What do you do?

**Spoken answer:**
"There are three layers. The reader can merge plain payloads from a stream before Velox vectors exist
(`streamMerge`, default off). A resize operator can combine batches after the read (`shuffleOutput`, off by default;
the input-side one is on). And a copyRanges fast path makes that resize cheap for flat vectors. The root cause is
usually too many partitions making each map output slice tiny; on the writer side the merge buffer and 25% thresholds
try to coalesce before writing. Complex types and dictionary payloads are not merged by the fast path."

### Q10. How does memory get returned, and what can leak?

**Spoken answer:**
"Teardown order matters. The Java `NativeMemoryManager.release` calls the native release, then warns if the
reservation listener still has nonzero bytes. Natively, `tryDestructSafe` refuses to destroy the Velox manager while
any pool has reserved bytes or extra pools remain, and logs an error instead of crashing. On the Java side the tree
factory warns at task end if bytes are still held. The failure modes we've seen are `hold` after `release` (now a
clear exception rather than SIGSEGV), pools kept alive by async I/O that never ran, and a use-after-free from a debug
dumper."

### Q11. How do you make sure a new spillable component is safe in this design?

**Spoken answer:**
"Four rules. One: register through `NativeMemoryManager.addSpiller`, and respond only in the phase you handle, as the
three existing spillers do. Two: assume the spill call can come from any thread while you're mid-operation, so use a
state flag like the hash writer's evict state and make `reclaim` return 0 when you can't safely release. Three:
spills can re-enter. Allocating inside a spill can trigger another spill, so guard against it. Four: account in what
was granted and freed, not what was requested; that was the dynamic-sizing ledger bug."

### Q12. (DataPelago-relevant) How would this change for a GPU or accelerator engine?

**Spoken answer [General, design discussion; not a statement about Nucleus]:**
"The ledger idea transfers: every device allocation must be visible to the scheduler's memory manager. What changes is
that there are two or three tiers (device, pinned host, disk) and spill is a data movement problem, so a spiller needs
a destination tier and a cost. Shrink-before-spill still applies for pooled device allocators. For shuffle, the
columnar-to-row conversion in the sort writer is probably wrong for an accelerator, so you'd keep hash-style columnar
buffers but must bound `partitions x columns` some other way, for example by two-pass partitioning into a few large
device buffers. Reader-side batch merging becomes more important because kernel launch overhead punishes small
batches."

**Follow-ups:**
- *Where does `GPUStageMode` show up in Gluten?* `getShuffleWriterType` forces hash writers for GPU stages, and there is
  `VeloxGpuShuffleReader` / `VeloxGpuAsyncShuffleReader` with prefetch limits [Source]. A separate GPU hash writer type
  was added then removed (`204b8d712`, `717d3e47f`).
- *Ask them:* how Nucleus handles its own device memory against Spark's accounting **[Ask]**.

---

## 7. Debugging Playbook

| Symptom | First look | Likely cause | Knob |
|---|---|---|---|
| Task fails with "Not enough spark off-heap execution memory" | Tree dump in the error | Unspillable holder, too many slots, small off-heap | `spark.memory.offHeap.size`, slots per executor, `spark.gluten.memory.isolation` |
| Container killed by YARN/K8s with no Spark OOM | Executor logs, RSS vs off-heap setting | Native allocations not tracked (untracked mode, allocator bypass), or non-native memory (JVM, libraries) | Check `spark.gluten.memory.untracked`, memory overhead |
| Spark UI shows huge spill on shuffle | `spillTime`, `bytesSpilled`, `peakBytes` shuffle metrics | Many partitions, large rows, buffers too big for budget | `partitionBufferEvictThreshold`, `shuffleWriter.bufferSize`, partition count |
| Slow map stage with thousands of partitions | `splitTime`, writer type | Hash writer with many partitions, tiny buffers | sort threshold (`sort.partitions.threshold`) |
| Many tiny batches after shuffle | Operator input row counts | Reader not merging | `streamMerge.enabled`, `resizeBatches.shuffleOutput` |
| Shuffle reader hang or wrong data on Celeborn RSS-sort | Reader logs, partition data length | Pre-`48b5fca64` / `cf7650496` builds | Upgrade |
| Disk fills on decommissioning executors | `shuffle` dirs on migrated executors | Pre-`8b8586340` shared-map bug | Upgrade |
| Native crash at task end | Stack in `hold`/`release` | Teardown ordering, pool still referenced | Pre-`de3b4c40d` = SIGSEGV, now an exception |
| "Reported OOM" for a killed task | Task kill reason | Pre-`e06ce2994` | Upgrade |

Metrics emitted by the shuffle writer (from `ColumnarShuffleWriter.scala`): `shuffleWallTime`, `splitTime` (computed as
wall minus spill, write and compress time), `spillTime`, `bytesSpilled`, `dataSize`, `compressTime`, `peakBytes`,
`avgDictionaryFields`, `dictionarySize`, plus `sortTime` and `c2rTime` for the sort writer. [Source]

---

## 8. Design Exercises

### Exercise A: Add a spillable operator to the tree

Prompt: "You add a native top-N operator with a large heap. Make it cooperate with Spark's memory manager."

Cover: reserve through the existing listener (do not talk to Spark directly); register a spiller that answers only
`SPILL`; return actual freed bytes; handle re-entrancy; account granted-not-requested; teardown so reserved bytes are 0
before `release()`; test with a tiny pool and many concurrent tasks; consider whether it should fail fast instead of
spilling when spill is not supported (`spillStrategy` unset).

### Exercise B: Shuffle writer for 100,000 partitions

Prompt: "Hash writer memory is `partitions x columns`. Design for 100k partitions."

Cover: why the sort writer exists; the id-packing trade-off (24/13/27) and how limits become hard failures; two-pass
approaches (counting sort into a few large buffers, then slice); output must still be one data file plus index for
Spark compatibility; behavior under spill (runs per partition, merge at the end); how to bound reader memory (many
tiny blocks).

### Exercise C: Prove a shuffle change did not corrupt data

Cover: row-based checksum on writer and reader (Gluten has an opt-in row checksum in the map status); randomized tests
with huge rows, tiny budgets, empty partitions, nulls in binary columns; compare against vanilla Spark shuffle on the
same plan; fault injection (truncated stream, EOS mid-page, as in `48b5fca64`).

---

## 9. Questions to Ask DataPelago

All are **[Ask]**: the answers are not in public sources.

1. How much of Gluten's JVM-side shuffle and memory machinery do you reuse versus replace (memory targets, spill
   coordination, `ColumnarShuffleManager`)?
2. How does device memory get reported into Spark's `TaskMemoryManager`, and what does spill mean for a GPU-resident
   operator?
3. Do you ship your own shuffle writer, or leave Spark's shuffle and convert at the boundary? How do you avoid the
   columnar-to-row cost the sort writer pays?
4. What is the largest shuffle bug you have shipped, and how was it detected (checksums, customer report, test)?
5. How do you track upstream Gluten fixes such as the ones in this guide?

---

## 10. Cheat Sheet

| Item | Value | Source |
|---|---|---|
| Reservation block | 8 MB | `GlutenCoreConfig` |
| Over-acquire ratio | 0.3 | `GlutenCoreConfig` |
| OOM retry | 9 sleeps, 1 to 256 ms, 1 s cap, `System.gc()` first | `ThrowOnOomMemoryTarget` |
| Spill phases | SHRINK then SPILL, largest child first | `TreeMemoryTargets` |
| Hash writer memory floor | 128 MB (`kMinMemLimit`) | `ShuffleWriter.h` |
| Hash buffer rows | `memLimit / bytesPerRow / partitions >> 2`, capped by 4096 | `VeloxHashShuffleWriter.cc` |
| Sort threshold | 4000 partitions or 100000 columns | `GlutenConfig` |
| Compact row id | 24 partition, 13 page, 27 offset bits | `VeloxSortShuffleWriter.cc` |
| Merge threshold / realloc threshold | 0.25 / 0.25 | `GlutenConfig` |
| Compression threshold | 100 rows | `GlutenConfig` |
| Reader buffers | 1 MB | `GlutenConfig` |
| Hash reclaim ladder | cached payloads, shrink buffers, evict buffers | `reclaimFixedSize` |

Three sentences to land:
1. "Native memory is only safe if every byte is on Spark's ledger and Spark's spill requests can reach native code."
2. "Most of the recent bugs are accounting, concurrency or range-check bugs, not algorithm bugs."
3. "The worst ones were silent: truncated offsets and an EOS loop, so I'd want checksums and fault-injection tests."

---

## 11. References

Primary source: `apache/gluten` `main` at `258bf8ec8` (2026-10-26), local checkout `~/github/gluten`.

Commits cited (all `git show <hash>` in that repo):

| Hash | Date | Subject |
|---|---|---|
| `03440927e` | 2026-09-30 | Sort shuffle row ID truncates offsets above 128 MiB (#13121) |
| `a69348d37` | 2026-06-10 | Fix hash shuffle spill ordering when evicting partition buffers (#12231) |
| `0b1e5115e`, `4df6e9cd1` | 2026-05 | Evict large partition buffers after split (#12089, #12156) |
| `ad03c2d99` | 2026-05-19 | Restore hash shuffle reader payload merging (#12097) |
| `48b5fca64`, `cf7650496` | 2026-09 | RSS-sort reader mid-page EOS and multi-window pages (#12986, #12984) |
| `8b8586340` | 2026-09-23 | Share taskIdMapsForShuffle between manager and resolver (#13072) |
| `e06ce2994` | 2026-09-30 | Do not report an interrupted task as off-heap OOM (#13147) |
| `4876f4e55` | 2026-10-21 | Fix two dynamic off-heap sizing memory bugs (#13094) |
| `08d401926` | 2026-07-03 | Fix dynamic sizing ledger drift (#12425) |
| `a03f6bf7c` | 2026-02-04 | Make TreeMemoryConsumer thread-safe (#11553) |
| `413901f9d` | 2026-08-11 | Make AppendableSpillerList safe to append during a spill (#12717) |
| `de3b4c40d` | 2026-08-14 | Fail fast when `hold()` follows `release()` |
| `44c604825` | 2026-08-06 | Read MiB-declared Spark memory configs with matching accessors (#12677) |
| `f350a440d` | 2026-04-01 | Fix `allocateZeroFilled` byte accounting (#11855) |
| `1b6197dec` | 2026-10-22 | Keep alignment when shrinking an aligned block (#13360) |
| `0209ec8e1` | 2026-07-15 | Log error when `-Xms` equals `-Xmx` |
| `810810ce9`, `f04968b10` | 2026-09 | Async load pool holding fix, then its revert (#12919, #13029) |
| `933315982` | 2026-10-09 | Fix use-after-free in micro-benchmark input dump (#13227) |

Issue numbers named in source comments: #3030 (task memory isolation), #6622 (ABBA deadlock), #8128 and #8132
(retry on OOM in legacy mode), #8247 (over-acquire as a probe), #5439 (dynamic off-heap sizing).
