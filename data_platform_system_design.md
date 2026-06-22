# Data Platform & Infrastructure System Design

Design problems for data platform / data infrastructure roles — processing
engines, table formats, warehouses, and ingestion/serving pipelines. Companion to
[`common_system_design_questions.md`](common_system_design_questions.md) (general
systems) and the conceptual deep-dives in
[`iceberg_spark_interview_questions.md`](iceberg_spark_interview_questions.md).

Each problem follows the same spine: Requirements → Capacity → Architecture →
core components → tradeoff tables → failure modes → monitoring → security →
deep-dive questions.

---

## Table of Contents

### Processing Engines
1. [Batch Processing Engine](#1-batch-processing-engine)
2. [Stream Processing Engine](#2-stream-processing-engine)
3. [Lakehouse Table Format](#3-lakehouse-table-format)
4. [OLAP Data Warehouse](#4-olap-data-warehouse)

### Pipelines & Serving
5. [Change Data Capture (CDC) Pipeline](#5-change-data-capture-cdc-pipeline)
6. [Real-Time OLAP Serving](#6-real-time-olap-serving)
7. [Metrics / Time-Series Database](#7-metrics--time-series-database)
8. [Distributed Query Engine](#8-distributed-query-engine)

---

## 1. Batch Processing Engine

**Problem:** Design a distributed batch processing engine like Apache Spark or MapReduce that runs large data transformations across a cluster.

### Requirements

**Functional:**
- Run transformations (map, filter, join, groupBy, aggregate) over datasets that don't fit on one machine
- Express jobs as a DAG of operations; execute lazily
- Read/write many sources (object store, HDFS, JDBC, columnar files)
- Recover automatically from worker failures without restarting the whole job
- Support caching/persisting intermediate results for reuse

**Non-Functional:**
- Scale horizontally to thousands of cores / petabytes
- High throughput for sequential scans; tolerate stragglers
- Fault tolerant (commodity hardware fails routinely)
- Efficient memory use with graceful spill to disk

### Capacity Estimation

```
Dataset: 10 TB input, 200 executors × 4 cores = 800 task slots
Default partition size: 128 MB  ->  10 TB / 128 MB ≈ 80,000 partitions (tasks)
Waves: 80,000 tasks / 800 slots ≈ 100 waves

Shuffle: a wide transformation re-partitions all 10 TB across the network.
  At 10 GB/s aggregate cluster bandwidth -> ~1,000s just to move shuffle data.
  => Minimizing shuffle volume is the dominant performance lever.
```

### High-Level Architecture

```
                       +------------------------+
                       |        Driver          |
                       |  +------------------+  |
                       |  |  DAG Scheduler   |  |  job -> stages (at shuffle
                       |  +------------------+  |        boundaries)
                       |  |  Task Scheduler  |  |  stage -> tasks (1 per partition)
                       |  +------------------+  |
                       +-----------+------------+
                                   | launch tasks / track lineage
        +--------------------------+--------------------------+
        |                          |                          |
+-------v--------+        +--------v-------+         +--------v-------+
|   Executor 1   |        |   Executor 2   |         |   Executor 3   |
| task task task |        | task task task |         | task task task |
| block manager  |        | block manager  |         | block manager  |
| (cache+shuffle)|        | (cache+shuffle)|         | (cache+shuffle)|
+-------+--------+        +--------+-------+         +--------+-------+
        |                          |                          |
        +--------------------------+--------------------------+
                                   |
                       +-----------v------------+
                       |   Shuffle Service /    |
                       |   Object Store (data)  |
                       +------------------------+
   Cluster manager (YARN / K8s / standalone) allocates executors.
```

### Job → Stages → Tasks

The engine compiles the DAG into **stages** split at **shuffle boundaries**.
Operations with **narrow** dependencies (each output partition depends on one
input partition — `map`, `filter`) pipeline within a stage. **Wide**
dependencies (each output depends on many inputs — `groupByKey`, `join`) require
a shuffle and start a new stage.

```python
class DAGScheduler:
    def submit_job(self, final_rdd, partitions):
        # Walk the lineage backwards, cutting a new stage at each wide dependency
        stages = self._build_stages(final_rdd)
        # Stages form their own DAG; run a stage only once all parents finish
        for stage in self._topological_order(stages):
            self._submit_missing_tasks(stage)

    def _build_stages(self, rdd, stages=None):
        stages = stages or []
        for dep in rdd.dependencies:
            if dep.is_shuffle:           # wide dep => stage boundary
                shuffle_stage = self._get_or_create_stage(dep.rdd)
                stages.append(shuffle_stage)
            else:                        # narrow dep => same stage, recurse
                self._build_stages(dep.rdd, stages)
        return stages

    def _submit_missing_tasks(self, stage):
        # One task per partition; locations chosen for data locality
        tasks = [
            Task(stage.id, p, preferred_locs=self._locality(stage.rdd, p))
            for p in stage.missing_partitions()
        ]
        self.task_scheduler.submit(tasks)
```

### Shuffle

The shuffle is the engine's most expensive and failure-prone operation:
map tasks partition their output by the reduce key and write **shuffle blocks**
to local disk; reduce tasks **fetch** the blocks for their partition from every
mapper.

```python
class ShuffleMapTask:
    def run(self, partition):
        # Map side: partition records by target reducer, write sorted blocks
        buckets = [[] for _ in range(self.num_reducers)]
        for record in self.compute(partition):
            r = self.partitioner(record.key)        # e.g. hash(key) % num_reducers
            buckets[r].append(record)
        # Spill to disk if buckets exceed memory; write one index + data file
        return self.block_manager.write_shuffle(self.shuffle_id, partition, buckets)

class ShuffleReduceTask:
    def run(self, reducer_id):
        # Reduce side: fetch this reducer's block from every map output
        blocks = self.fetch_blocks(self.shuffle_id, reducer_id, self.map_locations)
        merged = self.merge_and_aggregate(blocks)   # external merge-sort if large
        return merged
```

### Fault Tolerance via Lineage

Instead of replicating intermediate data, the engine records **lineage** (how
each partition was derived). If an executor dies, the driver **recomputes** only
the lost partitions from their parents.

```python
class Partition:
    def compute(self):
        # Recompute on demand from parent partitions + the transformation
        if self.cached_value is not None:
            return self.cached_value
        parent_data = [p.compute() for p in self.parents]
        return self.transformation(parent_data)
```

- **Narrow lineage** recovers cheaply (one parent partition per lost partition).
- **Wide lineage** can cascade — a lost shuffle output may force re-running the
  whole map stage. **Checkpointing** (persist to durable storage, truncate
  lineage) bounds recomputation for long pipelines.

### Performance: Skew, Spill, and Adaptive Execution

```python
# Skew mitigation: salt a hot key so it spreads across reducers
from pyspark.sql.functions import col, concat, lit, floor, rand

salted = df.withColumn("salted_key", concat(col("key"), lit("_"),
                                             floor(rand() * 16)))
# join against an exploded lookup table replicated 16x, then drop the salt
```

| Technique | Problem solved | Mechanism |
|-----------|----------------|-----------|
| **Salting** | One key dominates a reducer | Spread the hot key across N synthetic keys |
| **Broadcast join** | Small × large join shuffles both sides | Ship the small side to every executor; no shuffle |
| **Map-side combine** | Large shuffle volume | Pre-aggregate per partition before shuffle |
| **AQE coalesce** | Too many tiny post-shuffle partitions | Merge small partitions at runtime |
| **AQE skew join** | Straggler reducer on a skewed key | Split the skewed partition into sub-tasks |
| **Spill to disk** | Aggregation exceeds memory | External merge-sort, bounded memory |

### Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Recovery | Lineage + recompute | Avoids replicating large intermediate data |
| Execution unit | Stage split at shuffle | Pipelines narrow ops; isolates expensive shuffle |
| Scheduling | Locality-aware tasks | Move compute to data, not data to compute |
| Memory | Unified cache + execution, spill | Use RAM when possible, degrade gracefully |
| Straggler handling | Speculative execution | Re-launch slow tasks elsewhere, take first finish |

### Failure Scenarios & Mitigation

| Failure Mode | Impact | Detection | Mitigation |
|--------------|--------|-----------|------------|
| Executor crash | Lost cached + shuffle blocks | Heartbeat timeout | Recompute from lineage; external shuffle service preserves blocks |
| Straggler task | Stage stalls on slowest task | Task runtime vs median | Speculative execution |
| Data skew | One reducer OOM / runs hours | Per-task input size | Salting, AQE skew join |
| Driver failure | Whole job lost | Driver health | Checkpoint + restart; HA driver in cluster mode |
| Shuffle fetch failure | Reduce stage can't proceed | Fetch errors | Retry; re-run upstream map stage if blocks gone |
| Small files / tiny partitions | Scheduler overhead dominates | Partition count vs data size | Coalesce, target ~128 MB partitions |

### Monitoring & Observability

**Key Metrics:**
- **Stage/task duration distribution** — straggler and skew detection
- **Shuffle read/write bytes** — dominant cost driver
- **Spill (memory → disk) bytes** — memory pressure
- **GC time per executor** — JVM health
- **Task failure / retry rate** — instability
- **Data locality ratio** (PROCESS_LOCAL vs ANY)

**Alerting:**
- Job runtime exceeds 2× historical baseline
- Any stage with max-task-time > 5× median (skew)
- Executor OOM kills
- Shuffle fetch failure rate > 0

### Security Considerations

- **Multi-tenant isolation**: per-job containers/cgroups (YARN/K8s), resource quotas
- **Data access**: short-lived credentials per job; encrypt shuffle data in transit and at rest
- **Untrusted code**: UDFs run in the executor JVM/process — sandbox or restrict in shared clusters
- **Audit**: log table/path access for governance (lineage to a catalog)

### Interview Deep-Dive Questions

4. **Why does Spark recover faster than MapReduce on failure?**
   - MapReduce materializes every map output to disk and re-reads between stages
   - Spark keeps narrow chains in memory and recomputes only lost partitions via lineage
   - Tradeoff: Spark must recompute (not re-read) lost wide-dependency outputs

5. **How do you decide the number of partitions?**
   - Target ~100–200 MB per partition for scan-bound work
   - Enough partitions for parallelism (≥ 2–3× total cores) but not so many that scheduling overhead dominates
   - Let AQE coalesce post-shuffle; repartition before a skewed wide op

6. **When is a broadcast join unsafe?**
   - When the "small" side isn't actually small — broadcasting a multi-GB table OOMs every executor
   - Use a size threshold; fall back to sort-merge join; AQE can switch at runtime after filtering

7. **How would you make a long multi-stage pipeline resilient?**
   - Checkpoint after expensive wide stages to truncate lineage
   - Persist hot intermediate datasets; use the external shuffle service so executor loss doesn't drop shuffle blocks
   - Idempotent, atomic output commit (write to temp, atomic rename/commit)

---

## 2. Stream Processing Engine

**Problem:** Design a stateful stream processing engine like Apache Flink or Spark Structured Streaming that computes continuous results over unbounded data.

### Requirements

**Functional:**
- Continuous transformations over unbounded streams (map, filter, join, aggregate)
- **Windowed** aggregations (tumbling, sliding, session)
- **Event-time** processing with correct handling of out-of-order/late data
- Large keyed **state** (e.g., per-user counters, joins)
- **Exactly-once** end-to-end semantics

**Non-Functional:**
- Low latency (sub-second to seconds)
- High throughput (millions of events/sec)
- Fault tolerant with fast recovery; no data loss or duplication on failure
- Rescalable (change parallelism without losing state)

### Capacity Estimation

```
Ingest: 1M events/sec × 1 KB = 1 GB/s
Keyed state: 50M keys × 200 bytes = 10 GB state -> spill to RocksDB on disk
Checkpoint interval: 30s -> incremental checkpoint of changed state to object store
Parallelism: 1M eps / ~50K eps per task = ~20 parallel subtasks per operator
```

### High-Level Architecture

```
  Sources (Kafka)                                       Sinks
      |                                                   ^
      v                                                   |
+-----------+    +----------------------------------+   +-+--------+
| Source    |--> | Operator subtasks (parallel)     |-->| Sink     |
| (offsets) |    |  keyBy -> window -> aggregate     |   | (2PC /   |
+-----------+    |  +--------------------------+     |   | idemp.)  |
                 |  | keyed state (RocksDB)     |     |   +----------+
                 |  +--------------------------+     |
                 +----------------+-----------------+
                                  | periodic barriers
                       +----------v-----------+
                       | Checkpoint Coordinator|
                       +----------+-----------+
                                  v
                       +----------------------+
                       |  Durable State Store |  (S3/HDFS: checkpoints,
                       |  (snapshots)         |   savepoints)
                       +----------------------+
```

### Event Time and Watermarks

Records carry an **event timestamp** (when it happened), not processing time
(when it arrives). A **watermark** of time `T` asserts "no more events with
timestamp ≤ T are expected," letting windows ≤ T fire while bounding wait for
stragglers.

```python
class WatermarkGenerator:
    def __init__(self, max_out_of_orderness_ms: int):
        self.max_lateness = max_out_of_orderness_ms
        self.max_seen_ts = 0

    def on_event(self, event):
        self.max_seen_ts = max(self.max_seen_ts, event.timestamp)

    def current_watermark(self) -> int:
        # Allow events up to max_lateness behind the newest seen event
        return self.max_seen_ts - self.max_lateness

class EventTimeWindow:
    def on_watermark(self, watermark: int):
        # Fire and emit any window whose end is now <= watermark
        for w in list(self.windows):
            if w.end <= watermark:
                self.emit(w, self.state.get(w))
                self.state.clear(w)
        # Events later than the watermark are "late": drop, side-output,
        # or allow within an explicit allowed-lateness grace period.
```

### State and Checkpointing (exactly-once)

Operators keep **keyed state** in an embedded store (e.g., RocksDB) so state can
exceed memory. Consistency comes from **distributed snapshots** (Chandy-Lamport):
the coordinator injects **barriers** into the stream; each operator snapshots its
state when barriers from all inputs align, producing a globally consistent
checkpoint without stopping the world.

```python
class CheckpointBarrierHandler:
    def on_barrier(self, barrier_id, input_channel):
        self.received[barrier_id].add(input_channel)
        # Align: wait for the barrier on every input before snapshotting
        if self.received[barrier_id] == self.all_inputs:
            snapshot = self.state.snapshot()          # async, incremental
            self.coordinator.ack(barrier_id, snapshot_handle=snapshot)
            self.forward(barrier_id)                  # pass barrier downstream
```

**End-to-end exactly-once** requires the sink to participate:
- **Idempotent sink** (keyed upsert): replays overwrite, so duplicates are harmless.
- **Transactional sink (two-phase commit)**: pre-commit on checkpoint, commit only
  after the checkpoint completes; on recovery, replay from the last checkpoint and
  abort uncommitted transactions. Source offsets are part of the checkpoint, so
  replay resumes exactly where the snapshot was taken.

### Micro-batch vs Continuous

| Aspect | Micro-batch (Structured Streaming) | Continuous (Flink) |
|--------|-----------------------------------|--------------------|
| Latency | 100ms–seconds (batch interval) | Milliseconds |
| Throughput | Very high (batch efficiency) | High |
| Model | Repeated small batch jobs | Long-lived operator graph |
| State | Versioned state store per batch | Embedded RocksDB + checkpoints |
| Backpressure | Natural (next batch waits) | Credit-based flow control |
| Best for | High-throughput ETL, exactly-once to lakehouse | Low-latency event-driven apps |

### Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Time semantics | Event time + watermarks | Correct results despite out-of-order arrival |
| State store | Embedded RocksDB | State larger than memory; fast local access |
| Consistency | Asynchronous barrier checkpoints | Exactly-once without stopping the stream |
| Sink | 2PC or idempotent upsert | Extend exactly-once past the engine |
| Rescaling | Keyed state by key-group | Redistribute state when parallelism changes |

### Failure Scenarios & Mitigation

| Failure Mode | Impact | Detection | Mitigation |
|--------------|--------|-----------|------------|
| Task failure | State/processing lost | Heartbeat | Restore from last checkpoint, replay from source offsets |
| Late data | Wrong window result | Watermark vs event ts | Allowed lateness + side output for very late |
| State too large | Slow checkpoints / OOM | State size, checkpoint duration | Incremental checkpoints, TTL on state, key pruning |
| Backpressure | Latency grows, lag rises | Consumer lag, buffer usage | Scale up, flow control, shed/throttle source |
| Slow checkpoint | Recovery far behind | Checkpoint duration trend | Incremental + async snapshots; tune interval |
| Non-idempotent sink | Duplicates on replay | Downstream dupes | Switch to upsert or transactional sink |

### Monitoring & Observability

**Key Metrics:**
- **Consumer lag** (records behind source head) — the headline health metric
- **End-to-end latency** (event time → emit)
- **Watermark lag** (processing-time − watermark)
- **Checkpoint duration & size**; checkpoint failure rate
- **State size per operator**; RocksDB compaction time
- **Records in/out, backpressure ratio**

**Alerting:**
- Consumer lag growing monotonically (can't keep up)
- Checkpoint duration approaching the interval
- Checkpoint failures > 0
- Watermark stalled (a source partition went idle)

### Security Considerations

- **Source/sink auth**: mTLS/SASL to Kafka; scoped credentials per job
- **State encryption**: checkpoints/savepoints on object store encrypted at rest
- **PII in state**: TTL and right-to-erasure handling for long-lived keyed state
- **Multi-tenancy**: per-job slots/containers; quota on state size

### Interview Deep-Dive Questions

4. **How does exactly-once survive a sink that isn't transactional?**
   - Make writes idempotent: deterministic keys + upsert, so replayed records overwrite
   - Or dedupe downstream by an event ID within a window
   - True 2PC requires sink support (e.g., Kafka transactions, JDBC XA)

5. **Watermark too aggressive vs too lax — what breaks?**
   - Too aggressive (small lateness): windows fire early, late events dropped → undercount
   - Too lax (large lateness): windows hold open, latency and state grow
   - Tune from the observed lateness distribution; use allowed-lateness as a safety valve

6. **How do you rescale a stateful job without losing state?**
   - Partition keyed state into fixed **key groups**; reassign groups to new subtasks
   - Restore from a **savepoint** (manual, portable checkpoint) at the new parallelism
   - Max parallelism is fixed at job start by the number of key groups

7. **How to join two streams with different delays?**
   - Interval/windowed join: buffer each side within a time bound keyed on the join key
   - State retention = the join window; evict on watermark to bound memory
   - For stream-to-static enrichment, broadcast the dimension or use async lookup with cache

---

## 3. Lakehouse Table Format

**Problem:** Design an open table format like Apache Iceberg, Delta Lake, or Hudi that brings ACID transactions, schema evolution, and time travel to files on object storage.

### Requirements

**Functional:**
- ACID commits over files in object storage (S3/GCS/ADLS)
- Concurrent readers and writers with **snapshot isolation**
- Schema evolution (add/drop/rename/reorder columns) without rewriting data
- **Time travel** (query/rollback to a past snapshot)
- Efficient reads via partition + file **pruning**; row-level updates/deletes/merge

**Non-Functional:**
- Scale to petabytes / millions of files per table
- Reads don't require a always-on metastore for data correctness
- Cheap metadata operations (planning shouldn't list all files)
- Works on eventually-consistent, rename-expensive object stores

### Why Object Storage Makes This Hard

Object stores have **no atomic multi-file rename** and **no directory locks**.
A "table = directory of Parquet files" (Hive) has no way to atomically add a set
of files, so readers see partial writes and there's no isolation. The format's
job is to provide atomicity via a **single atomic metadata pointer swap** and to
make planning cheap with a **metadata tree** instead of file listing.

### Metadata Tree (Iceberg model)

```
        catalog (table -> current metadata pointer)   <-- atomic swap here
                         |
                  metadata.json  (schema, partition spec, snapshot list)
                         |
                  manifest list   (one per snapshot; points to manifests
                         |          with partition range + row-count stats)
              +----------+----------+
              |                     |
          manifest A            manifest B   (lists data files + per-column
              |                     |          min/max/null stats)
        +-----+-----+         +-----+-----+
     data.parquet  data    data.parquet  data   (immutable column files)
```

A **commit** writes new data files, new manifests, a new manifest list, and a
new `metadata.json`, then **atomically swaps** the table's pointer to it. Readers
resolve the current pointer once and get a consistent **snapshot** of the whole
table.

### Atomic Commit with Optimistic Concurrency

```python
class TableCommit:
    def commit(self, new_data_files, base_snapshot_id):
        for attempt in range(MAX_RETRIES):
            current = self.catalog.load_current_metadata(self.table)

            # Conflict check: did someone else commit since we planned?
            if self._conflicts(current.snapshot_id, base_snapshot_id, new_data_files):
                # e.g., they deleted files we're updating -> re-plan against `current`
                base_snapshot_id = current.snapshot_id
                continue

            new_snapshot = self._build_snapshot(current, new_data_files)
            new_metadata = current.with_snapshot(new_snapshot)

            # Atomic compare-and-swap of the table pointer (catalog enforces it)
            if self.catalog.commit(self.table,
                                    expected=current.metadata_location,
                                    new=self._write(new_metadata)):
                return new_snapshot.id
        raise CommitFailedException("too many concurrent writers")
```

The atomicity primitive lives in the **catalog**: a transactional metastore
(JDBC, Hive, Glue, Unity) does a conditional update, or DynamoDB/`PutObject`
with a precondition. Append-only commits rarely conflict; overwrite/delete
commits validate that the files they touch still exist.

### Read Path: Pruning

```python
class ScanPlanner:
    def plan_files(self, snapshot, predicate):
        files = []
        # 1. Manifest-list pruning: skip whole manifests by partition range
        for manifest in snapshot.manifests:
            if not self._partition_overlaps(manifest.partition_summary, predicate):
                continue
            # 2. Manifest entry pruning: skip data files by per-column min/max
            for entry in manifest.read_entries():
                if self._stats_overlap(entry.column_stats, predicate):
                    files.append(entry.data_file)
        return files   # only the surviving files are read
```

Pruning happens on **metadata**, not by listing the object store — planning a
query touches kilobytes of manifests, not millions of object keys.

### Row-Level Updates: Copy-on-Write vs Merge-on-Read

| Strategy | Write cost | Read cost | Best for |
|----------|-----------|-----------|----------|
| **Copy-on-write** | High (rewrite whole files touching changed rows) | Low (just read data) | Read-heavy, infrequent updates |
| **Merge-on-read** | Low (write delete files / delta logs) | Higher (merge deletes at read) | Write/update-heavy, streaming upserts |

### Format Comparison

| Aspect | Iceberg | Delta Lake | Hudi |
|--------|---------|-----------|------|
| Metadata | Manifest tree (snapshots) | `_delta_log` JSON + checkpoints | Timeline + file groups |
| Concurrency | Optimistic, catalog CAS | Optimistic, log-based | Optimistic; MOR/COW tables |
| Partitioning | Hidden partitioning (transforms) | Directory partitions | Directory partitions |
| Update model | COW + MOR (v2 deletes) | COW + deletion vectors | First-class MOR upserts |
| Engine support | Broad (Spark/Flink/Trino/…) | Spark-first, broadening | Spark/Flink |
| Niche | Vendor-neutral, large tables | Databricks ecosystem | Streaming upserts/CDC |

### Maintenance: Small Files & Snapshot Expiry

```python
# Compaction: rewrite many small files into fewer right-sized files
spark.sql("CALL catalog.system.rewrite_data_files("
          "table => 'db.t', options => map('target-file-size-bytes','536870912'))")

# Expire old snapshots so time-travel history and orphan files don't grow forever
spark.sql("CALL catalog.system.expire_snapshots('db.t', TIMESTAMP '2026-06-01')")
```

### Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Atomicity | Single metadata pointer swap | Object stores lack multi-file atomic rename |
| Planning | Metadata-tree pruning | Avoid listing millions of objects |
| Isolation | Immutable snapshots | Readers never see partial writes; free time travel |
| Concurrency | Optimistic + conflict check | Writers rarely conflict; no global lock |
| Data files | Immutable columnar (Parquet/ORC) | Append + new-snapshot model; cheap stats |

### Failure Scenarios & Mitigation

| Failure Mode | Impact | Detection | Mitigation |
|--------------|--------|-----------|------------|
| Concurrent commit conflict | Commit rejected | CAS failure | Retry: re-plan against latest snapshot |
| Orphan files (failed write) | Wasted storage | Orphan scan | Periodic orphan-file cleanup (older than retention) |
| Small-file explosion | Slow scans, planning bloat | Avg file size, file count | Scheduled compaction; right-size writes |
| Metadata bloat | Slow planning | Manifest count/size | Rewrite manifests; expire snapshots |
| Eventual-consistency read | Missing just-written file | Read-after-write gap | Reference files only via committed metadata, never list |
| Catalog unavailable | Can't commit/resolve pointer | Catalog health | HA catalog; reads can use a cached metadata location |

### Monitoring & Observability

**Key Metrics:**
- **Avg data file size** & **files per partition** (small-file health)
- **Manifest count / metadata size** (planning cost)
- **Commit retry / conflict rate** (write contention)
- **Snapshot count & oldest snapshot age** (history growth)
- **Scan files pruned %** (pruning effectiveness)

**Alerting:**
- Avg file size below threshold (compaction overdue)
- Commit conflict rate rising (writer contention)
- Metadata size growth outpacing data

### Security Considerations

- **Access control**: catalog-level table/column/row policies (Unity/Lake Formation); the format itself trusts the catalog
- **Encryption**: object-store SSE or per-file encryption keys (KMS)
- **Time-travel & GDPR**: erasure must rewrite affected files and expire old snapshots, or deleted data lingers in history
- **Credential scope**: writers need write to the data prefix + commit on the catalog only

### Interview Deep-Dive Questions

4. **How does time travel work and what limits it?**
   - Each snapshot is immutable and references a complete file set; query `AS OF` a snapshot/timestamp resolves that metadata
   - Limited by retention: `expire_snapshots` removes old snapshots and their now-unreferenced files
   - Holding long history blocks file cleanup (storage cost) and conflicts with erasure requirements

5. **Two writers update the same table concurrently — what happens?**
   - Both plan against snapshot N; first commits → N+1
   - Second's CAS fails; it re-reads N+1, re-validates its changes don't conflict (e.g., didn't delete files the other removed), and retries
   - Append-only writers almost never truly conflict; overwrites/deletes may

6. **Why is hidden partitioning better than Hive directory partitioning?**
   - Iceberg stores the partition transform in metadata (e.g., `days(ts)`); queries filtering on `ts` get pruned automatically — no `partition_col=` predicate required
   - Avoids the classic bug of forgetting the partition predicate and full-scanning
   - Partition spec can evolve without rewriting old data

7. **Copy-on-write vs merge-on-read — how do you choose?**
   - Streaming upserts / CDC sink → MOR (cheap writes; compact deletes periodically)
   - BI tables read far more than written → COW (clean reads, no merge cost)
   - Often: MOR for ingestion, scheduled compaction to COW-like read performance

---

## 4. OLAP Data Warehouse

**Problem:** Design a cloud data warehouse like Snowflake, BigQuery, or Redshift for fast analytical queries over large structured datasets.

### Requirements

**Functional:**
- SQL analytics (scans, joins, group-by, window functions) over TB–PB tables
- Load/ingest batch and streaming data
- Independent scaling of storage and compute; many concurrent workloads
- Result/metadata caching; materialized views

**Non-Functional:**
- Fast scans via columnar + pruning; seconds for interactive BI
- Elastic compute (scale up/out per workload, pay for use)
- High concurrency without workloads starving each other
- Durable storage, strong consistency for committed data

### High-Level Architecture (decoupled storage/compute)

```
            +------------------------------------------+
            |        Cloud Services Layer              |
            | (auth, SQL parse/optimize, txn mgr,      |
            |  metadata + statistics, result cache)    |
            +-------------------+----------------------+
                                |
     +--------------------------+--------------------------+
     |                          |                          |
+----v-----+              +-----v----+              +------v---+
| Virtual  |              | Virtual  |              | Virtual  |  <- elastic
| Warehouse|              | Warehouse|              | Warehouse|     compute
| (ETL)    |              | (BI)     |              | (ad-hoc) |     (MPP nodes)
+----+-----+              +-----+----+              +------+---+
     |  local SSD cache         |  local SSD cache         |
     +--------------------------+--------------------------+
                                |
                    +-----------v-----------+
                    | Object Store (S3/GCS) |  <- single source of truth
                    | columnar micro-partitions
                    +-----------------------+
```

Compute clusters are stateless and ephemeral; they cache hot micro-partitions on
local SSD but the durable copy lives in object storage. This is what lets two
warehouses read the same data without contention and scale independently.

### Columnar Storage + Pruning

Data is stored **columnar** in immutable blocks (Snowflake "micro-partitions",
~16 MB; BigQuery capacitor blocks). Each block carries per-column **min/max/zone
maps**, so the optimizer skips blocks that can't match a predicate without
reading them.

```python
class PartitionPruner:
    def prune(self, blocks, predicate):
        # e.g., predicate: order_date = '2026-06-21'
        surviving = []
        for block in blocks:
            stats = block.column_stats[predicate.column]
            if stats.min <= predicate.value <= stats.max:  # might contain
                surviving.append(block)
            # else: provably no match -> skip without reading
        return surviving
```

Columnar layout also enables **vectorized execution** and high compression
(run-length, dictionary), so scans are CPU- and IO-efficient.

### MPP Query Execution

```python
class MPPExecutor:
    def execute_join(self, left, right, join_key):
        # Cost-based choice of distribution strategy
        if self._size(right) < BROADCAST_THRESHOLD:
            # Broadcast small side to all nodes; no shuffle of the large side
            right_copy = self.broadcast(right)
            return self.map_nodes(lambda part: hash_join(part, right_copy, join_key))
        else:
            # Shuffle/redistribute BOTH sides by join_key so matching keys colocate
            left_r = self.redistribute(left, join_key)
            right_r = self.redistribute(right, join_key)
            return self.map_nodes(lambda l, r: hash_join(l, r, join_key),
                                  left_r, right_r)
```

### Caching Layers

| Cache | Scope | Hit avoids |
|-------|-------|-----------|
| **Result cache** | Identical query, unchanged data | Re-execution entirely (ms response) |
| **Metadata cache** | Pruning stats, table versions | Re-reading statistics |
| **Local SSD cache** | Per-warehouse hot micro-partitions | Object-store round trips |

### Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Architecture | Decouple storage & compute | Independent scaling; isolate workloads |
| Storage | Immutable columnar blocks + stats | Pruning, compression, vectorization |
| Concurrency | Multiple virtual warehouses | BI ≠ ETL; no resource contention |
| Joins | CBO: broadcast vs shuffle | Minimize network for the data shape |
| Consistency | Versioned metadata, atomic commit | Snapshot reads; no dirty reads |

### Comparison

| Aspect | Snowflake | BigQuery | Redshift (classic) |
|--------|-----------|----------|--------------------|
| Compute model | Virtual warehouses (provisioned) | Serverless slots | Provisioned cluster (RA3 decouples) |
| Storage | Object store + micro-partitions | Colossus columnar | Local (classic) / S3 (RA3) |
| Scaling | Resize / multi-cluster auto | Automatic slots | Resize nodes |
| Strength | Workload isolation, sharing | Fully serverless, big scans | Tight AWS integration |
| Pricing | Per-compute-second | Per-byte-scanned (or slots) | Per-node-hour |

### Failure Scenarios & Mitigation

| Failure Mode | Impact | Detection | Mitigation |
|--------------|--------|-----------|------------|
| Compute node failure | Query restarts | Node health | Retry on remaining nodes; stateless compute |
| Skewed join/group | Straggler node OOM | Per-node bytes | Better distribution key; CBO; salting |
| Concurrency spike | Queue / slow queries | Queue depth | Auto-suspend/resume; multi-cluster scale-out |
| Bad statistics | Wrong plan (e.g., shuffle huge table) | Plan vs actual rows | Refresh stats; runtime adaptive re-plan |
| Hot result cache miss after load | Latency spike | Cache hit ratio | Warmups; cache invalidation only on changed micro-partitions |
| Object store throttling | Scan slowdown | S3 5xx/latency | Local SSD cache, request spreading |

### Monitoring & Observability

**Key Metrics:**
- **Bytes scanned per query** (cost) and **partitions pruned %** (efficiency)
- **Queue time vs execution time** (concurrency pressure)
- **Spilling to local/remote** (memory pressure)
- **Result-cache hit ratio**
- **Warehouse utilization / credits consumed**

**Alerting:**
- Query queue time > threshold (under-provisioned)
- Repeated full scans of large tables (missing clustering/pruning)
- Spill-to-remote events (undersized warehouse)

### Security Considerations

- **RBAC** on databases/schemas/tables; column- and row-level policies; data masking
- **Encryption** at rest (per-table keys, KMS) and in transit; key rotation
- **Data sharing** without copies (governed shares) — scope reader access carefully
- **Audit**: query history, access logs; PII tagging and classification

### Interview Deep-Dive Questions

4. **Why decouple storage and compute?**
   - Independent scaling: petabytes of storage with right-sized compute per workload
   - Isolation: ETL and BI run on separate warehouses over the same data, no contention
   - Elasticity: spin compute up for a load, suspend when idle (pay per use)
   - Tradeoff: object-store latency, mitigated by SSD caching

5. **What makes pruning effective, and how can it fail?**
   - Effective when data is **clustered/sorted** on the filtered columns so block min/max ranges are narrow and non-overlapping
   - Fails when data is randomly ordered (every block's range spans all values → no skipping)
   - Fix: define clustering keys; periodic re-clustering

6. **How is high concurrency achieved without queries fighting?**
   - Separate virtual warehouses per workload class; auto-scale out (more clusters) under queueing
   - Result/metadata cache offloads repeated queries
   - Workload management: queues, priorities, per-query resource limits

7. **How would you ingest streaming data into a warehouse?**
   - Micro-batch loads (e.g., Snowpipe/streaming inserts) appending new micro-partitions
   - Stage to object storage, commit atomically into table metadata
   - Tradeoff: more frequent loads → more small partitions → schedule compaction/re-clustering

---

## 5. Change Data Capture (CDC) Pipeline

**Problem:** Design a pipeline that captures every row-level change from an operational database and delivers it to a data lake/warehouse and downstream consumers, in order and without loss.

### Requirements

**Functional:**
- Capture inserts/updates/deletes from source DBs (MySQL, Postgres) with low lag
- Initial **snapshot** (backfill existing rows) + ongoing **streaming** of changes
- Preserve per-row ordering; carry before/after images
- Handle source **schema changes**
- Deliver to multiple sinks (lakehouse, warehouse, search, cache)

**Non-Functional:**
- Low end-to-end latency (seconds)
- Exactly-once (or effectively-once) at the sink — no dup/lost changes
- Minimal load on the source database
- Resumable after failures without re-snapshotting

### Why Log-Based Capture

| Approach | Mechanism | Pros | Cons |
|----------|-----------|------|------|
| **Query-based** | Poll `WHERE updated_at > last` | Simple, no DB privileges | Misses deletes & intermediate states; load on source; clock issues |
| **Trigger-based** | DB triggers write to audit table | Captures all changes | Write amplification; invasive; slows source |
| **Log-based** | Read WAL/binlog/redo | All changes incl. deletes; no source query load; ordered | Needs replication privileges; per-DB connector |

Log-based wins: the replication log already records every committed change in
commit order, so capture is complete and cheap.

### High-Level Architecture

```
+-----------+   read WAL/binlog   +-------------+   topic per table   +--------+
|  Source   |-------------------> | CDC         |-------------------> | Kafka  |
|  DB (WAL) |   (replication      | Connector   |  key = primary key  | (log)  |
+-----------+    slot/binlog pos) | (Debezium)  |                     +---+----+
                                  +------+------+                         |
                                         | offsets (resume point)         |
                                  +------v------+              +----------v---------+
                                  | Offset/State|              |  Sink Connectors   |
                                  | Store       |              | (lakehouse upsert, |
                                  +-------------+              |  warehouse, search)|
                                                               +--------------------+
```

### Snapshot → Streaming Handoff (the hard part)

You must backfill existing rows **and** stream new changes without a gap or
overlap at the boundary.

```python
class CdcConnector:
    def start(self):
        # 1. Record the current log position BEFORE snapshotting
        log_pos = self.source.current_log_position()

        # 2. Consistent snapshot of existing rows (read at a consistent point)
        with self.source.consistent_read():
            for row in self.source.scan_all_tables():
                self.emit(ChangeEvent(op="r", after=row))  # "r" = snapshot read

        # 3. Stream the log starting from the position captured in step 1.
        #    Any change between snapshot and now is replayed; sinks upsert by
        #    primary key so re-applying a row already in the snapshot is a no-op.
        self.stream_from(log_pos)

    def stream_from(self, log_pos):
        for change in self.source.read_log(log_pos):
            self.emit(ChangeEvent(
                op=change.op,              # c=create, u=update, d=delete
                before=change.before,
                after=change.after,
                source_ts=change.commit_ts,
                pk=change.primary_key,
            ))
            self.commit_offset(change.log_pos)   # resume point on restart
```

### Ordering and Exactly-Once at the Sink

- **Ordering**: key each change by the row's **primary key** so all changes to a
  row land on the same partition and stay ordered. Global cross-row ordering is
  not preserved (and rarely needed).
- **Exactly-once**: changes are idempotent if applied as **upserts** keyed by PK
  with a monotonic version (log position / commit LSN). The sink keeps only the
  latest version per key; replays are no-ops.

```python
class IdempotentUpsertSink:
    def apply(self, change):
        existing_version = self.sink.get_version(change.pk)
        # Drop stale/duplicate replays (out-of-order or re-delivered)
        if existing_version is not None and change.log_pos <= existing_version:
            return
        if change.op == "d":
            self.sink.delete(change.pk, version=change.log_pos)   # tombstone
        else:
            self.sink.upsert(change.pk, change.after, version=change.log_pos)
```

### Application-Level CDC: the Outbox Pattern

When the change originates in a service (not just the DB), the **outbox pattern**
avoids the dual-write problem: write the business row and an `outbox` event in
**one local transaction**; a CDC connector tails the `outbox` table. Either both
commit or neither — no lost or phantom events.

### Schema Evolution

```python
def handle_schema_change(self, ddl_event):
    # Emit a schema-change marker so sinks can evolve their target
    self.schema_history.append(ddl_event)
    # Additive changes (new nullable column) -> sink adds column
    # Renames/drops -> reconcile via compatibility rules or manual review
    self.emit(SchemaChangeEvent(ddl=ddl_event, position=ddl_event.log_pos))
```

### Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Capture | Log-based (WAL/binlog) | Complete, ordered, low source load |
| Transport | Kafka, key=PK | Per-row ordering, replay, fan-out to sinks |
| Delivery | Idempotent upsert + version | Effectively-once despite at-least-once transport |
| Backfill | Snapshot + log-position handoff | No gap/overlap; consistent start |
| App changes | Outbox pattern | Atomic with business transaction |

### Failure Scenarios & Mitigation

| Failure Mode | Impact | Detection | Mitigation |
|--------------|--------|-----------|------------|
| Connector restart | Possible re-delivery | Offset gap | Resume from committed offset; idempotent sink |
| Source log retention exceeded | Connector can't resume | Position older than retained log | Increase log retention; alert on lag; re-snapshot if lost |
| Out-of-order delivery | Wrong final row state | Version regression | Version/LSN check; drop stale |
| Schema change | Sink write fails | DDL event / write errors | Schema registry + compatibility rules |
| Sink unavailable | Backlog grows | Sink lag | Buffer in Kafka (retention), backpressure |
| Large transaction | Burst of changes, latency spike | Throughput spike | Batch apply; rate limit; partition parallelism |

### Monitoring & Observability

**Key Metrics:**
- **Replication lag** (source commit time → captured) and **end-to-end lag** (→ sink applied)
- **Connector offset vs source log head** (resume safety margin)
- **Events/sec by op type** (c/u/d) and per table
- **Sink apply errors / dropped stale events**
- **Snapshot progress** during backfill

**Alerting:**
- End-to-end lag exceeds SLA
- Connector offset approaching the source log retention edge (risk of unrecoverable gap)
- Snapshot running unexpectedly (a connector lost its position)

### Security Considerations

- **Source privileges**: replication-only account; no broad read/write
- **PII in change events**: mask/encrypt sensitive columns in the pipeline; respect erasure (tombstone + downstream delete)
- **Transport**: encrypted, authenticated Kafka; per-sink credentials
- **Audit**: change events are a sensitive audit trail — restrict topic access

### Interview Deep-Dive Questions

4. **How do you guarantee no gap between snapshot and streaming?**
   - Capture the log position *before* the snapshot; stream from that position afterward
   - Overlap (changes during snapshot) is safe because sinks upsert by PK with versioning — re-applying is idempotent
   - Never the reverse order (snapshot then read position), which would lose changes made during the snapshot

5. **Why key by primary key in Kafka?**
   - Guarantees all changes for a row are in one partition, preserving per-row order
   - Enables log compaction (keep only latest per key) for a compacted "current state" topic
   - Cross-row global order is sacrificed (acceptable; consumers care about per-entity order)

6. **How do deletes propagate, especially to a lakehouse?**
   - Emit a delete event (`op=d`) with the PK; sink writes a tombstone / MOR delete file
   - For compacted topics, a null value tombstones the key
   - Periodic compaction reconciles deletes into the table

7. **How would you handle a 500 GB initial snapshot without locking the source?**
   - Chunked, incremental snapshot (read PK ranges) interleaved with streaming (Debezium incremental snapshots)
   - Read from a replica to offload the primary
   - Watermark-based dedup so concurrent live changes during a chunk are handled correctly

---

## 6. Real-Time OLAP Serving

**Problem:** Design a real-time analytics store like Apache Druid, Apache Pinot, or ClickHouse that answers sub-second aggregation queries over freshly ingested event data.

### Requirements

**Functional:**
- Ingest streaming events (from Kafka) and make them queryable within seconds
- Sub-second slice-and-dice aggregations (group-by, filter, top-N) over billions of rows
- Combine **real-time** (just-ingested) and **historical** (batch) data in one query
- Time-range queries; high-cardinality dimensions

**Non-Functional:**
- Query latency P95 < 1s at high concurrency (dashboards)
- High ingestion throughput (millions of events/sec)
- Horizontally scalable storage and query
- Tolerate node failures without query gaps

### High-Level Architecture (Druid-style)

```
  Kafka ---> +------------------+        +------------------+
             | Real-time/Ingest |  hand  | Deep Storage     |
             | nodes (recent    | off    | (S3/HDFS:        |
             | data, mutable)   +------->|  immutable       |
             +--------+---------+        |  segments)       |
                      |                  +---------+--------+
   query (recent)     |                            | load
                      |                  +---------v--------+
             +--------v---------+         | Historical nodes |
             |     Broker       |<------->| (immutable       |
             | (scatter-gather, |  query  |  segments, cached|
             |  merge results)  | (old)   |  on local disk)  |
             +--------+---------+         +------------------+
                      ^                   Coordinator: assigns segments,
                 client query             balances, manages tiers
```

A query is **scatter-gathered** by the broker across whichever real-time and
historical nodes hold segments in the time range, then results are merged. This
is a serving-layer realization of the lambda idea: fresh data from ingest nodes,
durable history from historical nodes, unified at query time.

### Segments and Columnar Indexing

Data is partitioned by time into immutable **segments**; within a segment,
columns are stored separately with **dictionary encoding** + **bitmap (inverted)
indexes** on dimensions, so filters become fast bitmap intersections.

```python
class Segment:
    """Immutable, time-bounded, columnar block (e.g., 1 hour, ~5M rows)."""
    def __init__(self):
        self.interval = None                 # [start, end)
        self.dimensions = {}                  # col -> dictionary-encoded values
        self.bitmap_index = {}               # (col, value) -> roaring bitmap of rows
        self.metrics = {}                     # pre-aggregated measures

    def filter(self, predicate) -> "RoaringBitmap":
        # AND/OR of per-value bitmaps -> matching row set, no row scan
        result = None
        for col, val in predicate.equals:
            bm = self.bitmap_index.get((col, val), EMPTY_BITMAP)
            result = bm if result is None else result & bm
        return result
```

### Rollup / Pre-Aggregation at Ingest

Optional **rollup** aggregates rows sharing the same dimensions + time bucket at
ingest time, trading raw-row fidelity for massive size and query-speed gains.

```python
class IngestRollup:
    def __init__(self, granularity="minute"):
        self.granularity = granularity
        self.buffer = {}   # (truncated_ts, dims...) -> aggregated metrics

    def add(self, event):
        bucket = self.truncate(event.timestamp, self.granularity)
        key = (bucket,) + tuple(event[d] for d in self.dimension_cols)
        agg = self.buffer.setdefault(key, {"count": 0, "sum": 0})
        agg["count"] += 1
        agg["sum"] += event.value
        # 1000 raw events with the same key -> 1 stored row
```

For high-cardinality distinct counts, store **sketches** (HyperLogLog, Theta) so
`COUNT(DISTINCT)` is approximate but cheap and mergeable across segments.

### Comparison

| Aspect | Druid | Pinot | ClickHouse |
|--------|-------|-------|-----------|
| Architecture | Many roles (broker/historical/realtime/coordinator) | Broker/server/controller | Shared-nothing shards/replicas |
| Ingestion | Native Kafka, rollup | Native Kafka, star-tree index | Inserts, materialized views |
| Strength | Time-series rollup, tiering | Low-latency, star-tree pre-agg | Raw-event SQL, flexibility |
| Updates | Append (segment replace) | Append (upsert tables) | Mutations (heavy), MergeTree |
| Best for | Operational dashboards | User-facing analytics | Ad-hoc + reporting on raw data |

### Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Storage unit | Immutable time segments | Easy replication, tiering, replacement |
| Indexing | Columnar + bitmap | Filters = bitmap ops, not scans |
| Freshness | Real-time + historical merge | Seconds-fresh without sacrificing durability |
| Size/speed | Rollup + sketches | Pre-aggregate; approximate high-cardinality |
| Durability | Deep storage as source of truth | Nodes are caches; rebuildable |

### Failure Scenarios & Mitigation

| Failure Mode | Impact | Detection | Mitigation |
|--------------|--------|-----------|------------|
| Historical node loss | Segments unavailable | Health, segment availability | Replicas (RF≥2); coordinator reloads from deep storage |
| Real-time node loss | Recent data gap | Ingest lag | Replica ingest tasks; replay from Kafka offsets |
| Hot segment (viral time range) | Skewed query load | Per-segment QPS | More replicas of hot segments; broker caching |
| High cardinality blowup | Memory/latency spike | Dimension cardinality | Sketches; drop/limit dimensions; pre-agg |
| Slow handoff | Recent data stuck on real-time tier | Handoff duration | Tune segment granularity; scale ingest |
| Broker overload | Query queueing | Broker CPU/queue | Scale brokers; result cache; tiered routing |

### Monitoring & Observability

**Key Metrics:**
- **Query latency P95/P99** and **QPS**
- **Ingestion lag** (Kafka offset → queryable)
- **Segment count / size / availability** (under-replicated segments)
- **Rows scanned vs returned** (index effectiveness)
- **Cache hit ratio** (broker/segment)

**Alerting:**
- Under-replicated segments > 0
- Ingestion lag exceeds freshness SLA
- Query P99 over threshold
- Cardinality of a dimension spiking

### Security Considerations

- **Access control**: table/column policies; row-level filters per tenant
- **Multi-tenancy**: isolate query pools; per-tenant rate limits to stop noisy-neighbor
- **Encryption**: deep storage + transit
- **PII**: avoid storing raw PII in dimensions; hash/limit; retention via segment expiry

### Interview Deep-Dive Questions

4. **How do real-time and historical data combine in one query?**
   - The broker fans out to real-time nodes (recent, mutable segments) and historical nodes (sealed segments) covering the time range
   - Each returns partial aggregates; broker merges
   - Once a real-time segment is handed off to deep storage and loaded by historicals, the broker stops querying the real-time copy

5. **When does rollup hurt?**
   - When you need raw per-event detail (rollup is lossy) or high-cardinality dimensions where rows rarely share a key (little reduction)
   - Mitigate: store raw for recent/hot tables, rolled-up for older; or dual tables

6. **How to keep P99 low under dashboard concurrency?**
   - Pre-aggregation + bitmap indexes reduce per-query work
   - Replicate hot segments; cache broker results; tier cold data to cheaper/slower nodes
   - Approximate queries (sketches) for distinct counts and quantiles

7. **How would you handle late-arriving events?**
   - Re-ingest into the affected time segment and replace it (segments are immutable; you swap a new version)
   - Coordinator atomically promotes the new segment version
   - For append-only stores, a compaction/merge task rebuilds the segment

---

## 7. Metrics / Time-Series Database

**Problem:** Design a monitoring metrics system like Prometheus (with long-term storage) that ingests, stores, and queries high-volume time-series data.

### Requirements

**Functional:**
- Collect time-series (metric name + label set → timestamped values) from many targets
- Query with aggregation over time and labels (rate, sum, histogram quantiles)
- Alerting rules and recording (pre-computed) rules
- Configurable retention; long-term storage and downsampling

**Non-Functional:**
- Ingest millions of samples/sec
- Low-latency queries for dashboards/alerts (seconds)
- Storage-efficient (years of data); high compression
- Resilient to target churn and cardinality spikes

### Capacity Estimation

```
Targets: 10,000 hosts × 1,000 series each = 10M active series
Scrape interval: 15s -> 10M / 15 ≈ 670K samples/sec
Raw sample on disk: ~1–2 bytes (after Gorilla compression) vs 16 bytes uncompressed
  10M series × (4 bytes/min after compression) ... ~ a few hundred GB/year compressed
Cardinality is the killer: a single label with 1M values (e.g., user_id) explodes series count.
```

### Pull vs Push

| Model | Mechanism | Pros | Cons |
|-------|-----------|------|------|
| **Pull (Prometheus)** | Server scrapes `/metrics` of targets | Server controls rate; easy target health (up/down); no client buffering | Needs service discovery; hard for short-lived/firewalled jobs |
| **Push** | Clients send to a gateway | Works for batch/serverless/ephemeral | Server can't detect target down; client must handle backpressure |

Most systems are pull-first with a **push gateway** for short-lived jobs.

### Data Model and Cardinality

```
http_requests_total{method="GET", status="200", instance="host1"}  -> series A
http_requests_total{method="POST", status="500", instance="host1"} -> series B
```

A **series** is uniquely identified by its metric name + the full label set. The
**central scaling problem is cardinality**: total series ≈ product of label
value counts. Putting an unbounded value (user ID, request ID, full URL) in a
label multiplies series and can OOM the database.

```python
class CardinalityGuard:
    def admit(self, series_labels) -> bool:
        # Reject/limit ingestion of metrics whose label sets explode cardinality
        metric = series_labels["__name__"]
        if self.active_series_for(metric) > self.limit(metric):
            self.metrics.increment("dropped_high_cardinality", metric=metric)
            return False
        return True
```

### TSDB Storage (Gorilla-style compression)

In-memory **head block** absorbs recent writes (backed by a WAL for crash
recovery); periodically it's flushed to immutable on-disk **blocks** (e.g.,
2-hour blocks) with an index.

```python
class GorillaChunk:
    """Compress (timestamp, value) pairs: delta-of-delta on time, XOR on value."""
    def append(self, ts: int, value: float):
        if self.count == 0:
            self._write_full(ts, value)
        else:
            # Timestamps are near-regular (15s) -> delta-of-delta ~ 0 bits
            dod = (ts - self.last_ts) - (self.last_ts - self.prev_ts)
            self._write_varint(dod)
            # Values change little -> XOR with previous, store only changed bits
            xored = float_bits(value) ^ float_bits(self.last_value)
            self._write_xor(xored)
        self.prev_ts, self.last_ts, self.last_value = self.last_ts, ts, value
        self.count += 1
```

This gets steady metrics down to ~1–2 bytes per sample (vs 16 raw).

### Retention, Downsampling, Long-Term Storage

```
[ recent: full resolution, local TSDB, 15d ]
        | remote write
[ long-term: Thanos/Cortex/Mimir over object storage ]
        | compaction + downsampling
[ 5m-resolution blocks (90d) ] -> [ 1h-resolution blocks (years) ]
```

Queries transparently route to the resolution that satisfies the time range;
old data is downsampled so dashboards over a year don't scan raw samples.

### Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Collection | Pull + push gateway | Target health, controlled rate; cover ephemeral jobs |
| Identity | metric + label set | Flexible dimensional queries |
| Storage | Head block + WAL → immutable blocks | Fast writes, crash-safe, compressible |
| Compression | Delta-of-delta + XOR (Gorilla) | ~10× smaller for regular series |
| Long-term | Remote write + downsampling tiers | Bounded cost over years |

### Failure Scenarios & Mitigation

| Failure Mode | Impact | Detection | Mitigation |
|--------------|--------|-----------|------------|
| Cardinality explosion | OOM, ingestion stalls | Active-series growth | Per-metric limits, relabel/drop high-card labels |
| Scrape gap (target down) | Missing samples | `up == 0` | Staleness handling; alert on target down |
| TSDB node loss | Recent data gap | Health checks | HA pairs (dual scrape) or replicated remote write |
| Slow/expensive query | Dashboard timeout, server load | Query duration | Recording rules pre-compute; limit range/step |
| Disk full | Ingestion stops | Disk usage | Retention enforcement; remote write offload |
| Clock skew | Out-of-order samples | Sample timestamp checks | NTP; reject far-future samples |

### Monitoring & Observability

**Key Metrics (meta-monitoring):**
- **Samples ingested/sec** and **active series count** (cardinality)
- **Scrape duration / failures** per target
- **Query duration P99** and **query concurrency**
- **WAL replay time** (recovery cost)
- **Remote-write queue/backlog** (long-term offload health)

**Alerting:**
- Active series growth rate spiking (incoming cardinality bomb)
- Ingestion or scrape failure rate rising
- Remote-write backlog growing (long-term store falling behind)

### Security Considerations

- **Endpoint exposure**: `/metrics` may leak internals — auth/network-restrict scrape targets
- **Multi-tenancy**: per-tenant series limits and isolation (Cortex/Mimir tenant headers)
- **Sensitive labels**: never put secrets/PII in labels (they're stored and queryable)
- **Alerting integrity**: protect alertmanager config and routing (paging is security-relevant)

### Interview Deep-Dive Questions

4. **Why is cardinality the central scaling concern?**
   - Storage, memory (active-series index), and query cost all scale with the number of series, not just sample volume
   - One bad label (user_id, request_id, raw path) turns 1 series into millions
   - Mitigate: bound label values, relabel/drop at scrape, enforce per-metric limits

5. **Pull vs push — which and why?**
   - Pull by default: server controls scrape rate, gets free up/down health, no client-side buffering
   - Push gateway for short-lived/serverless jobs that die before a scrape
   - Push-only systems can't tell "no data" from "target down"

6. **How does Gorilla compression achieve ~10×?**
   - Timestamps are near-regular → delta-of-delta is usually 0 (a single bit)
   - Values change slowly → XOR with previous stores only the differing bits
   - Works because monitoring data is highly regular and slowly-varying

7. **How would you query a year of data without scanning raw samples?**
   - Downsample into coarser-resolution blocks (5m, 1h) during compaction
   - Route queries to the lowest resolution that satisfies the step/range
   - Recording rules pre-compute expensive expressions; long-term store (Thanos/Mimir) serves the history

---

## 8. Distributed Query Engine

**Problem:** Design an interactive distributed SQL engine like Trino (Presto) that queries data in place across many sources without owning storage.

### Requirements

**Functional:**
- ANSI SQL over external sources (lakehouse, warehouse, RDBMS, Kafka) — **federation**
- Joins, aggregations, window functions across large datasets
- **No data ingestion** — query data where it lives
- Cross-source joins in a single query

**Non-Functional:**
- Interactive latency (seconds) for ad-hoc analytics
- High concurrency; scale workers horizontally
- Push work down to sources to minimize data movement
- Memory-aware execution (large joins/aggregations)

### High-Level Architecture

```
                 SQL
                  |
        +---------v----------+
        |    Coordinator     |  parse -> analyze -> plan -> optimize ->
        |  (planner, sched)  |  distributed stages; schedules splits;
        +---------+----------+  tracks query state
                  | distributed plan + splits
     +------------+------------+-----------------+
     |            |            |                 |
+----v---+   +----v---+   +----v---+   pipelined exchange (in memory,
| Worker |<->| Worker |<->| Worker |   streamed between stages — not
+----+---+   +----+---+   +----+---+   materialized to disk)
     |            |            |
+----v------------v------------v----+
|        Connectors / sources       |  pushdown: filters, projections,
| (Iceberg, Hive, MySQL, Kafka, S3) |  partial aggregates, limits
+-----------------------------------+
```

### Connectors and Pushdown

A **connector** abstracts a source: it exposes metadata, lists **splits** (units
of parallel read), and accepts **pushed-down** operations so less data crosses
the network.

```python
class Connector:
    def get_splits(self, table, predicate):
        # A split = a parallelizable chunk (a file, a partition, a key range)
        # Predicate is pushed in so the connector can prune splits up front
        return self.metadata.plan_splits(table, predicate)

    def supports_pushdown(self, operation):
        # Push filters/projections/partial-aggregates to the source when it can
        # do them (e.g., a SQL DB runs the WHERE; object store prunes by stats)
        return operation in self.capabilities

class PushdownOptimizer:
    def optimize(self, plan):
        # Move filters below the table scan into the connector
        for scan in plan.find_table_scans():
            pushable = self._extract_pushable_filters(scan.parent_filters)
            if scan.connector.supports_pushdown("filter"):
                scan.pushed_predicate = pushable
                self._remove_redundant_filter(scan, pushable)
            # Projection pushdown: read only needed columns (columnar sources)
            scan.projected_columns = self._needed_columns(scan)
        return plan
```

### Pipelined (Streaming) Execution

Unlike a batch engine that materializes each stage to disk, the engine streams
data between stages through **in-memory exchanges**: as soon as a stage produces
rows, downstream stages consume them. This is why it's fast and interactive —
and why it's **memory-bound**: a query that needs more memory than the cluster
has historically fails rather than spilling (modern versions add spill).

```python
class StageExecutor:
    def run(self, stage):
        # Splits scheduled across workers; results stream upward via exchange
        for split in stage.splits:
            self.assign_to_worker(split)
        # Exchange buffers shuffle data between stages in memory (back-pressured),
        # rather than writing intermediate results to durable storage
        return self.exchange.stream(stage.output_partitioning)
```

### Joins: Broadcast vs Partitioned

```python
class JoinPlanner:
    def choose_strategy(self, left, right):
        if self.estimate_size(right) < self.broadcast_threshold:
            return BroadcastJoin(build=right)   # ship small side to all workers
        # Otherwise repartition both sides by join key (network shuffle)
        return PartitionedJoin(key=self.join_key)
```

### Trino vs Other Engines

| Aspect | Trino / Presto | Spark SQL | Warehouse-native (Snowflake) |
|--------|----------------|-----------|------------------------------|
| Owns storage | No (federated) | No | Yes |
| Execution | Pipelined, in-memory | Stage-materialized, fault-tolerant | MPP over own storage |
| Fault tolerance | Query fails & retries (historically) | Task-level recovery (lineage) | Managed retries |
| Latency | Low (interactive) | Higher (batch) | Low |
| Best for | Ad-hoc, cross-source federation | Large fault-tolerant ETL | Curated BI on warehouse data |

### Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Storage | None — query in place | Federation; no ingestion/ETL latency |
| Execution | Pipelined in-memory exchange | Low latency for interactive queries |
| Source access | Connector + pushdown | Move predicates to data, minimize transfer |
| Joins | CBO broadcast vs partitioned | Avoid shuffling a large side unnecessarily |
| Fault model | Fail-fast (historically) | Simplicity/latency over long-job resilience |

### Failure Scenarios & Mitigation

| Failure Mode | Impact | Detection | Mitigation |
|--------------|--------|-----------|------------|
| Worker failure | Query fails (no mid-query recovery) | Heartbeat | Retry query; fault-tolerant execution mode (spool exchange) |
| Memory exceeded | Query killed | Memory tracking | Spill-to-disk; resource groups; smaller broadcast |
| Slow/limited source | Whole query stalls | Per-split latency | Concurrency limits per source; pushdown; caching |
| Coordinator overload | Planning bottleneck | Coordinator CPU/queue | Scale coordinator; query queueing; fault-tolerant dispatch |
| Skewed join key | Straggler worker | Per-worker rows | Better distribution; partitioned join tuning |
| Bad source stats | Wrong plan (huge shuffle) | Plan vs actual | CBO with fresh stats; adaptive re-planning |

### Monitoring & Observability

**Key Metrics:**
- **Query latency P50/P95** and **queued vs running** queries
- **Bytes read per source** & **bytes shuffled** (pushdown effectiveness)
- **Peak memory per query** and **spill bytes**
- **Worker CPU / split queue depth**
- **Failed-query rate** by reason (memory, source, worker)

**Alerting:**
- Query queue time over threshold (under-provisioned)
- Memory-kill rate rising (resource-group tuning needed)
- A connector's read latency dominating queries

### Security Considerations

- **Federated access control**: enforce per-source credentials and per-catalog/table/column policies at the coordinator
- **Credential passthrough vs service account**: decide whether queries run as the user (impersonation) or a shared principal
- **Network**: workers reach many sources — segment and least-privilege the egress
- **Audit**: log query text, accessed catalogs/tables, and result row counts

### Interview Deep-Dive Questions

4. **Why is Trino fast for interactive queries but risky for long ETL?**
   - Pipelined in-memory execution skips per-stage disk materialization → low latency
   - But no intermediate checkpoints → a worker loss historically fails the whole query
   - Long multi-hour jobs prefer Spark (lineage recovery) or fault-tolerant execution mode

5. **What does pushdown buy you, concretely?**
   - Filter pushdown: an RDBMS connector runs the `WHERE`, returning thousands not millions of rows
   - Projection pushdown: columnar sources read only needed columns
   - Partial aggregation pushdown: pre-aggregate at the source, shuffle less
   - Without it, the engine drags entire tables across the network

6. **Broadcast vs partitioned join — failure modes of each?**
   - Broadcast: cheap if the build side is truly small; OOMs every worker if it isn't (bad stats)
   - Partitioned: shuffles both sides (network cost) but scales to large×large; skewed keys create stragglers
   - CBO chooses from stats; refresh stats and cap broadcast size

7. **How would you add fault tolerance without losing interactivity?**
   - Fault-tolerant execution: spool exchange data to durable storage so failed tasks/stages retry instead of failing the query
   - Tradeoff: spooling adds latency and storage — enable for batch/large queries, keep pipelined mode for interactive ones
   - Resource groups isolate the two workload classes

---

## Summary: Key Tradeoffs

| System | Key Tradeoff |
|--------|-------------|
| Batch Processing Engine | Recompute (lineage) vs replicate intermediate data |
| Stream Processing Engine | Latency vs exactly-once / state-checkpoint overhead |
| Lakehouse Table Format | Read performance (COW) vs write performance (MOR) |
| OLAP Data Warehouse | Storage/compute decoupling: elasticity vs object-store latency |
| CDC Pipeline | Completeness/low-source-load (log-based) vs simplicity (query-based) |
| Real-Time OLAP Serving | Freshness/speed (rollup, sketches) vs raw-event fidelity |
| Metrics / Time-Series DB | Dimensional flexibility vs cardinality cost |
| Distributed Query Engine | Interactive latency (pipelined) vs long-job fault tolerance |

---

## Interview Tips

1. **Name the storage/compute boundary** — most modern data systems separate durable storage (object store) from elastic compute; state which is the source of truth.
2. **Identify the expensive operation** — shuffle (batch), checkpoint/state (streaming), file listing (lakehouse), cardinality (metrics), data movement (query engine). Optimize around it.
3. **Be explicit about consistency & delivery semantics** — exactly-once vs at-least-once, snapshot isolation, ordering guarantees, and *where* they're enforced.
4. **Show the failure-and-recovery story** — lineage recompute, checkpoint restore, offset replay, segment reload from deep storage.
5. **Quantify** — partitions, segment sizes, series cardinality, scan bytes. Data-infra interviews reward back-of-envelope numbers.
6. **Discuss maintenance** — compaction, snapshot expiry, downsampling, re-clustering. Data platforms live or die by background housekeeping.
