# Apache Iceberg & Spark Interview Questions

Comprehensive interview questions covering Apache Iceberg table format and Apache Spark processing engine.

---

## Table of Contents
1. [Apache Iceberg Fundamentals](#1-apache-iceberg-fundamentals)
2. [Iceberg Architecture & Internals](#2-iceberg-architecture--internals)
3. [Iceberg Operations & Features](#3-iceberg-operations--features)
4. [Apache Spark Core Concepts](#4-apache-spark-core-concepts)
5. [Spark Architecture & Execution](#5-spark-architecture--execution)
6. [Spark SQL & DataFrames](#6-spark-sql--dataframes)
7. [Spark Performance & Optimization](#7-spark-performance--optimization)
8. [Spark Streaming](#8-spark-streaming)
9. [Iceberg + Spark Integration](#9-iceberg--spark-integration)
10. [Scenario-Based Questions](#10-scenario-based-questions)

---

## 1. Apache Iceberg Fundamentals

### Q: What is Apache Iceberg and what problems does it solve?

**Answer:** Apache Iceberg is an open table format for large analytic datasets. It solves several problems with traditional Hive tables:

1. **Schema Evolution** - Add, drop, rename, or reorder columns without rewriting data
2. **Partition Evolution** - Change partitioning schemes without data migration
3. **Hidden Partitioning** - Users don't need to know partition layout for queries
4. **Time Travel** - Query historical snapshots of data
5. **ACID Transactions** - Reliable concurrent reads and writes
6. **Reliable Deletes** - Row-level deletes without full partition rewrites
7. **Engine Independence** - Works with Spark, Flink, Trino, Hive, etc.

### Q: How does Iceberg differ from Hive tables?

**Answer:**

| Feature | Hive | Iceberg |
|---------|------|---------|
| File Tracking | Directory listing | Manifest files |
| Schema Evolution | Limited (append only) | Full support |
| Partition Evolution | Requires data rewrite | In-place evolution |
| Partition Pruning | User must filter on partition columns | Automatic (hidden partitioning) |
| ACID | Limited (Hive 3+) | Full support |
| Time Travel | Not supported | Built-in |
| Concurrent Writes | Locking-based | Optimistic concurrency |
| Small Files | Manual compaction | Automatic compaction support |

### Q: What are the key components of an Iceberg table?

**Answer:**

1. **Catalog** - Tracks table metadata location (Hive Metastore, AWS Glue, Nessie, etc.)
2. **Metadata Files** - JSON files containing table schema, partition spec, snapshots
3. **Manifest Lists** - Track which manifest files belong to a snapshot
4. **Manifest Files** - Avro files listing data files with statistics
5. **Data Files** - Actual data in Parquet, ORC, or Avro format

```
catalog
  └── table metadata pointer
        └── metadata.json (schema, partitions, snapshots)
              └── manifest-list.avro (snapshot → manifests)
                    └── manifest.avro (file list + stats)
                          └── data files (parquet/orc/avro)
```

### Q: Explain Iceberg's hidden partitioning.

**Answer:** Hidden partitioning separates the physical partitioning from user queries:

```sql
-- Traditional Hive: Users must know and filter on partition columns
SELECT * FROM events WHERE year = 2024 AND month = 1 AND day = 15;

-- Iceberg: Users write natural queries, Iceberg handles partitioning
SELECT * FROM events WHERE event_time = '2024-01-15';
```

Iceberg uses **partition transforms** to derive partition values:
- `year(ts)`, `month(ts)`, `day(ts)`, `hour(ts)` - Time-based
- `bucket(n, col)` - Hash bucketing
- `truncate(width, col)` - String/numeric truncation
- `identity(col)` - Direct value (like Hive)

Benefits:
- No partition column in schema
- Queries are simpler and less error-prone
- Partition scheme can evolve without query changes

---

## 2. Iceberg Architecture & Internals

### Q: Explain Iceberg's snapshot isolation and how it enables time travel.

**Answer:** Each write operation creates a new **snapshot** - an immutable view of the table at a point in time.

```
Snapshot 1 (t=100) → manifest-list-1 → [manifest-a, manifest-b]
     ↓
Snapshot 2 (t=200) → manifest-list-2 → [manifest-a, manifest-c]  (manifest-b removed, c added)
     ↓
Snapshot 3 (t=300) → manifest-list-3 → [manifest-a, manifest-c, manifest-d]
```

**Time Travel Queries:**
```sql
-- Query by snapshot ID
SELECT * FROM table VERSION AS OF 12345;

-- Query by timestamp
SELECT * FROM table TIMESTAMP AS OF '2024-01-15 10:00:00';

-- Spark syntax
spark.read.option("snapshot-id", 12345).table("db.table")
spark.read.option("as-of-timestamp", "1705312800000").table("db.table")
```

Snapshots enable:
- Consistent reads during writes
- Audit and debugging
- Rollback to previous states
- Incremental processing

### Q: How does Iceberg handle schema evolution?

**Answer:** Iceberg tracks schema changes in metadata without rewriting data files:

**Supported Operations:**
- **Add column** - New columns read as NULL in old files
- **Drop column** - Column ignored in reads, data remains in files
- **Rename column** - Tracked by unique column ID, not name
- **Reorder columns** - Changes projection order
- **Widen type** - int → long, float → double
- **Make nullable** - required → optional

**Key Mechanism:** Each column has a unique ID that persists through renames:
```
Schema v1: {id: 1, name: "user_id"}, {id: 2, name: "email"}
Schema v2: {id: 1, name: "customer_id"}, {id: 2, name: "email"}  -- renamed
Schema v3: {id: 1, name: "customer_id"}, {id: 2, name: "email"}, {id: 3, name: "phone"}  -- added
```

Old data files still use column IDs, so reads work correctly regardless of current names.

### Q: Explain partition evolution in Iceberg.

**Answer:** Partition evolution allows changing partitioning without data migration:

```sql
-- Initial: partition by day
ALTER TABLE events SET PARTITION SPEC (day(event_time));

-- Later: add hour partitioning for recent data
ALTER TABLE events SET PARTITION SPEC (hour(event_time));
```

**How it works:**
1. Each data file is written with a specific partition spec
2. Manifest files track which spec was used for each file
3. Queries evaluate partition predicates against each spec
4. Old files remain partitioned by day, new files by hour

**Example Query Planning:**
```
Query: WHERE event_time BETWEEN '2024-01-15 10:00' AND '2024-01-15 11:00'

Old files (day partition): Scan day=2024-01-15, filter rows
New files (hour partition): Scan only hour=10 and hour=11
```

### Q: How does Iceberg handle concurrent writes?

**Answer:** Iceberg uses **optimistic concurrency control**:

1. Writer reads current metadata pointer from catalog
2. Writer creates new snapshot based on current state
3. Writer attempts atomic swap of metadata pointer
4. If another writer committed first, retry with conflict resolution

**Conflict Resolution:**
- **Append only** - Usually no conflict, both appends included
- **Delete/Update** - Check if affected files overlap
- **Rewrite** - May need to retry if same files modified

```
Writer A: Read meta v1, write files, commit meta v2 ✓
Writer B: Read meta v1, write files, commit meta v2 ✗ (conflict)
Writer B: Retry - read meta v2, check conflicts, commit meta v3 ✓
```

**Isolation Levels:**
- Serializable (default) - Detects all conflicts
- Snapshot - Allows concurrent modifications to different rows

### Q: What are manifest files and why are they important?

**Answer:** Manifest files are Avro files that list data files with rich metadata:

**Contents:**
```
- File path
- Partition values
- Record count
- File size
- Column-level statistics (min, max, null count, distinct count)
- Sort order
```

**Importance:**

1. **Query Planning** - Statistics enable predicate pushdown without opening files
   ```
   Query: WHERE amount > 1000
   Manifest shows: file1 max(amount)=500 → skip file1
   ```

2. **Efficient Listing** - No need for expensive directory listing
   ```
   Hive: LIST s3://bucket/table/year=2024/month=01/day=*/* (slow)
   Iceberg: Read manifest file (fast, cached)
   ```

3. **Incremental Processing** - Track added/deleted files between snapshots

4. **Compaction Planning** - Identify small files to merge

---

## 3. Iceberg Operations & Features

### Q: Explain the different write modes in Iceberg.

**Answer:**

1. **Append** - Add new data files
   ```sql
   INSERT INTO table VALUES (...)
   ```
   ```python
   df.writeTo("db.table").append()
   ```

2. **Overwrite (Dynamic)** - Replace partitions that appear in the data
   ```sql
   INSERT OVERWRITE table SELECT * FROM source
   ```
   ```python
   df.writeTo("db.table").overwritePartitions()
   ```

3. **Overwrite (Static)** - Replace specific partitions
   ```python
   df.writeTo("db.table").option("overwrite-filter", "date = '2024-01-15'").overwrite()
   ```

4. **Upsert/Merge** - Update existing rows, insert new ones
   ```sql
   MERGE INTO target USING source
   ON target.id = source.id
   WHEN MATCHED THEN UPDATE SET *
   WHEN NOT MATCHED THEN INSERT *
   ```

### Q: How do deletes work in Iceberg (Copy-on-Write vs Merge-on-Read)?

**Answer:**

**Copy-on-Write (CoW):**
- Rewrites entire data files excluding deleted rows
- Higher write cost, no read overhead
- Best for: Batch updates, low delete rate

```
Delete row 5 from file A:
1. Read file A (rows 1-10)
2. Write file A' (rows 1-4, 6-10)
3. New snapshot references A' instead of A
```

**Merge-on-Read (MoR):**
- Writes delete files (position or equality deletes)
- Lower write cost, read-time merge overhead
- Best for: Streaming updates, high delete rate

```
Delete row 5 from file A:
1. Write delete file: "file A, position 5"
2. New snapshot references A + delete file
3. Reads merge delete file with data file
```

**Delete File Types:**
- **Position deletes** - File path + row position (faster reads)
- **Equality deletes** - Column values to match (more flexible)

**Configuration:**
```sql
ALTER TABLE t SET TBLPROPERTIES (
  'write.delete.mode' = 'merge-on-read',  -- or 'copy-on-write'
  'write.update.mode' = 'merge-on-read'
);
```

### Q: What is table maintenance in Iceberg and why is it important?

**Answer:** Table maintenance optimizes table performance over time:

**1. Compaction (Rewrite Data Files)**
```sql
CALL catalog.system.rewrite_data_files('db.table');
```
- Combines small files into larger ones
- Sorts data for better query performance
- Removes deleted rows (MoR cleanup)

**2. Expire Snapshots**
```sql
CALL catalog.system.expire_snapshots('db.table', TIMESTAMP '2024-01-01');
```
- Removes old snapshots
- Allows garbage collection of unreferenced files
- Maintains time travel for recent history

**3. Remove Orphan Files**
```sql
CALL catalog.system.remove_orphan_files('db.table');
```
- Deletes files not referenced by any snapshot
- Cleans up failed writes

**4. Rewrite Manifests**
```sql
CALL catalog.system.rewrite_manifests('db.table');
```
- Combines small manifest files
- Improves query planning performance

### Q: Explain Iceberg's row-level operations (MERGE, UPDATE, DELETE).

**Answer:**

**DELETE:**
```sql
DELETE FROM events WHERE event_date < '2023-01-01';
```

**UPDATE:**
```sql
UPDATE users SET status = 'inactive' WHERE last_login < '2023-01-01';
```

**MERGE (Upsert):**
```sql
MERGE INTO target t
USING source s
ON t.id = s.id
WHEN MATCHED AND s.op = 'delete' THEN DELETE
WHEN MATCHED THEN UPDATE SET t.value = s.value, t.updated_at = s.ts
WHEN NOT MATCHED THEN INSERT (id, value, updated_at) VALUES (s.id, s.value, s.ts);
```

**CDC Pattern with MERGE:**
```sql
MERGE INTO customers target
USING (
  SELECT id, name, email, op, ts,
         ROW_NUMBER() OVER (PARTITION BY id ORDER BY ts DESC) as rn
  FROM cdc_events
) source
ON target.id = source.id AND source.rn = 1
WHEN MATCHED AND source.op = 'D' THEN DELETE
WHEN MATCHED THEN UPDATE SET *
WHEN NOT MATCHED AND source.op != 'D' THEN INSERT *;
```

---

## 4. Apache Spark Core Concepts

### Q: What is Apache Spark and what are its key components?

**Answer:** Apache Spark is a unified analytics engine for large-scale data processing.

**Key Components:**
1. **Spark Core** - Distributed task execution, memory management, fault tolerance
2. **Spark SQL** - Structured data processing with DataFrames and SQL
3. **Spark Streaming** - Real-time stream processing
4. **MLlib** - Machine learning library
5. **GraphX** - Graph processing

**Key Abstractions:**
- **RDD (Resilient Distributed Dataset)** - Low-level immutable distributed collection
- **DataFrame** - Distributed table with named columns and schema
- **Dataset** - Type-safe DataFrame (Scala/Java)
- **SparkSession** - Entry point for all Spark functionality

### Q: Explain the difference between RDD, DataFrame, and Dataset.

**Answer:**

| Feature | RDD | DataFrame | Dataset |
|---------|-----|-----------|---------|
| Type Safety | Compile-time | Runtime | Compile-time |
| Schema | No schema | Has schema | Has schema |
| Optimization | No Catalyst | Catalyst optimizer | Catalyst optimizer |
| Serialization | Java serialization | Tungsten (efficient) | Tungsten |
| API | Functional (map, filter) | Declarative (SQL-like) | Both |
| Language | All | All | Scala/Java only |

**When to use:**
- **RDD**: Low-level control, unstructured data, legacy code
- **DataFrame**: Most common, SQL-like operations, Python/R
- **Dataset**: Type safety needed, Scala/Java projects

```scala
// RDD
val rdd = sc.parallelize(Seq(1, 2, 3))
rdd.map(_ * 2).filter(_ > 2)

// DataFrame
val df = spark.read.json("data.json")
df.select("name").filter($"age" > 21)

// Dataset
case class Person(name: String, age: Int)
val ds = spark.read.json("data.json").as[Person]
ds.filter(_.age > 21)
```

### Q: What are transformations and actions in Spark?

**Answer:**

**Transformations** - Lazy operations that define a new RDD/DataFrame
- Don't execute immediately, build a DAG
- **Narrow**: Each input partition contributes to one output partition (map, filter)
- **Wide**: Input partitions contribute to multiple output partitions (groupBy, join)

**Actions** - Trigger execution and return results
- Force evaluation of the DAG
- Return values to driver or write to storage

| Transformations | Actions |
|-----------------|---------|
| map, flatMap | collect |
| filter, where | count |
| select, withColumn | show |
| groupBy, join | take, first |
| orderBy, sort | write, save |
| distinct, union | foreach |
| repartition, coalesce | reduce |

**Lazy Evaluation Benefits:**
1. Optimization - Catalyst can optimize entire query plan
2. Efficiency - Skip unnecessary computations
3. Fault tolerance - Can recompute from lineage

```python
# No execution yet (transformations)
df = spark.read.json("data.json")
filtered = df.filter(df.age > 21)
selected = filtered.select("name")

# Execution triggered (action)
result = selected.collect()
```

### Q: What is a Spark job, stage, and task?

**Answer:**

**Job** - Created by an action, represents full computation
**Stage** - Set of tasks that can run in parallel (bounded by shuffles)
**Task** - Unit of work on one partition, runs on one executor

```
Action (collect)
    ↓
Job (full DAG execution)
    ↓
Stages (split at shuffle boundaries)
    ↓
Tasks (one per partition per stage)
```

**Example:**
```python
df.filter(...).groupBy("key").count().show()
#    Stage 1: filter (narrow)
#    -- shuffle --
#    Stage 2: groupBy + count (wide) + show (action)
```

**Stage Boundaries:** Created at:
- `groupBy`, `reduceByKey`, `aggregateByKey`
- `join`, `cogroup`
- `repartition`, `coalesce`
- `sortBy`, `orderBy`

### Q: Explain Spark's cluster architecture.

**Answer:**

```
┌─────────────────────────────────────────────────────────┐
│                    Driver Program                        │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐ │
│  │ SparkContext│  │ DAG         │  │ Task            │ │
│  │             │  │ Scheduler   │  │ Scheduler       │ │
│  └─────────────┘  └─────────────┘  └─────────────────┘ │
└───────────────────────────┬─────────────────────────────┘
                            │
              ┌─────────────┼─────────────┐
              ↓             ↓             ↓
    ┌─────────────┐ ┌─────────────┐ ┌─────────────┐
    │  Executor   │ │  Executor   │ │  Executor   │
    │ ┌────┬────┐ │ │ ┌────┬────┐ │ │ ┌────┬────┐ │
    │ │Task│Task│ │ │ │Task│Task│ │ │ │Task│Task│ │
    │ └────┴────┘ │ │ └────┴────┘ │ │ └────┴────┘ │
    │   Cache     │ │   Cache     │ │   Cache     │
    └─────────────┘ └─────────────┘ └─────────────┘
         Worker          Worker          Worker
```

**Components:**

1. **Driver** - Main program, creates SparkContext, coordinates execution
   - DAG Scheduler: Builds execution graph, creates stages
   - Task Scheduler: Assigns tasks to executors

2. **Cluster Manager** - Allocates resources (YARN, Kubernetes, Mesos, Standalone)

3. **Executors** - JVM processes on worker nodes
   - Run tasks
   - Store cached data
   - Report status to driver

4. **Workers** - Physical/virtual machines hosting executors

---

## 5. Spark Architecture & Execution

### Q: Explain the Catalyst optimizer in Spark SQL.

**Answer:** Catalyst is Spark SQL's query optimizer that transforms logical plans into optimized physical plans.

**Optimization Phases:**

```
SQL Query / DataFrame API
         ↓
1. Parsing → Unresolved Logical Plan
         ↓
2. Analysis → Resolved Logical Plan (schema binding)
         ↓
3. Logical Optimization → Optimized Logical Plan
         ↓
4. Physical Planning → Physical Plans (candidates)
         ↓
5. Cost-Based Selection → Selected Physical Plan
         ↓
6. Code Generation → RDDs (execution)
```

**Key Optimizations:**

1. **Predicate Pushdown** - Push filters close to data source
   ```sql
   SELECT * FROM (SELECT * FROM t) WHERE id = 1
   → SELECT * FROM t WHERE id = 1
   ```

2. **Column Pruning** - Only read required columns
   ```sql
   SELECT name FROM users  -- only reads 'name' column from Parquet
   ```

3. **Constant Folding** - Evaluate constant expressions at compile time
   ```sql
   WHERE date > '2024-01-01' AND 1 = 1
   → WHERE date > '2024-01-01'
   ```

4. **Join Reordering** - Optimize join order based on statistics

5. **Broadcast Join Selection** - Use broadcast for small tables

### Q: What is Tungsten and how does it improve performance?

**Answer:** Tungsten is Spark's execution engine focused on CPU and memory efficiency.

**Key Features:**

1. **Memory Management**
   - Off-heap memory allocation
   - Explicit memory management (no GC overhead)
   - Binary data format (compact, cache-friendly)

2. **Cache-Aware Computation**
   - Data structures fit in CPU cache
   - Sequential memory access patterns

3. **Whole-Stage Code Generation**
   - Generates JVM bytecode for entire query stages
   - Eliminates virtual function calls
   - Loop unrolling and SIMD optimizations

**Example - Code Generation:**
```sql
SELECT a + b FROM t WHERE c > 10
```

Without codegen (volcano model):
```
for each row:
  evaluate c > 10 (virtual call)
  if true:
    evaluate a + b (virtual call)
    emit result
```

With codegen (fused):
```java
while (input.hasNext()) {
  Row row = input.next();
  if (row.getInt(2) > 10) {
    emit(row.getInt(0) + row.getInt(1));
  }
}
```

### Q: Explain shuffle in Spark and its performance implications.

**Answer:** Shuffle is the process of redistributing data across partitions, required for wide transformations.

**Shuffle Process:**
```
Map Stage                    Reduce Stage
┌─────────┐                  ┌─────────┐
│Partition│ ──┬──────────┬── │Partition│
│    0    │   │          │   │    0    │
└─────────┘   │   Hash   │   └─────────┘
┌─────────┐   │    By    │   ┌─────────┐
│Partition│ ──┼── Key ───┼── │Partition│
│    1    │   │          │   │    1    │
└─────────┘   │          │   └─────────┘
┌─────────┐   │          │   ┌─────────┐
│Partition│ ──┴──────────┴── │Partition│
│    2    │  (write shuffle  │    2    │
└─────────┘   files to disk) └─────────┘
              (read shuffle
               files)
```

**Performance Implications:**
1. **Disk I/O** - Shuffle writes intermediate files to disk
2. **Network I/O** - Data transferred between executors
3. **Serialization** - Data must be serialized/deserialized
4. **Memory** - Buffers needed for sorting/aggregation

**Optimization Strategies:**
```python
# Reduce shuffle data volume
df.filter(condition).groupBy("key")  # Filter before groupBy

# Broadcast small tables
from pyspark.sql.functions import broadcast
df1.join(broadcast(small_df), "key")

# Use appropriate partitioning
df.repartition(200, "key")  # Repartition by join key before join

# Combine operations
df.groupBy("key").agg(sum("a"), count("b"))  # Single shuffle

# Use map-side aggregation
spark.conf.set("spark.sql.shuffle.partitions", 200)
```

### Q: What is data skew and how do you handle it?

**Answer:** Data skew occurs when data is unevenly distributed across partitions, causing some tasks to take much longer than others.

**Detection:**
- Spark UI shows tasks with vastly different durations
- One partition much larger than others
- OOM errors on specific executors

**Solutions:**

**1. Salting (Add random prefix to skewed keys)**
```python
from pyspark.sql.functions import concat, lit, rand, floor

# Add salt to skewed key
salt_buckets = 10
salted_df = df.withColumn("salted_key",
    concat(col("key"), lit("_"), floor(rand() * salt_buckets)))

# Join with salted key (explode lookup table)
lookup_exploded = lookup.crossJoin(
    spark.range(salt_buckets).withColumnRenamed("id", "salt")
).withColumn("salted_key", concat(col("key"), lit("_"), col("salt")))

result = salted_df.join(lookup_exploded, "salted_key")
```

**2. Broadcast Join (for small skewed table)**
```python
result = large_df.join(broadcast(small_df), "key")
```

**3. Adaptive Query Execution (AQE) - Spark 3.0+**
```python
spark.conf.set("spark.sql.adaptive.enabled", True)
spark.conf.set("spark.sql.adaptive.skewJoin.enabled", True)
```

**4. Isolate Skewed Keys**
```python
# Process skewed keys separately
skewed_keys = ["hot_key_1", "hot_key_2"]
skewed_df = df.filter(col("key").isin(skewed_keys))
normal_df = df.filter(~col("key").isin(skewed_keys))

# Use broadcast for skewed, regular join for normal
skewed_result = skewed_df.join(broadcast(lookup), "key")
normal_result = normal_df.join(lookup, "key")
result = skewed_result.union(normal_result)
```

**5. Increase Parallelism**
```python
spark.conf.set("spark.sql.shuffle.partitions", 1000)
df.repartition(1000, "key")
```

### Q: Explain Adaptive Query Execution (AQE) in Spark 3.x.

**Answer:** AQE optimizes queries at runtime based on actual data statistics.

**Key Features:**

**1. Dynamically Coalescing Shuffle Partitions**
```python
# Before: 200 partitions, some very small
# After: Coalesces small partitions automatically
spark.conf.set("spark.sql.adaptive.coalescePartitions.enabled", True)
spark.conf.set("spark.sql.adaptive.advisoryPartitionSizeInBytes", "64MB")
```

**2. Dynamically Switching Join Strategies**
```python
# Runtime decision: If one side is small after filtering, use broadcast
spark.conf.set("spark.sql.adaptive.enabled", True)
# No need to hint, AQE decides at runtime
```

**3. Dynamically Optimizing Skew Joins**
```python
spark.conf.set("spark.sql.adaptive.skewJoin.enabled", True)
spark.conf.set("spark.sql.adaptive.skewJoin.skewedPartitionFactor", 5)
spark.conf.set("spark.sql.adaptive.skewJoin.skewedPartitionThresholdInBytes", "256MB")
```

**Configuration:**
```python
spark.conf.set("spark.sql.adaptive.enabled", True)  # Master switch
spark.conf.set("spark.sql.adaptive.localShuffleReader.enabled", True)
```

---

## 6. Spark SQL & DataFrames

### Q: What are the different types of joins in Spark?

**Answer:**

**Join Types:**
```python
# Inner Join (default)
df1.join(df2, "key")
df1.join(df2, df1.key == df2.key, "inner")

# Left/Right/Full Outer
df1.join(df2, "key", "left")
df1.join(df2, "key", "right")
df1.join(df2, "key", "outer")  # full outer

# Left Semi (exists)
df1.join(df2, "key", "left_semi")  # rows in df1 that have match in df2

# Left Anti (not exists)
df1.join(df2, "key", "left_anti")  # rows in df1 with no match in df2

# Cross Join
df1.crossJoin(df2)
```

**Join Strategies (Physical):**

| Strategy | When Used | Characteristics |
|----------|-----------|-----------------|
| Broadcast Hash | Small table (< 10MB default) | No shuffle, fast |
| Shuffle Hash | Medium tables, equal keys | One-side shuffle |
| Sort Merge | Large tables | Both sides shuffle + sort |
| Broadcast Nested Loop | Cross join, non-equi | Broadcast + nested loop |
| Cartesian | Cross join, no condition | Full cartesian product |

**Hints:**
```python
from pyspark.sql.functions import broadcast

# Force broadcast
df1.join(broadcast(df2), "key")

# SQL hints
spark.sql("""
  SELECT /*+ BROADCAST(small) */ *
  FROM large JOIN small ON large.key = small.key
""")

# Merge hint
spark.sql("""
  SELECT /*+ MERGE(t1, t2) */ *
  FROM t1 JOIN t2 ON t1.key = t2.key
""")
```

### Q: Explain window functions in Spark.

**Answer:** Window functions perform calculations across a set of rows related to the current row.

**Basic Structure:**
```python
from pyspark.sql import Window
from pyspark.sql.functions import *

# Define window
window = Window.partitionBy("department").orderBy("salary")

# Apply window function
df.withColumn("rank", rank().over(window))
```

**Window Specifications:**
```python
# Partition + Order
Window.partitionBy("dept").orderBy("date")

# Row-based frame
Window.partitionBy("dept").orderBy("date").rowsBetween(-2, 2)  # 5 rows

# Range-based frame
Window.partitionBy("dept").orderBy("amount").rangeBetween(-100, 100)

# Unbounded
Window.partitionBy("dept").orderBy("date").rowsBetween(
    Window.unboundedPreceding, Window.currentRow
)
```

**Common Window Functions:**
```python
# Ranking
rank()          # 1, 2, 2, 4 (gaps after ties)
dense_rank()    # 1, 2, 2, 3 (no gaps)
row_number()    # 1, 2, 3, 4 (unique)
ntile(4)        # quartiles
percent_rank()  # percentile rank

# Analytic
lag("col", 1)       # previous row value
lead("col", 1)      # next row value
first("col")        # first in window
last("col")         # last in window

# Aggregate (running)
sum("amount").over(window)
avg("amount").over(window)
count("*").over(window)
```

**Example - Running Total:**
```python
window = Window.partitionBy("account").orderBy("date").rowsBetween(
    Window.unboundedPreceding, Window.currentRow
)
df.withColumn("running_balance", sum("amount").over(window))
```

### Q: How do you handle NULL values in Spark?

**Answer:**

**Checking for NULL:**
```python
from pyspark.sql.functions import col, isnan, isnull, when

# Filter nulls
df.filter(col("value").isNull())
df.filter(col("value").isNotNull())

# Check for NaN (floating point)
df.filter(isnan("value"))
```

**Replacing NULL:**
```python
# Fill all columns
df.na.fill(0)
df.na.fill("unknown")

# Fill specific columns
df.na.fill({"age": 0, "name": "unknown"})

# Fill with column-specific logic
df.withColumn("value", when(col("value").isNull(), 0).otherwise(col("value")))

# Fill with coalesce
from pyspark.sql.functions import coalesce
df.withColumn("value", coalesce(col("value"), col("default_value"), lit(0)))
```

**Drop rows with NULL:**
```python
df.na.drop()                    # any column null
df.na.drop(how="all")          # all columns null
df.na.drop(subset=["col1"])    # specific columns
df.na.drop(thresh=2)           # at least 2 non-null
```

**NULL in comparisons:**
```python
# NULL comparisons return NULL (not True/False)
# Use null-safe equals
df.filter(col("a").eqNullSafe(col("b")))

# In SQL
spark.sql("SELECT * FROM t WHERE a <=> b")  # null-safe equals
```

### Q: Explain User-Defined Functions (UDFs) in Spark.

**Answer:**

**Python UDF:**
```python
from pyspark.sql.functions import udf
from pyspark.sql.types import StringType

# Register UDF
@udf(returnType=StringType())
def upper_case(s):
    return s.upper() if s else None

# Use in DataFrame
df.withColumn("name_upper", upper_case(col("name")))

# Register for SQL
spark.udf.register("upper_case", upper_case)
spark.sql("SELECT upper_case(name) FROM users")
```

**Pandas UDF (Vectorized - Much faster):**
```python
from pyspark.sql.functions import pandas_udf
import pandas as pd

# Scalar Pandas UDF
@pandas_udf("string")
def upper_case_pandas(s: pd.Series) -> pd.Series:
    return s.str.upper()

# Grouped Map
@pandas_udf("id long, v double")
def normalize(pdf: pd.DataFrame) -> pd.DataFrame:
    v = pdf['v']
    pdf['v'] = (v - v.mean()) / v.std()
    return pdf

df.groupBy("group").applyInPandas(normalize, schema="id long, v double")
```

**Performance Comparison:**
| UDF Type | Serialization | Speed | Use Case |
|----------|---------------|-------|----------|
| Python UDF | Row-by-row | Slow | Simple logic |
| Pandas UDF | Batch (Arrow) | Fast | Vectorizable operations |
| Scala UDF | Native | Fastest | Performance critical |

**Best Practices:**
1. Prefer built-in functions over UDFs
2. Use Pandas UDFs over Python UDFs
3. Avoid UDFs in joins and filters (can't be pushed down)
4. Handle NULL explicitly in UDF code

---

## 7. Spark Performance & Optimization

### Q: What are the key Spark configuration parameters for performance?

**Answer:**

**Memory Configuration:**
```python
# Executor memory
spark.executor.memory = "8g"
spark.executor.memoryOverhead = "1g"  # Off-heap

# Memory fractions
spark.memory.fraction = 0.6  # Execution + storage
spark.memory.storageFraction = 0.5  # Storage within fraction

# Off-heap
spark.memory.offHeap.enabled = true
spark.memory.offHeap.size = "4g"
```

**Parallelism:**
```python
# Default parallelism for RDD operations
spark.default.parallelism = 200

# Shuffle partitions for SQL/DataFrame
spark.sql.shuffle.partitions = 200

# Adaptive (Spark 3.0+)
spark.sql.adaptive.enabled = true
```

**Serialization:**
```python
# Use Kryo (faster than Java serialization)
spark.serializer = "org.apache.spark.serializer.KryoSerializer"
spark.kryo.registrationRequired = false
```

**Shuffle:**
```python
spark.shuffle.compress = true
spark.shuffle.spill.compress = true
spark.sql.shuffle.partitions = 200
```

**Broadcast:**
```python
spark.sql.autoBroadcastJoinThreshold = 10485760  # 10MB
```

### Q: How do you tune the number of partitions?

**Answer:**

**Guidelines:**
- 2-4 partitions per CPU core
- Each partition: 128MB - 1GB
- More partitions = more parallelism but more overhead

**Check Partitions:**
```python
df.rdd.getNumPartitions()
spark.sql("SELECT count(*) FROM table").explain()  # Shows partitions
```

**Change Partitions:**
```python
# Increase partitions (shuffle)
df.repartition(200)
df.repartition(200, "key")  # Repartition by column

# Decrease partitions (no shuffle if reducing)
df.coalesce(10)

# For shuffle operations
spark.conf.set("spark.sql.shuffle.partitions", 200)
```

**Partition Sizing:**
```python
# Calculate target partitions
data_size_bytes = df.rdd.map(lambda x: len(str(x))).sum()
target_partition_size = 128 * 1024 * 1024  # 128MB
num_partitions = max(data_size_bytes // target_partition_size, 1)
```

### Q: Explain caching in Spark and when to use it.

**Answer:**

**Storage Levels:**
```python
from pyspark import StorageLevel

df.cache()  # MEMORY_AND_DISK (default for DataFrame)
df.persist()  # Same as cache()
df.persist(StorageLevel.MEMORY_ONLY)
df.persist(StorageLevel.MEMORY_AND_DISK_SER)
df.persist(StorageLevel.DISK_ONLY)
df.unpersist()  # Remove from cache
```

| Level | Space | CPU | In Memory | On Disk |
|-------|-------|-----|-----------|---------|
| MEMORY_ONLY | High | Low | Yes | No |
| MEMORY_AND_DISK | High | Medium | Yes | Spill |
| MEMORY_ONLY_SER | Low | High | Yes (serialized) | No |
| DISK_ONLY | Low | High | No | Yes |

**When to Cache:**
```python
# Good: DataFrame used multiple times
df = spark.read.parquet("large_data")
filtered = df.filter(condition)
filtered.cache()

result1 = filtered.groupBy("a").count()
result2 = filtered.groupBy("b").sum("c")

# Bad: DataFrame used once
df = spark.read.parquet("data")
df.cache()  # Unnecessary
df.write.parquet("output")
```

**Cache vs Checkpoint:**
```python
# Cache: Stored in executor memory/disk, lineage preserved
df.cache()

# Checkpoint: Written to reliable storage, lineage truncated
spark.sparkContext.setCheckpointDir("/checkpoint")
df.checkpoint()  # Eager
df.checkpoint(eager=False)  # Lazy

# Use checkpoint for:
# - Very long lineages (prevents stack overflow)
# - Iterative algorithms
# - When you need fault tolerance beyond lineage
```

### Q: What are broadcast variables and accumulators?

**Answer:**

**Broadcast Variables** - Efficiently share read-only data with all executors:
```python
# Without broadcast: Data sent with each task
lookup = {"a": 1, "b": 2}
rdd.map(lambda x: lookup.get(x))  # lookup serialized per task

# With broadcast: Data sent once per executor
lookup_bc = spark.sparkContext.broadcast(lookup)
rdd.map(lambda x: lookup_bc.value.get(x))

# Clean up
lookup_bc.unpersist()
lookup_bc.destroy()
```

**Accumulators** - Aggregators that tasks can add to:
```python
# Numeric accumulator
error_count = spark.sparkContext.accumulator(0)

def process_record(record):
    try:
        return transform(record)
    except:
        error_count.add(1)
        return None

results = rdd.map(process_record)
results.count()  # Trigger execution
print(f"Errors: {error_count.value}")

# Custom accumulator
from pyspark.accumulators import AccumulatorParam

class SetAccumulatorParam(AccumulatorParam):
    def zero(self, v):
        return set()
    def addInPlace(self, v1, v2):
        return v1.union(v2)

unique_keys = spark.sparkContext.accumulator(set(), SetAccumulatorParam())
```

**Important Notes:**
- Accumulators only guaranteed accurate with actions (not transformations due to re-execution)
- Use for debugging/monitoring, not business logic

---

## 8. Spark Streaming

### Q: Compare Spark Streaming (DStreams) vs Structured Streaming.

**Answer:**

| Feature | DStreams | Structured Streaming |
|---------|----------|---------------------|
| API | RDD-based | DataFrame/Dataset |
| Processing Model | Micro-batch | Micro-batch + Continuous |
| Exactly-once | With checkpointing | Built-in |
| Event Time | Manual | Native support |
| Watermarks | Manual | Built-in |
| State Management | updateStateByKey | Native stateful ops |
| Output Modes | N/A | Append, Complete, Update |

**Recommendation:** Use Structured Streaming for new applications.

### Q: Explain the key concepts in Structured Streaming.

**Answer:**

**Basic Structure:**
```python
# Read stream
df = spark.readStream \
    .format("kafka") \
    .option("subscribe", "topic") \
    .load()

# Transform (same as batch)
result = df.select("value").groupBy("key").count()

# Write stream
query = result.writeStream \
    .format("console") \
    .outputMode("complete") \
    .trigger(processingTime="10 seconds") \
    .start()

query.awaitTermination()
```

**Output Modes:**
```python
# Append - Only new rows (default, for non-aggregation)
.outputMode("append")

# Complete - Entire result table (for aggregations)
.outputMode("complete")

# Update - Only changed rows
.outputMode("update")
```

**Triggers:**
```python
# Fixed interval micro-batch
.trigger(processingTime="10 seconds")

# Once (for testing/backfill)
.trigger(once=True)

# Available-now (process all available, then stop)
.trigger(availableNow=True)

# Continuous (experimental, low latency)
.trigger(continuous="1 second")
```

### Q: Explain watermarks and late data handling.

**Answer:**

**Watermark** - Threshold for how late data can arrive:
```python
from pyspark.sql.functions import window

df.withWatermark("event_time", "10 minutes") \
  .groupBy(window("event_time", "5 minutes")) \
  .count()
```

**How Watermarks Work:**
```
Max event time seen: 12:30:00
Watermark: 12:20:00 (max - 10 minutes)

Events with event_time < 12:20:00 may be dropped
Events with event_time >= 12:20:00 will be processed
```

**State Cleanup:**
- State for windows before watermark can be cleaned up
- Prevents unbounded state growth

**Late Data Handling Strategies:**
1. **Drop late data** (default after watermark)
2. **Output late data** (use update mode)
3. **Longer watermark** (tradeoff: more state)

### Q: How do you handle stateful operations in Structured Streaming?

**Answer:**

**Built-in Stateful Operations:**
```python
# Aggregations
df.groupBy("key").count()
df.groupBy(window("time", "10 minutes")).sum("value")

# Deduplication
df.dropDuplicates(["id"])
df.withWatermark("time", "10 minutes").dropDuplicates(["id", "time"])
```

**Arbitrary Stateful Processing:**
```python
from pyspark.sql.streaming import GroupState

def update_state(key, values, state: GroupState):
    # Get previous state
    if state.exists:
        prev = state.get
    else:
        prev = 0

    # Compute new state
    total = prev + sum(v.value for v in values)
    state.update(total)

    # Handle timeout
    if state.hasTimedOut:
        state.remove()
        return (key, -1)  # Final output

    # Set timeout
    state.setTimeoutDuration("1 hour")

    return (key, total)

df.groupByKey(lambda x: x.key) \
  .mapGroupsWithState(
      update_state,
      outputMode="update",
      timeoutConf=GroupStateTimeout.ProcessingTimeTimeout
  )
```

---

## 9. Iceberg + Spark Integration

### Q: How do you configure Spark to work with Iceberg?

**Answer:**

**Spark Session Configuration:**
```python
from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName("IcebergApp") \
    .config("spark.jars.packages", "org.apache.iceberg:iceberg-spark-runtime-3.5_2.12:1.5.0") \
    .config("spark.sql.extensions", "org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions") \
    .config("spark.sql.catalog.spark_catalog", "org.apache.iceberg.spark.SparkSessionCatalog") \
    .config("spark.sql.catalog.spark_catalog.type", "hive") \
    .config("spark.sql.catalog.local", "org.apache.iceberg.spark.SparkCatalog") \
    .config("spark.sql.catalog.local.type", "hadoop") \
    .config("spark.sql.catalog.local.warehouse", "s3://bucket/warehouse") \
    .getOrCreate()
```

**Catalog Types:**
- **Hive** - Uses Hive Metastore
- **Hadoop** - Uses HDFS/S3 directories
- **AWS Glue** - Uses AWS Glue Catalog
- **Nessie** - Git-like versioning
- **REST** - Generic REST catalog

### Q: How do you perform CRUD operations on Iceberg tables with Spark?

**Answer:**

**Create Table:**
```sql
-- SQL
CREATE TABLE catalog.db.table (
    id BIGINT,
    data STRING,
    ts TIMESTAMP
) USING iceberg
PARTITIONED BY (days(ts))
TBLPROPERTIES (
    'write.format.default' = 'parquet',
    'write.parquet.compression-codec' = 'snappy'
);
```

```python
# DataFrame API
df.writeTo("catalog.db.table") \
  .using("iceberg") \
  .partitionedBy(days("ts")) \
  .createOrReplace()
```

**Read:**
```python
df = spark.table("catalog.db.table")
df = spark.read.format("iceberg").load("catalog.db.table")

# Time travel
df = spark.read.option("snapshot-id", 123456).table("catalog.db.table")
df = spark.read.option("as-of-timestamp", "2024-01-15 10:00:00").table("catalog.db.table")
```

**Insert:**
```python
# Append
df.writeTo("catalog.db.table").append()

# Overwrite partitions
df.writeTo("catalog.db.table").overwritePartitions()
```

**Update/Delete/Merge:**
```sql
-- Delete
DELETE FROM catalog.db.table WHERE ts < '2023-01-01';

-- Update
UPDATE catalog.db.table SET status = 'archived' WHERE ts < '2023-01-01';

-- Merge (Upsert)
MERGE INTO catalog.db.target t
USING catalog.db.source s
ON t.id = s.id
WHEN MATCHED THEN UPDATE SET *
WHEN NOT MATCHED THEN INSERT *;
```

### Q: How do you perform incremental reads with Iceberg?

**Answer:**

**Incremental Scan (Between Snapshots):**
```python
# Read changes between snapshots
df = spark.read.format("iceberg") \
    .option("start-snapshot-id", start_id) \
    .option("end-snapshot-id", end_id) \
    .load("catalog.db.table")

# Streaming reads
df = spark.readStream.format("iceberg") \
    .option("stream-from-timestamp", "2024-01-01 00:00:00") \
    .load("catalog.db.table")
```

**Change Data Capture:**
```python
# Read all changes (added + deleted rows)
spark.read.format("iceberg") \
    .option("start-snapshot-id", start_id) \
    .option("end-snapshot-id", end_id) \
    .load("catalog.db.table.changes")  # .changes suffix

# Result includes _change_type column: 'insert', 'delete', 'update_before', 'update_after'
```

### Q: How do you optimize Iceberg tables with Spark?

**Answer:**

**Compaction:**
```sql
-- Rewrite data files (compaction)
CALL catalog.system.rewrite_data_files(
    table => 'db.table',
    strategy => 'binpack',
    options => map('target-file-size-bytes', '134217728')  -- 128MB
);

-- Sort during compaction
CALL catalog.system.rewrite_data_files(
    table => 'db.table',
    strategy => 'sort',
    sort_order => 'id ASC NULLS FIRST, data ASC NULLS FIRST'
);
```

**Expire Snapshots:**
```sql
CALL catalog.system.expire_snapshots(
    table => 'db.table',
    older_than => TIMESTAMP '2024-01-01 00:00:00',
    retain_last => 10
);
```

**Remove Orphan Files:**
```sql
CALL catalog.system.remove_orphan_files(
    table => 'db.table',
    older_than => TIMESTAMP '2024-01-01 00:00:00'
);
```

**Rewrite Manifests:**
```sql
CALL catalog.system.rewrite_manifests('db.table');
```

---

## 10. Scenario-Based Questions

### Q: Design a data pipeline that ingests CDC events into an Iceberg table.

**Answer:**

```python
from pyspark.sql.functions import *

# Read CDC events from Kafka
cdc_stream = spark.readStream \
    .format("kafka") \
    .option("subscribe", "cdc-events") \
    .load() \
    .select(from_json(col("value").cast("string"), schema).alias("data")) \
    .select("data.*")

# Deduplicate within batch (keep latest per key)
deduplicated = cdc_stream \
    .withWatermark("event_time", "10 minutes") \
    .dropDuplicates(["id", "event_time"])

# Micro-batch processing with foreachBatch
def upsert_to_iceberg(batch_df, batch_id):
    if batch_df.isEmpty():
        return

    # Deduplicate within batch (latest event wins)
    deduped = batch_df.orderBy(col("event_time").desc()) \
        .dropDuplicates(["id"])

    # Write to temp view for MERGE
    deduped.createOrReplaceTempView("updates")

    spark.sql("""
        MERGE INTO catalog.db.target t
        USING updates s
        ON t.id = s.id
        WHEN MATCHED AND s.op = 'D' THEN DELETE
        WHEN MATCHED THEN UPDATE SET *
        WHEN NOT MATCHED AND s.op != 'D' THEN INSERT *
    """)

# Start streaming query
query = deduplicated.writeStream \
    .foreachBatch(upsert_to_iceberg) \
    .option("checkpointLocation", "/checkpoint/cdc") \
    .trigger(processingTime="1 minute") \
    .start()
```

### Q: How would you handle a slowly changing dimension (SCD Type 2) with Iceberg?

**Answer:**

```sql
-- SCD Type 2: Track history with effective dates
CREATE TABLE catalog.db.dim_customer (
    customer_key BIGINT,     -- Surrogate key
    customer_id STRING,      -- Natural key
    name STRING,
    email STRING,
    effective_from TIMESTAMP,
    effective_to TIMESTAMP,
    is_current BOOLEAN
) USING iceberg;

-- Process changes
MERGE INTO catalog.db.dim_customer target
USING (
    SELECT
        customer_id,
        name,
        email,
        change_timestamp
    FROM staging_changes
) source
ON target.customer_id = source.customer_id AND target.is_current = true

-- Close existing record if changed
WHEN MATCHED AND (
    target.name != source.name OR
    target.email != source.email
) THEN UPDATE SET
    effective_to = source.change_timestamp,
    is_current = false

-- Insert new version (handled separately due to MERGE limitations)
;

-- Insert new current records
INSERT INTO catalog.db.dim_customer
SELECT
    uuid() as customer_key,
    s.customer_id,
    s.name,
    s.email,
    s.change_timestamp as effective_from,
    CAST('9999-12-31' AS TIMESTAMP) as effective_to,
    true as is_current
FROM staging_changes s
LEFT JOIN catalog.db.dim_customer t
ON s.customer_id = t.customer_id AND t.is_current = true
WHERE t.customer_id IS NULL
   OR t.name != s.name
   OR t.email != s.email;
```

### Q: You have a Spark job that's running slowly. How would you diagnose and optimize it?

**Answer:**

**1. Check Spark UI:**
- **Jobs Tab**: Identify slow stages
- **Stages Tab**: Look for data skew (task duration variance)
- **Storage Tab**: Check cache utilization
- **SQL Tab**: Examine query plans

**2. Common Issues and Fixes:**

**Data Skew:**
```python
# Before: Skewed join
df1.join(df2, "key")

# After: Salted join or broadcast
df1.join(broadcast(df2), "key")  # If df2 is small
# Or use AQE
spark.conf.set("spark.sql.adaptive.skewJoin.enabled", True)
```

**Too Many/Few Partitions:**
```python
# Check partition count
df.rdd.getNumPartitions()

# Repartition if needed
df.repartition(200)  # For parallelism
df.coalesce(50)      # Reduce small files
```

**Shuffle Optimization:**
```python
# Filter before join
df_filtered = df.filter(condition)
df_filtered.join(other, "key")

# Broadcast small tables
from pyspark.sql.functions import broadcast
large_df.join(broadcast(small_df), "key")
```

**Memory Issues:**
```python
# Increase memory
spark.executor.memory = "8g"
spark.executor.memoryOverhead = "2g"

# Reduce memory pressure
spark.conf.set("spark.sql.shuffle.partitions", 400)  # More partitions
df.persist(StorageLevel.MEMORY_AND_DISK_SER)  # Serialized caching
```

**3. Query Plan Analysis:**
```python
df.explain(True)  # Show all plans
df.explain("cost")  # Show cost estimates
df.explain("formatted")  # Pretty print
```

### Q: How would you implement exactly-once semantics with Spark and Iceberg?

**Answer:**

```python
# Structured Streaming with Iceberg provides exactly-once via checkpointing

# Read from Kafka (with offsets tracked)
stream = spark.readStream \
    .format("kafka") \
    .option("subscribe", "events") \
    .option("startingOffsets", "earliest") \
    .load()

# Process
processed = stream.select(
    from_json(col("value").cast("string"), schema).alias("data")
).select("data.*")

# Write to Iceberg with checkpointing
query = processed.writeStream \
    .format("iceberg") \
    .outputMode("append") \
    .option("checkpointLocation", "s3://bucket/checkpoint") \
    .option("fanout-enabled", "true") \
    .toTable("catalog.db.events")

# Exactly-once guarantees:
# 1. Kafka offsets tracked in checkpoint
# 2. Iceberg commits are atomic
# 3. On failure, reprocessing starts from last committed offset
```

**For Batch Jobs:**
```python
# Use job-level idempotency
job_id = "daily_load_20240115"

# Check if already processed
existing = spark.sql(f"""
    SELECT 1 FROM catalog.db.target
    WHERE _job_id = '{job_id}' LIMIT 1
""")

if existing.isEmpty():
    # Add job ID for idempotency
    df_with_job = df.withColumn("_job_id", lit(job_id))

    # Atomic write
    df_with_job.writeTo("catalog.db.target").append()
```

---

## Quick Reference

### Iceberg Commands
```sql
-- Table maintenance
CALL catalog.system.rewrite_data_files('db.table');
CALL catalog.system.expire_snapshots('db.table', TIMESTAMP '2024-01-01');
CALL catalog.system.remove_orphan_files('db.table');

-- Time travel
SELECT * FROM table VERSION AS OF 12345;
SELECT * FROM table TIMESTAMP AS OF '2024-01-01';

-- Schema evolution
ALTER TABLE t ADD COLUMN new_col STRING;
ALTER TABLE t ALTER COLUMN col TYPE BIGINT;
ALTER TABLE t RENAME COLUMN old TO new;

-- Partition evolution
ALTER TABLE t ADD PARTITION FIELD hour(ts);
ALTER TABLE t DROP PARTITION FIELD month(ts);
```

### Spark Performance Configs
```python
spark.sql.shuffle.partitions = 200
spark.sql.adaptive.enabled = true
spark.sql.adaptive.skewJoin.enabled = true
spark.sql.autoBroadcastJoinThreshold = 10485760
spark.serializer = "org.apache.spark.serializer.KryoSerializer"
```

### Key Formulas
```
Partitions = max(data_size / 128MB, num_cores * 2-4)
Executor memory = (node_memory - overhead) / executors_per_node
Parallelism = num_executors * executor_cores
```

---

*Good luck with your interviews!*
