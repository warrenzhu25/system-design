# Database and Systems Paper Notes

This directory contains readable Markdown notes for influential database, storage, and serving
systems papers. Each note links to the authoritative publication and clearly identifies its venue
and award status.

## Collection

| System | Venue and status | Theme | Original | Notes |
| --- | --- | --- | --- | --- |
| **Aurora** | SIGMOD 2017, Industrial | Cloud OLTP, log-as-database | [Full text](original/aurora-sigmod-2017.md) | [Notes](notes/aurora-sigmod-2017.md) |
| **ClickHouse** | VLDB 2024 | Columnar OLAP, merge-time transformation | [Full text](original/clickhouse-vldb-2024.md) | [Notes](notes/clickhouse-vldb-2024.md) |
| **CockroachDB** | SIGMOD 2020, Industrial | Geo-distributed serializable SQL | [Full text](original/cockroachdb-sigmod-2020.md) | [Notes](notes/cockroachdb-sigmod-2020.md) |
| **Dash** | VLDB 2020 | Hashing on persistent memory | [Full text](original/dash-vldb-2020.md) | [Notes](notes/dash-vldb-2020.md) |
| **DataFusion** | SIGMOD 2024, Industrial | Embeddable modular query engine | [Full text](original/datafusion-sigmod-2024.md) | [Notes](notes/datafusion-sigmod-2024.md) |
| **DynamoDB** | USENIX ATC 2022 | Managed NoSQL, admission control | [Full text](original/dynamodb-atc-2022.md) | [Notes](notes/dynamodb-atc-2022.md) |
| **FASTER** | SIGMOD 2018 | Concurrent KV store, hybrid log | [Full text](original/faster-sigmod-2018.md) | [Notes](notes/faster-sigmod-2018.md) |
| **FoundationDB Record Layer** | SIGMOD 2019, Industrial | Multi-tenant structured store | [Full text](original/foundationdb-record-layer-sigmod-2019.md) | [Notes](notes/foundationdb-record-layer-sigmod-2019.md) |
| **Kora** | VLDB 2023 | Cloud-native Kafka platform | [Full text](original/kora-vldb-2023.md) | [Notes](notes/kora-vldb-2023.md) |
| **Masstree** | EuroSys 2012 | Cache-crafty multicore KV index | [Full text](original/masstree-eurosys-2012.md) | [Notes](notes/masstree-eurosys-2012.md) |
| **MICA** | NSDI 2014 | Kernel-bypass in-memory KV store | [Full text](original/mica-nsdi-2014.md) | [Notes](notes/mica-nsdi-2014.md) |
| **ScyllaDB** | Vendor whitepaper (not peer reviewed) | Shard-per-core NoSQL design | [Full text](original/scylladb-seven-design-principles.md) | [Notes](notes/scylladb-seven-design-principles.md) |
| **Silo** | SOSP 2013 | Multicore serializable transactions | [Full text](original/silo-sosp-2013.md) | [Notes](notes/silo-sosp-2013.md) |
| **Ursa** | **VLDB 2025 Best Industry Paper** | Lakehouse-native Kafka engine | [Full text](original/ursa-vldb-2025.md) | [Notes](notes/ursa-vldb-2025.md) |
| **Velox** | VLDB 2022 | Unified execution engine library | [Full text](original/velox-vldb-2022.md) | [Notes](notes/velox-vldb-2022.md) |
| **VLL** | VLDB Journal 24(5), 2015 | Lightweight lock manager | [Full text](original/vll-vldb-journal-2015.md) | [Notes](notes/vll-vldb-journal-2015.md) |
| **vLLM** | SOSP 2023 | LLM serving, PagedAttention | [Full text](original/vllm-sosp-2023.md) | [Notes](notes/vllm-sosp-2023.md) |

All notes are also available as a single combined volume:
**[notes/all-notes.md](notes/all-notes.md)**. Regenerate it after editing any note with:

```sh
python3 scripts/merge_markdown_papers.py --source notes --output notes/all-notes.md \
  --title "Database and Systems Paper Notes" \
  --subtitle "Combined reading notes for all 17 papers. Study aids, not substitutes for the papers; check claims and benchmark numbers against the linked sources."
```

Only Ursa is a VLDB Best Industry Paper. The other entries are included for their technical
relevance, and their venues are stated exactly so no award or status is attributed to the wrong
work. ScyllaDB's entry is a corporate whitepaper rather than a peer-reviewed paper; its notes flag
this and treat its performance claims accordingly.

## Reading paths

Papers in this collection cluster into a few recurring themes. These orderings work well as
sequences.

**Separating the log from its storage.** [Aurora](notes/aurora-sigmod-2017.md) →
[FASTER](notes/faster-sigmod-2018.md) → [Kora](notes/kora-vldb-2023.md) →
[Ursa](notes/ursa-vldb-2025.md). All four decide that the durable log is the real database and
that materialized state is a cache over it; they differ in where the log lives and who applies it.

**Concurrency without contention on multicore.** [Masstree](notes/masstree-eurosys-2012.md) →
[Silo](notes/silo-sosp-2013.md) → [VLL](notes/vll-vldb-journal-2015.md) →
[Dash](notes/dash-vldb-2020.md). A progression from "readers must never write shared memory" to
"never take a global counter" to "compress the lock table itself," ending with the same lessons
re-derived for persistent memory.

**Partition or share?** [Masstree](notes/masstree-eurosys-2012.md) §6.6 →
[MICA](notes/mica-nsdi-2014.md) → [Silo](notes/silo-sosp-2013.md) §5.4 →
[ScyllaDB](notes/scylladb-seven-design-principles.md). The same question — should cores share data
or own partitions? — answered four ways, with measurements showing the crossover depends on skew
and cross-partition rate.

**Analytical execution engines.** [ClickHouse](notes/clickhouse-vldb-2024.md) →
[Velox](notes/velox-vldb-2022.md) → [DataFusion](notes/datafusion-sigmod-2024.md). Monolithic and
highly tuned, versus deliberately componentized; both argue vectorization plus runtime adaptivity
over plan-time omniscience.

**Operating a system at scale.** [DynamoDB](notes/dynamodb-atc-2022.md) →
[Kora](notes/kora-vldb-2023.md) → [CockroachDB](notes/cockroachdb-sigmod-2020.md) §7. Retrospectives
rather than designs: admission control, deployment safety, gray-failure detection, and the things
that turned out to matter more than the algorithms.

**OS ideas borrowed by data systems.** [vLLM](notes/vllm-sosp-2023.md) (virtual memory and paging) →
[FASTER](notes/faster-sigmod-2018.md) (epoch protection) →
[FoundationDB Record Layer](notes/foundationdb-record-layer-sigmod-2019.md) (layering).

## Note format

Every paper entry has two Markdown files:

1. A faithful full-content conversion under `original/`, including figures, tables, code, and
   references from the published paper.
2. Reading notes under `notes/` containing:

   - complete citation, DOI, venue, award, and primary-source links;
   - a one-sentence summary and the problem being solved;
   - architecture and important data flows, with the paper's own mechanisms preserved;
   - correctness model and failure behavior;
   - evaluation results, with the authors' assumptions and hardware stated;
   - limitations, open questions, and practical lessons;
   - a "takeaways" section distilling the transferable ideas.

The `original/` files are format conversions and may contain extraction errors; the official PDF
in `pdf/` remains authoritative. Reading notes are study aids rather than substitutes for the
paper. Claims and benchmark numbers should be checked against the linked source before production
use.

## Directory layout

```text
paper/
├── README.md      this index
├── notes/         reading notes, one per paper (+ all-notes.md, the merged volume)
├── original/      full-text Markdown conversions (+ all-papers.md, the merged volume)
├── pdf/           source PDFs
├── scripts/       PDF → Markdown conversion and merge tooling
└── tests/         tests for the scripts
```
