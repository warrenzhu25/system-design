# Apache Spark Upgrade Agent — System Design

An agentic system that automates upgrading Apache Spark applications (PySpark / Scala)
across major Spark versions — analyzing code, transforming it to resolve API and
dependency incompatibilities, validating on real clusters, and verifying data-quality
parity, all under human approval.

Modeled on the [Apache Spark Upgrade Agent for Amazon EMR](https://docs.aws.amazon.com/emr/latest/ReleaseGuide/spark-upgrades.html).

## Table of Contents
1. [Problem & Requirements](#1-problem--requirements)
2. [Why This Is Hard](#2-why-this-is-hard)
3. [High-Level Architecture](#3-high-level-architecture)
4. [The Agent Loop](#4-the-agent-loop)
5. [Workflow Phases](#5-workflow-phases)
6. [Tool Catalog (MCP Server)](#6-tool-catalog-mcp-server)
7. [Knowledge Base: Breaking-Change Corpus](#7-knowledge-base-breaking-change-corpus)
8. [Data Quality Validation](#8-data-quality-validation)
9. [Data Models](#9-data-models)
10. [Reliability, Safety & Cost](#10-reliability-safety--cost)
11. [Observability](#11-observability)
12. [Troubleshooting Agent (Diagnose Failed / Slow Jobs)](#12-troubleshooting-agent-diagnose-failed--slow-jobs)
13. [Trade-offs & Extensions](#13-trade-offs--extensions)

---

## 1. Problem & Requirements

Upgrading a Spark application across a major version (e.g. EMR 6.x / Spark 3.1 →
EMR 7.x / Spark 3.5) traditionally takes weeks-to-months of engineering: read release
notes, fix removed/renamed APIs, resolve transitive dependency conflicts, rebuild,
re-run jobs, and prove output didn't silently change. The agent compresses this into a
supervised, mostly-automated loop.

### Functional Requirements
- **Analyze** a Spark project (build files, source, tests) and produce an **upgrade plan**.
- **Transform code** to fix build-time *and* runtime incompatibilities between source and target Spark versions.
- **Update build config & environment**: dependency versions, Scala/Java/Python versions, plugins.
- **Validate** by submitting real jobs to a target-version cluster; iterate on failures.
- **Verify data quality**: confirm upgraded job output matches the baseline (no silent semantic drift).
- **Human-in-the-loop**: user approves the plan and every code change; nothing auto-merges.
- **Resumable & observable**: progress survives restarts; status is queryable at any time.
- Support **PySpark and Scala/Java**, batch jobs first.

### Non-Functional Requirements
- **Correctness over speed** — a wrong fix that compiles is worse than no fix.
- **Idempotent & reversible** — every change is a reviewable diff in version control; safe rollback.
- **Bounded** — capped iterations / cost per phase; never loops forever.
- **Secure** — operates on customer code and data with least-privilege, auditable access.
- **Extensible** — new Spark versions = new knowledge-base entries + tools, not a rewrite.

### Scope Boundaries (non-goals)
- Not a generic refactoring tool; scoped to version-migration changes.
- Not autonomous deploy — it prepares and validates; promotion to prod is the user's call.
- Streaming / structured-streaming and ML pipelines are phase-2.

---

## 2. Why This Is Hard

| Challenge | Example |
|---|---|
| **Removed/renamed APIs** | `SQLContext` → `SparkSession`; deprecated RDD methods removed. |
| **Silent semantic changes** | Calendar switch (Julian→Proleptic Gregorian) changes date parsing; `spark.sql.legacy.*` flags. |
| **Dependency hell** | Scala 2.12→2.13, Jackson/Guava conflicts, Hadoop client bumps, connector versions. |
| **Build vs. runtime errors** | Code compiles but throws `NoSuchMethodError` only when the job runs on the cluster. |
| **Data correctness** | Output schema/values drift with no exception — only caught by comparison. |
| **Config defaults change** | AQE on by default, ANSI SQL mode, default shuffle partitions. |

The key insight: **a compile pass is necessary but nowhere near sufficient.** Real
validation requires *running the job on the target runtime* and *comparing output data*.

---

## 3. High-Level Architecture

Three planes: the **client** (where the user and the LLM live), the **control plane**
(tools + orchestration), and the **execution plane** (real Spark clusters + data).

```
┌──────────────────────────────────────────────────────────────────────┐
│  DEVELOPER ENVIRONMENT (laptop / cloud IDE)                            │
│                                                                        │
│   ┌─────────────┐   natural language    ┌──────────────────────────┐  │
│   │   User      │◀────────────────────▶│  AI Assistant (LLM agent) │  │
│   └─────────────┘   plan / diffs /      │  - reasons, calls tools   │  │
│                     approvals           │  - edits local files      │  │
│   ┌─────────────┐                       └────────────┬─────────────┘  │
│   │ Project repo │◀── reads/writes diffs ────────────┘                │
│   │ (git)        │                            │ MCP (tool calls)      │
│   └─────────────┘                             ▼                       │
└───────────────────────────────────────┬───────────────────────────────┘
                                         │  MCP Proxy (authN/Z, transport)
                                         ▼
┌──────────────────────────────────────────────────────────────────────┐
│  CONTROL PLANE  —  Managed MCP Server (Spark Upgrade Tools)            │
│                                                                        │
│  Planner  │  Build  │  CodeEdit  │  Test/Validate  │  Observability    │
│     │          │         │              │                  │           │
│     ▼          ▼         ▼              ▼                  ▼           │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │ Knowledge Base: version diffs, migration rules, fix patterns,    │  │
│  │                 dependency compatibility matrix                  │  │
│  └────────────────────────────────────────────────────────────────┘  │
│  ┌──────────────────┐  ┌──────────────────┐  ┌───────────────────┐    │
│  │ Analysis Store    │  │ Artifact Store   │  │ Data-Quality Svc  │    │
│  │ (state, history)  │  │ (S3: jars, logs) │  │ (output compare)  │    │
│  └──────────────────┘  └──────────────────┘  └───────────────────┘    │
└───────────────────────────────────────┬───────────────────────────────┘
                                         │ submit jobs / fetch logs
                                         ▼
┌──────────────────────────────────────────────────────────────────────┐
│  EXECUTION PLANE                                                       │
│   Target-version Spark cluster (EMR-EC2 / EMR-Serverless / k8s)        │
│   + sample input data  →  job runs  →  output + driver/executor logs   │
└──────────────────────────────────────────────────────────────────────┘
```

**Why this split.** The LLM stays client-side so it edits the user's actual working
tree and the user reviews every diff locally. The MCP server is **stateless per call
but state-backed** — it exposes deterministic tools and persists analysis state, so the
non-deterministic LLM does the *reasoning* while the server does the *privileged,
repeatable actions* (submit job, query logs, look up migration rules). The execution
plane is deliberately the real target runtime — the only ground truth for runtime and
data-quality correctness.

### Why MCP instead of a monolithic service
- The reasoning model is swappable and runs in the user's chosen assistant/IDE.
- Tools are individually permissioned and auditable (every call is a logged event).
- Same tools work for interactive use and batch/CI automation.

---

## 4. The Agent Loop

The agent is a **plan-act-observe loop** with a hard iteration budget per phase.

```
   ┌──────────── PLAN ─────────────┐
   │ analyze project → upgrade plan│◀───┐ (regenerate on feedback)
   └───────────────┬───────────────┘    │
                   │ user approves       │ feedback
                   ▼                     │
   ┌──────────── ACT ──────────────┐    │
   │ pick next step → call tool /   │    │
   │ edit code → produce diff       │    │
   └───────────────┬───────────────┘    │
                   ▼                     │
   ┌──────────── OBSERVE ──────────┐    │
   │ build/test/job result + logs  │    │
   └───────────────┬───────────────┘    │
                   ▼                     │
        ┌──────────────────────┐         │
        │ success?             ─┼── no ──▶│ (analyze failure, retry,
        └──────────┬───────────┘            up to N attempts)
                   │ yes
                   ▼
            next phase / done
```

**Loop invariants**
- Each iteration ends in a concrete observable: a build result, a test result, a job
  status, or a data-quality report — never "looks right."
- Failure → `fix_upgrade_failure` takes the *actual error + logs* (not a guess) and
  proposes a targeted patch, consulting the knowledge base for known signatures.
- A per-phase attempt cap (e.g. 3–5) prevents infinite loops; on exhaustion the agent
  stops and surfaces the blocker to the user with full context.
- Approval gate before any irreversible-ish action (plan acceptance; optionally each diff).

---

## 5. Workflow Phases

### Phase 0 — Prerequisites
- Project cloned locally with **git initialized** (so every change is a reviewable, revertable diff).
- A **target-version cluster** provisioned and reachable for validation.
- A **staging path** (e.g. S3) for built artifacts, logs, and the upgrade summary.

Kickoff is a natural-language prompt, e.g.:
> "Upgrade my Spark app at `./etl` from EMR 6.0 (Spark 3.0) to EMR 7.12 (Spark 3.5).
> Use cluster `j-ABC123` for validation and `s3://my-bucket/upgrade/` for artifacts."

### Phase 1 — Planning
1. Scan project structure: build files (`pom.xml`/`build.sbt`/`requirements.txt`/`setup.py`), source tree, tests, entry points, current Spark/Scala/Java/Python versions.
2. Diff source vs. target version using the knowledge base → enumerate expected changes (API, config, dependency, behavior).
3. Emit a **structured upgrade plan**: ordered steps, files likely affected, risk notes.
4. **Approval gate.** User can: accept as-is; remove steps (e.g. skip integration tests); add steps (run a specific test before validation); or constrain the approach (pin Python 3.10 + Java 17). Agent regenerates and re-asks until approved.

### Phase 2 — Compile & Build
1. `update_build_configuration`: bump Spark + transitive deps, Scala version, plugins.
2. `check_and_update_build_environment` / `check_and_update_python_environment`: align JDK / Python / venv.
3. `compile_and_build_project`: build; on failure, feed errors to `fix_upgrade_failure`, patch, rebuild. Iterate to a clean build.

### Phase 3 — Unit & Integration Tests
- If tests exist, run them post-build. On failure, iteratively edit source until green (or until the user-approved plan said to skip). Tests are the cheapest correctness signal before paying for cluster runs.

### Phase 4 — Runtime Validation
1. `run_validation_job`: submit the built artifact to the target cluster (EMR step / Serverless run / k8s submit) against sample input.
2. `check_job_status`: poll to terminal state.
3. On runtime failure (the `NoSuchMethodError`, serialization, config-default class): pull driver/executor logs, `fix_upgrade_failure` → patch → rebuild → resubmit. Iterate within budget.

### Phase 5 — Data Quality Validation
- Run baseline (source version) and upgraded (target version) jobs on the same input; compare outputs (schema, row counts, value-level). See [§8](#8-data-quality-validation). Mismatches are surfaced, not silently accepted.

### Phase 6 — Summary
- Produce a consolidated report: all code diffs, config/dependency changes, env changes, validation results, and any data-quality mismatches with explanations (e.g. "date parsing changed due to calendar switch; set `spark.sql.legacy.timeParserPolicy=LEGACY` or migrate format"). User reviews and merges.

---

## 6. Tool Catalog (MCP Server)

Tools are the **deterministic, privileged primitives** the LLM orchestrates. Grouped by phase:

| Category | Tool | Responsibility |
|---|---|---|
| **Planner** | `generate_spark_upgrade_plan` | Analyze project, produce ordered upgrade plan. |
| | `reuse_existing_spark_upgrade_plan` | Resume from a previously saved local plan. |
| **Build** | `update_build_configuration` | Rewrite `pom.xml`/`build.sbt`/`requirements.txt` deps & versions. |
| | `check_and_update_build_environment` | Verify/switch JDK & Scala for target Spark. |
| | `compile_and_build_project` | Compile/build; return structured errors. |
| **Code Edit** | `check_and_update_python_environment` | Align Python version / venv / pip deps. |
| | `fix_upgrade_failure` | Take an error + logs, return a targeted source patch. |
| **Test / Validate** | `run_validation_job` | Submit Spark app to target cluster (EMR-EC2 / EMR-S / k8s). |
| | `check_job_status` | Poll step/job-run status to terminal state. |
| **Observability** | `get_data_quality_summary` | Fetch output-comparison report for the upgrade. |
| | `describe_upgrade_analysis` | Detail a single analysis run. |
| | `list_upgrade_analyses` | List all analyses for the account/user. |

**Design notes.**
- Tools return **structured, machine-parseable results** (error class, file, line, stack
  frame) — not prose — so the model patches precisely instead of guessing.
- `fix_upgrade_failure` is the workhorse: it's where error → knowledge-base lookup →
  candidate patch happens. It's *advisory* — the LLM applies the edit to the local tree,
  keeping the human-reviewable diff in the user's repo.
- Submit/status are split (`run_validation_job` + `check_job_status`) so long cluster
  jobs are polled asynchronously rather than holding a connection.

---

## 7. Knowledge Base: Breaking-Change Corpus

The quality ceiling of the agent is its migration knowledge. The KB is curated, not
hallucinated.

**Contents (keyed by `(source_version, target_version)`):**
- **API changes**: removed/renamed/deprecated symbols with replacement patterns.
- **Config changes**: changed defaults (AQE, ANSI mode, shuffle partitions) and the legacy flags that restore old behavior.
- **Behavioral changes**: semantic shifts (calendar, null handling, decimal precision, timestamp parsing).
- **Dependency compatibility matrix**: Spark ↔ Scala ↔ Java ↔ Hadoop ↔ connector versions known to coexist.
- **Fix patterns**: error-signature → transformation, mined from release notes, migration guides, and prior successful upgrades.

**How it's used:** in Planning (predict changes), in `fix_upgrade_failure` (match error
signatures), and as **retrieval context** for the LLM (RAG) so prompts carry the exact
relevant migration rules rather than relying on the model's training memory — critical
because Spark version specifics are precise and the model must not invent APIs.

**Maintenance:** new Spark/EMR release → add a KB entry set + dependency matrix rows.
This is how the system scales to new versions without code changes. Successful and
failed upgrades feed back as new fix patterns (with human review).

> Reasoning model: use a strong, current model for the agent (e.g. a latest Claude
> model such as Opus/Sonnet 4.x) — code transformation and multi-step planning reward
> reasoning quality, and tool-use reliability matters more than raw latency here.

---

## 8. Data Quality Validation

The defining feature: catching changes that **compile, run, and produce wrong data.**

```
   same sample input
        │
   ┌────┴─────┐
   ▼          ▼
 baseline   upgraded
 (source)   (target)
  job         job
   │          │
   ▼          ▼
 output A   output B
   └────┬─────┘
        ▼
   COMPARATOR  ──▶  data-quality report
   - schema diff (added/removed/retyped cols)
   - row count delta
   - value-level diff (keyed join + per-column mismatch counts)
   - aggregate checks (sums, min/max, null rates per column)
```

**Mechanics**
- Compare on a **deterministic sample** (or full data if cheap); use a stable key for row-level alignment, and tolerance bands for floats.
- Classify diffs: *expected* (documented behavioral change → recommend legacy flag or code fix) vs. *unexpected* (likely a bad transformation → block & escalate).
- Output feeds `get_data_quality_summary`; mismatches appear in the final summary with a suggested remediation.

**Why sample-based:** full reprocessing of production data per iteration is cost-prohibitive
and slow; a representative sample catches the vast majority of semantic regressions cheaply,
with an optional full run as a final gate.

---

## 9. Data Models

```jsonc
// UpgradeAnalysis — the unit of state, one per upgrade attempt
{
  "analysisId": "ua-7f3c...",
  "project": { "path": "./etl", "lang": "scala", "vcsCommit": "abc123" },
  "source": { "emr": "6.0.0", "spark": "3.0.0", "scala": "2.12", "java": "8" },
  "target": { "emr": "7.12.0", "spark": "3.5.0", "scala": "2.12", "java": "17" },
  "validation": { "type": "EMR-EC2", "clusterId": "j-ABC", "stagingPath": "s3://.../" },
  "status": "VALIDATING",       // PLANNING|BUILDING|TESTING|VALIDATING|DQ_CHECK|DONE|BLOCKED
  "phase": "RUNTIME_FIX",
  "createdAt": "...", "updatedAt": "..."
}

// UpgradePlan
{
  "analysisId": "ua-7f3c...",
  "approved": true,
  "steps": [
    { "id": 1, "phase": "BUILD",   "action": "Bump spark-core 3.0→3.5, Jackson 2.10→2.15", "files": ["pom.xml"], "risk": "med" },
    { "id": 2, "phase": "CODE",    "action": "Replace SQLContext with SparkSession",        "files": ["src/.../Etl.scala"], "risk": "low" },
    { "id": 3, "phase": "VALIDATE","action": "Run on cluster, compare output",              "files": [], "risk": "high" }
  ]
}

// ChangeRecord — every applied edit, for the summary & audit
{
  "analysisId": "ua-7f3c...",
  "kind": "CODE|BUILD|CONFIG|ENV",
  "file": "src/main/scala/Etl.scala",
  "diff": "@@ -12,3 +12,3 @@ ...",
  "reason": "SQLContext removed in Spark 3.x; use SparkSession.builder",
  "kbRuleId": "spark-3.0-3.5/api/sqlcontext-removed",
  "appliedBy": "agent", "approved": true
}

// ValidationRun
{
  "analysisId": "ua-7f3c...", "runId": "step-9",
  "target": "EMR-EC2:j-ABC",
  "result": "FAILED",
  "errorClass": "java.lang.NoSuchMethodError",
  "logsUri": "s3://.../logs/step-9/",
  "iteration": 2
}

// DataQualityReport
{
  "analysisId": "ua-7f3c...",
  "schemaDiffs": [{ "col": "event_date", "change": "type string→date" }],
  "rowCountDelta": 0,
  "valueMismatches": [{ "col": "event_date", "mismatchRows": 1432, "cause": "calendar switch" }],
  "verdict": "MISMATCH_EXPLAINED",     // MATCH | MISMATCH_EXPLAINED | MISMATCH_UNEXPECTED
  "recommendations": ["set spark.sql.legacy.timeParserPolicy=LEGACY or migrate date format"]
}
```

---

## 10. Reliability, Safety & Cost

**Human-in-the-loop guardrails**
- Plan must be explicitly approved; optionally every code diff too.
- All edits land as git diffs in the user's tree — nothing is force-merged; rollback = `git restore`.
- Agent never deploys to production; it hands off a validated branch.

**Bounded execution**
- Per-phase attempt caps; global cost/time budget on cluster jobs (the expensive resource).
- Cheapest-signal-first ordering: static analysis → compile → unit tests → cluster run → data-quality. Fail fast before spending on cluster time.

**Idempotency & resumability**
- `UpgradeAnalysis` is durable state; `reuse_existing_spark_upgrade_plan` resumes after interruption without redoing approved work.

**Security**
- Least-privilege IAM for cluster submit + log/artifact access; private VPC endpoints for the MCP server.
- Every tool call is audit-logged (e.g. CloudTrail) with caller identity.
- Customer code/data stay in the customer's account; the model sees only what tools return.

**Failure modes & responses**
| Failure | Response |
|---|---|
| Build can't be fixed in N tries | Stop, surface error + attempted patches to user. |
| Job fails after repeated fixes | Escalate with logs; suggest manual intervention. |
| Data mismatch unexplained | Mark `BLOCKED`; do not declare success. |
| Cluster unavailable | Retry with backoff; pause analysis, keep state. |
| LLM proposes invalid edit | Build/test catches it; iteration self-corrects. |

---

## 11. Observability

- **Status anytime**: `list_upgrade_analyses` / `describe_upgrade_analysis` expose phase, current step, iteration counts, and blockers.
- **Per-iteration trace**: every tool call, error, and patch is recorded against the `analysisId` — full provenance for why each change was made (incl. `kbRuleId`).
- **Metrics worth tracking**: % upgrades fully automated, avg iterations per phase, cluster-cost per upgrade, data-quality pass rate, most-common failure signatures (→ feeds KB improvement).
- **Final summary artifact** persisted to the staging path: human-readable changelog + machine-readable change records.

---

## 12. Troubleshooting Agent (Diagnose Failed / Slow Jobs)

A sibling capability to the upgrade agent. Where the upgrade agent *transforms* code
across versions, the **troubleshooting agent** *diagnoses* a Spark job that already
failed or runs too slowly — turning hours of manual log/metric spelunking into a
conversational root-cause analysis with concrete code/config fixes.

Modeled on the [Apache Spark Troubleshooting Agent for EMR & Glue](https://docs.aws.amazon.com/emr/latest/ReleaseGuide/spark-troubleshoot.html).
It is **analysis-only** — it recommends; the user runs the job to validate fixes.

### What it shares with the upgrade agent
Same shape: client-side LLM + MCP Proxy + managed MCP server + Spark knowledge base.
It reuses the **Analysis Store**, **Observability tools**, and the **execution plane**.
The difference is the inputs (a *job/step identifier*, not a project path) and the
outputs (a *root-cause report + fix recommendation*, not a validated branch).

### Kickoff
The user points the agent at a finished run, e.g.:
> "Analyze my EMR step execution failure, step `s-XYZ` on cluster `j-ABC123`."

Inputs needed: a **failed application identifier** (EMR step id / EMR-S job-run id /
Glue job run / SageMaker notebook), plus accessible **logs, Spark History Server
event logs, and configuration**. No source-tree changes are made by the agent.

### Workflow (3 steps, iterative)

```
   job/step id
        │
        ▼
 ┌─────────────────────────────┐   tools gather telemetry from the run
 │ 1. FEATURE EXTRACTION &      │   - History/event-server logs
 │    CONTEXT BUILDING          │   - configs (executors, memory, AQE, shuffle)
 │                              │   - error traces + stage/task metrics
 └──────────────┬──────────────┘   → failure signature + perf profile
                ▼
 ┌─────────────────────────────┐   correlate features vs. Spark KB
 │ 2. ROOT-CAUSE ANALYSIS       │   → Analysis Insights (what was found)
 │    (analyze_spark_workload)  │   → Root Cause (what & why)
 │                              │   → Class: code | config | resource
 └──────────────┬──────────────┘
                ▼ (if code-related)
 ┌─────────────────────────────┐   inspect code patterns, inefficient ops
 │ 3. CODE RECOMMENDATION       │   → exact before/after snippets,
 │    (spark_code_recommendation)│     config adjustments, arch advice
 └─────────────────────────────┘
                │
                ▼
   conversational follow-up: dive deeper, refine, re-ask
```

1. **Feature Extraction & Context Building** — automatically pull telemetry: History
   Server / event logs, configuration, error traces; extract performance metrics,
   resource-utilization patterns, and a **failure signature**. Builds the context
   profile the model reasons over (so it diagnoses from real data, not guesses).
2. **Root-Cause Analysis** (`analyze_spark_workload`) — correlate the extracted
   features against the Spark knowledge base to produce: **Analysis Insights**
   (technical findings), **Root Cause** (clear what/why), and an **Initial Assessment**
   classifying the issue as **code-, config-, or resource-related** with mitigation
   guidance.
3. **Code Recommendation** (`spark_code_recommendation`, when code-related) — analyze
   the existing code, find the inefficient/incorrect operations, and emit **exact
   before/after** code plus config/architectural suggestions.

The loop is conversational and iterative — the user can drill into a specific stage,
ask "why did this shuffle spill," or apply tools interactively during local development.

### Tool catalog

| Tool | Category | Responsibility |
|---|---|---|
| `analyze_spark_workload` | Root-cause analysis | Diagnose a failed/slow Spark workload from its telemetry. |
| `spark_code_recommendation` | Code-fix recommendation | Produce concrete before/after code fixes for the diagnosed issue. |
| *(reused)* `describe_upgrade_analysis` / `list_upgrade_analyses` | Observability | Inspect/list prior analyses. |

### Diagnostic taxonomy (what root cause maps to)

| Class | Typical signatures | Example recommendation |
|---|---|---|
| **Code** | data skew, exploding joins, UDF misuse, wide transformations, `collect()` on large data | salt the skewed key / broadcast the small side / replace UDF with native expr |
| **Config** | OOM, excessive GC, too few/many partitions, AQE off, small-files | tune `executor.memory`, enable AQE, set `shuffle.partitions`, coalesce output |
| **Resource** | executor loss, disk spill, throttling, insufficient cores | resize cluster / add disk / adjust dynamic allocation |

### Where it plugs into the upgrade flow
A natural composition: when **Phase 4 (Runtime Validation)** of an upgrade fails, the
upgrade agent can hand the failed `ValidationRun` to `analyze_spark_workload` to get a
root cause, then route the result back into `fix_upgrade_failure` for a targeted patch.
The two agents share the same KB and execution plane — troubleshooting is essentially
the upgrade agent's runtime-fix step, exposed as a standalone capability for any job
(not just ones being upgraded).

### Data model addition

```jsonc
// TroubleshootingAnalysis
{
  "analysisId": "ts-9a1b...",
  "target": "EMR-EC2:j-ABC/step-s-XYZ",     // or EMR-S job-run / Glue / SageMaker
  "inputs": { "logsUri": "s3://.../logs/", "eventLogUri": "s3://.../spark-events/" },
  "features": {
    "failureSignature": "OutOfMemoryError: GC overhead limit",
    "skewRatio": 14.2, "shuffleSpillGB": 88, "maxTaskDurationMs": 920000,
    "configHighlights": { "spark.sql.adaptive.enabled": false, "executor.memory": "4g" }
  },
  "rootCause": {
    "class": "CODE",                         // CODE | CONFIG | RESOURCE
    "summary": "Skewed join on customer_id; one partition holds 40% of rows",
    "insights": ["stage 7 had 1 task running 15min vs. p50 30s", "AQE disabled"]
  },
  "recommendations": [
    { "kind": "CODE",   "before": "a.join(b, \"customer_id\")",
      "after": "a.join(broadcast(b), \"customer_id\")  // b is small", "confidence": "high" },
    { "kind": "CONFIG", "change": "spark.sql.adaptive.skewJoin.enabled=true", "confidence": "high" }
  ],
  "status": "DONE"                           // EXTRACTING | ANALYZING | RECOMMENDING | DONE
}
```

### Design notes
- **Analysis-only by default** — it never edits or redeploys; the user owns applying and
  re-running. This keeps a low-risk surface (read logs/metrics, emit advice).
- **Telemetry-grounded, not vibes** — the model reasons over extracted, structured
  features (skew ratio, spill, GC time, stage metrics), so root causes are evidence-backed
  and cite the specific stage/task; KB supplies the fix patterns via retrieval.
- **Same KB, different index** — keyed by *failure signature + metric pattern* rather than
  *(source, target) version*; successful diagnoses feed back as new signature→fix patterns.

---

## 13. Trade-offs & Extensions

**Key trade-offs**
- **Real-cluster validation vs. cost/speed.** Running actual jobs is the only true runtime signal but it's slow and costs money → mitigated by signal ordering and sampling.
- **Automation vs. control.** Full autonomy would be faster but version migrations have subtle, business-critical correctness implications → mandatory human gates.
- **LLM flexibility vs. determinism.** The model handles open-ended code changes; deterministic tools + curated KB constrain it from hallucinating APIs.
- **Sample DQ vs. full DQ.** Sampling is cheap and catches most regressions; offer full-data run as an opt-in final gate.

**Extensions**
- Structured Streaming & MLlib pipeline migrations (stateful semantics, checkpoint compatibility).
- Cross-engine migrations (e.g. Hive→Spark, Spark→Spark-on-k8s).
- CI/CD mode: run the agent headless on a PR, post the plan + diffs + DQ report as a review.
- Org-wide KB sharing: aggregate anonymized fix patterns across upgrades to raise automation rates over time.
- Performance regression check alongside data-quality (runtime/shuffle/cost deltas).

---

### One-paragraph summary
A client-side reasoning model (the agent) drives a plan→act→observe loop over a set of
deterministic, permissioned **MCP tools** backed by a curated **breaking-change knowledge
base**. It plans the upgrade, fixes build and runtime incompatibilities iteratively against
a **real target-version cluster**, and — crucially — proves correctness with **data-quality
comparison** rather than trusting a clean compile. Humans approve the plan and review every
diff; all changes are git-tracked and reversible; nothing auto-deploys. The KB, not the
code, is what scales the system to new Spark versions.
