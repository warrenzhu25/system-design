# Affirm System Design Interview Questions

System-design answers split out of [`affirm_interview_questions.md`](./affirm_interview_questions.md), which keeps the coding, concurrency/low-level design and behavioral questions.

---

## Contents

1. [A/B Testing & Experimentation Platform](#1-system-design--ab-testing--experimentation-platform)
2. [Payment Messaging & Scheduling System](#2-system-design--payment-messaging--scheduling-system)
3. [Payment Data Model & Schema Design](#3-system-design--payment-data-model--schema-design)
4. [Real-Time Fraud Detection System](#4-system-design--real-time-fraud-detection-system)

---

## 1. System Design — A/B Testing & Experimentation Platform

**Problem Statement:**
Reported directly as an Affirm system-design round (tagged `ab-testing, metrics, experiment-design`).
Design an experimentation platform that assigns users (or merchants) to experiment variants
consistently, logs exposure and outcome events, and computes metrics per variant — the kind of platform
that would sit behind decisions like "does this new BNPL checkout flow increase approval rate."

**Functional Requirements:**
- `assign(experiment_id, user_id) -> variant` — deterministic and sticky: the same user always gets the
  same variant for the lifetime of an experiment, without needing to persist a per-user record on first
  assignment.
- Log exposure events (`user was shown variant X`) and outcome/metric events (`user completed
  checkout`, `loan was approved`) tied to the experiment and variant.
- Compute per-variant aggregate metrics (conversion rate, approval rate) with statistical significance,
  on demand or on a schedule.

**Non-Functional Requirements:**
- Assignment must not add meaningful latency to the checkout path it's embedded in (sub-millisecond,
  no network round-trip on the hot path).
- No sample-ratio mismatch: the actual variant split must match the configured split (e.g., a true 50/50)
  at scale, not drift due to a bad hashing scheme.
- Metrics must be auditable back to raw exposure/outcome events for a specific experiment, given the
  regulatory scrutiny fintech decisions (like loan approval logic) draw.

**High-Level Design:**
1. **Deterministic bucketing, not stored assignment**: `variant = hash(experiment_id + user_id) % 100`
   mapped into configured variant ranges — stateless, sticky by construction (same inputs always hash the
   same way), and needs no database read on the hot path.
2. **Exposure/outcome logging as events, not row updates**: every assignment and outcome is an
   append-only event (experiment_id, user_id, variant, event_type, timestamp) shipped to a durable log
   (e.g., Kafka), decoupling the checkout hot path from the analytics pipeline entirely.
3. **Metrics computation as an offline/streaming aggregation job** over the event log, joining exposures
   to outcomes per variant and computing rates plus a significance test (e.g., a z-test on conversion
   proportions) — not computed synchronously per request.
4. **Experiment configuration service**: a small control-plane store (experiment_id, variant
   definitions, traffic split, start/end time, targeting rules) that the bucketing hash function reads,
   cached aggressively at the edge since it changes rarely relative to assignment volume.
5. **Guardrail metrics**: track a small set of "don't break these" metrics (e.g., overall approval rate,
   error rate) across every experiment, not just the ones being tested — critical in a lending context
   where an experiment regression could mean a compliance problem, not just a bad UX metric.

**Data Model (sketch):**
```
experiments(experiment_id, variants[], traffic_split, targeting_rule, start_ts, end_ts)
exposure_events(experiment_id, user_id, variant, ts)          # append-only
outcome_events(experiment_id, user_id, metric_name, value, ts) # append-only
variant_metrics(experiment_id, variant, metric_name, count, sum, computed_at)  # materialized
```

**Detailed Component Breakdown:**

| Component | Responsibility | Tech choice (typical) |
|---|---|---|
| Experiment config service + UI | CRUD for experiments: variants, split, targeting, start/end, owner, guardrails; versioned, audited | Postgres/MySQL + admin UI; changes published as immutable versions |
| Config distribution | Pushes config snapshots to every app server so assignment is local | Pub/sub (Kafka) or poll an S3/CDN JSON snapshot; in-memory cache in SDK |
| Assignment SDK (in checkout service) | `hash(salt + user_id) -> bucket -> variant`; evaluates targeting; emits exposure | Library, no network call on hot path |
| Event collector | Receives exposure + outcome events, validates schema, batches | Stateless HTTP/gRPC service -> Kafka |
| Event log | Durable, replayable, partitioned by `user_id` | Kafka (topics `exposures`, `outcomes`) |
| Stream/batch pipeline | Dedupe exposures, join exposure to outcome within attribution window, aggregate | Flink/Spark Structured Streaming + nightly batch backfill |
| Warehouse / OLAP | Raw events + per-variant aggregates, auditable | Snowflake/BigQuery (raw), ClickHouse/Druid (dashboards) |
| Stats engine | Z-test/t-test, CUPED variance reduction, SRM check, sequential testing | Python/Spark job writing `variant_metrics` |
| Monitoring/alerting | SRM alarm, guardrail breach auto-pause | Metrics store + pager |

**Step-by-Step Flow (assign -> log -> analyze):**
1. **Create**: PM defines `checkout_v2` with variants `control 50 / treatment 50`, targeting (`country=US`,
   `loan_amount>100`), primary metric (approval rate), guardrails (error rate, default-risk proxy). Config
   service stores version `v1` and publishes it.
2. **Distribute**: every checkout service instance receives the snapshot (push, with a periodic poll as a
   fallback) and keeps it in memory. A stale snapshot is safe because assignment is a pure function of config.
3. **Request arrives**: checkout calls `sdk.assign("checkout_v2", user_id)`.
4. **Targeting**: SDK evaluates targeting rules; if the user is ineligible it returns `not_in_experiment`
   and logs nothing (so ineligible users never pollute the denominator).
5. **Bucket**: `bucket = murmur3(experiment_salt + ":" + user_id) % 10000`; the variant is the range
   containing `bucket`. A per-experiment salt keeps assignments independent across experiments (otherwise
   the same users are always "treatment" everywhere, correlating results).
6. **Exposure log**: SDK emits `{experiment_id, version, user_id, variant, ts, request_id}` asynchronously
   (in-process buffer -> collector -> Kafka). Log at the point the user actually *sees* the variant
   ("exposure"), not at assignment, to avoid diluting the metric with users who never reached the feature.
7. **Outcome**: later services (`loan_approved`, `checkout_completed`) emit outcome events keyed by `user_id`.
   They don't need to know about experiments; the pipeline joins them.
8. **Pipeline**: dedupe to first exposure per `(experiment_id, user_id)`; join outcomes whose `ts >=
   first_exposure_ts` and inside the attribution window; aggregate to `(variant, metric) -> n, sum, sum_sq`.
9. **Stats**: compute lift, confidence interval, p-value; run the **SRM chi-squared** check first and
   refuse to publish results if it fails.
10. **Guardrails**: if a guardrail regresses beyond threshold, auto-pause (set traffic to 0 / kill switch in
    config) and page the owner.
11. **Audit**: any reported number can be recomputed from raw Kafka/warehouse events plus the config version.

**Design Rules Added After Failure Analysis (what keeps results trustworthy):**
1. **X1 — Declared randomization unit.** Each experiment declares its unit (user, device, merchant). Pre-login traffic
   buckets on device ID, post-login on account ID, with an identity-merge table; users seen in two variants
   ("flickers") are excluded from analysis, never silently reassigned.
2. **X2 — Immutable salt and ranges.** The experiment salt never changes after launch, and ramp-ups only *widen*
   bucket ranges (10% -> 50%) so already-assigned users keep their variant. Config validation rejects any edit that
   would move an assigned bucket.
3. **X3 — Sticky eligibility.** If targeting uses a mutable attribute (loan amount, country), a user could drift in or
   out. Evaluate targeting at first exposure and persist `(experiment, unit) -> variant` in a small exposure store; analysis
   uses the **first** exposure. Pure-hash assignment remains the default when targeting uses immutable attributes.
4. **X4 — Log exposure at render, stamp the config version.** Log when the user actually *sees* the variant, with
   `config_version`, so analysis knows exactly which rules produced each exposure.
5. **X5 — Fail safe to control.** If the SDK has no valid config, or the config is staler than a bounded age (e.g.
   5 minutes), it serves control and logs `config_missing`/`config_stale`, never an arbitrary variant.
6. **X6 — Kill switch is tested.** Emergency disable travels on a fast path (push plus short TTL), and is exercised
   regularly, since a kill switch that has never been used is an assumption.

**Detailed Failure Scenarios:**

| # | Failure | Effect on results | Detection | Handling |
|---|---|---|---|---|
| 1 | SDK cannot load config at startup | Users get no/unknown variant | `config_missing` rate | Serve control; exclude from analysis; alert; keep last-known-good on disk |
| 2 | Config push partly fails (instances on old/new versions) | Mixed split, SRM | `config_version` distribution in events; SRM check | Max-staleness SLO (X5); stamp version on events; analysis can filter by version |
| 3 | Hash salt or bucket range changed mid-experiment | Users reshuffle, contaminating both arms | Flicker rate spikes; SRM | Reject at config validation (X2); if it happened, restart the experiment, don't "fix" the analysis |
| 4 | Treatment code crashes or is slower, so exposures aren't logged | **Differential event loss**: SRM that looks like a real effect | SRM; compare server-side vs client-side exposure counts | Log exposure server-side where possible; client beacons with sequence numbers to estimate loss; treat SRM as invalid result |
| 5 | Collector or Kafka outage | Lost events, possibly asymmetric | Gaps in per-client sequence numbers; producer error metrics | Bounded local buffer, drop *events* never requests; backfill from service logs; mark the affected window |
| 6 | At-least-once duplicates | Inflated counts | Duplicate `event_id` rate | Dedupe on `event_id`; exposures deduped to first per `(experiment, unit)` |
| 7 | Late-arriving outcomes (approval decisions, repayments, defaults) | Premature readout | Metric maturity flag | Attribution windows with maturity; short-horizon proxy metrics for early read, delayed metrics (default rate) for final; never call a result before maturity |
| 8 | Bots, internal/test users | Noise, bias if filtered unevenly | Traffic-source breakdown | Apply the same filter to all arms, defined by attributes not by outcomes |
| 9 | Overlapping experiments interact | Confounded effects | Interaction test between concurrent experiments | Layers for mutually exclusive features; independent salts otherwise |
| 10 | Identity changes (anonymous to logged in) | Flickers; unit double counted | Flicker report | X1 unit rules; merge table; exclude flickers |
| 11 | Stats job fails or runs on partial data | Wrong lift shown | Row-count/completeness checks vs expected | Results marked stale, never display a partial day; reruns idempotent from raw events |
| 12 | Peeking / multiple comparisons | False positives | n/a | Sequential tests (mSPRT-style) or fixed horizon; correction across metrics (Benjamini-Hochberg) |
| 13 | Guardrail breach but alert pipeline lags | Harm continues | Guardrail latency SLO | Key guardrails computed on the streaming path; auto-pause at hard thresholds |
| 14 | Kill switch ineffective (cache TTL too long) | Bad variant keeps serving | Kill-switch drill | X6: fast-path push, short TTL for emergency flags, regular drills |
| 15 | Experiment ended but code still reads config | Unclear behavior | Stale-flag report | Defined end state (ship winner / revert to control); cleanup ticket required to close |
| 16 | Experiment affects credit decisions | Fair-lending/compliance exposure | n/a | Pre-launch compliance review gate in the config service; decision reason codes logged; no experiments on protected-class proxies without sign-off (coordinate with the compliance team) |
| 17 | Audit asks "recompute this number" | Irreproducible result | Reproducibility test | Raw events immutable; config versions retained; analysis job parameterized and versioned |

**How Real Production Systems Do It (and what to copy):**

*Uber XP — [Uber's Experimentation Platform](https://www.uber.com/blog/xp/): 1,000+ concurrent experiments, 1,000+ metrics.*
- **Statistics engine owns the math, not the experimenter.** Metrics are predefined as one of three types:
  continuous, proportion (binary) or ratio (numerator/denominator). The engine picks the test: Welch's t-test for
  continuous, Mann-Whitney U for skewed data, chi-squared for proportions; delta method or bootstrap for ratio
  metrics; Benjamini-Hochberg correction when there are 2+ treatments. Alpha = 0.05, sizing targets power = 0.8.
  **Copy**: put these in the stats-engine box of our design and make "metric type" a field in the metric registry,
  so "approval rate" (proportion) and "average APR" (continuous) are tested differently.
- **Pre-processing before testing**: outlier removal (clustering-based), **CUPED** variance reduction (use
  pre-experiment behavior as a covariate so you reach significance with less traffic) and difference-in-differences
  to correct pre-existing bias between groups. **Copy**: for BNPL, covariates such as prior approval rate or
  prior GMV; this matters because lending experiments have low base rates and long outcome delays.
- **Health checks that invalidate results**: (1) **sample-size imbalance (SRM)** when the control:treatment ratio
  differs from configured, which triggers a randomization investigation; (2) **flickers**, users who switch groups
  (e.g. moved device), who are excluded from analysis. **Copy**: log `variant` on every exposure and exclude any
  user seen in two variants.
- **Sequential monitoring**: mixture SPRT, evaluated daily on cumulative treatment-minus-control difference, with
  jackknife/block bootstrap for correlated observations from the same user. This lets you stop early without the
  "peeking" false-positive problem of repeatedly running a fixed-horizon test.
- **Outage monitoring system**: a separate sequential test on business metrics (crash rate, trip frequency) that
  alerts or auto-pauses an experiment. This is the guardrail box in our design, and it is why guardrails are
  independent of the experiment's own success metric.
- **Universal holdout**: a small configurable slice of users excluded from all experiments in a domain, used to
  measure long-term cumulative effect. **Copy**: a permanent 1-2% holdout of borrowers for checkout experiments.
- **Continuous experiments**: Thompson sampling, UCB and contextual bandits for things like picking among 10 email
  subject lines. Mention bandits as the alternative when you want to *optimize* instead of *measure*.
- Uber's post does not describe hashing/bucketing or the logging pipeline (it only says users are "randomly
  split"), so the deterministic-hash assignment and Kafka event log above are standard industry design, not
  Uber's disclosed internals.

*Affirm*: Affirm's engineering blog has a post on re-architecting its upfunnel platform to cut experiment cycle time
from months to days ([Medium](https://medium.com/affirmengineering/re-architecting-affirms-upfunnel-platform-how-we-cut-experiment-cycle-time-from-months-to-days-20a047eb6599)).
I could not read its internals, so don't cite specifics; use it as evidence that experiment velocity is a stated
Affirm priority.

**Follow-Up Questions:**
1. How do you prevent a user from being double-counted if they're exposed to an experiment twice? →
   dedupe exposure events on `(experiment_id, user_id)` at the aggregation layer — the first exposure
   timestamp is authoritative, later ones are logged but excluded from the primary metric.
2. How would you detect a sample-ratio mismatch before it invalidates results? → continuously compare
   observed variant proportions against configured splits with a chi-squared check, alerting rather than
   waiting for the end of the experiment to notice a hashing or targeting bug skewed the split.
3. How does this interact with a loan-approval model that's itself being experimented on? → the
   guardrail metrics (approval rate, default-risk proxies) need to be computed with tighter monitoring
   cadence and lower tolerance for regression than a typical UX experiment, since a skewed approval-model
   experiment has real credit-risk consequences, not just a worse click-through rate.

---

## 2. System Design — Payment Messaging & Scheduling System

**Problem Statement:**
Matches a reported system-design phone screen "focused on loan repayment and payment batching," tagged
`payment, messaging, scheduling`. Design the system that schedules and executes a borrower's future
loan-repayment installments: given a repayment schedule (e.g., 4 biweekly installments), reliably
initiate each payment attempt on its due date, handle retries on failure, and notify the borrower and
internal systems of outcomes — without ever double-charging or silently dropping a scheduled payment.

**Functional Requirements:**
- Given a loan's repayment schedule, enqueue each installment for execution at its due date/time.
- On due date, attempt the payment via a payment processor; on failure, retry with backoff up to a
  configured attempt limit, then flag for collections/dunning.
- Publish payment-outcome events (`succeeded`, `failed`, `retrying`) that other services (notifications,
  loan-balance ledger, dunning) consume.

**Non-Functional Requirements:**
- **Exactly-once effective execution**: a scheduler crash/restart or duplicate message must never result
  in the same installment being charged twice.
- Scheduled payments must fire within a tight window of their due time (minutes, not hours) even under
  scheduler restarts or partial outages.
- Every payment attempt (and its outcome) must be durably logged for audit and dispute resolution.

**High-Level Design:**
1. **Durable schedule store**: each installment is a row (`loan_id, installment_id, due_ts, status,
   attempt_no`) in a database, not just an in-memory timer — a scheduler process that only holds
   timers in memory loses every pending payment on crash.
2. **Polling/dispatch loop with a due-time index**: a worker polls `WHERE status = 'PENDING' AND due_ts
   <= now()` (indexed on `due_ts`), claims a batch (e.g., via `SELECT ... FOR UPDATE SKIP LOCKED` or an
   equivalent claim mechanism), and dispatches each to the payment-execution path.
3. **Idempotent payment execution**: every attempt carries an idempotency key
   (`loan_id:installment_id:attempt_no`) passed through to the payment processor, so a retried request
   after an ambiguous failure (timeout, not an explicit decline) can't double-charge — the processor itself
   de-dupes on that key. **`attempt_no` is a logical attempt**: it advances *only* after the previous attempt
   reached a definitive "no money moved" outcome. A worker crash, a lease expiry or a redelivered message
   must **never** produce a new key (see "Attempt Rules" below — getting this wrong is the classic
   double-charge bug).
4. **Atomic state transition, events for everything downstream**: the executor records the attempt result
   *and* the installment's state transition in **one transaction** together with an outbox row, so there is
   no window where the attempt is terminal but the installment still says `PROCESSING`. Ledger,
   notifications and dunning consume the outbox event asynchronously and idempotently, so they can lag or be
   down without affecting charging correctness.
5. **Retry with backoff, bounded**: on a *definitive* retryable failure (insufficient funds flagged as
   "try later"), set `due_ts` to a later time and advance `attempt_no`; past the configured max attempts,
   transition to a terminal `FAILED` status and hand off to dunning rather than retrying forever. Our own
   infrastructure failures (processor outage, open circuit breaker) use a separate counter and do not consume
   the borrower's attempts.

**Data Model (sketch):**
```
installments(installment_id PK, loan_id, seq, due_ts, amount_due, status, attempt_no, infra_retry_count,
             claim_token, lease_expires_at, version, UNIQUE (loan_id, seq))
payment_attempts(idempotency_key PK, installment_id, attempt_no, status, request_fingerprint,
                 processor_ref, created_at, lease_until, result_code, UNIQUE (installment_id, attempt_no))
outbox(event_id PK, aggregate_id, type, payload, published_at NULL)
```
State machines:
- Installment: `SCHEDULED -> PROCESSING -> PAID | SCHEDULED (retry, later due_ts) | FAILED | PENDING_SETTLEMENT`;
  `PENDING_SETTLEMENT -> PAID | RETURNED`; `PAID -> RETURNED` (ACH return / chargeback).
- Attempt: `IN_FLIGHT -> SUCCEEDED | DECLINED_HARD | FAILED_RETRYABLE | PENDING_SETTLEMENT | UNKNOWN`;
  `UNKNOWN -> (resolved to one of the above)`. Transitions are monotonic and guarded by `WHERE status IN (...)`
  so a late or duplicate writer can't move state backwards.

**Attempt Rules (the invariants that make double-charging impossible):**
1. **R1 — Key stability.** The idempotency key is derived from `(loan_id, installment_id, attempt_no)`. `attempt_no`
   advances only when the current attempt is `FAILED_RETRYABLE` (the processor definitively said
   no money moved).
2. **R2 — Claiming never changes the key.** The dispatcher sets `PROCESSING`, `claim_token` and a lease; it does
   **not** increment `attempt_no`. Redelivery and lease recovery therefore reuse the same key.
3. **R3 — Resolve before re-driving.** If the current attempt is `IN_FLIGHT`, `UNKNOWN` or terminal-but-not-applied,
   recovery first **resolves** it (look up the key at the processor / replay the stored result). It never starts a
   fresh charge under a new key.
4. **R4 — Never "retry" an UNKNOWN with a new key.** An unresolved attempt stays `UNKNOWN`, is escalated, and blocks
   further charging of that installment.
5. **R5 — Mind the processor's key-retention window.** Processors remember idempotency keys only for a limited time
   (often around 24 hours; check your processor's documentation). Resolve `UNKNOWN` attempts well inside the window
   (e.g. a 4-hour SLA). Past it, do not re-send the same key; resolve through the processor's search by our metadata
   reference (`installment_id`) or the settlement report first.
6. **R6 — Request fingerprint.** Store `hash(amount, currency, method)` with the key. Reusing a key with different
   parameters is rejected locally (and by the processor) instead of silently charging a different amount.
7. **R7 — Write-ahead intent.** The `IN_FLIGHT` row is committed (and synchronously replicated) **before** the
   processor call, so a crash can never leave a charge with no local record.
8. **R8 — Re-validate at execution time.** In the same pre-call transaction, re-read the installment (`version`,
   `status`, `amount_due`) and the loan state. If the loan was cancelled, refunded, paid off early, rescheduled or the
   amount changed, abort without calling the processor.
9. **R9 — Infrastructure failures are not borrower failures.** Processor 5xx/timeouts-with-definitive-no-charge and
   circuit-breaker rejections increment `infra_retry_count` (own cap and alert), not `attempt_no`.
10. **R10 — No cross-processor failover mid-attempt.** Failing over to a second processor while the first attempt is
    unresolved can charge twice, because the two processors share no idempotency state. Resolve first, then route.

**Detailed Component Breakdown:**

| Component | Responsibility | Tech choice (typical) |
|---|---|---|
| Loan servicing API | Creates loan + installment schedule at origination; handles reschedule/early payoff | Stateless service + Postgres |
| Schedule store | Source of truth for `installments` (status, due_ts, attempt_no) | Postgres/MySQL, index on `(status, due_ts)`, partitioned by due date |
| Scheduler / dispatcher | Polls due rows, claims a batch, enqueues work | N workers using `FOR UPDATE SKIP LOCKED` or lease column |
| Work queue | Buffers payment jobs, absorbs due-date spikes (e.g. the 1st of month) | Kafka / SQS |
| Payment executor | Calls processor with idempotency key; classifies the result | Stateless workers |
| Payment method / processor adapter | Cards, ACH, debit; each has different latency and failure semantics | Adapter per rail (card network, ACH bank partner) |
| Attempt log | Append-only record of every attempt and outcome (audit) | Same DB table + Kafka topic |
| Outbox relay + outcome consumers | Publish committed results; ledger/notification/dunning consume idempotently | Relay process + Kafka consumers |
| Ledger service | Double-entry record of money movement (see #3) | Append-only DB |
| Notification service | Email/push: upcoming payment, success, failure | Consumes outcome events |
| Dunning / collections | Handles terminal failures: grace period, late fee, outreach | Workflow engine / state machine |
| Reconciliation job | Matches processor settlement files to the ledger and flags breaks | Nightly batch |

**Step-by-Step Flow (one installment, happy path):**
1. **Origination**: loan is approved; servicing API writes `loans` plus 4 `installments` rows
   (`status=SCHEDULED`, `due_ts`, `amount`) in **one transaction**.
2. **Pre-notify** (T-3 days): a scan finds `due_ts` within 3 days and emits `payment.upcoming`; notification
   service emails the borrower (also a regulatory courtesy).
3. **Due time**: a dispatcher polls
   `SELECT ... WHERE status='SCHEDULED' AND due_ts<=now() ORDER BY due_ts LIMIT 500 FOR UPDATE SKIP LOCKED`.
4. **Claim**: in the same transaction it sets `status=PROCESSING`, a fresh `claim_token` and `lease_expires_at`
   (using the database clock) and commits. It does **not** touch `attempt_no` (R2). The queue message
   `{installment_id, claim_token}` is written through the outbox in that same transaction.
5. **Execute**: the executor re-validates the installment (R8), builds `idempotency_key =
   loan_id:installment_id:attempt_no` from the *stored* `attempt_no`, inserts the `payment_attempts` row
   (`IN_FLIGHT`, with fingerprint and lease) and commits durably **before** calling out (R7), then calls the
   processor.
6. **Result**: the processor returns `SUCCEEDED`, `DECLINED_HARD`, `FAILED_RETRYABLE` (e.g. insufficient funds) or
   (for ACH) `PENDING_SETTLEMENT`, with settlement confirmed days later by a webhook or return file. A **timeout is
   none of these**: it is `UNKNOWN` until resolved (R3-R4).
7. **Record + transition atomically**: in one transaction the executor updates the attempt (guarded by `WHERE
   status IN (IN_FLIGHT, UNKNOWN)`), applies the installment transition (`PAID`, `SCHEDULED` with new `due_ts`,
   `FAILED`, `PENDING_SETTLEMENT`) guarded by `version`, and inserts the outbox event
   `payment.succeeded|failed|pending`. The outbox relay publishes it (at-least-once).
8. **Apply downstream**: the ledger consumer (idempotent on `idempotency_key`) posts entries (debit cash / credit loan
   receivable); notification and dunning consumers react. None of them can change charging state.
9. **Retry path**: on a definitive `FAILED_RETRYABLE`, set `status=SCHEDULED`, `attempt_no += 1`,
   `due_ts = now + backoff(attempt_no)` (e.g. 1d, 3d, 5d), up to a max; at the limit set `FAILED` and emit
   `payment.exhausted` to dunning.
10. **Lease recovery (the corrected protocol)**: a sweeper selects `PROCESSING` rows with `lease_expires_at < now()`
    and inspects the *current* attempt: (a) no attempt row -> safe to re-queue with a new `claim_token`, same
    `attempt_no`; (b) `IN_FLIGHT`/`UNKNOWN` -> enqueue a **resolve** job (R3), never a fresh charge; (c) terminal
    but installment not transitioned -> apply the stored result. Because the key is unchanged, even a racing
    original executor is harmless: both hit the same processor key.
11. **Reconciliation** (nightly): compare processor settlement report vs `payment_attempts`/ledger; any
    mismatch (charged but we show failed, or the reverse) opens a ticket and posts a correcting ledger entry.

**Detailed Failure Scenarios:**

*A. Scheduler worker (dispatcher) and sweeper*

| # | Failure | What state is left behind | Handling | Residual risk |
|---|---|---|---|---|
| D1 | Crash before the claim commits | Nothing changed | Next poll by any worker picks the row up | None |
| D2 | Crash after the claim commits, before the message is delivered | Row `PROCESSING`, no message in flight | Message is written via the **outbox in the claim transaction**, so the relay delivers it after restart; the lease sweeper is the second net | Payment late by at most one lease period |
| D3 | Message delivered twice (relay retry, queue redelivery) | Two executors for one claim | Executor checks `claim_token` + attempt row; second one sees `IN_FLIGHT`/terminal and exits or replays | None, same key either way |
| D4 | **Slow executor outlives its lease; sweeper re-drives** | Attempt may be `IN_FLIGHT` with a charge in progress | Sweeper inspects the attempt (rule R3): enqueues **resolve**, not a new charge; any second executor reuses the **same key** (R2) | Needs the processor to honour key dedupe (R5 window) |
| D5 | Sweeper resets a row whose attempt is terminal but not applied (e.g. relay lag) | Attempt `SUCCEEDED`, installment `PROCESSING` | Atomic record+transition in the executor (step 7) removes this window; if it still occurs, sweeper applies the stored result | None |
| D6 | Dispatcher reads from a lagging replica | Row looks unclaimed | Claims are done only on the primary; reads for display may use replicas | None if enforced |
| D7 | Clock skew between workers | Rows appear due too early or late | All due and lease comparisons use the **database clock** (`now()`), never worker clocks | None |
| D8 | Scheduler outage, then recovery | Large backlog of due rows | Normal poll catches up, ordered by `due_ts`; batch size cap and per-processor rate limit prevent a thundering herd; borrower-facing grace avoids late fees caused by our outage | Charges are late; policy must not penalize borrowers |
| D9 | Poison row (always fails before reaching the executor) | Retried forever, blocks batch head | Per-row dispatch attempt cap -> quarantine state + alert; claim query skips quarantined rows | Needs a human |
| D10 | Two sweepers run concurrently | Duplicate recovery work | Sweeper claims rows with `SKIP LOCKED`; recovery actions are idempotent | None |
| D11 | Hot partition at month start (1st/15th) | Index contention, long scans | Partition schedule by `due_date`; shard claims by `hash(installment_id) % N` worker lanes; jitter charges within the allowed window | Capacity planning |

*B. Executor and processor interaction*

| # | Failure | Money moved? | Handling |
|---|---|---|---|
| E1 | Crash before the `IN_FLIGHT` row commits | No | Redelivery starts clean; nothing to undo |
| E2 | Crash after `IN_FLIGHT` commits, before the processor call | No | Resolve finds "not found" at the processor -> re-send **same key** (within retention window) |
| E3 | **Crash during or after the processor call, before the result is recorded** | **Maybe** | Resolve: `lookup(key)`; found -> record result; not found within window -> re-send same key; else `UNKNOWN` + escalate (R3-R5) |
| E4 | Processor call times out | Maybe | Same as E3. Never treat a timeout as failure; never mint a new key (R4) |
| E5 | Processor returns 5xx / is unreachable (definitive no charge) | No | `infra_retry_count++`, backoff, circuit breaker; borrower attempt not consumed (R9) |
| E6 | Processor returns a soft decline (insufficient funds) | No | `FAILED_RETRYABLE`: advance `attempt_no`, schedule retry per policy |
| E7 | Processor returns a hard decline (closed account, fraud) | No | `DECLINED_HARD`: stop retrying, prompt borrower to update payment method, notify |
| E8 | Decline code misclassified | Depends | Maintain an explicit code->class table with a default of "UNKNOWN: do not retry automatically"; alert on unmapped codes. Misclassifying hard as soft keeps charging a bad instrument; the reverse sends borrowers to collections early |
| E9 | ACH/bank rail accepts the debit (`PENDING_SETTLEMENT`) | Pending | No retry until a final result; webhook or return file settles it; installment shows pending, not paid |
| E10 | ACH return arrives days later (NSF, unauthorized) | Reversed | Reversal event -> installment `RETURNED`, reversing ledger entry, retry/dunning per return code (some codes forbid re-presentment) |
| E11 | Same key sent with different amount/method | n/a | Local fingerprint check (R6) rejects; processor also errors |
| E12 | Processor key-retention window expires while `UNKNOWN` | Maybe | Do not re-send the old key; search by metadata reference / settlement report; hold until resolved (R5) |
| E13 | Failover to a second processor while the first is unresolved | Maybe | Forbidden until resolved (R10); only new installments route to the new processor |
| E14 | Executor crashes after the result is recorded but before downstream effects | Yes | Outbox event already committed with the result; relay publishes it later |

*C. Events, ledger and downstream*

| # | Failure | Handling |
|---|---|---|
| C1 | Outbox relay down | Events accumulate in the outbox table; relay resumes and publishes in order per aggregate; alert on outbox age |
| C2 | Duplicate event delivery | Consumers dedupe on `event_id`/`idempotency_key`; ledger has a unique constraint on the key |
| C3 | Events arrive out of order (`pending` after `succeeded`) | State transitions are monotonic and version-guarded; stale events are ignored |
| C4 | Ledger service down | Charging continues; ledger consumer catches up from the topic; reconciliation compares ledger totals to attempts |
| C5 | Notification provider down | Independent retry with its own DLQ; never blocks charging; a missed "payment failed" notice triggers a re-send job |
| C6 | Poison event | After N failures -> DLQ + alert; partition keeps moving |

*D. Database, queue and platform*

| # | Failure | Handling |
|---|---|---|
| P1 | Primary DB failover | Promote the synchronous replica (no acknowledged `IN_FLIGHT` row is lost, which is why R7 requires sync commit); fence the old primary; executors reconnect and resolve any open attempts |
| P2 | Split-brain (two primaries) | Fencing tokens/leases at the storage layer; claim and attempt writes carry `version` so a stale primary's writes fail |
| P3 | Queue/Kafka unavailable | Outbox keeps messages durably; sweeper re-drives due rows; no charge depends on the queue being up |
| P4 | Executor deployment bug charges wrong amount | Amount is read from the installment in the same transaction as the intent row; invariant alert: `charged_amount != amount_due`; kill switch pauses dispatch; automatic refund workflow for confirmed over-charges |
| P5 | Replica/analytics lag shows stale status to support | UI labels data freshness; support actions read from the primary |
| P6 | Region outage | Pending schedule is in a replicated store; failover region resumes polling; all open attempts go through resolve; processor-side keys protect against re-sends |

*E. Business-rule races*

| # | Race | Handling |
|---|---|---|
| B1 | Borrower pays early or in full while a debit is in flight | R8 check before the call; if the debit is already in flight, the ledger records an overpayment and an automatic refund/credit is issued |
| B2 | Loan cancelled or merchant refund processed while due | `version` bump on the installment; executor aborts (R8); refund path posts reversing entries |
| B3 | Borrower changes payment method between attempts | New method applies to the next `attempt_no` only; the old attempt keeps its key and fingerprint |
| B4 | Reschedule requests (change due date) | Update `due_ts` with a `version` check; if `PROCESSING`, the request is queued until the attempt resolves |
| B5 | Due date vs time zone/business days | Store `due_ts` in UTC computed from the borrower's local date; ACH cutoffs and bank holidays adjust the effective date; tests around DST |
| B6 | Duplicate schedule rows from a retried origination call | `UNIQUE (loan_id, seq)` and idempotent origination |
| B7 | Regulatory limits on re-presentment/retries | Policy table per rail/return code, enforced in the retry scheduler, with audit |

*F. Safety nets and detection*

- **Invariant monitors** (page on breach): more than one `SUCCEEDED` attempt for an installment; processor settlement
  with no local attempt row; installment `PROCESSING` longer than 2x the lease; any attempt `UNKNOWN` older than the SLA;
  outbox oldest unpublished age; charge amount differs from amount due; sudden spike in a decline code.
- **Reconciliation (nightly, plus intra-day for card rails)**: three-way match of processor settlement files, local
  attempts and ledger entries. Breaks open tickets; a confirmed **duplicate charge** triggers an automatic refund and
  an incident review.
- **Kill switches**: pause dispatch globally, per processor, per merchant, or per rail without redeploying.
- **Game days**: kill executors mid-call, drop the processor connection after the request is sent, expire leases
  artificially, and replay duplicate messages; the expected result is zero duplicate settlements.

**How Real Production Systems Do It (and what to copy):**

*Stripe — [idempotency](https://stripe.com/blog/idempotency).* The client sends a unique key
(`Idempotency-Key: AGJ6FJMkGQIpHUTX`) with each charge. The server correlates the key with the state of that
request and handles three failure shapes: (1) **connection failure before reaching the server**: the retry is the
first time the server sees the key, so it just runs; (2) **failure mid-operation**: if the earlier attempt was rolled
back in an ACID database it is safe to rerun fully, otherwise the server recovers state and continues; (3) **response
lost after success**: the server returns the cached result of the first success instead of charging again. Clients
retry until success is *verified*, with exponential backoff (about 2^n) plus jitter so recovering servers aren't hit by
a synchronized herd. Stripe also separates **definitive** failures (could not connect, safe to retry) from
**ambiguous** ones (connection dropped mid-exchange, safe only because of the key). **Copy**: our executor treats
a timeout as *ambiguous* and always reuses the same key.

*Airbnb — payment orchestration and the Orpheus idempotency framework
([Rebuilding Payment Orchestration](https://airbnb.tech/?p=49), summarized in
[this write-up](https://biweekly-engineering.beehiiv.com/p/handling-double-payments-airbnb-biweekly-engineering-episode-32)).*
Every payment call is split into three phases:
1. **Pre-RPC**: look up the idempotency key in the **master** database; if that request already completed, return
   its stored result; if not, insert the new request record.
2. **RPC**: call the payment service provider (PSP), outside any open DB transaction.
3. **Post-RPC**: write the PSP's response, or the failure, back to the database so a later retry can decide.
Errors are classified: **5xx = retryable, 4xx = non-retryable** (the request itself is wrong). Idempotency reads and
writes always go to the primary, never a replica, because replica lag could let a duplicate slip through. Larger
workflows are modeled as a **DAG of retryable, idempotent steps**; Airbnb reports five 9s (99.999%) payment
consistency after the rebuild. **Copy** the three-phase executor, shown next.

*Executor sketch assembled from those two designs, with the Attempt Rules applied (pseudo-code, not either company's source):*
```
def execute(msg):                                   # msg = {installment_id, claim_token}
    # PRE-RPC: one transaction on the primary, committed with synchronous replication (R7)
    with db.txn(sync_commit=True):
        inst = installments.get(msg.installment_id, for_update=True)
        if inst.status != PROCESSING or inst.claim_token != msg.claim_token: return   # stale/duplicate message
        if loan_changed(inst):                                                         # R8
            installments.release(inst); return
        key = f"{inst.loan_id}:{inst.installment_id}:{inst.attempt_no}"                # R1: stored attempt_no
        row = attempts.get(key, for_update=True)
        if row and row.status in TERMINAL: return apply_stored_result(inst, row)       # replay, no processor call
        if row and row.status in (IN_FLIGHT, UNKNOWN): return resolve(inst, row)       # R3: never re-charge blindly
        attempts.insert(key, status=IN_FLIGHT, fingerprint=hash(inst.amount_due, method),
                        lease_until=db.now() + 60s)                                    # R6, R7

    # RPC: no DB transaction held open across the network
    try:
        result = processor.charge(key, inst.amount_due, method, metadata={"installment_id": inst.installment_id})
    except (Timeout, ConnectionError):
        result = resolve_with_lookup(key)           # FOUND(result) | NOT_FOUND | UNKNOWN

    # POST-RPC: attempt result + installment transition + outbox, atomically
    with db.txn():
        finalize(inst, key, result)

def resolve(inst, row):
    r = processor.lookup(row.key)
    if r.found:                      return finalize(inst, row.key, r.result)
    if r.not_found and row.age < KEY_RETENTION_SAFE and processor.lookup_is_authoritative:
        return redo_charge_same_key(inst, row)      # same key => cannot double charge
    attempts.mark(row.key, UNKNOWN); escalate(row)  # R4/R5: no new key, no new charge

def finalize(inst, key, result):
    cls = classify(result)                          # unmapped codes -> UNKNOWN (E8)
    attempts.update(key, status=cls) if attempts.status(key) in (IN_FLIGHT, UNKNOWN)   # monotonic
    installments.transition(inst, cls, expected_version=inst.version)   # PAID | SCHEDULED(attempt_no+1 on FAILED_RETRYABLE)
                                                                         # | FAILED | PENDING_SETTLEMENT; infra errors -> infra_retry_count++
    outbox.insert(event_for(cls))
```

*Uber — [Payments Platform](https://www.uber.com/blog/ubers-payments-platform).* Uber splits the money lifecycle into
**stateless microservices** (money order creation, processing, collection, disbursement) joined by **Kafka**, with two
shared stores as sources of truth (the money-order store and the entity-balance store). Flows where a rider is
waiting use **Cadence**, Uber's workflow engine, so the interactive path is a durable workflow while the backbone stays
asynchronous. Uber Pay integrates 50+ payment methods behind a generic interface (charge, disburse, refund) so rail
specifics stay out of the core. **Copy**: (a) the stage-per-service split maps to our dispatcher / executor /
outcome consumer / ledger; (b) a durable-workflow engine (Cadence/Temporal) is a legitimate alternative to the
polling dispatcher: one workflow per installment, a durable timer until `due_ts`, activities for charge and retry, and
the workflow history is itself the audit trail. Trade-off to say out loud: polling is simpler to operate and easy to
reason about at 100M installments; workflows give clearer per-installment logic but add an engine to run.

*Affirm*: Affirm's postings for its Repayment Scheduling team describe it as the system that lets users schedule
automatic payments to repay loans, at tens of billions of dollars a year. I found no public Affirm design write-up,
so the above is the industry pattern; present it as such.

**Follow-Up Questions:**
1. How do you avoid two dispatch workers claiming and double-charging the same installment? →
   `SKIP LOCKED`-style claiming (or an equivalent leased/claimed-row mechanism) makes a claim atomic across
   workers; the idempotency key on the actual charge is the second, independent line of defense if a claim
   ever races anyway.
2. What happens if the payment processor call times out — did it charge or not? → treat a timeout as
   "unknown," not "failed" — query the processor's own idempotency-key lookup (or a status endpoint)
   before deciding to retry, rather than blindly retrying and relying on the processor's dedup alone.
3. How do you handle a due date landing on a scheduler outage window? → the durable due-time index means
   a recovered scheduler's next poll picks up every installment whose `due_ts` has passed, regardless of
   how long the outage was — the design should explicitly call out that "catch-up" is a normal poll, not
   a special recovery path. Borrower-facing policy (grace period, no late fee for our outage) is part of the answer.
4. A worker is slow and its lease expires while the charge is in flight. How do you avoid a second charge? →
   the lease only authorizes *re-driving*; recovery inspects the current attempt, never mints a new key (R2-R4),
   and any second executor reuses the same key, so the processor returns the original result. The unsafe design is
   the one where claiming increments the attempt number and therefore changes the key.
5. What if the processor forgets idempotency keys after a day? → resolve `UNKNOWN` attempts inside the retention
   window, and after that look up by our metadata reference or the settlement file before any new charge (R5).
6. How do you prove it works? → invariant monitors (one success per installment), three-way reconciliation, and
   fault-injection tests that kill executors mid-call and expire leases, expecting zero duplicate settlements.

---

## 3. System Design — Payment Data Model & Schema Design

**Problem Statement:**
Reported as a distinct Affirm system-design round (tagged `payment, data-modeling, schema-design`),
separate from the messaging/scheduling round above and focused purely on **data modeling**: design the
core schema for representing loans, installments, payments, and merchants such that it supports
accurate balance computation, partial payments, refunds, and multi-merchant reconciliation, under a
schema that will inevitably need to evolve.

**Functional Requirements:**
- Represent a loan's principal, its installment schedule, and every payment applied against it,
  including partial payments and refunds.
- Support querying a loan's current outstanding balance at any point in time, and reconstructing that
  balance as of any past date (for disputes/audits).
- Support a merchant having many loans across many borrowers, and a borrower having many loans across
  many merchants.

**Non-Functional Requirements:**
- Balance computation must be correct even under out-of-order payment application (e.g., a delayed ACH
  settlement recorded after a later payment already posted).
- Schema changes (new payment types, new fee structures) shouldn't require rewriting historical records.
- Auditable: every balance must be reconstructable from an immutable trail, not just a mutable running
  total.

**High-Level Design:**
1. **Event-sourced ledger, not a mutable balance column**: `loans.outstanding_balance` as a single
   mutable field is the tempting-but-wrong default — instead, every balance-affecting action (charge
   applied, payment received, refund issued, fee assessed) is an immutable `ledger_entry` row, and the
   current balance is a **derived** sum, not stored truth.
2. **Core entities**: `merchants`, `loans` (principal, term, merchant_id, borrower_id), `installments`
   (loan_id, due_ts, scheduled_amount), `ledger_entries` (loan_id, entry_type, amount, effective_ts,
   recorded_ts, reference_id) — note the **two timestamps**: `effective_ts` (when it actually happened,
   e.g. the ACH settlement date) vs. `recorded_ts` (when the system learned about it), which is exactly
   what makes out-of-order settlement correct.
3. **Balance as of any date** = sum of `ledger_entries` with `effective_ts <= target_date`, independent
   of `recorded_ts` — this is precisely how a delayed-ACH entry, recorded late but effective earlier,
   correctly retroactively adjusts a historical balance snapshot instead of corrupting today's balance.
4. **Materialized current-balance cache**: for read performance, maintain a `loan_balances` table updated
   by a trigger/consumer on `ledger_entries` insert — but treat it explicitly as a cache rebuildable from
   the ledger, never as the source of truth.
5. **Schema evolution via entry_type + a flexible metadata column** (e.g., JSON) on `ledger_entries`
   rather than adding new top-level columns per new fee/payment type — new entry types are additive, and
   historical entries never need migration.

**Data Model (sketch):**
```
merchants(merchant_id, name, ...)
loans(loan_id, merchant_id, borrower_id, principal, term, created_ts)
installments(loan_id, installment_id, due_ts, scheduled_amount)
ledger_entries(entry_id, loan_id, entry_type, amount, effective_ts, recorded_ts, reference_id, metadata)
loan_balances(loan_id, outstanding_balance, last_entry_id)  # materialized cache, rebuildable
```

**Detailed Schema & Components:**

```sql
-- Reference data
merchants(merchant_id PK, name, settlement_account_id, status)
borrowers(borrower_id PK, ...)
accounts(account_id PK, owner_type, owner_id, account_type)   -- LOAN_RECEIVABLE, CASH, MERCHANT_PAYABLE,
                                                              -- INTEREST_INCOME, FEE_INCOME, REFUND_LIABILITY
-- Contract
loans(loan_id PK, borrower_id, merchant_id, principal, apr, term_count, status, created_ts)
installments(installment_id PK, loan_id FK, seq, due_date, principal_due, interest_due, fee_due, status)
-- Money movement (immutable)
transactions(txn_id PK, loan_id, txn_type, effective_ts, recorded_ts, idempotency_key UNIQUE, reverses_txn_id NULL)
ledger_entries(entry_id PK, txn_id FK, account_id FK, direction CHECK IN ('DEBIT','CREDIT'),
               amount_minor BIGINT, currency, installment_id NULL, metadata JSONB)
-- Derived
account_balances(account_id PK, balance_minor, as_of_entry_id)   -- cache, rebuildable
```

Invariants enforced in code and by DB constraints:
- For each `txn_id`, `SUM(debits) = SUM(credits)` (zero-sum, double entry), checked before commit.
- Rows are insert-only; **corrections are new reversing transactions** (`reverses_txn_id`), never updates.
- Money is integer minor units (`amount_minor`), never floats; one currency per transaction.
- `idempotency_key` is unique, so a retried "apply payment" cannot post twice.

**Components around the schema:**

| Component | Role |
|---|---|
| Ledger service | The only writer of `transactions`/`ledger_entries`; validates zero-sum and idempotency |
| Allocation engine | Decides how a payment splits across fees -> interest -> principal -> oldest installment |
| Balance projector | Consumes ledger inserts, maintains `account_balances` and per-installment status |
| Statement/query API | Point-in-time balance, payoff quote, payment history |
| Reconciliation service | Compares ledger with processor files and merchant settlement reports |
| Data warehouse sink | CDC (Debezium -> Kafka -> warehouse) for finance and analytics |

**Step-by-Step Flow (borrower pays $30 of a $30 installment, with a late fee):**
1. Payment succeeds at the processor (event from #2 with `idempotency_key`).
2. Ledger service opens a DB transaction and checks `idempotency_key`; if present, return the existing `txn_id`.
3. Allocation engine reads the installment (fee $5, interest $2, principal $23) and allocates the $30
   fees -> interest -> principal.
4. Ledger posts one `transactions` row and balanced entries: `DEBIT CASH 30`, `CREDIT FEE_INCOME 5`,
   `CREDIT INTEREST_INCOME 2`, `CREDIT LOAN_RECEIVABLE 23`. Zero-sum check passes; commit.
5. Projector updates `account_balances` and marks the installment `PAID` (idempotent via `as_of_entry_id`).
6. Balance as of date `D`: `SUM(signed amount) WHERE account=LOAN_RECEIVABLE AND loan_id=? AND effective_ts<=D`.
7. **Late ACH settlement**: entry is recorded on day 5 with `effective_ts` = day 2; projections for day 2..5 are
   recomputed from the ledger; `recorded_ts` shows when we knew, which answers "what did we tell the borrower".
8. **Refund**: merchant refunds $50 of a $100 loan; post a reversing transaction (credit receivable, debit
   merchant payable), then recompute the remaining schedule via a *new* schedule version.
9. **Settlement to merchant**: group entries by `merchant_id` and period, produce a settlement batch, pay out,
   post `DEBIT MERCHANT_PAYABLE / CREDIT CASH`.

**Scaling notes:** shard by `loan_id` (all of a loan's entries co-locate, so a single-shard transaction posts
the whole transaction); hot merchant accounts that aggregate many loans should not be a single balance row, so
use per-loan accounts plus an aggregated view, or batch-serialized writes for hot accounts.

**Posting Rules Added After Failure Analysis:**
1. **L1 — One transaction posts a whole business event.** `transactions` and all its `ledger_entries` commit together or
   not at all. A deferred constraint (or trigger) verifies `SUM(debits) = SUM(credits)` per transaction at commit, in
   addition to the application check.
2. **L2 — Serialize postings per loan.** Take a row lock on the loan (or route all writes for a `loan_id` to one partition
   writer) while allocating a payment, so two concurrent payments cannot both allocate to the same installment. Use
   `version` checks for optimistic retries.
3. **L3 — Deterministic allocation and rounding.** Money is integer minor units. Allocation order (fees -> interest ->
   principal -> oldest installment) is code-versioned; remainders use largest-remainder rounding; the engine version
   is stored with each transaction so results are reproducible.
4. **L4 — Immutable, insert-only.** The application role has `INSERT`/`SELECT` only. Corrections are reversing
   transactions that reference `reverses_txn_id`; a constraint forbids reversing more than the original amount.
5. **L5 — Direction is explicit.** Each entry stores `direction` and `account_id`, so balance sums never depend on
   inferring sign from `entry_type`; a new entry type cannot silently flip a sign.
6. **L6 — PII stays out of the ledger.** Entries reference tokens/ids; personal data lives in a separate store so
   deletion requests can be honored without rewriting financial history.
7. **L7 — Derived data is rebuildable.** `loan_balances`, installment statuses and merchant settlement batches are
   projections with `as_of_entry_id`, and a scheduled job recomputes a sample (and periodically everything) from the
   ledger and compares.

**Detailed Failure Scenarios:**

| # | Failure | Impact | Handling |
|---|---|---|---|
| 1 | Duplicate posting request (client or consumer retry) | Double credit | `UNIQUE (idempotency_key)` on `transactions`; the second insert returns the first result |
| 2 | Two payments land at once for the same installment | Overallocation, wrong paid status | L2 serialization; second payment allocates to the remainder or becomes a credit/refund; both visible in the ledger |
| 3 | Crash mid-posting | Partial entries | L1: single DB transaction, nothing visible until commit |
| 4 | A bug produces an unbalanced transaction | Money created/destroyed | L1 commit-time check; nightly global invariant `SUM(debit) = SUM(credit)`; page on violation |
| 5 | Projector crashes or lags | Stale `loan_balances` | Reads that must be exact compute from entries or check `as_of_entry_id`; the projector is idempotent on `entry_id`; lag alert |
| 6 | Cache/projection drifts from the ledger | Wrong balance shown | L7 recompute-and-compare; auto-heal by rebuilding from the ledger; drift counter on a dashboard |
| 7 | Late ACH settlement or backdated entry (`effective_ts` earlier than recorded) | Historical balances, interest and late fees change | Re-derive dependent values deterministically from entries; emit correction events; notify only if the borrower-visible result changes; `recorded_ts` answers "what did we know when" |
| 8 | Backdated entry lands in a **closed accounting period** | Finance reports change retroactively | Post an adjustment dated in the open period referencing the original; approval workflow; never rewrite closed periods |
| 9 | Reversal of a reversal / refund after partial payments | Negative or inconsistent balances | Every action is a new linked transaction; constraint caps total reversals; refund allocation rules documented (pro-rata to paid installments) |
| 10 | Refund larger than amount paid | Merchant owes us | Cap at paid amount; create a merchant receivable; flag for review |
| 11 | Hot merchant account (many loans) | Row contention | Per-loan accounts plus an aggregated view; batch-serialized writes for the aggregate account (the pattern Uber describes for hot ledger entities) |
| 12 | Settlement spanning many loans/shards | Partial settlement | Settlement batch derived from the ledger, then executed as a saga with idempotent steps and a reconciliation record |
| 13 | New entry type breaks an old consumer | Wrong totals downstream | Additive schema only; versioned `metadata`; consumers fail loudly (not silently skip) on unknown types; contract tests |
| 14 | CDC/warehouse feed lags or loses rows | Finance reports wrong | Per-day row-count and checksum comparison between OLTP and warehouse; replay from the log |
| 15 | Deletion/privacy request | Conflict with immutability | L6: delete or anonymize the token mapping; ledger rows remain valid |
| 16 | Accidental or malicious update/delete | Audit failure | L4 permissions; optionally hash-chain entries for tamper evidence (Uber built a custom tamper-evident store for this need); alert on any `UPDATE/DELETE` attempt |
| 17 | Reconciliation break vs processor/bank | Unknown money | Classify (timing difference, missing, amount); auto-clear timing breaks within tolerance; unresolved breaks open tickets; never plug silently with a manual entry without approval |
| 18 | Historical loan migration | Opening balances wrong | Dual-run, compare balances loan by loan, cut over only on zero unexplained differences |
| 19 | Interest/day-count disagreement | Borrower disputes | Versioned calculation engine; store inputs and version with each accrual; replayable |
| 20 | Shard imbalance or hot loan | Latency | Shard by `loan_id` hash; split hot shards; move heavy merchants' reporting to the warehouse |

**How Real Production Systems Do It (and what to copy):**

*Uber — [Payments Platform](https://www.uber.com/blog/ubers-payments-platform), ledger design.*
- **Money order** = the unit of work: debits and credits between two or more entities for one commerce transaction
  (a trip, an Eats order, a freight shipment). It is **immutable**; an adjustment is a *new* money order. This is our
  `transactions` + `ledger_entries` with `reverses_txn_id`.
- **Double-entry with zero-sum enforced before commit**: each participant's amount is explicit (an "entity fare"),
  and the sum across entities must be zero, so ambiguity upstream can't create or destroy money.
- **Ledger = entity -> sub-ledger accounts -> balance.** Originally an entity was a single ~1 KB DynamoDB row;
  zero-balance sub-ledgers were pruned to keep rows small. They later hit limits (rows, hot keys, audit) and moved to
  **LedgerStore**, built with Uber's Storage Platform team for auditability and tamper-evident records.
- **Entity changelog**: every balance update is recorded, so any ledger can be rebuilt from inception. This is our
  "balance is derived; rebuild from entries" principle.
- **Strongly consistent balances** for 1.2B+ entities, with async cross-region replication; **hot entities** (B2B,
  Uber Direct) used a **serialized batch-write** path for about 10x throughput. **Copy**: a merchant-payable
  account touched by every loan is exactly this hot-row problem; batch writes per account in one writer, don't
  let thousands of concurrent transactions contend on one row.
- Scale reference: gross bookings grew from $50B to $217B annualized, total money movement nearly 2x that, and
  several multi-tenant ledgers have run 7+ years, which is why they chose immutability and schema-flexible entries.
- Not disclosed in the post: idempotency mechanics and reconciliation, so those pieces of our design come from
  Stripe/Airbnb (see #2) and general practice.

*Airbnb — [Rebuilding Payment Orchestration](https://airbnb.tech/?p=49).* Payment workflows as DAGs of idempotent
steps that end in ledger postings; the stated outcome is 99.999% consistency. **Copy**: quote this as the target and
tie it to why the unique `idempotency_key` on `transactions` exists.

*Affirm*: no public Affirm ledger post found. Third-party guides say interviewers press on idempotency keys, double
entry, and reconciling state when a downstream call fails midway; the schema above plus #2's executor answer all three.

**Follow-Up Questions:**
1. Why not just decrement a `balance` column on every payment? → it collapses the effective/recorded
   timestamp distinction, makes historical/point-in-time balance reconstruction impossible without a
   separate audit log anyway, and turns every concurrent payment into a race on a single row — the ledger
   design gets you correctness on all three for the cost of one derived-value computation.
2. How do you reconcile a merchant's expected settlement total against what actually posted? → sum
   `ledger_entries` by `merchant_id` (via `loan_id -> merchant_id`) over a settlement window, filtered by
   `entry_type`, and diff against the merchant's own reported totals — the append-only ledger is exactly
   the audit trail a reconciliation job needs, with no separate logging system required.
3. How would you add a new fee type (e.g., a late fee) without a migration? → it's a new `entry_type`
   value with fee-specific fields in the `metadata` column — no schema migration, no change to how balance
   summation works, since summation is agnostic to which `entry_type` it's adding.

---

## 4. System Design — Real-Time Fraud Detection System

**Problem Statement:**
A recurring system-design theme across Affirm prep guides, and the natural system-level counterpart to
the coding-round fraud problems in [coding #4](./affirm_interview_questions.md#4-live-fraud-detector-streaming-suspicious-pii-propagation) and [coding #10](./affirm_interview_questions.md#10-fraud-detection-match-transactions-to-fraud-events): design a real-time fraud detection system that scores
incoming loan applications and transactions, combining rule-based checks (velocity limits, PII reuse
across flagged applications, device/IP reputation) with a machine-learning risk score, and decides to
approve, decline, or route to manual review — within the latency budget of a live checkout flow.

**Functional Requirements:**
- Score an incoming application/transaction against both deterministic rules (e.g., "3+ applications
  from this device in 10 minutes") and an ML model's risk score.
- Combine rule hits and the model score into a decision: `approve`, `decline`, or `manual_review`.
- Propagate a confirmed-fraud signal (an analyst or chargeback confirms fraud after the fact) back into
  future scoring — the system-level version of the PII-propagation coding problem in [coding #4](./affirm_interview_questions.md#4-live-fraud-detector-streaming-suspicious-pii-propagation).

**Non-Functional Requirements:**
- Scoring latency in the low tens of milliseconds — it sits directly in the checkout path, not an
  offline batch job.
- False-positive rate (blocking legitimate borrowers) must stay low enough not to hurt approval rate/
  conversion, while false negatives carry direct loss exposure — an explicit precision/recall tradeoff to
  discuss, not just "minimize fraud."
- New fraud patterns (rules or retrained models) must be deployable without a full system redeploy.

**High-Level Design:**
1. **Feature store for low-latency lookups**: velocity counts (applications per device/IP/PII value in
   trailing windows), historical fraud flags, and account-age signals are precomputed and kept in a
   low-latency key-value store (not queried from a data warehouse synchronously) — this is the same
   windowed-counting shape as the Hit Counter ([coding #11](./affirm_interview_questions.md#11-thread-safe-sliding-window-hit-counter)) and PII propagator ([coding #4](./affirm_interview_questions.md#4-live-fraud-detector-streaming-suspicious-pii-propagation)), reused at system scale.
2. **Rules engine**: a set of independently deployable, versioned rules evaluated against the feature
   store in parallel; each rule contributes a hit/no-hit signal plus a severity, so adding a rule doesn't
   require redeploying the scoring service.
3. **ML scoring service**: a separately deployed model-serving endpoint returns a risk score for the
   same request; called in parallel with the rules engine (not sequentially) to stay within the latency
   budget, with a fallback to rules-only if the model service is slow/unavailable (graceful degradation
   over blocking checkout).
4. **Decision layer**: combines rule hits (some rules are hard-decline, e.g. a confirmed-fraud PII match;
   most are soft signals) with the ML score against a threshold, producing `approve` / `decline` /
   `manual_review` — hard-decline rules should short-circuit before even waiting on the ML call, to keep
   the common "obviously fraudulent" case fast.
5. **Feedback loop**: confirmed-fraud outcomes (from manual review or later chargebacks) are written back
   into the feature store (flagging the PII/device involved) and queued for periodic model retraining —
   closing the loop between the propagation logic in [coding #4](./affirm_interview_questions.md#4-live-fraud-detector-streaming-suspicious-pii-propagation) and the model that scores future applications.

**Data Model (sketch):**
```
feature_store: device_velocity(device_id, window) -> count
                pii_fraud_flags(pii_value) -> bool          # fed by the feedback loop
rules(rule_id, version, definition, severity, enabled)
scoring_requests(request_id, application_id, ts, rule_hits[], ml_score, decision)  # audit trail
```

**Detailed Component Breakdown:**

| Component | Responsibility | Tech choice (typical) |
|---|---|---|
| Decision API | Entry point from checkout/underwriting; owns the latency budget (e.g. p99 < 100 ms) and fallbacks | Stateless gRPC service |
| Feature assembler | Fans out to online stores, merges request-time features | Parallel async calls with per-call timeouts |
| Online feature store | Velocity counters, PII/device flags, account age | Redis/Cassandra/DynamoDB, TTL'd windows |
| Streaming aggregator | Computes windowed counts from event stream into the online store | Kafka + Flink (sliding/tumbling windows) |
| Offline feature store | Same features computed in batch for training (avoids train/serve skew) | Warehouse + feature definitions shared with online |
| Rules engine | Versioned rules as data; hard-decline vs soft-signal | Rule DSL/JSON evaluated in-process; config pushed |
| Model serving | Returns risk score for the feature vector | Hosted model service (GBDT/NN), shadow + canary versions |
| Decision combiner | Rules + score + policy thresholds -> approve/decline/review | Pure function, config-driven thresholds |
| Case management | Manual review queue, analyst tooling, label capture | Internal app + DB |
| Decision log | Immutable record of inputs, features, rule hits, score, model version, decision | Kafka -> warehouse; required for adverse-action/regulatory explanation |
| Label pipeline + training | Chargebacks, confirmed fraud, repayment defaults -> labels -> retrain | Airflow + Spark; delayed labels (weeks/months) |
| Graph/link analysis | Shared device/PII/address clusters (the [coding #4](./affirm_interview_questions.md#4-live-fraud-detector-streaming-suspicious-pii-propagation) problem at scale) | Graph store or union-find in stream job |

**Step-by-Step Flow (one application at checkout):**
1. Checkout sends `{application_id, user, PII hashes, device_id, ip, cart, merchant}` to the Decision API.
2. **Normalize/enrich**: hash PII (never log raw SSN), resolve device/IP reputation.
3. **Fan-out in parallel** (each with a hard timeout, e.g. 20-30 ms): velocity counters, PII fraud flags, graph
   neighbours, account history.
4. **Hard rules first**: if a hard-decline rule hits (confirmed-fraud PII, sanctions match), return `decline`
   immediately and skip the model.
5. **Model call**: send the assembled vector to model serving. If it times out or fails, use the degraded path
   (rules only, wider manual-review band) and tag the decision `model_unavailable`.
6. **Combine**: `score >= T_decline -> decline`; `T_review <= score < T_decline -> manual_review`; else
   `approve`; soft rule hits can push a borderline score up a band.
7. **Respond** to checkout and **async-log** the full decision record (inputs, features, versions, latency).
8. **Stream update**: the application event updates velocity counters and the PII/device graph for the *next*
   request.
9. **Manual review**: analyst sees evidence, decides; the decision is a label.
10. **Feedback**: confirmed fraud writes PII/device flags into the online store (propagation as in [coding #4](./affirm_interview_questions.md#4-live-fraud-detector-streaming-suspicious-pii-propagation)), and
    labels join the training set. Delayed labels (chargeback, first-payment default) arrive weeks later, so
    training uses label maturity windows.
11. **Release**: a new model runs in **shadow** (scored, not acted on), then canary on a % of traffic, compared on
    precision/recall and approval rate before full rollout.

**Safety Rules Added After Failure Analysis:**
1. **F1 — Every decision is explainable and logged.** Persist inputs (feature snapshot), rule hits, model version, score,
   thresholds and the final decision synchronously with a decision id; reason codes are needed for adverse-action
   notices. If the decision log write fails, the request fails closed to manual review rather than proceeding unlogged.
2. **F2 — Rules and models ship progressively.** shadow (evaluate, don't act) -> canary (a small percentage) -> full,
   with automatic rollback when decline rate, review rate or score distribution moves beyond a bound.
3. **F3 — Counters are atomic and idempotent.** Velocity updates use atomic increment-then-compare keyed by
   `request_id`, so retries don't double count and parallel requests can't both read the same stale count.
4. **F4 — Propagated signals have confidence, scope and TTL.** A confirmed-fraud flag on a PII value is a hard signal;
   a flag inherited through a *shared device or address* is only a review-band signal and expires, to avoid punishing
   family members or shared networks.
5. **F5 — Learn from unbiased data.** Route a small random sample of approvals and declines to review (or hold out a
   control policy) so labels are not only from the cases the current policy already flagged.
6. **F6 — Degraded modes are explicit.** Each dependency (feature store, model, device intelligence, bureau) has a
   timeout, a fallback and a tag in the decision log; degraded decisions have tighter exposure limits (lower credit
   limit, step-up verification) instead of blind approval.

**Detailed Failure Scenarios:**

| # | Failure | Impact | Handling |
|---|---|---|---|
| 1 | Feature store slow/unavailable | Missing features | Per-feature timeouts; defaults plus explicit "missing" indicators (the model is trained with missingness); wider review band; tag `features_degraded`; limit exposure (F6) |
| 2 | Streaming aggregates lag | Velocity undercounted, so attackers burst through | Monitor freshness; if lag exceeds a bound, treat velocity as unknown and apply conservative thresholds |
| 3 | Concurrent requests race on a counter | All requests pass at count=2 | F3: atomic increment-then-check (e.g. a Lua script or conditional write) |
| 4 | Checkout retries the same request | Double-counted velocity, spurious declines | Idempotent on `request_id`; cached decision returned for the same id |
| 5 | Model service down or slow | No score | Timeout + circuit breaker -> rules-only decision with widened manual-review band; hedged request to a second replica when latency allows |
| 6 | Bad model deployment (feature order, scaling bug) | Score distribution shifts, approvals or declines surge | Feature-schema contract tests; shadow then canary (F2); PSI/score-drift and approval-rate alarms with auto-rollback; model registry with signed model+schema |
| 7 | Bad rule (typo declines everyone) | Revenue loss, customer harm | Rule validation, dry-run against the last 24h of traffic, staged rollout, per-rule hit-rate alarm with auto-disable, per-rule kill switch |
| 8 | Propagation flags innocent users | Wrongful declines, fairness risk | F4; appeal and manual review path; track false-positive rate of propagated flags |
| 9 | Feedback-loop bias | Model learns only what the old policy already caught | F5 exploration sample; reject inference; evaluate on the holdout |
| 10 | Labels arrive late (chargebacks, first-payment default) | Recent cohorts look "good" | Label-maturity windows in training and evaluation; proxy labels for early signal; distribution monitoring until labels arrive |
| 11 | Attackers probe thresholds | Model/rules reverse engineered | Rate limit decision queries per device/IP, randomize inside the review band, minimize reason detail returned to applicants beyond what regulation requires |
| 12 | Drift/seasonal spikes (holidays, new merchant) | Precision drops | Drift monitors, champion/challenger, scheduled retraining, per-merchant calibration |
| 13 | Manual-review queue overwhelmed | Checkout stalls | Prioritize by risk and order value; SLA timeouts with a default action; step-up verification (ID check, micro-deposit) instead of an unbounded queue; capacity alerts |
| 14 | Decision log write fails | Cannot explain decisions later | F1: fail closed to review; buffer with a durable local queue; reconcile logs vs outcomes daily |
| 15 | PII exposure in features/logs | Privacy incident | Hash/tokenize at ingress with salted hashing; field-level encryption; log redaction; access audit |
| 16 | Third-party data (device intelligence, bureau) outage | Missing high-signal features | Per-vendor circuit breaker, short-lived caching, fallbacks, vendor SLAs |
| 17 | Event-time vs processing-time errors in windows | Wrong velocity | Event-time windows with watermarks; late events update counters but are flagged |
| 18 | Fair-lending and explainability | Regulatory risk | Feature review excluding protected attributes and close proxies; periodic bias testing; documented reason codes (coordinate with the compliance/legal team) |
| 19 | New user or device (cold start) | No history | Treat as unknown, not as fraud; rely on other signals; avoid blanket declines |
| 20 | Config error (wrong merchant threshold) | Systematic mis-decisions | Config validation, review and audit; canary per merchant |
| 21 | Train/serve skew (online and offline features computed differently) | Model underperforms in production | One feature definition feeds both paths; compare online vs offline values on a sample; alert on divergence |

**How Real Production Systems Do It (and what to copy):**

*Uber — Michelangelo (ML platform) and Palette (feature store): [Palette meta store](https://eng.uber.com/blog/palette-meta-store-journey/),
[Michelangelo overview](https://eng.uber.com/blog/from-predictive-to-generative-ai/).*
- **Online scoring path**: for a request, the serving layer fetches precomputed features from the online store
  (reported as **Cassandra** in a third-party write-up), **joins them with features sent in the request payload**,
  assembles the full feature vector, scores the model, and returns the prediction. This is our "feature assembler ->
  model serving" boxes; request-time features (device, IP, cart) come in the payload and don't need a lookup.
- **Palette** holds curated features (20,000+) with **batch and near-real-time** computation, including custom feature
  sets for Fraud, Eats and Comms; a metastore handles feature discovery, creation, deprecation, and offline/online
  serving setup. **Copy**: one feature definition feeds both training (offline) and serving (online), which is the
  fix for train/serve skew.
- Michelangelo scale: 5K+ models in production, ~10 million real-time predictions/second at peak, so model serving is
  a shared platform, not each team's service.
- **Delayed labels**: for fraud, outcome labels can take up to ~90 days, so you also monitor feature and prediction
  *distributions* over time to catch drift before labels arrive. **Copy**: add a drift monitor and label-maturity
  windows to training.
- **Uber Risk platform** ([Feature Store Summit 2023 slides](https://content.hopsworks.ai/hubfs/Feature%20Store%20Summit%202023/FS%20Summit%2023%20-%20Uber%20(Risk%20Platform).pdf)):
  a dedicated risk feature platform, "uFlow", with batch, near-real-time, aggregator and real-time computation and
  fetch stages, plus a graph feature store ("uGraph"). **Copy**: velocity counters = aggregator stage, shared
  device/PII clusters = graph store, and the same-request features = real-time stage. I only saw the slide
  outline, so latency numbers are not claimed.

*Affirm*: public job postings describe fraud work as adversarial, with continuous attacker pressure on identity and
synthetic-identity fraud. I found no public Affirm fraud architecture post; state that the above is the industry
pattern and ask the interviewer about latency budget and manual-review capacity.

**Follow-Up Questions:**
1. How do you deploy a new fraud rule without redeploying the scoring service? → rules live as versioned
   config/data (not code) evaluated by a generic rule-evaluation engine, so publishing a new rule is a
   config change picked up on the next evaluation, not a service deploy.
2. What happens if the ML scoring service is down? → the decision layer falls back to rules-only scoring
   (documented as a wider decline/manual-review band to compensate for the missing signal) rather than
   either blocking checkout or silently approving everything — an explicit degraded mode, not an implicit
   one.
3. How fast does a newly confirmed fraud PII value affect in-flight scoring? → as fast as the feature
   store write propagates — if it's the same low-latency store the rules engine reads synchronously, a
   confirmed flag can affect the very next request, mirroring the "retroactive + prospective" propagation
   behavior required in the coding-round version of this problem ([coding #4](./affirm_interview_questions.md#4-live-fraud-detector-streaming-suspicious-pii-propagation)).

---

## References

Sources used for these design answers:
- [Affirm System Design Technical Phone Interview Experience — 1point3acres](https://www.1point3acres.com/interview/thread/1032443)
- [Uber's Payments Platform](https://www.uber.com/blog/ubers-payments-platform)
- [Uber's Experimentation Platform (XP)](https://www.uber.com/blog/xp/)
- [Uber Palette meta store](https://eng.uber.com/blog/palette-meta-store-journey/)
- [Stripe: Designing robust and predictable APIs with idempotency](https://stripe.com/blog/idempotency)
- [Airbnb: Rebuilding Payment Orchestration (Orpheus)](https://airbnb.tech/?p=49)
- [Affirm System Design Interview: The Complete Guide — System Design Handbook](https://www.systemdesignhandbook.com/guides/affirm-system-design-interview/)

Note: the design problems were reconstructed and expanded from publicly visible question titles, tags and summaries into complete answers; they are not verbatim interview transcripts.
