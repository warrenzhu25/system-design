# CoreWeave Interview Questions

---

## Contents

**Coding**
1. [Find a Directed Resource Access Path (Shortest Path in an Access Graph)](#1-find-a-directed-resource-access-path-shortest-path-in-an-access-graph)
2. [Boolean Filter Engine for Experiment Tracking Data](#2-boolean-filter-engine-for-experiment-tracking-data)
3. [Concurrent Server Reboot CLI](#3-concurrent-server-reboot-cli)
4. [Rate-Limited Concurrent Web Scraper](#4-rate-limited-concurrent-web-scraper)

**System Design**
5. [Multi-Tenant GPU Cluster Scheduler](#5-system-design--multi-tenant-gpu-cluster-scheduler)
6. [Kubernetes Control Plane for a GPU Fleet](#6-system-design--kubernetes-control-plane-for-a-gpu-fleet)
7. [High-Throughput Storage for ML Training Data](#7-system-design--high-throughput-storage-for-ml-training-data)

**Process & Behavioral**
8. [Interview Process Overview](#8-interview-process-overview)
9. [Behavioral Themes](#9-behavioral-themes)

---

## 1. Find a Directed Resource Access Path (Shortest Path in an Access Graph)

**Problem Statement:**
A CoreWeave online-assessment problem (the only publicly catalogued OA problem for the company, per
FastPrep). You're given `access: list[list[int]]`, a list of directed pairs where `[a, b]` means resource
`a` can access resource `b`, plus a `source` and `target` resource id. Return **one shortest path** (as a
list of resource ids, including both endpoints) from `source` to `target`. When multiple shortest paths
exist, break ties by the order edges appear in the input — i.e. run BFS and enqueue each node's neighbors
in the order their edges were given. Return `[]` if `target` is unreachable, and `[source]` if
`source == target`.

**Constraints:**
- Up to 200,000 access pairs.
- Resource ids range over `0` to `10^9` (sparse — an adjacency **map**, not an array, is required).
- The graph may contain cycles, duplicate edges, and self-loops.

**Example:**
```
access = [[1, 2], [1, 3], [2, 4], [3, 4]]
source, target = 1, 4
# Two shortest paths exist: [1,2,4] and [1,3,4].
# [1,2] appears before [1,3] in the input, so BFS reaches 4 via 2 first.
shortest_access_path(access, 1, 4)  # -> [1, 2, 4]

shortest_access_path([[5, 7]], 7, 5)  # -> []  (access is directed; 7 -> 5 was never given)
```

**Test Cases:**

| Scenario | Expectation |
|---|---|
| `source == target`, no self-loop edge needed | `[source]` |
| Two equal-length paths reachable via different first edges | The one whose first edge appears earlier in `access` |
| `target` unreachable from `source` | `[]` |
| Self-loop `[x, x]` present, unrelated to the query path | Ignored, doesn't affect shortest-path length |
| Duplicate edge `[a, b]` listed twice | No effect on correctness or tie-breaking (first occurrence already set adjacency order) |

**Key Insights:**
1. Resource ids span `0..10^9`, so the adjacency structure must be a `dict[int, list[int]]`, not an
   array indexed by id — a detail worth stating out loud before coding, since it's an easy place to
   silently assume small dense ids.
2. Plain BFS from `source` with a `parent` map gives shortest-path-by-edge-count for free; the tie-breaking
   rule ("earlier edge in the input wins") falls out automatically as long as each node's outgoing edges are
   appended to its adjacency list **in input order** and neighbors are visited in that same order during
   the BFS expansion — no separate tie-break logic needed.
3. Cycles and duplicate edges need no special-casing: the `parent`/visited map already prevents
   revisiting a node, so a cycle just means BFS never re-enqueues a node it has already assigned a parent
   to, and a duplicate edge is a no-op the second time it's processed.

**Python Solution:**
```python
from collections import defaultdict, deque


def shortest_access_path(access: list[list[int]], source: int, target: int) -> list[int]:
    """
    Time:  O(V + E) — E = len(access), V = distinct resource ids touched.
    Space: O(V + E) for the adjacency map and BFS parent/visited bookkeeping.
    """
    if source == target:
        return [source]

    graph: dict[int, list[int]] = defaultdict(list)
    for a, b in access:
        graph[a].append(b)  # preserve input order so BFS tie-breaks correctly

    parent: dict[int, int] = {source: source}
    queue = deque([source])
    while queue:
        node = queue.popleft()
        if node == target:
            break
        for nxt in graph.get(node, []):
            if nxt not in parent:
                parent[nxt] = node
                queue.append(nxt)

    if target not in parent:
        return []

    path = [target]
    while path[-1] != source:
        path.append(parent[path[-1]])
    path.reverse()
    return path
```

---

## 2. Boolean Filter Engine for Experiment Tracking Data

**Problem Statement:**
A CoreWeave technical phone screen reported in early 2026: design a filtering engine over a table of ML
**experiment run** records (the kind of metadata a training-job dashboard tracks — accuracy, status,
GPU type, hyperparameters), where the filter itself is an arbitrary boolean expression combining field
comparisons with `AND` / `OR` / `NOT`. The report specifically calls out **type handling** as a named focus
area: fields on different runs aren't guaranteed to be the same type (a run that crashed before logging a
metric might have that field missing entirely; a `"gpu_count"` field might arrive as either `int` or `str`
depending on the client that wrote it).

**Filter expression shape:**
```python
{"op": "and", "children": [
    {"field": "accuracy", "cmp": ">=", "value": 0.9},
    {"op": "or", "children": [
        {"field": "status", "cmp": "=", "value": "completed"},
        {"field": "gpu_type", "cmp": "in", "value": ["H100", "B200"]},
    ]},
]}
```
Leaf nodes are `{"field", "cmp", "value"}` with `cmp` in `{"=", "!=", "<", "<=", ">", ">=", "in"}`. Internal
nodes are `{"op": "and" | "or" | "not", "children": [...]}` (`"not"` takes exactly one child). Return the
indices of every record that matches, in original order.

**Type-handling rules (state these explicitly before coding):**
- A record missing the field a leaf references does **not** match that leaf — including for `!=`, which is
  the case candidates most often get wrong by assuming "missing != value" should be `True`.
- A comparison between incompatible types (e.g. `"gpu_count": "8"` vs. `value: 8` for `<`) evaluates to
  `False` rather than raising — a bad type is a non-match, not a crash.
- `bool` is **not** treated as a subtype of `int` for comparisons, even though Python's own `==` would say
  `True == 1` — otherwise a `"completed": True` field would silently satisfy a numeric filter.

**Test Cases:**

| Scenario | Expectation |
|---|---|
| Leaf field missing from a record | Record doesn't match that leaf, for any `cmp` including `!=` |
| `"in"` against a list containing mixed types | Matches only same-type members, no crash on the rest |
| `<` between a `str` and an `int` value | `False`, not a `TypeError` |
| `not` wrapping a leaf that would match | Record excluded |
| Deeply nested `and`/`or`/`not` | Evaluated recursively, short-circuiting `and`/`or` where possible |

**Key Insights:**
1. Treat the whole thing as a small **recursive expression tree evaluator** — one function that dispatches
   on `"op"` vs. `"field"`/`"cmp"`, not a flat rule list. This is the same shape as a mini query-plan
   evaluator, which is exactly the "explain your design before coding" hook the interviewer is fishing for
   given the infra-adjacent framing.
2. Centralize the type-safety check in **one** comparison helper that every leaf and every `"in"` check
   funnels through, so "incompatible types don't crash" is enforced in exactly one place instead of
   re-implemented per operator.
3. `and`/`or` should short-circuit (stop evaluating children once the result is determined) — not a
   performance requirement at small scale, but worth naming since a filter tree over a large record set is
   evaluated per-record, and short-circuiting is free correctness-preserving cleanup once the recursive
   shape is already in place.

**Python Solution:**
```python
def _safe_compare(cmp: str, actual, expected) -> bool:
    """One choke point for type-safety: incompatible types -> False, never raise."""
    if cmp == "in":
        return any(type(actual) is type(v) and actual == v for v in expected)
    if type(actual) is not type(expected):
        return False
    if cmp == "=":
        return actual == expected
    if cmp == "!=":
        return actual != expected
    try:
        return {"<": actual < expected, "<=": actual <= expected,
                ">": actual > expected, ">=": actual >= expected}[cmp]
    except TypeError:
        return False


def _evaluate(node: dict, record: dict) -> bool:
    if "op" in node:
        if node["op"] == "not":
            return not _evaluate(node["children"][0], record)
        if node["op"] == "and":
            return all(_evaluate(child, record) for child in node["children"])
        return any(_evaluate(child, record) for child in node["children"])  # "or"

    field = node["field"]
    if field not in record:
        return False
    return _safe_compare(node["cmp"], record[field], node["value"])


def filter_runs(records: list[dict], expression: dict) -> list[int]:
    """
    Time:  O(n * s) — n records, s = size of the expression tree per record.
    Space: O(1) beyond the output index list (evaluation is a plain recursive walk).
    """
    return [i for i, record in enumerate(records) if _evaluate(expression, record)]
```

---

## 3. Concurrent Server Reboot CLI

**Problem Statement:**
A CoreWeave SDE2 onsite round, reported as a 60-minute CoderPad exercise: build a **CLI tool** that
reboots a batch of servers by calling a real **HTTP JSON** reboot endpoint, in parallel, and prints a
final status per server. The report names three explicit focus areas: **API calls**, **error handling**,
and **`ThreadPoolExecutor` concurrency** — and other candidates' notes on this round specifically describe
the API as an HTTP endpoint (status codes + a JSON body), not a pre-wrapped exception-raising client, so
part of the exercise is translating raw HTTP semantics into retry/no-retry decisions yourself, then wiring
the result into a CLI entrypoint — matching CoreWeave's own fleet-operations surface (rebooting GPU nodes
at scale), where a single hung or failing call must never block the rest of the batch.

**API contract (given):**
```
POST /v1/servers/{server_id}/reboot

200 OK                {"status": "ok"}
429 Too Many Requests  header "Retry-After": "<seconds>", body {"error": "rate_limited"}
500 / 502 / 503 / 504  {"error": "transient", "message": "..."}   -> retryable
404 / 400              {"error": "not_found" | "invalid_request"} -> not retryable
```
You're given an `HttpClient` with `post(path: str) -> HttpResponse`, where `HttpResponse` exposes
`.status_code: int`, `.headers: dict`, and `.json() -> dict`.

**Requirements:**
- Reboot up to `max_workers` servers concurrently (bounded, not one thread per server).
- Classify every response by **status code**: `200` succeeds; `429`/`5xx` are retryable (`429` waits the
  server-given `Retry-After` seconds, `5xx` backs off exponentially up to `max_retries`); `4xx` other than
  `429` fails immediately with no retry.
- A CLI entrypoint (`main(argv, client)`) that reads a `--ids=s1,s2,s3` flag, reboots that batch, and
  prints the final `{server_id: status}` map to stdout as a single JSON object.
- `dict[str, str]` result values are `"success"` or `"failed: <reason>"` (`reason` pulled from the JSON
  error body) — one call failing or exhausting retries must not prevent results for the others.

**Test Cases:**

| Scenario | Expectation |
|---|---|
| All servers return `200` on first call | All `"success"`, results present for every input id |
| One server returns `429` with `Retry-After: 1`, then `200` | `"success"`, and the `429` did not consume a retry attempt |
| One server returns `503` twice then `200` | `"success"` after 3 total attempts, other servers unaffected |
| One server returns `404` | `"failed: not_found"` immediately, no retry attempted for it |
| One server returns `500` past `max_retries` | `"failed: transient"` after the last retry, others still complete normally |
| `main(["--ids=s1,s3"], client)` | Prints one JSON object to stdout with exactly those two keys |
| 50 servers, `max_workers=10` | No more than 10 `post()` calls in flight at any instant (verified via a counting mock) |

**Key Insights:**
1. The retry decision is a **pure function of the status code**, not of exception types — centralize it in
   one `RETRYABLE_STATUSES` set/branch so "which codes get retried" is stated once, not re-derived per
   caller. This is the concrete version of the same "one choke point for a policy decision" pattern as the
   filter engine's type-safety helper above.
2. `429` and `5xx` need **different** backoff strategies: `429` means "the server told you exactly how
   long to wait" (respect `Retry-After` literally, don't also apply exponential backoff on top of it),
   while `5xx` means "the server didn't say" (exponential backoff is your own choice to make, and doesn't
   consume the same budget as a rate-limit wait).
3. `ThreadPoolExecutor(max_workers=N)` + `as_completed` gives bounded concurrency and independent-failure
   isolation for free — same shape as the reboot-by-exception version of this problem, just with the retry
   classifier reading `response.status_code` instead of catching typed exceptions. The CLI layer (`main`)
   should stay a thin wrapper around `reboot_servers` — parse argv, call it, serialize the result to JSON —
   so the concurrency/retry logic stays independently testable without going through argv or stdout at all.

**Python Solution:**
```python
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

RETRYABLE_STATUSES = {429, 500, 502, 503, 504}


def _reboot_with_retry(client: "HttpClient", server_id: str, max_retries: int) -> tuple[str, str]:
    attempt = 0
    while True:
        response = client.post(f"/v1/servers/{server_id}/reboot")

        if response.status_code == 200:
            return server_id, "success"

        if response.status_code not in RETRYABLE_STATUSES:
            return server_id, f"failed: {response.json().get('error', 'unknown_error')}"

        if response.status_code == 429:
            time.sleep(float(response.headers.get("Retry-After", 1)))
            continue  # server-given wait, doesn't consume a retry attempt

        attempt += 1
        if attempt > max_retries:
            return server_id, f"failed: {response.json().get('error', 'unknown_error')}"
        time.sleep(2 ** attempt * 0.1)  # exponential backoff for unspecified 5xx


def reboot_servers(
    server_ids: list[str], client: "HttpClient", max_workers: int = 10, max_retries: int = 3
) -> dict[str, str]:
    """
    Time:  O(n / max_workers) wall-clock, bounded by the pool size.
    Space: O(n) for the result map and in-flight futures.
    """
    results: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(_reboot_with_retry, client, sid, max_retries): sid for sid in server_ids
        }
        for future in as_completed(futures):
            sid, status = future.result()
            results[sid] = status
    return results


def main(argv: list[str], client: "HttpClient") -> None:
    """CLI entrypoint: `--ids=s1,s2,s3` -> one JSON object on stdout."""
    ids_flag = next(arg for arg in argv if arg.startswith("--ids="))
    server_ids = ids_flag.split("=", 1)[1].split(",")
    print(json.dumps(reboot_servers(server_ids, client), sort_keys=True))
```

---

## 4. Rate-Limited Concurrent Web Scraper

**Problem Statement:**
A CoreWeave full-time onsite, reported as one leg of a loop that also covered system design and
management/director discussions: implement a scraper that fans out over a list of URLs, respects a global
rate limit (no more than `N` requests per second across all workers, not per worker), retries on
transient HTTP failures, and returns each URL's fetched content or final error — the same "bounded
concurrency, independent failure isolation" shape as the reboot CLI above, but with a shared rate limiter
instead of just a bounded worker pool.

**Requirements:**
- Fetch a batch of URLs concurrently, but never exceed `N` requests/second in aggregate across all
  in-flight workers.
- Retry a failed fetch (timeout, 5xx) up to `max_retries` times with backoff; a 4xx (client error) is not
  retried.
- Return `dict[str, str | None]` — page content on success, `None` on exhausted retries — without one
  slow or failing URL blocking the others.

**Test Cases:**

| Scenario | Expectation |
|---|---|
| 20 URLs, rate limit 5/sec | Total wall-clock time is bounded below by ~4 seconds (20 / 5), not instantaneous |
| One URL returns 404 | `None` immediately, no retries burned on it |
| One URL times out twice then succeeds | Content returned, counted as 3 attempts against that URL only |
| One URL exhausts all retries | `None` for it; other URLs' results unaffected |
| Rate limiter shared across worker threads | No burst of more than `N` requests within any 1-second window, verified via timestamped mock calls |

**Key Insights:**
1. The rate limit is **global**, not per-worker — a naive "sleep 1/N seconds inside each worker" throttles
   each thread independently and the aggregate rate scales with worker count, which is wrong. A single
   shared **token bucket** (or a lock-guarded sliding window) that every worker acquires from before firing
   a request is the only way to bound the *aggregate* rate.
2. Same failure-isolation principle as the reboot CLI: retry logic lives inside each URL's task, and results
   are collected via `as_completed`, so a URL stuck retrying doesn't hold up URLs that already resolved.
3. Distinguishing retryable (timeout, 5xx) from non-retryable (4xx) failures up front avoids wasting rate-
   limited request budget hammering a URL that will never succeed — worth calling out as the same "don't
   retry what can't succeed" principle as `PermanentError` in the reboot problem.

**Python Solution:**
```python
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed


class TokenBucket:
    """Shared global rate limiter: at most `rate` acquisitions per second, across all threads."""

    def __init__(self, rate: float):
        self.rate = rate
        self._lock = threading.Lock()
        self._tokens = rate
        self._last = time.monotonic()

    def acquire(self) -> None:
        while True:
            with self._lock:
                now = time.monotonic()
                self._tokens = min(self.rate, self._tokens + (now - self._last) * self.rate)
                self._last = now
                if self._tokens >= 1:
                    self._tokens -= 1
                    return
                wait = (1 - self._tokens) / self.rate
            time.sleep(wait)


def _fetch_with_retry(fetch, bucket: TokenBucket, url: str, max_retries: int):
    attempt = 0
    while True:
        bucket.acquire()
        status, body = fetch(url)
        if 200 <= status < 300:
            return url, body
        if 400 <= status < 500:
            return url, None  # not retryable
        attempt += 1
        if attempt > max_retries:
            return url, None
        time.sleep(2 ** attempt * 0.1)


def scrape_urls(
    urls: list[str], fetch, rate: float, max_workers: int = 10, max_retries: int = 3
) -> dict[str, str | None]:
    """
    Time:  wall-clock bounded below by len(urls) / rate.
    Space: O(n) for results and in-flight futures.
    """
    bucket = TokenBucket(rate)
    results: dict[str, str | None] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_fetch_with_retry, fetch, bucket, u, max_retries): u for u in urls}
        for future in as_completed(futures):
            url, content = future.result()
            results[url] = content
    return results
```

---

## 5. System Design — Multi-Tenant GPU Cluster Scheduler

**Problem Statement:**
CoreWeave's signature system-design prompt, consistent across every account of the onsite loop's design
round: design the scheduler that places customer training/inference jobs onto a shared fleet of GPU
machines. Unlike a generic batch scheduler, correctness here is inseparable from **hardware topology** —
which GPUs share a fast NVLink domain, which machines share a rack/InfiniBand leaf switch — because a job
split across a badly-chosen set of GPUs runs far slower than one packed onto topologically-close hardware.

**Functional Requirements:**
- `submit_job(job)` — a job requests `k` GPUs and declares whether it needs them co-located within one
  NVLink domain (a "gang" requirement).
- **Gang scheduling**: all of a job's requested GPUs start together, or none do — no partial starts.
- Support job priority/preemption: a higher-priority job can preempt a lower-priority job's GPUs, with the
  preempted job checkpointed and requeued rather than silently killed.
- Per-tenant quotas so one customer can't starve others of the shared fleet.

**Non-Functional Requirements:**
- Placement decisions must account for GPU topology, not treat all idle GPUs as interchangeable.
- A single dead/unhealthy GPU must not be schedulable, and must not silently stall a gang-scheduled job
  waiting on it forever.
- Scheduling latency for a typical job (find + reserve GPUs) should be low relative to job runtime — jobs
  run for hours, so seconds of scheduling latency are acceptable; minutes are not.

**High-Level Design:**
1. **Topology-aware inventory**: model the fleet as a tree — datacenter → rack → machine → GPU — annotated
   with interconnect type at each level (NVLink within a machine, InfiniBand across a rack). The scheduler's
   placement search walks this tree top-down, preferring the tightest-scoped subtree that satisfies a gang
   request, rather than treating GPUs as a flat pool.
2. **Gang scheduling via all-or-nothing reservation**: for a job needing `k` co-located GPUs, the scheduler
   finds a candidate subtree with `k` free, topologically-adjacent GPUs and reserves all of them in one
   atomic step (e.g., a compare-and-swap against each GPU's state) before starting any of the job's
   workers — if reservation of any GPU in the set fails (raced by another scheduling pass), the whole
   reservation is rolled back and retried, never left half-started.
3. **Priority queue with preemption**: pending jobs are ordered by priority (and fairness within a
   priority tier, e.g. weighted by tenant quota headroom); when no idle topology-adjacent set satisfies a
   high-priority job, the scheduler selects a lower-priority running job occupying a suitable set, signals
   it to checkpoint, reclaims its GPUs once the checkpoint completes, and requeues the preempted job at the
   front of its priority tier.
4. **Health-aware placement**: each GPU/machine reports periodic health checks (NVLink link errors, ECC
   errors, thermal throttling); unhealthy GPUs are removed from the schedulable inventory immediately, and
   any job already gang-scheduled onto a GPU that goes unhealthy mid-run is treated as a fault requiring
   checkpoint-and-reschedule onto a replacement set — not left to hang indefinitely.
5. **Per-tenant quotas**: enforced at admission (reject/queue a submission exceeding a tenant's GPU-hour
   or concurrent-GPU quota) rather than only after scheduling, so quota violations don't consume placement
   search effort before being rejected.

**Data Model (sketch):**
```
gpu_inventory(gpu_id, machine_id, rack_id, nvlink_domain_id, status, health)
jobs(job_id, tenant_id, priority, gpu_count, gang_required, status, checkpoint_uri)
reservations(job_id, gpu_id[], reserved_at, epoch)   # epoch fences stale reservation attempts
tenant_quotas(tenant_id, max_concurrent_gpus, max_gpu_hours_per_day)
```

**Scaling & Reliability:**
- Reservation is fenced with an epoch/version per GPU (similar in spirit to leader fencing in a
  replicated log) so a scheduling pass that raced against a concurrent reservation and lost fails cleanly
  instead of double-booking a GPU.
- Checkpoint storage for preempted/faulted jobs must itself be high-throughput (see the storage design
  below) — a preemption that takes minutes to checkpoint defeats the purpose of fast preemption for
  higher-priority work.
- Fleet-wide topology data is read far more often (every scheduling pass) than it changes (hardware
  additions/removals) — cache it in the scheduler's memory and update via an event stream from the
  inventory/health system, rather than querying a central store per decision.

**Follow-Up Questions:**
1. What happens if a job's gang requirement can never be satisfied because the fleet has no single
   NVLink domain with enough free GPUs? → surface a clear "cannot be gang-scheduled at current fleet
   shape" rejection rather than queuing indefinitely; optionally offer the tenant a relaxed (multi-domain,
   InfiniBand-only) placement if their workload tolerates the slower interconnect.
2. How do you avoid low-priority jobs starving forever under constant preemption? → age-based priority
   boost (a job's effective priority rises the longer it's queued/preempted) is the standard fix, worth
   naming explicitly since pure priority-preemption without it is a textbook starvation bug.
3. How is a "stopped training job wastes GPU-hours" cost made visible to the scheduling decision itself,
   not just reported after the fact? → preemption cost isn't free — factor the checkpoint/restart overhead
   of the victim job into the preemption decision (e.g., avoid preempting a job seconds after its last
   checkpoint if a slightly-larger wait would let it reach its next natural checkpoint) rather than
   preempting purely on priority the instant a higher-priority job arrives.

---

## 6. System Design — Kubernetes Control Plane for a GPU Fleet

**Problem Statement:**
A recurring variant of the design round, framed specifically around **operating** Kubernetes as the
orchestration layer over a GPU fleet, rather than the scheduling algorithm itself: how does Kubernetes
know which nodes have healthy GPUs, how are GPUs exposed as a schedulable resource, and how does the
control plane detect and replace a failed GPU node without a human paging at 3am for every hardware fault
— a near-certainty at fleets of tens of thousands of GPUs, where "hardware fails constantly" is treated as
a given, not an edge case.

**Functional Requirements:**
- Expose each node's GPUs (count, model, NVLink/InfiniBand topology, health) as schedulable Kubernetes
  resources that pods can request via standard `resources.limits`.
- Detect a node whose GPU(s) have gone unhealthy (Xid errors, ECC failures, thermal shutdown) and cordon
  it — stop scheduling new pods there — without waiting on a generic node-level `NotReady` signal that
  might never fire if the rest of the node (CPU, network) is fine.
- Automate replacement: drain, deprovision, and trigger re-provisioning of a confirmed-bad node with
  minimal manual intervention.

**Non-Functional Requirements:**
- GPU health detection latency should be on the order of the device plugin's health-check interval
  (seconds), not dependent on a workload crashing first to reveal the fault.
- Node replacement must not evict healthy, unrelated workloads on the same physical machine if only one
  GPU (of several) is unhealthy — evict only what actually depends on the bad device.
- The control plane's view of fleet topology (which nodes/GPUs are on the same NVLink domain or
  InfiniBand leaf switch) must stay consistent enough for the scheduler layer above it to make correct
  topology-aware placement decisions.

**High-Level Design:**
1. **Device plugin per node**: a DaemonSet-style device plugin advertises each node's GPUs as an
   extended/custom Kubernetes resource (e.g., `nvidia.com/gpu`) and continuously runs low-level health
   checks (Xid error codes, NVLink link state, ECC error counters) directly against the hardware — this is
   the mechanism that turns "GPU health" into something Kubernetes' scheduler can act on, since the
   default `kubelet`/node-health signals have no concept of individual device health.
2. **Custom resource + controller for topology**: model NVLink domains and InfiniBand leaf-switch groups as
   their own custom resources (CRDs), populated by the device plugin/inventory system, so pod placement
   (and the layer-5 scheduler above Kubernetes' default one) can express "these GPUs must be
   topologically co-located" as a real constraint rather than an opaque label.
3. **Fine-grained cordoning**: on a device-plugin-reported GPU failure, cordon scheduling of *that GPU
   resource specifically* (remove it from the node's advertised allocatable count) rather than cordoning
   the whole node — a healthy node with one bad GPU (of e.g. 8) should keep serving workloads that don't
   need that specific device.
4. **Automated drain-and-replace controller**: a controller watches for GPUs confirmed unhealthy past a
   debounce window (to avoid flapping on a transient blip), evicts only the pods actually bound to that
   device, cordons the node if the failure is node-level (not device-level), and files a replacement
   request against the underlying fleet-provisioning system — with a human in the loop only for the
   physical hardware swap, not for the detection-and-drain decision.
5. **NCCL/InfiniBand-aware readiness**: a node isn't marked schedulable for multi-node distributed jobs
   until its NCCL/InfiniBand connectivity to its topology peers is verified post-boot, not just once
   `kubelet` reports `Ready` — a node that's up but can't talk to its NVLink/InfiniBand peers is worse
   than a node that's simply absent, since a distributed job scheduled onto it fails instead of never
   starting.

**Data Model (sketch):**
```
node_gpu_inventory(node_id, gpu_id, model, nvlink_domain_id, ib_leaf_switch_id, health_status)
gpu_health_events(gpu_id, event_type, error_code, timestamp)
replacement_requests(node_id, reason, status, opened_at)
```

**Scaling & Reliability:**
- Health-check and event volume scales with fleet size — route GPU health events through the same durable,
  partitioned ingestion pattern used for other high-volume telemetry (rather than synchronous API calls
  from every device plugin into the control plane), so a burst of simultaneous hardware faults during, say,
  a power event doesn't overwhelm the control plane's API server.
- Debounce/hysteresis on health-based cordoning is essential at this scale: a flapping GPU that cordons and
  uncordons repeatedly generates constant pod churn — require a fault to persist past a threshold before
  acting, and require sustained health before uncordoning.
- Replacement automation should rate-limit itself (cap concurrent in-flight replacements) so a correlated
  hardware failure (e.g. a bad batch, a cooling incident) doesn't trigger a replacement stampede that
  itself destabilizes the fleet-provisioning system.

**Follow-Up Questions:**
1. How do you avoid false-positive cordoning from a transient, self-recovering GPU blip? → debounce window
   plus requiring the health check to fail N consecutive times, not once — the same anti-flapping logic
   named above, worth restating as the direct answer when asked.
2. How does the layer-5 (job) scheduler stay consistent with the Kubernetes-level view of topology and
   health if they're separate systems? → the topology/health CRDs are the single source of truth both
   layers read from — the job scheduler must not maintain its own independently-updated copy that can
   drift from what Kubernetes actually has scheduled.
3. What's the blast radius if the device plugin itself crashes on a node? → that node's GPUs should fail
   safe (become non-schedulable, e.g., via a liveness-based resource withdrawal), not fail open
   (workloads keep getting scheduled onto GPUs no one is currently health-checking) — an explicit design
   choice worth stating, not an assumed default.

---

## 7. System Design — High-Throughput Storage for ML Training Data

**Problem Statement:**
Design the storage layer feeding CoreWeave's training clusters: thousands of GPUs across many nodes need
to read the **same** large dataset simultaneously at the start of (and throughout) a training job, and
periodically write **checkpoints** (multi-gigabyte model-state snapshots) without slowing the job down —
described consistently across sources as a defining CoreWeave design prompt because it's the literal
substrate their GPU customers run on top of.

**Functional Requirements:**
- Serve read access to shared training datasets to thousands of concurrent GPU-attached readers, where
  most readers request largely overlapping (often identical) data.
- Accept periodic checkpoint writes from training jobs (large, bursty, write-once) without those writes
  starving concurrent dataset reads on the same storage tier.
- Support datasets ranging from many-TB training corpora to much smaller fine-tuning sets, without forcing
  small-dataset jobs to pay large-dataset provisioning overhead.

**Non-Functional Requirements:**
- Aggregate read throughput must scale with GPU count — a training job's GPUs should not sit idle waiting
  on data, since idle GPU-hours are the direct cost this design is graded against.
- Checkpoint writes must be durable before a job can safely resume from them — a checkpoint that's
  "written" but not actually durable defeats the purpose of checkpointing for fault recovery.
- Cost-aware tiering: not all data is equally hot, and treating a multi-TB corpus as uniformly
  high-performance storage is wasteful when only a working-set slice is actively read per epoch.

**High-Level Design:**
1. **Origin store + aggressive local/regional caching**: durable datasets live in an object-storage-style
   origin (cheap, durable, not optimized for massive fan-out reads). In front of it, a caching tier —
   ideally NVMe local to or near the GPU nodes — serves the actual hot reads; since most readers in a job
   request the same shards, cache hit rate should be very high after the first epoch's worth of cold
   reads.
2. **Sharded, parallel dataset layout**: datasets are pre-sharded so that thousands of readers pull
   **different shards in parallel** from the origin on a cold cache (rather than all serializing behind
   one file/object), and the caching tier itself is distributed/sharded across nodes so no single cache
   node becomes the bottleneck for a dataset every GPU in the job needs.
3. **Read/write path separation**: checkpoint writes go through a separate path (or at minimum separate
   bandwidth/IOPS budget) from dataset reads, so a burst of checkpoint traffic from one job doesn't degrade
   read throughput for datasets other jobs are actively training on — the same "don't let bursty writes
   starve steady reads" principle as isolating an indexing pipeline from a query path.
4. **Checkpoint durability**: a checkpoint write is only acknowledged as complete once replicated
   (or otherwise made durable, e.g. erasure-coded) — a job resuming from a checkpoint that turns out to be
   partially written or lost is strictly worse than the job having no checkpoint at all, since it can
   silently resume from corrupted state.
5. **Tiering by access temperature**: an actively-training job's current-epoch working set sits on the
   fastest tier available; older checkpoints and cold/archival datasets roll down to cheaper tiers
   automatically, with an explicit (and visibly slower) rehydration path back to hot storage when needed —
   the same hot/cold tiering shape as a log-search platform, applied to training artifacts instead of logs.

**Data Model (sketch):**
```
dataset_shards(dataset_id, shard_id, origin_uri, size, cache_status)
checkpoint_manifest(job_id, checkpoint_id, shard_uris[], durable, created_at)
cache_nodes(node_id, capacity, current_occupancy, colocated_gpu_nodes[])
```

**Scaling & Reliability:**
- Cache placement should be topology-aware in the same sense as GPU scheduling — a cache shard colocated
  with (or network-close to) the GPU nodes actually consuming it avoids turning the cache tier's own
  network into the new bottleneck.
- A cold-start job (nothing yet cached) is the worst case for read throughput; pre-warming the cache from
  the scheduler's own knowledge of "this dataset is about to be needed by a job about to be gang-scheduled"
  turns a foreseeable cold start into a non-event.
- Checkpoint write bursts are naturally correlated across many concurrently-running jobs (everyone tends to
  checkpoint on similar cadences); rate-limit or stagger checkpoint scheduling per job slightly rather than
  letting every job's checkpoint timer coincide and saturate the write path simultaneously.

**Follow-Up Questions:**
1. How do you keep a single popular dataset from making one cache shard a hotspot? → replicate hot shards
   across multiple cache nodes (not just the one that happened to cache it first) once read frequency
   crosses a threshold, and load-balance readers across the replicas.
2. What happens to a training job if the storage layer can't keep up and GPUs start stalling on reads? →
   surface this as a first-class, visible signal (GPU utilization dipping due to data-starvation, not
   compute) back to the customer/scheduler, rather than a silent slowdown indistinguishable from normal
   variance — matches the broader theme of making infrastructure cost (wasted GPU-hours) legible.
3. How would this design change for a dataset far larger than any single node's local cache? → lean harder
   on the sharded/parallel-read origin path and accept a lower cache-hit ratio, but keep the same
   separation of read vs. checkpoint-write bandwidth — the tiering strategy changes, the isolation
   principle doesn't.

---

## 8. Interview Process Overview

Reports converge on a loop that's fairly consistent for software-engineering roles, with some variation by
level and by track (Production Engineer loops run a slightly different shape than core Software Engineer
loops):

- **Recruiter screen**, followed by a **~60-minute technical phone screen** (candidate's choice of Go or
  Python in reports), typically a medium-difficulty data-structures/algorithms problem framed around
  CoreWeave's actual domain — scheduling, resource access graphs, log/telemetry parsing — rather than a
  generic LeetCode-style prompt with no context.
- **Virtual onsite**, commonly structured as: **two coding rounds**, **one system design round** (usually
  distributed-systems- or infrastructure-flavored, per the problems above), **one "craft"/technical
  deep-dive round**, and **one behavioral round** — though individual reports describe anywhere from four
  to five rounds, and at least one report explicitly names separate **leadership** and **technical expert**
  rounds distinct from the standard behavioral round.
- Multiple reports mention direct **discussions with management and/or an engineering director** as part
  of the loop, not just peer-level interviewers — consistent with a still-fast-growing company where senior
  leadership stays involved in hiring decisions.
- **Production Engineer** loops (a distinct track from core Software Engineer) follow a similar
  recruiter → panel structure but weight operational/on-call judgment more heavily relative to algorithmic
  coding depth.
- End-to-end timeline is reported around **3–5 weeks**, and CoreWeave is described across multiple reports
  as moving **faster than typical** post-onsite (debrief/decision calls within about a week), which is
  worth knowing so a quick turnaround isn't mistaken for a red flag.

---

## 9. Behavioral Themes

CoreWeave's behavioral and leadership rounds lean heavily on the operational reality of running GPU
infrastructure that customers' expensive, time-sensitive training jobs depend on — see
[`behavioral_interview.md`](./behavioral_interview.md) for general STAR-method prep. Themes specific to
CoreWeave's loop:

- **Incident ownership, start to finish**: have a story that covers the full arc — detection, response,
  and a genuine postmortem/follow-up — not just "I fixed a bug." The infrastructure framing (GPU nodes,
  clusters, training jobs) means interviewers are listening for how you reason about blast radius and
  customer impact under pressure, not just root-causing.
- **On-call discipline**: expect direct questions about sustaining a healthy on-call rotation — how you've
  handled being paged, how you've reduced page volume for a team, or how you've balanced on-call load
  against feature work. This shows up more heavily in Production Engineer loops but appears in SWE
  behavioral rounds too.
- **Customer-impact framing, quantified**: CoreWeave's own interview guidance frames technical decisions in
  terms of concrete cost — "a stopped training job wastes thousands of expensive GPU-hours." A story that
  connects a technical decision to a quantified customer or cost impact lands better here than one framed
  purely in engineering terms.
- **Operating through hypergrowth**: CoreWeave has scaled extremely fast; reports describing separate
  leadership/director conversations suggest interviewers probe for comfort operating with less process and
  more ambiguity than a mature org — have a story about shipping or leading through a period of fast
  organizational change.
- **Working session, not just Q&A**: several reports describe management/director rounds as genuine
  technical discussions rather than scripted behavioral questions — be ready to defend a design decision
  conversationally with a senior leader in the room, not just recite a prepared STAR answer.

---

## References

Sources used for compiling these questions:
- [CoreWeave Interview Questions — 1point3acres](https://www.1point3acres.com/interview/company/coreweave)
- [CoreWeave Technical Phone Interview: Data Filtering Algorithm — 1point3acres](https://www.1point3acres.com/interview/post/7502269)
- [CoreWeave SDE 2 Interview Experience: CLI Tool with Concurrency — 1point3acres](https://www.1point3acres.com/interview/post/7518478)
- [CoreWeave Onsite Interview Experience for Software Engineer Role — 1point3acres](https://www.1point3acres.com/interview/thread/1169570)
- [CoreWeave Onsite Software Engineer Interview Experience — 1point3acres](https://www.1point3acres.com/interview/thread/1156266)
- [CoreWeave Onsite Interview Experience for Fulltime SDE Position — 1point3acres](https://www.1point3acres.com/interview/thread/1143428)
- [Find a Directed Resource Access Path (CoreWeave Online Assessment) — FastPrep](https://www.fastprep.io/problems/coreweave-resource-access-path)
- [CoreWeave Online Assessment and Interview Insights — FastPrep](https://www.fastprep.io/hiring-insights/coreweave)
- [What to Expect in the CoreWeave System Design Interview — DesignGurus](https://www.designgurus.io/answers/detail/what-to-expect-in-the-coreweave-system-design-interview)
- [CoreWeave Interview Guide (2026): Specialty GPU Cloud — TechInterview.org](https://www.techinterview.org/companies/coreweave-interview-guide/)

Note: 1point3acres' individual interview-report threads require forum membership to view full question
text/discussion, and third-party prep-guide sites (DesignGurus, TechInterview.org) synthesize aggregated,
unverified community reports rather than publishing firsthand transcripts. The problems above were
reconstructed and expanded from the publicly visible report summaries and FastPrep's fuller public problem
writeup into complete, solvable problem statements with original test cases and solutions; the system
design topics reflect the recurring pattern independently corroborated across multiple sources rather than
a single verbatim transcript.
