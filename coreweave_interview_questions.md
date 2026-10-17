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
8. [Notification System](#8-system-design--notification-system)

**Database Design**
9. [Database Design Interview — Database System](#9-database-design-interview--database-system)

**Process & Behavioral**
10. [Interview Process Overview](#10-interview-process-overview)
11. [Behavioral Themes](#11-behavioral-themes)

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
   re-implemented per operator. Dispatch the range operators (`<`,`<=`,`>`,`>=`) with a plain `if/elif`
   ladder inside that single `try/except`, not a dict literal like `{"<": a < b, ">": a > b, ...}[cmp]` —
   a dict literal evaluates every branch's value before indexing into it, so it silently does 4x the
   comparisons needed and can raise from an operator nobody asked for; an `if/elif` only ever evaluates
   the one operator actually requested.
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
        if cmp == "<":
            return actual < expected
        if cmp == "<=":
            return actual <= expected
        if cmp == ">":
            return actual > expected
        return actual >= expected  # cmp == ">="
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
| One server returns `429` with `Retry-After: 1`, then `200` | `"success"`, and the `429` did not consume the `5xx` retry counter |
| One server returns `503` twice then `200` | `"success"` after 3 total attempts, other servers unaffected |
| One server returns `404` | `"failed: not_found"` immediately, no retry attempted for it |
| One server returns `500` past `max_retries` | `"failed: transient"` after the last retry, others still complete normally |
| One server returns `429` forever | `"failed: rate_limited"` once its own rate-limit budget is exhausted — does not hang `reboot_servers` for the other servers |
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
   consume the same budget as a rate-limit wait). "Doesn't consume the same budget" means **separately
   tracked**, not **unbounded** — give `429` its own counter capped at `max_retries` too. Without that
   cap, a server that's persistently (or adversarially) rate-limited loops forever, and since
   `ThreadPoolExecutor.__exit__` blocks until every submitted future completes, one such server hangs
   `reboot_servers` for the *entire* batch — every other server's result included — not just its own.
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
    rate_limit_waits = 0  # tracked separately from `attempt` — different budget, still bounded
    while True:
        response = client.post(f"/v1/servers/{server_id}/reboot")

        if response.status_code == 200:
            return server_id, "success"

        if response.status_code not in RETRYABLE_STATUSES:
            return server_id, f"failed: {response.json().get('error', 'unknown_error')}"

        if response.status_code == 429:
            rate_limit_waits += 1
            if rate_limit_waits > max_retries:
                return server_id, f"failed: {response.json().get('error', 'unknown_error')}"
            time.sleep(float(response.headers.get("Retry-After", 1)))
            continue  # server-given wait, doesn't consume the 5xx `attempt` counter

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
| 20 URLs, rate limit 5/sec | Total wall-clock time is bounded below by ~(20 − 5) / 5 = 3 seconds — the bucket starts full, so the first `rate` requests burst through instantly and only the remainder are throttled |
| One URL returns 404 | `None` immediately, no retries burned on it |
| One URL times out twice then succeeds | Content returned, counted as 3 attempts against that URL only |
| One URL exhausts all retries | `None` for it; other URLs' results unaffected |
| `fetch(url)` raises (a real socket timeout, connection reset) rather than returning a status | Treated as a retryable failure, not an uncaught exception that crashes `scrape_urls` |
| Rate limiter shared across worker threads, sustained/steady-state traffic | No more than `N` requests/sec once the initial burst (see Key Insight #5) has drained, verified via timestamped mock calls |

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
4. A real `fetch` typically signals a timeout or connection failure by **raising**, not by returning a
   status code (that's how `requests`, `httpx`, and friends actually behave) — wrap the `fetch(url)` call
   itself in `try/except` and route a raised exception into the same retryable path as a `5xx`. Skipping
   this means the one failure mode named explicitly in the requirements (timeout) is exactly the one that
   isn't handled: the exception propagates out of the task, and `future.result()` re-raises it in the
   collecting thread, crashing `scrape_urls` and losing every already-fetched result, not just the failing
   URL's.
5. The bucket is initialized full (`_tokens = rate`), so it grants an initial burst of up to `rate`
   requests before steady-state throttling kicks in — this is standard, correct token-bucket behavior
   (bursting is the point of a token bucket over a naive fixed-rate sleep), but it means wall-clock time
   for a batch is closer to `(len(urls) - rate) / rate` than the naive `len(urls) / rate` — worth being
   precise about that when asked to justify a timing assertion. **Be precise about what "no burst" means
   if an interviewer pushes on it**: with the bucket starting full, a 1-second *sliding* window that starts
   right at the burst can see up to `2 * rate` requests (the instant burst plus a full second of steady
   refill immediately after) — verified empirically (5 instant + ~5 more trickling in over the next
   second, for `rate=5`). The correct, defensible claim is "steady-state throughput never exceeds `rate`/
   sec," not "no 1-second window ever exceeds `rate` requests." Start `_tokens = 0` instead if the
   interviewer specifically wants zero burst, throttled from the very first request.

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


# Variant: `self._tokens = 0` in __init__ instead of `self.rate` throttles from the very first
# request, with no initial burst — trade the burst-tolerant throughput above for a strict "never
# more than `rate` in any window" guarantee (verified: max ~`rate + 1` per 1-second window, the
# +1 being ordinary sampling/boundary noise, not a burst).


def _fetch_with_retry(fetch, bucket: TokenBucket, url: str, max_retries: int):
    attempt = 0
    while True:
        bucket.acquire()
        try:
            status, body = fetch(url)
        except Exception:
            status, body = 599, None  # network-level failure (timeout, reset): retryable, like a 5xx
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

**Detailed Component Breakdown:**

| Component | Responsibility | Tech choice (typical) |
|---|---|---|
| API / admission | AuthN/Z, validates job spec, checks tenant quota, assigns priority class | Stateless API + policy engine |
| Job store | Durable job state machine: `PENDING -> RESERVED -> RUNNING -> (PREEMPTING) -> DONE/FAILED` | etcd / Postgres; optimistic concurrency via version |
| Scheduling queue | Per-priority, per-tenant queues with fair-share and aging | In-memory, rebuilt from the job store on start |
| Scheduler (leader) | Scoring/placement loop, one active leader, hot standbys | Leader election (lease); single-writer for reservations |
| Topology/inventory cache | Tree: DC -> rack/IB leaf -> node -> NVLink domain -> GPU, with free/used/health | In-memory, fed by informer/watch streams |
| Reservation manager | Atomic all-or-nothing reservation with epoch fencing | CAS on GPU records / transaction |
| Node agent (kubelet + device plugin) | Starts/stops workers, reports health, performs checkpoint signal | DaemonSet |
| Preemption controller | Picks victims, signals checkpoint, enforces grace period, requeues | Part of scheduler |
| Health/lifecycle controller | Cordons/drains/replaces bad nodes (see #6) | Controller + CRDs |
| Checkpoint service | Receives/stores checkpoints (see #7) | Object store + cache |
| Quota & accounting | Tracks GPU-hours per tenant, billing and fairness inputs | Time-series + ledger table |
| Observability | Queue depth, wait time, fragmentation, GPU utilization, preemption rate | Prometheus/Grafana |

**Step-by-Step Flow (a 64-GPU gang job):**
1. **Submit**: tenant calls `submit_job{gpus=64, gang=true, topology=nvlink-per-node, priority=P2, ckpt_uri}`.
2. **Admit**: API validates, checks `tenant_quota` (concurrent GPUs, daily GPU-hours); over quota -> reject
   or queue with reason. Job persisted as `PENDING` (durable before ack).
3. **Enqueue**: scheduler's watch picks it up into the tenant's queue; effective priority = base + aging.
4. **Pick next**: each cycle, choose the head job by priority then fair-share (lowest usage/quota ratio).
5. **Filter** candidate nodes: healthy, not cordoned, right GPU model, enough free GPUs, matching labels.
6. **Score/place** (topology): 64 GPUs = 8 nodes of 8; prefer 8 nodes under one IB leaf switch/rack, then
   one pod/DC. Score = fewer switch hops, then best-fit (minimize fragmentation), then power/cost.
7. **Reserve atomically**: write reservations for all 64 GPUs with each GPU's `epoch`; CAS fails if any changed.
   On failure: release everything, back off, retry. Job -> `RESERVED`.
8. **Launch**: node agents pull their assignments, start containers, set up NCCL environment (rank, world size,
   rendezvous address); a barrier waits until all 8 nodes report ready, with a timeout; if not, tear down and
   rescore, so the job never half-starts. Job -> `RUNNING`.
9. **No fit**: choose the lowest-priority victim set whose eviction frees a valid topology; send `SIGTERM`/
   checkpoint request, wait up to the grace period (hard kill after), release GPUs, requeue victim with
   `resume_from=latest_ckpt`.
10. **Fault mid-run**: node reports unhealthy GPU -> controller marks job `INTERRUPTED`, restarts from the last
    checkpoint on a fresh reservation (replacement node swapped in), with retry budget.
11. **Complete**: release reservations, write usage records to accounting, free capacity triggers the next cycle.

**Failure Modes:**
- Scheduler leader dies: standby takes the lease and rebuilds queues + inventory from the stores; running jobs
  are untouched because workers don't depend on the scheduler once started. Epochs fence the old leader's writes.
- Fragmentation: lots of 1-GPU jobs scattered across nodes block 8-GPU gangs. Mitigate with bin-packing small
  jobs onto already-partial nodes, reserving whole nodes for big gangs, and optional defragmentation by
  checkpoint-and-move.
- Gang deadlock: two big jobs each hold half of what they need. Prevent with all-or-nothing reservation
  (no hold-and-wait) or **backfill with a reservation for the head job** so small jobs fill gaps without delaying it.

**How Real Production Systems Do It (and what to copy):**

*CoreWeave SUNK — [Slurm on Kubernetes](https://wf.coreweave.com/blog/sunk-slurm-on-kubernetes-implementations)
("two solutions, one platform, one pool of compute").*
- **Slurm control plane as pods**: standard Slurm daemons (controller, accounting) run as Kubernetes workloads from a
  Helm chart with configurable resource requests; **login pods** give users a normal Slurm interface (`sbatch`,
  `srun`) with Kubernetes hidden behind it. Slurm config, **topology**, prolog/epilog scripts and secrets live in
  ConfigMaps/Secrets mounted into every component.
- **Compute = one `slurmd` pod per bare-metal node (1:1)**, running `munged`, with shared filesystems mounted as
  PVCs. A **Nodeset CRD** defines Slurm nodes in Kubernetes (like a DaemonSet in that it maps one pod per node, but with
  protected updates and scaling) and reports status: candidate nodes matching affinity, nodes assigned as Slurm nodes,
  and ready / running / drain conditions.
- **Syncer**: a Slurm REST client (with caching for large clusters) plus a pod controller that reconciles pod state
  and Slurm state both ways. If Slurm drains a node, the Syncer annotates the pod; if Kubernetes cordons a node (e.g.
  a hardware fault), the matching Slurm node is drained with a reason. This is the "single source of truth for
  health" answer in the follow-ups.
- **Slurm as a Kubernetes scheduler**: a custom scheduler (own namespace) places *Kubernetes* pods using Slurm
  state. A pod scheduled this way (e.g. a Knative inference pod) creates a **Slurm job**; Slurm can **preempt
  lower-priority work** to satisfy it, and once Slurm allocates the nodes the scheduler places the pod there. Result:
  reserved/on-demand GPUs serve both batch training and serverless inference from one pool, raising utilization.
- **Topology**: generated automatically and stored in Slurm config; [SUNK docs](https://docs.coreweave.com/docs/products/sunk/concepts/overview)
  state support for Slurm's **tree** and **block** topology plugins.
- **Why this is a good answer**: it reuses Slurm's decades of gang scheduling, backfill, fair-share, accounting and
  preemption (all hard to rebuild) and uses Kubernetes for what it is good at: containers, lifecycle, HA, GitOps.
  If asked "build vs reuse", say you'd reuse Slurm (or Kueue/Volcano on plain Kubernetes) and invest in topology
  data, health integration and quota policy, which is where the CoreWeave-specific value is.

*How this maps to our steps:* admission and quotas = Slurm accounting/QOS; queue + priority + aging = Slurm
multifactor priority; filter/score with topology (step 6) = tree/block topology plugin; atomic reservation (step 7)
= Slurm's node allocation before launch; preemption (step 9) = Slurm preemption with checkpoint; node health (step
10) = Syncer + drain integration (details in #6).

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

**Detailed Component Breakdown:**

| Component | Responsibility | Notes |
|---|---|---|
| GPU device plugin (DaemonSet) | Advertises `nvidia.com/gpu`, runs low-level GPU health checks (Xid errors, ECC, NVLink state) | Per-device health, not just node `Ready` |
| Node health checker | Periodic deeper checks: GPU burn/diagnostic, NCCL/InfiniBand connectivity, disk, driver | Runs on idle nodes and after each job |
| Node conditions / taints | Publishes failures as Kubernetes node conditions; cordon/taint | Automation keys off conditions |
| Topology CRDs + controller | NVLink domain, IB leaf-switch and rack membership as custom resources | Inventory source for scheduler |
| Lifecycle controller | Debounce -> cordon -> drain -> reboot/reprovision/replace; rate-limited | The "no 3am pager" box |
| Fleet provisioning / repair | Reimage, reboot, RMA ticket, physical swap, rejoin | Humans only for hardware |
| Scheduler sync (e.g. Syncer) | Mirrors drain/cordon state between Kubernetes and the job scheduler | One truth for health |
| Telemetry pipeline | DCGM/exporter metrics, events -> Prometheus/Kafka | Detect flapping, correlated failures |
| Alerting | Pages on fleet-level symptoms (replacement rate, failures per rack), not per-GPU blip | Reduces page volume |

**Step-by-Step Flow (a GPU develops an uncorrectable ECC error):**
1. Device plugin sees an **Xid/ECC error** from the driver within its health-check interval (seconds).
2. It marks that device `Unhealthy`; the node's advertised GPU count drops 8 -> 7 so **no new pod can request it**.
   A pod already using that GPU is flagged.
3. Health checker confirms (re-runs diagnostic N times / waits the debounce window) to rule out a transient blip.
4. Confirmed: lifecycle controller sets a **node condition** (e.g. `GpuUnhealthy`) and, if the failure is
   node-wide (all GPUs, NVLink fabric, NIC), **cordons** the node; if it is one GPU, only that device is withdrawn.
5. The job layer is told through the scheduler sync: the affected job is checkpointed/failed and **requeued** on
   healthy hardware; unrelated pods on the same node continue.
6. **Drain**: workloads depending on the bad device are evicted; the controller waits for them to exit.
7. **Remediate** by class: transient/software -> reboot and re-verify; persistent -> **triage**: remove the node
   from the cluster and request replacement; physical fault -> repair ticket / swap with a spare node.
8. **Rejoin gate**: a rebooted or replaced node is not schedulable until it passes health + NCCL/InfiniBand
   verification with its topology peers.
9. **Undrain** only after sustained health (hysteresis), then topology CRDs are updated and the scheduler sees capacity.
10. Everything is logged as events for fleet analytics (failure rate by rack, driver version, batch).

**Failure Modes:**
- Correlated failures (power/cooling/bad firmware): cap concurrent drains and replacements (e.g. never drain more
  than X% of a rack at once); alert on *rate*, not individual events.
- Silent data corruption/straggler GPU: a GPU that is "healthy" but slow stalls a synchronous job; detect with
  per-rank step-time monitoring and pre/post-job diagnostics.
- Device plugin crash: fail closed, so GPUs withdraw from allocatable until it returns.

**How Real Production Systems Do It (and what to copy):**

*CoreWeave — SUNK node lifecycle ([drain/undrain docs](https://docs.coreweave.com/products/sunk/manage_sunk/drain-and-undrain-nodes.md)).*
- **Health check after each job**: Node Health Check (NHC) runs in the **Slurm epilog** after a job, but only when no
  other job is running on that node. A failed check **drains** the node and the drain reason starts with `NHC:`.
- **Drain semantics**: a Slurm-drained node accepts no *new* jobs but lets running jobs finish; unlike Kubernetes
  drain it does not evict. (Copy this distinction: "stop new work" and "evict existing work" are separate
  steps and a design should state which one applies.)
- **Periodic verification of idle nodes**: CoreWeave's **HPC Verification** framework tests idle nodes hourly. If a
  drain reason contains `sunk:verify-undrain`, SUNK **auto-undrains** the node after the next passing test; if it
  fails or the node never becomes idle, it stays drained. This is the "sustained health before uncordoning" rule.
- **Kubernetes-driven drains**: when a Kubernetes node condition fires, CoreWeave's lifecycle automation **cordons**
  the node and the Syncer mirrors it into Slurm as a `k8s: cordon:` drain reason. Documented outcomes: **reboot**
  (node returns to production and the drain clears; allow about an hour once idle, about three hours before it is
  considered stuck), or **triage** (node removed for repair and replaced; the new node joins Slurm as a *new* node).
  Administrator reasons (`AdminMaintenanceMode`, `AdminPermanentFailure`) persist until a person clears them, so a
  human decision can't be overridden by automation. **Rolling updates** undrain a node once its `slurmd` pod
  restarts.
- **Failed prolog/epilog** can also drain a node, and those cases may need manual troubleshooting.
- The [SUNK overview](https://docs.coreweave.com/docs/products/sunk/concepts/overview) states SUNK is integrated with
  CoreWeave Kubernetes Service (CKS) hardware lifecycle management: node operators act on unhealthy nodes, and that
  information propagates both ways between CKS and SUNK.

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

**Detailed Component Breakdown:**

| Component | Responsibility | Notes |
|---|---|---|
| Object store (origin) | Durable dataset + checkpoint storage; S3-compatible API | Erasure-coded, multi-AZ |
| Gateway + index layer | Authenticates, maps bucket/key -> object location, handles metadata | Scales independently of data |
| Per-node cache proxy (LOTA-style) | Local S3-compatible proxy on every GPU node, caches objects on node NVMe | Training code only changes endpoint URL |
| Distributed cache index | Knows which node holds which object so peers/clients find it | Global cache across nodes |
| Dataset loader (client) | Sharded, parallel, prefetching reader (WebDataset/Mosaic-style) | Keeps GPUs fed |
| Checkpoint writer | Async, sharded write of model state; atomic manifest commit | Separate path/QoS from reads |
| Cache pre-warmer | Loads upcoming dataset into cache before job start | Driven by scheduler |
| Tiering/lifecycle manager | Moves old checkpoints/data to cheaper tiers, expires | Policy-driven |
| Quota/QoS | Per-tenant bandwidth and IOPS limits | Prevents noisy neighbors |
| Metrics | Per-GPU data-wait time, cache hit rate, per-shard latency | Makes starvation visible |

**Step-by-Step Flow:**
*Read path (training epoch)*
1. Scheduler places the job and notifies the **pre-warmer** with the dataset id; the cache starts pulling shards.
2. Each rank's dataloader is assigned a **disjoint set of shards** (rank-based sharding) and starts N parallel
   prefetch threads against the local proxy endpoint.
3. Proxy checks the local NVMe cache: **hit** -> serve at local-disk speed. **Miss** -> fetch from the object store
   (via gateway/index), write to NVMe, serve; a read of *any part* of an object can cache the whole object.
4. Subsequent epochs hit the cache; steady state GPUs never touch the origin.
5. Metrics record hit rate and per-step data-wait; sustained wait > threshold raises "data starved" to the job owner.

*Write path (checkpoint)*
1. Every K steps each rank snapshots its shard of model/optimizer state to local memory (fast, short pause).
2. A background thread writes shards to the object store in parallel as multipart uploads, **separate from the read
   path/QoS**, while training continues.
3. When all shards are durable, a writer commits a **manifest** (list of shard URIs + checksums + step number) with
   an atomic put; only a checkpoint with a committed manifest is considered valid.
4. On failure/preemption, the new job reads the latest *committed* manifest and loads shards in parallel.
5. Lifecycle policy keeps the last N checkpoints hot, tiers older ones down, and deletes beyond retention.

**Failure Modes:**
- Partial checkpoint (crash mid-write): no manifest -> ignored; the previous manifest remains valid.
- Cache node lost: objects refetch from origin; hot shards re-replicated; the job slows but doesn't fail.
- Origin throttling on cold start: stagger and prefetch; pre-warm from the scheduler's knowledge of upcoming jobs.
- Checkpoint storms: jitter per-job checkpoint intervals and cap write bandwidth per tenant.

**How Real Production Systems Do It (and what to copy):**

*CoreWeave AI Object Storage (CAIOS) with LOTA —
[benchmark write-up](https://coreweave.com/blog/benchmark-results-coreweave-ai-object-storage-delivers-2-gb-s-per-gpu-throughput-across-any-number-of-gpus),
[CAIOS docs](https://docs.coreweave.com/products/storage).*
- **Architecture**: a **gateway and index layer**, an **object repository**, and a **LOTA proxy on each GPU node**;
  together they form a **global cache across all GPU nodes**. Applications keep using the **S3-compatible** API and
  just point at the LOTA endpoint (`cwlota.com`) with no custom code; the proxy decides cache vs origin.
- **Per-node NVMe cache**: 1 TiB per node in the benchmark (adjustable up to local disk), so 20 nodes = 20 TiB global
  cache. **Whole-object caching**: accessing any segment of an object caches the whole object for later reads.
  **Pre-staging**: users can load datasets into the cache ahead of time, recommended for large multimedia files,
  inference workloads and container images.
- **Hardware path**: benchmark nodes had 8x H200, one **BlueField-3 DPU with dual 100 Gbps links** for storage
  traffic, and separate ConnectX-7 InfiniBand adapters for GPU-to-GPU traffic. **Copy**: storage and
  GPU-collective traffic ride **separate networks**, so reads never compete with NCCL all-reduce.
- **Measured behavior** (20 nodes, Warp S3 benchmark, 10,000 objects of 50 MiB written then random-read for 10 min):
  while the cache was filling (first minute) about **24 GiB/s aggregate**; once warm and served from local NVMe
  **368 GiB/s aggregate = 18.4 GiB/s per node = 2.3 GiB/s per GPU**. The write phase used 10 MiB multipart parts with
  100 threads; **write throughput was not benchmarked** in that post. The cold-vs-warm ~15x gap is the number to quote
  for why caching and pre-warming matter. The "scales to any number of GPUs" claim is stated as expected, with
  customer corroboration, not as part of the 20-node test.
- **Vendor-reported scale** (CoreWeave materials, not independently verified): a frontier-model customer running LOTA
  across 15,000+ GPUs with ~20 PB of cache at about 99.7% hit rate; LOTA can also accelerate reads of a bucket in
  another CoreWeave region without keeping a second copy.
- **Design lessons to say out loud**: cache on the node (NVMe is the cheapest high-bandwidth tier), make the cache
  transparent behind S3 so users change one URL, separate storage and collective networks, and measure
  cold vs warm separately.

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

## 8. System Design — Notification System

**Problem Statement:**
From CoreWeave's own interview-prep note for the loop (not a third-party report): *"System Design Interview
(notification system): review the fundamentals of designing scalable distributed systems. Ask clarifying questions
before proposing a solution, define your assumptions and key tradeoffs, and discuss how your design addresses
scalability, reliability, fault tolerance, and varying traffic patterns... explain how components interact and how the
design can evolve as requirements change. The interview is collaborative."*
So the grading is on **process**: clarify, state assumptions, justify tradeoffs, handle bursts and failures, and evolve
the design. The prompt is generic, so frame it for CoreWeave's world: customers (and internal on-call) must be told
about GPU job events, node and rack failures, maintenance windows, quota and billing thresholds, and incidents.

### Step 0. Clarifying questions to ask (say these out loud first)
1. **Who are the recipients?** External customers (tenants with many users), internal engineers/on-call, or both?
2. **What triggers notifications?** System events (job failed, node drained), scheduled (maintenance reminders), user-
   configured alerts, billing/quota thresholds? Do customers define their own alert rules, or only subscribe?
3. **Channels?** Email, webhook, Slack/Teams, SMS, PagerDuty, in-app inbox? Is webhook (machine consumption) first-class?
4. **Priority classes?** Is there a "critical, deliver in seconds" tier (node failure killing a training job) versus
   "bulk, can wait" (weekly usage report)?
5. **Delivery guarantee?** At-least-once with dedupe, or best-effort? Is a missed critical alert acceptable? (Never.)
6. **Latency targets** per priority, **volume**, and **burst** shape (a rack failure produces a storm).
7. **User controls:** preferences, quiet hours, digests/batching, mute, per-tenant rate caps?
8. **Compliance/audit:** must we prove a notification was sent (SLA credits, incident records)?
9. **Multi-region** and data-residency constraints?

### Assumptions (stated, hypothetical, to anchor the math)
- 5,000 tenants, 100,000 end users; ~2 channels per subscription on average.
- Steady state 100 events/s ingested -> ~300 deliveries/s after fan-out. **Incident burst**: a rack/power event or a
  bad rollout raises events 50-100x for 5-10 minutes: ~10,000 events/s, which would fan out to 30,000 deliveries/s
  if not collapsed.
- Critical delivery within 10 s (p99); standard within 60 s; bulk/digest within minutes.
- Retention: delivery records 90 days hot, 1 year archived (audit).
- Storage back-of-envelope: 100 events/s x 86,400 = 8.6M events/day x ~1 KB = ~9 GB/day of events, ~3x that for
  delivery records; fits a single sharded relational cluster, not a data-warehouse problem.

**Functional Requirements:**
- Producers publish events; users subscribe by tenant/project/resource/severity and channel.
- Route each event to the right recipients and channels, rendering from templates.
- Support dedupe, aggregation (many events -> one incident message), digests, quiet hours, and escalation.
- Provide delivery status, retries, in-app inbox, and an audit trail.

**Non-Functional Requirements:**
- Critical notifications are **not delayed by bulk traffic** (isolation) and **survive failures** of the notification
  system's own components; no silent loss.
- Idempotent: the same event never produces duplicate messages (at-least-once internally, effectively-once to users).
- Elastic under bursts; protects external providers and customer endpoints from overload (rate limits).
- Observability: end-to-end latency per priority, queue lag, provider error rates.

### Component Breakdown

| Component | Responsibility | Tech choice (typical) |
|---|---|---|
| Event producers | Scheduler, node-health controller (see #6), billing, SUNK/Kubernetes events, API | Publish via an **outbox** so the DB write and the event cannot diverge |
| Ingestion API + schema registry | Validates events (`event_id`, `type`, `tenant_id`, `severity`, `subject`, `dedupe_key`, `ts`) | Stateless gRPC/HTTP; Avro/Protobuf |
| Event bus | Durable, replayable, partitioned by `tenant_id` (or subject) | Kafka; separate topics per priority tier |
| Dedupe + aggregation | Drops repeats, groups storms into incidents | Stateful stream job (Flink/Kafka Streams) or Redis with TTL |
| Subscription / preference service | Who wants what, on which channel, quiet hours, escalation policy | Relational DB (source of truth) + cache |
| Router / fan-out | Resolves recipients, picks channels, renders template, writes delivery records | Stateless workers consuming per-priority topics |
| Delivery queues | Per-channel, per-priority queues with delayed-retry topics | Kafka topics (`email.critical`, `webhook.standard`, `retry.1m`...) |
| Channel workers | Email, webhook, Slack, SMS, PagerDuty adapters; per-destination rate limit, circuit breaker | Worker pools scaled per channel |
| Scheduler | Digests, quiet-hours release, maintenance reminders, escalation timers | Durable due-time index (poll), not in-memory timers |
| Delivery store | `deliveries` state machine + audit trail | Sharded relational DB; archive to object storage |
| In-app inbox | Per-user feed, read/unread | KV/wide-column store keyed by user + WebSocket/SSE push |
| Template service | Localized templates, versioned | Config store |
| Observability + meta-monitoring | Metrics, tracing by `event_id`, **dead-man's-switch** | Prometheus + a *separate* alert path |

### Step-by-Step Flow (a rack loses power; 64 nodes go unhealthy)
1. **Detect/publish**: the node-health controller sees 64 nodes `NotReady` and each affected job fails. It writes
   events through its outbox: `node.unhealthy` x64 (`subject=node/<id>`, `rack=R12`, `severity=critical`) and
   `job.failed` per job. The relay publishes them to Kafka (`events.critical`).
2. **Validate**: ingestion rejects malformed events to a dead-letter topic with an alert; accepted events get a
   server timestamp.
3. **Dedupe**: a short-window check on `dedupe_key = type:subject:state` (Redis `SET NX` with TTL ~10 min, or
   stream-state) drops flapping repeats (a node going NotReady->Ready->NotReady).
4. **Aggregate**: the stream job groups by `(tenant_id, incident_class, rack)` using a **group-wait** window
   (e.g. 20-30 s): 64 node events plus 40 job failures for one tenant become **one incident notification** ("rack R12
   power event: 64 nodes affected, 40 jobs failed, checkpoints available"). The first notification still goes out fast
   (send at first event, then update); later changes are batched into follow-ups on a **group-interval** (see
   Alertmanager below). Critical events skip long batching windows.
5. **Resolve recipients**: the router loads the tenant's subscriptions (cached, invalidated on change), applies
   severity thresholds, quiet hours (critical bypasses), and the escalation policy (on-call primary, then secondary).
6. **Fan-out + render**: for each recipient/channel it creates a `deliveries` row (`delivery_id =
   hash(event_or_incident_id, recipient, channel)`, `status=QUEUED`) in the same transaction as enqueueing (outbox
   again), renders the template, and publishes to `webhook.critical`, `email.critical`, etc.
7. **Send**: a channel worker takes the message, checks the **per-destination token bucket** (per customer webhook,
   per provider), sets `status=SENDING`, and calls out with a timeout, passing the `delivery_id` as the provider
   idempotency key where supported.
8. **Outcome**: success -> `DELIVERED` (webhooks: 2xx; email: provider accepted, with bounce/complaint webhooks
   updating status later). Retryable failure (timeout, 5xx, 429) -> publish to a delayed retry topic with
   exponential backoff plus jitter (`retry.5s` -> `1m` -> `10m` -> `1h`); non-retryable (4xx, hard bounce) -> `FAILED`
   and suppress the address.
9. **Escalate**: if a critical page isn't acknowledged within N minutes, the scheduler's due-time index fires the next
   step in the escalation policy (secondary on-call, then SMS/phone).
10. **Customer webhook broken**: a per-endpoint circuit breaker opens after repeated failures; deliveries park with
    backoff; after days of failure the endpoint is disabled and the tenant is notified through a different channel.
11. **Audit/inbox**: every state change appends to the audit log; the in-app inbox gets the incident entry and pushes
    over WebSocket/SSE.
12. **Resolution**: when nodes recover, a `resolved` event updates the same incident thread and sends one "resolved"
    message (not 64).

### Data Model (sketch)
```
subscriptions(sub_id, tenant_id, user_id, event_filter, min_severity, channel, destination, quiet_hours, escalation_policy_id)
events(event_id PK, tenant_id, type, subject, severity, payload, dedupe_key, ts)          -- retained 7-30 days, source: Kafka
incidents(incident_id PK, tenant_id, group_key, status, first_ts, last_ts, event_count)  -- aggregation state
deliveries(delivery_id PK, incident_or_event_id, recipient, channel, status, attempt, next_attempt_ts, last_error, provider_ref)
     INDEX (status, next_attempt_ts)      -- retry/sweeper scan
     PARTITION BY tenant_id hash / time   -- 90 days hot
inbox(user_id, notif_id, ts, read)         -- wide-column, partitioned by user_id
```

### Handling Varying Traffic Patterns
- **Burst absorption**: Kafka buffers; consumers autoscale on lag; critical topic has reserved capacity so bulk lag
  never delays it (priority isolation by topic and consumer pool, not by in-queue ordering).
- **Collapse the fan-out**: aggregation turns a 10,000 events/s storm into hundreds of incident updates (the single
  most effective burst control).
- **Load shedding**: under sustained overload, drop or delay *bulk* and digests first; never shed critical.
- **Per-tenant fairness**: tenant rate limits and weighted fair consumption so one noisy tenant (a flapping alert)
  can't starve the others.
- **Backpressure to external systems**: token buckets per provider and per customer endpoint; honor `429/Retry-After`.
- **Retry storms**: jittered backoff and retry topics with their own bounded consumers, so retries don't amplify
  an outage.
- **Daily/weekly patterns**: digests and reports scheduled off-peak, spread with jitter, not at 09:00:00 sharp.

### Reliability and Fault Tolerance
- **At-least-once + idempotency**: producers retry; `event_id`/`delivery_id` uniqueness constraints make duplicates
  harmless. Exactly-once to a human is approximated, not guaranteed (an email can be sent but the ack lost); say so.
- **No single points**: stateless services behind load balancers; Kafka replicated (RF=3, `acks=all`,
  min.insync.replicas=2); DB with synchronous replica for the delivery store (a lost `deliveries` row loses a page).
- **Crash recovery**: a sweeper requeues deliveries stuck in `SENDING` past a lease; because sends are idempotent, a
  second attempt is safe.
- **Circular dependency**: the notification system must not run on the infrastructure it reports on. If GPU clusters
  or the control plane are down, alerts must still flow: deploy in separate failure domains/regions, use an
  independent secondary path (e.g. a provider-hosted paging service) for critical alerts, and run a **dead-man's
  switch**: a heartbeat that pages if the notification pipeline itself goes quiet.
- **Multi-region**: active-active ingestion; each region delivers its tenants; on failover the idempotency key and
  delivery store prevent re-sends. Cross-region Kafka replication (or dual-publish) for critical events.
- **Poison messages**: after N failures send to a DLQ with an alert; never block the partition.

### Evolution Path ("how can the design change as requirements change?")
1. **MVP**: single Kafka topic, router, email + webhook workers, relational delivery table.
2. Add priority topics, rate limits, retries, preferences UI.
3. Add dedupe + aggregation (storm protection) and escalation policies.
4. Add digests/batching, in-app inbox, more channels via an adapter interface.
5. Customer-defined alert rules (query-based alerting on metrics) feeding the same bus.
6. Multi-region active-active, tenant-level SLAs, and per-tenant dedicated capacity for large customers.

### Trade-offs to State Explicitly
| Decision | Option A | Option B | Our choice |
|---|---|---|---|
| Delivery semantic | At-most-once (never duplicate) | At-least-once + dedupe | B, because missing a critical alert is worse than a duplicate |
| Batching window | Immediate, one message per event | Group over a window | Hybrid: first message immediate, updates batched |
| Dedupe state | In-memory/Redis (fast, can lose) | Durable in the DB (slower) | Redis for short windows; DB uniqueness as the backstop |
| Scheduling | In-memory timers | Durable due-time index | Durable index (survives restarts) |
| Queue per priority | One queue with priorities | Separate topics/pools | Separate topics, because ordering within one queue still lets bulk delay critical |

### How Real Production Systems Do It (and what to copy)

*Prometheus Alertmanager (the open-source standard for alert routing; documented behavior).* It takes alerts from many
sources and applies, in order: **grouping** (`group_by` labels, so many alerts become one notification), timing knobs
(`group_wait` to collect the first batch before sending, `group_interval` between updates for the same group,
`repeat_interval` before re-notifying unresolved alerts), **inhibition** (suppress downstream alerts when a root-cause
alert is firing, e.g. suppress per-node alerts when the whole-rack alert fires), **silences** (time-boxed mutes), and
**routing trees** to receivers (email, Slack, PagerDuty, webhook). **Copy**: the rack-power walkthrough above uses
exactly grouping + inhibition + group timers; name Alertmanager in the interview as the reference design.

*Knock's batched-notification engine ([engineering write-up](https://knock.app/blog/building-a-batched-notification-engine)).*
A vendor's published design for the digest/batching piece: **batch on write** (as each event arrives, find an open
batch for `(recipient, batch_key)`; if none, create one with `closes_at = now + window`; attach the event) versus
**batch on read** (periodically query unsent notifications and group them). Write-time batching costs extra storage but
keeps flush queries cheap as data grows. Flushing is either a delayed job at window end (risk: the queue drops
scheduled jobs) or a cron scan for due, unprocessed batches (robust, with up to one polling interval of extra
latency); `processed_at` records completion so failures can be found and reprocessed. They list the open problems
too: **race conditions creating duplicate batches**, **exactly-once sends**, **delivery retries**, and **unbounded batch
size** (cap and flush early, keep a total count separate from displayed items). **Copy**: use the cron/due-time index
flush with `processed_at`, protect batch creation with a unique constraint on `(recipient, batch_key, open)`, and cap
batch size.

*Provider and queue practices from public notification-system write-ups (secondary, not one company's design).*
Separate Kafka topics per priority so a promotional burst cannot block one-time codes; per-channel rate caps set from
provider limits (email/SMS providers enforce quotas); at-least-once delivery with idempotency keys; and the **outbox
pattern** because writing straight to Kafka can lose messages during a Kafka outage. I found no primary engineering
post from Uber/Slack/LinkedIn for their notification internals, so don't attribute these to a named company.

*CoreWeave-specific hook.* CoreWeave's node lifecycle already generates the events this system consumes: SUNK health
checks drain nodes with reasons (`NHC:`, `k8s: cordon:`), and the lifecycle automation reboots or replaces them (see
#6). Mention that the notification system should subscribe to these state transitions and **aggregate by rack/job** so
a customer sees one message about their affected job, with the checkpoint status, not a flood of node events.

### Follow-Up Questions
1. How do you guarantee a critical alert is never lost if Kafka is down? -> outbox in the producer's DB means the event
   waits durably and is relayed when Kafka returns; for the pipeline's own failure, an independent secondary path and
   the dead-man's switch catch silence.
2. How do you prevent alert fatigue? -> aggregation, inhibition, per-tenant caps, severity thresholds, digests for low
   severity, and escalation only on non-acknowledgement.
3. How would you support customer-defined alert rules? -> a rules evaluator reads metrics/events, emits events into
   the same bus; keep evaluation and delivery decoupled.
4. How do you test this? -> replay recorded event storms against a staging cluster, chaos-test provider outages and
   duplicate delivery, and load-test the critical path at 100x steady state.
5. How do you handle ordering (resolved arriving before firing)? -> carry a monotonic per-subject sequence/state
   version; the router ignores older states.

---

## 9. Database Design Interview — Database System

**Problem Statement:**
From CoreWeave's own prep note: *"Database Design Interview (database system): review the fundamentals of database
system design and distributed data management. Be prepared to discuss how you evaluate different database technologies
based on workload and requirements, model and organize data effectively, and think through indexing, replication,
partitioning, consistency, and scalability. We value your reasoning and communication as much as the final solution,
so explain the tradeoffs behind your design decisions."*
The prompt doesn't say which database, so prepare **two modes** and ask the interviewer which they want:
(A) **choose and model data stores** for a described product (most likely, given "evaluate technologies based on
workload"), and (B) **design a database system** itself (a distributed key-value/relational store). Section A first.

### Step 0. Clarifying questions (always ask)
1. What are the **entities and access patterns**? List the top reads and writes by frequency and latency target.
2. **Read/write ratio**, record size, growth rate, and total size in 1 and 3 years.
3. **Consistency needs**: which operations need strong consistency or transactions (money, quotas, reservations),
   and which tolerate staleness (dashboards)?
4. **Query shape**: point lookups, range scans, joins, aggregations, full-text, time-series windows?
5. **Availability / RPO / RTO**, multi-region, and data residency?
6. **Retention and compliance**: deletion rights, audit immutability?
7. **Operational constraints**: team skills, managed vs self-hosted, existing stack?

### Part A. Case study: the data layer for a GPU cloud platform

**Entities and access patterns (stated assumptions):**

| Data | Access pattern | Volume (assumed) | Consistency need |
|---|---|---|---|
| Tenants, users, projects, quotas | Point lookups; low write rate; joins with permissions | Millions of rows | Strong; quota updates transactional |
| Fleet inventory (datacenter, rack, node, GPU, topology, health state) | Frequent reads by scheduler; moderate writes; hierarchical queries | ~10^5-10^6 GPUs | Strong for allocation, cached reads |
| Jobs and reservations | Create/update state machine; lookups by tenant, state; reservation must be atomic | ~10^5 jobs/day, 10^7 historical | Strong (no double-booking) |
| Health events and telemetry | Very high-rate append; time-window queries, retention | ~10^6 samples/s (metrics), ~10^3 events/s | Eventual; ordered per source |
| Usage metering and billing | Append-only usage records, aggregations per tenant per hour, immutable and auditable | ~10^7-10^9 records/day | Exactly-once accounting; strong on invoices |
| Notifications / deliveries (see #8) | Append + status updates; query by tenant, time | ~10^7/day | Strong per row; eventual across |
| Checkpoint and dataset metadata (see #7) | Point lookup by job/checkpoint, manifest commits | ~10^6 | Strong, atomic manifest commit |

**Technology choices (decision table):**

| Workload | Good fit | Why | Avoid when |
|---|---|---|---|
| Transactional entities with joins, constraints (tenants, quotas, jobs, invoices) | **PostgreSQL / MySQL** (sharded by `tenant_id` when it outgrows one primary) | ACID, indexes, constraints, mature tooling; the working set is modest | Write rate or size exceeds one shard and you can't shard by tenant |
| Distributed config, leader election, reservations needing linearizable CAS | **etcd / Postgres row locks** | Raft-based linearizable KV with watch | Large data volumes (etcd is for small, critical state) |
| Massive append + point/range reads by key, high write availability | **Cassandra / ScyllaDB / DynamoDB** | Linear scale-out, tunable consistency, partition-key access | Ad-hoc queries/joins, strong multi-row transactions |
| Time-series and event analytics (telemetry, usage rollups) | **ClickHouse / TSDB** (e.g. Prometheus for metrics, ClickHouse for events) | Columnar compression, fast aggregates, TTL/downsampling | Frequent row-level updates, strong transactions |
| Hot cache, rate limiting, ephemeral state | **Redis** | Sub-millisecond reads, atomic ops, TTL | As the only copy of important data |
| Large blobs (checkpoints, datasets) | **Object store** with metadata in a relational DB | Cheap, durable, parallel throughput (see #7) | Small random updates |
| Search over logs/job names | **OpenSearch/Elasticsearch** (derived index) | Inverted index | As system of record |

**Principle to say out loud:** *one source of truth per datum*, with derived stores (caches, search indexes, analytics)
fed by events/CDC and always rebuildable. Don't pick one database for everything, and don't introduce a store without a
workload that justifies its operational cost.

**Schema sketches and indexes (transactional core):**
```sql
tenants(tenant_id PK, name, plan, created_at)
quotas(tenant_id, resource_type, limit_value, PRIMARY KEY (tenant_id, resource_type))

nodes(node_id PK, rack_id, dc_id, gpu_model, nvlink_domain, state, updated_at)
  INDEX (rack_id), INDEX (state, gpu_model)                     -- scheduler lists free nodes by model
gpus(gpu_id PK, node_id FK, slot, health, epoch)                -- epoch fences concurrent reservations

jobs(job_id PK, tenant_id, state, priority, gpu_count, submitted_at, started_at, finished_at)
  INDEX (tenant_id, state, submitted_at DESC)                   -- "my running jobs" (composite: equality, equality, range)
  INDEX (state, priority DESC, submitted_at) WHERE state='PENDING'   -- partial index: the scheduler's queue only
reservations(job_id, gpu_id, epoch, reserved_at, PRIMARY KEY (job_id, gpu_id), UNIQUE (gpu_id) WHERE released_at IS NULL)

usage_records(record_id PK, tenant_id, job_id, gpu_model, start_ts, end_ts, gpu_seconds, source_event_id UNIQUE)
-- append-only; UNIQUE source_event_id makes metering idempotent
invoices(invoice_id PK, tenant_id, period, total_minor BIGINT, status)       -- money as integer minor units
```
Indexing notes: composite index order is *equality columns first, then range/sort*; use **partial indexes** for hot
subsets (pending jobs); the `UNIQUE (gpu_id) WHERE released_at IS NULL` constraint makes **double-booking impossible at
the database level**, not just in application code; avoid over-indexing write-heavy tables (each index taxes writes).

**Partitioning (sharding) strategy:**
- Transactional core: shard by `tenant_id` (hash). Tenant-scoped queries stay on one shard; cross-tenant admin queries
  go to a read replica or the analytics store. Large tenants can be given a dedicated shard.
- `usage_records` and telemetry: **time-partition** (daily/hourly) for cheap retention (drop a partition, don't delete
  rows) plus hash on tenant to spread writes. Avoid a pure time key for writes because "now" becomes a hot partition.
- Avoid unbounded partitions: bucket large partitions (see Discord's channel + time bucket below).

**Replication and consistency:**
- Transactional core: single primary per shard, **synchronous replica** in another zone for zero-loss failover (RPO ~0),
  asynchronous replicas for reads and DR region. Reads that must not be stale (check quota before admit) use the
  primary; dashboards use replicas and tolerate lag.
- Quorum math for leaderless stores: with N=3 replicas, writes W=2 and reads R=2 give R+W>N, so a read overlaps the
  latest acknowledged write; W=1/R=1 is faster but can return stale data. Choose per query, not globally.
- Linearizable reservations: rely on the primary's transaction (or etcd CAS) with the epoch/unique constraint;
  never implement a distributed lock on an eventually consistent store.
- Cross-shard operations (quota transfer between tenants) are rare; use a saga with idempotent steps and
  compensations, not distributed two-phase commit.

**Metering pipeline (exactly-once accounting):**
1. Scheduler/node agents emit `job.started`, `job.stopped`, periodic `heartbeat` usage events with unique `event_id`s.
2. Kafka -> a metering consumer writes `usage_records` with `INSERT ... ON CONFLICT (source_event_id) DO NOTHING`.
3. Hourly rollups (ClickHouse or SQL) compute `gpu_hours` per tenant; invoices read from the immutable records.
4. Late/corrected events create adjustment records (never edit history), mirroring ledger design.
5. Reconciliation compares billed GPU-hours with scheduler allocation logs and flags drift.

**Failure scenarios:**
- Primary dies: promote the synchronous replica; fence the old primary (STONITH/lease) to prevent split brain.
- Hot tenant: per-tenant rate limits, move to a dedicated shard, cache reads.
- Bad migration: expand/contract schema changes (add nullable column -> backfill -> switch -> drop), online index builds.
- Region loss: async cross-region replica with a documented RPO; reservations re-reconciled with actual hardware state
  from the node agents (the hardware is the truth).

### Part B. If asked to design a database system itself (distributed KV store)

**Components:** client/driver, routing layer (partition map), storage nodes (WAL + memtable + SSTables or B-tree pages),
replication module (Raft or quorum), coordinator/placement service (metadata), compaction/GC, rebalancer,
backup/restore, monitoring.

**Write path (leader-based, Raft):** (1) client hashes the key to a partition and finds its leader; (2) leader appends
to its **write-ahead log** and replicates to followers; (3) once a **majority** acknowledges, the entry is committed;
(4) leader applies to the memtable (LSM) and acknowledges the client; (5) memtable flushes to immutable SSTables,
background **compaction** merges them, with bloom filters to skip files on reads.
**Read path:** route to leader (linearizable) or a follower (stale-tolerant, or use read leases / ReadIndex);
check memtable, then SSTables newest to oldest using bloom filters and block indexes.
**Partitioning:** consistent hashing with virtual nodes (even spread, cheap rebalancing) or range partitioning (ordered
scans, risk of hot ranges, automatic split/merge). **Rebalancing:** move partitions by adding a learner replica,
catching up, then promoting. **Secondary indexes:** local (per partition, scatter-gather reads) vs global (separate
partitioned index, async or transactional updates). **Transactions:** single-partition ACID via the leader; cross-
partition via two-phase commit over Raft groups (Spanner-style) or avoided via data modeling.
**Storage engine trade-off:** LSM trees favor write-heavy workloads (sequential writes, higher read/space amplification
and compaction cost); B-trees favor read-heavy and in-place updates (write amplification and random I/O).
**Consistency menu:** linearizable (leader reads), sequential/read-your-writes (session tokens), eventual (any replica).

### Spoken Answers to Likely Questions

**"How do you decide between SQL and NoSQL?"**
"I start from the access patterns, not the technology. If the data is relational, needs multi-row transactions and
ad-hoc queries, and fits on a handful of shards, I choose PostgreSQL: ACID, constraints and mature tooling remove whole
classes of bugs, which matters for things like quotas and invoices. If the pattern is massive append and lookups by a
known key, with a need for write availability across zones, a partition-key store like Cassandra, Scylla or DynamoDB
scales linearly, but I give up joins and cross-partition transactions and must design tables around queries. For
time-series and event analytics a columnar store like ClickHouse beats both. And I keep one source of truth per fact,
with derived stores fed by events. For this platform: Postgres for tenants, quotas, jobs and invoices; etcd or database
constraints for reservations; ClickHouse for telemetry and usage rollups; object storage for checkpoints with metadata
in Postgres."

**"How do you choose indexes?"**
"I list the top queries, then build the smallest set of indexes that serves them. A composite index puts equality
columns first and the range or sort column last, so `(tenant_id, state, submitted_at)` serves 'this tenant's running
jobs, newest first' from one index scan. I add a partial index for hot subsets like pending jobs so the scheduler's
queue is tiny. A covering index avoids heap lookups for the hottest reads. Every index slows writes and costs storage, so
on write-heavy tables I keep them minimal and verify with `EXPLAIN`. In LSM stores the equivalent is choosing the
partition key and clustering order, because you can't index arbitrary columns cheaply."

**"How do you partition, and what about hot partitions?"**
"I pick a key that matches the access pattern and spreads load: tenant ID for the transactional core, tenant plus time
bucket for usage data. Hash partitioning spreads evenly but loses range scans; range partitioning keeps ordering but can
create hot ranges, so I'd rather hash a prefix and cluster by time inside it. The classic failure is one key getting
disproportionate traffic. Discord's messages are keyed by channel plus a time bucket so no partition grows without
bound, and when a popular channel still created hot partitions they added a data-service layer that coalesces concurrent
reads for the same row into one database query and routes by channel ID so identical requests meet at the same instance.
The general toolbox: bucketing, caching, request coalescing, rate limits, and splitting or isolating the hot tenant."

**"Explain replication and consistency trade-offs."**
"Synchronous replication gives zero data loss on failover but adds write latency and ties availability to the replica;
asynchronous is faster but can lose the last few writes. So I use a synchronous replica in a second zone for the
transactional core, and async replicas for reads and cross-region DR. With quorum systems, N=3 with W=2 and R=2
guarantees a read overlaps the latest acknowledged write; I can drop to R=1 for dashboards. For anything that must never
double-book, like GPU reservations, I use a single-leader linearizable path with a database constraint as the final
guard, not eventual consistency plus hope. Cross-shard workflows use sagas with idempotent steps. And I state the
CAP stance per operation: during a partition, billing and reservations choose consistency, telemetry chooses availability."

**"How do you evolve the schema without downtime?"**
"Expand and contract. Add the new nullable column or table, deploy code that writes both, backfill in batches with
throttling, switch reads, then drop the old field in a later release. Build indexes online or concurrently, and test
migrations on a production-sized copy. For wide-column stores I add new columns freely and version the payload. Rollback
must stay possible until the contract step."

**"How would you migrate a live database to a new one?"** (Discord is the reference.)
"Dual-write new data to both systems, backfill history behind it with a checkpointed, resumable migrator, then validate
by sending a small share of reads to both and comparing results, and only then cut over with a rollback path. Discord
did this to move trillions of messages from Cassandra to ScyllaDB."

### How Real Production Systems Do It (and what to copy)

*Discord: messages on Cassandra then ScyllaDB ([Discord engineering post](https://discord.com/blog/how-discord-stores-trillions-of-messages)).*
- **Data model**: messages are partitioned by **channel ID plus a bucket (static time window)**, replicated across three
  nodes by default; IDs are Snowflakes, so they sort chronologically. The bucket keeps any one partition from growing
  without bound while keeping a channel's recent messages together for fast reads. (The exact bucket size and clustering
  key were not in the text I read.)
- **Why it hurt**: by early 2022 the Cassandra cluster had **177 nodes** and was high-toil. Concurrent reads of one
  hot channel-bucket raised latency across the node and, through quorum reads/writes, across the cluster; falling behind
  on **compaction** made reads costlier; JVM GC tuning and a manual "gossip dance" consumed on-call time; uncompacted
  **tombstones** caused migration timeouts.
- **Mitigation**: Rust **data services** between the API monolith and the databases, with about one gRPC endpoint per
  query and no business logic. They **coalesce** concurrent requests for the same row (the first spawns a worker, later
  requests subscribe to its result) and use **consistent-hash routing** by channel ID so identical requests reach the same
  instance. This reduced spikes but did not eliminate hot partitions; it bought time.
- **Migration**: new ScyllaDB cluster; **dual writes**; a Rust migrator that reads token ranges, **checkpoints progress**,
  and writes up to 3.2 million messages/second (estimate cut from ~3 months with a Spark migrator to ~9 days); validate by
  sending a small share of reads to both and comparing; cut over in May 2022.
- **Result**: **177 -> 72 nodes** (9 TB per node vs ~4 TB), historical-fetch p99 from 40-125 ms to ~15 ms, insert p99
  from 5-70 ms to a steady ~5 ms. These are Discord's reported numbers.
- **Copy**: bucket by time to bound partitions; put a thin coalescing layer in front of hot rows; migrate with dual
  write + backfill + shadow-read validation + cutover.

*Uber payments ledger ([Uber's Payments Platform](https://www.uber.com/blog/ubers-payments-platform)).* Immutable,
append-only money records with a changelog that can rebuild any balance; started on DynamoDB and moved to a custom
tamper-evident store (LedgerStore) as audit needs grew; hot entities needed a serialized batch-write path. **Copy**:
for GPU billing use immutable usage records, adjustments instead of edits, and idempotent ingestion
(`UNIQUE source_event_id`). See the Affirm ledger section in `affirm_interview_questions.md` (#15) for the full pattern.

*Postgres to ClickHouse for telemetry ([Trigger.dev case on ClickHouse's blog](https://clickhouse.com/blog/trigger-dev-scaling-observability), vendor-hosted).*
A team that began with Postgres moved high-volume telemetry to ClickHouse because Postgres wasn't scaling for larger
customers; reported P95 queries around 200 ms and a much smaller storage footprint. **Copy**: separate analytical from
operational workloads. Treat the numbers as vendor-reported.

*CoreWeave's own stack.* I found no public CoreWeave engineering post describing its databases, so don't claim what
CoreWeave uses; frame choices from workload and say you'd want to learn their current stack.

### Follow-Up Questions
1. Why not put everything in one Postgres? -> works early; at scale telemetry and usage volume (10^6 samples/s)
   overwhelm row-store write and storage costs, and analytics queries hurt OLTP latency, so split by workload.
2. How do you prevent two schedulers double-booking a GPU? -> a unique partial index on active reservations plus
   epoch-based CAS; the application check alone isn't enough.
3. How do you keep the search index/cache consistent with the source of truth? -> CDC/outbox events update derived
   stores; they are rebuildable; reads tolerate bounded staleness.
4. What changes at 100x scale? -> shard the core by tenant with a routing layer, move more reads to replicas/caches,
   adopt partitioned log + stream processing for metering, and automate rebalancing.
5. How would you back up and test restores? -> continuous WAL archiving plus periodic snapshots, restore drills in a
   scratch environment with measured RTO, and checksums on backups.

---

## 10. Interview Process Overview

Reports converge on a loop that's fairly consistent for software-engineering roles, with some variation by
level and by track (Production Engineer loops run a slightly different shape than core Software Engineer
loops):

- **Recruiter screen**, followed by a **~60-minute technical phone screen** (candidate's choice of Go or
  Python in reports), typically a medium-difficulty data-structures/algorithms problem framed around
  CoreWeave's actual domain — scheduling, resource access graphs, log/telemetry parsing — rather than a
  generic LeetCode-style prompt with no context.
- **CoreWeave's own prep note** for the loop names two design-flavored rounds: a **System Design** interview on a
  **notification system** (clarify, state assumptions and tradeoffs, cover scalability/reliability/fault tolerance/varying
  traffic; see #8) and a **Database Design** interview on a **database system** (evaluate technologies by workload, data
  modeling, indexing, replication, partitioning, consistency; see #9). Both stress collaboration and explaining reasoning.
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

## 11. Behavioral Themes

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
- [CoreWeave SUNK: Slurm on Kubernetes](https://wf.coreweave.com/blog/sunk-slurm-on-kubernetes-implementations)
- [CoreWeave SUNK drain/undrain docs](https://docs.coreweave.com/products/sunk/manage_sunk/drain-and-undrain-nodes.md)
- [CoreWeave AI Object Storage benchmark (LOTA)](https://coreweave.com/blog/benchmark-results-coreweave-ai-object-storage-delivers-2-gb-s-per-gpu-throughput-across-any-number-of-gpus)
- [How Discord Stores Trillions of Messages — Discord Engineering](https://discord.com/blog/how-discord-stores-trillions-of-messages)
- [Building a batched notification engine — Knock](https://knock.app/blog/building-a-batched-notification-engine)
- [Prometheus Alertmanager docs](https://prometheus.io/docs/alerting/latest/alertmanager/)
- [Trigger.dev: scaling observability with ClickHouse](https://clickhouse.com/blog/trigger-dev-scaling-observability)
- [What to Expect in the CoreWeave System Design Interview — DesignGurus](https://www.designgurus.io/answers/detail/what-to-expect-in-the-coreweave-system-design-interview)
- [CoreWeave Interview Guide (2026): Specialty GPU Cloud — TechInterview.org](https://www.techinterview.org/companies/coreweave-interview-guide/)

Note: 1point3acres' individual interview-report threads require forum membership to view full question
text/discussion, and third-party prep-guide sites (DesignGurus, TechInterview.org) synthesize aggregated,
unverified community reports rather than publishing firsthand transcripts. The problems above were
reconstructed and expanded from the publicly visible report summaries and FastPrep's fuller public problem
writeup into complete, solvable problem statements with original test cases and solutions; the system
design topics reflect the recurring pattern independently corroborated across multiple sources rather than
a single verbatim transcript.
