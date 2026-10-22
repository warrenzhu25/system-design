# CoreWeave Interview Questions

---

## Contents

**Coding**
1. [Find a Directed Resource Access Path (Shortest Path in an Access Graph)](#1-find-a-directed-resource-access-path-shortest-path-in-an-access-graph)
2. [Boolean Filter Engine for Experiment Tracking Data](#2-boolean-filter-engine-for-experiment-tracking-data)
3. [Concurrent Server Reboot CLI](#3-concurrent-server-reboot-cli)
4. [Rate-Limited Concurrent Web Scraper](#4-rate-limited-concurrent-web-scraper)

**System Design and Database Design** — moved to [`coreweave_system_design.md`](./coreweave_system_design.md)

**Process & Behavioral**
5. [Interview Process Overview](#5-interview-process-overview)
6. [Behavioral Themes](#6-behavioral-themes)

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

## 5. Interview Process Overview

Reports converge on a loop that's fairly consistent for software-engineering roles, with some variation by
level and by track (Production Engineer loops run a slightly different shape than core Software Engineer
loops):

- **Recruiter screen**, followed by a **~60-minute technical phone screen** (candidate's choice of Go or
  Python in reports), typically a medium-difficulty data-structures/algorithms problem framed around
  CoreWeave's actual domain — scheduling, resource access graphs, log/telemetry parsing — rather than a
  generic LeetCode-style prompt with no context.
- **CoreWeave's own prep note** for the loop names two design-flavored rounds: a **System Design** interview on a
  **notification system** (clarify, state assumptions and tradeoffs, cover scalability/reliability/fault tolerance/varying
  traffic; see [Notification System](./coreweave_system_design.md#4-system-design--notification-system)) and a **Database Design** interview on a **database system** (evaluate technologies by workload, data
  modeling, indexing, replication, partitioning, consistency; see [Database Design](./coreweave_system_design.md#5-database-design-interview--database-system)). Both stress collaboration and explaining reasoning.
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

## 6. Behavioral Themes

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

Sources used for compiling these questions (design-answer sources are in [`coreweave_system_design.md`](./coreweave_system_design.md)):
- [CoreWeave Interview Questions — 1point3acres](https://www.1point3acres.com/interview/company/coreweave)
- [CoreWeave Technical Phone Interview: Data Filtering Algorithm — 1point3acres](https://www.1point3acres.com/interview/post/7502269)
- [CoreWeave SDE 2 Interview Experience: CLI Tool with Concurrency — 1point3acres](https://www.1point3acres.com/interview/post/7518478)
- [CoreWeave Onsite Interview Experience for Software Engineer Role — 1point3acres](https://www.1point3acres.com/interview/thread/1169570)
- [CoreWeave Onsite Software Engineer Interview Experience — 1point3acres](https://www.1point3acres.com/interview/thread/1156266)
- [CoreWeave Onsite Interview Experience for Fulltime SDE Position — 1point3acres](https://www.1point3acres.com/interview/thread/1143428)
- [Find a Directed Resource Access Path (CoreWeave Online Assessment) — FastPrep](https://www.fastprep.io/problems/coreweave-resource-access-path)
- [CoreWeave Online Assessment and Interview Insights — FastPrep](https://www.fastprep.io/hiring-insights/coreweave)
- [CoreWeave Interview Guide (2026): Specialty GPU Cloud — TechInterview.org](https://www.techinterview.org/companies/coreweave-interview-guide/)

Note: 1point3acres' individual interview-report threads require forum membership to view full question
text/discussion, and third-party prep-guide sites (DesignGurus, TechInterview.org) synthesize aggregated,
unverified community reports rather than publishing firsthand transcripts. The problems above were
reconstructed and expanded from the publicly visible report summaries and FastPrep's fuller public problem
writeup into complete, solvable problem statements with original test cases and solutions; the system
design topics reflect the recurring pattern independently corroborated across multiple sources rather than
a single verbatim transcript.
