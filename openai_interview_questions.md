# OpenAI Interview Questions

---

## Contents

**Coding**
1. [Infection Spread on a Grid (Escalating Multi-Source BFS)](#1-infection-spread-on-a-grid-escalating-multi-source-bfs)
2. [Cross-Entropy Loss (Masking and Label Smoothing)](#2-cross-entropy-loss-masking-and-label-smoothing)
3. [1-Nearest Neighbor Classification (Brute-Force and Vectorized)](#3-1-nearest-neighbor-classification-brute-force-and-vectorized)
4. [Prefix Product Autograd](#4-prefix-product-autograd)
5. [Dependency-Aware Multi-Agent Request Scheduler](#5-dependency-aware-multi-agent-request-scheduler)
6. [GPU Credit System with Validity Windows](#6-gpu-credit-system-with-validity-windows)
7. [Durable In-Memory Key-Value Store (Serialization, Segmented WAL, Log Recovery)](#7-durable-in-memory-key-value-store-serialization-segmented-wal-log-recovery)
8. [Work Queue with Leases, Retries, and a Dead-Letter Queue](#8-work-queue-with-leases-retries-and-a-dead-letter-queue)
9. [Versioned Data Structure with Point-in-Time Snapshots](#9-versioned-data-structure-with-point-in-time-snapshots)
10. [Memory Allocator: First-Fit and Best-Fit](#10-memory-allocator-first-fit-and-best-fit)
11. [Sharded Matrix Multiplication and Backpropagation](#11-sharded-matrix-multiplication-and-backpropagation)
12. [Message Event Aggregation in a Sliding Window](#12-message-event-aggregation-in-a-sliding-window)
13. [Restore Valid IPv4 Addresses](#13-restore-valid-ipv4-addresses)
14. [Maximum Falling Path with Limited Vertical Jumps and Bonus Scoring](#14-maximum-falling-path-with-limited-vertical-jumps-and-bonus-scoring)

**System Design**
15. [GPU Scheduling Across Competing Jobs](#15-system-design--gpu-scheduling-across-competing-jobs)

**Behavioral**
16. [Behavioral Themes](#16-behavioral-themes)

---

## 1. Infection Spread on a Grid (Escalating Multi-Source BFS)

**Problem Statement:**
Reported as very-high-frequency across SWE/MLE/RE/RS/EM roles. You're given an `M x N` grid of infected
cells (`X`) and susceptible cells (`.`), and told that infection propagates day by day under "escalating
rules." The exact escalation rule isn't publicly visible on the source listing beyond that phrase — this
is exactly the kind of ambiguity to resolve with the interviewer up front rather than guess through
silently. The assumption used below (stated explicitly, as you should in the real interview):

- Each day, every currently-infected cell infects all orthogonally-adjacent susceptible cells.
- Starting on day 3, infection also spreads diagonally (the "escalating" part).
- The grid may also contain `#` wall cells that infection can never cross.

Return the day number on which every susceptible cell has become infected, or `-1` if some susceptible
cells are permanently unreachable (isolated by walls, or there's no infected cell to begin with).

**Example:**
```
grid = ["X..",
        "...",
        "..."]
Output: 4   # day 1: (0,1),(1,0); day 2: (0,2),(1,1),(2,0); day 3: (1,2),(2,1); day 4: (2,2)
```

**Test Cases:**

| Grid | Result |
|---|---|
| All `.`, no `X` | `-1` (nothing can ever infect them) |
| All `X`, no `.` | `0` (already fully infected) |
| `["X#."]` (wall blocks the only path) | `-1` |
| Small grid fully reachable via orthogonal spread alone | matches plain multi-source BFS distance |
| A cell first reached on day ≥ 3 | its own further spread should reach neighbors diagonally, not just orthogonally — verify by tracing the `day` field carried in the queue, not just Manhattan distance |

**Key Insights:**
1. This is multi-source BFS: push every initially-infected cell into the queue at day 0 simultaneously,
   rather than running a separate BFS per source and merging.
2. Because BFS with a FIFO queue processes cells in non-decreasing "day" order (a full day's layer is
   exhausted before the next day's cells are dequeued), checking `day >= 3` on the cell being expanded
   correctly reflects the *global* elapsed day count — no separate day counter needed outside the queue.
3. State the escalation-rule assumption to the interviewer before writing code; a "Very High frequency"
   problem like this one means the real rule will be given precisely, and guessing wrong wastes the
   round on the wrong problem.

**Python Solution:**
```python
from collections import deque


def days_to_infect_all(grid: list[str]) -> int:
    """
    Time:  O(rows * cols)
    Space: O(rows * cols)
    """
    rows, cols = len(grid), len(grid[0])
    grid = [list(row) for row in grid]
    queue = deque()
    susceptible = 0

    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == "X":
                queue.append((r, c, 0))
            elif grid[r][c] == ".":
                susceptible += 1

    if susceptible == 0:
        return 0

    orthogonal = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    diagonal = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    infected_count = 0
    last_day = 0

    while queue:
        r, c, day = queue.popleft()
        deltas = orthogonal + (diagonal if day >= 3 else [])
        for dr, dc in deltas:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] == ".":
                grid[nr][nc] = "X"
                infected_count += 1
                last_day = day + 1
                queue.append((nr, nc, day + 1))

    return last_day if infected_count == susceptible else -1
```

**Follow-Up Questions:**
1. What if the real rule is distance-based (infection radius grows by one cell per day) rather than
   orthogonal/diagonal? → swap the neighbor-generation function; the BFS skeleton (multi-source, day
   carried per queue entry) is unchanged.
2. What if diagonal movement should be blocked when both orthogonal cells adjacent to that diagonal are
   walls (no "cutting the corner")? → add that check only when expanding a diagonal delta.
3. How would this scale to a much larger grid where most cells eventually get infected? → at that point
   a dense synchronous array sweep per day can beat event-driven BFS, since BFS's advantage is skipping
   work for cells that never activate — discuss the crossover point explicitly rather than assuming BFS
   is always better.
4. The bank lists several sibling titles on this same theme — "Minimum Time to Infect a Network,"
   "Plant Infection by Neighbor Count," and "Infection Spread with Immune Units and Expiring
   Contagiousness." Treat these as variants of the same multi-source-BFS skeleton with a different rule
   plugged in: a graph instead of a grid (BFS over adjacency lists instead of 4/8-directional deltas), a
   per-cell infection *threshold* (a cell only converts once enough of its neighbors are infected, which
   turns this into a bootstrap-percolation problem — track an infected-neighbor-count per cell and only
   enqueue it once the count crosses the threshold), and infected cells that themselves expire back to
   susceptible/immune after some number of days (carry an expiry alongside `day` in the queue, and allow
   a cell to be re-infected after its immunity window if the rules call for that).

---

## 2. Cross-Entropy Loss (Masking and Label Smoothing)

**Problem Statement:**
Implement categorical cross-entropy loss for a batch of classification logits, then extend it to (a)
mask out padding positions that shouldn't contribute to the loss and (b) support label smoothing.

**Requirements (staged, as an interviewer typically adds them):**
1. **Base case**: given `logits` of shape `(N, C)` (raw, unnormalized scores) and integer `targets` of
   shape `(N,)`, compute the mean negative log-likelihood of the correct class under the softmax
   distribution, numerically stably (no raw `exp()` overflow for large logits).
2. **Masking**: some rows of `targets` carry a sentinel `ignore_index` (e.g., padding tokens in a
   sequence-to-sequence batch) and must be excluded from both the sum and the averaging denominator.
3. **Label smoothing**: instead of a one-hot target, the true class gets probability `1 - eps + eps/C`
   and every other class gets `eps/C`; the loss becomes the cross-entropy against this smoothed
   distribution instead of the hard one-hot target.

**Example:**
```
logits = [[0, 0, 0]], targets = [0]   # uniform logits -> uniform softmax over 3 classes
loss = ln(3) ≈ 1.0986
```

**Test Cases:**

| logits | targets | ignore_index | label_smoothing | loss |
|---|---|---|---|---|
| `[[10,0,0]]` | `[0]` | — | `0` | ≈ `0` (softmax nearly all mass on class 0) |
| `[[0,0,0]]` | `[0]` | — | `0` | `ln(3) ≈ 1.0986` |
| `[[0,0,0],[0,0,0]]` | `[0, -100]` | `-100` | `0` | same as a single-row batch of `[[0,0,0]]`, `[0]` — the ignored row contributes nothing |
| `[[0,0,0]]` | `[0]` | — | `0` (eps=0) | identical to the un-smoothed case — smoothing with `eps=0` must reduce exactly to plain cross-entropy |

**Key Insights:**
1. Numerical stability: subtract the row-wise max from `logits` before exponentiating (the standard
   log-sum-exp trick) so large logits don't overflow `exp()`.
2. Label smoothing's loss decomposes into two terms that don't require iterating per-class: writing
   `smooth_neg = eps/C` and `smooth_pos = 1 - eps + eps/C`, the per-row loss is
   `-(smooth_neg * sum(log_probs) + (smooth_pos - smooth_neg) * log_probs[true_class])` — fully
   vectorizable, no inner loop over classes.
3. Masking must exclude ignored rows from *both* the numerator sum and the count used for the mean —
   a common bug is excluding them from the sum but still dividing by the original `N`.

**Python Solution:**
```python
import numpy as np


def cross_entropy_loss(logits: np.ndarray, targets: np.ndarray,
                        ignore_index: int | None = None,
                        label_smoothing: float = 0.0) -> float:
    """
    Time:  O(N * C)
    Space: O(N * C)
    """
    n, c = logits.shape
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    log_probs = shifted - np.log(np.sum(np.exp(shifted), axis=1, keepdims=True))

    mask = np.ones(n, dtype=bool)
    if ignore_index is not None:
        mask = targets != ignore_index
    if not mask.any():
        raise ValueError("no valid (non-ignored) rows to compute loss over")

    safe_targets = np.where(mask, targets, 0)  # avoid indexing with the sentinel value
    true_log_probs = log_probs[np.arange(n), safe_targets]

    if label_smoothing > 0:
        eps = label_smoothing
        smooth_neg = eps / c
        smooth_pos = 1 - eps + smooth_neg
        row_sum_log_probs = log_probs.sum(axis=1)
        losses = -(smooth_neg * row_sum_log_probs + (smooth_pos - smooth_neg) * true_log_probs)
    else:
        losses = -true_log_probs

    return float(losses[mask].mean())
```

**Follow-Up Questions:**
1. Sequence-level loss (2D `targets` over a batch of token sequences) → flatten `(batch, seq_len, C)` to
   `(batch*seq_len, C)` before applying the same function; masking naturally handles padded positions.
2. Class weighting (some classes matter more, e.g. rare classes upweighted) → multiply each row's loss by
   `class_weight[true_class]` before averaging, and use the sum of weights (not the row count) as the
   denominator.
3. How would you verify this against a reference implementation (e.g. a deep learning framework's
   built-in loss)? → compare on random inputs across a range of magnitudes, including very large/very
   negative logits, specifically to catch numerical-stability regressions the naive formula would miss.

---

## 3. 1-Nearest Neighbor Classification (Brute-Force and Vectorized)

**Problem Statement:**
Given a labeled training set `X_train` (`N x D`) with labels `y_train` (`N,`), classify each row of a
query set `X_query` (`M x D`) by the label of its nearest training point under Euclidean distance.
Implement a brute-force version, then a fully vectorized version using matrix operations instead of
nested loops.

**Example:**
```
X_train = [[0,0], [10,10]], y_train = [0, 1]
X_query = [[1,1], [9,9]]
Output: [0, 1]
```

**Test Cases:**

| X_train | y_train | X_query | predictions |
|---|---|---|---|
| `[[0,0],[10,10]]` | `[0,1]` | `[[1,1],[9,9]]` | `[0,1]` |
| `[[0,0]]` | `[7]` | `[[100,100]]` | `[7]` (only one training point — always wins) |
| `[[0,0],[1,1]]` | `[0,1]` | `[[0.4,0.4]]` | `[0]` (closer to `[0,0]`) |
| Brute-force and vectorized versions on the same random inputs | identical predictions |

**Key Insights:**
1. Brute force is `O(M*N*D)`: for every query point, scan every training point.
2. The vectorized version avoids nested Python loops via the identity
   `||a-b||^2 = ||a||^2 + ||b||^2 - 2·a·b`, turning the all-pairs distance computation into one matrix
   multiply (`X_query @ X_train.T`) plus two broadcasted vector additions — the FLOP count is the same,
   but it runs orders of magnitude faster in practice since the multiply is BLAS-accelerated instead of
   interpreted.
3. This is worth connecting explicitly to the "neural network formulation": if you drop the "nearest"
   step and instead take a *softmax* over the negative distances, you get exactly a radial-basis-function
   layer — 1-NN is the zero-temperature (hard-max) limit of that soft, differentiable layer.

**Python Solution:**
```python
import numpy as np


def one_nn_predict_bruteforce(X_train, y_train, X_query):
    """
    Time:  O(M * N * D)
    Space: O(1) beyond the output
    """
    preds = []
    for q in X_query:
        dists = np.sum((X_train - q) ** 2, axis=1)
        preds.append(y_train[np.argmin(dists)])
    return np.array(preds)


def one_nn_predict_vectorized(X_train, y_train, X_query):
    """
    Time:  O(M * N * D) FLOPs, but as one BLAS matmul instead of M*N Python-level ops
    Space: O(M * N) for the pairwise distance matrix
    """
    train_sq = np.sum(X_train ** 2, axis=1)          # (N,)
    query_sq = np.sum(X_query ** 2, axis=1)          # (M,)
    cross = X_query @ X_train.T                       # (M, N)
    dists = query_sq[:, None] + train_sq[None, :] - 2 * cross
    nearest_idx = np.argmin(dists, axis=1)
    return y_train[nearest_idx]
```

**Follow-Up Questions:**
1. Extend to k-NN (majority vote over the k nearest) → replace `argmin` with `argpartition`/`argsort`
   over the top-k indices, then majority-vote (or distance-weighted vote) the corresponding labels.
2. `N` too large to hold all pairwise distances in memory → process `X_query` in batches, or build an
   approximate index (KD-tree for low `D`, or an ANN structure like HNSW for high-dimensional embeddings)
   instead of exact brute-force search.
3. Ties (two training points equidistant with different labels) → the naive `argmin` picks the first;
   discuss whether the problem wants a deterministic tie-break rule or a specified one (e.g. lowest
   index, or a fallback vote among all tied points).

---

## 4. Prefix Product Autograd

**Problem Statement:**
Implement forward and backward passes for a prefix-product (`cumprod`) operation, without using an
existing autograd library: `y[i] = x[0] * x[1] * ... * x[i]`. Given `grad_y` (the upstream gradient
w.r.t. each `y[i]`), compute `grad_x`. The operation must remain correct even when some `x[i] == 0` — the
naive approach of recovering "the product excluding index i" via `y[j] / x[i]` breaks with a
division-by-zero exactly there.

**Example:**
```
x = [2, 3, 4]
y = cumprod_forward(x)   # [2, 6, 24]
grad_y = [1, 1, 1]
grad_x = cumprod_backward(x, grad_y)   # [16, 10, 6]
```

**Test Cases:**

| x | grad_y | grad_x |
|---|---|---|
| `[2,3,4]` | `[1,1,1]` | `[16,10,6]` |
| `[1,1,1]` | `[1,1,1]` | `[3,2,1]` |
| `[2,0,4]` | `[1,1,1]` | must not raise `ZeroDivisionError`, and must match the analytic gradient computed without ever dividing by `x[1]` |

**Key Insights:**
1. `dL/dx_i = sum_{j>=i} grad_y[j] * (product of all x_k for k in [0,j] except k=i)`. The naive
   implementation of "product excluding i" as `y[j] / x[i]` is exactly what fails at a zero — instead
   build the excluded product directly as `prefix(0..i-1) * suffix(i+1..j)`, which never divides by
   anything.
2. Precompute `prefix[i] = product(x[0..i-1])` once (`prefix[0] = 1`). Then for each `i`, sweep `j` from
   `i` to `n-1`, incrementally extending a running suffix product — this keeps the whole computation at
   `O(n^2)` (fine for interview scale) with zero divisions anywhere.
3. This mirrors how real autograd engines (e.g. PyTorch's `cumprod` backward) handle zeros: they
   special-case the segment following a zero rather than dividing through it.

**Python Solution:**
```python
def cumprod_forward(x: list[float]) -> list[float]:
    """
    Time:  O(n)
    Space: O(n) for the output
    """
    y = []
    running = 1.0
    for xi in x:
        running *= xi
        y.append(running)
    return y


def cumprod_backward(x: list[float], grad_y: list[float]) -> list[float]:
    """
    Time:  O(n^2)
    Space: O(n) for prefix products + O(n) output
    """
    n = len(x)
    prefix = [1.0] * n
    for i in range(1, n):
        prefix[i] = prefix[i - 1] * x[i - 1]

    grad_x = [0.0] * n
    for i in range(n):
        running = 1.0    # product of x[i+1 .. j], empty when j == i
        total = 0.0
        for j in range(i, n):
            total += grad_y[j] * prefix[i] * running
            if j + 1 < n:
                running *= x[j + 1]
        grad_x[i] = total
    return grad_x
```

**Follow-Up Questions:**
1. Can this be done in `O(n)` instead of `O(n^2)`? → yes, by splitting `x` at zero positions: for a
   segment with no zeros, the division-based formula (`y[j]/x[i]`) is safe and gives an `O(n)` pass;
   only the handful of positions adjacent to an actual zero need the explicit no-division treatment —
   a hybrid gets amortized `O(n)` in the common (mostly-nonzero) case.
2. What if `x` requires gradient itself to be used again downstream (need the *graph*, not just one
   backward call)? → this is where a real autograd system diverges: it records the operation and its
   inputs/outputs on a tape at forward time so backward can be triggered generically later, rather than
   this function being hand-called once.
3. Batch this over a `(batch, n)` tensor instead of a single vector → vectorize the same prefix/suffix
   trick along the last axis with `numpy`/tensor broadcasting instead of Python loops.

---

## 5. Dependency-Aware Multi-Agent Request Scheduler

**Problem Statement:**
Given a set of agent requests, each with an `id`, a `priority`, and a list of other request ids it
depends on (must complete first), produce a valid execution order that respects all dependencies and, at
every point where more than one request is ready to run, always picks the highest-priority ready request
next. Raise an error if the dependency graph has a cycle.

**Example:**
```
requests = [("a", 1, []), ("b", 5, []), ("c", 3, ["a", "b"])]
Output: ["b", "a", "c"]   # b and a are both ready initially; b's higher priority runs first
```

**Test Cases:**

| requests (id, priority, deps) | order |
|---|---|
| `[("a",1,[]), ("b",5,[]), ("c",3,["a","b"])]` | `["b","a","c"]` |
| `[("a",1,["b"]), ("b",1,["a"])]` | raises (cycle) |
| `[("a",5,[]), ("b",1,["a"]), ("c",9,["a"])]` | `["a","c","b"]` (both b,c ready after a; c has higher priority) |
| Single request, no deps | `[that id]` |

**Key Insights:**
1. This is Kahn's algorithm (topological sort via in-degree counting) with the FIFO queue replaced by a
   max-heap keyed on priority — "ready" (in-degree 0) requests are always available in the heap so the
   highest-priority one is popped next, rather than whichever happened to become ready first.
2. Cycle detection falls out for free: if the final order has fewer entries than total requests, some
   requests never reached in-degree 0, meaning a cycle exists among them.
3. Python's `heapq` is a min-heap, so push `(-priority, id)` to get max-priority-first behavior without
   writing a custom comparator.

**Python Solution:**
```python
import heapq
from collections import defaultdict


def schedule_requests(requests: list[tuple[str, int, list[str]]]) -> list[str]:
    """
    Time:  O((V + E) log V)
    Space: O(V + E)
    """
    priority = {}
    dependents = defaultdict(list)
    indegree = {}

    for rid, prio, deps in requests:
        priority[rid] = prio
        indegree[rid] = len(deps)
        for d in deps:
            dependents[d].append(rid)

    heap = [(-priority[rid], rid) for rid, indeg in indegree.items() if indeg == 0]
    heapq.heapify(heap)

    order = []
    while heap:
        _, rid = heapq.heappop(heap)
        order.append(rid)
        for nxt in dependents[rid]:
            indegree[nxt] -= 1
            if indegree[nxt] == 0:
                heapq.heappush(heap, (-priority[nxt], nxt))

    if len(order) != len(requests):
        raise ValueError("cycle detected among agent requests")
    return order
```

**Follow-Up Questions:**
1. Requests can also carry a resource requirement (e.g. a specific model/GPU pool) → this becomes the
   dependency-aware version of the GPU Scheduling design (problem 15) — a ready request additionally
   needs its resource to be free, not just its dependencies satisfied.
2. Ties in priority → add a stable tiebreaker (e.g. submission order) to the heap key so the schedule is
   deterministic and reproducible for the same input.
3. Dynamic arrival (new requests submitted while the scheduler is running, possibly depending on
   already-completed ones) → treat already-completed ids as pre-satisfied dependencies (indegree
   contribution of 0) when a new request arrives, and push it straight onto the ready heap if all its
   deps are already done.

---

## 6. GPU Credit System with Validity Windows

**Problem Statement:**
Implement a credit system for GPU usage: credits are granted in batches, each with an amount and an
expiration time. Consuming credits should always draw from the **earliest-expiring** valid batch first
(to minimize credits wasted to expiry), expired batches become unusable, and consuming more than the
currently-valid balance should fail cleanly rather than allowing negative balance.

**Example:**
```
grant(10, expires_at=100)
grant(5, expires_at=50)
consume(3, now=10)          # draws from the batch expiring at 50 first
query_balance(now=10)       # -> 12   (2 left in the 50-batch, 10 in the 100-batch)
query_balance(now=60)       # -> 10   (the 50-batch expired, its remaining 2 are gone)
```

**Test Cases:**

| Operations | Result |
|---|---|
| `grant(10,100); grant(5,50); consume(3, now=10)` then `query_balance(10)` | `12` |
| Same, then `query_balance(60)` | `10` (2 expired credits lost) |
| `consume(10, now=60)` after the above, then another `consume(1, now=60)` | second call raises (insufficient balance) |
| `consume` amount larger than total valid balance | raises without partially deducting anything |

**Key Insights:**
1. A min-heap keyed by expiry time gives "earliest-expiring batch" in `O(log n)`, and lazily dropping
   expired batches from the heap root on every operation avoids maintaining a separate expiry sweep.
2. Consuming may need to partially drain a batch (leaving a smaller remainder) or fully drain several —
   loop, taking `min(batch_amount, remaining_needed)` from the heap root until the request is satisfied
   or the heap is exhausted.
3. If a request can't be fully satisfied, don't leave the store partially debited — validate against
   `query_balance` first, or track how much would be consumed before committing any mutation (the
   solution below opts for "fail without partial deduction" by construction, since it aborts as soon as
   it can't find enough remaining valid credit).

**Python Solution:**
```python
import heapq


class GPUCreditSystem:
    """
    grant:         O(log n)
    consume:       O(k log n), k = number of expired/drained batches touched
    query_balance: O(log n) amortized (purges expired batches lazily)
    """

    def __init__(self):
        self._heap: list[list[float]] = []  # [expires_at, amount], min-heap by expiry

    def grant(self, amount: float, expires_at: float) -> None:
        heapq.heappush(self._heap, [expires_at, amount])

    def _purge_expired(self, now: float) -> None:
        while self._heap and self._heap[0][0] <= now:
            heapq.heappop(self._heap)

    def consume(self, amount: float, now: float) -> None:
        self._purge_expired(now)
        if self.query_balance(now) < amount:
            raise ValueError("insufficient valid credit balance")

        remaining = amount
        while remaining > 0:
            batch = self._heap[0]
            take = min(batch[1], remaining)
            batch[1] -= take
            remaining -= take
            if batch[1] == 0:
                heapq.heappop(self._heap)

    def query_balance(self, now: float) -> float:
        self._purge_expired(now)
        return sum(amount for _, amount in self._heap)
```

**Follow-Up Questions:**
1. Refunds (an unused reservation returns credit) → push the refunded amount back as a new batch with
   its *original* batch's remaining expiry, not a fresh one, so it can't outlive what was originally
   granted.
2. Multiple credit "types" (e.g. training vs. inference credits, non-fungible) → key the whole structure
   by type, running an independent heap per type.
3. High-throughput concurrent grant/consume → the heap needs a lock (or a lock-free structure) around
   the check-then-mutate sequence in `consume`, since two concurrent consumers could otherwise both pass
   the `query_balance` check before either deducts.

---

## 7. Durable In-Memory Key-Value Store (Serialization, Segmented WAL, Log Recovery)

**Problem Statement:**
Build an in-memory key-value store whose state survives a process restart, developed in the staged order
an interviewer typically presents it:

1. **Base**: `get`/`set`/`delete` over an in-memory dict.
2. **Durability via serialization**: every mutation is also appended, serialized, to a write-ahead log
   (WAL) — the in-memory dict is a cache that can always be rebuilt from the log.
3. **Segmentation**: the log rotates into a new segment once the current one reaches a size threshold,
   so no single file grows unbounded and old segments become independently manageable.
4. **Log-based recovery**: on startup (or after simulating a crash by dropping `self.data`), replay every
   segment's entries in order to rebuild the exact pre-crash state.

**Example:**
```
store = DurableKVStore(segment_size=2)
store.set("a", 1); store.set("b", 2); store.set("c", 3)   # segment rotates after 2 entries
store.delete("b")
store.data = {}          # simulate a crash: in-memory state lost, log segments survive
store.recover()
store.get("a")  # -> 1
store.get("b")  # -> None (deleted)
store.get("c")  # -> 3
```

**Test Cases:**

| Scenario | Expectation |
|---|---|
| `set` then `get` | returns the value just set |
| `delete` then `get` | returns `None` |
| Enough `set` calls to exceed `segment_size` | `len(store.segments) > 1` |
| Clear `store.data` and call `recover()` | state matches what it was before clearing, replayed purely from the segments |
| `compact(1)` after taking a snapshot covering segment 0 | `store.segments` drops the first segment; recovery from segments alone is then only valid in combination with that snapshot (documented limitation, not silently incorrect) |

**Key Insights:**
1. Treat the WAL as the source of truth and the in-memory dict as a derived cache — `recover()` is
   simply "replay the log from empty state," which only works if *every* mutation, without exception,
   goes through the log-append path first.
2. Segmentation is just "start a new list once the current one is full" — recovery doesn't care how many
   segments there are, it just replays them all in order.
3. Compaction (dropping old segments) is only safe once a full snapshot of the state as of that point has
   been durably persisted elsewhere — the code below implements the mechanical drop but the snapshot
   itself is explicitly out of scope, and that gap is worth naming rather than hiding.

**Python Solution:**
```python
import json


class DurableKVStore:
    """
    get:              O(1)
    set/delete:       O(1) amortized (append-only; occasional segment rotation)
    recover:          O(total entries across all retained segments)
    Space: O(total log entries retained)
    """

    def __init__(self, segment_size: int = 1000):
        self.data: dict = {}
        self.segment_size = segment_size
        self.segments: list[list[str]] = [[]]

    def _current_segment(self) -> list[str]:
        if len(self.segments[-1]) >= self.segment_size:
            self.segments.append([])
        return self.segments[-1]

    def _append_log(self, op: str, key: str, value=None) -> None:
        self._current_segment().append(json.dumps({"op": op, "key": key, "value": value}))

    def set(self, key: str, value) -> None:
        self._append_log("SET", key, value)
        self.data[key] = value

    def delete(self, key: str) -> None:
        self._append_log("DELETE", key)
        self.data.pop(key, None)

    def get(self, key: str):
        return self.data.get(key)

    def recover(self) -> None:
        self.data = {}
        for segment in self.segments:
            for raw in segment:
                entry = json.loads(raw)
                if entry["op"] == "SET":
                    self.data[entry["key"]] = entry["value"]
                elif entry["op"] == "DELETE":
                    self.data.pop(entry["key"], None)

    def compact(self, snapshot_covers_up_to_segment: int) -> None:
        """Drop segments already captured by a durable snapshot taken elsewhere."""
        self.segments = self.segments[snapshot_covers_up_to_segment:]
```

**Follow-Up Questions:**
1. Add real snapshotting so `compact` is safe standalone → periodically serialize `self.data` itself to
   a snapshot file alongside a marker of "log position at snapshot time," and have `recover()` load the
   latest snapshot before replaying only the segments created after it.
2. Every write currently serializes and appends synchronously — how would you make this durable *and*
   fast under high write throughput? → batch/group commit (buffer a short window of writes, flush and
   `fsync` once), trading a small durability window for much higher throughput.
3. Concurrent writers → the append-to-log-then-apply-to-dict sequence needs to be atomic per key (or
   globally), otherwise two concurrent `set`s on the same key could interleave and leave the log and the
   in-memory dict disagreeing about final order.

---

## 8. Work Queue with Leases, Retries, and a Dead-Letter Queue

**Problem Statement:**
Implement a work queue where consumers `pull` a task under a time-bound **lease**: the task becomes
invisible to other consumers until the lease expires or the consumer explicitly `ack`s it. If the lease
expires without an ack, the task becomes available again for redelivery and its retry count increments.
Once retries exceed `max_retries`, the task is moved to a dead-letter queue instead of being redelivered
again.

**Example:**
```
q = WorkQueue(max_retries=1)
q.push("t1", "payload")
q.pull(now=0, lease_duration=10)     # -> ("t1", "payload"); leased until t=10
# no ack happens
q.pull(now=11, lease_duration=10)    # lease expired -> redelivered: -> ("t1", "payload") again, retries=1
# still no ack
q.pull(now=22, lease_duration=10)    # retries would exceed max_retries=1 -> moved to dead_letter, not redelivered
```

**Test Cases:**

| Scenario | Expectation |
|---|---|
| `push` then `pull` before any expiry | returns the task; a second immediate `pull` returns `None` (already leased) |
| Lease expires without `ack` | task becomes pullable again, `retries` incremented |
| `ack` before lease expiry | task is gone; it never becomes pullable again, even after the original lease time passes |
| Retries exceed `max_retries` | task appears in `dead_letter`, is never redelivered again |

**Key Insights:**
1. Track a **lease version** per task, incremented on every `pull`. When a lease entry's expiry comes
   due in the heap, only act on it if its version still matches the task's *current* lease version —
   otherwise it's a stale heap entry left over from an earlier pull that was already acked or already
   re-leased, and acting on it would double-process the task.
2. "Redeliver on expiry" and "evaluate whether this exceeds max_retries" both happen at the point a
   lease is discovered to be expired — not at `pull` time in general — since expiry is a passive,
   time-driven event rather than something a caller triggers directly.
3. A min-heap ordered by lease-expiry time, checked lazily (only when its earliest entry's expiry has
   actually passed), avoids polling or timers entirely.

**Python Solution:**
```python
import heapq
from collections import deque


class WorkQueue:
    """
    push: O(1)
    pull: O(log n) amortized (may also drain several expired-lease heap entries)
    ack:  O(1)
    """

    def __init__(self, max_retries: int = 3):
        self.max_retries = max_retries
        self._ready: deque[str] = deque()
        self._task_data: dict[str, dict] = {}       # task_id -> {"payload", "retries"}
        self._lease_version: dict[str, int] = {}
        self._leases: list[tuple[float, str, int]] = []  # (expires_at, task_id, version)
        self.dead_letter: list[str] = []

    def push(self, task_id: str, payload) -> None:
        self._task_data[task_id] = {"payload": payload, "retries": 0}
        self._lease_version[task_id] = 0
        self._ready.append(task_id)

    def pull(self, now: float, lease_duration: float):
        self._recover_expired(now)
        if not self._ready:
            return None
        task_id = self._ready.popleft()
        version = self._lease_version[task_id] + 1
        self._lease_version[task_id] = version
        heapq.heappush(self._leases, (now + lease_duration, task_id, version))
        return task_id, self._task_data[task_id]["payload"]

    def ack(self, task_id: str) -> None:
        self._task_data.pop(task_id, None)
        self._lease_version.pop(task_id, None)

    def _recover_expired(self, now: float) -> None:
        while self._leases and self._leases[0][0] <= now:
            _, task_id, version = heapq.heappop(self._leases)
            if task_id not in self._task_data:
                continue                                    # already acked
            if self._lease_version[task_id] != version:
                continue                                    # stale entry, superseded by a newer lease
            self._task_data[task_id]["retries"] += 1
            if self._task_data[task_id]["retries"] > self.max_retries:
                self.dead_letter.append(task_id)
                del self._task_data[task_id]
                del self._lease_version[task_id]
            else:
                self._ready.append(task_id)
```

**Follow-Up Questions:**
1. Priority among ready tasks → swap `_ready` (a FIFO deque) for a heap keyed by priority, same as
   problem 5's scheduler.
2. Visibility timeout that extends while a consumer is actively working (heartbeating) → add an
   `extend_lease(task_id, now, new_duration)` that bumps the version and pushes a new heap entry, so the
   old, shorter lease's eventual expiry is recognized as stale via the version check and ignored.
3. Exactly-once processing guarantees → a lease alone only gives "at-least-once" (a task could be
   processed, then its ack lost, then redelivered) — true exactly-once needs idempotent task processing
   or a dedup key on the consumer side, not just lease/retry bookkeeping.

---

## 9. Versioned Data Structure with Point-in-Time Snapshots

**Problem Statement:**
Implement a set (or graph edge set) that supports point-in-time queries: `add`/`remove` elements, take a
`snapshot()` that labels "now," and later ask `contains(x, version)` — was `x` present as of that
snapshot — without copying the entire structure on every snapshot. Apply the same technique to a
"versioned follow graph" (edges are `(follower, followee)` pairs instead of arbitrary elements) to
support "was A following B as of version V."

**Example:**
```
s = SnapshotSet()
s.add("a")            # version becomes 1
v1 = s.snapshot()     # v1 = 1
s.remove("a")         # version becomes 2
v2 = s.snapshot()     # v2 = 2
s.contains("a", v1)   # -> True
s.contains("a", v2)   # -> False
```

**Test Cases:**

| Operations | Query | Result |
|---|---|---|
| `add("a")` (v1), `remove("a")` (v2), `add("a")` (v3) | `contains("a", v1)` | `True` |
| same | `contains("a", v2)` | `False` |
| same | `contains("a", v3)` | `True` |
| `contains("a", 0)` before any operation | `False` |
| `contains` on an element never touched | `False` |

**Key Insights:**
1. Don't snapshot the whole structure — store, per element, an append-only list of `(version, is_present)`
   change points; a query binary-searches for the latest change at or before the requested version.
2. A single global monotonically-increasing version counter (incremented on every mutation, across all
   keys) gives a total order across the whole structure, which is what makes cross-key point-in-time
   comparisons ("what was true at version V") well-defined.
3. Store each key's versions and flags in two parallel lists (not a list of tuples) so `bisect` can
   search directly on the versions list in `O(log k)` without rebuilding anything per query.

**Python Solution:**
```python
import bisect
from collections import defaultdict
from typing import Any


class SnapshotSet:
    """
    add/remove: O(1) amortized
    snapshot:   O(1)
    contains:   O(log k), k = number of recorded changes for that element
    """

    def __init__(self):
        self._version = 0
        self._versions: dict[Any, list[int]] = defaultdict(list)
        self._flags: dict[Any, list[bool]] = defaultdict(list)

    def snapshot(self) -> int:
        return self._version

    def add(self, x: Any) -> None:
        self._version += 1
        self._versions[x].append(self._version)
        self._flags[x].append(True)

    def remove(self, x: Any) -> None:
        self._version += 1
        self._versions[x].append(self._version)
        self._flags[x].append(False)

    def contains(self, x: Any, version: int) -> bool:
        versions = self._versions.get(x)
        if not versions:
            return False
        idx = bisect.bisect_right(versions, version) - 1
        return idx >= 0 and self._flags[x][idx]


class VersionedFollowGraph:
    """Same technique, keyed by (follower, followee) edges."""

    def __init__(self):
        self._edges = SnapshotSet()

    def follow(self, follower: str, followee: str) -> None:
        self._edges.add((follower, followee))

    def unfollow(self, follower: str, followee: str) -> None:
        self._edges.remove((follower, followee))

    def is_following(self, follower: str, followee: str, version: int) -> bool:
        return self._edges.contains((follower, followee), version)
```

**Follow-Up Questions:**
1. "List all followees of A as of version V" (not just a single point query) → this needs a secondary
   index from follower to candidate followees, each still checked with `contains` — the point-query
   structure alone doesn't give an efficient enumeration.
2. Unbounded history growth → periodically compact each key's change list by dropping all but the latest
   entry at or before some "oldest queryable version" horizon, once nothing will ever query further back.
3. Compare this to a copy-on-write persistent tree (e.g. a persistent balanced BST) → the change-list
   approach is simpler and faster for point queries on a flat key space, but a persistent tree
   generalizes better to ordered/range queries across versions, which this technique doesn't support.

---

## 10. Memory Allocator: First-Fit and Best-Fit

**Problem Statement:**
Implement a simple memory allocator over a fixed-size buffer: `malloc(size)` returns a start address for
a free block of at least `size` (or `None` if none fits), and `free(address)` releases a previously
allocated block, coalescing it with any adjacent free blocks. Support both a **First-Fit** strategy
(take the first sufficiently large free block) and a **Best-Fit** strategy (take the smallest
sufficiently large free block, to reduce wasted space) — selectable at construction.

**Test Cases:**

| Scenario | Expectation |
|---|---|
| `malloc` larger than any single free block (even if total free space is enough) | returns `None` |
| First-Fit over free blocks `[(0,50), (60,10), (80,30)]`, `malloc(10)` | picks the block at `0` (first sufficient), leaving `(10,40)` |
| Best-Fit over the same free blocks, `malloc(10)` | picks the block at `60` (smallest sufficient, exact fit), consuming it entirely |
| `free()` of a block adjacent to an existing free block | the two coalesce into one larger free block |
| `free()` of an address that was never allocated (or already freed) | raises `ValueError` |

**Key Insights:**
1. Represent free space as a list of `(start, size)` blocks; `malloc` filters to blocks large enough,
   then First-Fit takes the first match and Best-Fit takes the `min` by size among matches — the only
   difference between the two strategies is that one selection rule.
2. On `free`, re-sort the free list by start address and merge any run of blocks where one's end exactly
   equals the next one's start — this is what prevents free space from fragmenting into unusable slivers
   over time.
3. Track allocated blocks in a separate `address -> size` map so `free` doesn't need the caller to
   remember the size, and so double-frees / bad addresses can be detected and rejected.

**Python Solution:**
```python
class MemoryAllocator:
    """
    malloc: O(n) scan over the free list, n = number of free blocks
    free:   O(n log n) due to re-sorting on release (a balanced interval structure gets this to O(log n))
    """

    def __init__(self, total_size: int, strategy: str = "first_fit"):
        self.strategy = strategy
        self.free_blocks: list[tuple[int, int]] = [(0, total_size)]
        self.allocated: dict[int, int] = {}

    def malloc(self, size: int) -> int | None:
        candidates = [(i, blk) for i, blk in enumerate(self.free_blocks) if blk[1] >= size]
        if not candidates:
            return None

        if self.strategy == "first_fit":
            idx, (start, blk_size) = candidates[0]
        else:  # best_fit
            idx, (start, blk_size) = min(candidates, key=lambda ib: ib[1][1])

        self.allocated[start] = size
        remaining = blk_size - size
        if remaining > 0:
            self.free_blocks[idx] = (start + size, remaining)
        else:
            self.free_blocks.pop(idx)
        return start

    def free(self, address: int) -> None:
        size = self.allocated.pop(address, None)
        if size is None:
            raise ValueError(f"address {address} is not currently allocated")
        self.free_blocks.append((address, size))
        self.free_blocks.sort()
        self._coalesce()

    def _coalesce(self) -> None:
        merged: list[tuple[int, int]] = []
        for start, size in self.free_blocks:
            if merged and merged[-1][0] + merged[-1][1] == start:
                prev_start, prev_size = merged.pop()
                merged.append((prev_start, prev_size + size))
            else:
                merged.append((start, size))
        self.free_blocks = merged
```

**Follow-Up Questions:**
1. Worst-Fit (always take the *largest* sufficient block, to keep remainders large and usable) → same
   `candidates` filter, `max` instead of `min`/first — worth discussing the tradeoff explicitly (fewer
   tiny unusable remainders, but faster exhaustion of large blocks for big future requests).
2. `O(log n)` malloc/free instead of scanning the whole free list → keep free blocks in a size-indexed
   balanced structure (e.g. a sorted list of `(size, start)` for Best-Fit) alongside the start-indexed
   one used for coalescing, updating both on every mutation.
3. Fragmentation over long runs of alloc/free churn → discuss compaction (relocating allocated blocks to
   defragment), which requires the caller to support pointer updates — a much bigger design point worth
   naming rather than solving in-line.

---

## 11. Sharded Matrix Multiplication and Backpropagation

**Problem Statement:**
Given `A` (`M x K`) and `B` (`K x N`), implement matrix multiplication `C = A @ B` sharded across
row-blocks of `A` (simulating multiple workers, each handling a slice of rows), then implement the
backward pass: given `grad_C`, compute `grad_A` and `grad_B`, also sharded the same way.

**Example:**
```
A = [[1,2],[3,4],[5,6]]   # 3x2
B = [[1,0],[0,1]]          # 2x2 identity
C = sharded_matmul_forward(A, B, num_shards=2)   # -> A itself, since B is identity
```

**Test Cases:**

| A | B | num_shards | forward result |
|---|---|---|---|
| `[[1,2],[3,4],[5,6]]` | `[[1,0],[0,1]]` (identity) | `2` | equals `A` |
| any `A, B` | any | `1` | equals a plain (unsharded) `A @ B` |
| any `A, B, grad_C` | — | any `num_shards` | `grad_A`/`grad_B` from the sharded backward match `grad_C @ B.T` / `A.T @ grad_C` computed directly |

**Key Insights:**
1. Forward sharding by rows of `A` is embarrassingly parallel: each shard needs its own row-slice of `A`
   plus the *entire* `B`, and produces an independent row-slice of `C` — no cross-shard communication.
2. `grad_A = grad_C @ B.T` is likewise row-independent — each shard computes its own `grad_A` slice from
   its own `grad_C` slice, again with no cross-shard communication.
3. `grad_B = A.T @ grad_C`, however, is **not** row-independent: it's a reduction over the shared `K`/row
   axis. Each shard can only compute a *local partial* `grad_B` from its own row-slice, and the true
   `grad_B` is the **sum** of every shard's partial — exactly the all-reduce step in real data-parallel
   training, where each worker computes a local gradient and the framework sums them across workers
   before the optimizer step.

**Python Solution:**
```python
import numpy as np


def sharded_matmul_forward(A: np.ndarray, B: np.ndarray, num_shards: int) -> np.ndarray:
    """
    Time:  O(M*K*N) total FLOPs, partitioned across shards
    Space: O(M*N) for the output
    """
    m = A.shape[0]
    shard_size = (m + num_shards - 1) // num_shards
    shards = [A[i:i + shard_size] @ B for i in range(0, m, shard_size)]
    return np.vstack(shards)


def sharded_matmul_backward(A: np.ndarray, B: np.ndarray, grad_C: np.ndarray,
                             num_shards: int) -> tuple[np.ndarray, np.ndarray]:
    """
    grad_A: row-sharded independently, no cross-shard step.
    grad_B: each shard computes a LOCAL partial; true grad_B is the sum ("all-reduce") of all partials.
    """
    m = A.shape[0]
    shard_size = (m + num_shards - 1) // num_shards
    grad_A_shards = []
    grad_B_total = np.zeros((A.shape[1], B.shape[1]))

    for i in range(0, m, shard_size):
        a_shard = A[i:i + shard_size]
        grad_c_shard = grad_C[i:i + shard_size]
        grad_A_shards.append(grad_c_shard @ B.T)
        grad_B_total += a_shard.T @ grad_c_shard  # local partial, reduced across shards

    return np.vstack(grad_A_shards), grad_B_total
```

**Follow-Up Questions:**
1. Shard along `K` instead of `M` (splitting the reduction dimension of the forward matmul itself) →
   forward now needs a sum-reduction across shards (each computes a partial `C` that must be summed, not
   concatenated) — the roles of forward and `grad_B` essentially swap in terms of which needs a reduction.
2. Communication cost: how would you estimate the network cost of the `grad_B` all-reduce at scale? →
   it's proportional to `K*N` per shard (the size of one partial `grad_B`), independent of `M` — call out
   that this is why data-parallel training's communication cost doesn't grow with dataset size, only
   with model size.
3. Numerical precision across many shards summed in a different order than a single-machine computation
   → floating-point addition isn't associative, so distributed and single-machine results can differ
   slightly; worth naming as an expected (not a bug) source of small numerical divergence.

---

## 12. Message Event Aggregation in a Sliding Window

**Problem Statement:**
Given a stream of `(timestamp, value)` events arriving in non-decreasing timestamp order, support
`add(timestamp, value)` and `query(now)`, where `query` returns the count, sum, and average of all
events with `timestamp` in `(now - window, now]`, evicting events that have fallen out of the window.

**Example:**
```
agg = SlidingWindowAggregator(window_seconds=300)
agg.add(0, 10); agg.add(100, 20); agg.add(250, 30)
agg.query(now=250)   # -> {count: 3, sum: 60, avg: 20.0}   (nothing has expired yet)
agg.query(now=350)   # -> {count: 2, sum: 50, avg: 25.0}   (the t=0 event fell out of the window)
```

**Test Cases:**

| Events added | query(now) | Result |
|---|---|---|
| `(0,10),(100,20),(250,30)` | `250` | `count=3, sum=60, avg=20.0` |
| same | `350` | `count=2, sum=50, avg=25.0` |
| same | `600` | `count=0, sum=0, avg=0.0` |
| no events added | any `now` | `count=0, sum=0, avg=0.0` |

**Key Insights:**
1. Because events arrive in non-decreasing timestamp order, a deque works as the window buffer: stale
   events are always at the front, so eviction is a simple `popleft()` loop, never a scan of the middle.
2. Maintain a running `sum` incrementally (add on insert, subtract on eviction) instead of recomputing
   `sum(events)` on every query — a query is then `O(k)` only for the events actually evicted that call,
   not `O(window size)` every time.
3. The non-decreasing-arrival assumption is load-bearing — state it explicitly, since it's what makes the
   deque-based eviction correct at all.

**Python Solution:**
```python
from collections import deque


class SlidingWindowAggregator:
    """
    add:   O(1) amortized
    query: O(k) amortized, k = events evicted during that call
    Space: O(w), w = events currently within the window
    """

    def __init__(self, window_seconds: float = 300):
        self.window = window_seconds
        self._events: deque[tuple[float, float]] = deque()
        self._sum = 0.0

    def add(self, timestamp: float, value: float) -> None:
        self._events.append((timestamp, value))
        self._sum += value

    def _evict_stale(self, now: float) -> None:
        while self._events and self._events[0][0] <= now - self.window:
            _, value = self._events.popleft()
            self._sum -= value

    def query(self, now: float) -> dict:
        self._evict_stale(now)
        count = len(self._events)
        return {
            "count": count,
            "sum": self._sum,
            "avg": self._sum / count if count else 0.0,
        }
```

**Follow-Up Questions:**
1. Events can arrive out of order (clock skew across producers) → a deque no longer suffices for
   eviction; use a structure that supports removal of arbitrary stale entries efficiently (e.g. a
   min-heap by timestamp, with lazy skipping of entries already evicted by a later `query`), or buffer a
   short reordering window before admitting events at all (a watermark).
2. Very high event rate, exact per-query correctness not required → replace the exact deque with a
   fixed set of time buckets (e.g. one bucket per 10-second interval) that roll off as they age out,
   trading a small amount of boundary imprecision for `O(1)` eviction independent of event volume.
3. Multiple independent windows (1-minute and 5-minute simultaneously) → maintain one aggregator per
   window size rather than trying to derive one from the other; they evict independently.

---

## 13. Restore Valid IPv4 Addresses

**Problem Statement:**
Given a string `s` of digits, return every way to insert exactly three dots into it so the result is a
valid IPv4 address — four segments, each `0`–`255`, with no segment having a leading zero unless the
segment is exactly `"0"`.

**Example:**
```
Input:  "25525511135"
Output: ["255.255.11.135", "255.255.111.35"]
```

**Test Cases:**

| Input | Output |
|---|---|
| `"25525511135"` | `["255.255.11.135", "255.255.111.35"]` |
| `"0000"` | `["0.0.0.0"]` |
| `"101023"` | `["1.0.10.23", "1.0.102.3", "10.1.0.23", "10.10.2.3", "101.0.2.3"]` |
| `"1111"` | `["1.1.1.1"]` |
| `""` or too few/many digits to ever form 4 valid segments | `[]` |

**Key Insights:**
1. Bounded backtracking: each of the 4 segments can only be 1–3 characters long, so the search tree has
   at most `3^4 = 81` leaves regardless of input length — no pruning heuristics needed for performance.
2. A segment is valid iff it's non-empty, at most 3 characters, has no leading zero unless it's exactly
   `"0"`, and its integer value is `0`–`255` — check all of these before recursing, not after.
3. The base case must check both "4 segments chosen" *and* "the whole string has been consumed" — a
   valid 4-segment split that doesn't use every character is not a valid answer.

**Python Solution:**
```python
def restore_ip_addresses(s: str) -> list[str]:
    """
    Time:  O(1) — at most 3^4 candidate splits regardless of input length
    Space: O(1) beyond the output
    """
    n = len(s)
    results: list[str] = []

    def is_valid_octet(segment: str) -> bool:
        if not segment or len(segment) > 3:
            return False
        if segment[0] == "0" and len(segment) > 1:
            return False
        return 0 <= int(segment) <= 255

    def backtrack(start: int, parts: list[str]) -> None:
        if len(parts) == 4:
            if start == n:
                results.append(".".join(parts))
            return
        for length in range(1, 4):
            if start + length > n:
                break
            segment = s[start:start + length]
            if is_valid_octet(segment):
                parts.append(segment)
                backtrack(start + length, parts)
                parts.pop()

    backtrack(0, [])
    return results
```

**Follow-Up Questions:**
1. IPv6 instead of IPv4 → same backtracking skeleton, but 8 segments of 1–4 hex characters each with a
   different validity check (hex digits, no per-segment numeric range check).
2. Return addresses ranked by some criterion (e.g. fewest leading zeros total) → generate all valid
   candidates as above, then sort by the criterion instead of trying to bake ranking into the recursion.
3. Extremely large `n` — does the `O(1)` bound still hold? → yes, since exactly 4 segments are required
   regardless of `n`; the function returns `[]` immediately in practice once `n` exceeds `12` (max 3
   digits × 4 segments) with no combinatorial blowup.

---

## 14. Maximum Falling Path with Limited Vertical Jumps and Bonus Scoring

**Problem Statement:**
Given a grid of scores, find the maximum total achievable by picking one cell in the first row and, on
each subsequent row, moving to a column within `max_jump` of the previous column (not just adjacent, as
in the classic "falling path" problem). Landing on a cell that holds the maximum value in its own row
earns a bonus: its score is multiplied by `bonus_mult` instead of counted at face value.

**Example:**
```
grid = [[1, 2, 3],
        [10, 1, 1]]
max_jump = 1, bonus_mult = 2.0
Output: 22   # start at (0,1)=2 (no bonus, row max is 3) -> jump to (1,0)=10, row max, bonus -> 20; total 22
```

**Test Cases:**

| grid | max_jump | bonus_mult | result |
|---|---|---|---|
| `[[1,2,3]]` | `1` | `2.0` | `6` (single row: land on the row-max cell, `3 * 2.0`) |
| `[[1,2,3],[10,1,1]]` | `1` | `2.0` | `22` |
| `[[5]]` | any | any | `10` (single cell, always the row max) |
| `max_jump` ≥ number of columns | any | any | reduces to "best bonus-adjusted cell per row, freely chosen," since every column is reachable from every other |

**Key Insights:**
1. This is the classic falling-path DP (`dp[r][c] = value(r,c) + max(dp[r-1][c'] for reachable c')`)
   with the neighbor window widened from `{-1,0,1}` to `{-max_jump,...,max_jump}`, and a bonus applied to
   `value(r,c)` before adding in the best incoming path.
2. Computing the row-max once per row (not per cell) before applying the bonus avoids `O(rows*cols)`
   redundant scans.
3. The naive per-cell window-max scan is `O(rows * cols * max_jump)`; a sliding-window-maximum (monotonic
   deque) over each row's `dp` values from the previous row removes the `max_jump` factor, giving
   `O(rows * cols)` — worth mentioning as the optimization even if you implement the simpler version
   first.

**Python Solution:**
```python
def max_falling_path_with_jumps(grid: list[list[int]], max_jump: int,
                                 bonus_mult: float = 2.0) -> float:
    """
    Time:  O(rows * cols * max_jump)  (a monotonic-deque sliding-window-max removes the max_jump factor)
    Space: O(cols)
    """
    rows, cols = len(grid), len(grid[0])

    def bonus_value(r: int, c: int) -> float:
        row_max = max(grid[r])
        val = grid[r][c]
        return val * bonus_mult if val == row_max else val

    prev = [bonus_value(0, c) for c in range(cols)]
    for r in range(1, rows):
        curr = []
        for c in range(cols):
            lo = max(0, c - max_jump)
            hi = min(cols - 1, c + max_jump)
            curr.append(bonus_value(r, c) + max(prev[lo:hi + 1]))
        prev = curr

    return max(prev)
```

**Follow-Up Questions:**
1. Implement the `O(rows * cols)` sliding-window-maximum optimization → maintain a monotonic deque of
   indices over `prev` as `c` advances, popping indices that fall outside `[c - max_jump, c + max_jump]`
   from the front and popping smaller values from the back before pushing — standard sliding-window-max.
2. Multiple bonus tiers (top-1 gets `2x`, top-3 get `1.5x`) → replace the single `row_max` check with a
   precomputed rank (via `sorted`/`argsort` per row) and a tier-lookup instead of an equality check.
3. Path reconstruction (not just the max value) → track a `parent` pointer per `dp[r][c]` (which `c'` in
   the previous row it came from) alongside the value, then walk backward from the best final cell.

---

## 15. System Design — GPU Scheduling Across Competing Jobs

**Problem Statement:**
Reported as very-high-frequency across SWE/Infra Eng/EM roles. Design a scheduler that allocates GPUs
across many competing training and inference jobs in a shared cluster, where some jobs need multiple
GPUs simultaneously and jobs carry different priority levels.

**Functional Requirements:**
- Accept job submissions, each requesting some number of GPUs and a priority class.
- Multi-GPU distributed training jobs need **gang scheduling**: all requested GPUs allocated together,
  or none — a partial allocation is useless and wastes reserved-but-idle GPUs on other jobs.
- Support a cluster with heterogeneous GPU generations/memory sizes.
- Higher-priority jobs (e.g. a production inference service) can preempt lower-priority ones (e.g. batch
  training).

**Non-Functional Requirements:**
- Minimize fragmentation — a naive scheduler can leave many small, individually-unusable GPU gaps.
- Bound wait time for gang-scheduled jobs so they aren't starved indefinitely under sustained load.
- Preemption should lose as little progress as possible — prefer checkpoint-and-resume over killing a
  job outright.

**High-Level Design:**
1. **Job queue**: incoming jobs land in a priority queue; gang-scheduled (multi-GPU) jobs are tracked
   separately from single-GPU jobs since their admission test is different (need K simultaneous free
   GPUs, not just any 1).
2. **Scheduling loop**: triggered on new submissions and on GPU release. For each pending job in
   priority order, check whether enough free, compatible GPUs exist to satisfy its full requirement
   (gang jobs may also need GPUs within the same interconnect domain for bandwidth); if so, allocate
   atomically.
3. **Placement/bin-packing**: prefer packing jobs onto GPUs of the same generation/interconnect domain
   rather than round-robin placement, so large contiguous blocks stay available for future gang jobs
   instead of getting fragmented across many partially-used nodes.
4. **Preemption**: if a high-priority job can't be satisfied by free capacity alone, select lower-priority
   running job(s) to preempt; signal them to checkpoint to durable storage, reclaim their GPUs once
   checkpointed, allocate to the high-priority job, and re-queue the preempted job to resume from its
   last checkpoint when capacity frees up again.
5. **Anti-starvation**: age-based priority boost for jobs that have waited a long time, so a steady
   stream of high-priority submissions can't starve lower-priority jobs forever.

**Data Model (sketch):**
```
gpus(gpu_id, node_id, generation, interconnect_domain, status)   # FREE / ALLOCATED / DRAINING
jobs(job_id, priority, gpus_required, is_gang, status, submitted_at, checkpoint_ref)
allocations(job_id, gpu_id, allocated_at)
```

**Scaling & Reliability:**
- Cluster state (which GPUs are free/allocated) needs to be read and updated on every scheduling
  decision — keep it in a fast in-memory index or a coordination service (e.g. etcd), not a full
  database scan per attempt.
- Checkpoint-and-resume requires the training framework to support cheap periodic checkpointing to an
  object store; a job that checkpoints expensively is a poor preemption candidate, so factor checkpoint
  cost into which job gets preempted.
- With multiple scheduler replicas, use optimistic concurrency (compare-and-swap on GPU status) or a
  single active leader with fast failover to avoid double-allocating the same GPU.

**Follow-Up Questions:**
1. A job's GPUs must sit on the same physical rack/NVLink domain for interconnect bandwidth → the
   free-capacity search becomes topology-aware, turning placement into a harder constrained-bin-packing
   problem, not a flat "N free GPUs anywhere" check.
2. How do you prevent thrashing (a job being preempted and resumed repeatedly)? → a cooldown period
   after preemption before that job is itself eligible to preempt something else, and weighting
   checkpoint/resume overhead into the preemption decision, not just raw priority.
3. How does this compare to general-purpose cluster schedulers (Kubernetes, YARN)? → the core
   priority-queue-plus-preemption loop is the same shape; gang scheduling and interconnect-topology-aware
   placement are the GPU-specific pieces most general-purpose schedulers don't natively handle well.

---

## 16. Behavioral Themes

See [`behavioral_interview.md`](./behavioral_interview.md) for general STAR-method prep. OpenAI's loop
is reported to weight two closely related but distinct behavioral angles heavily:

- **Company fit ("why OpenAI")**: reported as a major gate on its own (~60 minutes) — have a genuinely
  differentiated answer for why OpenAI specifically, not a generic "AI is important" answer that would
  work interchangeably for any AI lab.
- **Career motivation ("why now")**: a separate, shorter round on what's driving a career move at this
  particular point — distinct from company fit, so don't conflate the two into one rehearsed answer.
- **Leadership**: standard ownership/impact/collaboration themes, tagged alongside the two above on the
  same round.

---

## References

Sources used for compiling these questions:
- [OpenAI Interview Questions - 1point3acres](https://www.1point3acres.com/interview/problems/company/openai)

Note: the source page requires forum membership to view full question text/discussion threads for the
reported interview-loop bank (30 coding, 18 system design, 3 behavioral, 1 other — all but one gated).
Problems 2–14 above were instead built from real, publicly-visible titles in that same page's attached
OJ (practice problem) bank of ~159 titles associated with OpenAI, which skews toward ML/LLM-infra themes
(loss functions, autograd, GPU/agent scheduling, distributed matrix ops) — each was expanded from its
title into a complete, solvable problem statement with original test cases and a verified solution, and
near-duplicate title variants (e.g. several infection-spread and memory-allocator titles) were
consolidated into one entry with follow-ups rather than repeated as separate problems.
