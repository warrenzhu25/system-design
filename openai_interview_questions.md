# OpenAI Interview Questions

---

## Contents

**Coding**
1. [Infection Spread on a Grid (Escalating Multi-Source BFS)](#1-infection-spread-on-a-grid-escalating-multi-source-bfs)

**System Design**
2. [GPU Scheduling Across Competing Jobs](#2-system-design--gpu-scheduling-across-competing-jobs)

**Behavioral**
3. [Behavioral Themes](#3-behavioral-themes)

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

---

## 2. System Design — GPU Scheduling Across Competing Jobs

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

## 3. Behavioral Themes

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

Note: the source page requires forum membership to view full question text/discussion threads; the
problems above were reconstructed and expanded from the publicly visible question titles/tags into
complete, solvable problem statements with original test cases and solutions.
