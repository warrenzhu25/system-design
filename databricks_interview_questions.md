# Databricks Interview Questions

This document consolidates the question-and-answer material currently stored under the [`DataBricks`](/Users/warren/github/system-design/DataBricks) folder.

---

## 1. Maximal Square

**Question:**
Given a matrix of `'0'` and `'1'`, find the area of the largest square containing only `'1'`.

**Answer:**
Use dynamic programming where `dp[i][j]` is the side length of the largest square whose bottom-right corner is `(i, j)`.

```python
def maximal_square(matrix: list[list[str]]) -> int:
    if not matrix or not matrix[0]:
        return 0

    rows, cols = len(matrix), len(matrix[0])
    dp = [[0] * cols for _ in range(rows)]
    max_side = 0

    for i in range(rows):
        for j in range(cols):
            if matrix[i][j] == "1":
                if i == 0 or j == 0:
                    dp[i][j] = 1
                else:
                    dp[i][j] = min(
                        dp[i - 1][j],
                        dp[i][j - 1],
                        dp[i - 1][j - 1],
                    ) + 1
                max_side = max(max_side, dp[i][j])

    return max_side * max_side
```

**Follow-up: Maximal Rectangle**

```python
def maximal_rectangle(matrix: list[list[str]]) -> int:
    if not matrix or not matrix[0]:
        return 0

    cols = len(matrix[0])
    heights = [0] * cols
    best = 0

    for row in matrix:
        for j in range(cols):
            if row[j] == "1":
                heights[j] += 1
            else:
                heights[j] = 0
        best = max(best, largest_rectangle_in_histogram(heights))

    return best


def largest_rectangle_in_histogram(heights: list[int]) -> int:
    stack = []
    best = 0
    extended = heights + [0]

    for i, h in enumerate(extended):
        while stack and extended[stack[-1]] > h:
            height = extended[stack.pop()]
            width = i if not stack else i - stack[-1] - 1
            best = max(best, height * width)
        stack.append(i)

    return best
```

**Complexity:**
- Square: `O(m * n)` time, `O(m * n)` space
- Rectangle: `O(m * n)` time, `O(n)` space

---

## 2. Lazy Array With Deferred `map`

Source: [DataBricks/lazy_array.py](/Users/warren/github/system-design/DataBricks/lazy_array.py)

**Question:**
Design a lazy array that supports chained `map(fn)` calls without eagerly applying transformations. Implement `indexOf(target)` to find the first index whose transformed value equals `target`.

**Answer:**
Store the original array plus a linked list of deferred operations. When `indexOf` is called, replay the operation chain only as needed.

```python
class OpNode:
    def __init__(self, func, pre=None):
        self.func = func
        self.pre = pre


class LazyArray:
    def __init__(self, arr, ops=None):
        self.arr = arr
        self.end_op = ops
        self.cache = {}

    def map(self, fn):
        new_ops = OpNode(fn, self.end_op)
        return LazyArray(self.arr, new_ops)

    def indexOf(self, target):
        funcs = []
        node = self.end_op
        while node:
            funcs.append(node.func)
            node = node.pre
        funcs.reverse()

        for i in range(len(self.arr)):
            cur = self.cache.get(i, self.arr[i])
            if i not in self.cache:
                for fn in funcs:
                    cur = fn(cur)
                self.cache[i] = cur
            if cur == target:
                return i
        return -1
```

**Complexity:**
- `map(fn)`: `O(1)`
- `indexOf(target)`: `O(n * k)` where `k` is number of deferred transforms

---

## 3. Snapshot Set Iterator

Source: [DataBricks/snap_shot_iterator.py](/Users/warren/github/system-design/DataBricks/snap_shot_iterator.py)

**Question:**
Design a set supporting `add`, `remove`, `contains`, and `getIterator()`, where each iterator is a snapshot of the set at the moment it is created.

**Answer:**
Track each element with a start version and end version. A snapshot iterator stores the current version and only yields elements alive in that version.

```python
class SnapshotSet:
    def __init__(self):
        self.version = 0
        self.look_up = {}
        self.tracker = []

    def add(self, n: int) -> bool:
        if n in self.look_up:
            return False
        self.look_up[n] = len(self.tracker)
        self.tracker.append([n, self.version, float("inf")])
        self.version += 1
        return True

    def remove(self, n: int) -> bool:
        if n not in self.look_up:
            return False
        index = self.look_up.pop(n)
        self.tracker[index][2] = self.version
        self.version += 1
        return True

    def contains(self, n: int) -> bool:
        return n in self.look_up

    def getIterator(self):
        return self.SnapshotIterator(self)

    class SnapshotIterator:
        def __init__(self, outer):
            self.index = 0
            self.cur_version = outer.version
            self.tracker_info = outer.tracker
            self._advance()

        def _advance(self):
            while self.index < len(self.tracker_info):
                number, start, end = self.tracker_info[self.index]
                if start < self.cur_version <= end:
                    break
                self.index += 1

        def __iter__(self):
            return self

        def __next__(self):
            self._advance()
            if self.index >= len(self.tracker_info):
                raise StopIteration
            value = self.tracker_info[self.index][0]
            self.index += 1
            return value
```

**Complexity:**
- `add` / `remove` / `contains`: `O(1)`
- Iterator creation: `O(1)`
- Full iteration: `O(total tracked elements)`

---

## 4. Generalized Tic-Tac-Toe

Source: [DataBricks/tic_tac_toe.py](/Users/warren/github/system-design/DataBricks/tic_tac_toe.py)

**Question:**
Design a generalized tic-tac-toe game on an `n x m` board where a player wins after connecting `k` marks in a row, column, diagonal, or anti-diagonal.

**Answer:**
For each player, keep a set of occupied cells. After each move, expand in four directions and count consecutive marks in both forward and backward directions.

```python
import collections


class TicTacToe:
    def __init__(self, n: int, m: int, k: int):
        self.log_info = collections.defaultdict(set)
        self.n = n
        self.m = m
        self.k = k
        self.winner = -1

    def move(self, row: int, col: int, player: int) -> int:
        if self.winner != -1:
            return self.winner
        if row < 0 or row >= self.n or col < 0 or col >= self.m:
            return 0

        self.log_info[player].add((row, col))
        seen = self.log_info[player]

        for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
            count = 1

            r, c = row + dr, col + dc
            while (r, c) in seen:
                count += 1
                r += dr
                c += dc

            r, c = row - dr, col - dc
            while (r, c) in seen:
                count += 1
                r -= dr
                c -= dc

            if count >= self.k:
                self.winner = player
                return player

        return 0
```

**Complexity:**
- Worst-case per move: `O(k)` average interview answer, up to line length in a dense board

---

## 5. Optimal Commute Path

Source: [DataBricks/find_ optimal_path.py](/Users/warren/github/system-design/DataBricks/find_%20optimal_path.py)

**Question:**
Given a grid with start `S`, destination `D`, blocked cells `X`, and transport-mode cells, find the optimal commute. The file contains two variants:
- choose the best single transport mode
- allow mixed modes and minimize total time, then cost

**Answer 1: Best Single Mode**

Run BFS for each mode independently. Compute path length, then convert to total time and total cost using that mode's per-step values.

```python
from collections import deque


def find_optimal_single_mode(grid, modes, costs, times):
    start = end = (-1, -1)
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            if grid[i][j] == "S":
                start = (i, j)
            elif grid[i][j] == "D":
                end = (i, j)

    best_mode = ""
    best_time = float("inf")
    best_cost = float("inf")

    for idx, mode in enumerate(modes):
        target = str(idx + 1)
        q = deque([(start, 0)])
        visited = {start}
        steps = -1

        while q:
            node, dist = q.popleft()
            if node == end:
                steps = dist
                break
            for dr, dc in ((0, 1), (0, -1), (-1, 0), (1, 0)):
                nr, nc = node[0] + dr, node[1] + dc
                if 0 <= nr < len(grid) and 0 <= nc < len(grid[0]) and (nr, nc) not in visited:
                    if grid[nr][nc] == target or grid[nr][nc] == "D":
                        visited.add((nr, nc))
                        q.append(((nr, nc), dist + 1))

        if steps != -1:
            total_time = steps * times[idx]
            total_cost = steps * costs[idx]
            if (total_time, total_cost) < (best_time, best_cost):
                best_time, best_cost = total_time, total_cost
                best_mode = mode

    return best_mode
```

**Answer 2: Mixed Modes**

Use Dijkstra with state priority `(time, cost)` to minimize time first and cost second.

```python
import heapq


def find_optimal_commute(grid, costs, times):
    start = end = (-1, -1)
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            if grid[i][j] == "S":
                start = (i, j)
            elif grid[i][j] == "D":
                end = (i, j)

    pq = [(0, 0, start)]
    best = {start: (0, 0)}

    while pq:
        time, cost, node = heapq.heappop(pq)
        if (time, cost) > best.get(node, (float("inf"), float("inf"))):
            continue
        if node == end:
            return [time, cost]

        if node == start:
            step_time = step_cost = 0
        else:
            idx = int(grid[node[0]][node[1]]) - 1
            step_time = times[idx]
            step_cost = costs[idx]

        for dr, dc in ((0, 1), (0, -1), (-1, 0), (1, 0)):
            nr, nc = node[0] + dr, node[1] + dc
            if 0 <= nr < len(grid) and 0 <= nc < len(grid[0]) and grid[nr][nc] != "X":
                cand = (time + step_time, cost + step_cost)
                if cand < best.get((nr, nc), (float("inf"), float("inf"))):
                    best[(nr, nc)] = cand
                    heapq.heappush(pq, (cand[0], cand[1], (nr, nc)))

    return [-1, -1]
```

**Cost model note:** the step cost charged when moving to a neighbor is the
*current* cell's transport mode (`grid[node]`), i.e. you pay for the cell you are
leaving. The start cell costs 0, and the destination cell's own mode is never
charged. If you instead want to charge for the cell you enter, compute
`step_time`/`step_cost` from `grid[nr][nc]` inside the neighbor loop.

---

## 6. Hit Counter With Variable Window Queries

Source: [DataBricks/hit_counter.py](/Users/warren/github/system-design/DataBricks/hit_counter.py)

**Question:**
Design a hit counter supporting:
- `hit(timestamp)`
- `get_load(seconds)`
- `get_qps(seconds)`

Also discuss how to optimize for high QPS queries and variable time-window queries.

**Answer:**
The file contains two versions:
- exact prefix-sum based counter with caching
- bucketed rollup version for longer windows

**Exact Version**

```python
class HitCounter:
    def __init__(self):
        self.timestamps = []
        self.current_time = 0
        self.prefix_count = []
        self.cache = {}

    def hit(self, timestamp):
        self.current_time = max(self.current_time, timestamp)
        prev = self.prefix_count[-1] if self.prefix_count else 0

        if not self.timestamps or self.timestamps[-1] != timestamp:
            self.timestamps.append(timestamp)
            self.prefix_count.append(prev + 1)
        else:
            self.prefix_count[-1] += 1

        self.cache.clear()

    def get_load(self, seconds):
        if not self.timestamps or seconds <= 0:
            return 0
        if seconds in self.cache:
            return self.cache[seconds]

        target = self.current_time - seconds
        left, right = 0, len(self.timestamps) - 1
        idx = -1
        while left <= right:
            mid = left + (right - left) // 2
            if self.timestamps[mid] <= target:
                idx = mid
                left = mid + 1
            else:
                right = mid - 1

        cutoff = self.prefix_count[idx] if idx != -1 else 0
        total = self.prefix_count[-1]
        self.cache[seconds] = total - cutoff
        return total - cutoff

    def get_qps(self, seconds):
        return 0 if seconds <= 0 else self.get_load(seconds) / seconds
```

**Discussion:**
- Cache recent `get_load` results for repeated windows
- Use multi-level buckets for large time ranges with bounded memory
- Window boundary: `get_load(seconds)` counts hits with `timestamp > current_time - seconds`,
  i.e. the half-open window `(current_time - seconds, current_time]`. If you want an
  inclusive last-`N`-seconds window, search for `timestamps[mid] < target` instead.
- Assumption: `hit()` is called with non-decreasing timestamps, so `timestamps`/`prefix_count`
  stay sorted for the binary search. An out-of-order hit appended at the tail would break that
  invariant — handle it by inserting in sorted position (`bisect`) if clocks can go backwards.

---

## 7. IP Firewall With CIDR Rules

Source: [DataBricks/ip_cird.py](/Users/warren/github/system-design/DataBricks/ip_cird.py)

**Question:**
Given firewall rules such as `ALLOW 1.2.3.0/24` and `DENY 1.2.3.4`, determine whether access should be allowed for an IP. The file contains trie-based solutions for matching CIDR prefixes with rule priority.

**Answer:**
Convert IPs to 32-bit binary strings and store rule prefixes in a trie. During lookup, walk the trie while tracking the highest-priority matching rule.

```python
class Trie:
    def __init__(self):
        self.children = {}
        self.priority = float("inf")
        self.action = None


class IpFirewall:
    def __init__(self, rules):
        self.root = Trie()
        for index, (action, ip_mask) in enumerate(rules):
            allow = action == "ALLOW"
            if "/" in ip_mask:
                ip, mask = ip_mask.split("/")
                mask = int(mask)
            else:
                ip, mask = ip_mask, 32

            bits = self._ip_to_bits(ip)
            node = self.root
            for i in range(mask):
                bit = bits[i]
                if bit not in node.children:
                    node.children[bit] = Trie()
                node = node.children[bit]

            if index < node.priority:
                node.priority = index
                node.action = allow

    def _ip_to_bits(self, ip):
        a, b, c, d = map(int, ip.split("."))
        return format((a << 24) | (b << 16) | (c << 8) | d, "032b")

    def allowAccess(self, ip):
        bits = self._ip_to_bits(ip)
        node = self.root
        best_priority = float("inf")
        best_action = False

        if node.priority < best_priority:
            best_priority = node.priority
            best_action = node.action

        for bit in bits:
            if bit not in node.children:
                break
            node = node.children[bit]
            if node.priority < best_priority:
                best_priority = node.priority
                best_action = node.action

        return best_action
```

**Complexity:**
- Insert: `O(32)`
- Query: `O(32)`

**Note on precedence:** this uses *first-match-by-rule-order* — the lowest rule
index along the matched path wins. So with `ALLOW 1.2.3.0/24` (index 0) then
`DENY 1.2.3.4` (index 1), the IP `1.2.3.4` is **allowed**: the broader earlier
rule wins. To deny a specific IP inside an allowed subnet, list the `DENY` first,
or switch the precedence rule to *longest-prefix wins* (take the action of the
deepest matched node instead of the lowest index).

---

## 8. Circuit Breaker Gateway

Source: [DataBricks/circuit_break.py](/Users/warren/github/system-design/DataBricks/circuit_break.py)

**Question:**
Implement a gateway with primary and secondary backends, each protected by a circuit breaker. Route requests according to circuit state and failure thresholds.

**Answer:**
Each breaker tracks:
- `failureThreshold`
- `resetThreshold`
- current consecutive failure count
- current skipped-request count
- whether the breaker is open

If the primary is unavailable or fails, try the secondary. If both are open, reject the request.

```python
class Server:
    def __init__(self, outcomes):
        self.outcomes = outcomes
        self.callCount = 0

    def handle(self, requestId):
        self.callCount += 1
        return self.outcomes[requestId]


class CircuitBreaker:
    def __init__(self, server, failureThreshold, resetThreshold):
        self.server = server
        self.failureThreshold = failureThreshold
        self.resetThreshold = resetThreshold
        self.is_open = False
        self.fail_count = 0
        self.skip_count = 0


class Gateway:
    def __init__(self, primaryBreaker, secondaryBreaker):
        self.primaryBreaker = primaryBreaker
        self.secondaryBreaker = secondaryBreaker

    def routeRequests(self, totalRequests):
        res = []
        for r in range(totalRequests):
            process_primary = False
            process_secondary = False
            fail_primary = True
            fail_secondary = True

            if not self.primaryBreaker.is_open:
                process_primary = True
                ok = self.primaryBreaker.server.handle(r)
                if ok:
                    fail_primary = False
                    self.primaryBreaker.fail_count = 0
                else:
                    self.primaryBreaker.fail_count += 1
                    if self.primaryBreaker.fail_count == self.primaryBreaker.failureThreshold:
                        self.primaryBreaker.is_open = True
                        self.primaryBreaker.fail_count = 0
            else:
                self.primaryBreaker.skip_count += 1
                if self.primaryBreaker.skip_count == self.primaryBreaker.resetThreshold:
                    self.primaryBreaker.is_open = False
                    self.primaryBreaker.skip_count = 0

            if (not process_primary) or fail_primary:
                if not self.secondaryBreaker.is_open:
                    process_secondary = True
                    ok = self.secondaryBreaker.server.handle(r)
                    if ok:
                        fail_secondary = False
                        self.secondaryBreaker.fail_count = 0
                    else:
                        self.secondaryBreaker.fail_count += 1
                        if self.secondaryBreaker.fail_count == self.secondaryBreaker.failureThreshold:
                            self.secondaryBreaker.is_open = True
                            self.secondaryBreaker.fail_count = 0
                else:
                    self.secondaryBreaker.skip_count += 1
                    if self.secondaryBreaker.skip_count == self.secondaryBreaker.resetThreshold:
                        self.secondaryBreaker.is_open = False
                        self.secondaryBreaker.skip_count = 0

            # Label by what actually SUCCEEDED, not just what was attempted.
            # A primary failure with no working fallback must be "Rejected",
            # not "Primary" (which previously looked like a success).
            if process_primary and not fail_primary:
                res.append("Primary")
            elif process_secondary and not fail_secondary:
                res.append("Primary -> Secondary" if process_primary else "Secondary")
            else:
                res.append("Rejected")

        return res
```

---

## 9. Bottleneck Nodes in a DAG

Source: [DataBricks/bottleneck.py](/Users/warren/github/system-design/DataBricks/bottleneck.py)

**Question:**
Given a DAG with `n` nodes and directed edges, find the bottleneck nodes. In this solution, a bottleneck is a layer in topological order that contains exactly one available node.

**Answer:**
Run topological sorting with indegree counting. Whenever the queue size is exactly `1`, record that node as a bottleneck candidate.

```python
import collections
from collections import deque


class Solution:
    def findBottlenecks(self, n, edges):
        graph = collections.defaultdict(list)
        indegree = [0] * n

        for u, v in edges:
            graph[u].append(v)
            indegree[v] += 1

        q = deque(i for i in range(n) if indegree[i] == 0)
        res = []
        processed = 0

        while q:
            if len(q) == 1:
                res.append(q[0])

            for _ in range(len(q)):
                node = q.popleft()
                processed += 1
                for nxt in graph[node]:
                    indegree[nxt] -= 1
                    if indegree[nxt] == 0:
                        q.append(nxt)

        return res if processed == n else []
```

**Complexity:**
- Time: `O(n + e)`
- Space: `O(n + e)`

**Note on the definition:** this finds *topological layers of size 1*, which is
**not** the same as a graph dominator (a node every path must cross). Example:
edges `0->1, 1->2, 0->3` report node `2` as a bottleneck, but the path `0->3`
never passes through it. The single-node-layer heuristic is fine if that is the
stated definition; for true must-pass-through nodes you need a dominator-tree
algorithm.

---

## 10. Customer Revenue System

Source: [DataBricks/customer_revenue.py](/Users/warren/github/system-design/DataBricks/customer_revenue.py)

**Question:**
Design a system that:
- adds a customer with initial revenue
- adds a customer by referral and increases the referrer's revenue
- returns the top `k` customers above a revenue threshold
- returns referral relationships level by level

**Answer:**
Use:
- `id_to_revenue` for direct lookup
- a sorted ranking structure ordered by revenue descending
- an adjacency list for referral relationships

```python
from sortedcontainers import SortedSet
import collections
from collections import deque


class RevenueSystem:
    def __init__(self):
        self.customer_id = -1
        self.id_to_revenue = {}
        self.ranking = SortedSet()
        self.children = collections.defaultdict(list)

    def add(self, revenue):
        self.customer_id += 1
        self.id_to_revenue[self.customer_id] = revenue
        self.ranking.add((-revenue, self.customer_id))
        return self.customer_id

    def addByReferral(self, revenue, referrerId):
        if referrerId not in self.id_to_revenue:
            return -1

        old_revenue = self.id_to_revenue[referrerId]
        self.ranking.remove((-old_revenue, referrerId))
        new_revenue = old_revenue + revenue
        self.ranking.add((-new_revenue, referrerId))
        self.id_to_revenue[referrerId] = new_revenue

        child = self.add(revenue)
        self.children[referrerId].append(child)
        return child

    def getTopKCustomer(self, k, minRevenue):
        res = []
        for neg_revenue, customer_id in self.ranking:
            revenue = -neg_revenue
            if len(res) == k or revenue < minRevenue:
                break
            res.append(customer_id)
        return res

    def getRelations(self, customerId):
        if customerId not in self.id_to_revenue:
            return []

        q = deque([customerId])
        res = []
        while q:
            cur = []
            for _ in range(len(q)):
                node = q.popleft()
                if node in self.children:
                    cur.extend(self.children[node])
                    for nxt in self.children[node]:
                        q.append(nxt)
            if cur:
                res.append(sorted(cur))
        return res
```

---

## 11. Delete Index From Interval List

Source: [DataBricks/delete_index_from_interval.py](/Users/warren/github/system-design/DataBricks/delete_index_from_interval.py)

**Question:**
A sorted interval list represents a flattened set of indices. Remove the element at a given logical index and return the updated interval list.

The file contains two variants:
- intervals inclusive: `[start, end]`
- intervals half-open: `[start, end)`

**Answer: Inclusive Intervals**

```python
def remove_intervals_include(intervals, index):
    found = False
    res = []

    for start, end in intervals:
        if found:
            res.append([start, end])
            continue

        length = end - start + 1
        if index < length:
            found = True
            remove_point = start + index
            if remove_point > start:
                res.append([start, remove_point - 1])
            if remove_point < end:
                res.append([remove_point + 1, end])
        else:
            index -= length
            res.append([start, end])

    return res
```

**Answer: Half-Open Intervals**

```python
def remove_intervals_not_include(intervals, index):
    found = False
    res = []

    for start, end in intervals:
        if found:
            res.append([start, end])
            continue

        length = end - start
        if index < length:
            found = True
            remove_point = start + index
            if remove_point > start:
                res.append([start, remove_point])
            if remove_point + 1 < end:
                res.append([remove_point + 1, end])
        else:
            index -= length
            res.append([start, end])

    return res
```

---

## 12. Encode / Decode With Hybrid RLE and Bit-Packing

Source: [DataBricks/encode_decode.py](/Users/warren/github/system-design/DataBricks/encode_decode.py)

**Question:**
Design an encoder/decoder for integer sequences using:
- `RLE[value, count]` for long runs
- `BP[...]` for short non-run groups

The solution uses RLE when a run length is at least `8`, otherwise it falls back to bit-packing groups.

**Answer:**

```python
class Solution:
    def encode(self, values):
        if len(values) < 1:
            return []

        index = 0
        res = []
        n = len(values)

        while index < n:
            count = 1
            while index + count < n and values[index + count] == values[index]:
                count += 1

            if count >= 8 or index + count == n:
                res.append(f"RLE[{values[index]}, {count}]")
                index += count
            else:
                bp_item = []
                while index < n and len(bp_item) < 8:
                    new_count = 1
                    while index + new_count < n and values[index + new_count] == values[index]:
                        new_count += 1

                    if new_count >= 8 or (index + new_count == n and new_count > 1):
                        break

                    bp_item.append(values[index])
                    index += 1

                if bp_item:
                    res.append(f"BP{bp_item}")

        return res

    def decode(self, runs):
        res = []
        for log in runs:
            if log.startswith("RLE"):
                info = log[4:-1]
                number, count = info.split(",")
                # Cast number to int so RLE and BP decode to the same type
                # (the original returned ints for BP but strings for RLE).
                res += [int(number)] * int(count)
            elif log.startswith("BP"):
                info = log[3:-1]
                number_list = info.split(",")
                for n in number_list:
                    res.append(int(n))
        return res
```

**Note:**
The string parsing mirrors the source file. The one change from the original is
casting the RLE `number` to `int`, so `decode` returns a consistent list of ints
(the original returned ints for `BP` groups but strings for `RLE` runs).

---

## 13. Fibonacci Tree Path

Source: [DataBricks/fibonacci_tree.py](/Users/warren/github/system-design/DataBricks/fibonacci_tree.py)

**Question:**
Find the path between two nodes in an implicit Fibonacci tree. The file also includes the standard binary-tree `getDirections` solution as a reference pattern.

**Answer:**
The Fibonacci tree version computes subtree sizes recursively and uses them to determine whether a target lies in the left or right subtree.

```python
class Solution:
    def findPath(self, order: int, source: int, dest: int) -> str:
        cache = {}

        def get_cache(n):
            if n < 0:
                return 0
            if n <= 1:
                return 1
            if n in cache:
                return cache[n]
            cache[n] = 1 + get_cache(n - 1) + get_cache(n - 2)
            return cache[n]

        def find_path(n, root, target):
            if root == target:
                return ""

            left_root = root + 1
            left_size = get_cache(n - 2)

            if left_root <= target < left_root + left_size:
                return "L" + find_path(n - 2, left_root, target)
            else:
                right_root = left_root + left_size
                return "R" + find_path(n - 1, right_root, target)

        source_path = find_path(order, 0, source)
        dest_path = find_path(order, 0, dest)

        i = 0
        while i < min(len(source_path), len(dest_path)) and source_path[i] == dest_path[i]:
            i += 1

        return "U" * (len(source_path) - i) + dest_path[i:]
```

**Idea:**
- compute the root-to-source path
- compute the root-to-destination path
- strip their common prefix
- move up with `U`, then follow destination suffix

---

## 14. Time-Based Key-Value Store

Source: [DataBricks/kv_storage.py](/Users/warren/github/system-design/DataBricks/kv_storage.py)

**Question:**
Implement a time-based key-value store supporting:
- `set(key, value, timestamp)`
- `get(key, timestamp)` returning the most recent value at or before `timestamp`

**Answer:**
Store timestamp-sorted values per key and binary search during `get`.

```python
class TimeMap:
    def __init__(self):
        self.key_time_map = {}

    def set(self, key, value, timestamp):
        if key not in self.key_time_map:
            self.key_time_map[key] = []
        self.key_time_map[key].append([timestamp, value])

    def get(self, key, timestamp):
        if key not in self.key_time_map:
            return ""
        if timestamp < self.key_time_map[key][0][0]:
            return ""

        left, right = 0, len(self.key_time_map[key]) - 1
        idx = -1

        while left <= right:
            mid = left + (right - left) // 2
            if self.key_time_map[key][mid][0] <= timestamp:
                idx = mid
                left = mid + 1
            else:
                right = mid - 1

        return self.key_time_map[key][idx][1] if idx != -1 else ""
```

**Complexity:**
- `set`: `O(1)` amortized
- `get`: `O(log n)` per key history

**Assumption:** `set` is called with non-decreasing timestamps per key (the usual
contract), so each key's list stays sorted and `get` can binary search. If writes
can arrive out of order, insert with `bisect.insort` (making `set` `O(n)`), or
sort lazily before the first `get`.

---

## Notes

- The markdown above is derived from the current Python files under [`DataBricks`](/Users/warren/github/system-design/DataBricks).
- Some source files contain rough or interview-draft code. This doc preserves the intended question and the answer pattern from those files rather than rewriting the entire set into production-quality implementations.

---

## 15. Tagged Command Undo

**Question:**
Design a `CommandLog` supporting:
- `execute(command, tags)` to append a command with zero or more tags
- `undo()` to undo the most recent active command
- `undo(tag)` to undo the most recent active command containing `tag`

Each command can be undone only once. All operations should remain efficient up to `10^5` calls.

**Answer:**
Use:
- a global append-only command array
- one stack of command indices per tag
- a boolean `active` array
- a pointer for global undo cleanup

For `undo(tag)`, keep popping stale indices from that tag's stack until the top points to an active command. For plain `undo()`, walk backward from the global tail until an active command is found.

```python
from collections import defaultdict


class CommandLog:
    def __init__(self):
        self.commands = []
        self.tags = []
        self.active = []
        self.tag_to_stack = defaultdict(list)
        self.last_active = -1

    def execute(self, command: str, tags: list[str]) -> None:
        idx = len(self.commands)
        self.commands.append(command)
        self.tags.append(tags)
        self.active.append(True)
        self.last_active = idx

        for tag in tags:
            self.tag_to_stack[tag].append(idx)

    def undo(self, tag: str | None = None) -> str:
        if tag is None:
            while self.last_active >= 0 and not self.active[self.last_active]:
                self.last_active -= 1
            if self.last_active < 0:
                return ""

            idx = self.last_active
            self.active[idx] = False
            self.last_active -= 1
            return self.commands[idx]

        stack = self.tag_to_stack[tag]
        while stack and not self.active[stack[-1]]:
            stack.pop()
        if not stack:
            return ""

        idx = stack.pop()
        self.active[idx] = False
        return self.commands[idx]
```

**Why this works:**
- Each command index is pushed once per tag
- Each stale stack entry is popped at most once
- The global pointer only moves left
- This gives amortized `O(1)` per operation

**Complexity:**
- `execute`: `O(t)` where `t` is number of tags on the command
- `undo()`: amortized `O(1)`
- `undo(tag)`: amortized `O(1)`

**Example:**

```python
log = CommandLog()
log.execute("A", ["x"])
log.execute("B", ["y"])
log.execute("C", ["x", "y"])

assert log.undo() == "C"
assert log.undo("x") == "A"
assert log.undo("y") == "B"
```

---

## 16. Throttling System Design

**Question:**
Design a throttling system that protects both:
- incoming traffic at the gateway
- outgoing traffic from API servers to databases, internal services, and third-party APIs

The goal is to prevent overload and cascading failure across the stack.

**Clarifying Questions:**
- Is the traffic steady or spiky?
- How much is internal vs external?
- What are the downstream capacity limits?
- Should internal traffic get priority?
- Do we need per-user, per-tier, or per-endpoint limits?
- Can limits change dynamically without restart?
- What should clients see when throttled: `429`, queueing, or degraded service?

### Requirements

**Functional Requirements:**
- enforce incoming rate limits at the gateway
- enforce outgoing limits toward each dependency
- support different quotas for free, paid, and internal users
- isolate tenants so one noisy client cannot starve others
- degrade gracefully with `429` or fallback responses
- support dynamic configuration updates

**Non-Functional Requirements:**
- low latency, ideally under `5 ms` per check
- high availability even if the limiter datastore degrades
- near-accurate enforcement, not necessarily perfect
- strong observability and alerting
- fairness under bursty traffic

### High-Level Design

```text
Client
  -> Gateway / Load Balancer
     -> Incoming Rate Limiter
        -> API Servers
           -> Outgoing Proxy / Sidecar
              -> DB / Internal APIs / 3P APIs

Config Service -> pushes limits to Gateway + Proxy fleet
Redis / local cache -> token state
Metrics pipeline -> Prometheus / dashboards / alerts
```

### Incoming Throttling

**Recommended algorithm: Token Bucket**
- each principal gets tokens refilled at a steady rate
- each request consumes a token
- bursts are allowed up to bucket capacity
- when empty, reject with `429 Too Many Requests`

**Why Token Bucket:**
- simple
- supports bursts better than fixed window
- predictable and cheap to implement

**Enforcement order:**
1. global system limit
2. tenant or tier limit
3. user or API key limit
4. endpoint-specific limit

Reject on the first violated limit.

**Distributed setup:**
- gateways share bucket state through Redis
- use Lua scripts for atomic refill-plus-consume
- maintain a short-lived local cache for ultra-hot keys

### Outgoing Throttling

For downstream dependencies, the limiter should sit in an on-host proxy or sidecar near the API server.

**Controls to combine:**
- per-dependency token buckets
- concurrency limits for expensive calls
- circuit breakers
- bounded retries with exponential backoff and jitter

**Circuit breaker states:**
- `CLOSED`: requests flow normally
- `OPEN`: fail fast because dependency is unhealthy
- `HALF_OPEN`: allow limited probes to test recovery

This prevents one failing database or third-party service from taking the entire system down.

### Data Model

**ThrottleConfig**
- principal id
- tier
- refill rate
- burst capacity
- per-endpoint overrides
- dependency-specific outgoing limits

**RateLimitState**
- tokens remaining
- last refill timestamp
- optional rolling error counters

**CircuitBreakerState**
- dependency name
- state
- recent failure count
- probe window
- last state transition time

### Storage Choices

- config of record: PostgreSQL or another durable store
- fast counters: Redis
- local hot cache: in-memory per gateway / sidecar
- metrics: Prometheus or another time-series system

### Dynamic Configuration

Limits should update without restart.

**Approach:**
- admins write config to the config service
- config service persists to durable storage
- publish change events over pub/sub
- gateways and sidecars subscribe and refresh local caches

This supports near-real-time rollout of new limits.

### Failure Handling

**Redis unavailable**
- fall back to local best-effort limits
- prefer partial protection over no protection
- alert aggressively because enforcement accuracy is degraded

**Hot key / noisy tenant**
- shard counters
- use local admission checks before Redis
- enforce stricter tenant-level caps

**Thundering herd**
- include jitter in `Retry-After`
- spread retries with randomized backoff
- optionally use client-visible quota headers

### Observability

Track:
- allowed vs throttled requests
- bucket exhaustion rate
- per-tenant reject rate
- downstream error rates and latency
- open circuit count
- retry volume
- queue depth and concurrency saturation

Alert on:
- sudden spike in throttles
- dependency saturation
- rising tail latency
- fallback mode activation

### Tradeoffs and Interview Discussion

**Token Bucket vs Sliding Window**
- token bucket is better for controlled bursts
- sliding window is more exact but usually more expensive

**Incoming vs Outgoing**
- incoming protects your fleet from clients
- outgoing protects dependencies from your own fleet
- you generally need both

**Accuracy vs availability**
- a globally consistent limiter is slower and more fragile
- a slightly approximate limiter with local fallback is usually the better operational choice

### Good Final Answer Summary

Use token buckets at the gateway for user-facing rate limits, plus sidecar or proxy-based throttling and circuit breakers for downstream calls. Store fast-changing counters in Redis with local caching, keep configs in a durable config service, push updates dynamically, and design the system to fail gracefully with `429`, jittered retries, and observability around saturation and cascading failure.

---

## 17. House Robber

**Question:**
You are given an array `nums` where `nums[i]` is the amount of money in house `i`. You cannot rob two adjacent houses. Return the maximum amount of money you can rob.

**Core DP Relation:**

```python
dp[i] = max(dp[i - 1], nums[i] + dp[i - 2])
```

At each house:
- skip it, keeping the best answer up to `i - 1`
- rob it, adding its value to the best answer up to `i - 2`

### Best Solution: O(1) Space

```python
def rob(nums: list[int]) -> int:
    prev, curr = 0, 0

    for num in nums:
        prev, curr = curr, max(curr, prev + num)

    return curr
```

**Complexity:**
- Time: `O(n)`
- Space: `O(1)`

### Recursive + Memo

```python
def rob_memo(nums: list[int]) -> int:
    memo = {}

    def dp(i: int) -> int:
        if i < 0:
            return 0
        if i == 0:
            return nums[0]
        if i in memo:
            return memo[i]

        memo[i] = max(nums[i] + dp(i - 2), dp(i - 1))
        return memo[i]

    return dp(len(nums) - 1)
```

**Complexity:**
- Time: `O(n)`
- Space: `O(n)`

### Iterative DP Table

```python
def rob_table(nums: list[int]) -> int:
    n = len(nums)
    if n == 0:
        return 0
    if n == 1:
        return nums[0]

    dp = [0] * n
    dp[0] = nums[0]
    dp[1] = max(nums[0], nums[1])

    for i in range(2, n):
        dp[i] = max(dp[i - 1], nums[i] + dp[i - 2])

    return dp[-1]
```

### Variation 1: Circular Street

If houses form a circle, you cannot rob both the first and last house.

**Idea:**
Solve two linear cases:
- rob from `0..n-2`
- rob from `1..n-1`

```python
def rob_circular(nums: list[int]) -> int:
    n = len(nums)
    if n == 0:
        return 0
    if n == 1:
        return nums[0]

    def rob_linear(arr: list[int]) -> int:
        prev, curr = 0, 0
        for num in arr:
            prev, curr = curr, max(curr, prev + num)
        return curr

    return max(rob_linear(nums[:-1]), rob_linear(nums[1:]))
```

**Complexity:**
- Time: `O(n)`
- Space: `O(1)`

### Variation 2: Gap Constraint `k`

If robbing house `i` means you must skip the next `k` houses, then:

```python
dp[i] = max(dp[i - 1], nums[i] + dp[i - k - 1])
```

```python
def rob_with_gap(nums: list[int], k: int) -> int:
    n = len(nums)
    if n == 0:
        return 0

    dp = [0] * n
    for i in range(n):
        take = nums[i]
        if i - k - 1 >= 0:
            take += dp[i - k - 1]
        skip = dp[i - 1] if i > 0 else 0
        dp[i] = max(skip, take)

    return dp[-1]
```

**Complexity:**
- Time: `O(n)`
- Space: `O(n)`

### Variation 3: Houses in a Tree

If houses are arranged in a binary tree, you cannot rob a node and its direct children.

**Idea:**
For each node, return:
- best value if you rob this node
- best value if you skip this node

```python
def rob_tree(root) -> int:
    def dfs(node):
        if not node:
            return 0, 0

        left_rob, left_skip = dfs(node.left)
        right_rob, right_skip = dfs(node.right)

        rob = node.val + left_skip + right_skip
        skip = max(left_rob, left_skip) + max(right_rob, right_skip)

        return rob, skip

    rob, skip = dfs(root)
    return max(rob, skip)
```

### Common Edge Cases

- empty array -> `0`
- one house -> that value
- two houses -> `max(nums[0], nums[1])`
- all zero values -> `0`
- circular case with two houses -> choose the larger one

### Interview Summary

This is a classic dynamic programming problem. The key observation is that every house creates a binary choice: rob it or skip it. The linear version reduces to keeping only the previous two states, which gives the optimal `O(n)` time and `O(1)` space solution. The circular, gap, and tree variants all reuse the same underlying include-or-exclude idea with different state transitions.

---

## 18. Durable Concurrent Data Writer

**Question:**
Design a thread-safe library that writes logs to a file on a single server.

```java
class DataWriter {
    public DataWriter(String filePathOnDisk) {
    }

    public void push(byte[] data) {
        // must block until data is safely on disk
    }
}
```

Requirements:
- thousands of threads may call `push()` concurrently
- `push()` must return only after the data is durable on disk
- writes from the same thread must preserve order
- writes from different threads may interleave
- the system must handle crash recovery if the process dies mid-write

### Core Design

The right design is **group commit**:
- calling threads enqueue requests
- a single background writer thread drains requests into a batch
- the writer appends the batch to the file
- the writer calls `fsync` once for the whole batch
- all waiting callers are released only after `fsync` succeeds

This keeps durability while amortizing the cost of `fsync`.

### Why Per-Write `fsync` Fails

If `fsync` costs about `5 ms`, then one-thread-at-a-time writing gives only about:

- `200 fsyncs / second`
- `200 writes / second` if each write gets its own `fsync`

With batching, one `fsync` can cover tens or hundreds of writes, so throughput rises dramatically while keeping the durability contract.

### High-Level Architecture

```text
Caller Threads
  -> enqueue WriteRequest
  -> block on per-request latch / future

Single Writer Thread
  -> drain queue into batch
  -> serialize records
  -> append to file
  -> fsync / force
  -> mark all requests in batch complete
```

### Recommended Java Implementation

```java
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.channels.FileChannel;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.TimeUnit;
import java.util.zip.CRC32;

public class DataWriter {
    private final FileChannel channel;
    private final BlockingQueue<WriteRequest> queue = new LinkedBlockingQueue<>();
    private final Thread writerThread;
    private volatile boolean running = true;

    private static final int MAX_BATCH_SIZE = 1024;
    private static final int MAX_BATCH_BYTES = 4 * 1024 * 1024;
    private static final long BATCH_TIMEOUT_MS = 1;

    public DataWriter(String filePathOnDisk) throws IOException {
        this.channel = FileChannel.open(
            Path.of(filePathOnDisk),
            StandardOpenOption.CREATE,
            StandardOpenOption.WRITE,
            StandardOpenOption.APPEND
        );

        this.writerThread = new Thread(this::writerLoop, "data-writer");
        this.writerThread.start();
    }

    public void push(byte[] data) throws IOException, InterruptedException {
        WriteRequest req = new WriteRequest(data);
        queue.put(req);
        req.await();

        if (req.error != null) {
            throw new IOException("write failed", req.error);
        }
    }

    private void writerLoop() {
        List<WriteRequest> batch = new ArrayList<>(MAX_BATCH_SIZE);

        while (running) {
            try {
                batch.clear();
                collectBatch(batch);
                if (!batch.isEmpty()) {
                    processBatch(batch);
                }
            } catch (Exception e) {
                for (WriteRequest req : batch) {
                    req.fail(e);
                }
            }
        }
    }

    private void collectBatch(List<WriteRequest> batch) throws InterruptedException {
        long totalBytes = 0;
        long deadline = System.currentTimeMillis() + BATCH_TIMEOUT_MS;

        WriteRequest first = queue.poll(BATCH_TIMEOUT_MS, TimeUnit.MILLISECONDS);
        if (first == null) {
            return;
        }

        batch.add(first);
        totalBytes += first.data.length;

        while (batch.size() < MAX_BATCH_SIZE
                && totalBytes < MAX_BATCH_BYTES
                && System.currentTimeMillis() < deadline) {
            WriteRequest next = queue.poll();
            if (next == null) {
                break;
            }
            batch.add(next);
            totalBytes += next.data.length;
        }
    }

    private void processBatch(List<WriteRequest> batch) throws IOException {
        int totalSize = 0;
        for (WriteRequest req : batch) {
            totalSize += 4 + req.data.length + 4;
        }

        ByteBuffer buffer = ByteBuffer.allocateDirect(totalSize);
        CRC32 crc = new CRC32();

        for (WriteRequest req : batch) {
            buffer.putInt(req.data.length);
            buffer.put(req.data);
            crc.reset();
            crc.update(req.data);
            buffer.putInt((int) crc.getValue());
        }

        buffer.flip();
        while (buffer.hasRemaining()) {
            channel.write(buffer);
        }

        channel.force(true);

        for (WriteRequest req : batch) {
            req.complete();
        }
    }

    public void close() throws IOException, InterruptedException {
        running = false;
        writerThread.interrupt();
        writerThread.join();
        channel.close();
    }

    private static class WriteRequest {
        final byte[] data;
        final CountDownLatch done = new CountDownLatch(1);
        volatile Exception error;

        WriteRequest(byte[] data) {
            this.data = data;
        }

        void await() throws InterruptedException {
            done.await();
        }

        void complete() {
            done.countDown();
        }

        void fail(Exception e) {
            error = e;
            done.countDown();
        }
    }
}
```

### Ordering Guarantee

For a single caller thread, ordering is preserved naturally because:
- `push()` blocks until durability completes
- the thread cannot issue the next logically dependent write until the previous call returns

That means if a thread calls `push(d1)` and then `push(d2)`, `d1` must enter the file before `d2`.

Across different threads, ordering is intentionally flexible. Any interleaving is acceptable as long as each individual thread's order is preserved.

### Durability Guarantee

`push()` must not return after only calling `write()`.

That would only place bytes in the kernel page cache. To meet the requirement, the implementation must call:

```java
channel.force(true);
```

and only then release waiting callers.

### File Format

A simple append-only record format works well:

```text
[length:4 bytes][payload:length bytes][crc32:4 bytes]
```

Benefits:
- length tells recovery where the record ends
- checksum detects torn or corrupted writes
- append-only layout keeps the implementation simple

### Crash Recovery

On startup:
1. scan from the beginning of the file
2. read `length`
3. read `payload`
4. read stored `crc32`
5. recompute checksum
6. if anything is incomplete or invalid, truncate the file at the last good offset

This handles:
- process crash during write
- machine restart during append
- partial tail corruption

### Concurrency Strategy

Using one file-writing thread is usually the correct choice because:
- file appends must serialize anyway
- it avoids lock contention among thousands of callers
- it centralizes batching and durability

Calling threads do not touch the file directly. They only:
- allocate a request object
- enqueue it
- block waiting for completion

### Performance Discussion

Let:
- `fsync = 5 ms`
- `200 fsyncs / second`

Then approximate throughput becomes:
- without batching: `200 writes / second`
- batch size `50`: about `10,000 writes / second`
- batch size `250`: about `50,000 writes / second`

Tradeoff:
- larger batches improve throughput
- larger batches also increase tail latency

So the usual tuning knobs are:
- max batch size
- max batch bytes
- max batch wait time

### Why This Design Is Good

- durable: callers return only after `fsync`
- thread-safe: callers do not share mutable file-write state
- ordered per thread: blocking semantics preserve program order
- high throughput: expensive `fsync` cost is amortized
- recoverable: append-only framing plus checksum enables tail repair

### Common Bad Designs

**Per-call synchronized write + fsync**
- correct but far too slow
- all threads serialize behind disk latency

**Async enqueue and return immediately**
- high throughput but violates durability
- caller cannot know whether data survived a crash

**Write without fsync**
- appears fast but only persists to page cache
- power loss can lose acknowledged data

### Interview Summary

Use a single background writer with group commit. Let callers enqueue requests and block on a latch or future. Serialize each record with length and checksum, append batches to the file, call `fsync` once per batch, then wake all waiting callers. On restart, scan and truncate the file at the last valid record. This is the standard way to get both durability and high throughput on a single machine.

---

## 19. Circuit Breaker

**Question:**
Design a thread-safe generic circuit breaker `CircuitBreaker<T>` for distributed systems. It should protect a failing dependency by blocking calls after repeated failures and allowing timed recovery attempts.

Required states:
- `CLOSED`
- `OPEN`
- `HALF_OPEN`

Required API:

```java
T call(Supplier<T> supplier)
String getState()
```

Configuration:
- `failureThreshold`
- `recoveryTimeout`

### State Machine

- `CLOSED -> OPEN`
  when failures reach the threshold
- `OPEN -> HALF_OPEN`
  after the recovery timeout expires
- `HALF_OPEN -> CLOSED`
  if the probe call succeeds
- `HALF_OPEN -> OPEN`
  if the probe call fails

### Core Design

Use atomic primitives:
- `AtomicReference<State>` for the breaker state
- `AtomicInteger` for failure count
- `AtomicReference<Instant>` for last failure time

This avoids a coarse synchronized lock around the entire call path.

### Reference Java Implementation

```java
import java.time.Duration;
import java.time.Instant;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;
import java.util.function.Supplier;

public class CircuitBreaker<T> {

    private enum State {
        CLOSED, OPEN, HALF_OPEN
    }

    private final AtomicReference<State> state = new AtomicReference<>(State.CLOSED);
    private final AtomicInteger failureCount = new AtomicInteger(0);
    private final AtomicReference<Instant> lastFailureTime = new AtomicReference<>(null);
    private final AtomicBoolean probeInFlight = new AtomicBoolean(false);

    private final int failureThreshold;
    private final Duration recoveryTimeout;

    public CircuitBreaker(int failureThreshold, Duration recoveryTimeout) {
        this.failureThreshold = failureThreshold;
        this.recoveryTimeout = recoveryTimeout;
    }

    public T call(Supplier<T> supplier) throws Exception {
        State current = state.get();

        // OPEN: block until the recovery timeout elapses, then let exactly one
        // caller flip the breaker to HALF_OPEN to probe the dependency.
        if (current == State.OPEN) {
            Instant lastFailure = lastFailureTime.get();
            boolean timedOut = lastFailure != null
                    && Duration.between(lastFailure, Instant.now()).compareTo(recoveryTimeout) > 0;
            if (!timedOut) {
                throw new RuntimeException("Circuit is OPEN. Request blocked.");
            }
            state.compareAndSet(State.OPEN, State.HALF_OPEN);  // one winner; others see HALF_OPEN
            current = state.get();
        }

        // HALF_OPEN: admit only ONE probe at a time. Without this gate, every
        // concurrent caller would slip through once the state flips to HALF_OPEN
        // and flood the still-recovering dependency.
        if (current == State.HALF_OPEN) {
            if (!probeInFlight.compareAndSet(false, true)) {
                throw new RuntimeException("Circuit is HALF_OPEN. Probe already in progress.");
            }
            return runProbe(supplier);
        }

        // CLOSED: normal path.
        if (current == State.CLOSED) {
            try {
                T result = supplier.get();
                onSuccess();
                return result;
            } catch (Exception e) {
                onFailure();
                throw e;
            }
        }

        // State changed concurrently (e.g. a probe just reopened it): block.
        throw new RuntimeException("Circuit is OPEN. Request blocked.");
    }

    private T runProbe(Supplier<T> supplier) throws Exception {
        try {
            T result = supplier.get();
            onSuccess();   // HALF_OPEN -> CLOSED
            return result;
        } catch (Exception e) {
            onFailure();   // HALF_OPEN -> OPEN
            throw e;
        } finally {
            probeInFlight.set(false);  // release the single probe permit
        }
    }

    private void onSuccess() {
        failureCount.set(0);
        if (state.get() == State.HALF_OPEN) {
            state.compareAndSet(State.HALF_OPEN, State.CLOSED);
        }
    }

    private void onFailure() {
        int failures = failureCount.incrementAndGet();
        lastFailureTime.set(Instant.now());

        if (failures >= failureThreshold) {
            state.set(State.OPEN);
        } else if (state.get() == State.HALF_OPEN) {
            state.compareAndSet(State.HALF_OPEN, State.OPEN);
        }
    }

    public String getState() {
        return state.get().name();
    }
}
```

### Why It Works

**CLOSED**
- requests are allowed through
- successes reset the failure counter
- failures increment the counter

**OPEN**
- requests are blocked immediately
- after `recoveryTimeout`, one thread can move the breaker to `HALF_OPEN`

**HALF_OPEN**
- a recovery probe is allowed
- success closes the circuit
- failure reopens it

### Complexity

- Time: `O(1)` per call
- Space: `O(1)`

### Thread-Safety Notes

Atomic operations are preferred here because:
- they avoid serializing all callers on one monitor
- state transitions are small and well-scoped
- compare-and-set allows safe race handling during `OPEN -> HALF_OPEN`

Compared with `synchronized`, atomics usually scale better under contention, though they make the implementation slightly harder to reason about.

### Example Behavior

**Open after threshold failures**

```java
CircuitBreaker<String> breaker = new CircuitBreaker<>(3, Duration.ofSeconds(5));
```

After 3 failed calls:
- breaker state becomes `OPEN`
- the next call is rejected immediately

**Recover after timeout**

If timeout expires, the next caller attempts:
- `OPEN -> HALF_OPEN`
- probe request
- success => `CLOSED`
- failure => `OPEN`

### Unit Test Ideas

- successful call keeps state `CLOSED`
- repeated failures open the circuit
- open circuit rejects calls before timeout
- successful half-open probe closes the circuit
- failed half-open probe reopens the circuit
- success resets failure count
- concurrent callers do not corrupt state

### Bonus 1: Metrics

A production breaker should record:
- success count
- failure count
- rejection count
- call latency
- state transitions

Example metrics interface:

```java
public interface CircuitBreakerMetrics {
    void recordSuccess(long durationMs);
    void recordFailure(long durationMs, Throwable error);
    void recordRejection();
    void recordStateTransition(String fromState, String toState);
}
```

This helps answer:
- Is the dependency healthy?
- How often are we rejecting?
- Are recoveries succeeding?

### Bonus 2: Failure Classification

Not every exception should open the circuit.

Examples:
- validation errors usually should not count
- timeouts often should count
- HTTP 5xx might count, HTTP 4xx might not

Use a predicate:

```java
import java.util.function.Predicate;

private final Predicate<Throwable> shouldCountAsFailure;
```

Then count only matching exceptions toward the threshold.

### Bonus 3: Sliding Window Breaker

Instead of counting consecutive failures, you can evaluate the last `N` requests.

Example policy:
- open if `5` of the last `10` calls failed

This is more robust for noisy systems because one success does not erase the entire failure trend.

Typical implementation:
- fixed-size ring buffer
- each slot stores success/failure
- evaluate failure rate after each call

Tradeoff:
- more accurate
- more memory and more logic than a simple consecutive-failure counter

### Bonus 4: Fallback Strategy

When the breaker is open, common fallback options are:
- return cached data
- return a default value
- call a secondary service
- degrade functionality gracefully

This can be wrapped on top of the breaker rather than embedded directly in the core state machine.

### Interview Summary

The essential design is a three-state machine guarded by atomic state transitions. In `CLOSED`, calls flow normally and failures are counted. In `OPEN`, calls are rejected until the timeout expires. In `HALF_OPEN`, a probe determines whether the dependency has recovered. For production use, add metrics, exception classification, and possibly a sliding-window policy or fallback behavior.

---

## 20. Cheapest Book Purchase Service

**Question:**
Design a service that helps customers buy a book at the cheapest available price from external sellers.

Customer request contains:
- `bookId`
- `maxPrice`
- payment information

System behavior:
- query roughly `50-200` seller APIs for quotes
- find the cheapest valid offer
- if cheapest price `<= maxPrice`, purchase the book
- otherwise return the cheapest quote as a suggestion

Constraints:
- `1-2 million` books
- `50-200` sellers today, possibly much more later
- end-to-end SLA around `10-20 seconds`
- asynchronous request model is acceptable

### Clarifying Questions

- Is the user flow synchronous or asynchronous?
- Do we need to return the actual purchase result immediately?
- Can sellers reserve inventory, or only quote and purchase?
- Are seller rate limits known and different per seller?
- Should we charge the customer before or after seller confirmation?
- Do quotes expire?
- How should we handle duplicate requests for the same book?

### Core Requirements

**Functional Requirements:**
- accept purchase-intent requests
- query many sellers for price and availability
- sort quotes by price
- attempt purchase from cheapest to more expensive sellers
- stop once one purchase succeeds
- return a suggestion if all acceptable offers exceed `maxPrice`
- safely handle payment authorization and capture

**Non-Functional Requirements:**
- high availability
- bounded latency within `10-20 seconds`
- protect seller APIs from overload
- tolerate slow or broken sellers
- provide observability for quote quality and purchase success

### Recommended User Flow

Use an **asynchronous API**:

1. user submits purchase request
2. service validates request and authorizes payment hold
3. service stores request and returns `requestId`
4. background workers fetch quotes and attempt purchase
5. final result is available via polling, webhook, or notification

This is safer than forcing the user to wait on a long HTTP request and gives more control over retries and backpressure.

### Public API

**Create purchase request**

```http
POST /api/v1/purchase-requests
```

Request:

```json
{
  "bookId": "isbn-123456",
  "maxPrice": 29.99,
  "paymentInfo": {
    "cardToken": "tok_abc123"
  }
}
```

Response:

```json
{
  "requestId": "req-xyz",
  "status": "pending"
}
```

**Get request status**

```http
GET /api/v1/purchase-requests/{requestId}
```

Possible statuses:
- `pending`
- `purchased`
- `price_not_met`
- `failed`

### High-Level Architecture

```text
Client
  -> API Gateway
  -> Purchase Request Service
     -> DB + payment authorization
     -> Message Queue

Workers
  -> Price Aggregator
     -> Seller adapters / quote clients
  -> Purchase Orchestrator
     -> cheapest-first purchase attempts
  -> Notification Service

Supporting systems
  -> PostgreSQL
  -> Redis cache
  -> Kafka / SQS
  -> Metrics / tracing / alerting
```

### Main Components

**Purchase Request Service**
- validate request
- authenticate customer
- tokenize or validate payment token
- place an authorization hold instead of capturing funds
- persist request
- enqueue work

**Price Aggregator**
- fetch candidate sellers
- issue quote requests in parallel
- collect successful responses within timeout
- store quotes
- sort by price and availability

**Purchase Orchestrator**
- iterate sellers from cheapest to most expensive
- attempt purchase
- if cheapest seller is sold out, try next seller
- on success, capture payment
- on failure, release authorization hold

**Notification / Status Service**
- expose request status
- push result via email, webhook, or app notification

### Quote Fan-Out Strategy

Sequential calls are too slow.

Use:
- async I/O
- bounded parallelism
- seller-specific timeouts
- seller-specific rate limits

A practical strategy:
- send quote requests in parallel to all candidate sellers
- wait up to a bounded deadline such as `2-3 seconds`
- process partial results rather than waiting forever

This fits the SLA and avoids one slow seller blocking the whole request.

### Seller Protection

Each seller should have:
- rate limiting
- circuit breaker
- retry policy with exponential backoff and jitter
- concurrency cap

This prevents:
- overloading a seller
- cascading failures from unhealthy sellers
- self-inflicted latency spikes

If a seller is consistently failing:
- open its circuit
- skip it temporarily
- probe again later using half-open logic

### Data Model

Important tables:
- `purchase_requests`
- `seller_quotes`
- `purchase_attempts`
- seller health / rate-limit metadata

Typical fields:

**purchase_requests**
- request id
- customer id
- book id
- max price
- request status
- timestamps

**seller_quotes**
- request id
- seller id
- price
- availability
- quote id
- quote expiry
- latency
- error details

**purchase_attempts**
- request id
- seller id
- attempt order
- status
- timestamps

SQL is a good fit here because:
- purchases and payment states need transactional consistency
- request and attempt state transitions are relational and auditable

### Payment Strategy

Do **authorize first, capture later**:

1. authorize card for `maxPrice` or reasonable reserve amount
2. find a seller and complete purchase attempt
3. capture actual amount only after seller confirms
4. void or release the hold if purchase fails

This avoids refund-heavy flows and keeps money handling cleaner.

### Handling Inventory Races

The quote may say "available," but the book may be gone by purchase time.

Best approach:
- keep sorted quotes
- try purchase in cheapest-first order
- if seller returns out-of-stock or quote-expired, move to next quote

If seller APIs support reservation, even better:
- reserve item briefly after quote
- then finalize purchase

### Caching

Cache quotes briefly in Redis:

```text
quote:{bookId}:{sellerId}
```

TTL can be short, such as `30-60 seconds`, depending on how quickly seller prices change.

Benefits:
- lower seller QPS
- faster repeat lookups
- better performance for popular books

Be careful:
- cached quotes can become stale
- purchase step must still tolerate price or inventory drift

### Coalescing Duplicate Requests

If many users request the same book at the same time:
- deduplicate in-flight quote work
- share the same quote fan-out result

This avoids repeatedly querying all sellers for identical work.

You can implement this with:
- an in-flight map keyed by `bookId`
- short-lived request coalescing windows

### Scaling Beyond 200 Sellers

At `10,000` sellers, querying everyone is wasteful.

Use seller selection:
- top sellers by historical availability
- category affinity
- geography
- price competitiveness
- ML ranking if scale justifies it

Then:
- query top `N` likely sellers first
- expand only if needed

This reduces both cost and latency.

### Reliability and Failure Handling

Common failure modes:
- seller timeout
- seller 5xx errors
- stale quote
- payment provider outage
- DB / queue degradation

Mitigations:
- queue-based async workflow
- retries with jitter
- circuit breakers
- idempotency keys for purchase attempts
- transactional request state updates
- dead-letter queue for unrecoverable work

### Observability

Track:
- quote latency by seller
- quote success rate
- purchase success rate
- seller failure and timeout rate
- cache hit rate
- number of fallback-to-next-seller cases
- authorization vs capture success rate
- end-to-end request completion latency

Alert on:
- rising seller timeout rates
- circuit breakers opening frequently
- growing queue backlog
- falling purchase conversion

### Good Interview Tradeoffs

**Async vs sync**
- async is better given a `10-20 second` SLA and external dependencies

**Ask all sellers vs subset**
- all sellers is fine at `50-200`
- ranking becomes necessary at larger scale

**Wait for all responses vs deadline**
- use a deadline and accept partial responses
- the slowest seller should not define the whole request

**Immediate charge vs delayed capture**
- hold first, capture after confirmed seller purchase

### Interview Summary

Use an asynchronous purchase-request workflow backed by a queue. Fan out quote requests to sellers in parallel with per-seller rate limiting, timeouts, and circuit breakers. Store and sort quotes, then attempt purchase from cheapest to most expensive seller until one succeeds. Use authorization-before-capture for payments, short-lived quote caching, and strong observability around seller health, queue depth, and purchase outcomes.

---

## 21. Revenue System With Direct Referrals

**Question:**
Design a `RevenueSystem` with these operations:

- `insert(revenue) -> customer_id`
- `insert(revenue, referrer_id) -> customer_id`
- `get_lowest_k_by_total_revenue(k, min_total_revenue) -> Set[int]`

Revenue rule:

```text
total_revenue(customer) =
    personal_revenue + sum(revenue of direct referrals only)
```

Important:
- only direct referrals count
- grandchildren do not contribute in the base version
- results should be ordered by:
  1. total revenue ascending
  2. customer id ascending

### Core Data Model

```python
class Customer:
    def __init__(self, customer_id: int, revenue: int):
        self.id = customer_id
        self.revenue = revenue
        self.total_revenue = revenue
        self.referrals = []
        self.parent = None   # referrer id; None for a top-level customer
```

### Approach 1: Hash Map + Sort on Query

Best when inserts are frequent and queries are relatively rare.

```python
class Customer:
    def __init__(self, customer_id: int, revenue: int):
        self.id = customer_id
        self.revenue = revenue
        self.total_revenue = revenue
        self.referrals = []


class RevenueSystem:
    def __init__(self):
        self.customers = {}
        self.next_id = 0

    def insert(self, revenue: int, referrer_id: int = None) -> int:
        customer_id = self.next_id
        self.next_id += 1

        customer = Customer(customer_id, revenue)
        self.customers[customer_id] = customer

        if referrer_id is not None:
            referrer = self.customers[referrer_id]
            referrer.total_revenue += revenue
            referrer.referrals.append(customer_id)

        return customer_id

    def get_lowest_k_by_total_revenue(self, k: int, min_total_revenue: int) -> set[int]:
        eligible = [
            customer
            for customer in self.customers.values()
            if customer.total_revenue >= min_total_revenue
        ]

        eligible.sort(key=lambda c: (c.total_revenue, c.id))
        return set(customer.id for customer in eligible[:k])
```

**Complexity:**
- `insert`: `O(1)`
- `get_lowest_k_by_total_revenue`: `O(n log n)`
- Space: `O(n)`

### Approach 2: Maintain a Sorted Set

Best when read queries are frequent.

Use:
- hash map for direct lookup
- balanced tree or sorted list keyed by `(total_revenue, id)`

Python version using `SortedList`:

```python
from sortedcontainers import SortedList


class Customer:
    def __init__(self, customer_id: int, revenue: int):
        self.id = customer_id
        self.revenue = revenue
        self.total_revenue = revenue
        self.referrals = []

    def __lt__(self, other):
        if self.total_revenue != other.total_revenue:
            return self.total_revenue < other.total_revenue
        return self.id < other.id

    def __eq__(self, other):
        return self.id == other.id

    def __hash__(self):
        return hash(self.id)


class RevenueSystem:
    def __init__(self):
        self.customers = {}
        self.sorted_customers = SortedList()
        self.next_id = 0

    def insert(self, revenue: int, referrer_id: int = None) -> int:
        customer_id = self.next_id
        self.next_id += 1

        customer = Customer(customer_id, revenue)
        self.customers[customer_id] = customer
        self.sorted_customers.add(customer)

        if referrer_id is not None:
            referrer = self.customers[referrer_id]
            self.sorted_customers.remove(referrer)
            referrer.total_revenue += revenue
            referrer.referrals.append(customer_id)
            self.sorted_customers.add(referrer)

        return customer_id

    def get_lowest_k_by_total_revenue(self, k: int, min_total_revenue: int) -> set[int]:
        dummy = Customer(-1, min_total_revenue)
        dummy.total_revenue = min_total_revenue
        start = self.sorted_customers.bisect_left(dummy)

        result = set()
        for i in range(start, len(self.sorted_customers)):
            if len(result) == k:
                break
            result.add(self.sorted_customers[i].id)
        return result
```

**Complexity:**
- `insert`: `O(log n)`
- query: `O(log n + k)` with binary search plus scan
- Space: `O(n)`

### Approach 3: Lazy Sorting

Best when writes come in bursts and reads are occasional.

```python
class RevenueSystem:
    def __init__(self):
        self.customers = {}
        self.next_id = 0
        self.sorted_cache = None

    def insert(self, revenue: int, referrer_id: int = None) -> int:
        customer_id = self.next_id
        self.next_id += 1

        customer = Customer(customer_id, revenue)
        self.customers[customer_id] = customer

        if referrer_id is not None:
            referrer = self.customers[referrer_id]
            referrer.total_revenue += revenue
            referrer.referrals.append(customer_id)

        self.sorted_cache = None
        return customer_id

    def get_lowest_k_by_total_revenue(self, k: int, min_total_revenue: int) -> set[int]:
        if self.sorted_cache is None:
            self.sorted_cache = sorted(
                self.customers.values(),
                key=lambda c: (c.total_revenue, c.id),
            )

        result = set()
        for customer in self.sorted_cache:
            if customer.total_revenue >= min_total_revenue:
                result.add(customer.id)
                if len(result) == k:
                    break
        return result
```

### Example Walkthrough

```python
system = RevenueSystem()

id0 = system.insert(300)       # total(0) = 300
id1 = system.insert(200)       # total(1) = 200
id2 = system.insert(100, 0)    # total(2) = 100, total(0) = 400
id3 = system.insert(150, 0)    # total(3) = 150, total(0) = 550
id4 = system.insert(50, 2)     # total(4) = 50, total(2) = 150
```

Totals:
- customer `0` -> `550`
- customer `1` -> `200`
- customer `2` -> `150`
- customer `3` -> `150`
- customer `4` -> `50`

Query:

```python
system.get_lowest_k_by_total_revenue(3, 100)
```

Eligible ordered list:
- `(150, id=2)`
- `(150, id=3)`
- `(200, id=1)`
- `(550, id=0)`

Result IDs:

```python
{2, 3, 1}
```

### Tradeoffs

**Hash map + sort**
- simplest
- best for many writes and few reads

**Sorted set**
- more complex
- best for frequent queries

**Lazy cache**
- good compromise when writes are bursty

### Follow-Up 1: Real-Time Top-K

If the interviewer wants top or bottom K maintained continuously:
- use a balanced tree
- or use a heap plus index tracking

The difficult part is that referrer revenue updates mean existing entries move when new customers are inserted.

### Follow-Up 2: Multi-Level Referrals

If indirect referrals count too, there are two main designs:

**Compute on demand**
- DFS or BFS through descendants
- simple
- expensive query time

**Bubble updates upward**
- when inserting a referred customer, walk up the parent chain
- fast queries
- slower inserts

```python
def insert(self, revenue: int, referrer_id: int = None) -> int:
    customer_id = self.next_id
    self.next_id += 1

    customer = Customer(customer_id, revenue)
    customer.parent = referrer_id   # record the referrer so we can walk upward
    self.customers[customer_id] = customer

    if referrer_id is not None:
        # Bubble this revenue up the entire referral chain
        current_id = referrer_id
        while current_id is not None:
            self.customers[current_id].total_revenue += revenue
            current_id = self.customers[current_id].parent

    return customer_id
```

### Follow-Up 3: Referral Levels

Return descendants grouped by generation using BFS:

```python
from collections import deque


def get_referral_levels(self, customer_id: int) -> list[list[int]]:
    if customer_id not in self.customers:
        return []

    result = []
    queue = deque([customer_id])

    while queue:
        level = []
        for _ in range(len(queue)):
            current_id = queue.popleft()
            customer = self.customers[current_id]

            if current_id != customer_id:
                level.append(current_id)

            for child_id in customer.referrals:
                queue.append(child_id)

        if level:
            result.append(level)

    return result
```

### Follow-Up 4: Revenue Up To N Levels Deep

Use BFS or DFS with depth tracking:

```python
from collections import deque


def get_revenue_n_levels(self, customer_id: int, n: int) -> int:
    if customer_id not in self.customers:
        return 0

    total = 0
    queue = deque([(customer_id, 0)])

    while queue:
        current_id, level = queue.popleft()
        customer = self.customers[current_id]
        total += customer.revenue

        if level < n:
            for child_id in customer.referrals:
                queue.append((child_id, level + 1))

    return total
```

### Interview Summary

This problem is mainly about choosing the right tradeoff between write speed and query speed. If inserts dominate, use a hash map and sort at query time. If reads dominate, maintain a sorted structure keyed by `(total_revenue, id)`. The core business rule is simple: direct referrals immediately increase the referrer's total revenue, but not the revenue of higher ancestors in the base version.

---

## 22. Merge Graph

*From interview-experience notes ([coding](https://garnet-bluebell-0e5.notion.site/coding-3403f77c7145801c90dfd948bebbd4e0)). The source's "Remove Covered Point" and "Find Dependency Bottleneck" duplicate [#11](#11-delete-index-from-interval-list) and [#9](#9-bottleneck-nodes-in-a-dag) and are not repeated.*

**Problem:** You are given `N` already-connected graphs. Add new edges to merge them into a single connected graph using the **minimum** number of edges, choosing the new connections **randomly**. A `random(n)` primitive returns an int in `[0, n)`. Return the list of new edges.

**Key insight:** with `N` connected components, the minimum to connect them into one is exactly `N - 1` edges (this is not MST — there are no weights). So the task is: pick one representative node per component, then connect the components with `N - 1` edges. If the interviewer wants a *uniform* distribution over all ways to connect the components, sample a uniform random spanning tree of the `N` components via a **Prüfer sequence**.

```python
import random, heapq
from typing import List, Dict, Set, Tuple

Graph = Dict[int, Set[int]]
Edge = Tuple[int, int]

def merge_graphs_simple(graphs: List[Graph]) -> List[Edge]:
    """Connect N components with N-1 new edges. Random, but NOT uniform
    over all labeled trees of the components."""
    if len(graphs) <= 1:
        return []
    reps = [random.choice(list(g.keys())) for g in graphs]  # one node per component
    random.shuffle(reps)
    return [(reps[i - 1], reps[i]) for i in range(1, len(reps))]  # random chain


def merge_graphs_uniform(graphs: List[Graph]) -> List[Edge]:
    """Uniformly random spanning tree over the N components via a Prufer
    sequence (each of the N^(N-2) labeled trees is equally likely)."""
    n = len(graphs)
    if n <= 1:
        return []
    if n == 2:
        return [(random.choice(list(graphs[0])), random.choice(list(graphs[1])))]

    # 1. sample a Prufer sequence of length n-2 over component indices
    prufer = [random.randrange(n) for _ in range(n - 2)]

    # 2. decode it into n-1 component-level tree edges
    degree = [1] * n
    for x in prufer:
        degree[x] += 1
    leaves = [i for i in range(n) if degree[i] == 1]
    heapq.heapify(leaves)

    comp_edges = []
    for x in prufer:
        leaf = heapq.heappop(leaves)
        comp_edges.append((leaf, x))
        degree[leaf] -= 1
        degree[x] -= 1
        if degree[x] == 1:
            heapq.heappush(leaves, x)
    comp_edges.append((heapq.heappop(leaves), heapq.heappop(leaves)))

    # 3. realize each component edge as an edge between random nodes
    nodes = [list(g.keys()) for g in graphs]
    return [(random.choice(nodes[i]), random.choice(nodes[j])) for i, j in comp_edges]
```

**Complexity:** simple `O(V + N)`; uniform `O(V + N log N)`. The caller adds each returned edge to both endpoints' adjacency sets.

---

# System Programming

*From interview-experience notes ([System Programming](https://garnet-bluebell-0e5.notion.site/System-Programming-3763f77c7145801b8fc3ded3f95f7ab0)). Single-machine concurrency design problems; the interview usually starts single-threaded and then adds thread-safety. Solutions below are clean reference implementations in Python.*

## 23. Threading Primitives (Java & Python)

A quick reference of the building blocks interviewers expect you to name.

**Java:**
```java
// Bounded thread pool (prefer over `new Thread` per task)
ExecutorService pool = Executors.newFixedThreadPool(8);
Future<Integer> f = pool.submit(() -> compute());

synchronized (lock) { /* critical section */ }   // mutual exclusion
ReentrantLock lk = new ReentrantLock();           // more flexible lock
lk.lock();
try { /* ... */ } finally { lk.unlock(); }        // ALWAYS unlock in finally

BlockingQueue<Task> q = new ArrayBlockingQueue<>(1000);  // bounded producer-consumer
q.put(task);            // blocks if full (use a bounded queue for backpressure)
Task t = q.take();      // blocks if empty

Semaphore sem = new Semaphore(10);                // cap concurrent access
AtomicInteger counter = new AtomicInteger();      // lock-free counting
```

**Python** — mind the GIL: threads help **I/O-bound** work, not CPU-bound; use `multiprocessing` for CPU-bound.
```python
import threading, queue

t = threading.Thread(target=work, args=(path,)); t.start(); t.join()

lock = threading.Lock()
with lock:            # protect shared mutable state
    ...

q = queue.Queue()     # thread-safe producer-consumer
q.put(item); item = q.get()

cond = threading.Condition()
with cond:
    cond.wait_for(lambda: ready)   # sleep until shared state changes
    cond.notify_all()

sem = threading.Semaphore(10)      # cap concurrency
```
One-liner: *"In Python I use `threading` for I/O-bound concurrency, `Lock` to protect shared state, and `queue.Queue` for producer-consumer; for CPU-bound work I use `multiprocessing` because of the GIL."*

## 24. Command Execution System

**Problem:** `submit(files, limit)` returns immediately with a command id; a background system reads lines from the given text files and returns **at most `limit` total lines**, preserving file order. Clarify: `limit` is total (not per file); strict order required; minimize over-read.

```python
import threading, queue, uuid

class CommandSystem:
    def __init__(self, num_workers=4):
        self.status = {}                       # cmd_id -> "pending"|"done"
        self.results = {}                      # cmd_id -> list[str]
        self.lock = threading.Lock()
        self.jobs = queue.Queue()
        for _ in range(num_workers):
            threading.Thread(target=self._worker, daemon=True).start()

    def submit(self, files, limit) -> str:
        cmd_id = str(uuid.uuid4())
        with self.lock:
            self.status[cmd_id] = "pending"
        self.jobs.put((cmd_id, files, limit))
        return cmd_id                          # returns immediately

    def get_status(self, cmd_id):
        with self.lock:
            return self.status.get(cmd_id)

    def get_result(self, cmd_id):
        with self.lock:
            return self.results.get(cmd_id)        # None until status == "done"

    def _worker(self):
        while True:
            cmd_id, files, limit = self.jobs.get()
            lines = self._read_in_order(files, limit)
            with self.lock:
                self.results[cmd_id] = lines
                self.status[cmd_id] = "done"
            self.jobs.task_done()

    def _read_in_order(self, files, limit):
        """Total `limit` across files, preserving file order, stopping early."""
        out = []
        for path in files:
            if len(out) >= limit:
                break
            with open(path) as f:
                for line in f:
                    out.append(line.rstrip("\n"))
                    if len(out) >= limit:
                        break
        return out[:limit]
```

**Java:**
```java
import java.io.*;
import java.nio.file.*;
import java.util.*;
import java.util.concurrent.*;

class CommandSystem {
    enum Status { PENDING, DONE }
    private final Map<String, Status> status = new ConcurrentHashMap<>();
    private final Map<String, List<String>> results = new ConcurrentHashMap<>();
    private final ExecutorService pool;

    CommandSystem(int numWorkers) {
        this.pool = Executors.newFixedThreadPool(numWorkers);
    }

    String submit(List<String> files, int limit) {
        String cmdId = UUID.randomUUID().toString();
        status.put(cmdId, Status.PENDING);
        pool.submit(() -> {                         // background; returns immediately
            results.put(cmdId, readInOrder(files, limit));
            status.put(cmdId, Status.DONE);
        });
        return cmdId;
    }

    Status getStatus(String cmdId) { return status.get(cmdId); }
    List<String> getResult(String cmdId) { return results.get(cmdId); }  // null until DONE

    private List<String> readInOrder(List<String> files, int limit) {
        List<String> out = new ArrayList<>();
        for (String path : files) {
            if (out.size() >= limit) break;
            try (BufferedReader br = Files.newBufferedReader(Path.of(path))) {
                String line;
                while (out.size() < limit && (line = br.readLine()) != null) {
                    out.add(line);
                }
            } catch (IOException e) {
                throw new UncheckedIOException(e);
            }
        }
        return out;
    }
}
```

**Tradeoff:** sequential reading guarantees order + zero over-read but limits parallelism (you don't know how many lines earlier files contribute until you read them). To parallelize, read files into per-file buffers concurrently and merge in original order — at the cost of possible over-read, which you bound with a worker pool or small read windows.

## 25. Log Writer (Durable, Concurrent)

**Problem:** many producer threads call `write(record)`; the call must not return until the record is **durably persisted** (survives process crash *and* machine restart — not just a memory buffer), and records must not be corrupted or interleaved. Clarify ordering (global vs per-producer).

**Solution — group commit:** producers enqueue and block on a per-record event; a single writer thread drains the queue, frames each record, appends, calls `fsync` **once per batch**, then releases all waiters. (Same design as [#18 Durable Concurrent Data Writer](#18-durable-concurrent-data-writer).)

```python
import threading, queue, os, struct, zlib

class LogWriter:
    def __init__(self, path):
        self.f = open(path, "ab", buffering=0)
        self.q = queue.Queue()
        threading.Thread(target=self._run, daemon=True).start()

    def write(self, record: bytes) -> None:
        done = threading.Event()
        self.q.put((record, done))
        done.wait()                       # block until durably persisted

    def _run(self):
        while True:
            batch = [self.q.get()]
            try:                          # drain whatever else is queued
                while True:
                    batch.append(self.q.get_nowait())
            except queue.Empty:
                pass
            for record, _ in batch:
                self.f.write(self._frame(record))
            self.f.flush()
            os.fsync(self.f.fileno())     # ONE fsync for the whole batch
            for _, done in batch:
                done.set()                # only now release the producers

    @staticmethod
    def _frame(record: bytes) -> bytes:
        # [len:4][payload][crc32:4] -> detects torn/corrupt tail on recovery
        return struct.pack(">I", len(record)) + record + struct.pack(">I", zlib.crc32(record))
```

**Java** (group commit — the same design as [#18](#18-durable-concurrent-data-writer)):
```java
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.channels.FileChannel;
import java.nio.file.*;
import java.util.*;
import java.util.concurrent.*;
import java.util.zip.CRC32;

class LogWriter {
    private final FileChannel channel;
    private final BlockingQueue<Req> queue = new LinkedBlockingQueue<>();
    private volatile boolean running = true;

    private static final class Req {
        final byte[] data;
        final CountDownLatch done = new CountDownLatch(1);
        volatile IOException error;
        Req(byte[] data) { this.data = data; }
    }

    LogWriter(String path) throws IOException {
        this.channel = FileChannel.open(Path.of(path),
            StandardOpenOption.CREATE, StandardOpenOption.WRITE, StandardOpenOption.APPEND);
        Thread t = new Thread(this::run, "log-writer");
        t.setDaemon(true);
        t.start();
    }

    void write(byte[] record) throws IOException, InterruptedException {
        Req req = new Req(record);
        queue.put(req);
        req.done.await();                         // block until durably persisted
        if (req.error != null) throw req.error;
    }

    private void run() {
        List<Req> batch = new ArrayList<>();
        while (running) {
            try {
                batch.clear();
                Req first = queue.poll(1, TimeUnit.MILLISECONDS);
                if (first == null) continue;
                batch.add(first);
                queue.drainTo(batch);             // group commit: take everything queued
                for (Req r : batch) {
                    ByteBuffer buf = frame(r.data);
                    while (buf.hasRemaining()) channel.write(buf);
                }
                channel.force(true);              // ONE fsync for the whole batch
                for (Req r : batch) r.done.countDown();
            } catch (Exception e) {
                for (Req r : batch) {             // fail the in-flight batch, don't hang producers
                    r.error = (e instanceof IOException io) ? io : new IOException(e);
                    r.done.countDown();
                }
            }
        }
    }

    private static ByteBuffer frame(byte[] data) {
        CRC32 crc = new CRC32();
        crc.update(data);
        ByteBuffer buf = ByteBuffer.allocate(8 + data.length);  // [len:4][payload][crc:4]
        buf.putInt(data.length).put(data).putInt((int) crc.getValue());
        buf.flip();
        return buf;
    }
}
```

**Recovery:** scan from the start, validate each record's length + CRC, truncate at the first bad/partial frame. Batching amortizes the expensive `fsync` across many writes while keeping the durability contract.

## 26. Job Scheduler (DAG)

**Problem:** schedule tasks with dependencies (a DAG); run a task only after all upstreams finish; use a worker pool. Clarify: DAG (no cycles)? need status/progress?

**Solution:** Kahn's topological order with a thread-safe ready queue. Indegree-0 tasks are ready; on completion, decrement children's indegree and enqueue the newly-ready ones.

```python
import threading, queue
from collections import defaultdict

def run_dag(task_ids, dependencies, run, num_workers=4):
    children = defaultdict(list)
    indegree = defaultdict(int)
    for before, after in dependencies:
        children[before].append(after)
        indegree[after] += 1

    ready = queue.Queue()
    for t in task_ids:
        if indegree[t] == 0:
            ready.put(t)

    lock = threading.Lock()
    remaining = len(task_ids)
    done_event = threading.Event()

    def worker():
        nonlocal remaining
        while True:
            task_id = ready.get()
            run(task_id)                          # execute the task
            with lock:
                for child in children[task_id]:
                    indegree[child] -= 1
                    if indegree[child] == 0:
                        ready.put(child)
                remaining -= 1
                if remaining == 0:
                    done_event.set()
            ready.task_done()

    for _ in range(num_workers):
        threading.Thread(target=worker, daemon=True).start()
    done_event.wait()
```

**Java:**
```java
import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.Consumer;

class DagScheduler {
    void run(List<Integer> taskIds, int[][] deps, Consumer<Integer> run, int numWorkers)
            throws InterruptedException {
        Map<Integer, List<Integer>> children = new HashMap<>();
        Map<Integer, Integer> indegree = new HashMap<>();
        for (int t : taskIds) indegree.put(t, 0);
        for (int[] e : deps) {                                  // e = {before, after}
            children.computeIfAbsent(e[0], k -> new ArrayList<>()).add(e[1]);
            indegree.merge(e[1], 1, Integer::sum);
        }

        BlockingQueue<Integer> ready = new LinkedBlockingQueue<>();
        for (int t : taskIds) if (indegree.get(t) == 0) ready.add(t);

        AtomicInteger remaining = new AtomicInteger(taskIds.size());
        CountDownLatch done = new CountDownLatch(1);
        if (remaining.get() == 0) done.countDown();   // empty DAG -> nothing to run
        Object lock = new Object();
        ExecutorService pool = Executors.newFixedThreadPool(numWorkers);

        for (int i = 0; i < numWorkers; i++) {
            pool.submit(() -> {
                try {
                    while (true) {
                        int task = ready.take();
                        run.accept(task);                      // execute the task
                        synchronized (lock) {
                            for (int child : children.getOrDefault(task, List.of())) {
                                if (indegree.merge(child, -1, Integer::sum) == 0) ready.add(child);
                            }
                        }
                        if (remaining.decrementAndGet() == 0) done.countDown();
                    }
                } catch (InterruptedException ignored) {
                    Thread.currentThread().interrupt();        // shutdownNow stops idle workers
                }
            });
        }
        done.await();
        pool.shutdownNow();
    }
}
```

**Worker count:** more workers help I/O-bound tasks; CPU-bound tasks are bounded by cores (and the GIL — use processes). A cycle leaves some tasks with indegree > 0 forever (`remaining` never reaches 0) — detectable as a deadlock/timeout.

## 27. Concurrent HashMap

**Problem:** implement a thread-safe map and explain why a plain dict isn't thread-safe (a concurrent resize/rehash or read-modify-write race can corrupt state or lose updates). Show progressively better locking.

```python
import threading

# V1: one global lock — correct, but serializes everything
class GlobalLockMap:
    def __init__(self):
        self._d, self._lk = {}, threading.Lock()
    def get(self, k):
        with self._lk: return self._d.get(k)
    def put(self, k, v):
        with self._lk: self._d[k] = v

# V2: read-write lock — many readers OR one writer (best for read-heavy)

# V3: lock striping — N shards each with its own lock, so writes to
#     different shards proceed concurrently (how Java's ConcurrentHashMap
#     historically worked)
class StripedMap:
    def __init__(self, num_shards=16):
        self._shards = [{} for _ in range(num_shards)]
        self._locks = [threading.Lock() for _ in range(num_shards)]
    def _shard(self, k):
        i = hash(k) % len(self._shards)
        return self._locks[i], self._shards[i]
    def get(self, k):
        lk, d = self._shard(k)
        with lk: return d.get(k)
    def put(self, k, v):
        lk, d = self._shard(k)
        with lk: d[k] = v
```

**Java** (lock striping; in practice use `java.util.concurrent.ConcurrentHashMap`, which does this internally — striped locks historically, CAS + per-bin locking since Java 8):
```java
import java.util.*;

class StripedMap<K, V> {
    private final Object[] locks;
    private final Map<K, V>[] shards;

    @SuppressWarnings("unchecked")
    StripedMap(int numShards) {
        locks = new Object[numShards];
        shards = new HashMap[numShards];
        for (int i = 0; i < numShards; i++) {
            locks[i] = new Object();
            shards[i] = new HashMap<>();
        }
    }

    private int idx(K k) { return (k.hashCode() & 0x7fffffff) % shards.length; }

    V get(K k) {
        int i = idx(k);
        synchronized (locks[i]) { return shards[i].get(k); }
    }

    void put(K k, V v) {
        int i = idx(k);
        synchronized (locks[i]) { shards[i].put(k, v); }
    }
}
```

**Tradeoff:** global lock is simplest; striping gives concurrency proportional to the shard count with bounded memory overhead.

## 28. Durable KV Store

**Problem:** `put` / `get` / `delete`, where `put` is durable before it returns. Use a write-ahead log + in-memory map; recover by replaying the WAL.

```python
import threading, os, json

class DurableKV:
    def __init__(self, wal_path):
        self.map = {}
        self.lock = threading.Lock()
        self.wal = open(wal_path, "a+")
        self._recover()

    def _recover(self):
        self.wal.seek(0)
        for line in self.wal:                  # replay WAL from the beginning
            op = json.loads(line)
            if op["t"] == "put":
                self.map[op["k"]] = op["v"]
            elif op["t"] == "del":
                self.map.pop(op["k"], None)

    def put(self, k, v):
        with self.lock:
            self._append({"t": "put", "k": k, "v": v})
            self.map[k] = v

    def delete(self, k):
        with self.lock:
            self._append({"t": "del", "k": k})
            self.map.pop(k, None)

    def get(self, k):
        with self.lock:
            return self.map.get(k)

    def _append(self, op):
        self.wal.write(json.dumps(op) + "\n")
        self.wal.flush()
        os.fsync(self.wal.fileno())            # durable before returning
```

**Java** (read-write lock for read-heavy loads):
```java
import java.io.*;
import java.nio.file.*;
import java.util.*;
import java.util.concurrent.locks.ReentrantReadWriteLock;

class DurableKV {
    private final Map<String, String> map = new HashMap<>();
    private final ReentrantReadWriteLock rw = new ReentrantReadWriteLock();
    private final FileOutputStream wal;

    DurableKV(String walPath) throws IOException {
        recover(walPath);
        this.wal = new FileOutputStream(walPath, true);     // append
    }

    private void recover(String walPath) throws IOException {
        Path p = Path.of(walPath);
        if (!Files.exists(p)) return;
        for (String line : Files.readAllLines(p)) {          // replay WAL
            String[] f = line.split("\t", 3);                // op \t key [\t value]
            if (f[0].equals("put")) map.put(f[1], f[2]);
            else if (f[0].equals("del")) map.remove(f[1]);
        }
    }

    void put(String k, String v) throws IOException {
        rw.writeLock().lock();
        try { append("put\t" + k + "\t" + v); map.put(k, v); }
        finally { rw.writeLock().unlock(); }
    }

    void delete(String k) throws IOException {
        rw.writeLock().lock();
        try { append("del\t" + k); map.remove(k); }
        finally { rw.writeLock().unlock(); }
    }

    String get(String k) {
        rw.readLock().lock();
        try { return map.get(k); } finally { rw.readLock().unlock(); }
    }

    private void append(String record) throws IOException {
        wal.write((record + "\n").getBytes());
        wal.flush();
        wal.getFD().sync();                                  // durable before returning
    }
}
```
*(This naive WAL framing assumes keys/values contain no tab/newline; use length-prefixed or escaped records otherwise.)*

**Scaling the lock:** a read-write lock for read-heavy loads; sharded/striped locks; or a **single writer thread** that serializes WAL appends and **batches `fsync`** (higher throughput; write latency bounded by the writer). Periodically snapshot + truncate the WAL (compaction) so recovery stays fast.

## 29. Multi-Threaded Chat System (single machine)

**Problem:** users subscribe to channels; when a user publishes to a channel, all subscribers receive the message. Single machine, in-memory.

**Design:** per-channel subscriber set guarded by a **per-channel lock** (channels are independent, so no global lock); publish **enqueues** rather than broadcasting inline; a dispatcher fans out to each subscriber's **outbound queue** so one slow client can't block the channel.

```python
import threading, queue
from collections import defaultdict

class ChatServer:
    def __init__(self):
        self.subs = defaultdict(set)                 # channel -> set[user_id]
        self.locks = defaultdict(threading.Lock)     # per-channel lock
        self.outbox = defaultdict(queue.Queue)       # user_id -> outbound queue

    def subscribe(self, user_id, channel):
        with self.locks[channel]:
            self.subs[channel].add(user_id)

    def unsubscribe(self, user_id, channel):
        with self.locks[channel]:
            self.subs[channel].discard(user_id)

    def publish(self, channel, message):
        with self.locks[channel]:
            targets = list(self.subs[channel])       # snapshot to avoid mutate-while-iterate
        for user_id in targets:
            self.outbox[user_id].put(message)        # per-user queue: a slow client only backs up itself
```

**Java** (concurrent collections replace the explicit per-channel lock + snapshot):
```java
import java.util.*;
import java.util.concurrent.*;

class ChatServer {
    private final Map<String, Set<String>> subs = new ConcurrentHashMap<>();           // channel -> users
    private final Map<String, BlockingQueue<String>> outbox = new ConcurrentHashMap<>(); // user -> queue

    void subscribe(String userId, String channel) {
        subs.computeIfAbsent(channel, c -> ConcurrentHashMap.newKeySet()).add(userId);
    }

    void unsubscribe(String userId, String channel) {
        Set<String> s = subs.get(channel);
        if (s != null) s.remove(userId);
    }

    void publish(String channel, String message) {
        Set<String> targets = subs.getOrDefault(channel, Set.of());
        for (String user : targets) {                  // ConcurrentHashMap keySet is safe to iterate
            inbox(user).offer(message);                // per-user queue: a slow client only backs up itself
        }
    }

    BlockingQueue<String> inbox(String userId) {       // a per-user sender thread drains this to the socket
        return outbox.computeIfAbsent(userId, u -> new LinkedBlockingQueue<>());
    }
}
```

Each user has a sender thread draining its outbox to the socket. **Key tradeoffs:** per-channel lock (not global) so channels don't block each other; queue + dispatcher (not inline broadcast) so a slow client doesn't stall the publisher; snapshot the subscriber set so subscribe/unsubscribe can run concurrently with publish.

> **Caveat:** `defaultdict(threading.Lock)` can momentarily create two locks if two threads first-touch the same new channel concurrently (CPython's GIL usually hides it, but it isn't guaranteed). Pre-register channels, or use a small guarded `get_lock(channel)` helper, so each channel has exactly one lock.

## 30. File Block Cache

**Problem:** a `CacheFile` reads from a remote store by `(offset, length)`, with many random reads. Make client reads efficient — avoid re-fetching, and dedup concurrent fetches of the same region.

**Solution:** cache fixed-size **blocks** keyed by `(file, block_id)`. A read for `[offset, offset+length)` touches the blocks it overlaps and fetches only the missing ones. Under concurrency, an **in-flight map** ensures the first thread fetches a block while others wait on the same event, instead of all hitting the remote.

```python
import threading

BLOCK = 64 * 1024

class FileBlockCache:
    def __init__(self, remote):
        self.remote = remote
        self.cache = {}                 # (file, block_id) -> bytes  (+ LRU eviction)
        self.inflight = {}              # (file, block_id) -> Event
        self.lock = threading.Lock()

    def read(self, file, offset, length):
        out = bytearray()
        for b in range(offset // BLOCK, (offset + length - 1) // BLOCK + 1):
            data = self._get_block(file, b)
            lo = max(offset, b * BLOCK) - b * BLOCK
            hi = min(offset + length, (b + 1) * BLOCK) - b * BLOCK
            out += data[lo:hi]
        return bytes(out)

    def _get_block(self, file, b):
        key = (file, b)
        while True:
            with self.lock:
                if key in self.cache:
                    return self.cache[key]              # hit
                ev = self.inflight.get(key)
                leader = ev is None
                if leader:
                    ev = self.inflight[key] = threading.Event()
            if leader:
                try:
                    data = self.remote.fetch(file, b * BLOCK, BLOCK)  # only the leader fetches
                    with self.lock:
                        self.cache[key] = data
                    return data
                finally:
                    # ALWAYS clear in-flight + wake waiters, even if fetch raised,
                    # so a failed leader can't deadlock everyone waiting on this block.
                    with self.lock:
                        self.inflight.pop(key, None)
                    ev.set()
            else:
                ev.wait()                              # wait for the leader, then re-check
                # loop again: cache hit if the leader succeeded, else we become leader
```

**Java** (`CompletableFuture` makes the in-flight dedup clean — the leader fetches, others `join` the same future):
```java
import java.io.ByteArrayOutputStream;
import java.util.concurrent.*;

class FileBlockCache {
    interface Remote { byte[] fetch(String file, long offset, int length); }

    private static final int BLOCK = 64 * 1024;
    private final Remote remote;
    private final ConcurrentMap<String, byte[]> cache = new ConcurrentHashMap<>();             // + LRU eviction
    private final ConcurrentMap<String, CompletableFuture<byte[]>> inflight = new ConcurrentHashMap<>();

    FileBlockCache(Remote remote) { this.remote = remote; }

    byte[] read(String file, long offset, int length) {
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        for (long b = offset / BLOCK; b <= (offset + length - 1) / BLOCK; b++) {
            byte[] data = getBlock(file, b);
            int lo = (int) (Math.max(offset, b * BLOCK) - b * BLOCK);
            int hi = (int) (Math.min(offset + length, (b + 1) * BLOCK) - b * BLOCK);
            out.write(data, lo, hi - lo);
        }
        return out.toByteArray();
    }

    private byte[] getBlock(String file, long b) {
        String key = file + "#" + b;
        byte[] hit = cache.get(key);
        if (hit != null) return hit;

        CompletableFuture<byte[]> mine = new CompletableFuture<>();
        CompletableFuture<byte[]> leader = inflight.putIfAbsent(key, mine);
        if (leader != null) return leader.join();          // someone else is already fetching
        try {
            byte[] data = remote.fetch(file, b * BLOCK, BLOCK);   // only the leader fetches
            cache.put(key, data);
            mine.complete(data);
            return data;
        } catch (RuntimeException e) {
            mine.completeExceptionally(e);                 // wake waiters even on failure
            throw e;
        } finally {
            inflight.remove(key);
        }
    }
}
```

Add LRU eviction when the cache exceeds capacity. The in-flight dedup is the key concurrency win — without it, N threads needing the same hot block all fetch it remotely.

## 31. Multi-Threaded Web Crawler

**Problem:** crawl pages from a seed — fetch, parse links, continue — using multiple threads to improve I/O throughput.

**Design:** a thread-safe `visited` set (lock) and a `Queue` frontier; a fixed-size worker pool fetches different URLs concurrently (crawling is I/O-bound, so threads overlap network waits).

```python
import threading, queue

class Crawler:
    def __init__(self, fetch, parse_links, num_workers=10):
        self.fetch, self.parse_links = fetch, parse_links
        self.frontier = queue.Queue()
        self.visited = set()
        self.lock = threading.Lock()
        self.num_workers = num_workers

    def crawl(self, seeds):
        for url in seeds:
            self._enqueue(url)
        for _ in range(self.num_workers):
            threading.Thread(target=self._worker, daemon=True).start()
        self.frontier.join()                  # block until all tasks done

    def _enqueue(self, url):
        with self.lock:
            if url in self.visited:
                return
            self.visited.add(url)             # mark visited at enqueue time (dedup)
        self.frontier.put(url)

    def _worker(self):
        while True:
            url = self.frontier.get()
            try:
                html = self.fetch(url)                       # network I/O
                for link in self.parse_links(html, url):
                    self._enqueue(link)
            finally:
                self.frontier.task_done()
```

**Java** (Python's `queue.join()` has no direct equivalent — track outstanding work with an `AtomicInteger` and signal a latch when it hits zero):
```java
import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicInteger;

class Crawler {
    interface Fetcher { Iterable<String> fetchLinks(String url); }

    private final Fetcher fetcher;
    private final BlockingQueue<String> frontier = new LinkedBlockingQueue<>();
    private final Set<String> visited = ConcurrentHashMap.newKeySet();   // thread-safe set
    private final AtomicInteger outstanding = new AtomicInteger(0);
    private final CountDownLatch done = new CountDownLatch(1);
    private final ExecutorService pool;
    private final int numWorkers;

    Crawler(Fetcher fetcher, int numWorkers) {
        this.fetcher = fetcher;
        this.numWorkers = numWorkers;
        this.pool = Executors.newFixedThreadPool(numWorkers);
    }

    void crawl(List<String> seeds) throws InterruptedException {
        for (String url : seeds) enqueue(url);
        if (outstanding.get() == 0) done.countDown();   // nothing to crawl (empty/all-dup seeds)
        for (int i = 0; i < numWorkers; i++) pool.submit(this::worker);
        done.await();                       // block until all work drained
        pool.shutdownNow();
    }

    private void enqueue(String url) {
        if (visited.add(url)) {             // add() is false if already present -> dedup at enqueue
            outstanding.incrementAndGet();
            frontier.offer(url);
        }
    }

    private void worker() {
        try {
            while (true) {
                String url = frontier.take();
                try {
                    for (String link : fetcher.fetchLinks(url)) enqueue(link);  // children inc before parent dec
                } finally {
                    if (outstanding.decrementAndGet() == 0) done.countDown();
                }
            }
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }
    }
}
```

**Thread-safety:** `visited` is guarded and updated at *enqueue* time so the same URL isn't queued twice; a fixed worker pool bounds resource use. **Production follow-up:** a per-domain rate limiter (token bucket per host) so you don't overload one site — see the full distributed frontier design in `common_system_design_questions.md` §10.
