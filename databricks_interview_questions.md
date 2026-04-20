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

            if process_primary and process_secondary:
                res.append("Primary -> Secondary")
            elif process_primary:
                res.append("Primary")
            elif process_secondary:
                res.append("Secondary")
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
                res += [number] * int(count)
            elif log.startswith("BP"):
                info = log[3:-1]
                number_list = info.split(",")
                for n in number_list:
                    res.append(int(n))
        return res
```

**Note:**
The decode implementation mirrors the source file exactly, including its original string parsing behavior.

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

    private final int failureThreshold;
    private final Duration recoveryTimeout;

    public CircuitBreaker(int failureThreshold, Duration recoveryTimeout) {
        this.failureThreshold = failureThreshold;
        this.recoveryTimeout = recoveryTimeout;
    }

    public T call(Supplier<T> supplier) throws Exception {
        Instant now = Instant.now();
        State current = state.get();

        if (current == State.OPEN) {
            Instant lastFailure = lastFailureTime.get();
            if (lastFailure != null
                    && Duration.between(lastFailure, now).compareTo(recoveryTimeout) > 0) {
                if (!state.compareAndSet(State.OPEN, State.HALF_OPEN)) {
                    throw new RuntimeException("Circuit is OPEN. Request blocked.");
                }
            } else {
                throw new RuntimeException("Circuit is OPEN. Request blocked.");
            }
        }

        try {
            T result = supplier.get();
            onSuccess();
            return result;
        } catch (Exception e) {
            onFailure();
            throw e;
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
    self.customers[customer_id] = customer

    if referrer_id is not None:
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
