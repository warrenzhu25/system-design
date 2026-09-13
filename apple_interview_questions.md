# Apple Interview Questions

---

## Contents

**Coding**
1. [Two Sum & 3Sum: Hash Map and Two-Pointer Fundamentals](#1-two-sum--3sum-hash-map-and-two-pointer-fundamentals)
2. [Best Time to Buy and Sell Stock](#2-best-time-to-buy-and-sell-stock)
3. [Group Anagrams](#3-group-anagrams)
4. [Merge Overlapping Intervals](#4-merge-overlapping-intervals)
5. [Number of Islands (Grid Flood Fill)](#5-number-of-islands-grid-flood-fill)
6. [Course Schedule: Topological Sort and Cycle Detection](#6-course-schedule-topological-sort-and-cycle-detection)
7. [Clone Graph](#7-clone-graph)
8. [Linked List Fundamentals: Reverse and Merge Two Sorted Lists](#8-linked-list-fundamentals-reverse-and-merge-two-sorted-lists)
9. [Add Two Numbers: Linked List Arithmetic](#9-add-two-numbers-linked-list-arithmetic)
10. [Binary Tree Fundamentals: Same Tree and Invert Tree](#10-binary-tree-fundamentals-same-tree-and-invert-tree)
11. [Search in Rotated Sorted Array](#11-search-in-rotated-sorted-array)
12. [LRU Cache (Design)](#12-lru-cache-design)
13. [Top K Frequent Elements](#13-top-k-frequent-elements)
14. [View Hierarchy Hit-Testing (Apple UI Coordinate Query)](#14-view-hierarchy-hit-testing-apple-ui-coordinate-query)

**System Design**
15. [iCloud-Style File Sync and Conflict Resolution](#15-system-design--icloud-style-file-sync-and-conflict-resolution)
16. [Push Notification Service (APNs-Style)](#16-system-design--push-notification-service-apns-style)
17. [Typeahead Search with On-Device and Server-Side Ranking](#17-system-design--typeahead-search-with-on-device-and-server-side-ranking)
18. [App Store / Spotlight-Style Search Ranking](#18-system-design--app-store--spotlight-style-search-ranking)

**Behavioral**
19. [Behavioral Themes](#19-behavioral-themes)

---

## 1. Two Sum & 3Sum: Hash Map and Two-Pointer Fundamentals

**Problem Statement:**
Reported as one of Apple's most frequently asked warm-up problems (community trackers put "Two Sum"
among the single most-cited Apple questions). Part A: given an array `nums` and a `target`, return the
indices of the two numbers that add up to `target` (exactly one solution assumed, each element used
once). Part B ("3Sum," also commonly reported): given an array `nums`, return all unique triplets
`[a, b, c]` such that `a + b + c == 0`.

**Example:**
```
two_sum([2, 7, 11, 15], 9) -> [0, 1]
three_sum([-1, 0, 1, 2, -1, -4]) -> [[-1, -1, 2], [-1, 0, 1]]
```

**Test Cases:**

| Input | Output |
|---|---|
| `two_sum([3,2,4], 6)` | `[1,2]` |
| `two_sum([3,3], 6)` | `[0,1]` |
| `three_sum([0,0,0])` | `[[0,0,0]]` |
| `three_sum([0,1,1])` | `[]` |
| `three_sum([-2,0,1,1,2])` | `[[-2,0,2],[-2,1,1]]` |

**Key Insights:**
1. Two Sum: a single pass with a `value -> index` hash map turns an O(n²) pairwise check into O(n) —
   for each element, check whether its complement was already seen.
2. 3Sum: sort first, then fix one element and two-pointer-scan the rest — sorting turns "find a pair
   summing to `-nums[i]`" into a linear two-pointer scan instead of another hash lookup, which also makes
   duplicate-skipping straightforward (adjacent equal values after sorting).
3. Skip duplicates at three places: the outer anchor, and both inner pointers after a match — this is
   the detail interviewers most often probe when a candidate's 3Sum returns duplicate triplets.
4. Early-exit when the sorted anchor value is `> 0` — no triplet with a positive smallest element can
   sum to zero.

**Python Solution:**
```python
def two_sum(nums: list[int], target: int) -> list[int]:
    """
    Time:  O(n)
    Space: O(n)
    """
    seen: dict[int, int] = {}
    for i, num in enumerate(nums):
        complement = target - num
        if complement in seen:
            return [seen[complement], i]
        seen[num] = i
    return []


def three_sum(nums: list[int]) -> list[list[int]]:
    """
    Time:  O(n^2)
    Space: O(1) extra beyond the sort and output
    """
    nums = sorted(nums)
    n = len(nums)
    result: list[list[int]] = []

    for i in range(n):
        if i > 0 and nums[i] == nums[i - 1]:
            continue
        if nums[i] > 0:
            break

        lo, hi = i + 1, n - 1
        while lo < hi:
            total = nums[i] + nums[lo] + nums[hi]
            if total < 0:
                lo += 1
            elif total > 0:
                hi -= 1
            else:
                result.append([nums[i], nums[lo], nums[hi]])
                lo += 1
                hi -= 1
                while lo < hi and nums[lo] == nums[lo - 1]:
                    lo += 1
                while lo < hi and nums[hi] == nums[hi + 1]:
                    hi -= 1

    return result
```

**Follow-Up Questions:**
1. Return all pairs (not just one) for Two Sum, without duplicates → sort + two-pointer instead of a
   hash map, skipping duplicate values at both pointers (same technique as 3Sum's inner loop).
2. Generalize to 4Sum / kSum → recursively fix one more element and reduce to `(k-1)Sum`, bottoming out
   at the two-pointer 2Sum on a sorted array.
3. `nums` is too large to fit in memory / is a stream → Two Sum's hash-map approach still works
   online (process one pass, no need to look ahead); 3Sum's sort-then-scan does not translate to
   streaming without external sorting.

---

## 2. Best Time to Buy and Sell Stock

**Problem Statement:**
Given an array `prices` where `prices[i]` is the stock price on day `i`, find the maximum profit from
a single buy followed by a later sell. Return `0` if no profit is possible.

**Example:**
```
prices = [7, 1, 5, 3, 6, 4]
Output: 5   # buy at 1 (day 1), sell at 6 (day 4)
```

**Test Cases:**

| prices | Output |
|---|---|
| `[7,1,5,3,6,4]` | `5` |
| `[7,6,4,3,1]` | `0` |
| `[2,4,1]` | `2` |
| `[]` | `0` |
| `[5]` | `0` |

**Key Insights:**
1. Track the minimum price seen so far while scanning left to right; at each day, the best possible
   sell-today profit is `price - min_so_far`.
2. No need to consider every `(buy, sell)` pair — the optimal buy day for any sell day is always the
   minimum price *before* it, which the running minimum already captures in one pass.

**Python Solution:**
```python
def max_profit(prices: list[int]) -> int:
    """
    Time:  O(n)
    Space: O(1)
    """
    min_price = float("inf")
    best_profit = 0

    for price in prices:
        if price < min_price:
            min_price = price
        else:
            best_profit = max(best_profit, price - min_price)

    return best_profit
```

**Follow-Up Questions:**
1. Allow unlimited buy/sell transactions (not just one) → greedily sum every positive
   `prices[i+1] - prices[i]` day-over-day gain.
2. At most `k` transactions → this becomes a 2D DP over `(day, transactions used, holding stock?)`.
3. A cooldown day is required after selling before buying again → add a `cooldown` state to the DP's
   state machine (`hold`, `sold`, `rest`).

---

## 3. Group Anagrams

**Problem Statement:**
Given an array of strings `strs`, group the anagrams together (any order of groups, any order within
a group).

**Example:**
```
Input:  ["eat","tea","tan","ate","nat","bat"]
Output: [["eat","tea","ate"],["tan","nat"],["bat"]]
```

**Test Cases:**

| Input | Output (grouping) |
|---|---|
| `["eat","tea","tan","ate","nat","bat"]` | `{eat,tea,ate}, {tan,nat}, {bat}` |
| `[""]` | `{""}` |
| `["a"]` | `{"a"}` |
| `["abc","cba","bac","xyz"]` | `{abc,cba,bac}, {xyz}` |

**Key Insights:**
1. Two strings are anagrams iff their sorted character sequences are identical — use the sorted string
   as a hash-map key to bucket anagrams together in one pass.
2. For long strings/large alphabets where sorting cost matters, a fixed-size character-count tuple
   (e.g., 26 counts for lowercase English) is an O(k) alternative key instead of an O(k log k) sort.

**Python Solution:**
```python
from collections import defaultdict


def group_anagrams(strs: list[str]) -> list[list[str]]:
    """
    Time:  O(n * k log k) for n strings of max length k (sorting each string)
    Space: O(n * k)
    """
    groups: dict[str, list[str]] = defaultdict(list)
    for s in strs:
        key = "".join(sorted(s))
        groups[key].append(s)
    return list(groups.values())
```

**Follow-Up Questions:**
1. Strings can contain any Unicode character, not just lowercase letters → sorting still works as the
   key (no need to assume an alphabet size), but the fixed-count-tuple optimization no longer applies
   directly.
2. Very large `n` where the grouping itself must be distributed → shard by a hash of the sorted key
   across workers; each shard groups independently, no cross-shard coordination needed since the key
   fully determines the group.

---

## 4. Merge Overlapping Intervals

**Problem Statement:**
Given an array of intervals `[start, end]`, merge all overlapping intervals and return the resulting
non-overlapping set, sorted by start.

**Example:**
```
Input:  [[1,3],[2,6],[8,10],[15,18]]
Output: [[1,6],[8,10],[15,18]]
```

**Test Cases:**

| Input | Output |
|---|---|
| `[[1,3],[2,6],[8,10],[15,18]]` | `[[1,6],[8,10],[15,18]]` |
| `[[1,4],[4,5]]` | `[[1,5]]` (touching counts as overlapping) |
| `[[1,4],[2,3]]` | `[[1,4]]` (fully contained) |
| `[]` | `[]` |

**Key Insights:**
1. Sort by start; then a single pass suffices — if the next interval's start is `<= ` the current
   merged interval's end, extend it; otherwise close the current interval and start a new one.
2. Merging is a purely local decision once sorted — no need to compare against every previously merged
   interval, only the most recent one.

**Python Solution:**
```python
def merge_intervals(intervals: list[list[int]]) -> list[list[int]]:
    """
    Time:  O(n log n)
    Space: O(n)
    """
    if not intervals:
        return []

    intervals = sorted(intervals, key=lambda iv: iv[0])
    merged = [intervals[0]]

    for start, end in intervals[1:]:
        if start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])

    return merged
```

**Follow-Up Questions:**
1. Insert a new interval into an already-sorted, already-merged list → `O(n)` single pass (LeetCode 57)
   instead of re-sorting everything from scratch.
2. Intervals arrive as a live stream and queries ask "is point `x` covered right now?" → maintain the
   merged set in a sorted structure (e.g., a balanced BST/`sortedcontainers`) for `O(log n)` insert and
   point queries instead of re-merging the whole array per update.

---

## 5. Number of Islands (Grid Flood Fill)

**Problem Statement:**
Given an `m x n` binary grid where `'1'` is land and `'0'` is water, return the number of islands (a
group of `'1'`s connected 4-directionally, surrounded by water/grid edges).

**Example:**
```
grid = [
 ["1","1","0","0"],
 ["1","1","0","0"],
 ["0","0","1","0"],
 ["0","0","0","1"],
]
Output: 3
```

**Test Cases:**

| Grid | Output |
|---|---|
| All `"0"` | `0` |
| All `"1"` (single block) | `1` |
| Checkerboard pattern | one island per isolated `"1"` (no 4-directional adjacency) |
| Grid above | `3` |

**Key Insights:**
1. Scan every cell; whenever an unvisited `'1'` is found, that's a *new* island — flood-fill (BFS/DFS)
   from it to mark every connected `'1'` as visited so it's never double-counted.
2. Use an explicit stack for flood-fill instead of recursion for large grids, to avoid Python's
   recursion-depth limit on a grid with a long, winding island.

**Python Solution:**
```python
def num_islands(grid: list[list[str]]) -> int:
    """
    Time:  O(m * n)
    Space: O(m * n) worst case for the visited set / stack
    """
    if not grid or not grid[0]:
        return 0

    rows, cols = len(grid), len(grid[0])
    visited = [[False] * cols for _ in range(rows)]
    count = 0

    def flood_fill(sr: int, sc: int) -> None:
        stack = [(sr, sc)]
        visited[sr][sc] = True
        while stack:
            r, c = stack.pop()
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = r + dr, c + dc
                if (0 <= nr < rows and 0 <= nc < cols
                        and not visited[nr][nc] and grid[nr][nc] == "1"):
                    visited[nr][nc] = True
                    stack.append((nr, nc))

    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == "1" and not visited[r][c]:
                count += 1
                flood_fill(r, c)

    return count
```

**Follow-Up Questions:**
1. Return the size of the largest island, not just the count → have `flood_fill` return the number of
   cells it visited, track the max.
2. The grid updates one cell at a time (land added/removed) and you need the island count after each
   update → this becomes "Number of Islands II," solved with a Union-Find structure supporting
   incremental `union` per land addition rather than a full re-scan.

---

## 6. Course Schedule: Topological Sort and Cycle Detection

**Problem Statement:**
There are `num_courses` courses labeled `0` to `num_courses - 1`. Given a list of prerequisite pairs
`[course, prereq]` meaning `prereq` must be taken before `course`, return whether it's possible to
finish all courses (i.e., the prerequisite graph has no cycle). The identical shape is also reported
framed as build-system dependency resolution — given build targets and `[target, depends_on]` pairs,
return a valid build order or detect a cycle; the algorithm below is unchanged either way.

**Example:**
```
num_courses = 2, prerequisites = [[1, 0]]
Output: True   # take 0, then 1

num_courses = 2, prerequisites = [[1, 0], [0, 1]]
Output: False  # cycle: 0 needs 1, 1 needs 0
```

**Test Cases:**

| num_courses | prerequisites | Output |
|---|---|---|
| `2` | `[[1,0]]` | `True` |
| `2` | `[[1,0],[0,1]]` | `False` |
| `4` | `[[1,0],[2,0],[3,1],[3,2]]` | `True` |
| `3` | `[]` | `True` |

**Key Insights:**
1. This is exactly cycle detection on a directed graph, solvable with Kahn's algorithm: repeatedly
   remove nodes with in-degree 0; if every node is eventually removed, there's no cycle.
2. Build an adjacency list `prereq -> [dependent courses]` and an `in_degree` array; courses that start
   at in-degree 0 have no prerequisite and can always be taken first.
3. If the BFS processes fewer nodes than `num_courses`, the unprocessed remainder forms a cycle — no
   need for a separate DFS-based cycle check.

**Python Solution:**
```python
from collections import defaultdict, deque


def can_finish(num_courses: int, prerequisites: list[list[int]]) -> bool:
    """
    Time:  O(V + E)
    Space: O(V + E)
    """
    graph: dict[int, list[int]] = defaultdict(list)
    in_degree = [0] * num_courses

    for course, prereq in prerequisites:
        graph[prereq].append(course)
        in_degree[course] += 1

    queue = deque(c for c in range(num_courses) if in_degree[c] == 0)
    visited = 0

    while queue:
        node = queue.popleft()
        visited += 1
        for nxt in graph[node]:
            in_degree[nxt] -= 1
            if in_degree[nxt] == 0:
                queue.append(nxt)

    return visited == num_courses
```

**Follow-Up Questions:**
1. Return *a* valid course order, not just feasibility → the same BFS already produces one: append each
   `node` to an order list as it's dequeued.
2. Courses can have weighted durations, and you want the minimum total time to finish all courses
   respecting prerequisites → longest-path-in-a-DAG (critical path), a small extension on top of the
   same topological order.
3. Return **all** valid topological orders, not just one → backtrack over the same in-degree-driven
   candidate set at each step instead of dequeuing greedily; this is exponential in the worst case, so
   say so explicitly and discuss bounding it (return the first `K`, or only for small/sparse graphs).
4. Group targets/courses into "levels" that can be built/taken in parallel → process the queue in full
   BFS layers (every current in-degree-0 node at once) rather than one node at a time; each layer is
   safely parallelizable since none of its members depend on each other.

---

## 7. Clone Graph

**Problem Statement:**
Given a reference to a node in a connected undirected graph, return a deep copy (clone) of the entire
graph. Each node has a value and a list of neighbor references.

**Example:**
```
Graph: 1 -- 2
       |    |
       4 -- 3
clone_graph(node_1) -> a structurally identical graph with entirely new node objects
```

**Test Cases:**

| Input graph | Expectation |
|---|---|
| Single node, no neighbors | clone is a new object, same value, empty neighbor list |
| 4-cycle (as above) | clone has 4 new nodes wired in the same cycle |
| `None` | returns `None` |
| Node with a self-loop / repeated neighbor edges | clone preserves the same neighbor multiplicity |

**Key Insights:**
1. Use a `dict[original_node, cloned_node]` as both the "already visited" check and the lookup for
   already-created clones — this is what prevents infinite recursion on the graph's cycles.
2. DFS (or BFS) from the given node; for each neighbor, either return its existing clone or create and
   register a new one before recursing into *its* neighbors.

**Python Solution:**
```python
class GraphNode:
    def __init__(self, val: int = 0, neighbors: list["GraphNode"] | None = None):
        self.val = val
        self.neighbors = neighbors or []


def clone_graph(node: "GraphNode | None") -> "GraphNode | None":
    """
    Time:  O(V + E)
    Space: O(V) for the clone map plus recursion stack
    """
    if node is None:
        return None

    clones: dict[GraphNode, GraphNode] = {}

    def dfs(original: GraphNode) -> GraphNode:
        if original in clones:
            return clones[original]

        copy = GraphNode(original.val)
        clones[original] = copy
        for neighbor in original.neighbors:
            copy.neighbors.append(dfs(neighbor))
        return copy

    return dfs(node)
```

**Follow-Up Questions:**
1. The graph is too large for recursion depth → convert to an explicit-stack DFS or a BFS with a queue,
   same `clones` map for cycle-safety.
2. Clone only up to depth `k` from the starting node, leaving deeper parts of the graph referencing the
   originals → track depth alongside each node in the traversal and stop expanding neighbors past `k`.

---

## 8. Linked List Fundamentals: Reverse and Merge Two Sorted Lists

**Problem Statement:**
Two classic linked-list warm-ups Apple interviewers frequently pair together. Part A: reverse a singly
linked list in place. Part B: merge two already-sorted linked lists into one sorted list by splicing
existing nodes (no new nodes).

**Example:**
```
reverse_list(1 -> 2 -> 3 -> None) -> 3 -> 2 -> 1 -> None
merge_two_lists(1 -> 2 -> 4, 1 -> 3 -> 4) -> 1 -> 1 -> 2 -> 3 -> 4 -> 4
```

**Test Cases:**

| Operation | Input | Output |
|---|---|---|
| `reverse_list` | `[]` | `[]` |
| `reverse_list` | `[1]` | `[1]` |
| `reverse_list` | `[1,2,3]` | `[3,2,1]` |
| `merge_two_lists` | `[], []` | `[]` |
| `merge_two_lists` | `[], [0]` | `[0]` |
| `merge_two_lists` | `[1,2,4], [1,3,4]` | `[1,1,2,3,4,4]` |

**Key Insights:**
1. Reversing is a three-pointer walk (`prev`, `current`, `next`) — the classic bug is losing the
   reference to the rest of the list before rewiring `current.next`, so save `next` first every
   iteration.
2. Merging with a dummy head node avoids special-casing "which list starts the result" — always append
   to `tail.next` and advance `tail`, then attach whichever list still has leftover nodes at the end.

**Python Solution:**
```python
class ListNode:
    def __init__(self, val: int = 0, next: "ListNode | None" = None):
        self.val = val
        self.next = next


def reverse_list(head: "ListNode | None") -> "ListNode | None":
    """
    Time:  O(n)
    Space: O(1)
    """
    prev = None
    while head is not None:
        nxt = head.next
        head.next = prev
        prev = head
        head = nxt
    return prev


def merge_two_lists(l1: "ListNode | None", l2: "ListNode | None") -> "ListNode | None":
    """
    Time:  O(n + m)
    Space: O(1) extra — reuses input nodes
    """
    dummy = ListNode()
    tail = dummy

    while l1 is not None and l2 is not None:
        if l1.val <= l2.val:
            tail.next, l1 = l1, l1.next
        else:
            tail.next, l2 = l2, l2.next
        tail = tail.next

    tail.next = l1 if l1 is not None else l2
    return dummy.next
```

**Follow-Up Questions:**
1. Reverse only nodes `[left, right]` within the list, leaving the rest untouched → locate `left - 1`,
   reverse the sublist in place, then reattach both ends.
2. Merge `k` sorted lists, not just two → repeatedly pair-merge (`O(nk log k)`), or a min-heap over the
   current head of each list (`O(n log k)`), reusing `merge_two_lists` as the pairwise primitive either way.

---

## 9. Add Two Numbers: Linked List Arithmetic

**Problem Statement:**
Two non-negative integers are represented as linked lists where each node holds a single digit, stored
in *reverse* order (least significant digit first). Add the two numbers and return the sum as a linked
list in the same format.

**Example:**
```
l1 = 2 -> 4 -> 3   (represents 342)
l2 = 5 -> 6 -> 4   (represents 465)
Output: 7 -> 0 -> 8   (represents 807)
```

**Test Cases:**

| l1 | l2 | Output |
|---|---|---|
| `[2,4,3]` | `[5,6,4]` | `[7,0,8]` |
| `[0]` | `[0]` | `[0]` |
| `[9,9,9]` | `[1]` | `[0,0,0,1]` (carry propagates into a new node) |
| `[5]` | `[5]` | `[0,1]` |

**Key Insights:**
1. Because digits are least-significant-first, this is literally grade-school addition read left to
   right — no need to reverse anything first.
2. Track a `carry` across the loop; the loop condition must also check `carry` (not just `l1 or l2`),
   since a trailing carry (e.g. `999 + 1`) needs one more output digit after both lists are exhausted.

**Python Solution:**
```python
def add_two_numbers(l1: "ListNode | None", l2: "ListNode | None") -> "ListNode | None":
    """
    Time:  O(max(n, m))
    Space: O(max(n, m)) for the result list
    """
    dummy = ListNode()
    tail = dummy
    carry = 0

    while l1 is not None or l2 is not None or carry:
        v1 = l1.val if l1 is not None else 0
        v2 = l2.val if l2 is not None else 0
        total = v1 + v2 + carry
        carry, digit = divmod(total, 10)

        tail.next = ListNode(digit)
        tail = tail.next

        l1 = l1.next if l1 is not None else None
        l2 = l2.next if l2 is not None else None

    return dummy.next
```

**Follow-Up Questions:**
1. Digits are stored most-significant-first instead → reverse both lists first (or use two stacks to
   process from the end), then apply the same carry logic.
2. Support subtraction as well as addition → need a sign and borrow logic instead of carry; qualitatively
   more special-casing (which operand is larger, borrowing across zero digits).

---

## 10. Binary Tree Fundamentals: Same Tree and Invert Tree

**Problem Statement:**
Two paired binary-tree warm-ups. Part A: given the roots of two binary trees, determine if they are
structurally identical with the same node values. Part B: invert a binary tree (mirror it — swap every
node's left and right children, recursively) and return its root.

**Example:**
```
is_same_tree([1,2,3], [1,2,3]) -> True
is_same_tree([1,2], [1,None,2]) -> False   # same values, different shape

invert_tree([4,2,7,1,3,6,9]) -> [4,7,2,9,6,3,1]
```

**Test Cases:**

| Operation | Input | Output |
|---|---|---|
| `is_same_tree` | both `None` | `True` |
| `is_same_tree` | one `None`, one not | `False` |
| `is_same_tree` | same shape, differing value | `False` |
| `invert_tree` | `None` | `None` |
| `invert_tree` | single node | itself, unchanged |

**Key Insights:**
1. Both are textbook tree recursion: the answer for a (sub)tree is defined in terms of the same answer
   on its children, with `None` as the base case.
2. `is_same_tree` short-circuits as soon as either a value mismatch or a "one side is `None`, the other
   isn't" shape mismatch is found — no need to keep comparing the rest of the tree.
3. `invert_tree` swaps children *and* recurses into both — swapping without recursing only flips the
   top level, not the whole tree.

**Python Solution:**
```python
class TreeNode:
    def __init__(self, val: int = 0, left: "TreeNode | None" = None,
                 right: "TreeNode | None" = None):
        self.val = val
        self.left = left
        self.right = right


def is_same_tree(p: "TreeNode | None", q: "TreeNode | None") -> bool:
    """
    Time:  O(n)
    Space: O(h) recursion stack, h = tree height
    """
    if p is None and q is None:
        return True
    if p is None or q is None or p.val != q.val:
        return False
    return is_same_tree(p.left, q.left) and is_same_tree(p.right, q.right)


def invert_tree(root: "TreeNode | None") -> "TreeNode | None":
    """
    Time:  O(n)
    Space: O(h) recursion stack, h = tree height
    """
    if root is None:
        return None
    root.left, root.right = invert_tree(root.right), invert_tree(root.left)
    return root
```

**Follow-Up Questions:**
1. Check if a tree is a *mirror of itself* (symmetric) rather than comparing two separate trees →
   recurse comparing `left.left` vs `right.right` and `left.right` vs `right.left` on one tree.
2. Do both without recursion (interviewer probes stack-depth concerns on unbalanced trees) → iterative
   BFS/DFS with an explicit stack/queue holding pairs of nodes to compare (or single nodes to invert).

---

## 11. Search in Rotated Sorted Array

**Problem Statement:**
Given an array of distinct integers sorted in ascending order, then rotated at an unknown pivot, and a
`target` value, return its index, or `-1` if not present. Must run in O(log n).

**Example:**
```
nums = [4,5,6,7,0,1,2], target = 0
Output: 4
```

**Test Cases:**

| nums | target | Output |
|---|---|---|
| `[4,5,6,7,0,1,2]` | `0` | `4` |
| `[4,5,6,7,0,1,2]` | `3` | `-1` |
| `[1]` | `0` | `-1` |
| `[5,1,3]` | `5` | `0` |

**Key Insights:**
1. At any midpoint, at least one half (`[lo..mid]` or `[mid..hi]`) is guaranteed to be normally sorted —
   determine which half is sorted by comparing `nums[lo]` to `nums[mid]`.
2. Once you know which half is sorted, checking whether `target` falls in that half's value range is an
   ordinary comparison; if it does, recurse/iterate into that half, otherwise the other half.

**Python Solution:**
```python
def search_rotated(nums: list[int], target: int) -> int:
    """
    Time:  O(log n)
    Space: O(1)
    """
    lo, hi = 0, len(nums) - 1

    while lo <= hi:
        mid = (lo + hi) // 2
        if nums[mid] == target:
            return mid

        if nums[lo] <= nums[mid]:            # left half is sorted
            if nums[lo] <= target < nums[mid]:
                hi = mid - 1
            else:
                lo = mid + 1
        else:                                  # right half is sorted
            if nums[mid] < target <= nums[hi]:
                lo = mid + 1
            else:
                hi = mid - 1

    return -1
```

**Follow-Up Questions:**
1. The array may contain duplicates → `nums[lo] <= nums[mid]` no longer reliably identifies the sorted
   half; fall back to `lo += 1` when `nums[lo] == nums[mid] == nums[hi]` to shrink ambiguity, which
   degrades worst case to O(n).
2. Find the rotation pivot (index of the minimum element) as a subroutine → a simpler standalone binary
   search comparing `nums[mid]` to `nums[hi]`.

---

## 12. LRU Cache (Design)

**Problem Statement:**
Design a data structure for a Least Recently Used (LRU) cache. Implement:
- `LRUCache(capacity)` — initialize with positive size `capacity`.
- `get(key)` — return the value if present (and mark it most recently used), else `-1`.
- `put(key, value)` — insert or update the value; if this exceeds `capacity`, evict the least recently
  used entry first.

Both operations must run in O(1) average time — reported as one of Apple's most common "design"
questions (alongside close relatives like a time-based key-value store).

**Example:**
```
cache = LRUCache(2)
cache.put(1, 1)
cache.put(2, 2)
cache.get(1)      # -> 1, and 1 is now most-recently-used
cache.put(3, 3)   # evicts key 2 (least recently used)
cache.get(2)      # -> -1 (evicted)
```

**Test Cases:**

| Sequence | Result |
|---|---|
| `put(1,1); put(2,2); get(1); put(3,3); get(2)` | `1`, then `-1` |
| `get(1)` on empty cache | `-1` |
| `put(1,1); put(1,2); get(1)` | `2` (update, not a second entry) |
| capacity `1`: `put(1,1); put(2,2); get(1)` | `-1` (1 was evicted) |

**Key Insights:**
1. O(1) `get`/`put` needs both O(1) lookup (hash map) and O(1) "move to most-recent" / "evict least
   recent" (a doubly linked list, so no shifting is required) — neither alone is sufficient.
2. Python's `collections.OrderedDict` already implements exactly this combination internally
   (hash map + doubly linked list), so it's a legitimate and expected O(1) implementation in an
   interview — just be ready to explain what it's doing under the hood if asked to build it from scratch.

**Python Solution:**
```python
from collections import OrderedDict


class LRUCache:
    """
    get/put: O(1) average
    Space:   O(capacity)
    """

    def __init__(self, capacity: int):
        self.capacity = capacity
        self._cache: "OrderedDict[int, int]" = OrderedDict()

    def get(self, key: int) -> int:
        if key not in self._cache:
            return -1
        self._cache.move_to_end(key)
        return self._cache[key]

    def put(self, key: int, value: int) -> None:
        if key in self._cache:
            self._cache.move_to_end(key)
        self._cache[key] = value
        if len(self._cache) > self.capacity:
            self._cache.popitem(last=False)
```

**Follow-Up Questions:**
1. Implement it without `OrderedDict` (the usual follow-up) → a hash map of `key -> node` plus a
   hand-rolled doubly linked list with dummy head/tail sentinels for O(1) unlink/relink.
2. Make it thread-safe for concurrent `get`/`put` → guard the linked-list mutation and map access with a
   lock; note that this serializes all access, so discuss sharding the cache by key hash for concurrency.
3. Add a per-entry TTL on top of LRU eviction → store an expiry timestamp per node; check-and-evict
   lazily on access, plus an optional background sweep for entries nobody accesses again.

---

## 13. Top K Frequent Elements

**Problem Statement:**
Given an integer array `nums` and an integer `k`, return the `k` most frequent elements, in any order.

**Example:**
```
nums = [1,1,1,2,2,3], k = 2
Output: [1, 2]
```

**Test Cases:**

| nums | k | Output (set) |
|---|---|---|
| `[1,1,1,2,2,3]` | `2` | `{1,2}` |
| `[1]` | `1` | `{1}` |
| `[4,4,4,6,6,7,7,7,7]` | `1` | `{7}` |
| `[1,2,3]` | `3` | `{1,2,3}` |

**Key Insights:**
1. Count frequencies with a hash map, then only need the top `k` of those — a full sort of all distinct
   values is `O(d log d)` (d = distinct values) and is fine, but a heap of size `k` (`O(d log k)`) is the
   asymptotically better answer when `k` is small relative to `d`.
2. For a strictly O(n) solution, bucket sort by frequency (bucket index = count, values up to `n`) and
   read off the top `k` buckets from the high end — worth mentioning as the optimal follow-up even if
   the heap-based version is what you code first.

**Python Solution:**
```python
import heapq
from collections import Counter


def top_k_frequent(nums: list[int], k: int) -> list[int]:
    """
    Time:  O(n + d log k), d = number of distinct elements
    Space: O(d)
    """
    counts = Counter(nums)
    return [num for num, _ in heapq.nlargest(k, counts.items(), key=lambda item: item[1])]
```

**Follow-Up Questions:**
1. Do it in guaranteed O(n) (not `O(d log k)`) → bucket sort by count: `buckets[count].append(value)`
   for `count` in `1..n`, then walk buckets from `n` down to `1` collecting values until `k` are found.
2. The stream is unbounded / can't hold exact counts in memory → approximate with a Count-Min Sketch for
   frequency estimates plus a small heap of current top candidates (standard "heavy hitters" approach).

---

## 14. View Hierarchy Hit-Testing (Apple UI Coordinate Query)

**Problem Statement:**
Reported as an Apple-specific onsite prompt: "given a view hierarchy and a coordinate point, return all
views containing that point." Each `View` has a frame `(x, y, width, height)` expressed relative to its
*parent's* origin (the way UIKit/AppKit view frames actually work), and a list of child views. Given the
root view and a point in the root's coordinate space, return every view (from the root down to the
deepest match) whose frame contains that point.

**Example:**
```
root (0,0,100,100)
├── header (0,0,100,20)
└── body (0,20,100,80)
    └── button (10,10,30,15)   # relative to body -> absolute (10,30)-(40,45)

hit_test(root, 20, 35) -> [root, body, button]   # header excluded, x/y outside its frame
```

**Test Cases:**

| Point | Result |
|---|---|
| `(20, 35)` | `[root, body, button]` |
| `(20, 5)` | `[root, header]` |
| `(5, 25)` | `[root, body]` (inside body, misses button) |
| `(500, 500)` | `[]` (outside root entirely) |

**Key Insights:**
1. Coordinates are relative to each view's *parent*, not global — you must accumulate an absolute offset
   while descending the tree (`abs = parent_abs + view.local_origin`), not compare the point against each
   view's raw `(x, y)` directly.
2. If a point falls outside a view's frame, it's guaranteed to fall outside every descendant of that
   view too (children are laid out within their parent's bounds) — so a miss lets you prune the entire
   subtree instead of still checking every child.
3. Sibling subtrees are independent: if views overlap (a real UI scenario — floating panels, badges),
   the point can match views down more than one branch; don't stop after the first match found.

**Python Solution:**
```python
from dataclasses import dataclass, field


@dataclass
class View:
    name: str
    x: float          # frame origin, relative to the parent's origin
    y: float
    width: float
    height: float
    children: list["View"] = field(default_factory=list)


def hit_test(root: View, point_x: float, point_y: float) -> list[View]:
    """
    Time:  O(V) worst case; a miss prunes its whole subtree
    Space: O(h) recursion stack + O(k) result, h = tree height, k = matching views
    """
    hits: list[View] = []

    def walk(view: View, parent_abs_x: float, parent_abs_y: float) -> None:
        abs_x = parent_abs_x + view.x
        abs_y = parent_abs_y + view.y

        if not (abs_x <= point_x < abs_x + view.width
                and abs_y <= point_y < abs_y + view.height):
            return  # outside this view -> outside every descendant too

        hits.append(view)
        for child in view.children:
            walk(child, abs_x, abs_y)

    walk(root, 0.0, 0.0)
    return hits
```

**Follow-Up Questions:**
1. Touch dispatch usually wants only the single front-most view, not the whole ancestor chain → that's
   `hits[-1]` from this same traversal (deepest match found), assuming later-drawn siblings are meant to
   be in front — see the z-order follow-up below for when that assumption doesn't hold.
2. Siblings can overlap and have an explicit z-order (drawn front-to-back or back-to-front) → add a
   `z_index`/paint-order field; among sibling matches at the same level, prefer the one with the higher
   z-index rather than assuming child-list order reflects visual stacking.
3. The hierarchy is very deep/wide and hit-testing runs on every touch move (performance-sensitive) →
   maintain a spatial index (e.g., a bounding-volume/quad-tree over view frames) so a touch move doesn't
   re-walk the full tree from the root every time.

---

## 15. System Design — iCloud-Style File Sync and Conflict Resolution

**Problem Statement:**
A recurring Apple system-design theme: design a cloud storage/sync service (an iCloud Drive-style
system) that keeps a folder in sync across multiple devices (Mac, iPhone, iPad), each of which can go
offline and edit files independently.

**Functional Requirements:**
- Propagate file creates/modifies/deletes from any device to the cloud and out to every other device.
- Support large files via chunked, resumable upload/download.
- Detect and resolve conflicts when the same file is edited on two devices while both are offline.

**Non-Functional Requirements:**
- Devices are frequently offline or on slow/metered connections — sync must be bandwidth-efficient
  (only transfer what changed) and resumable after interruption.
- Minimize what the server needs to see of file contents (privacy-first design is an explicit
  evaluation criterion in Apple's actual loop, not an afterthought).
- Eventually consistent: devices converge without requiring the user to manually intervene for the
  common (non-conflicting) case.

**High-Level Design:**
1. **Change detection**: a local sync agent watches the filesystem; on a change, it chunks the file
   using content-defined chunking (a rolling hash finds chunk boundaries based on content, not fixed
   offsets) so a small edit only produces a small delta, not a whole re-upload.
2. **Versioning**: each file tracks a per-device version vector (not a single global timestamp — clocks
   across devices can't be trusted for ordering). A linear history (one device's vector strictly
   dominates) is a clean fast-forward; divergent vectors on both sides mean a genuine conflict.
3. **Storage**: chunks are pushed to a content-addressed blob store keyed by chunk hash — identical
   content (even across different files/users) is naturally deduplicated. A separate metadata service
   tracks, per file, the ordered list of chunk hashes making up its current version.
4. **Conflict handling**: on a detected conflict, keep both versions (e.g., surface a "(conflicted copy
   from iPhone)" file) rather than silently picking a winner and losing data; some structured/text
   formats can attempt an automatic 3-way merge against the common ancestor version first.
5. **Notify & pull**: a push notification tells other devices "something changed, go check" (a wake-up
   hint only); the device still pulls from the metadata service to fetch the authoritative current
   version and diff against its local state.

**Data Model (sketch):**
```
files(file_id PK, path, latest_version_id, deleted)
file_versions(version_id PK, file_id, device_id, version_vector, created_at, chunk_hashes[])
chunks(chunk_hash PK, size, storage_ref)          # content-addressed, deduplicated
device_sync_state(device_id, file_id, last_synced_version_id)
```

**Scaling & Reliability:**
- Content-addressed chunk storage deduplicates across users and devices, cutting both storage and
  transfer bandwidth for common content (OS files, shared documents, re-added photos).
- The metadata service (small records: hashes + version vectors) is the hot path; bulky chunk bytes sit
  in cheap object storage, fetched directly by the device rather than proxied through metadata servers.
- A missed or duplicate push notification is harmless — it's only a hint to go pull, and the pull always
  compares against the authoritative metadata state.

**Follow-Up Questions:**
1. How do you tell a genuine conflict apart from two devices making the same edit independently and
   arriving at identical content? → compare resulting content hash, not just the version vectors; if the
   chunk list is identical, converge silently instead of surfacing a false conflict.
2. How does client-side end-to-end encryption change the design? → chunk hashes then operate on
   ciphertext, which breaks cross-user/cross-file deduplication unless using convergent encryption — a
   privacy/efficiency tradeoff worth naming explicitly rather than glossing over.
3. A user has a folder of 50,000 small files (e.g., a source tree) → per-file metadata overhead can
   dominate actual content size; batch metadata updates across many small files in one round trip rather
   than one request per file.

---

## 16. System Design — Push Notification Service (APNs-Style)

**Problem Statement:**
Design a push notification service (the shape of Apple's own APNs): third-party app servers submit
notifications addressed to a device; the service delivers them to that device in near-real-time,
including while the device is backgrounded, and queues them for delivery once a currently-offline
device reconnects.

**Functional Requirements:**
- Accept a notification submission `(device_token, payload, priority, collapse_id)` from an app server
  and deliver it to the addressed device.
- Support both user-visible alerts and silent/background pushes.
- Coalesce redundant pending notifications sharing the same `collapse_id` (only the latest matters, e.g.
  "3 new messages" superseding "2 new messages").

**Non-Functional Requirements:**
- Massive fan-in (many app servers sending) and fan-out (huge device population); sub-second delivery
  while a device is reachable.
- Devices are routinely offline (locked, no signal, low battery) — undelivered notifications must queue
  and flush on reconnect rather than being dropped.
- At-least-once delivery is acceptable; silent, indefinite loss is not.

**High-Level Design:**
1. **Persistent connection tier**: each device holds one long-lived, authenticated, multiplexed
   connection to a nearby gateway node — this always-on channel is what makes near-real-time delivery
   possible without the device polling.
2. **Submission API**: an app server calls the submission endpoint; the service validates the token,
   looks up which gateway node (if any) currently holds that device's connection via a connection
   registry, and hands the notification off for delivery.
3. **Delivery routing**: if the device is connected, forward directly to its gateway node for immediate
   push over the open connection. If not connected, persist the notification in a small, bounded
   per-device pending queue — a new notification with the same `collapse_id` replaces the pending one
   rather than appending, keeping the queue naturally bounded regardless of sender chattiness.
4. **Reconnect flush**: the moment a device reconnects (to any gateway node), its pending queue is
   delivered and cleared.
5. **Priority tiers**: high-priority pushes (calls, security alerts) bypass coalescing/batching
   entirely; low-priority background pushes can be batched into periodic delivery windows to reduce how
   often the device's radio has to wake up.

**Data Model (sketch):**
```
device_connections(device_token, gateway_node_id, connected_at)      # ephemeral, TTL'd
pending_notifications(device_token, collapse_id, payload, priority, enqueued_at)
delivery_receipts(notification_id, device_token, delivered_at, status)
```

**Scaling & Reliability:**
- Shard the gateway tier by `device_token` hash so a device's live connection and its pending queue sit
  close together operationally.
- Coalescing by `collapse_id` bounds per-device queue size independent of how often a sender pushes.
- The connection registry is the one piece of fast-changing shared state — keep it minimal
  (`token -> node`) and TTL-based, so a crashed gateway's devices are naturally reassigned the moment
  they reconnect elsewhere, with no manual failover step.

**Follow-Up Questions:**
1. A device is in poor coverage and its connection flaps every few seconds → don't replay the full
   pending queue on every micro-reconnect; wait for the connection to hold for a short grace period
   before flushing, to avoid redundant delivery storms.
2. How do you stop a buggy or malicious app server from spamming one device? → enforce per-app rate
   limits at the submission API, independent of and upstream from the per-device delivery/coalescing
   logic.
3. Silent background pushes shouldn't wake the device as often as user-visible alerts → batch low
   priority pushes into shared, periodic wake windows across apps on a device, instead of letting each
   app trigger its own radio wake-up independently.

---

## 17. System Design — Typeahead Search with On-Device and Server-Side Ranking

**Problem Statement:**
Reported as an Apple onsite prompt: "design a typeahead box for a search engine." Design search
autocomplete that returns ranked suggestions as a user types, blending the user's own (privacy-sensitive)
local context with global, server-backed trending suggestions — with Apple's actual evaluation explicitly
weighting the privacy tradeoffs of what leaves the device, not just latency and ranking quality.

**Functional Requirements:**
- On each keystroke, return top-N ranked suggestions with minimal perceptible delay.
- Blend personal sources (recent searches, contacts, on-device content) with global popularity/trending
  terms.
- Degrade gracefully to local-only suggestions when offline or the network call is slow.

**Non-Functional Requirements:**
- Per-keystroke latency budget in the tens of milliseconds — typing must never feel like it's waiting on
  the network.
- Minimize raw per-keystroke query traffic leaving the device, and never let personal/sensitive local
  sources (contacts, recent history) be required to leave the device at all.
- Rankings must reflect shifting trends without requiring an app/index update to roll out.

**High-Level Design:**
1. **On-device index**: an in-memory prefix index (trie) over local sources — recent searches,
   contacts, installed app names, on-device documents — built and incrementally updated on the device.
   Every keystroke queries this synchronously with zero network round trip, and this alone covers the
   privacy-sensitive personal-suggestion case entirely on-device.
2. **Debounced server call**: only send the current prefix to the server after a short pause in typing
   (or after a minimum character count), not on every keystroke — this both cuts perceived latency
   pressure on the network path and minimizes how much partial-query telemetry ever leaves the device.
3. **Server-side global index**: a compressed trie/FST built offline (batch, or near-real-time for
   trending terms) from aggregated, privacy-preserved query statistics — thresholded/differentially
   private so rare, potentially identifying queries never surface as suggestions to anyone. Serving is a
   pure precomputed lookup, no per-query ranking computation.
4. **Merge & progressive render**: the client shows local results immediately (available with zero
   latency), then merges in server suggestions as they arrive, re-ranking the combined list by a blend
   of personal relevance and global popularity — a soft upgrade, never a blocking wait.
5. **Client-side response cache**: cache recent server responses per-prefix so backspacing/retyping a
   prefix already queried this session is instant and doesn't repeat the network round trip.

**Data Model (sketch):**
```
local_index (on-device):  trie<term, {source, recency, frequency}>
server_index (offline-built): compressed trie/FST<prefix, top_completions[], popularity_score>
query_aggregates (server, privacy-preserved): term, thresholded_count_bucket, last_updated
```

**Scaling & Reliability:**
- The server-side index is read-only and precomputed; serving cost per request is a cheap lookup, not a
  ranking computation, so it scales horizontally by simply replicating the index.
- Debouncing keeps server QPS proportional to "pauses in typing" rather than "total keystrokes typed
  globally," which is the dominant cost lever at Apple's device scale.
- A slow/failed server call never blocks the UI — local results are already rendered, and server
  suggestions arrive as a strictly additive upgrade.

**Follow-Up Questions:**
1. How do you surface a fast-breaking trending term before the next offline index rebuild? → maintain a
   small, frequently-refreshed "hot terms" overlay merged on top of the base offline index, rebuilt on a
   much shorter cycle than the full index.
2. How do you prevent the on-device suggestion history from leaking to a second person who picks up an
   unlocked device? → scope the local index to the authenticated session/user context, and exclude
   sources flagged as sensitive from autocomplete outright rather than relying on UI-level hiding.
3. Typo tolerance → cheap edit-distance-1 fallback on the local index when the exact prefix has no
   matches; server-side, this is normally baked into how the FST/index itself is constructed (mapping
   common misspellings at build time) rather than done as an online fuzzy search per query, which
   doesn't scale at Apple's query volume.

---

## 18. System Design — App Store / Spotlight-Style Search Ranking

**Problem Statement:**
Reported Apple onsite theme: design the search ranking backend behind a large catalog search
(App Store app search, or a Spotlight-style system search) that returns relevant results within a tight
latency budget, distinct from the typeahead problem above — this is ranking full query results against
a large corpus, not prefix autocomplete.

**Functional Requirements:**
- Given a query, return the top-N most relevant results from a catalog of millions of items.
- Support both lexical matching (exact/partial name matches) and semantic relevance (intent beyond exact
  keyword overlap).
- Support safe rollout of ranking model changes without regressing result quality for all users at once.

**Non-Functional Requirements:**
- Tight end-to-end latency budget (results must feel instantaneous after a query is submitted).
- Index must reflect new/updated catalog items without a full rebuild for every change.
- Ranking must be explainable/debuggable enough to diagnose "why did this result rank where it did."

**High-Level Design:**
1. **Candidate generation (recall stage)**: an inverted index (lexical/BM25-style match) and/or an ANN
   index over item embeddings narrows millions of catalog items down to a few hundred candidates cheaply
   — this stage optimizes for recall, not precision.
2. **Feature fetch**: pull each candidate's ranking features (click-through history, recency, category
   popularity, personalization signals) from a precomputed, low-latency feature store — never compute
   these from raw logs synchronously in the request path.
3. **Ranking (precision stage)**: a heavier ML ranker (gradient-boosted trees or a small neural ranker)
   re-scores only the shortlist from step 1 using richer cross-features (query-item interaction terms),
   since running it over the full catalog would blow the latency budget.
4. **Indexing pipeline**: catalog updates flow through a near-real-time or micro-batch indexing pipeline
   so new/changed items become searchable within minutes, not on the next full rebuild.
5. **Experimentation**: new ranking models are validated via interleaving or a holdout traffic split
   against the current production ranker before a full rollout, with explicit guardrail metrics
   (click-through rate, zero-result rate, abandonment).

**Data Model (sketch):**
```
catalog_items(item_id, name, category, metadata, embedding_id, indexed_at)
item_features(item_id, ctr_7d, popularity_score, recency, ...)   # feature store, low-latency reads
ranking_logs(query_id, query_text, candidates[], scores[], final_order[], ts)
```

**Scaling & Reliability:**
- Candidate generation and feature fetch are independently horizontally scalable (sharded index,
  replicated feature store); the ranker only ever processes a bounded shortlist, so its cost doesn't grow
  with catalog size.
- Explicitly split the total latency budget across candidate generation, feature fetch, and ranking —
  naming concrete numbers per stage is what separates a strong answer from a hand-wavy one.
- Fall back to a cheaper heuristic ranking (lexical match + popularity) if the ML ranker times out or
  errors, rather than failing the search entirely.

**Follow-Up Questions:**
1. A brand-new catalog item has no click history to rank on → fall back to content-based features
   (category, metadata, developer/publisher reputation) until enough interaction data accumulates.
2. How do you debug a specific bad-looking result ranking too high? → the `ranking_logs` table should
   retain per-candidate feature values and score contributions, not just the final order, so a specific
   query's ranking decision can be replayed and inspected after the fact.
3. Query volume spikes heavily around specific real-world events (e.g. a keynote, a major app launch) →
   candidate-generation and feature-store layers need headroom/autoscaling sized for burst traffic, not
   just steady-state average QPS.

---

## 19. Behavioral Themes

Apple's behavioral round is a standard STAR-format interview — see
[`behavioral_interview.md`](./behavioral_interview.md) for general STAR-method prep. Apple's process is
notably less centralized than most large tech companies: each team's interviewers write their own
questions rather than drawing from one company-wide bank, but a few themes recur consistently across
reported loops:

- **Woven into the project deep-dive, not a standalone segment**: because loops are team-owned, the
  behavioral questions are often interleaved into your technical project walkthrough rather than run as
  a separate round — be ready to pause mid-explanation of a technical decision and answer "how did you
  handle disagreement here" or "who did you need to convince" without switching gears.
- **Ownership and craftsmanship**: real reported prompts include *"walk me through your resume and the
  work you're most proud of"* and *"walk me through a key technical decision and why you made it"* —
  be ready to go deep on the reasoning and tradeoffs behind one piece of work, not just what shipped.
- **Progress under ambiguity**: *"what was the most challenging project you've worked on"* — Apple
  teams are often small and move on incomplete specs; have an example of making forward progress without
  a fully defined problem.
- **Cross-functional and cross-discipline collaboration**: many Apple SWE roles sit adjacent to
  hardware, design, or ML teams with very different working styles and constraints — have an example of
  aligning with a team that didn't share your discipline's assumptions.
- **Confidentiality and compartmentalization**: Apple's culture places unusually high weight on
  need-to-know information handling; be prepared to discuss how you've protected sensitive information
  or worked effectively within a need-to-know structure, if relevant to your experience.
- **Attention to detail**: be ready to discuss a time a small, easy-to-miss detail mattered a lot to the
  outcome — this maps directly to Apple's own self-described engineering culture, and interviewers
  reportedly probe for it explicitly.

---

## References

Sources used for compiling these questions:
- [Apple Software Engineer Interview Questions - Glassdoor](https://www.glassdoor.com/Interview/Apple-Software-Engineer-Interview-Questions-EI_IE1138.0,5_KO6,23.htm)
- [Top 30 Apple Coding Interview Questions (with solutions) - Educative](https://www.educative.io/blog/apple-coding-interview-questions)
- [30 Apple LeetCode Interview Questions for 2026 - Verve AI](https://www.vervecopilot.com/blog/apple-leetcode-interview-questions)
- [Apple System Design Interview (2026 Guide) - Exponent](https://www.tryexponent.com/blog/apple-system-design-interview)
- [Apple Interview Guide: Process, Questions, Salaries & Prep Tips - InterviewQuery](https://www.interviewquery.com/interview-guides/apple)

Note: unlike some companies' interview loops, Apple does not appear to rely on a large bank of
proprietary "Code Craft"-style prompts — reported questions are predominantly standard, well-known
algorithm problems (per the sources above) plus team-specific system-design and behavioral rounds. The
solutions above are original implementations written for this file, not copied from any source.
