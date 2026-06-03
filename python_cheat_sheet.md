# Python Coding Interview Cheat Sheet

A comprehensive reference for coding interviews covering data structures, algorithms, and common patterns.

---

## Table of Contents
1. [Data Structures & Operations](#1-data-structures--operations)
2. [String Manipulation](#2-string-manipulation)
3. [Built-in Functions & Modules](#3-built-in-functions--modules)
4. [Common Algorithm Patterns](#4-common-algorithm-patterns)
5. [Graph & Tree Essentials](#5-graph--tree-essentials)
6. [Time Complexity Quick Reference](#6-time-complexity-quick-reference)
7. [Concurrency & Async](#7-concurrency--async)
8. [Tips & Gotchas](#8-tips--gotchas)

---

## 1. Data Structures & Operations

### Lists

```python
# Slicing
arr = [0, 1, 2, 3, 4, 5]
arr[1:4]      # [1, 2, 3] - elements 1 to 3
arr[::2]      # [0, 2, 4] - every 2nd element
arr[::-1]     # [5, 4, 3, 2, 1, 0] - reverse
arr[-3:]      # [3, 4, 5] - last 3 elements

# List Comprehensions
squares = [x**2 for x in range(10)]
evens = [x for x in range(10) if x % 2 == 0]
matrix = [[0] * cols for _ in range(rows)]  # 2D list (correct way)

# Common Methods
arr.append(x)       # O(1) - add to end
arr.pop()           # O(1) - remove from end
arr.pop(0)          # O(n) - remove from front (use deque instead)
arr.insert(i, x)    # O(n) - insert at index
arr.remove(x)       # O(n) - remove first occurrence
arr.index(x)        # O(n) - find index of x
arr.sort()          # O(n log n) - in-place sort
arr.sort(key=lambda x: x[1], reverse=True)  # custom sort
sorted(arr)         # returns new sorted list
arr.reverse()       # O(n) - in-place reverse
```

### Dictionaries

```python
# Basic Operations
d = {}
d = {'a': 1, 'b': 2}
d.get('key', default)  # returns default if key not found
d.setdefault('key', []).append(val)  # initialize if missing

# Iteration
for key in d:
for key, val in d.items():
for val in d.values():

# defaultdict - auto-initializes missing keys
from collections import defaultdict
graph = defaultdict(list)
graph['a'].append('b')  # no KeyError

count = defaultdict(int)
count['x'] += 1  # starts at 0

# Counter - counting elements
from collections import Counter
cnt = Counter([1, 1, 2, 3, 3, 3])  # Counter({3: 3, 1: 2, 2: 1})
cnt.most_common(2)  # [(3, 3), (1, 2)]
cnt['x']            # 0 (no KeyError for missing)
cnt.update([1, 2])  # add more counts

# OrderedDict (Python 3.7+ dict maintains order, but useful for move_to_end)
from collections import OrderedDict
od = OrderedDict()
od.move_to_end('key')        # move to end
od.move_to_end('key', False) # move to front
od.popitem(last=False)       # pop from front (FIFO)
```

### Sets

```python
s = set()
s = {1, 2, 3}
s.add(x)        # O(1)
s.remove(x)     # O(1), raises KeyError if missing
s.discard(x)    # O(1), no error if missing
s.pop()         # remove arbitrary element

# Set Operations
a | b   # union
a & b   # intersection
a - b   # difference (in a but not b)
a ^ b   # symmetric difference (in a or b but not both)
a <= b  # subset
a < b   # proper subset

# Frozen Set (immutable, hashable - can be dict key or set element)
fs = frozenset([1, 2, 3])
```

### Heaps (heapq - Min Heap by default)

```python
import heapq

# Min Heap
heap = []
heapq.heappush(heap, val)
min_val = heapq.heappop(heap)
peek = heap[0]  # peek without removing

# Heapify existing list
arr = [3, 1, 4, 1, 5]
heapq.heapify(arr)  # O(n)

# Max Heap (negate values)
heapq.heappush(heap, -val)
max_val = -heapq.heappop(heap)

# N largest/smallest
heapq.nlargest(k, arr)
heapq.nsmallest(k, arr)
heapq.nlargest(k, arr, key=lambda x: x[1])  # with key

# Heap with custom comparison (tuple trick)
heapq.heappush(heap, (priority, index, item))  # index breaks ties
```

### Deque (Double-Ended Queue)

```python
from collections import deque

dq = deque()
dq = deque([1, 2, 3])
dq = deque(maxlen=3)  # bounded deque, auto-removes from opposite end

# Operations (all O(1))
dq.append(x)       # add to right
dq.appendleft(x)   # add to left
dq.pop()           # remove from right
dq.popleft()       # remove from left
dq.extend([4, 5])  # extend right
dq.extendleft([0]) # extend left (reverses order)
dq.rotate(1)       # rotate right
dq.rotate(-1)      # rotate left
```

### Stacks & Queues

```python
# Stack (LIFO) - use list
stack = []
stack.append(x)  # push
stack.pop()      # pop
stack[-1]        # peek

# Queue (FIFO) - use deque
from collections import deque
queue = deque()
queue.append(x)   # enqueue
queue.popleft()   # dequeue
queue[0]          # peek

# Priority Queue - use heapq (see above)
```

---

## 2. String Manipulation

### Common Methods

```python
s = "  Hello, World!  "

# Whitespace & Case
s.strip()       # "Hello, World!" - remove leading/trailing whitespace
s.lstrip()      # "Hello, World!  "
s.rstrip()      # "  Hello, World!"
s.lower()       # "  hello, world!  "
s.upper()       # "  HELLO, WORLD!  "
s.capitalize()  # "  hello, world!  " -> "  Hello, world!  "
s.title()       # "  Hello, World!  "

# Split & Join
"a,b,c".split(",")      # ['a', 'b', 'c']
"a b c".split()         # ['a', 'b', 'c'] - splits on whitespace
"a,,b".split(",")       # ['a', '', 'b'] - keeps empty strings
",".join(['a', 'b'])    # "a,b"
"".join(['a', 'b'])     # "ab"

# Find & Replace
s.find("World")         # 9 (index) or -1 if not found
s.index("World")        # 9 (raises ValueError if not found)
s.count("l")            # 3
s.replace("World", "Python")
s.startswith("  He")    # True
s.endswith("!  ")       # True

# Checks
s.isalpha()    # only letters
s.isdigit()    # only digits
s.isalnum()    # letters or digits
s.isspace()    # only whitespace
s.islower()
s.isupper()
```

### Character Operations

```python
ord('a')       # 97 - character to ASCII
chr(97)        # 'a' - ASCII to character

# Check character type
c.isalpha()
c.isdigit()
c.isalnum()
c.islower()
c.isupper()

# Convert case
c.lower()
c.upper()

# Alphabet tricks
ord('z') - ord('a')  # 25 (0-indexed position in alphabet)
chr(ord('a') + i)    # i-th letter (0-indexed)
```

### String Formatting

```python
# f-strings (Python 3.6+)
name = "Alice"
age = 30
f"Name: {name}, Age: {age}"

# Formatting numbers
f"{3.14159:.2f}"      # "3.14" - 2 decimal places
f"{42:05d}"           # "00042" - zero-padded
f"{1000000:,}"        # "1,000,000" - thousands separator
f"{255:b}"            # "11111111" - binary
f"{255:x}"            # "ff" - hex

# Alignment
f"{s:>10}"  # right align
f"{s:<10}"  # left align
f"{s:^10}"  # center
```

---

## 3. Built-in Functions & Modules

### Essential Built-ins

```python
# sorted - returns new sorted list
sorted(arr)
sorted(arr, reverse=True)
sorted(arr, key=lambda x: x[1])
sorted(arr, key=lambda x: (x[0], -x[1]))  # multi-key sort

# enumerate - index + element
for i, val in enumerate(arr):
for i, val in enumerate(arr, start=1):  # 1-indexed

# zip - combine iterables
for a, b in zip(list1, list2):
list(zip([1, 2], [3, 4]))  # [(1, 3), (2, 4)]
dict(zip(keys, values))    # create dict from lists

# Unzip
pairs = [(1, 'a'), (2, 'b')]
nums, chars = zip(*pairs)  # (1, 2), ('a', 'b')

# map - apply function
list(map(int, ["1", "2"]))  # [1, 2]
list(map(lambda x: x*2, [1, 2]))  # [2, 4]

# filter - filter elements
list(filter(lambda x: x > 0, [-1, 0, 1, 2]))  # [1, 2]

# reduce - accumulate
from functools import reduce
reduce(lambda x, y: x + y, [1, 2, 3])  # 6
reduce(lambda x, y: x * y, [1, 2, 3])  # 6

# any/all
any([False, True, False])  # True - at least one True
all([True, True, False])   # False - all must be True

# min/max with key
max(arr, key=lambda x: x[1])
min(arr, key=abs)

# sum
sum(arr)
sum(arr, start=10)  # start value
```

### itertools

```python
from itertools import permutations, combinations, product, groupby, accumulate, chain

# Permutations - all orderings
list(permutations([1, 2, 3]))      # 6 items (n!)
list(permutations([1, 2, 3], 2))   # 6 items (nPr)

# Combinations - choose without order
list(combinations([1, 2, 3], 2))   # [(1,2), (1,3), (2,3)]
list(combinations_with_replacement([1, 2], 2))  # [(1,1), (1,2), (2,2)]

# Product - Cartesian product
list(product([1, 2], [3, 4]))      # [(1,3), (1,4), (2,3), (2,4)]
list(product([0, 1], repeat=3))   # all 3-bit combinations

# Groupby (must be sorted first!)
from itertools import groupby
data = [('a', 1), ('a', 2), ('b', 3)]
for key, group in groupby(data, key=lambda x: x[0]):
    print(key, list(group))

# Accumulate - running totals
list(accumulate([1, 2, 3, 4]))     # [1, 3, 6, 10]
list(accumulate([1, 2, 3], lambda x, y: x * y))  # [1, 2, 6]

# Chain - combine iterables
list(chain([1, 2], [3, 4]))        # [1, 2, 3, 4]
list(chain.from_iterable([[1, 2], [3, 4]]))  # [1, 2, 3, 4]
```

### bisect (Binary Search)

```python
import bisect

arr = [1, 3, 5, 7, 9]

# Find insertion point
bisect.bisect_left(arr, 5)   # 2 - leftmost position
bisect.bisect_right(arr, 5)  # 3 - rightmost position
bisect.bisect(arr, 5)        # same as bisect_right

# Insert while maintaining order
bisect.insort_left(arr, 4)   # arr becomes [1, 3, 4, 5, 7, 9]
bisect.insort_right(arr, 4)

# Find if element exists
def binary_search(arr, x):
    i = bisect.bisect_left(arr, x)
    return i < len(arr) and arr[i] == x

# Find leftmost/rightmost occurrence
def find_leftmost(arr, x):
    i = bisect.bisect_left(arr, x)
    return i if i < len(arr) and arr[i] == x else -1

def find_rightmost(arr, x):
    i = bisect.bisect_right(arr, x) - 1
    return i if i >= 0 and arr[i] == x else -1
```

### functools

```python
from functools import lru_cache, cache, cmp_to_key

# Memoization
@lru_cache(maxsize=None)  # or @cache in Python 3.9+
def fib(n):
    if n < 2:
        return n
    return fib(n-1) + fib(n-2)

# For methods that take unhashable args (like lists)
# Convert to tuple: fib(tuple(arr))

# Custom comparison for sorting
def compare(a, b):
    return a - b  # negative: a < b, zero: equal, positive: a > b

sorted(arr, key=cmp_to_key(compare))

# Example: sort numbers to form largest number
def compare(a, b):
    if a + b > b + a:
        return -1  # a should come first
    return 1
sorted(nums, key=cmp_to_key(lambda a, b: -1 if a+b > b+a else 1))
```

### collections Summary

```python
from collections import Counter, defaultdict, deque, OrderedDict, namedtuple

Counter([1, 1, 2])        # {1: 2, 2: 1}
defaultdict(list)         # auto-initialize with empty list
defaultdict(int)          # auto-initialize with 0
defaultdict(set)          # auto-initialize with empty set
deque()                   # double-ended queue
OrderedDict()             # ordered dict with move_to_end

# namedtuple - lightweight class
Point = namedtuple('Point', ['x', 'y'])
p = Point(1, 2)
p.x, p.y  # 1, 2
```

---

## 4. Common Algorithm Patterns

### Binary Search Templates

```python
# Template 1: Find exact match
def binary_search(arr, target):
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1

# Template 2: Find leftmost (first occurrence)
def find_leftmost(arr, target):
    left, right = 0, len(arr)
    while left < right:
        mid = (left + right) // 2
        if arr[mid] < target:
            left = mid + 1
        else:
            right = mid
    return left  # returns insertion point if not found

# Template 3: Find rightmost (last occurrence)
def find_rightmost(arr, target):
    left, right = 0, len(arr)
    while left < right:
        mid = (left + right) // 2
        if arr[mid] <= target:
            left = mid + 1
        else:
            right = mid
    return left - 1  # returns -1 if not found

# Template 4: Find first True (condition-based)
def first_true(lo, hi, condition):
    while lo < hi:
        mid = (lo + hi) // 2
        if condition(mid):
            hi = mid
        else:
            lo = mid + 1
    return lo
```

### Two Pointers

```python
# Opposite direction - e.g., two sum in sorted array
def two_sum_sorted(arr, target):
    left, right = 0, len(arr) - 1
    while left < right:
        total = arr[left] + arr[right]
        if total == target:
            return [left, right]
        elif total < target:
            left += 1
        else:
            right -= 1
    return [-1, -1]

# Same direction - e.g., remove duplicates
def remove_duplicates(arr):
    if not arr:
        return 0
    write = 1
    for read in range(1, len(arr)):
        if arr[read] != arr[read - 1]:
            arr[write] = arr[read]
            write += 1
    return write

# Fast and slow - e.g., cycle detection
def has_cycle(head):
    slow = fast = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        if slow == fast:
            return True
    return False
```

### Sliding Window

```python
# Fixed size window
def max_sum_subarray(arr, k):
    window_sum = sum(arr[:k])
    max_sum = window_sum
    for i in range(k, len(arr)):
        window_sum += arr[i] - arr[i - k]
        max_sum = max(max_sum, window_sum)
    return max_sum

# Variable size window - e.g., smallest subarray with sum >= target
def min_subarray_len(arr, target):
    left = 0
    current_sum = 0
    min_len = float('inf')

    for right in range(len(arr)):
        current_sum += arr[right]
        while current_sum >= target:
            min_len = min(min_len, right - left + 1)
            current_sum -= arr[left]
            left += 1

    return min_len if min_len != float('inf') else 0

# Variable size with condition - e.g., longest substring with at most k distinct
def longest_k_distinct(s, k):
    from collections import defaultdict
    count = defaultdict(int)
    left = 0
    max_len = 0

    for right in range(len(s)):
        count[s[right]] += 1
        while len(count) > k:
            count[s[left]] -= 1
            if count[s[left]] == 0:
                del count[s[left]]
            left += 1
        max_len = max(max_len, right - left + 1)

    return max_len
```

### BFS / DFS Templates

```python
from collections import deque

# BFS - iterative (shortest path in unweighted graph)
def bfs(graph, start, target):
    visited = {start}
    queue = deque([(start, 0)])  # (node, distance)

    while queue:
        node, dist = queue.popleft()
        if node == target:
            return dist
        for neighbor in graph[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, dist + 1))
    return -1

# BFS - level by level
def bfs_levels(root):
    if not root:
        return []
    result = []
    queue = deque([root])

    while queue:
        level = []
        for _ in range(len(queue)):  # process entire level
            node = queue.popleft()
            level.append(node.val)
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
        result.append(level)
    return result

# DFS - recursive
def dfs_recursive(graph, node, visited=None):
    if visited is None:
        visited = set()
    visited.add(node)
    for neighbor in graph[node]:
        if neighbor not in visited:
            dfs_recursive(graph, neighbor, visited)
    return visited

# DFS - iterative
def dfs_iterative(graph, start):
    visited = set()
    stack = [start]

    while stack:
        node = stack.pop()
        if node in visited:
            continue
        visited.add(node)
        for neighbor in graph[node]:
            if neighbor not in visited:
                stack.append(neighbor)
    return visited

# DFS on grid
def dfs_grid(grid, row, col, visited):
    if (row < 0 or row >= len(grid) or
        col < 0 or col >= len(grid[0]) or
        (row, col) in visited or grid[row][col] == 0):
        return

    visited.add((row, col))
    for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
        dfs_grid(grid, row + dr, col + dc, visited)
```

### Backtracking

```python
# General template
def backtrack(candidates, path, result):
    if is_solution(path):
        result.append(path[:])  # copy of path
        return

    for candidate in get_candidates(candidates, path):
        if is_valid(candidate, path):
            path.append(candidate)
            backtrack(candidates, path, result)
            path.pop()  # backtrack

# Example: Subsets
def subsets(nums):
    result = []

    def backtrack(start, path):
        result.append(path[:])
        for i in range(start, len(nums)):
            path.append(nums[i])
            backtrack(i + 1, path)
            path.pop()

    backtrack(0, [])
    return result

# Example: Permutations
def permutations(nums):
    result = []

    def backtrack(path, remaining):
        if not remaining:
            result.append(path[:])
            return
        for i in range(len(remaining)):
            path.append(remaining[i])
            backtrack(path, remaining[:i] + remaining[i+1:])
            path.pop()

    backtrack([], nums)
    return result

# Example: Combinations
def combinations(n, k):
    result = []

    def backtrack(start, path):
        if len(path) == k:
            result.append(path[:])
            return
        for i in range(start, n + 1):
            path.append(i)
            backtrack(i + 1, path)
            path.pop()

    backtrack(1, [])
    return result
```

### Dynamic Programming Patterns

```python
# 1D DP - e.g., Fibonacci / Climbing Stairs
def climb_stairs(n):
    if n <= 2:
        return n
    dp = [0] * (n + 1)
    dp[1], dp[2] = 1, 2
    for i in range(3, n + 1):
        dp[i] = dp[i-1] + dp[i-2]
    return dp[n]

# Space-optimized 1D DP
def climb_stairs_optimized(n):
    if n <= 2:
        return n
    prev1, prev2 = 2, 1
    for i in range(3, n + 1):
        prev1, prev2 = prev1 + prev2, prev1
    return prev1

# 2D DP - e.g., Longest Common Subsequence
def lcs(text1, text2):
    m, n = len(text1), len(text2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if text1[i-1] == text2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    return dp[m][n]

# 0/1 Knapsack
def knapsack(weights, values, capacity):
    n = len(weights)
    dp = [[0] * (capacity + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        for w in range(capacity + 1):
            if weights[i-1] <= w:
                dp[i][w] = max(
                    dp[i-1][w],  # don't take
                    dp[i-1][w - weights[i-1]] + values[i-1]  # take
                )
            else:
                dp[i][w] = dp[i-1][w]
    return dp[n][capacity]

# Space-optimized Knapsack (single row)
def knapsack_optimized(weights, values, capacity):
    dp = [0] * (capacity + 1)
    for i in range(len(weights)):
        for w in range(capacity, weights[i] - 1, -1):  # reverse order!
            dp[w] = max(dp[w], dp[w - weights[i]] + values[i])
    return dp[capacity]

# Unbounded Knapsack (forward order)
def unbounded_knapsack(weights, values, capacity):
    dp = [0] * (capacity + 1)
    for w in range(1, capacity + 1):
        for i in range(len(weights)):
            if weights[i] <= w:
                dp[w] = max(dp[w], dp[w - weights[i]] + values[i])
    return dp[capacity]
```

### Union-Find (Disjoint Set Union)

```python
class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n
        self.count = n  # number of components

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])  # path compression
        return self.parent[x]

    def union(self, x, y):
        px, py = self.find(x), self.find(y)
        if px == py:
            return False
        # union by rank
        if self.rank[px] < self.rank[py]:
            px, py = py, px
        self.parent[py] = px
        if self.rank[px] == self.rank[py]:
            self.rank[px] += 1
        self.count -= 1
        return True

    def connected(self, x, y):
        return self.find(x) == self.find(y)
```

### Topological Sort

```python
from collections import deque, defaultdict

# Kahn's Algorithm (BFS)
def topological_sort_bfs(num_nodes, edges):
    graph = defaultdict(list)
    in_degree = [0] * num_nodes

    for u, v in edges:
        graph[u].append(v)
        in_degree[v] += 1

    queue = deque([i for i in range(num_nodes) if in_degree[i] == 0])
    result = []

    while queue:
        node = queue.popleft()
        result.append(node)
        for neighbor in graph[node]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    return result if len(result) == num_nodes else []  # empty if cycle

# DFS-based (reverse post-order)
def topological_sort_dfs(num_nodes, edges):
    graph = defaultdict(list)
    for u, v in edges:
        graph[u].append(v)

    visited = [0] * num_nodes  # 0: unvisited, 1: visiting, 2: visited
    result = []

    def dfs(node):
        if visited[node] == 1:
            return False  # cycle detected
        if visited[node] == 2:
            return True

        visited[node] = 1
        for neighbor in graph[node]:
            if not dfs(neighbor):
                return False
        visited[node] = 2
        result.append(node)
        return True

    for i in range(num_nodes):
        if visited[i] == 0:
            if not dfs(i):
                return []  # cycle

    return result[::-1]  # reverse to get correct order
```

---

## 5. Graph & Tree Essentials

### Graph Representation

```python
# Adjacency List (most common)
from collections import defaultdict

graph = defaultdict(list)
for u, v in edges:
    graph[u].append(v)
    graph[v].append(u)  # for undirected

# Weighted graph
graph = defaultdict(list)
for u, v, weight in edges:
    graph[u].append((v, weight))

# Adjacency Matrix (for dense graphs)
n = len(nodes)
matrix = [[0] * n for _ in range(n)]
for u, v in edges:
    matrix[u][v] = 1
    matrix[v][u] = 1  # for undirected
```

### Tree Traversals

```python
# Binary tree node
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

# Preorder (Root, Left, Right)
def preorder(root):
    if not root:
        return []
    return [root.val] + preorder(root.left) + preorder(root.right)

def preorder_iterative(root):
    if not root:
        return []
    result, stack = [], [root]
    while stack:
        node = stack.pop()
        result.append(node.val)
        if node.right:
            stack.append(node.right)
        if node.left:
            stack.append(node.left)
    return result

# Inorder (Left, Root, Right) - sorted for BST
def inorder(root):
    if not root:
        return []
    return inorder(root.left) + [root.val] + inorder(root.right)

def inorder_iterative(root):
    result, stack = [], []
    current = root
    while stack or current:
        while current:
            stack.append(current)
            current = current.left
        current = stack.pop()
        result.append(current.val)
        current = current.right
    return result

# Postorder (Left, Right, Root)
def postorder(root):
    if not root:
        return []
    return postorder(root.left) + postorder(root.right) + [root.val]

def postorder_iterative(root):
    if not root:
        return []
    result, stack = [], [root]
    while stack:
        node = stack.pop()
        result.append(node.val)
        if node.left:
            stack.append(node.left)
        if node.right:
            stack.append(node.right)
    return result[::-1]

# Level Order (BFS)
def level_order(root):
    if not root:
        return []
    result = []
    queue = deque([root])
    while queue:
        level = []
        for _ in range(len(queue)):
            node = queue.popleft()
            level.append(node.val)
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
        result.append(level)
    return result
```

### Common Tree Patterns

```python
# Tree Height
def height(root):
    if not root:
        return 0
    return 1 + max(height(root.left), height(root.right))

# Is Balanced
def is_balanced(root):
    def check(node):
        if not node:
            return 0
        left = check(node.left)
        right = check(node.right)
        if left == -1 or right == -1 or abs(left - right) > 1:
            return -1
        return 1 + max(left, right)
    return check(root) != -1

# Lowest Common Ancestor (Binary Tree)
def lca(root, p, q):
    if not root or root == p or root == q:
        return root
    left = lca(root.left, p, q)
    right = lca(root.right, p, q)
    if left and right:
        return root
    return left or right

# Validate BST
def is_valid_bst(root, min_val=float('-inf'), max_val=float('inf')):
    if not root:
        return True
    if root.val <= min_val or root.val >= max_val:
        return False
    return (is_valid_bst(root.left, min_val, root.val) and
            is_valid_bst(root.right, root.val, max_val))

# Serialize/Deserialize
def serialize(root):
    if not root:
        return "null"
    return f"{root.val},{serialize(root.left)},{serialize(root.right)}"

def deserialize(data):
    values = iter(data.split(","))

    def build():
        val = next(values)
        if val == "null":
            return None
        node = TreeNode(int(val))
        node.left = build()
        node.right = build()
        return node

    return build()
```

### Dijkstra's Algorithm (Shortest Path)

```python
import heapq
from collections import defaultdict

def dijkstra(graph, start, end):
    # graph: {node: [(neighbor, weight), ...]}
    distances = {start: 0}
    heap = [(0, start)]

    while heap:
        dist, node = heapq.heappop(heap)

        if node == end:
            return dist

        if dist > distances.get(node, float('inf')):
            continue

        for neighbor, weight in graph[node]:
            new_dist = dist + weight
            if new_dist < distances.get(neighbor, float('inf')):
                distances[neighbor] = new_dist
                heapq.heappush(heap, (new_dist, neighbor))

    return -1  # no path found
```

---

## 6. Time Complexity Quick Reference

### Data Structure Operations

| Operation | Array | Linked List | Hash Table | BST (balanced) | Heap |
|-----------|-------|-------------|------------|----------------|------|
| Access | O(1) | O(n) | O(1) | O(log n) | O(1) peek |
| Search | O(n) | O(n) | O(1) | O(log n) | O(n) |
| Insert | O(n) | O(1)* | O(1) | O(log n) | O(log n) |
| Delete | O(n) | O(1)* | O(1) | O(log n) | O(log n) |

*With reference to node

### Sorting Algorithms

| Algorithm | Best | Average | Worst | Space | Stable |
|-----------|------|---------|-------|-------|--------|
| Quick Sort | O(n log n) | O(n log n) | O(n²) | O(log n) | No |
| Merge Sort | O(n log n) | O(n log n) | O(n log n) | O(n) | Yes |
| Heap Sort | O(n log n) | O(n log n) | O(n log n) | O(1) | No |
| Tim Sort (Python) | O(n) | O(n log n) | O(n log n) | O(n) | Yes |
| Counting Sort | O(n+k) | O(n+k) | O(n+k) | O(k) | Yes |

### Common Pattern Complexities

| Pattern | Time | Space |
|---------|------|-------|
| Two Pointers | O(n) | O(1) |
| Sliding Window | O(n) | O(k) |
| Binary Search | O(log n) | O(1) |
| BFS/DFS | O(V + E) | O(V) |
| DP (2D table) | O(m×n) | O(m×n) or O(n) |
| Backtracking | O(k^n) | O(n) |
| Union Find | O(α(n)) ≈ O(1) | O(n) |

---

## 7. Concurrency & Async

### Threading

```python
import threading
from threading import Thread, Lock, RLock, Semaphore, Condition, Event

# Basic Thread
def worker(name):
    print(f"Thread {name} running")

thread = Thread(target=worker, args=("A",))
thread.start()
thread.join()  # wait for completion

# Multiple Threads
threads = []
for i in range(5):
    t = Thread(target=worker, args=(i,))
    threads.append(t)
    t.start()
for t in threads:
    t.join()

# Lock - mutual exclusion
lock = Lock()
counter = 0

def increment():
    global counter
    with lock:  # context manager (preferred)
        counter += 1

# Manual lock (less preferred)
lock.acquire()
try:
    counter += 1
finally:
    lock.release()

# RLock - reentrant lock (same thread can acquire multiple times)
rlock = RLock()
def recursive_function(n):
    with rlock:
        if n > 0:
            recursive_function(n - 1)

# Semaphore - limit concurrent access
semaphore = Semaphore(3)  # max 3 threads

def limited_access():
    with semaphore:
        # only 3 threads can be here at once
        do_work()

# Condition - wait/notify pattern
condition = Condition()
queue = []

def producer():
    with condition:
        queue.append(item)
        condition.notify()  # or notify_all()

def consumer():
    with condition:
        while not queue:
            condition.wait()  # releases lock and waits
        item = queue.pop(0)

# Event - simple signaling
event = Event()

def waiter():
    event.wait()  # blocks until set
    print("Event triggered!")

def trigger():
    event.set()    # unblocks all waiters
    event.clear()  # reset for reuse
```

### Thread-Safe Data Structures

```python
from queue import Queue, LifoQueue, PriorityQueue
from collections import deque
import threading

# Thread-safe Queue (blocking)
q = Queue(maxsize=10)
q.put(item)           # blocks if full
q.put(item, block=False)  # raises Full
q.get()               # blocks if empty
q.get(timeout=1.0)    # timeout in seconds
q.task_done()         # signal item processed
q.join()              # wait for all items processed

# LIFO Queue (stack)
stack = LifoQueue()

# Priority Queue
pq = PriorityQueue()
pq.put((priority, item))  # lower priority = higher precedence

# Thread-safe deque operations
dq = deque()
# append/appendleft/pop/popleft are atomic
# but compound operations need lock
lock = threading.Lock()
with lock:
    if dq:
        item = dq.popleft()
```

### Multiprocessing

```python
from multiprocessing import Process, Pool, Queue, Value, Array, Manager
import multiprocessing as mp

# Basic Process
def worker(x):
    return x * x

process = Process(target=worker, args=(5,))
process.start()
process.join()

# Pool - parallel execution
with Pool(processes=4) as pool:
    results = pool.map(worker, [1, 2, 3, 4])     # blocking
    results = pool.map_async(worker, [1, 2, 3])  # non-blocking
    results.get()  # wait and get results

    # Single execution
    result = pool.apply(worker, (5,))       # blocking
    result = pool.apply_async(worker, (5,)) # non-blocking

# Shared Memory
counter = Value('i', 0)  # 'i' = integer
with counter.get_lock():
    counter.value += 1

arr = Array('d', [1.0, 2.0, 3.0])  # 'd' = double

# Queue for inter-process communication
queue = Queue()
def producer(q):
    q.put("data")
def consumer(q):
    data = q.get()

# Manager for complex shared objects
manager = Manager()
shared_list = manager.list()
shared_dict = manager.dict()
```

### Asyncio

```python
import asyncio

# Basic async function
async def fetch_data(url):
    await asyncio.sleep(1)  # simulate I/O
    return f"data from {url}"

# Running async code
async def main():
    result = await fetch_data("http://example.com")
    print(result)

asyncio.run(main())

# Concurrent execution with gather
async def main():
    results = await asyncio.gather(
        fetch_data("url1"),
        fetch_data("url2"),
        fetch_data("url3"),
    )
    return results

# Create tasks for concurrent execution
async def main():
    task1 = asyncio.create_task(fetch_data("url1"))
    task2 = asyncio.create_task(fetch_data("url2"))

    # Do other work while tasks run
    await asyncio.sleep(0.5)

    result1 = await task1
    result2 = await task2

# Timeout
async def main():
    try:
        result = await asyncio.wait_for(fetch_data("url"), timeout=2.0)
    except asyncio.TimeoutError:
        print("Timed out!")

# AsyncIO Primitives
lock = asyncio.Lock()
async def protected_operation():
    async with lock:
        await do_work()

semaphore = asyncio.Semaphore(3)
async def limited_operation():
    async with semaphore:
        await do_work()

event = asyncio.Event()
async def waiter():
    await event.wait()
async def trigger():
    event.set()

# Queue for async producer-consumer
queue = asyncio.Queue(maxsize=10)
async def producer():
    await queue.put(item)
async def consumer():
    item = await queue.get()
    queue.task_done()
```

### Common Concurrency Patterns

```python
# Producer-Consumer with threading
from threading import Thread, Lock
from queue import Queue

def producer(q, n_items):
    for i in range(n_items):
        q.put(f"item_{i}")
    q.put(None)  # sentinel

def consumer(q):
    while True:
        item = q.get()
        if item is None:
            break
        process(item)
        q.task_done()

# Reader-Writer Lock (multiple readers, single writer)
class ReadWriteLock:
    def __init__(self):
        self.readers = 0
        self.readers_lock = Lock()
        self.writer_lock = Lock()

    def acquire_read(self):
        with self.readers_lock:
            self.readers += 1
            if self.readers == 1:
                self.writer_lock.acquire()

    def release_read(self):
        with self.readers_lock:
            self.readers -= 1
            if self.readers == 0:
                self.writer_lock.release()

    def acquire_write(self):
        self.writer_lock.acquire()

    def release_write(self):
        self.writer_lock.release()

# Thread Pool Executor (high-level)
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed

with ThreadPoolExecutor(max_workers=4) as executor:
    # Submit tasks
    futures = [executor.submit(worker, i) for i in range(10)]

    # Get results as they complete
    for future in as_completed(futures):
        result = future.result()

    # Or use map for ordered results
    results = list(executor.map(worker, range(10)))
```

---

## 8. Tips & Gotchas

### Mutable Default Arguments

```python
# BAD - mutable default is shared across calls!
def append_to(element, lst=[]):
    lst.append(element)
    return lst

append_to(1)  # [1]
append_to(2)  # [1, 2] - not [2]!

# GOOD - use None as default
def append_to(element, lst=None):
    if lst is None:
        lst = []
    lst.append(element)
    return lst
```

### Integer Division Behavior

```python
# Python 3 floor division rounds toward negative infinity
7 // 3    # 2
-7 // 3   # -3 (not -2!)
7 // -3   # -3

# Use int() for truncation toward zero
int(-7 / 3)  # -2

# Modulo follows floor division
-7 % 3   # 2 (not -1)
7 % -3   # -2
```

### Copy vs Deepcopy

```python
import copy

# Shallow copy - nested objects are shared
original = [[1, 2], [3, 4]]
shallow = original[:]  # or list(original) or original.copy()
shallow[0][0] = 99
print(original)  # [[99, 2], [3, 4]] - modified!

# Deep copy - all nested objects are copied
deep = copy.deepcopy(original)
deep[0][0] = 100
print(original)  # [[99, 2], [3, 4]] - unchanged
```

### Common Edge Cases to Check

```python
# Empty input
if not arr:
    return []

# Single element
if len(arr) == 1:
    return arr[0]

# Two elements (important for many algorithms)

# All same elements
# All negative numbers
# Integer overflow (Python handles this, but be aware)
# Very large inputs (consider time complexity)

# Strings
if not s or len(s) == 0:
    return ""

# Trees
if not root:
    return None
```

### Python-Specific Tips

```python
# Swap without temp variable
a, b = b, a

# Chained comparisons
if 0 <= x < 10:

# Multiple assignment
a = b = c = 0

# Infinity
float('inf')
float('-inf')

# Check if all elements satisfy condition
all(x > 0 for x in arr)
any(x > 0 for x in arr)

# Flatten nested list
flat = [item for sublist in nested for item in sublist]

# Dictionary from two lists
d = dict(zip(keys, values))

# Reverse string/list
s[::-1]
arr[::-1]

# Check if string is palindrome
s == s[::-1]

# Get index of max/min element
max_idx = arr.index(max(arr))
# More efficient for large arrays:
max_idx = max(range(len(arr)), key=lambda i: arr[i])

# Sort dict by value
sorted_dict = dict(sorted(d.items(), key=lambda x: x[1]))

# Remove duplicates while preserving order
list(dict.fromkeys(arr))

# Transpose matrix
transposed = list(zip(*matrix))  # returns tuples
transposed = [list(row) for row in zip(*matrix)]  # returns lists
```

### GIL (Global Interpreter Lock)

```python
# GIL prevents true parallel execution of Python bytecode
# This means:

# CPU-bound tasks: Use multiprocessing, not threading
from multiprocessing import Pool
with Pool(4) as p:
    results = p.map(cpu_intensive_function, data)

# I/O-bound tasks: Threading works fine (GIL released during I/O)
from concurrent.futures import ThreadPoolExecutor
with ThreadPoolExecutor(4) as executor:
    results = list(executor.map(io_bound_function, urls))

# Async I/O: Best for high-concurrency I/O
import asyncio
async def main():
    results = await asyncio.gather(*[fetch(url) for url in urls])

# Summary:
# - CPU-bound: multiprocessing
# - I/O-bound (few tasks): threading
# - I/O-bound (many tasks): asyncio
# - Mixed: combine as needed
```

### Common Mistakes to Avoid

```python
# Don't modify list while iterating
# BAD
for item in arr:
    if condition:
        arr.remove(item)

# GOOD - iterate over copy or use list comprehension
for item in arr[:]:
    if condition:
        arr.remove(item)
# OR
arr = [item for item in arr if not condition]

# Don't use == for None comparison
# BAD
if x == None:
# GOOD
if x is None:

# Don't forget to return in recursive functions
# BAD
def search(node, target):
    if node.val == target:
        return True
    if node.left:
        search(node.left, target)  # missing return!

# GOOD
def search(node, target):
    if node.val == target:
        return True
    if node.left:
        if search(node.left, target):
            return True
    return False

# Be careful with default argument evaluation
import time
# BAD - evaluated once at function definition
def log(msg, timestamp=time.time()):
    print(f"{timestamp}: {msg}")

# GOOD
def log(msg, timestamp=None):
    if timestamp is None:
        timestamp = time.time()
    print(f"{timestamp}: {msg}")
```

---

## Quick Reference Card

```python
# Data Structures
from collections import defaultdict, Counter, deque, OrderedDict
import heapq
import bisect

# Algorithms
from functools import lru_cache, cmp_to_key
from itertools import permutations, combinations, product, groupby, accumulate

# Concurrency
from threading import Thread, Lock, Semaphore, Condition
from multiprocessing import Pool, Process, Queue
import asyncio
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

# Common Patterns
# - Binary Search: left, right = 0, len(arr) - 1
# - Two Pointers: left, right = 0, len(arr) - 1
# - Sliding Window: for right in range(len(arr)): while condition: left += 1
# - BFS: queue = deque([start]); while queue: node = queue.popleft()
# - DFS: stack = [start]; while stack: node = stack.pop()
# - Backtracking: path.append(x); recurse(); path.pop()
# - DP: dp[i] = f(dp[i-1], dp[i-2], ...)
# - Union-Find: find with path compression, union by rank
```

---

*Good luck with your interviews!*
