# Pinterest Interview Questions

## Table of Contents

- [Interview Format](#interview-format)
- [Quick-Recall Cheat Sheet](#quick-recall-cheat-sheet)
- [Coding Questions](#coding-questions)
  - [Put Boxes Into the Warehouse (LC 1564)](#put-boxes-into-the-warehouse-lc-1564)
    - [Follow-up: Maximum Total Box Height](#follow-up-maximum-total-box-height)
    - [Variation: Boxes from Both Ends](#variation-boxes-from-both-ends)
  - [Lighthouse Light Propagation (Pinterest Custom)](#lighthouse-light-propagation-pinterest-custom)
  - [Convert BST to Sorted Doubly Linked List (LC 426)](#convert-bst-to-sorted-doubly-linked-list-lc-426)
  - [Escape Room / Room-by-Room Race (Pinterest Custom)](#escape-room--room-by-room-race-pinterest-custom)
  - [Cheapest Flights Within K Stops (LC 787)](#cheapest-flights-within-k-stops-lc-787)
  - [Odd Even Linked List (LC 328)](#odd-even-linked-list-lc-328)
  - [Reconstruct Itinerary (LC 332)](#reconstruct-itinerary-lc-332)
  - [Violation Log Counter](#violation-log-counter)
- [High-Frequency Coding Problems](#high-frequency-coding-problems)
  - [ACL / Permission System](#acl--permission-system-high-frequency)
  - [LC 815 — Bus Routes](#lc-815--bus-routes-high-frequency)
  - [LC 1055 — Shortest Way to Form String](#lc-1055--shortest-way-to-form-string-high-frequency)
  - [Pixie-like Random Walk](#pixie-like-random-walk)
  - [Expression Add Operators — Left-to-Right](#expression-add-operators--left-to-right-simplified-lc-282)
  - [Insert into Sorted Circular DLL (LC 708)](#insert-into-sorted-circular-doubly-linked-list-lc-708--lc-426-follow-up)
  - [LC 465 — Optimal Account Balancing](#lc-465--optimal-account-balancing)
  - [First Word Containing a Prefix](#first-word-containing-a-prefix)
  - [LC 84 — Largest Rectangle in Histogram](#lc-84--largest-rectangle-in-histogram-monotonic-stack)
  - [LC 392 — Is Subsequence](#lc-392--is-subsequence-two-pointers)
  - [Weighted Sampling (softmax → CDF + binary search)](#weighted-sampling-softmax--cdf--binary-search)
  - [LC 1526 — Minimum Operations to Form a Target Array](#lc-1526--minimum-operations-to-form-a-target-array)
  - [LC 1723 — Find Minimum Time to Finish All Jobs](#lc-1723--find-minimum-time-to-finish-all-jobs)
- [System Design Questions](#system-design-questions)
  - [Design Pinterest Home Feed](#design-pinterest-home-feed)
  - [Design Pinterest Search Engine](#design-pinterest-search-engine)
  - [Design Personalized Pin Recommendation Chatbot](#design-personalized-pin-recommendation-chatbot)
  - [Design Pinterest Notifications](#design-pinterest-notifications)
- [Behavioral Questions](#behavioral-questions)
- [Tips for Pinterest Interviews](#tips-for-pinterest-interviews)

---

## Interview Format

- **Phone Screening**: 1 hour, coding focused (LeetCode style)
- **Onsite**: Multiple rounds including coding, system design, and behavioral

---

## Quick-Recall Cheat Sheet

One-glance revision table: the **pattern** is what to recognize; the **recipe** is the few steps to recall under pressure. Skim this right before the interview; the full intuition + worked example for each is in the sections below.

**Coding Questions**

| Problem | Pattern | Recall in one line |
|---------|---------|--------------------|
| Put Boxes Into Warehouse (LC 1564) | Greedy + prefix-min | Effective height = prefix-min of ceilings; sort boxes **asc**, fill from the **rightmost** fitting room. |
| Max Total Box Height (follow-up) | Greedy | Same effective heights, but place the **largest** boxes first to maximize the summed height. |
| Boxes from Both Ends | Two pointers | Sort boxes **desc**; converge pointers from both raw ends, place each at whichever end still fits. |
| Lighthouse Light Propagation | Ray cast / BFS | Per lighthouse, cast a ray each direction until wall/edge; union lit cells. Use BFS over `(cell, dir)` if mirrors. |
| BST → Sorted DLL (LC 426) | In-order threading | In-order traversal links `prev ↔ node`; close the ring (`head.left = tail`) at the end. |
| Escape Room / Room Race | DLL-per-bucket + map | One FIFO doubly-linked list per room + `player → node` map; `getTop` scans rooms high→low. |
| Cheapest Flights ≤ K Stops (LC 787) | Bellman-Ford | Relax all edges `k+1` times using a **temp copy** of prices (or stop-aware Dijkstra). |
| Odd-Even Linked List (LC 328) | Two pointers | `odd`/`even` pointers step by 2; save `even_head`; splice even chain after the odd tail. |
| Reconstruct Itinerary (LC 332) | Eulerian path (Hierholzer) | DFS popping sorted edges, append node **post-order**, then **reverse** the result. |
| Violation Log Counter | Sliding window | Per-user sorted timestamps; window of `t` seconds, flag if count `> k` (two-pointer or bisect). |

**High-Frequency Problems**

| Problem | Pattern | Recall in one line |
|---------|---------|--------------------|
| ACL / Permission System | Tree walk **up** | `child → parent` map + per-advertiser grant set; `check_access` walks up ancestors for any grant. |
| Bus Routes (LC 815) | BFS, **routes** are nodes | `stop → routes` index; BFS boards unvisited routes (+1 bus per level); mark **routes** visited, not stops. |
| Shortest Way to Form String (LC 1055) | Greedy two-pointer | One pass over `source` per copy, advancing `target`; `-1` if any target char ∉ source. |
| Pixie-like Random Walk | Random walk **with restart** | Walk from query pins, `alpha`-restart; visit counts = relevance; boost multi-query hits. |
| Expression Add Operators LTR | DFS, **running value** | Split digits, apply `+ - *` to the running value (no precedence ⇒ no prev-term bookkeeping). |
| Insert Circular Sorted DLL (LC 708) | One loop, two valid slots | Insert at normal slot `prev ≤ val ≤ cur` **or** the max→min wrap seam; self-loop if empty. |
| Optimal Account Balancing (LC 465) | Net balances + backtrack | Collapse to net balances, drop zeros; backtrack settling the first debt vs each opposite-sign debt. |
| First Word with Prefix | Scan / trie | `word.startswith(prefix)` scan; trie with min word-index per node for many queries. |
| Largest Rectangle (LC 84) | Monotonic stack | Increasing index stack; pop on a shorter bar → `height × width` (back to prev); trailing sentinel `0`. |
| Is Subsequence (LC 392) | Two pointers | Advance `s` pointer on match while scanning `t`; many queries ⇒ `char → positions` + binary search. |
| Weighted Sampling | softmax → CDF + bisect | softmax (subtract max) → build CDF **once** → `bisect` a uniform `u` per draw. |
| Min Ops to Form Array (LC 1526) | Difference array | Answer = `target[0]` + sum of **positive** consecutive differences. |
| Min Time to Finish Jobs (LC 1723) | Backtrack + pruning | Assign jobs→workers; **bound** prune + skip equal-load workers; sort jobs **desc** first. |

---

## Coding Questions

### Put Boxes Into the Warehouse (LC 1564)

**Problem:** You are given two arrays: `boxes` and `warehouse` representing the heights of boxes and the ceiling heights of warehouse rooms respectively. The warehouse rooms are numbered `0` to `n-1` from left to right where `warehouse[i]` is the height of the `i`th room.

Boxes are pushed into the warehouse from left to right. A box can only enter a room if:
1. The box height is less than or equal to the room's ceiling height
2. The box height is less than or equal to all rooms it passes through

Return the maximum number of boxes you can put into the warehouse.

**Example:**
```
Input: boxes = [4,3,4,1], warehouse = [5,3,3,4,1]
Output: 3

Warehouse:    5   3   3   4   1
              |   |   |   |   |
Box heights:  |   3   3   |   1   (can fit boxes of height 3, 3, 1)
              |   |   |   |   |
              +---+---+---+---+
              0   1   2   3   4

Explanation:
- Box of height 1 can go to room 4
- Box of height 3 can go to room 1 or 2
- Box of height 4 cannot fit (room 1 has ceiling 3)
```

**Solution:**

*Main logic:* a box must clear **every** room it passes, so first turn the warehouse into **effective heights** (prefix-min of ceilings from the left). Then greedily match: sort boxes ascending and fill from the **rightmost** room with the smallest box that fits. (The two-pointer alternative sorts boxes descending and walks rooms left-to-right instead.)

```python
def maxBoxesInWarehouse(boxes: list[int], warehouse: list[int]) -> int:
    """
    Greedy approach:
    1. Preprocess warehouse to get effective heights (limited by rooms to the left)
    2. Sort boxes in ascending order
    3. Greedily place smallest boxes from the rightmost room

    Time: O(n log n + m) where n = len(boxes), m = len(warehouse)
    Space: O(m) for effective heights (can be O(1) if we modify warehouse in-place)
    """
    # Step 1: Calculate effective height for each room
    # A room's effective height is min of its height and all rooms to its left
    effective = warehouse[:]
    for i in range(1, len(warehouse)):
        effective[i] = min(effective[i], effective[i-1])

    # Step 2: Sort boxes (smallest first)
    boxes.sort()

    # Step 3: Greedily place boxes from rightmost room
    # Try to fit each box starting from the rightmost available room
    count = 0
    room = len(warehouse) - 1

    for box in boxes:
        # Find rightmost room that can fit this box
        while room >= 0 and effective[room] < box:
            room -= 1

        if room < 0:
            break  # No more rooms available

        count += 1
        room -= 1  # Use this room

    return count
```

**Alternative Solution (Two Pointers):**

```python
def maxBoxesInWarehouse(boxes: list[int], warehouse: list[int]) -> int:
    """
    Two pointers without preprocessing:
    - Sort boxes descending
    - Try to place largest boxes first from the left
    - If box doesn't fit, skip to next smaller box

    Time: O(n log n + m)
    Space: O(1) extra space (sorting may use O(n))
    """
    boxes.sort(reverse=True)

    count = 0
    box_idx = 0

    for room_height in warehouse:
        # Skip boxes that are too tall
        while box_idx < len(boxes) and boxes[box_idx] > room_height:
            box_idx += 1

        if box_idx >= len(boxes):
            break

        # Place this box
        count += 1
        box_idx += 1

    return count
```

---

### Follow-up: Maximum Total Box Height

**Problem:** Instead of maximizing the number of boxes, maximize the total sum of heights of boxes placed in the warehouse.

**Example:**
```
Input: boxes = [4,3,4,1], warehouse = [5,3,3,4,1]
Output: 7

Explanation:
- We can place boxes [4, 3] for total height = 7
- Box 4 goes to room 0 (ceiling 5)
- Box 3 goes to room 1 (ceiling 3)
- This is better than placing [3, 3, 1] = 7 (same in this case)
- Or [4, 1] = 5
```

**Solution 1: Greedy with Sorting (Largest First)**

*Main logic:* now maximize total *height*, not count — so place the **largest** boxes first, each into the leftmost room (by effective height) that can still fit it. Bigger boxes contribute more to the sum, so they get priority for the scarce tall rooms.

```python
def maxTotalBoxHeight(boxes: list[int], warehouse: list[int]) -> int:
    """
    Greedy: Place largest boxes first in leftmost valid room.

    Key insight: To maximize sum, prioritize placing larger boxes.
    Place each box in the leftmost room it can fit.

    Time: O(n log n + n * m) - can be optimized
    Space: O(m)
    """
    # Calculate effective heights
    effective = warehouse[:]
    for i in range(1, len(warehouse)):
        effective[i] = min(effective[i], effective[i-1])

    # Sort boxes descending (largest first)
    boxes.sort(reverse=True)

    total = 0
    used = [False] * len(warehouse)

    for box in boxes:
        # Find leftmost room that can fit this box
        for i in range(len(warehouse)):
            if not used[i] and effective[i] >= box:
                used[i] = True
                total += box
                break

    return total
```

**Solution 2: Optimized Greedy with Two Pointers**

```python
def maxTotalBoxHeight(boxes: list[int], warehouse: list[int]) -> int:
    """
    Optimized greedy using the insight that we should fill from left
    with the largest possible boxes.

    Key insight: Process rooms left to right, for each room pick the
    largest box that fits (considering effective height).

    Time: O(n log n + m)
    Space: O(m)
    """
    # Calculate effective heights
    effective = warehouse[:]
    for i in range(1, len(warehouse)):
        effective[i] = min(effective[i], effective[i-1])

    # Sort boxes descending
    boxes.sort(reverse=True)

    total = 0
    box_idx = 0

    # For each room (left to right), try to place the largest remaining box
    for room_height in effective:
        # Skip boxes that are too tall for this room
        while box_idx < len(boxes) and boxes[box_idx] > room_height:
            box_idx += 1

        if box_idx >= len(boxes):
            break

        total += boxes[box_idx]
        box_idx += 1

    return total
```

**Solution 3: Dynamic Programming (More General)**

```python
def maxTotalBoxHeightDP(boxes: list[int], warehouse: list[int]) -> int:
    """
    DP approach - useful if there are additional constraints.

    dp[i][j] = max total height using first i boxes for first j rooms

    Time: O(n * m)
    Space: O(n * m), can be optimized to O(m)
    """
    # Calculate effective heights
    effective = warehouse[:]
    for i in range(1, len(warehouse)):
        effective[i] = min(effective[i], effective[i-1])

    # Sort boxes descending for optimal assignment
    boxes.sort(reverse=True)

    n, m = len(boxes), len(warehouse)

    # dp[j] = max height achievable using some boxes for first j rooms
    # We'll use a different formulation: assign boxes to rooms greedily

    # Actually for this problem, greedy is optimal
    # DP is overkill but shown for completeness

    dp = [[0] * (m + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            # Don't place box i in room j
            dp[i][j] = max(dp[i-1][j], dp[i][j-1])

            # Place box i in room j (if it fits)
            if boxes[i-1] <= effective[j-1]:
                dp[i][j] = max(dp[i][j], dp[i-1][j-1] + boxes[i-1])

    return dp[n][m]
```

**Comparison of Approaches:**

| Approach | Time | Space | Notes |
|----------|------|-------|-------|
| Greedy (naive) | O(n log n + nm) | O(m) | Simple, works for small inputs |
| Greedy (optimized) | O(n log n + m) | O(m) | Best for this problem |
| DP | O(nm) | O(nm) | Generalizable to more constraints |

---

### Variation: Boxes from Both Ends

**Problem:** What if boxes can be pushed from either the left or right side of the warehouse?

*Main logic:* a room is now reachable from the left **or** the right edge, so its effective height is the better of prefix-min (from the left) and suffix-min (from the right). Rather than materialize both passes, sort boxes **descending** and walk two pointers inward from both raw ends: place each box at whichever end's room can hold it. The descending order makes the raw ends behave like the effective heights — a box too tall for the current `left` room is also too tall for every room behind it (they're only ever shorter on the prefix-min), so discarding it is safe; same on the right.

```python
def maxBoxesBothEnds(boxes: list[int], warehouse: list[int]) -> int:
    """
    When boxes can enter from both ends, we can place more boxes.

    Sort boxes descending, converge two pointers from both raw ends, and
    place each box at whichever end still fits it. Processing largest-first
    means a box rejected at an end is also rejected by every room behind it,
    so the raw ceilings act as effective (prefix/suffix-min) heights for free
    — no separate effective-height passes needed.

    Time: O(n log n + m)
    Space: O(1)
    """
    boxes.sort(reverse=True)

    left, right = 0, len(warehouse) - 1
    count = 0

    for box in boxes:
        if left > right:
            break

        # Place at whichever end's room can still hold this box
        if warehouse[left] >= box:
            count += 1
            left += 1
        elif warehouse[right] >= box:
            count += 1
            right -= 1
        # else: box fits at neither end -> discard (all remaining are smaller)

    return count
```

**Why the raw ends act like effective heights:**

In the original (left-only) problem, a box must clear *every* room from the entrance to its slot, so the binding constraint at room `i` is the prefix-min `min(warehouse[0..i])`, not `warehouse[i]` alone. Here a box can enter from either side, so its constraint is `min(prefix-min[i], suffix-min[i])` — whichever approach is shorter.

The descending-sort trick lets us skip building those min-arrays:

- The `left` pointer only moves rightward, and we always try the **largest** remaining box first. If `warehouse[left] < box`, that box (and every later, smaller box) can never be placed *at or before* `left` from the left side — so we never need to revisit those rooms. Effectively, `warehouse[left]` already represents "the tightest ceiling encountered entering from the left so far."
- The `right` pointer is symmetric for the suffix side.

So the two raw pointers, walking inward, reproduce the prefix-min / suffix-min behavior implicitly — without ever materializing the effective-height arrays.

**Worked example:**
```
warehouse = [3, 1, 3]   boxes = [3, 3, 1]

Effective heights (for intuition):
  prefix-min (from left):  [3, 1, 1]
  suffix-min (from right): [1, 1, 3]
  reachable = max of the two: [3, 1, 3]   <- rooms 0 and 2 can hold a 3

Sort boxes descending: [3, 3, 1]
left = 0, right = 2, count = 0

box = 3:  warehouse[left=0] = 3 >= 3  -> place from left.  count=1, left=1
box = 3:  warehouse[left=1] = 1 <  3  -> try right.
          warehouse[right=2] = 3 >= 3 -> place from right. count=2, right=1
box = 1:  left=1, right=1 (still valid). warehouse[1] = 1 >= 1 -> place. count=3, left=2

Result: 3   (one 3 from each end into rooms 0 and 2, the 1 into room 1)
```

Notice the middle room (raw ceiling 1) is never a blocker for the two 3-boxes: each entered from an end whose path stayed tall, exactly what the prefix/suffix-min intuition predicts.

**Edge cases:**
- `left > right` mid-loop — every room is taken; stop early (the `break`).
- A box larger than both current ends is silently dropped; since boxes are sorted descending, no later box could fit there either.
- Single room: both pointers start equal; one box may be placed if it fits, then `left > right` halts the rest.

---

### Lighthouse Light Propagation (Pinterest Custom)

**Problem:** Simulate light propagation on a 2D grid. Lighthouses emit beams in defined directions. Beams travel in straight lines until hitting a wall or grid edge. Determine which cells receive light.

**Grid Encoding (Example):**
```
0 = empty cell
1 = wall
2 = lighthouse (emits in all 4 directions)
3 = lighthouse (emits right and down only)
... (clarify with interviewer)
```

**Example:**
```
Grid:
  0 1 2 3 4
0 . . L . .
1 . W . . .
2 . . . W .
3 L . . . .
4 . . . . .

L = lighthouse (all directions)
W = wall
. = empty

Light reaches:
  0 1 2 3 4
0 * * L * *    (row 0: lighthouse lights entire row except blocked)
1 . W * . .    (wall blocks, but column 2 lit from above)
2 * . * W .    (row 2, col 0 from lighthouse at (3,0))
3 L * * * *    (lighthouse lights right)
4 * . . . .    (column 0 lit from lighthouse above)
```

**Clarifying Questions to Ask:**
1. What directions can lighthouses emit? (4-directional, 8-directional, specific?)
2. Can light pass through other lighthouses?
3. Are there reflective surfaces (mirrors)?
4. What happens at corners/edges?
5. Can lighthouses rotate (state changes over time)?

---

**Solution 1: Ray Casting (Per-Lighthouse)**

*Main logic:* for each lighthouse, **cast a ray** in each of its emit directions, marking cells lit until the ray hits a wall or the grid edge. Union the lit cells across all lighthouses. O(lighthouses × ray length).

```python
def simulate_light(grid: list[list[int]]) -> list[list[bool]]:
    """
    For each lighthouse, cast rays in all valid directions.
    Mark cells until hitting wall or boundary.

    Time: O(L * max(R, C)) where L = lighthouses
    Space: O(R * C) for lit matrix
    """
    rows, cols = len(grid), len(grid[0])
    lit = [[False] * cols for _ in range(rows)]

    # Direction vectors: up, down, left, right
    DIRECTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    # For 8-directional, add diagonals:
    # DIRECTIONS += [(-1, -1), (-1, 1), (1, -1), (1, 1)]

    EMPTY = 0
    WALL = 1
    LIGHTHOUSE = 2

    def cast_ray(start_r: int, start_c: int, dr: int, dc: int):
        """Cast a ray from (start_r, start_c) in direction (dr, dc)."""
        r, c = start_r + dr, start_c + dc

        while 0 <= r < rows and 0 <= c < cols:
            if grid[r][c] == WALL:
                break  # Wall blocks light

            lit[r][c] = True

            # Optional: if lighthouses block light
            # if grid[r][c] == LIGHTHOUSE:
            #     break

            r += dr
            c += dc

    # Find all lighthouses and cast rays
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == LIGHTHOUSE:
                lit[r][c] = True  # Lighthouse itself is lit

                for dr, dc in DIRECTIONS:
                    cast_ray(r, c, dr, dc)

    return lit
```

**Solution 2: Directional BFS**

```python
from collections import deque

def simulate_light_bfs(grid: list[list[int]]) -> list[list[bool]]:
    """
    BFS where each entry carries (cell, direction).
    Light propagates only along its direction.

    Useful when light can change direction (reflections).

    Time: O(R * C * D) where D = directions
    Space: O(R * C * D) for visited states
    """
    rows, cols = len(grid), len(grid[0])
    lit = [[False] * cols for _ in range(rows)]

    DIRECTIONS = {
        'U': (-1, 0),
        'D': (1, 0),
        'L': (0, -1),
        'R': (0, 1),
    }

    EMPTY = 0
    WALL = 1
    LIGHTHOUSE = 2

    # State: (row, col, direction)
    visited = set()
    queue = deque()

    # Initialize: add all lighthouse beams
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == LIGHTHOUSE:
                lit[r][c] = True
                for dir_name, (dr, dc) in DIRECTIONS.items():
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rows and 0 <= nc < cols:
                        queue.append((nr, nc, dir_name))
                        visited.add((nr, nc, dir_name))

    while queue:
        r, c, direction = queue.popleft()

        if grid[r][c] == WALL:
            continue

        lit[r][c] = True

        # Continue in same direction
        dr, dc = DIRECTIONS[direction]
        nr, nc = r + dr, c + dc

        if 0 <= nr < rows and 0 <= nc < cols:
            state = (nr, nc, direction)
            if state not in visited:
                visited.add(state)
                queue.append(state)

    return lit
```

---

**Variation: Reflective Mirrors**

```python
def simulate_with_mirrors(grid: list[list[int]]) -> list[list[bool]]:
    """
    Grid includes mirrors that reflect light 90 degrees.

    Mirror types:
    '/' : U->R, D->L, L->D, R->U
    '\\': U->L, D->R, L->U, R->D
    """
    rows, cols = len(grid), len(grid[0])
    lit = [[False] * cols for _ in range(rows)]

    DIRECTIONS = {
        'U': (-1, 0),
        'D': (1, 0),
        'L': (0, -1),
        'R': (0, 1),
    }

    # Reflection mappings
    REFLECT_SLASH = {'U': 'R', 'D': 'L', 'L': 'D', 'R': 'U'}    # '/'
    REFLECT_BACK = {'U': 'L', 'D': 'R', 'L': 'U', 'R': 'D'}     # '\'

    EMPTY = '.'
    WALL = '#'
    LIGHTHOUSE = 'L'
    MIRROR_SLASH = '/'
    MIRROR_BACK = '\\'

    visited = set()
    queue = deque()

    # Initialize lighthouses
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == LIGHTHOUSE:
                lit[r][c] = True
                for dir_name in DIRECTIONS:
                    queue.append((r, c, dir_name))
                    visited.add((r, c, dir_name))

    while queue:
        r, c, direction = queue.popleft()

        dr, dc = DIRECTIONS[direction]
        nr, nc = r + dr, c + dc

        if not (0 <= nr < rows and 0 <= nc < cols):
            continue

        cell = grid[nr][nc]

        if cell == WALL:
            continue

        lit[nr][nc] = True

        # Determine next direction
        if cell == MIRROR_SLASH:
            next_dir = REFLECT_SLASH[direction]
        elif cell == MIRROR_BACK:
            next_dir = REFLECT_BACK[direction]
        else:
            next_dir = direction

        state = (nr, nc, next_dir)
        if state not in visited:
            visited.add(state)
            queue.append(state)

    return lit
```

---

**Variation: Rotating Lighthouses**

```python
def simulate_rotating(
    grid: list[list[int]],
    lighthouse_dirs: dict[tuple[int, int], str],
    steps: int
) -> list[list[set]]:
    """
    Lighthouses rotate 90° clockwise each step.
    Track which cells are lit at each step.

    lighthouse_dirs: {(r, c): initial_direction}
    """
    rows, cols = len(grid), len(grid[0])

    DIRECTIONS = ['U', 'R', 'D', 'L']  # Clockwise order
    DIR_VECTORS = {'U': (-1, 0), 'R': (0, 1), 'D': (1, 0), 'L': (0, -1)}

    def rotate_cw(d):
        idx = DIRECTIONS.index(d)
        return DIRECTIONS[(idx + 1) % 4]

    def cast_single_ray(r, c, direction):
        """Cast ray in one direction, return lit cells."""
        lit_cells = set()
        dr, dc = DIR_VECTORS[direction]
        r, c = r + dr, c + dc

        while 0 <= r < rows and 0 <= c < cols and grid[r][c] != 1:
            lit_cells.add((r, c))
            r += dr
            c += dc

        return lit_cells

    results = []  # List of lit cells per step

    current_dirs = lighthouse_dirs.copy()

    for step in range(steps):
        lit = set()

        for (r, c), direction in current_dirs.items():
            lit.add((r, c))
            lit.update(cast_single_ray(r, c, direction))

        results.append(lit)

        # Rotate all lighthouses
        current_dirs = {pos: rotate_cw(d) for pos, d in current_dirs.items()}

    return results
```

---

**Helper: Count Lit Cells**

```python
def count_lit_cells(grid: list[list[int]]) -> int:
    """Count total cells that receive light."""
    lit = simulate_light(grid)
    return sum(sum(row) for row in lit)


def get_lit_positions(grid: list[list[int]]) -> list[tuple[int, int]]:
    """Return list of (row, col) positions that receive light."""
    lit = simulate_light(grid)
    return [(r, c) for r in range(len(lit))
                   for c in range(len(lit[0])) if lit[r][c]]
```

---

**Complexity Analysis:**

| Approach | Time | Space | Use Case |
|----------|------|-------|----------|
| Ray Casting | O(L × max(R,C)) | O(R×C) | Simple, no reflections |
| Directional BFS | O(R×C×D) | O(R×C×D) | With reflections |
| Rotating | O(S×L×max(R,C)) | O(R×C) | Time-varying simulation |

L = lighthouses, R = rows, C = cols, D = directions, S = steps

**Common Bugs:**
1. Off-by-one in boundary checks
2. Forgetting to mark lighthouse cell itself as lit
3. Infinite loop with mirrors (cycle detection needed)
4. Not handling light passing through lighthouses

**Testing:**
```python
# Test 1: Simple horizontal beam
grid = [
    [2, 0, 0, 1, 0],  # Lighthouse at (0,0), wall at (0,3)
]
# Expected: cells (0,0), (0,1), (0,2) lit

# Test 2: Multiple lighthouses, overlapping beams
grid = [
    [2, 0, 0],
    [0, 0, 0],
    [0, 0, 2],
]
# Both lighthouses contribute

# Test 3: Lighthouse blocked immediately
grid = [
    [1, 2, 1],  # Walls on both sides
]
# Only lighthouse cell lit
```

---

### Convert BST to Sorted Doubly Linked List (LC 426)

**Problem:** Convert a BST into a sorted **circular** doubly-linked list **in place**.
- `left` pointer → predecessor
- `right` pointer → successor
- Return the head (smallest element)
- The list should be circular (head.left = tail, tail.right = head)

**Example:**
```
        4
       / \
      2   5
     / \
    1   3

Output: 1 <-> 2 <-> 3 <-> 4 <-> 5 (circular)
        ^                       |
        |_______________________|
```

**Solution 1: Recursive In-Order Traversal**

*Main logic:* an **in-order** traversal visits BST nodes in sorted order, so thread each visited node to its predecessor as you go (`prev.right = node`, `node.left = prev`). Keep the first and last nodes, then close the ring at the end (`head.left = tail`, `tail.right = head`).

```python
class Node:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left   # predecessor in DLL
        self.right = right # successor in DLL


def treeToDoublyList(root: Node) -> Node:
    """
    In-order traversal with prev pointer to link nodes.

    Key: Process left subtree BEFORE modifying current node's pointers,
    since the tree structure is destroyed during conversion.

    Time: O(n)
    Space: O(h) recursion stack, where h = height
    """
    if not root:
        return None

    # Track first (head) and last (prev) nodes
    first = None
    prev = None

    def inorder(node):
        nonlocal first, prev

        if not node:
            return

        # Process left subtree first (before modifying pointers!)
        inorder(node.left)

        # Process current node
        if prev:
            # Link prev <-> current
            prev.right = node
            node.left = prev
        else:
            # First node (smallest) - save as head
            first = node

        prev = node

        # Process right subtree
        inorder(node.right)

    inorder(root)

    # Close the circular link: connect head <-> tail
    first.left = prev
    prev.right = first

    return first
```

**Solution 2: Iterative In-Order (Explicit Stack)**

```python
def treeToDoublyList(root: Node) -> Node:
    """
    Iterative in-order using stack.
    Useful when recursion depth is a concern (skewed tree).

    Time: O(n)
    Space: O(h) for stack
    """
    if not root:
        return None

    first = None
    prev = None
    stack = []
    current = root

    while stack or current:
        # Go to leftmost node
        while current:
            stack.append(current)
            current = current.left

        # Process current node
        current = stack.pop()

        if prev:
            prev.right = current
            current.left = prev
        else:
            first = current

        prev = current

        # Move to right subtree
        current = current.right

    # Close circular link
    first.left = prev
    prev.right = first

    return first
```

**Walkthrough:**
```
        4
       / \
      2   5
     / \
    1   3

In-order: 1, 2, 3, 4, 5

Step 1: Visit 1 (leftmost)
  first = 1, prev = 1

Step 2: Visit 2
  prev(1).right = 2, 2.left = 1
  prev = 2
  Result: 1 <-> 2

Step 3: Visit 3
  prev(2).right = 3, 3.left = 2
  prev = 3
  Result: 1 <-> 2 <-> 3

Step 4: Visit 4
  prev(3).right = 4, 4.left = 3
  prev = 4
  Result: 1 <-> 2 <-> 3 <-> 4

Step 5: Visit 5
  prev(4).right = 5, 5.left = 4
  prev = 5
  Result: 1 <-> 2 <-> 3 <-> 4 <-> 5

Close circular:
  first(1).left = prev(5)
  prev(5).right = first(1)
```

---

**Follow-up: Insert into Circular Sorted DLL**

```python
def insert(head: Node, new_val: int) -> Node:
    """
    Insert a new node into circular sorted DLL while maintaining order.

    Cases:
    1. Empty list: create single-node circular list
    2. Insert at head (new_val <= head.val)
    3. Insert at tail (new_val >= tail.val)
    4. Insert in middle (find position where prev.val <= new_val <= next.val)

    Time: O(n) - linear search
    Space: O(1)
    """
    new_node = Node(new_val)

    # Case 1: Empty list
    if not head:
        new_node.left = new_node
        new_node.right = new_node
        return new_node

    # Find insertion point
    prev = head
    curr = head.right

    while curr != head:
        # Found position: prev <= new_val <= curr
        if prev.val <= new_val <= curr.val:
            break

        # Handle wrap-around (insert at tail/head boundary)
        # This is when we're at the max->min transition
        if prev.val > curr.val:  # At the wrap point
            if new_val >= prev.val or new_val <= curr.val:
                break

        prev = curr
        curr = curr.right

    # Insert between prev and curr
    new_node.left = prev
    new_node.right = curr
    prev.right = new_node
    curr.left = new_node

    # Update head if new node is smallest
    if new_val < head.val:
        return new_node

    return head


def insert_simple(head: Node, new_val: int) -> Node:
    """
    Simplified version: always find correct position.

    Time: O(n)
    Space: O(1)
    """
    new_node = Node(new_val)

    if not head:
        new_node.left = new_node
        new_node.right = new_node
        return new_node

    # Find tail (node before head in circular list)
    tail = head.left

    # Case: insert before head (new smallest)
    if new_val <= head.val:
        new_node.right = head
        new_node.left = tail
        tail.right = new_node
        head.left = new_node
        return new_node

    # Case: insert after tail (new largest)
    if new_val >= tail.val:
        new_node.left = tail
        new_node.right = head
        tail.right = new_node
        head.left = new_node
        return head

    # Case: insert in middle - find position
    curr = head
    while curr.right != head and curr.right.val < new_val:
        curr = curr.right

    # Insert after curr
    new_node.left = curr
    new_node.right = curr.right
    curr.right.left = new_node
    curr.right = new_node

    return head
```

**Insert Examples:**
```
List: 1 <-> 3 <-> 5 (circular)

insert(head, 2):
  Find: 1.val <= 2 <= 3.val
  Result: 1 <-> 2 <-> 3 <-> 5

insert(head, 0):
  New smallest, becomes new head
  Result: 0 <-> 1 <-> 2 <-> 3 <-> 5

insert(head, 6):
  New largest, insert at tail
  Result: 0 <-> 1 <-> 2 <-> 3 <-> 5 <-> 6
```

---

**Common Bugs:**
1. **Forgetting empty tree check** - return None
2. **Forgetting circular link** - must connect head.left = tail and tail.right = head
3. **Modifying pointers before recursive call** - tree structure destroyed mid-traversal
4. **Insert wrap-around case** - circular list has no natural "end"

**Complexity:**

| Operation | Time | Space |
|-----------|------|-------|
| treeToDoublyList | O(n) | O(h) |
| insert | O(n) | O(1) |

**Edge Cases:**
```python
# Empty tree
root = None → return None

# Single node
root = Node(1) → circular self-loop: 1.left = 1.right = 1

# Skewed tree (all left or all right)
# Recursion depth = n, consider iterative for large trees
```

---

### Escape Room / Room-by-Room Race (Pinterest Custom)

**Problem:** Design a game-state data structure with `n` rooms and `m` players starting in room 0.

```python
Game(n_rooms, n_players)              # All players start in room 0
void proceedToNextRoom(playerId)      # Player advances by one room
int  getPeople(roomId)                # Current headcount in roomId
vector<int> getTop(k)                 # Top-k player IDs by rank
```

**Ranking Rules:**
1. Higher-numbered room = higher rank
2. Same room: earlier arrival ranks higher (FIFO tiebreaker)

**Complexity Targets:**
- `proceedToNextRoom(playerId)` → O(1) amortized
- `getPeople(roomId)` → O(1)
- `getTop(k)` → O(rooms + k) or O(k)

**Example:**
```
Game(5, 3)
proceedToNextRoom(0)  // player 0: room 0 → 1
proceedToNextRoom(1)  // player 1: room 0 → 1
proceedToNextRoom(2)  // player 2: room 0 → 1
proceedToNextRoom(0)  // player 0: room 1 → 2

getTop(2)             // [0, 1] — player 0 in room 2, player 1 arrived in room 1 before player 2
getPeople(1)          // 2 (players 1 and 2)
getPeople(2)          // 1 (player 0)
```

**Solution: Doubly-Linked List Per Room + Player Map**

*Main logic:* keep a **doubly-linked list per room** in FIFO arrival order, plus a `player → node` map. `proceedToNextRoom` unlinks the player from its current room list and appends to the next room's list (O(1)); `getPeople` reads a per-room size counter; `getTop(k)` scans rooms from highest to lowest, taking players in arrival order (higher room, then earlier arrival, ranks first).

```python
class ListNode:
    """Node in doubly-linked list for a room."""
    def __init__(self, player_id: int):
        self.player_id = player_id
        self.prev = None
        self.next = None


class RoomList:
    """Doubly-linked list tracking players in arrival order (FIFO)."""
    def __init__(self):
        # Sentinel nodes for easier insert/remove
        self.head = ListNode(-1)  # dummy head
        self.tail = ListNode(-1)  # dummy tail
        self.head.next = self.tail
        self.tail.prev = self.head
        self.count = 0

    def append(self, node: ListNode) -> None:
        """Add player to end of list (most recent arrival). O(1)"""
        node.prev = self.tail.prev
        node.next = self.tail
        self.tail.prev.next = node
        self.tail.prev = node
        self.count += 1

    def remove(self, node: ListNode) -> None:
        """Remove player from list. O(1)"""
        node.prev.next = node.next
        node.next.prev = node.prev
        node.prev = None
        node.next = None
        self.count -= 1

    def __len__(self) -> int:
        return self.count

    def __iter__(self):
        """Iterate players in arrival order (head to tail)."""
        curr = self.head.next
        while curr != self.tail:
            yield curr.player_id
            curr = curr.next


class Game:
    def __init__(self, n_rooms: int, n_players: int):
        self.n_rooms = n_rooms
        self.n_players = n_players

        # Per-room doubly-linked list of players (arrival order)
        self.rooms = [RoomList() for _ in range(n_rooms)]

        # Player -> (current_room, ListNode)
        self.player_info = {}

        # Initialize: all players in room 0
        for player_id in range(n_players):
            node = ListNode(player_id)
            self.rooms[0].append(node)
            self.player_info[player_id] = (0, node)

    def proceedToNextRoom(self, player_id: int) -> None:
        """
        Move player to the next room. O(1)
        """
        current_room, node = self.player_info[player_id]

        # Check if already in last room
        if current_room >= self.n_rooms - 1:
            return  # Can't proceed further

        next_room = current_room + 1

        # Remove from current room's list
        self.rooms[current_room].remove(node)

        # Add to next room's list (at tail = most recent arrival)
        self.rooms[next_room].append(node)

        # Update player info
        self.player_info[player_id] = (next_room, node)

    def getPeople(self, room_id: int) -> int:
        """
        Get headcount in a room. O(1)
        """
        if 0 <= room_id < self.n_rooms:
            return len(self.rooms[room_id])
        return 0

    def getTop(self, k: int) -> list[int]:
        """
        Get top-k players by rank. O(rooms + k)

        Walk rooms from highest to lowest, emit players in arrival order.
        """
        result = []

        # Iterate from highest room to lowest
        for room_id in range(self.n_rooms - 1, -1, -1):
            for player_id in self.rooms[room_id]:
                result.append(player_id)
                if len(result) >= k:
                    return result

        return result
```

**Optimized Version: Track Non-Empty Rooms**

For sparse rooms (many empty), maintain a sorted set of non-empty room IDs:

```python
from sortedcontainers import SortedList

class GameOptimized:
    def __init__(self, n_rooms: int, n_players: int):
        self.n_rooms = n_rooms
        self.rooms = [RoomList() for _ in range(n_rooms)]
        self.player_info = {}

        # Track non-empty rooms (sorted in descending order for leaderboard)
        self.non_empty_rooms = SortedList()

        # Initialize all players in room 0
        for player_id in range(n_players):
            node = ListNode(player_id)
            self.rooms[0].append(node)
            self.player_info[player_id] = (0, node)

        if n_players > 0:
            self.non_empty_rooms.add(0)

    def proceedToNextRoom(self, player_id: int) -> None:
        """O(log N) where N = non-empty rooms, due to sorted set operations."""
        current_room, node = self.player_info[player_id]

        if current_room >= self.n_rooms - 1:
            return

        next_room = current_room + 1

        # Remove from current room
        self.rooms[current_room].remove(node)

        # Update non-empty rooms tracking
        if len(self.rooms[current_room]) == 0:
            self.non_empty_rooms.remove(current_room)

        # Add to next room
        if len(self.rooms[next_room]) == 0:
            self.non_empty_rooms.add(next_room)

        self.rooms[next_room].append(node)
        self.player_info[player_id] = (next_room, node)

    def getPeople(self, room_id: int) -> int:
        """O(1)"""
        return len(self.rooms[room_id]) if 0 <= room_id < self.n_rooms else 0

    def getTop(self, k: int) -> list[int]:
        """O(N + k) where N = number of non-empty rooms."""
        result = []

        # Iterate non-empty rooms from highest to lowest
        for room_id in reversed(self.non_empty_rooms):
            for player_id in self.rooms[room_id]:
                result.append(player_id)
                if len(result) >= k:
                    return result

        return result
```

**Walkthrough:**
```
Game(5, 3)  # rooms 0-4, players 0-2

Initial state:
  Room 0: [0, 1, 2] (all players, arrival order)
  Room 1-4: []
  player_info: {0: (0, node0), 1: (0, node1), 2: (0, node2)}

proceedToNextRoom(0):
  Remove player 0 from room 0, add to room 1
  Room 0: [1, 2]
  Room 1: [0]

proceedToNextRoom(1):
  Room 0: [2]
  Room 1: [0, 1]

proceedToNextRoom(2):
  Room 0: []
  Room 1: [0, 1, 2]

proceedToNextRoom(0):
  Room 1: [1, 2]
  Room 2: [0]

getTop(2):
  Start from room 4 (empty) → room 3 (empty) → room 2: yield 0
  → room 1: yield 1
  Result: [0, 1]

getPeople(1) → 2
getPeople(2) → 1
```

**Common Bugs to Avoid:**
1. **Shared list node state** - Each room must have independent doubly-linked list
2. **Forgetting to update both prev and next pointers** on remove/insert
3. **Not handling boundary** (player in last room trying to proceed)
4. **Off-by-one in room iteration** for getTop

**Complexity Summary:**

| Operation | Basic | Optimized |
|-----------|-------|-----------|
| proceedToNextRoom | O(1) | O(log N)* |
| getPeople | O(1) | O(1) |
| getTop(k) | O(rooms + k) | O(N + k)** |

*N = non-empty rooms (due to sorted set)
**N = non-empty rooms only

**Simpler Phone Screen Variant (No Leaderboard):**

```python
class GameSimple:
    """Phone screen version: just proceedToNextRoom + getPeople."""

    def __init__(self, n_rooms: int, n_players: int):
        self.room_count = [0] * n_rooms
        self.room_count[0] = n_players
        self.player_room = {i: 0 for i in range(n_players)}
        self.n_rooms = n_rooms

    def proceedToNextRoom(self, player_id: int) -> None:
        current = self.player_room[player_id]
        if current < self.n_rooms - 1:
            self.room_count[current] -= 1
            self.room_count[current + 1] += 1
            self.player_room[player_id] = current + 1

    def getPeople(self, room_id: int) -> int:
        return self.room_count[room_id]
```

---

### Cheapest Flights Within K Stops (LC 787)

**Problem:** Find the cheapest price from `src` to `dst` with at most `k` stops (k+1 edges). Return -1 if no such route exists.

**Example 1:**
```
n = 4, flights = [[0,1,100],[1,2,100],[2,0,100],[1,3,600],[2,3,200]]
src = 0, dst = 3, k = 1

    100     600
0 -----> 1 -----> 3
|        |        ^
|100     |100     |200
v        v        |
2 <----- 2 -------+

Output: 700 (0 -> 1 -> 3)
With k=1 stop, can only use 2 flights
```

**Example 2:**
```
Same graph, but k = 2
Output: 400 (0 -> 1 -> 2 -> 3)
With k=2 stops, can use 3 flights
```

**Solution 1: BFS (Level-by-Level)**

*Main logic:* BFS where **each level = one more flight**, so after `k + 1` levels you stop. Track the cheapest cost seen per city and only relax a neighbor when a cheaper cost arrives — this is Bellman-Ford bounded to `k + 1` edges.

```python
from collections import defaultdict, deque

def findCheapestPrice(n: int, flights: list[list[int]], src: int, dst: int, k: int) -> int:
    """
    BFS level-by-level where each level = one flight.
    Track minimum cost to reach each node.

    Time: O(k * E) where E = number of flights
    Space: O(n)
    """
    graph = defaultdict(list)
    for u, v, price in flights:
        graph[u].append((v, price))

    # prices[i] = minimum cost to reach node i
    prices = [float('inf')] * n
    prices[src] = 0

    queue = deque([(src, 0)])  # (node, cost)
    stops = 0

    while queue and stops <= k:
        # Process all nodes at current level (same number of stops)
        for _ in range(len(queue)):
            node, cost = queue.popleft()

            for neighbor, price in graph[node]:
                new_cost = cost + price

                # Only add to queue if we found a better price
                if new_cost < prices[neighbor]:
                    prices[neighbor] = new_cost
                    queue.append((neighbor, new_cost))

        stops += 1

    return prices[dst] if prices[dst] != float('inf') else -1
```

> **Caveat:** this BFS keeps a single global `prices[]` and prunes with `new_cost < prices[neighbor]`. Pruning by a global best-cost (rather than best-cost *per stop count*) can occasionally block a valid fewer-stops path that a node already reached more cheaply with more stops. The level-by-level loop usually saves it, but the **Bellman-Ford temp-array version (Solution 2/4)** and the **stop-aware Dijkstra (Solution 3)** are the canonical, provably-correct choices — prefer those.

**Solution 2: Bellman-Ford (K+1 Relaxations)**

```python
def findCheapestPrice(n: int, flights: list[list[int]], src: int, dst: int, k: int) -> int:
    """
    Bellman-Ford limited to k+1 iterations (at most k+1 edges).

    Key: Use previous iteration's prices to avoid using more edges than allowed.

    Time: O(k * E)
    Space: O(n)
    """
    # prices[i] = min cost to reach node i
    prices = [float('inf')] * n
    prices[src] = 0

    # Relax edges k+1 times (k stops = k+1 edges)
    for _ in range(k + 1):
        # Use copy to ensure we only use paths from previous iteration
        temp = prices[:]

        for u, v, price in flights:
            if prices[u] != float('inf'):
                temp[v] = min(temp[v], prices[u] + price)

        prices = temp

    return prices[dst] if prices[dst] != float('inf') else -1
```

**Solution 3: Modified Dijkstra**

```python
import heapq
from collections import defaultdict

def findCheapestPrice(n: int, flights: list[list[int]], src: int, dst: int, k: int) -> int:
    """
    Dijkstra's with stops constraint.

    Key difference from standard Dijkstra: A node can be visited multiple times
    with different stop counts (might find cheaper path with more stops later).

    Time: O(E * k * log(E * k))
    Space: O(n * k)
    """
    graph = defaultdict(list)
    for u, v, price in flights:
        graph[u].append((v, price))

    # (cost, node, stops_remaining)
    heap = [(0, src, k + 1)]

    # Track minimum stops to reach each node
    # (can revisit with more stops if cheaper path exists)
    visited = {}  # node -> min stops used to reach it with better/equal cost

    while heap:
        cost, node, stops = heapq.heappop(heap)

        if node == dst:
            return cost

        if stops <= 0:
            continue

        # Skip if we've visited this node with more stops remaining
        # (we already found a path that's at least as good)
        if node in visited and visited[node] >= stops:
            continue
        visited[node] = stops

        for neighbor, price in graph[node]:
            heapq.heappush(heap, (cost + price, neighbor, stops - 1))

    return -1
```

**Solution 4: Dynamic Programming**

```python
def findCheapestPrice(n: int, flights: list[list[int]], src: int, dst: int, k: int) -> int:
    """
    DP: dp[t][i] = min cost to reach node i using exactly t flights.

    Time: O(k * E)
    Space: O(n) with space optimization
    """
    # dp[i] = min cost to reach node i
    dp = [float('inf')] * n
    dp[src] = 0

    for _ in range(k + 1):
        temp = dp[:]
        for u, v, price in flights:
            if dp[u] != float('inf'):
                temp[v] = min(temp[v], dp[u] + price)
        dp = temp

    return dp[dst] if dp[dst] != float('inf') else -1
```

**Comparison:**

| Approach | Time | Space | Notes |
|----------|------|-------|-------|
| BFS | O(k * E) | O(n) | Level-by-level, intuitive |
| Bellman-Ford | O(k * E) | O(n) | Simple, uses temp array |
| Dijkstra | O(Ek log Ek) | O(nk) | Fastest for sparse graphs |
| DP | O(k * E) | O(n) | Same as Bellman-Ford |

**Key Insight: Why Standard Dijkstra Fails**

Standard Dijkstra marks nodes as visited once. But here, a path with more stops might be cheaper:
```
src=0, dst=2, k=2
0 --500--> 2 (1 flight, cost 500)
0 --100--> 1 --100--> 2 (2 flights, cost 200)

Standard Dijkstra might visit node 2 first with cost 500,
then never explore the cheaper path through node 1.
```

**Edge Cases:**
- `src == dst`: Return 0
- No path exists: Return -1
- k = 0: Direct flight only

---

### Odd Even Linked List (LC 328)

**Problem:** Given the head of a singly linked list, group all nodes at odd indices together followed by nodes at even indices. The first node is odd (index 1), second is even (index 2), etc. Preserve relative order within each group.

**Example 1:**
```
Input:  1 -> 2 -> 3 -> 4 -> 5
Output: 1 -> 3 -> 5 -> 2 -> 4

Odd indices:  1, 3, 5 (positions 1, 3, 5)
Even indices: 2, 4    (positions 2, 4)
```

**Example 2:**
```
Input:  2 -> 1 -> 3 -> 5 -> 6 -> 4 -> 7
Output: 2 -> 3 -> 6 -> 7 -> 1 -> 5 -> 4
```

**Solution: Two Pointers**

*Main logic:* weave two chains in a single pass — an `odd` pointer and an `even` pointer each advancing by two. Remember the even head, then splice the even chain onto the tail of the odd chain. O(n) time, O(1) space.

```python
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

def oddEvenList(head: ListNode) -> ListNode:
    """
    Build two separate lists (odd and even), then connect them.

    Time: O(n) - single pass
    Space: O(1) - only pointers, no extra nodes
    """
    if not head or not head.next:
        return head

    odd = head          # First node (odd index)
    even = head.next    # Second node (even index)
    even_head = even    # Save to connect later

    while even and even.next:
        # Link odd to next odd (skip one)
        odd.next = even.next
        odd = odd.next

        # Link even to next even (skip one)
        even.next = odd.next
        even = even.next

    # Connect odd list to even list
    odd.next = even_head

    return head
```

**Walkthrough:**
```
Initial: 1 -> 2 -> 3 -> 4 -> 5
         ^    ^
        odd  even
        even_head = 2

Step 1: odd.next = 3, odd = 3
        even.next = 4, even = 4
        1 -> 3 -> 4 -> 5
             ^    ^
            odd  even
        2 -> 4 (even list)

Step 2: odd.next = 5, odd = 5
        even.next = None, even = None
        1 -> 3 -> 5
                  ^
                 odd
        2 -> 4 -> None (even list)

Connect: odd.next = even_head
         1 -> 3 -> 5 -> 2 -> 4
```

**Why `while even and even.next`?**
- `even` check: handles odd-length lists (even becomes None)
- `even.next` check: ensures there's a next odd node to process

**Edge Cases:**
```python
# Empty list
head = None → return None

# Single node
head = [1] → return [1]

# Two nodes
head = [1, 2] → return [1, 2]  # Already odd, even order
```

**Common Mistakes:**
1. Forgetting to save `even_head` before modifying pointers
2. Wrong loop condition (causes null pointer)
3. Not handling edge cases (empty, single node)

**Complexity:**
- Time: O(n) - visit each node once
- Space: O(1) - only use pointers

---

### Reconstruct Itinerary (LC 332)

**Problem:** Given a list of airline tickets `[from, to]`, reconstruct the itinerary starting from "JFK". Return the itinerary with the smallest lexical order. All tickets must be used exactly once.

**Example 1:**
```
Input: tickets = [["MUC","LHR"],["JFK","MUC"],["SFO","SJC"],["LHR","SFO"]]
Output: ["JFK","MUC","LHR","SFO","SJC"]
```

**Example 2:**
```
Input: tickets = [["JFK","SFO"],["JFK","ATL"],["SFO","ATL"],["ATL","JFK"],["ATL","SFO"]]
Output: ["JFK","ATL","JFK","SFO","ATL","SFO"]

Explanation: ["JFK","ATL","JFK","SFO","ATL","SFO"] < ["JFK","SFO","ATL","JFK","ATL","SFO"]
```

**Key Insight:** This is finding an **Eulerian Path** (visit every edge exactly once). Use **Hierholzer's Algorithm**.

**Why Hierholzer's Algorithm?**
- Greedy DFS might get stuck at dead ends
- Solution: Build path in reverse (post-order), add node to result when backtracking
- When stuck (no outgoing edges), that node must be the end of the path

**Solution 1: Hierholzer's Algorithm (Recursive)**

*Main logic:* this is an **Eulerian path** (use every ticket exactly once). Sort each origin's destinations so DFS explores them lexicographically, and add a node to the result **in post-order** — only once it has no unused outgoing edges — then reverse at the end. Post-order is what makes dead-ends land in the right place.

```python
from collections import defaultdict

def findItinerary(tickets: list[list[str]]) -> list[str]:
    """
    Hierholzer's algorithm for Eulerian path.

    Key insight: Add nodes to result in post-order (after visiting all edges).
    This handles dead-ends correctly - they get added to result first.

    Time: O(E log E) for sorting edges
    Space: O(E) for graph and result
    """
    # Build adjacency list with sorted destinations (reverse for pop efficiency)
    graph = defaultdict(list)
    for src, dst in sorted(tickets, reverse=True):
        graph[src].append(dst)

    result = []

    def dfs(airport):
        # Visit all outgoing edges in lexical order
        while graph[airport]:
            next_airport = graph[airport].pop()  # Pop smallest (we sorted reverse)
            dfs(next_airport)
        # Add to result when no more outgoing edges (backtracking)
        result.append(airport)

    dfs("JFK")

    return result[::-1]  # Reverse to get correct order
```

**Solution 2: Iterative with Stack**

```python
from collections import defaultdict

def findItinerary(tickets: list[list[str]]) -> list[str]:
    """
    Iterative version using explicit stack.

    Time: O(E log E)
    Space: O(E)
    """
    # Build graph with sorted destinations (reverse order)
    graph = defaultdict(list)
    for src, dst in sorted(tickets, reverse=True):
        graph[src].append(dst)

    stack = ["JFK"]
    result = []

    while stack:
        # If current airport has outgoing flights, continue DFS
        while graph[stack[-1]]:
            next_airport = graph[stack[-1]].pop()
            stack.append(next_airport)

        # No more outgoing flights, add to result (backtrack)
        result.append(stack.pop())

    return result[::-1]
```

**Walkthrough Example 2:**
```
tickets = [["JFK","SFO"],["JFK","ATL"],["SFO","ATL"],["ATL","JFK"],["ATL","SFO"]]

Graph (sorted reverse):
  JFK -> [SFO, ATL]  (will pop ATL first)
  ATL -> [SFO, JFK]  (will pop JFK first)
  SFO -> [ATL]

DFS trace:
  JFK -> ATL (pop ATL from JFK)
  ATL -> JFK (pop JFK from ATL)
  JFK -> SFO (pop SFO from JFK, JFK now empty)
  SFO -> ATL (pop ATL from SFO, SFO now empty)
  ATL -> SFO (pop SFO from ATL, ATL now empty)
  SFO has no edges -> add SFO to result
  backtrack to ATL -> add ATL to result
  backtrack to SFO -> add SFO to result
  backtrack to JFK -> add JFK to result
  backtrack to ATL -> add ATL to result
  backtrack to JFK -> add JFK to result

result (reverse order built): [SFO, ATL, SFO, JFK, ATL, JFK]
final (reversed): [JFK, ATL, JFK, SFO, ATL, SFO]
```

**Why This Works:**
1. When we have no more outgoing edges, we're at a "dead end"
2. Dead ends must be at the END of the final path
3. By adding nodes post-order, dead ends get added first
4. Reversing gives us the correct path order

**Common Mistakes:**
1. Using simple DFS without post-order addition (gets stuck at dead ends)
2. Not sorting in reverse (efficiency for pop)
3. Forgetting to reverse the result

**Complexity:**
- Time: O(E log E) where E = number of tickets (sorting)
- Space: O(E) for graph storage and result

---

### Violation Log Counter

**Problem:** Given a list of log entries where each entry contains a `timestamp` and `user_id`, implement a system to detect violations.

**Part 1:** Count total actions per user.

**Part 2:** Find all users who performed more than `k` actions within any `t` second window.

**Example:**
```
Input:
logs = [
    (1, "user1"),   # timestamp=1, user_id="user1"
    (2, "user1"),
    (3, "user2"),
    (4, "user1"),
    (5, "user1"),
    (10, "user1"),
    (11, "user2"),
]
k = 3, t = 5

Output: ["user1"]

Explanation:
- user1 has actions at timestamps [1, 2, 4, 5] within window [1, 6] (i.e. [ts, ts+t]) -> 4 actions > 3
- user2 only has 1-2 actions in any 5-second window
```

**Part 1 Solution: HashMap Count**

*Main logic:* Part 1 is a plain per-user counter. Part 2 is a **sliding window** per user over that user's sorted timestamps: advance a left pointer to keep the window within `t` seconds; if any window ever holds more than `k` actions, flag the user.

```python
from collections import defaultdict

def count_actions_per_user(logs: list[tuple[int, str]]) -> dict[str, int]:
    """
    Simple hashmap to count total actions per user.

    Time: O(n)
    Space: O(u) where u = unique users
    """
    counts = defaultdict(int)
    for timestamp, user_id in logs:
        counts[user_id] += 1
    return dict(counts)
```

**Part 2 Solution: Binary Search for Sliding Window**

```python
from collections import defaultdict
import bisect

def find_violating_users(
    logs: list[tuple[int, str]],
    k: int,
    t: int
) -> list[str]:
    """
    Find users with more than k actions in any t-second window.

    Approach:
    1. Group timestamps by user
    2. Sort timestamps for each user
    3. For each timestamp, use binary search to count actions in [ts, ts+t]
    4. If count > k, user is violating

    Time: O(n log n) for sorting + O(n log n) for binary searches
    Space: O(n)
    """
    # Group timestamps by user
    user_timestamps = defaultdict(list)
    for timestamp, user_id in logs:
        user_timestamps[user_id].append(timestamp)

    violating_users = []

    for user_id, timestamps in user_timestamps.items():
        timestamps.sort()

        # Check each timestamp as start of window
        for i, ts in enumerate(timestamps):
            # Find rightmost timestamp <= ts + t
            window_end = ts + t
            right_idx = bisect.bisect_right(timestamps, window_end)

            # Count actions in window [ts, ts+t]
            count = right_idx - i

            if count > k:
                violating_users.append(user_id)
                break  # Found violation, no need to check more

    return violating_users
```

**Alternative: Sliding Window with Two Pointers**

```python
def find_violating_users_two_pointers(
    logs: list[tuple[int, str]],
    k: int,
    t: int
) -> list[str]:
    """
    Two pointers approach - more efficient if checking all windows.

    Time: O(n log n) for sorting, O(n) for scanning
    Space: O(n)
    """
    user_timestamps = defaultdict(list)
    for timestamp, user_id in logs:
        user_timestamps[user_id].append(timestamp)

    violating_users = []

    for user_id, timestamps in user_timestamps.items():
        timestamps.sort()

        left = 0
        max_count = 0

        for right in range(len(timestamps)):
            # Shrink window if too large
            while timestamps[right] - timestamps[left] > t:
                left += 1

            # Current window size
            max_count = max(max_count, right - left + 1)

            if max_count > k:
                violating_users.append(user_id)
                break

    return violating_users
```

**Follow-up: Real-time Violation Detection**

```python
from collections import defaultdict, deque

class ViolationDetector:
    """
    Real-time violation detection with streaming logs.

    Uses deque to maintain sliding window per user.
    """
    def __init__(self, k: int, t: int):
        self.k = k
        self.t = t
        self.user_windows = defaultdict(deque)  # user_id -> deque of timestamps

    def process_log(self, timestamp: int, user_id: str) -> bool:
        """
        Process a log entry and return True if user is now violating.

        Time: O(w) where w = window size, amortized O(1)
        """
        window = self.user_windows[user_id]

        # Remove expired timestamps
        while window and window[0] < timestamp - self.t:
            window.popleft()

        # Add new timestamp
        window.append(timestamp)

        # Check violation
        return len(window) > self.k
```

---

## High-Frequency Coding Problems

A consolidated list of frequently-asked Pinterest coding problems. Already covered above and not repeated here: **LC 332 Reconstruct Itinerary**, **LC 426 Convert BST to DLL** (its insert follow-up is below), **LC 1564 Put Boxes Into Warehouse** (LC 1580 "II" is the *Boxes from Both Ends* variation), and **Escape Room**.

### ACL / Permission System (high frequency)

**Question:** Implement `grant_access(advertiser, group)`, `check_access(advertiser, group)`, `revoke_access(advertiser, group)` over a **group hierarchy** — granting access to a group implies access to everything nested under it.

**Example:**
```
Groups nest like folders:   org_root > brand_X > campaign_A
grant_access("alice", "brand_X")     # alice gets brand_X and everything under it
check_access("alice", "campaign_A")  -> True   (campaign_A is inside brand_X)
check_access("alice", "brand_X")     -> True   (the granted node itself)
check_access("alice", "org_root")    -> False  (org_root is ABOVE the grant, not under it)
revoke_access("alice", "brand_X")
check_access("alice", "campaign_A")  -> False  (grant removed)
```

**Intuition — push the work to read time, walk *up* not *down*:**

The cascade rule ("access to a group implies access to all descendants") tempts you to expand a grant downward — when alice gets `brand_X`, mark `brand_X` and every descendant as granted. But the tree can be huge and changes over time (new campaigns appear under `brand_X` later), so eager expansion is expensive and goes stale.

The trick is to invert it. Store the hierarchy as simple `child → parent` pointers and store each grant as just the single node it was made on. Then `check_access(advertiser, group)` walks **upward** from `group` through its ancestors: if the advertiser was granted *any* ancestor (or the node itself), access is allowed. "Granted an ancestor" is exactly equivalent to "the target is in that ancestor's subtree" — so the upward walk reproduces the cascade without ever materializing it. Cost is O(tree depth) per check, and grants/revokes stay O(1).

```python
from collections import defaultdict

class ACL:
    def __init__(self):
        self.parent = {}                    # group -> parent group (hierarchy)
        self.grants = defaultdict(set)      # advertiser -> set of granted groups

    def add_group(self, group, parent=None):
        self.parent[group] = parent

    def grant_access(self, advertiser, group):
        self.grants[advertiser].add(group)

    def revoke_access(self, advertiser, group):
        self.grants[advertiser].discard(group)

    def check_access(self, advertiser, group):
        granted = self.grants[advertiser]
        cur = group
        while cur is not None:              # access via the group or any ancestor
            if cur in granted:
                return True
            cur = self.parent.get(cur)
        return False
```
`check_access` is O(hierarchy depth). If a group can have multiple parents (a DAG), replace the upward walk with a BFS/DFS over ancestors plus a visited set.

**Walkthrough:**
```
Hierarchy (child -> parent):
  campaign_A -> brand_X -> org_root
  campaign_B -> brand_Y -> org_root

grant_access("alice", "brand_X")
check_access("alice", "campaign_A"):
  cur=campaign_A  granted? no  -> up
  cur=brand_X     granted? YES -> True     (inherited from an ancestor)
check_access("alice", "campaign_B"):
  campaign_B -> brand_Y -> org_root -> None,  never granted -> False
revoke_access("alice", "brand_X"); check_access("alice","campaign_A") -> False
```

**Edge cases & gotchas:**
- Granting a leaf grants only that node; granting a high ancestor cascades to its whole subtree — that asymmetry is the entire point of the design.
- `revoke` removes only an *explicit* grant; it cannot subtract access that still flows from a granted ancestor.
- A cycle in the parent map would loop forever — assume a tree (or a DAG, where the visited set also guards against cycles).

### LC 815 — Bus Routes (high frequency)

**Question:** You're given `routes`, where `routes[i]` is the list of stops that bus *i* repeatedly cycles through. You start at `source` and want to reach `target`. Return the **fewest number of buses** you must ride (or `-1` if impossible). Note you may board/exit a bus at any stop on its route.

**Example:**
```
routes = [[1,2,7], [3,6,7]], source = 1, target = 6
Answer: 2
  - Ride bus 0 from stop 1 to stop 7 (1 bus so far).
  - Stop 7 is shared by bus 0 and bus 1, so transfer to bus 1.
  - Ride bus 1 from stop 7 to stop 6 (2 buses total).
```

**The key modeling decision — what is a "node"?**

The naive instinct is to make *stops* the graph nodes and connect adjacent stops. That works but is wasteful: a single route with 1000 stops creates ~1000 edges, and "fewest buses" doesn't map cleanly to "fewest stop-to-stop hops."

The cleaner model: treat each **bus route as a node**. Two routes are connected if they share at least one stop (you can transfer between them there). The answer "fewest buses" then becomes "fewest route-nodes on a path" — a shortest-path-in-an-unweighted-graph problem, which is exactly **BFS**.

We don't even build the route-to-route graph explicitly. Instead we BFS outward from `source`:
- We keep a `stop → set of routes passing through it` index so that, standing at any stop, we instantly know every bus we could board.
- Each BFS "level" = riding one more bus. From the current stop, we board every not-yet-taken route through it; every stop on those routes is now reachable with `buses + 1`.
- The first time we touch `target`, the current level count is the answer (BFS visits nearer levels first, so it's minimal).

The crucial detail: mark **routes** as visited, not just stops. Once you've ridden a bus, re-boarding it can never help — you've already reached all its stops. Forgetting this re-explores entire routes and blows up the runtime.

```python
from collections import defaultdict, deque

def numBusesToDestination(routes, source, target):
    if source == target:
        return 0
    stop_to_routes = defaultdict(set)
    for i, route in enumerate(routes):
        for stop in route:
            stop_to_routes[stop].add(i)

    visited_routes, visited_stops = set(), {source}
    q = deque([(source, 0)])
    while q:
        stop, buses = q.popleft()
        for r in stop_to_routes[stop]:
            if r in visited_routes:
                continue
            visited_routes.add(r)
            for nxt in routes[r]:
                if nxt == target:
                    return buses + 1
                if nxt not in visited_stops:
                    visited_stops.add(nxt)
                    q.append((nxt, buses + 1))
    return -1
```

**Walkthrough:**
```
routes = [[1,2,7],[3,6,7]], source = 1, target = 6
stop_to_routes: 1->{0}, 2->{0}, 7->{0,1}, 3->{1}, 6->{1}

BFS from stop 1 (buses=0):
  board route 0 (stops 1,2,7): none is 6; enqueue 2,7 at buses=1
  pop 2 (buses=1): route 0 already visited -> nothing
  pop 7 (buses=1): board route 1 (stops 3,6,7): 6 == target -> return 2
```
Two buses: route 0 from stop 1 to the shared stop 7, then route 1 to stop 6.

**Edge cases & gotchas:**
- `source == target` returns 0 before any BFS runs.
- Mark **routes** visited, not just stops — re-boarding a fully-explored route is the classic cause of TLE.
- The level counter measures *buses*, so increment when you board a new route, not when you step to a stop.
- Unreachable target returns -1.

### LC 1055 — Shortest Way to Form String (high frequency)

**Question:** Given strings `source` and `target`, you want to build `target` by concatenating **subsequences of `source`**. (A subsequence keeps order but may skip characters — e.g. `"ac"` is a subsequence of `"abc"`.) Return the **minimum number of copies of `source`** whose subsequences, glued together, equal `target`. Return `-1` if it's impossible.

**Example:**
```
source = "abc", target = "abcbc"
Answer: 2
  - From copy 1 of "abc", take the subsequence "abc"  -> covers target[0:3] = "abc"
  - From copy 2 of "abc", take the subsequence "bc"   -> covers target[3:5] = "bc"
  Concatenated: "abc" + "bc" = "abcbc" = target, using 2 copies.

source = "abc", target = "abx" -> -1   ('x' never appears in source)
```

**Intuition — why a simple greedy is optimal:**

Think of laying `target` down left to right, consuming one fresh copy of `source` at a time. Within a single copy you scan `source` once; every time the current `source` char matches the next unmatched `target` char, you "consume" that target char and advance. When you reach the end of `source`, you've extracted the *longest possible* prefix of the remaining target that this one copy can cover — so you start a new copy and continue.

Why is "grab as much as you can per copy" optimal? Because matching greedily never hurts: taking the earliest possible match for each target character leaves the *most* of `source` available for later characters. There's no scenario where deliberately skipping a valid match lets you finish in fewer copies. So greedy = minimal copies.

Feasibility and the bound both fall out naturally: if any `target` character doesn't appear in `source` at all, it can never be matched ⇒ return `-1`. Otherwise every copy advances `target` by at least one character, so you need at most `len(target)` copies.

```python
def shortestWay(source, target):
    source_set = set(source)
    count, i = 0, 0                       # i = index into target
    while i < len(target):
        if target[i] not in source_set:
            return -1
        for ch in source:                 # one pass over source = one copy
            if i < len(target) and ch == target[i]:
                i += 1
        count += 1
    return count
```
**Follow-up (many targets):** precompute, for each char, its sorted positions in `source`, then binary-search the next position — same optimization as the LC 392 follow-up below.

**Walkthrough:**
```
source = "abc", target = "abcbc"

copy 1: scan a,b,c -> a(i:0->1), b(i:1->2), c(i:2->3);  i=3, count=1
copy 2: scan a,b,c -> a != target[3]='b' skip,
                      b matches (i:3->4), c matches (i:4->5);  i=5, count=2
i == len(target) -> answer 2
```
If `target` held a char missing from `source` (e.g. `"abx"`), the `not in source_set` check returns -1 immediately.

**Edge cases & gotchas:**
- Any target char absent from `source` ⇒ impossible (-1); check it up front.
- Worst case is `len(target)` copies — each copy may advance `target` by only one char, which also bounds the loop.
- Empty `target` ⇒ 0 copies.

### Pixie-like Random Walk

**Question:** Given a few "query" pins a user just engaged with, recommend related pins. The data is a **bipartite graph**: pins connect to the boards they're saved on, and boards connect to the pins saved on them (pin → board → pin → …). This is the core of Pinterest's *Pixie* recommender.

**Example:**
```
Pins P0,P5 are both saved on board B1; P5,P3 are both on board B2.
Query = [P0].
A walk P0 -> B1 -> P5 -> B2 -> P3 discovers P5 and P3 as related to P0,
with P5 (one hop away via a shared board) visited most -> ranked highest.
```

**Intuition — "related" = "frequently reachable by a short random walk":**

How do you define which pins are related to the query pins, using only the graph? A clean answer: pins you keep landing on when you wander randomly *starting from the query*. Pins that share many boards with the query (or share boards with pins that do) get visited far more often than unrelated pins, so **visit frequency becomes the relevance score**.

Two refinements make the raw walk practical:
- **Restart probability `alpha`:** at each step, with probability `alpha` teleport back to the query pin. Without this the walk drifts arbitrarily far and starts recommending globally-popular-but-irrelevant pins; restarting tethers the distribution to the neighborhood of the query (this is "random walk *with restart*", aka personalized PageRank).
- **Multi-hit boosting:** when there are several query pins, a candidate reached from *several* of them is a stronger signal than one reached from a single query. Combining per-query counts with a booster like `(Σ √count_q)²` rewards pins that are broadly reachable across the whole query set rather than spiking on one.

```python
import random
from collections import Counter

def pixie_walk(graph, query_pins, num_steps=10000, alpha=0.15, is_pin=lambda n: True):
    """graph: node -> list of neighbors (bipartite pin<->board). Returns ranked pins."""
    counts = Counter()
    for start in query_pins:
        node = start
        for _ in range(num_steps):
            if random.random() < alpha or not graph[node]:
                node = start              # restart toward the query pin
                continue
            node = random.choice(graph[node])
            if is_pin(node):
                counts[node] += 1
    return counts.most_common()
```
Talking points: walk length per query proportional to query-pin degree; combine per-query counts with a booster like `(sum of sqrt(count_q))^2` so broadly-reachable pins win.

**Walkthrough (intuition):**
```
query pin P0 sits on boards B1, B2.
A walk: P0 -> B1 -> P5 -> B1 -> P3 -> (restart) -> P0 -> B2 -> P5 -> ...
Pins sharing boards with P0 (like P5) get revisited -> high counts -> top recs.

With two query pins P0 and P9, a pin reachable from BOTH (a "multi-hit")
is boosted above one reachable from only one query, even at similar raw count.
```

**Edge cases & gotchas:**
- Dangling node (no neighbors) ⇒ force a restart, otherwise the walk gets stuck.
- `alpha` trades locality vs. discovery: higher = stays near the query (more personalized), lower = explores farther.
- Count only *pins*, never boards — boards are just the bipartite bridge between pins.
- It is a sampling method: results are approximate and vary run-to-run; more steps = more stable ranking.

### Expression Add Operators — Left-to-Right (simplified LC 282)

**Question:** Given a string of digits, insert `+ - *` (or nothing) between digits so the resulting expression equals `target`. Return all such expressions. The twist vs. normal math: operators associate **strictly left-to-right with no precedence**, so `2+3*2` is read as `(2+3)*2 = 10`, **not** `2+(3*2)=8`.

**Example:**
```
num = "232", target = 10
One valid expression: "2+3*2"  ->  ((2+3)*2) = 10
  (digits may also be grouped: "23"|"2" gives operands 23 and 2)
```

**Intuition — no precedence means "no memory" backtracking:**

You explore every way to (a) split the digit string into operands and (b) put an operator before each operand after the first. That's a standard DFS: at index `idx`, try every prefix `num[idx:j]` as the next operand, and for each, recurse having applied `+`, `-`, or `*`.

The thing that makes this *easier* than real LC 282 is the left-to-right rule. Because there's no precedence, an operator acts on the **running value immediately** — once you compute it, it's locked in. So multiplication is just `value * operand`, same shape as `value + operand`. You carry only a single number (`value`) down the recursion. (In real LC 282, `*` binds tighter than `+`, so `2+3*2` needs the `3*2` evaluated *before* the `+`; you must remember the last operand and undo it: `value - prev + prev*o`. The left-to-right variant deletes that whole complication.)

```python
def add_operators_ltr(num, target):
    res = []
    def dfs(idx, expr, value):
        if idx == len(num):
            if value == target:
                res.append(expr)
            return
        for j in range(idx + 1, len(num) + 1):
            s = num[idx:j]
            o = int(s)
            if idx == 0:
                dfs(j, s, o)
            else:
                dfs(j, expr + "+" + s, value + o)
                dfs(j, expr + "-" + s, value - o)
                dfs(j, expr + "*" + s, value * o)   # multiplies the running value, not the last term
            if num[idx] == "0":            # no leading-zero multi-digit operands
                break
    dfs(0, "", 0)
    return res
```
Contrast with real LC 282: there, `*` binds tighter, so you must carry the last multiplicand and undo it (`value - prev + prev * o`). Left-to-right removes that bookkeeping entirely.

**Walkthrough:**
```
num = "232", target = 10

split "2" | "3" | "2":  value: 2 -> (2+3)=5 -> (5*2)=10  ✓  expr "2+3*2"
                              2 -> (2+3)=5 -> (5-2)=3
                              2 -> (2*3)=6 -> (6+2)=8 ...

"2+3*2" is read LEFT-TO-RIGHT as (2+3)*2 = 10, NOT 2+(3*2)=8.
```
Real LC 282 would evaluate `2+3*2` as `2+6=8`; here each operator simply transforms the running `value`.

**Edge cases & gotchas:**
- Leading zeros: `if num[idx]=="0": break` permits the single digit `0` but forbids multi-digit operands like `0X`.
- The first operand (`idx == 0`) takes no operator — seed `value` with it directly.
- Don't conflate this with real LC 282's `*`-precedence "undo the last term" trick — the whole simplification is that no previous-term tracking is needed.

### Insert into Sorted Circular Doubly Linked List (LC 708) — LC 426 follow-up

**Question:** Insert a value into a **sorted circular** doubly-linked list (the structure produced by Convert-BST-to-DLL above) so it stays sorted, and return a pointer to the list. You may be given a pointer to *any* node, not necessarily the smallest.

**Example:**
```
Ring (ascending, wraps around):  3 <-> 5 <-> 1 <-> (back to 3)
insert(4):  goes between 3 and 5      -> 3 <-> 4 <-> 5 <-> 1
insert(0):  smaller than the min (1)  -> inserts at the 5->1 seam, before 1
insert(6):  larger than the max (5)   -> inserts at the same seam, after 5
```

**Intuition — one loop, two kinds of valid slots:**

Because the list is circular and sorted, walk pointer pairs `(prev, cur)` once around the ring looking for where `val` fits. There are two situations where `(prev, cur)` is a legal insertion gap:

1. **Normal slot:** `prev.val <= val <= cur.val` — `val` sits between two ascending neighbors.
2. **The wrap seam:** the one place where the ring "rolls over" from its maximum back to its minimum (`prev.val > cur.val`). Any value `>= max` or `<= min` belongs here — that's where a new global-max or global-min is spliced in.

If you complete a full loop without matching either (which happens when *all values are equal*), just insert anywhere — the list stays valid. The empty-list case is special: a lone node points to itself to form a one-element ring.

```python
def insert(head, val):
    node = Node(val)
    if not head:                          # empty -> single-element ring
        node.next = node.prev = node
        return node
    prev, cur = head, head.next
    while True:
        if prev.val <= val <= cur.val:                       # normal slot
            break
        if prev.val > cur.val and (val >= prev.val or val <= cur.val):
            break                          # wrap point (between max and min)
        prev, cur = cur, cur.next
        if prev is head:                  # full loop -> insert before head
            break
    prev.next = node; node.prev = prev
    node.next = cur; cur.prev = node
    return head
```

**Walkthrough:**
```
Ring: 1 <-> 3 <-> 5 <-> (back to 1)

insert(4): prev=1,cur=3 -> 1<=4<=3? no -> advance
           prev=3,cur=5 -> 3<=4<=5? YES -> splice -> 1<->3<->4<->5
insert(6): no normal slot; at prev=5,cur=1 wrap point (5>1 and 6>=5)
           -> insert after max -> 1<->3<->5<->6
insert(0): wrap point (5>1 and 0<=1) -> insert before min -> 0<->1<->3<->5
```

**Edge cases & gotchas:**
- Empty list: point the node at itself (`node.next = node.prev = node`).
- All-equal values (e.g. every node is 2): no `prev <= val <= cur` ever fires — the full-loop guard `prev is head` breaks out and inserts anywhere, which is still valid.
- The wrap-point test handles values *larger than the max* or *smaller than the min*; both belong at the max→min seam.

### LC 465 — Optimal Account Balancing

**Question:** You're given a list of transactions `[lender, borrower, amount]`. Return the **minimum number of transactions** needed to settle all debts so everyone's balance is zero. People can pay anyone, not only their original counterparty.

**Example:**
```
transactions = [[0,1,10], [2,0,5]]
  person0: paid 10 to p1, received 5 from p2  -> net -5
  person1: received 10                        -> net +10
  person2: paid 5                             -> net -5
Net balances: {0: -5, 1: +10, 2: -5}
Answer: 2   (p0 pays p1 5, then p2 pays p1 5 — two transfers settle everyone)
```

**Intuition — forget the transactions, settle the net balances:**

First insight: the original transaction list is a red herring. All that matters is each person's **net balance** (sum owed minus sum owed to them). Two people who exchanged ten payments are equivalent to their single net figure. So collapse everyone to one number and drop the zeros (already settled) — you're left with a multiset of positive (creditors) and negative (debtors) balances that sums to zero.

Second insight: now it's "settle this multiset in the fewest transfers." Each transfer moves money from one person to another and zeroes out *at least one* of them. To minimize transfers you want each one to cancel as much as possible — ideally a debt exactly offsetting a credit. There's no simple greedy that's always optimal (greedily matching biggest-vs-biggest can miss a better grouping), so you **backtrack**: take the first unsettled balance and try cancelling it against every opposite-sign balance, recursing on the rest, and keep the minimum. This is NP-hard in general, but the number of distinct non-zero people is small in practice, and the opposite-sign filter prunes hard.

```python
from collections import defaultdict

def minTransfers(transactions):
    bal = defaultdict(int)
    for a, b, amt in transactions:
        bal[a] -= amt
        bal[b] += amt
    debts = [v for v in bal.values() if v != 0]

    def dfs(i):
        while i < len(debts) and debts[i] == 0:
            i += 1
        if i == len(debts):
            return 0
        best = float("inf")
        for j in range(i + 1, len(debts)):
            if debts[j] * debts[i] < 0:    # opposite signs can cancel
                debts[j] += debts[i]
                best = min(best, 1 + dfs(i + 1))
                debts[j] -= debts[i]
        return best
    return dfs(0)
```

**Walkthrough:**
```
transactions = [[0,1,10],[2,0,5]]
balances: person0: -10+5 = -5,  person1: +10,  person2: -5
debts (nonzero) = [-5, +10, -5]

dfs settles debts[0] = -5 against each opposite-sign debt:
  pair with +10 -> +10 becomes +5; recurse on [0, +5, -5]
     settle +5 against -5 -> both zero; recurse -> 0 more transfers
     => 1 + 1 = 2 transfers
minimum = 2
```

**Edge cases & gotchas:**
- Drop zero balances first — they're settled already and only add branching.
- It's the *net* balance per person that matters, not the raw transaction list (multiple debts collapse to one number).
- NP-hard in general, but the number of distinct debtors is tiny in interviews; the `debts[j]*debts[i] < 0` opposite-sign filter prunes hard.
- A greedy "biggest creditor pays biggest debtor" is **not** always optimal — that's exactly why we backtrack.

### First Word Containing a Prefix

**Question:** Given a list of words and a prefix string, return the index of the **first** word that starts with that prefix, or `-1` if none does.

**Example:**
```
words = ['a', 'apple', 'appz', 'b'], prefix = 'ap'
  'a'      -> too short to start with 'ap'        -> no
  'apple'  -> starts with 'ap'                     -> YES, return index 1
```

**Intuition — the real point is "does a word start with a prefix?":**

For a single query this is almost trivial: scan the list and return the first index where `word.startswith(prefix)`. The reason it shows up in interviews is the implementation detail they want you to get right: a word **shorter than the prefix** must fail. If you compare character by character with a plain `zip(prefix, word)`, the loop stops at the shorter string and you'd wrongly conclude `'a'` matches prefix `'ap'` (you only compared the `'a'`). The fix is `zip_longest(prefix, word, fillvalue='#')`: the prefix's extra characters now pair against a sentinel `'#'` that the word can never equal, so the shorter word correctly fails. (`str.startswith` already handles this, which is why it's the clean answer — the manual version is just to show you understand the edge.)

The follow-up that makes it interesting: **many prefix queries** against the same word list. Then build a **trie** of the words, each node caching the smallest word-index in its subtree; each query walks the prefix (O(prefix length)) and reads off the cached index instead of re-scanning the whole list.

```python
from itertools import zip_longest

def has_prefix(word, prefix):
    return all(wc == pc for pc, wc in zip_longest(prefix, word, fillvalue="#") if pc != "#")

def first_with_prefix(words, prefix):
    for i, w in enumerate(words):
        if w.startswith(prefix):          # or: has_prefix(w, prefix)
            return i
    return -1
```
**For many prefix queries:** build a trie of the words (each node storing the smallest word-index in its subtree) → each prefix lookup is O(prefix length).

**Walkthrough:**
```
words = ['a','apple','appz','b'], prefix = 'ap'
  i=0 'a'      startswith 'ap'? no  (word shorter than prefix)
  i=1 'apple'  startswith 'ap'? YES -> return 1

has_prefix('a','ap') via zip_longest(fillvalue='#'):
  ('a','a') ok; ('p','#') -> 'p' != '#' -> fails  (shorter word correctly rejected)
```

**Edge cases & gotchas:**
- A word *shorter* than the prefix must fail — a naive zip without a fill value stops early and wrongly passes; `zip_longest` with a sentinel fixes it.
- Empty prefix matches the first word (index 0).
- No match ⇒ -1.
- For repeated queries, a trie storing the min word-index per node beats re-scanning the list each time.

### LC 84 — Largest Rectangle in Histogram (monotonic stack)

**Question:** Given `heights`, where `heights[i]` is the height of the *i*-th bar in a histogram (each bar 1 unit wide), find the area of the **largest rectangle** that fits entirely under the bars.

**Example:**
```
heights = [2,1,5,6,2,3]
            _
        _  | |
       | | | | _
       | | | || |
 _     | | | || |
| |_   | | | || |
Largest rectangle = bars 5 and 6 (indices 2,3) at height 5 over width 2 = 10.
```

**Intuition — every bar is the height of *some* rectangle; how wide can it go?**

For each bar, imagine the widest rectangle that uses that bar as its *limiting (shortest) height*. It can extend left and right until it hits a bar shorter than itself. The answer is the max over all bars of `height × that width`. The challenge is computing, for every bar, "how far left/right until something shorter" efficiently.

A **monotonic increasing stack** does it in one pass. We push bar indices while heights keep rising. The moment a bar `h` arrives that's *shorter* than the top of the stack, that top bar can't extend any further right — `h` is its right wall. So we pop it and finalize its rectangle: its height is the popped bar's height, and its width spans from the current index back to the bar now under it on the stack (the nearest shorter bar to the left). We keep popping while the stack top is taller than `h`. A trailing sentinel of height `0` at the end forces every remaining bar to be popped and measured. Each index is pushed and popped once → O(n).

```python
def largestRectangleArea(heights):
    stack = []                            # indices with increasing heights
    best = 0
    for i, h in enumerate(heights + [0]): # sentinel flushes everything
        while stack and heights[stack[-1]] > h:
            height = heights[stack.pop()]
            width = i if not stack else i - stack[-1] - 1
            best = max(best, height * width)
        stack.append(i)
    return best
```

**Walkthrough:**
```
heights = [2,1,5,6,2,3]  (+ trailing sentinel 0)

push 0(h2)
i=1 h=1 < 2: pop idx0 -> h2, width=i=1 -> area 2; push 1(h1)
push 2(h5), push 3(h6)
i=4 h=2: pop idx3(h6) width=4-2-1=1 -> 6
         pop idx2(h5) width=4-1-1=2 -> 10   <- best
         push 4(h2)
push 5(h3)
i=6 h=0 (sentinel): pop everything -> nothing beats 10
answer = 10  (bars of height 5,6 spanning width 2)
```

**Edge cases & gotchas:**
- The trailing `0` sentinel guarantees the stack fully flushes; without it, a strictly increasing histogram never pops.
- Width formula: `i` when the stack empties (rectangle reaches the left edge), else `i - stack[-1] - 1`.
- Popping with `>` (not `>=`) leaves equal-height bars stacked — still correct, because the later equal bar computes the full combined width.
- Empty input / single bar are handled naturally.

### LC 392 — Is Subsequence (two pointers)

**Question:** Return whether `s` is a **subsequence** of `t` — i.e. whether you can delete some (possibly zero) characters from `t`, without reordering, and get `s`.

**Example:**
```
s = "ace", t = "abcde" -> True   ("a_c_e" — keep a, c, e in order)
s = "aec", t = "abcde" -> False  (e appears after c in t, so "aec" can't be formed)
```

**Intuition — greedily match left to right:**

Keep a pointer into `s` and scan `t` once. Every time the current `t` character equals the character `s` is waiting for, advance the `s` pointer. `s` is a subsequence exactly when that pointer reaches the end of `s`. Greedy works because matching the *earliest* available occurrence of each `s` character is never worse — it leaves the most of `t` for the remaining characters. O(|t|) time, O(1) space.

The **follow-up** is where it gets interesting and matches a real Pinterest-scale concern: you have **many** strings `s₁, s₂, …` to test against one large fixed `t`. Re-scanning all of `t` per query is wasteful. Instead preprocess `t` once into `char → sorted list of positions`. For each character of `s`, binary-search the *next* position strictly after the one you last used — O(|s| log |t|) per query (same trick as the LC 1055 follow-up).

```python
def isSubsequence(s, t):
    i = 0
    for c in t:
        if i < len(s) and s[i] == c:
            i += 1
    return i == len(s)
```
**Follow-up (many `s` against one big `t`):** precompute each char's sorted positions in `t`; for each char of `s`, binary-search the next position strictly after the last — O(|s| log |t|) per query (same idea as LC 1055's follow-up).

```python
import bisect
from collections import defaultdict

def make_matcher(t):
    pos = defaultdict(list)
    for i, c in enumerate(t):
        pos[c].append(i)
    def is_subseq(s):
        prev = -1
        for c in s:
            lst = pos.get(c)
            if not lst:
                return False
            j = bisect.bisect_right(lst, prev)   # next position after prev
            if j == len(lst):
                return False
            prev = lst[j]
        return True
    return is_subseq
```

**Walkthrough:**
```
s = "ace", t = "abcde"
  expect 'a': t='a' match -> i=1
  expect 'c': t='b' no, t='c' match -> i=2
  expect 'e': t='d' no, t='e' match -> i=3 == len(s) -> True

Follow-up matcher on the same t:
  pos = {a:[0], b:[1], c:[2], d:[3], e:[4]}
  is_subseq("ace"): a -> 0; c -> first pos > 0 = 2; e -> first pos > 2 = 4 -> True
```

**Edge cases & gotchas:**
- Empty `s` is trivially a subsequence (the pointer is already at the end).
- The follow-up wins when testing many short `s` against one large fixed `t`: O(|s| log|t|) per query after an O(|t|) preprocess, versus an O(|t|) scan each time.
- `bisect_right(lst, prev)` (strictly *after* `prev`) is required — `bisect_left` could reuse the same position twice for repeated characters.

### Weighted Sampling (softmax → CDF + binary search)

**Question:** Given a list of `logits` (raw model scores), sample an index so that index `i` is chosen with probability `softmax(logits)[i]` — i.e. proportional to `exp(logit_i)`. Support drawing many samples efficiently.

**Example:**
```
logits = [1.0, 2.0, 3.0]
softmax  ≈ [0.09, 0.245, 0.665]   (index 2 is most likely, ~66% of draws)
Over many calls, sample() returns 0/1/2 in roughly 9% / 24% / 66% of cases.
```

**Intuition — turn probabilities into a number line, then binary-search:**

Picture the probabilities laid end to end on `[0, 1)`: index 0 owns `[0, 0.09)`, index 1 owns `[0.09, 0.335)`, index 2 owns `[0.335, 1.0)`. The width of each segment *is* its probability. So if you draw a uniform `u ∈ [0, 1)` and find which segment it lands in, you've sampled with exactly the right probabilities. Those segment boundaries are the **cumulative distribution (CDF)** — a sorted array — so "which segment" is a **binary search** for the first CDF entry `≥ u`.

Two details interviewers look for:
- **Numerical stability:** subtract `max(logits)` before exponentiating. `exp` of a large logit overflows; subtracting the max shifts everything to `≤ 0` (so `exp ≤ 1`) without changing the softmax ratios.
- **Build once, sample many:** the CDF costs O(n) to build but then each draw is O(log n). Rebuilding it on every sample is the classic inefficiency.

```python
import math, bisect, random

def build_sampler(logits):
    m = max(logits)
    exps = [math.exp(x - m) for x in logits]   # subtract max for stability
    total = sum(exps)
    cdf, acc = [], 0.0
    for e in exps:
        acc += e / total
        cdf.append(acc)
    def sample():
        return bisect.bisect_left(cdf, random.random())   # index ∝ softmax(logits)
    return sample
```

**Walkthrough:**
```
logits = [1.0, 2.0, 3.0]
shift by max=3:  exps = [e^-2, e^-1, e^0] = [0.135, 0.368, 1.0], total = 1.503
probs ≈ [0.090, 0.245, 0.665]
cdf   ≈ [0.090, 0.335, 1.000]

sample(): draw u in [0,1)
  u=0.05 -> bisect_left -> index 0
  u=0.20 -> index 1
  u=0.50 -> index 2
fraction of u landing in each bucket == that bucket's probability.
```

**Edge cases & gotchas:**
- Subtract `max(logits)` before `exp` — without it, large logits overflow `math.exp`.
- Build the CDF once and reuse `sample()` for many draws (O(log n) each); rebuilding it per draw is the common inefficiency.
- Floating-point: the last CDF entry should be ~1.0, and since `random()` < 1 you can't index past the end.
- `bisect_left` vs `bisect_right` both work here — landing exactly on a CDF boundary has probability zero.

### LC 1526 — Minimum Operations to Form a Target Array

**Question:** You start with an all-zero array. Each operation picks a contiguous subarray and adds 1 to every element of it. Return the **minimum number of operations** to turn the zero array into `target`.

**Example:**
```
target = [3,1,1,2]
Answer: 4
  e.g. +1 over [0..3] three?  No — think of it as horizontal layers:
  height profile  3,1,1,2  needs 4 "stripe" operations total.
```

**Intuition — count only the *rises* (think in differences):**

Visualize `target` as a bar chart and each operation as painting a horizontal stripe over a contiguous run. To build up bar `i`, you only need *new* stripes when it's **taller than its left neighbor** — if it's the same height or shorter, the stripes covering the taller neighbor already extend into it (you just stop some of them, which is free).

So the answer is `target[0]` (the stripes needed to raise the first bar from zero) plus, for every later position, `max(0, target[i] - target[i-1])` — the extra height you have to add beyond what's already flowing in from the left. Equivalently: `target[0]` + the sum of all positive consecutive differences. O(n), no DP needed.

```python
def minNumberOperations(target):
    ops = target[0]
    for i in range(1, len(target)):
        if target[i] > target[i - 1]:
            ops += target[i] - target[i - 1]
    return ops
```
**LC 3229 (array → target with +1/-1 on subarrays)** generalizes this to the difference array `d[i] = target[i] - arr[i]`: the answer is `|d[0]|` plus, for each `i`, the increase in `|d[i]|` whenever consecutive diffs share the same sign (and the full `|d[i]|` when the sign flips).

**Walkthrough:**
```
target = [3,1,1,2]
  ops = target[0] = 3            (lay 3 increments over the whole prefix)
  i=1: 1 < 3 -> no new ops       (some subarrays just end here)
  i=2: 1 == 1 -> no new ops
  i=3: 2 > 1 -> += (2-1) = 1
  total = 4
```
Only *rises* above the previous height need fresh increment-subarrays; *falls* are free — you simply stop extending some subarrays.

**Edge cases & gotchas:**
- Equivalent to `target[0]` plus the sum of positive consecutive differences.
- Single element ⇒ `target[0]`.
- The LC 3229 generalization keys off the difference array `d = target - arr` and the *sign* of consecutive diffs (see note above).

### LC 1723 — Find Minimum Time to Finish All Jobs

**Question:** You have `jobs` (each `jobs[i]` is a processing time) and `k` workers. Assign **every** job to exactly one worker. A worker's "load" is the sum of its jobs' times. Minimize the **maximum** load across all workers (the time when the last worker finishes).

**Example:**
```
jobs = [3,2,3], k = 2
Best assignment: {3,2} and {3} -> loads 5 and 3 -> max = 5
(Any split here gives max 5; answer = 5.)
```

**Intuition — try every assignment, but prune hard:**

There's no clean greedy (assigning each job to the currently-least-loaded worker can be suboptimal), so you **backtrack**: place job `i` on some worker, recurse on job `i+1`, undo. Naively that's `kⁿ`, so two prunes make it tractable:

- **Bound prune:** track the best (smallest max-load) found so far. If putting a job on a worker would push that worker's load to `≥ best`, this branch can't improve the answer — skip it.
- **Symmetry prune:** if two workers currently have the *same* load, placing the job on either produces an identical subtree. Use a per-step `seen` set of loads so you try each distinct load only once.

Finally, **sort jobs descending** first: placing big jobs early makes loads exceed `best` sooner, so the bound prune fires much earlier. (Alternative framing some prefer: binary-search the answer `T`, then feasibility-check "can all jobs fit in `k` bins each ≤ `T`?" by backtracking — better when job times are large but `k` is small.)

```python
def minimumTimeRequired(jobs, k):
    jobs.sort(reverse=True)
    loads = [0] * k
    best = [sum(jobs)]
    def dfs(i):
        if i == len(jobs):
            best[0] = min(best[0], max(loads))
            return
        seen = set()
        for w in range(k):
            if loads[w] in seen:           # symmetry prune: identical workers
                continue
            seen.add(loads[w])
            if loads[w] + jobs[i] < best[0]:   # bound prune
                loads[w] += jobs[i]
                dfs(i + 1)
                loads[w] -= jobs[i]
    dfs(0)
    return best[0]
```
Alternative: binary-search the answer `T` and greedily/backtrack-check feasibility (can all jobs fit in `k` workers each ≤ `T`).

**Walkthrough:**
```
jobs = [3,2,3], k = 2   (sorted desc -> [3,3,2])

place 3 on w0: loads [3,0]
  place 3 on w1: loads [3,3]
    place 2 on w0: [5,3] -> max 5
    place 2 on w1: [3,5] -> max 5   (symmetric; seen-set skips the identical worker)
best max load = 5
```
The `seen` set skips workers with identical current load (placing the job on either yields the same subtree), and the `loads[w] + jobs[i] < best` bound prunes branches that already exceed the incumbent.

**Edge cases & gotchas:**
- Sort jobs **descending** first — large jobs trigger the bound prune early, drastically shrinking the search.
- `k >= len(jobs)` ⇒ answer is `max(jobs)` (each job gets its own worker).
- The symmetry prune (skip equal-load workers) is what makes this exponential search tractable.
- Alternative framing: binary-search the answer `T`, feasibility-check by backtracking — preferable when loads are large but `k` is small.

---

## System Design Questions

### Design Pinterest Home Feed

**Problem:** Design the Pinterest home feed — the personalized, infinitely-scrolling grid of pins each user sees.

**Requirements:**

**Functional:**
- Show a personalized, ranked grid of pins (not purely chronological)
- Support infinite scroll with stable pagination
- Surface fresh pins (newly created, trending) alongside evergreen recommendations
- Avoid showing pins the user has already seen or dismissed
- A "Following" view for pins from boards/users the person follows

**Non-Functional:**
- Feed load P99 < 300ms; first screen feels instant
- 500M+ users, billions of pins; ~hundreds of K feed requests/sec at peak
- Eventually consistent (a new pin appearing seconds late is fine)
- Personalization quality is the product

**Key insight — Pinterest is a *recommendation* feed, not a follow feed.** Unlike Twitter/Facebook, most of the home feed comes from *interest-based recommendation*, not a follow graph. So the dominant pattern is **candidate generation → ranking**, not classic fan-out. Fan-out matters mainly for the smaller "Following" surface.

### Capacity Estimation

```
Users: 500M, DAU ~150M
Feed requests: 150M DAU × ~10 sessions × ~5 page loads ≈ 7.5B/day ≈ 90K rps avg, ~250K peak
Pins scored per request: ~1-2K candidates -> rank -> top ~25 returned
Pin metadata: ~2 KB × tens of billions of pins -> tens of TB (sharded KV)
Embeddings: pin embedding 256-512 dims × billions -> vector index in the hundreds of GB-TB
```

### High-Level Architecture

```
+--------+    +-------------+    +----------------------+
| Client | -> | API Gateway | -> |   Feed Service       |
+--------+    +-------------+    +----------+-----------+
                                            |
        +-----------------------------------+-----------------------------------+
        |                     |                       |                         |
+-------v-------+   +---------v--------+    +---------v---------+      +---------v--------+
| Candidate Gen |   |  Ranking Service |    |  Seen/Dedup Store |      |  Feed Cache      |
| - your boards |   | (two-tower +     |    | (Bloom + recent   |      | (materialized    |
| - interests   |   |  gradient-boost/ |    |  seen pins/user)  |      |  page per user)  |
| - follow graph|   |  neural ranker)  |    +-------------------+      +------------------+
| - trending    |   +---------+--------+
| - candidate   |             |
|   ANN (vector)|             v
+-------+-------+    +---------+--------+    +------------------+
        |            |  Pin Metadata    |    |   CDN (images)   |
        +----------->|  Store (KV)      |    +------------------+
                     +------------------+
```

### Feed Generation: Candidate Gen + Ranking (vs Fan-out)

| Surface | Approach | Why |
|---------|----------|-----|
| **Home (For You)** | Candidate generation + ML ranking, computed at request time (with cached candidate sets) | Interest-driven; the "interesting" pins aren't tied to who you follow |
| **Following tab** | Fan-out-on-write into a per-user feed list for low-follow users; pull-on-read for accounts following many | Classic feed tradeoff (see §8 News Feed) |
| **Trending / fresh injection** | Pull a small set of globally/again topically trending pins, blended in at ranking time | Freshness and discovery |

```python
def build_home_feed(user_id, cursor, limit=25):
    profile = profile_service.get(user_id)           # interests, recent activity

    # 1. Candidate generation (union of cheap retrievers, ~1-2K candidates)
    candidates = set()
    candidates |= board_based_candidates(user_id)            # pins similar to your boards
    candidates |= interest_candidates(profile.interests)     # topic clusters
    candidates |= ann_candidates(profile.embedding, top=500) # vector nearest-neighbors
    candidates |= follow_graph_candidates(user_id)           # recent pins from follows
    candidates |= trending_candidates(profile.locale)        # freshness/discovery

    # 2. Filter already-seen / dismissed (Bloom filter is the cheap first pass)
    candidates = [p for p in candidates if not seen_store.probably_seen(user_id, p)]

    # 3. Heavy ML ranking (engagement + diversity + freshness)
    ranked = ranking_service.rank(user_id, candidates, profile)

    # 4. Diversity pass: avoid clustering same board/creator/topic
    ranked = diversify(ranked)

    # 5. Record what we're about to show (so next page dedups) and paginate by cursor
    page = paginate(ranked, cursor, limit)
    seen_store.add(user_id, [p.id for p in page])
    return page
```

### Ranking Signals

| Signal | Example |
|--------|---------|
| Engagement prediction | P(save), P(click), P(close-up), P(hide) from a multi-task model |
| Personalization | Affinity between user embedding and pin embedding |
| Freshness | Recency boost for new/trending pins |
| Diversity | Penalize repeated creator/board/topic in one page |
| Quality | Image quality, spam/low-quality demotion |

### Infinite Scroll & Pagination

- **Cursor-based**, not offset — the cursor encodes the ranked position + a feed-session id so new pins arriving mid-scroll don't shift or duplicate results.
- Snapshot the ranked candidate set per feed session (cache it) so pages 2..N are cheap and consistent.

### Caching & CDN

- **Feed cache:** materialize the next 1–2 pages per active user (Redis) so scroll is instant; recompute on session start or after significant new activity.
- **Pin metadata cache:** hot pins in Redis in front of the KV store.
- **CDN:** images served from edge, with responsive sizes; the feed returns image URLs, not bytes.

### Failure Scenarios & Mitigation

| Failure Mode | Impact | Mitigation |
|--------------|--------|------------|
| Ranking service down | No personalization | Fall back to trending/topic feed (still useful) |
| Candidate gen partial | Thinner feed | Union of independent retrievers degrades gracefully |
| Seen-store unavailable | Repeats shown | Bloom filter is best-effort; accept some repeats |
| Hot pin (viral) | Metadata hotspot | Local cache + replication for hot keys |
| Embedding/index stale | Slightly worse recs | Acceptable; refresh embeddings on a schedule |

### Monitoring & Observability

- **Feed latency P50/P99**, candidates scored per request, ranking-model latency
- **Engagement per impression** (save rate, click rate, hide rate) — the quality metric
- **Repeat-pin rate** (seen-store effectiveness), **diversity score** per page
- **Candidate-gen coverage** (% requests with enough candidates)

### Interview Deep-Dive Questions

1. **Why not fan-out-on-write like Twitter?** Pinterest's value is interest-based discovery; most relevant pins aren't from accounts you follow, so a precomputed follow-feed would miss the point. Candidate-gen + ranking is the right primitive; fan-out is reserved for the "Following" tab.
2. **How do you keep scroll consistent as new pins arrive?** Snapshot the ranked set per feed session and paginate by cursor over that snapshot; inject "new pins" only on explicit refresh.
3. **How do you avoid showing the same pin repeatedly?** Per-user seen-store (Bloom filter for the cheap probabilistic check + a bounded recent-seen set), updated as pages are served.
4. **How do you cold-start a new user?** Onboarding interest picker → topic-based candidates; lean on trending/popular until enough interaction signal accrues.

### Design Pinterest Search Engine

**Requirements:**
- Text search for pins, boards, users
- Visual search (search by image)
- Autocomplete and suggestions
- Handle typos and synonyms
- Scale: 500M+ users, billions of pins

**Functional Requirements:**
1. Text-based search with ranking
2. Image-based visual search
3. Real-time autocomplete (<100ms)
4. Personalized results
5. Filter by category, time, popularity

**Non-Functional Requirements:**
- Latency: P99 < 200ms
- Availability: 99.99%
- Scale: 100K QPS

**High-Level Architecture:**

```
┌─────────────┐     ┌─────────────┐     ┌─────────────────────┐
│   Client    │────▶│  API Gateway │────▶│  Search Service     │
└─────────────┘     └─────────────┘     └──────────┬──────────┘
                                                    │
                    ┌───────────────────────────────┼───────────────────────┐
                    │                               │                       │
                    ▼                               ▼                       ▼
          ┌─────────────────┐           ┌─────────────────┐      ┌─────────────────┐
          │ Query Processing│           │  Text Search    │      │ Visual Search   │
          │    Service      │           │  (Elasticsearch)│      │ (Vector DB)     │
          └─────────────────┘           └─────────────────┘      └─────────────────┘
                    │                               │                       │
                    │                               ▼                       ▼
                    │                   ┌─────────────────────────────────────────┐
                    │                   │           Ranking Service               │
                    │                   │    (ML-based personalized ranking)     │
                    │                   └─────────────────────────────────────────┘
                    │                                       │
                    ▼                                       ▼
          ┌─────────────────┐                   ┌─────────────────┐
          │  Autocomplete   │                   │   Result Cache  │
          │   (Trie/Redis)  │                   │     (Redis)     │
          └─────────────────┘                   └─────────────────┘
```

**Key Components:**

**1. Query Processing Service**
- Tokenization, stemming, lemmatization
- Spell correction (edit distance, phonetic matching)
- Query expansion (synonyms, related terms)
- Intent classification (product, board, user search)

**2. Text Search (Elasticsearch)**
```json
{
  "index": "pins",
  "mappings": {
    "properties": {
      "title": {"type": "text", "analyzer": "english"},
      "description": {"type": "text"},
      "tags": {"type": "keyword"},
      "category": {"type": "keyword"},
      "created_at": {"type": "date"},
      "engagement_score": {"type": "float"},
      "embedding": {"type": "dense_vector", "dims": 512}
    }
  }
}
```

**3. Visual Search Pipeline**
```
Image Upload → CNN Feature Extraction → Vector Embedding (512d)
                      ↓
              Vector Database (Pinecone/Milvus)
                      ↓
              Approximate Nearest Neighbor (HNSW/IVF)
                      ↓
              Top-K Similar Images
```

**4. Ranking Service**
- Two-stage ranking: Candidate retrieval → Fine ranking
- Features: Relevance score, freshness, engagement, personalization
- Model: Learning-to-Rank (LambdaMART, Neural ranker)

**5. Autocomplete**
- Trie-based prefix matching
- Popularity-weighted suggestions
- Personalized based on user history
- Redis sorted sets for hot queries

**Data Flow:**
1. User enters query
2. Query processing: spell check, tokenize, expand
3. Parallel fetch: text search + visual search (if image)
4. Merge and dedupe results
5. Ranking service applies personalization
6. Cache results, return to user

**Scaling Considerations:**
- Elasticsearch: Shard by pin_id hash, replicas for read scaling
- Vector DB: Partition by category, approximate search for speed
- Caching: Query result cache (Redis), embedding cache
- CDN: Cache autocomplete suggestions

---

### Design Personalized Pin Recommendation Chatbot

**Requirements:**
- Conversational interface for pin discovery
- Understand user intent from natural language
- Recommend relevant pins based on conversation context
- Learn from user feedback (likes, saves, dismisses)

**Functional Requirements:**
1. Natural language understanding
2. Context-aware recommendations
3. Multi-turn conversations
4. Explain recommendations
5. Handle follow-up queries ("show me more like this")

**Non-Functional Requirements:**
- Response time: < 2 seconds
- Personalization: Based on user history
- Scale: 10M daily active chatbot users

**High-Level Architecture:**

```
┌─────────────┐     ┌─────────────────────────────────────────────────────┐
│   Client    │────▶│                  Chatbot Service                    │
└─────────────┘     └──────────────────────────┬──────────────────────────┘
                                               │
          ┌────────────────────────────────────┼────────────────────────────┐
          │                                    │                            │
          ▼                                    ▼                            ▼
┌─────────────────┐              ┌─────────────────────┐        ┌─────────────────┐
│   NLU Service   │              │ Conversation Manager│        │  Response Gen   │
│  (Intent/Entity)│              │  (Context Tracking) │        │    (LLM)        │
└─────────────────┘              └─────────────────────┘        └─────────────────┘
          │                                    │                            │
          │                                    ▼                            │
          │                      ┌─────────────────────┐                    │
          │                      │   User Profile      │                    │
          │                      │   Service           │                    │
          │                      └─────────────────────┘                    │
          │                                    │                            │
          ▼                                    ▼                            ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           Recommendation Engine                                  │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐│
│  │ Collaborative│  │Content-Based│  │  Knowledge  │  │   Real-time Ranking    ││
│  │  Filtering   │  │  Filtering  │  │   Graph     │  │   (Context + Intent)   ││
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────────────────┘│
└─────────────────────────────────────────────────────────────────────────────────┘
                                               │
                                               ▼
                                    ┌─────────────────┐
                                    │   Pin Database  │
                                    │   (Metadata +   │
                                    │   Embeddings)   │
                                    └─────────────────┘
```

**Key Components:**

**1. NLU Service (Natural Language Understanding)**
```python
# Intent Classification Examples
intents = {
    "explore": "Show me home decor ideas",
    "similar": "More like this one",
    "filter": "Only show DIY projects",
    "specific": "Find minimalist kitchen designs",
    "feedback": "I don't like this style"
}

# Entity Extraction
entities = {
    "category": "home decor",
    "style": "minimalist",
    "color": "blue",
    "room": "kitchen"
}
```

**2. Conversation Manager**
```python
from dataclasses import dataclass, field

@dataclass
class ConversationContext:
    user_id: str
    session_id: str
    turns: list[Turn]  # History of user/bot messages
    current_intent: str
    entities: dict[str, str]  # Accumulated entities
    shown_pins: set[str]  # Avoid repeats
    liked_pins: list[str]
    disliked_pins: list[str]

    def update(self, user_message: str, extracted_intent: str, entities: dict):
        # Merge new entities, update intent
        # Track context for multi-turn
        pass
```

**3. Recommendation Engine**

```python
def get_recommendations(context: ConversationContext) -> list[Pin]:
    # 1. Build query from context
    query_embedding = encode_query(context)

    # 2. Candidate retrieval
    candidates = []
    candidates += collaborative_filter(context.user_id)
    candidates += content_based_filter(context.liked_pins)
    candidates += semantic_search(query_embedding)

    # 3. Filter shown/disliked
    candidates = [p for p in candidates
                  if p.id not in context.shown_pins
                  and p.id not in context.disliked_pins]

    # 4. Rank with context
    ranked = rank_with_context(
        candidates,
        user_profile=get_profile(context.user_id),
        conversation_context=context,
        intent=context.current_intent
    )

    return ranked[:10]
```

**4. Response Generation (LLM)**
```python
def generate_response(pins: list[Pin], context: ConversationContext) -> str:
    prompt = f"""
    User is looking for: {context.current_intent}
    Preferences: {context.entities}

    Recommended pins:
    {format_pins(pins)}

    Generate a friendly response introducing these pins.
    Explain why they match the user's request.
    Only reference pins from the list above by their id/title; do not invent pins.
    """
    return llm.generate(prompt)
```

> **Grounding note:** the recommendations come from the retrieval/ranking engine, **not** the LLM — the LLM only *narrates* the already-chosen pins. Pass the concrete pin ids/titles and instruct the model to reference only those, so it can't hallucinate pins that don't exist. For production, also stream the response and add output guardrails.

**Personalization Signals:**
- Historical pin interactions (saves, clicks, time spent)
- Board themes
- Search history
- Explicit preferences from conversation
- Real-time session behavior

**Feedback Loop:**
```
User Action (like/save/dismiss)
         ↓
Update User Profile (short-term + long-term)
         ↓
Retrain Recommendation Models (batch)
         ↓
A/B Test New Models
```

**Scaling Considerations:**
- LLM inference: Batch requests, model serving with GPUs
- Embeddings: Pre-compute and cache pin embeddings
- Conversation state: Redis for session storage
- Recommendations: Pre-compute candidate sets, real-time ranking

---

### Design Pinterest Notifications

**Problem:** Design the notification system that drives users back to Pinterest — push, email, and in-app — without spamming them.

**Requirements:**

**Functional:**
- Channels: mobile push (APNs/FCM), email, in-app inbox
- Triggered notifications (someone saved your pin, new follower) and recommendation notifications ("ideas for you", boards you might like)
- Per-user, per-channel, per-type preferences and unsubscribe
- Aggregation/digest ("12 people saved your pin") and frequency throttling
- Scheduled/batched sends respecting the user's timezone and quiet hours

**Non-Functional:**
- Millions of notifications/minute at peak; high deliverability
- Eventually consistent; at-least-once delivery with dedup
- Low cost (email/push are cheap, but volume is huge)
- Notifications must be *useful* — over-notifying churns users

### High-Level Architecture

```
+-------------+    +------------------+    +------------------+
| Event       | -> | Notification     | -> |  Kafka topic     |
| sources     |    | Service (API)    |    |  per channel     |
| (saves,     |    | - preference     |    +--------+---------+
|  follows,   |    |   check          |             |
|  rec engine)|    | - rate limit     |   +---------+---------+----------+
+-------------+    | - dedup          |   |         |                    |
                   | - aggregation    | +-v---+  +--v---+           +-----v----+
                   +------------------+ |Push |  |Email |           | In-app   |
                                        |Worker| |Worker|           | Worker   |
                                        +--+--+  +--+---+           +----+-----+
                                           |        |                    |
                                        APNs/FCM  SES/SendGrid     Inbox store (KV)
```

This is the §6 Notification System pattern, specialized for Pinterest. The interesting Pinterest-specific parts are **aggregation, send-time optimization, and fatigue control** — Pinterest sends a lot of "recommendation" notifications, so quality gating matters more than raw delivery.

### Key Components

**1. Notification Service (the gate)** — before anything is queued, apply, in order:
1. **Preference check** — channel + type enabled? quiet hours? unsubscribed?
2. **Frequency cap** — per-user-per-channel budget (e.g. ≤ N pushes/day); recommendation notifications are capped harder than transactional ones.
3. **Dedup** — idempotency key per (user, event) so retries/duplicate events don't double-send.
4. **Aggregation** — buffer similar events in a short window and collapse ("12 people saved *Autumn Recipes*").

**2. Aggregation / digest**

```python
class NotificationAggregator:
    """Collapse bursts of similar events into one notification."""
    def add(self, user_id, event):
        key = (user_id, event.type, event.target_id)  # e.g. (u, "save", pin_id)
        self.buffer[key].append(event)
        # Flush after a quiet window or when the buffer is large enough
        if self.should_flush(key):
            events = self.buffer.pop(key)
            actor_count = len({e.actor_id for e in events})
            self.enqueue(summarize(user_id, event.type, event.target_id, actor_count))
```

**3. Send-time optimization** — for non-urgent (recommendation) notifications, schedule for when the user is most likely to engage (per-user model of active hours), respecting timezone and quiet hours. Transactional notifications (new follower) send promptly.

**4. Channel workers** — pull from the per-channel Kafka topic, render the payload, call the provider (APNs/FCM/SES), handle invalid tokens (deactivate) and transient failures (retry with backoff). In-app notifications write to a per-user inbox KV store read by the app.

### Data Model (sketch)

```sql
notification_preferences(user_id PK, channel, type, enabled, quiet_hours, freq_cap)
device_tokens(user_id, platform, token, is_active)        -- push targets
notifications(id PK, user_id, type, channel, payload, status, created_at, sent_at)
```

### Fatigue Control (the Pinterest-specific crux)

- **Frequency caps** per channel and per notification *class* (transactional vs recommendation).
- **Engagement-based backoff** — if a user repeatedly ignores or dismisses a notification type, suppress or downrank it.
- **Importance scoring** — only send a recommendation push if its predicted engagement clears a bar; otherwise leave it for the in-app inbox.
- **Global send budget** per user per day across all channels.

### Failure Scenarios & Mitigation

| Failure Mode | Impact | Mitigation |
|--------------|--------|------------|
| Provider outage (APNs/FCM) | Push not delivered | Retry queue; fall back to email/in-app for important events |
| Duplicate events | Double notifications | Idempotency key per (user, event) |
| Invalid/expired tokens | Wasted sends, errors | Deactivate token on provider rejection; cleanup job |
| Aggregation worker down | Notification storm | Cap per-user sends even if aggregation is unavailable |
| Preference store down | Risk of spamming opted-out users | Fail *closed* for marketing/rec notifications (don't send if unsure) |

### Monitoring & Observability

- **Delivery rate by channel**, provider error/throttle rate
- **Open/click/save rate** per notification type (usefulness), **unsubscribe rate** (fatigue)
- **Send volume vs cap** per user, aggregation collapse ratio
- **Queue depth** per channel

### Interview Deep-Dive Questions

1. **How do you prevent notification fatigue?** Per-channel + per-type frequency caps, engagement-based suppression, importance scoring so low-value recommendation pushes are downgraded to the in-app inbox, and a global daily budget per user.
2. **How do you aggregate "12 people saved your pin"?** Buffer events keyed by (user, type, target) in a short window, collapse to one summary with an actor count, flush on a quiet timer or size threshold.
3. **How do you pick send time?** Per-user active-hours model for non-urgent notifications, respecting timezone and quiet hours; transactional notifications send immediately.
4. **At-least-once vs exactly-once?** At-least-once delivery with an idempotency key per (user, event) for dedup — exactly-once across external providers isn't achievable, so make the consumer idempotent instead.

---

## Behavioral Questions

- Tell me about a time you had to make a technical decision with incomplete information
- Describe a project where you had to balance speed vs quality
- How do you handle disagreements with teammates on technical approaches?
- Tell me about a time you improved a system's performance significantly

---

## Tips for Pinterest Interviews

1. **Coding**: Focus on clean, efficient solutions. Pinterest values code quality
2. **System Design**: Think about scale (billions of users/pins) and ML integration
3. **Culture Fit**: Pinterest values creativity, collaboration, and user focus
4. **Product Sense**: Understand how technical decisions impact user experience

---

*Good luck with your interview!*
