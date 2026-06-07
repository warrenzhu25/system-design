# Pinterest Interview Questions

## Interview Format

- **Phone Screening**: 1 hour, coding focused (LeetCode style)
- **Onsite**: Multiple rounds including coding, system design, and behavioral

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

```python
def maxBoxesBothEnds(boxes: list[int], warehouse: list[int]) -> int:
    """
    When boxes can enter from both ends, we can place more boxes.

    Key insight: Find the peak (max height room), split into two problems.
    Or use two pointers from both ends.

    Time: O(n log n + m)
    Space: O(1)
    """
    boxes.sort(reverse=True)

    left, right = 0, len(warehouse) - 1
    count = 0

    for box in boxes:
        if left > right:
            break

        # Try to place from the side with larger ceiling
        if warehouse[left] >= box:
            count += 1
            left += 1
        elif warehouse[right] >= box:
            count += 1
            right -= 1
        # else: box doesn't fit anywhere

    return count
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
- user1 has actions at timestamps [1, 2, 4, 5] within window [1, 5] -> 4 actions > 3
- user2 only has 1-2 actions in any 5-second window
```

**Part 1 Solution: HashMap Count**

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

## System Design Questions

### Design Pinterest Home Feed

**Requirements:**
- Show personalized pins to users
- Support infinite scroll
- Handle billions of pins
- Real-time updates for new pins

**Key Components:**
1. **Pin Storage**: Distributed storage for pin metadata and images
2. **Recommendation Engine**: ML-based pin ranking
3. **Feed Generation**: Pre-computed vs real-time feed
4. **Caching**: Multi-layer caching for hot pins
5. **CDN**: Image delivery optimization

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
    """
    return llm.generate(prompt)
```

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

**Requirements:**
- Push notifications (mobile)
- Email notifications
- In-app notifications
- User preferences and throttling

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
