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

### Design Pinterest Search

**Requirements:**
- Text search for pins, boards, users
- Visual search (search by image)
- Autocomplete and suggestions
- Handle typos and synonyms

**Key Components:**
1. **Inverted Index**: Elasticsearch for text search
2. **Visual Embeddings**: CNN-based image feature extraction
3. **Vector Search**: Approximate nearest neighbor search
4. **Query Understanding**: NLP for intent classification

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
