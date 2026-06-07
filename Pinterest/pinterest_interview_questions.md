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
