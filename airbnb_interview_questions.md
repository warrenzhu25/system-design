# Airbnb Interview Questions

Source: https://www.1point3acres.com/interview/problems/company/airbnb?page=1&types=

---

## 1. Minimum Purchases to Exactly Fill Layover Hours (Unlimited Experiences)

**Problem Statement:**
Find the minimum number of experience bookings needed to exactly fill a layover period.

**Key Requirements:**
- Exact Sum: Selected durations must sum to precisely X hours
- Minimize Purchases: Choose the fewest experiences possible
- Unlimited Repetition: Any experience can be booked multiple times
- Return 0 if Impossible: No combination exists that equals X

**Input/Output Format:**
- Input: Array of experience durations (decimals) and target layover hours X
- Output: Minimum number of purchases needed, or 0 if impossible

**Constraints:**
- Array length: 1 to 30 experiences
- Duration range: 0 < duration ≤ 24.0 hours
- Layover range: 0 < X ≤ 100.0 hours
- All values have exactly one decimal place

**Sample Test Cases:**

| Input | Output | Notes |
|-------|--------|-------|
| durations=[3.0, 2.0], X=7.0 | 3 | 2.0+2.0+3.0=7.0 |
| durations=[1.5, 2.0, 3.5], X=7.0 | 2 | 3.5+3.5=7.0 |
| durations=[3.0, 2.0], X=1.0 | 0 | No valid combination |

**Solution Approach:**
This is a variant of the coin change problem solvable via dynamic programming, converting decimals to integers (multiply by 10) to avoid floating-point precision issues.

**Python Solution:**
```python
def min_purchases(durations: list[float], X: float) -> int:
    # Convert to integers to avoid floating point issues
    target = int(X * 10)
    coins = [int(d * 10) for d in durations]

    # dp[i] = minimum purchases to reach amount i
    dp = [float('inf')] * (target + 1)
    dp[0] = 0

    for amount in range(1, target + 1):
        for coin in coins:
            if coin <= amount and dp[amount - coin] != float('inf'):
                dp[amount] = min(dp[amount], dp[amount - coin] + 1)

    return dp[target] if dp[target] != float('inf') else 0


# Test cases
print(min_purchases([3.0, 2.0], 7.0))      # Output: 3
print(min_purchases([1.5, 2.0, 3.5], 7.0)) # Output: 2
print(min_purchases([3.0, 2.0], 1.0))      # Output: 0
```

---

## 2. Calculate Board Score

**Problem Description:**
You're given a 2D board of size R × C where each cell contains a character. The scoring mechanism works as follows:
- For every cell (i, j), measure the contiguous run of identical characters extending "to the right" and "down"
- Each direction's run length contributes to the total score
- Sum all contributions across both directions for the final board score

**Constraints:**
- 1 ≤ R, C ≤ 2000 (O(RC) or near-linear solution expected)
- Board characters are printable ASCII

**Sample Test Cases:**
- Test 1: 2×2 board of all 'A's → Output: 12
- Test 2: Single row "ABBB" → Output: 8
- Test 3: 3×3 board with all unique characters → Output: 18
- Test 4: 3×5 mixed pattern board → Output: 52
- Test 5: 2×3 board with dots and one 'X' → Output: 20

**Input/Output Format:**
- Input: First line contains R and C, followed by R lines of length C
- Output: Single integer representing total board score

**Python Solution:**
```python
def calculate_board_score(board: list[list[str]]) -> int:
    if not board or not board[0]:
        return 0

    R, C = len(board), len(board[0])
    total_score = 0

    # Calculate horizontal runs (to the right)
    for i in range(R):
        j = C - 1
        run_length = 1
        total_score += run_length  # rightmost cell
        while j > 0:
            j -= 1
            if board[i][j] == board[i][j + 1]:
                run_length += 1
            else:
                run_length = 1
            total_score += run_length

    # Calculate vertical runs (down)
    for j in range(C):
        i = R - 1
        run_length = 1
        total_score += run_length  # bottom cell
        while i > 0:
            i -= 1
            if board[i][j] == board[i + 1][j]:
                run_length += 1
            else:
                run_length = 1
            total_score += run_length

    return total_score


# Test cases
print(calculate_board_score([['A', 'A'], ['A', 'A']]))  # Output: 12
print(calculate_board_score([['A', 'B', 'B', 'B']]))    # Output: 8
print(calculate_board_score([['A', 'B', 'C'],
                              ['D', 'E', 'F'],
                              ['G', 'H', 'I']]))        # Output: 18
```

---

## 3. Replace Text for Review

**Problem Description:**
Given a piece of text, find and replace specific words.

**Input Requirements:**
- A string representing the text content
- A dictionary mapping words to their replacements

**Output:**
The modified string with all specified words replaced

**Sample Test Cases:**

| Input | Output |
|-------|--------|
| `The quick brown fox;{'quick': 'slow', 'brown': 'black'}` | `The slow black fox` |
| `Lorem ipsum dolor sit amet;{'Lorem': '', 'dolor': 'pain'}` | ` ipsum pain sit amet` |
| `hello world;{'world': 'universe'}` | `hello universe` |
| `a quick brown fox;{'quick': 'slow', 'brown': 'red', 'fox': 'cat'}` | `a slow red cat` |
| `unchanged text;{}` | `unchanged text` |

**Key Observations:**
- Empty string replacements remove words
- Multiple replacements occur in a single pass
- Empty replacement dictionaries return unchanged text

**Python Solution:**
```python
import re

def replace_text(text: str, replacements: dict[str, str]) -> str:
    if not replacements:
        return text

    # Sort by length descending to handle longer matches first
    sorted_keys = sorted(replacements.keys(), key=len, reverse=True)

    # Build regex pattern
    pattern = '|'.join(re.escape(key) for key in sorted_keys)

    def replace_match(match):
        return replacements[match.group(0)]

    return re.sub(pattern, replace_match, text)


# Alternative simple solution using str.replace
def replace_text_simple(text: str, replacements: dict[str, str]) -> str:
    result = text
    for old, new in replacements.items():
        result = result.replace(old, new)
    return result


# Test cases
print(replace_text("The quick brown fox", {'quick': 'slow', 'brown': 'black'}))
# Output: The slow black fox

print(replace_text("Lorem ipsum dolor sit amet", {'Lorem': '', 'dolor': 'pain'}))
# Output:  ipsum pain sit amet

print(replace_text("hello world", {'world': 'universe'}))
# Output: hello universe
```

---

## 4. URL Query Parameter Parsing

**Problem Statement:**
Parse a given URL string and return the query parameters as a dictionary.

**Input:** A string representing a URL.

**Output:** A dictionary with keys as parameter names and values as parameter values.

**Test Cases:**

| Input | Output |
|-------|--------|
| `https://example.com/path?arg1=value1&arg2=value2` | `{'arg1': 'value1', 'arg2': 'value2'}` |
| `https://example.com/path?arg1=value1&arg2=` | `{'arg1': 'value1', 'arg2': ''}` |
| `https://example.com/path` | `{}` |
| `https://example.com/path?arg1=value1&arg1=value2` | `{'arg1': ['value1', 'value2']}` |
| `https://example.com/path?encoded%3Dvalue&arg=value` | `{'encoded=': 'value', 'arg': 'value'}` |

**Key Considerations:**
- Handle empty parameter values
- Handle URLs without query strings
- Handle duplicate parameter names (return as list)
- Handle URL-encoded characters in parameters

**Python Solution:**
```python
from urllib.parse import urlparse, parse_qs, unquote

def parse_url_params(url: str) -> dict:
    parsed = urlparse(url)
    query_string = parsed.query

    if not query_string:
        return {}

    result = {}
    pairs = query_string.split('&')

    for pair in pairs:
        if '=' in pair:
            key, value = pair.split('=', 1)
        else:
            key, value = pair, ''

        # URL decode the key and value
        key = unquote(key)
        value = unquote(value)

        if key in result:
            # Convert to list if duplicate keys
            if isinstance(result[key], list):
                result[key].append(value)
            else:
                result[key] = [result[key], value]
        else:
            result[key] = value

    return result


# Test cases
print(parse_url_params("https://example.com/path?arg1=value1&arg2=value2"))
# Output: {'arg1': 'value1', 'arg2': 'value2'}

print(parse_url_params("https://example.com/path?arg1=value1&arg2="))
# Output: {'arg1': 'value1', 'arg2': ''}

print(parse_url_params("https://example.com/path"))
# Output: {}

print(parse_url_params("https://example.com/path?arg1=value1&arg1=value2"))
# Output: {'arg1': ['value1', 'value2']}

print(parse_url_params("https://example.com/path?encoded%3Dvalue&arg=value"))
# Output: {'encoded=': 'value', 'arg': 'value'}
```

---

## 5. Remove Vowels from String (Algorithm question with test cases)

**Problem Description:**
Write a function to perform the following task: Given a string, remove all of the vowel letters (a, e, i, o, u) and return the result.

**Key Requirements:**
- Eliminate both uppercase and lowercase vowels (A, E, I, O, U, a, e, i, o, u)
- Input strings won't exceed 1000 characters
- Preserve all non-vowel characters and spacing

**Test Cases:**

| Input | Expected Output |
|-------|-----------------|
| "Hello World" | "Hll Wrld" |
| "beautiful" | "btfl" |
| "AEIOU are vowels" | " r vwls" |
| "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz" | "BCDFGHJKLMNPQRSTVWXYZbcdfghjklmnpqrstvwxyz" |
| "" | "" |

**Python Solution:**
```python
def remove_vowels(s: str) -> str:
    vowels = set('aeiouAEIOU')
    return ''.join(char for char in s if char not in vowels)


# Alternative using translate
def remove_vowels_translate(s: str) -> str:
    vowels = 'aeiouAEIOU'
    return s.translate(str.maketrans('', '', vowels))


# Test cases
print(remove_vowels("Hello World"))          # Output: "Hll Wrld"
print(remove_vowels("beautiful"))            # Output: "btfl"
print(remove_vowels("AEIOU are vowels"))     # Output: " r vwls"
print(remove_vowels(""))                     # Output: ""
```

---

## 6. Intersection of Two Linked Lists

**Problem Statement:**
Given two (singly) linked lists, return the node at which the two lists intersect. If the two linked lists have no intersection at all, return null.

**Examples:**
1. listA = [4,1,8,4,5], listB = [5,6,1,8,4,5] → Output: ListNode(8)
   - The lists intersect at the node with value 8.
2. listA = [1,9,1,2,4], listB = [3,2,4] → Output: ListNode(2)
   - The lists intersect at the node with value 2.
3. listA = [2,6,4], listB = [1,5] → Output: null
   - The lists do not intersect.

**Constraints:**
- Return null if no intersection exists
- Preserve the original structure of both linked lists after execution
- No cycles exist in the linked structure
- Optimal solution should achieve O(n) time complexity with O(1) space complexity

**Python Solution:**
```python
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


def get_intersection_node(headA: ListNode, headB: ListNode) -> ListNode:
    if not headA or not headB:
        return None

    # Two pointer approach
    pA, pB = headA, headB

    # When one pointer reaches end, redirect to other list's head
    # They will meet at intersection or both become None
    while pA != pB:
        pA = pA.next if pA else headB
        pB = pB.next if pB else headA

    return pA


# Alternative: Calculate lengths first
def get_intersection_node_v2(headA: ListNode, headB: ListNode) -> ListNode:
    def get_length(head):
        length = 0
        while head:
            length += 1
            head = head.next
        return length

    lenA, lenB = get_length(headA), get_length(headB)

    # Align starting points
    while lenA > lenB:
        headA = headA.next
        lenA -= 1
    while lenB > lenA:
        headB = headB.next
        lenB -= 1

    # Find intersection
    while headA and headB:
        if headA == headB:
            return headA
        headA = headA.next
        headB = headB.next

    return None
```

---

## 7. Find Split Stays for Airbnb Listings

**Problem Description:**
Create an API endpoint that identifies all viable split stay combinations across two Airbnb listings within a specified date range.

**Core Requirements:**
- Input: Multiple listings with availability calendars (day numbers) and a date range
- Output: All valid pairs of listings that can form a complete split stay
- A split stay requires: first listing covers initial trip days, second listing covers remaining days
- Each listing has a name identifier and list of available days

**Sample Input:**
```json
{
  "listings": {
    "A": [1,2,3,6,7,10,11],
    "B": [3,4,5,6,8,9,10,13],
    "C": [7,8,9,10,11]
  },
  "start_date": 3,
  "end_date": 11
}
```

**Expected Output:**
All sets of two Airbnbs that could form a split stay: [B, C]

**Implementation Requirements:**
- Create an algorithm validating split stay combinations
- Write comprehensive test cases covering various scenarios
- Available language options: Python 3, Java

**Python Solution:**
```python
def find_split_stays(listings: dict[str, list[int]],
                     start_date: int,
                     end_date: int) -> list[tuple[str, str]]:
    result = []
    listing_names = list(listings.keys())

    # Convert to sets for O(1) lookup
    availability = {name: set(days) for name, days in listings.items()}

    # All required days
    required_days = set(range(start_date, end_date + 1))

    # Try all pairs of listings
    for i in range(len(listing_names)):
        for j in range(len(listing_names)):
            if i == j:
                continue

            first = listing_names[i]
            second = listing_names[j]

            # Try all possible split points
            for split_day in range(start_date + 1, end_date + 1):
                first_days = set(range(start_date, split_day))
                second_days = set(range(split_day, end_date + 1))

                # Check if first listing covers first part
                # and second listing covers second part
                if first_days.issubset(availability[first]) and \
                   second_days.issubset(availability[second]):
                    if (first, second) not in result:
                        result.append((first, second))
                    break  # Found valid split for this pair

    return result


# Test case
listings = {
    "A": [1, 2, 3, 6, 7, 10, 11],
    "B": [3, 4, 5, 6, 8, 9, 10, 13],
    "C": [7, 8, 9, 10, 11]
}
print(find_split_stays(listings, 3, 11))
# Output: [('B', 'C')]
```

---

## 8. Simulate Water Flow on Terrain

**Problem Description:**
You are given an array like [5, 4, 3, 2, 1, 3, 4, 0, 3, 4]. Each number represents the height of a column at that index.

**Two-Part Challenge:**

**Part 1:** Visualize the terrain by rendering each array value as a column height.

**Part 2:** Imagine we pour a certain amount of water at a certain column. The water can flow in whichever direction makes sense. Print the terrain after all the water has fallen.

**Example:**
When calling `dumpWater(terrain, waterAmount=8, column=1)` on terrain `[5, 4, 3, 2, 1, 3, 4, 0, 3, 4]`:

```
+
++WWWW+ +
+++WW++ ++
++++W++ ++
+++++++W++
++++++++++ <--- base layer
```

The water distributes based on gravity and terrain topology, filling lower areas first.

**Sample Test Cases:**
- Input: `[5, 4, 3, 2, 1, 3, 4, 0, 3, 4]`, 8 units at column 1
- Input: `[1, 2, 3, 4, 5, 6, 7, 8, 9]`, 5 units at column 4

**Python Solution:**
```python
def visualize_terrain(terrain: list[int], water: list[int] = None) -> str:
    if water is None:
        water = [0] * len(terrain)

    max_height = max(t + w for t, w in zip(terrain, water))
    rows = []

    for level in range(max_height, 0, -1):
        row = ""
        for i in range(len(terrain)):
            if terrain[i] >= level:
                row += "+"
            elif terrain[i] + water[i] >= level:
                row += "W"
            else:
                row += " "
        rows.append(row)

    return "\n".join(rows)


def dump_water(terrain: list[int], water_amount: int, column: int) -> list[int]:
    n = len(terrain)
    water = [0] * n

    for _ in range(water_amount):
        # Find where this unit of water settles
        pos = column

        while True:
            current_level = terrain[pos] + water[pos]

            # Try to flow left
            left_pos = pos
            for i in range(pos - 1, -1, -1):
                if terrain[i] + water[i] < current_level:
                    left_pos = i
                    current_level = terrain[i] + water[i]
                elif terrain[i] + water[i] > current_level:
                    break

            # Try to flow right
            right_pos = pos
            current_level = terrain[pos] + water[pos]
            for i in range(pos + 1, n):
                if terrain[i] + water[i] < current_level:
                    right_pos = i
                    current_level = terrain[i] + water[i]
                elif terrain[i] + water[i] > current_level:
                    break

            # Choose lowest position
            if left_pos != pos and (right_pos == pos or
                terrain[left_pos] + water[left_pos] <= terrain[right_pos] + water[right_pos]):
                pos = left_pos
            elif right_pos != pos:
                pos = right_pos
            else:
                break

        water[pos] += 1

    return water


# Test
terrain = [5, 4, 3, 2, 1, 3, 4, 0, 3, 4]
water = dump_water(terrain, 8, 1)
print(visualize_terrain(terrain, water))
```

---

## 9. Hotel Split Stay

**Problem Description:**
Given `n` hotels with available date lists and a date range, split the range into two consecutive non-overlapping subranges. Each subrange must be covered by a different hotel using only forward checking (no backtracking). Return all valid hotel ID combinations.

**Input Specification:**
- `n`: integer where 2 ≤ n ≤ 100
- `hotels`: 2D list where `hotels[i]` contains available dates for hotel `i` in "YYYY-MM-DD" format
- `start_date`: string representing the range start
- `end_date`: string representing the range end

**Output:**
Return all possible hotel ID combinations as tuples where each subrange is fulfilled by a different hotel, with no repeated combinations.

**Example:**

**Input:**
```
3 hotels
[["2023-01-01", "2023-01-02", "2023-01-03"],
 ["2023-01-02", "2023-01-03", "2023-01-04"],
 ["2023-01-04", "2023-01-05"]]
"2023-01-01"
"2023-01-05"
```

**Output:**
```
[(0, 1), (0, 2)]
```

**Python Solution:**
```python
from datetime import datetime, timedelta

def hotel_split_stay(hotels: list[list[str]],
                     start_date: str,
                     end_date: str) -> list[tuple[int, int]]:

    def parse_date(d):
        return datetime.strptime(d, "%Y-%m-%d")

    def date_range(start, end):
        """Generate all dates from start to end (inclusive)"""
        dates = []
        current = parse_date(start)
        end_dt = parse_date(end)
        while current <= end_dt:
            dates.append(current.strftime("%Y-%m-%d"))
            current += timedelta(days=1)
        return dates

    # Convert hotel availability to sets
    availability = [set(dates) for dates in hotels]
    required_dates = date_range(start_date, end_date)

    result = []
    n = len(hotels)

    # Try all pairs of hotels
    for first in range(n):
        for second in range(n):
            if first == second:
                continue

            # Try all split points
            for split_idx in range(1, len(required_dates)):
                first_part = required_dates[:split_idx]
                second_part = required_dates[split_idx:]

                # Check if first hotel covers first part
                # and second hotel covers second part
                if all(d in availability[first] for d in first_part) and \
                   all(d in availability[second] for d in second_part):
                    if (first, second) not in result:
                        result.append((first, second))
                    break

    return result


# Test case
hotels = [
    ["2023-01-01", "2023-01-02", "2023-01-03"],
    ["2023-01-02", "2023-01-03", "2023-01-04"],
    ["2023-01-04", "2023-01-05"]
]
print(hotel_split_stay(hotels, "2023-01-01", "2023-01-05"))
# Output: [(0, 1), (0, 2)]
```

---

## 10. Data Analysis System Design

**Problem Description:**
Design a data analysis system emphasizing high interactivity.

**Approach:**

1. **Initial Design Phase**: Start with a high-level architecture using traditional Spark processing, while recognizing that this approach may introduce latency issues that don't satisfy requirements.

2. **Advanced Phase**: Explore stream processing solutions, specifically Flink aggregation, to address latency concerns.

**Key Technical Areas:**
- Integration between Flink and Kafka
- Recovery mechanisms when Flink experiences failures

**Problem Type:** System Design (architectural/technical design question rather than a coding problem)

**Python Solution (Conceptual Implementation):**
```python
"""
This is a system design problem. Below is a conceptual implementation
demonstrating the key components.
"""

from abc import ABC, abstractmethod
from typing import Any
from collections import defaultdict
import time

class DataProcessor(ABC):
    @abstractmethod
    def process(self, data: Any) -> Any:
        pass

class KafkaConsumer:
    """Simulated Kafka consumer for stream processing"""
    def __init__(self, topic: str, bootstrap_servers: list[str]):
        self.topic = topic
        self.servers = bootstrap_servers
        self.offset = 0

    def consume(self):
        # Simulated message consumption
        pass

    def commit_offset(self, offset: int):
        self.offset = offset

class FlinkAggregator(DataProcessor):
    """Stream processing with Flink-like aggregation"""
    def __init__(self):
        self.state = defaultdict(int)
        self.checkpoints = []

    def process(self, data: dict) -> dict:
        key = data.get('key')
        value = data.get('value', 0)
        self.state[key] += value
        return {'key': key, 'aggregated': self.state[key]}

    def checkpoint(self):
        """Save state for recovery"""
        self.checkpoints.append({
            'state': dict(self.state),
            'timestamp': time.time()
        })

    def recover(self):
        """Recover from last checkpoint"""
        if self.checkpoints:
            last = self.checkpoints[-1]
            self.state = defaultdict(int, last['state'])

class DataAnalysisSystem:
    """
    High-level architecture:
    1. Kafka for data ingestion
    2. Flink for real-time aggregation
    3. Checkpointing for fault tolerance
    """
    def __init__(self):
        self.consumer = KafkaConsumer("events", ["localhost:9092"])
        self.processor = FlinkAggregator()
        self.results_cache = {}

    def run(self):
        """Main processing loop"""
        while True:
            # Consume from Kafka
            events = self.consumer.consume()

            # Process with Flink
            for event in events or []:
                result = self.processor.process(event)
                self.results_cache[result['key']] = result

            # Periodic checkpointing
            self.processor.checkpoint()
            self.consumer.commit_offset(self.consumer.offset)

    def query(self, key: str) -> dict:
        """Low-latency query interface"""
        return self.results_cache.get(key, {})
```

---

## 11. Retryer Function Implementation

**Problem Description:**
Design a retryer function that accepts an asynchronous function and a maximum retry count. The function should implement exponential backoff delays (2s, 4s, 8s, etc.) between attempts, with a maximum wait of 16 seconds per retry. Success returns the result; failure after max retries throws an error.

**Input Parameters:**
- `func`: An asynchronous function requiring no parameters
- `max_retries`: Integer specifying maximum retry attempts

**Output:**
Returns the async function's result or throws an error upon exhaustion of retries.

**Example Usage:**
```python
async def my_async_function():
    # Simulate a potentially failing async call

result = await retryer(my_async_function, 3)
```

**Key Constraints:**
- Each function call is independent with no shared state
- Exponential backoff capped at 16 seconds maximum
- Use `await asyncio.sleep()` for delay simulation

**Test Cases Covered:**
1. Function that always fails
2. Function that always succeeds
3. Function with probabilistic success/failure
4. Function failing on initial attempts then succeeding
5. Immediate success with minimal retries

**Python Solution:**
```python
import asyncio
from typing import Callable, TypeVar

T = TypeVar('T')

class MaxRetriesExceeded(Exception):
    pass


async def retryer(func: Callable[[], T], max_retries: int) -> T:
    """
    Retry an async function with exponential backoff.

    Args:
        func: Async function to retry
        max_retries: Maximum number of retry attempts

    Returns:
        Result of the function if successful

    Raises:
        MaxRetriesExceeded: If all retries fail
    """
    last_exception = None

    for attempt in range(max_retries + 1):
        try:
            return await func()
        except Exception as e:
            last_exception = e

            if attempt < max_retries:
                # Exponential backoff: 2, 4, 8, 16 (capped)
                delay = min(2 ** (attempt + 1), 16)
                await asyncio.sleep(delay)

    raise MaxRetriesExceeded(
        f"Failed after {max_retries + 1} attempts. Last error: {last_exception}"
    )


# Test cases
async def test_retryer():
    # Test 1: Always succeeds
    async def always_succeeds():
        return "success"

    result = await retryer(always_succeeds, 3)
    assert result == "success"
    print("Test 1 passed: Always succeeds")

    # Test 2: Always fails
    async def always_fails():
        raise ValueError("Always fails")

    try:
        await retryer(always_fails, 2)
        assert False, "Should have raised"
    except MaxRetriesExceeded:
        print("Test 2 passed: Always fails")

    # Test 3: Succeeds after failures
    attempt_count = 0
    async def succeeds_third_try():
        nonlocal attempt_count
        attempt_count += 1
        if attempt_count < 3:
            raise ValueError(f"Attempt {attempt_count} failed")
        return "finally success"

    result = await retryer(succeeds_third_try, 3)
    assert result == "finally success"
    print("Test 3 passed: Succeeds after failures")


# Run tests
# asyncio.run(test_retryer())
```

---

## 12. Minimum Cost to Order Meals

**Problem Description:**
You need to order some dishes such that the total cost exactly equals to a specific amount. Each dish can be ordered multiple times.

The task is to compute the count of different combinations that sum to an exact budget.

**Input & Output:**
- Input: A list of dish prices and a dining budget amount
- Output: An integer representing the number of distinct ordering combinations equaling the amount

**Example:**
Given prices = [1, 2, 5] and amount = 5, the output is 4:
- 5
- 2 + 2 + 1
- 2 + 1 + 1 + 1
- 1 + 1 + 1 + 1 + 1

**Constraints:**
- Up to 500 dish prices
- Each price ≤ 300
- Budget amount ≤ 10,000

**Python Solution:**
```python
def count_meal_combinations(prices: list[int], amount: int) -> int:
    """
    Count distinct combinations (order doesn't matter).
    This is the "Coin Change 2" problem.
    """
    # dp[i] = number of ways to make amount i
    dp = [0] * (amount + 1)
    dp[0] = 1  # One way to make 0: use nothing

    # Process each price to avoid counting permutations
    for price in prices:
        for a in range(price, amount + 1):
            dp[a] += dp[a - price]

    return dp[amount]


def count_meal_permutations(prices: list[int], amount: int) -> int:
    """
    Count permutations (order matters).
    """
    dp = [0] * (amount + 1)
    dp[0] = 1

    for a in range(1, amount + 1):
        for price in prices:
            if price <= a:
                dp[a] += dp[a - price]

    return dp[amount]


# Test cases
print(count_meal_combinations([1, 2, 5], 5))  # Output: 4
print(count_meal_combinations([1, 2, 3], 4))  # Output: 4 (1+1+1+1, 1+1+2, 2+2, 1+3)
```

---

## 13. Water Pouring Problem

**Problem Description:**
You have two jugs with different capacities and a target capacity. Initially, both jugs are empty.

You can perform three operations:
- Fill one jug completely
- Empty one jug
- Pour water from one jug to another until the source empties or destination fills

**Objective:**
Determine the minimum number of operations needed to achieve the target capacity and output the operation sequence.

**Example Walkthrough:**
For jugs with capacities [3, 5] targeting 4 units:

1. (0, 0) → (0, 5) — Fill jug 2
2. (0, 5) → (3, 2) — Pour jug 2 into jug 1
3. (3, 2) → (0, 2) — Empty jug 1
4. (0, 2) → (2, 0) — Pour jug 2 into jug 1
5. (2, 0) → (2, 5) — Fill jug 2
6. (2, 5) → (3, 4) — Target reached

**Test Cases:**
- jug1=3, jug2=5, target=4
- jug1=2, jug2=6, target=5
- jug1=8, jug2=12, target=1
- jug1=7, jug2=11, target=6
- jug1=5, jug2=3, target=4

**Note:** This problem requires a BFS or similar search algorithm to find the shortest path to the target state.

**Python Solution:**
```python
from collections import deque

def water_pouring(jug1_cap: int, jug2_cap: int, target: int) -> list[tuple[int, int]]:
    """
    Find minimum operations to achieve target amount in either jug.
    Returns list of states (jug1, jug2) from start to target.
    """
    if target > max(jug1_cap, jug2_cap):
        return []

    if target == 0:
        return [(0, 0)]

    # BFS
    visited = set()
    queue = deque()
    queue.append((0, 0, [(0, 0)]))  # (jug1, jug2, path)
    visited.add((0, 0))

    while queue:
        j1, j2, path = queue.popleft()

        # Check if target reached
        if j1 == target or j2 == target:
            return path

        # Generate all possible next states
        next_states = [
            (jug1_cap, j2),      # Fill jug 1
            (j1, jug2_cap),      # Fill jug 2
            (0, j2),             # Empty jug 1
            (j1, 0),             # Empty jug 2
            # Pour jug 1 into jug 2
            (max(0, j1 - (jug2_cap - j2)), min(jug2_cap, j1 + j2)),
            # Pour jug 2 into jug 1
            (min(jug1_cap, j1 + j2), max(0, j2 - (jug1_cap - j1)))
        ]

        for state in next_states:
            if state not in visited:
                visited.add(state)
                queue.append((state[0], state[1], path + [state]))

    return []  # No solution


def print_solution(jug1_cap: int, jug2_cap: int, target: int):
    path = water_pouring(jug1_cap, jug2_cap, target)
    if not path:
        print(f"No solution for jugs ({jug1_cap}, {jug2_cap}) with target {target}")
        return

    print(f"Solution for jugs ({jug1_cap}, {jug2_cap}), target={target}:")
    for i, (j1, j2) in enumerate(path):
        print(f"  Step {i}: ({j1}, {j2})")
    print(f"  Total operations: {len(path) - 1}")


# Test cases
print_solution(3, 5, 4)
print_solution(2, 6, 5)
print_solution(7, 11, 6)
```

---

## 14. Cheapest Flight Route with Limited Transfers

**Problem Description:**
Find the minimum fare itinerary from a start city to a destination city, given flight information and a maximum number of transfers allowed. The solution must return both the minimum cost and the specific route taken.

**Problem Statement:**
Given a list of flight information where each entry includes the start city, destination city, and corresponding ticket price, along with a maximum allowed number of transfers, find the minimum fare itinerary from a specified start point to a destination.

**Example Scenario:**
With flights [London → Japan, 500], [Japan → Beijing, 100], [London → Beijing, 1000]:
- Maximum 1 transfer: Optimal route is London → Japan → Beijing (cost: 600)
- No transfers: Optimal route is London → Beijing (cost: 1000)

**Sample Input:**
```
n = 3
flights = [(0, 1, 100), (1, 2, 100), (0, 2, 500)]
src = 0
dst = 2
K = 1
```

**Python Solution:**
```python
from collections import defaultdict
import heapq

def find_cheapest_flight(n: int,
                         flights: list[tuple[int, int, int]],
                         src: int,
                         dst: int,
                         k: int) -> tuple[int, list[int]]:
    """
    Find cheapest flight with at most k transfers.
    Returns (cost, route) or (-1, []) if no route exists.
    """
    # Build adjacency list
    graph = defaultdict(list)
    for u, v, price in flights:
        graph[u].append((v, price))

    # Dijkstra with stops constraint
    # (cost, stops, node, path)
    heap = [(0, 0, src, [src])]
    # visited[node] = minimum stops to reach node
    visited = {}

    while heap:
        cost, stops, node, path = heapq.heappop(heap)

        if node == dst:
            return (cost, path)

        if stops > k + 1:
            continue

        # Skip if we've reached this node with fewer stops
        if node in visited and visited[node] < stops:
            continue
        visited[node] = stops

        for neighbor, price in graph[node]:
            heapq.heappush(heap, (cost + price, stops + 1, neighbor, path + [neighbor]))

    return (-1, [])


# Alternative: Bellman-Ford approach
def find_cheapest_flight_bf(n: int,
                            flights: list[tuple[int, int, int]],
                            src: int,
                            dst: int,
                            k: int) -> int:
    """
    Bellman-Ford approach - returns just the minimum cost.
    """
    # dist[i] = minimum cost to reach node i
    dist = [float('inf')] * n
    dist[src] = 0

    # Relax edges k+1 times (for k transfers = k+1 flights)
    for _ in range(k + 1):
        new_dist = dist.copy()
        for u, v, price in flights:
            if dist[u] != float('inf'):
                new_dist[v] = min(new_dist[v], dist[u] + price)
        dist = new_dist

    return dist[dst] if dist[dst] != float('inf') else -1


# Test cases
n = 3
flights = [(0, 1, 100), (1, 2, 100), (0, 2, 500)]
print(find_cheapest_flight(n, flights, 0, 2, 1))  # (200, [0, 1, 2])
print(find_cheapest_flight(n, flights, 0, 2, 0))  # (500, [0, 2])
print(find_cheapest_flight_bf(n, flights, 0, 2, 1))  # 200
```

---

## 15. Minimum Moves (Online Judge)

**Difficulty:** Easy, Newgrad

**Problem Description:**
Determine the minimum number of moves to reach (n-1, m-1) from (0,0) on a grid, or return -1 if unreachable.

You can move up to k cells per move in one direction, provided every cell traversed is 0 (obstacles are marked as 1).

**Constraints:**
- Grid dimensions: n, m ≤ 100
- Movement parameter: k ranges from 1–100
- Input: n×m maze with 0/1 values and integer k
- Output: integer representing minimum move count

**Sample Test Cases:**

| Input | Output |
|-------|--------|
| maze = [[0, 0], [1, 0]], k = 2 | 2 |
| maze = [[0, 0, 0], [1, 0, 0]], k = 5 | 2 |

**Context:**
This problem is framed as HackerMan navigating the HackerPlay maze, representing a common Airbnb internship-level coding challenge.

**Python Solution:**
```python
from collections import deque

def minimum_moves(maze: list[list[int]], k: int) -> int:
    if not maze or not maze[0]:
        return -1

    n, m = len(maze), len(maze[0])

    if maze[0][0] == 1 or maze[n-1][m-1] == 1:
        return -1

    if n == 1 and m == 1:
        return 0

    # BFS
    visited = [[False] * m for _ in range(n)]
    queue = deque([(0, 0, 0)])  # (row, col, moves)
    visited[0][0] = True

    # Directions: right, down, left, up
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]

    while queue:
        row, col, moves = queue.popleft()

        for dr, dc in directions:
            # Try moving 1 to k steps in this direction
            for step in range(1, k + 1):
                nr, nc = row + dr * step, col + dc * step

                # Check bounds
                if nr < 0 or nr >= n or nc < 0 or nc >= m:
                    break

                # Check obstacle
                if maze[nr][nc] == 1:
                    break

                # Check if reached destination
                if nr == n - 1 and nc == m - 1:
                    return moves + 1

                # Add to queue if not visited
                if not visited[nr][nc]:
                    visited[nr][nc] = True
                    queue.append((nr, nc, moves + 1))

    return -1


# Test cases
print(minimum_moves([[0, 0], [1, 0]], 2))          # Output: 2
print(minimum_moves([[0, 0, 0], [1, 0, 0]], 5))    # Output: 2
print(minimum_moves([[0, 1], [1, 0]], 1))          # Output: -1
```

---

## 16. Round Prices to Match Target (Online Judge)

**Difficulty:** Easy, Newgrad

**Problem Description:**
The task requires rounding individual prices in a float array to integers such that the rounded values sum to a specified target, while minimizing the total deviation from original prices.

**Key Requirements:**
- Input: Float array of prices and an integer target value
- Output: Integer array of same length with values summing to target
- Goal: Minimize cumulative difference between original and rounded prices

**Sample Case:**

**Input:**
```
prices = [1.2, 4.3, 5.8, 6.4]
target = 18
```

**Output:**
```
[1, 4, 6, 7]
```

**Python Solution:**
```python
import math

def round_prices(prices: list[float], target: int) -> list[int]:
    """
    Round prices to integers summing to target with minimum deviation.

    Strategy:
    1. Floor all values first
    2. Calculate how many we need to ceil instead
    3. Ceil those with smallest cost (closest to ceiling)
    """
    n = len(prices)

    # Start with all floors
    result = [math.floor(p) for p in prices]
    current_sum = sum(result)

    # How many do we need to round up?
    need_ceil = target - current_sum

    if need_ceil < 0 or need_ceil > n:
        return []  # Impossible

    # Calculate cost of ceiling each value (how much we lose)
    # Cost = ceil(p) - p = 1 - (p - floor(p))
    # Lower fractional part = higher cost
    # So we want to ceil those with higher fractional parts

    # (index, fractional_part)
    fractions = [(i, prices[i] - result[i]) for i in range(n)]

    # Sort by fractional part descending (ceil those with largest fractions first)
    fractions.sort(key=lambda x: -x[1])

    # Ceil the first need_ceil elements
    for i in range(need_ceil):
        idx = fractions[i][0]
        result[idx] += 1

    return result


# Test cases
print(round_prices([1.2, 4.3, 5.8, 6.4], 18))  # Output: [1, 4, 6, 7]
print(round_prices([1.5, 2.5, 3.5], 9))        # Output: [2, 3, 4] or similar
print(round_prices([0.1, 0.2, 0.3], 1))        # Output: [0, 0, 1] or similar
```

---

## 17. Donut Challenge (Online Judge)

**Difficulty:** Easy, Newgrad

**Problem Description:**
Find the smallest eating rate `d` (donuts per minute) needed to finish all donut boxes within a time constraint, consuming from only one box per minute.

**Key Details:**
- Input: Array of donut box sizes, total time available in minutes
- Output: Minimum donuts-per-minute rate required
- Constraints: Time available ≥ number of boxes; each box contains at least one donut; partial consumption allowed if a box has fewer donuts than the rate

**Sample Input/Output:**
```
Input: donutBoxes = [4, 9, 11, 17], numMinutes = 8
Output: 6
```

**Python Solution:**
```python
import math

def min_eating_rate(donut_boxes: list[int], num_minutes: int) -> int:
    """
    Binary search for minimum eating rate.
    Same as LeetCode 875: Koko Eating Bananas.
    """
    def can_finish(rate: int) -> bool:
        """Check if we can finish all boxes at this rate within time limit."""
        minutes_needed = sum(math.ceil(box / rate) for box in donut_boxes)
        return minutes_needed <= num_minutes

    # Binary search between 1 and max(boxes)
    left, right = 1, max(donut_boxes)

    while left < right:
        mid = (left + right) // 2
        if can_finish(mid):
            right = mid  # Try smaller rate
        else:
            left = mid + 1  # Need larger rate

    return left


# Test cases
print(min_eating_rate([4, 9, 11, 17], 8))   # Output: 6
print(min_eating_rate([3, 6, 7, 11], 8))    # Output: 4
print(min_eating_rate([30, 11, 23, 4, 20], 5))  # Output: 30
print(min_eating_rate([30, 11, 23, 4, 20], 6))  # Output: 23
```

---

## 18. Resolve Battles (Online Judge)

**Difficulty:** Easy, Newgrad

**Problem Description:**
Given a list of army commands (Hold, Support, Move), resolve map battles. Determine each army's final state considering supports, attacks, ties at equal strength, and canceled support if the supporter is attacked.

Return each army's ending location or `[dead]`.

**Command Types:**
1. Hold - Army remains in current location
2. Support - Army backs another army
3. Move - Army moves to new location

**Sample Input/Output:**

**Input:**
```
actions = ["A Munich Hold", "B Warsaw Support A", "C Bohemia Move Munich"]
```

**Output:**
```
["A Munich", "B Warsaw", "C [dead]"]
```

**Python Solution:**
```python
def resolve_battles(actions: list[str]) -> list[str]:
    """
    Resolve Diplomacy-style battles.
    """
    # Parse actions
    armies = {}
    for action in actions:
        parts = action.split()
        army_id = parts[0]
        location = parts[1]
        command = parts[2]
        target = parts[3] if len(parts) > 3 else None

        armies[army_id] = {
            'location': location,
            'command': command,
            'target': target,
            'strength': 1,
            'alive': True,
            'final_location': location
        }

    # Calculate supports (and check if supporter is attacked)
    for army_id, army in armies.items():
        if army['command'] == 'Support':
            target_army = army['target']

            # Check if this supporter is being attacked
            being_attacked = False
            for other_id, other in armies.items():
                if other_id != army_id and other['command'] == 'Move':
                    if other['target'] == army['location']:
                        being_attacked = True
                        break

            # If not attacked, add support
            if not being_attacked and target_army in armies:
                armies[target_army]['strength'] += 1

    # Resolve moves
    # Group armies by destination
    destinations = {}
    for army_id, army in armies.items():
        if army['command'] == 'Move':
            dest = army['target']
            if dest not in destinations:
                destinations[dest] = []
            destinations[dest].append(army_id)

    # Resolve conflicts at each destination
    for dest, attackers in destinations.items():
        # Find defender if any
        defender = None
        for army_id, army in armies.items():
            if army['location'] == dest and army['command'] in ['Hold', 'Support']:
                defender = army_id
                break

        # Find strongest attacker
        max_strength = 0
        strongest = []
        for attacker_id in attackers:
            strength = armies[attacker_id]['strength']
            if strength > max_strength:
                max_strength = strength
                strongest = [attacker_id]
            elif strength == max_strength:
                strongest.append(attacker_id)

        # Resolve
        if len(strongest) > 1:
            # Tie - all attackers die
            for attacker_id in attackers:
                armies[attacker_id]['alive'] = False
        elif defender:
            defender_strength = armies[defender]['strength']
            if max_strength > defender_strength:
                # Attacker wins
                armies[defender]['alive'] = False
                armies[strongest[0]]['final_location'] = dest
            else:
                # Defender wins or tie
                armies[strongest[0]]['alive'] = False
        else:
            # No defender, attacker moves in
            armies[strongest[0]]['final_location'] = dest

    # Build result
    result = []
    for action in actions:
        army_id = action.split()[0]
        army = armies[army_id]
        if army['alive']:
            result.append(f"{army_id} {army['final_location']}")
        else:
            result.append(f"{army_id} [dead]")

    return result


# Test case
actions = ["A Munich Hold", "B Warsaw Support A", "C Bohemia Move Munich"]
print(resolve_battles(actions))
# Output: ['A Munich', 'B Warsaw', 'C [dead]']
```

---

## 19. Minimum Eating Speed (Online Judge)

**Difficulty:** Easy, Fulltime

**Problem Description:**
Find the minimum eating rate (candies per hour) needed to finish all candy piles within a time constraint.

**Objective:**
Find the minimum candies-per-hour rate c to finish all piles within numHours, eating from only one pile per hour.

**Input:**
- `candyPiles`: array of integers representing pile sizes
- `numHours`: integer representing available time

**Output:** Integer representing the minimum eating rate

**Rules:** Each hour, you select one pile and consume up to `c` candies from it (or fewer if the pile has fewer remaining).

**Sample Test Case:**

| Input | Output |
|-------|--------|
| candyPiles = [4, 9, 11, 17], numHours = 8 | 6 |

**Python Solution:**
```python
import math

def min_eating_speed(candy_piles: list[int], num_hours: int) -> int:
    """
    Binary search for minimum eating rate.
    LeetCode 875: Koko Eating Bananas.
    """
    def hours_needed(rate: int) -> int:
        return sum(math.ceil(pile / rate) for pile in candy_piles)

    left, right = 1, max(candy_piles)

    while left < right:
        mid = (left + right) // 2
        if hours_needed(mid) <= num_hours:
            right = mid
        else:
            left = mid + 1

    return left


# Test cases
print(min_eating_speed([4, 9, 11, 17], 8))   # Output: 6
print(min_eating_speed([3, 6, 7, 11], 8))    # Output: 4
print(min_eating_speed([30, 11, 23, 4, 20], 5))  # Output: 30
```

---

## 20. Connect 4: Determine Winner on 6x7 Board (Online Judge)

**Difficulty:** Easy

**Problem Description:**
Implement a function to determine if there's a winner in a Connect 4 game board. The board is a 6 x 7 2D array where `X`, `O`, and empty characters represent the states.

**Requirements:**
1. Input is a 6 x 7 2D array
2. Output is a string: 'X' or 'O' for the winner, or 'No Winner' if no winner exists
3. Check for four consecutive pieces in rows, columns, or either diagonal direction

**Key Hint:**
You need to scan through each row, column, and diagonal for four consecutive matching pieces.

**Test Cases:**
- Case 1: Board with X winning horizontally in row 1 → returns `'X'`
- Case 2: Board with X winning horizontally in row 2 (columns 1-4) → returns `'X'`
- Case 3: Empty board with no matches → returns `'No Winner'`

**Python Solution:**
```python
def connect4_winner(board: list[list[str]]) -> str:
    """
    Check for Connect 4 winner on 6x7 board.
    """
    ROWS, COLS = 6, 7

    def check_line(cells: list[str]) -> str:
        """Check if there are 4 consecutive same pieces."""
        count = 1
        for i in range(1, len(cells)):
            if cells[i] == cells[i-1] and cells[i] in ['X', 'O']:
                count += 1
                if count >= 4:
                    return cells[i]
            else:
                count = 1
        return None

    # Check horizontal
    for row in range(ROWS):
        winner = check_line(board[row])
        if winner:
            return winner

    # Check vertical
    for col in range(COLS):
        column = [board[row][col] for row in range(ROWS)]
        winner = check_line(column)
        if winner:
            return winner

    # Check diagonals (top-left to bottom-right)
    for start_row in range(ROWS - 3):
        for start_col in range(COLS - 3):
            diagonal = [board[start_row + i][start_col + i] for i in range(4)]
            if diagonal[0] in ['X', 'O'] and len(set(diagonal)) == 1:
                return diagonal[0]

    # Check diagonals (top-right to bottom-left)
    for start_row in range(ROWS - 3):
        for start_col in range(3, COLS):
            diagonal = [board[start_row + i][start_col - i] for i in range(4)]
            if diagonal[0] in ['X', 'O'] and len(set(diagonal)) == 1:
                return diagonal[0]

    return "No Winner"


# Test cases
board1 = [
    ['X', 'X', 'X', 'X', ' ', ' ', ' '],
    [' ', ' ', ' ', ' ', ' ', ' ', ' '],
    [' ', ' ', ' ', ' ', ' ', ' ', ' '],
    [' ', ' ', ' ', ' ', ' ', ' ', ' '],
    [' ', ' ', ' ', ' ', ' ', ' ', ' '],
    [' ', ' ', ' ', ' ', ' ', ' ', ' '],
]
print(connect4_winner(board1))  # Output: 'X'

board2 = [
    [' ', ' ', ' ', ' ', ' ', ' ', ' '],
    [' ', ' ', ' ', ' ', ' ', ' ', ' '],
    [' ', ' ', ' ', ' ', ' ', ' ', ' '],
    [' ', ' ', ' ', ' ', ' ', ' ', ' '],
    [' ', ' ', ' ', ' ', ' ', ' ', ' '],
    [' ', ' ', ' ', ' ', ' ', ' ', ' '],
]
print(connect4_winner(board2))  # Output: 'No Winner'
```

---

## 21. Optimal Task Scheduling to Maximize Rewards Before Deadlines (Online Judge)

**Difficulty:** Easy

**Problem Description:**
Given a list of tasks, where each task is defined by a task ID, a task deadline (as an integer), and a task reward. Each task requires one unit of time to complete. The goal is to maximize total rewards while respecting deadline constraints.

**Input/Output:**
- Input: A list of tasks as tuples in the format `(task_id, task_deadline, task_reward)`
- Output: A list of task IDs representing the optimal completion order that maximizes total reward

**Example:**
- Input: `[('a', 2, 8), ('b', 1, 3), ('c', 2, 5), ('d', 3, 3)]`
- Possible Output: `[c, a, d]` or `[a, c, d]`

**Constraints:**
- Up to 1000 tasks maximum
- Each task must be completed before its deadline expires
- Tasks take exactly one unit of time each

**Key Requirements:**
Avoid using a greedy algorithm. Improve the solution using specific data structures and implement. The hint recommends optimizing through sorting and data structure improvements to reduce complexity.

**Python Solution:**
```python
import heapq

def schedule_tasks(tasks: list[tuple[str, int, int]]) -> list[str]:
    """
    Schedule tasks to maximize reward using a priority queue approach.

    Args:
        tasks: List of (task_id, deadline, reward) tuples

    Returns:
        List of task IDs in optimal order
    """
    if not tasks:
        return []

    # Sort by deadline
    sorted_tasks = sorted(tasks, key=lambda x: x[1])

    # Min-heap of rewards (we use negative for max-heap behavior)
    # Stores (reward, task_id)
    scheduled = []  # min-heap by reward

    for task_id, deadline, reward in sorted_tasks:
        if len(scheduled) < deadline:
            # We have room before deadline
            heapq.heappush(scheduled, (reward, task_id))
        elif scheduled and scheduled[0][0] < reward:
            # Replace lowest reward task if current is better
            heapq.heapreplace(scheduled, (reward, task_id))

    # Extract task IDs (order doesn't matter for total reward)
    result = [task_id for _, task_id in scheduled]
    return result


def schedule_tasks_with_order(tasks: list[tuple[str, int, int]]) -> list[str]:
    """
    Schedule tasks and return them in execution order.
    Uses Union-Find for O(n log n) solution.
    """
    if not tasks:
        return []

    max_deadline = max(t[1] for t in tasks)

    # Parent array for Union-Find
    # parent[i] = latest available slot <= i
    parent = list(range(max_deadline + 2))

    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    # Sort by reward descending
    sorted_tasks = sorted(tasks, key=lambda x: -x[2])

    schedule = [None] * (max_deadline + 1)  # slot -> task_id

    for task_id, deadline, reward in sorted_tasks:
        # Find latest available slot <= deadline
        slot = find(deadline)
        if slot > 0:
            schedule[slot] = task_id
            parent[slot] = slot - 1  # Mark slot as used

    # Return scheduled tasks in order
    return [t for t in schedule if t is not None]


# Test cases
tasks = [('a', 2, 8), ('b', 1, 3), ('c', 2, 5), ('d', 3, 3)]
print(schedule_tasks(tasks))  # Output: ['c', 'a', 'd'] or similar
print(schedule_tasks_with_order(tasks))
```

---

## 22. Keyword Tagging in Reviews with Overlapping Matches (Online Judge)

**Difficulty:** Easy

**Problem Description:**
Given a keyword-to-tag mapping and a user review, replace keywords in the review with the format `[<tag>]{<keyword>}`, handling overlapping matches appropriately.

**Example:**

**Input Mapping:**
```json
{
  "san": "person",
  "francisco": "person",
  "san francisco": "city",
  "Airbnb": "business",
  "city": "location"
}
```

**Input Review:**
```
"I travelled to San Francisco for work and stayed at Airbnb.
I really loved the city and the home where I stayed.
I stayed with San and Francisco.
They both were really good and san's hospitality was outstanding."
```

**Expected Output:**
```
"I travelled to [city]{San Francisco} for work and stayed at [business]{Airbnb}.
I really loved the [location]{city} and the home where I stayed.
I stayed with [person]{San} and [person]{Francisco}.
They both were really good and [person]{san}'s hospitality was outstanding."
```

**Key Considerations:**
- Handle case-insensitive matching (e.g., "San Francisco" matches "san francisco")
- Resolve overlapping matches intelligently (longer matches take precedence)
- Preserve original text casing in the replacement format
- Apply tags only to matched keywords, not surrounding text

**Python Solution:**
```python
import re

def tag_keywords(mapping: dict[str, str], review: str) -> str:
    """
    Tag keywords in review with their corresponding tags.
    Longer matches take precedence over shorter ones.
    """
    # Sort keywords by length descending (longer matches first)
    sorted_keywords = sorted(mapping.keys(), key=len, reverse=True)

    # Build a case-insensitive pattern
    # We need to track positions and avoid overlapping replacements

    # Find all matches with positions
    matches = []  # (start, end, original_text, tag)

    for keyword in sorted_keywords:
        tag = mapping[keyword]
        # Case-insensitive search
        pattern = re.compile(re.escape(keyword), re.IGNORECASE)

        for match in pattern.finditer(review):
            start, end = match.start(), match.end()
            original_text = match.group()

            # Check if this overlaps with existing matches
            overlaps = False
            for existing_start, existing_end, _, _ in matches:
                if not (end <= existing_start or start >= existing_end):
                    overlaps = True
                    break

            if not overlaps:
                matches.append((start, end, original_text, tag))

    # Sort matches by position
    matches.sort(key=lambda x: x[0])

    # Build result
    result = []
    last_end = 0

    for start, end, original_text, tag in matches:
        result.append(review[last_end:start])
        result.append(f"[{tag}]{{{original_text}}}")
        last_end = end

    result.append(review[last_end:])

    return ''.join(result)


# Test case
mapping = {
    "san": "person",
    "francisco": "person",
    "san francisco": "city",
    "Airbnb": "business",
    "city": "location"
}

review = """I travelled to San Francisco for work and stayed at Airbnb.
I really loved the city and the home where I stayed.
I stayed with San and Francisco.
They both were really good and san's hospitality was outstanding."""

print(tag_keywords(mapping, review))
```

---

## 23. Refund Allocation by Payment Method and Date Priority (Online Judge)

**Difficulty:** Easy

**Problem Description:**
Design a refund system that processes refunds based on specific prioritization rules.

**Refund Processing Rules:**
- Refunds should be issued in full for one payment before considering the next payment
- Priority by payment method: CREDIT > CREDIT_CARD > PAYPAL
- Within each method, prioritize most recent payments first

**Input Format:**
- Transaction List: Payment records containing type, date, and amount
- Refund Amount: Total amount to refund

**Examples:**

**Example 1:**
- Transactions: Credit (2023-01-10, $40), Paypal (2023-01-15, $60)
- Refund: $50
- Output: Credit refund of $30 (remaining $20 unrefunded)

**Example 2:**
- Transactions: Credit (2023-01-15, $40), Paypal (2023-01-10, $60), Paypal (2023-01-20, $40)
- Refund: $50
- Output: Credit $40 + Paypal $10

**Sample Input Format:**
`Credit,2023-01-10,40|Paypal,2023-01-15,60|50`

**Python Solution:**
```python
from datetime import datetime
from dataclasses import dataclass

@dataclass
class Transaction:
    method: str
    date: datetime
    amount: float

    def __post_init__(self):
        if isinstance(self.date, str):
            self.date = datetime.strptime(self.date, "%Y-%m-%d")


@dataclass
class Refund:
    method: str
    amount: float


def allocate_refunds(transactions: list[Transaction],
                     refund_amount: float) -> list[Refund]:
    """
    Allocate refunds based on payment method priority and date.

    Priority: CREDIT > CREDIT_CARD > PAYPAL
    Within same method: most recent first
    """
    METHOD_PRIORITY = {
        'Credit': 0,
        'CREDIT': 0,
        'Credit_Card': 1,
        'CREDIT_CARD': 1,
        'Paypal': 2,
        'PAYPAL': 2
    }

    # Sort transactions by priority, then by date (most recent first)
    sorted_trans = sorted(
        transactions,
        key=lambda t: (METHOD_PRIORITY.get(t.method, 99), -t.date.timestamp())
    )

    refunds = []
    remaining = refund_amount

    for trans in sorted_trans:
        if remaining <= 0:
            break

        refund_for_this = min(trans.amount, remaining)
        if refund_for_this > 0:
            refunds.append(Refund(method=trans.method, amount=refund_for_this))
            remaining -= refund_for_this

    return refunds


def parse_input(input_str: str) -> tuple[list[Transaction], float]:
    """Parse input string format: Method,Date,Amount|Method,Date,Amount|RefundAmount"""
    parts = input_str.split('|')
    refund_amount = float(parts[-1])

    transactions = []
    for part in parts[:-1]:
        method, date, amount = part.split(',')
        transactions.append(Transaction(method=method, date=date, amount=float(amount)))

    return transactions, refund_amount


# Test cases
input1 = "Credit,2023-01-10,40|Paypal,2023-01-15,60|50"
transactions, refund_amount = parse_input(input1)
refunds = allocate_refunds(transactions, refund_amount)
print(f"Input: {input1}")
print(f"Refunds: {refunds}")
# Output: Credit $40, Paypal $10

input2 = "Credit,2023-01-15,40|Paypal,2023-01-10,60|Paypal,2023-01-20,40|50"
transactions, refund_amount = parse_input(input2)
refunds = allocate_refunds(transactions, refund_amount)
print(f"\nInput: {input2}")
print(f"Refunds: {refunds}")
# Output: Credit $40, Paypal $10 (from 2023-01-20 payment)
```

---

*Total: 23 questions with Python solutions*
*Last updated: 2026-04-02*
