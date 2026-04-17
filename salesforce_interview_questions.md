# Salesforce Interview Questions

---

## 1. Minimum Removals to Make Array Almost Sorted

**Problem Statement:**
An array of integers is almost sorted if at most one element can be deleted from it to make it perfectly sorted in ascending order.

**Examples of Almost Sorted Arrays:**
- `[2, 1, 7]` - remove 2 to get [1, 7] (sorted)
- `[1, 5, 6]` - already sorted (remove 0 elements)
- `[3, 1, 2]` - remove 3 to get [1, 2] (sorted)

**Examples of NOT Almost Sorted Arrays:**
- `[4, 2, 1]` - cannot make sorted by removing just one element
- `[1, 2, 6, 4, 3]` - more than one element out of place

**Task:**
Given an array of n unique integers, determine the minimum number of elements to remove so that the remaining array becomes almost sorted.

**Input/Output:**
- Input: Array of n unique integers
- Output: Minimum number of elements to remove

**Sample Test Cases:**

| Input | Output | Explanation |
|-------|--------|-------------|
| `[1, 2, 3, 4, 5]` | 0 | Already sorted (almost sorted with 0 removals needed) |
| `[5, 4, 3, 2, 1]` | 3 | Remove 3 elements to get [2, 1] or [4, 3], which are almost sorted |
| `[1, 3, 2, 4, 5]` | 0 | Already almost sorted (remove 3 or 2 to get sorted) |
| `[4, 2, 1]` | 1 | Remove one element to get [2, 1] or [4, 1], which are almost sorted |
| `[1, 2, 6, 4, 3]` | 2 | Remove 2 elements to get an almost sorted array |

**Key Insight:**
An array is "almost sorted" if it has at most one descent (position where `arr[i] > arr[i+1]`). To minimize removals:
1. Find the longest subsequence that is either:
   - Strictly increasing (0 descents), OR
   - Has exactly one descent (can remove one element to be sorted)

**Solution Approach:**
This problem can be solved using dynamic programming, building on the Longest Increasing Subsequence (LIS) concept. We need to find the longest "almost increasing" subsequence - one that has at most one violation.

**Python Solution:**
```python
def min_removals_almost_sorted(arr: list[int]) -> int:
    """
    Find minimum removals to make array almost sorted.
    Almost sorted = at most one element can be removed to make it sorted.

    Approach: Find the longest subsequence with at most one "descent"
    (position where arr[i] > arr[i+1]).
    """
    n = len(arr)
    if n <= 2:
        return 0  # Any array of length <= 2 is almost sorted

    # dp0[i] = length of longest increasing subsequence ending at i
    # dp1[i] = length of longest "almost increasing" subsequence ending at i
    #          (has exactly one descent)

    dp0 = [1] * n  # LIS ending at each position
    dp1 = [1] * n  # Almost increasing ending at each position

    for i in range(1, n):
        for j in range(i):
            if arr[j] < arr[i]:
                # Extend increasing subsequence
                dp0[i] = max(dp0[i], dp0[j] + 1)
                # Extend almost increasing subsequence
                dp1[i] = max(dp1[i], dp1[j] + 1)
            else:
                # arr[j] >= arr[i]: this creates a descent
                # Can only extend dp0 to dp1 (adding first descent)
                dp1[i] = max(dp1[i], dp0[j] + 1)

    # Maximum length of almost sorted subsequence
    max_almost_sorted = max(max(dp0), max(dp1))

    return n - max_almost_sorted


def min_removals_almost_sorted_v2(arr: list[int]) -> int:
    """
    Alternative approach using binary search for O(n log n) complexity.

    Key insight: An almost sorted array has at most one descent.
    We find the longest subsequence where removing at most one element
    makes it strictly increasing.
    """
    from bisect import bisect_left

    n = len(arr)
    if n <= 2:
        return 0

    def lis_length(nums):
        """Standard LIS using binary search - O(n log n)"""
        tails = []
        for num in nums:
            pos = bisect_left(tails, num)
            if pos == len(tails):
                tails.append(num)
            else:
                tails[pos] = num
        return len(tails)

    # For almost sorted, we need longest subsequence with at most 1 descent
    # Method: For each position, compute:
    # - LIS ending at that position (from left)
    # - LIS starting at that position (from right, reversed)

    # lis_left[i] = length of LIS ending at index i
    lis_left = [1] * n
    tails = []
    for i in range(n):
        pos = bisect_left(tails, arr[i])
        lis_left[i] = pos + 1
        if pos == len(tails):
            tails.append(arr[i])
        else:
            tails[pos] = arr[i]

    # lis_right[i] = length of LIS starting at index i
    lis_right = [1] * n
    tails = []
    for i in range(n - 1, -1, -1):
        # We need increasing from i onwards, so we look for
        # longest increasing subsequence in arr[i:]
        # Use negative values for reverse processing
        pos = bisect_left(tails, -arr[i])
        lis_right[i] = pos + 1
        if pos == len(tails):
            tails.append(-arr[i])
        else:
            tails[pos] = -arr[i]

    # Case 1: The longest increasing subsequence (no descent)
    max_len = max(lis_left)

    # Case 2: Allow one descent - try connecting lis_left[i] with lis_right[j]
    # where i < j and we "skip" elements between them
    # This effectively allows one element to be removed from the final result

    # For each pair (i, j) where arr[i] < arr[j], we can form
    # a valid almost-sorted by taking LIS ending at i + 1 + LIS starting at j
    # (the +1 accounts for the potential removal point)

    for i in range(n):
        for j in range(i + 1, n):
            if arr[i] < arr[j]:
                # Valid connection: can form almost sorted of length
                # lis_left[i] + lis_right[j]
                max_len = max(max_len, lis_left[i] + lis_right[j])

    return n - max_len


def min_removals_simple(arr: list[int]) -> int:
    """
    Simplified O(n^2) approach focusing on clarity.

    An array is almost sorted if it has at most one inversion point
    (where arr[i] > arr[i+1]).

    Find longest subsequence with at most one such inversion.
    """
    n = len(arr)
    if n <= 2:
        return 0

    max_len = 0

    # Try all possible subsequences represented by keeping track of:
    # - Last value in the increasing part
    # - Whether we've used our "one descent" allowance

    # dp[last_val][used_descent] = max length
    # But values can be large, so we use index-based DP

    # For each starting position, find longest almost-increasing subsequence
    for start in range(n):
        # Greedy extension from start
        length = 1
        descents = 0
        last = arr[start]

        for i in range(start + 1, n):
            if arr[i] > last:
                length += 1
                last = arr[i]
            elif descents == 0:
                # Use our one descent allowance
                # But we need to be smart about whether to include this element
                # Option 1: Skip this element entirely (don't use descent)
                # Option 2: Include it and mark descent used
                pass  # This greedy approach is incomplete

        max_len = max(max_len, length)

    # Full DP solution
    # State: (index, last_included_index, descent_used)
    from functools import lru_cache

    @lru_cache(maxsize=None)
    def dp(idx, last_idx, descent_used):
        """
        Returns max length of almost-sorted subsequence starting from idx,
        with last included element at last_idx,
        and descent_used indicating if we've had a descent.
        """
        if idx == n:
            return 0

        # Option 1: Skip current element
        result = dp(idx + 1, last_idx, descent_used)

        # Option 2: Include current element
        if last_idx == -1:
            # First element
            result = max(result, 1 + dp(idx + 1, idx, descent_used))
        elif arr[idx] > arr[last_idx]:
            # Continues increasing
            result = max(result, 1 + dp(idx + 1, idx, descent_used))
        elif not descent_used:
            # Creates a descent (first one)
            result = max(result, 1 + dp(idx + 1, idx, True))

        return result

    max_len = dp(0, -1, False)
    return n - max_len


# Test cases
print(min_removals_almost_sorted([1, 2, 3, 4, 5]))     # 0
print(min_removals_almost_sorted([5, 4, 3, 2, 1]))     # 3
print(min_removals_almost_sorted([1, 3, 2, 4, 5]))     # 0
print(min_removals_almost_sorted([4, 2, 1]))           # 1
print(min_removals_almost_sorted([1, 2, 6, 4, 3]))     # 2
print(min_removals_almost_sorted([2, 1, 7]))           # 0
```

**Time Complexity:** O(n²) for the DP solution, can be optimized to O(n log n) with binary search

**Space Complexity:** O(n) for the DP arrays

**Key Points:**
1. An array is "almost sorted" if it has at most one descent
2. We need to find the longest subsequence with ≤1 descent
3. The answer is n minus the length of this longest subsequence
4. The DP tracks both regular LIS and "almost increasing" subsequences

---

## 2. String Compression

**Problem Statement:**
Given a string, write a function to compress it by shortening every sequence of the same character to that character followed by the number of repetitions. If the compressed string is longer than the original, return the original string.

**Function Signature:**
```python
def compressString(input: str) -> str
```

**Parameters:**
- `input`: the string to compress

**Returns:**
- The compressed string, or the original string if the compressed one is longer

**Example:**
```
Input: "abaasass"
Output: "a1b1a2s1a1s2"

Breakdown:
- 'a' (1) → "a1"
- 'b' (1) → "b1"
- 'a' (2) → "a2"
- 's' (1) → "s1"
- 'a' (1) → "a1"
- 's' (2) → "s2"
```

**Sample Test Cases:**

| Input | Output | Explanation |
|-------|--------|-------------|
| `"abaasass"` | `"a1b1a2s1a1s2"` | Each run compressed with count |
| `"aabcccccaaa"` | `"a2b1c5a3"` | Compressed (10) shorter than original (11) |
| `"abcdef"` | `"abcdef"` | Compressed "a1b1c1d1e1f1" (12) > original (6) |
| `"aaaaaa"` | `"a6"` | Significant compression |
| `""` | `""` | Empty string edge case |
| `"a"` | `"a"` | Single char: "a1" (2) > "a" (1) |

**Key Observations:**
1. Compression is beneficial when there are repeated characters
2. For all unique characters, compressed string is always longer (each char becomes 2+ chars)
3. Need to handle edge cases: empty string, single character

**Python Solution:**
```python
def compressString(input: str) -> str:
    """
    Compress string by replacing consecutive characters with char + count.
    Returns original if compressed is longer.
    """
    if not input:
        return input

    compressed = []
    count = 1

    for i in range(1, len(input)):
        if input[i] == input[i - 1]:
            count += 1
        else:
            compressed.append(input[i - 1] + str(count))
            count = 1

    # Don't forget the last group
    compressed.append(input[-1] + str(count))

    result = ''.join(compressed)

    # Return original if compressed is not shorter
    return result if len(result) < len(input) else input


def compressString_v2(input: str) -> str:
    """
    Alternative using itertools.groupby for cleaner code.
    """
    from itertools import groupby

    if not input:
        return input

    compressed = ''.join(f"{char}{len(list(group))}"
                         for char, group in groupby(input))

    return compressed if len(compressed) < len(input) else input


def compressString_v3(input: str) -> str:
    """
    Optimized version that stops early if compressed exceeds original length.
    """
    if not input:
        return input

    n = len(input)
    compressed = []
    total_len = 0
    count = 1

    for i in range(1, n + 1):
        if i < n and input[i] == input[i - 1]:
            count += 1
        else:
            # Add character and count
            compressed.append(input[i - 1])
            count_str = str(count)
            compressed.append(count_str)

            total_len += 1 + len(count_str)

            # Early termination if already longer
            if total_len >= n:
                return input

            count = 1

    return ''.join(compressed)


# Test cases
print(compressString("abaasass"))      # "a1b1a2s1a1s2"
print(compressString("aabcccccaaa"))   # "a2b1c5a3"
print(compressString("abcdef"))        # "abcdef" (original)
print(compressString("aaaaaa"))        # "a6"
print(compressString(""))              # ""
print(compressString("a"))             # "a" (original)
print(compressString("aa"))            # "aa" (original, "a2" same length)
print(compressString("aaa"))           # "a3" (compressed shorter)
```

**Time Complexity:** O(n) where n is the length of the input string

**Space Complexity:** O(n) for storing the compressed result

**Follow-up Variations:**
1. What if counts > 9? (Already handled - counts can be multi-digit)
2. What if we only write count when > 1? (e.g., "aabbb" → "a2b3" or "a2b3")
3. In-place compression with fixed-size buffer?

---

## 3. Longest Subsequence Which Is Substring

**Problem Statement:**
You are given two strings `x` and `y`. Find the length of the longest subsequence of `x` that is also a substring of `y`.

**Definitions:**
- **Subsequence**: A sequence derived from another string by deleting some or no elements without changing the order of remaining elements
- **Substring**: A contiguous part of a string

**Function Signature:**
```python
def longestSubsequenceWhichIsSubstring(x: str, y: str) -> int
```

**Parameters:**
- `x`: the first string (we take subsequences from this)
- `y`: the second string (we look for substrings in this)

**Returns:**
- The length of the longest subsequence of `x` that is also a substring of `y`

**Example:**
```
Input: x = "abcd", y = "abdc"
Output: 3

Explanation:
- Substrings of y: "a", "b", "d", "c", "ab", "bd", "dc", "abd", "bdc", "abdc"
- Check which are subsequences of x = "abcd":
  - "abd" ✓ (a→b→d exists in order in "abcd")
  - "bdc" ✓ (b→d→c exists in order in "abcd")
  - "abdc" ✗ (would need d before c, but in "abcd" c comes before d)
- Longest valid: "abd" or "bdc" with length 3
```

**Sample Test Cases:**

| Input | Output | Explanation |
|-------|--------|-------------|
| `x="abcd", y="abdc"` | 3 | "abd" or "bdc" |
| `x="abc", y="abc"` | 3 | "abc" is both subsequence and substring |
| `x="abc", y="def"` | 0 | No common characters |
| `x="abcde", y="ace"` | 3 | "ace" is substring of y and subsequence of x |
| `x="ab", y="ba"` | 1 | Only single chars match |

**Approach:**
1. Generate all substrings of `y` (O(N²) substrings where N = len(y))
2. For each substring, check if it's a subsequence of `x` using two pointers (O(M) where M = len(x))
3. Track the maximum length found

**Python Solution:**
```python
def longestSubsequenceWhichIsSubstring(x: str, y: str) -> int:
    """
    Find longest subsequence of x that is also a substring of y.

    Approach: Generate all substrings of y, check if each is a subsequence of x.
    Time: O(N² * M) where N = len(y), M = len(x)
    """

    def is_subsequence(subseq: str, string: str) -> bool:
        """Check if subseq is a subsequence of string using two pointers."""
        if not subseq:
            return True

        j = 0  # Pointer for subseq
        for char in string:
            if char == subseq[j]:
                j += 1
                if j == len(subseq):
                    return True
        return False

    max_length = 0
    n = len(y)

    # Generate all substrings of y
    for i in range(n):
        for j in range(i + 1, n + 1):
            substring = y[i:j]

            # Early termination: skip if length <= current max
            if len(substring) <= max_length:
                continue

            # Check if this substring is a subsequence of x
            if is_subsequence(substring, x):
                max_length = len(substring)

    return max_length


def longestSubsequenceWhichIsSubstring_optimized(x: str, y: str) -> int:
    """
    Optimized version: iterate substrings by decreasing length.
    Returns as soon as we find a valid one.
    """

    def is_subsequence(subseq: str, string: str) -> bool:
        j = 0
        for char in string:
            if j < len(subseq) and char == subseq[j]:
                j += 1
        return j == len(subseq)

    n = len(y)

    # Check substrings from longest to shortest
    for length in range(n, 0, -1):
        for start in range(n - length + 1):
            substring = y[start:start + length]
            if is_subsequence(substring, x):
                return length

    return 0


def longestSubsequenceWhichIsSubstring_dp(x: str, y: str) -> int:
    """
    DP approach using preprocessing for faster subsequence checks.

    Precompute: next[i][c] = next position of character c at or after index i in x
    Time: O(M * 26 + N² * N) preprocessing + checking
    """
    m, n = len(x), len(y)

    if m == 0 or n == 0:
        return 0

    # Precompute next occurrence of each character in x
    # next_pos[i][c] = smallest j >= i such that x[j] == c, or -1 if none
    INF = m + 1
    next_pos = [[INF] * 26 for _ in range(m + 1)]

    # Fill from right to left
    for i in range(m - 1, -1, -1):
        for c in range(26):
            next_pos[i][c] = next_pos[i + 1][c]
        next_pos[i][ord(x[i]) - ord('a')] = i

    def is_subsequence_fast(subseq: str) -> bool:
        """Check subsequence using precomputed next positions."""
        pos = 0
        for char in subseq:
            c = ord(char) - ord('a')
            if pos >= m or next_pos[pos][c] == INF:
                return False
            pos = next_pos[pos][c] + 1
        return True

    max_length = 0

    for i in range(n):
        for j in range(i + 1, n + 1):
            if j - i <= max_length:
                continue
            if is_subsequence_fast(y[i:j]):
                max_length = j - i

    return max_length


# Test cases
print(longestSubsequenceWhichIsSubstring("abcd", "abdc"))     # 3
print(longestSubsequenceWhichIsSubstring("abc", "abc"))       # 3
print(longestSubsequenceWhichIsSubstring("abc", "def"))       # 0
print(longestSubsequenceWhichIsSubstring("abcde", "ace"))     # 3
print(longestSubsequenceWhichIsSubstring("ab", "ba"))         # 1
print(longestSubsequenceWhichIsSubstring("", "abc"))          # 0
print(longestSubsequenceWhichIsSubstring("abc", ""))          # 0
```

**Time Complexity:**
- Basic: O(N² × M) where N = len(y), M = len(x)
- Optimized with early return: Still O(N² × M) worst case, but faster in practice
- DP with preprocessing: O(M × 26 + N² × M) but subsequence checks are O(length) instead of O(M)

**Space Complexity:** O(1) for basic, O(M × 26) for DP version

**Key Insights:**
1. We iterate over substrings of y (not x) because substrings are contiguous
2. Subsequence check uses two-pointer technique in O(M) time
3. Optimization: check longer substrings first and return early
4. Further optimization: precompute character positions for O(1) lookups

---

## 4. Minimum Operations to Reduce an Integer to 0

**LeetCode 2571**

**Problem Statement:**
You are given a positive integer `n`. You can perform the following operation any number of times:
- Add or subtract a power of 2 from `n`

Return the minimum number of operations to make `n` equal to 0.

**Function Signature:**
```python
def minOperations(n: int) -> int
```

**Examples:**

```
Example 1:
Input: n = 39
Output: 3
Explanation:
- 39 + 1 = 40  (40 = 2^5 + 2^3)
- 40 - 8 = 32  (32 = 2^5)
- 32 - 32 = 0
Total: 3 operations

Example 2:
Input: n = 54
Output: 3
Explanation:
- 54 + 2 = 56  (56 = 2^5 + 2^4 + 2^3)
- 56 - 8 = 48
- 48 - 48 = 0  (or 56 + 8 = 64, 64 - 64 = 0)
```

**Sample Test Cases:**

| Input | Binary | Output | Explanation |
|-------|--------|--------|-------------|
| 39 | 100111 | 3 | Add 1 → subtract 8 → subtract 32 |
| 54 | 110110 | 3 | Handle consecutive 1s efficiently |
| 7 | 111 | 2 | 7+1=8, 8-8=0 (better than 3 subtractions) |
| 8 | 1000 | 1 | Single power of 2: just subtract |
| 1 | 1 | 1 | Subtract 1 |
| 15 | 1111 | 2 | 15+1=16, 16-16=0 |

**Key Insight:**
The problem becomes clear when looking at binary representation:
- A **lone 1 bit** (surrounded by 0s): subtract that power of 2 (1 operation)
- **Consecutive 1s** (2+ in a row): add 1 to flip them all to 0s with a carry (often more efficient)

**Why consecutive 1s benefit from adding:**
- `7 = 111` in binary
- Subtracting each: 7-1-2-4 = 0 → 3 operations
- Adding first: 7+1 = 8 = 1000, then 8-8 = 0 → 2 operations

**Algorithm:**
```
while n > 0:
    if last two bits are "11" (consecutive 1s):
        add 1 (to flip consecutive 1s)
        operations++
    else if last bit is "1" (lone 1):
        subtract 1
        operations++
    right shift n
```

**Python Solution:**
```python
def minOperations(n: int) -> int:
    """
    Greedy bit manipulation approach.

    Key insight:
    - Lone 1 bit: subtract (1 op)
    - Consecutive 1s: add 1 to flip them (often saves operations)

    Time: O(log n)
    Space: O(1)
    """
    ops = 0

    while n:
        if n & 1:  # LSB is 1
            if n & 2:  # Second bit also 1 → consecutive 1s
                n += 1  # Add 1 to handle consecutive 1s
            else:  # Lone 1
                n -= 1  # Subtract 1
            ops += 1
        n >>= 1  # Right shift

    return ops


def minOperations_v2(n: int) -> int:
    """
    Alternative using popcount insight.

    Observation: We're essentially counting "groups" of 1s,
    where each group costs 1 or 2 operations depending on length.

    Using lowbit trick: n & (-n) gives the lowest set bit.
    """
    ops = 0

    while n:
        lowbit = n & (-n)

        # Check if there are consecutive 1s
        if n & (lowbit << 1):
            # Consecutive 1s: add lowbit to flip them
            n += lowbit
        else:
            # Lone 1: subtract it
            n -= lowbit
        ops += 1

    return ops


def minOperations_recursive(n: int) -> int:
    """
    Recursive approach with memoization.

    At each step, if n is odd:
    - Option 1: subtract 1 (go to n-1)
    - Option 2: add 1 (go to n+1)
    If n is even, just divide by 2 (no cost, it's a right shift).
    """
    from functools import lru_cache

    @lru_cache(maxsize=None)
    def dp(num):
        if num == 0:
            return 0
        if num == 1:
            return 1

        if num % 2 == 0:
            return dp(num // 2)
        else:
            # Try both adding and subtracting
            return 1 + min(dp((num + 1) // 2), dp((num - 1) // 2))

    return dp(n)


def minOperations_bfs(n: int) -> int:
    """
    BFS approach - guaranteed shortest path.

    States: current value
    Transitions: add/subtract any power of 2

    Note: This is less efficient but demonstrates the BFS pattern.
    """
    from collections import deque

    if n == 0:
        return 0

    visited = {n}
    queue = deque([(n, 0)])  # (value, operations)

    while queue:
        val, ops = queue.popleft()

        # Try all powers of 2 up to a reasonable limit
        power = 1
        while power <= 2 * val:
            for next_val in [val + power, val - power]:
                if next_val == 0:
                    return ops + 1
                if next_val > 0 and next_val not in visited:
                    visited.add(next_val)
                    queue.append((next_val, ops + 1))
            power <<= 1

    return -1  # Should never reach here


# Test cases
print(minOperations(39))   # 3
print(minOperations(54))   # 3
print(minOperations(7))    # 2
print(minOperations(8))    # 1
print(minOperations(1))    # 1
print(minOperations(15))   # 2
print(minOperations(27))   # 3  (11011 -> add 1 -> 11100, sub 4 -> 11000, sub 8+16=24? or different path)

# Trace through n=39 (binary: 100111):
# n=100111, bits 0&1 both 1 → add 1, ops=1, n=101000
# n=101000 → shift → n=10100
# n=10100 → shift → n=1010
# n=1010 → shift → n=101
# n=101, bit 0=1, bit 1=0 → subtract 1, ops=2, n=100
# n=100 → shift → n=10
# n=10 → shift → n=1
# n=1, bit 0=1, bit 1=0 → subtract 1, ops=3, n=0
# Result: 3 ✓
```

**Time Complexity:** O(log n) - we process each bit once (with potential carry)

**Space Complexity:** O(1) for iterative, O(log n) for recursive with memoization

**Key Points:**
1. Binary representation reveals the structure of the problem
2. Consecutive 1s benefit from "add 1" to create a carry that clears them
3. Lone 1s are best handled by direct subtraction
4. The greedy approach is optimal because we're minimizing operations per "group" of 1s
5. Each group of consecutive 1s costs at most 2 operations (add + subtract the result)

**Follow-up Questions:**
1. What if we can only add powers of 2? (Different problem - may not reach 0)
2. What if we want the actual sequence of operations, not just the count?
3. How does this relate to Gray code or binary representations?

---

## 5. Maximum Requests in a Time Window

**Problem Statement:**
You are given an integer array `timestamp`, where `timestamp[i]` is the time when a request was received, and an integer `windowSize`.

A time window is an inclusive interval: `[start, start + windowSize - 1]`

You may choose any valid start. Return the maximum number of requests whose timestamps fall within some window of size `windowSize`.

**Function Signature:**
```python
def maxRequests(timestamp: List[int], windowSize: int) -> int
```

**Input:**
- `timestamp`: an array of integers (not necessarily sorted; duplicates allowed)
- `windowSize`: a positive integer

**Output:**
- An integer: the maximum number of timestamps in any such window

**Constraints:**
- 1 ≤ n ≤ 2 × 10⁵
- 1 ≤ timestamp[i] ≤ 10⁹
- 1 ≤ windowSize ≤ 10⁹

**Examples:**

```
Example 1:
Input: timestamp = [1, 3, 7, 5], windowSize = 4
Output: 2

Explanation:
- Window [1, 4] contains timestamps: 1, 3 → 2 requests
- Window [3, 6] contains timestamps: 3, 5 → 2 requests
- Window [5, 8] contains timestamps: 5, 7 → 2 requests
Maximum is 2.

Example 2:
Input: timestamp = [2, 2, 3], windowSize = 1
Output: 2

Explanation:
- Window [2, 2] contains timestamps: 2, 2 → 2 requests
- Window [3, 3] contains timestamps: 3 → 1 request
Maximum is 2.
```

**Sample Test Cases:**

| Input | windowSize | Output | Explanation |
|-------|------------|--------|-------------|
| `[1, 3, 7, 5]` | 4 | 2 | Windows [1,4] or [3,6] each have 2 |
| `[2, 2, 3]` | 1 | 2 | Window [2,2] has both 2s |
| `[1, 2, 3, 4, 5]` | 3 | 3 | Window [1,3] or [2,4] or [3,5] |
| `[1, 1, 1, 1]` | 1 | 4 | All timestamps fit in [1,1] |
| `[1, 100]` | 5 | 1 | Timestamps too far apart |

**Key Insight:**
The optimal window will always have its **left edge aligned with some timestamp**. If no timestamp is at the left edge, we can shift the window left until one is, without losing any timestamps inside.

**Approach: Sort + Sliding Window (Two Pointers)**
1. Sort the timestamps
2. For each timestamp as the potential left edge of a window, count how many timestamps fall within `[timestamp[i], timestamp[i] + windowSize - 1]`
3. Use two pointers to efficiently count

**Python Solution:**
```python
def maxRequests(timestamp: list[int], windowSize: int) -> int:
    """
    Sort + Two Pointers sliding window approach.

    Key insight: Optimal window has left edge at some timestamp.

    Time: O(n log n) for sorting + O(n) for two pointers
    Space: O(1) extra (O(n) if counting sorted array)
    """
    if not timestamp:
        return 0

    timestamp.sort()
    n = len(timestamp)
    max_count = 0
    right = 0

    for left in range(n):
        # Window: [timestamp[left], timestamp[left] + windowSize - 1]
        window_end = timestamp[left] + windowSize - 1

        # Expand right while timestamps fit in window
        while right < n and timestamp[right] <= window_end:
            right += 1

        # Count of timestamps in window = right - left
        max_count = max(max_count, right - left)

    return max_count


def maxRequests_binary_search(timestamp: list[int], windowSize: int) -> int:
    """
    Sort + Binary Search approach.

    For each timestamp as window start, binary search for the rightmost
    timestamp that fits within the window.

    Time: O(n log n)
    Space: O(1) extra
    """
    import bisect

    if not timestamp:
        return 0

    timestamp.sort()
    max_count = 0

    for i, start in enumerate(timestamp):
        window_end = start + windowSize - 1
        # Find rightmost index where timestamp <= window_end
        right_idx = bisect.bisect_right(timestamp, window_end)
        max_count = max(max_count, right_idx - i)

    return max_count


def maxRequests_brute_force(timestamp: list[int], windowSize: int) -> int:
    """
    Brute force: Try every possible window start position.

    Only practical for small inputs due to O(n²) complexity.
    """
    if not timestamp:
        return 0

    max_count = 0

    for start in timestamp:
        window_end = start + windowSize - 1
        count = sum(1 for t in timestamp if start <= t <= window_end)
        max_count = max(max_count, count)

    return max_count


# Test cases
print(maxRequests([1, 3, 7, 5], 4))      # 2
print(maxRequests([2, 2, 3], 1))         # 2
print(maxRequests([1, 2, 3, 4, 5], 3))   # 3
print(maxRequests([1, 1, 1, 1], 1))      # 4
print(maxRequests([1, 100], 5))          # 1
print(maxRequests([], 5))                # 0
print(maxRequests([5], 10))              # 1

# Trace through Example 1: [1, 3, 7, 5] → sorted: [1, 3, 5, 7], windowSize=4
# left=0 (ts=1): window [1,4], right→2 (1,3 fit; 5>4), count=2
# left=1 (ts=3): window [3,6], right→3 (5 fits; 7>6), count=2
# left=2 (ts=5): window [5,8], right→4 (7 fits), count=2
# left=3 (ts=7): window [7,10], right=4, count=1
# Max = 2 ✓
```

**Time Complexity:** O(n log n) - dominated by sorting

**Space Complexity:** O(1) extra space (excluding the sorted array)

**Why Two Pointers Works:**
- After sorting, timestamps are in order
- As we move `left` forward, `right` never needs to go backward
- Each element is visited at most twice (once by left, once by right)
- Total: O(n) for the two-pointer scan

**Variations:**
1. **Sliding window with fixed count**: Find minimum window size containing at least k requests
2. **Rate limiting**: Count requests in rolling windows (real-time streaming)
3. **Multiple windows**: Find top-k window sizes with most requests

---

## 6. Maximum Number of Palindromic Strings (Cross-String Swaps)

**Problem Statement:**
You are given an array `arr` of `n` strings consisting only of lowercase English letters.

In one operation, you may choose two **distinct** strings `arr[x]` and `arr[y]`, pick an index `i` in `arr[x]` and an index `j` in `arr[y]`, and swap the characters `arr[x][i]` and `arr[y][j]`.

- You may perform any number of operations
- You may only swap characters **between different strings** (no swaps within the same string)

Return the maximum number of strings that can be made palindromes after performing any number of operations.

**Function Signature:**
```python
def countPalindromes(arr: List[str]) -> int
```

**Examples:**

```
Example 1:
Input: arr = ["pass", "sas", "asps", "df"]
Output: 3

Explanation:
- Swap 'p' from "pass" with 's' from "asps" → ["sass", "sas", "aspp", "df"]
- Continue swapping to get: ["paap", "sas", "ssss", "df"]
- Palindromes: "paap", "sas", "ssss" → 3 strings

Example 2:
Input: arr = ["xy", "tz", "abab"]
Output: 2

Explanation:
- Through swaps, we can make "xy" → "aa" and "tz" → "bb"
- "abab" cannot be made a palindrome with remaining characters
- Palindromes: "aa", "bb" → 2 strings
```

**Key Insight:**
Since we can swap characters **freely between any strings**, we essentially have a **global pool of characters** that we can redistribute. The only constraint is that **each string has a fixed length**.

**Palindrome Requirements:**
- **Even-length string (L):** Need exactly `L/2` pairs of characters
- **Odd-length string (L):** Need `(L-1)/2` pairs + 1 middle character

**Resource Analysis:**
From the global character pool:
- **P** = total pairs = `Σ(count[c] // 2)` for each character `c`
- **O** = odd-count characters = count of characters with odd frequency

For a selection of strings with total length **S** and **M** odd-length strings:
- Pairs needed = `(S - M) / 2`
- Middles needed = `M`

**Constraint:**
```
If M ≤ O:  cost = pairs_needed
If M > O:  cost = pairs_needed + ceil((M - O) / 2)
           (need to break pairs to create extra middles)

Feasible if: cost ≤ P
```

**Algorithm: Greedy by Length**
1. Compute P (total pairs) and O (odd-count characters) from all strings
2. Sort strings by length (ascending) - shorter strings use fewer pairs
3. Greedily add strings, checking feasibility after each addition
4. Count maximum strings that satisfy the constraint

**Python Solution:**
```python
def countPalindromes(arr: list[str]) -> int:
    """
    Maximum palindromes via cross-string character swaps.

    Key insight: We can redistribute characters freely between strings.
    Each string has fixed length, so we need enough pairs and middles.

    Time: O(n log n) for sorting + O(n) for greedy
    Space: O(26) for character counts
    """
    from collections import Counter

    # Count all characters across all strings
    total_count = Counter()
    for s in arr:
        total_count.update(s)

    # P = total pairs available
    # O = count of characters with odd frequency (free middle chars)
    P = sum(count // 2 for count in total_count.values())
    O = sum(1 for count in total_count.values() if count % 2 == 1)

    # Sort by length - shorter strings are cheaper (fewer pairs needed)
    lengths = sorted(len(s) for s in arr)

    result = 0
    S = 0  # sum of lengths of selected strings
    M = 0  # count of odd-length strings selected

    for L in lengths:
        # Try adding this string
        new_S = S + L
        new_M = M + (L % 2)

        # Calculate total cost (pairs needed)
        pairs_needed = (new_S - new_M) // 2

        if new_M <= O:
            # All middles come from odd-count characters (free)
            cost = pairs_needed
        else:
            # Need to break pairs for extra middles
            extra_middles = new_M - O
            pairs_for_middles = (extra_middles + 1) // 2
            cost = pairs_needed + pairs_for_middles

        if cost <= P:
            S, M = new_S, new_M
            result += 1

    return result


def countPalindromes_detailed(arr: list[str]) -> int:
    """
    Same algorithm with detailed tracing for understanding.
    """
    from collections import Counter

    # Build global character pool
    total_count = Counter()
    for s in arr:
        total_count.update(s)

    print(f"Character counts: {dict(total_count)}")

    # Calculate resources
    P = sum(count // 2 for count in total_count.values())
    O = sum(1 for count in total_count.values() if count % 2 == 1)

    print(f"Total pairs (P): {P}")
    print(f"Odd-count chars (O): {O}")

    # Greedy selection
    lengths = sorted(len(s) for s in arr)
    print(f"Sorted lengths: {lengths}")

    result = 0
    S = M = 0

    for L in lengths:
        new_S, new_M = S + L, M + (L % 2)
        pairs_needed = (new_S - new_M) // 2

        if new_M <= O:
            cost = pairs_needed
        else:
            cost = pairs_needed + (new_M - O + 1) // 2

        print(f"  L={L}: S={new_S}, M={new_M}, cost={cost}, P={P}", end="")

        if cost <= P:
            S, M = new_S, new_M
            result += 1
            print(" ✓")
        else:
            print(" ✗")

    return result


# Test cases
print(countPalindromes(["pass", "sas", "asps", "df"]))  # 3
print(countPalindromes(["xy", "tz", "abab"]))           # 2
print(countPalindromes(["a", "a", "aa"]))               # 3 (all can be palindromes)
print(countPalindromes(["abc", "def"]))                 # 0 (no pairs, can't make palindromes)
print(countPalindromes(["aa", "bb", "cc"]))             # 3 (all already palindromes)
print(countPalindromes(["ab"]))                         # 0 (need pair, have none)

# Trace Example 1: ["pass", "sas", "asps", "df"]
# Characters: p=2, a=3, s=6, d=1, f=1
# P = 1+1+3+0+0 = 5 pairs
# O = 3 (a, d, f have odd counts)
# Sorted lengths: [2, 3, 4, 4]
#
# L=2: S=2, M=0, cost=1 <= 5 ✓
# L=3: S=5, M=1, cost=2 <= 5 ✓
# L=4: S=9, M=1, cost=4 <= 5 ✓
# L=4: S=13, M=1, cost=6 > 5 ✗
# Answer: 3 ✓

# Trace Example 2: ["xy", "tz", "abab"]
# Characters: x=1, y=1, t=1, z=1, a=2, b=2
# P = 0+0+0+0+1+1 = 2 pairs
# O = 4 (x, y, t, z have odd counts)
# Sorted lengths: [2, 2, 4]
#
# L=2: S=2, M=0, cost=1 <= 2 ✓
# L=2: S=4, M=0, cost=2 <= 2 ✓
# L=4: S=8, M=0, cost=4 > 2 ✗
# Answer: 2 ✓
```

**Time Complexity:** O(n log n) - sorting dominates

**Space Complexity:** O(26) = O(1) for character frequency counts

**Why Greedy by Length Works:**
1. Shorter strings always use fewer or equal pairs than longer strings
2. The middle character constraint rarely bottlenecks (breaking 1 pair gives 2 middles)
3. There's no benefit to skipping a short string for a longer one

**Key Formula Recap:**
```
Total characters T = 2P + O

For selected strings (length sum S, odd-count M):
  pairs_needed = (S - M) / 2

  if M <= O: cost = pairs_needed
  if M > O:  cost = pairs_needed + ⌈(M - O) / 2⌉

Feasible if cost <= P
```

---

## 7. Find All Anagrams in a String

**LeetCode 438**

**Problem Statement:**
Given two strings `s` and `p`, find all starting indices in `s` where a substring is an anagram of `p`.

An **anagram** is a rearrangement of letters using all original letters exactly once.

Return a list of starting indices (order doesn't matter).

**Function Signature:**
```python
def findAnagrams(s: str, p: str) -> List[int]
```

**Examples:**

```
Example 1:
Input: s = "cbaebabacd", p = "abc"
Output: [0, 6]

Explanation:
- Index 0: "cba" is an anagram of "abc" ✓
- Index 6: "bac" is an anagram of "abc" ✓

Example 2:
Input: s = "abab", p = "ab"
Output: [0, 1, 2]

Explanation:
- Index 0: "ab" is an anagram ✓
- Index 1: "ba" is an anagram ✓
- Index 2: "ab" is an anagram ✓
```

**Constraints:**
- 1 ≤ len(s), len(p) ≤ 3 × 10⁴
- Both strings contain only lowercase English letters

**Key Insight:**
Two strings are anagrams if and only if they have the **same character frequency counts**. Use a **sliding window** of size `len(p)` and efficiently update counts as the window slides.

**Approach: Sliding Window with Frequency Matching**
1. Build frequency count for pattern `p`
2. Maintain frequency count for current window in `s`
3. Slide window: add new char, remove old char
4. Compare counts (or track number of matching characters)

**Python Solution:**
```python
def findAnagrams(s: str, p: str) -> list[int]:
    """
    Sliding window with character frequency comparison.

    Time: O(n) where n = len(s)
    Space: O(1) - only 26 lowercase letters
    """
    from collections import Counter

    if len(p) > len(s):
        return []

    p_count = Counter(p)
    window_count = Counter()
    result = []
    p_len = len(p)

    for i, char in enumerate(s):
        # Add new character to window
        window_count[char] += 1

        # Remove character that's no longer in window
        if i >= p_len:
            left_char = s[i - p_len]
            window_count[left_char] -= 1
            if window_count[left_char] == 0:
                del window_count[left_char]

        # Check if current window is an anagram
        if window_count == p_count:
            result.append(i - p_len + 1)

    return result


def findAnagrams_optimized(s: str, p: str) -> list[int]:
    """
    Optimized: Track number of matching characters instead of
    comparing full dictionaries each time.

    Time: O(n)
    Space: O(1)
    """
    if len(p) > len(s):
        return []

    p_len = len(p)
    result = []

    # Frequency arrays for 26 letters
    p_count = [0] * 26
    w_count = [0] * 26

    for c in p:
        p_count[ord(c) - ord('a')] += 1

    # Track how many character types have matching counts
    matches = 0
    for i in range(26):
        if p_count[i] == 0:
            matches += 1  # Both are 0, they match

    for i, char in enumerate(s):
        idx = ord(char) - ord('a')

        # Add new character
        if w_count[idx] == p_count[idx]:
            matches -= 1  # Was matching, now won't be
        w_count[idx] += 1
        if w_count[idx] == p_count[idx]:
            matches += 1  # Now matches

        # Remove old character (if window is full)
        if i >= p_len:
            left_idx = ord(s[i - p_len]) - ord('a')
            if w_count[left_idx] == p_count[left_idx]:
                matches -= 1
            w_count[left_idx] -= 1
            if w_count[left_idx] == p_count[left_idx]:
                matches += 1

        # All 26 characters match means it's an anagram
        if matches == 26:
            result.append(i - p_len + 1)

    return result


def findAnagrams_simple(s: str, p: str) -> list[int]:
    """
    Simple approach: Sort and compare (less efficient but intuitive).

    Time: O(n * k log k) where k = len(p)
    Space: O(k)
    """
    if len(p) > len(s):
        return []

    p_sorted = sorted(p)
    p_len = len(p)
    result = []

    for i in range(len(s) - p_len + 1):
        if sorted(s[i:i + p_len]) == p_sorted:
            result.append(i)

    return result


# Test cases
print(findAnagrams("cbaebabacd", "abc"))  # [0, 6]
print(findAnagrams("abab", "ab"))          # [0, 1, 2]
print(findAnagrams("aa", "bb"))            # []
print(findAnagrams("a", "a"))              # [0]
print(findAnagrams("ab", "abc"))           # [] (p longer than s)

# Trace through "cbaebabacd" with p="abc":
# p_count = {a:1, b:1, c:1}
#
# i=0 'c': window="c", count={c:1}
# i=1 'b': window="cb", count={c:1,b:1}
# i=2 'a': window="cba", count={c:1,b:1,a:1} == p_count ✓ → result=[0]
# i=3 'e': window="bae", remove 'c', add 'e', count={b:1,a:1,e:1} ✗
# i=4 'b': window="aeb", count={a:1,e:1,b:1} ✗
# i=5 'a': window="eba", count={e:1,b:1,a:1} ✗
# i=6 'b': window="bab", count={b:2,a:1} ✗
# i=7 'a': window="aba", count={a:2,b:1} ✗
# i=8 'c': window="bac", count={b:1,a:1,c:1} == p_count ✓ → result=[0,6]
# i=9 'd': window="acd", count={a:1,c:1,d:1} ✗
#
# Final: [0, 6] ✓
```

**Time Complexity:** O(n) where n = len(s)
- Each character is processed exactly once (added and removed from window)

**Space Complexity:** O(1)
- Only 26 possible characters, so frequency arrays are constant size

**Why the Optimized Version is O(n):**
- Instead of comparing two dictionaries (O(26) each time)
- Track `matches` = count of character types with equal frequencies
- When `matches == 26`, all characters match → anagram found
- Each add/remove operation updates `matches` in O(1)

**Related Problems:**
1. **Permutation in String** (LeetCode 567) - Check if any permutation of s1 is substring of s2
2. **Minimum Window Substring** (LeetCode 76) - Find smallest window containing all characters
3. **Longest Substring Without Repeating Characters** (LeetCode 3)

---

## 8. Grep With Context Lines

**Problem Statement:**
Implement a grep-like function that searches for lines containing a target string and returns those lines along with surrounding context lines.

**Requirements:**
1. Search for lines containing a specific `target` string
2. Include `lines_around` lines before and after each match
3. Do not print the same line twice (handle overlapping contexts)
4. Maintain original line order

**Function Signature:**
```python
def grep_with_context(lines: list[str], search_target: str, lines_around: int) -> list[str]
```

**Example:**
```
Input:
lines = [
    "good morning",
    "hello there",
    "my name is Alex",
    "my friend is albert",
    "it is nice to meet you Alex",
]
search_target = "Alex"
lines_around = 1

Output:
[
    "hello there",          # 1 line before "my name is Alex"
    "my name is Alex",      # match
    "my friend is albert",  # 1 line after (also 1 line before next match)
    "it is nice to meet you Alex",  # match
]

Note: "my friend is albert" appears only once despite being in context of both matches.
```

**Interview Follow-ups:**
1. Solve for static list input
2. Solve for streaming input (lines arrive one by one)
3. Optimize for large `lines_around` values
4. Design a multithreaded solution

---

### Part 1: Static Input Solution

**Approach:** Use a boolean array to mark lines to include.

```python
def grep_with_context(
    lines: list[str],
    search_target: str,
    lines_around: int
) -> list[str]:
    """
    Basic solution using boolean marking.

    Time: O(n * k) where n = lines, k = lines_around
    Space: O(n) for the marked array
    """
    if lines_around < 0:
        raise ValueError("lines_around must be >= 0")

    n = len(lines)
    marked = [False] * n
    k = lines_around

    # Mark matching lines and their context
    for i, line in enumerate(lines):
        if search_target in line:
            left = max(0, i - k)
            right = min(n - 1, i + k)
            for j in range(left, right + 1):
                marked[j] = True

    # Collect marked lines in order
    return [line for i, line in enumerate(lines) if marked[i]]
```

---

### Part 2: Streaming Input Solution

**Problem:** Lines arrive one at a time. Cannot see the whole list upfront.

**Approach:**
- Use a deque buffer to store recent `k` lines (for "before" context)
- Track `emit_until` index to handle "after" context
- Mark lines as printed to avoid duplicates

```python
from collections import deque

class StreamingGrep:
    """
    Process lines one at a time with O(k) memory.

    Time per line: O(k) worst case
    Space: O(k) for the buffer
    """

    def __init__(self, search_target: str, lines_around: int):
        if lines_around < 0:
            raise ValueError("lines_around must be >= 0")
        self.search_target = search_target
        self.k = lines_around
        self.idx = -1
        self.emit_until = -1  # Print lines up to this index
        # Buffer entries: [index, line, is_printed]
        self.buffer = deque()

    def process_line(self, line: str) -> list[str]:
        """Process one line and return any lines to output."""
        self.idx += 1
        output = []

        # Add new line to buffer
        self.buffer.append([self.idx, line, False])

        # Remove lines too old for "before" context
        min_keep_idx = self.idx - self.k
        while self.buffer and self.buffer[0][0] < min_keep_idx:
            self.buffer.popleft()

        # Check if current line matches
        if self.search_target in line:
            # Update "after" context boundary
            self.emit_until = max(self.emit_until, self.idx + self.k)

            # Emit all buffered lines (before context + match)
            for entry in self.buffer:
                if not entry[2]:  # Not printed yet
                    output.append(entry[1])
                    entry[2] = True

        # Emit lines within "after" context range
        for entry in self.buffer:
            if entry[0] <= self.emit_until and not entry[2]:
                output.append(entry[1])
                entry[2] = True

        return output


# Usage example
def grep_streaming(lines: list[str], target: str, k: int) -> list[str]:
    grep = StreamingGrep(target, k)
    result = []
    for line in lines:
        result.extend(grep.process_line(line))
    return result
```

---

### Part 3: Optimized Solution (Large k)

**Problem:** When `k` is very large, marking each line individually is slow O(n * k).

**Approach:** Use interval merging instead of individual marking.

```python
def grep_with_context_optimized(
    lines: list[str],
    search_target: str,
    lines_around: int
) -> list[str]:
    """
    Optimized solution using interval merging.

    Time: O(n) for scanning + O(r) for output
    Space: O(t) for intervals, where t = number of matches
    """
    if lines_around < 0:
        raise ValueError("lines_around must be >= 0")

    n = len(lines)
    k = lines_around
    intervals = []  # List of [start, end] ranges

    # Find matches and create intervals
    for i, line in enumerate(lines):
        if search_target in line:
            left = max(0, i - k)
            right = min(n - 1, i + k)

            # Merge with previous interval if overlapping
            if intervals and left <= intervals[-1][1] + 1:
                intervals[-1][1] = max(intervals[-1][1], right)
            else:
                intervals.append([left, right])

    # Collect lines from merged intervals
    result = []
    for left, right in intervals:
        for i in range(left, right + 1):
            result.append(lines[i])

    return result
```

**Complexity Comparison:**

| Solution | Time | Space |
|----------|------|-------|
| Basic (Part 1) | O(n × k) | O(n) |
| Streaming (Part 2) | O(k) per line | O(k) |
| Optimized (Part 3) | O(n + r) | O(t) |

Where: n = lines, k = lines_around, r = result size, t = match count

---

### Part 4: Multithreaded Solution

**Problem:** Process a massive file using multiple CPU threads.

**Design: Map-Reduce Approach**

```
┌─────────────────────────────────────────────────────────────┐
│                        INPUT FILE                           │
│  [line0, line1, line2, ..., lineN-1]                       │
└─────────────────────────────────────────────────────────────┘
                            │
              ┌─────────────┼─────────────┐
              ▼             ▼             ▼
        ┌──────────┐  ┌──────────┐  ┌──────────┐
        │ Worker 1 │  │ Worker 2 │  │ Worker 3 │
        │ [0..99]  │  │[100..199]│  │[200..299]│
        └────┬─────┘  └────┬─────┘  └────┬─────┘
             │             │             │
             ▼             ▼             ▼
        intervals     intervals     intervals
              └─────────────┼─────────────┘
                            ▼
                    ┌──────────────┐
                    │ COORDINATOR  │
                    │ Merge & Sort │
                    └──────────────┘
                            │
                            ▼
                    Final Output Lines
```

**Implementation:**

```python
from concurrent.futures import ThreadPoolExecutor
from typing import List, Tuple

def worker(
    lines: list[str],
    search_target: str,
    k: int,
    start: int,
    end: int
) -> list[tuple[int, int]]:
    """
    Worker scans a chunk and returns intervals (using global indices).
    """
    n = len(lines)
    local_intervals = []

    for i in range(start, end + 1):
        if search_target in lines[i]:
            left = max(0, i - k)
            right = min(n - 1, i + k)
            local_intervals.append((left, right))

    return local_intervals


def merge_intervals(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Merge overlapping intervals."""
    if not intervals:
        return []

    sorted_intervals = sorted(intervals)
    merged = [list(sorted_intervals[0])]

    for left, right in sorted_intervals[1:]:
        if left <= merged[-1][1] + 1:
            merged[-1][1] = max(merged[-1][1], right)
        else:
            merged.append([left, right])

    return [tuple(iv) for iv in merged]


def grep_multithreaded(
    lines: list[str],
    search_target: str,
    lines_around: int,
    num_workers: int = 4
) -> list[str]:
    """
    Multithreaded grep using map-reduce pattern.

    Time: O(n/p) per worker + O(m log m) for merge
    Space: O(m) for intervals
    """
    n = len(lines)
    if n == 0:
        return []

    chunk_size = (n + num_workers - 1) // num_workers
    all_intervals = []

    # MAP: Parallel scanning
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        for i in range(num_workers):
            start = i * chunk_size
            end = min(start + chunk_size - 1, n - 1)
            if start <= end:
                futures.append(
                    executor.submit(
                        worker, lines, search_target, lines_around, start, end
                    )
                )

        # Collect results
        for future in futures:
            all_intervals.extend(future.result())

    # REDUCE: Merge intervals
    merged = merge_intervals(all_intervals)

    # Collect final output
    result = []
    for left, right in merged:
        for i in range(left, right + 1):
            result.append(lines[i])

    return result
```

---

### Edge Cases

```python
def test_grep_with_context():
    lines = ["a", "b Alex", "c", "Alex d", "e"]

    # Normal case with overlap
    assert grep_with_context(lines, "Alex", 1) == \
        ["a", "b Alex", "c", "Alex d", "e"]

    # Zero context
    assert grep_with_context(lines, "Alex", 0) == ["b Alex", "Alex d"]

    # No match
    assert grep_with_context(lines, "zzz", 3) == []

    # Empty input
    assert grep_with_context([], "Alex", 2) == []

    # Context larger than document
    assert grep_with_context(["only line"], "only", 100) == ["only line"]

    # Match at boundaries
    boundary = ["Alex start", "middle", "end Alex"]
    assert grep_with_context(boundary, "Alex", 1) == \
        ["Alex start", "middle", "end Alex"]

    # Test streaming
    sg = StreamingGrep("Alex", 1)
    out = []
    for line in lines:
        out.extend(sg.process_line(line))
    assert out == ["a", "b Alex", "c", "Alex d", "e"]

    # Test optimized
    assert grep_with_context_optimized(lines, "Alex", 1) == \
        ["a", "b Alex", "c", "Alex d", "e"]

    print("All tests passed!")

test_grep_with_context()
```

---

### Summary

| Part | Use Case | Key Technique | Time | Space |
|------|----------|---------------|------|-------|
| 1 | Static list | Boolean marking | O(nk) | O(n) |
| 2 | Streaming | Deque buffer | O(k)/line | O(k) |
| 3 | Large k | Interval merging | O(n) | O(t) |
| 4 | Massive file | Map-reduce threads | O(n/p) | O(m) |

---

*Total: 8 questions*
*Last updated: 2026-04-16*
