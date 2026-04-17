# Netflix Interview Questions

Source: https://www.1point3acres.com/interview/problems/company/netflix

> **Note:** Full problem details require login to 1point3acres.com. This document contains publicly available summaries and structure.

---

## Table of Contents

### System Design (8)
1. [Netflix Sentiment Tracking](#2-netflix-sentiment-tracking)
2. [Design the Data Model for an Ads Demand Platform](#3-design-the-data-model-for-an-ads-demand-platform)
3. [Home Page Video Recommendation](#5-home-page-video-recommendation)
4. [Design an Ads Audience Targeting System](#11-design-an-ads-audience-targeting-system)
5. [Design a Billing System for 300M Subscribers](#12-design-a-billing-system-for-300m-subscribers)
6. [Design an Ads Frequency Cap System](#18-design-an-ads-frequency-cap-system)
7. [ML Job Scheduler](#19-ml-job-scheduler)
8. [Design a WAL Log Enrichment Pipeline](#21-design-a-wal-log-enrichment-pipeline)
9. [Design the Data Model for a Promotion Posting System](#26-design-the-data-model-for-a-promotion-posting-system)

### Coding - Arrays & Strings (12)
10. [Contains Duplicate](#25-contains-duplicate)
11. [Contains Duplicate II](#6-contains-duplicate-ii)
12. [Contains Duplicate III](#14-contains-duplicate-iii)
13. [First Missing Positive](#30-first-missing-positive)
14. [Longest Palindromic Substring](#37-longest-palindromic-substring)
15. [Longest Substring Without Repeating Characters](#15-longest-substring-without-repeating-characters)
16. [Maximum Subarray](#35-maximum-subarray)
17. [Median of Two Sorted Arrays](#34-median-of-two-sorted-arrays)
18. [Meeting Rooms](#17-meeting-rooms)
19. [Number Pairs That Match Target](#29-number-pairs-that-match-target)
20. [Sort by User Preference](#13-sort-by-user-preference)
21. [User Engagement Patterns](#27-user-engagement-patterns)

### Coding - Graph & Search (6)
22. [Number of Islands](#36-number-of-islands)
23. [Parallel Courses](#4-parallel-courses)
24. [Parallel Courses II](#31-parallel-courses-ii)
25. [Reconstruct Itinerary](#23-reconstruct-itinerary)
26. [Movie History Friends](#7-movie-history-friends)
27. [Movie History Friends II](#22-movie-history-friends-ii)

### Coding - Data Structures & Cache (6)
28. [LRU Cache](#32-lru-cache)
29. [Merge K Sorted Lists](#33-merge-k-sorted-lists)
30. [Versioned File System](#8-versioned-file-system)
31. [Auto-Expire Cache](#24-auto-expire-cache)
32. [Music Playlist](#9-music-playlist)
33. [Homepage Title Deduplication](#1-homepage-title-deduplication)

### Coding - Concurrency (2)
34. [Countdown Latch](#20-countdown-latch)
35. [Timer Function](#10-timer-function)

### Coding - ML & Analytics (2)
36. [Error Rate Monitor](#16-error-rate-monitor)
37. [Spam Email Detection](#28-spam-email-detection)

---

## 1. Homepage Title Deduplication

**Type:** Coding Problem

**Problem:** The Netflix homepage organizing structure involves a vertical list of shelves (rows), where each shelf contains a horizontal list of titles (movies or shows). Remove duplicate titles that appear across multiple shelves.

**Sections:**
- Netflix Homepage: Removing Duplicate Titles

---

## 2. Netflix Sentiment Tracking

**Type:** System Design

**Problem:** Design a system that tracks overall public sentiment toward Netflix on social media over time.

**Sections:**
- Netflix Sentiment Tracker
- The Challenge
- Interview Story 1

---

## 3. Design the Data Model for an Ads Demand Platform

**Type:** System Design (Data Modeling)

**Problem:** Create a database schema for Netflix's advertising demand platform.

**Phases:**
- Phase 1: Problem Requirements
- Phase 2: Database Structure
- Phase 3: Technical Discussion Topics
- Interview Success Checklist
- Important Takeaways

---

## 4. Parallel Courses

**Type:** Coding Problem (Graph)

**Problem:** You are given an integer n, which indicates that there are n courses labeled from 1 to n. Determine course scheduling with prerequisites.

**Focus:** Topological sort, cycle detection, BFS level-order traversal

### The Challenge

You are given an integer `n`. This number tells you there are `n` courses you need to take, labeled from 1 to n.

You are also given a list called `relations`. Inside this list, each item is a pair of numbers: `[prevCourse, nextCourse]`. This represents a rule: you must finish `prevCourse` before you can start `nextCourse`.

**How Semesters Work:** In a single semester, you can take any number of courses. However, you can only take a course if you have already finished all of its prerequisites in a previous semester.

**Your Goal:** Return the minimum number of semesters needed to finish every course. If there is no way to finish all the courses (for example, if the prerequisites form a loop), return -1.

### Examples

**Example 1:**
```
Input: n = 3, relations = [[1,3], [2,3]]
Output: 2
Explanation:
- Semester 1: Take courses 1 and 2 (no prerequisites)
- Semester 2: Take course 3 (1 and 2 are now finished)
Total: 2 semesters
```

**Example 2:**
```
Input: n = 3, relations = [[1,2], [2,3], [3,1]]
Output: -1
Explanation:
Course 1 needs 2. Course 2 needs 3. Course 3 needs 1.
This forms a cycle, so no course can be started.
```

### Constraints

- `1 <= n <= 5000`
- `1 <= relations.length <= 5000`
- `relations[i].length == 2`
- `1 <= prevCourse, nextCourse <= n`
- `prevCourse != nextCourse` (no self-loops)
- All pairs are unique

### Key Insight

This is a **topological sort** problem where we need to find the longest path in the DAG. The minimum number of semesters equals the number of "levels" when processing courses in topological order using BFS (Kahn's algorithm).

### Python Solution

```python
from collections import deque, defaultdict

def minimumSemesters(n: int, relations: list[list[int]]) -> int:
    # Build adjacency list and in-degree count
    graph = defaultdict(list)
    in_degree = [0] * (n + 1)

    for prev_course, next_course in relations:
        graph[prev_course].append(next_course)
        in_degree[next_course] += 1

    # Start with courses that have no prerequisites
    queue = deque()
    for course in range(1, n + 1):
        if in_degree[course] == 0:
            queue.append(course)

    semesters = 0
    courses_taken = 0

    # BFS level by level (each level = one semester)
    while queue:
        semesters += 1
        # Process all courses available this semester
        for _ in range(len(queue)):
            course = queue.popleft()
            courses_taken += 1

            # Unlock dependent courses
            for next_course in graph[course]:
                in_degree[next_course] -= 1
                if in_degree[next_course] == 0:
                    queue.append(next_course)

    # Check if all courses were taken (cycle detection)
    return semesters if courses_taken == n else -1
```

### TypeScript Solution

```typescript
function minimumSemesters(n: number, relations: number[][]): number {
    const graph: Map<number, number[]> = new Map();
    const inDegree: number[] = new Array(n + 1).fill(0);

    // Build graph
    for (const [prev, next] of relations) {
        if (!graph.has(prev)) graph.set(prev, []);
        graph.get(prev)!.push(next);
        inDegree[next]++;
    }

    // Initialize queue with courses having no prerequisites
    const queue: number[] = [];
    for (let course = 1; course <= n; course++) {
        if (inDegree[course] === 0) {
            queue.push(course);
        }
    }

    let semesters = 0;
    let coursesTaken = 0;

    while (queue.length > 0) {
        semesters++;
        const levelSize = queue.length;

        for (let i = 0; i < levelSize; i++) {
            const course = queue.shift()!;
            coursesTaken++;

            for (const nextCourse of graph.get(course) || []) {
                inDegree[nextCourse]--;
                if (inDegree[nextCourse] === 0) {
                    queue.push(nextCourse);
                }
            }
        }
    }

    return coursesTaken === n ? semesters : -1;
}
```

### Walkthrough

For `n = 3, relations = [[1,3], [2,3]]`:

| Step | Queue | Semester | Action |
|------|-------|----------|--------|
| Init | `[1, 2]` | 0 | Courses 1, 2 have in-degree 0 |
| Process | `[]` | 1 | Take 1, 2. Decrement in-degree of 3 twice |
| Process | `[3]` | 1 | Course 3 now has in-degree 0 |
| Process | `[]` | 2 | Take course 3 |

**Result:** 2 semesters

### Interview Questions

1. **Why use BFS instead of DFS?**
   - BFS processes nodes level by level, naturally giving us the minimum number of semesters. DFS would require additional tracking to find the longest path.

2. **How do you detect a cycle?**
   - After BFS completes, if `courses_taken < n`, some courses were never added to the queue, meaning they're stuck in a cycle.

3. **What if we want to limit courses per semester?**
   - This becomes a harder problem (NP-hard). You'd need to prioritize courses that unlock the most dependencies, potentially using a priority queue.

4. **Time complexity?**
   - O(V + E) where V = n courses and E = number of relations. Each course and relation is processed exactly once.

### Complexity Analysis

| Metric | Value |
|--------|-------|
| Time | O(n + relations.length) |
| Space | O(n + relations.length) |

**Note:** LeetCode problem #1136

---

## 5. Home Page Video Recommendation

**Type:** System Design

**Problem:** Design Netflix's home page video recommendation system.

**Sections:**
- Netflix Home Page Video Recommendations
- The Challenge
- Helpful Learning Materials
- Candidate Stories

---

## 6. Contains Duplicate II

**Type:** Coding Problem (Array/HashMap)

**Problem:** Given an integer array `nums` and an integer `k`, return true if there are two distinct indices `i` and `j` in the array such that `nums[i] == nums[j]` and `abs(i - j) <= k`.

**Sections:**
- Problem Breakdown
- Example Scenarios
- Input Constraints
- Solution Approach
- Code Implementation
- Complexity Analysis

---

## 7. Movie History Friends

**Type:** Coding Problem

**Problem:** Analyze customer movie viewing patterns. Each customer maintains a list representing the movies they've watched in the order they were viewed.

**Sections:**
- Problem Challenge: Movie History Friends
- Sample Cases
- Input Limits

---

## 8. Versioned File System

**Type:** Coding Problem (Data Structure Design)

**Problem:** Design and implement an in-memory file management system with versioning support.

**Sections:**
- What We Are Building
- Step 1: Simple File System
- Step 2: Adding Versions
- Step 3: Handling Crashes
- Real-World Problems
- Full Code
- Interview Questions
- Big-O Summary

---

## 9. Music Playlist

**Type:** Coding Problem (Data Structure)

**Problem:** Design a data structure that tracks music listening history with timestamps.

**Sections:**
- Problem Requirements
- Example Walkthrough
- Input Limits

---

## 10. Timer Function

**Type:** Coding Problem

**Problem:** Create a function named `timer` accepting a single parameter—seconds (non-negative integer)—that outputs a human-readable time string representation.

**Sections:**
- Building a Recursive Timer
- Problem Statement
- Rules to Follow
- Conversion Values
- Sample Inputs and Outputs
- Input Limits

---

## 11. Design an Ads Audience Targeting System

**Type:** System Design

**Problem:** Design a system that enables advertisers to upload custom audience lists, package them into segments, and leverage them for precise ad targeting purposes.

**Phases:**
- Phase 1: Project Requirements
- Phase 2: Data Model
- Phase 3: API Design
- Phase 4: High-Level Design
- Phase 5: Scaling & Trade-offs
- Review Checklist
- Important Takeaways

---

## 12. Design a Billing System for 300M Subscribers

**Type:** System Design

**Problem:** Architect a billing infrastructure for Netflix capable of processing monthly subscription charges across 300 million users.

**Phases:**
- Phase 1: What We Need to Build
- Phase 2: Database Design
- Phase 3: How Systems Talk
- Phase 4: System Architecture
- Phase 5: Handling Growth and Problems
- Interview Checklist
- Main Takeaways

**Scale:** 300 million subscribers, monthly subscription-based billing

---

## 13. Sort by User Preference

**Type:** Coding Problem

**Problem:** Design a solution that organizes shows from a user interface and returns them as title IDs based on user preferences.

**Sections:**
- Sorting Shows Based on User Preferences

---

## 14. Contains Duplicate III

**Type:** Coding Problem (Array/Sliding Window)

**Problem:** You are given an integer array `nums` and two integers `indexDiff` and `valueDiff`. Find duplicates within index and value constraints.

**Sections:**
- Problem Breakdown
- Illustrative Examples
- Technical Constraints

---

## 15. Longest Substring Without Repeating Characters

**Type:** Coding Problem (Sliding Window)

**Problem:** Find the length of the longest substring without duplicate characters from a given string input.

**Sections:**
- Longest Substring Without Repeating Characters

**Note:** Classic LeetCode problem #3

---

## 16. Error Rate Monitor

**Type:** Coding Problem (Monitoring)

**Problem:** Monitor a system's error rates over time.

**Sections:**
- System Health Monitor

---

## 17. Meeting Rooms

**Type:** Coding Problem (Intervals)

**Problem:** Given an array of meeting time interval objects consisting of start and end times `[[start1,end1],[start2,end2],...]` (start_i < end_i), determine if a person could add all meetings to their schedule without any conflicts.

**Sections:**
- Problem Requirements
- Sample Cases
- Input Limits

**Note:** Classic interval scheduling problem

---

## 18. Design an Ads Frequency Cap System

**Type:** System Design

**Problem:** Design a system that restricts ad exposure by limiting how many times a specific ad (or category of ads) is shown to a user within a time window.

**Sections:**
- System Overview
- Scope of the Design
- Why Interviewers Ask This

---

## 19. ML Job Scheduler

**Type:** System Design

**Problem:** Design a distributed job scheduler for ML workloads at Netflix.

**Sections:**
- System Design: ML Job Scheduler
- The Task
- Helpful Study Materials

---

## 20. Countdown Latch

**Type:** Coding Problem (Concurrency)

**Problem:** Design and implement a thread-safe counter that allows multiple threads to wait until the counter reaches zero.

**Focus:** Thread synchronization, blocking mechanisms, counter management

### The Challenge

You need to build a thread-safe counter. This counter allows several threads to pause and wait until the count drops to zero.

You must create a CountdownLatch class with these methods:

```python
class CountdownLatch:
    def __init__(self, count: int):
        """Start the latch with a specific number (must be >= 0)."""
        pass

    def count_up(self) -> None:
        """Add 1 to the count."""
        pass

    def count_down(self) -> None:
        """
        Subtract 1 from the count.
        If the count hits zero, wake up every thread that is waiting.
        """
        pass

    def wait(self) -> None:
        """
        Stop the current thread here until the count is zero.
        If the count is already zero, do not stop.
        """
        pass
```

### Common Use Cases

- **Parallel Downloads:** Wait for multiple files to finish downloading before combining them.
- **Starting Services:** Wait for all small services to be ready before letting users connect.
- **Batch Jobs:** Wait for a group of items to finish processing.

### Code Example

Here is how you use the latch. We create a latch with a count of 3. Then, we start 3 worker threads. The main thread waits until all workers represent they are done.

```python
import threading
import time

def worker(latch: CountdownLatch, worker_id: int):
    """Act like a worker doing a job."""
    print(f"Worker {worker_id} starting...")
    time.sleep(0.5)  # Pretend to work
    print(f"Worker {worker_id} done!")
    latch.count_down()

# Create latch with count = 3 (we expect 3 workers)
latch = CountdownLatch(3)

# Start 3 worker threads
for i in range(3):
    t = threading.Thread(target=worker, args=(latch, i))
    t.start()

# Main thread pauses here until workers finish
print("Main thread waiting for workers...")
latch.wait()
print("All workers completed!")
```

**Output:**
```
Main thread waiting for workers...
Worker 0 starting...
Worker 1 starting...
Worker 2 starting...
Worker 0 done!
Worker 1 done!
Worker 2 done!
All workers completed!
```

### Python Solution

We use a Condition variable to handle the threads safely.

```python
import threading

class CountdownLatch:
    def __init__(self, count: int):
        if count < 0:
            raise ValueError("Count must be non-negative")

        self._count = count
        self._lock = threading.Lock()
        self._condition = threading.Condition(self._lock)

    def count_up(self) -> None:
        """Add 1 to the count."""
        with self._condition:
            self._count += 1

    def count_down(self) -> None:
        """Subtract 1. If zero, tell everyone to wake up."""
        with self._condition:
            if self._count > 0:
                self._count -= 1
                if self._count == 0:
                    self._condition.notify_all()

    def wait(self) -> None:
        """Block here until count is zero."""
        with self._condition:
            while self._count > 0:
                self._condition.wait()

    def get_count(self) -> int:
        """Get the current number (for checking status)."""
        with self._lock:
            return self._count
```

**Key Implementation Details:**

- **threading.Condition:** This acts like a normal lock, but it also lets threads wait for a signal.
- **notify_all():** This wakes up ALL the sleeping threads once the count hits zero.
- **While loop in wait():** This is a safety check. Sometimes threads wake up without a signal (called spurious wakeups). The loop forces the thread to check the count again.
- **Locking:** We use a lock to make sure the count is accurate when multiple threads change it at the same time.

### TypeScript Solution

Here is how you can do this in JavaScript/TypeScript using Promises.

```typescript
class CountdownLatch {
  private count: number;
  private resolvers: Array<() => void> = [];

  constructor(count: number) {
    if (count < 0) throw new Error("Count must be non-negative");
    this.count = count;
  }

  countUp(): void {
    this.count += 1;
  }

  countDown(): void {
    if (this.count > 0) {
      this.count -= 1;
      if (this.count === 0) {
        this.resolvers.forEach(resolve => resolve());
        this.resolvers = [];
      }
    }
  }

  async wait(): Promise<void> {
    if (this.count === 0) return;

    return new Promise<void>((resolve) => {
      this.resolvers.push(resolve);
    });
  }

  getCount(): number {
    return this.count;
  }
}
```

### Interview Questions

1. **Why use Condition instead of Lock?**
   - A Lock only stops two threads from touching data at the exact same time.
   - A Condition does that too, but it also helps threads coordinate by letting them "sleep" and "wake up" based on a signal.

2. **Why do we need a while loop in wait()?**
   - This handles spurious wakeups. Occasionally, the operating system wakes a thread up for no reason. The loop checks the condition again to ensure the thread only proceeds if the count is actually zero.

3. **What is the difference between notify() and notify_all()?**
   - `notify()` wakes up only ONE thread.
   - `notify_all()` wakes up EVERY waiting thread.
   - We use `notify_all()` because when the count hits zero, everyone who is waiting needs to move forward.

4. **How would you add a timeout?**
   - You would change the `wait()` method to accept a time limit. If the time runs out before the count hits zero, the method returns or throws an error.

5. **How would you make this work across different servers?**
   - This requires a distributed system, like Redis or Zookeeper, to share the count state across the network.

### Performance Analysis

| Operation | Time | Space |
| --- | --- | --- |
| count_up | O(1) | O(1) |
| count_down | O(1) | O(1) |
| wait | O(1) | O(1) |

**Space:** O(W), where W is the number of waiting threads.

---

## 21. Design a WAL Log Enrichment Pipeline

**Type:** System Design

**Problem:** Build a system that captures Write-Ahead Log (WAL) entries from a source database, enriches them with supplementary context, and transmits them to a target database.

**Phases:**
- Phase 1: What We Need to Build
- Phase 2: How We Store Data
- Phase 3: Interface Design
- Phase 4: System Architecture
- Phase 5: Handling High Load & Problems
- Review Checklist
- Important Takeaways

---

## 22. Movie History Friends II

**Type:** Coding Problem

**Problem:** Given a map of customer IDs linked to their viewing history, plus two integers `k` and `m` (with `m <= k`), return all pairs of customer IDs that are friends based on viewing patterns.

**Sections:**
- Movie History Friends II (main problem)

---

## 23. Reconstruct Itinerary

**Type:** Coding Problem (Graph)

**Problem:** You are given a list of airline tickets where `tickets[i] = [from_i, to_i]` representing flight departure and arrival airports. Reconstruct the itinerary.

**Sections:**
- Problem Requirements
- Sample Inputs and Outputs
- Technical Limits
- Interview Follow-Up Question

**Note:** Classic graph traversal problem (Hierholzer's algorithm)

---

## 24. Auto-Expire Cache

**Type:** Coding Problem (Data Structure)

**Problem:** Design and implement a key-value cache with automatic expiration.

**Sections:**
- Problem Requirements
- Part 1: Basic Implementation
- Part 2: Fixing Memory Leaks
- Part 3: Limited Size Cache (LRU)
- Full Code Examples
- Extra Interview Topics
- Summary of Approaches

---

## 25. Contains Duplicate

**Type:** Coding Problem (Array/HashSet)

**Problem:** Given an integer array `nums`, return true if any value appears more than once in the array, otherwise return false.

**Sections:**
- Problem Requirements
- Sample Cases
- Solution Strategy
- Code Implementation

**Note:** Classic LeetCode problem #217

---

## 26. Design the Data Model for a Promotion Posting System

**Type:** System Design (Data Modeling)

**Problem:** Create a database schema for Netflix's internal promotion posting system.

**Sections:**
- Phase 1: What We Need to Build
- Phase 2: Database Tables
- Phase 3: Detailed Explanations
- Review Checklist
- Important Takeaways

---

## 27. User Engagement Patterns

**Type:** Coding Problem (Analytics)

**Problem:** Analyze user engagement patterns across different content.

**Sections:**
- User Engagement Analysis
- Data Format
- The Goal
- Example Walkthrough
- Input Constraints

---

## 28. Spam Email Detection

**Type:** Coding Problem (ML/Design)

**Problem:** Given a large corpus of email correspondence history, design and implement a system to detect spam emails.

**Sections:**
- Detecting Spam Emails

---

## 29. Number Pairs That Match Target

**Type:** Coding Problem (Array)

**Problem:** Given an array of integers `nums` and an integer `target`, return all pairs of numbers where one of the four arithmetic operations (+, -, *, /) produces exactly the target value.

**Sections:**
- Problem Statement
- Sample Cases
- Input Limits

---

## 30. First Missing Positive

**Type:** Coding Problem (Array)

**Problem:** Given an unsorted integer array `nums`, return the smallest positive integer that does not appear in `nums`.

Your solution must run in **O(n) time** and use **O(1) extra space**.

### Examples

**Example 1:**
```
Input: nums = [1,2,0]
Output: 3
Explanation: The values 1 and 2 are present, so the smallest missing positive is 3.
```

**Example 2:**
```
Input: nums = [3,4,-1,1]
Output: 2
```

**Example 3:**
```
Input: nums = [7,8,9,11,12]
Output: 1
```

### Constraints

- `1 <= nums.length <= 10^5`
- `-2^31 <= nums[i] <= 2^31 - 1`

### Key Insight

For an array of length `n`, the answer must be in the range `[1, n+1]`. If all numbers 1 to n are present, the answer is `n+1`. Otherwise, it's some number in `[1, n]`.

The trick to achieve O(1) space is to use the array itself as a hash table by placing each number in its "correct" position (number `i` at index `i-1`).

### Python Solution

```python
def firstMissingPositive(nums: list[int]) -> int:
    n = len(nums)

    # Step 1: Place each number in its correct position
    # Number i should be at index i-1
    for i in range(n):
        while 1 <= nums[i] <= n and nums[nums[i] - 1] != nums[i]:
            # Swap nums[i] to its correct position
            correct_idx = nums[i] - 1
            nums[i], nums[correct_idx] = nums[correct_idx], nums[i]

    # Step 2: Find the first position where nums[i] != i + 1
    for i in range(n):
        if nums[i] != i + 1:
            return i + 1

    # All positions are correct, answer is n + 1
    return n + 1
```

### TypeScript Solution

```typescript
function firstMissingPositive(nums: number[]): number {
    const n = nums.length;

    // Step 1: Place each number in its correct position
    for (let i = 0; i < n; i++) {
        while (nums[i] >= 1 && nums[i] <= n && nums[nums[i] - 1] !== nums[i]) {
            const correctIdx = nums[i] - 1;
            [nums[i], nums[correctIdx]] = [nums[correctIdx], nums[i]];
        }
    }

    // Step 2: Find first position where nums[i] != i + 1
    for (let i = 0; i < n; i++) {
        if (nums[i] !== i + 1) {
            return i + 1;
        }
    }

    return n + 1;
}
```

### Walkthrough

For `nums = [3, 4, -1, 1]`:

| Step | Array State | Action |
|------|-------------|--------|
| Initial | `[3, 4, -1, 1]` | Start at index 0 |
| Swap | `[-1, 4, 3, 1]` | 3 goes to index 2 |
| Swap | `[-1, 1, 3, 4]` | 4 goes to index 3 |
| Swap | `[1, -1, 3, 4]` | 1 goes to index 0 |
| Scan | `[1, -1, 3, 4]` | Index 1 has -1 ≠ 2 |

**Result:** 2

### Interview Questions

1. **Why does this approach work in O(n) time despite nested loops?**
   - Each element is swapped at most once to its correct position. After being placed correctly, it never moves again. So total swaps across all iterations is at most `n`.

2. **Why check `nums[nums[i] - 1] != nums[i]` in the while condition?**
   - This prevents infinite loops when duplicates exist. If the target position already has the correct value, we stop swapping.

3. **What if we used a HashSet instead?**
   - HashSet would give O(n) time but O(n) space. The in-place swap technique achieves O(1) space.

4. **Can negative numbers or zeros affect the result?**
   - No. We only care about positive integers in range `[1, n]`. Negatives and zeros are ignored during placement.

### Complexity Analysis

| Metric | Value |
|--------|-------|
| Time | O(n) |
| Space | O(1) |

**Note:** Classic LeetCode problem #41

---

## 31. Parallel Courses II

**Type:** Coding Problem (Graph/Bitmask DP)

**Problem:** Given n courses with prerequisites and a limit of k courses per semester, return the minimum number of semesters needed to complete all courses.

**Focus:** Bitmask dynamic programming, subset enumeration, NP-hard optimization

### The Challenge

You are given an integer `n` (courses labeled 1 to n) and an array `relations` where `relations[i] = [prevCourse, nextCourse]` means you must complete `prevCourse` before `nextCourse`.

In one semester, you can take **at most k courses** as long as all prerequisites are completed.

Return the minimum number of semesters needed to take all courses.

### Examples

**Example 1:**
```
Input: n = 4, relations = [[2,1],[3,1],[1,4]], k = 2
Output: 3
Explanation:
- Semester 1: Take courses 2 and 3
- Semester 2: Take course 1
- Semester 3: Take course 4
```

**Example 2:**
```
Input: n = 5, relations = [[2,1],[3,1],[4,1],[1,5]], k = 2
Output: 4
Explanation:
- Semester 1: Take 2, 3
- Semester 2: Take 4 (still need 1 more prereq for course 1)
- Semester 3: Take 1
- Semester 4: Take 5
```

**Example 3:**
```
Input: n = 11, relations = [], k = 2
Output: 6
Explanation: No prerequisites, just ceil(11/2) = 6 semesters.
```

### Constraints

- `1 <= n <= 15`
- `1 <= k <= n`
- `0 <= relations.length <= n * (n - 1) / 2`
- The graph is a **DAG** (no cycles guaranteed)

### Key Insight

The constraint `n <= 15` strongly hints at **bitmask DP**. Unlike Parallel Courses I, greedy doesn't work here because choosing which k courses to take affects future options.

**State:** A bitmask representing which courses have been completed.
**Transition:** For each state, find all available courses (prerequisites satisfied), try all subsets of size ≤ k.

### Python Solution

```python
def minNumberOfSemesters(n: int, relations: list[list[int]], k: int) -> int:
    # prereq[i] = bitmask of prerequisites for course i
    prereq = [0] * n
    for prev, next_course in relations:
        prereq[next_course - 1] |= (1 << (prev - 1))

    # dp[mask] = minimum semesters to complete courses in mask
    dp = [float('inf')] * (1 << n)
    dp[0] = 0

    for mask in range(1 << n):
        if dp[mask] == float('inf'):
            continue

        # Find all available courses (prerequisites satisfied)
        available = 0
        for i in range(n):
            if not (mask & (1 << i)):  # Course i not taken yet
                if (mask & prereq[i]) == prereq[i]:  # All prereqs done
                    available |= (1 << i)

        # Try all subsets of available courses with size <= k
        subset = available
        while subset > 0:
            if bin(subset).count('1') <= k:
                new_mask = mask | subset
                dp[new_mask] = min(dp[new_mask], dp[mask] + 1)
            # Iterate to next subset
            subset = (subset - 1) & available

    return dp[(1 << n) - 1]
```

### TypeScript Solution

```typescript
function minNumberOfSemesters(n: number, relations: number[][], k: number): number {
    // prereq[i] = bitmask of prerequisites for course i
    const prereq: number[] = new Array(n).fill(0);
    for (const [prev, next] of relations) {
        prereq[next - 1] |= (1 << (prev - 1));
    }

    // dp[mask] = minimum semesters to complete courses in mask
    const dp: number[] = new Array(1 << n).fill(Infinity);
    dp[0] = 0;

    for (let mask = 0; mask < (1 << n); mask++) {
        if (dp[mask] === Infinity) continue;

        // Find available courses
        let available = 0;
        for (let i = 0; i < n; i++) {
            if (!(mask & (1 << i))) {  // Not taken
                if ((mask & prereq[i]) === prereq[i]) {  // Prereqs done
                    available |= (1 << i);
                }
            }
        }

        // Try all subsets of size <= k
        let subset = available;
        while (subset > 0) {
            if (countBits(subset) <= k) {
                const newMask = mask | subset;
                dp[newMask] = Math.min(dp[newMask], dp[mask] + 1);
            }
            subset = (subset - 1) & available;
        }
    }

    return dp[(1 << n) - 1];
}

function countBits(n: number): number {
    let count = 0;
    while (n) {
        count += n & 1;
        n >>= 1;
    }
    return count;
}
```

### Walkthrough

For `n = 4, relations = [[2,1],[3,1],[1,4]], k = 2`:

| State (binary) | Completed | Available | Best Subset | Semesters |
|----------------|-----------|-----------|-------------|-----------|
| `0000` | none | 2, 3 | {2,3} | 0 → 1 |
| `0110` | 2, 3 | 1 | {1} | 1 → 2 |
| `0111` | 1, 2, 3 | 4 | {4} | 2 → 3 |
| `1111` | all | - | - | **3** |

### Interview Questions

1. **Why doesn't greedy work for this problem?**
   - With k limit, taking courses that unlock the most dependencies isn't always optimal. You might need to delay certain courses to parallelize better in future semesters.

2. **How does the subset enumeration trick work?**
   - `subset = (subset - 1) & available` generates all non-empty subsets of `available` in decreasing order. This is a standard bitmask technique.

3. **What's the time complexity?**
   - O(3^n) because for each of 2^n states, we enumerate subsets of available courses. The total number of (mask, subset) pairs is bounded by 3^n.

4. **How would you optimize for larger n?**
   - For n > 15, this approach won't work. You'd need heuristics, branch-and-bound, or approximate algorithms. This is NP-hard in general.

5. **Difference from Parallel Courses I?**
   - Part I has no k limit, so BFS level-order traversal works in O(V+E). Part II requires exponential-time DP due to the k constraint.

### Complexity Analysis

| Metric | Value |
|--------|-------|
| Time | O(3^n) |
| Space | O(2^n) |

**Note:** LeetCode problem #1494

---

## 32. LRU Cache

**Type:** Coding Problem (Data Structure Design)

**Problem:** Design a data structure that follows the constraints of a Least Recently Used (LRU) cache. Implement the `LRUCache` class with `get` and `put` operations in O(1) time complexity.

**Focus:** Hash map + doubly linked list, cache eviction policy

### The Challenge

Implement the `LRUCache` class:

- `LRUCache(int capacity)` - Initialize the cache with positive capacity
- `int get(int key)` - Return the value if key exists, otherwise return -1
- `void put(int key, int value)` - Update or insert the value. If capacity is exceeded, evict the least recently used key.

Both operations must run in **O(1)** average time complexity.

### Examples

```
Input:
["LRUCache", "put", "put", "get", "put", "get", "put", "get", "get", "get"]
[[2], [1, 1], [2, 2], [1], [3, 3], [2], [4, 4], [1], [3], [4]]

Output: [null, null, null, 1, null, -1, null, -1, 3, 4]

Explanation:
LRUCache cache = new LRUCache(2);
cache.put(1, 1);    // cache: {1=1}
cache.put(2, 2);    // cache: {1=1, 2=2}
cache.get(1);       // returns 1, cache: {2=2, 1=1}
cache.put(3, 3);    // evicts key 2, cache: {1=1, 3=3}
cache.get(2);       // returns -1 (not found)
cache.put(4, 4);    // evicts key 1, cache: {3=3, 4=4}
cache.get(1);       // returns -1
cache.get(3);       // returns 3
cache.get(4);       // returns 4
```

### Constraints

- `1 <= capacity <= 3000`
- `0 <= key <= 10^4`
- `0 <= value <= 10^5`
- At most `2 * 10^5` calls to `get` and `put`

### Key Insight

Use a **hash map** for O(1) lookups and a **doubly linked list** for O(1) insertion/deletion. The list maintains access order - most recently used at head, least recently used at tail.

### Python Solution

```python
class Node:
    def __init__(self, key: int = 0, val: int = 0):
        self.key = key
        self.val = val
        self.prev = None
        self.next = None

class LRUCache:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache = {}  # key -> Node

        # Dummy head and tail for easier list manipulation
        self.head = Node()
        self.tail = Node()
        self.head.next = self.tail
        self.tail.prev = self.head

    def _remove(self, node: Node) -> None:
        """Remove node from linked list."""
        node.prev.next = node.next
        node.next.prev = node.prev

    def _add_to_head(self, node: Node) -> None:
        """Add node right after head (most recent)."""
        node.next = self.head.next
        node.prev = self.head
        self.head.next.prev = node
        self.head.next = node

    def get(self, key: int) -> int:
        if key not in self.cache:
            return -1

        node = self.cache[key]
        # Move to head (most recently used)
        self._remove(node)
        self._add_to_head(node)
        return node.val

    def put(self, key: int, value: int) -> None:
        if key in self.cache:
            # Update existing node
            node = self.cache[key]
            node.val = value
            self._remove(node)
            self._add_to_head(node)
        else:
            # Create new node
            node = Node(key, value)
            self.cache[key] = node
            self._add_to_head(node)

            # Evict if over capacity
            if len(self.cache) > self.capacity:
                lru = self.tail.prev
                self._remove(lru)
                del self.cache[lru.key]
```

### TypeScript Solution

```typescript
class LRUNode {
    key: number;
    val: number;
    prev: LRUNode | null = null;
    next: LRUNode | null = null;

    constructor(key: number = 0, val: number = 0) {
        this.key = key;
        this.val = val;
    }
}

class LRUCache {
    private capacity: number;
    private cache: Map<number, LRUNode> = new Map();
    private head: LRUNode = new LRUNode();
    private tail: LRUNode = new LRUNode();

    constructor(capacity: number) {
        this.capacity = capacity;
        this.head.next = this.tail;
        this.tail.prev = this.head;
    }

    private remove(node: LRUNode): void {
        node.prev!.next = node.next;
        node.next!.prev = node.prev;
    }

    private addToHead(node: LRUNode): void {
        node.next = this.head.next;
        node.prev = this.head;
        this.head.next!.prev = node;
        this.head.next = node;
    }

    get(key: number): number {
        if (!this.cache.has(key)) return -1;

        const node = this.cache.get(key)!;
        this.remove(node);
        this.addToHead(node);
        return node.val;
    }

    put(key: number, value: number): void {
        if (this.cache.has(key)) {
            const node = this.cache.get(key)!;
            node.val = value;
            this.remove(node);
            this.addToHead(node);
        } else {
            const node = new LRUNode(key, value);
            this.cache.set(key, node);
            this.addToHead(node);

            if (this.cache.size > this.capacity) {
                const lru = this.tail.prev!;
                this.remove(lru);
                this.cache.delete(lru.key);
            }
        }
    }
}
```

### Interview Questions

1. **Why use dummy head and tail nodes?**
   - They eliminate edge cases for empty list or single-element operations. No null checks needed.

2. **Why store the key in the Node?**
   - When evicting the LRU node, we need its key to remove it from the hash map.

3. **How would you implement this thread-safe?**
   - Add a lock around get/put operations, or use a concurrent hash map with fine-grained locking per node.

4. **Netflix use case?**
   - Caching thumbnails, user session data, recently watched titles, or CDN content metadata.

### Complexity Analysis

| Operation | Time | Space |
|-----------|------|-------|
| get | O(1) | O(capacity) |
| put | O(1) | O(capacity) |

**Note:** LeetCode problem #146

---

## 33. Merge K Sorted Lists

**Type:** Coding Problem (Heap/Divide & Conquer)

**Problem:** You are given an array of `k` linked lists, each sorted in ascending order. Merge all the linked lists into one sorted linked list and return it.

**Focus:** Min-heap, divide and conquer, pointer manipulation

### Examples

**Example 1:**
```
Input: lists = [[1,4,5],[1,3,4],[2,6]]
Output: [1,1,2,3,4,4,5,6]
Explanation: Merging the three sorted lists produces one sorted list.
```

**Example 2:**
```
Input: lists = []
Output: []
```

**Example 3:**
```
Input: lists = [[]]
Output: []
```

### Constraints

- `k == lists.length`
- `0 <= k <= 10^4`
- `0 <= lists[i].length <= 500`
- `-10^4 <= lists[i][j] <= 10^4`
- Each `lists[i]` is sorted in ascending order
- Total nodes across all lists ≤ 10^4

### Key Insight

Use a **min-heap** to always get the smallest element among the k list heads. This gives O(N log k) time where N is total elements.

### Python Solution

```python
import heapq
from typing import List, Optional

class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

def mergeKLists(lists: List[Optional[ListNode]]) -> Optional[ListNode]:
    # Min-heap: (value, index, node)
    # Index is used as tiebreaker since nodes aren't comparable
    heap = []

    for i, node in enumerate(lists):
        if node:
            heapq.heappush(heap, (node.val, i, node))

    dummy = ListNode()
    current = dummy

    while heap:
        val, i, node = heapq.heappop(heap)
        current.next = node
        current = current.next

        if node.next:
            heapq.heappush(heap, (node.next.val, i, node.next))

    return dummy.next
```

### TypeScript Solution

```typescript
class ListNode {
    val: number;
    next: ListNode | null;
    constructor(val?: number, next?: ListNode | null) {
        this.val = val ?? 0;
        this.next = next ?? null;
    }
}

function mergeKLists(lists: Array<ListNode | null>): ListNode | null {
    // Filter out null lists
    const heads = lists.filter(node => node !== null) as ListNode[];
    if (heads.length === 0) return null;

    // Use divide and conquer approach
    while (heads.length > 1) {
        const merged: ListNode[] = [];
        for (let i = 0; i < heads.length; i += 2) {
            const l1 = heads[i];
            const l2 = i + 1 < heads.length ? heads[i + 1] : null;
            merged.push(mergeTwoLists(l1, l2));
        }
        heads.length = 0;
        heads.push(...merged);
    }

    return heads[0];
}

function mergeTwoLists(l1: ListNode | null, l2: ListNode | null): ListNode {
    const dummy = new ListNode();
    let current = dummy;

    while (l1 && l2) {
        if (l1.val <= l2.val) {
            current.next = l1;
            l1 = l1.next;
        } else {
            current.next = l2;
            l2 = l2.next;
        }
        current = current.next;
    }

    current.next = l1 || l2;
    return dummy.next!;
}
```

### Interview Questions

1. **Why use a heap instead of merging pairs?**
   - Both work. Heap is O(N log k), divide-and-conquer is also O(N log k). Heap uses O(k) space vs O(1) for iterative merge.

2. **What's the index in the heap tuple for?**
   - Python's heapq compares tuples element by element. If values are equal, it compares the next element. Nodes aren't comparable, so we add an index as a tiebreaker.

3. **Netflix use case?**
   - Merging sorted streams of user activity logs, combining ranked recommendation lists from different algorithms.

### Complexity Analysis

| Metric | Value |
|--------|-------|
| Time | O(N log k) |
| Space | O(k) for heap |

**Note:** LeetCode problem #23

---

## 34. Median of Two Sorted Arrays

**Type:** Coding Problem (Binary Search)

**Problem:** Given two sorted arrays `nums1` and `nums2`, return the median of the two sorted arrays. The overall runtime must be **O(log(m+n))**.

**Focus:** Binary search on partitions, edge case handling

### Examples

**Example 1:**
```
Input: nums1 = [1,3], nums2 = [2]
Output: 2.0
Explanation: Merged = [1,2,3], median = 2.
```

**Example 2:**
```
Input: nums1 = [1,2], nums2 = [3,4]
Output: 2.5
Explanation: Merged = [1,2,3,4], median = (2+3)/2 = 2.5.
```

### Constraints

- `0 <= m, n <= 1000`
- `1 <= m + n <= 2000`
- `-10^6 <= nums1[i], nums2[i] <= 10^6`

### Key Insight

Binary search on the smaller array to find a partition where:
- All elements on the left side ≤ all elements on the right side
- Left side has exactly `(m + n + 1) / 2` elements

### Python Solution

```python
def findMedianSortedArrays(nums1: list[int], nums2: list[int]) -> float:
    # Ensure nums1 is the smaller array
    if len(nums1) > len(nums2):
        nums1, nums2 = nums2, nums1

    m, n = len(nums1), len(nums2)
    half = (m + n + 1) // 2

    left, right = 0, m

    while left <= right:
        i = (left + right) // 2  # Partition index in nums1
        j = half - i              # Partition index in nums2

        # Edge values (use infinity for out of bounds)
        nums1_left = nums1[i - 1] if i > 0 else float('-inf')
        nums1_right = nums1[i] if i < m else float('inf')
        nums2_left = nums2[j - 1] if j > 0 else float('-inf')
        nums2_right = nums2[j] if j < n else float('inf')

        # Check if partition is correct
        if nums1_left <= nums2_right and nums2_left <= nums1_right:
            # Found correct partition
            if (m + n) % 2 == 1:
                return max(nums1_left, nums2_left)
            else:
                return (max(nums1_left, nums2_left) +
                        min(nums1_right, nums2_right)) / 2
        elif nums1_left > nums2_right:
            right = i - 1  # Move partition left in nums1
        else:
            left = i + 1   # Move partition right in nums1

    return 0.0  # Should never reach here
```

### TypeScript Solution

```typescript
function findMedianSortedArrays(nums1: number[], nums2: number[]): number {
    // Ensure nums1 is smaller
    if (nums1.length > nums2.length) {
        [nums1, nums2] = [nums2, nums1];
    }

    const m = nums1.length, n = nums2.length;
    const half = Math.floor((m + n + 1) / 2);

    let left = 0, right = m;

    while (left <= right) {
        const i = Math.floor((left + right) / 2);
        const j = half - i;

        const nums1Left = i > 0 ? nums1[i - 1] : -Infinity;
        const nums1Right = i < m ? nums1[i] : Infinity;
        const nums2Left = j > 0 ? nums2[j - 1] : -Infinity;
        const nums2Right = j < n ? nums2[j] : Infinity;

        if (nums1Left <= nums2Right && nums2Left <= nums1Right) {
            if ((m + n) % 2 === 1) {
                return Math.max(nums1Left, nums2Left);
            }
            return (Math.max(nums1Left, nums2Left) +
                    Math.min(nums1Right, nums2Right)) / 2;
        } else if (nums1Left > nums2Right) {
            right = i - 1;
        } else {
            left = i + 1;
        }
    }

    return 0;
}
```

### Interview Questions

1. **Why binary search on the smaller array?**
   - Reduces the search space. If we search on size m, we do O(log m) operations. Choosing the smaller array minimizes this.

2. **Why use infinity for out-of-bounds?**
   - It ensures comparisons always work correctly at boundaries without special cases.

3. **Netflix use case?**
   - Finding median latency from multiple data centers, computing percentiles for A/B test metrics.

### Complexity Analysis

| Metric | Value |
|--------|-------|
| Time | O(log(min(m, n))) |
| Space | O(1) |

**Note:** LeetCode problem #4

---

## 35. Maximum Subarray

**Type:** Coding Problem (Dynamic Programming)

**Problem:** Given an integer array `nums`, find the subarray with the largest sum and return its sum.

**Focus:** Kadane's algorithm, DP optimization

### Examples

**Example 1:**
```
Input: nums = [-2,1,-3,4,-1,2,1,-5,4]
Output: 6
Explanation: Subarray [4,-1,2,1] has the largest sum 6.
```

**Example 2:**
```
Input: nums = [1]
Output: 1
```

**Example 3:**
```
Input: nums = [5,4,-1,7,8]
Output: 23
Explanation: Entire array has the largest sum.
```

### Constraints

- `1 <= nums.length <= 10^5`
- `-10^4 <= nums[i] <= 10^4`

### Key Insight

**Kadane's Algorithm:** At each position, decide whether to:
1. Extend the previous subarray (add current element)
2. Start a new subarray from current element

The choice depends on whether the previous sum is positive.

### Python Solution

```python
def maxSubArray(nums: list[int]) -> int:
    max_sum = nums[0]
    current_sum = nums[0]

    for i in range(1, len(nums)):
        # Either extend previous subarray or start fresh
        current_sum = max(nums[i], current_sum + nums[i])
        max_sum = max(max_sum, current_sum)

    return max_sum
```

### TypeScript Solution

```typescript
function maxSubArray(nums: number[]): number {
    let maxSum = nums[0];
    let currentSum = nums[0];

    for (let i = 1; i < nums.length; i++) {
        currentSum = Math.max(nums[i], currentSum + nums[i]);
        maxSum = Math.max(maxSum, currentSum);
    }

    return maxSum;
}
```

### Divide and Conquer Alternative

```python
def maxSubArrayDivideConquer(nums: list[int]) -> int:
    def helper(left: int, right: int) -> int:
        if left == right:
            return nums[left]

        mid = (left + right) // 2

        # Max sum crossing the middle
        left_sum = float('-inf')
        curr = 0
        for i in range(mid, left - 1, -1):
            curr += nums[i]
            left_sum = max(left_sum, curr)

        right_sum = float('-inf')
        curr = 0
        for i in range(mid + 1, right + 1):
            curr += nums[i]
            right_sum = max(right_sum, curr)

        cross_sum = left_sum + right_sum

        return max(helper(left, mid), helper(mid + 1, right), cross_sum)

    return helper(0, len(nums) - 1)
```

### Interview Questions

1. **Why is this O(n) instead of O(n²)?**
   - We make a single pass, deciding at each element whether to extend or restart. No nested loops.

2. **How would you return the actual subarray?**
   - Track start/end indices. Update start when we restart, update end when we find a new maximum.

3. **Follow-up: What if the array is circular?**
   - Two cases: max subarray doesn't wrap (use Kadane's) or wraps (total sum minus minimum subarray). Take the maximum.

### Complexity Analysis

| Approach | Time | Space |
|----------|------|-------|
| Kadane's | O(n) | O(1) |
| Divide & Conquer | O(n log n) | O(log n) |

**Note:** LeetCode problem #53

---

## 36. Number of Islands

**Type:** Coding Problem (Graph/DFS)

**Problem:** Given an `m x n` 2D grid of `'1'`s (land) and `'0'`s (water), return the number of islands. An island is surrounded by water and is formed by connecting adjacent lands horizontally or vertically.

**Focus:** DFS/BFS flood fill, connected components

### Examples

**Example 1:**
```
Input: grid = [
  ["1","1","1","1","0"],
  ["1","1","0","1","0"],
  ["1","1","0","0","0"],
  ["0","0","0","0","0"]
]
Output: 1
```

**Example 2:**
```
Input: grid = [
  ["1","1","0","0","0"],
  ["1","1","0","0","0"],
  ["0","0","1","0","0"],
  ["0","0","0","1","1"]
]
Output: 3
```

### Constraints

- `m == grid.length`
- `n == grid[i].length`
- `1 <= m, n <= 300`
- `grid[i][j]` is `'0'` or `'1'`

### Key Insight

Use DFS/BFS to "flood fill" each island. When we find a '1', increment count and mark all connected '1's as visited (change to '0' or use a visited set).

### Python Solution

```python
def numIslands(grid: list[list[str]]) -> int:
    if not grid:
        return 0

    rows, cols = len(grid), len(grid[0])
    count = 0

    def dfs(r: int, c: int) -> None:
        # Boundary check and water check
        if r < 0 or r >= rows or c < 0 or c >= cols or grid[r][c] == '0':
            return

        # Mark as visited (sink the island)
        grid[r][c] = '0'

        # Explore all 4 directions
        dfs(r + 1, c)
        dfs(r - 1, c)
        dfs(r, c + 1)
        dfs(r, c - 1)

    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == '1':
                count += 1
                dfs(r, c)

    return count
```

### TypeScript Solution

```typescript
function numIslands(grid: string[][]): number {
    if (grid.length === 0) return 0;

    const rows = grid.length;
    const cols = grid[0].length;
    let count = 0;

    function dfs(r: number, c: number): void {
        if (r < 0 || r >= rows || c < 0 || c >= cols || grid[r][c] === '0') {
            return;
        }

        grid[r][c] = '0';  // Mark visited

        dfs(r + 1, c);
        dfs(r - 1, c);
        dfs(r, c + 1);
        dfs(r, c - 1);
    }

    for (let r = 0; r < rows; r++) {
        for (let c = 0; c < cols; c++) {
            if (grid[r][c] === '1') {
                count++;
                dfs(r, c);
            }
        }
    }

    return count;
}
```

### BFS Alternative

```python
from collections import deque

def numIslandsBFS(grid: list[list[str]]) -> int:
    if not grid:
        return 0

    rows, cols = len(grid), len(grid[0])
    count = 0

    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == '1':
                count += 1
                queue = deque([(r, c)])
                grid[r][c] = '0'

                while queue:
                    row, col = queue.popleft()
                    for dr, dc in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                        nr, nc = row + dr, col + dc
                        if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] == '1':
                            queue.append((nr, nc))
                            grid[nr][nc] = '0'

    return count
```

### Interview Questions

1. **DFS vs BFS - which is better?**
   - Same time complexity. DFS uses call stack (risk of stack overflow for huge grids), BFS uses explicit queue. BFS is iterative and safer for very large inputs.

2. **How to avoid modifying the input grid?**
   - Use a separate `visited` set of coordinates, but this increases space to O(m×n).

3. **Netflix use case?**
   - Clustering user segments, identifying connected regions in recommendation graphs, analyzing content viewing patterns.

4. **Follow-up: Count the size of the largest island?**
   - Return the count from DFS instead of just marking visited.

### Complexity Analysis

| Metric | Value |
|--------|-------|
| Time | O(m × n) |
| Space | O(m × n) worst case for DFS stack |

**Note:** LeetCode problem #200

---

## 37. Longest Palindromic Substring

**Type:** Coding Problem (String/DP)

**Problem:** Given a string `s`, return the longest palindromic substring in `s`.

**Focus:** Expand around center, dynamic programming

### Examples

**Example 1:**
```
Input: s = "babad"
Output: "bab" (or "aba")
```

**Example 2:**
```
Input: s = "cbbd"
Output: "bb"
```

### Constraints

- `1 <= s.length <= 1000`
- `s` consists of only digits and English letters

### Key Insight

**Expand Around Center:** A palindrome mirrors around its center. There are `2n - 1` possible centers (n single characters + n-1 gaps between characters for even-length palindromes).

### Python Solution

```python
def longestPalindrome(s: str) -> str:
    if len(s) < 2:
        return s

    start, max_len = 0, 1

    def expand_around_center(left: int, right: int) -> int:
        """Expand and return the length of palindrome."""
        while left >= 0 and right < len(s) and s[left] == s[right]:
            left -= 1
            right += 1
        return right - left - 1

    for i in range(len(s)):
        # Odd length palindrome (single center)
        len1 = expand_around_center(i, i)
        # Even length palindrome (gap center)
        len2 = expand_around_center(i, i + 1)

        curr_len = max(len1, len2)
        if curr_len > max_len:
            max_len = curr_len
            start = i - (curr_len - 1) // 2

    return s[start:start + max_len]
```

### TypeScript Solution

```typescript
function longestPalindrome(s: string): string {
    if (s.length < 2) return s;

    let start = 0, maxLen = 1;

    function expandAroundCenter(left: number, right: number): number {
        while (left >= 0 && right < s.length && s[left] === s[right]) {
            left--;
            right++;
        }
        return right - left - 1;
    }

    for (let i = 0; i < s.length; i++) {
        const len1 = expandAroundCenter(i, i);      // Odd
        const len2 = expandAroundCenter(i, i + 1);  // Even

        const currLen = Math.max(len1, len2);
        if (currLen > maxLen) {
            maxLen = currLen;
            start = i - Math.floor((currLen - 1) / 2);
        }
    }

    return s.substring(start, start + maxLen);
}
```

### DP Alternative

```python
def longestPalindromeDP(s: str) -> str:
    n = len(s)
    if n < 2:
        return s

    # dp[i][j] = True if s[i:j+1] is palindrome
    dp = [[False] * n for _ in range(n)]
    start, max_len = 0, 1

    # All single characters are palindromes
    for i in range(n):
        dp[i][i] = True

    # Check for length 2 to n
    for length in range(2, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1

            if length == 2:
                dp[i][j] = (s[i] == s[j])
            else:
                dp[i][j] = (s[i] == s[j] and dp[i + 1][j - 1])

            if dp[i][j] and length > max_len:
                start = i
                max_len = length

    return s[start:start + max_len]
```

### Interview Questions

1. **Why expand around center instead of DP?**
   - Same O(n²) time, but expand uses O(1) space vs O(n²) for DP table. Expand is also more intuitive.

2. **Why 2n-1 centers?**
   - n centers for odd-length palindromes (each character), n-1 centers for even-length (gaps between characters).

3. **What is Manacher's Algorithm?**
   - An O(n) algorithm that uses previously computed palindrome information to skip redundant comparisons. Complex but optimal.

4. **Netflix use case?**
   - Pattern matching in subtitles, detecting repeated sequences in user search queries.

### Complexity Analysis

| Approach | Time | Space |
|----------|------|-------|
| Expand Around Center | O(n²) | O(1) |
| DP | O(n²) | O(n²) |
| Manacher's | O(n) | O(n) |

**Note:** LeetCode problem #5

---

## Summary by Category

| Category | Count | Questions |
|----------|-------|-----------|
| System Design | 9 | Sentiment Tracking, Ads Platform, Video Recommendation, Audience Targeting, Billing (300M), Frequency Cap, ML Scheduler, WAL Pipeline, Promotion System |
| Arrays & Strings | 12 | Contains Duplicate (I, II, III), First Missing Positive, Longest Palindromic Substring, Longest Substring, Maximum Subarray, Median of Two Sorted Arrays, Meeting Rooms, Number Pairs, Sort by Preference, User Engagement |
| Graph & Search | 6 | Number of Islands, Parallel Courses (I, II), Reconstruct Itinerary, Movie History Friends (I, II) |
| Data Structures | 6 | LRU Cache, Merge K Sorted Lists, Versioned File System, Auto-Expire Cache, Music Playlist, Homepage Deduplication |
| Concurrency | 2 | Countdown Latch, Timer Function |
| ML & Analytics | 2 | Error Rate Monitor, Spam Email Detection |

**Total: 37 questions**

---

## Common Patterns at Netflix

1. **Ads/Advertising Systems** - Multiple questions on ad targeting, frequency capping, demand platforms
2. **Content Recommendation** - Homepage recommendations, user preferences, engagement patterns
3. **Data Modeling** - Focus on database schema design for various systems
4. **Scalability** - 300M subscriber billing system shows emphasis on scale
5. **Classic Algorithms** - Contains Duplicate series, Meeting Rooms, Reconstruct Itinerary, Maximum Subarray
6. **Caching** - LRU Cache, Auto-expire cache variations
7. **Concurrency** - Thread-safe implementations (Countdown Latch)
8. **Bitmask DP** - Small constraint problems (Parallel Courses II with n ≤ 15)
9. **Binary Search** - Median of Two Sorted Arrays, partition-based solutions
10. **Heap/Priority Queue** - Merge K Sorted Lists, top-K problems

---

*Last updated: 2026-04-17*
