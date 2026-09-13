# Google Interview Questions

---

## Contents

**Coding**
1. [Minimum Meeting Rooms (Interval Scheduling)](#1-minimum-meeting-rooms-interval-scheduling)
2. [Accounts Merge (Union-Find)](#2-accounts-merge-union-find)
3. [Top-K Frequent Items (Streaming)](#3-top-k-frequent-items-streaming)
4. [Word Dictionary with Wildcard Search (Trie + DFS)](#4-word-dictionary-with-wildcard-search-trie--dfs)
5. [Decode String (Nested Stack Decoding)](#5-decode-string-nested-stack-decoding)
6. [3Sum (Zero-Sum Triplets)](#6-3sum-zero-sum-triplets)
7. [Longest Increasing Subarray / Subsequence](#7-longest-increasing-subarray--subsequence)
8. [Implement a Queue Using Two Stacks](#8-implement-a-queue-using-two-stacks)

**System Design**
9. [Retrieval-Augmented Generation (RAG)](#9-system-design--retrieval-augmented-generation-rag)
10. [Distributed Message Queue (Kafka-Style)](#10-system-design--distributed-message-queue-kafka-style)
11. [URL Shortener](#11-system-design--url-shortener)
12. [Distributed Cache (Redis-Style)](#12-system-design--distributed-cache-redis-style)

**Behavioral**
13. [Behavioral Themes](#13-behavioral-themes)

---

## 1. Minimum Meeting Rooms (Interval Scheduling)

**Problem Statement:**
Google's public listing for this one only exposes the tags (`interval`, `heap`, `greedy`, medium
difficulty) — the full wording is paywalled on the source forum. This is a reconstructed, standard
version consistent with those tags, not a verbatim transcript: given a list of meetings as `(start, end)`
intervals, find the minimum number of conference rooms required so that no two meetings needing a room
at the same time share one.

**Example:**
```
Input:  [(0,30), (5,10), (15,20)]
Output: 2   # (5,10) and (15,20) both fit inside (0,30), but not simultaneously in one room
```

**Test Cases:**

| Intervals | Rooms |
|---|---|
| `[(0,30),(5,10),(15,20)]` | `2` |
| `[(7,10),(2,4)]` | `1` |
| `[(1,5),(5,10)]` | `1` (touching endpoints don't overlap) |
| `[]` | `0` |

**Key Insights:**
1. Sort meetings by start time; track room end-times in a min-heap.
2. For each meeting, if the room that frees up earliest (`heap[0]`) does so at or before this meeting's
   start, reuse that room (`heapreplace`); otherwise allocate a new room (`heappush`).
3. The heap's final size is the answer — no need to track room identities unless a follow-up asks for
   the actual assignment.

**Python Solution:**
```python
import heapq


def min_meeting_rooms(intervals: list[tuple[int, int]]) -> int:
    """
    Time:  O(n log n)
    Space: O(n)
    """
    if not intervals:
        return 0

    intervals = sorted(intervals, key=lambda iv: iv[0])
    heap: list[int] = []  # end times of rooms currently in use

    for start, end in intervals:
        if heap and heap[0] <= start:
            heapq.heapreplace(heap, end)
        else:
            heapq.heappush(heap, end)

    return len(heap)
```

**Follow-Up Questions:**
1. Return the actual room assignment per meeting, not just the count → push `(end_time, room_id)` tuples
   onto the heap instead of bare end times, reusing the popped `room_id` when a room is freed.
2. Meetings are added/removed dynamically rather than given as a fixed batch → maintain a live sorted
   structure (e.g., a balanced tree of active intervals) instead of a one-shot sort + heap pass.

---

## 2. Accounts Merge (Union-Find)

**Problem Statement:**
Also reconstructed from paywalled tags (`graph`, `union-find`, hard) — this is the classic "Accounts
Merge" problem, which matches those tags closely. Given `n` accounts, each a list `[name, email1,
email2, ...]`, two accounts belong to the same person if they share at least one email (even
transitively, through a third account). Merge accounts belonging to the same person and return each
merged account as `[name, sorted_deduplicated_emails...]`.

**Example:**
```
Input:
[["John","johnsmith@mail.com","john_ny@mail.com"],
 ["John","johnsmith@mail.com","john00@mail.com"],
 ["Mary","mary@mail.com"],
 ["John","johnnybravo@mail.com"]]

Output:
[["John","john00@mail.com","john_ny@mail.com","johnsmith@mail.com"],
 ["Mary","mary@mail.com"],
 ["John","johnnybravo@mail.com"]]
```

**Test Cases:**

| Accounts | Result groups |
|---|---|
| Two accounts sharing one email | merged into one, emails unioned |
| Three accounts chained by shared emails (A-B share one email, B-C share another) | all three merge into one |
| Account with no shared email with anyone | stays its own singleton group |
| Same account list, different input order | same groups, just possibly different output order |

**Key Insights:**
1. Model each account index as a node; union two account indices whenever they share an email.
2. A single pass builds an `email -> first account index seen` map; whenever an email is seen again on a
   different account, union that account with the first one that had it.
3. After unioning, group emails by each account's root parent — the tricky part is doing the
   email→account bookkeeping correctly, not the union-find itself.

**Python Solution:**
```python
class DSU:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]  # path compression
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb


def accounts_merge(accounts: list[list[str]]) -> list[list[str]]:
    """
    Time:  O(N * K * alpha(N)) for N accounts averaging K emails each
    Space: O(N * K)
    """
    dsu = DSU(len(accounts))
    email_to_acct: dict[str, int] = {}

    for i, account in enumerate(accounts):
        for email in account[1:]:
            if email in email_to_acct:
                dsu.union(i, email_to_acct[email])
            else:
                email_to_acct[email] = i

    groups: dict[int, set[str]] = {}
    for email, i in email_to_acct.items():
        root = dsu.find(i)
        groups.setdefault(root, set()).add(email)

    return [[accounts[root][0]] + sorted(emails) for root, emails in groups.items()]
```

**Follow-Up Questions:**
1. Accounts merge by shared phone number too, not just email → union-find is unchanged; just also feed
   phone numbers through the same "seen before → union" logic as a second identity key.
2. Millions of accounts, too large for one machine → union-find doesn't parallelize cleanly; this becomes
   a connected-components problem better suited to a distributed graph framework (e.g., iterative
   label-propagation over a Spark/graph-processing job) rather than in-memory DSU.

---

## 3. Top-K Frequent Items (Streaming)

**Problem Statement:**
Reconstructed from paywalled tags (`heap`, `hashmap`, `scaling`, medium). Design a `TopKFrequent`
structure that supports streaming `add(item)` calls and a `top_k()` query returning the `k` most
frequent items seen so far. The "scaling" tag implies the interviewer wants a discussion of how this
holds up when `add()` is called far more often than `top_k()`.

**Example:**
```
add("a"); add("a"); add("b"); add("c"); add("c"); add("c")
top_k(2) -> [("c", 3), ("a", 2)]
```

**Test Cases:**

| Operations | `top_k(k)` |
|---|---|
| `add(a); add(a); add(b); top_k(1)` | `[(a,2)]` |
| `add(a); add(b); add(c); top_k(2)` | any 2 of the 3, all tied at count 1 |
| `top_k(1)` with no adds | `[]` |
| `add(x) * 100; top_k(5)` when only 1 distinct item exists | `[(x,100)]` |

**Key Insights:**
1. A hashmap (`Counter`) tracks running counts — `add()` must be O(1) since it's the hot path.
2. Don't maintain a heap continuously on every `add()` if `top_k()` is called far less often — that
   defeats the purpose of the "scaling" framing. Build the heap lazily, only inside `top_k()`.
3. `heapq.nlargest(k, ...)` is `O(n log k)`, not `O(n log n)` — it maintains a size-`k` heap internally
   rather than sorting everything, which matters when `n` (distinct items) is large but `k` is small.

**Python Solution:**
```python
import heapq
from collections import Counter


class TopKFrequent:
    """
    add():   O(1) amortized
    top_k(): O(n log k) for n distinct items seen
    Space:   O(n)
    """

    def __init__(self):
        self.counts: Counter = Counter()

    def add(self, item) -> None:
        self.counts[item] += 1

    def top_k(self, k: int) -> list[tuple]:
        return heapq.nlargest(k, self.counts.items(), key=lambda kv: kv[1])
```

**Follow-Up Questions:**
1. The item universe is huge and effectively unbounded (can't afford an exact counter per item) →
   switch to an approximate structure like Count-Min Sketch or the Space-Saving algorithm, trading exact
   counts for bounded memory.
2. Counts need to be maintained across many machines (e.g., sharded by item hash) → each shard keeps
   local counts; a periodic merge step combines shard-local top-k lists into a global approximate top-k
   (exact global top-k generally requires collecting full counts, not just each shard's local top-k).

---

## 4. Word Dictionary with Wildcard Search (Trie + DFS)

**Problem Statement:**
Reconstructed from paywalled tags (`trie`, `tree`, `dfs`, medium). Implement a `WordDictionary`
supporting `add_word(word)` and `search(pattern)`, where `pattern` may contain `.` as a wildcard matching
any single character.

**Example:**
```
add_word("bad"); add_word("dad"); add_word("mad")
search("pad")  -> False
search("bad")  -> True
search(".ad")  -> True
search("b..")  -> True
```

**Test Cases:**

| Words added | Query | Result |
|---|---|---|
| `bad, dad, mad` | `pad` | `False` |
| `bad, dad, mad` | `.ad` | `True` |
| `bad, dad, mad` | `b..` | `True` |
| `bad` | `ba.d` (wrong length) | `False` |
| (none added) | `.` | `False` |

**Key Insights:**
1. Standard trie for `add_word`; `search` is a DFS over the trie instead of a simple walk, because `.`
   must branch into every child at that position rather than following one fixed edge.
2. The recursion naturally short-circuits with `any(...)` — stop exploring siblings as soon as one branch
   finds a match.
3. This degrades badly with many wildcards: a pattern of all `.` is `O(26^L)` in the worst case (or
   `O(b^L)` for alphabet size `b`), since every position branches into every child — call this out
   explicitly rather than letting it look like it's always `O(L)`.

**Python Solution:**
```python
class TrieNode:
    def __init__(self):
        self.children: dict[str, "TrieNode"] = {}
        self.is_word = False


class WordDictionary:
    """
    add_word(): O(L) for word length L
    search():   O(L) best case, O(b^L) worst case with b = branching factor, many wildcards
    Space:      O(total characters across all added words)
    """

    def __init__(self):
        self.root = TrieNode()

    def add_word(self, word: str) -> None:
        node = self.root
        for ch in word:
            node = node.children.setdefault(ch, TrieNode())
        node.is_word = True

    def search(self, word: str) -> bool:
        def dfs(node: TrieNode, i: int) -> bool:
            if i == len(word):
                return node.is_word
            ch = word[i]
            if ch == ".":
                return any(dfs(child, i + 1) for child in node.children.values())
            child = node.children.get(ch)
            return dfs(child, i + 1) if child else False

        return dfs(self.root, 0)
```

**Follow-Up Questions:**
1. When does this degrade, and how would you bound it? → the crux follow-up: bound the number of
   wildcards allowed per query, or cap total DFS branches explored, to keep worst-case latency bounded
   for a shared/multi-tenant service.
2. Support prefix search (`starts_with`) in addition to exact/wildcard match → same trie, just stop the
   walk at the end of the prefix and return `True` without checking `is_word`.

---

## 5. Decode String (Nested Stack Decoding)

**Problem Statement:**
A real, commonly-reported Google interview question (corroborated across Glassdoor and multiple
interview-prep aggregators, not from a paywalled source). Given an encoded string with the pattern
`k[encoded_string]` — meaning `encoded_string` repeats exactly `k` times — decode it fully. Encodings can
nest arbitrarily deep, e.g. `3[a2[c]]`.

**Example:**
```
Input:  "3[a2[c]]"
Output: "accaccacc"

Input:  "2[abc]3[cd]ef"
Output: "abcabccdcdcdef"
```

**Test Cases:**

| Input | Output |
|---|---|
| `"3[a]2[bc]"` | `"aaabcbc"` |
| `"3[a2[c]]"` | `"accaccacc"` |
| `"2[abc]3[cd]ef"` | `"abcabccdcdcdef"` |
| `"abc"` (no brackets) | `"abc"` |
| `"10[a]"` (multi-digit count) | `"aaaaaaaaaa"` |

**Key Insights:**
1. Two stacks (or one stack of `(count, partial_string)` pairs): one tracks the repeat counts seen so
   far, the other tracks the string built up before entering each bracket level.
2. On `[`, push the current count and the string built so far, then reset both to start accumulating the
   bracket's contents fresh.
3. On `]`, pop the saved count and prefix string, and append `count * current_string` onto the popped
   prefix — this is what correctly handles arbitrary nesting depth without recursion (though a recursive
   solution mirroring the same push/pop structure on the call stack is equally valid).
4. Multi-digit counts (`"10[a]"`) require accumulating consecutive digit characters, not just reading one
   digit at a time.

**Python Solution:**
```python
def decode_string(s: str) -> str:
    """
    Time:  O(n * m) where m is the maximum expansion multiplier along any nesting chain
           (dominated by the size of the final decoded output)
    Space: O(n) for the stacks
    """
    count_stack: list[int] = []
    string_stack: list[str] = []
    current = ""
    count = 0

    for ch in s:
        if ch.isdigit():
            count = count * 10 + int(ch)
        elif ch == "[":
            count_stack.append(count)
            string_stack.append(current)
            count = 0
            current = ""
        elif ch == "]":
            prev_count = count_stack.pop()
            prev_string = string_stack.pop()
            current = prev_string + current * prev_count
        else:
            current += ch

    return current
```

**Follow-Up Questions:**
1. What if the input can be malformed (unbalanced brackets, a count with no following `[`)? → validate
   during the same pass: an unmatched `]` means `count_stack` is empty on pop, an unmatched `[` means the
   stacks are non-empty at the end — raise on either.
2. How would you decode without materializing the full expanded string (if it's astronomically large,
   e.g. deeply nested counts)? → this becomes a "query the i-th character without expanding" problem,
   solved by tracking expansion *lengths* (which can overflow a normal string but fit in an integer) and
   recursing/jumping directly to the segment containing index `i`.

---

## 6. 3Sum (Zero-Sum Triplets)

**Problem Statement:**
A classic, widely-reported Google interview question. Given an integer array, return all unique triplets
`[a, b, c]` such that `a + b + c == 0`. The result must not contain duplicate triplets.

**Example:**
```
Input:  [-1, 0, 1, 2, -1, -4]
Output: [[-1, -1, 2], [-1, 0, 1]]
```

**Test Cases:**

| Input | Output |
|---|---|
| `[-1,0,1,2,-1,-4]` | `[[-1,-1,2],[-1,0,1]]` |
| `[0,1,1]` | `[]` |
| `[0,0,0]` | `[[0,0,0]]` |
| `[0,0,0,0]` | `[[0,0,0]]` (still just one unique triplet) |

**Key Insights:**
1. Sort the array first, then fix each element in turn as the smallest of the triplet and two-pointer
   the remaining subarray for a pair summing to `-fixed` — this turns an `O(n^3)` brute force into
   `O(n^2)`.
2. Skip duplicate values for the fixed element (`if i > 0 and nums[i] == nums[i-1]: continue`) and skip
   duplicates for the two pointers after finding a match, to avoid emitting the same triplet twice.
3. Sorting first is also what makes the two-pointer sweep possible at all — without it you'd need a
   hashset-based approach that's harder to dedupe cleanly.

**Python Solution:**
```python
def three_sum(nums: list[int]) -> list[list[int]]:
    """
    Time:  O(n^2)
    Space: O(1) extra beyond the output (O(n) or O(log n) for the sort, depending on implementation)
    """
    nums = sorted(nums)
    n = len(nums)
    result = []

    for i in range(n - 2):
        if i > 0 and nums[i] == nums[i - 1]:
            continue
        if nums[i] > 0:
            break  # smallest element positive => no triplet can sum to zero

        left, right = i + 1, n - 1
        while left < right:
            total = nums[i] + nums[left] + nums[right]
            if total < 0:
                left += 1
            elif total > 0:
                right -= 1
            else:
                result.append([nums[i], nums[left], nums[right]])
                left += 1
                right -= 1
                while left < right and nums[left] == nums[left - 1]:
                    left += 1
                while left < right and nums[right] == nums[right + 1]:
                    right -= 1

    return result
```

**Follow-Up Questions:**
1. Generalize to `k`-Sum → recurse: reduce `kSum` to `(k-1)Sum` on a fixed element, bottoming out at
   `2Sum` via two pointers on the sorted array.
2. Return the *count* of triplets rather than the triplets themselves, at very large `n` → the two-pointer
   approach still works, just accumulate a count instead of materializing triplets (careful to count
   duplicate-value groups combinatorially rather than iterating each one).

---

## 7. Longest Increasing Subarray / Subsequence

**Problem Statement:**
A reported Google question; the exact wording (contiguous "subarray" vs. non-contiguous "subsequence")
varies by source, so both are covered here — clarify which one is meant with the interviewer up front,
since they have very different solutions.

**Longest Increasing *Subarray*** (contiguous): return the length of the longest contiguous run of
strictly increasing elements.
```
Input:  [1, 3, 5, 4, 7]
Output: 3   # [1, 3, 5]
```

**Longest Increasing *Subsequence*** (not necessarily contiguous, classic LIS): return the length of the
longest strictly increasing subsequence.
```
Input:  [10, 9, 2, 5, 3, 7, 101, 18]
Output: 4   # [2, 3, 7, 101] (or [2, 3, 7, 18])
```

**Test Cases:**

| Input | Subarray answer | Subsequence answer |
|---|---|---|
| `[1,3,5,4,7]` | `3` | `4` (`1,3,5,7` or `1,3,4,7`) |
| `[5,4,3,2,1]` | `1` | `1` |
| `[10,9,2,5,3,7,101,18]` | `3` (`3,7,101`) | `4` |
| `[]` | `0` | `0` |

**Key Insights:**
1. The subarray version is a single `O(n)` linear scan tracking a running streak length — no DP needed,
   since a contiguous run can't "skip back" to extend an earlier one.
2. The subsequence version is the classic `O(n^2)` DP (`dp[i]` = length of the longest increasing
   subsequence ending at `i`) — but the interview bar is usually the `O(n log n)` patience-sorting
   variant: maintain a `tails` array where `tails[k]` is the smallest possible tail value of an
   increasing subsequence of length `k+1`, and binary-search each new element's insertion point.
3. `tails` itself is not a valid subsequence of the input — it's an auxiliary structure; don't try to read
   the actual LIS elements off it without extra bookkeeping (a follow-up point below).

**Python Solution:**
```python
import bisect


def longest_increasing_subarray(nums: list[int]) -> int:
    """
    Time:  O(n)
    Space: O(1)
    """
    if not nums:
        return 0
    best = cur = 1
    for i in range(1, len(nums)):
        cur = cur + 1 if nums[i] > nums[i - 1] else 1
        best = max(best, cur)
    return best


def longest_increasing_subsequence(nums: list[int]) -> int:
    """
    Time:  O(n log n)
    Space: O(n) for the tails array
    """
    tails: list[int] = []
    for x in nums:
        pos = bisect.bisect_left(tails, x)
        if pos == len(tails):
            tails.append(x)
        else:
            tails[pos] = x
    return len(tails)
```

**Follow-Up Questions:**
1. Reconstruct the actual longest increasing subsequence, not just its length → track, alongside `tails`,
   a parent pointer per element pointing at the element it extended; walk parents back from the last
   element placed at the final `tails` position.
2. Longest *non-decreasing* subsequence (equal values allowed to extend) → change `bisect_left` to
   `bisect_right` so equal values are treated as extending the run rather than replacing a tail.

---

## 8. Implement a Queue Using Two Stacks

**Problem Statement:**
A classic, widely-reported Google question. Implement a FIFO queue (`push`, `pop`, `peek`, `empty`) using
only two stacks (LIFO structures) as the underlying storage.

**Test Cases:**

| Operations | Result |
|---|---|
| `push(1); push(2); peek()` | `1` |
| `push(1); push(2); pop(); pop()` | `1`, then `2` |
| `push(1); pop(); push(2); peek()` | `2` |
| `empty()` on a fresh queue | `True` |

**Key Insights:**
1. Use an "in" stack for pushes and an "out" stack for pops/peeks. Pushing is always `O(1)` — just push
   onto "in".
2. On `pop`/`peek`, if "out" is empty, dump all of "in" onto "out" (reversing order so the oldest pushed
   element ends up on top of "out") — then pop/peek from "out" as normal.
3. This gives amortized `O(1)` per operation: each element is moved from "in" to "out" at most once over
   its lifetime, even though a single dump can be `O(n)` in the worst case.

**Python Solution:**
```python
class QueueWithTwoStacks:
    """
    push():  O(1)
    pop()/peek(): O(1) amortized (occasional O(n) dump from in-stack to out-stack)
    Space: O(n)
    """

    def __init__(self):
        self._in: list[int] = []
        self._out: list[int] = []

    def push(self, x: int) -> None:
        self._in.append(x)

    def _shift(self) -> None:
        if not self._out:
            while self._in:
                self._out.append(self._in.pop())

    def pop(self) -> int:
        self._shift()
        return self._out.pop()

    def peek(self) -> int:
        self._shift()
        return self._out[-1]

    def empty(self) -> bool:
        return not self._in and not self._out
```

**Follow-Up Questions:**
1. Do the reverse — implement a stack using two queues → push onto one queue, then rotate all-but-the-last
   pushed element to the back so the most recent push ends up at the front (making `pop` = dequeue).
2. Prove the amortized `O(1)` bound formally → an amortized analysis (potential/banker's method) showing
   each element crosses from "in" to "out" exactly once, so total moves across `n` operations is bounded
   by `n`, not `n^2`.

---

## 9. System Design — Retrieval-Augmented Generation (RAG)

**Problem Statement:**
Design a retrieval-augmented generation system: given a large corpus of documents, support low-latency
semantic search to retrieve the top-k relevant chunks for a user query, then pass them to an LLM to
generate a grounded answer.

**Functional Requirements:**
- Ingest a large, growing corpus of documents; chunk and embed them for retrieval.
- Given a query, return the top-k most relevant chunks via semantic (embedding) search.
- Construct an LLM prompt from the retrieved chunks and generate an answer, ideally with citations back
  to source chunks.
- Support adding, updating, and deleting documents over time.

**Non-Functional Requirements:**
- Low end-to-end query latency (embedding the query, ANN search, optional re-ranking, and generation all
  fit within an interactive budget).
- Scale to a corpus far larger than one machine's memory — the index must be shardable.
- Bounded staleness: new/updated documents should become searchable within a known delay, even if not
  instantly.

**High-Level Design:**
1. **Ingestion pipeline**: split documents into chunks (fixed-size windows vs. semantic/sentence-boundary
   chunking, with some overlap between adjacent chunks so context isn't cut mid-thought), run a batch
   embedding job over new/changed chunks, and write vectors + metadata into the index and a document store.
2. **Vector index**: an approximate nearest-neighbor (ANN) index (HNSW or IVF) rather than exact search —
   discuss the recall/latency/memory tradeoff explicitly. Shard the index across nodes once the corpus
   exceeds one machine's memory; replicate shards for read availability and throughput.
3. **Query path**: embed the incoming query with the same embedding model used at ingestion time, run
   ANN search per shard, merge the per-shard top-k candidates, and optionally re-rank the merged shortlist
   with a more expensive cross-encoder (cheap embedding model for broad recall, expensive model only on
   the shortlist for precision).
4. **Prompt construction & generation**: assemble the LLM prompt from the top-ranked chunks (with source
   attribution), call the LLM, and return the answer alongside citations for verification.
5. **Freshness pipeline**: either near-real-time streaming ingestion for individual document updates, or
   periodic batch re-indexing — pick based on how stale the corpus is allowed to get, and say so
   explicitly as a tradeoff.

**Data Model (sketch):**
```
documents(doc_id, source, version, updated_at)
chunks(chunk_id, doc_id, text, position, embedding_id)
vector_index: chunk_id -> embedding vector   # ANN index, sharded by corpus partition
```

**Scaling & Reliability:**
- Shard the vector index (e.g., by document source or a coarse embedding-space partition); fan out a
  query to all shards and merge top-k results.
- Cache frequently-repeated or recent queries' retrieved chunk sets to cut ANN search load.
- If the retrieval or re-ranking stage times out, fall back to a smaller or cached candidate set rather
  than failing the whole request outright.
- Tombstone deleted/updated chunks in the index and filter them at query time until the next compaction
  pass physically removes them, rather than doing expensive in-place index surgery per delete.

**Follow-Up Questions:**
1. How do you evaluate retrieval quality? → build an offline eval set of (query, relevant chunk) pairs
   and measure recall@k; online, track downstream answer quality via human eval or user feedback signals.
2. The corpus doesn't fit in memory on one node → shard by document source or an embedding-space
   partition, query all shards in parallel, and merge the per-shard top-k results before re-ranking.
3. How do you keep the index from serving stale/deleted content? → tombstone at write time, filter
   tombstoned chunks at query time, and periodically compact the index to physically drop them.

---

## 10. System Design — Distributed Message Queue (Kafka-Style)

**Problem Statement:**
Design a distributed message queue (a simplified Kafka) that guarantees at-least-once delivery, supports
multiple independent consumer groups, and scales to high write throughput.

**Functional Requirements:**
- Producers publish messages to named topics.
- Multiple independent consumer groups can each read the full topic at their own pace, without
  interfering with each other.
- Delivery is at-least-once even across broker or consumer failures.

**Non-Functional Requirements:**
- High, horizontally-scalable write throughput.
- Ordering is guaranteed only within a partition, not across an entire topic.
- Producers can choose a durability/latency tradeoff per write (acknowledgment level).

**High-Level Design:**
1. **Partitioning**: each topic is split into a fixed number of partitions for parallelism. A producer's
   partition key (e.g., hash of a message key) determines which partition a message lands in, which is
   what gives ordering *within* that key.
2. **Replication**: each partition has one leader broker and some number of follower replicas. Writes go
   to the leader and are replicated to followers before being acknowledged, depending on the configured
   acknowledgment level (`ack=all` waits for all replicas — safer, slower; `ack=1` waits for the leader
   only — faster, riskier on leader failure).
3. **Storage engine**: each partition is an append-only log on disk (sequential writes are fast even on
   spinning disks), split into segment files, with a retention or compaction policy (delete old segments,
   or keep only the latest value per key) configured per topic.
4. **Consumer offsets**: each consumer group tracks its own committed offset per partition, so
   independent groups read the same log at their own pace. Committing the offset only after processing a
   message (not before) gives at-least-once delivery — a crash between processing and committing causes
   reprocessing, never silent message loss.
5. **Failure handling**: a coordination layer (a Raft-based controller, or historically ZooKeeper) handles
   leader election when a broker holding a partition's leader fails; consumer group membership changes
   trigger a rebalance of partition ownership among the group's remaining members.

**Data Model (sketch):**
```
topics(topic_name, num_partitions, replication_factor)
partitions(topic_name, partition_id, leader_broker, replica_brokers[])
segments(topic_name, partition_id, segment_file, base_offset, size_bytes)
consumer_offsets(consumer_group, topic_name, partition_id, committed_offset)
```

**Scaling & Reliability:**
- Throughput scales by adding partitions (more parallel writers/readers) and brokers.
- A stuck or slow consumer in one group never blocks other groups (independent offsets) or other
  partitions (independent, parallel consumption).
- Segment-based storage makes retention and compaction cheap — dropping or compacting whole segment
  files, not scanning and rewriting the entire log.
- Partition count is chosen up front with future scale in mind — increasing it later reshuffles which
  keys map to which partition, breaking existing ordering guarantees for those keys.

**Follow-Up Questions:**
1. How do you get exactly-once processing on top of at-least-once delivery? → make consumers idempotent
   (dedupe by a message id) or tie the consumer's offset commit to its output write in one transaction.
2. A single hot partition (skewed key) becomes a throughput bottleneck → improve the partition key
   choice, or shard the hot key further (e.g., salt it) at the cost of losing strict ordering for that key.
3. Frequent consumer group membership churn causes rebalancing storms → use a cooperative/incremental
   rebalancing protocol instead of a stop-the-world rebalance on every join/leave.

---

## 11. System Design — URL Shortener

**Problem Statement:**
A widely-reported, classic Google system design question. Design a service like bit.ly: given a long
URL, generate a short, unique alias that redirects to it, at very high read (redirect) volume relative to
writes (shortens).

**Functional Requirements:**
- `shorten(long_url) -> short_code` — generate a short, unique code for a URL.
- `redirect(short_code) -> long_url` — resolve a short code back to its original URL with a fast HTTP
  redirect.
- Optional: custom aliases, expiration, click analytics.

**Non-Functional Requirements:**
- Read-heavy: redirects vastly outnumber shortens (often 100:1 or more) — the read path must be very low
  latency and horizontally scalable.
- Short codes must be effectively unique with no coordination bottleneck at write time.
- High availability for redirects — a shortener outage breaks every link using it across the web.

**High-Level Design:**
1. **Code generation**: two common approaches — (a) hash the long URL (e.g., base62-encode a slice of an
   MD5/SHA hash) and handle collisions with a retry/salt, or (b) a centralized counter (or pre-allocated
   ranges of counter values handed out to stateless app servers) base62-encoded into a code, guaranteeing
   uniqueness without a global lock on every request. Range-based counter allocation is the standard
   answer for avoiding a single point of write contention.
2. **Write path**: `shorten` allocates/derives a code, writes `(short_code -> long_url)` to the primary
   datastore, and returns the code.
3. **Read path**: `redirect` looks up `short_code` — this is almost entirely a cache-friendly key-value
   lookup, so put a cache (e.g., Redis) in front of the datastore; a very large fraction of redirect
   traffic hits a small fraction of popular links (power-law distribution), so cache hit rates are high.
4. **Storage**: a simple key-value store or a sharded relational table keyed by `short_code` is sufficient
   — this workload doesn't need complex joins or transactions.
5. **Redirect mechanic**: use a 301 (permanent) redirect if analytics on each click aren't needed (browsers
   cache it, reducing load further) or a 302 (temporary) redirect if you need every click to hit your
   service for analytics — call out this tradeoff explicitly.

**Data Model (sketch):**
```
urls(short_code PK, long_url, created_at, expires_at, owner_id)
click_events(short_code, ts, referrer, user_agent)   # optional, for analytics; write-heavy, separate store
```

**Scaling & Reliability:**
- Shard the datastore by `short_code` (e.g., consistent hashing) once it exceeds one machine.
- Cache aggressively on the read path; a cache miss falls through to the sharded datastore.
- Pre-allocate counter ranges per app server (e.g., server claims codes 1,000,000-1,999,999) to avoid a
  shared counter becoming a write bottleneck, at the cost of codes not being generated in strict global
  order (acceptable, since order doesn't matter here).
- Replicate the datastore for read availability; a brief window of eventual consistency on a
  just-created short code is usually acceptable (the creator can tolerate a moment's lag before their own
  link resolves, but that's a real tradeoff to state explicitly rather than assume away).

**Follow-Up Questions:**
1. How do you prevent short-code collisions with the hash-based approach? → check-and-retry with a
   different salt/slice on collision, or just use the counter-based approach, which structurally can't
   collide.
2. Support custom user-chosen aliases → same datastore, just skip code generation when a custom alias is
   supplied and validate uniqueness against the existing table before insert.
3. Add click analytics without slowing down the redirect path → fire-and-forget the click event onto a
   queue (Kafka) from the redirect handler rather than writing synchronously; a separate consumer
   aggregates analytics asynchronously.

---

## 12. System Design — Distributed Cache (Redis-Style)

**Problem Statement:**
A widely-reported Google system design question. Design a distributed, in-memory key-value cache (think:
a simplified Redis/Memcached) that scales beyond one machine's memory and supports the standard
`get`/`set`/`delete` operations plus TTL-based expiration.

**Functional Requirements:**
- `get(key)`, `set(key, value, ttl=None)`, `delete(key)`.
- Keys automatically expire after their TTL.
- Scale to a keyspace and read/write volume far beyond one machine.

**Non-Functional Requirements:**
- Very low latency (sub-millisecond) reads and writes — this is a cache, not a system of record.
- High throughput per node, since the whole point of a cache is absorbing load that would otherwise hit a
  slower backing store.
- Tunable consistency/durability — a cache is allowed to lose data on a crash (it's re-derivable from the
  backing store), which is exactly what makes it able to trade durability for speed.

**High-Level Design:**
1. **Sharding**: partition the keyspace across many cache nodes via consistent hashing (with virtual
   nodes for load smoothing) — a client or a thin routing layer hashes the key to find its owning node,
   giving `O(1)` routing without a centralized lookup on the hot path.
2. **Per-node storage**: an in-memory hashmap per node for `O(1)` get/set; TTL expiration via a
   combination of lazy expiration (check the timestamp on read, evict if expired) and a periodic active
   sweep (avoid unbounded memory growth from keys nobody ever reads again).
3. **Eviction policy**: when a node approaches its memory limit even after TTL expiration, evict under a
   policy like LRU (approximate LRU via sampling is what real systems like Redis use, since exact LRU
   bookkeeping on every access is itself overhead).
4. **Replication (optional, for availability)**: each shard can have a replica; a cache miss due to a
   crashed primary can be tolerated (it just falls through to the backing store and repopulates the
   cache) — this is a deliberate contrast with a system-of-record, where losing the replica would be
   unacceptable.
5. **Client-side routing vs. proxy layer**: either the client library embeds the consistent-hash ring and
   talks to nodes directly (lower latency, more client complexity), or a thin proxy layer does the routing
   (simpler clients, one more network hop) — state this as an explicit tradeoff.

**Data Model (sketch):**
```
# per-node, in-memory only:
store: key -> (value, expires_at)
# cluster metadata (small, replicated everywhere or in a coordination service):
ring: hash_position -> node_id
```

**Scaling & Reliability:**
- Adding/removing a node only remaps `~1/N` of keys thanks to consistent hashing — the rest of the
  cluster's cached data stays valid.
- A cache miss is never a correctness failure — the caller falls through to the backing store and
  repopulates the cache — which is what allows the cache tier to trade durability for raw speed.
- Hot-key skew (one key vastly more popular than others) can overwhelm a single shard despite good
  overall key distribution — mitigate with client-side local caching of the hottest keys, or by
  replicating just that key across multiple nodes.

**Follow-Up Questions:**
1. How do you handle a "thundering herd" when a very hot key expires and many clients simultaneously miss
   and hit the backing store at once? → have the first miss take a lock/lease to repopulate while other
   concurrent requests wait briefly or serve a slightly-stale value, rather than all of them hitting the
   backing store simultaneously.
2. Strong vs. eventual consistency across replicas → most cache use cases accept eventual consistency
   (a replica might briefly serve a stale value after a write) in exchange for lower write latency; call
   out when that's unacceptable (e.g., a cache used for rate-limiting counters, where staleness causes
   incorrect limit enforcement) and what you'd change (synchronous replication, at a latency cost).
3. Cache stampede on cold-start (cluster restarts with empty caches) → gradual cache warming, or a
   temporary higher rate limit tolerance on the backing store immediately after a cluster restart.

---

## 13. Behavioral Themes

Google's behavioral round is largely a standardized assessment rather than a single freeform interview —
see [`behavioral_interview.md`](./behavioral_interview.md) for general STAR-method prep. Themes specific
to Google's loop:

- **"Googleyness" and general work style**: expect a mix of short-answer and multiple-choice-style
  behavioral questions covering collaboration style and comfort with ambiguity, rather than one deep
  freeform conversation.
- **Ambiguity tolerance**: a time you made progress on a project with unclear or incomplete requirements,
  and how you resolved the ambiguity rather than stalling on it.
- **Disagreement and response**: a time you disagreed with a decision — what you did about it, and what
  the outcome was, regardless of whether you ultimately prevailed.
- **Failure and learning**: a genuine failure (not a thinly-disguised success story) and what specifically
  you changed afterward.
- **Self-driven execution**: a project you drove with minimal outside guidance, showing you can operate
  without heavy process or oversight.

---

## References

Sources used for compiling these questions:
- [Google Interview Questions - 1point3acres](https://www.1point3acres.com/interview/problems/company/google)
- [Google Software Engineer Interview Questions - Glassdoor](https://www.glassdoor.com/Interview/Google-Software-Engineer-Interview-Questions-EI_IE9079.0,6_KO7,24.htm)
- [Google Interview questions and preparation - LeetCode Discuss](https://leetcode.com/discuss/interview-question/5547675/Google-Interview-questions-and-preparation/)
- [Most asked System Design questions - LeetCode Discuss](https://leetcode.com/discuss/interview-question/5806013/Most-asked-System-Design-questions/)

Note: 1point3acres requires forum membership to view full question text/discussion threads — as of this
writing its public Google page shows only the Google Hiring Assessment (GHA) fully, plus generic tag
counts (487 total questions, ~66 coding across easy/medium/hard, 4 system design, mostly gated). Problems
#1-4 and #9-10 are reconstructed from those paywalled tag categories into complete, standard, solvable
versions (not verbatim transcripts). Problems #5-8 and #11-12 are real, independently-corroborated
questions pulled from public sources (Glassdoor, LeetCode company-tagged discussions) rather than from
1point3acres, since its own bank was too gated to expand further on its own.
