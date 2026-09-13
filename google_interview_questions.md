# Google Interview Questions

---

## Contents

**Coding**
1. [Minimum Meeting Rooms (Interval Scheduling)](#1-minimum-meeting-rooms-interval-scheduling)
2. [Accounts Merge (Union-Find)](#2-accounts-merge-union-find)
3. [Top-K Frequent Items (Streaming)](#3-top-k-frequent-items-streaming)
4. [Word Dictionary with Wildcard Search (Trie + DFS)](#4-word-dictionary-with-wildcard-search-trie--dfs)

**System Design**
5. [Retrieval-Augmented Generation (RAG)](#5-system-design--retrieval-augmented-generation-rag)
6. [Distributed Message Queue (Kafka-Style)](#6-system-design--distributed-message-queue-kafka-style)

**Behavioral**
7. [Behavioral Themes](#7-behavioral-themes)

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

## 5. System Design — Retrieval-Augmented Generation (RAG)

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

## 6. System Design — Distributed Message Queue (Kafka-Style)

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

## 7. Behavioral Themes

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

Note: the source page requires forum membership to view full question text/discussion threads; the
problems above were reconstructed from the publicly visible tags/categories into complete, standard,
solvable problem statements with original test cases and solutions — they are not verbatim transcripts
of the reported questions.
