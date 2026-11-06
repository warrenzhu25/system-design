# Anthropic Interview Questions

Source: https://www.1point3acres.com/interview/problems/company/anthropic

> **Note:** Full problem details require login to 1point3acres.com. This document contains publicly available summaries and structure, expanded with full solutions where a complete, verifiable problem statement is publicly visible.

---

## Table of Contents

### System Design
1. [Design a 1-to-1 Chat System](#1-design-a-1-to-1-chat-system)
2. [Inference API System Design](#8-inference-api-system-design)
3. [Prompt Playground System Design](#9-prompt-playground-system-design)
4. [Distributed Model Deployment System Design](#12-distributed-model-deployment-system-design)
5. [LLM Request Batching API System Design](#17-llm-request-batching-api-system-design)
6. [Real-Time Chat Architecture (Streaming/Token Fan-Out)](#26-real-time-chat-architecture-streamingtoken-fan-out)
7. [Large-File Distribution & Bandwidth Optimization](#27-large-file-distribution--bandwidth-optimization)

### Online Assessments (OA)
8. [Task Management System](#2-task-management-system-online-assessment)
9. [Banking System](#3-banking-system-online-assessment)
10. [Cloud Storage System](#4-cloud-storage-system-online-assessment)
11. [Employee Management System](#7-employee-management-system-online-assessment)
12. [Recipe Manager](#10-recipe-manager-online-assessment)
13. [In-memory Database](#20-in-memory-database-online-assessment)

### Coding Problems
14. [Web Crawler](#5-web-crawler)
15. [LRU Cache (Python)](#11-lru-cache-python)
16. [Deduplicate Files](#13-deduplicate-files)
17. [Batch Image Processor](#15-batch-image-processor)
18. [Tokenize (Python)](#16-tokenize-python)
19. [Converting Stack Samples to Trace Events](#18-converting-stack-samples-to-trace-events)
20. [Distributed Mode and Median](#19-distributed-mode-and-median)
21. [LLM Agent Tool-Use Loop](#21-llm-agent-tool-use-loop)
22. [String Processing & Stack: Escaped Bracket Segments](#22-string-processing--stack-escaped-bracket-segments)
23. [Memoize Decorator with LRU Eviction and Disk Persistence](#23-memoize-decorator-with-lru-eviction-and-disk-persistence)
24. [Distributed Map-Reduce Simulation](#24-distributed-map-reduce-simulation)
25. [Tokenizer with Unknown Merging (Trie-Based Greedy Matching)](#25-tokenizer-with-unknown-merging-trie-based-greedy-matching)

### Coding Problems — November 2026 Refresh
28. [Bootloader Instruction Simulator: Fix the Loop](#28-bootloader-instruction-simulator-fix-the-loop-low)
29. [DNS Resolver: Aliases, Fallback, Cycles, Caching](#29-dns-resolver-aliases-fallback-cycles-caching-low-fellows-oa-round-1)
30. [Weighted Data Batcher with Checkpointing](#30-weighted-data-batcher-with-checkpointing-research-track-medium)
31. [In-Memory Key-Value Store with Nested Transactions and Value Counts](#31-in-memory-key-value-store-with-nested-transactions-and-value-counts-medium)
32. [Simulate a Fair Coin from a Biased Coin](#32-simulate-a-fair-coin-from-a-biased-coin-medium)
33. [Hashtag Autocomplete](#33-hashtag-autocomplete-medium)
34. [Onsite Design Doc Review](#34-onsite-design-doc-review-newer-round-high)

### Behavioral
26. [Culture & Behavioral Interview Questions](#6-culture--behavioral-interview-questions)
27. [Hiring Manager Interview Questions](#14-hiring-manager-interview-questions)

---

## 1. Design a 1-to-1 Chat System

**Type:** System Design

**Problem:** Design a chat system that supports only 1-to-1 messaging between users.

**Structure (5 Phases):**
- Phase 1: Define the Goals (~5 minutes)
- Phase 2: Database Schema & Entities (~5 minutes)
- Phase 3: How Client and Server Talk (~5 minutes)
- Phase 4: System Architecture (~15-25 minutes)
- Phase 5: Handling Scale & Challenges (~15-20 minutes)

**Key Areas:**
- Functional and non-functional requirements
- Data model design
- Real-time communication protocols
- Scalability considerations
- Edge cases and challenges

---

## 2. Task Management System (Online Assessment)

**Type:** Online Assessment (OA)

**Problem:** Implement a task management system.

**Levels:**
- Level 1: Basics - Foundational task management features
- Level 2: Search & Sort - Query and organization capabilities
- Level 3: Users & Assignments - Multi-user functionality
- Level 4: Completion & History - Tracking and audit trails

---

## 3. Banking System (Online Assessment)

**Type:** Online Assessment (OA)

**Problem:** Implement an in-memory banking system.

**Levels:**
- Level 1: Basic Actions
- Level 2: Ranking Spenders
- Level 3: Payments and Cashback
- Level 4: Merging and History

**Additional Sections:**
- Special Rules & Edge Cases
- System Constraints

---

## 4. Cloud Storage System (Online Assessment)

**Type:** Online Assessment (OA)

**Problem:** Implement a simple cloud storage system that maps objects (files) to their metainformation.

**Parts:**
- Part 1: Basic File Management - Core file operations
- Part 2: Searching for Files - Query and retrieval functionality
- Part 3: Managing Users and Storage Limits - Multi-user support with quotas
- Part 4: Compressing Files - File compression features

**Additional Sections:**
- Problem Summary
- Data Format specifications

---

## 5. Web Crawler

**Type:** Coding Problem

**Problem:** Given a starting URL `startUrl` and an interface `HtmlParser` that can fetch all URLs from a given web page (or a plain `fetch(url) -> List[str]` function returning outbound links), implement a web crawler that returns all URLs reachable from the starting point that share the same hostname, using up to `N` worker threads. Avoid re-fetching a URL twice, and avoid races/deadlock when multiple threads discover the same new URL simultaneously.

**Approach:** BFS/graph traversal with a shared thread-safe "visited" set and a bounded worker pool. The classic trap is a race between "check if visited" and "mark as visited" — two threads can both see a URL as unvisited and double-fetch it. Guard that check-and-set with a lock (or use a concurrent set whose `add` atomically reports whether the item was new).

```python
import threading
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from urllib.parse import urlparse

class ConcurrentCrawler:
    def __init__(self, fetch, num_workers=8):
        self.fetch = fetch
        self.num_workers = num_workers
        self.visited = set()
        self.lock = threading.Lock()
        self.result = []

    def _same_domain(self, seed, url):
        return urlparse(seed).netloc == urlparse(url).netloc

    def _mark_if_new(self, url):
        with self.lock:
            if url in self.visited:
                return False
            self.visited.add(url)
            return True

    def crawl(self, seed_url):
        with ThreadPoolExecutor(max_workers=self.num_workers) as pool:
            pending = set()

            def submit(url):
                pending.add(pool.submit(self._process, url, seed_url))

            if self._mark_if_new(seed_url):
                submit(seed_url)

            while pending:
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                for fut in done:
                    for new_url in fut.result():
                        if self._same_domain(seed_url, new_url) and self._mark_if_new(new_url):
                            submit(new_url)
        return self.result

    def _process(self, url, seed_url):
        links = self.fetch(url)
        with self.lock:
            self.result.append(url)
        return links
```

**Complexity:** O(V+E) fetches/work total; wall-clock roughly `(V / num_workers) * fetch_latency` under even load.

**Follow-up (System Design) Questions:**
1. Why does the lock only need to guard the visited-set mutation, not the fetch itself? Because the fetch is the I/O-bound work concurrency is meant to parallelize — locking around it would serialize the crawl and defeat the point of threading.
2. How would you add a max-depth limit or a per-domain rate limit? Track `(url, depth)` pairs instead of bare URLs, and add a token-bucket/sleep gate per hostname before dispatching a fetch.
3. How does this generalize to a distributed crawler across multiple machines? Replace the in-process `ThreadPoolExecutor` + in-memory `visited` set with a shared work queue (Kafka/Redis/SQS) and a distributed set (Redis `SADD` or a Bloom filter for approximate dedup at very large scale).

---

## 6. Culture & Behavioral Interview Questions

**Type:** Behavioral Interview

**Focus:** Anthropic places heavy emphasis on culture fit, AI safety values, and how candidates think about ethical considerations. This is often split across two distinct rounds:

- **AI-Safety Screening** (~30 min, very high frequency): assesses whether your values around AI development are compatible with Anthropic's mission before investing further interview time. Expect direct questions like "why do you want to work on AI safety specifically" and "tell me about a time you raised a concern about a system's risk or correctness even though it slowed things down." A genuine, specific answer tied to your own trajectory reads far better than a generic "I think AI is important."
- **Culture & AI-Safety Values** (~55 min, very high frequency): probes critical thinking about AI risk/safety trade-offs rather than recitation of talking points — expect scenario questions ("how would you balance shipping a feature quickly against a safety concern you're not 100% sure about") rather than pure definitional ones. Reasoning out loud through the ambiguity matters more than landing on a single "correct" answer.

**Topic Areas:**
1. Introduction
2. Views on AI Safety and Company Mission — see "AI-Safety Screening" above
3. Handling Feedback and Disagreements
4. Standing Up for Your Beliefs — see "Culture & AI-Safety Values" above
5. Interest in Anthropic
6. Life Goals and Personal Changes
7. Final Advice

---

## 7. Employee Management System (Online Assessment)

**Type:** Online Assessment (OA)

**Problem:** Implement a simplified employee management system.

**Parts:**
- Part 1: Basic Features
- Part 2: Tracking Time
- Part 3: Raises and Pay Checks
- Part 4: Bonus Pay Periods

---

## 8. Inference API System Design

**Type:** System Design

**Problem:** Design a high-concurrency inference API system that can handle massive concurrent requests efficiently.

**Structure:**
- Step 1: Defining the Scope
- Step 2: Estimating Scale and Capacity
- Step 3: Designing the API
- Step 4: Database and Data Structure
- Step 5: System Overview
- Step 6: Deep Dive into Key Components
- Step 7: Finding and Fixing Weak Spots

**Additional Sections:**
- Problem Requirements
- Sample Solution
- Extra Discussion Points
- Mistakes to Avoid
- How to Pass the Interview
- Practice Questions
- Study Materials

---

## 9. Prompt Playground System Design

**Type:** System Design

**Problem:** Design a prompt engineering playground similar to ChatGPT Playground or Anthropic Console.

**Structure:**
- Step 1: Defining the Requirements
- Step 2: Estimating Scale and Costs
- Step 3: API Definition
- Step 4: Database Schema Design
- Step 5: Architecture Overview
- Step 6: Deep Dive into Key Components
- Step 7: Fixing Performance Issues

**Additional Sections:**
- The Design Problem
- Additional Design Details
- Comparing Different Approaches
- Interview Advice

---

## 10. Recipe Manager (Online Assessment)

**Type:** Online Assessment (OA)

**Problem:** Implement an in-memory recipe management system.

**Levels:**
- Level 1: Basic Operations
- Level 2: Finding and Organizing Data
- Level 3: Adding Users
- Level 4: History and Rollbacks

**Additional Sections:**
- Critical Rules
- System Limitations
- Code Solution

---

## 11. LRU Cache (Python)

**Type:** Coding Problem

**Problem:** You are given an existing in-memory LRU (Least Recently Used) cache implementation in Python.

**Parts:**
- The Problem - Understanding the existing implementation
- Part 2: Saving Data to Disk
- Extra Questions

---

## 12. Distributed Model Deployment System Design

**Type:** System Design

**Problem:** Design a system that efficiently downloads and distributes a large ML model (e.g., 500GB) from external storage to all GPU workers in a data center cluster.

**Structure:**
- The Challenge
- Proposed Solution
- Understanding the Requirements
- Mathematical analysis
- API Design
- Data Structure Design
- System Architecture
- Deep Dive analysis
- Bottleneck identification and fixes
- Interview Tips

---

## 13. Deduplicate Files

**Type:** Coding Problem

**Problem:** Given a root folder/directory, find all sets of duplicate files (identical content, filename may differ) and return groups of duplicate paths. Discuss how you'd scale this to millions of files / terabytes of data.

**Faster Solution (two-phase hashing to avoid reading every byte of every file):**
1. Group by file size (cheap `stat` call) — files with a unique size can't have duplicates.
2. Within a size group, hash a small prefix (e.g., first 4KB) to cheaply split further.
3. Only for files matching on size + prefix hash, compute a full cryptographic hash (SHA-256) to confirm.

```python
import os, hashlib
from collections import defaultdict

def _hash_file(path, chunk_size=None):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        if chunk_size:
            h.update(f.read(chunk_size))
        else:
            for block in iter(lambda: f.read(1 << 20), b""):
                h.update(block)
    return h.digest()

def find_duplicates(root):
    by_size = defaultdict(list)
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            path = os.path.join(dirpath, name)
            try:
                size = os.path.getsize(path)
            except OSError:
                continue
            by_size[size].append(path)

    duplicates = []
    for size, paths in by_size.items():
        if len(paths) < 2:
            continue
        by_prefix = defaultdict(list)
        for p in paths:
            by_prefix[_hash_file(p, chunk_size=4096)].append(p)
        for prefix_group in by_prefix.values():
            if len(prefix_group) < 2:
                continue
            by_full_hash = defaultdict(list)
            for p in prefix_group:
                by_full_hash[_hash_file(p)].append(p)
            duplicates.extend(g for g in by_full_hash.values() if len(g) > 1)
    return duplicates
```

**Follow-up Questions:**
1. How would you scale this to millions of files / terabytes of data? Shard the size-grouping across workers (map-reduce style: mapper emits `(size, path)`, reducer groups), store hashes in a distributed KV store instead of an in-memory dict, and stream-hash rather than loading files fully.
2. Why hash a 4KB prefix before the full file? It cheaply eliminates most size-matched non-duplicates without paying the cost of hashing entire large files that will turn out to differ early on.

---

## 14. Hiring Manager Interview Questions

**Type:** Behavioral Interview

**Focus:** Anthropic's hiring manager interview emphasizes project deep dives, technical leadership, collaboration skills, and career alignment. Two of its topic areas map to distinct, commonly-reported rounds:

- **Technical Presentation** (~55 min, very high frequency): can you explain a complex system you built to a technical audience, including honest discussion of what didn't work? Structure it as: problem/motivation → your specific contribution → one concrete technical decision and its trade-off → results with real numbers → what you'd do differently. Expect deep technical follow-up questions on the hardest part, not just the summary.
- **Leadership & Collaboration** (~55 min, very high frequency): how you influence and support others without formal authority. Strong story shapes: mentoring someone through a hard debugging problem, driving consensus across teams on a technical direction, or handling disagreement with a peer/manager on approach. A story where you updated your own view based on someone else's input reads as more mature than one where you simply won the argument.

**Topic Areas:**
1. Summary - Introduction section
2. Technical Experience & Projects — see "Technical Presentation" above
3. Working with Others — see "Leadership & Collaboration" above
4. Leading and Teaching — see "Leadership & Collaboration" above
5. Your Goals and Work Style - Career aspirations and work preferences
6. Key Advice for Success - Interview tips

---

## 15. Batch Image Processor

**Type:** Coding Problem

**Problem:** Given a large list of image file paths and a set of transformations (resize, grayscale, watermark), apply all transformations to all images and write outputs, using multiple processes to parallelize CPU-bound image work. Handle partial failures (a corrupt image shouldn't kill the batch) and report progress.

**Approach:** Image processing (e.g. via Pillow) is CPU-bound and holds the GIL during pixel operations, so use `multiprocessing.Pool` instead of threads. Each worker processes one `(path, transforms)` unit and returns a success/failure result; the parent aggregates results and logs failures without crashing the pool.

```python
from multiprocessing import Pool
from dataclasses import dataclass

@dataclass
class Result:
    path: str
    ok: bool
    error: str = None

def _process_one(args):
    path, transforms = args
    try:
        from PIL import Image
        img = Image.open(path)
        for t in transforms:
            img = t(img)
        out_path = path + ".out.jpg"
        img.save(out_path)
        return Result(path, True)
    except Exception as e:
        return Result(path, False, str(e))

def run_pipeline(paths, transforms, num_workers=8):
    results = []
    with Pool(num_workers) as pool:
        for i, r in enumerate(pool.imap_unordered(_process_one, [(p, transforms) for p in paths]), 1):
            results.append(r)
            if i % 100 == 0:
                print(f"progress: {i}/{len(paths)}")
    failures = [r for r in results if not r.ok]
    return results, failures
```

**Interview Follow-up Questions:**
1. Why processes instead of threads here, versus threads for an I/O-bound crawler? CPU-bound work holds the GIL during pixel operations, so threads wouldn't actually run in parallel; separate processes each get their own interpreter/GIL.
2. Why `imap_unordered` instead of `map`? It yields results as soon as any worker finishes, enabling progress reporting without waiting on the slowest task.
3. How would you cut IPC overhead when images are numerous but small? Pass `chunksize=N` to `pool.imap` so workers pull batches of tasks per round-trip instead of one at a time.

---

## 16. Tokenize (Python)

**Type:** Coding Problem

**Problem:** You are given two functions: `tokenize` and `detokenize`.

**Parts:**
- Part 1: Understanding the Code
- Part 2: Reviewing a Proposed Fix
- Part 3: Writing a Better Solution
- Part 4: Follow-Up Questions

---

## 17. LLM Request Batching API System Design

**Type:** System Design

**Problem:** Design an HTTP API / serving system that batches concurrent user requests to an LLM to maximize GPU throughput while keeping per-request latency acceptable.

**Framework to hit:**
- **Requirements:** target p99 latency (e.g. <500ms time-to-first-token), throughput (tokens/sec/GPU), variable prompt/response lengths.
- **Continuous (in-flight) batching:** static batching wastes GPU cycles waiting for the longest sequence in a batch to finish; continuous batching admits new requests into a running batch as soon as any sequence finishes, keeping GPU utilization high.
- **KV-cache management:** each request's attention KV cache grows with generated tokens and consumes GPU memory — discuss paged KV-cache (vLLM-style) to avoid fragmentation, and an eviction/preemption policy when memory is exhausted (swap to CPU, or abort the lowest-priority request).
- **Scheduling:** priority queues by SLA class, fairness across tenants, and separating prefill (compute-bound, parallelizable across the prompt) from decode (memory-bandwidth-bound, sequential) — some systems run these on separate GPU pools.
- **Autoscaling:** the same bursty-load, cost-of-over/under-provisioning, cold-start-latency problem that applies to any compute fleet applies here — queue-depth-based autoscaling signals tend to react faster than naive CPU/GPU-utilization-based signals.
- **Failure modes:** a single stuck/slow request degrading the whole batch, GPU OOM from a bad batch-size estimate, and backpressure/admission control at the front door.

**Structure:**
- The Challenge (problem introduction)
- Sample Solution (reference implementation)
- Step 1: Clarifying the Requirements
- Step 2: Estimating Scale and Resources
- Step 3: API Design
- Step 4: Data Storage
- Step 5: Basic System Architecture
- Step 6: Deep Dive into Components (see "Framework to hit" above)
- Step 7: Fixing Potential Problems (see "Failure modes" above)

---

## 18. Converting Stack Samples to Trace Events

**Type:** Coding Problem

**Problem:** A sampling profiler periodically records the full call stack at a timestamp.

**Sections:**
- Problem Statement
- Follow-up Question: Reducing Noise
- System Design Discussion

---

## 19. Distributed Mode and Median

**Type:** Coding Problem

**Problem:** A very large dataset is distributed across multiple machines (typically 10 workers). Each machine has a portion of the dataset stored locally. Using pre-built interface functions `send(workerid, data)` and `recv()`, implement distributed algorithms.

**Objectives:**
1. Finding the Mode - Identify the most frequently occurring element across the distributed dataset
2. Finding the Median - Compute the median value across all machines

**Key Constraints:**
- Multiple worker nodes (~10) hold partitioned data
- Must use provided communication primitives (send/recv functions)
- Very large dataset implies efficiency considerations

---

## 20. In-memory Database (Online Assessment)

**Type:** Online Assessment (OA)

**Problem:** Implement a simplified version of an in-memory database.

**Levels:**
- Level 1: Core Features - Foundational database operations
- Level 2: Filtering Data - Query capabilities with conditions
- Level 3: Automatic Expiration (TTL) - Time-to-live functionality for data
- Level 4: Historical Data - Tracking and retrieving past states

---

## 21. LLM Agent Tool-Use Loop

**Type:** Coding Problem
*Tags: llm, agents, tool-use, prompt-engineering · ~55 min · High frequency*

**Problem:** Design and implement a simplified "agent loop": given a user query, a set of callable tools (each with a name, description, and JSON schema for arguments), and an LLM that can be called with `llm(messages) -> {tool_call | final_answer}`, implement the loop that calls tools, feeds results back to the model, and terminates with a final answer or a max-iteration guard.

**Approach:** This is a state machine, not really an ML problem — the coding bar is about clean control flow, error handling per tool call, and termination guarantees.

```python
def run_agent(query, tools_by_name, llm, max_iters=10):
    messages = [{"role": "user", "content": query}]
    for _ in range(max_iters):
        response = llm(messages)
        if response["type"] == "final_answer":
            return response["content"]

        tool_name = response["tool_call"]["name"]
        args = response["tool_call"]["arguments"]
        messages.append({"role": "assistant", "content": None, "tool_call": response["tool_call"]})

        tool = tools_by_name.get(tool_name)
        if tool is None:
            observation = {"error": f"unknown tool {tool_name}"}
        else:
            try:
                observation = {"result": tool.fn(**args)}
            except Exception as e:
                observation = {"error": str(e)}

        messages.append({"role": "tool", "name": tool_name, "content": observation})
    return {"error": "max iterations exceeded"}
```

**Follow-up Questions:**
1. How do you handle malformed tool-call JSON from the model? Wrap the parse in a try/except and feed a structured error back as the "observation" so the model can self-correct, rather than crashing the loop.
2. How do you guard against infinite loops beyond the iteration cap? Detect repeated identical tool calls (same name + args) and break early with an explicit error.
3. What belongs in a tool's schema/docstring vs. the system prompt? Per-tool usage constraints belong in the schema/docstring (so they travel with the tool); cross-tool policy (e.g. "confirm before destructive actions") belongs in the system prompt.
4. Should a tool with side effects (send_email, delete_file) require confirmation? Yes — flag side-effecting tools and require an explicit confirmation step (or a human-in-the-loop gate) before executing them.

---

## 22. String Processing & Stack: Escaped Bracket Segments

**Type:** Coding Problem
*Tags: two-pointer, string-processing, stack · ~55 min · High frequency*

**Problem:** Given a string with nested brackets and escape characters, e.g. a mini templating language like `"a(b(c)d)e\\(f"`, implement a function that validates balanced parentheses (respecting `\(` / `\)` as literal, non-structural characters) and returns the top-level segments (content between matching outer parens, plus text outside any parens).

**Approach:** Single pass with a stack tracking open-paren positions, and a one-character lookahead to detect an escaping backslash.

```python
def parse_segments(s):
    stack = []
    segments = []
    i = 0
    n = len(s)
    buf = []
    while i < n:
        c = s[i]
        if c == '\\' and i + 1 < n and s[i+1] in '()':
            buf.append(s[i+1])
            i += 2
            continue
        if c == '(':
            if not stack:
                if buf:
                    segments.append(''.join(buf)); buf = []
            else:
                buf.append(c)
            stack.append(i)
            i += 1
            continue
        if c == ')':
            if not stack:
                raise ValueError(f"unmatched ')' at index {i}")
            stack.pop()
            if not stack:
                segments.append(''.join(buf)); buf = []
            else:
                buf.append(c)
            i += 1
            continue
        buf.append(c)
        i += 1
    if stack:
        raise ValueError("unmatched '(' remaining")
    if buf:
        segments.append(''.join(buf))
    return segments
```

**Complexity:** O(n) time, O(n) space worst case for the stack (deeply nested input).

**Follow-up Questions:**
1. What are the key edge cases? Empty string, unmatched closing paren, trailing backslash, and an escaped-but-not-a-paren backslash (`\\n`).

---

## 23. Memoize Decorator with LRU Eviction and Disk Persistence

**Type:** Coding Problem
*Tags: caching, persistence, kwargs · ~55 min · High frequency*

**Problem:** Implement a `@memoize` decorator that caches a function's return value keyed on its arguments (including keyword arguments), supports an LRU eviction policy with a max size, and can optionally persist the cache to disk so it survives process restarts.

**Approach:** Key generation must canonicalize `*args`/`**kwargs` (sorted kwargs) so that call-order of kwargs doesn't create spurious cache misses; use `functools.lru_cache`-style semantics implemented over `collections.OrderedDict` so persistence stays under your control; serialize with `json` on every write (simple but slower) or periodically (faster, small durability window).

```python
import functools, json, os
from collections import OrderedDict

def memoize(maxsize=128, persist_path=None):
    def decorator(fn):
        cache = OrderedDict()
        if persist_path and os.path.exists(persist_path):
            with open(persist_path) as f:
                cache.update(json.load(f))

        def make_key(args, kwargs):
            return json.dumps([args, sorted(kwargs.items())], sort_keys=True)

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            key = make_key(args, kwargs)
            if key in cache:
                cache.move_to_end(key)
                return cache[key]
            result = fn(*args, **kwargs)
            cache[key] = result
            cache.move_to_end(key)
            if len(cache) > maxsize:
                cache.popitem(last=False)
            if persist_path:
                with open(persist_path, "w") as f:
                    json.dump(dict(cache), f)
            return result
        wrapper.cache = cache
        return wrapper
    return decorator
```

**Follow-up Questions:**
1. What happens with unhashable/non-JSON-serializable arguments? JSON-based keys break on arbitrary objects — fall back to `repr()` for the key, or require args be JSON-safe and document the constraint.
2. Why is write-on-every-call persistence a concern? It's simple but slow under high call volume; a production system would batch/flush periodically or use a write-ahead log, trading a small durability window for throughput.

---

## 24. Distributed Map-Reduce Simulation

**Type:** Coding Problem
*Tags: distributed-systems, map-reduce, bandwidth · ~55 min · Medium frequency*

**Problem:** Implement a simplified single-machine simulation of a map-reduce framework: given `N` input splits, a `map_fn`, and a `reduce_fn`, partition mapper output into `R` reduce buckets (by hash of key), run reducers, and return the final key→value results. Then discuss how you'd handle a mapper or reducer failure in a real distributed deployment, and how you'd minimize network shuffle bandwidth.

```python
from collections import defaultdict

def map_reduce(splits, map_fn, reduce_fn, num_reducers):
    # Map phase: each split emits (key, value) pairs
    partitions = [defaultdict(list) for _ in range(num_reducers)]
    for split in splits:
        for k, v in map_fn(split):
            bucket = hash(k) % num_reducers
            partitions[bucket][k].append(v)

    # Shuffle is implicit here (single machine); in reality this is
    # where each reducer pulls its partition from every mapper's local disk.

    # Reduce phase
    output = {}
    for partition in partitions:
        for k, values in partition.items():
            output[k] = reduce_fn(k, values)
    return output
```

**Follow-up Questions:**
1. How do you handle a mapper or reducer failure in a real distributed deployment? A mapper failure re-runs that map task (safe since input splits are immutable); a reducer failure re-runs the reduce task once its input partitions are re-materialized. Contrast this with lineage-based recomputation (recompute only the lost partition from its dependency graph) versus classic MapReduce's write-everything-to-disk-between-stages approach.
2. How do you minimize network shuffle bandwidth? Combiners (local pre-aggregation before shuffle — the same idea as a map-side combine for a `reduceByKey`-style operation versus a plain `groupByKey`), a partitioning strategy that co-locates related keys, compression of shuffle data, and speculative execution for stragglers.

---

## 25. Tokenizer with Unknown Merging (Trie-Based Greedy Matching)

**Type:** Coding Problem
*Tags: string-processing, greedy, trie · ~55 min · Medium frequency*

**Problem:** Implement a simplified BPE-style tokenizer: given a vocabulary of known tokens (strings) and an input string, greedily tokenize left-to-right by always matching the longest known vocabulary token at the current position; any character with no matching token becomes an `<UNK>` token. Optimize for repeated queries against a large, fixed vocabulary.

**Approach:** Build a trie (prefix tree) over the vocabulary once; for each position in the input, walk the trie to find the longest matching prefix (greedy longest-match), consuming that many characters; if no match, emit `<UNK>` and advance by one character.

```python
class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_token = False

class Tokenizer:
    def __init__(self, vocab):
        self.root = TrieNode()
        for token in vocab:
            node = self.root
            for ch in token:
                node = node.children.setdefault(ch, TrieNode())
            node.is_token = True

    def tokenize(self, text):
        tokens = []
        i, n = 0, len(text)
        while i < n:
            node = self.root
            j = i
            last_match_end = None
            while j < n and text[j] in node.children:
                node = node.children[text[j]]
                j += 1
                if node.is_token:
                    last_match_end = j
            if last_match_end is not None:
                tokens.append(text[i:last_match_end])
                i = last_match_end
            else:
                tokens.append("<UNK>")
                i += 1
        return tokens
```

**Complexity:** O(V·L) to build the trie (V vocab entries, average length L), O(n·L_max) to tokenize a string of length n.

**Follow-up Questions:**
1. Is greedy longest-match always optimal? No — real BPE actually merges by learned frequency-rank, not greedy-longest; be ready to name that distinction if asked.
2. How would you serve this at scale? Shard/cache the trie across a service fleet rather than rebuilding it per request, since the vocabulary is fixed and large.

---

## 26. Real-Time Chat Architecture (Streaming/Token Fan-Out)

**Type:** System Design
*Tags: chat, websocket, kafka, redis · High frequency*

**Problem:** Design a real-time chat backend supporting streaming token-by-token responses to the client, message history persistence, and horizontal scaling across many concurrent sessions.

**Framework to hit:**
- **Client connection:** WebSocket (or SSE) per active session for token streaming. SSE is often preferable for one-directional server→client streaming (simpler infra, works through more proxies) versus WebSocket's bidirectionality.
- **Fan-out/session routing:** with many stateless API servers behind a load balancer, a client's connection can land on any server, but the inference worker producing tokens for that session is a specific backend — use Redis pub/sub or Kafka to route generated tokens from the inference worker back to the correct connection-holding server.
- **Persistence:** message history in a database with append-only log semantics (avoid update-in-place for messages), with the ability to reconstruct a conversation on client reconnect (durable topic per session, or a DB read).
- **Backpressure/reconnect:** if the client disconnects mid-stream, buffer generated tokens for a grace period and resume, rather than dropping generation work.
- **Scaling numbers:** estimate concurrent connections per server (WebSocket memory footprint), and where the pub/sub layer becomes the bottleneck at scale (partition count, consumer group rebalancing).

---

## 27. Large-File Distribution & Bandwidth Optimization

**Type:** System Design
*Tags: distribution, bandwidth, tree-broadcast, bittorrent · High frequency*

**Problem:** Design a system to distribute a large file (e.g., a multi-GB model checkpoint) from one source to thousands of machines as fast as possible, minimizing total network bandwidth and avoiding a bottleneck at the source.

**Framework to hit:**
- **Naive baseline:** source pushes to every node directly — O(N) load on the source's uplink, which clearly won't scale to thousands of nodes.
- **Tree broadcast:** organize nodes into a broadcast tree; each node re-serves the data to its children once received, turning source load into O(log N) hops with O(1) fan-out load per node. There's a fan-out-factor tradeoff (wider tree = fewer hops but more load per node).
- **BitTorrent-style swarm:** split the file into chunks; nodes exchange different chunks with each other (not just from the source), so aggregate bandwidth scales with the number of peers rather than just the source's uplink. This beats a single tree at very large N, at the cost of coordination complexity (tracker/piece-selection logic, e.g. rarest-piece-first). This pattern is why some distributed compute frameworks added torrent-based broadcast specifically for distributing large binaries to many workers without saturating a central node.
- **Failure handling:** node dropout mid-transfer (peers reroute to other sources of that chunk), corruption detection (checksum per chunk), and stragglers (timeout + refetch from a different peer).

---

## Additions — November 2026 Refresh

Source: the Anthropic bank on 1point3acres (35 questions: 21 coding, 7 system design, 4 behavioral, 3
other), read on 2026-11-06. The page shows only a summary card per question; full text is member-only. Each
problem below is restated from its card, so **clarify requirements with the interviewer** — signatures and
follow-ups are my reconstruction. Every Python block runs and passes its own asserts.

### 28. Bootloader Instruction Simulator: Fix the Loop (low)

Card: a list of `plus` / `next` / `jump` instructions mutates an accumulator and program counter; detect the
loop, then identify or repair the single bad instruction that makes execution cycle.

Assumed instruction set: `plus n` adds `n` to the accumulator and moves to the next line; `next` does
nothing and moves to the next line (a no-op that still carries an argument, `next n`); `jump n` moves by
`n`. The program halts when the counter walks off the end. Part 1: run until an instruction repeats and
return the accumulator. Part 2: swap exactly one `jump` ↔ `next` so that the program halts.

```python
def parse(lines):
    program = []
    for line in lines:
        op, arg = line.split()
        program.append((op, int(arg)))
    return program


def run(program):
    """Returns (halted, accumulator). halted is False when an instruction repeats."""
    acc, pc, seen = 0, 0, set()
    while pc < len(program):
        if pc in seen:
            return False, acc
        seen.add(pc)
        op, arg = program[pc]
        if op == "plus":
            acc += arg
            pc += 1
        elif op == "jump":
            pc += arg
        else:                       # next
            pc += 1
    return True, acc


def fix_loop(program):
    """Try flipping each jump/next; O(n^2) worst case, fine for this size."""
    swap = {"jump": "next", "next": "jump"}
    for i, (op, arg) in enumerate(program):
        if op not in swap:
            continue
        patched = program[:i] + [(swap[op], arg)] + program[i + 1:]
        halted, acc = run(patched)
        if halted:
            return i, acc
    return None


code = parse([
    "next 0",
    "plus 1",
    "jump 4",
    "plus 3",
    "jump -3",
    "plus -99",
    "plus 1",
    "jump -4",
    "plus 6",
])
assert run(code) == (False, 5)                 # loops; accumulator is 5 when it repeats
assert fix_loop(code) == (7, 8)                # flipping line 7 lets it fall off the end
assert run(parse(["plus 2", "plus 3"])) == (True, 5)
assert fix_loop(parse(["jump 0"])) == (0, 0)   # self-loop fixed by turning it into a next
```

**Optimisation to mention:** instead of re-simulating per candidate, mark all lines that can reach the end
by running the graph backwards (reverse edges, BFS from "end"), then only flip an instruction whose changed
target is in that set — O(n). Follow-up: what if more than one instruction is wrong? (Then it is a
shortest-path problem over "number of edits used".)

### 29. DNS Resolver: Aliases, Fallback, Cycles, Caching (low, Fellows OA round 1)

Card: simplified DNS resolver built step by step against a unit-test harness; later steps add aliasing,
fallback, error handling, cycle detection, caching and concurrency.

Model: `records` maps a name to `("A", ip)` or `("CNAME", other_name)`. Resolve follows aliases up to a hop
limit, detects cycles, caches results with a TTL, and falls back to a secondary resolver on a miss.

```python
import threading


class ResolveError(Exception):
    pass


class Resolver:
    def __init__(self, records, clock, fallback=None, ttl=60, max_hops=8):
        self.records = records
        self.clock = clock
        self.fallback = fallback            # another Resolver-like object, or None
        self.ttl = ttl
        self.max_hops = max_hops
        self._cache = {}                    # name -> (ip, expires_at)
        self._lock = threading.Lock()

    def resolve(self, name):
        now = self.clock()
        with self._lock:
            hit = self._cache.get(name)
            if hit and hit[1] > now:
                return hit[0]
        ip = self._resolve_uncached(name)
        with self._lock:
            self._cache[name] = (ip, now + self.ttl)
        return ip

    def _resolve_uncached(self, name):
        seen = []
        current = name
        for _ in range(self.max_hops + 1):
            if current in seen:
                raise ResolveError("cycle: " + " -> ".join(seen + [current]))
            seen.append(current)
            record = self.records.get(current)
            if record is None:
                if self.fallback is not None:
                    return self.fallback.resolve(current)
                raise ResolveError("NXDOMAIN " + current)
            kind, value = record
            if kind == "A":
                return value
            current = value                 # CNAME: keep walking
        raise ResolveError("too many hops from " + name)


class Clock:
    now = 0

    def __call__(self):
        return self.now


clock = Clock()
upstream = Resolver({"ext.example": ("A", "9.9.9.9")}, clock)
r = Resolver({
    "www": ("CNAME", "web"),
    "web": ("CNAME", "lb"),
    "lb": ("A", "10.0.0.1"),
    "a": ("CNAME", "b"),
    "b": ("CNAME", "a"),
}, clock, fallback=upstream, ttl=30, max_hops=3)

assert r.resolve("www") == "10.0.0.1"                 # two-hop alias chain
assert r.resolve("ext.example") == "9.9.9.9"          # served by the fallback
for bad, text in (("a", "cycle"), ("nope", "NXDOMAIN")):
    try:
        r.resolve(bad)
        raise AssertionError("expected failure for " + bad)
    except ResolveError as error:
        assert text in str(error)

r.records["lb"] = ("A", "10.0.0.2")                   # change the record; cache still serves the old one
assert r.resolve("www") == "10.0.0.1"
clock.now = 31                                         # TTL expired
assert r.resolve("www") == "10.0.0.2"

deep = Resolver({"n0": ("A", "1.1.1.1"), **{f"n{i}": ("CNAME", f"n{i-1}") for i in range(1, 6)}},
                clock, max_hops=3)
try:
    deep.resolve("n5")
    raise AssertionError("expected hop limit")
except ResolveError as error:
    assert "too many hops" in str(error)
```

Concurrency follow-up: the lock guards the cache only; two threads may resolve the same cold name at once
(harmless duplicate work). To coalesce them, keep a per-name in-flight future ("single flight"). Cache negative
answers with a shorter TTL if the interviewer asks about NXDOMAIN storms.

### 30. Weighted Data Batcher with Checkpointing (research track, medium)

Card: a `DataBatcher` samples weighted batches from a `DataRegistry`, supports deterministic save/resume via
an `offset`, and must handle `batch_size` not divisible by the sum of weights.

Make batch composition a pure function of the batch index so resume is just "continue from index `offset`":
within every cycle of `sum(weights)` slots each dataset appears `weight` times, so slot `s` maps to a dataset
by a precomputed schedule. Use largest-remainder allocation for a partial batch.

```python
from itertools import accumulate


class DataBatcher:
    def __init__(self, datasets, weights, batch_size):
        # datasets: {name: list of examples}; weights: {name: int}
        self.names = sorted(datasets)
        self.data = datasets
        self.weights = weights
        self.batch_size = batch_size
        self.cycle = []                               # one full weight cycle of dataset names
        for name in self.names:
            self.cycle += [name] * weights[name]
        # Interleave rather than block: spread each dataset evenly across the cycle.
        self.cycle = self._interleave()
        self.offset = 0                               # number of slots consumed so far

    def _interleave(self):
        slots = []
        for name in self.names:
            w = self.weights[name]
            for k in range(w):
                slots.append(((k + 0.5) / w, name))
        return [name for _, name in sorted(slots)]

    def next_batch(self):
        batch = []
        for _ in range(self.batch_size):
            name = self.cycle[self.offset % len(self.cycle)]
            # Per-dataset position = how many times this dataset's slot appeared before.
            full_cycles, within = divmod(self.offset, len(self.cycle))
            per_cycle = self.weights[name]
            before = sum(1 for n in self.cycle[:within] if n == name)
            index = (full_cycles * per_cycle + before) % len(self.data[name])
            batch.append((name, self.data[name][index]))
            self.offset += 1
        return batch

    def state(self):
        return {"offset": self.offset}

    def load_state(self, state):
        self.offset = state["offset"]


data = {"web": list(range(100)), "code": [f"c{i}" for i in range(50)]}
batcher = DataBatcher(data, {"web": 3, "code": 1}, batch_size=6)   # 6 is not a multiple of 4
first = batcher.next_batch()
names = [n for n, _ in first]
assert names.count("web") + names.count("code") == 6
assert names.count("web") in (4, 5)                  # close to the 3:1 ratio inside a partial cycle

saved = batcher.state()
after = batcher.next_batch()
replay = DataBatcher(data, {"web": 3, "code": 1}, batch_size=6)
replay.load_state(saved)
assert replay.next_batch() == after                  # deterministic resume

long_run = DataBatcher(data, {"web": 3, "code": 1}, batch_size=4)
counts = {"web": 0, "code": 0}
for _ in range(25):
    for n, _ in long_run.next_batch():
        counts[n] += 1
assert counts == {"web": 75, "code": 25}             # exact ratio over whole cycles
```

State to checkpoint: only `offset` (plus the weights and seed if shuffling). If the interviewer wants random
sampling instead of a fixed schedule, derive each slot from `random.Random(seed + offset)` so it is still
reproducible. Edge cases to raise: zero-weight datasets, a dataset smaller than its share (wrap or stop?),
and changing weights mid-run (checkpoint must record the weight version).

### 31. In-Memory Key-Value Store with Nested Transactions and Value Counts (medium)

Card: `set` / `get` / `delete`, nested `begin` / `commit` / `rollback`, and a `count(value)` query.

Keep a base dict and a stack of per-transaction undo logs. Maintain `value → count` incrementally so
`count` is O(1).

```python
from collections import Counter

_MISSING = object()


class TxnStore:
    def __init__(self):
        self.data = {}
        self.counts = Counter()
        self.undo = []                      # stack of {key: previous value or _MISSING}

    def _put(self, key, value):
        if self.undo and key not in self.undo[-1]:
            self.undo[-1][key] = self.data.get(key, _MISSING)
        old = self.data.get(key, _MISSING)
        if old is not _MISSING:
            self.counts[old] -= 1
        if value is _MISSING:
            self.data.pop(key, None)
        else:
            self.data[key] = value
            self.counts[value] += 1

    def set(self, key, value):
        self._put(key, value)

    def get(self, key):
        return self.data.get(key)

    def delete(self, key):
        self._put(key, _MISSING)

    def count(self, value):
        return self.counts[value]

    def begin(self):
        self.undo.append({})

    def rollback(self):
        if not self.undo:
            return False
        log = self.undo.pop()
        for key, previous in log.items():
            # Restore without recording into an outer log: the outer log already has the
            # first-seen value from before the whole stack began for keys it touched.
            old = self.data.get(key, _MISSING)
            if old is not _MISSING:
                self.counts[old] -= 1
            if previous is _MISSING:
                self.data.pop(key, None)
            else:
                self.data[key] = previous
                self.counts[previous] += 1
        return True

    def commit(self):
        if not self.undo:
            return False
        log = self.undo.pop()
        if self.undo:                       # fold into the parent; keep the parent's older value
            for key, previous in log.items():
                self.undo[-1].setdefault(key, previous)
        return True


s = TxnStore()
s.set("a", 10)
s.set("b", 10)
assert s.count(10) == 2
s.begin()
s.set("a", 20)
s.begin()
s.delete("b")
assert s.get("b") is None and s.count(10) == 0
s.rollback()                                # undo the inner delete only
assert s.get("b") == 10 and s.get("a") == 20
s.commit()                                  # fold the outer transaction in
assert s.get("a") == 20 and s.count(20) == 1 and s.count(10) == 1
assert s.rollback() is False                # nothing open

s.begin()
s.set("a", 30)
s.begin()
s.set("a", 40)
s.commit()                                  # inner commit merges into the outer
s.rollback()                                # outer rollback must restore a to 20
assert s.get("a") == 20 and s.count(20) == 1 and s.count(30) == 0 and s.count(40) == 0
```

The subtle bug candidates hit: on an inner `commit`, the parent must keep **its own** older value for a key
(`setdefault`), not overwrite it with the inner one — otherwise the outer rollback restores the wrong state
(the last block above tests exactly that).

### 32. Simulate a Fair Coin from a Biased Coin (medium)

Card (coding-problems list): "Simulate a Fair Coin Using a Biased Coin."

Von Neumann's trick: flip twice; `HT → 0`, `TH → 1`, otherwise retry. Both outcomes have probability
`p(1−p)` whatever `p` is, so the result is exactly fair.

```python
import random


def fair_bit(biased):
    """biased() returns 1 with unknown probability p (0 < p < 1)."""
    while True:
        a, b = biased(), biased()
        if a != b:
            return a                         # (1,0) -> 1, (0,1) -> 0


def make_biased(p, seed):
    rng = random.Random(seed)
    return lambda: 1 if rng.random() < p else 0


coin = make_biased(0.8, seed=7)
ones = sum(fair_bit(coin) for _ in range(20000))
assert abs(ones / 20000 - 0.5) < 0.02        # statistical check, ~7 sigma of slack

# Expected flips per output bit is 1 / (p * (1 - p)): 6.25 for p = 0.8, 4 for p = 0.5.
calls = 0


def counting():
    global calls
    calls += 1
    return coin()


for _ in range(5000):
    fair_bit(counting)
assert 5.0 < calls / 5000 < 7.5
```

Follow-ups: wasted flips (use the pairs `00`/`11` recursively — Peres' extractor — to approach the entropy
bound), a fair die from a fair coin (rejection sampling on `ceil(log2 n)` bits), and proving fairness in one
sentence: `P(HT) = P(TH) = p(1−p)`.

### 33. Hashtag Autocomplete (medium)

Card: "Hashtag Autocomplete" in the coding list. Assumed: given a stream of hashtags with counts, return the
top-k most frequent completions for a prefix, ties alphabetical.

A trie where each node caches its top-k makes queries O(|prefix|); updates re-rank along the path.

```python
class TrieNode:
    __slots__ = ("children", "top")

    def __init__(self):
        self.children = {}
        self.top = []                        # up to k (-count, word), sorted ascending


class Autocomplete:
    def __init__(self, k=3):
        self.k = k
        self.root = TrieNode()
        self.counts = {}

    def add(self, word, times=1):
        self.counts[word] = self.counts.get(word, 0) + times
        entry = (-self.counts[word], word)
        node = self.root
        self._update(node, word, entry)
        for ch in word:
            node = node.children.setdefault(ch, TrieNode())
            self._update(node, word, entry)

    def _update(self, node, word, entry):
        node.top = [e for e in node.top if e[1] != word]
        node.top.append(entry)
        node.top.sort()
        del node.top[self.k:]

    def suggest(self, prefix):
        node = self.root
        for ch in prefix:
            node = node.children.get(ch)
            if node is None:
                return []
        return [word for _, word in node.top]


ac = Autocomplete(k=3)
for tag, n in [("#python", 5), ("#pytorch", 9), ("#pyspark", 5), ("#java", 7), ("#pyramid", 1)]:
    ac.add(tag, n)
assert ac.suggest("#py") == ["#pytorch", "#pyspark", "#python"]      # tie 5/5 broken alphabetically
assert ac.suggest("#j") == ["#java"]
assert ac.suggest("#z") == []
ac.add("#pyramid", 20)                                               # now the top hit
assert ac.suggest("#py")[0] == "#pyramid" and len(ac.suggest("#py")) == 3
assert ac.suggest("") [0] == "#pyramid"
```

Cost: memory is O(total characters × k); each update is O(|word| × k log k). Follow-ups: time-decayed counts,
a distributed trie (shard by prefix, merge top-k from shards), and updating lazily from a batch job instead of
on every event.

### 34. Onsite Design Doc Review (newer round, high)

Card: the interviewer hands over a design doc (sometimes from the team's own backlog) and asks for a
critique — surface risks, propose alternatives, identify gaps. Common for EM and senior IC loops.

This is a reading-and-judgement round, so rehearse a **fixed pass order** rather than reacting to whatever
you read first [General]:

1. **Restate** the problem and goals in two sentences; ask what is out of scope and which numbers are real
   versus guessed.
2. **Requirements vs design:** is every requirement addressed, and does the design add anything nobody asked
   for?
3. **Failure modes:** what happens on partial failure, retry, duplicate delivery, region loss, a slow
   dependency, a bad deploy? Where is the single point of failure?
4. **Data and consistency:** source of truth, ordering guarantees, migration and backfill, schema evolution.
5. **Scale and cost:** the first bottleneck, and what breaks at 10×.
6. **Security, privacy and safety:** access control, abuse, data retention — weigh this for an AI lab.
7. **Operability:** metrics, alerts, rollout and rollback, on-call burden.
8. **Alternatives:** name at least one simpler option and say why you would or would not pick it.

Deliver findings ranked by severity ("blocking / should fix / nit") and end with the one question whose answer
would change your recommendation most. Say what you would test or prototype before committing.

---

## Summary by Category

| Category | Count | Questions |
|----------|-------|-----------|
| System Design | 7 | Chat System, Inference API, Prompt Playground, Model Deployment, LLM Batching, Real-Time Chat Architecture, Large-File Distribution |
| Online Assessment | 6 | Task Management, Banking, Cloud Storage, Employee Management, Recipe Manager, In-memory DB |
| Coding Problems | 18 | Web Crawler, LRU Cache, Deduplicate Files, Batch Image Processor, Tokenize, Stack Samples, Distributed Mode/Median, LLM Agent Tool-Use, String Processing & Stack, Memoize Decorator, Distributed Map-Reduce, Trie Tokenizer, Bootloader Fix-the-Loop, DNS Resolver, Weighted Data Batcher, Nested-Transaction KV, Fair Coin from Biased Coin, Hashtag Autocomplete |
| Behavioral | 2 | Culture & Behavioral, Hiring Manager |
| Design Review | 1 | Onsite Design Doc Review |

---

*Last updated: 2026-11-06*
