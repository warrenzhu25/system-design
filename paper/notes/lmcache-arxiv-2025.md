# LMCache: An Efficient KV Cache Layer for Enterprise-Scale LLM Inference

> arXiv preprint (2025, not peer reviewed), Tensormesh and University of Chicago — structured reading notes

Full paper content: [Markdown conversion](../original/lmcache-arxiv-2025.md)

## Paper information

- **Authors:** Yuhan Liu*, Jiayi Yao*, Yihua Cheng*, Yuwei An, Xiaokun Chen, Shaoting Feng, Yuyang
  Huang, Samuel Shen, Rui Zhang, Kuntai Du, Junchen Jiang (*equal contribution)
- **Affiliations:** Tensormesh Inc.; University of Chicago
- **Venue:** arXiv preprint, [2510.09665](https://arxiv.org/abs/2510.09665). The converted text is
  the 2025 preprint, version 1 as I downloaded it.
- **Code:** [LMCache/LMCache](https://github.com/LMCache/LMCache) ·
  **Docs:** [docs.lmcache.ai](https://docs.lmcache.ai) ·
  **Blog:** [blog.lmcache.ai](https://blog.lmcache.ai)
- **Caution:** the authors are the project's creators and company, and the commercial baselines are
  tested as black boxes. Treat the headline numbers as vendor-reported.

## One-sentence summary

LMCache is an open-source layer that sits between inference engines (vLLM, SGLang) and tiered
storage and network devices, which extracts KV cache from the engine, stores it in GPU/CPU/disk/
remote tiers, shares it across queries and engines, and makes this fast with large batched chunk
transfers, compute/I/O overlap and zero-copy, while exposing a stable connector interface to
the engines and a control API for routers and operators.

## Vocabulary you need first

| Term | Meaning |
|---|---|
| **KV cache** | The K and V attention tensors of every token seen so far, kept so they are not recomputed. "An LLM-native representation of knowledge". |
| **Page / block** | The fixed-size unit in which vLLM/SGLang allocate KV cache in GPU memory (PagedAttention). One vLLM page is 16 tokens for one layer. |
| **Prefix caching / context caching** | Reuse a stored KV cache for a later query that starts with the same tokens. |
| **Cache offloading** | Move KV cache out of GPU memory to CPU, disk or remote storage, and load it back when needed. |
| **PD disaggregation** | Run prefill and decode on different GPUs/nodes, which requires transferring KV cache from the prefill GPUs to the decode GPUs. |
| **TTFT / ITL** | Time to first token (prefill delay) / inter-token latency (average gap between output tokens). |
| **Chunk** | LMCache's transfer and storage unit, larger than a page (default 256 tokens). |
| **Connector** | The API boundary between an engine (vLLM) and LMCache. |
| **Controller** | LMCache's centralized metadata/management component. |

## Problem and motivation

### Two reasons KV cache must leave the GPU

1. **Cross-query caching** avoids recomputing shared prefixes (document analysis where the same
   document is asked about repeatedly, multi-turn chat with a fixed system prompt or long preamble).
   This needs the cache to be persisted past the lifetime of one query.
2. **Prefill-decode disaggregation** keeps latency-sensitive decode from being disturbed by
   throughput-oriented prefill. This needs KV cache to be shipped from prefill GPUs to decode GPUs.

### Evidence from LMCache's own usage statistics

Collected by a usage tracker that users turn on voluntarily (so not a random sample):

- **Stored KV cache has outgrown GPU memory.** Weekly growth over five weeks shows the share that
  does not fit in GPU memory rising significantly (Figure 3).
- **Reuse per token is rising.** Ratio of reused tokens to stored tokens beyond GPU memory grew over
  several weeks for the top 10 users (Figure 4, left). **More than 19% of users** reuse stored tokens
  more than 1.5 times on average (right).

Conclusion: cache beyond GPU capacity is growing and is increasingly re-read, so it must be moved
out and back efficiently. Figure 1 also shows growth in users, stored cache size and Docker image
pulls (no numbers are given in the text).

### Three systems challenges

1. **I/O inefficiency under paged memory.**
   - Older approaches (`torch.save`/`torch.load`, primitive tensor copies) reach only sub-1 GB/s and
     lack zero-copy.
   - vLLM/SGLang use small pages (typically 16-64 KB; vLLM uses a **62.5 KB** page for
     Llama-3.1-8B). A KV cache is scattered over many non-contiguous pages, so persisting or sending
     it means very many small I/Os.
   - Small transfers underuse bandwidth. Table 1 (RCCL transfer library, two AMD GPU nodes with
     eight Broadcom Thor-2 400 Gbps NICs):

     | Message size | 64 KB | 256 KB | 1 MB | 10 MB | 16 MB | 100 MB |
     |---|---|---|---|---|---|---|
     | Throughput (GB/s) | 4 | 13 | 30 | 46 | 49 | 49 |

     So you need at least ~16 MB per transfer to saturate that network. Also, 1-2 MB transfers reach
     only 75-80% of theoretical PCIe 5.0 bandwidth.
2. **Keeping up with fast-moving engines.** 15-20 new open-weight models per week in 2025 (the paper
   also says one prominent LLM every ~4 days). Engines change attention kernels and memory layouts,
   which changes the KV format, so an external cache library keeps breaking unless the interface is
   decoupled.
3. **No management APIs.** Without a way to locate, evict, pin or compress caches, routers and ops
   teams can't make cache-aware decisions. Two examples from customers: a financial company asked
   to *explicitly pin* frequently used financial documents; an agent company asked for APIs to
   identify the KV cache of a given content, compress it, and move it across nodes.

### Related work as the authors frame it

| Category | Examples | Gap per the authors |
|---|---|---|
| Inference stacks | vLLM Production Stack, NVIDIA Dynamo, AIBrix, llm-d, SGLang OME, KServe | They focus on K8s deployment and routing; LMCache is used inside vLLM Production Stack, Dynamo, llm-d and KServe |
| Engine-native offload | vLLM, SGLang GPU-to-CPU transfer | Single-node only; no cross-node optimization or hierarchical storage |
| KV storage layers | Mooncake, Redis, InfiniStore, 3FS | Provide distributed storage but lack the glue layer for moving small tensors across tiers, or are tied to one framework |
| Proprietary | Fireworks AI, Together AI | Closed-source stacks |
| Research prototypes | Prompt Cache, prefix-caching, PD-disaggregation and KV-compression prototypes | Usually built on research frameworks, not enterprise-ready or maintained alongside vLLM/SGLang |

## Architecture

LMCache sits between inference engines and heterogeneous storage/network devices (Figure 5).

- **Storage tiers:** GPU memory, CPU memory, local disk, remote disk, Redis (and, per Section 9,
  more, see "Community" below).
- **Networks:** Ethernet, RDMA, NVLink.

### Components named in the workflows (Figure 6)

| Component | Role |
|---|---|
| **KV connector** | Entry point from the engine. Prepares metadata: tokenized prompt, GPU memory addresses of the relevant pages. |
| **Token processor** | Works out how many tokens of a query are already in the backend (prefix match) and how many are new. |
| **Storage manager** | Saves new KV to the backend, or looks up addresses of stored KV. |
| **Transfer channel** | The data-transfer logic. |
| **Event manager** | Tracks query ids; launches asynchronous layer-wise loading events. |
| **GPU connector** | Loads KV into GPU memory (into the engine's paged layout). |
| **LMCache worker** | One per engine instance; does local work and reports to the controller. |
| **Cache controller (manager)** | Standalone global process holding the **token pool**: a record of all tokens currently stored in the backend, updated whenever a worker stores or evicts. |

### The three workflows

- **Store.** A new query goes through the KV connector (metadata), then the token processor (how
  many new tokens are not yet in the backend), then the storage manager saves the KV for those new
  tokens through the transfer channel.
- **Retrieve.** KV connector prepares metadata; the token processor finds the number of
  prefix-matched tokens. The event manager checks whether this query id was seen before. If yes, the
  memory addresses are already tracked and go straight to the GPU connector. It also launches
  asynchronous, layer-wise loading events. If the query id is new, it goes to the storage manager to
  look up the CPU-memory addresses of the stored KV.
- **Lookup.** Higher-level components like routers ask the **cache controller**, which answers from
  the token pool.

## Performance optimizations (Section 5)

Three observed problems drive this section: pages are too small to saturate bandwidth; transfers
must run alongside inference without stalling it or burning CPU on CUDA launches; and duplicating KV
on every storage device wastes space and slows inference.

### 5.1 Batched operations

- **Configurable chunk size.** Group multiple pages from multiple layers into a **chunk, default 256
  tokens** (configurable to match I/O speed). Mechanism: an intermediate **streaming GPU buffer**.
  - *Store:* a custom CUDA kernel copies the scattered paged KV into a contiguous buffer, then DMA
    engines offload the chunk to lower tiers.
  - *Load:* DMA engines bring chunks into the GPU buffer, then CUDA kernels split them back into
    paged memory.
- **Parallel store/load across tiers.** The store/load APIs take multiple source and destination
  devices, so you can copy a hot context GPU to CPU *while* offloading a cold one CPU to disk. When
  the link is full-duplex (PCIe), reads and writes run in parallel. Supported targets include local
  or remote CPU DRAM, local or remote disks, and object storage such as S3.
- **Delayed decode-time storing.** Don't offload each newly generated token's KV immediately (many
  tiny writes). Buffer them and write once a chunk's worth of tokens is generated.

### 5.2 Compute-I/O overlap

- **Layer-wise pipelining.** Separate CUDA streams for compute and data movement. Before layer 1
  runs, its KV is loaded into the GPU buffer and converted to pages. While layer 1 computes, layer 2
  is fetched and converted; layer 2's load starts only after layer 1's KV has been placed in paged
  memory. Only a **fixed-size buffer of one layer's KV** is needed on the GPU.
- **Asynchronous prefetch.** A query admitted while the engine is busy waits in a queue. During that
  gap LMCache prefetches its KV from slow tiers into faster ones (for example remote disk to local
  CPU memory or GPU memory). The target tier is user-configurable according to latency SLO and
  resource limits.

### 5.3 Minimum data copy

- **Zero-copy with a reference counter.** When one buffer is written to several places (CPU memory
  to local disk *and* remote object storage), LMCache increments a counter on the shared data rather
  than copying. Each finished read/write decrements it; at zero the buffer is released. The authors
  compare it to reference counting in operating systems.
- **Dynamic offloading.** The engine keeps a pool of **free pages** in GPU memory whose KV is not
  used by an active query. Copying all of them to CPU would waste space, so LMCache duplicates only a
  window, tracked with three pointers: **start** (beginning of the free-page region), **current**
  (how many free pages are already offloaded) and **end** (end of the pages scheduled for offload).
  Four states (Figure 7):
  1. *Initialization:* start and current coincide; start..end is the pending-duplication region.
  2. *In progress:* current moves toward end; pages before current are already offloaded.
  3. *Query arrival:* queries take free pages, so end moves forward by the number allocated, keeping
     enough duplicated pages ahead for future allocations.
  4. *Steady state:* current meets end; everything scheduled is duplicated.

  **Trade-off:** if a query wants pages beyond `current`, allocation **stalls** until the copy
  catches up. A small duplication window saves space but risks stalls. Example from the paper: if only
  one page is duplicated and a query needs three, it waits for two more; with three duplicated pages
  it proceeds at once, at a higher duplication ratio. Only GPU to CPU is implemented; extending to
  other tiers is mentioned but "not supported".

## Standardized connector interface (Section 6)

Why: engines change constantly (sliding-window attention, Multi-Head Latent Attention, new kernels),
each change can alter how KV is laid out, and ad-hoc patching of LMCache would never keep up.
LMCache's team initiated the API, and the vLLM team and LMCache team implement and maintain it
together.

**Design objectives:**
1. Maximum flexibility: enable as many KV operations as possible.
2. vLLM-native: keep the strict scheduler/worker separation, prefix caching as a first-class
   feature, and piece-wise CUDA graphs (vLLM captures graphs only for non-attention operations).
3. Friendly to **out-of-tree** connectors: no vLLM code changes needed.
4. Minimum API-level overhead: no inter-process communication at the API level.

**The two sides (Table 2):**

| Side | Function | What it does |
|---|---|---|
| Scheduler | `get_num_new_matched_tokens(query)` | Returns the number of tokens that hit in LMCache's backend. May return `None` to tell vLLM to put the request back on the waiting queue and run others first, **overlapping this request's I/O with other requests' compute**. |
| Scheduler | `update_state_after_alloc(query, blocks, num_external_blocks)` | Decides whether each vLLM page must be loaded from the external backend. |
| Scheduler | `build_connector_meta(scheduler_output)` | Builds metadata (including GPU addresses of KV pages) for transfers between LMCache and GPU. |
| Model runner | `start_load_kv(kv_pointers)` | Starts loading KV from lower tiers into GPU before inference. |
| Model runner | `wait_load_kv(kv_pointers, layer_id)` | Blocks until that layer's data has arrived. |
| Model runner | `start_store_kv(kv_pointer)` | Starts offloading KV after compute. |
| Model runner | `wait_store_kv(kv_pointer, layer_id)` | Blocks until that layer's store finished. |

Extra cache hits from the connector are treated by vLLM as ordinary prefix-cached tokens, so they
directly change scheduling (fewer tokens to prefill).

**End-to-end call sequence for one query:**
1. Scheduler calls `get_num_new_matched_tokens` (may defer the request).
2. `update_state_after_alloc` marks which pages need loading.
3. If matched tokens > 0, `build_connector_meta` prepares transfer metadata.
4. In the model runner, **layer-wise mode:** `start_load_kv` loads layer 1; before each layer,
   `wait_load_kv` synchronizes that layer and kicks off the next layer's load; after each layer,
   `wait_store_kv` waits for the previous layer's store to finish and `start_store_kv` starts
   storing the newly produced KV.
5. **Non-layer-wise mode:** `start_load_kv` blocks until the whole KV is in the correct paged
   addresses, then inference runs, and afterwards `start_store_kv` stores synchronously.

**Impact:** The API had been public in vLLM for over six months at time of writing. Adopters named:
NVIDIA Dynamo, Red Hat's llm-d, ByteDance's AIBrix and vLLM Production Stack, plus multiple
proprietary connectors from companies using the same API.

## Controller interfaces (Section 7)

Two layers: a standalone **controller manager** (global coordination, in-memory view of KV cache
state across all instances) and **per-instance workers** collocated with each LMCache instance.
External API calls go to the manager, which dispatches to workers when needed; workers also call
the manager through internal APIs for metadata updates and lookups.

**Internal APIs (used by instances):**
- `batched_admit` / `batched_evict(hashes, inst_id, device)`: report admissions and evictions to the manager.
- `batched_p2p_lookup(hashes) -> list[inst_id, device, hit_chunks]`: look up which peers hold chunks.

**External APIs (used by users, operators, routers) (Table 3):**
- `lookup(tokens) -> list[inst_id, device, hit_tokens]`
- `move((src_inst, src_device), (dst_inst, dst_device), tokens)`
- `clear(tokens, inst_id, device)`
- `pin` / `unpin(tokens, instance, storage_device)`
- `compress` / `decompress(tokens, instance, device, method)`

**Four uses the paper walks through:**
1. **KV-cache-aware routing.** Instances report admits and evictions; the manager keeps a global
   in-memory view; a router calls `lookup(tokens)` and sends the query to the instance with the most hits.
2. **KV cache migration.** When an instance is scaling down or load needs balancing, the manager
   dispatches `move(...)` to the source instance, which connects to the destination if no connection
   exists and transfers the KV.
3. **P2P sharing.** On a local miss, an instance's worker calls `batched_p2p_lookup`, gets
   (instance, device, hit chunks), and may fetch from the peer with the most hit chunks.
4. **Clearance.** On model switch or memory reclaim, `clear` is dispatched to the right instance's
   worker, which removes those tokens from the given storage device.

`compress`/`pin` have no walkthrough; users call them directly.

## Evaluation (Section 8)

### Setup

| Item | Value |
|---|---|
| Models | Llama-3.1-8B-Instruct, Sao10K-L3-8B, Llama-3.1-70B-Instruct, Qwen2.5-Coder-32B-Instruct, Qwen3-Coder-480B-A35B-Instruct-FP8, Qwen2.5-72B-Instruct |
| Datasets | Emulated multi-round QA; TriviaQA from LongBench (long-context QA); random dataset from vLLM's benchmark script |
| Hardware | 8x H100 server (GMI Cloud), using the minimum number of GPUs each model needs. Multi-node uses the same GPU count plus a remote storage backend on CPU memory. For PD disaggregation, prefiller and decoder are connected by NVLink |
| Metrics | TTFT and ITL; per-component breakdowns for CPU offload and PD disaggregation |
| Baselines | **Basic vLLM** v0.10.2 (prefix caching on, GPU-only). **vLLM CPU offloading** v0.11.0. **Commercial options #1 and #2** (dedicated endpoints reserving GPUs for a user-defined model, accessed September 10). LMCache itself is **v0.3.6** |

Three scenarios (Table 4): single-node CPU offloading, centralized storage server over Ethernet,
and PD disaggregation over NVLink.

### 8.2 Single-node CPU offloading

Workload: emulated chatbot doing document analysis. Each query ~10K tokens (a ~12-page PDF plus a
short unique question; **20K tokens for Llama-3.1-8B**), answers at most 100 tokens, 40 initial
users and more joining at a QPS rate, **500 GB** of CPU memory available for offload.

Results (Figure 8, across five models):
- At QPS = 1, LMCache has **1.9-8.1x smaller TTFT** than the best baselines.
- LMCache sustains **2.3-14x higher throughput** at the same TTFT than the strongest baseline.
- ITL is **7-92% smaller** than the best baseline at QPS = 1. (Slow first token causes queueing that
  slows later tokens, so ITL also improves.)
- vLLM's native CPU offloading fails to run Qwen3-Coder-480B; the commercial options don't host it.

Why it wins, per the authors: CPU memory holds far more KV than GPU, so the hit ratio is much higher
than GPU-only vLLM; and vLLM's native offload moves data per layer and per 16 tokens, which can't
fill the bandwidth, whereas LMCache moves **chunks** with high-performance CUDA kernels.
Black-box inference about commercial options: #1 likely lacks KV offloading to secondary storage; #2
likely has it but is still slower.

### 8.3 Real-trace-driven evaluation

Traces drawn from the token-length distributions of "company F" and "company G". Since company F's
proprietary models aren't available, five open models are used. The original trace spans several
days and is stretched to complete in one hour; LMCache uses at most 500 GB CPU DRAM; baseline is
current vLLM with GPU prefix caching.

Results: at high QPS, LMCache cuts TTFT by **at least 3.7-6.8x** and ITL by **at least 19-58%** across
the five models. (Figure 9's caption for three models says 4.4-6.6x TTFT and 34-58% ITL; Figure 10's
says 3.7-6.8x and 19-44%. These are the same range split across figures.)

### 8.4 Centralized storage server

KV cache shared through a remote server connected at **15 Gbps**; TriviaQA from LongBench; Poisson
arrivals at a given QPS. Result: **1.3-3x higher throughput** at the same TTFT (Figure 11). The gain
comes from the remote store holding much more than CPU memory, hence higher hit ratios.

**Caveat the authors state:** loading from remote adds more latency than loading from CPU memory
because of lower bandwidth, so loading can be **slower than prefill** for short contexts or small
models (prefill is already fast). So the choice between loading and recomputing must be adaptive
(see sensitivity study).

### 8.5 PD disaggregation

Against vLLM's native PD disaggregation, with the official benchmark script (random data; 8K input,
200 output tokens). Results over four models: mean TTFT down **1.53-1.84x**, mean ITL down
**1.12-1.66x**, and better P95 TTFT.

Why: LMCache copies each chunk of the prefiller's KV (produced during chunked prefill) into a
buffer in prefiller GPU memory, sends it to a matching buffer on the decoder, and the decoder
copies it into its paged memory. vLLM's native path uses NIXL to copy **page by page** from the
prefiller's scattered page addresses to the decoder's, so bandwidth is underused.

### 8.6 Component-wise analysis

- **PD disaggregation breakdown (Figure 14):** prefill and decode compute times are identical in
  both; the difference is purely the KV transmission time, which LMCache cuts through coarser
  granularity.
- **CPU load bandwidth (Table 5):**

  | Method | Achieved bandwidth |
  |---|---|
  | LMCache | 400 Gbps |
  | vLLM native CPU offloading | 88 Gbps |

  Reason: each transfer triggers a CUDA memcpy with metadata prep beforehand and a completion signal
  after; per-page copies pay that fixed cost many times, per-chunk copies pay it rarely.
- **Async overlap (Figure 13):** overlapping KV loading with inference compute (prefill or decode)
  reduces end-to-end delay by **1.46x**.

### 8.7 Sensitivity to context length and bandwidth

On B200 machines (Figure 15):
- At **32 Gbps**, LMCache's loading beats plain prefill only if the input is longer than about
  **256K tokens**.
- At **64 or 128 Gbps**, loading beats prefill at **every** context length tested.

Takeaway: make loading **adaptive**. Under low bandwidth, load only above the crossover length.

### 8.8 SGLang

Qwen3-32B on 2 H100s (TP=2) with LMCache CPU offloading: better throughput and lower mean TTFT
and end-to-end latency than SGLang without offload; **comparable** to SGLang's native CPU offload.
The authors note SGLang's native offload lacks a distributed, hierarchical backend (local disk, remote CPU/disk).

## Real-world lessons (Section 9)

1. **Remote loading can beat prefill.** The old belief was that remote storage only buys capacity and
   hit rate at a latency cost, because object stores like S3 once gave ~100 MB/s. S3 Express has
   raised that to nearly 1 GB/s. A user ("Company C") loading KV caches from its own remote object
   store reported **22-32% lower TTFT** than full prefill. So remote backends can improve hit rate
   *and* TTFT.
2. **Context truncation destroys prefix hits.** Many companies keep only the latest tokens when input
   exceeds the window (sliding window). Truncated inputs no longer match earlier prefixes. In
   company F's real trace, hit ratio drops from about **85% to 45%** when only the latest tokens are
   kept. Lesson: avoid dynamically adding or removing context tokens.
3. **Containers are the norm.** Production is Kubernetes with Docker images; many users never touch
   the source, just the official images.
4. **Hit rates exceed expectations.** Customers expected KV reuse only for fixed system prompts, but
   company G saw about **50%** hit rate in production. Reusable context is now often *dynamic*:
   coding assistant conversation histories, chat apps, RAG pipelines.
5. **Industry vs academia.** LMCache began in May 2024 as a research prototype for the authors' own
   work. Industry wanted performance, stability and compatibility, so flexible APIs for custom
   attention (such as selective token dropping) were deprioritized, which made it less popular in
   academia. They plan more flexible APIs.
6. **Python over Rust/C++.** Many companies rewrite ML libraries in Rust or C++. LMCache keeps
   Python with careful optimization, to evolve faster and accept community contributions while
   keeping comparable performance.
7. **Community effort.** A year earlier it supported only local CPU, local disk and Redis with vLLM
   on NVIDIA GPUs. Now it adds **eight more storage backends** (NFS, WEKA, GPU-Direct Storage,
   Mooncake Store, NIXL, S3, InfiniStore, Valkey), **four processor types** (NVIDIA, AMD, Ascend,
   TPU) and **two engines** (vLLM and SGLang), upstreamed by industry partners.

The conclusion also mentions KV cache reuse in recommendation systems and lossy compression in
open-ended chatbots as emerging uses, in trillion-token-scale deployments.

## Limitations, critique and open questions

Stated or implied by the authors:
- Remote loading can be slower than prefill for short contexts, small models or low bandwidth;
  adaptivity is required but is shown as a study, not a built-in policy.
- Dynamic offloading is only implemented for GPU to CPU. Extending it to other tiers is only mentioned.
- The real-trace evaluation uses substitute open models, and the traces are reshaped (stretched to
  one hour). Commercial baselines are black boxes and can't be configured the same way.
- Academic flexibility (custom attention hooks) was deprioritized.

My own observations (not claims from the paper):
- The largest quantitative claims (up to 15x throughput, 2.3-14x at equal TTFT) come from the
  authors' own multi-round QA workload with large reusable contexts and a 500 GB CPU budget. Gains
  depend heavily on how much reuse the workload has.
- The abstract says "up to 15x" while the body reports 2.3-14x for single-node CPU offload and 1.3-3x
  for remote storage. The 15x figure isn't reproduced in the body text I read, so quote the
  body numbers.
- No failure-handling or consistency discussion: nothing on what happens when a worker dies, when
  the controller's token pool goes stale, or how conflicting `move`/`clear` operations are ordered.
- The controller is a single global in-memory service; its scalability and availability are not evaluated.

## Takeaways for interviews and design work

1. **Make transfers big.** Page-sized I/O (20-60 KB) wastes the network; ~16 MB+ is needed to saturate
   a 400 Gbps NIC. Use a staging buffer with a gather/scatter kernel to turn scattered pages into
   contiguous chunks.
2. **Overlap I/O with compute** with per-layer pipelining on separate streams; you then need just one
   layer of buffer space on the GPU.
3. **Use idle time.** While a request waits in the queue, prefetch its cache into a faster tier.
4. **Count copies.** Reference-counted buffers avoid duplicating data when fanning out to several
   tiers.
5. **Isolate from engine churn** with a narrow connector API (scheduler side vs model-runner side).
6. **Expose a control plane** (lookup, move, pin, clear, compress), because routers, autoscalers and
   operators need to reason about where the cache is.
7. **Loading vs recomputing is a bandwidth-and-length decision.** There is a crossover context
   length per bandwidth; make the choice adaptive.
8. **Truncation and sliding windows kill prefix reuse.** Keep prompts append-only where possible.

### Short answers to likely questions

- *Why is vLLM's native CPU offload slower?* It copies per page and per layer, so fixed per-copy
  overhead dominates (88 Gbps achieved vs 400 Gbps for chunked copies in Table 5).
- *How does LMCache keep up with engine changes?* A small connector API with scheduler-side and
  model-runner-side hooks, co-maintained with the vLLM team and supporting out-of-tree connectors.
- *How would a router use LMCache?* Query the controller's `lookup(tokens)` for per-instance hit
  counts and send the request where the hit is largest; use `move` to rebalance.
- *When is fetching a remote KV cache worse than prefill?* Short contexts, small models, or low
  bandwidth. At 32 Gbps on B200 the crossover was ~256K tokens; at 64-128 Gbps loading won at all
  lengths tested.
- *Why can offloading to CPU stall the engine?* Dynamic offloading mirrors only some free pages; if
  a query needs pages that haven't been copied yet, allocation waits for the copy to catch up.

## Related notes

- [Mooncake (arXiv 2024)](mooncake-arxiv-2024.md): the system-level design (global KVCache-aware
  scheduler); Mooncake Store is one of LMCache's backends
- [vLLM (SOSP 2023)](vllm-sosp-2023.md): PagedAttention, the small-page layout that makes KV transfer inefficient
