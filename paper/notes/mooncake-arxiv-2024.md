# Mooncake: A KVCache-centric Disaggregated Architecture for LLM Serving

> arXiv technical report (2024), Moonshot AI and Tsinghua University — structured reading notes

Full paper content: [Markdown conversion](../original/mooncake-arxiv-2024.md)

## Paper information

- **Authors:** Ruoyu Qin, Zheming Li, Weiran He, Mingxing Zhang, Yongwei Wu, Weimin Zheng, Xinran Xu
- **Affiliations:** Moonshot AI; MadSys group at Tsinghua University
- **Venue:** arXiv preprint, [2407.00079](https://arxiv.org/abs/2407.00079). The text converted here
  is titled "Mooncake: A KVCache-centric Disaggregated Architecture for LLM Serving" and calls itself
  a "Preprint Tech Report".
- **Which version this is:** a later, revised version of this work was published at FAST 2025 under
  a different title. I did **not** read that version, and these notes do not cover it. Anything
  below about the evaluation, numbers or design is from the tech report.
- **System:** the serving platform behind Kimi, Moonshot AI's chatbot
- **Code and trace:** [kvcache-ai/Mooncake](https://github.com/kvcache-ai/Mooncake); the request trace
  is open-sourced there

## One-sentence summary

Mooncake argues that in LLM serving the **KVCache is the thing to schedule around**: it splits
prefill and decode onto separate node pools, builds a cluster-wide KVCache pool out of the idle CPU
DRAM, SSD and RDMA bandwidth of the GPU machines, and uses a global scheduler (Conductor) that
routes each request by *where its cache is* while rejecting requests early, using predicted load,
when the cluster cannot meet its latency SLOs.

## Vocabulary you need first

| Term | Meaning |
|---|---|
| **LLM prefill** | Process all input tokens in parallel; produces the first output token and the **KVCache** (the K and V tensors of every input token at every layer). Compute-bound. |
| **LLM decode** | Generate one token per step, reusing and appending to the KVCache. Memory-bandwidth-bound; batching more requests raises utilization. |
| **TTFT** | Time to first token. The latency SLO for prefill. |
| **TBT** | Time between tokens. The latency SLO for decode. |
| **MFU** | Model FLOPs utilization: how much of the GPU's peak compute is actually used. |
| **Continuous batching** | Before each decode iteration the scheduler adds newly arrived requests and removes finished ones. |
| **Prefix caching** | If a new request starts with the same tokens as an earlier one, reuse that earlier KVCache instead of recomputing it. |
| **MaaS** | Model as a Service: selling inference to many users under SLOs. |
| **Goodput / effective throughput** | Throughput counting only requests that finish **within SLO**. |
| **Conductor** | Mooncake's global scheduler. |
| **Messenger** | Mooncake's per-node RDMA service that moves KVCache between machines. |
| **CPP** | Chunked pipeline parallelism, Mooncake's way to run one long prefill across several nodes. |

## Problem and goals

1. **Objective.** Maximize overall *effective* throughput (directly tied to revenue) subject to TTFT
   and TBT SLOs. The paper frames Kimi's job as a constrained optimization with different SLO levels.
2. **Two throughput levers conflict with latency.**
   - *Reuse KVCache* to cut compute. But if the cache lives on a remote or slower tier, fetching it
     lengthens TTFT.
   - *Make decode batches bigger* to raise MFU. But bigger batches raise TBT.
3. **Overload is the normal case.** GPU supply is limited and Kimi's traffic grew very quickly, so
   the cluster is often overloaded at peak. You cannot just add machines. Most prior work assumes
   every request will be served. Mooncake has to decide **which requests to refuse**.
4. **What counts.** A request only contributes to goodput if it fully completes within SLO. If it is
   dropped halfway, every token already computed is wasted. So a doomed request should be rejected
   *as early as possible*.

SLO specification used in the paper: `TTFT P90 = 10x` and `TBT P90 = 5x` in the end-to-end
experiments, meaning 90% of requests must have TTFT within 10 times, and TBT within 5 times, that of
a single request running alone. In real deployments Kimi uses fixed absolute SLOs; if monitoring
sees them missed, the options are add capacity (usually impossible) or reject requests.

## Background: why prefill and decode want different treatment

- Prefill computes all input tokens at once. Attention cost grows quadratically with length and the
  MLP cost grows linearly, so prefill time grows **superlinearly** with input length (Figure 2,
  left). Except for very short requests, prefill is compute-intensive.
- Decode processes one token per request per step, so it is memory-bound and its step time grows
  **sublinearly** with batch size (Figure 2, right). Larger batches improve utilization until VRAM
  for KVCache runs out or the TBT SLO is hit.
- Putting both on the same GPU lets long prefills interrupt decode steps and blow TBT. That is the
  case for **disaggregation**, also argued by Splitwise, DistServe and TetriInfer (cited in the
  paper).

## Architecture

The paper's Figure 1 is the architecture diagram; the pieces are described here in words.
Mooncake has two kinds of worker nodes and three supporting pieces:

| Piece | Role |
|---|---|
| **Prefill instances / pool** | Nodes that only run prefill. Each machine in the paper's testbed has 8 GPUs. |
| **Decoding instances / pool** | Nodes that only run decode with continuous batching. |
| **Disaggregated KVCache pool** | Built from CPU DRAM and SSD of the GPU machines themselves. Paged blocks; eviction can be LRU, LFU or request-aware. Described as "near-GPU prefix caching without additional costs". Also lets Kimi offer a *context caching API* to users. |
| **Messenger** | A separate (GPUDirect) RDMA-based process on each node. Receives signals and moves KVCache across machines asynchronously. |
| **Conductor** | Global scheduler. Dispatches requests using current cache distribution and workload, and decides when to **replicate or swap** blocks. |

### KVCache block identity: prefix hashing

KVCache is stored in fixed-size **blocks**. Each block is identified by a hash of *its own tokens
concatenated with the hash of the previous block*. So the hash identifies "this block **given
everything before it**". Two requests can share a block only if all earlier blocks are identical
too, which is exactly the prefix-caching rule. Blocks with identical hashes are stored once.

Worked example from the released trace (block size 512 tokens): two requests both start with hash
ids 46-57. That is 12 identical blocks, so the first `12 x 512 = 6,144` tokens can be shared, even
though the requests then diverge (ids 2353, 2354 versus 2366).

### The four-step request workflow (Figure 4)

After tokenizing, Conductor picks **a prefill node (group) and a decoding node**, then:

1. **KVCache reuse.** The prefill node receives the raw input, the block ids of the reusable prefix,
   and the block ids of the full cache allocated for the request. It loads the prefix cache from
   *remote CPU memory* into GPU memory. Skipped if there is no prefix hit. The choice of node
   balances three goals: reuse as much cache as possible, balance prefill load, and meet the TTFT
   SLO (details in "Scheduling" below).
2. **Incremental prefill.** The node computes only the tokens not already cached, then stores the new
   KVCache in CPU memory. If the number of uncached tokens exceeds `prefill_chunk` (typically more
   than 1,000 tokens, chosen to saturate the GPU), prefill is split into chunks and pipelined (CPP).
3. **KVCache transfer.** The Messenger streams the KVCache *layer by layer* to the decoding node's
   CPU memory. This is asynchronous and overlaps with step 2.
4. **Decoding.** Once all KVCache has arrived in the decode node's DRAM, the request joins the next
   continuous batch. The decode node was pre-selected by Conductor based on load so as not to break
   TBT, but the **local scheduler re-checks the TBT SLO**, because load may have changed during
   prefill. This re-check can reject the request, in which case all the prefill work is wasted. That
   waste is what the "early rejection" work later in the paper tries to avoid.

Decode-side detail from the Figure 4 caption: on decoding instances, loading KVCache from CPU to GPU
is asynchronous and concurrent with GPU decoding, so the GPU does not sit idle waiting.

## The sampled real-world trace

The authors release one hour of sampled Kimi traffic. Because users in a session share prefixes,
they preferentially sampled requests from the same session so cache relationships survive.

- **Size:** 23,608 entries (the evaluation says "23,000 requests" in round numbers)
- **Fields:** `timestamp` (ms, 0 to 3,600,000), `input_length`, `output_length` (token counts only,
  no text), and `hash_ids` (remapped prefix-block hashes, block size 512)
- **Statistics:** average input 8,590 tokens, average output 182 tokens, an input:output ratio of
  about 720. The authors stress this reflects Kimi's long-context focus and is not universal.
- **Privacy:** no real text or tokens, only counts and remapped hashes. Evaluations run a *dummy*
  model with the LLaMA2-70B architecture.
- **Claim:** the first open-source dataset usable for real-world KVCache reuse analysis.

### What the trace shows about caching

Table 1 simulates a single global cache pool, giving the hit ratio by policy and block capacity:

| Block capacity | Inf | 100000 | 50000 | 30000 | 10000 | 1000 |
|---|---|---|---|---|---|---|
| LRU | 0.55 | 0.55 | 0.55 | 0.54 | 0.46 | 0.34 |
| LFU | 0.55 | 0.55 | 0.55 | 0.50 | 0.40 | 0.34 |
| LengthAwareCache (LFU that prefers blocks late in a request) | 0.55 | 0.54 | 0.53 | 0.50 | 0.40 | 0.34 |

- Growing capacity from 1,000 to 50,000 blocks lifts hit ratio from 34% to 55%, then it flattens.
  **Even with infinite capacity the ceiling is 55%.** The authors warn this is a sample and the
  real required capacity scales up with the real workload.
- **LRU wins** on this trace, probably because of temporal locality between related requests.
- **Popularity is heavily skewed:** over 50% of cache blocks are never reused, while some blocks are
  hit tens of thousands of times (Figure 6). So hot blocks must be replicated or the machines
  holding them become congested.

## Prefill pool design

### Why keep prefill separate, given chunked prefill exists?

Chunked prefill (Sarathi-Serve) splits prompts into small chunks that ride along in decode batches.
Benefits: all nodes look the same (simpler scheduling), and mixing prefill chunks into decode
batches raises the decode batch's compute intensity (better MFU).

Mooncake still disaggregates. A request's prefill is inlined into the decode batch **only if it can
go without chunking and without hurting TBT**. Two reasons for keeping a prefill pool:

1. Prefill nodes need different cross-node parallelism for long contexts (next section).
2. It opens a chance to save VRAM (layer-wise prefill, below).

### Multi-node prefill: why CPP instead of TP or SP

Context windows have grown from 8K to 128K to 1M, so a long-context request has 10-100x more input
tokens than output tokens, and TTFT dominates. One 8-GPU node is not enough, so prefill should span
nodes.

| Option | Problem |
|---|---|
| **Tensor parallelism across nodes** | Two RDMA all-reduces per layer. MFU of prefill nodes drops a lot. |
| **Sequence parallelism (SP)**, e.g. Ring/Striped Attention, DeepSpeed Ulysses | Splits the sequence across nodes; needs cross-node communication at least once per layer. Better than TP but MFU is still worse than single-node TP. You'd want two groups (TP-only and SP) and route to SP only when TTFT demands it, which leads to hard dynamic resizing. *Elastic* SP (LoongServe) exists but needs a global communication group set up in advance and complicates Conductor's cache- and SLO-aware decisions. SP's per-layer traffic also competes with KVCache transfers for the network. |
| **Chunked pipeline parallelism (CPP)** (Mooncake's choice) | Group every X prefill nodes into a pipeline. A request's tokens are split into chunks (each at most `prefill_chunk`). Different chunks of the **same request** run at the same time on different pipeline stages. |

CPP exploits the **autoregressive property** of decoder-only models (a chunk only depends on
earlier chunks' KV). Benefits:
1. Communication is needed only at pipeline stage boundaries and can overlap with compute, giving
   better MFU and less network contention with KVCache transfer.
2. It suits both short and long contexts without a separate configuration, so no frequent dynamic
   re-partitioning of nodes.

The authors say this is, to their knowledge, the first application of this training-style pipeline
idea (cf. TeraPipe) to inference.

### Layer-wise prefill: saving VRAM by streaming the cache

Reasoning: if a request's KVCache has size `S` and is held for time `T`, its VRAM occupancy cost is
`S x T`. Chunked prefill inlined with decode stretches `T`, so it costs more.

Mooncake instead overlaps KVCache movement with computation, layer by layer:

- Before layer *l*'s attention starts, wait for layer *l*'s async load to finish, and trigger the
  async load of layer *l+1*.
- After layer *l*'s attention finishes, launch the async **store** of layer *l*'s KVCache.
- After the last layer, wait for all outstanding stores.

Effect: the prefill instance's execution time is about **max(KVCache load time, normal prefill
time)**, depending on how large the cached prefix is relative to the input. Figure 7 shows layer-wise
storing adds little latency even for long contexts.

Consequence: prefill scheduling can **ignore VRAM capacity** as long as VRAM can hold one request.
Prefill scheduling considers only the KVCache distribution and the available **DRAM** of the node.
The authors suggest using the freed VRAM later, for example to inline the decode of low-priority
batch-API requests (no strict TBT) into prefill nodes.

## Scheduling (normal load)

### Cache-aware prefill scheduling (Algorithm 1)

Goal: for each request, pick (prefill instance `p`, decode instance `d`) to minimize TTFT subject to
SLOs. Steps in plain language:

1. Split the prompt into blocks and compute chained prefix hashes (`block_keys`).
2. Across all prefill instances, find the **best prefix match** (longest) and who holds it.
3. For each prefill instance, estimate TTFT:
   - `T_queue` = estimated wait (sum of predicted prefill times of already queued requests).
   - **If the best remote match is not much longer than this instance's local match** (the ratio is
     below `kvcache_balancing_threshold`): use the local prefix. `TTFT = T_queue + T_prefill(len, local_prefix_len)`.
   - **Otherwise**: consider pulling the best cache from the holder.
     `TTFT = T_transfer + T_queue + T_prefill(len, best_prefix_len)`.
   - Keep the instance with the smallest TTFT.
4. Choose a decode instance by load (`SelectDecodingInstance`), which also yields a predicted TBT.
5. If predicted TTFT exceeds `TTFT_SLO` **or** predicted TBT exceeds `TBT_SLO`, **reject** (HTTP
   429 to the upper layer).
6. If the best match was remote and longer than the threshold, issue a **hot-spot migration**:
   transfer the KVCache from the holder to the chosen instance.

How the estimates are made:
- **Prefill time:** an offline-trained predictor on request length and prefix-hit length. Because
  Transformer computation is regular, the error is small given enough offline data.
- **Queue time:** sum of the predicted prefill times of everything queued ahead. TTFTs for all
  candidate instances are computed in parallel, so the scheduling cost is negligible next to
  inference time.
- **Transfer time:** the hard one. It depends on data size *and* current network state, especially
  whether the sender is congested. This is why hot-block replication matters.

Prefix matching like this already exists in vLLM, but the open-source version only caches locally.

### Cache load balancing and hot-spot migration

Each prefill machine owns its own local prefix cache, and usage varies enormously. System prompts are
touched by nearly every request, while a cache for one user's long document may be used by one user.
So where copies live determines whether routing can achieve both **high hit rate and low load**.

The "straw man" is to collect global per-block usage, predict future use, and place replicas
accordingly. The authors reject it: Kimi's workload is highly dynamic and fast-growing, so
prediction is unreliable. Instead, a **heuristic that replicates as a side effect of routing**:

- If a request goes to an instance that doesn't hold the best cache (because the best holder is
  busy), Conductor forwards the request **plus the cache's location**. That instance pulls the
  KVCache from the holder and stores it locally, if the extra prefill time it would otherwise spend
  is longer than the transfer time.
- If the best remote prefix is no more than local prefix times a threshold, it just **recomputes**.

Either way popular blocks are copied to more machines over time, without ever predicting
popularity. The threshold is currently tuned manually (footnote: could be made adaptive).

### Scheduling experiment (Figure 8)

Cluster of 8 prefill + 8 decode instances, built overnight from idle machines, replaying 23,000
real requests. Compared: random, load-balancing, cache-aware (6.1) and KVCache-centric (6.1 + 6.2).
Metrics: average TTFT and TTFT SLO attainment. Result: both cache-aware and cache-balancing
strategies cut TTFT a lot, and the full KVCache-centric policy beats random and load-balancing on
both metrics. The text gives this **qualitatively** (figure only), without exact numbers.

## Overload-oriented scheduling

Premise: for a fast-growing service the cluster grows slower than demand, so overload is routine.
The system should accept requests until load hits a threshold and then reject (or defer for retry).

### Defining load by SLO

Mooncake measures load as **predicted SLO satisfaction**: compare the maximum predicted TTFT of a
prefill instance with `l_ttft`, and maximum predicted TBT of a decode instance with `l_tbt`. This is
easier than in coupled systems, where prefill and decode interfere and load is just
"requests in flight / max capacity". Two decisions follow: accept the prefill based on prefill
load, and proceed to decode based on decode load.

### Early Rejection

A time lag separates the prefill and decode decisions. If a request passes the prefill check, is
prefilled, and is then rejected by a now-overloaded decode node, the prefill work is wasted, so the
number of requests really finishing is less than prefill load suggests.

**Early Rejection:** evaluate the *decode* instance's load **before** starting prefill. On arrival,
Conductor accepts or rejects based on the **greater** load of the prefill and decode pools. This
cuts wasted prefill computation and improves balance.

### The side effect: load oscillation

Observed on a 20-machine cluster over 20 minutes (Figure 9): prefill and decode load swing in
**anti-phase**. Worse with fewer prefill machines and longer prefill. The cause is the lag between
the decode-load observation and its actual execution. The four stages (Figure 10a):

1. Both pools lightly loaded, so Conductor accepts a lot until prefill is full.
2. Those requests arrive at decode, making decode load high, so Conductor now rejects most requests;
   prefill load falls.
3. Nothing new enters decode, so decode load drains; Conductor sees low load and accepts a flood
   again until prefill is full.
4. Decode load climbs again, Conductor rejects, prefill sits idle. Repeat.

Net effect: poor utilization of the whole cluster.

### Early Rejection Based on Prediction

Fix: predict the decode load **at the time the request would reach decode**, and use that to decide
admission. Two ways to predict:

- **Request-level:** predict each request's output length (unknown in advance), then the TBT and
  completions. Hard: costly or inaccurate, especially in overload. Left to future work.
- **System-level (used):** don't track individual requests; estimate aggregate load a short time
  ahead. Assume every request's decode takes a **uniform time `t_d`**. For a future time `t`:
  1. add requests that prefill will have finished by `t` to the decode pool,
  2. remove requests whose decode will have finished (running longer than `t_d`) before `t`,
  3. compute the average ratio of predicted TBT to `l_tbt` across decode instances. That ratio is
     the predicted load.

This needs less precision, and is better suited to overload where accuracy per request is
unattainable.

## Evaluation

### Setup

| Item | Value |
|---|---|
| Nodes | 8x NVIDIA A800-SXM4-80GB (NVLink), RDMA up to 800 Gbps between nodes |
| Deployment | Each node runs either a prefill instance or a decode instance |
| Model | **Dummy** model with LLaMA2-70B architecture |
| Baseline | vLLM (continuous batching + PagedAttention; prefill and decode coupled) |
| Metrics | P90 TTFT and P90 TBT, normalized to the SLO limit (1.0 = limit), as a function of RPS |
| SLO thresholds | Lowest observed value x10 for TTFT, x5 for TBT (see "SLO specification" above) |
| Arrivals | Poisson for public/simulated data; timestamp replay for the real trace |

| Dataset | Avg input | Avg output | Cache ratio |
|---|---|---|---|
| ArXiv Summarization | 8,088 | 229 | ~0% |
| L-Eval | 19,019 | 72 | >80% |
| Simulated | 16k / 32k / 64k / 128k | 512 | 50% |
| Real (Kimi trace) | 7,955 | 194 | ~50% |

Note the real-trace evaluation set (avg input 7,955) differs from the whole released trace's
statistics (8,590), as sampled in the paper.

### Results

**Public datasets** (vLLM-[4M] = 4 vLLM instances; Mooncake-[3P+1D] and [2P+2D]):
- Mooncake-[3P+1D] reaches **+20% throughput on ArXiv Summarization and +40% on L-Eval** under SLOs.
  L-Eval gains more because its high cache ratio makes prefix caching cut prefill time.
- Mooncake-[2P+2D] has lower TBT but worse TTFT than [3P+1D] and vLLM, because prefill and decode
  capacity are imbalanced. The authors say real demand for each is stable over a period, so the ratio
  can be preset, and dynamic conversion between roles is future work.

**Simulated long-context data** (16k-128k prompts, 50% cache):
- Long prompts disrupt vLLM's decode, so to hold TBT the baseline processes requests individually.
- Mooncake still batches, because disaggregation shields decode from prefill, and never breaks the
  TBT SLO.
- Throughput gains of **50% to 525%** under the same TTFT and TBT limits.

**Real workload** (Mooncake-[10P+10D] vs vLLM-[20M]; TTFT limit 30 s, TBT limit 0.1 s/token):
- TTFT CDFs are nearly identical; almost 100% of requests meet the TTFT SLO in both.
- Nearly 100% of Mooncake's requests satisfy TBT, but only **57% of vLLM's** do, with some having
  extremely high TBT.
- Mooncake processes about **75% more requests** within SLO.

**Overload** (8P+8D, 23,000 requests replayed at **2x speed**):

| Strategy | Requests rejected |
|---|---|
| Baseline (reject by load before each stage, prefill can still be wasted) | 4,183 |
| Early Rejection | 3,771 |
| Early Rejection based on Prediction | 3,589 |

Prediction-based rejection rejects 594 fewer requests than baseline (about 14% fewer) and 182 fewer
than plain early rejection. The paper reports rejection counts here, not throughput or SLO
attainment, so "fewer rejections" is the proxy for "more useful work done".

## Related work and positioning

- **Engines:** FasterTransformer, TensorRT-LLM, DeepSpeed Inference, Orca (iteration-level
  scheduling), vLLM (KVCache memory management), FlexGen, Sarathi, FastServe. Mooncake builds on
  vLLM's open-source community.
- **Disaggregation:** Splitwise (early motivation), DistServe (resource allocation per phase for
  goodput), TetriInfer (chunked prefill + disaggregation + predictive scheduling).
- **Prefix caching:** Prompt Cache, SGLang's RadixAttention (LRU over a radix tree).
- **AttentionStore** (concurrent): hierarchical KVCache using cheap memory/storage, shares many
  design choices. The paper's distinction: in long-context inference KVCache is huge, so you need
  capacity and fast transfer *plus KVCache-centric global scheduling*. **Mooncake is not a
  standalone cache service**; it couples a cache storage mechanism with a cache-aware scheduler.
- **Preble** (prompt scheduling) is essentially KVCache-centric scheduling too. Mooncake notes that
  real reusability in its online trace is much lower than open-source benchmarks suggest: **at most
  55% theoretical reuse** even with infinite storage and TTFT budget. It can reach **~90%** for
  some apps, such as Kimi's chat-with-paper service.

## Future work the authors list

- **Heterogeneous accelerators:** use bandwidth-oriented chips for decode (GDDR/LPDDR can be an
  order of magnitude better per dollar or per watt than flagship HBM parts); processing-in-memory
  and hybrid bonding for memory-bound decode.
- **Disaggregate attention from linear operators in decode:** decode attention's arithmetic
  intensity is proportional to heads divided by KV heads and cannot be raised with batch size. DeepSeek's
  MLA raises intensity from the other side.
- **Smaller KVCache** (compression, token selection, cross-layer sharing, hybrid architectures) helps
  twice: bigger batches and higher cache hit ratio.
- **Scheduling:** priority-aware, per-request SLO tiers; eviction for partial hits and expiry;
  dynamic prefill/decode rebalancing; use idle resources for batch jobs.

## Limitations, critique and open questions

Stated by the authors:
- Dummy model and content-free trace, so results cannot be checked against real outputs.
- The prefill:decode ratio is preset; dynamic conversion is future work (the [2P+2D] result shows
  the cost of getting it wrong).
- Migration threshold is manual; transfer-time prediction suffers from congestion; load prediction
  assumes uniform decode time; request-level length prediction is not done.
- Rejection by the decode-side local scheduler after prefill is still possible, so some waste remains.

My own observations (not claims from the paper):
- Only vLLM is the baseline, and vLLM's coupled design is the weak spot being measured. No direct
  comparison with Splitwise, DistServe or TetriInfer is shown.
- Largest cluster tested is 20 machines. The Figure 8 scheduling comparison is only given in a plot.
- The 55% reuse ceiling is specific to Kimi's sampled traffic. The benefit of the KVCache pool is
  workload-dependent: ~0% cache (ArXiv) saw only +20%, >80% cache (L-Eval) saw +40%.
- The architecture trades **network and DRAM capacity for GPU compute**. It assumes a fast RDMA
  fabric (up to 800 Gbps per node here) and that CPU DRAM is largely idle.

## Takeaways for interviews and design work

1. **Treat the cache as a first-class scheduling input.** Estimate TTFT = queue + transfer + compute
   for each candidate node and choose the minimum. Compare "pull the cache" against "recompute it".
2. **Replicate as a by-product of routing,** not through a popularity predictor. When you route away
   from the cache holder, make the new node fetch and keep a copy.
3. **Disaggregate by resource profile.** Prefill is compute-bound and long-context-sensitive; decode
   is memory-bound and latency-sensitive. Each pool gets its own parallelism strategy (CPP for
   prefill) and its own SLO.
4. **Overlap data movement with compute** layer by layer, so capacity planning can ignore VRAM
   beyond one request.
5. **Under overload, reject early and cheaply,** but beware feedback delay. Admission control driven
   by a lagging load signal oscillates; predict the load at the time the work will arrive.
6. **Define goodput strictly:** only completed-within-SLO requests count. Rejecting at the door is
   better than failing in the middle.

### Short answers to likely questions

- *Why not just use chunked prefill and keep everything colocated?* Long contexts need cross-node
  parallelism for TTFT, which colocated chunked prefill does not give, and disaggregation lets
  prefill drop its VRAM constraint via layer-wise streaming. Mooncake still inlines short, unchunked
  prefills into decode if TBT allows.
- *Why CPP rather than sequence parallelism?* Less communication (only at stage boundaries,
  overlappable), no per-layer cross-node traffic competing with KVCache transfers, and no need to
  dynamically resize groups.
- *What goes wrong with naive early rejection?* A lag between observing decode load and acting on it
  makes prefill and decode load oscillate in anti-phase, so utilization is poor.
- *How does Mooncake avoid hot-spot nodes?* Heuristic migration: when a request is routed away from
  the best holder, the chosen node copies the cache, so hot blocks spread without prediction.
- *Where does the throughput gain come from?* Three places: decode never interrupted by prefill
  (TBT SLO holds with large batches), prefix reuse from a cluster-wide cache (more with higher cache
  ratio), and wasted work avoided by early, prediction-based rejection.

## Related notes

- [LMCache (arXiv 2025)](lmcache-arxiv-2025.md): a KV cache layer that can use Mooncake Store as a backend
- [vLLM (SOSP 2023)](vllm-sosp-2023.md): PagedAttention and continuous batching, which Mooncake builds on and compares against
