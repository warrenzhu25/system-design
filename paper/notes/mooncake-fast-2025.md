# Mooncake: A KVCache-centric Disaggregated Architecture for LLM Serving

> FAST 2025 (Best Paper Award per the project's update log) — structured reading notes

Paper: [arXiv 2407.00079](https://arxiv.org/abs/2407.00079) ·
Code: [kvcache-ai/Mooncake](https://github.com/kvcache-ai/Mooncake) ·
Docs: [kvcache-ai.github.io/Mooncake](https://kvcache-ai.github.io/Mooncake)

These notes summarize arXiv v2. Check numbers against the paper before quoting them.

## Paper information

- **Organizations:** Moonshot AI and Tsinghua University
- **System:** the serving platform behind the Kimi chatbot
- **Original text:** not vendored in `../original/`; read it at the arXiv link above

## One-sentence summary

Mooncake treats the KVCache as the central resource of LLM serving: it separates prefill and
decode onto different nodes, pools idle CPU DRAM, SSD and RDMA bandwidth into a cluster-wide
KVCache, and uses a global scheduler that places requests where their cache is, rejecting early
when the system is overloaded.

## Problem

- The goal is **effective throughput**: only requests that finish within their latency SLOs count.
  The SLOs are **TTFT** (time to first token) and **TBT** (time between tokens).
- Throughput optimizations conflict with SLOs. Fetching a remote KVCache can lengthen TTFT, and
  large decode batches raise TBT.
- GPUs are scarce, so the system is often overloaded at peak. Most prior work assumes every
  request will be served; Mooncake has to decide which to drop.

## Architecture

| Component | Role |
|---|---|
| Prefill pool | Nodes that run prefill only |
| Decode pool | Nodes that run decode only, with continuous batching |
| KVCache pool | Paged blocks in underused CPU DRAM and SSD across the cluster |
| Messenger | Per-node GPUDirect RDMA service that moves KVCache asynchronously |
| Conductor | Global scheduler: picks a prefill and decode instance for each request and decides when to replicate or swap blocks |

**KVCache blocks.** Each block's hash includes its prefix, so identical prefixes deduplicate.
Eviction can use LRU, LFU or request-aware policies.

**Request workflow**
1. Load the reusable prefix KVCache from remote CPU memory.
2. Run incremental prefill. Inputs above a chunk threshold (typically over 1,000 tokens) go
   through the pipeline.
3. Stream each layer's KVCache to the decode node while prefill continues.
4. Join continuous batching on the decode node. The local scheduler re-checks the TBT SLO, so a
   request can still be rejected after prefill has used resources.

## Key techniques

- **Chunked pipeline parallelism (CPP).** Long prompts are split into chunks processed across a
  group of prefill nodes. Cross-node communication is needed only at stage boundaries, unlike
  tensor or sequence parallelism.
- **Layer-wise prefill.** KVCache loading and storing overlap with computation, so VRAM only has
  to hold one request.
- **Cache-aware prefill scheduling.**
  1. Hash each prompt block chained with the previous block.
  2. Find the longest prefix match on each instance.
  3. Estimate TTFT as queue time + transfer time + prefill time.
  4. Pick the instance with the lowest estimate. If none meets the SLO, return HTTP 429.

  Prefill time comes from an offline-trained predictor.
- **Hot-spot migration.** A heuristic, not a forecast. If a remote prefix match is much longer
  than the local one (by a manually tuned `kvcache_balancing_threshold`), move the cache;
  otherwise recompute. Hot blocks get replicated over time.
- **Early rejection.** Decide at arrival whether to accept, based on the more loaded of the
  prefill and decode pools, to avoid wasted prefill work.
- **Prediction-based early rejection.** Plain early rejection makes prefill and decode load
  oscillate in anti-phase. The fix predicts future decode load, assuming a uniform decode time.

## Evaluation

**Setup:** nodes with 8× A800-80GB (NVLink) and RDMA up to 800 Gbps. A dummy model with the
LLaMA2-70B architecture, with traces replayed. Baseline: vLLM. SLO thresholds are 10× TTFT and
5× TBT of the lowest observed values, with P90 reported.

| Setting | Workload | Result |
|---|---|---|
| vLLM-[4M] vs Mooncake-[3P+1D] | ArXiv Summarization (8,088 in / 229 out, ~0% cache) | +20% throughput |
| Same | L-Eval (19,019 in / 72 out, >80% cache) | +40% throughput |
| Simulated | 16k–128k prompts, 50% cache, Poisson arrivals | +50% to +525% throughput under the same SLOs |
| vLLM-[20M] vs Mooncake-[10P+10D] | Real trace, 23,000 requests (7,955 in / 194 out, ~50% cache) | ~100% of Mooncake requests meet TBT vs 57% for vLLM; ~75% more requests within SLOs |
| 8P+8D, 2× replay | Same trace, overload | Rejections: baseline 4,183; early 3,771; prediction-based 3,589 |

In the overload test, prediction-based rejection rejects about 14% fewer requests than the
baseline. The paper reports rejection counts there, not throughput. The comparison against random
and load-balancing scheduling is reported only qualitatively.

## Limitations

- **Synthetic model.** Results use a dummy model and a trace with only timing and token counts,
  so they can't be checked against real outputs.
- **Limited reuse.** Even with unlimited capacity, only about 50% of KVCache is reusable in their
  online workloads.
- **Static prefill:decode ratio.** A 2P+2D setup showed TTFT problems from imbalance. Dynamic
  conversion is future work.
- **Heuristics.** The migration threshold is hand-tuned, transfer-time prediction suffers from
  network congestion, and the load prediction assumes uniform decode time.
- **Residual waste.** Rejection on the decode side after prefill still wastes prefill compute.
- **Narrow baseline and scale.** Only vLLM is compared, at up to 20 machines. DistServe and
  Splitwise are not compared.

## What to take into a storage or system design interview

- The KVCache is a tiered cache problem: HBM, DRAM, SSD and remote, connected by RDMA. This is the
  inference-side analog of dataset and checkpoint caching.
- Scheduling and cache placement are one problem. Estimate TTFT as queue + transfer + compute
  and route to the minimum.
- Under overload, reject early and cheaply. Work you will throw away costs more than a fast 429.
- Replicate hot blocks on demand instead of pre-placing them.

## Related notes

- [vLLM (SOSP 2023)](vllm-sosp-2023.md): PagedAttention, the block layout Mooncake builds on
- [LMCache (arXiv 2025)](lmcache-arxiv-2025.md): a cache layer you add to an existing engine
