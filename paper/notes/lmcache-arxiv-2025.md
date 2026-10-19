# LMCache: An Efficient KV Cache Layer for Enterprise-Scale LLM Inference

> arXiv preprint (October 2025, not peer reviewed) — structured reading notes

Paper: [arXiv 2510.09665](https://arxiv.org/abs/2510.09665) ·
Code: [LMCache/LMCache](https://github.com/LMCache/LMCache) ·
Docs: [docs.lmcache.ai](https://docs.lmcache.ai) ·
Blog: [blog.lmcache.ai](https://blog.lmcache.ai)

These notes summarize arXiv v1. Check numbers against the paper before quoting them. The
throughput claims are the authors' own.

## Paper information

- **Authors:** Cheng, Liu, Yao et al.
- **Affiliations:** TensorMesh and the University of Chicago
- **Original text:** not vendored in `../original/`; read it at the arXiv link above

## One-sentence summary

LMCache is a KV cache layer that sits between inference engines (vLLM, SGLang) and tiered storage
(GPU, CPU DRAM, disk, remote), making cross-query reuse and prefill-decode disaggregation
practical by moving KV data in large batched chunks and exposing a control API for routers and
operators.

## Problem

- Engines treat each query independently, so shared prefixes are re-prefilled and KV caches are
  discarded when a query ends.
- Cross-query prefix caching and prefill-decode (PD) disaggregation both depend on moving KV cache
  efficiently across GPU, CPU, storage and network. Three obstacles:
  1. **I/O inefficiency under paged memory.** vLLM and SGLang use small pages (about 20–63 KB in
     the paper's examples). The paper's Table 1 shows throughput rising from 4 GB/s at 64 KB to
     about 49 GB/s at 16 MB.
  2. **Fast-changing engines.** New models and engine changes can alter KV layout.
  3. **No management APIs.** Routers and operators can't locate, pin, move or compress caches.

## Architecture

- **Position:** a layer between the engine and heterogeneous storage and network devices.
- **Tiers:** GPU memory, CPU DRAM, local disk, remote disk and Redis. Transports include
  Ethernet, RDMA and NVLink.
- **Data plane (LMCache worker):** one per engine instance. Handles offload to CPU or disk and
  GPU-to-GPU transfer for PD disaggregation.
- **Control plane (LMCache controller):** a central manager that dispatches operations to workers
  and keeps a token pool recording what is stored where, exposed as a virtualized namespace.

**Connector interface** (what the engine calls)

| Side | Calls |
|---|---|
| Scheduler | `get_num_new_matched_tokens`, `update_state_after_alloc`, `build_connector_meta` |
| Model runner | `start_load_kv`, `wait_load_kv`, `start_store_kv`, `wait_store_kv` |

**Controller API:** `lookup`, `query_ip`, `move`, `clear`, `pin`, `compress`.

**Example uses:** KV-aware routing (lookup, map to instance, forward to the one with most hits),
migration on failure or imbalance, clearing on model switch, pinning system prompts.

## Key techniques

- **Chunking.** Pages are grouped into chunks (default 256 tokens), staged in a GPU buffer and
  copied with a custom CUDA kernel.
- **Batched store and load.** Operations across tiers are batched so CPU and disk transfers run in
  parallel.
- **Delayed decode storing.** New decode-time pages are aggregated before writing.
- **Layer-wise pipelining.** Separate CUDA streams load layer *n+1* while layer *n* computes, so
  only a one-layer GPU buffer is needed.
- **Asynchronous prefetch.** Queued queries' KV is promoted from slow tiers to fast ones while
  they wait.
- **Process separation.** Data movement runs in its own process, avoiding 5–10% contention in the
  engine and allowing a CPU memory pool shared across instances.
- **Zero-copy.** A reference counter lets several destinations share one buffer.
- **Dynamic offloading.** Only part of the free GPU page pool is mirrored to CPU. The trade-off is
  duplication overhead against allocation stalls.

## Evaluation

**Setup:** 8×H100 for most runs, B200 for context-length tests. Models include Llama-3.1 8B and
70B, Qwen2.5 32B and 72B, and Qwen3-Coder-480B. Baselines are vLLM with prefix caching and three
anonymized commercial APIs.

| Scenario | Setup | Result |
|---|---|---|
| CPU offload | Multi-round QA, 10K-token contexts, 500 GB CPU cap | 2.3–14× throughput at equal TTFT (at QPS=1) |
| Central storage | TriviaQA (LongBench), 15 Gbps link | 1.3–3× throughput |
| PD disaggregation | 8K in / 200 out, NVLink | Mean TTFT down 1.53–1.84×; mean ITL down 1.12–1.66× |
| Real trace | About 4K average input | 25% better at QPS 2–5; 49% better at QPS 6 vs naive vLLM |
| Load bandwidth | From CPU | 400 Gbps vs 88 Gbps for vLLM native offload |
| Async overlap | Load/compute overlap | 1.46× end-to-end delay reduction |
| Context sensitivity | B200 | At 32 Gbps, loading beats prefill only above 256K tokens; at 64–128 Gbps it wins at all lengths |
| SGLang | Qwen3-32B, 2×H100 | Better than no offload; comparable to SGLang's native offload |

The headline claim is up to 15× throughput and at least 2× lower latency.

## Deployment lessons (section 8)

- Tiered storage is necessary. Some teams found remote fetches beat recomputation once pipelined.
- Hybrid use works, such as CPU offload plus PD disaggregation on the prefiller.
- Keep the full history. Sliding windows hurt quality and break prefix matching.
- Lossy compression is often acceptable. Quantized KV caches were tolerated in some chatbot and
  finance settings.
- A failed cache read reports what was read, so it does not crash the engine.
- Python for integration and community contribution, with CUDA kernels only on hot paths.

## Limitations

- Assumes input and output token counts match, which rules out token-dropping compression.
- Intermediate attention states are hidden from LMCache.
- Remote loading can be slower than prefill for short contexts, small models or low bandwidth.
- The real trace was replayed on a substitute open model, and commercial baselines were black
  boxes. The SGLang comparison covers one configuration.
- Full decoupling of the engine and the cache layer is not solved.
- The paper doesn't cover non-prefix reuse (CacheBlend) or a named peer-to-peer sharing protocol.
  It describes a shared CPU pool and controller `move`.

## What to take into a storage or system design interview

- Large transfers win. Chunking turns 20–60 KB pages into multi-MB transfers.
- Overlap data movement with compute (layer-wise pipelining) so cache hits don't add latency.
- Loading from a remote tier only beats recompute above a size or bandwidth threshold. Make that
  decision adaptive.
- A control API (lookup, pin, move) is what turns a cache into something a router can use.

## Related notes

- [Mooncake (FAST 2025)](mooncake-fast-2025.md): the system-level design with a global scheduler
- [vLLM (SOSP 2023)](vllm-sosp-2023.md): PagedAttention and the small-page layout LMCache works around
