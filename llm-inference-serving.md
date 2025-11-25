# High-Throughput LLM Inference Serving System

## Table of Contents
1. [Problem Statement](#problem-statement)
2. [Requirements Analysis](#requirements-analysis)
3. [LLM Inference Fundamentals](#llm-inference-fundamentals)
4. [System Architecture](#system-architecture)
5. [Optimization Techniques](#optimization-techniques)
6. [Hardware Considerations](#hardware-considerations)
7. [Scalability Strategies](#scalability-strategies)
8. [Cost Optimization](#cost-optimization)
9. [Production Considerations](#production-considerations)
10. [Framework Comparison](#framework-comparison)
11. [Performance Benchmarks](#performance-benchmarks)

## Problem Statement

**Goal:** Design a production system to serve Large Language Model (LLM) inference requests with high throughput, low latency, and cost efficiency.

**Challenges:**
- **Model Size**: 7B - 175B+ parameters (14GB - 350GB+ in memory)
- **Latency**: User-facing applications need < 1s for short responses
- **Throughput**: Handle thousands of concurrent requests
- **Cost**: GPU inference is expensive ($2-10 per million tokens)
- **Variable Input/Output**: Requests vary from 100 to 4000+ tokens
- **Memory Constraints**: KV cache grows with sequence length
- **Batching Complexity**: Different requests have different lengths

**Example Use Cases:**
- Chatbots and conversational AI
- Code generation and completion
- Content generation and summarization
- Question answering systems
- Real-time translation

## Requirements Analysis

### Functional Requirements
- **Model Support**: Serve models from 7B to 70B+ parameters
- **Streaming**: Support token-by-token streaming responses
- **Multi-tenancy**: Serve multiple models simultaneously
- **API Compatibility**: OpenAI-compatible API
- **Prompt Management**: Support system prompts, few-shot examples
- **Safety**: Content filtering, rate limiting
- **Monitoring**: Request metrics, token usage tracking

### Non-Functional Requirements

**Scale Assumptions:**
```
Model: Llama-2-70B (70B parameters)
Precision: FP16 (2 bytes per parameter)
Model Size: 140 GB
Context Window: 4096 tokens
Target QPS: 100 queries per second
Average Input Tokens: 500
Average Output Tokens: 200
Target Latency:
  - Time to First Token (TTFT): < 500ms
  - Inter-Token Latency: < 50ms
  - Total Time (200 tokens): < 10s
Availability: 99.9% uptime
```

**Performance Targets:**
- **Throughput**: 100+ QPS for production workload
- **Latency**:
  - P50 TTFT: < 300ms
  - P95 TTFT: < 500ms
  - P99 TTFT: < 1000ms
- **GPU Utilization**: > 70%
- **Token Throughput**: 10,000+ tokens/second per GPU

## LLM Inference Fundamentals

### Autoregressive Generation

**How LLMs Generate Text:**

```python
# Simplified generation loop
def generate(model, prompt, max_tokens=100):
    # 1. Encode prompt
    input_ids = tokenizer.encode(prompt)

    # 2. Prefill phase: Process entire prompt
    # Compute attention for all input tokens
    past_key_values = None
    outputs = model(input_ids, past_key_values=past_key_values)
    next_token = outputs.logits[:, -1, :].argmax(dim=-1)
    past_key_values = outputs.past_key_values  # Cache KV states

    generated_tokens = [next_token.item()]

    # 3. Decode phase: Generate one token at a time
    for _ in range(max_tokens - 1):
        # Only process the new token, reuse cached KV
        outputs = model(
            next_token.unsqueeze(0),
            past_key_values=past_key_values
        )
        next_token = outputs.logits[:, -1, :].argmax(dim=-1)
        past_key_values = outputs.past_key_values  # Update cache

        generated_tokens.append(next_token.item())

        if next_token == tokenizer.eos_token_id:
            break

    return tokenizer.decode(generated_tokens)
```

**Two Phases:**

1. **Prefill Phase** (Prompt Processing):
   - Process all input tokens in parallel
   - Compute attention across entire prompt
   - High compute utilization (matrix multiplication)
   - Latency: ~100-500ms for 500 tokens

2. **Decode Phase** (Token Generation):
   - Generate one token at a time (autoregressive)
   - Reuse KV cache from previous tokens
   - Memory-bandwidth bound (not compute-bound)
   - Latency: ~20-50ms per token

### KV Cache

**Purpose:** Avoid recomputing attention for previous tokens.

**Memory Calculation:**
```
For Llama-70B with 4096 context:
  - Layers: 80
  - Attention heads: 64
  - Head dimension: 128
  - Precision: FP16 (2 bytes)

KV Cache per token = 2 (K+V) × 80 layers × 64 heads × 128 dim × 2 bytes
                   = 2 × 80 × 64 × 128 × 2
                   = 2,621,440 bytes
                   ≈ 2.5 MB per token

For 4096 tokens: 2.5 MB × 4096 = 10.24 GB per request
```

**Memory Bottleneck:**
- Model weights: 140 GB (Llama-70B in FP16)
- KV cache for batch size 32 with 4096 tokens: 327 GB
- **Total: 467 GB** (exceeds single A100 80GB capacity!)

**Solution:** Efficient KV cache management (PagedAttention, continuous batching)

### Compute Characteristics

**Prefill (Prompt Processing):**
- **Compute-bound**: Matrix multiplications dominate
- **GPU utilization**: High (70-90%)
- **Parallelizable**: Process multiple tokens simultaneously
- **Latency**: Scales with prompt length

**Decode (Token Generation):**
- **Memory-bandwidth bound**: Loading model weights dominates
- **GPU utilization**: Low (20-40%) with naive batching
- **Sequential**: Generate one token at a time per sequence
- **Latency**: Constant per token (~50ms)

**Key Insight:**
- Decode phase is underutilized in terms of compute
- Opportunity for batching multiple requests
- Continuous batching can dramatically improve throughput

## System Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      API Gateway / Load Balancer                 │
│                         (Rate Limiting)                          │
└────────────────────┬────────────────────────────────────────────┘
                     │
        ┌────────────┴─────────────┐
        │                          │
┌───────▼──────────┐        ┌──────▼─────────────┐
│  Request Router  │        │  Model Registry    │
│  & Queue Manager │        │  (Model Versions)  │
└───────┬──────────┘        └────────────────────┘
        │
        │     ┌──────────────────────────────────┐
        │     │                                  │
        ├────→│  Inference Engine Pool (vLLM)   │
        │     │  ┌────────────┬────────────┐    │
        │     │  │  GPU 0     │  GPU 1     │    │
        │     │  │ Model A    │ Model A    │    │
        │     │  │ (Replica)  │ (Replica)  │    │
        │     │  └────────────┴────────────┘    │
        │     └──────────────────────────────────┘
        │
        │     ┌──────────────────────────────────┐
        └────→│  Inference Engine Pool (TGI)    │
              │  ┌────────────┬────────────┐    │
              │  │  GPU 2-5   │  GPU 6-9   │    │
              │  │ Model B    │ Model B    │    │
              │  │ (4-way TP) │ (4-way TP) │    │
              │  └────────────┴────────────┘    │
              └──────────────────────────────────┘
                        │
                        ↓
              ┌──────────────────┐
              │  Response Cache  │
              │     (Redis)      │
              └──────────────────┘
                        │
                        ↓
              ┌──────────────────┐
              │   Monitoring &   │
              │    Metrics DB    │
              │  (Prometheus)    │
              └──────────────────┘
```

### Component Details

#### 1. API Gateway

**Responsibilities:**
- Request authentication and authorization
- Rate limiting per user/API key
- Request validation and preprocessing
- Response formatting
- SSL termination

**Implementation:**
```python
from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel
from typing import Optional, List
import asyncio

app = FastAPI()

class CompletionRequest(BaseModel):
    prompt: str
    max_tokens: int = 100
    temperature: float = 0.7
    top_p: float = 0.9
    stream: bool = False
    stop: Optional[List[str]] = None

class CompletionResponse(BaseModel):
    id: str
    choices: List[dict]
    usage: dict
    model: str

# Rate limiter
rate_limiter = RateLimiter(
    requests_per_minute=60,
    tokens_per_minute=100000
)

@app.post("/v1/completions")
async def create_completion(
    request: CompletionRequest,
    authorization: str = Header(...)
):
    # 1. Authenticate
    api_key = authorization.replace("Bearer ", "")
    user = authenticate(api_key)

    # 2. Rate limit
    if not rate_limiter.allow(user.id, estimate_tokens(request.prompt)):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")

    # 3. Route to inference engine
    if request.stream:
        return StreamingResponse(
            stream_completion(request),
            media_type="text/event-stream"
        )
    else:
        result = await inference_router.complete(request)
        return result

async def stream_completion(request):
    """Stream tokens as they are generated"""
    async for token in inference_router.stream(request):
        yield f"data: {json.dumps(token)}\n\n"
    yield "data: [DONE]\n\n"
```

**Rate Limiting:**
```python
from datetime import datetime, timedelta
from collections import defaultdict

class RateLimiter:
    def __init__(self, requests_per_minute=60, tokens_per_minute=100000):
        self.rpm_limit = requests_per_minute
        self.tpm_limit = tokens_per_minute
        self.request_counts = defaultdict(list)
        self.token_counts = defaultdict(list)

    def allow(self, user_id: str, estimated_tokens: int) -> bool:
        now = datetime.now()
        minute_ago = now - timedelta(minutes=1)

        # Clean old entries
        self.request_counts[user_id] = [
            t for t in self.request_counts[user_id] if t > minute_ago
        ]
        self.token_counts[user_id] = [
            (t, tokens) for t, tokens in self.token_counts[user_id]
            if t > minute_ago
        ]

        # Check limits
        requests_in_window = len(self.request_counts[user_id])
        tokens_in_window = sum(tokens for _, tokens in self.token_counts[user_id])

        if requests_in_window >= self.rpm_limit:
            return False
        if tokens_in_window + estimated_tokens > self.tpm_limit:
            return False

        # Record
        self.request_counts[user_id].append(now)
        self.token_counts[user_id].append((now, estimated_tokens))

        return True
```

#### 2. Request Router & Queue Manager

**Responsibilities:**
- Route requests to appropriate model endpoints
- Queue management and prioritization
- Load balancing across replicas
- Request batching coordination

**Implementation:**
```python
import asyncio
from collections import deque
from typing import List

class RequestRouter:
    def __init__(self, inference_engines: List['InferenceEngine']):
        self.engines = inference_engines
        self.request_queue = asyncio.Queue()
        self.routing_strategy = "least_loaded"  # or "round_robin"

    async def complete(self, request: CompletionRequest):
        # Select engine based on strategy
        engine = self._select_engine()

        # Submit request
        result = await engine.generate(request)
        return result

    def _select_engine(self) -> 'InferenceEngine':
        if self.routing_strategy == "least_loaded":
            # Route to engine with shortest queue
            return min(self.engines, key=lambda e: e.queue_size())
        elif self.routing_strategy == "round_robin":
            # Simple round-robin
            engine = self.engines[self.current_index]
            self.current_index = (self.current_index + 1) % len(self.engines)
            return engine

    async def stream(self, request: CompletionRequest):
        engine = self._select_engine()
        async for token in engine.generate_stream(request):
            yield token
```

**Priority Queue:**
```python
import heapq
from dataclasses import dataclass, field
from typing import Any

@dataclass(order=True)
class PrioritizedRequest:
    priority: int
    timestamp: float = field(compare=False)
    request: Any = field(compare=False)

class PriorityRequestQueue:
    def __init__(self):
        self.heap = []
        self.counter = 0

    def add(self, request, priority=0):
        """Lower priority number = higher priority"""
        item = PrioritizedRequest(
            priority=priority,
            timestamp=time.time(),
            request=request
        )
        heapq.heappush(self.heap, item)

    def get(self):
        if self.heap:
            return heapq.heappop(self.heap).request
        return None

    def size(self):
        return len(self.heap)
```

#### 3. Inference Engine

**Core serving engine (vLLM, TGI, TensorRT-LLM)**

**vLLM Example:**
```python
from vllm import LLM, SamplingParams
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.utils import random_uuid

class VLLMInferenceEngine:
    def __init__(self, model_name: str, gpu_memory_utilization: float = 0.9):
        self.model_name = model_name

        # Initialize async engine with continuous batching
        self.engine = AsyncLLMEngine.from_engine_args(
            EngineArgs(
                model=model_name,
                gpu_memory_utilization=gpu_memory_utilization,
                max_num_batched_tokens=4096,
                max_num_seqs=256,  # Max concurrent sequences
                tensor_parallel_size=1,
                dtype="float16"
            )
        )

    async def generate(self, request: CompletionRequest) -> CompletionResponse:
        request_id = random_uuid()

        # Configure sampling parameters
        sampling_params = SamplingParams(
            temperature=request.temperature,
            top_p=request.top_p,
            max_tokens=request.max_tokens,
            stop=request.stop
        )

        # Submit to engine (continuous batching happens internally)
        results_generator = self.engine.generate(
            request.prompt,
            sampling_params,
            request_id
        )

        # Collect all tokens
        final_output = None
        async for output in results_generator:
            final_output = output

        # Format response
        return CompletionResponse(
            id=request_id,
            choices=[{
                "text": final_output.outputs[0].text,
                "finish_reason": final_output.outputs[0].finish_reason,
                "index": 0
            }],
            usage={
                "prompt_tokens": len(final_output.prompt_token_ids),
                "completion_tokens": len(final_output.outputs[0].token_ids),
                "total_tokens": len(final_output.prompt_token_ids) +
                                len(final_output.outputs[0].token_ids)
            },
            model=self.model_name
        )

    async def generate_stream(self, request: CompletionRequest):
        """Stream tokens as they are generated"""
        request_id = random_uuid()

        sampling_params = SamplingParams(
            temperature=request.temperature,
            top_p=request.top_p,
            max_tokens=request.max_tokens,
            stop=request.stop
        )

        results_generator = self.engine.generate(
            request.prompt,
            sampling_params,
            request_id
        )

        # Stream tokens
        prev_text = ""
        async for output in results_generator:
            current_text = output.outputs[0].text
            new_text = current_text[len(prev_text):]
            prev_text = current_text

            yield {
                "id": request_id,
                "choices": [{
                    "text": new_text,
                    "index": 0,
                    "finish_reason": output.outputs[0].finish_reason
                }]
            }

    def queue_size(self) -> int:
        return self.engine.get_num_unfinished_requests()
```

#### 4. Model Registry

**Track available models and their configurations:**

```python
from dataclasses import dataclass
from typing import Dict, Optional

@dataclass
class ModelConfig:
    name: str
    path: str
    tensor_parallel_size: int
    max_model_len: int
    gpu_memory_utilization: float
    quantization: Optional[str] = None  # "awq", "gptq", None
    endpoints: List[str] = None  # Available serving endpoints

class ModelRegistry:
    def __init__(self):
        self.models: Dict[str, ModelConfig] = {}
        self.load_configs()

    def load_configs(self):
        # Load from config file or database
        self.models = {
            "llama-2-7b": ModelConfig(
                name="llama-2-7b",
                path="/models/llama-2-7b-hf",
                tensor_parallel_size=1,
                max_model_len=4096,
                gpu_memory_utilization=0.9,
                endpoints=["http://inference-0:8000", "http://inference-1:8000"]
            ),
            "llama-2-70b": ModelConfig(
                name="llama-2-70b",
                path="/models/llama-2-70b-hf",
                tensor_parallel_size=4,
                max_model_len=4096,
                gpu_memory_utilization=0.9,
                quantization="awq",
                endpoints=["http://inference-2:8000"]
            )
        }

    def get_model(self, name: str) -> Optional[ModelConfig]:
        return self.models.get(name)

    def list_models(self) -> List[str]:
        return list(self.models.keys())
```

## Optimization Techniques

### 1. Continuous Batching (PagedAttention)

**Problem with Static Batching:**
```
Batch = [Request A (100 tokens), Request B (500 tokens), Request C (50 tokens)]

With static batching:
- Must wait for longest request (B) to finish
- Requests A and C idle after completion
- GPU underutilized
- Batch throughput = 500 iterations (limited by slowest)
```

**Continuous Batching Solution:**
```
Iteration 1: Process [A, B, C]
Iteration 50: A finishes → Replace with new request D → Process [D, B, C]
Iteration 100: D finishes → Replace with E → Process [E, B, C]
...
Iteration 500: B finishes

Result:
- No idle time
- Continuous GPU utilization
- Much higher throughput
```

**PagedAttention (vLLM):**

**Concept:**
- KV cache divided into fixed-size blocks (pages)
- Pages allocated on-demand (like virtual memory)
- Enables non-contiguous memory allocation
- Reduces fragmentation and memory waste

**Benefits:**
```
Traditional: Each sequence needs contiguous memory for max_len
- Max 32 sequences with 4096 tokens each = 327 GB

PagedAttention: Allocate blocks as needed
- 200+ concurrent sequences possible
- ~5x memory efficiency improvement
- Enables much larger batch sizes
```

**Implementation Concept:**
```python
class PagedAttentionKVCache:
    def __init__(self, block_size=16, num_blocks=1000):
        self.block_size = block_size  # tokens per block
        self.num_blocks = num_blocks

        # Physical KV cache: [num_blocks, block_size, num_layers, num_heads, head_dim]
        self.physical_cache = torch.zeros(
            num_blocks, block_size, num_layers, num_heads, head_dim
        )

        # Block allocation table
        self.free_blocks = set(range(num_blocks))
        self.sequence_blocks = {}  # seq_id -> [block_ids]

    def allocate_blocks(self, seq_id, num_tokens):
        """Allocate blocks for a sequence"""
        num_blocks_needed = (num_tokens + self.block_size - 1) // self.block_size

        if len(self.free_blocks) < num_blocks_needed:
            raise MemoryError("Not enough free blocks")

        allocated = []
        for _ in range(num_blocks_needed):
            block_id = self.free_blocks.pop()
            allocated.append(block_id)

        self.sequence_blocks[seq_id] = allocated
        return allocated

    def free_sequence(self, seq_id):
        """Free blocks when sequence completes"""
        if seq_id in self.sequence_blocks:
            blocks = self.sequence_blocks[seq_id]
            self.free_blocks.update(blocks)
            del self.sequence_blocks[seq_id]
```

### 2. Quantization

**Reduce memory and computation by using lower precision:**

**Precision Options:**
- **FP32**: 4 bytes (baseline, not used for LLMs)
- **FP16/BF16**: 2 bytes (standard for inference)
- **INT8**: 1 byte (2x memory reduction)
- **INT4**: 0.5 bytes (4x memory reduction)

**Quantization Methods:**

**A. Post-Training Quantization (PTQ):**

```python
# Example: Load quantized model with bitsandbytes
from transformers import AutoModelForCausalLM
import torch

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-70b-hf",
    device_map="auto",
    load_in_8bit=True,  # INT8 quantization
    # or load_in_4bit=True for INT4
)

# INT4 with NormalFloat (NF4) quantization
from transformers import BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,  # Nested quantization
    bnb_4bit_quant_type="nf4"  # NormalFloat quantization
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-70b-hf",
    quantization_config=quantization_config,
    device_map="auto"
)
```

**B. GPTQ (Accurate Post-Training Quantization):**

```python
# Load GPTQ quantized model
from auto_gptq import AutoGPTQForCausalLM

model = AutoGPTQForCausalLM.from_quantized(
    "TheBloke/Llama-2-70B-GPTQ",
    device="cuda:0",
    use_safetensors=True,
    use_triton=True  # Use Triton kernels for speedup
)
```

**C. AWQ (Activation-aware Weight Quantization):**

```python
# AWQ preserves important weights
from awq import AutoAWQForCausalLM

model = AutoAWQForCausalLM.from_quantized(
    "TheBloke/Llama-2-70B-AWQ",
    fuse_layers=True  # Fuse layers for speedup
)
```

**Memory Savings:**
```
Llama-70B:
- FP16: 140 GB
- INT8: 70 GB (2x reduction)
- INT4 (GPTQ/AWQ): 35 GB (4x reduction)

Result: 70B model fits on single A100 80GB GPU with INT4!
```

**Performance Impact:**
```
Precision | Throughput | Quality Loss
----------|------------|-------------
FP16      | 100%       | 0%
INT8      | 120%       | <1%
INT4-GPTQ | 150%       | 2-3%
INT4-AWQ  | 150%       | 1-2%
```

### 3. Model Parallelism

**For models too large for single GPU:**

**A. Tensor Parallelism (Intra-layer):**

Split individual layers across multiple GPUs:

```
Single GPU:
[Attention Layer (140GB)] → Too large!

Tensor Parallel (4 GPUs):
GPU 0: [Attention heads 0-15]
GPU 1: [Attention heads 16-31]
GPU 2: [Attention heads 32-47]
GPU 3: [Attention heads 48-63]

All-reduce after each layer
```

**Implementation with vLLM:**
```python
from vllm import LLM

# 4-way tensor parallelism
llm = LLM(
    model="meta-llama/Llama-2-70b-hf",
    tensor_parallel_size=4,  # Split across 4 GPUs
    dtype="float16"
)
```

**B. Pipeline Parallelism (Inter-layer):**

Split layers across GPUs:

```
GPU 0: Layers 0-19
GPU 1: Layers 20-39
GPU 2: Layers 40-59
GPU 3: Layers 60-79

Sequential processing (lower utilization)
```

**C. Sequence Parallelism:**

Split along sequence dimension (used with tensor parallelism):

```python
# Combine tensor + pipeline parallelism
from vllm import LLM

llm = LLM(
    model="meta-llama/Llama-2-70b-hf",
    tensor_parallel_size=4,
    pipeline_parallel_size=2,  # 8 GPUs total
    dtype="float16"
)
```

**Trade-offs:**
```
Strategy              | GPUs | Communication | Efficiency
----------------------|------|---------------|------------
Tensor Parallel (TP)  | 2-8  | High (all-reduce) | 85-95%
Pipeline Parallel (PP)| 2-16 | Low (P2P)     | 60-75%
TP + PP              | 8+   | Medium        | 70-85%
```

### 4. Flash Attention

**Problem:** Standard attention has O(N²) memory complexity

**Flash Attention:**
- Fused attention kernels
- Tiled computation (block by block)
- Reduces HBM memory reads/writes
- 2-4x speedup for long sequences

**Usage:**
```python
# Automatically used in vLLM and modern frameworks
from vllm import LLM

llm = LLM(
    model="meta-llama/Llama-2-7b-hf",
    # Flash Attention 2 automatically used if available
)
```

**Performance:**
```
Sequence Length | Standard Attention | Flash Attention | Speedup
----------------|-------------------|-----------------|--------
512             | 100ms             | 80ms            | 1.25x
2048            | 450ms             | 200ms           | 2.25x
4096            | 1800ms            | 500ms           | 3.6x
```

### 5. Speculative Decoding

**Idea:** Use small "draft" model to predict multiple tokens, verify with large model

**Algorithm:**
```python
def speculative_decoding(large_model, small_model, prompt, k=4):
    """
    k = number of tokens to speculate
    """
    tokens = tokenize(prompt)

    while not done:
        # 1. Small model generates k candidate tokens (fast)
        candidates = small_model.generate(tokens, num_tokens=k)

        # 2. Large model verifies in parallel (single forward pass)
        logits = large_model(torch.cat([tokens, candidates]))

        # 3. Accept tokens while predictions match
        accepted = 0
        for i in range(k):
            predicted_token = logits[len(tokens) + i].argmax()
            if predicted_token == candidates[i]:
                accepted += 1
                tokens.append(candidates[i])
            else:
                # Reject rest, sample from large model logits
                tokens.append(predicted_token)
                break

        # Result: Accept 1-k tokens per iteration instead of 1
```

**Speedup:**
```
If 50% of speculations accepted (k=4):
- Average accepted: 2 tokens per iteration
- Speedup: ~2x for decode phase
- Total speedup: ~1.5x (decode is ~70% of time)
```

**Trade-off:**
- Requires running two models
- Speedup depends on draft model quality
- Works best when draft model is similar to target

### 6. KV Cache Compression

**Techniques to reduce KV cache memory:**

**A. Multi-Query Attention (MQA):**
- Share K/V across attention heads
- Reduce KV cache by ~8x

**B. Grouped-Query Attention (GQA):**
- Share K/V across groups of heads
- Llama-2 uses GQA (middle ground)

**C. Streaming LLM:**
- Keep only recent tokens + attention sinks
- Discard middle tokens for very long contexts

## Hardware Considerations

### GPU Selection

**Common GPU Options:**

| GPU | Memory | FP16 TFLOPS | Memory BW | Cost | Best For |
|-----|--------|-------------|-----------|------|----------|
| A10 | 24 GB | 125 | 600 GB/s | $1.50/hr | 7B models |
| L4 | 24 GB | 121 | 300 GB/s | $1.00/hr | 7B inference |
| A100 40GB | 40 GB | 312 | 1555 GB/s | $3.00/hr | 13B models |
| A100 80GB | 80 GB | 312 | 2000 GB/s | $4.00/hr | 70B quantized |
| H100 | 80 GB | 989 | 3350 GB/s | $8.00/hr | 70B, fastest |

**Model Fit Analysis:**

```python
def estimate_gpu_memory(
    num_params_billions: float,
    precision: str = "fp16",
    batch_size: int = 1,
    max_seq_len: int = 4096,
    num_layers: int = 80,
    num_heads: int = 64,
    head_dim: int = 128
):
    """Estimate GPU memory requirements"""

    # Model weights
    bytes_per_param = {
        "fp32": 4,
        "fp16": 2,
        "int8": 1,
        "int4": 0.5
    }[precision]

    model_memory_gb = (num_params_billions * 1e9 * bytes_per_param) / 1e9

    # KV cache
    kv_cache_per_token = 2 * num_layers * num_heads * head_dim * 2  # 2 bytes (FP16)
    kv_cache_gb = (batch_size * max_seq_len * kv_cache_per_token) / 1e9

    # Activation memory (rough estimate)
    activation_gb = (batch_size * max_seq_len * num_layers * 4096 * 2) / 1e9

    # Overhead (10%)
    overhead_gb = (model_memory_gb + kv_cache_gb + activation_gb) * 0.1

    total_gb = model_memory_gb + kv_cache_gb + activation_gb + overhead_gb

    return {
        "model": model_memory_gb,
        "kv_cache": kv_cache_gb,
        "activations": activation_gb,
        "overhead": overhead_gb,
        "total": total_gb
    }

# Example: Llama-70B
memory = estimate_gpu_memory(
    num_params_billions=70,
    precision="int4",
    batch_size=32,
    max_seq_len=2048
)

print(f"Model: {memory['model']:.1f} GB")        # 35 GB
print(f"KV Cache: {memory['kv_cache']:.1f} GB")  # 82 GB
print(f"Total: {memory['total']:.1f} GB")        # ~130 GB
# Requires 2x A100 80GB with tensor parallelism!
```

### Multi-GPU Setup

**Network Topology:**

```
NVLink/NVSwitch (preferred for tensor parallelism):
  - Bandwidth: 600 GB/s between GPUs
  - Latency: ~2-5 μs
  - Best for: Tensor parallelism (tight coupling)

PCIe 4.0:
  - Bandwidth: 64 GB/s
  - Latency: ~10-20 μs
  - Best for: Pipeline parallelism, data parallelism

InfiniBand/RoCE (multi-node):
  - Bandwidth: 200-400 Gb/s (25-50 GB/s)
  - Latency: ~1-5 μs
  - Best for: Distributed inference across nodes
```

**Recommended Configurations:**

```
7B models:
  - 1x A10 (24GB) or 1x L4 (24GB)
  - No parallelism needed
  - Cost: ~$1/hour

13B models:
  - 1x A100 40GB
  - FP16 or INT8
  - Cost: ~$3/hour

70B models (Option 1):
  - 2x A100 80GB with tensor parallelism
  - INT4 quantization
  - Cost: ~$8/hour

70B models (Option 2):
  - 4x A100 40GB with tensor parallelism
  - FP16
  - Cost: ~$12/hour
```

## Scalability Strategies

### Horizontal Scaling

**Replica-based Scaling:**

```
                    Load Balancer
                         │
        ┌────────────────┼────────────────┐
        │                │                │
    ┌───▼───┐       ┌────▼───┐      ┌────▼───┐
    │ Rep 0 │       │ Rep 1  │      │ Rep 2  │
    │ GPU 0 │       │ GPU 1  │      │ GPU 2  │
    │ 7B    │       │ 7B     │      │ 7B     │
    └───────┘       └────────┘      └────────┘

Each replica is independent
Linear scaling for throughput
No inter-GPU communication
```

**Auto-scaling Policy:**

```python
class AutoScaler:
    def __init__(self, min_replicas=2, max_replicas=10):
        self.min_replicas = min_replicas
        self.max_replicas = max_replicas
        self.current_replicas = min_replicas

    async def scale_decision(self, metrics):
        """Decide whether to scale up or down"""

        # Metrics to consider
        avg_queue_length = metrics['avg_queue_length']
        avg_gpu_utilization = metrics['avg_gpu_utilization']
        p95_latency = metrics['p95_latency_ms']

        # Scale up conditions
        if (avg_queue_length > 50 or
            p95_latency > 1000 or
            avg_gpu_utilization > 0.85):
            if self.current_replicas < self.max_replicas:
                await self.scale_up()

        # Scale down conditions
        elif (avg_queue_length < 10 and
              p95_latency < 500 and
              avg_gpu_utilization < 0.4):
            if self.current_replicas > self.min_replicas:
                await self.scale_down()

    async def scale_up(self):
        """Add a new replica"""
        new_replica_id = self.current_replicas

        # Launch new instance (Kubernetes pod, EC2 instance, etc.)
        await k8s_client.create_pod(
            name=f"inference-{new_replica_id}",
            image="vllm:latest",
            gpu_count=1,
            env={
                "MODEL_NAME": "meta-llama/Llama-2-7b-hf",
                "TENSOR_PARALLEL_SIZE": "1"
            }
        )

        self.current_replicas += 1
        logger.info(f"Scaled up to {self.current_replicas} replicas")

    async def scale_down(self):
        """Remove a replica"""
        replica_to_remove = self.current_replicas - 1

        # Graceful shutdown: wait for in-flight requests
        await self.drain_replica(replica_to_remove)

        # Remove instance
        await k8s_client.delete_pod(f"inference-{replica_to_remove}")

        self.current_replicas -= 1
        logger.info(f"Scaled down to {self.current_replicas} replicas")
```

### Request Routing Strategies

**1. Least-Loaded Routing:**
```python
def route_request(request, replicas):
    # Route to replica with shortest queue
    return min(replicas, key=lambda r: r.queue_length)
```

**2. Latency-aware Routing:**
```python
def route_request(request, replicas):
    # Estimate latency for each replica
    estimates = []
    for replica in replicas:
        estimated_latency = (
            replica.avg_latency *
            (1 + replica.queue_length / replica.capacity)
        )
        estimates.append((estimated_latency, replica))

    return min(estimates, key=lambda x: x[0])[1]
```

**3. Prompt-length Aware Routing:**
```python
def route_request(request, replicas):
    # Route long prompts to less-loaded replicas
    prompt_length = len(request.prompt.split())

    if prompt_length > 1000:  # Long prompt
        # Prefer less loaded replicas
        return min(replicas, key=lambda r: r.queue_length)
    else:  # Short prompt
        # Any replica is fine (round-robin)
        return replicas[current_index % len(replicas)]
```

### Caching Strategies

**1. Prompt Prefix Caching:**

**Idea:** Cache KV states for common prompt prefixes

```python
class PrefixCache:
    def __init__(self, max_size_gb=10):
        self.cache = {}  # prefix_hash -> KV tensors
        self.max_size = max_size_gb * 1e9
        self.current_size = 0

    def get(self, prefix: str):
        """Get cached KV states for prefix"""
        prefix_hash = hash(prefix)
        return self.cache.get(prefix_hash)

    def put(self, prefix: str, kv_states):
        """Cache KV states for prefix"""
        prefix_hash = hash(prefix)

        # Estimate size
        kv_size = kv_states[0].element_size() * kv_states[0].nelement()

        # Evict if necessary (LRU)
        while self.current_size + kv_size > self.max_size and self.cache:
            oldest = min(self.cache.items(), key=lambda x: x[1]['timestamp'])
            del self.cache[oldest[0]]
            self.current_size -= oldest[1]['size']

        # Add to cache
        self.cache[prefix_hash] = {
            'kv_states': kv_states,
            'size': kv_size,
            'timestamp': time.time()
        }
        self.current_size += kv_size

# Usage
prefix = "You are a helpful assistant. User: "
cached_kv = prefix_cache.get(prefix)

if cached_kv:
    # Reuse cached computation
    # Only process new tokens after prefix
    new_prompt = prefix + user_input
    outputs = model.generate(new_prompt, past_key_values=cached_kv)
else:
    # Full computation
    outputs = model.generate(prefix + user_input)
    # Cache for future requests
    prefix_cache.put(prefix, outputs.past_key_values)
```

**Benefits:**
- 50-90% reduction in prefill time for requests with common prefixes
- Common use case: System prompts, few-shot examples

**2. Response Caching:**

```python
class ResponseCache:
    def __init__(self, ttl=3600):
        self.cache = {}
        self.ttl = ttl

    def get_cache_key(self, prompt, params):
        """Generate cache key from prompt and parameters"""
        # Include temperature=0 deterministic requests only
        if params.temperature != 0:
            return None

        key_data = {
            'prompt': prompt,
            'max_tokens': params.max_tokens,
            'model': params.model
        }
        return hashlib.md5(json.dumps(key_data).encode()).hexdigest()

    def get(self, prompt, params):
        key = self.get_cache_key(prompt, params)
        if not key:
            return None

        if key in self.cache:
            entry = self.cache[key]
            if time.time() - entry['timestamp'] < self.ttl:
                return entry['response']
            else:
                del self.cache[key]

        return None

    def put(self, prompt, params, response):
        key = self.get_cache_key(prompt, params)
        if key:
            self.cache[key] = {
                'response': response,
                'timestamp': time.time()
            }
```

## Cost Optimization

### Cost Analysis

**GPU Costs (Cloud Pricing):**

```
GPU Type    | $/hour | Tokens/sec | Cost per 1M tokens
------------|--------|------------|-------------------
L4          | $1.00  | 100        | $2.78
A10         | $1.50  | 120        | $3.47
A100 (40GB) | $3.00  | 200        | $4.17
A100 (80GB) | $4.00  | 250        | $4.44
H100        | $8.00  | 500        | $4.44
```

**Example Calculation:**
```python
def calculate_cost(
    requests_per_day: int,
    avg_tokens_per_request: int,
    tokens_per_second: int,
    gpu_hourly_cost: float
):
    """Calculate daily inference cost"""

    # Total tokens per day
    total_tokens = requests_per_day * avg_tokens_per_request

    # Time needed (seconds)
    total_seconds = total_tokens / tokens_per_second
    total_hours = total_seconds / 3600

    # Cost
    daily_cost = total_hours * gpu_hourly_cost

    # Cost per million tokens
    cost_per_million = (gpu_hourly_cost / (tokens_per_second * 3600)) * 1e6

    return {
        'daily_cost': daily_cost,
        'monthly_cost': daily_cost * 30,
        'cost_per_million_tokens': cost_per_million
    }

# Example: 10K requests/day, 500 tokens avg, A100 40GB
costs = calculate_cost(
    requests_per_day=10000,
    avg_tokens_per_request=500,
    tokens_per_second=200,
    gpu_hourly_cost=3.00
)

print(f"Daily cost: ${costs['daily_cost']:.2f}")
print(f"Monthly cost: ${costs['monthly_cost']:.2f}")
print(f"Cost per 1M tokens: ${costs['cost_per_million_tokens']:.2f}")
# Output: Daily: $20.83, Monthly: $625, Per 1M: $4.17
```

### Optimization Strategies

**1. Quantization for Cost Reduction:**

```
Llama-70B serving options:

Option A: FP16, 4x A100 80GB tensor parallel
  - Cost: $16/hour
  - Throughput: 200 tokens/sec
  - Cost per 1M tokens: $22.22

Option B: INT4-AWQ, 1x A100 80GB
  - Cost: $4/hour
  - Throughput: 150 tokens/sec (quantization overhead)
  - Cost per 1M tokens: $7.41
  - Savings: 67%!

Option C: INT4-AWQ, 2x A100 40GB tensor parallel
  - Cost: $6/hour
  - Throughput: 180 tokens/sec
  - Cost per 1M tokens: $9.26
  - Good balance
```

**2. Spot Instances:**

```python
# Use spot instances for batch workloads
# (not for real-time serving due to interruption risk)

class SpotInstanceManager:
    def __init__(self):
        self.on_demand_cost = 4.00  # $/hour
        self.spot_cost = 1.20  # $/hour (70% savings typical)

    def run_batch_job(self, num_hours_estimate):
        """
        For batch inference (not latency-sensitive):
        - Use spot instances
        - Handle interruptions gracefully
        - Save checkpoints frequently
        """
        cost_savings = (self.on_demand_cost - self.spot_cost) * num_hours_estimate
        print(f"Estimated savings: ${cost_savings:.2f}")

        # Launch spot instance with interruption handler
        # ... implementation
```

**3. Model Selection:**

```
Use case                    | Model Size | Why
----------------------------|------------|---------------------------
Simple classification       | 7B         | Adequate quality, 4x cheaper
Complex reasoning           | 70B        | Worth the cost for quality
Code generation             | 13B-34B    | Good balance
Chat/Conversation           | 7B-13B     | Fast, interactive
High-stakes (legal, medical)| 70B+       | Quality critical
```

**4. Batching Optimization:**

```python
def optimal_batch_size(gpu_memory_gb, model_memory_gb, avg_seq_len):
    """
    Find optimal batch size for throughput
    Larger batch = better GPU utilization but higher latency
    """

    # Available memory for KV cache
    available_memory = gpu_memory_gb - model_memory_gb

    # KV cache per sequence (rough estimate)
    kv_cache_per_seq = (avg_seq_len * 2.5) / 1000  # GB

    # Max batch size
    max_batch_size = int(available_memory / kv_cache_per_seq)

    # Optimal is typically 50-80% of max for stability
    optimal = int(max_batch_size * 0.7)

    return optimal

# Example: A100 80GB, Llama-70B INT4 (35GB), 2048 avg seq
batch_size = optimal_batch_size(
    gpu_memory_gb=80,
    model_memory_gb=35,
    avg_seq_len=2048
)
print(f"Optimal batch size: {batch_size}")  # ~12-15
```

## Production Considerations

### Monitoring & Observability

**Key Metrics to Track:**

```python
from prometheus_client import Counter, Histogram, Gauge
import time

# Request metrics
requests_total = Counter('llm_requests_total', 'Total requests', ['model', 'status'])
request_duration = Histogram('llm_request_duration_seconds', 'Request duration', ['model'])

# Token metrics
tokens_generated = Counter('llm_tokens_generated_total', 'Total tokens generated', ['model'])
tokens_per_second = Gauge('llm_tokens_per_second', 'Token generation rate', ['model'])

# Latency metrics
time_to_first_token = Histogram('llm_ttft_seconds', 'Time to first token', ['model'])
inter_token_latency = Histogram('llm_inter_token_latency_ms', 'Inter-token latency', ['model'])

# Resource metrics
gpu_utilization = Gauge('gpu_utilization_percent', 'GPU utilization', ['gpu_id'])
gpu_memory_used = Gauge('gpu_memory_used_bytes', 'GPU memory used', ['gpu_id'])
queue_length = Gauge('llm_queue_length', 'Request queue length', ['model'])
active_requests = Gauge('llm_active_requests', 'Active requests', ['model'])

# Example instrumentation
@app.post("/v1/completions")
async def completion(request: CompletionRequest):
    start_time = time.time()

    try:
        # Track request
        active_requests.labels(model=request.model).inc()

        # Generate
        first_token_time = None
        token_count = 0

        async for token in engine.generate_stream(request):
            if first_token_time is None:
                first_token_time = time.time()
                ttft = first_token_time - start_time
                time_to_first_token.labels(model=request.model).observe(ttft)

            token_count += 1
            yield token

        # Record metrics
        duration = time.time() - start_time
        request_duration.labels(model=request.model).observe(duration)
        tokens_generated.labels(model=request.model).inc(token_count)
        tokens_per_second.labels(model=request.model).set(token_count / duration)
        requests_total.labels(model=request.model, status='success').inc()

    except Exception as e:
        requests_total.labels(model=request.model, status='error').inc()
        raise

    finally:
        active_requests.labels(model=request.model).dec()
```

**Grafana Dashboard:**

```
┌─────────────────────────────────────────────────────────────┐
│  LLM Inference Dashboard                                    │
├─────────────────────────────────────────────────────────────┤
│  Throughput: 1,247 req/min  ↑ 12%                          │
│  P95 Latency: 324ms                                         │
│  Active Requests: 47                                        │
│  GPU Utilization: 78%                                       │
│                                                             │
│  [Graph: Requests/sec over time]                           │
│  [Graph: Latency percentiles (P50, P95, P99)]              │
│  [Graph: Tokens/sec per GPU]                               │
│  [Graph: Queue length]                                      │
│  [Graph: GPU memory usage]                                  │
│  [Graph: Cost per 1M tokens]                               │
└─────────────────────────────────────────────────────────────┘
```

### Health Checks & Failover

```python
from fastapi import FastAPI

app = FastAPI()

@app.get("/health")
async def health_check():
    """Kubernetes health check"""
    try:
        # Check model is loaded
        if not model_loaded:
            return {"status": "unhealthy", "reason": "model_not_loaded"}

        # Check GPU is accessible
        if not torch.cuda.is_available():
            return {"status": "unhealthy", "reason": "gpu_unavailable"}

        # Check queue is not overloaded
        if queue_length > MAX_QUEUE_SIZE:
            return {"status": "unhealthy", "reason": "queue_overloaded"}

        return {"status": "healthy"}

    except Exception as e:
        return {"status": "unhealthy", "reason": str(e)}

@app.get("/ready")
async def readiness_check():
    """Check if ready to serve traffic"""
    return {
        "ready": model_loaded and torch.cuda.is_available(),
        "queue_length": queue_length,
        "active_requests": active_requests
    }
```

### Safety & Content Filtering

```python
from transformers import pipeline

class ContentFilter:
    def __init__(self):
        # Load toxicity classifier
        self.toxicity_classifier = pipeline(
            "text-classification",
            model="unitary/toxic-bert"
        )

    def is_safe(self, text: str) -> bool:
        """Check if text is safe to generate/return"""
        result = self.toxicity_classifier(text[:512])[0]

        # Reject if toxic with high confidence
        if result['label'] == 'toxic' and result['score'] > 0.7:
            return False

        return True

    def filter_request(self, request: CompletionRequest):
        """Filter input prompt"""
        if not self.is_safe(request.prompt):
            raise HTTPException(
                status_code=400,
                detail="Request contains inappropriate content"
            )

    def filter_response(self, response: str):
        """Filter generated output"""
        if not self.is_safe(response):
            return "[Content filtered due to safety policies]"
        return response

# Usage
content_filter = ContentFilter()

@app.post("/v1/completions")
async def completion(request: CompletionRequest):
    # Filter input
    content_filter.filter_request(request)

    # Generate
    response = await engine.generate(request)

    # Filter output
    response.text = content_filter.filter_response(response.text)

    return response
```

## Framework Comparison

### Popular Serving Frameworks

| Framework | Developer | Key Features | Best For |
|-----------|-----------|--------------|----------|
| vLLM | UC Berkeley | PagedAttention, continuous batching | High throughput |
| TGI (Text Generation Inference) | Hugging Face | Easy deployment, streaming | Production ease |
| TensorRT-LLM | NVIDIA | Optimized kernels, quantization | Max performance |
| DeepSpeed-MII | Microsoft | DeepSpeed integration | Training + serving |
| Ray Serve | Anyscale | Distributed, multi-model | Complex deployments |

### Detailed Comparison

**vLLM:**

```python
from vllm import LLM, SamplingParams

# Simple API
llm = LLM(model="meta-llama/Llama-2-7b-hf")
prompts = ["Hello, my name is", "The capital of France is"]
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

outputs = llm.generate(prompts, sampling_params)
for output in outputs:
    print(output.outputs[0].text)
```

**Pros:**
- State-of-the-art throughput (PagedAttention)
- Simple Python API
- Good documentation
- Active development

**Cons:**
- Less mature than some alternatives
- Limited quantization support (improving)
- Primarily optimized for NVIDIA GPUs

**TGI (Text Generation Inference):**

```bash
# Docker deployment
docker run --gpus all --shm-size 1g \
  -p 8080:80 \
  -v $PWD/data:/data \
  ghcr.io/huggingface/text-generation-inference:latest \
  --model-id meta-llama/Llama-2-7b-hf \
  --num-shard 1
```

```python
# Client usage
from huggingface_hub import InferenceClient

client = InferenceClient(model="http://localhost:8080")
response = client.text_generation(
    "What is deep learning?",
    max_new_tokens=100
)
```

**Pros:**
- Production-ready out of the box
- Excellent streaming support
- Built-in metrics and monitoring
- OpenAI-compatible API

**Cons:**
- Slightly lower throughput than vLLM
- Less flexible for customization

**TensorRT-LLM:**

```python
# More complex setup but maximum performance
import tensorrt_llm

# Build optimized engine
builder = tensorrt_llm.Builder()
network = builder.create_network()
# ... build network
engine = builder.build_engine(network, config)

# Run inference
session = tensorrt_llm.InferenceSession(engine)
outputs = session.infer(inputs)
```

**Pros:**
- Maximum performance on NVIDIA GPUs
- Advanced optimizations (fusion, quantization)
- Lowest latency

**Cons:**
- Complex setup
- Requires engine building
- NVIDIA-only

### Benchmark Comparison

**Setup:**
- Model: Llama-2-7B
- Hardware: 1x A100 40GB
- Input: 512 tokens, Output: 128 tokens
- Batch size: 32

```
Framework       | Throughput (req/s) | Latency P95 (ms) | GPU Util
----------------|--------------------|--------------------|----------
vLLM            | 42                | 850                | 82%
TGI             | 38                | 920                | 76%
TensorRT-LLM    | 47                | 780                | 85%
Vanilla PyTorch | 12                | 2800               | 45%
```

**Recommendation:**
- **Production (ease)**: TGI
- **Production (performance)**: vLLM
- **Maximum performance**: TensorRT-LLM
- **Research/flexibility**: Raw PyTorch/DeepSpeed

## Performance Benchmarks

### Real-World Performance

**Llama-2-7B (Single A100 40GB):**

```
Configuration: vLLM, FP16, continuous batching

Input Length | Output Length | Batch Size | Throughput | Latency P95
-------------|---------------|------------|------------|-------------
128          | 128           | 64         | 3200 tok/s | 450ms
512          | 128           | 32         | 2800 tok/s | 680ms
1024         | 256           | 16         | 2400 tok/s | 1200ms
2048         | 512           | 8          | 2000 tok/s | 2500ms
```

**Llama-2-70B (4x A100 40GB, Tensor Parallel):**

```
Configuration: vLLM, INT4-AWQ, 4-way tensor parallelism

Input Length | Output Length | Batch Size | Throughput | Latency P95
-------------|---------------|------------|------------|-------------
128          | 128           | 32         | 800 tok/s  | 1200ms
512          | 128           | 16         | 650 tok/s  | 1800ms
1024         | 256           | 8          | 550 tok/s  | 3000ms
2048         | 512           | 4          | 400 tok/s  | 5500ms
```

### Scaling Results

**Horizontal Scaling (Llama-2-7B):**

```
GPUs | Throughput | Cost/1M tokens | Efficiency
-----|------------|----------------|------------
1    | 2800 t/s   | $3.75          | 100%
2    | 5500 t/s   | $3.82          | 98%
4    | 11000 t/s  | $3.82          | 98%
8    | 21500 t/s  | $3.91          | 96%

Scaling is nearly linear for independent replicas!
```

## Conclusion

### Recommended Architecture

**Small Scale (< 100 QPS):**
```
Model: Llama-2-7B or similar
Infrastructure: 2x A100 40GB (replicas)
Framework: vLLM or TGI
Optimizations: FP16, continuous batching
Cost: ~$150/day
Latency: P95 < 500ms
```

**Medium Scale (100-1000 QPS):**
```
Model: Multiple models (7B-70B)
Infrastructure:
  - 5x A100 40GB for 7B models (replicas)
  - 2x (4x A100 40GB) for 70B models (tensor parallel)
Framework: vLLM
Optimizations: INT8/INT4 for large models, prefix caching
Load Balancing: Latency-aware routing
Cost: ~$1500/day
Latency: P95 < 800ms
```

**Large Scale (1000+ QPS):**
```
Model: Fleet of models
Infrastructure: Auto-scaling cluster (50-100 GPUs)
Framework: vLLM with custom orchestration
Optimizations:
  - Aggressive quantization (INT4)
  - Prefix caching
  - Speculative decoding
  - Response caching
Load Balancing: Multi-tier routing
Cost: ~$10,000/day
Latency: P95 < 1000ms
```

### Key Takeaways

1. **Continuous batching is crucial**: 3-5x throughput improvement over static batching

2. **Memory is the bottleneck**: KV cache dominates memory usage for long contexts

3. **Quantization is cost-effective**: INT4 enables 4x cost reduction with minimal quality loss

4. **Choose the right model size**: Don't use 70B when 7B suffices (4x cost difference)

5. **Optimize for your use case**:
   - Latency-critical → Smaller models, more replicas
   - Throughput-focused → Larger batches, continuous batching
   - Cost-sensitive → Quantization, model selection, caching

6. **Monitor everything**: Latency, throughput, cost per token, GPU utilization

7. **Framework choice matters**: vLLM for throughput, TGI for ease, TensorRT-LLM for max performance

This design provides a production-ready, scalable LLM inference serving system capable of handling thousands of QPS with sub-second latency while optimizing for cost efficiency.
