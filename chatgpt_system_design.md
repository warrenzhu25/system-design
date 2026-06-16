# System Design: ChatGPT-like Conversational AI System

## Problem Statement

Design a large-scale conversational AI system similar to ChatGPT that can:
- Handle millions of concurrent users
- Provide real-time streaming responses
- Maintain conversation context
- Support multiple AI models
- Ensure safety and content moderation

---

## 1. Requirements

### Functional Requirements

1. **Chat Interface**
   - Send messages and receive AI responses
   - Stream responses token-by-token (real-time)
   - Support multi-turn conversations with context

2. **Conversation Management**
   - Create, list, delete conversations
   - Retrieve conversation history
   - Share conversations (optional)

3. **Model Selection**
   - Support multiple models (GPT-4, GPT-3.5, etc.)
   - Model-specific capabilities (vision, code, etc.)

4. **User Management**
   - Authentication and authorization
   - Usage tracking and rate limiting
   - Subscription tiers (free, plus, enterprise)

5. **Safety & Moderation**
   - Content filtering (input and output)
   - Prompt injection detection
   - PII detection and redaction

### Non-Functional Requirements

| Requirement | Target |
|-------------|--------|
| Latency (Time to First Token) | < 500ms P95 |
| Throughput | 100K+ concurrent chats |
| Availability | 99.9% |
| Token Generation Rate | 50-100 tokens/second |
| Context Window | Up to 128K tokens |
| Message History | Unlimited (paginated) |

---

## 2. Capacity Estimation

### Traffic Estimates

```
Daily Active Users (DAU): 100 million
Avg conversations/user/day: 5
Avg messages/conversation: 10
Avg tokens/message: 100 (input) + 500 (output)

Daily messages: 100M × 5 × 10 = 5 billion messages
Daily tokens: 5B × 600 = 3 trillion tokens

Peak QPS: 5B / 86400 × 3 (peak factor) ≈ 175K messages/second
```

### Storage Estimates

```
Per message: ~2KB (content + metadata)
Daily storage: 5B × 2KB = 10 TB/day
Monthly storage: 300 TB
With 3x replication: ~1 PB/month
```

### Compute Estimates

```
GPU inference:
- 1 A100 GPU: ~50 tokens/second for GPT-4 class model
- Target: 3T tokens/day ÷ 86400 = 35M tokens/second
- Required GPUs: 35M / 50 = 700K GPU-seconds/second
- With batching (8x efficiency): ~90K A100 GPUs

Note: Actual deployment uses optimizations:
- Speculative decoding
- KV cache optimization
- Model parallelism
```

---

## 3. High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                                    Clients                                       │
│                    (Web, Mobile, API, IDE Plugins)                              │
└─────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              CDN / Edge Network                                  │
│                         (Static assets, WebSocket edges)                        │
└─────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              API Gateway / Load Balancer                         │
│                    (Rate limiting, Auth, Request routing)                       │
└─────────────────────────────────────────────────────────────────────────────────┘
                                        │
                    ┌───────────────────┼───────────────────┐
                    ▼                   ▼                   ▼
          ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
          │  Chat Service   │ │ Conversation    │ │  User Service   │
          │  (Orchestrator) │ │    Service      │ │                 │
          └────────┬────────┘ └────────┬────────┘ └────────┬────────┘
                   │                   │                   │
                   │         ┌─────────┴─────────┐         │
                   │         ▼                   ▼         │
                   │  ┌─────────────┐   ┌─────────────┐    │
                   │  │ Conversation│   │    User     │    │
                   │  │   Database  │   │  Database   │────┘
                   │  │  (Messages) │   │             │
                   │  └─────────────┘   └─────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                            Inference Gateway                                     │
│              (Load balancing, Model routing, Request batching)                  │
└─────────────────────────────────────────────────────────────────────────────────┘
                                        │
        ┌───────────────────────────────┼───────────────────────────────┐
        ▼                               ▼                               ▼
┌───────────────┐               ┌───────────────┐               ┌───────────────┐
│  Model Pool   │               │  Model Pool   │               │  Model Pool   │
│   (GPT-4)     │               │  (GPT-3.5)    │               │   (Vision)    │
│               │               │               │               │               │
│ ┌───────────┐ │               │ ┌───────────┐ │               │ ┌───────────┐ │
│ │ GPU Node  │ │               │ │ GPU Node  │ │               │ │ GPU Node  │ │
│ │ (A100×8)  │ │               │ │ (A100×8)  │ │               │ │ (A100×8)  │ │
│ └───────────┘ │               │ └───────────┘ │               │ └───────────┘ │
│ ┌───────────┐ │               │ ┌───────────┐ │               │ ┌───────────┐ │
│ │ GPU Node  │ │               │ │ GPU Node  │ │               │ │ GPU Node  │ │
│ └───────────┘ │               │ └───────────┘ │               │ └───────────┘ │
└───────────────┘               └───────────────┘               └───────────────┘
                                        │
                                        ▼
                            ┌───────────────────┐
                            │   Safety Layer    │
                            │  (Content Filter) │
                            └───────────────────┘
```

---

## 4. Core Components

### 4.1 API Gateway

**Responsibilities:**
- Authentication (JWT, API keys)
- Rate limiting per user/tier
- Request validation
- WebSocket connection management
- Geographic routing

```python
class APIGateway:
    def handle_request(self, request):
        # 1. Authenticate
        user = self.auth_service.authenticate(request.token)

        # 2. Rate limit check
        if not self.rate_limiter.allow(user.id, user.tier):
            raise RateLimitExceeded()

        # 3. Validate request
        self.validate_request(request)

        # 4. Route to appropriate service
        if request.type == "chat":
            return self.route_to_chat_service(request)
        elif request.type == "conversation":
            return self.route_to_conversation_service(request)
```

**Rate Limiting by Tier:**

| Tier | Requests/min | Tokens/min | Concurrent |
|------|-------------|------------|------------|
| Free | 3 | 40K | 1 |
| Plus | 60 | 100K | 3 |
| API | 500 | 1M | 10 |
| Enterprise | Unlimited | Custom | Custom |

### 4.2 Chat Service (Orchestrator)

The central orchestrator that handles the chat flow:

```python
class ChatService:
    def __init__(self):
        self.conversation_service = ConversationService()
        self.inference_gateway = InferenceGateway()
        self.safety_service = SafetyService()
        self.token_counter = TokenCounter()

    async def handle_chat(self, request: ChatRequest) -> AsyncIterator[str]:
        """
        Main chat handler with streaming response.
        """
        # 1. Safety check on input
        safety_result = await self.safety_service.check_input(request.message)
        if not safety_result.is_safe:
            yield "[Content blocked due to policy violation]"
            return

        # 2. Load conversation history
        history = await self.conversation_service.get_history(
            request.conversation_id,
            max_tokens=request.model.context_window - request.max_tokens
        )

        # 3. Build prompt with context
        prompt = self.build_prompt(history, request.message, request.system_prompt)

        # 4. Count tokens and validate
        input_tokens = self.token_counter.count(prompt)
        if input_tokens > request.model.context_window:
            raise ContextTooLong()

        # 5. Stream inference
        response_tokens = []
        async for token in self.inference_gateway.generate(
            model=request.model,
            prompt=prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature
        ):
            # Safety check on partial output (periodically)
            response_tokens.append(token)

            if len(response_tokens) % 50 == 0:
                partial = "".join(response_tokens)
                if not await self.safety_service.check_output_partial(partial):
                    yield "[Response stopped due to content policy]"
                    return

            yield token

        # 6. Save to conversation history
        full_response = "".join(response_tokens)
        await self.conversation_service.add_messages(
            request.conversation_id,
            [
                Message(role="user", content=request.message),
                Message(role="assistant", content=full_response)
            ]
        )

        # 7. Log usage for billing
        await self.usage_service.log(
            user_id=request.user_id,
            model=request.model,
            input_tokens=input_tokens,
            output_tokens=len(response_tokens)
        )
```

### 4.3 Conversation Service

Manages conversation state and history:

```python
class ConversationService:
    def __init__(self):
        self.db = ConversationDatabase()  # Cassandra/DynamoDB
        self.cache = Redis()

    async def get_history(
        self,
        conversation_id: str,
        max_tokens: int
    ) -> list[Message]:
        """
        Get conversation history, truncated to fit context window.
        Uses sliding window - keeps recent messages.
        """
        # Try cache first
        cached = await self.cache.get(f"conv:{conversation_id}")
        if cached:
            messages = deserialize(cached)
        else:
            messages = await self.db.get_messages(conversation_id)
            await self.cache.setex(f"conv:{conversation_id}", 300, serialize(messages))

        # Truncate to fit context window (keep most recent)
        return self.truncate_to_tokens(messages, max_tokens)

    def truncate_to_tokens(self, messages: list[Message], max_tokens: int) -> list[Message]:
        """
        Keep most recent messages that fit within token budget.
        Always keep system message if present.
        """
        result = []
        current_tokens = 0

        # Always include system message
        system_msg = next((m for m in messages if m.role == "system"), None)
        if system_msg:
            current_tokens += count_tokens(system_msg.content)
            result.append(system_msg)

        # Add messages from most recent, going backwards
        for msg in reversed(messages):
            if msg.role == "system":
                continue

            msg_tokens = count_tokens(msg.content)
            if current_tokens + msg_tokens > max_tokens:
                break

            result.insert(1 if system_msg else 0, msg)
            current_tokens += msg_tokens

        return result

    async def create_conversation(self, user_id: str, title: str = None) -> Conversation:
        conv = Conversation(
            id=generate_uuid(),
            user_id=user_id,
            title=title or "New conversation",
            created_at=now(),
            updated_at=now()
        )
        await self.db.insert(conv)
        return conv
```

**Database Schema:**

```sql
-- Conversations table (partitioned by user_id)
CREATE TABLE conversations (
    conversation_id UUID PRIMARY KEY,
    user_id UUID,
    title TEXT,
    model TEXT,
    created_at TIMESTAMP,
    updated_at TIMESTAMP,
    is_archived BOOLEAN DEFAULT FALSE,

    INDEX idx_user_conversations (user_id, updated_at DESC)
);

-- Messages table (partitioned by conversation_id, clustered by created_at)
CREATE TABLE messages (
    conversation_id UUID,
    message_id UUID,
    role TEXT,  -- 'user', 'assistant', 'system'
    content TEXT,
    tokens INT,
    created_at TIMESTAMP,

    PRIMARY KEY (conversation_id, created_at, message_id)
) WITH CLUSTERING ORDER BY (created_at ASC);
```

### 4.4 Inference Gateway

Routes requests to appropriate model pools and handles batching:

```python
class InferenceGateway:
    def __init__(self):
        self.model_pools = {
            "gpt-4": ModelPool("gpt-4", min_replicas=100),
            "gpt-3.5-turbo": ModelPool("gpt-3.5-turbo", min_replicas=500),
            "gpt-4-vision": ModelPool("gpt-4-vision", min_replicas=50),
        }
        self.request_batcher = RequestBatcher()

    async def generate(
        self,
        model: str,
        prompt: str,
        max_tokens: int,
        temperature: float
    ) -> AsyncIterator[str]:
        """
        Route to model pool and stream response.
        """
        pool = self.model_pools[model]

        # Get healthy node with lowest load
        node = await pool.get_best_node()

        # Stream tokens from inference node
        async for token in node.generate_stream(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature
        ):
            yield token


class ModelPool:
    """Manages a pool of GPU nodes for a specific model."""

    def __init__(self, model_name: str, min_replicas: int):
        self.model_name = model_name
        self.nodes: list[InferenceNode] = []
        self.min_replicas = min_replicas

    async def get_best_node(self) -> InferenceNode:
        """
        Select node using weighted round-robin based on:
        - Current queue depth
        - GPU memory utilization
        - Recent latency
        """
        healthy_nodes = [n for n in self.nodes if n.is_healthy()]

        if not healthy_nodes:
            raise NoAvailableNodes()

        # Score nodes (lower is better)
        def score(node):
            return (
                node.queue_depth * 10 +
                node.gpu_memory_pct * 5 +
                node.avg_latency_ms / 100
            )

        return min(healthy_nodes, key=score)


class InferenceNode:
    """Single GPU node running model inference."""

    async def generate_stream(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float
    ) -> AsyncIterator[str]:
        """
        Generate tokens using the model.
        Uses KV cache for efficient autoregressive generation.
        """
        # Tokenize input
        input_ids = self.tokenizer.encode(prompt)

        # Initialize KV cache
        kv_cache = self.model.init_kv_cache(input_ids)

        # Generate tokens one at a time
        for _ in range(max_tokens):
            # Forward pass (uses cached KV states)
            logits = self.model.forward(input_ids[-1:], kv_cache)

            # Sample next token
            next_token = self.sample(logits, temperature)

            # Check for end of sequence
            if next_token == self.tokenizer.eos_token_id:
                break

            # Decode and yield
            token_str = self.tokenizer.decode([next_token])
            yield token_str

            # Update for next iteration
            input_ids = torch.cat([input_ids, next_token.unsqueeze(0)])
```

### 4.5 Streaming Response (Server-Sent Events)

```python
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import asyncio

app = FastAPI()

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    if request.stream:
        return StreamingResponse(
            generate_stream(request),
            media_type="text/event-stream"
        )
    else:
        return await generate_full(request)


async def generate_stream(request: ChatCompletionRequest):
    """
    Server-Sent Events format for streaming.
    """
    chat_service = ChatService()

    async for token in chat_service.handle_chat(request):
        # SSE format
        chunk = {
            "id": f"chatcmpl-{generate_id()}",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": request.model,
            "choices": [{
                "index": 0,
                "delta": {"content": token},
                "finish_reason": None
            }]
        }
        yield f"data: {json.dumps(chunk)}\n\n"

        # Small delay to prevent overwhelming client
        await asyncio.sleep(0.01)

    # Final chunk with finish_reason
    final_chunk = {
        "id": f"chatcmpl-{generate_id()}",
        "object": "chat.completion.chunk",
        "choices": [{
            "index": 0,
            "delta": {},
            "finish_reason": "stop"
        }]
    }
    yield f"data: {json.dumps(final_chunk)}\n\n"
    yield "data: [DONE]\n\n"
```

---

## 5. Safety & Content Moderation

### Multi-Layer Safety System

```
┌─────────────────────────────────────────────────────────────┐
│                      Input Safety                            │
├─────────────────────────────────────────────────────────────┤
│  1. Prompt Injection Detection                              │
│  2. PII Detection (SSN, credit cards, etc.)                │
│  3. Harmful Content Classification                          │
│  4. Rate-based Abuse Detection                              │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
                    [Model Inference]
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                     Output Safety                            │
├─────────────────────────────────────────────────────────────┤
│  1. Harmful Content Detection                               │
│  2. Hallucination Detection (optional)                      │
│  3. PII Leakage Detection                                   │
│  4. Code Safety (malware patterns)                          │
└─────────────────────────────────────────────────────────────┘
```

```python
class SafetyService:
    def __init__(self):
        self.content_classifier = ContentClassifier()
        self.pii_detector = PIIDetector()
        self.prompt_injection_detector = PromptInjectionDetector()

    async def check_input(self, message: str) -> SafetyResult:
        """
        Check user input for safety issues.
        Run checks in parallel for speed.
        """
        results = await asyncio.gather(
            self.content_classifier.classify(message),
            self.pii_detector.detect(message),
            self.prompt_injection_detector.detect(message),
        )

        content_result, pii_result, injection_result = results

        # Block if any high-severity issue
        if content_result.category in ["violence", "hate", "sexual_minors"]:
            return SafetyResult(is_safe=False, reason="content_policy")

        if injection_result.is_injection and injection_result.confidence > 0.9:
            return SafetyResult(is_safe=False, reason="prompt_injection")

        # Warn but allow for PII (redact in logs)
        if pii_result.has_pii:
            return SafetyResult(is_safe=True, pii_detected=True)

        return SafetyResult(is_safe=True)

    async def check_output_partial(self, partial_response: str) -> bool:
        """
        Fast check on partial output during streaming.
        Uses lightweight classifier for speed.
        """
        result = await self.content_classifier.classify_fast(partial_response)
        return result.category not in ["violence", "hate", "illegal"]
```

---

## 6. Scaling Strategies

### 6.1 GPU Cluster Management

```python
class GPUClusterManager:
    """
    Manages GPU nodes across multiple regions/zones.
    Handles scaling, health checks, and failover.
    """

    def __init__(self):
        self.nodes: dict[str, InferenceNode] = {}
        self.autoscaler = Autoscaler()

    async def scale_model_pool(self, model: str, target_replicas: int):
        """
        Scale up/down based on demand.
        """
        current = len([n for n in self.nodes.values() if n.model == model])

        if target_replicas > current:
            # Scale up
            for _ in range(target_replicas - current):
                node = await self.provision_node(model)
                self.nodes[node.id] = node
        else:
            # Scale down (graceful)
            excess = current - target_replicas
            nodes_to_remove = self.select_nodes_for_removal(model, excess)
            for node in nodes_to_remove:
                await node.drain()  # Wait for in-flight requests
                await self.terminate_node(node)

    def calculate_target_replicas(self, model: str) -> int:
        """
        Based on current queue depth and latency targets.
        """
        pool = self.model_pools[model]
        avg_queue_depth = pool.get_avg_queue_depth()
        avg_latency = pool.get_avg_latency()

        # Scale up if queue is building or latency is high
        if avg_queue_depth > 10 or avg_latency > 1000:
            return int(pool.current_replicas * 1.5)
        elif avg_queue_depth < 2 and avg_latency < 200:
            return max(pool.min_replicas, int(pool.current_replicas * 0.8))

        return pool.current_replicas
```

### 6.2 Request Batching (Continuous Batching)

```python
class ContinuousBatcher:
    """
    Continuous batching for efficient GPU utilization.
    New requests can join a batch even while generation is in progress.
    """

    def __init__(self, model, max_batch_size=32):
        self.model = model
        self.max_batch_size = max_batch_size
        self.active_requests: dict[str, Request] = {}
        self.pending_queue = asyncio.Queue()

    async def run(self):
        """
        Main loop: process batches continuously.
        """
        while True:
            # Gather requests for next batch
            batch = await self.gather_batch()

            if not batch:
                await asyncio.sleep(0.001)
                continue

            # Process one step for all requests in batch
            await self.process_batch_step(batch)

            # Remove completed requests
            self.remove_completed()

    async def gather_batch(self) -> list[Request]:
        """
        Collect requests up to max_batch_size.
        Include both new and in-progress requests.
        """
        batch = list(self.active_requests.values())

        # Add new requests from queue
        while len(batch) < self.max_batch_size:
            try:
                req = self.pending_queue.get_nowait()
                batch.append(req)
                self.active_requests[req.id] = req
            except asyncio.QueueEmpty:
                break

        return batch

    async def process_batch_step(self, batch: list[Request]):
        """
        Generate one token for each request in batch.
        Uses batched forward pass for efficiency.
        """
        # Pad sequences for batched inference
        input_ids = pad_sequences([r.current_ids for r in batch])
        kv_caches = [r.kv_cache for r in batch]

        # Batched forward pass
        logits = self.model.forward_batch(input_ids, kv_caches)

        # Sample next token for each request
        for i, req in enumerate(batch):
            next_token = self.sample(logits[i], req.temperature)

            if next_token == EOS_TOKEN or len(req.generated) >= req.max_tokens:
                req.mark_complete()
            else:
                req.add_token(next_token)
                # Notify streaming client
                await req.token_queue.put(self.tokenizer.decode([next_token]))
```

### 6.3 KV Cache Optimization

```python
class PagedKVCache:
    """
    Paged attention for efficient KV cache management.
    Allows non-contiguous memory allocation.
    """

    def __init__(self, num_layers, num_heads, head_dim, block_size=16):
        self.block_size = block_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim

        # Physical blocks (pre-allocated GPU memory)
        self.physical_blocks = self.allocate_physical_blocks(1000)
        self.free_blocks = list(range(len(self.physical_blocks)))

        # Logical to physical mapping per sequence
        self.block_tables: dict[str, list[int]] = {}

    def allocate_for_sequence(self, seq_id: str, num_tokens: int):
        """
        Allocate blocks for a new sequence.
        """
        num_blocks = (num_tokens + self.block_size - 1) // self.block_size

        if len(self.free_blocks) < num_blocks:
            raise OutOfMemory()

        allocated = [self.free_blocks.pop() for _ in range(num_blocks)]
        self.block_tables[seq_id] = allocated

    def extend_sequence(self, seq_id: str):
        """
        Add another block when sequence grows.
        """
        if not self.free_blocks:
            # Trigger eviction of completed sequences
            self.evict_completed()

        new_block = self.free_blocks.pop()
        self.block_tables[seq_id].append(new_block)

    def free_sequence(self, seq_id: str):
        """
        Return blocks when sequence completes.
        """
        blocks = self.block_tables.pop(seq_id)
        self.free_blocks.extend(blocks)
```

---

## 7. Context Window Management

### Long Context Handling

```python
class ContextManager:
    """
    Manages context window for long conversations.
    """

    def __init__(self, max_context: int = 128000):
        self.max_context = max_context

    def prepare_context(
        self,
        system_prompt: str,
        history: list[Message],
        user_message: str,
        max_output: int
    ) -> str:
        """
        Build context that fits within token limit.
        """
        available = self.max_context - max_output

        # Reserve space for system prompt and current message
        system_tokens = count_tokens(system_prompt)
        user_tokens = count_tokens(user_message)
        history_budget = available - system_tokens - user_tokens - 100  # buffer

        # Select history using recency + importance
        selected_history = self.select_history(history, history_budget)

        # Build final prompt
        prompt = f"{system_prompt}\n\n"
        for msg in selected_history:
            prompt += f"{msg.role}: {msg.content}\n\n"
        prompt += f"user: {user_message}\nassistant:"

        return prompt

    def select_history(
        self,
        history: list[Message],
        budget: int
    ) -> list[Message]:
        """
        Select messages to include in context.

        Strategy:
        1. Always include most recent messages
        2. For older messages, use summarization or sampling
        """
        result = []
        used_tokens = 0

        # Most recent messages first
        for msg in reversed(history):
            msg_tokens = count_tokens(msg.content)

            if used_tokens + msg_tokens > budget:
                break

            result.insert(0, msg)
            used_tokens += msg_tokens

        return result


class ConversationSummarizer:
    """
    Summarize older parts of conversation to save context.
    """

    async def summarize_if_needed(
        self,
        conversation_id: str,
        history: list[Message],
        threshold: int = 50000
    ) -> list[Message]:
        """
        If history exceeds threshold, summarize older messages.
        """
        total_tokens = sum(count_tokens(m.content) for m in history)

        if total_tokens < threshold:
            return history

        # Split into old (to summarize) and recent (to keep)
        split_point = len(history) // 2
        old_messages = history[:split_point]
        recent_messages = history[split_point:]

        # Generate summary
        summary = await self.generate_summary(old_messages)

        # Store summary as system context
        summary_message = Message(
            role="system",
            content=f"[Summary of earlier conversation: {summary}]"
        )

        return [summary_message] + recent_messages
```

---

## 8. Caching Strategies

### Multi-Level Caching

```
┌─────────────────────────────────────────────────────────────┐
│                    L1: Prompt Cache                          │
│           (Exact match, short TTL, per-node)                │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                   L2: KV Cache                               │
│        (Prefix sharing across requests, GPU memory)         │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                L3: Semantic Cache                            │
│         (Similar prompts, embedding-based, Redis)           │
└─────────────────────────────────────────────────────────────┘
```

```python
class PromptCache:
    """
    Cache exact prompt matches.
    Useful for repeated queries (e.g., "What is Python?")
    """

    def __init__(self):
        self.cache = LRUCache(maxsize=10000)

    def get(self, prompt: str, params: dict) -> Optional[str]:
        key = self.make_key(prompt, params)
        return self.cache.get(key)

    def set(self, prompt: str, params: dict, response: str, ttl: int = 3600):
        key = self.make_key(prompt, params)
        self.cache.set(key, response, ttl=ttl)

    def make_key(self, prompt: str, params: dict) -> str:
        # Include temperature, model, etc. in key
        return hashlib.sha256(
            f"{prompt}:{json.dumps(params, sort_keys=True)}".encode()
        ).hexdigest()


class SemanticCache:
    """
    Cache semantically similar prompts.
    Uses embedding similarity for matching.
    """

    def __init__(self, similarity_threshold: float = 0.95):
        self.threshold = similarity_threshold
        self.embeddings_db = VectorDatabase()  # Pinecone, Milvus, etc.

    async def get(self, prompt: str) -> Optional[str]:
        # Get embedding for prompt
        embedding = await self.embed(prompt)

        # Search for similar
        results = await self.embeddings_db.search(
            embedding,
            top_k=1,
            threshold=self.threshold
        )

        if results:
            return results[0].cached_response
        return None

    async def set(self, prompt: str, response: str):
        embedding = await self.embed(prompt)
        await self.embeddings_db.insert(
            embedding=embedding,
            metadata={"prompt": prompt, "response": response}
        )
```

---

## 9. Monitoring & Observability

### Key Metrics

```python
# Latency Metrics
time_to_first_token_ms = Histogram("ttft_ms", buckets=[100, 200, 500, 1000, 2000])
tokens_per_second = Gauge("tps")
end_to_end_latency_ms = Histogram("e2e_latency_ms")

# Throughput Metrics
requests_per_second = Counter("rps")
tokens_generated = Counter("tokens_generated")
concurrent_requests = Gauge("concurrent_requests")

# Error Metrics
error_rate = Counter("errors", labels=["type"])
safety_blocks = Counter("safety_blocks", labels=["reason"])
rate_limit_hits = Counter("rate_limits", labels=["tier"])

# Resource Metrics
gpu_utilization = Gauge("gpu_util", labels=["node"])
gpu_memory_used = Gauge("gpu_memory", labels=["node"])
queue_depth = Gauge("queue_depth", labels=["model"])
```

### Alerting Rules

```yaml
alerts:
  - name: HighLatency
    condition: p95(ttft_ms) > 1000
    for: 5m
    severity: warning

  - name: HighErrorRate
    condition: rate(errors[5m]) / rate(requests[5m]) > 0.01
    for: 2m
    severity: critical

  - name: GPUMemoryHigh
    condition: gpu_memory_used / gpu_memory_total > 0.95
    for: 5m
    severity: warning

  - name: QueueBacklog
    condition: queue_depth > 100
    for: 2m
    severity: warning
```

---

## 10. Cost Optimization

### Token-Based Pricing Model

```python
class BillingService:
    """
    Track and bill based on token usage.
    """

    PRICING = {
        "gpt-4": {"input": 0.03, "output": 0.06},       # per 1K tokens
        "gpt-3.5-turbo": {"input": 0.001, "output": 0.002},
        "gpt-4-vision": {"input": 0.01, "output": 0.03},
    }

    async def calculate_cost(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int
    ) -> float:
        pricing = self.PRICING[model]
        input_cost = (input_tokens / 1000) * pricing["input"]
        output_cost = (output_tokens / 1000) * pricing["output"]
        return input_cost + output_cost

    async def log_usage(
        self,
        user_id: str,
        model: str,
        input_tokens: int,
        output_tokens: int
    ):
        cost = await self.calculate_cost(model, input_tokens, output_tokens)

        await self.usage_db.insert({
            "user_id": user_id,
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost": cost,
            "timestamp": now()
        })
```

### Cost Reduction Strategies

1. **Prompt Caching**: Reuse KV cache for common prefixes
2. **Model Routing**: Use cheaper models for simple queries
3. **Batching**: Maximize GPU utilization
4. **Speculative Decoding**: Use small model to draft, large model to verify
5. **Quantization**: INT8/INT4 inference for cost reduction

---

## 11. Summary

### Architecture Decisions

| Component | Choice | Rationale |
|-----------|--------|-----------|
| API Protocol | SSE/WebSocket | Real-time streaming |
| Message Store | Cassandra | Write-heavy, time-series |
| Cache | Redis Cluster | Session state, rate limiting |
| Inference | Custom GPU cluster | Control over batching, caching |
| Load Balancing | Least-connections + health | GPU-aware routing |

### Key Trade-offs

| Trade-off | Decision | Impact |
|-----------|----------|--------|
| Latency vs Throughput | Continuous batching | Higher GPU util, slight latency |
| Cost vs Quality | Model routing | Use GPT-3.5 for simple queries |
| Context vs Speed | Sliding window | Lose old context, faster response |
| Safety vs UX | Multi-layer checks | Some false positives |

### Scaling Path

1. **10K users**: Single region, few GPU nodes
2. **1M users**: Multi-region, auto-scaling GPU pools
3. **100M users**: Global edge, aggressive caching, model optimization

---

## Interview Tips

1. **Start with requirements**: Clarify streaming, latency targets, scale
2. **Draw high-level first**: API → Orchestrator → Inference → Safety
3. **Deep dive on inference**: Batching, KV cache, GPU management
4. **Discuss trade-offs**: Latency vs throughput, cost vs quality
5. **Address safety**: Content moderation is critical for LLM systems
6. **Consider costs**: Token-based billing, optimization strategies

---

*This design covers the core components of a production ChatGPT-like system. Actual implementations involve additional complexity around model training, fine-tuning pipelines, and enterprise features.*
