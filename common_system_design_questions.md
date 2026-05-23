# Common System Design Interview Questions

A comprehensive guide to frequently asked system design questions with detailed solutions, tradeoffs, and discussion points.

---

## Table of Contents

### Core Infrastructure
1. [URL Shortener](#1-url-shortener)
2. [Rate Limiter](#2-rate-limiter)
3. [Distributed Cache](#3-distributed-cache)
4. [Key-Value Store](#4-key-value-store)

### Communication Systems
5. [Message Queue](#5-message-queue)
6. [Notification System](#6-notification-system)
7. [Real-Time Chat](#7-real-time-chat)

### Data Systems
8. [News Feed / Timeline](#8-news-feed--timeline)
9. [Search Autocomplete](#9-search-autocomplete)
10. [Web Crawler](#10-web-crawler)

### Media & Storage
11. [Video Streaming Platform](#11-video-streaming-platform)
12. [Distributed File Storage](#12-distributed-file-storage)

---

## 1. URL Shortener

**Problem:** Design a URL shortening service like bit.ly or TinyURL.

### Requirements

**Functional:**
- Generate short URL from long URL
- Redirect short URL to original
- Custom short URLs (optional)
- URL expiration (optional)
- Analytics (click count, geography)

**Non-Functional:**
- 100M URLs created per day
- 10:1 read/write ratio (1B redirects/day)
- Low latency redirects (< 100ms)
- High availability (99.99%)
- URLs should not be predictable

### Capacity Estimation

```
Write: 100M / 86400 ≈ 1,200 URLs/second
Read: 1B / 86400 ≈ 12,000 redirects/second

Storage (5 years):
- 100M × 365 × 5 = 182.5B URLs
- Average URL: 500 bytes
- Total: 182.5B × 500 = ~90 TB
```

### High-Level Architecture

```
┌──────────┐     ┌──────────────┐     ┌─────────────┐
│  Client  │────▶│ Load Balancer│────▶│  API Server │
└──────────┘     └──────────────┘     └──────┬──────┘
                                             │
                      ┌──────────────────────┼──────────────────────┐
                      │                      │                      │
               ┌──────▼──────┐       ┌───────▼───────┐      ┌───────▼───────┐
               │    Cache    │       │   Database    │      │  Analytics    │
               │   (Redis)   │       │  (Cassandra)  │      │   (Kafka)     │
               └─────────────┘       └───────────────┘      └───────────────┘
```

### URL Encoding Approaches

**Option 1: Base62 Encoding**
```python
CHARSET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"

def encode(num: int) -> str:
    if num == 0:
        return CHARSET[0]
    result = []
    while num > 0:
        result.append(CHARSET[num % 62])
        num //= 62
    return ''.join(reversed(result))

# 7 characters = 62^7 = 3.5 trillion unique URLs
```

**Option 2: MD5/SHA256 Hash + Truncation**
```python
import hashlib

def generate_short_url(long_url: str) -> str:
    hash_bytes = hashlib.md5(long_url.encode()).digest()
    # Take first 7 characters of base62 encoding
    return base62_encode(int.from_bytes(hash_bytes[:6], 'big'))[:7]
```

**Option 3: Pre-generated Keys (Key Generation Service)**
```python
class KeyGenerationService:
    def __init__(self):
        self.unused_keys = Queue()  # Pre-generated keys
        self.used_keys = set()

    def get_key(self) -> str:
        key = self.unused_keys.get()
        self.used_keys.add(key)
        return key
```

### Tradeoffs: Encoding Approaches

| Approach | Pros | Cons |
|----------|------|------|
| **Counter + Base62** | Simple, no collisions | Single point of failure, predictable |
| **Hash Truncation** | Distributed, simple | Collision handling needed |
| **Pre-generated Keys** | Fast, no runtime collision | Complex key management, storage overhead |
| **Snowflake ID** | Distributed, time-sorted | Longer URLs (10+ chars) |

### Database Schema

```sql
CREATE TABLE urls (
    short_code VARCHAR(10) PRIMARY KEY,
    long_url TEXT NOT NULL,
    user_id UUID,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    expires_at TIMESTAMPTZ,
    click_count BIGINT DEFAULT 0
);

CREATE INDEX idx_urls_user ON urls(user_id);
CREATE INDEX idx_urls_expires ON urls(expires_at) WHERE expires_at IS NOT NULL;
```

### Database Choice Tradeoffs

| Database | Pros | Cons |
|----------|------|------|
| **PostgreSQL** | ACID, familiar, good tooling | Scaling challenges at extreme scale |
| **Cassandra** | Linear scalability, high write throughput | Eventually consistent, no joins |
| **DynamoDB** | Managed, auto-scaling | Vendor lock-in, cost at scale |

### Caching Strategy

```python
class URLShortener:
    def redirect(self, short_code: str) -> str:
        # Check cache first
        cached = self.redis.get(f"url:{short_code}")
        if cached:
            return cached

        # Cache miss - query database
        url = self.db.get_url(short_code)
        if url:
            self.redis.setex(f"url:{short_code}", 3600, url.long_url)
            return url.long_url

        raise NotFoundError()
```

### Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| URL Length | 7 characters | 3.5T URLs, balance of brevity and capacity |
| Encoding | Base62 | URL-safe, case-sensitive for more combinations |
| Database | Cassandra | Write-heavy, easy horizontal scaling |
| Cache | Redis | Sub-ms reads, high hit rate expected |
| ID Generation | Snowflake | Distributed, time-ordered, no coordination |

### Handling Collisions

```python
def create_short_url(long_url: str, max_retries: int = 3) -> str:
    for attempt in range(max_retries):
        short_code = generate_code(long_url + str(attempt))
        try:
            self.db.insert(short_code, long_url)
            return short_code
        except DuplicateKeyError:
            continue
    raise CollisionError("Max retries exceeded")
```

### Interview Discussion Points

1. **How to handle hot URLs (viral content)?**
   - Cache with longer TTL
   - CDN edge caching
   - Rate limiting per URL

2. **How to prevent abuse?**
   - Rate limiting per user/IP
   - URL validation (block malicious domains)
   - CAPTCHA for anonymous users

3. **How to support custom URLs?**
   - Separate table/namespace for custom URLs
   - Availability check before creation
   - Premium feature with reservations

---

## 2. Rate Limiter

**Problem:** Design a rate limiting system to protect APIs from abuse and ensure fair usage.

### Requirements

**Functional:**
- Limit requests per user/IP/API key
- Support multiple time windows (second, minute, hour)
- Return meaningful error responses (429)
- Support different limits for different tiers

**Non-Functional:**
- < 1ms latency overhead
- High availability
- Distributed across multiple servers
- Accurate within acceptable margin (±5%)

### Rate Limiting Algorithms

#### 1. Token Bucket

```python
class TokenBucket:
    def __init__(self, capacity: int, refill_rate: float):
        self.capacity = capacity
        self.tokens = capacity
        self.refill_rate = refill_rate  # tokens per second
        self.last_refill = time.time()

    def allow_request(self, tokens: int = 1) -> bool:
        self._refill()
        if self.tokens >= tokens:
            self.tokens -= tokens
            return True
        return False

    def _refill(self):
        now = time.time()
        elapsed = now - self.last_refill
        self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_rate)
        self.last_refill = now
```

**Pros:** Allows bursts, smooth rate limiting, memory efficient
**Cons:** Harder to reason about exact limits

#### 2. Sliding Window Log

```python
class SlidingWindowLog:
    def __init__(self, window_size: int, max_requests: int):
        self.window_size = window_size  # seconds
        self.max_requests = max_requests
        self.requests = []  # timestamps

    def allow_request(self) -> bool:
        now = time.time()
        cutoff = now - self.window_size

        # Remove old entries
        self.requests = [ts for ts in self.requests if ts > cutoff]

        if len(self.requests) < self.max_requests:
            self.requests.append(now)
            return True
        return False
```

**Pros:** Precise, no boundary issues
**Cons:** Memory intensive for high-volume APIs

#### 3. Sliding Window Counter

```python
class SlidingWindowCounter:
    def __init__(self, window_size: int, max_requests: int):
        self.window_size = window_size
        self.max_requests = max_requests

    def allow_request(self, redis, key: str) -> bool:
        now = time.time()
        current_window = int(now // self.window_size)
        previous_window = current_window - 1

        # Get counts from current and previous windows
        current_count = int(redis.get(f"{key}:{current_window}") or 0)
        previous_count = int(redis.get(f"{key}:{previous_window}") or 0)

        # Calculate weighted count
        elapsed_in_window = now % self.window_size
        weight = elapsed_in_window / self.window_size
        weighted_count = current_count + previous_count * (1 - weight)

        if weighted_count < self.max_requests:
            redis.incr(f"{key}:{current_window}")
            redis.expire(f"{key}:{current_window}", self.window_size * 2)
            return True
        return False
```

**Pros:** Memory efficient, relatively precise
**Cons:** Approximate (but usually acceptable)

### Algorithm Comparison

| Algorithm | Memory | Precision | Burst Handling | Complexity |
|-----------|--------|-----------|----------------|------------|
| Token Bucket | O(1) | Approximate | Allows bursts | Low |
| Leaky Bucket | O(1) | Approximate | Smooths bursts | Low |
| Fixed Window | O(1) | Boundary issues | Allows 2x burst | Low |
| Sliding Log | O(n) | Exact | No bursts | Medium |
| Sliding Counter | O(1) | Approximate | Limited bursts | Medium |

### Distributed Rate Limiting

```python
class DistributedRateLimiter:
    def __init__(self, redis_cluster):
        self.redis = redis_cluster

    def is_allowed(self, key: str, limit: int, window: int) -> bool:
        # Lua script for atomic check-and-increment
        lua_script = """
        local key = KEYS[1]
        local limit = tonumber(ARGV[1])
        local window = tonumber(ARGV[2])

        local current = redis.call('INCR', key)
        if current == 1 then
            redis.call('EXPIRE', key, window)
        end

        if current > limit then
            return 0
        end
        return 1
        """
        return bool(self.redis.eval(lua_script, 1, key, limit, window))
```

### High-Level Architecture

```
┌──────────┐     ┌──────────────┐     ┌─────────────┐
│  Client  │────▶│   Gateway    │────▶│  API Server │
└──────────┘     │ (Rate Check) │     └─────────────┘
                 └──────┬───────┘
                        │
         ┌──────────────┼──────────────┐
         │              │              │
  ┌──────▼──────┐ ┌─────▼─────┐ ┌──────▼──────┐
  │ Redis Node 1│ │Redis Node 2│ │Redis Node 3│
  └─────────────┘ └───────────┘ └─────────────┘
                        │
                 ┌──────▼──────┐
                 │   Config    │
                 │   Service   │
                 └─────────────┘
```

### Handling Failures

```python
class ResilientRateLimiter:
    def is_allowed(self, key: str) -> bool:
        try:
            return self._check_redis(key)
        except RedisConnectionError:
            # Fallback strategies:

            # Option 1: Fail open (allow all)
            return True

            # Option 2: Fail closed (deny all)
            # return False

            # Option 3: Local rate limiting
            # return self._local_limiter.is_allowed(key)
```

### Tradeoffs: Fail Open vs Fail Closed

| Strategy | Pros | Cons |
|----------|------|------|
| **Fail Open** | Service continues, better UX | Risk of abuse during outage |
| **Fail Closed** | Protects backend | Blocks legitimate traffic |
| **Local Fallback** | Best of both | Inconsistent limits across nodes |

### Response Headers

```http
HTTP/1.1 429 Too Many Requests
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 0
X-RateLimit-Reset: 1640000000
Retry-After: 60
```

### Interview Discussion Points

1. **How to handle distributed rate limiting accurately?**
   - Accept slight inaccuracy for performance
   - Use sticky sessions for per-user limits
   - Sync counters asynchronously with eventual consistency

2. **How to rate limit by multiple dimensions?**
   - Hierarchical limits: global → tenant → user → endpoint
   - Check all limits, reject on first violation
   - Use composite keys: `{tenant}:{user}:{endpoint}`

3. **How to handle rate limit changes?**
   - Config service with pub/sub notifications
   - Gradual rollout to avoid thundering herd
   - Grace period for limit decreases

---

## 3. Distributed Cache

**Problem:** Design a distributed caching system like Memcached or Redis.

### Requirements

**Functional:**
- GET, SET, DELETE operations
- TTL (time-to-live) support
- Support various data types (strings, lists, sets, hashes)
- Atomic operations (INCR, DECR)

**Non-Functional:**
- Sub-millisecond latency
- High throughput (millions of ops/second)
- High availability (99.99%)
- Horizontal scalability
- Memory efficient

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         Clients                                  │
└────────────────────────────┬────────────────────────────────────┘
                             │
                    ┌────────▼────────┐
                    │  Client Library │
                    │ (Consistent Hash)│
                    └────────┬────────┘
                             │
        ┌────────────────────┼────────────────────┐
        │                    │                    │
┌───────▼───────┐   ┌────────▼────────┐  ┌────────▼────────┐
│  Cache Node 1 │   │  Cache Node 2   │  │  Cache Node 3   │
│   (Primary)   │   │   (Primary)     │  │   (Primary)     │
└───────┬───────┘   └────────┬────────┘  └────────┬────────┘
        │                    │                    │
┌───────▼───────┐   ┌────────▼────────┐  ┌────────▼────────┐
│   Replica 1   │   │    Replica 2    │  │    Replica 3    │
└───────────────┘   └─────────────────┘  └─────────────────┘
```

### Data Partitioning: Consistent Hashing

```python
import hashlib
from bisect import bisect_right

class ConsistentHash:
    def __init__(self, nodes: list, virtual_nodes: int = 150):
        self.virtual_nodes = virtual_nodes
        self.ring = []
        self.node_map = {}

        for node in nodes:
            self.add_node(node)

    def _hash(self, key: str) -> int:
        return int(hashlib.md5(key.encode()).hexdigest(), 16)

    def add_node(self, node: str):
        for i in range(self.virtual_nodes):
            virtual_key = f"{node}:{i}"
            hash_val = self._hash(virtual_key)
            self.ring.append(hash_val)
            self.node_map[hash_val] = node
        self.ring.sort()

    def remove_node(self, node: str):
        for i in range(self.virtual_nodes):
            virtual_key = f"{node}:{i}"
            hash_val = self._hash(virtual_key)
            self.ring.remove(hash_val)
            del self.node_map[hash_val]

    def get_node(self, key: str) -> str:
        if not self.ring:
            return None
        hash_val = self._hash(key)
        idx = bisect_right(self.ring, hash_val) % len(self.ring)
        return self.node_map[self.ring[idx]]
```

### Eviction Policies

```python
class LRUCache:
    """Least Recently Used - evict oldest accessed item"""
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache = OrderedDict()

    def get(self, key: str):
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        return None

    def put(self, key: str, value: any):
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)


class LFUCache:
    """Least Frequently Used - evict least accessed item"""
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache = {}
        self.freq = defaultdict(OrderedDict)
        self.min_freq = 0

    def get(self, key: str):
        if key not in self.cache:
            return None

        value, freq = self.cache[key]
        del self.freq[freq][key]
        if not self.freq[freq]:
            del self.freq[freq]
            if self.min_freq == freq:
                self.min_freq += 1

        self.freq[freq + 1][key] = True
        self.cache[key] = (value, freq + 1)
        return value
```

### Eviction Policy Comparison

| Policy | Use Case | Pros | Cons |
|--------|----------|------|------|
| **LRU** | General purpose | Simple, good for temporal locality | Scan resistance issue |
| **LFU** | Stable access patterns | Keeps popular items | Slow to adapt to changes |
| **FIFO** | Simple workloads | Very simple | Ignores access patterns |
| **Random** | Uniform access | No overhead | Unpredictable |
| **TTL** | Time-sensitive data | Automatic cleanup | May evict hot data |

### Cache Invalidation Strategies

```python
class CacheInvalidation:
    # 1. Write-through: Write to cache and DB simultaneously
    def write_through(self, key: str, value: any):
        self.cache.set(key, value)
        self.db.write(key, value)

    # 2. Write-behind: Write to cache, async write to DB
    def write_behind(self, key: str, value: any):
        self.cache.set(key, value)
        self.queue.enqueue(WriteTask(key, value))

    # 3. Cache-aside: Application manages cache
    def cache_aside_read(self, key: str):
        value = self.cache.get(key)
        if value is None:
            value = self.db.read(key)
            self.cache.set(key, value)
        return value

    def cache_aside_write(self, key: str, value: any):
        self.db.write(key, value)
        self.cache.delete(key)  # Invalidate, not update
```

### Invalidation Strategy Tradeoffs

| Strategy | Consistency | Performance | Complexity |
|----------|-------------|-------------|------------|
| **Write-through** | Strong | Slower writes | Low |
| **Write-behind** | Eventual | Fast writes | High (async) |
| **Cache-aside** | Eventual | Fast reads | Medium |
| **Read-through** | Eventual | Simple client | Medium |

### Replication Strategies

```python
class ReplicatedCache:
    def __init__(self, primary, replicas):
        self.primary = primary
        self.replicas = replicas

    # Synchronous replication
    def sync_write(self, key: str, value: any):
        self.primary.set(key, value)
        for replica in self.replicas:
            replica.set(key, value)  # Wait for all

    # Asynchronous replication
    def async_write(self, key: str, value: any):
        self.primary.set(key, value)
        for replica in self.replicas:
            self.queue.enqueue(ReplicateTask(replica, key, value))

    # Read from replica (load distribution)
    def read(self, key: str, prefer_replica: bool = True):
        if prefer_replica and self.replicas:
            return random.choice(self.replicas).get(key)
        return self.primary.get(key)
```

### Hot Key Handling

```python
class HotKeyHandler:
    def __init__(self):
        self.local_cache = TTLCache(maxsize=1000, ttl=1)
        self.hot_keys = set()

    def get(self, key: str):
        # Check local cache for hot keys
        if key in self.hot_keys:
            value = self.local_cache.get(key)
            if value:
                return value

        # Fetch from distributed cache
        value = self.distributed_cache.get(key)

        # Detect and cache hot keys locally
        if self._is_hot(key):
            self.hot_keys.add(key)
            self.local_cache[key] = value

        return value

    def _is_hot(self, key: str) -> bool:
        # Track access frequency
        self.access_count[key] += 1
        return self.access_count[key] > HOT_THRESHOLD
```

### Interview Discussion Points

1. **How to handle cache stampede?**
   - Locking: Only one request fetches from DB
   - Early expiration: Refresh before TTL expires
   - Probabilistic early expiration: Random refresh window

2. **How to ensure consistency between cache and DB?**
   - Delete cache on write (cache-aside)
   - Use distributed transactions (expensive)
   - Accept eventual consistency with short TTLs

3. **How to handle node failures?**
   - Consistent hashing minimizes redistribution
   - Replicas take over reads
   - Graceful degradation to database

---

## 4. Key-Value Store

**Problem:** Design a distributed key-value store like DynamoDB or Cassandra.

### Requirements

**Functional:**
- PUT(key, value), GET(key), DELETE(key)
- Support large values (up to 1MB)
- Range queries (optional)
- Versioning / conflict resolution

**Non-Functional:**
- High availability (99.99%)
- Tunable consistency (strong to eventual)
- Horizontal scalability
- Low latency (< 10ms p99)
- Durable (no data loss)

### CAP Theorem Considerations

```
         Consistency
            /\
           /  \
          /    \
         /  CA  \
        /________\
       /\        /\
      /  \  CP  /  \
     / AP \    /    \
    /______\  /______\
Availability  Partition Tolerance
```

**Design Choice:** AP with tunable consistency (like Cassandra/DynamoDB)

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                          Clients                                 │
└────────────────────────────┬────────────────────────────────────┘
                             │
                    ┌────────▼────────┐
                    │   Coordinator   │
                    │     Node        │
                    └────────┬────────┘
                             │
        ┌────────────────────┼────────────────────┐
        │                    │                    │
┌───────▼───────┐   ┌────────▼────────┐  ┌────────▼────────┐
│   Storage     │   │    Storage      │  │    Storage      │
│   Node 1      │   │    Node 2       │  │    Node 3       │
│ [A-F keys]    │   │  [G-M keys]     │  │  [N-Z keys]     │
└───────────────┘   └─────────────────┘  └─────────────────┘
```

### Data Partitioning

```python
class PartitionManager:
    def __init__(self, nodes: list, replication_factor: int = 3):
        self.hash_ring = ConsistentHash(nodes)
        self.rf = replication_factor

    def get_nodes_for_key(self, key: str) -> list:
        """Return N nodes responsible for this key (N = replication factor)"""
        primary = self.hash_ring.get_node(key)
        replicas = self._get_next_nodes(primary, self.rf - 1)
        return [primary] + replicas

    def _get_next_nodes(self, start_node: str, count: int) -> list:
        """Get next N unique physical nodes on the ring"""
        nodes = []
        # Walk the ring to find next unique nodes
        # (skip virtual nodes of same physical node)
        ...
        return nodes
```

### Quorum-Based Consistency

```python
class QuorumCoordinator:
    def __init__(self, n: int = 3, w: int = 2, r: int = 2):
        self.n = n  # Total replicas
        self.w = w  # Write quorum
        self.r = r  # Read quorum
        # Note: W + R > N ensures consistency

    def write(self, key: str, value: any) -> bool:
        nodes = self.get_replica_nodes(key)
        responses = []

        for node in nodes:
            try:
                response = node.write(key, value, version=time.time())
                responses.append(response)
            except NodeUnavailable:
                continue

        # Success if W nodes acknowledged
        return len([r for r in responses if r.success]) >= self.w

    def read(self, key: str) -> any:
        nodes = self.get_replica_nodes(key)
        responses = []

        for node in nodes:
            try:
                response = node.read(key)
                responses.append(response)
            except NodeUnavailable:
                continue

        if len(responses) < self.r:
            raise InsufficientReplicas()

        # Return most recent version
        return max(responses, key=lambda r: r.version).value
```

### Consistency Level Tradeoffs

| Config | W | R | Consistency | Availability | Latency |
|--------|---|---|-------------|--------------|---------|
| Strong | N | 1 | Strong | Low | High write |
| Strong | 1 | N | Strong | Low | High read |
| Quorum | ⌈N/2⌉+1 | ⌈N/2⌉+1 | Strong | Medium | Medium |
| Eventual | 1 | 1 | Eventual | High | Low |

### Write Path (LSM-Tree)

```python
class LSMTree:
    def __init__(self):
        self.memtable = SortedDict()  # In-memory, sorted
        self.wal = WriteAheadLog()    # Durability
        self.sstables = []            # On-disk, immutable

    def write(self, key: str, value: any):
        # 1. Write to WAL for durability
        self.wal.append(key, value)

        # 2. Write to memtable
        self.memtable[key] = value

        # 3. Flush to SSTable if memtable is full
        if len(self.memtable) > MEMTABLE_THRESHOLD:
            self._flush_to_sstable()

    def _flush_to_sstable(self):
        # Create new SSTable from memtable
        sstable = SSTable.from_dict(self.memtable)
        self.sstables.append(sstable)
        self.memtable.clear()
        self.wal.clear()

    def read(self, key: str):
        # 1. Check memtable first
        if key in self.memtable:
            return self.memtable[key]

        # 2. Search SSTables (newest first)
        for sstable in reversed(self.sstables):
            value = sstable.get(key)
            if value is not None:
                return value

        return None
```

### SSTable with Bloom Filter

```python
class SSTable:
    def __init__(self, data: dict):
        self.data = sorted(data.items())
        self.index = self._build_sparse_index()
        self.bloom_filter = BloomFilter()

        for key in data:
            self.bloom_filter.add(key)

    def get(self, key: str):
        # Quick negative check
        if not self.bloom_filter.might_contain(key):
            return None

        # Binary search using sparse index
        return self._binary_search(key)


class BloomFilter:
    def __init__(self, size: int = 10000, hash_count: int = 3):
        self.size = size
        self.hash_count = hash_count
        self.bits = bitarray(size)

    def add(self, key: str):
        for i in range(self.hash_count):
            idx = self._hash(key, i) % self.size
            self.bits[idx] = 1

    def might_contain(self, key: str) -> bool:
        for i in range(self.hash_count):
            idx = self._hash(key, i) % self.size
            if not self.bits[idx]:
                return False
        return True  # Might be false positive
```

### Conflict Resolution

```python
class VectorClock:
    """Track causality for conflict detection"""
    def __init__(self):
        self.clock = defaultdict(int)

    def increment(self, node_id: str):
        self.clock[node_id] += 1

    def merge(self, other: 'VectorClock'):
        for node, time in other.clock.items():
            self.clock[node] = max(self.clock[node], time)

    def compare(self, other: 'VectorClock') -> str:
        less = more = equal = True
        for node in set(self.clock.keys()) | set(other.clock.keys()):
            if self.clock[node] < other.clock[node]:
                more = False
            elif self.clock[node] > other.clock[node]:
                less = False
            if self.clock[node] != other.clock[node]:
                equal = False

        if equal:
            return "EQUAL"
        if less:
            return "BEFORE"
        if more:
            return "AFTER"
        return "CONCURRENT"  # Conflict!


class ConflictResolver:
    def resolve(self, values: list) -> any:
        # Strategy 1: Last-write-wins (timestamp)
        return max(values, key=lambda v: v.timestamp).value

        # Strategy 2: Return all (let application decide)
        # return [v.value for v in values]

        # Strategy 3: Merge (for CRDTs)
        # return self.merge_crdt(values)
```

### Interview Discussion Points

1. **How to handle node failures?**
   - Hinted handoff: Temporarily store data on healthy nodes
   - Anti-entropy: Background sync with Merkle trees
   - Read repair: Fix inconsistencies during reads

2. **How to handle hot partitions?**
   - Add random suffix to keys (scatter-gather)
   - Client-side caching
   - Read from replicas

3. **LSM vs B-Tree tradeoffs?**
   - LSM: Better write throughput, space amplification
   - B-Tree: Better read performance, in-place updates

---

## 5. Message Queue

**Problem:** Design a distributed message queue like Kafka or RabbitMQ.

### Requirements

**Functional:**
- Producers publish messages to topics
- Consumers subscribe and receive messages
- Message ordering guarantees (within partition)
- At-least-once / exactly-once delivery
- Message retention and replay

**Non-Functional:**
- High throughput (millions of messages/second)
- Low latency (< 10ms p99)
- Durable (no message loss)
- Horizontally scalable
- Fault tolerant

### High-Level Architecture

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  Producer 1  │     │  Producer 2  │     │  Producer 3  │
└──────┬───────┘     └──────┬───────┘     └──────┬───────┘
       │                    │                    │
       └────────────────────┼────────────────────┘
                            │
                   ┌────────▼────────┐
                   │  Load Balancer  │
                   └────────┬────────┘
                            │
        ┌───────────────────┼───────────────────┐
        │                   │                   │
┌───────▼───────┐  ┌────────▼────────┐  ┌───────▼───────┐
│   Broker 1    │  │    Broker 2     │  │   Broker 3    │
│ ┌───────────┐ │  │ ┌───────────┐   │  │ ┌───────────┐ │
│ │Partition 0│ │  │ │Partition 1│   │  │ │Partition 2│ │
│ │ (Leader)  │ │  │ │ (Leader)  │   │  │ │ (Leader)  │ │
│ └───────────┘ │  │ └───────────┘   │  │ └───────────┘ │
│ ┌───────────┐ │  │ ┌───────────┐   │  │ ┌───────────┐ │
│ │Partition 1│ │  │ │Partition 2│   │  │ │Partition 0│ │
│ │ (Replica) │ │  │ │ (Replica) │   │  │ │ (Replica) │ │
│ └───────────┘ │  │ └───────────┘   │  │ └───────────┘ │
└───────────────┘  └─────────────────┘  └───────────────┘
        │                   │                   │
        └───────────────────┼───────────────────┘
                            │
       ┌────────────────────┼────────────────────┐
       │                    │                    │
┌──────▼───────┐     ┌──────▼───────┐     ┌──────▼───────┐
│  Consumer 1  │     │  Consumer 2  │     │  Consumer 3  │
│ (Group A)    │     │ (Group A)    │     │ (Group B)    │
└──────────────┘     └──────────────┘     └──────────────┘
```

### Message Storage

```python
class Partition:
    def __init__(self, partition_id: int):
        self.id = partition_id
        self.log = []  # Append-only log
        self.offset = 0
        self.index = {}  # offset -> file position

    def append(self, message: bytes) -> int:
        """Append message, return offset"""
        entry = LogEntry(
            offset=self.offset,
            timestamp=time.time(),
            key_size=len(message.key),
            value_size=len(message.value),
            key=message.key,
            value=message.value
        )
        self.log.append(entry)
        self.index[self.offset] = len(self.log) - 1
        self.offset += 1
        return entry.offset

    def read(self, start_offset: int, max_bytes: int) -> list:
        """Read messages starting from offset"""
        messages = []
        current_bytes = 0

        for i in range(start_offset, self.offset):
            entry = self.log[self.index[i]]
            entry_size = entry.size()

            if current_bytes + entry_size > max_bytes:
                break

            messages.append(entry)
            current_bytes += entry_size

        return messages
```

### Consumer Groups

```python
class ConsumerGroup:
    def __init__(self, group_id: str, topic: str):
        self.group_id = group_id
        self.topic = topic
        self.consumers = {}  # consumer_id -> assigned partitions
        self.offsets = {}    # partition -> committed offset

    def rebalance(self):
        """Redistribute partitions among consumers"""
        partitions = self.get_topic_partitions()
        consumers = list(self.consumers.keys())

        if not consumers:
            return

        # Round-robin assignment
        assignments = defaultdict(list)
        for i, partition in enumerate(partitions):
            consumer = consumers[i % len(consumers)]
            assignments[consumer].append(partition)

        self.consumers = dict(assignments)

    def commit_offset(self, consumer_id: str, partition: int, offset: int):
        """Commit consumed offset"""
        self.offsets[partition] = offset

    def get_offset(self, partition: int) -> int:
        """Get last committed offset"""
        return self.offsets.get(partition, 0)
```

### Delivery Guarantees

```python
class Producer:
    def send(self, topic: str, key: bytes, value: bytes,
             acks: str = "all") -> Future:
        """
        acks options:
        - "0": Fire and forget (no guarantee)
        - "1": Leader acknowledged (at-least-once)
        - "all": All replicas acknowledged (strongest)
        """
        partition = self._select_partition(topic, key)
        broker = self._get_leader(topic, partition)

        request = ProduceRequest(topic, partition, key, value)
        return broker.send(request, acks=acks)

    def _select_partition(self, topic: str, key: bytes) -> int:
        if key:
            # Hash key for consistent partition
            return hash(key) % self.num_partitions[topic]
        else:
            # Round-robin for keyless messages
            return self._round_robin_counter.next() % self.num_partitions[topic]


class Consumer:
    def poll(self, timeout_ms: int) -> list:
        """Fetch messages from assigned partitions"""
        messages = []

        for partition in self.assigned_partitions:
            offset = self.group.get_offset(partition)
            batch = self.broker.fetch(
                self.topic, partition, offset, self.max_bytes
            )
            messages.extend(batch)

        return messages

    def commit(self):
        """Commit current offsets"""
        for partition in self.assigned_partitions:
            self.group.commit_offset(
                self.consumer_id, partition, self.current_offset[partition]
            )
```

### Delivery Guarantee Tradeoffs

| Guarantee | Implementation | Tradeoff |
|-----------|---------------|----------|
| **At-most-once** | Commit before processing | May lose messages |
| **At-least-once** | Commit after processing | May duplicate messages |
| **Exactly-once** | Idempotent producers + transactions | Higher latency, complexity |

### Replication

```python
class ReplicationManager:
    def __init__(self, replication_factor: int = 3):
        self.rf = replication_factor

    def replicate(self, partition: Partition, message: bytes):
        leader = partition.leader
        followers = partition.followers

        # Write to leader first
        offset = leader.append(message)

        # Replicate to followers
        acks = 1  # Leader already acked
        for follower in followers:
            try:
                follower.replicate(message, offset)
                acks += 1
            except FollowerUnavailable:
                continue

        # Check if enough replicas acknowledged
        if acks < self.min_insync_replicas:
            raise InsufficientReplicas()

        return offset

    def elect_leader(self, partition: Partition):
        """Elect new leader from in-sync replicas"""
        isr = partition.in_sync_replicas

        if not isr:
            # Unclean leader election (may lose data)
            new_leader = partition.followers[0]
        else:
            # Clean election from ISR
            new_leader = isr[0]

        partition.leader = new_leader
        self._notify_clients(partition)
```

### Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Storage | Append-only log | Fast sequential writes, easy replay |
| Ordering | Per-partition only | Scalability vs global ordering |
| Retention | Time or size based | Balance storage vs replay ability |
| Replication | Sync to ISR | Durability with acceptable latency |

### Interview Discussion Points

1. **How to handle slow consumers?**
   - Consumer groups with independent offsets
   - Increase partitions for parallelism
   - Message retention allows catch-up

2. **How to achieve exactly-once?**
   - Idempotent producers (sequence numbers)
   - Transactional writes
   - Consumer idempotency

3. **Kafka vs RabbitMQ tradeoffs?**
   - Kafka: Higher throughput, log retention, replay
   - RabbitMQ: Lower latency, flexible routing, message acknowledgment

---

## 6. Notification System

**Problem:** Design a notification system supporting push, SMS, and email.

### Requirements

**Functional:**
- Send push notifications (iOS, Android, Web)
- Send SMS and email notifications
- Support scheduled notifications
- User notification preferences
- Rate limiting and deduplication

**Non-Functional:**
- High throughput (millions/day)
- Low latency for real-time notifications
- High deliverability
- Fault tolerant
- Scalable

### High-Level Architecture

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   Service A  │     │   Service B  │     │   Service C  │
└──────┬───────┘     └──────┬───────┘     └──────┬───────┘
       │                    │                    │
       └────────────────────┼────────────────────┘
                            │
                   ┌────────▼────────┐
                   │ Notification API │
                   └────────┬────────┘
                            │
                   ┌────────▼────────┐
                   │   Kafka Queue   │
                   └────────┬────────┘
                            │
        ┌───────────────────┼───────────────────┐
        │                   │                   │
┌───────▼───────┐  ┌────────▼────────┐  ┌───────▼───────┐
│  Push Worker  │  │   SMS Worker    │  │ Email Worker  │
└───────┬───────┘  └────────┬────────┘  └───────┬───────┘
        │                   │                   │
┌───────▼───────┐  ┌────────▼────────┐  ┌───────▼───────┐
│  APNs / FCM   │  │ Twilio/Nexmo    │  │  SendGrid     │
└───────────────┘  └─────────────────┘  └───────────────┘
```

### Data Model

```sql
-- User notification preferences
CREATE TABLE notification_preferences (
    user_id UUID PRIMARY KEY,
    push_enabled BOOLEAN DEFAULT TRUE,
    email_enabled BOOLEAN DEFAULT TRUE,
    sms_enabled BOOLEAN DEFAULT FALSE,
    quiet_hours_start TIME,
    quiet_hours_end TIME,
    frequency_limit INTEGER DEFAULT 10,  -- per hour
    channel_preferences JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Device tokens for push notifications
CREATE TABLE device_tokens (
    id UUID PRIMARY KEY,
    user_id UUID NOT NULL,
    platform VARCHAR(20) NOT NULL,  -- ios, android, web
    token TEXT NOT NULL,
    app_version VARCHAR(20),
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMPTZ DEFAULT NOW(),

    UNIQUE(user_id, token)
);

-- Notification log
CREATE TABLE notifications (
    id UUID PRIMARY KEY,
    user_id UUID NOT NULL,
    type VARCHAR(50) NOT NULL,
    channel VARCHAR(20) NOT NULL,  -- push, sms, email
    title VARCHAR(255),
    body TEXT,
    data JSONB,
    status VARCHAR(20) DEFAULT 'pending',
    sent_at TIMESTAMPTZ,
    delivered_at TIMESTAMPTZ,
    read_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW(),

    INDEX idx_notifications_user (user_id, created_at DESC)
);
```

### Notification Service

```python
class NotificationService:
    def __init__(self):
        self.queue = KafkaProducer()
        self.rate_limiter = RateLimiter()
        self.preference_cache = RedisCache()

    async def send(self, request: NotificationRequest):
        # 1. Check user preferences
        prefs = await self._get_preferences(request.user_id)
        if not self._should_send(request, prefs):
            return NotificationResult(status="filtered")

        # 2. Rate limiting
        if not self.rate_limiter.allow(request.user_id):
            return NotificationResult(status="rate_limited")

        # 3. Deduplication
        if self._is_duplicate(request):
            return NotificationResult(status="deduplicated")

        # 4. Queue for async processing
        await self.queue.send(
            topic=f"notifications.{request.channel}",
            key=request.user_id,
            value=request.to_json()
        )

        return NotificationResult(status="queued")

    def _should_send(self, request, prefs) -> bool:
        # Check channel enabled
        if request.channel == "push" and not prefs.push_enabled:
            return False

        # Check quiet hours
        if self._in_quiet_hours(prefs):
            if request.priority != "urgent":
                return False

        return True
```

### Push Notification Worker

```python
class PushWorker:
    def __init__(self):
        self.apns = APNsClient()
        self.fcm = FCMClient()

    async def process(self, notification: Notification):
        # Get user's device tokens
        tokens = await self.get_device_tokens(notification.user_id)

        results = []
        for token in tokens:
            try:
                if token.platform == "ios":
                    result = await self.apns.send(
                        token=token.token,
                        payload=self._build_apns_payload(notification)
                    )
                elif token.platform == "android":
                    result = await self.fcm.send(
                        token=token.token,
                        payload=self._build_fcm_payload(notification)
                    )

                results.append(result)

            except InvalidTokenError:
                # Mark token as inactive
                await self.deactivate_token(token)

            except ProviderError as e:
                # Retry with backoff
                await self.retry_queue.send(notification, delay=60)

        return results

    def _build_apns_payload(self, notification):
        return {
            "aps": {
                "alert": {
                    "title": notification.title,
                    "body": notification.body
                },
                "sound": "default",
                "badge": notification.badge_count
            },
            "data": notification.data
        }
```

### Priority and Batching

```python
class NotificationBatcher:
    def __init__(self):
        self.batches = defaultdict(list)
        self.batch_size = 1000
        self.flush_interval = 5  # seconds

    async def add(self, notification: Notification):
        priority = notification.priority

        if priority == "urgent":
            # Send immediately
            await self.send_single(notification)
        else:
            # Batch for efficiency
            self.batches[notification.channel].append(notification)

            if len(self.batches[notification.channel]) >= self.batch_size:
                await self.flush(notification.channel)

    async def flush(self, channel: str):
        batch = self.batches[channel]
        self.batches[channel] = []

        if channel == "email":
            # Batch send via SendGrid
            await self.email_client.send_batch(batch)
        elif channel == "sms":
            # SMS typically not batched
            for notification in batch:
                await self.sms_client.send(notification)
```

### Delivery Tracking

```python
class DeliveryTracker:
    def __init__(self):
        self.metrics = PrometheusMetrics()

    async def track_sent(self, notification_id: str, channel: str):
        await self.db.update(
            notification_id,
            status="sent",
            sent_at=datetime.now()
        )
        self.metrics.increment("notifications_sent", channel=channel)

    async def track_delivered(self, notification_id: str):
        """Called via webhook from provider"""
        await self.db.update(
            notification_id,
            status="delivered",
            delivered_at=datetime.now()
        )

    async def track_failed(self, notification_id: str, reason: str):
        await self.db.update(
            notification_id,
            status="failed",
            error=reason
        )
        self.metrics.increment("notifications_failed", reason=reason)
```

### Channel Selection Strategy

| Channel | Use Case | Latency | Cost | Deliverability |
|---------|----------|---------|------|----------------|
| **Push** | Real-time alerts | < 1s | Free | Medium (token expiry) |
| **SMS** | Critical/auth | < 5s | High | High |
| **Email** | Non-urgent, long content | Minutes | Low | Medium (spam filters) |

### Interview Discussion Points

1. **How to handle notification storms?**
   - Aggregate similar notifications
   - Implement digest mode (daily/weekly summary)
   - Rate limit per user and globally

2. **How to ensure high deliverability?**
   - Multiple providers with fallback
   - Clean invalid tokens regularly
   - Monitor delivery rates and bounce backs

3. **How to handle user timezone?**
   - Store user timezone in preferences
   - Schedule based on local time
   - Respect quiet hours

---

## 7. Real-Time Chat

**Problem:** Design a real-time messaging system like WhatsApp or Slack.

### Requirements

**Functional:**
- 1:1 and group messaging
- Real-time message delivery
- Message history and search
- Online presence indicators
- Read receipts and typing indicators
- File/media sharing

**Non-Functional:**
- Low latency (< 100ms)
- High availability (99.99%)
- Message ordering guaranteed
- End-to-end encryption (optional)
- Millions of concurrent connections

### High-Level Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                          Clients                                  │
└────────────────────────────┬─────────────────────────────────────┘
                             │ WebSocket
                    ┌────────▼────────┐
                    │  Load Balancer  │
                    │  (L4 / Sticky)  │
                    └────────┬────────┘
                             │
        ┌────────────────────┼────────────────────┐
        │                    │                    │
┌───────▼───────┐   ┌────────▼────────┐   ┌───────▼───────┐
│  Chat Server 1│   │  Chat Server 2  │   │  Chat Server 3│
│  (WebSocket)  │   │   (WebSocket)   │   │  (WebSocket)  │
└───────┬───────┘   └────────┬────────┘   └───────┬───────┘
        │                    │                    │
        └────────────────────┼────────────────────┘
                             │
                    ┌────────▼────────┐
                    │   Redis Pub/Sub │
                    │  (Message Bus)  │
                    └────────┬────────┘
                             │
        ┌────────────────────┼────────────────────┐
        │                    │                    │
┌───────▼───────┐   ┌────────▼────────┐   ┌───────▼───────┐
│  Message DB   │   │  Presence       │   │  Media Store  │
│  (Cassandra)  │   │  Service        │   │    (S3)       │
└───────────────┘   └─────────────────┘   └───────────────┘
```

### Connection Management

```python
class ChatServer:
    def __init__(self):
        self.connections = {}  # user_id -> WebSocket
        self.redis = Redis()

    async def handle_connect(self, websocket, user_id: str):
        # Store connection
        self.connections[user_id] = websocket

        # Register in Redis (for cross-server routing)
        await self.redis.hset("user_servers", user_id, self.server_id)

        # Update presence
        await self.presence_service.set_online(user_id)

        # Subscribe to user's message channel
        await self.redis.subscribe(f"user:{user_id}")

    async def handle_disconnect(self, user_id: str):
        del self.connections[user_id]
        await self.redis.hdel("user_servers", user_id)
        await self.presence_service.set_offline(user_id)

    async def send_to_user(self, user_id: str, message: dict):
        if user_id in self.connections:
            # User connected to this server
            await self.connections[user_id].send_json(message)
        else:
            # Route to correct server via Redis
            await self.redis.publish(f"user:{user_id}", json.dumps(message))
```

### Message Flow

```python
class MessageService:
    async def send_message(self, sender_id: str, chat_id: str,
                          content: str, message_type: str = "text"):
        # 1. Generate message ID and timestamp
        message = Message(
            id=snowflake.generate(),
            chat_id=chat_id,
            sender_id=sender_id,
            content=content,
            type=message_type,
            timestamp=time.time(),
            status="sent"
        )

        # 2. Persist message
        await self.db.insert(message)

        # 3. Get chat participants
        participants = await self.get_chat_participants(chat_id)

        # 4. Deliver to online recipients
        for user_id in participants:
            if user_id != sender_id:
                await self.deliver_message(user_id, message)

        # 5. Send delivery confirmation to sender
        await self.send_ack(sender_id, message.id, "sent")

        return message

    async def deliver_message(self, user_id: str, message: Message):
        # Check if user is online
        server_id = await self.redis.hget("user_servers", user_id)

        if server_id:
            # User is online - deliver in real-time
            await self.redis.publish(
                f"user:{user_id}",
                message.to_json()
            )
        else:
            # User offline - queue for push notification
            await self.push_queue.send(
                PushNotification(user_id, message)
            )
```

### Message Storage Schema

```sql
-- Cassandra schema (optimized for chat queries)
CREATE TABLE messages (
    chat_id UUID,
    message_id TIMEUUID,
    sender_id UUID,
    content TEXT,
    message_type VARCHAR,
    status VARCHAR,
    created_at TIMESTAMP,
    PRIMARY KEY ((chat_id), message_id)
) WITH CLUSTERING ORDER BY (message_id DESC);

-- Chat metadata
CREATE TABLE chats (
    chat_id UUID PRIMARY KEY,
    type VARCHAR,  -- direct, group
    name VARCHAR,
    participants SET<UUID>,
    created_at TIMESTAMP,
    last_message_at TIMESTAMP
);

-- User's chat list (denormalized for fast access)
CREATE TABLE user_chats (
    user_id UUID,
    last_message_at TIMESTAMP,
    chat_id UUID,
    unread_count INT,
    PRIMARY KEY ((user_id), last_message_at, chat_id)
) WITH CLUSTERING ORDER BY (last_message_at DESC);
```

### Read Receipts

```python
class ReadReceiptService:
    async def mark_read(self, user_id: str, chat_id: str, message_id: str):
        # Update read pointer
        await self.redis.hset(
            f"read_pointers:{chat_id}",
            user_id,
            message_id
        )

        # Notify other participants
        participants = await self.get_participants(chat_id)
        for participant_id in participants:
            if participant_id != user_id:
                await self.send_receipt(
                    participant_id,
                    ReadReceipt(chat_id, user_id, message_id)
                )

    async def get_read_status(self, chat_id: str, message_id: str):
        """Get who has read this message"""
        pointers = await self.redis.hgetall(f"read_pointers:{chat_id}")
        readers = []
        for user_id, last_read in pointers.items():
            if last_read >= message_id:
                readers.append(user_id)
        return readers
```

### Presence Service

```python
class PresenceService:
    def __init__(self):
        self.redis = Redis()
        self.heartbeat_interval = 30  # seconds

    async def set_online(self, user_id: str):
        await self.redis.setex(
            f"presence:{user_id}",
            self.heartbeat_interval * 2,
            "online"
        )
        await self._notify_contacts(user_id, "online")

    async def heartbeat(self, user_id: str):
        await self.redis.expire(
            f"presence:{user_id}",
            self.heartbeat_interval * 2
        )

    async def set_offline(self, user_id: str):
        await self.redis.delete(f"presence:{user_id}")
        await self.redis.set(
            f"last_seen:{user_id}",
            time.time()
        )
        await self._notify_contacts(user_id, "offline")

    async def get_presence(self, user_ids: list) -> dict:
        pipe = self.redis.pipeline()
        for user_id in user_ids:
            pipe.get(f"presence:{user_id}")
            pipe.get(f"last_seen:{user_id}")

        results = await pipe.execute()
        presence = {}
        for i, user_id in enumerate(user_ids):
            status = results[i * 2]
            last_seen = results[i * 2 + 1]
            presence[user_id] = {
                "status": status or "offline",
                "last_seen": last_seen
            }
        return presence
```

### Scaling WebSocket Connections

| Approach | Pros | Cons |
|----------|------|------|
| **Sticky Sessions** | Simple, connection affinity | Uneven load distribution |
| **Redis Pub/Sub** | Flexible routing | Redis becomes bottleneck |
| **Consistent Hashing** | Predictable routing | Rebalancing on node changes |
| **Service Mesh** | Modern, feature-rich | Complexity overhead |

### Interview Discussion Points

1. **How to handle message ordering in groups?**
   - Lamport timestamps or vector clocks
   - Single writer per group (leader-based)
   - Accept eventual consistency, resolve client-side

2. **How to scale to millions of connections?**
   - One server: ~100K connections
   - Stateless chat servers with Redis routing
   - Regional deployment to reduce latency

3. **How to implement end-to-end encryption?**
   - Signal Protocol (Double Ratchet)
   - Key exchange on device, server never sees plaintext
   - Store encrypted messages only

---

## 8. News Feed / Timeline

**Problem:** Design a social media news feed like Facebook or Twitter.

### Requirements

**Functional:**
- Users follow other users
- Posts appear in followers' feeds
- Feed shows posts in reverse chronological order (or ranked)
- Support for likes, comments, shares
- Real-time feed updates

**Non-Functional:**
- Low latency feed generation (< 200ms)
- High read throughput (100K+ reads/second)
- Eventually consistent (acceptable delay)
- Support for viral content (millions of followers)

### Feed Generation Approaches

#### 1. Pull Model (Fan-out on Read)

```python
class PullFeedService:
    def get_feed(self, user_id: str, limit: int = 20) -> list:
        # 1. Get users I follow
        following = self.get_following(user_id)

        # 2. Fetch recent posts from each
        all_posts = []
        for followed_id in following:
            posts = self.get_posts(followed_id, limit=100)
            all_posts.extend(posts)

        # 3. Sort and truncate
        all_posts.sort(key=lambda p: p.timestamp, reverse=True)
        return all_posts[:limit]
```

**Pros:** Simple, no storage overhead, fresh data
**Cons:** Slow for users following many accounts, high read latency

#### 2. Push Model (Fan-out on Write)

```python
class PushFeedService:
    async def create_post(self, user_id: str, content: str):
        post = Post(id=generate_id(), author=user_id, content=content)
        await self.post_db.insert(post)

        # Fan-out to all followers
        followers = await self.get_followers(user_id)
        for follower_id in followers:
            await self.feed_cache.lpush(
                f"feed:{follower_id}",
                post.id
            )
            # Trim to keep only recent posts
            await self.feed_cache.ltrim(f"feed:{follower_id}", 0, 1000)

    def get_feed(self, user_id: str, limit: int = 20) -> list:
        # Just read from pre-computed feed
        post_ids = self.feed_cache.lrange(f"feed:{user_id}", 0, limit)
        return self.post_db.get_batch(post_ids)
```

**Pros:** Fast reads, simple feed retrieval
**Cons:** Expensive for celebrities (millions of followers), storage heavy

#### 3. Hybrid Model

```python
class HybridFeedService:
    CELEBRITY_THRESHOLD = 10000

    async def create_post(self, user_id: str, content: str):
        post = Post(id=generate_id(), author=user_id, content=content)
        await self.post_db.insert(post)

        follower_count = await self.get_follower_count(user_id)

        if follower_count < self.CELEBRITY_THRESHOLD:
            # Regular user: push to all followers
            await self._fan_out_to_followers(user_id, post)
        else:
            # Celebrity: store in celebrity posts table
            await self.celebrity_posts.insert(post)

    async def get_feed(self, user_id: str, limit: int = 20) -> list:
        # Get pre-computed feed (from regular users)
        feed_posts = await self.feed_cache.lrange(f"feed:{user_id}", 0, limit)

        # Get celebrity posts (pull on read)
        celebrities = await self.get_followed_celebrities(user_id)
        celebrity_posts = await self.get_recent_celebrity_posts(celebrities)

        # Merge and sort
        all_posts = feed_posts + celebrity_posts
        all_posts.sort(key=lambda p: p.timestamp, reverse=True)
        return all_posts[:limit]
```

### Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                           Clients                                 │
└────────────────────────────┬─────────────────────────────────────┘
                             │
                    ┌────────▼────────┐
                    │   API Gateway   │
                    └────────┬────────┘
                             │
        ┌────────────────────┼────────────────────┐
        │                    │                    │
┌───────▼───────┐   ┌────────▼────────┐   ┌───────▼───────┐
│ Feed Service  │   │  Post Service   │   │ Graph Service │
│ (Read Path)   │   │  (Write Path)   │   │ (Following)   │
└───────┬───────┘   └────────┬────────┘   └───────┬───────┘
        │                    │                    │
        │           ┌────────▼────────┐           │
        │           │   Fan-out       │           │
        │           │   Workers       │           │
        │           └────────┬────────┘           │
        │                    │                    │
┌───────▼───────┐   ┌────────▼────────┐   ┌───────▼───────┐
│  Feed Cache   │   │   Post Store    │   │  Graph Store  │
│   (Redis)     │   │  (Cassandra)    │   │   (Neo4j)     │
└───────────────┘   └─────────────────┘   └───────────────┘
```

### Ranking Algorithm

```python
class FeedRanker:
    def rank(self, posts: list, user_id: str) -> list:
        scored_posts = []

        for post in posts:
            score = self._calculate_score(post, user_id)
            scored_posts.append((score, post))

        scored_posts.sort(reverse=True)
        return [post for score, post in scored_posts]

    def _calculate_score(self, post: Post, user_id: str) -> float:
        # Time decay (newer = higher score)
        age_hours = (time.time() - post.timestamp) / 3600
        time_score = 1 / (1 + age_hours)

        # Engagement signals
        engagement_score = (
            post.likes * 1.0 +
            post.comments * 2.0 +
            post.shares * 3.0
        ) / 1000

        # Affinity with author
        affinity = self._get_affinity(user_id, post.author)

        # Content type boost
        content_boost = 1.2 if post.has_media else 1.0

        return (
            time_score * 0.3 +
            engagement_score * 0.3 +
            affinity * 0.3 +
            content_boost * 0.1
        )
```

### Model Comparison

| Aspect | Pull | Push | Hybrid |
|--------|------|------|--------|
| Write Cost | Low | High (fan-out) | Medium |
| Read Cost | High | Low | Low-Medium |
| Storage | Low | High | Medium |
| Latency | High | Low | Low |
| Celebrity Handling | Good | Poor | Good |

### Interview Discussion Points

1. **How to handle celebrities with millions of followers?**
   - Hybrid approach: pull for celebrities, push for regular users
   - Async fan-out with priority queues
   - Cache celebrity posts separately

2. **How to implement real-time feed updates?**
   - WebSocket for active users
   - Long polling as fallback
   - Push notification for important updates

3. **How to rank posts fairly?**
   - Multi-factor scoring (time, engagement, affinity)
   - A/B testing different algorithms
   - User feedback loop for personalization

---

## 9. Search Autocomplete

**Problem:** Design a typeahead/autocomplete system for search suggestions.

### Requirements

**Functional:**
- Return top suggestions as user types
- Rank by popularity/relevance
- Support personalization
- Handle typos and fuzzy matching

**Non-Functional:**
- Ultra-low latency (< 50ms)
- High availability
- Scale to billions of queries
- Real-time trend updates

### High-Level Architecture

```
┌──────────────┐
│    Client    │
│ (Debounced)  │
└──────┬───────┘
       │
┌──────▼───────┐
│     CDN      │
│ (Edge Cache) │
└──────┬───────┘
       │
┌──────▼───────┐
│ Autocomplete │
│   Service    │
└──────┬───────┘
       │
┌──────▼───────┐     ┌──────────────┐
│  Trie Store  │────▶│  Analytics   │
│   (Redis)    │     │   Pipeline   │
└──────────────┘     └──────────────┘
```

### Trie Data Structure

```python
class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end = False
        self.suggestions = []  # Top suggestions at this prefix

class AutocompleteTrie:
    def __init__(self, max_suggestions: int = 10):
        self.root = TrieNode()
        self.max_suggestions = max_suggestions

    def insert(self, word: str, weight: int):
        node = self.root
        for char in word.lower():
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]

            # Update suggestions at each prefix
            self._update_suggestions(node, word, weight)

        node.is_end = True

    def _update_suggestions(self, node: TrieNode, word: str, weight: int):
        # Keep top N suggestions sorted by weight
        existing = [s for s in node.suggestions if s[1] != word]
        existing.append((weight, word))
        existing.sort(reverse=True)
        node.suggestions = existing[:self.max_suggestions]

    def search(self, prefix: str) -> list:
        node = self.root
        for char in prefix.lower():
            if char not in node.children:
                return []
            node = node.children[char]

        return [word for weight, word in node.suggestions]
```

### Redis Implementation

```python
class RedisAutocomplete:
    def __init__(self, redis_client):
        self.redis = redis_client

    def index(self, term: str, weight: int):
        """Index a term with all its prefixes"""
        term_lower = term.lower()

        for i in range(1, len(term_lower) + 1):
            prefix = term_lower[:i]
            # Use sorted set for weighted suggestions
            self.redis.zadd(
                f"autocomplete:{prefix}",
                {term: weight}
            )
            # Keep only top N
            self.redis.zremrangebyrank(
                f"autocomplete:{prefix}",
                0, -11  # Keep top 10
            )

    def search(self, prefix: str, limit: int = 10) -> list:
        """Get suggestions for prefix"""
        key = f"autocomplete:{prefix.lower()}"
        # Get top N by score (weight)
        results = self.redis.zrevrange(key, 0, limit - 1)
        return results

    def increment_weight(self, term: str, delta: int = 1):
        """Increase weight when term is selected"""
        term_lower = term.lower()
        for i in range(1, len(term_lower) + 1):
            prefix = term_lower[:i]
            self.redis.zincrby(f"autocomplete:{prefix}", delta, term)
```

### Handling Scale

```python
class ShardedAutocomplete:
    def __init__(self, num_shards: int = 16):
        self.shards = [Redis(f"shard-{i}") for i in range(num_shards)]

    def _get_shard(self, prefix: str) -> Redis:
        shard_id = hash(prefix[0]) % len(self.shards)
        return self.shards[shard_id]

    def search(self, prefix: str, limit: int = 10) -> list:
        shard = self._get_shard(prefix)
        return shard.zrevrange(f"autocomplete:{prefix}", 0, limit - 1)


class TieredAutocomplete:
    """Two-tier: Hot prefixes in memory, rest in Redis"""

    def __init__(self):
        self.hot_cache = LRUCache(maxsize=100000)  # In-memory
        self.redis = Redis()

    def search(self, prefix: str, limit: int = 10) -> list:
        # Check hot cache first
        if prefix in self.hot_cache:
            return self.hot_cache[prefix]

        # Fall back to Redis
        results = self.redis.zrevrange(
            f"autocomplete:{prefix}", 0, limit - 1
        )

        # Cache popular prefixes
        if len(prefix) <= 3:  # Short prefixes are frequently queried
            self.hot_cache[prefix] = results

        return results
```

### Real-Time Trend Updates

```python
class TrendingAutocomplete:
    def __init__(self):
        self.redis = Redis()
        self.time_window = 3600  # 1 hour

    def record_search(self, term: str):
        """Record search for trending calculation"""
        timestamp = int(time.time())
        bucket = timestamp // 60  # Per-minute buckets

        # Increment in current time bucket
        self.redis.zincrby(f"trending:{bucket}", 1, term)
        self.redis.expire(f"trending:{bucket}", self.time_window)

    def get_trending(self, limit: int = 10) -> list:
        """Get currently trending searches"""
        current_bucket = int(time.time()) // 60
        buckets = [current_bucket - i for i in range(60)]  # Last hour

        # Merge all buckets
        self.redis.zunionstore(
            "trending:merged",
            [f"trending:{b}" for b in buckets]
        )

        return self.redis.zrevrange("trending:merged", 0, limit - 1)
```

### Fuzzy Matching

```python
class FuzzyAutocomplete:
    def search(self, query: str, limit: int = 10) -> list:
        # 1. Exact prefix match
        exact = self.trie.search(query)

        if len(exact) >= limit:
            return exact[:limit]

        # 2. Fuzzy match with edit distance
        fuzzy = self._fuzzy_search(query, max_distance=1)

        # 3. Phonetic matching (Soundex/Metaphone)
        phonetic = self._phonetic_search(query)

        # Combine and deduplicate
        all_results = exact + fuzzy + phonetic
        seen = set()
        unique = []
        for term in all_results:
            if term not in seen:
                seen.add(term)
                unique.append(term)

        return unique[:limit]

    def _fuzzy_search(self, query: str, max_distance: int) -> list:
        """Find terms within edit distance"""
        results = []

        # Generate possible corrections
        candidates = self._generate_edits(query, max_distance)

        for candidate in candidates:
            matches = self.trie.search(candidate)
            results.extend(matches)

        return results
```

### Optimization Techniques

| Technique | Benefit | Tradeoff |
|-----------|---------|----------|
| **Prefix sharding** | Distribute load | Cross-shard aggregation |
| **Edge caching (CDN)** | Ultra-low latency | Stale data for trending |
| **Bloom filters** | Quick negative lookup | False positives |
| **Compression** | Memory efficiency | CPU overhead |

### Interview Discussion Points

1. **How to handle different languages?**
   - Language-specific tokenization
   - Unicode normalization
   - Per-language indexes

2. **How to personalize suggestions?**
   - User search history
   - Blend global + personal results
   - Privacy considerations

3. **How to prevent gaming/spam?**
   - Rate limiting on indexing
   - Profanity filters
   - Human review for trending

---

## 10. Web Crawler

**Problem:** Design a web crawler that indexes the internet for a search engine.

### Requirements

**Functional:**
- Crawl billions of web pages
- Extract and follow links
- Handle different content types
- Respect robots.txt
- Detect and handle duplicates

**Non-Functional:**
- High throughput (1000+ pages/second)
- Politeness (don't overload sites)
- Fault tolerant
- Scalable horizontally
- Fresh content prioritization

### High-Level Architecture

```
┌────────────────────────────────────────────────────────────────────┐
│                         URL Frontier                                │
│                    (Priority Queue + Politeness)                    │
└────────────────────────────┬───────────────────────────────────────┘
                             │
        ┌────────────────────┼────────────────────┐
        │                    │                    │
┌───────▼───────┐   ┌────────▼────────┐   ┌───────▼───────┐
│  Crawler 1    │   │   Crawler 2     │   │   Crawler 3   │
│   Worker      │   │    Worker       │   │    Worker     │
└───────┬───────┘   └────────┬────────┘   └───────┬───────┘
        │                    │                    │
        └────────────────────┼────────────────────┘
                             │
                    ┌────────▼────────┐
                    │   DNS Resolver  │
                    │    (Cached)     │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │  Content Store  │
                    │   (Raw HTML)    │
                    └────────┬────────┘
                             │
        ┌────────────────────┼────────────────────┐
        │                    │                    │
┌───────▼───────┐   ┌────────▼────────┐   ┌───────▼───────┐
│  Link         │   │   Duplicate     │   │   Content     │
│  Extractor    │   │   Detector      │   │   Parser      │
└───────┬───────┘   └────────┬────────┘   └───────┬───────┘
        │                    │                    │
        ▼                    ▼                    ▼
   URL Frontier         Discard            Search Index
```

### URL Frontier

```python
class URLFrontier:
    def __init__(self, num_queues: int = 1000):
        self.priority_queue = PriorityQueue()  # Global priority
        self.host_queues = defaultdict(deque)   # Per-host queues
        self.last_crawl = {}                    # Host -> timestamp
        self.crawl_delay = 1.0                  # Seconds between requests

    def add_url(self, url: str, priority: int):
        parsed = urlparse(url)
        host = parsed.netloc

        # Add to host queue
        self.host_queues[host].append((priority, url))

        # Add host to priority queue if not already waiting
        if host not in self.priority_queue:
            next_crawl_time = self.last_crawl.get(host, 0) + self.crawl_delay
            self.priority_queue.put((next_crawl_time, priority, host))

    def get_next_url(self) -> str:
        while True:
            # Get next host to crawl
            next_time, priority, host = self.priority_queue.get()

            # Wait if necessary (politeness)
            now = time.time()
            if next_time > now:
                time.sleep(next_time - now)

            # Get URL from host queue
            if self.host_queues[host]:
                _, url = self.host_queues[host].popleft()
                self.last_crawl[host] = time.time()

                # Re-add host if more URLs
                if self.host_queues[host]:
                    next_crawl = time.time() + self.crawl_delay
                    self.priority_queue.put((next_crawl, priority, host))

                return url
```

### Crawler Worker

```python
class CrawlerWorker:
    def __init__(self):
        self.frontier = URLFrontier()
        self.dns_cache = DNSCache()
        self.robots_cache = RobotsCache()
        self.seen_urls = BloomFilter(capacity=10_000_000_000)

    async def crawl(self):
        while True:
            url = self.frontier.get_next_url()

            # Check if already seen
            if url in self.seen_urls:
                continue
            self.seen_urls.add(url)

            # Check robots.txt
            if not self._is_allowed(url):
                continue

            try:
                # Fetch page
                content = await self._fetch(url)

                # Store content
                await self._store(url, content)

                # Extract and queue links
                links = self._extract_links(url, content)
                for link in links:
                    self.frontier.add_url(link, self._calculate_priority(link))

            except Exception as e:
                self._handle_error(url, e)

    async def _fetch(self, url: str) -> bytes:
        parsed = urlparse(url)

        # DNS resolution (cached)
        ip = await self.dns_cache.resolve(parsed.netloc)

        # HTTP request with timeout
        async with aiohttp.ClientSession() as session:
            async with session.get(
                url,
                timeout=aiohttp.ClientTimeout(total=30),
                headers={"User-Agent": "MyBot/1.0"}
            ) as response:
                return await response.read()
```

### Duplicate Detection

```python
class DuplicateDetector:
    def __init__(self):
        self.url_seen = BloomFilter(capacity=10_000_000_000)
        self.content_hashes = {}  # SimHash -> canonical URL

    def is_url_duplicate(self, url: str) -> bool:
        normalized = self._normalize_url(url)
        if normalized in self.url_seen:
            return True
        self.url_seen.add(normalized)
        return False

    def is_content_duplicate(self, content: str) -> tuple:
        # Compute SimHash for near-duplicate detection
        simhash = self._compute_simhash(content)

        # Check against existing hashes
        for existing_hash, canonical_url in self.content_hashes.items():
            if self._hamming_distance(simhash, existing_hash) < 3:
                return True, canonical_url

        self.content_hashes[simhash] = content
        return False, None

    def _compute_simhash(self, content: str) -> int:
        """64-bit SimHash for near-duplicate detection"""
        tokens = content.lower().split()
        v = [0] * 64

        for token in tokens:
            token_hash = hash(token)
            for i in range(64):
                if token_hash & (1 << i):
                    v[i] += 1
                else:
                    v[i] -= 1

        simhash = 0
        for i in range(64):
            if v[i] > 0:
                simhash |= (1 << i)

        return simhash
```

### Robots.txt Parser

```python
class RobotsCache:
    def __init__(self):
        self.cache = TTLCache(maxsize=100000, ttl=86400)  # 24 hour cache

    async def is_allowed(self, url: str, user_agent: str = "*") -> bool:
        parsed = urlparse(url)
        robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"

        # Fetch robots.txt if not cached
        if robots_url not in self.cache:
            try:
                content = await self._fetch(robots_url)
                rules = self._parse_robots(content)
                self.cache[robots_url] = rules
            except:
                # If can't fetch, assume allowed
                self.cache[robots_url] = RobotsRules(allow_all=True)

        rules = self.cache[robots_url]
        return rules.is_allowed(parsed.path, user_agent)

    def get_crawl_delay(self, host: str) -> float:
        robots_url = f"https://{host}/robots.txt"
        if robots_url in self.cache:
            return self.cache[robots_url].crawl_delay
        return 1.0  # Default
```

### Priority Calculation

```python
class PriorityCalculator:
    def calculate(self, url: str, parent_url: str = None) -> int:
        score = 0

        parsed = urlparse(url)

        # Domain authority (pre-computed)
        score += self.domain_scores.get(parsed.netloc, 0) * 10

        # URL depth (shorter = higher priority)
        depth = len(parsed.path.split('/'))
        score -= depth * 2

        # Content type hints
        if parsed.path.endswith(('.html', '.htm', '/')):
            score += 5
        elif parsed.path.endswith(('.pdf', '.doc')):
            score += 2

        # Freshness (if known)
        if parent_url:
            parent_freshness = self.freshness_scores.get(parent_url, 0)
            score += parent_freshness

        return max(0, score)
```

### Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| URL Dedup | Bloom Filter | Memory efficient, false positive OK |
| Content Dedup | SimHash | Near-duplicate detection |
| Frontier | Per-host queues | Politeness enforcement |
| Storage | Distributed file system | Handle petabytes of data |
| DNS | Local cache | Reduce DNS queries |

### Interview Discussion Points

1. **How to handle dynamic content (JavaScript)?**
   - Headless browser for JS rendering
   - Separate queue for JS-heavy sites
   - Accept that some content won't be indexed

2. **How to prioritize freshness?**
   - Track content change frequency
   - Prioritize frequently updated sites
   - Different crawl schedules per site

3. **How to handle crawler traps?**
   - URL length limits
   - Depth limits per domain
   - Pattern detection for generated URLs

---

## 11. Video Streaming Platform

**Problem:** Design a video streaming service like YouTube or Netflix.

### Requirements

**Functional:**
- Upload and process videos
- Stream videos with adaptive quality
- Support multiple devices and resolutions
- Video recommendations
- Comments, likes, view counts

**Non-Functional:**
- Low latency playback start (< 2s)
- Smooth streaming (minimal buffering)
- Global availability via CDN
- Support millions of concurrent viewers
- Cost-efficient storage and delivery

### High-Level Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                          Upload Flow                              │
└────────────────────────────┬─────────────────────────────────────┘
                             │
┌──────────────┐    ┌────────▼────────┐    ┌──────────────┐
│   Creator    │───▶│  Upload Service │───▶│  Object Store │
└──────────────┘    └────────┬────────┘    │    (S3)       │
                             │             └───────┬───────┘
                    ┌────────▼────────┐            │
                    │  Transcoding    │◀───────────┘
                    │  Pipeline       │
                    └────────┬────────┘
                             │
        ┌────────────────────┼────────────────────┐
        │                    │                    │
┌───────▼───────┐   ┌────────▼────────┐   ┌───────▼───────┐
│  1080p/H.264  │   │   720p/H.264    │   │   480p/H.264  │
└───────┬───────┘   └────────┬────────┘   └───────┬───────┘
        │                    │                    │
        └────────────────────┼────────────────────┘
                             │
                    ┌────────▼────────┐
                    │      CDN        │
                    │  (Edge Caches)  │
                    └────────┬────────┘
                             │
┌──────────────────────────────────────────────────────────────────┐
│                          Playback Flow                            │
└──────────────────────────────────────────────────────────────────┘
```

### Video Processing Pipeline

```python
class VideoProcessor:
    PROFILES = [
        {"resolution": "1080p", "bitrate": 5000, "codec": "h264"},
        {"resolution": "720p", "bitrate": 2500, "codec": "h264"},
        {"resolution": "480p", "bitrate": 1000, "codec": "h264"},
        {"resolution": "360p", "bitrate": 500, "codec": "h264"},
    ]

    async def process(self, video_id: str, source_path: str):
        # 1. Validate video
        metadata = await self._extract_metadata(source_path)
        if not self._is_valid(metadata):
            raise InvalidVideoError()

        # 2. Transcode to multiple resolutions
        tasks = []
        for profile in self.PROFILES:
            task = self._transcode(video_id, source_path, profile)
            tasks.append(task)

        outputs = await asyncio.gather(*tasks)

        # 3. Generate HLS/DASH segments
        for output in outputs:
            await self._segment(output)

        # 4. Generate thumbnails
        await self._generate_thumbnails(source_path, video_id)

        # 5. Update database
        await self._update_status(video_id, "ready")

    async def _transcode(self, video_id: str, source: str, profile: dict):
        output = f"{video_id}/{profile['resolution']}.mp4"

        cmd = [
            "ffmpeg", "-i", source,
            "-c:v", profile["codec"],
            "-b:v", f"{profile['bitrate']}k",
            "-vf", f"scale=-2:{profile['resolution'][:-1]}",
            "-c:a", "aac", "-b:a", "128k",
            output
        ]

        await asyncio.create_subprocess_exec(*cmd)
        return output

    async def _segment(self, video_path: str):
        """Create HLS segments for adaptive streaming"""
        cmd = [
            "ffmpeg", "-i", video_path,
            "-hls_time", "6",
            "-hls_list_size", "0",
            "-hls_segment_filename", f"{video_path}_%03d.ts",
            f"{video_path}.m3u8"
        ]
        await asyncio.create_subprocess_exec(*cmd)
```

### Adaptive Bitrate Streaming

```python
# HLS Master Playlist (m3u8)
"""
#EXTM3U
#EXT-X-STREAM-INF:BANDWIDTH=5000000,RESOLUTION=1920x1080
1080p/playlist.m3u8
#EXT-X-STREAM-INF:BANDWIDTH=2500000,RESOLUTION=1280x720
720p/playlist.m3u8
#EXT-X-STREAM-INF:BANDWIDTH=1000000,RESOLUTION=854x480
480p/playlist.m3u8
#EXT-X-STREAM-INF:BANDWIDTH=500000,RESOLUTION=640x360
360p/playlist.m3u8
"""

class AdaptivePlayer:
    def __init__(self):
        self.current_quality = "720p"
        self.buffer_health = 10  # seconds

    def select_quality(self, bandwidth: int) -> str:
        """Select quality based on available bandwidth"""
        if bandwidth > 6000:
            return "1080p"
        elif bandwidth > 3000:
            return "720p"
        elif bandwidth > 1500:
            return "480p"
        else:
            return "360p"

    def adjust_quality(self, measured_bandwidth: int, buffer_level: float):
        """Adaptive bitrate selection"""
        target = self.select_quality(measured_bandwidth)

        # Be conservative when buffer is low
        if buffer_level < 5:
            target = self._lower_quality(target)

        # Smooth transitions
        if target != self.current_quality:
            self.current_quality = target
```

### CDN and Caching

```python
class CDNManager:
    def __init__(self):
        self.edge_locations = ["us-east", "us-west", "eu-west", "ap-east"]

    def get_stream_url(self, video_id: str, user_location: str) -> str:
        # Select nearest edge location
        edge = self._nearest_edge(user_location)

        # Return CDN URL with signed token
        token = self._generate_token(video_id, expires=3600)
        return f"https://{edge}.cdn.example.com/v/{video_id}/master.m3u8?token={token}"

    def warm_cache(self, video_id: str, predicted_popularity: float):
        """Pre-populate cache for expected popular videos"""
        if predicted_popularity > 0.8:
            # Push to all edge locations
            for edge in self.edge_locations:
                self._push_to_edge(edge, video_id)
        elif predicted_popularity > 0.5:
            # Push to high-traffic edges only
            self._push_to_edge("us-east", video_id)
            self._push_to_edge("eu-west", video_id)
```

### View Count System

```python
class ViewCounter:
    def __init__(self):
        self.redis = Redis()
        self.batch_size = 1000

    async def record_view(self, video_id: str, user_id: str):
        # Deduplicate views (same user within window)
        view_key = f"viewed:{video_id}:{user_id}"
        if await self.redis.exists(view_key):
            return

        await self.redis.setex(view_key, 3600, "1")  # 1 hour window

        # Increment counter in Redis
        await self.redis.incr(f"views:{video_id}")

        # Batch sync to database
        count = await self.redis.get(f"views:{video_id}")
        if int(count) % self.batch_size == 0:
            await self._sync_to_db(video_id, count)

    async def _sync_to_db(self, video_id: str, count: int):
        await self.db.execute(
            "UPDATE videos SET view_count = %s WHERE id = %s",
            [count, video_id]
        )
```

### Storage Strategy

| Content Type | Storage | Retention | Access Pattern |
|--------------|---------|-----------|----------------|
| Original upload | S3 Standard | Forever | Rare |
| Transcoded (popular) | S3 Standard + CDN | Forever | Frequent |
| Transcoded (old) | S3 Glacier | Forever | Rare |
| Thumbnails | S3 + CDN | Forever | Very frequent |
| Metadata | Database | Forever | Frequent |

### Interview Discussion Points

1. **How to handle live streaming?**
   - RTMP ingest, HLS/DASH output
   - Lower segment duration (2s vs 6s)
   - Edge transcoding for latency

2. **How to optimize for mobile?**
   - Smaller segments for faster start
   - Lower resolution defaults
   - Offline download support

3. **How to handle copyright?**
   - Content ID fingerprinting
   - Hash-based duplicate detection
   - DMCA takedown workflow

---

## 12. Distributed File Storage

**Problem:** Design a distributed file system like HDFS or Google File System.

### Requirements

**Functional:**
- Store and retrieve large files (GB to TB)
- Support append operations
- Handle concurrent readers
- Maintain file metadata

**Non-Functional:**
- High throughput for large sequential reads/writes
- Fault tolerant (survive hardware failures)
- Scalable to petabytes
- Strong consistency for metadata

### High-Level Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                          Clients                                  │
└────────────────────────────┬─────────────────────────────────────┘
                             │
                    ┌────────▼────────┐
                    │   Master Node   │
                    │   (Metadata)    │
                    │ ┌─────────────┐ │
                    │ │  Namespace  │ │
                    │ │   (Files)   │ │
                    │ └─────────────┘ │
                    │ ┌─────────────┐ │
                    │ │   Chunk     │ │
                    │ │   Mapping   │ │
                    │ └─────────────┘ │
                    └────────┬────────┘
                             │
        ┌────────────────────┼────────────────────┐
        │                    │                    │
┌───────▼───────┐   ┌────────▼────────┐   ┌───────▼───────┐
│  Chunk Server │   │  Chunk Server   │   │  Chunk Server │
│      1        │   │       2         │   │       3       │
│ ┌───────────┐ │   │ ┌───────────┐   │   │ ┌───────────┐ │
│ │  Chunk A  │ │   │ │  Chunk A  │   │   │ │  Chunk B  │ │
│ │  Chunk C  │ │   │ │  Chunk B  │   │   │ │  Chunk A  │ │
│ └───────────┘ │   │ └───────────┘   │   │ └───────────┘ │
└───────────────┘   └─────────────────┘   └───────────────┘
```

### Chunk Management

```python
class MasterNode:
    def __init__(self):
        self.namespace = {}  # file_path -> FileMetadata
        self.chunk_locations = {}  # chunk_id -> [chunk_servers]
        self.chunk_servers = {}  # server_id -> ChunkServerInfo

    def create_file(self, path: str) -> FileHandle:
        if path in self.namespace:
            raise FileExistsError()

        file_meta = FileMetadata(
            path=path,
            chunks=[],
            size=0,
            created_at=time.time()
        )
        self.namespace[path] = file_meta
        return FileHandle(path, file_meta)

    def allocate_chunk(self, file_path: str) -> ChunkAllocation:
        # Generate unique chunk ID
        chunk_id = self._generate_chunk_id()

        # Select chunk servers (considering load, rack diversity)
        servers = self._select_chunk_servers(count=3)

        # Record allocation
        self.chunk_locations[chunk_id] = servers

        # Add to file's chunk list
        self.namespace[file_path].chunks.append(chunk_id)

        return ChunkAllocation(
            chunk_id=chunk_id,
            primary=servers[0],
            secondaries=servers[1:]
        )

    def _select_chunk_servers(self, count: int) -> list:
        """Select servers with rack awareness"""
        available = list(self.chunk_servers.values())

        # Sort by available space
        available.sort(key=lambda s: s.available_space, reverse=True)

        selected = []
        racks_used = set()

        for server in available:
            if len(selected) >= count:
                break
            # Prefer different racks for fault tolerance
            if server.rack not in racks_used or len(selected) < count:
                selected.append(server)
                racks_used.add(server.rack)

        return selected
```

### Write Path

```python
class ChunkClient:
    def __init__(self, master: MasterNode):
        self.master = master
        self.chunk_size = 64 * 1024 * 1024  # 64 MB

    async def write_file(self, path: str, data: bytes):
        # 1. Create file on master
        handle = self.master.create_file(path)

        # 2. Split into chunks
        chunks = self._split_into_chunks(data)

        for chunk_data in chunks:
            # 3. Get chunk allocation from master
            allocation = self.master.allocate_chunk(path)

            # 4. Write to primary, which replicates to secondaries
            await self._write_chunk(allocation, chunk_data)

    async def _write_chunk(self, allocation: ChunkAllocation, data: bytes):
        # Push data to all replicas first (pipeline)
        for server in [allocation.primary] + allocation.secondaries:
            await server.receive_data(allocation.chunk_id, data)

        # Commit on primary (which coordinates with secondaries)
        await allocation.primary.commit(allocation.chunk_id)


class ChunkServer:
    def __init__(self):
        self.chunks = {}  # chunk_id -> ChunkData
        self.pending = {}  # chunk_id -> received data

    async def receive_data(self, chunk_id: str, data: bytes):
        """Receive data (not yet committed)"""
        self.pending[chunk_id] = data

    async def commit(self, chunk_id: str):
        """Commit chunk to stable storage"""
        if chunk_id not in self.pending:
            raise ChunkNotFoundError()

        data = self.pending.pop(chunk_id)

        # Write to disk
        chunk = ChunkData(
            id=chunk_id,
            data=data,
            checksum=self._compute_checksum(data)
        )
        await self._write_to_disk(chunk)
        self.chunks[chunk_id] = chunk
```

### Read Path

```python
class ChunkClient:
    async def read_file(self, path: str) -> bytes:
        # 1. Get file metadata from master
        file_meta = self.master.get_file(path)

        # 2. Read each chunk
        data_parts = []
        for chunk_id in file_meta.chunks:
            chunk_data = await self._read_chunk(chunk_id)
            data_parts.append(chunk_data)

        return b''.join(data_parts)

    async def _read_chunk(self, chunk_id: str) -> bytes:
        # Get chunk locations from master
        locations = self.master.get_chunk_locations(chunk_id)

        # Try each replica until success
        for server in locations:
            try:
                data = await server.read_chunk(chunk_id)
                if self._verify_checksum(data):
                    return data.content
            except (ChunkCorruptError, ServerUnavailableError):
                continue

        raise ChunkUnavailableError(chunk_id)
```

### Heartbeat and Lease

```python
class ChunkServer:
    async def heartbeat_loop(self):
        while True:
            # Report status to master
            status = ChunkServerStatus(
                server_id=self.id,
                chunks=list(self.chunks.keys()),
                available_space=self._get_available_space(),
                load=self._get_load()
            )
            await self.master.heartbeat(status)
            await asyncio.sleep(10)  # Every 10 seconds


class MasterNode:
    def __init__(self):
        self.server_last_seen = {}
        self.lease_duration = 60  # seconds

    async def heartbeat(self, status: ChunkServerStatus):
        self.server_last_seen[status.server_id] = time.time()
        self._update_chunk_locations(status)

    async def monitor_loop(self):
        while True:
            now = time.time()
            for server_id, last_seen in list(self.server_last_seen.items()):
                if now - last_seen > self.lease_duration:
                    await self._handle_server_failure(server_id)
            await asyncio.sleep(10)

    async def _handle_server_failure(self, server_id: str):
        # Re-replicate chunks from failed server
        affected_chunks = self._get_chunks_on_server(server_id)
        for chunk_id in affected_chunks:
            await self._re_replicate(chunk_id)
```

### Replication and Recovery

```python
class ReplicationManager:
    def __init__(self, replication_factor: int = 3):
        self.rf = replication_factor

    async def re_replicate(self, chunk_id: str):
        """Re-replicate under-replicated chunks"""
        current_replicas = self.master.chunk_locations[chunk_id]

        if len(current_replicas) >= self.rf:
            return  # Sufficient replicas

        # Select new server
        existing_racks = {s.rack for s in current_replicas}
        new_server = self._select_new_server(exclude_racks=existing_racks)

        # Copy from existing replica
        source = current_replicas[0]
        await self._copy_chunk(chunk_id, source, new_server)

        # Update metadata
        self.master.chunk_locations[chunk_id].append(new_server)

    async def balance_chunks(self):
        """Background job to balance chunk distribution"""
        server_loads = self._calculate_server_loads()
        avg_load = sum(server_loads.values()) / len(server_loads)

        overloaded = [s for s, l in server_loads.items() if l > avg_load * 1.2]
        underloaded = [s for s, l in server_loads.items() if l < avg_load * 0.8]

        for source in overloaded:
            if not underloaded:
                break
            target = underloaded.pop()
            await self._migrate_chunk(source, target)
```

### Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Chunk Size | 64 MB | Balance metadata overhead vs parallelism |
| Replication | 3 replicas | Survive 2 failures, rack-aware |
| Consistency | Single master | Simplify metadata consistency |
| Writes | Primary-based | Ordered writes, easy conflict resolution |

### HDFS vs GFS vs Alternatives

| Aspect | HDFS | GFS | Object Storage (S3) |
|--------|------|-----|---------------------|
| Chunk Size | 128 MB | 64 MB | Variable |
| Consistency | Strong (metadata) | Strong (metadata) | Eventual |
| Use Case | Batch processing | Batch processing | General purpose |
| Append | Supported | Optimized | Not efficient |

### Interview Discussion Points

1. **How to handle master failure?**
   - Hot standby with replicated state
   - Paxos/Raft for leader election
   - Transaction log for recovery

2. **How to optimize for small files?**
   - Combine small files into larger blocks
   - Separate metadata service
   - Or use different storage system

3. **How to handle hot spots?**
   - Increase replication for hot chunks
   - Client-side caching
   - Load-based replica selection

---

## Summary: Key Tradeoffs

| System | Key Tradeoff |
|--------|-------------|
| URL Shortener | Collision handling vs simplicity |
| Rate Limiter | Accuracy vs latency |
| Distributed Cache | Consistency vs availability |
| Key-Value Store | CAP theorem choices |
| Message Queue | Delivery guarantees vs performance |
| Notification System | Reliability vs speed |
| Real-Time Chat | Scalability vs consistency |
| News Feed | Push vs pull model |
| Search Autocomplete | Latency vs freshness |
| Web Crawler | Throughput vs politeness |

---

## Interview Tips

1. **Start with requirements** - Clarify functional and non-functional requirements
2. **Do back-of-envelope calculations** - Estimate scale, storage, bandwidth
3. **Draw high-level architecture first** - Show the big picture
4. **Deep dive on key components** - Show depth in critical areas
5. **Discuss tradeoffs explicitly** - There's no perfect solution
6. **Consider failure scenarios** - How does the system degrade?
7. **Think about operations** - Monitoring, deployment, debugging
