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

### Workflow & Orchestration
13. [Task Scheduler](#13-task-scheduler)

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

### Extended Tradeoffs

#### ID Generation: Base62 vs Base64 vs UUID vs Snowflake
| Aspect | Base62 | Base64 | UUID | Snowflake |
|--------|--------|--------|------|-----------|
| URL Length | 7 chars (3.5T) | 6 chars (68B) | 22 chars | 11 chars |
| URL Safety | Yes (alphanumeric) | No (+, /, =) | Yes | Yes |
| Predictability | Sequential = predictable | Hash = unpredictable | Random | Time-ordered |
| Collision Risk | None (counter) | Hash truncation | Negligible | None |
| Coordination | Requires counter sync | Stateless | Stateless | Clock + machine ID |
| When to use | Single DC, simple | Avoid for URLs | Distributed, no order needed | Distributed, time-sorted |

#### Database: SQL vs NoSQL vs NewSQL
| Aspect | PostgreSQL | Cassandra | CockroachDB | DynamoDB |
|--------|------------|-----------|-------------|----------|
| Consistency | Strong | Eventual/Tunable | Strong | Eventual/Strong |
| Write Throughput | 10K-50K/s | 100K+/s | 20K-50K/s | Unlimited (auto-scale) |
| Read Latency | 1-5ms | 1-5ms | 5-15ms | 1-10ms |
| Scaling | Vertical + read replicas | Linear horizontal | Horizontal | Auto |
| Operational Cost | Medium | High | Medium | Low (managed) |
| When to use | <100M URLs, need ACID | Massive scale, write-heavy | Global, strong consistency | AWS-native, managed |

### Failure Scenarios & Mitigation
| Failure Mode | Impact | Detection | Mitigation |
|--------------|--------|-----------|------------|
| Database down | No new URLs, no redirects | Health checks, connection timeouts | Multi-region replicas, failover |
| Cache miss storm | DB overload on cache expiry | Spike in DB queries | Staggered TTLs, probabilistic refresh |
| Hot key (viral URL) | Single cache node overload | Key access metrics | Local caching, key replication |
| Key collision | Write failures | Collision counter metrics | Retry with salt, longer codes |
| DNS failure | Service unreachable | External monitoring | Multiple DNS providers, low TTL |
| Malicious URLs | Phishing, malware distribution | User reports, scanning | URL validation, blacklist, preview pages |

### Monitoring & Observability
**Key Metrics:**
- **Redirects per second**: Overall throughput and load
- **P99 redirect latency**: User experience (target <50ms)
- **Cache hit ratio**: Efficiency (target >95%)
- **URL creation rate**: Growth and capacity planning
- **Error rate by type**: 404s, 500s, rate limit hits
- **Short code length distribution**: Capacity utilization

**Alerting:**
- Cache hit ratio drops below 90%
- P99 latency exceeds 100ms
- Error rate exceeds 0.1%
- Database replication lag exceeds 1s
- Disk usage exceeds 80%

**Dashboards:**
- Real-time traffic map (geographic distribution)
- Top URLs by click count (detect viral content)
- URL creation funnel (success rate, error breakdown)

### Security Considerations
**Malicious URL Detection:**
- Integration with Google Safe Browsing API
- Real-time URL scanning before redirect
- User reporting mechanism with review queue
- Pattern detection for phishing domains

**Rate Limiting & Abuse Prevention:**
- Per-IP creation limits (10/min anonymous, 100/min authenticated)
- CAPTCHA for anonymous users after threshold
- Account suspension for repeated violations
- Honeypot URLs to detect automated abuse

**Analytics Privacy:**
- Hash or truncate IP addresses after geolocation
- Configurable analytics opt-out per URL
- Data retention policies (delete after 90 days)
- GDPR-compliant data export/deletion

**Infrastructure Security:**
- HTTPS-only redirects
- Signed URLs for private content
- API key rotation and scoping
- Audit logging for all administrative actions

### Interview Deep-Dive Questions

4. **How to migrate short codes when changing encoding scheme?**
   - Maintain dual-read: check old format, then new format
   - Background migration job with progress tracking
   - Use redirect chains temporarily (old → new → destination)
   - Version prefix in URLs (e.g., `v2/abc123`)

5. **How to handle 10x traffic growth overnight (viral event)?**
   - Auto-scaling groups for API servers
   - Cache warming for predicted viral URLs
   - CDN edge caching for redirect responses
   - Circuit breakers to protect database
   - Graceful degradation: serve cached data, queue writes

6. **How to handle link rot (destination URL becomes invalid)?**
   - Periodic health checks on popular URLs
   - Store multiple snapshots via Wayback Machine integration
   - Notify URL creators of broken links
   - Option to redirect to archived version

7. **How to implement geographic-specific redirects?**
   - Store geo-rules in URL metadata: `{US: url1, EU: url2, default: url3}`
   - GeoIP lookup at redirect time
   - CDN-level geo-routing for edge performance
   - Allow creators to configure per URL

8. **How would you optimize cost for billions of URLs?**
   - Tiered storage: hot (SSD) for recent, warm (HDD) for older
   - Compress long URLs in storage
   - Archive rarely-accessed URLs to cold storage
   - Delete expired URLs in background jobs
   - Use spot instances for analytics processing

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

### Extended Tradeoffs

#### Algorithm Deep Dive: Token Bucket vs Leaky Bucket vs Sliding Window
| Aspect | Token Bucket | Leaky Bucket | Sliding Window Log | Sliding Window Counter |
|--------|--------------|--------------|--------------------|-----------------------|
| Burst Handling | Allows bursts up to bucket size | Smooths all traffic | No bursts | Limited bursts |
| Memory | O(1) | O(1) | O(n) per user | O(1) |
| Precision | Approximate | Approximate | Exact | ~99% accurate |
| Implementation | Simple | Simple | Complex | Medium |
| Use Case | APIs with occasional spikes | Steady rate needed | Audit logging | General purpose |
| Boundary Issues | None | None | None | Minor at window edges |

#### Storage: Redis vs Memcached vs In-Memory
| Aspect | Redis Cluster | Memcached | In-Memory (Local) |
|--------|--------------|-----------|-------------------|
| Latency | 0.5-2ms | 0.5-1ms | <0.1ms |
| Consistency | Strong per key | Eventual | None (per-node) |
| Persistence | Optional | No | No |
| Data Structures | Rich (sorted sets, hashes) | Key-value only | Custom |
| Scaling | Sharding built-in | Client sharding | N/A |
| Failure Impact | Partial outage | Partial outage | Inconsistent limits |
| When to use | Distributed, precise | Distributed, simple | Single node, ultra-fast |

### Failure Scenarios & Mitigation
| Failure Mode | Impact | Detection | Mitigation |
|--------------|--------|-----------|------------|
| Redis cluster down | Rate limiting disabled | Connection failures, health checks | Fail open/closed policy, local fallback |
| Network partition | Split-brain limiting | Cross-DC health checks | Accept temporary inaccuracy, reconcile |
| Clock skew | Incorrect window calculations | NTP monitoring | Use Redis server time, not client |
| Hot key (single user) | Redis node hotspot | Key distribution metrics | Shard by user ID, local caching |
| Memory exhaustion | Redis OOM | Memory alerts | TTL on all keys, eviction policy |
| Config service down | Can't update limits | Config service health | Cache config locally, use stale |

### Monitoring & Observability
**Key Metrics:**
- **Requests allowed/denied per second**: Rate limiting effectiveness
- **Limit utilization by tier**: How close users are to limits
- **Redis operation latency**: Overhead introduced
- **False positive rate**: Legitimate requests blocked
- **Burst detection**: Identify attack patterns

**Alerting:**
- Denial rate exceeds 10% globally (possible attack or misconfiguration)
- Redis latency exceeds 5ms P99
- Any rate limit rule blocking >50% of requests for a user
- Config sync failures exceed 1 minute

**Dashboards:**
- Real-time rate limit status by endpoint
- Top users by request volume
- Geographic distribution of blocked requests
- Rate limit configuration history

### Security Considerations
**IP Spoofing Prevention:**
- Use X-Forwarded-For with trusted proxy list only
- Client certificate authentication for API keys
- Rate limit by multiple dimensions (IP + API key + user)

**Distributed Attack Protection:**
- Global rate limits across all endpoints
- Adaptive limits based on traffic patterns
- Integration with WAF for L7 filtering
- Automatic blocklisting of repeated offenders

**API Key Security:**
- Hash API keys in storage
- Separate rate limits per key scope
- Key rotation without downtime
- Audit logging of key usage

### Interview Deep-Dive Questions

4. **How to rate limit by multiple keys simultaneously (IP + user + endpoint)?**
   - Check all limits in parallel: `MULTI/EXEC` in Redis
   - Use composite keys: `{ip}:{user}:{endpoint}`
   - Hierarchical limits: global → tenant → user → endpoint
   - Return most restrictive limit in response headers

5. **How to handle graceful degradation when Redis is unavailable?**
   - Fail open with monitoring: allow requests, alert ops
   - Local in-memory rate limiting as fallback (per-node limits)
   - Circuit breaker pattern: after N failures, stop checking
   - Async write-behind: queue rate limit checks, process later

6. **How to implement adaptive rate limiting?**
   - Monitor backend latency and error rates
   - Dynamically reduce limits when backend stressed
   - Use control theory (PID controllers) for smooth adjustments
   - Per-endpoint limits based on resource cost

7. **How to avoid rate limiting legitimate traffic during load spikes?**
   - Token bucket allows controlled bursts
   - Implement "credit" system for good actors
   - Dynamic limits based on time of day
   - Whitelist known good clients

8. **How would you implement rate limiting for a real-time bidding system (10ms budget)?**
   - In-memory rate limiting only (no network calls)
   - Pre-compute and cache user limits locally
   - Async sync to central store
   - Accept slight over-limit during sync gaps
   - Sample-based limiting for extreme throughput

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

### Extended Tradeoffs

#### Cache Technologies: Redis vs Memcached vs Hazelcast vs Local Cache
| Aspect | Redis | Memcached | Hazelcast | Local (Caffeine) |
|--------|-------|-----------|-----------|------------------|
| Latency | 0.5-2ms | 0.3-1ms | 1-5ms | <0.1ms |
| Data Structures | Rich (lists, sets, sorted sets) | Key-value only | Rich + compute | Custom |
| Persistence | Optional RDB/AOF | No | Yes | No |
| Clustering | Built-in | Client-side | Built-in + auto-discovery | N/A |
| Memory Efficiency | Moderate | High | Moderate | High |
| Max Item Size | 512MB | 1MB default | Configurable | Unlimited |
| Use Case | Feature-rich caching | Simple, ultra-fast | Distributed compute | Single-node, ultra-fast |

#### Caching Strategies Deep Dive
| Strategy | Consistency | Write Perf | Read Perf | Complexity | Best For |
|----------|-------------|------------|-----------|------------|----------|
| Cache-aside | Eventual | Fast | Cache miss = slow | Low | Read-heavy, tolerant |
| Write-through | Strong | Slow (sync) | Fast | Medium | Consistency critical |
| Write-behind | Eventual | Fast (async) | Fast | High | Write-heavy, batch |
| Read-through | Eventual | N/A | Fast | Medium | Simplified client |
| Refresh-ahead | Eventual | N/A | Always fast | High | Predictable access |

### Failure Scenarios & Mitigation
| Failure Mode | Impact | Detection | Mitigation |
|--------------|--------|-----------|------------|
| Cache stampede | DB overload when cache expires | Spike in DB queries | Locking, probabilistic refresh, staggered TTL |
| Thundering herd | Mass cache invalidation | Mass cache misses | Batch invalidation, jitter on expiry |
| Split brain | Inconsistent data across regions | Cross-region health checks | Single source of truth, conflict resolution |
| Hot key | Single node overload | Key access metrics | Key replication, local caching, sharding |
| Cold start | High latency on new deploy | Cache hit ratio drop | Cache warming, gradual traffic shift |
| Memory pressure | Eviction of needed data | Memory utilization alerts | Right-size cache, LRU tuning |

### Monitoring & Observability
**Key Metrics:**
- **Hit ratio**: Overall cache effectiveness (target >90%)
- **Miss latency**: Time to populate cache (DB query time)
- **Eviction rate**: Memory pressure indicator
- **Memory utilization**: Capacity planning
- **Hot key detection**: Keys with >1% of total traffic
- **TTL distribution**: Expiration clustering risk

**Alerting:**
- Hit ratio drops below 80%
- Eviction rate exceeds 1000/second
- Memory utilization exceeds 85%
- Single key exceeds 10% of traffic
- Cross-region replication lag exceeds 100ms

**Debugging Tools:**
- Key access frequency histograms
- Slow query log for cache misses
- Memory fragmentation analysis
- Cluster slot distribution

### Security Considerations
**Cache Poisoning Prevention:**
- Validate data before caching
- Signed cache entries for sensitive data
- Input sanitization for cache keys
- TTL limits to bound poison impact

**Data Leakage Protection:**
- Encrypt sensitive data in cache
- Per-tenant cache isolation (key prefixes)
- Audit logging for sensitive key access
- Automatic PII redaction in logs

**Access Control:**
- Authentication (Redis AUTH, TLS)
- Network segmentation (VPC, security groups)
- Role-based access to admin commands
- Disable dangerous commands (KEYS, FLUSHALL)

### Interview Deep-Dive Questions

4. **How to handle cache warming on new deployments?**
   - Pre-populate from DB during deploy (offline warming)
   - Shadow traffic to new nodes before cutover
   - Gradual traffic shift with increasing cache population
   - Predictive warming based on access patterns

5. **How to implement multi-region caching?**
   - Regional caches with cross-region async replication
   - Write to local region, replicate to others
   - Conflict resolution: last-write-wins or vector clocks
   - Read from local, fallback to remote on miss

6. **How to handle cache versioning for schema changes?**
   - Include version in cache key: `user:v2:{id}`
   - Dual-read during migration (v1, then v2)
   - Background migration job for popular keys
   - TTL-based natural expiration of old versions

7. **How to implement distributed locking with cache?**
   - Redis SETNX with expiration for simple locks
   - Redlock algorithm for distributed consensus
   - Fencing tokens to prevent stale lock holders
   - Consider alternatives: ZooKeeper, etcd for critical paths

8. **How would you design a cache for a social network's user timeline?**
   - Cache recent N posts per user (list structure)
   - Write-through on new post creation
   - Fan-out updates to follower caches (async)
   - Separate hot/cold tiers: recent in memory, older on SSD
   - Personalization layer on top of cached base timeline

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

### Extended Tradeoffs

#### KV Store Technologies: DynamoDB vs Cassandra vs Redis vs etcd
| Aspect | DynamoDB | Cassandra | Redis | etcd |
|--------|----------|-----------|-------|------|
| Consistency | Eventual/Strong | Tunable | Strong (single) | Strong (Raft) |
| Latency | 1-10ms | 1-5ms | <1ms | 1-10ms |
| Throughput | Unlimited (auto) | 100K+/s per node | 100K+/s | 10K+/s |
| Max Value Size | 400KB | 2GB (not recommended) | 512MB | 1.5MB |
| Transactions | Yes (limited) | Lightweight | Yes (Lua/MULTI) | Yes (STM) |
| Secondary Indexes | GSI (eventual) | Yes (local/global) | No | No |
| Operations | Managed | Self-managed | Managed/Self | Self-managed |
| Cost | Pay-per-request | Infrastructure | Infrastructure | Infrastructure |
| Use Case | Serverless, AWS | Massive write scale | Caching, real-time | Config, coordination |

#### LSM-Tree vs B-Tree Deep Dive
| Aspect | LSM-Tree | B-Tree |
|--------|----------|--------|
| Write Pattern | Append-only, sequential | Random I/O |
| Write Throughput | High (batch writes) | Medium |
| Read Latency | Higher (multiple levels) | Lower (single tree) |
| Space Amplification | Higher (multiple copies) | Lower |
| Write Amplification | Higher (compaction) | Lower |
| Range Queries | Efficient after compaction | Very efficient |
| Best For | Write-heavy, time-series | Read-heavy, OLTP |
| Examples | Cassandra, RocksDB, LevelDB | MySQL InnoDB, PostgreSQL |

### Failure Scenarios & Mitigation
| Failure Mode | Impact | Detection | Mitigation |
|--------------|--------|-----------|------------|
| Node failure | Reduced capacity, rebalancing | Heartbeat timeout | Hinted handoff, re-replication |
| Network partition | Split-brain, inconsistency | Cross-DC pings | Quorum requirements, partition tolerance |
| Data corruption | Silent data loss | Checksums, Merkle trees | Repair from replicas, anti-entropy |
| Hot partition | Single node overload | Partition metrics | Key design, scatter-gather pattern |
| Compaction lag | Read amplification | Compaction metrics | Tuning, separate compaction resources |
| Tombstone buildup | Query slowdown | Tombstone metrics | TTL, regular cleanup, DELETE patterns |

### Monitoring & Observability
**Key Metrics:**
- **Read/Write latency P50, P99, P999**: Performance SLOs
- **Throughput**: Requests per second by operation type
- **Replication lag**: Consistency risk indicator
- **Compaction pending**: LSM-tree health
- **Disk utilization**: Capacity planning
- **Partition distribution**: Load balance health

**Alerting:**
- P99 latency exceeds 100ms
- Replication lag exceeds 10 seconds
- Disk utilization exceeds 80%
- Node heartbeat missing for 30 seconds
- Write failures exceed 0.1%

**Operational Tools:**
- Nodetool (Cassandra) for cluster operations
- Consistent hashing visualization
- Hot partition detection
- Tombstone analysis

### Security Considerations
**Encryption:**
- Encryption at rest (disk-level or application-level)
- TLS for data in transit
- Key management integration (KMS, Vault)
- Field-level encryption for sensitive data

**Access Control:**
- Role-based access control (RBAC)
- Row-level security where supported
- API key scoping per keyspace/table
- Audit logging for all operations

**Data Protection:**
- Backup and point-in-time recovery
- Cross-region replication for disaster recovery
- Data retention policies
- Secure deletion (crypto-shredding)

### Interview Deep-Dive Questions

4. **How to handle schema evolution in a schemaless KV store?**
   - Version field in value: `{version: 2, data: {...}}`
   - Migration on read (lazy): transform old to new format
   - Background migration for frequently accessed keys
   - Dual-write during transition period
   - Avoid breaking changes: additive only

5. **How to implement cross-region replication with conflict resolution?**
   - Async replication to minimize latency impact
   - Vector clocks to detect concurrent writes
   - Conflict resolution strategies: last-write-wins, merge, custom
   - Conflict-free replicated data types (CRDTs) for specific use cases
   - Tombstones with grace periods for deletes

6. **How to design a KV store for time-series data?**
   - Partition by time bucket + entity ID
   - Optimize for append-only writes
   - TTL for automatic data expiration
   - Downsampling for historical data
   - Separate hot (recent) and cold (historical) tiers

7. **What backup strategies work for a petabyte-scale KV store?**
   - Incremental snapshots using LSM-tree properties
   - Cross-region replication as live backup
   - Point-in-time recovery using write-ahead logs
   - Backup to object storage (S3, GCS)
   - Validate backups with periodic restore tests

8. **How to implement transactions across multiple keys?**
   - Two-phase commit for distributed transactions
   - Paxos/Raft for consensus across partitions
   - Optimistic concurrency control with version checks
   - Saga pattern for eventual consistency
   - Keep transactions within single partition when possible

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

### Extended Tradeoffs

#### Message Queue Technologies: Kafka vs RabbitMQ vs SQS vs Pulsar
| Aspect | Kafka | RabbitMQ | SQS | Pulsar |
|--------|-------|----------|-----|--------|
| Throughput | 1M+ msg/s | 50K msg/s | Auto-scale | 1M+ msg/s |
| Latency | 5-10ms | 1-5ms | 20-100ms | 5-10ms |
| Ordering | Per-partition | Per-queue | FIFO queues only | Per-partition |
| Retention | Log-based (days/weeks) | Until consumed | 14 days max | Tiered (infinite) |
| Replay | Yes | No (without plugins) | No | Yes |
| Exactly-Once | Yes (with transactions) | No | FIFO only | Yes |
| Protocol | Custom binary | AMQP, STOMP, MQTT | HTTP | Custom binary |
| Operations | Complex | Medium | Managed | Complex |
| Use Case | Event streaming, logs | Task queues, RPC | Simple queues, serverless | Unified messaging |

#### Delivery Semantics Deep Dive
| Guarantee | Implementation | Performance | Complexity | Use Case |
|-----------|---------------|-------------|------------|----------|
| At-most-once | Fire and forget, no acks | Fastest | Low | Metrics, logs (loss OK) |
| At-least-once | Ack after process, retry on failure | Fast | Medium | Most applications |
| Exactly-once | Idempotent producer + transactions | Slowest | High | Financial, inventory |

**Exactly-Once Components:**
- **Producer idempotency**: Sequence numbers per producer session
- **Transactions**: Atomic writes across partitions
- **Consumer idempotency**: Dedupe using message IDs
- **Outbox pattern**: DB + queue in single transaction

### Failure Scenarios & Mitigation
| Failure Mode | Impact | Detection | Mitigation |
|--------------|--------|-----------|------------|
| Broker failure | Partition unavailable | Leader election, health checks | Replication, automatic failover |
| Consumer lag | Processing delays | Lag metrics | Auto-scaling consumers, alerts |
| Message loss | Data loss on crash | Gap detection | Replication factor ≥3, sync replication |
| Poison message | Consumer stuck | Same message redelivered | Dead letter queue, max retries |
| Producer backpressure | Publish failures | Publish latency spike | Batching, async sends, circuit breaker |
| Disk full | Broker crash | Disk utilization monitoring | Retention policies, alerts |

### Monitoring & Observability
**Key Metrics:**
- **Consumer lag**: Messages behind real-time (critical for latency)
- **Throughput**: Messages per second (produce/consume)
- **Broker disk utilization**: Capacity planning
- **Under-replicated partitions**: Durability risk
- **Consumer group health**: Active members, rebalances
- **Message age**: Time since produce (staleness)

**Alerting:**
- Consumer lag exceeds 10,000 messages
- Under-replicated partitions > 0
- Producer error rate exceeds 0.1%
- Consumer rebalance frequency exceeds 1/hour
- Disk utilization exceeds 80%

**Debugging Tools:**
- Consumer offset tracking
- Message tracing (correlation IDs)
- Partition distribution visualization
- Consumer group state inspection

### Security Considerations
**Message Encryption:**
- TLS for data in transit (required)
- Application-level encryption for sensitive payloads
- Key rotation without downtime
- Envelope encryption for audit trail

**Access Control:**
- Topic-level ACLs (produce, consume, admin)
- Consumer group authorization
- Service-to-service authentication (mTLS, OAuth)
- Audit logging for all operations

**Compliance:**
- Message retention policies for data residency
- PII handling: encryption or exclusion
- Audit trail for all message operations
- GDPR: right to deletion considerations

### Interview Deep-Dive Questions

4. **How to handle poison messages (messages that always fail)?**
   - Configure max retry attempts per message
   - Move to dead letter queue (DLQ) after max retries
   - Alert on DLQ depth, manual intervention
   - Root cause analysis: schema issues, missing dependencies
   - Replay from DLQ after fix

5. **How to implement message ordering across partitions?**
   - Single partition for strict ordering (limits throughput)
   - Sequence numbers in message, reassemble at consumer
   - Causal ordering using vector clocks
   - Accept partial ordering per entity (partition by entity ID)

6. **How to scale consumers without losing messages?**
   - Consumer groups with partition assignment
   - Rebalance protocol for member changes
   - Commit offsets before rebalance completes
   - Cooperative rebalancing to minimize disruption
   - Pre-scale before expected load spikes

7. **How to implement request-reply pattern over async queue?**
   - Correlation ID in request and response
   - Reply-to queue per request or per client
   - Timeout handling for missing responses
   - Consider RPC frameworks (gRPC) for synchronous needs

8. **How would you migrate from RabbitMQ to Kafka?**
   - Dual-write: produce to both systems
   - Shadow consumer: consume from both, compare
   - Gradual traffic shift with feature flags
   - Schema compatibility between systems
   - Rollback plan with bidirectional sync

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

### Extended Tradeoffs

#### Push Providers: APNs vs FCM vs Web Push
| Aspect | APNs (Apple) | FCM (Google) | Web Push |
|--------|--------------|--------------|----------|
| Platform | iOS, macOS | Android, iOS, Web | Browsers |
| Payload Size | 4KB | 4KB | 4KB |
| Reliability | High | High | Medium (browser dependent) |
| Latency | <1s | <1s | 1-10s |
| Token Management | Complex (refresh needed) | Simpler | Subscription-based |
| Delivery Receipt | Yes | Yes | No |
| Silent Push | Yes | Yes | Limited |
| Priority Levels | High/Normal | High/Normal | Urgent/Normal |

#### Email Providers: SendGrid vs SES vs Mailgun
| Aspect | SendGrid | Amazon SES | Mailgun |
|--------|----------|------------|---------|
| Deliverability | High | High | High |
| Price (per 10K) | $0.25+ | $0.10 | $0.80 |
| Templates | Yes | Yes | Yes |
| Analytics | Detailed | Basic | Detailed |
| Dedicated IPs | Yes | Yes | Yes |
| API Quality | Excellent | Good | Excellent |
| Webhook Support | Full | Limited | Full |
| Best For | General purpose | AWS ecosystem, cost | Transactional focus |

### Failure Scenarios & Mitigation
| Failure Mode | Impact | Detection | Mitigation |
|--------------|--------|-----------|------------|
| Provider outage (APNs/FCM) | Push delivery stops | Provider health endpoints | Multi-provider fallback, retry queue |
| Token expiration | Delivery failures | Invalid token responses | Regular token refresh, cleanup job |
| Rate limiting (provider) | Throttled delivery | 429 responses | Batching, backoff, multiple credentials |
| Email bounce | Failed delivery, reputation damage | Bounce webhooks | List hygiene, bounce handling |
| Spam filtering | User never sees notification | Engagement metrics | Domain reputation, content quality |
| Template error | Malformed notifications | Pre-send validation | Template testing, staging environment |

### Monitoring & Observability
**Key Metrics:**
- **Delivery rate by channel**: Success percentage (target >98%)
- **Delivery latency by priority**: Time to deliver
- **Open/Click rates**: Engagement effectiveness
- **Bounce/Complaint rates**: Reputation health
- **Provider API latency**: Integration health
- **Queue depth by priority**: Processing capacity

**Alerting:**
- Delivery rate drops below 95%
- Bounce rate exceeds 2%
- Complaint rate exceeds 0.1%
- Provider API errors exceed 1%
- Queue depth exceeds 100K for >5 minutes

**Dashboards:**
- Real-time delivery funnel (sent → delivered → opened → clicked)
- Channel breakdown by notification type
- Provider health comparison
- User preference distribution

### Security Considerations
**Notification Spoofing Prevention:**
- Signed messages from trusted senders
- Rate limiting per sender service
- Content validation (no external links in sensitive flows)
- Audit trail of all notifications

**PII Handling:**
- Minimize PII in notification content
- Encrypt sensitive data in payload
- Short-lived URLs for personalized content
- Log redaction for notification content

**User Privacy:**
- Granular opt-in/opt-out controls
- Clear unsubscribe mechanisms
- Preference sync across devices
- Data retention limits on notification history

### Interview Deep-Dive Questions

4. **How to handle notification aggregation (100 likes → "100 people liked your post")?**
   - Buffer notifications in short window (30s-5min)
   - Aggregate by type and target object
   - Summary notification with breakdown on expand
   - User-configurable aggregation thresholds
   - Real-time count updates for active users

5. **How to implement A/B testing for notifications?**
   - Hash user ID for consistent bucketing
   - Test: subject lines, timing, channels
   - Measure: open rate, conversion, unsubscribe
   - Statistical significance calculator
   - Automatic winner selection after confidence threshold

6. **How to scale to billions of notifications (New Year's midnight)?**
   - Pre-compute and queue notifications hours ahead
   - Time-zone aware scheduling
   - Staggered delivery windows (±30 min)
   - Dedicated capacity for peak events
   - Circuit breakers to protect downstream systems

7. **How to prevent notification fatigue?**
   - ML-based send time optimization
   - Frequency caps per user per channel
   - Importance scoring to filter low-value notifications
   - Digest mode for non-urgent notifications
   - User engagement signals to adjust frequency

8. **How would you design cross-platform notification sync?**
   - Central notification inbox service
   - Real-time sync via WebSocket for active devices
   - Read/dismiss status propagation across devices
   - Last-N notifications cached per device
   - Push notification as wake-up for inbox sync

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

### Extended Tradeoffs

#### Real-Time Protocols: WebSocket vs SSE vs Long Polling vs gRPC Streaming
| Aspect | WebSocket | SSE (Server-Sent Events) | Long Polling | gRPC Streaming |
|--------|-----------|-------------------------|--------------|----------------|
| Bidirectional | Yes | No (server→client only) | No (request-response) | Yes |
| Latency | <50ms | <100ms | 100-500ms | <50ms |
| Connection Overhead | Low (persistent) | Low (persistent) | High (reconnect) | Low (HTTP/2) |
| Browser Support | All modern | All modern | All | Limited (needs proxy) |
| Scalability | 100K+ per server | 100K+ per server | 10K per server | 100K+ per server |
| Load Balancer | L4 or L7 (sticky) | L7 | L7 | L7 (HTTP/2) |
| Firewall Friendly | Some issues | HTTP-based | HTTP-based | HTTP/2 based |
| Use Case | Full-duplex chat | Live feeds, notifications | Legacy support | Internal services |

#### Message Routing: Redis Pub/Sub vs Kafka vs Custom
| Aspect | Redis Pub/Sub | Kafka | Custom (Consistent Hashing) |
|--------|---------------|-------|---------------------------|
| Latency | <1ms | 5-10ms | <1ms |
| Durability | No (fire-and-forget) | Yes (persisted) | Depends on impl |
| Message History | No | Yes | Custom |
| Scalability | Moderate | High | High |
| Ordering | Per-channel | Per-partition | Custom |
| Operations | Simple | Complex | Custom |
| Use Case | Real-time, no history | Event sourcing, replay | Full control needed |

### Failure Scenarios & Mitigation
| Failure Mode | Impact | Detection | Mitigation |
|--------------|--------|-----------|------------|
| WebSocket disconnect | User appears offline, message delays | Heartbeat timeout | Auto-reconnect, exponential backoff |
| Chat server crash | Connected users lose connection | Health checks | Sticky session migration, connection redistribution |
| Message ordering issues | Confusing conversation flow | Out-of-order detection | Lamport timestamps, client-side sorting |
| Redis Pub/Sub overload | Message delays or loss | Pub/Sub metrics | Shard by chat ID, use Kafka for scale |
| Database write failures | Messages not persisted | Write error rate | Retry queue, write-ahead log, user notification |
| Media upload failure | Broken attachments | Upload tracking | Resumable uploads, CDN fallback |

### Monitoring & Observability
**Key Metrics:**
- **Connected users**: WebSocket connection count
- **Message delivery latency**: Time from send to receive
- **Presence update latency**: Online status propagation time
- **Messages per second**: Throughput
- **Reconnection rate**: Connection stability
- **Undelivered message rate**: Reliability

**Alerting:**
- Message delivery latency P99 exceeds 500ms
- Connection drop rate exceeds 1%
- Undelivered messages exceed 0.1%
- Chat server memory exceeds 80%
- Database replication lag exceeds 1s

**Debugging Tools:**
- Message tracing (correlation IDs)
- Connection state inspection
- Chat room activity heatmaps
- User session timeline

### Security Considerations
**End-to-End Encryption (E2EE):**
- Signal Protocol (Double Ratchet) for key management
- Pre-keys for offline message encryption
- Identity key verification (safety numbers)
- Server never sees plaintext content
- Group encryption with sender keys

**Message Tampering Prevention:**
- Digital signatures on messages
- Hash chain for conversation integrity
- Tamper-evident logging
- Client-side verification

**Access Control:**
- Authentication for WebSocket connections
- Authorization per chat room
- Admin controls for group management
- Rate limiting per user

### Interview Deep-Dive Questions

4. **How to implement typing indicators efficiently?**
   - Throttle typing events (send every 2-3s while typing)
   - Short TTL (3s) - auto-clear if no update
   - Broadcast only to active participants in view
   - Don't persist typing state
   - Debounce on client side

5. **How to scale group chats with thousands of members?**
   - Fan-out on read for very large groups
   - Tiered architecture: active participants vs lurkers
   - Pagination for member list
   - Selective presence updates (nearby members only)
   - Message sampling for activity indicators

6. **How to implement message search across chat history?**
   - Elasticsearch index for message content
   - Index on write (async for non-blocking)
   - Per-user access control in search
   - E2EE consideration: client-side search or keyword encryption
   - Recent messages in hot storage, older in cold

7. **How to handle message edits and deletes?**
   - Version history for edits (store diffs)
   - Soft delete with tombstone
   - Propagate edit/delete to all recipients
   - Time limit for editing (e.g., 15 minutes)
   - "Edited" indicator in UI

8. **How would you implement read receipts at scale?**
   - Batch read updates (aggregate per chat room)
   - Store per-user read pointer, not per-message
   - Propagate read status asynchronously
   - Limit history for read receipt display
   - Opt-out option for privacy

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

### Extended Tradeoffs

#### Fan-Out Models with Numerical Analysis
| Aspect | Fan-Out on Write | Fan-Out on Read | Hybrid |
|--------|------------------|-----------------|--------|
| Write Cost | O(followers) | O(1) | O(non-celebrity followers) |
| Read Cost | O(1) | O(following) | O(celebrity following) |
| Storage | O(users × avg posts) | O(posts) | O(users × avg non-celeb posts) |
| Write Latency | High for celebrities | Low | Medium |
| Read Latency | Low | High for heavy users | Low-Medium |
| Example Numbers | 10M followers = 10M writes | 1K following = 1K reads | Celebrity threshold: 10K |
| Consistency | Eventually consistent | Always fresh | Hybrid consistency |
| Best For | Majority of users | Very few followers | Production systems |

#### Ranking Algorithm Components
| Signal | Weight | Description | Update Frequency |
|--------|--------|-------------|------------------|
| Recency | 30% | Time decay function | Real-time |
| Engagement | 25% | Likes, comments, shares | Near real-time |
| Affinity | 20% | Relationship strength with author | Daily batch |
| Content Type | 10% | Photo, video, text | Static per post |
| Author Quality | 10% | Past engagement, credibility | Daily batch |
| Diversity | 5% | Avoid same author clustering | Real-time |

### Failure Scenarios & Mitigation
| Failure Mode | Impact | Detection | Mitigation |
|--------------|--------|-----------|------------|
| Celebrity posting (fan-out storm) | Queue backup, delays | Queue depth spike | Async fan-out, priority queues, hybrid model |
| Viral content | Cache stampede, DB overload | Traffic spike on content ID | Cache replication, rate limiting reads |
| Feed inconsistency | Missing or duplicate posts | User reports, consistency checks | Idempotent writes, deduplication |
| Ranking service down | No personalization | Health checks | Fallback to chronological feed |
| Graph service slow | Slow following list lookup | Latency metrics | Cache follower lists aggressively |
| Storage corruption | Lost feed data | Checksum failures | Rebuild from posts + graph |

### Monitoring & Observability
**Key Metrics:**
- **Feed generation latency**: P50, P99 (target <200ms)
- **Fan-out lag**: Time from post to all feeds updated
- **Engagement rate**: Clicks, likes, time spent per feed load
- **Scroll depth**: How far users scroll (feed quality indicator)
- **Cache hit ratio**: Feed cache effectiveness
- **Diversity score**: Content variety in feeds

**Alerting:**
- Feed latency P99 exceeds 500ms
- Fan-out queue depth exceeds 10M
- Engagement rate drops 10% from baseline
- Cache hit ratio drops below 80%
- Feed generation errors exceed 0.1%

**A/B Testing:**
- Ranking algorithm variants
- Feed refresh behavior
- Content type weighting
- Personalization intensity

### Security Considerations
**Content Moderation:**
- Pre-publish content scanning (text, images)
- User reporting mechanism with priority queue
- ML-based spam and abuse detection
- Account reputation scoring
- Appeals process for false positives

**Spam Detection:**
- Velocity checks (posts per hour)
- Duplicate content detection
- Bot behavior patterns (engagement timing)
- Network analysis (coordinated behavior)
- Progressive enforcement (warning → throttle → ban)

**Privacy:**
- Private account handling in fan-out
- Block/mute respect in feed generation
- Content visibility rules per relationship
- Data retention for feed history

### Interview Deep-Dive Questions

4. **How to handle unfollows efficiently?**
   - Lazy removal: filter during read, not immediate cleanup
   - Background job to clean up feed caches
   - For hybrid model: just stop future fan-out
   - Consider: keep posts from before unfollow?

5. **How to implement infinite scroll with consistent experience?**
   - Cursor-based pagination (not offset)
   - Snapshot feed state at session start
   - Handle new posts: "N new posts" banner
   - Deduplicate across page loads
   - Cache multiple pages ahead

6. **How to debug why a specific post isn't appearing in a feed?**
   - Tracing: follow post through fan-out pipeline
   - Check: user relationship (following, blocked?)
   - Check: post visibility (deleted, private?)
   - Check: ranking (scored too low?)
   - Check: pagination (past cursor?)

7. **How to handle time-travel debugging ("what did feed look like yesterday")?**
   - Event sourcing: log all feed changes
   - Point-in-time snapshots (hourly/daily)
   - Reconstruction from posts + graph history
   - Replay ranking with historical signals

8. **How would you implement "Close Friends" or audience selection?**
   - Separate fan-out path for restricted content
   - Filter at read time based on audience list
   - Cache per audience segment
   - Combine with regular feed at read time
   - Privacy: hide audience selection from viewers

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

### Extended Tradeoffs

#### Data Structures: Trie vs Ternary Search Tree vs Elasticsearch
| Aspect | Trie | Ternary Search Tree | Elasticsearch |
|--------|------|---------------------|---------------|
| Memory | High (many pointers) | Medium | Low (inverted index) |
| Prefix Lookup | O(k) k=prefix length | O(k + log n) | O(log n) |
| Fuzzy Matching | Complex to add | Complex to add | Built-in |
| Updates | Moderate | Moderate | Fast (append-only) |
| Scaling | In-memory, single node | In-memory, single node | Distributed |
| Operations | Simple | Simple | Complex |
| Use Case | Small-medium, fast | Medium, memory-constrained | Large scale, fuzzy |

#### Fuzzy Matching Algorithms
| Algorithm | Accuracy | Speed | Handles | Use Case |
|-----------|----------|-------|---------|----------|
| Levenshtein (Edit Distance) | High | Slow | Typos | Spell correction |
| Soundex | Medium | Fast | Phonetic | Name search |
| Metaphone | Higher | Fast | Phonetic | Better phonetic |
| N-grams | High | Medium | Partial matches | Substring search |
| BK-Tree | High | Medium | Edit distance | Dictionary lookup |

### Failure Scenarios & Mitigation
| Failure Mode | Impact | Detection | Mitigation |
|--------------|--------|-----------|------------|
| Index corruption | Wrong/missing suggestions | Index validation | Rebuild from source, checksums |
| Stale suggestions | Outdated trending | Freshness checks | Real-time index updates, TTL |
| Cold cache | High latency | Cache miss rate spike | Cache warming, predictive loading |
| Hot prefix ("the") | Single shard overload | Prefix query distribution | Prefix sharding, in-memory hot cache |
| Suggestion bombing | Spam in suggestions | Anomaly detection | Rate limiting, human review queue |
| Personalization service down | No personalization | Health checks | Fallback to global suggestions |

### Monitoring & Observability
**Key Metrics:**
- **P99 latency**: Target <50ms including network
- **Suggestion click-through rate (CTR)**: Quality indicator
- **Query abandonment rate**: User giving up on search
- **Index freshness**: Age of newest suggestions
- **Cache hit ratio**: Efficiency
- **Prefix coverage**: % of queries with suggestions

**Alerting:**
- P99 latency exceeds 100ms
- CTR drops 10% from baseline
- Index update lag exceeds 1 hour
- Cache hit ratio drops below 90%
- Suggestion coverage drops below 95%

**Quality Metrics:**
- Mean Reciprocal Rank (MRR): Position of clicked suggestion
- Suggestion diversity score
- Offensive content detection rate

### Security Considerations
**Profanity and Content Filtering:**
- Blocklist for offensive terms
- ML-based toxicity detection
- Real-time filtering in suggestion pipeline
- Admin tools for manual review and blocking

**Suggestion Bombing Prevention:**
- Rate limiting on suggestion indexing
- Anomaly detection for sudden popularity
- Human review for trending suggestions
- Source verification (organic vs manipulated)

**Privacy:**
- Don't expose other users' searches in suggestions
- Anonymize before aggregating
- User opt-out for search history personalization
- Data retention limits on search logs

### Interview Deep-Dive Questions

4. **How to handle multiple languages and scripts?**
   - Language detection on query
   - Per-language index shards
   - Unicode normalization (NFC/NFKC)
   - Language-specific tokenization (CJK, Arabic)
   - Cross-language suggestions for common terms

5. **How to implement personalization while respecting privacy?**
   - On-device processing for sensitive history
   - Server-side: hash user ID, don't log queries
   - Blend personal + global (e.g., 30% personal weight)
   - Easy opt-out mechanism
   - Differential privacy for aggregate analytics

6. **How to handle real-time trending suggestions?**
   - Sliding window aggregation (last 15 min, 1 hour)
   - Exponential decay weighting
   - Burst detection for viral queries
   - Geographic segmentation for local trends
   - Cooldown for yesterday's trends

7. **How would you implement query understanding (beyond prefix)?**
   - Intent classification (navigational, informational)
   - Entity recognition in queries
   - Query expansion (synonyms, related terms)
   - Session context (previous queries)
   - User profile context

8. **How to optimize for mobile autocomplete?**
   - Client-side prefix caching (common prefixes)
   - Adaptive result count (fewer on small screens)
   - Touch-friendly tap targets
   - Predictive pre-fetching based on typing patterns
   - Bandwidth-aware response compression

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

### Extended Tradeoffs

#### Crawling Strategies: BFS vs DFS vs Priority-Based
| Aspect | BFS (Breadth-First) | DFS (Depth-First) | Priority-Based |
|--------|---------------------|-------------------|----------------|
| Coverage | Broad, shallow | Deep, narrow | Important pages first |
| Discovery | Fast for popular | Good for complete sites | Best balance |
| Memory | High (large frontier) | Low (stack-based) | Medium |
| Freshness | Good for homepages | Poor for popular | Best |
| Use Case | General crawling | Site-specific | Production crawlers |
| Politeness | Easier to manage | Per-site polite | Complex but optimal |

#### Bloom Filter Sizing Analysis
| URLs (billions) | Bits per URL | False Positive Rate | Memory |
|-----------------|--------------|---------------------|--------|
| 1 | 10 | 1% | 1.2 GB |
| 1 | 15 | 0.1% | 1.9 GB |
| 10 | 10 | 1% | 12 GB |
| 10 | 15 | 0.1% | 19 GB |
| 100 | 10 | 1% | 120 GB |

**Formula:** m = -n * ln(p) / (ln(2))² where n=items, p=false positive rate

### Failure Scenarios & Mitigation
| Failure Mode | Impact | Detection | Mitigation |
|--------------|--------|-----------|------------|
| Crawler traps | Infinite crawling | URL pattern detection | Depth limits, pattern blocklist |
| DNS failures | Can't resolve hosts | DNS query failures | DNS caching, multiple resolvers |
| Rate limiting (by site) | Blocked IP, 429s | Response code monitoring | Respect robots.txt, exponential backoff |
| Spider trap detection | Soft blocking | Unusual link patterns | CAPTCHA detection, JS analysis |
| Content explosion | Disk full | Disk utilization alerts | Content size limits, deduplication |
| Frontier corruption | Lost URLs | Checkpoint validation | Regular checkpoints, idempotent replay |

### Monitoring & Observability
**Key Metrics:**
- **Pages per second**: Crawl throughput
- **Unique domains discovered**: Coverage breadth
- **DNS cache hit ratio**: Efficiency
- **Error rate by type**: 4xx, 5xx, timeout breakdown
- **Content type distribution**: HTML vs JS vs other
- **Duplicate detection rate**: Bloom filter effectiveness

**Alerting:**
- Crawl rate drops 50% from baseline
- Error rate exceeds 10%
- Disk utilization exceeds 80%
- DNS failure rate exceeds 5%
- Single domain exceeds 10% of frontier

**Operational Tools:**
- Frontier visualization by domain
- Crawl delay heatmap
- Content type and size distributions
- robots.txt compliance reports

### Security Considerations
**Robots.txt Compliance:**
- Always fetch and respect robots.txt
- Cache robots.txt per domain (24-hour TTL)
- Respect Crawl-delay directive
- Honor noindex meta tags

**Crawl Ethics:**
- Identify crawler in User-Agent
- Provide contact information
- Respect rate limits gracefully
- Don't crawl password-protected content

**Infrastructure Security:**
- Sandbox untrusted content execution
- Prevent SSRF via URL validation
- Content scanning for malware
- Isolated network for fetchers

### Interview Deep-Dive Questions

4. **How to handle JavaScript-rendered content?**
   - Headless browser (Puppeteer, Playwright) for JS rendering
   - Separate queue for JS-heavy sites (more resources)
   - Smart rendering: detect if JS needed per page
   - Pre-rendering services for common frameworks
   - Accept that some content won't be indexed

5. **How to implement incremental crawling (only fetch changed pages)?**
   - Store Last-Modified and ETag headers
   - Conditional requests (If-Modified-Since)
   - Content hashing to detect changes
   - Change frequency estimation per URL
   - Adaptive recrawl intervals based on change rate

6. **How to optimize crawl budget for a large site?**
   - Prioritize by PageRank or site structure
   - Focus on recently updated sections
   - Sample deep pages, crawl important ones
   - Use sitemaps for guided discovery
   - A/B test crawl strategies

7. **How to detect and handle soft 404s?**
   - Content similarity to known 404 pages
   - "Page not found" text detection
   - HTTP 200 with empty content
   - Unusual response time patterns
   - Hash-based duplicate detection

8. **How would you design a recrawl scheduler?**
   - Track content change frequency per URL
   - Exponential backoff for stable pages
   - Immediate recrawl for known-fresh (RSS, sitemaps)
   - Time-decay priority scoring
   - Separate queues by priority tier

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

### Extended Tradeoffs

#### Streaming Protocols: HLS vs DASH vs CMAF
| Aspect | HLS | DASH | CMAF |
|--------|-----|------|------|
| Developer | Apple | MPEG | Apple + Microsoft |
| Segment Format | .ts (MPEG-TS) | .m4s (fMP4) | fMP4 |
| Manifest | .m3u8 | .mpd (XML) | Both supported |
| DRM Support | FairPlay | Widevine, PlayReady | All |
| Latency | 15-30s | 10-20s | 2-5s (LL-HLS/DASH) |
| Browser Support | Safari native, others via JS | All via JS | All via JS |
| Device Support | iOS, Apple TV native | Android, Smart TVs | Universal |
| Use Case | Apple ecosystem | Cross-platform | Unified delivery |

#### Encoding Profiles (Bitrate Ladder)
| Resolution | Bitrate (H.264) | Bitrate (H.265) | Target Device |
|------------|-----------------|-----------------|---------------|
| 2160p (4K) | 15-20 Mbps | 8-12 Mbps | Smart TV, High-end |
| 1080p | 4-6 Mbps | 2-4 Mbps | Desktop, Tablet |
| 720p | 2-3 Mbps | 1-2 Mbps | Mobile WiFi |
| 480p | 1-1.5 Mbps | 0.5-1 Mbps | Mobile data |
| 360p | 0.5-0.7 Mbps | 0.3-0.5 Mbps | Poor connection |
| 240p | 0.3 Mbps | 0.15 Mbps | Extreme conditions |

### Failure Scenarios & Mitigation
| Failure Mode | Impact | Detection | Mitigation |
|--------------|--------|-----------|------------|
| Transcoding failure | Video not available | Job failure alerts | Retry queue, manual intervention |
| CDN outage | Buffering, playback failure | Edge health checks | Multi-CDN with failover |
| Origin overload | CDN cache misses slow | Origin latency spike | Origin shielding, cache warming |
| DRM license server down | Premium content blocked | License fetch failures | License caching, graceful degradation |
| Segment corruption | Playback artifacts | Checksum validation | Regenerate segment, CDN purge |
| Popular video spike | CDN capacity exceeded | Bandwidth alerts | Pre-warming, traffic shaping |

### Monitoring & Observability
**Key Metrics:**
- **Video Start Time (VST)**: Time to first frame (target <2s)
- **Rebuffering ratio**: % of playback time buffering (target <0.5%)
- **Bitrate switches**: Quality stability indicator
- **CDN hit ratio**: Efficiency (target >95%)
- **Encoding queue depth**: Processing capacity
- **Error rate by type**: Playback, license, network

**Alerting:**
- VST P95 exceeds 4 seconds
- Rebuffering ratio exceeds 1%
- CDN hit ratio drops below 90%
- Transcoding queue exceeds 1 hour
- License fetch failures exceed 1%

**Quality of Experience (QoE):**
- VMAF/PSNR scores for quality assessment
- Playback completion rate
- User engagement (watch time, drop-off points)
- Device/browser breakdown

### Security Considerations
**DRM Implementation:**
- Multi-DRM: Widevine (Chrome, Android), FairPlay (Safari, iOS), PlayReady (Edge)
- Hardware-backed key storage where available
- License server rate limiting and validation
- Token-based license requests

**Content Protection:**
- Signed URLs with expiration
- Geographic restrictions enforcement
- Device limits per account
- Screen recording detection

**Piracy Prevention:**
- Forensic watermarking in video stream
- Content ID fingerprinting on upload
- Takedown automation via APIs
- Quality degradation for suspicious accounts

### Interview Deep-Dive Questions

4. **How to handle live streaming vs VOD differently?**
   - **Live**: Edge transcoding, low-latency protocols (LL-HLS), 2s segments
   - **VOD**: Pre-encoded, 6s segments, full ABR ladder
   - **Live DVR**: Sliding window of past N minutes
   - **Live-to-VOD**: Archive and re-encode after broadcast
   - Different CDN caching strategies (no-cache vs aggressive)

5. **How to implement offline download support?**
   - License with offline capability (DRM expiry extension)
   - Download encrypted segments to device
   - Secure local storage with key management
   - Periodic license renewal on reconnect
   - Quality selection for storage constraints

6. **How to design a multi-CDN strategy?**
   - Real-time performance monitoring per CDN
   - Traffic steering at DNS or application layer
   - Automatic failover on CDN health issues
   - Cost optimization (mix of CDN tiers)
   - Geographic affinity with performance override

7. **How to optimize for first frame time?**
   - Pre-fetch first segment on hover/tap
   - Start at lower quality, ramp up
   - Optimize manifest/license server latency
   - Use HTTP/2 or HTTP/3 for multiplexing
   - Progressive segment loading

8. **How would you handle a live event with 10M concurrent viewers?**
   - Pre-warm CDN edges with promotional content
   - Regional ingest with edge transcoding
   - Staggered start (countdown buffer)
   - Auto-scaling for origin and encoding
   - Graceful degradation plan (lower max quality)

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

### Extended Tradeoffs

#### File Systems: HDFS vs GFS vs Ceph vs MinIO
| Aspect | HDFS | GFS (Google) | Ceph | MinIO |
|--------|------|--------------|------|-------|
| Architecture | Single NameNode | Single Master | Distributed (no SPOF) | Distributed |
| Chunk Size | 128 MB | 64 MB | 4 MB | Variable |
| Consistency | Strong (metadata) | Strong (metadata) | Strong (RADOS) | Strong |
| Append Support | Yes | Optimized | Yes | No (object) |
| Small Files | Poor | Poor | Better | Good |
| POSIX Compatible | No (HDFS API) | No | Yes (CephFS) | S3 API |
| Ecosystem | Hadoop, Spark | Google internal | OpenStack, K8s | K8s, S3 apps |
| Use Case | Big data analytics | Large-scale batch | General purpose | S3-compatible |

#### Consistency Models Analysis
| Model | Definition | Latency | Use Case |
|-------|------------|---------|----------|
| Strong | Read sees latest write | Highest | Financial, metadata |
| Linearizable | Strong + real-time ordering | Highest | Coordination |
| Sequential | All see same order | High | Append logs |
| Causal | Preserves cause-effect | Medium | Collaboration |
| Eventual | Eventually consistent | Lowest | Replicated data |

### Failure Scenarios & Mitigation
| Failure Mode | Impact | Detection | Mitigation |
|--------------|--------|-----------|------------|
| Master failure | No metadata operations | Heartbeat timeout | Hot standby, Paxos/Raft election |
| Chunk server failure | Reduced redundancy | Heartbeat timeout | Re-replication to healthy nodes |
| Data corruption | Silent data loss | Checksum verification | Background scrubbing, repair from replica |
| Network partition | Split-brain potential | Cross-rack health | Quorum requirements, fencing |
| Rack failure | Multiple chunk losses | Power monitoring | Rack-aware placement, ≥3 replicas |
| Disk failure | Chunk data loss | SMART monitoring | RAID, proactive replacement |

### Monitoring & Observability
**Key Metrics:**
- **Under-replicated chunks**: Durability risk indicator
- **Read/Write throughput**: Performance
- **Master operation latency**: Metadata health
- **Disk utilization distribution**: Balance
- **Chunk server health**: Availability
- **Replication queue depth**: Recovery progress

**Alerting:**
- Under-replicated chunks > 0 for > 30 minutes
- Master failover occurred
- Disk utilization exceeds 85% on any node
- Chunk server heartbeat missing > 60 seconds
- Replication throughput drops 50%

**Operational Tools:**
- Cluster topology visualization
- Rebalancing progress tracking
- Hot chunk identification
- Capacity planning projections

### Security Considerations
**Encryption:**
- Encryption at rest (disk-level or per-chunk)
- Encryption in transit (TLS between all components)
- Key management integration (KMS, Vault)
- Per-tenant encryption keys

**Access Control:**
- Path-based ACLs (read, write, execute)
- User/group permissions (POSIX-style)
- Service account authentication
- Audit logging for all operations

**Data Protection:**
- Backup to separate storage system
- Point-in-time snapshots
- Cross-datacenter replication
- Compliance: data retention, deletion policies

### Interview Deep-Dive Questions

4. **How to handle small files efficiently (small file problem)?**
   - HAR (Hadoop Archive): combine into large files
   - Federation: multiple namespaces
   - Memory-only metadata for small files
   - Separate small file store (object storage)
   - Merge on write, split on read

5. **How to implement cross-datacenter replication?**
   - Async replication to minimize latency impact
   - WAL shipping for metadata
   - Chunk-level replication for data
   - Conflict resolution policy (timestamp-based)
   - RPO/RTO tradeoffs

6. **How to handle quota management at scale?**
   - Per-directory quota tracking
   - Soft and hard limits
   - Background enforcement (not real-time)
   - Aggregation hierarchy caching
   - Graceful enforcement with user notification

7. **How to optimize for append-heavy workloads (e.g., logs)?**
   - Last chunk kept open for appends
   - Lease management for concurrent appends
   - Record append (atomic append operation)
   - Separate hot and cold data placement
   - Compaction for old log files

8. **How would you design file search/indexing on distributed storage?**
   - Metadata indexing in separate database
   - Full-text search with Elasticsearch
   - Incremental indexing on file changes
   - Distributed crawling with coordinator
   - Content-based vs metadata-only search

---

## 13. Task Scheduler

**Problem:** Design a distributed task scheduling system like Apache Airflow, Temporal, or Dagster.

### Requirements

**Functional:**
- Define workflows as directed acyclic graphs (DAGs) of tasks
- Schedule workflows based on time (cron) or events
- Execute tasks with dependency resolution
- Support retries, timeouts, and failure handling
- Provide visibility into workflow execution status
- Support parameterized workflow runs

**Non-Functional:**
- Handle 100K+ task executions per day
- Low scheduling latency (< 1s from trigger to execution start)
- High availability (99.9%)
- Exactly-once task execution semantics
- Horizontal scalability for workers
- Support long-running tasks (hours to days)

### Core Concepts

**DAG (Directed Acyclic Graph):** A workflow definition where nodes are tasks and edges are dependencies. Tasks execute only when all upstream dependencies complete successfully.

**Task:** A single unit of work (e.g., run a script, call an API, execute a query). Each task has inputs, outputs, retry policy, and timeout configuration.

**DAG Run:** A single execution instance of a DAG with specific parameters and execution timestamp.

**Task Instance:** A single execution of a task within a DAG run, tracking state (pending, running, success, failed, skipped).

### High-Level Architecture

**Architecture Flow:**
1. Users define DAGs via UI, API, or code (Python DSL)
2. DAG Parser validates and stores DAG definitions in Metadata DB
3. Scheduler scans for DAGs ready to run (time-based or event-triggered)
4. Scheduler creates DAG Runs and Task Instances in Metadata DB
5. Scheduler enqueues ready tasks (no pending dependencies) to Task Queue
6. Worker Pool consumes tasks from queue and executes them
7. Workers report status back to Metadata DB
8. Scheduler monitors task completion and enqueues newly ready downstream tasks
9. Web UI provides visibility into runs, logs, and metrics

### Scheduler Design

**Scheduler Responsibilities:**
1. **DAG Parsing:** Periodically scan DAG repository, parse definitions, detect changes
2. **Run Creation:** At scheduled time, create new DAG Run with initial task instances
3. **Dependency Resolution:** Identify tasks with all dependencies satisfied
4. **Task Queuing:** Push ready tasks to distributed queue with priority
5. **State Management:** Update task states, handle retries, timeouts
6. **Deadlock Detection:** Identify circular dependencies or stuck workflows

**Scheduler Loop:**
1. Check for DAGs due to run (cron evaluation)
2. Create DAG Runs for triggered DAGs
3. For each active DAG Run, find tasks where all upstreams succeeded
4. Enqueue ready tasks with execution context
5. Check for timed-out tasks, mark failed, trigger retries if configured
6. Clean up completed DAG Runs (archive or delete)

### Worker Design

**Worker Responsibilities:**
1. Poll task queue for available work
2. Deserialize task definition and parameters
3. Execute task in isolated environment
4. Stream logs to centralized log storage
5. Report heartbeats during execution
6. Report final status (success/failure) with outputs

**Task Execution Model:**
- **Process-based:** Each task runs in separate process (isolation, resource limits)
- **Container-based:** Each task runs in Docker/K8s pod (stronger isolation, reproducibility)
- **Serverless:** Tasks execute as Lambda/Cloud Functions (auto-scaling, pay-per-use)

### Task State Machine

**Task Instance States:**
- **none:** Task instance not yet created
- **scheduled:** Task is scheduled, waiting for dependencies
- **queued:** Dependencies met, waiting for worker
- **running:** Worker is executing the task
- **success:** Task completed successfully
- **failed:** Task failed after all retries exhausted
- **up_for_retry:** Task failed, retry scheduled
- **skipped:** Task skipped due to branching logic
- **upstream_failed:** Upstream dependency failed

**State Transitions:**
- scheduled → queued (when all upstream tasks succeed)
- queued → running (when worker picks up task)
- running → success | failed | up_for_retry
- up_for_retry → queued (after retry delay)
- failed → downstream tasks marked upstream_failed

### Distributed Locking and Leader Election

**Why Needed:**
- Only one scheduler should create DAG Runs for a given schedule
- Prevent duplicate task execution
- Coordinate failover when scheduler dies

**Implementation Options:**
- Database-based locks with expiration (simple, but DB becomes bottleneck)
- ZooKeeper/etcd for distributed coordination (reliable, additional infrastructure)
- Redis with Redlock algorithm (fast, good for most cases)

**Scheduler HA Pattern:**
Multiple scheduler instances run simultaneously. Only the leader creates DAG Runs. Followers wait in standby. On leader failure, new leader is elected via distributed lock.

### Dependency Resolution Strategies

**Static Dependencies:** Defined at DAG parse time, fixed task graph.

**Dynamic Dependencies:** Tasks can spawn subtasks at runtime (e.g., process each partition).

**Cross-DAG Dependencies:** Task in DAG A depends on task in DAG B (sensor pattern).

**Dependency Operators:**
- **all_success:** Run only if all upstreams succeeded (default)
- **all_failed:** Run only if all upstreams failed (error handling)
- **one_success:** Run if any upstream succeeded
- **none_failed:** Run if no upstream failed (includes skipped)

### Retry and Failure Handling

**Retry Configuration:**
- Max retry attempts (e.g., 3)
- Retry delay (fixed, exponential backoff)
- Retry exceptions (retry only on specific errors)

**Failure Handling Strategies:**
- **Stop DAG:** Fail entire DAG Run on first task failure
- **Continue:** Mark downstream as upstream_failed, continue parallel branches
- **Callback:** Execute failure callback for alerting

**Timeout Handling:**
- Task-level timeout: Kill task if exceeds duration
- DAG-level timeout: Fail entire DAG if total duration exceeded
- Sensor timeout: Stop waiting for external condition

### Task Queue Design

**Queue Requirements:**
- Distributed across workers
- Priority support (urgent tasks first)
- At-least-once delivery with visibility timeout
- Dead letter queue for poison tasks

**Queue Options:**
| Option | Pros | Cons |
|--------|------|------|
| Redis + Celery | Simple, fast, widely used | No persistence by default |
| RabbitMQ | Reliable, feature-rich | Complex operations |
| Amazon SQS | Managed, scalable | Higher latency |
| Kafka | High throughput, replay | Overkill for task queues |

### Metadata Storage

**Core Tables:**
- **dag:** DAG definitions (dag_id, schedule, default_args, is_paused)
- **dag_run:** DAG executions (run_id, dag_id, execution_date, state, start_date, end_date)
- **task_instance:** Task executions (task_id, dag_id, run_id, state, try_number, start_date, duration)
- **task_log:** Execution logs (task_instance_id, log_content, timestamp)
- **variable:** Key-value configuration store
- **connection:** External system credentials

**Database Choice:**
- PostgreSQL: Reliable, ACID, good for most deployments
- MySQL: Alternative, slightly less feature-rich
- Avoid NoSQL: Need transactions for state consistency

### Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Scheduling | Pull-based (workers pull) | Simpler, natural load balancing |
| Execution | Container per task | Isolation, reproducibility |
| State Store | PostgreSQL | ACID transactions for consistency |
| Queue | Redis + Celery | Simple, proven, fast |
| HA | Active-passive scheduler | Avoid duplicate scheduling |

### Interview Discussion Points

1. **How to handle exactly-once execution?**
   - Distributed lock per task instance
   - Idempotent task design (preferred)
   - Two-phase commit for critical tasks

2. **How to scale workers?**
   - Horizontal scaling based on queue depth
   - Worker pools per task type (CPU vs memory intensive)
   - Kubernetes autoscaling

3. **How to handle long-running tasks?**
   - Heartbeat mechanism for liveness
   - Checkpointing for resumability
   - Separate timeout from worker health

### Extended Tradeoffs

#### Scheduler Architectures: Centralized vs Distributed
| Aspect | Centralized (Airflow) | Distributed (Temporal) |
|--------|----------------------|------------------------|
| Consistency | Easier (single scheduler) | Requires consensus |
| Scalability | Limited by single scheduler | Horizontally scalable |
| Failure Handling | Failover to standby | Automatic partition healing |
| Complexity | Simpler | More complex |
| Latency | Higher (single bottleneck) | Lower (parallel scheduling) |
| Use Case | Medium scale (100K tasks/day) | Large scale (10M+ tasks/day) |

#### Execution Models: Process vs Container vs Serverless
| Aspect | Process | Container (K8s) | Serverless (Lambda) |
|--------|---------|-----------------|---------------------|
| Isolation | Process-level | Container-level | Function-level |
| Startup Time | Fast (<100ms) | Medium (1-10s) | Variable (cold start) |
| Resource Control | Limited | Full (CPU, memory, GPU) | Limited |
| Cost | Fixed (VMs) | Flexible | Pay-per-execution |
| Max Duration | Unlimited | Unlimited | Limited (15min Lambda) |
| Reproducibility | Low | High | Medium |
| Best For | Simple scripts | Complex pipelines | Event-driven, sporadic |

#### DAG Definition: Code vs Config vs UI
| Aspect | Code (Python DSL) | Config (YAML/JSON) | UI (Drag-and-Drop) |
|--------|-------------------|--------------------|--------------------|
| Flexibility | High (full language) | Medium | Low |
| Version Control | Easy (git) | Easy | Difficult |
| Learning Curve | Requires coding | Simpler | Easiest |
| Dynamic DAGs | Yes | Limited | No |
| Testing | Unit testable | Schema validation | Manual |
| Best For | Engineers | DevOps, data teams | Business users |

### Failure Scenarios & Mitigation
| Failure Mode | Impact | Detection | Mitigation |
|--------------|--------|-----------|------------|
| Scheduler crash | No new tasks scheduled | Health check, leader election | HA with standby, auto-failover |
| Worker crash | Running tasks lost | Heartbeat timeout | Task retry, state recovery |
| Queue unavailable | Tasks can't be dispatched | Queue health checks | Queue replication, fallback |
| Database failure | State lost, no coordination | Connection errors | DB replication, backups |
| Task stuck (no heartbeat) | Resources blocked | Heartbeat timeout | Kill task, retry or fail |
| Infinite retry loop | Resource exhaustion | Retry count monitoring | Max retries, dead letter queue |

### Monitoring & Observability

**Key Metrics:**
- **Task queue depth:** Backlog indicator
- **Task latency (queue to start):** Scheduling efficiency
- **Task duration by type:** Performance baseline
- **Success/failure rate:** Reliability
- **Scheduler lag:** Time between scheduled and actual start
- **Worker utilization:** Capacity planning

**Alerting:**
- Task queue depth exceeds threshold for >5 minutes
- Task failure rate exceeds 5%
- DAG Run duration exceeds 2× historical average
- Scheduler heartbeat missing
- No workers available for >1 minute

**Dashboards:**
- DAG Run history with status timeline
- Task instance Gantt chart (execution visualization)
- Worker pool status and utilization
- SLA tracking (on-time completion rate)

### Security Considerations

**Authentication & Authorization:**
- Role-based access control (view, trigger, edit DAGs)
- Per-DAG permissions for multi-tenant deployments
- API authentication (JWT, OAuth)
- Audit logging for all actions

**Secrets Management:**
- Encrypted connection credentials
- Integration with Vault, AWS Secrets Manager
- Secrets injection at runtime (not stored in DAG code)
- Credential rotation without DAG changes

**Execution Security:**
- Sandboxed task execution (containers)
- Network policies for task isolation
- Resource quotas per user/team
- Code scanning for DAG definitions

### Interview Deep-Dive Questions

4. **How to implement task prioritization across DAGs?**
   - Priority queues with multiple levels (critical, high, normal, low)
   - Weighted fair scheduling across teams/projects
   - Preemption for critical tasks (kill lower priority)
   - SLA-based priority boost as deadline approaches
   - Resource pools with guaranteed capacity per priority

5. **How to handle backfills (re-running historical dates)?**
   - Backfill creates DAG Runs for past execution dates
   - Limit concurrent backfill runs to avoid overload
   - Respect dependencies (backfill in chronological order)
   - Mark as backfill vs scheduled run for analytics
   - Consider: clear downstream before backfill?

6. **How to implement cross-DAG dependencies?**
   - Sensor tasks that poll for upstream completion
   - Event-driven triggers (upstream emits event)
   - External database for cross-DAG state
   - Dataset/data-aware scheduling (Airflow 2.4+)
   - Tradeoff: coupling vs flexibility

7. **How to support dynamic task generation at runtime?**
   - Task groups with variable membership
   - Mapped tasks (foreach over collection)
   - Subdag/taskflow for complex patterns
   - Challenge: dependency tracking for dynamic tasks
   - State management for partial completion

8. **How would you design a multi-tenant task scheduler?**
   - Namespace isolation (DAGs, connections, variables)
   - Resource quotas per tenant (max concurrent tasks)
   - Dedicated worker pools per tenant
   - Noisy neighbor prevention (fair scheduling)
   - Cost attribution and chargeback
   - Shared vs dedicated infrastructure tradeoffs

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
| Video Streaming | Quality vs bandwidth |
| Distributed File Storage | Consistency vs throughput |
| Task Scheduler | Exactly-once execution vs latency |

---

## Interview Tips

1. **Start with requirements** - Clarify functional and non-functional requirements
2. **Do back-of-envelope calculations** - Estimate scale, storage, bandwidth
3. **Draw high-level architecture first** - Show the big picture
4. **Deep dive on key components** - Show depth in critical areas
5. **Discuss tradeoffs explicitly** - There's no perfect solution
6. **Consider failure scenarios** - How does the system degrade?
7. **Think about operations** - Monitoring, deployment, debugging
