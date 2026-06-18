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

**Write load:** 100M URLs per day divided by 86,400 seconds equals approximately 1,200 URLs per second.

**Read load:** 1B redirects per day divided by 86,400 seconds equals approximately 12,000 redirects per second.

**Storage for 5 years:** 100M URLs per day times 365 days times 5 years equals 182.5 billion URLs. With an average URL size of 500 bytes, total storage is approximately 90 TB.

### High-Level Architecture

**Architecture Flow:**
1. Client sends request to Load Balancer
2. Load Balancer routes to API Server
3. API Server connects to three main backend services:
   - Cache (Redis) for fast URL lookups
   - Database (Cassandra) for persistent storage
   - Analytics (Kafka) for tracking clicks and metrics

### URL Encoding Approaches

**Option 1: Base62 Encoding**

The Base62 encoding algorithm converts a numeric ID to a short alphanumeric string using characters 0-9, A-Z, and a-z (62 total characters). The algorithm works by repeatedly dividing the number by 62 and mapping each remainder to the corresponding character. The result is reversed since digits are generated least-significant first. With 7 characters, this provides 62^7 (approximately 3.5 trillion) unique URLs.

**Option 2: MD5/SHA256 Hash with Truncation**

This approach takes the MD5 hash of the long URL, converts the first 6 bytes to base62, and takes the first 7 characters. This is simpler to implement but requires collision handling.

**Option 3: Pre-generated Keys (Key Generation Service)**

A separate service pre-generates unique keys and stores them in an unused keys queue. When a new short URL is needed, a key is retrieved from the queue and moved to the used keys set. This eliminates runtime collision handling but requires separate key management infrastructure.

### Tradeoffs: Encoding Approaches

| Approach | Pros | Cons |
|----------|------|------|
| **Counter + Base62** | Simple, no collisions | Single point of failure, predictable |
| **Hash Truncation** | Distributed, simple | Collision handling needed |
| **Pre-generated Keys** | Fast, no runtime collision | Complex key management, storage overhead |
| **Snowflake ID** | Distributed, time-sorted | Longer URLs (10+ chars) |

### Database Schema

**URL Table Schema:**
- short_code (primary key, VARCHAR(10)): The 7-character shortened URL identifier
- long_url (TEXT, NOT NULL): The original destination URL
- user_id (UUID): Optional owner of the URL
- created_at (TIMESTAMPTZ): Timestamp of creation, defaults to current time
- expires_at (TIMESTAMPTZ): Optional expiration time
- click_count (BIGINT): Number of redirects, defaults to 0

**Indexes:**
- Index on user_id for user-based queries
- Partial index on expires_at for non-null expiration values

### Database Choice Tradeoffs

| Database | Pros | Cons |
|----------|------|------|
| **PostgreSQL** | ACID, familiar, good tooling | Scaling challenges at extreme scale |
| **Cassandra** | Linear scalability, high write throughput | Eventually consistent, no joins |
| **DynamoDB** | Managed, auto-scaling | Vendor lock-in, cost at scale |

### Caching Strategy

**Redirect Flow with Caching:**
1. Check Redis cache for the short code
2. If found (cache hit), return the cached long URL
3. If not found (cache miss), query the database
4. If found in database, cache the result with a 1-hour TTL and return the long URL
5. If not found anywhere, raise a NotFoundError

### Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| URL Length | 7 characters | 3.5T URLs, balance of brevity and capacity |
| Encoding | Base62 | URL-safe, case-sensitive for more combinations |
| Database | Cassandra | Write-heavy, easy horizontal scaling |
| Cache | Redis | Sub-ms reads, high hit rate expected |
| ID Generation | Snowflake | Distributed, time-ordered, no coordination |

### Handling Collisions

**Collision Handling Algorithm:**
1. Generate a short code from the long URL
2. Attempt to insert into the database
3. If a duplicate key error occurs, retry with a modified input (append attempt number as salt)
4. After 3 failed attempts, raise a CollisionError

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
   - Store geo-rules in URL metadata: {US: url1, EU: url2, default: url3}
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

**Token Bucket Algorithm:**
The token bucket maintains a bucket with a maximum capacity and a refill rate (tokens per second). When a request arrives, tokens are first refilled based on elapsed time since the last refill. If enough tokens are available, they are consumed and the request is allowed. Otherwise, the request is denied. This algorithm allows bursts up to the bucket capacity while maintaining an average rate.

**Pros:** Allows bursts, smooth rate limiting, memory efficient
**Cons:** Harder to reason about exact limits

#### 2. Sliding Window Log

**Sliding Window Log Algorithm:**
This approach stores the timestamp of every request within the current window. When a new request arrives, old entries outside the window are removed, and if the count is under the limit, the request is allowed and its timestamp is added. This provides exact rate limiting with no boundary issues.

**Pros:** Precise, no boundary issues
**Cons:** Memory intensive for high-volume APIs

#### 3. Sliding Window Counter

**Sliding Window Counter Algorithm:**
This hybrid approach divides time into fixed windows and stores counts for each window. When checking a request, it calculates a weighted count using both the current and previous window counts, where the weight is based on how far into the current window we are. This provides a good approximation with O(1) memory per key.

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

**Distributed Rate Limiter with Redis:**
The distributed implementation uses a Lua script for atomic check-and-increment operations on Redis. The script increments a counter for the key, sets an expiration on first increment, and returns whether the request is within the limit. This ensures atomicity even across multiple application servers.

### High-Level Architecture

**Architecture Flow:**
1. Client sends request to Gateway
2. Gateway performs rate check before routing
3. Gateway connects to Redis Cluster (multiple nodes) for distributed state
4. If allowed, request proceeds to API Server
5. Config Service provides rate limit rules to the Gateway

### Handling Failures

**Resilient Rate Limiter Strategies:**
When Redis is unavailable, the rate limiter can use one of three fallback strategies:
- **Fail open:** Allow all requests (better UX, risk of abuse during outage)
- **Fail closed:** Deny all requests (protects backend, blocks legitimate traffic)
- **Local fallback:** Use in-memory rate limiting per node (inconsistent limits but functional)

### Tradeoffs: Fail Open vs Fail Closed

| Strategy | Pros | Cons |
|----------|------|------|
| **Fail Open** | Service continues, better UX | Risk of abuse during outage |
| **Fail Closed** | Protects backend | Blocks legitimate traffic |
| **Local Fallback** | Best of both | Inconsistent limits across nodes |

### Response Headers

**Rate Limit Response Format:**
When a rate limit is exceeded, the response includes HTTP status 429 (Too Many Requests) with headers indicating: the rate limit (X-RateLimit-Limit: 100), remaining requests (X-RateLimit-Remaining: 0), reset timestamp (X-RateLimit-Reset: 1640000000), and suggested retry delay (Retry-After: 60).

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
   - Check all limits in parallel: MULTI/EXEC in Redis
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

**Architecture Flow:**
1. Clients connect through a Client Library that implements consistent hashing
2. Client Library routes requests to the appropriate Cache Node based on key hash
3. Three Cache Nodes (each a Primary) handle different key ranges
4. Each Primary has a Replica for fault tolerance
5. Consistent hashing minimizes key redistribution when nodes are added or removed

### Data Partitioning: Consistent Hashing

**Consistent Hashing Algorithm:**
The consistent hash ring places both nodes and keys on a circular hash space. Each physical node is represented by multiple virtual nodes (e.g., 150) distributed around the ring to ensure even load distribution. To find the node for a key, hash the key and walk clockwise around the ring until finding the first node. When nodes are added or removed, only keys that hash between the affected node and its predecessor need to move.

### Eviction Policies

**LRU (Least Recently Used) Algorithm:**
Maintains an ordered dictionary where the most recently accessed items are at the end. On get, the item is moved to the end. On put, if capacity is exceeded, the oldest item (front) is evicted. This works well when recent access patterns predict future access.

**LFU (Least Frequently Used) Algorithm:**
Tracks access frequency for each item using a frequency map with ordered dictionaries at each frequency level. On access, items move to the next frequency level. Eviction removes items from the lowest non-empty frequency level. This keeps popular items longer but is slower to adapt to changing patterns.

### Eviction Policy Comparison

| Policy | Use Case | Pros | Cons |
|--------|----------|------|------|
| **LRU** | General purpose | Simple, good for temporal locality | Scan resistance issue |
| **LFU** | Stable access patterns | Keeps popular items | Slow to adapt to changes |
| **FIFO** | Simple workloads | Very simple | Ignores access patterns |
| **Random** | Uniform access | No overhead | Unpredictable |
| **TTL** | Time-sensitive data | Automatic cleanup | May evict hot data |

### Cache Invalidation Strategies

**Write-through:** Write to cache and database simultaneously. Ensures consistency but slows writes.

**Write-behind (Write-back):** Write to cache immediately, queue database write for async processing. Fast writes but risk of data loss and complexity.

**Cache-aside (Lazy loading):** On read, check cache first; on miss, read from database and populate cache. On write, update database and invalidate (delete) cache entry. Simple and flexible but has potential for stale reads.

### Invalidation Strategy Tradeoffs

| Strategy | Consistency | Performance | Complexity |
|----------|-------------|-------------|------------|
| **Write-through** | Strong | Slower writes | Low |
| **Write-behind** | Eventual | Fast writes | High (async) |
| **Cache-aside** | Eventual | Fast reads | Medium |
| **Read-through** | Eventual | Simple client | Medium |

### Replication Strategies

**Synchronous Replication:** Write to primary and wait for all replicas to acknowledge before returning success. Ensures strong consistency but increases write latency.

**Asynchronous Replication:** Write to primary and immediately return success, then queue replication to followers. Lower latency but risks data loss if primary fails before replication completes.

**Read Distribution:** For reads, optionally prefer replicas to distribute load, falling back to primary if needed.

### Hot Key Handling

**Hot Key Detection and Mitigation:**
1. Maintain a local cache (short TTL, e.g., 1 second) for hot keys
2. Track access frequency per key
3. When a key exceeds a threshold, mark it as hot
4. For hot keys, serve from local cache to avoid hammering the distributed cache
5. Periodically sync hot key values from the distributed cache

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

The CAP theorem states that a distributed system can only guarantee two of three properties: Consistency, Availability, and Partition Tolerance. Since network partitions are inevitable, the real choice is between CP (consistent but may be unavailable during partitions) and AP (available but may return stale data during partitions).

**Design Choice:** AP with tunable consistency (like Cassandra/DynamoDB), allowing applications to choose their consistency level per operation.

### High-Level Architecture

**Architecture Flow:**
1. Clients connect to any Coordinator Node
2. Coordinator routes requests to appropriate Storage Nodes based on key hash
3. Storage Nodes are partitioned by key range (e.g., Node 1 handles A-F, Node 2 handles G-M, Node 3 handles N-Z)
4. Each key is replicated to multiple nodes for fault tolerance
5. Coordinator handles quorum-based reads and writes

### Data Partitioning

**Partition Manager Algorithm:**
Uses consistent hashing with a configurable replication factor (default 3). For any key, the manager identifies the primary node using the hash ring, then walks the ring to find the next N-1 distinct physical nodes for replicas. This ensures replicas are on different physical machines, ideally in different racks.

### Quorum-Based Consistency

**Quorum Coordination:**
With N total replicas, W write quorum, and R read quorum:
- **Write:** Send to all N nodes, succeed if W acknowledge
- **Read:** Query all N nodes, return if R respond, pick value with highest version
- **Consistency guarantee:** If W + R > N, reads will see the latest write

For example, with N=3, W=2, R=2: any read must overlap with at least one node that has the latest write.

### Consistency Level Tradeoffs

| Config | W | R | Consistency | Availability | Latency |
|--------|---|---|-------------|--------------|---------|
| Strong | N | 1 | Strong | Low | High write |
| Strong | 1 | N | Strong | Low | High read |
| Quorum | ⌈N/2⌉+1 | ⌈N/2⌉+1 | Strong | Medium | Medium |
| Eventual | 1 | 1 | Eventual | High | Low |

### Write Path (LSM-Tree)

**LSM-Tree Write Algorithm:**
1. **Write to WAL:** Append to write-ahead log for durability
2. **Write to Memtable:** Insert into in-memory sorted structure
3. **Flush when full:** When memtable exceeds threshold, write to immutable SSTable on disk
4. **Clear and continue:** Clear memtable and WAL, start fresh

This provides fast writes (sequential I/O) at the cost of slower reads (may need to check multiple SSTables).

### SSTable with Bloom Filter

**SSTable Read Optimization:**
Each SSTable includes a sparse index and Bloom filter. On read:
1. Check Bloom filter first (quick negative lookup)
2. If Bloom filter says "maybe present," use sparse index for binary search
3. Bloom filters have false positives but no false negatives, avoiding unnecessary disk reads

**Bloom Filter Operation:**
Uses multiple hash functions to set bits in a bit array. To check membership, all corresponding bits must be set. False positives are possible (all bits set by different keys) but false negatives are not.

### Conflict Resolution

**Vector Clock Algorithm:**
Tracks causality using a map of node IDs to logical timestamps. When comparing two vector clocks:
- If all entries in A ≤ corresponding entries in B, A happened BEFORE B
- If all entries in A ≥ corresponding entries in B, A happened AFTER B
- If neither, the events are CONCURRENT (conflict!)

**Conflict Resolution Strategies:**
- **Last-write-wins:** Use timestamp, simpler but may lose data
- **Return all versions:** Let application merge (like shopping carts)
- **CRDTs:** Use conflict-free replicated data types for automatic merge

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

**Architecture Flow:**
1. Multiple Producers send messages through a Load Balancer
2. Load Balancer routes to Brokers based on partition assignment
3. Each Broker hosts multiple Partitions, each being either a Leader or Replica
4. Within a Broker: Partition 0 might be Leader, Partition 1 might be Replica
5. Replication ensures each partition's data exists on multiple Brokers
6. Consumers organized into Consumer Groups subscribe to partitions
7. Each partition is consumed by exactly one consumer per group

### Message Storage

**Partition Storage Model:**
Each partition maintains an append-only log with monotonically increasing offsets. Messages are appended with metadata (offset, timestamp, key size, value size, key, value) and indexed by offset for efficient seeking. Reading fetches messages starting from a given offset up to a byte limit.

### Consumer Groups

**Consumer Group Coordination:**
1. Track which partitions are assigned to which consumers
2. Store committed offsets per partition
3. On rebalance (consumer join/leave), redistribute partitions using round-robin or range assignment
4. Consumers commit offsets after processing to track progress

### Delivery Guarantees

**Producer Acknowledgment Levels:**
- **acks=0 (fire and forget):** No guarantee, highest throughput
- **acks=1 (leader acknowledged):** At-least-once, leader confirms
- **acks=all:** Strongest guarantee, all in-sync replicas confirm

**Partition Selection:**
- With key: hash(key) % num_partitions for consistent routing
- Without key: round-robin for load distribution

**Consumer Processing:**
Poll messages from assigned partitions, process, then commit offsets to record progress.

### Delivery Guarantee Tradeoffs

| Guarantee | Implementation | Tradeoff |
|-----------|---------------|----------|
| **At-most-once** | Commit before processing | May lose messages |
| **At-least-once** | Commit after processing | May duplicate messages |
| **Exactly-once** | Idempotent producers + transactions | Higher latency, complexity |

### Replication

**Replication Process:**
1. Write to leader partition first
2. Replicate to followers (can be sync or async)
3. Track acknowledgments from replicas
4. Fail if not enough replicas acknowledge (based on min.insync.replicas)

**Leader Election:**
On leader failure, elect new leader from in-sync replicas (ISR). If ISR is empty, may perform unclean election from any follower (risks data loss). Notify clients of new leader assignment.

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

**Architecture Flow:**
1. Services (A, B, C) send notification requests to Notification API
2. Notification API validates and publishes to Kafka Queue
3. Kafka routes to channel-specific Workers (Push, SMS, Email)
4. Push Worker connects to APNs/FCM for mobile delivery
5. SMS Worker connects to Twilio/Nexmo for text messages
6. Email Worker connects to SendGrid for email delivery

### Data Model

**Notification Preferences Table:**
- user_id (primary key, UUID)
- push_enabled (boolean, default true)
- email_enabled (boolean, default true)
- sms_enabled (boolean, default false)
- quiet_hours_start/end (TIME)
- frequency_limit (integer, per hour)
- channel_preferences (JSONB for granular settings)

**Device Tokens Table:**
- id (UUID primary key)
- user_id (UUID, foreign key)
- platform (ios, android, web)
- token (device push token)
- app_version, is_active
- Unique constraint on (user_id, token)

**Notifications Log Table:**
- id (UUID primary key)
- user_id, type, channel (push/sms/email)
- title, body, data (payload)
- status (pending/sent/delivered/read)
- sent_at, delivered_at, read_at timestamps
- Index on (user_id, created_at DESC)

### Notification Service

**Send Flow:**
1. Check user preferences (cached in Redis)
2. Apply rate limiting per user
3. Check for duplicate notifications (deduplication)
4. If all checks pass, queue to appropriate channel topic
5. Return status (queued, filtered, rate_limited, deduplicated)

**Should Send Logic:**
- Check if channel is enabled in preferences
- Check quiet hours (allow urgent messages through)
- Check notification type preferences

### Push Notification Worker

**Push Delivery Process:**
1. Fetch user's device tokens from database
2. For each token, send to appropriate provider:
   - iOS: Apple Push Notification service (APNs)
   - Android: Firebase Cloud Messaging (FCM)
3. Handle errors:
   - Invalid token: Mark as inactive, remove from future sends
   - Provider error: Queue for retry with exponential backoff
4. Build platform-specific payloads with title, body, sound, badge, custom data

### Priority and Batching

**Notification Batching Strategy:**
- **Urgent priority:** Send immediately, no batching
- **Normal priority:** Batch for efficiency, flush when batch reaches size limit or time interval expires
- **Email:** Typically batched for cost efficiency
- **SMS:** Usually not batched due to urgency expectations

### Delivery Tracking

**Tracking Metrics:**
- Track sent timestamp when notification is dispatched
- Track delivered timestamp via provider webhooks
- Track failed status with error reason
- Emit Prometheus metrics for sent/failed counts by channel

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

**Architecture Flow:**
1. Clients establish WebSocket connections through L4/Sticky Load Balancer
2. Load Balancer routes to Chat Servers that maintain WebSocket connections
3. Chat Servers communicate through Redis Pub/Sub for cross-server message routing
4. Backend services include:
   - Message DB (Cassandra) for persistent storage
   - Presence Service for online status
   - Media Store (S3) for file attachments

### Connection Management

**Chat Server Connection Handling:**
1. On connect: Store WebSocket in local connections map
2. Register user→server mapping in Redis for cross-server routing
3. Update presence service to mark user online
4. Subscribe to user's Redis channel for incoming messages
5. On disconnect: Clean up all registrations and mark offline

**Message Routing:**
- If recipient connected to same server: Send directly via WebSocket
- If recipient on different server: Publish to Redis channel for their server to deliver

### Message Flow

**Send Message Process:**
1. Generate unique message ID (Snowflake) and timestamp
2. Persist message to database
3. Get all chat participants
4. For each recipient (except sender):
   - If online: Deliver via real-time channel
   - If offline: Queue push notification
5. Send delivery acknowledgment to sender

### Message Storage Schema

**Messages Table (Cassandra, optimized for chat queries):**
- chat_id (partition key): Groups messages by conversation
- message_id (TIMEUUID, clustering key): Ordered by time
- sender_id, content, message_type, status, created_at
- Clustering order by message_id DESC for efficient recent queries

**Chats Table:**
- chat_id (primary key)
- type (direct/group), name, participants (SET)
- created_at, last_message_at

**User Chats Table (denormalized):**
- user_id (partition key)
- last_message_at, chat_id (clustering keys)
- unread_count
- Clustering order by last_message_at DESC for inbox view

### Read Receipts

**Read Receipt Flow:**
1. Update read pointer in Redis hash: chat_id → user_id → last_read_message_id
2. Notify other participants of read status change
3. To check who has read a message: Compare all read pointers with message_id

### Presence Service

**Presence Management:**
- On online: Set presence key with TTL (2× heartbeat interval), notify contacts
- On heartbeat: Refresh TTL
- On offline: Delete presence key, set last_seen timestamp, notify contacts
- To get presence for multiple users: Pipeline Redis queries for efficiency

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

**Pull Feed Algorithm:**
1. Get list of users the viewer follows
2. For each followed user, fetch their recent posts
3. Merge all posts, sort by timestamp (descending)
4. Return top N posts

**Pros:** Simple, no storage overhead, fresh data
**Cons:** Slow for users following many accounts, high read latency

#### 2. Push Model (Fan-out on Write)

**Push Feed Algorithm:**
On post creation:
1. Insert post into post database
2. Get all followers of the author
3. For each follower, prepend post ID to their feed cache (Redis list)
4. Trim feed cache to keep only recent N posts

On feed read:
1. Fetch post IDs from pre-computed feed cache
2. Batch fetch post details

**Pros:** Fast reads, simple feed retrieval
**Cons:** Expensive for celebrities (millions of followers), storage heavy

#### 3. Hybrid Model

**Hybrid Feed Algorithm:**
Define celebrity threshold (e.g., 10,000 followers).

On post creation:
- If author has fewer followers than threshold: Fan-out to all followers (push)
- If author is celebrity: Store in celebrity posts table (no fan-out)

On feed read:
1. Get pre-computed feed (from regular users' push)
2. Get followed celebrities list
3. Pull recent posts from celebrities
4. Merge, sort, and return top N

### Architecture

**Architecture Flow:**
1. Clients connect through API Gateway
2. Three main services:
   - Feed Service (Read Path): Assembles and returns feeds
   - Post Service (Write Path): Creates posts and triggers fan-out
   - Graph Service (Following): Manages follow relationships
3. Fan-out Workers process post distribution asynchronously
4. Storage layer:
   - Feed Cache (Redis): Pre-computed feed lists
   - Post Store (Cassandra): Post content
   - Graph Store (Neo4j): Follow relationships

### Ranking Algorithm

**Feed Ranking Score Calculation:**
The score combines multiple signals with weights:
- **Time decay (30%):** Newer posts score higher, exponential decay by hours
- **Engagement (30%):** Weighted sum of likes (1x), comments (2x), shares (3x)
- **Affinity (30%):** Pre-computed relationship strength with author
- **Content boost (10%):** Posts with media get 1.2x multiplier

Posts are sorted by score descending.

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

**Architecture Flow:**
1. Client sends debounced keystrokes
2. CDN (Edge Cache) serves cached popular prefixes
3. Autocomplete Service handles cache misses
4. Trie Store (Redis) provides prefix lookups
5. Analytics Pipeline tracks searches for trending and weight updates

### Trie Data Structure

**Autocomplete Trie:**
Each node in the trie stores children (character → node mapping), an end-of-word flag, and pre-computed top suggestions for that prefix. On insert, traverse/create path for each character, updating top suggestions at each prefix node. On search, traverse the prefix path and return pre-computed suggestions. This provides O(prefix_length) lookup time.

### Redis Implementation

**Redis Autocomplete with Sorted Sets:**
For indexing: For each prefix of a term (length 1 to full), add the term to a sorted set keyed by that prefix with the term's weight as score. Trim each set to keep only top N suggestions.

For search: Query the sorted set for the prefix, get top N by score descending.

For weight updates: When a term is selected, increment its score across all prefix sorted sets.

### Handling Scale

**Sharded Autocomplete:**
Shard by first character of prefix across multiple Redis instances. Route queries based on first character hash.

**Tiered Autocomplete:**
Use in-memory LRU cache for hot prefixes (short prefixes like 1-3 characters are most frequent). Fall back to Redis for longer/less common prefixes. Cache short prefixes locally since they're queried frequently.

### Real-Time Trend Updates

**Trending Search Tracking:**
Use time-bucketed sorted sets (per-minute buckets) to track search frequency. On each search, increment count in current bucket with TTL. For trending queries, union last N buckets (e.g., 60 minutes) with zunionstore. Return top K from merged results. Buckets naturally expire based on TTL.

### Fuzzy Matching

**Fuzzy Search Algorithm:**
1. Try exact prefix match first
2. If insufficient results, try fuzzy search with edit distance ≤ 1
3. Add phonetic matching (Soundex/Metaphone) for name searches
4. Combine, deduplicate, and return top results
5. Generate edit candidates by insertions, deletions, replacements, transpositions

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

**Architecture Flow:**
1. URL Frontier (Priority Queue + Politeness) manages URLs to crawl
2. Multiple Crawler Workers fetch pages in parallel
3. DNS Resolver (Cached) reduces DNS lookup overhead
4. Content Store saves raw HTML
5. Processing pipeline:
   - Link Extractor: Finds and queues new URLs
   - Duplicate Detector: Filters seen content
   - Content Parser: Extracts text for search index

### URL Frontier

**URL Frontier Design:**
Maintains separate per-host queues to enforce politeness (delay between requests to same host). A global priority queue schedules which host to crawl next based on last crawl time. When getting next URL:
1. Pop host with earliest eligible crawl time
2. Wait if not yet eligible
3. Get URL from that host's queue
4. Update last crawl time, re-add host if more URLs

### Crawler Worker

**Crawler Worker Process:**
1. Get next URL from frontier
2. Check seen URLs (Bloom filter) - skip if duplicate
3. Check robots.txt (cached per host) - skip if disallowed
4. Fetch page content with timeout and proper User-Agent
5. Store content for processing
6. Extract links, calculate priority, add to frontier
7. Handle errors appropriately (retry, skip, etc.)

**Fetch Process:**
Use cached DNS resolution, send HTTP request with timeout (30s), identify crawler via User-Agent header.

### Duplicate Detection

**URL Duplicate Detection:**
Normalize URLs (lowercase, remove fragments, sort parameters) and check against a Bloom filter (10B capacity). False positives OK (skip a few URLs), no false negatives.

**Content Duplicate Detection:**
Use SimHash (64-bit locality-sensitive hash) for near-duplicate detection. Compute SimHash of page content, compare with stored hashes. If Hamming distance < 3, pages are near-duplicates. Store canonical URL for each SimHash.

**SimHash Algorithm:**
For each token in content, compute hash. For each bit position, if bit is 1 add 1 to counter, else subtract 1. Final SimHash: bit is 1 if counter > 0. Similar documents have similar SimHashes.

### Robots.txt Parser

**Robots.txt Handling:**
Cache robots.txt per domain (24-hour TTL). Parse rules for User-Agent matching, check path against allow/disallow rules. If can't fetch robots.txt, assume allowed. Also extract Crawl-delay directive for politeness.

### Priority Calculation

**URL Priority Score:**
- Domain authority (pre-computed PageRank-like score) × 10
- URL depth penalty (shorter paths higher priority)
- Content type hints (+5 for HTML, +2 for PDF)
- Freshness signals from parent page
- Result: higher score = higher priority

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

**Formula:** m = -n × ln(p) / (ln(2))² where n=items, p=false positive rate

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

**Upload Flow:**
1. Creator uploads video to Upload Service
2. Upload Service stores original in Object Store (S3)
3. Transcoding Pipeline processes video into multiple resolutions
4. Outputs stored for each quality level (1080p, 720p, 480p)
5. Content distributed to CDN Edge Caches

**Playback Flow:**
1. Client requests video stream
2. CDN serves content from nearest edge location
3. Adaptive bitrate player selects quality based on bandwidth

### Video Processing Pipeline

**Video Processing Steps:**
1. **Validation:** Extract and validate video metadata
2. **Transcoding:** Convert to multiple resolution/bitrate profiles in parallel
   - 1080p at 5000 kbps (H.264)
   - 720p at 2500 kbps (H.264)
   - 480p at 1000 kbps (H.264)
   - 360p at 500 kbps (H.264)
3. **Segmentation:** Create HLS/DASH segments (6-second chunks)
4. **Thumbnails:** Generate preview images at intervals
5. **Update status:** Mark video as ready for playback

**Transcoding:** Uses ffmpeg with resolution scaling, bitrate targeting, and AAC audio encoding.

### Adaptive Bitrate Streaming

**HLS Master Playlist Structure:**
The master playlist (m3u8) lists available quality levels with their bandwidth and resolution. Each quality level has its own segment playlist. The player selects based on available bandwidth and can switch mid-stream.

**Adaptive Player Logic:**
- Select quality based on measured bandwidth thresholds
- Be conservative when buffer is low (prefer lower quality to avoid stalling)
- Smooth transitions (don't switch too frequently)
- Initial selection may start lower and ramp up

### CDN and Caching

**CDN Strategy:**
- Select nearest edge location based on user geography
- Generate signed URLs with expiration for security
- Pre-warm cache for predicted popular content
- High predicted popularity: push to all edges
- Medium popularity: push to high-traffic edges only

### View Count System

**View Counting Strategy:**
1. Deduplicate views per user within time window (1 hour)
2. Increment counter in Redis for real-time counts
3. Batch sync to database periodically (every 1000 views)
4. This balances real-time display with database efficiency

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

**Architecture Flow:**
1. Clients connect to Master Node for metadata operations
2. Master Node maintains:
   - Namespace (file paths → metadata)
   - Chunk Mapping (chunk IDs → server locations)
3. Data operations go directly to Chunk Servers
4. Each Chunk Server stores multiple chunks
5. Chunks are replicated across servers for fault tolerance
6. Example: Chunk A exists on Servers 1, 2, 3 for replication

### Chunk Management

**Master Node Operations:**
- Create file: Add to namespace, return file handle
- Allocate chunk: Generate unique chunk ID, select servers (considering load and rack diversity), record allocation
- Server selection: Prefer servers with available space, ensure rack diversity for fault tolerance

### Write Path

**Write Process:**
1. Create file on master (get file handle)
2. Split data into chunks (64 MB each)
3. For each chunk:
   - Get chunk allocation from master (primary + secondaries)
   - Pipeline data to all replicas
   - Commit on primary (coordinates with secondaries)

**Chunk Server Write:**
1. Receive data into pending buffer
2. On commit: write to stable storage with checksum
3. Acknowledge to primary/client

### Read Path

**Read Process:**
1. Get file metadata (chunk list) from master
2. For each chunk:
   - Get chunk locations from master
   - Try each replica until successful read with valid checksum
3. Concatenate chunks to return file content

### Heartbeat and Lease

**Chunk Server Heartbeat:**
Every 10 seconds, report to master:
- Server ID and available space
- List of chunks stored
- Current load metrics

**Master Monitoring:**
- Track last heartbeat time per server
- If no heartbeat for lease duration (60s), handle server failure
- Re-replicate affected chunks to maintain replication factor

### Replication and Recovery

**Re-replication Process:**
When a chunk becomes under-replicated:
1. Identify chunks on failed server
2. For each chunk, select new server (prefer different rack)
3. Copy data from existing replica to new server
4. Update master's chunk location map

**Background Balancing:**
Periodically redistribute chunks from overloaded servers to underloaded ones, targeting balanced disk utilization across cluster.

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
