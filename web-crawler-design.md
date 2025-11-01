# Web Crawler System Design

## Table of Contents
1. [Requirements](#requirements)
2. [High-Level Architecture](#high-level-architecture)
3. [Detailed Component Design](#detailed-component-design)
4. [Data Models](#data-models)
5. [Scalability Strategies](#scalability-strategies)
6. [High Availability & Fault Tolerance](#high-availability--fault-tolerance)
7. [Performance Optimizations](#performance-optimizations)
8. [Monitoring & Observability](#monitoring--observability)
9. [Trade-offs & Considerations](#trade-offs--considerations)

## Requirements

### Functional Requirements
- **Crawl web pages** starting from seed URLs
- **Extract links** from crawled pages for further crawling
- **Store page content** and metadata
- **Respect robots.txt** and crawl politeness policies
- **Support recrawling** for content freshness
- **Handle different content types** (HTML, images, PDFs, etc.)
- **Deduplication** to avoid crawling the same URL multiple times
- **Priority-based crawling** (important pages first)

### Non-Functional Requirements
- **Scalability**: Handle billions of URLs
- **High Availability**: 99.9%+ uptime
- **Fault Tolerance**: Graceful degradation, no data loss
- **Performance**: Thousands of pages per second
- **Politeness**: Rate limiting per domain
- **Extensibility**: Easy to add new parsers and plugins
- **Cost Efficiency**: Optimal resource utilization

### Scale Estimations
- **Total URLs**: 10 billion pages
- **Crawl frequency**: Average 1 month refresh
- **Pages per day**: ~330 million (10B / 30 days)
- **Pages per second**: ~3,800 pages/sec
- **Average page size**: 100 KB
- **Storage**: ~1 PB for raw content
- **Network bandwidth**: ~380 MB/sec download

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         Load Balancer                            │
└────────────────┬────────────────────────────────────────────────┘
                 │
    ┌────────────┴────────────┐
    │                         │
┌───▼────────┐          ┌────▼──────┐
│  API Layer │          │ Scheduler │
│  (REST)    │          │  Service  │
└───┬────────┘          └────┬──────┘
    │                        │
    │    ┌───────────────────┴─────────────────────┐
    │    │                                         │
┌───▼────▼─────┐    ┌──────────────┐    ┌─────────▼────────┐
│   URL         │◄───┤ Distributed  │◄───┤  Crawler Worker  │
│  Frontier     │    │    Queue     │    │      Pool        │
│  (Priority)   │    │   (Kafka)    │    │  (1000s nodes)   │
└───────────────┘    └──────────────┘    └─────────┬────────┘
                                                    │
    ┌───────────────────────────────────────────────┤
    │                                               │
┌───▼────────────┐    ┌──────────────┐    ┌────────▼─────────┐
│  DNS Resolver  │    │   Storage    │    │  Content Parser  │
│   (Cached)     │    │   Service    │    │    & Extractor   │
└────────────────┘    │  (S3/HDFS)   │    └──────────────────┘
                      └──────┬───────┘
                             │
                      ┌──────▼───────┐
                      │   Metadata   │
                      │   Database   │
                      │  (Cassandra) │
                      └──────────────┘
```

## Detailed Component Design

### 1. URL Frontier (Priority Queue System)

**Responsibilities:**
- Maintain queue of URLs to be crawled
- Implement priority-based selection
- Ensure politeness (delay between requests to same host)
- Deduplication

**Design:**
```
┌─────────────────────────────────────────────────┐
│              URL Frontier Manager                │
├─────────────────────────────────────────────────┤
│  ┌──────────────┐      ┌──────────────────┐    │
│  │ URL Queue    │      │  Prioritization  │    │
│  │ (per host)   │◄────►│     Engine       │    │
│  └──────────────┘      └──────────────────┘    │
│  ┌──────────────┐      ┌──────────────────┐    │
│  │  Politeness  │      │  Deduplication   │    │
│  │   Manager    │      │  (Bloom Filter)  │    │
│  └──────────────┘      └──────────────────┘    │
└─────────────────────────────────────────────────┘
```

**Implementation Details:**
- **Per-host queues**: Separate queue for each domain to enforce politeness
- **Front queues**: Priority-based queues (high, medium, low priority)
- **Back queues**: Per-host queues with rate limiting
- **Selector**: Routes URLs from back queues to front queues
- **Heap**: Priority heap to select next host queue to serve

**Politeness:**
- Minimum delay between requests: 1-5 seconds per host
- Configurable via robots.txt crawl-delay
- Exponential backoff on errors

**Deduplication:**
- **Bloom filter**: Fast probabilistic check (99.9% accuracy)
- **Checksum cache**: MD5/SHA256 of URL for exact deduplication
- **Content fingerprinting**: Avoid duplicate content with different URLs

### 2. Crawler Worker Nodes

**Responsibilities:**
- Fetch web pages via HTTP/HTTPS
- Handle redirects and errors
- Extract content and links
- Respect robots.txt

**Worker Architecture:**
```python
class CrawlerWorker:
    def __init__(self):
        self.http_client = AsyncHTTPClient(pool_size=100)
        self.robots_cache = RobotsCache()
        self.dns_cache = DNSCache()

    async def crawl(self, url_task):
        # 1. Check robots.txt
        if not self.robots_cache.is_allowed(url_task.url):
            return self.mark_forbidden(url_task)

        # 2. DNS resolution (cached)
        ip = await self.dns_cache.resolve(url_task.host)

        # 3. Fetch page
        response = await self.http_client.get(
            url_task.url,
            timeout=30,
            headers=self.get_headers()
        )

        # 4. Extract content and links
        parsed = await self.parse_content(response)

        # 5. Store results
        await self.store(parsed)

        # 6. Enqueue new URLs
        await self.enqueue_urls(parsed.links)

        # 7. Update metrics
        self.metrics.record_success(url_task)
```

**Key Features:**
- **Async I/O**: Non-blocking I/O for high concurrency
- **Connection pooling**: Reuse TCP connections
- **Timeout handling**: Configurable timeouts
- **Retry logic**: Exponential backoff with jitter
- **User-Agent**: Identify crawler and provide contact info

### 3. Distributed Queue (Kafka/RabbitMQ)

**Purpose:**
- Decouple URL frontier from crawler workers
- Provide buffering and load distribution
- Enable horizontal scaling

**Topic Structure:**
```
topics:
  - crawl.high_priority    (partition: 10)
  - crawl.medium_priority  (partition: 50)
  - crawl.low_priority     (partition: 100)
  - crawl.recrawl          (partition: 20)

consumer_groups:
  - crawler_worker_group (1000+ instances)
```

**Message Format:**
```json
{
  "url": "https://example.com/page",
  "priority": 8,
  "depth": 3,
  "parent_url": "https://example.com",
  "scheduled_at": 1698765432,
  "metadata": {
    "last_crawled": 1698000000,
    "crawl_frequency": "weekly"
  }
}
```

**Guarantees:**
- **At-least-once delivery**: Kafka consumer groups
- **Ordering**: Not strictly required (relaxed for performance)
- **Retention**: 7 days for replay capability

### 4. Content Parser & Extractor

**Responsibilities:**
- Parse HTML/XML content
- Extract links and resources
- Extract structured data
- Normalize and clean content

**Parser Pipeline:**
```
Raw HTML → Parser → Link Extractor → Content Extractor → Storage
            │
            ├─→ HTML Parser (BeautifulSoup/lxml)
            ├─→ PDF Parser
            ├─→ Image Parser
            └─→ Video Parser
```

**Link Extraction:**
```python
class LinkExtractor:
    def extract_links(self, html, base_url):
        links = []
        soup = BeautifulSoup(html, 'lxml')

        # Extract from various tags
        for tag in soup.find_all(['a', 'link', 'area']):
            href = tag.get('href')
            if href:
                absolute_url = urljoin(base_url, href)
                normalized = self.normalize_url(absolute_url)
                links.append({
                    'url': normalized,
                    'anchor_text': tag.get_text(),
                    'rel': tag.get('rel'),
                    'type': tag.name
                })

        return links

    def normalize_url(self, url):
        # Remove fragments, sort query params, lowercase
        parsed = urlparse(url.lower())
        query = urlencode(sorted(parse_qs(parsed.query).items()))
        return urlunparse((
            parsed.scheme,
            parsed.netloc,
            parsed.path.rstrip('/'),
            '',
            query,
            ''
        ))
```

### 5. DNS Resolver

**Purpose:**
- Resolve domain names to IP addresses
- Cache results to reduce latency
- Handle DNS failures gracefully

**Implementation:**
```python
class DNSCache:
    def __init__(self):
        self.cache = LRUCache(capacity=1_000_000)
        self.resolver = aiodns.DNSResolver()

    async def resolve(self, hostname):
        # Check cache first
        if hostname in self.cache:
            return self.cache[hostname]

        # Resolve and cache
        try:
            result = await self.resolver.gethostbyname(hostname)
            ip = result.addresses[0]
            self.cache[hostname] = ip
            return ip
        except Exception as e:
            # Fallback or raise
            raise DNSResolutionError(f"Failed to resolve {hostname}")
```

**Optimizations:**
- **TTL-aware caching**: Respect DNS TTL values
- **Prefetching**: Proactively resolve popular domains
- **Local DNS server**: Dedicated caching DNS resolver

### 6. Storage Service

**Two-tier storage strategy:**

**Hot Storage (Metadata):**
- **Database**: Cassandra / ScyllaDB
- **Purpose**: Store crawl metadata, URL state, scheduling info
- **Partition key**: URL hash
- **Replication**: RF=3 across multiple DCs

**Schema:**
```sql
CREATE TABLE crawled_urls (
    url_hash blob PRIMARY KEY,
    url text,
    last_crawled timestamp,
    next_crawl timestamp,
    http_status int,
    content_hash blob,
    title text,
    outlinks int,
    crawl_count int,
    error_count int,
    priority int
);

CREATE TABLE url_content (
    url_hash blob,
    crawled_at timestamp,
    content_type text,
    size int,
    storage_path text,
    checksum text,
    PRIMARY KEY (url_hash, crawled_at)
) WITH CLUSTERING ORDER BY (crawled_at DESC);
```

**Cold Storage (Content):**
- **Storage**: S3 / HDFS / Ceph
- **Purpose**: Store raw HTML, images, PDFs
- **Organization**: Partitioned by date and domain
- **Compression**: gzip/zstd
- **Path structure**: `s3://crawler-data/year=2024/month=11/domain=example.com/`

**Benefits:**
- Cost-effective for large content
- Scalable to petabytes
- Lifecycle policies for old data
- Easy backup and replication

### 7. Scheduler Service

**Responsibilities:**
- Determine what URLs to crawl and when
- Implement crawl frequency policies
- Prioritize important pages
- Handle recrawls

**Scheduling Strategies:**

**A. Freshness-based:**
```
Priority = Importance × (1 / FreshnessFactor)

where:
  FreshnessFactor = time_since_last_crawl / expected_update_frequency
```

**B. Change detection:**
- Track content change history
- Increase frequency for frequently changing pages
- Decrease frequency for static pages

**C. Priority factors:**
- PageRank / Domain authority
- Backlink count
- User engagement metrics
- Content type (news vs. static docs)

**Implementation:**
```python
class Scheduler:
    def calculate_next_crawl(self, url_data):
        # Base frequency
        base_interval = url_data.crawl_frequency

        # Adjust based on change rate
        if url_data.change_rate > 0.8:  # Changes often
            interval = base_interval * 0.5
        elif url_data.change_rate < 0.2:  # Rarely changes
            interval = base_interval * 2.0
        else:
            interval = base_interval

        # Apply priority boost
        priority_factor = url_data.priority / 10.0
        interval = interval / priority_factor

        next_crawl = url_data.last_crawled + timedelta(seconds=interval)
        return next_crawl

    def run_scheduler(self):
        # Periodic job (every hour)
        while True:
            # Find URLs due for recrawl
            due_urls = self.db.get_urls_due_for_crawl(
                before=datetime.now(),
                limit=100_000
            )

            # Enqueue with priority
            for url_data in due_urls:
                priority = self.calculate_priority(url_data)
                self.queue.enqueue(url_data.url, priority)

            time.sleep(3600)  # Run every hour
```

### 8. Robots.txt Handler

**Purpose:**
- Fetch and parse robots.txt
- Cache rules per domain
- Enforce crawl policies

**Implementation:**
```python
class RobotsCache:
    def __init__(self):
        self.cache = TTLCache(capacity=100_000, ttl=86400)  # 24h TTL

    async def is_allowed(self, url, user_agent='MyCrawler'):
        parsed = urlparse(url)
        domain = parsed.netloc

        # Check cache
        if domain not in self.cache:
            await self.fetch_robots(domain)

        robots = self.cache.get(domain)
        if robots is None:
            return True  # No robots.txt, allow

        return robots.can_fetch(user_agent, parsed.path)

    async def fetch_robots(self, domain):
        robots_url = f"https://{domain}/robots.txt"
        try:
            response = await self.http_client.get(robots_url, timeout=10)
            parser = RobotFileParser()
            parser.parse(response.text.splitlines())
            self.cache[domain] = parser
        except Exception:
            self.cache[domain] = None  # Allow by default on error
```

## Data Models

### URL State Machine
```
   NEW
    │
    ├──→ QUEUED ──→ FETCHING ──→ PARSED ──→ STORED
    │                  │                       │
    │                  ↓                       │
    │              ERROR ←──────────────────────
    │                  │
    │                  ↓
    └───────────→ SCHEDULED (for recrawl)
```

### Core Entities

**URL Entity:**
```
URL {
  id: UUID
  url: string
  url_hash: bytes
  domain: string
  status: enum(NEW, QUEUED, FETCHING, PARSED, STORED, ERROR)
  priority: int (0-10)
  depth: int
  created_at: timestamp
  last_crawled: timestamp
  next_crawl: timestamp
  crawl_count: int
  error_count: int
  http_status: int
  content_hash: bytes
  content_type: string
  size: int
  metadata: json
}
```

**Crawl Result:**
```
CrawlResult {
  url_hash: bytes
  crawled_at: timestamp
  http_status: int
  headers: map<string, string>
  content_type: string
  content_length: int
  storage_path: string
  checksum: string
  outlinks: list<string>
  duration_ms: int
  error: string
}
```

## Scalability Strategies

### Horizontal Scaling

**1. Stateless Workers:**
- All crawler workers are stateless
- Can add/remove workers dynamically
- Auto-scaling based on queue depth

**2. Partitioning:**
- **URL frontier**: Partition by domain hash
- **Queue**: Multiple topic partitions
- **Storage**: Shard by URL hash
- **Cache**: Consistent hashing

**3. Load Balancing:**
```
Request → Load Balancer → Worker Pool (1000s)
                          (Round-robin / Least-connections)
```

### Vertical Scaling Optimizations

**1. Memory Management:**
- Connection pooling
- Object pooling for parsers
- Memory-mapped files for large datasets

**2. CPU Optimization:**
- Async I/O (avoid blocking)
- Multi-threading for parsing
- Batch processing

**3. Network:**
- HTTP/2 for multiplexing
- Keep-alive connections
- Compression (gzip, brotli)

### Data Partitioning Strategies

**By Domain:**
```
Shard = hash(domain) % num_shards

Benefits:
  - Locality: All URLs from same domain on same shard
  - Politeness: Easier to enforce rate limits
  - Caching: DNS and robots.txt per shard
```

**By URL Hash:**
```
Shard = hash(url) % num_shards

Benefits:
  - Even distribution
  - No hotspots
  - Simple implementation
```

**Hybrid Approach:**
- Use domain-based partitioning for URL frontier
- Use hash-based partitioning for storage

## High Availability & Fault Tolerance

### Redundancy

**1. Component Redundancy:**
```
Every component has N+2 instances:
  - API Layer: 3+ instances behind LB
  - Scheduler: 3 instances (leader election)
  - Workers: 1000s of instances
  - Database: 3+ replicas
  - Queue: 3+ brokers with replication
  - Storage: Replicated (S3: 11 9s durability)
```

**2. Geographic Redundancy:**
```
Multi-region deployment:
  - Primary DC: us-east-1
  - Secondary DC: us-west-2
  - Tertiary DC: eu-west-1

Data replication across DCs
```

### Failure Handling

**Worker Failures:**
```python
# Message acknowledgment pattern
def process_url(url_task):
    try:
        result = crawl_url(url_task)
        store_result(result)
        queue.ack(url_task.id)  # Mark as completed
    except Exception as e:
        queue.nack(url_task.id)  # Return to queue
        log_error(e)
```

**Features:**
- **Heartbeat monitoring**: Detect dead workers
- **Task timeout**: Requeue if not completed in time
- **Max retries**: Give up after N failures
- **Circuit breaker**: Stop crawling problematic hosts

**Database Failures:**
- **Read replicas**: Failover to replica on primary failure
- **Write quorum**: W + R > N (consistency)
- **Eventual consistency**: Accept for non-critical reads

**Queue Failures:**
- **Replication**: Kafka replication factor = 3
- **ISR**: In-sync replicas for durability
- **Consumer groups**: Multiple consumers for failover

**Network Failures:**
- **Retry with exponential backoff**
- **Circuit breaker pattern**
- **Fallback mechanisms**

### Data Consistency

**URL Deduplication:**
```
1. Bloom filter (fast, probabilistic)
   → 99% of duplicates filtered

2. Distributed cache (Redis)
   → Check recent URLs (last 7 days)

3. Database lookup (authoritative)
   → Final check before crawl
```

**Eventual Consistency:**
- Acceptable for crawl scheduling
- Duplicate crawls occasionally OK (idempotent)
- Deduplication in storage layer

**Strong Consistency:**
- Required for: Critical metadata, billing, quotas
- Use distributed transactions (2PC) sparingly

### Disaster Recovery

**Backup Strategy:**
```
1. Metadata DB:
   - Daily full backup
   - Continuous incremental backup
   - Point-in-time recovery (PITR)

2. Content Storage:
   - Cross-region replication
   - Versioning enabled
   - Lifecycle policies

3. Configuration:
   - Version controlled (Git)
   - Infrastructure as Code (Terraform)
```

**Recovery Procedures:**
```
RTO (Recovery Time Objective): 1 hour
RPO (Recovery Point Objective): 15 minutes

Failure Scenarios:
1. Worker node failure → Auto-replace (1 min)
2. Database failure → Failover to replica (5 min)
3. Queue failure → Broker failover (2 min)
4. Full DC failure → Region failover (30 min)
5. Data corruption → Restore from backup (1 hour)
```

## Performance Optimizations

### Caching Strategy

**Multi-level Cache:**
```
Request
  ↓
L1: Worker Local Cache (in-memory, 1GB)
  ↓ (miss)
L2: Distributed Cache (Redis, 100GB)
  ↓ (miss)
L3: Database
```

**What to Cache:**
- DNS resolutions (TTL: 1 hour)
- Robots.txt (TTL: 24 hours)
- URL deduplication (TTL: 7 days)
- Page content fingerprints (TTL: 30 days)

### Connection Pooling

```python
class ConnectionPool:
    def __init__(self, max_size=100):
        self.pool = asyncio.Queue(maxsize=max_size)
        for _ in range(max_size):
            conn = self.create_connection()
            self.pool.put_nowait(conn)

    async def get_connection(self):
        return await self.pool.get()

    async def return_connection(self, conn):
        if conn.is_healthy():
            await self.pool.put(conn)
        else:
            conn = self.create_connection()
            await self.pool.put(conn)
```

**Benefits:**
- Reuse TCP connections
- Reduce handshake overhead
- Better throughput

### Batch Processing

**Batch URL Enqueuing:**
```python
# Instead of one-by-one
for url in urls:
    queue.enqueue(url)  # Bad: N network calls

# Batch insert
queue.enqueue_batch(urls)  # Good: 1 network call
```

**Batch Storage:**
```python
# Batch write to database
db.batch_insert(crawl_results, batch_size=1000)

# Batch upload to S3
s3.upload_batch(contents, parallel=10)
```

### Compression

**Content Compression:**
- Store compressed HTML (gzip/zstd)
- 5-10x space savings
- Transparent decompression on read

**Network Compression:**
- Accept-Encoding: gzip, br
- Reduce bandwidth by 70-80%

### Request Prioritization

**Priority Levels:**
```
Level 10: Critical (news sites, high-traffic)
Level 7-9: High (popular sites, frequent updates)
Level 4-6: Medium (regular sites)
Level 1-3: Low (infrequent updates, deep pages)
Level 0: Background (recrawl, archive)
```

**Implementation:**
```python
def calculate_priority(url_data):
    priority = 5  # Base

    # Factor 1: PageRank/Authority
    priority += min(url_data.pagerank * 3, 3)

    # Factor 2: Freshness requirement
    if url_data.is_news_site:
        priority += 2

    # Factor 3: Depth (penalize deep pages)
    priority -= min(url_data.depth * 0.5, 3)

    # Factor 4: Update frequency
    if url_data.changes_daily:
        priority += 1

    return max(0, min(10, priority))
```

## Monitoring & Observability

### Key Metrics

**Throughput Metrics:**
- Pages crawled per second
- Bytes downloaded per second
- URLs processed per second
- Queue depth (lag)

**Latency Metrics:**
- Average crawl time per page
- DNS resolution time
- HTTP request time
- Parse time
- P50, P95, P99 latencies

**Availability Metrics:**
- Worker node uptime
- Database availability
- Queue availability
- Success rate (%)

**Resource Metrics:**
- CPU utilization
- Memory usage
- Network bandwidth
- Disk I/O
- Queue size

### Alerting

**Critical Alerts:**
```yaml
alerts:
  - name: HighErrorRate
    condition: error_rate > 5%
    duration: 5m
    severity: critical

  - name: QueueDepthHigh
    condition: queue_depth > 10M
    duration: 10m
    severity: warning

  - name: WorkerNodeDown
    condition: worker_count < min_workers * 0.8
    duration: 2m
    severity: critical

  - name: DatabaseReplicationLag
    condition: replication_lag > 60s
    duration: 5m
    severity: warning
```

### Logging

**Structured Logging:**
```json
{
  "timestamp": "2024-11-01T10:30:45Z",
  "level": "INFO",
  "service": "crawler-worker",
  "worker_id": "worker-1234",
  "url": "https://example.com/page",
  "duration_ms": 245,
  "http_status": 200,
  "content_length": 45678,
  "event": "crawl_success"
}
```

**Log Aggregation:**
- Centralized logging (ELK stack / Datadog)
- Search and analysis capabilities
- Retention: 30 days hot, 1 year cold

### Distributed Tracing

**Trace Example:**
```
Request ID: abc123

1. API receives crawl request (0ms)
2. URL enqueued to Kafka (5ms)
3. Worker picks up task (10ms)
4. DNS resolution (15ms)
5. HTTP request (150ms)
6. Parse content (50ms)
7. Store in S3 (30ms)
8. Update metadata DB (20ms)
9. Complete (280ms total)
```

**Tools:**
- OpenTelemetry
- Jaeger / Zipkin
- AWS X-Ray

### Dashboards

**Real-time Dashboard:**
```
┌─────────────────────────────────────────────┐
│  Web Crawler Monitoring Dashboard           │
├─────────────────────────────────────────────┤
│  Throughput: 3,847 pages/sec  ↑ 5%         │
│  Queue Depth: 2.3M URLs                     │
│  Active Workers: 987 / 1000                 │
│  Success Rate: 94.2%                        │
│                                             │
│  [Graph: Pages Crawled Over Time]          │
│  [Graph: Error Rate]                        │
│  [Graph: Latency Distribution]             │
│  [Graph: Resource Utilization]             │
└─────────────────────────────────────────────┘
```

## Trade-offs & Considerations

### 1. Consistency vs. Availability

**Trade-off:**
- Strong consistency → Lower availability, higher latency
- Eventual consistency → Higher availability, potential duplicates

**Decision:**
- Use eventual consistency for crawl queue
- Accept occasional duplicate crawls (idempotent operations)
- Use strong consistency for critical metadata only

### 2. Freshness vs. Resource Usage

**Trade-off:**
- Frequent recrawls → Fresh content, high resource usage
- Infrequent recrawls → Stale content, low resource usage

**Decision:**
- Adaptive crawl frequency based on page change rate
- Prioritize high-value pages for frequent crawls
- Use change detection to optimize recrawl schedule

### 3. Depth vs. Breadth

**Trade-off:**
- Deep crawl → Complete site coverage, slow
- Broad crawl → Wide coverage, miss deep content

**Decision:**
- BFS (Breadth-First) for initial discovery
- Prioritize depth for high-value domains
- Configurable max depth per domain

### 4. Politeness vs. Throughput

**Trade-off:**
- Aggressive crawling → High throughput, risk of blocking
- Polite crawling → Lower throughput, better relationships

**Decision:**
- Respect robots.txt strictly
- Implement per-domain rate limiting
- Use distributed workers to maintain overall throughput
- Adaptive backoff on errors

### 5. Cost vs. Performance

**Trade-off:**
- More resources → Better performance, higher cost
- Fewer resources → Lower cost, degraded performance

**Decision:**
- Auto-scaling based on queue depth
- Use spot instances for cost savings
- Tiered storage (hot/cold) for optimization
- Compression to reduce storage costs

### 6. Centralized vs. Distributed Architecture

**Centralized:**
- Pros: Simple, easier to maintain
- Cons: Single point of failure, limited scale

**Distributed:**
- Pros: Scalable, fault-tolerant
- Cons: Complex, eventual consistency issues

**Decision:**
- Distributed architecture for core components
- Centralized coordination (scheduler) with HA failover

### 7. Push vs. Pull Model

**Push (Queue pushes to workers):**
- Pros: Better load balancing
- Cons: Workers can be overwhelmed

**Pull (Workers pull from queue):**
- Pros: Workers control their load
- Cons: Potential idle time

**Decision:**
- Pull model with consumer groups
- Workers pull at their own pace
- Queue retention for replay capability

## Additional Considerations

### Security

**1. Rate Limiting:**
- Prevent accidental DDoS
- Per-domain limits
- Global throttling

**2. Authentication:**
- API keys for crawl requests
- OAuth for user access
- mTLS for service-to-service

**3. Data Privacy:**
- Respect DNT (Do Not Track) headers
- GDPR compliance for EU sites
- Data retention policies

### Legal & Ethical

**1. robots.txt Compliance:**
- Strict adherence to exclusion rules
- Honor crawl-delay directives
- Respect meta robots tags

**2. Copyright:**
- Store content for indexing, not republishing
- Provide opt-out mechanisms
- Clear crawler identification

**3. Terms of Service:**
- Review ToS before crawling
- Obtain permission for aggressive crawls
- Provide contact information in User-Agent

### Extensibility

**Plugin Architecture:**
```python
class CrawlerPlugin:
    def on_before_crawl(self, url): pass
    def on_after_crawl(self, result): pass
    def on_parse(self, content): pass
    def on_error(self, error): pass

# Example: JavaScript rendering plugin
class JSRenderPlugin(CrawlerPlugin):
    def on_after_crawl(self, result):
        if self.needs_js_rendering(result):
            result.content = self.render_with_browser(result.url)
        return result
```

**Custom Parsers:**
- Plugin system for new content types
- Configurable extraction rules
- Domain-specific parsers

## Conclusion

This web crawler design provides:

**Scalability:**
- Horizontal scaling of all components
- Partitioning and sharding strategies
- Efficient resource utilization

**High Availability:**
- Redundancy at every layer
- Multi-region deployment
- Automatic failover mechanisms

**Fault Tolerance:**
- Graceful degradation
- Retry and circuit breaker patterns
- Data replication and backup

**Performance:**
- Caching at multiple levels
- Connection pooling and reuse
- Batch processing and compression

**Observability:**
- Comprehensive metrics and monitoring
- Distributed tracing
- Centralized logging

The system can handle billions of URLs, maintain high throughput (thousands of pages per second), and operate with 99.9%+ uptime while respecting web politeness policies and legal requirements.
