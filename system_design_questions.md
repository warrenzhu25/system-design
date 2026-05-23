# System Design Interview Questions

A comprehensive collection of system design interview questions from top tech companies.

---

## Table of Contents

### Netflix (9 questions)
1. [Netflix Sentiment Tracking](#1-netflix-sentiment-tracking)
2. [Design the Data Model for an Ads Demand Platform](#2-design-the-data-model-for-an-ads-demand-platform)
3. [Home Page Video Recommendation](#3-home-page-video-recommendation)
4. [Design an Ads Audience Targeting System](#4-design-an-ads-audience-targeting-system)
5. [Design a Billing System for 300M Subscribers](#5-design-a-billing-system-for-300m-subscribers)
6. [Design an Ads Frequency Cap System](#6-design-an-ads-frequency-cap-system)
7. [ML Job Scheduler](#7-ml-job-scheduler)
8. [Design a WAL Log Enrichment Pipeline](#8-design-a-wal-log-enrichment-pipeline)
9. [Design the Data Model for a Promotion Posting System](#9-design-the-data-model-for-a-promotion-posting-system)

### Anthropic (5 questions)
10. [Design a 1-to-1 Chat System](#10-design-a-1-to-1-chat-system)
11. [Inference API System Design](#11-inference-api-system-design)
12. [Prompt Playground System Design](#12-prompt-playground-system-design)
13. [Distributed Model Deployment System Design](#13-distributed-model-deployment-system-design)
14. [LLM Request Batching API System Design](#14-llm-request-batching-api-system-design)

### Databricks (1 question)
15. [Throttling System Design](#15-throttling-system-design)

### Airbnb (1 question)
16. [Data Analysis System Design](#16-data-analysis-system-design)

---

# Netflix

## 1. Netflix Sentiment Tracking

**Type:** System Design

**Problem:** Design a system that tracks overall public sentiment toward Netflix on social media over time.

### Functional Requirements

1. Ingest social media posts from Twitter, Reddit, Facebook, etc.
2. Analyze sentiment (positive, negative, neutral) of each post
3. Aggregate sentiment scores over time windows (hourly, daily, weekly)
4. Detect sudden sentiment changes (e.g., after show releases)
5. Dashboard for real-time and historical sentiment visualization

### Non-Functional Requirements

- **Scale:** Process 10M+ social posts per day
- **Latency:** Near real-time processing (< 5 minute delay)
- **Availability:** 99.9% uptime for dashboard
- **Storage:** 1 year of historical data

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     Data Ingestion Layer                        │
├─────────────┬─────────────┬─────────────┬─────────────┐         │
│ Twitter API │ Reddit API  │ Facebook API│ News RSS    │         │
└──────┬──────┴──────┬──────┴──────┬──────┴──────┬──────┘         │
       │             │             │             │                 │
       └─────────────┴─────────────┴─────────────┘                 │
                           │                                       │
                    ┌──────▼──────┐                                │
                    │ Kafka Queue │  (Raw posts)                   │
                    └──────┬──────┘                                │
                           │                                       │
              ┌────────────┼────────────┐                          │
              │            │            │                          │
       ┌──────▼──────┐ ┌───▼───┐ ┌──────▼──────┐                  │
       │ Flink/Spark │ │  ML   │ │ Aggregation │                  │
       │  Streaming  │ │Service│ │   Service   │                  │
       └──────┬──────┘ └───┬───┘ └──────┬──────┘                  │
              │            │            │                          │
              └────────────┼────────────┘                          │
                           │                                       │
                    ┌──────▼──────┐                                │
                    │  TimescaleDB │ (Time-series storage)         │
                    │  / ClickHouse│                               │
                    └──────┬──────┘                                │
                           │                                       │
                    ┌──────▼──────┐                                │
                    │  Dashboard  │ (Grafana/Custom)               │
                    │    API      │                                │
                    └─────────────┘                                │
└─────────────────────────────────────────────────────────────────┘
```

### Key Components

**1. Data Ingestion**
- API connectors for each social platform
- Rate limiting and backoff handling
- Deduplication using post ID hashing

**2. Sentiment Analysis ML Pipeline**
- Pre-trained transformer model (BERT/RoBERTa fine-tuned on social media)
- Batch inference for efficiency
- Output: sentiment score (-1 to 1), confidence, detected topics

**3. Stream Processing (Apache Flink)**
```python
# Pseudocode for stream processing
stream = kafka.consume("raw_posts")

enriched = stream
    .filter(is_netflix_related)
    .map(analyze_sentiment)
    .window(TumblingWindow(5_minutes))
    .aggregate(compute_avg_sentiment)

enriched.sink_to(timescale_db)
```

**4. Storage Schema**
```sql
CREATE TABLE sentiment_metrics (
    timestamp TIMESTAMPTZ NOT NULL,
    source VARCHAR(50),         -- twitter, reddit, etc.
    sentiment_avg FLOAT,
    sentiment_stddev FLOAT,
    post_count INTEGER,
    positive_count INTEGER,
    negative_count INTEGER,
    neutral_count INTEGER,
    top_topics JSONB,
    PRIMARY KEY (timestamp, source)
);

-- Hypertable for time-series optimization
SELECT create_hypertable('sentiment_metrics', 'timestamp');
```

### Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Message Queue | Kafka | High throughput, replay capability, exactly-once semantics |
| Stream Processing | Flink | Low latency, stateful processing, exactly-once guarantees |
| ML Inference | Batch micro-batches | Balance latency vs GPU efficiency |
| Time-series DB | TimescaleDB | SQL interface, automatic partitioning, good compression |
| Sentiment Model | Fine-tuned BERT | High accuracy for social media text, handles slang/emoji |

### Handling Spikes

- **Auto-scaling:** Kubernetes HPA based on Kafka lag
- **Backpressure:** Flink handles backpressure natively
- **Sampling:** During extreme spikes, sample 10% for real-time, process rest async

### Alert System

```python
# Anomaly detection for sentiment drops
def detect_anomaly(current_sentiment, historical_avg, stddev):
    z_score = (current_sentiment - historical_avg) / stddev
    if z_score < -2.5:  # 2.5 standard deviations below mean
        trigger_alert("Significant negative sentiment detected")
```

### Interview Discussion Points

1. **How to handle multiple languages?**
   - Language detection → route to language-specific models
   - Or use multilingual models (mBERT)

2. **How to attribute sentiment to specific shows?**
   - Named entity recognition to extract show titles
   - Topic modeling to cluster discussions

3. **How to handle bot/spam accounts?**
   - Account age, follower ratio, posting frequency analysis
   - ML-based spam detection as preprocessing step

---

## 2. Design the Data Model for an Ads Demand Platform

**Type:** System Design (Data Modeling)

**Problem:** Create a database schema for Netflix's advertising demand platform that supports advertisers creating campaigns, targeting audiences, and tracking performance.

### Functional Requirements

1. Advertisers can create and manage ad campaigns
2. Campaigns have budgets, schedules, and targeting criteria
3. Support multiple ad creatives per campaign
4. Track impressions, clicks, and conversions
5. Real-time budget tracking and pacing
6. Reporting and analytics

### Entity Relationship Diagram

```
┌─────────────┐       ┌─────────────┐       ┌─────────────┐
│  Advertiser │       │  Campaign   │       │  Ad Group   │
├─────────────┤       ├─────────────┤       ├─────────────┤
│ id (PK)     │──1:N──│ id (PK)     │──1:N──│ id (PK)     │
│ name        │       │ advertiser_id│      │ campaign_id │
│ email       │       │ name        │       │ name        │
│ billing_info│       │ budget      │       │ targeting   │
│ created_at  │       │ daily_budget│       │ bid_amount  │
└─────────────┘       │ start_date  │       │ status      │
                      │ end_date    │       └──────┬──────┘
                      │ status      │              │
                      └─────────────┘              │1:N
                                                   │
                      ┌─────────────┐       ┌──────▼──────┐
                      │  Creative   │       │  Ad Unit    │
                      ├─────────────┤       ├─────────────┤
                      │ id (PK)     │──N:1──│ id (PK)     │
                      │ ad_unit_id  │       │ ad_group_id │
                      │ media_url   │       │ creative_id │
                      │ format      │       │ placement   │
                      │ duration    │       │ status      │
                      └─────────────┘       └─────────────┘
```

### Database Schema

```sql
-- Advertisers
CREATE TABLE advertisers (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    company_name VARCHAR(255),
    billing_address JSONB,
    payment_method_id VARCHAR(100),
    account_status VARCHAR(20) DEFAULT 'active',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Campaigns
CREATE TABLE campaigns (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    advertiser_id UUID REFERENCES advertisers(id),
    name VARCHAR(255) NOT NULL,
    objective VARCHAR(50) NOT NULL,  -- awareness, consideration, conversion
    total_budget DECIMAL(12,2) NOT NULL,
    daily_budget DECIMAL(12,2),
    spent_budget DECIMAL(12,2) DEFAULT 0,
    start_date DATE NOT NULL,
    end_date DATE,
    status VARCHAR(20) DEFAULT 'draft',  -- draft, active, paused, completed
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),

    INDEX idx_campaigns_advertiser (advertiser_id),
    INDEX idx_campaigns_status (status),
    INDEX idx_campaigns_dates (start_date, end_date)
);

-- Ad Groups (targeting units within campaigns)
CREATE TABLE ad_groups (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    campaign_id UUID REFERENCES campaigns(id),
    name VARCHAR(255) NOT NULL,
    targeting_criteria JSONB NOT NULL,
    /*
    targeting_criteria example:
    {
        "demographics": {"age_min": 18, "age_max": 35, "genders": ["M", "F"]},
        "interests": ["comedy", "drama", "action"],
        "devices": ["smart_tv", "mobile", "web"],
        "geographies": ["US", "CA", "UK"],
        "dayparting": {"hours": [18, 19, 20, 21, 22], "days": ["Sat", "Sun"]}
    }
    */
    bid_strategy VARCHAR(50) DEFAULT 'cpm',  -- cpm, cpc, cpv
    bid_amount DECIMAL(10,4) NOT NULL,
    status VARCHAR(20) DEFAULT 'active',
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Creatives (ad content)
CREATE TABLE creatives (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    advertiser_id UUID REFERENCES advertisers(id),
    name VARCHAR(255) NOT NULL,
    format VARCHAR(50) NOT NULL,  -- video_15s, video_30s, banner, overlay
    media_url VARCHAR(500) NOT NULL,
    thumbnail_url VARCHAR(500),
    duration_seconds INTEGER,
    click_through_url VARCHAR(500),
    approval_status VARCHAR(20) DEFAULT 'pending',
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Ad Units (creative + ad group association)
CREATE TABLE ad_units (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    ad_group_id UUID REFERENCES ad_groups(id),
    creative_id UUID REFERENCES creatives(id),
    placement VARCHAR(50) NOT NULL,  -- pre_roll, mid_roll, pause_screen
    weight INTEGER DEFAULT 1,  -- for rotation
    status VARCHAR(20) DEFAULT 'active',

    UNIQUE(ad_group_id, creative_id, placement)
);

-- Impression Events (for analytics - likely in a separate analytics DB)
CREATE TABLE impressions (
    id UUID PRIMARY KEY,
    ad_unit_id UUID NOT NULL,
    campaign_id UUID NOT NULL,
    user_id UUID,
    session_id UUID,
    content_id UUID,  -- What content was the ad shown on
    device_type VARCHAR(50),
    country VARCHAR(2),
    timestamp TIMESTAMPTZ NOT NULL,

    -- Partitioned by time for efficiency
    PRIMARY KEY (id, timestamp)
) PARTITION BY RANGE (timestamp);

-- Budget Ledger (for real-time budget tracking)
CREATE TABLE budget_ledger (
    id BIGSERIAL PRIMARY KEY,
    campaign_id UUID NOT NULL,
    amount DECIMAL(12,4) NOT NULL,  -- positive = spend, negative = refund
    event_type VARCHAR(50) NOT NULL,
    event_id UUID,
    timestamp TIMESTAMPTZ DEFAULT NOW(),

    INDEX idx_ledger_campaign (campaign_id, timestamp)
);
```

### Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Primary Keys | UUID | Distributed generation, no central sequence |
| Targeting Storage | JSONB | Flexible schema for evolving targeting options |
| Budget Tracking | Ledger pattern | Audit trail, easy reconciliation |
| Impressions Table | Partitioned | Time-series queries, efficient pruning |
| Separate Analytics DB | Yes | OLTP vs OLAP separation |

### Real-Time Budget Enforcement

```python
# Redis for real-time budget checks
class BudgetService:
    def __init__(self, redis_client):
        self.redis = redis_client

    def can_serve_ad(self, campaign_id: str, cost: float) -> bool:
        """Check if campaign has budget for this impression."""
        key = f"budget:{campaign_id}"

        # Atomic check and decrement
        remaining = self.redis.incrbyfloat(key, -cost)

        if remaining < 0:
            # Rollback and reject
            self.redis.incrbyfloat(key, cost)
            return False

        return True

    def sync_to_db(self, campaign_id: str):
        """Periodic sync from Redis to PostgreSQL."""
        spent = self.redis.get(f"budget:{campaign_id}:spent")
        # Update campaigns.spent_budget in database
```

### Indexing Strategy

```sql
-- Frequently queried combinations
CREATE INDEX idx_campaigns_active ON campaigns(advertiser_id)
    WHERE status = 'active';

CREATE INDEX idx_ad_groups_targeting ON ad_groups
    USING GIN(targeting_criteria);

CREATE INDEX idx_impressions_reporting ON impressions(campaign_id, timestamp)
    INCLUDE (device_type, country);
```

### Interview Discussion Points

1. **How to handle high write throughput for impressions?**
   - Write to Kafka first, batch insert to analytics DB
   - Use ClickHouse or Druid for real-time analytics

2. **How to ensure budget isn't overspent?**
   - Redis for real-time tracking with atomic operations
   - Periodic reconciliation with source of truth in DB

3. **How to handle campaign updates mid-flight?**
   - Versioned configurations, effective timestamps
   - Cache invalidation via pub/sub

---

## 3. Home Page Video Recommendation

**Type:** System Design

**Problem:** Design Netflix's home page video recommendation system that personalizes content rows for 200M+ users.

### Functional Requirements

1. Generate personalized homepage with multiple rows of content
2. Each row has a theme (e.g., "Continue Watching", "Trending Now", "Because you watched X")
3. Real-time updates based on user interactions
4. A/B testing for recommendation algorithms
5. Handle cold start for new users

### Non-Functional Requirements

- **Scale:** 200M+ users, 15K+ titles
- **Latency:** < 200ms for homepage generation
- **Freshness:** Recommendations updated within hours of new viewing
- **Availability:** 99.99% uptime

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Client Request                              │
│                    (User opens Netflix app)                         │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                    ┌────────▼────────┐
                    │   API Gateway   │
                    │   (Edge Cache)  │
                    └────────┬────────┘
                             │
              ┌──────────────┼──────────────┐
              │              │              │
       ┌──────▼──────┐ ┌─────▼─────┐ ┌──────▼──────┐
       │  Homepage   │ │   User    │ │   Content   │
       │  Assembler  │ │  Profile  │ │   Service   │
       │   Service   │ │  Service  │ │             │
       └──────┬──────┘ └─────┬─────┘ └──────┬──────┘
              │              │              │
              └──────────────┼──────────────┘
                             │
                    ┌────────▼────────┐
                    │  Recommendation │
                    │     Engine      │
                    └────────┬────────┘
                             │
        ┌────────────────────┼────────────────────┐
        │                    │                    │
┌───────▼───────┐   ┌────────▼────────┐  ┌────────▼────────┐
│ Offline Model │   │ Feature Store   │  │ Real-time Model │
│   (Spark)     │   │ (Redis/Feast)   │  │   (Online)      │
└───────┬───────┘   └────────┬────────┘  └────────┬────────┘
        │                    │                    │
        └────────────────────┼────────────────────┘
                             │
                    ┌────────▼────────┐
                    │   Data Lake     │
                    │ (User history,  │
                    │  content meta)  │
                    └─────────────────┘
```

### Recommendation Layers

**Layer 1: Candidate Generation (Offline)**
- Collaborative filtering: Users who watched X also watched Y
- Content-based: Similar genres, actors, directors
- Output: ~1000 candidates per user, computed daily

**Layer 2: Ranking (Near Real-time)**
- ML model scores each candidate
- Features: user history, time of day, device, content freshness
- Output: Top 100-200 ranked titles

**Layer 3: Row Generation (Real-time)**
- Organize ranked titles into themed rows
- Diversification: Avoid duplicate genres in sequence
- Business rules: Promote originals, new releases

### Data Model

```python
# User Profile (cached in Redis)
user_profile = {
    "user_id": "u123",
    "viewing_history": [
        {"content_id": "c1", "watch_pct": 0.85, "timestamp": "..."},
        {"content_id": "c2", "watch_pct": 0.30, "timestamp": "..."},
    ],
    "genre_preferences": {"drama": 0.8, "comedy": 0.6, "action": 0.4},
    "taste_cluster": 42,  # Pre-computed cluster ID
    "last_active": "2024-01-15T20:30:00Z"
}

# Pre-computed recommendations (refreshed every few hours)
recommendations = {
    "user_id": "u123",
    "generated_at": "2024-01-15T18:00:00Z",
    "rows": [
        {
            "row_type": "continue_watching",
            "title": "Continue Watching",
            "items": ["c2", "c5", "c8"]  # Incomplete views
        },
        {
            "row_type": "because_you_watched",
            "title": "Because You Watched Stranger Things",
            "items": ["c10", "c15", "c20"],
            "seed_content": "stranger_things"
        },
        {
            "row_type": "trending",
            "title": "Trending Now",
            "items": ["c100", "c101", "c102"]
        }
    ]
}
```

### Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Pre-computation | Yes, offline + real-time hybrid | Can't compute for 200M users in real-time |
| Storage | Redis for hot data, S3 for offline models | Sub-ms latency for serving |
| ML Framework | Two-tower model | Efficient candidate retrieval via ANN |
| Personalization Granularity | User-level | Row-level personalization is key differentiator |
| Cold Start | Content popularity + demographics | New users get trending + demographic-based |

### Serving Flow

```python
class HomepageService:
    def get_homepage(self, user_id: str, device: str) -> Homepage:
        # 1. Fetch pre-computed recommendations
        cached_recs = self.redis.get(f"recs:{user_id}")

        if not cached_recs or self.is_stale(cached_recs):
            # 2. Real-time re-ranking if needed
            cached_recs = self.rerank_recommendations(user_id)

        # 3. Apply real-time signals
        rows = self.apply_realtime_signals(cached_recs, user_id)

        # 4. Add continue watching row (always real-time)
        continue_watching = self.get_continue_watching(user_id)
        rows.insert(0, continue_watching)

        # 5. Apply device-specific layout
        rows = self.adapt_to_device(rows, device)

        return Homepage(rows=rows)

    def apply_realtime_signals(self, recs, user_id):
        """Apply real-time signals like:
        - Remove just-watched content
        - Boost content with new episodes
        - Apply time-of-day preferences
        """
        recent_watches = self.get_recent_watches(user_id, hours=2)
        return [r for r in recs if r.content_id not in recent_watches]
```

### Handling Scale

**Pre-computation Pipeline (Spark)**
```
Daily: For each user
    1. Fetch user features
    2. Run candidate generation (ANN lookup)
    3. Run ranking model
    4. Generate row assignments
    5. Write to Redis/S3

Throughput: 200M users / 24 hours = 2,300 users/second
```

**Serving Infrastructure**
- CDN edge caching for static content metadata
- Redis cluster for user recommendations (sharded by user_id)
- Circuit breakers for graceful degradation

### A/B Testing Framework

```python
class ABTestingFramework:
    def get_recommendation_config(self, user_id: str) -> Config:
        # Consistent hashing for stable assignment
        bucket = hash(user_id) % 100

        if bucket < 10:
            return Config(model="new_ranking_v2", row_count=8)
        else:
            return Config(model="current_prod", row_count=6)
```

### Interview Discussion Points

1. **How to handle content that becomes unavailable?**
   - Real-time content availability check before serving
   - Background job to update pre-computed recs

2. **How to balance exploration vs exploitation?**
   - Epsilon-greedy: 10% of rows show exploratory content
   - Thompson sampling for row ordering

3. **How to measure recommendation quality?**
   - Primary: Watch time, completion rate
   - Secondary: Click-through rate, session length
   - Guardrail: Diversity metrics, coverage

---

## 4. Design an Ads Audience Targeting System

**Type:** System Design

**Problem:** Design a system that enables advertisers to upload custom audience lists, package them into segments, and leverage them for precise ad targeting purposes.

### Functional Requirements

1. Upload audience lists (hashed emails, device IDs, user IDs)
2. Create segments from multiple lists (union, intersection, exclusion)
3. Real-time audience membership lookup during ad serving
4. Audience size estimation before campaign launch
5. Support for lookalike audience expansion

### Non-Functional Requirements

- **Scale:** 1B+ user identifiers, 10K+ audience segments
- **Lookup Latency:** < 10ms for membership check
- **Upload:** Support files with 100M+ identifiers
- **Privacy:** All identifiers must be hashed/encrypted

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Audience Management UI                       │
└────────────────────────────┬────────────────────────────────────┘
                             │
              ┌──────────────┼──────────────┐
              │              │              │
       ┌──────▼──────┐ ┌─────▼─────┐ ┌──────▼──────┐
       │   Upload    │ │  Segment  │ │  Estimation │
       │   Service   │ │  Builder  │ │   Service   │
       └──────┬──────┘ └─────┬─────┘ └──────┬──────┘
              │              │              │
              └──────────────┼──────────────┘
                             │
                    ┌────────▼────────┐
                    │  Audience Store │
                    │   (Distributed) │
                    └────────┬────────┘
                             │
              ┌──────────────┼──────────────┐
              │              │              │
       ┌──────▼──────┐ ┌─────▼─────┐ ┌──────▼──────┐
       │   Bitmap    │ │   Bloom   │ │  Identity   │
       │    Store    │ │  Filters  │ │   Graph     │
       │  (Roaring)  │ │  (Cache)  │ │  (Matches)  │
       └─────────────┘ └───────────┘ └─────────────┘
                             │
                    ┌────────▼────────┐
                    │   Ad Serving    │
                    │  (Lookup API)   │
                    └─────────────────┘
```

### Data Model

```sql
-- Audience Lists (uploaded by advertisers)
CREATE TABLE audience_lists (
    id UUID PRIMARY KEY,
    advertiser_id UUID NOT NULL,
    name VARCHAR(255) NOT NULL,
    identifier_type VARCHAR(50) NOT NULL,  -- email_hash, device_id, user_id
    identifier_count BIGINT DEFAULT 0,
    upload_status VARCHAR(20) DEFAULT 'processing',
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- List Members (stored in distributed KV store, not SQL)
-- Key: list_id:identifier_hash
-- Value: 1 (or TTL for expiring audiences)

-- Segments (logical combinations of lists)
CREATE TABLE segments (
    id UUID PRIMARY KEY,
    advertiser_id UUID NOT NULL,
    name VARCHAR(255) NOT NULL,
    definition JSONB NOT NULL,
    /*
    definition example:
    {
        "operation": "AND",
        "operands": [
            {"type": "list", "list_id": "uuid1"},
            {"operation": "OR", "operands": [
                {"type": "list", "list_id": "uuid2"},
                {"type": "list", "list_id": "uuid3"}
            ]},
            {"operation": "NOT", "operands": [
                {"type": "list", "list_id": "uuid4"}
            ]}
        ]
    }
    */
    estimated_size BIGINT,
    status VARCHAR(20) DEFAULT 'active',
    created_at TIMESTAMPTZ DEFAULT NOW()
);
```

### Upload Processing Pipeline

```python
class AudienceUploadService:
    def process_upload(self, file_path: str, list_id: str):
        """Process large audience file upload."""

        # 1. Stream file from S3
        for batch in self.stream_file_batches(file_path, batch_size=100000):

            # 2. Hash identifiers (if not pre-hashed)
            hashed = [self.hash_identifier(id) for id in batch]

            # 3. Write to distributed store
            self.audience_store.batch_add(list_id, hashed)

            # 4. Update Bloom filter for fast negative lookups
            self.bloom_filter.add_batch(list_id, hashed)

        # 5. Build Roaring bitmap for set operations
        self.bitmap_store.build_bitmap(list_id)

        # 6. Update metadata
        self.db.update_list_status(list_id, 'ready')
```

### Real-Time Membership Lookup

```python
class AudienceLookupService:
    def is_member(self, user_id: str, segment_id: str) -> bool:
        """Check if user is in segment (< 10ms)."""

        # 1. Check Bloom filter first (fast negative)
        segment = self.get_segment(segment_id)

        if not self.bloom_check(user_id, segment):
            return False  # Definitely not a member

        # 2. Evaluate segment definition
        return self.evaluate_segment(user_id, segment.definition)

    def evaluate_segment(self, user_id: str, definition: dict) -> bool:
        """Recursively evaluate segment definition."""

        op = definition.get('operation')

        if definition.get('type') == 'list':
            # Direct list membership check
            return self.check_list_membership(user_id, definition['list_id'])

        if op == 'AND':
            return all(self.evaluate_segment(user_id, op)
                      for op in definition['operands'])
        elif op == 'OR':
            return any(self.evaluate_segment(user_id, op)
                      for op in definition['operands'])
        elif op == 'NOT':
            return not self.evaluate_segment(user_id, definition['operands'][0])
```

### Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Identifier Storage | Roaring Bitmaps | 10-100x compression, fast set operations |
| Fast Negative Check | Bloom Filters | O(1) lookup, 1% false positive acceptable |
| Segment Evaluation | Runtime evaluation | Flexibility over pre-materialization |
| ID Matching | Hash-based | Privacy-preserving, deterministic |

### Audience Size Estimation

```python
def estimate_segment_size(self, segment_definition: dict) -> int:
    """Estimate audience size without full evaluation."""

    # Use HyperLogLog for cardinality estimation
    # Combine HLL sketches based on set operations

    if segment_definition.get('type') == 'list':
        return self.hll_store.get_cardinality(segment_definition['list_id'])

    op = segment_definition['operation']
    operand_sizes = [self.estimate_segment_size(op)
                    for op in segment_definition['operands']]

    if op == 'AND':
        # Conservative estimate: min of operands
        return min(operand_sizes)
    elif op == 'OR':
        # Use inclusion-exclusion principle with HLL merge
        return self.hll_union_cardinality(segment_definition['operands'])
    elif op == 'NOT':
        # Total users - excluded
        return self.total_users - operand_sizes[0]
```

### Interview Discussion Points

1. **How to handle identity resolution across devices?**
   - Build identity graph linking email, device IDs, logged-in user IDs
   - Probabilistic matching for cross-device

2. **How to ensure privacy compliance (GDPR)?**
   - All identifiers hashed before storage
   - Support for deletion requests (tombstone + periodic cleanup)

3. **How to handle audience freshness?**
   - TTL on list memberships for time-sensitive audiences
   - Daily refresh from advertiser data pipelines

---

## 5. Design a Billing System for 300M Subscribers

**Type:** System Design

**Problem:** Architect a billing infrastructure for Netflix capable of processing monthly subscription charges across 300 million users.

### Functional Requirements

1. Process monthly recurring charges for all subscribers
2. Support multiple payment methods (credit card, PayPal, gift cards)
3. Handle plan changes (upgrades, downgrades) mid-cycle
4. Retry failed payments with exponential backoff
5. Generate invoices and payment receipts
6. Support multiple currencies and tax calculations

### Non-Functional Requirements

- **Scale:** 300M subscribers, ~10M transactions/day
- **Reliability:** 99.99% uptime, no double charges
- **Consistency:** Exactly-once payment processing
- **Compliance:** PCI-DSS for payment data

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     Billing Orchestrator                        │
│              (Schedules and coordinates billing)                │
└────────────────────────────┬────────────────────────────────────┘
                             │
        ┌────────────────────┼────────────────────┐
        │                    │                    │
┌───────▼───────┐   ┌────────▼────────┐  ┌────────▼────────┐
│ Subscription  │   │    Invoice      │  │    Payment      │
│   Service     │   │    Service      │  │    Service      │
└───────┬───────┘   └────────┬────────┘  └────────┬────────┘
        │                    │                    │
        │                    │                    │
┌───────▼───────┐   ┌────────▼────────┐  ┌────────▼────────┐
│  PostgreSQL   │   │   PostgreSQL    │  │   Payment       │
│ (Subscriptions)│  │   (Invoices)    │  │   Gateway       │
└───────────────┘   └─────────────────┘  │   Abstraction   │
                                         └────────┬────────┘
                                                  │
                           ┌──────────────────────┼──────────────┐
                           │                      │              │
                    ┌──────▼──────┐        ┌──────▼──────┐ ┌─────▼─────┐
                    │   Stripe    │        │   PayPal   │ │  Adyen    │
                    └─────────────┘        └────────────┘ └───────────┘
```

### Data Model

```sql
-- Subscriptions
CREATE TABLE subscriptions (
    id UUID PRIMARY KEY,
    user_id UUID NOT NULL UNIQUE,
    plan_id UUID NOT NULL,
    status VARCHAR(20) NOT NULL,  -- active, cancelled, past_due, suspended
    current_period_start TIMESTAMPTZ NOT NULL,
    current_period_end TIMESTAMPTZ NOT NULL,
    cancel_at_period_end BOOLEAN DEFAULT FALSE,
    payment_method_id UUID,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Plans
CREATE TABLE plans (
    id UUID PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    price_cents INTEGER NOT NULL,
    currency VARCHAR(3) NOT NULL,
    billing_interval VARCHAR(20) NOT NULL,  -- monthly, yearly
    features JSONB,
    active BOOLEAN DEFAULT TRUE
);

-- Invoices
CREATE TABLE invoices (
    id UUID PRIMARY KEY,
    subscription_id UUID REFERENCES subscriptions(id),
    user_id UUID NOT NULL,
    amount_cents INTEGER NOT NULL,
    currency VARCHAR(3) NOT NULL,
    tax_cents INTEGER DEFAULT 0,
    status VARCHAR(20) NOT NULL,  -- draft, open, paid, void, uncollectible
    due_date TIMESTAMPTZ NOT NULL,
    paid_at TIMESTAMPTZ,
    invoice_number VARCHAR(50) UNIQUE,
    line_items JSONB NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Payments
CREATE TABLE payments (
    id UUID PRIMARY KEY,
    invoice_id UUID REFERENCES invoices(id),
    amount_cents INTEGER NOT NULL,
    currency VARCHAR(3) NOT NULL,
    status VARCHAR(20) NOT NULL,  -- pending, succeeded, failed, refunded
    payment_method_type VARCHAR(50),
    gateway VARCHAR(50) NOT NULL,
    gateway_payment_id VARCHAR(255),
    failure_reason VARCHAR(500),
    attempt_number INTEGER DEFAULT 1,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Payment Methods (tokenized, no raw card data)
CREATE TABLE payment_methods (
    id UUID PRIMARY KEY,
    user_id UUID NOT NULL,
    type VARCHAR(50) NOT NULL,  -- card, paypal, bank_account
    gateway VARCHAR(50) NOT NULL,
    gateway_token VARCHAR(255) NOT NULL,  -- Tokenized reference
    last_four VARCHAR(4),
    expiry_month INTEGER,
    expiry_year INTEGER,
    is_default BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Idempotency Keys (prevent double charges)
CREATE TABLE idempotency_keys (
    key VARCHAR(255) PRIMARY KEY,
    request_hash VARCHAR(64),
    response JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
);
```

### Billing Orchestration

```python
class BillingOrchestrator:
    def run_daily_billing(self):
        """Process all subscriptions due for billing today."""

        # 1. Find subscriptions due for renewal
        due_subscriptions = self.db.query("""
            SELECT * FROM subscriptions
            WHERE status = 'active'
            AND current_period_end <= NOW()
            AND current_period_end > NOW() - INTERVAL '1 day'
            FOR UPDATE SKIP LOCKED  -- Parallel processing
        """)

        for subscription in due_subscriptions:
            self.process_subscription_billing(subscription)

    def process_subscription_billing(self, subscription):
        """Bill a single subscription with idempotency."""

        idempotency_key = f"billing:{subscription.id}:{subscription.current_period_end}"

        # Check idempotency
        if self.is_already_processed(idempotency_key):
            return

        try:
            # 1. Create invoice
            invoice = self.invoice_service.create_invoice(subscription)

            # 2. Attempt payment
            payment = self.payment_service.charge(
                invoice=invoice,
                payment_method=subscription.payment_method_id,
                idempotency_key=idempotency_key
            )

            if payment.status == 'succeeded':
                # 3. Extend subscription period
                self.extend_subscription(subscription)
                self.invoice_service.mark_paid(invoice)
            else:
                # 4. Schedule retry
                self.schedule_retry(subscription, payment)

        except Exception as e:
            self.alert_service.notify_billing_failure(subscription, e)
```

### Payment Retry Strategy

```python
class PaymentRetryService:
    RETRY_SCHEDULE = [
        (1, "day"),    # Retry after 1 day
        (3, "days"),   # Retry after 3 more days
        (7, "days"),   # Retry after 7 more days
        (14, "days"),  # Final retry after 14 more days
    ]

    def schedule_retry(self, subscription, failed_payment):
        attempt = failed_payment.attempt_number

        if attempt >= len(self.RETRY_SCHEDULE):
            # Max retries reached - suspend account
            self.subscription_service.suspend(subscription)
            self.notification_service.send_suspension_notice(subscription.user_id)
            return

        delay, unit = self.RETRY_SCHEDULE[attempt]
        retry_at = datetime.now() + timedelta(**{unit: delay})

        self.queue.schedule(
            task='retry_payment',
            payload={'subscription_id': subscription.id},
            execute_at=retry_at
        )

        # Notify user of failed payment
        self.notification_service.send_payment_failed(
            subscription.user_id,
            retry_at=retry_at
        )
```

### Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Idempotency | Idempotency keys table | Prevent double charges on retries |
| Payment Tokens | Gateway tokenization | PCI compliance, no card data stored |
| Billing Schedule | Daily batch + on-demand | Balance load, handle timezone complexity |
| Database | PostgreSQL with partitioning | ACID for financial data, partition by date |
| Queue | Kafka/SQS | Reliable retry scheduling, at-least-once delivery |

### Handling Plan Changes

```python
def change_plan(self, subscription_id: str, new_plan_id: str):
    """Handle mid-cycle plan changes with proration."""

    subscription = self.get_subscription(subscription_id)
    old_plan = self.get_plan(subscription.plan_id)
    new_plan = self.get_plan(new_plan_id)

    # Calculate proration
    days_remaining = (subscription.current_period_end - datetime.now()).days
    total_days = (subscription.current_period_end - subscription.current_period_start).days

    proration_factor = days_remaining / total_days

    credit = old_plan.price_cents * proration_factor
    charge = new_plan.price_cents * proration_factor
    net_amount = charge - credit

    if net_amount > 0:
        # Charge the difference
        self.charge_proration(subscription, net_amount)
    else:
        # Credit to next invoice
        self.apply_credit(subscription, abs(net_amount))

    subscription.plan_id = new_plan_id
    self.db.save(subscription)
```

### Interview Discussion Points

1. **How to handle timezone complexity for billing dates?**
   - Store all times in UTC, bill based on account creation timezone

2. **How to ensure no duplicate charges during system failures?**
   - Idempotency keys with TTL, gateway-level idempotency

3. **How to handle chargebacks?**
   - Webhook handlers for gateway chargeback notifications
   - Automatic account suspension, dispute management workflow

---

## 6. Design an Ads Frequency Cap System

**Type:** System Design

**Problem:** Design a system that restricts ad exposure by limiting how many times a specific ad (or category of ads) is shown to a user within a time window.

### Functional Requirements

1. Limit impressions per user per ad/campaign/advertiser
2. Support multiple time windows (per hour, per day, per week)
3. Real-time enforcement during ad serving (< 5ms overhead)
4. Support for hierarchical caps (ad < campaign < advertiser)
5. Near real-time cap updates when advertisers change settings

### Non-Functional Requirements

- **Scale:** 200M users, 10K campaigns, 1M QPS ad requests
- **Latency:** < 5ms for cap check
- **Consistency:** Eventual consistency acceptable (slight over-delivery OK)
- **Availability:** 99.99% uptime

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      Ad Serving Request                         │
└────────────────────────────┬────────────────────────────────────┘
                             │
                    ┌────────▼────────┐
                    │   Ad Server     │
                    │                 │
                    │  ┌───────────┐  │
                    │  │ Cap Check │◄─┼──── Before selecting ad
                    │  └─────┬─────┘  │
                    │        │        │
                    │  ┌─────▼─────┐  │
                    │  │Cap Update │◄─┼──── After impression served
                    │  └───────────┘  │
                    └────────┬────────┘
                             │
              ┌──────────────┼──────────────┐
              │              │              │
       ┌──────▼──────┐ ┌─────▼─────┐ ┌──────▼──────┐
       │   Local     │ │   Redis   │ │   Kafka     │
       │   Cache     │ │  Cluster  │ │  (Events)   │
       │ (per-server)│ │  (Caps)   │ │             │
       └─────────────┘ └───────────┘ └──────┬──────┘
                                            │
                                     ┌──────▼──────┐
                                     │   Flink     │
                                     │ Aggregator  │
                                     └─────────────┘
```

### Data Model

```python
# Cap Configuration (stored in database, cached in Redis)
cap_config = {
    "ad_id": "ad123",
    "campaign_id": "camp456",
    "advertiser_id": "adv789",
    "caps": [
        {"type": "ad", "entity_id": "ad123", "limit": 3, "window": "day"},
        {"type": "campaign", "entity_id": "camp456", "limit": 10, "window": "day"},
        {"type": "advertiser", "entity_id": "adv789", "limit": 20, "window": "week"}
    ]
}

# User Cap Counter (stored in Redis)
# Key format: cap:{user_id}:{entity_type}:{entity_id}:{window_id}
# Example: cap:user123:ad:ad456:2024-01-15
# Value: integer count
```

### Real-Time Cap Checking

```python
class FrequencyCapService:
    def __init__(self, redis_client):
        self.redis = redis_client
        self.local_cache = TTLCache(maxsize=10000, ttl=60)

    def can_serve_ad(self, user_id: str, ad: Ad) -> bool:
        """Check if ad can be served without exceeding caps."""

        caps = self.get_caps(ad)

        for cap in caps:
            key = self.build_key(user_id, cap)
            current_count = self.get_count(key)

            if current_count >= cap.limit:
                return False

        return True

    def build_key(self, user_id: str, cap: Cap) -> str:
        window_id = self.get_window_id(cap.window)
        return f"cap:{user_id}:{cap.type}:{cap.entity_id}:{window_id}"

    def get_window_id(self, window: str) -> str:
        now = datetime.utcnow()
        if window == "hour":
            return now.strftime("%Y-%m-%d-%H")
        elif window == "day":
            return now.strftime("%Y-%m-%d")
        elif window == "week":
            return now.strftime("%Y-W%W")

    def record_impression(self, user_id: str, ad: Ad):
        """Increment counters after serving ad."""
        caps = self.get_caps(ad)

        pipe = self.redis.pipeline()
        for cap in caps:
            key = self.build_key(user_id, cap)
            ttl = self.get_ttl(cap.window)
            pipe.incr(key)
            pipe.expire(key, ttl)
        pipe.execute()
```

### Optimizations for Scale

**1. Local Caching**
```python
class OptimizedCapService:
    def can_serve_ad(self, user_id: str, ad: Ad) -> bool:
        # Check local cache first (probabilistic early exit)
        local_key = f"{user_id}:{ad.id}"

        if self.local_cache.get(local_key, 0) >= ad.cap_limit:
            return False  # Likely capped, skip Redis call

        # Full check in Redis
        return self.redis_check(user_id, ad)
```

**2. Batch Counter Updates**
```python
class BatchedCapUpdater:
    def __init__(self):
        self.pending_updates = defaultdict(int)
        self.flush_interval = 1  # second

    def record_impression(self, user_id: str, ad: Ad):
        key = self.build_key(user_id, ad)
        self.pending_updates[key] += 1

    async def flush_loop(self):
        while True:
            await asyncio.sleep(self.flush_interval)
            updates = self.pending_updates.copy()
            self.pending_updates.clear()

            pipe = self.redis.pipeline()
            for key, count in updates.items():
                pipe.incrby(key, count)
            pipe.execute()
```

### Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Storage | Redis Cluster | Low latency, atomic increments, TTL support |
| Counter Strategy | Per-window keys | Automatic expiration, no cleanup needed |
| Consistency | Eventual | Slight over-delivery acceptable vs latency |
| Local Cache | Yes, short TTL | Reduce Redis calls for hot users |

### Handling Edge Cases

**Window Boundary**: When a window ends, new key is used automatically
**Over-delivery**: Accept small margin (1-2%) for performance
**Redis Failure**: Graceful degradation - serve ads, log for reconciliation

### Interview Discussion Points

1. **How to handle cross-datacenter consistency?**
   - Use local Redis per DC, async sync, accept small over-delivery

2. **How to support "lifetime" caps?**
   - Separate storage (database), check less frequently (cacheable)

3. **How to handle cap changes mid-campaign?**
   - Version cap configs, compare against user's impression timestamp

---

## 7. ML Job Scheduler

**Type:** System Design

**Problem:** Design a distributed job scheduler for ML workloads at Netflix that handles model training, batch inference, and feature engineering pipelines.

### Functional Requirements

1. Schedule recurring jobs (hourly, daily, weekly)
2. Handle job dependencies (DAG execution)
3. Support heterogeneous resources (CPU, GPU, memory)
4. Retry failed jobs with backoff
5. Real-time monitoring and alerting
6. Priority-based queue management

### Non-Functional Requirements

- **Scale:** 10K+ jobs/day, 1K concurrent jobs
- **Reliability:** No missed scheduled jobs, exactly-once execution
- **Latency:** Job start within 30 seconds of scheduled time
- **Resource Efficiency:** Maximize cluster utilization

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      User Interface                             │
│            (Job submission, monitoring, logs)                   │
└────────────────────────────┬────────────────────────────────────┘
                             │
                    ┌────────▼────────┐
                    │   API Server    │
                    │  (Job CRUD)     │
                    └────────┬────────┘
                             │
        ┌────────────────────┼────────────────────┐
        │                    │                    │
┌───────▼───────┐   ┌────────▼────────┐  ┌────────▼────────┐
│   Scheduler   │   │   Job Queue     │  │   Metadata      │
│   (Leader)    │   │   (Priority)    │  │   Store         │
└───────┬───────┘   └────────┬────────┘  └─────────────────┘
        │                    │
        │         ┌──────────┴──────────┐
        │         │                     │
        │    ┌────▼────┐          ┌─────▼────┐
        │    │ Worker  │          │  Worker  │
        │    │ Pool 1  │          │  Pool 2  │
        │    │ (GPU)   │          │  (CPU)   │
        │    └────┬────┘          └────┬─────┘
        │         │                    │
        │    ┌────▼────────────────────▼────┐
        │    │      Kubernetes Cluster      │
        │    │   (Resource orchestration)   │
        │    └──────────────────────────────┘
        │
┌───────▼───────┐
│   Executor    │
│   Monitor     │
│ (Health/Logs) │
└───────────────┘
```

### Data Model

```sql
-- Job Definitions
CREATE TABLE job_definitions (
    id UUID PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    owner_team VARCHAR(100),
    schedule_cron VARCHAR(100),  -- NULL for one-time jobs
    job_type VARCHAR(50) NOT NULL,  -- training, inference, etl
    config JSONB NOT NULL,
    /*
    config example:
    {
        "image": "ml-training:v1.2",
        "command": ["python", "train.py"],
        "resources": {"cpu": 4, "memory": "16Gi", "gpu": 1},
        "timeout_minutes": 120,
        "retry_policy": {"max_retries": 3, "backoff": "exponential"}
    }
    */
    dependencies JSONB,  -- List of job_ids that must complete first
    priority INTEGER DEFAULT 5,
    enabled BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Job Runs (instances of job executions)
CREATE TABLE job_runs (
    id UUID PRIMARY KEY,
    job_definition_id UUID REFERENCES job_definitions(id),
    scheduled_time TIMESTAMPTZ NOT NULL,
    start_time TIMESTAMPTZ,
    end_time TIMESTAMPTZ,
    status VARCHAR(20) NOT NULL,  -- pending, queued, running, succeeded, failed, cancelled
    worker_id VARCHAR(100),
    attempt_number INTEGER DEFAULT 1,
    exit_code INTEGER,
    error_message TEXT,
    metrics JSONB,  -- runtime metrics, resource usage
    created_at TIMESTAMPTZ DEFAULT NOW(),

    INDEX idx_runs_status (status, scheduled_time),
    INDEX idx_runs_job (job_definition_id, scheduled_time)
);

-- DAG Dependencies
CREATE TABLE job_dependencies (
    job_id UUID REFERENCES job_definitions(id),
    depends_on_job_id UUID REFERENCES job_definitions(id),
    PRIMARY KEY (job_id, depends_on_job_id)
);
```

### Scheduler Implementation

```python
class MLJobScheduler:
    def __init__(self):
        self.leader_election = LeaderElection("scheduler")
        self.job_queue = PriorityQueue()

    async def run(self):
        """Main scheduler loop."""
        while True:
            if not self.leader_election.is_leader():
                await asyncio.sleep(1)
                continue

            # 1. Find jobs due for scheduling
            due_jobs = await self.find_due_jobs()

            for job in due_jobs:
                # 2. Check dependencies
                if await self.dependencies_satisfied(job):
                    # 3. Create job run and enqueue
                    job_run = await self.create_job_run(job)
                    await self.job_queue.enqueue(job_run)

            await asyncio.sleep(1)

    async def find_due_jobs(self) -> List[JobDefinition]:
        """Find jobs that should run based on cron schedule."""
        now = datetime.utcnow()

        return await self.db.query("""
            SELECT jd.* FROM job_definitions jd
            WHERE jd.enabled = TRUE
            AND jd.schedule_cron IS NOT NULL
            AND NOT EXISTS (
                SELECT 1 FROM job_runs jr
                WHERE jr.job_definition_id = jd.id
                AND jr.scheduled_time = %s
            )
            AND cron_matches(jd.schedule_cron, %s)
        """, [now, now])

    async def dependencies_satisfied(self, job: JobDefinition) -> bool:
        """Check if all upstream jobs have completed successfully."""
        if not job.dependencies:
            return True

        for dep_id in job.dependencies:
            latest_run = await self.get_latest_run(dep_id)
            if not latest_run or latest_run.status != 'succeeded':
                return False

        return True
```

### Worker Implementation

```python
class MLJobWorker:
    def __init__(self, worker_id: str, resource_type: str):
        self.worker_id = worker_id
        self.resource_type = resource_type  # gpu, cpu
        self.k8s_client = KubernetesClient()

    async def run(self):
        """Worker loop - pull and execute jobs."""
        while True:
            # 1. Pull job matching our resource type
            job_run = await self.job_queue.dequeue(
                resource_type=self.resource_type
            )

            if not job_run:
                await asyncio.sleep(1)
                continue

            # 2. Execute job
            await self.execute_job(job_run)

    async def execute_job(self, job_run: JobRun):
        """Execute job in Kubernetes."""
        try:
            # Update status
            job_run.status = 'running'
            job_run.start_time = datetime.utcnow()
            job_run.worker_id = self.worker_id
            await self.db.save(job_run)

            # Create K8s Job
            k8s_job = self.build_k8s_job(job_run)
            await self.k8s_client.create_job(k8s_job)

            # Wait for completion
            result = await self.wait_for_completion(k8s_job)

            # Update status
            job_run.status = 'succeeded' if result.success else 'failed'
            job_run.end_time = datetime.utcnow()
            job_run.exit_code = result.exit_code
            await self.db.save(job_run)

            # Handle failure - maybe retry
            if not result.success:
                await self.handle_failure(job_run)

        except Exception as e:
            job_run.status = 'failed'
            job_run.error_message = str(e)
            await self.db.save(job_run)
            await self.handle_failure(job_run)

    def build_k8s_job(self, job_run: JobRun) -> dict:
        config = job_run.job_definition.config
        return {
            "apiVersion": "batch/v1",
            "kind": "Job",
            "metadata": {"name": f"ml-job-{job_run.id}"},
            "spec": {
                "template": {
                    "spec": {
                        "containers": [{
                            "name": "ml-job",
                            "image": config["image"],
                            "command": config["command"],
                            "resources": {
                                "requests": config["resources"],
                                "limits": config["resources"]
                            }
                        }],
                        "restartPolicy": "Never"
                    }
                }
            }
        }
```

### Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Orchestration | Kubernetes | Mature, handles resource allocation, scaling |
| Queue | Redis + PostgreSQL | Redis for speed, PG for persistence |
| Leader Election | ZooKeeper/etcd | Prevent duplicate scheduling |
| Job Execution | K8s Jobs | Isolation, resource limits, logs |
| DAG Resolution | At schedule time | Simple, handles dynamic dependencies |

### Monitoring & Alerting

```python
class JobMonitor:
    def check_health(self):
        alerts = []

        # 1. Jobs stuck in running too long
        stuck_jobs = self.db.query("""
            SELECT * FROM job_runs
            WHERE status = 'running'
            AND start_time < NOW() - (config->>'timeout_minutes' || ' minutes')::interval
        """)
        for job in stuck_jobs:
            alerts.append(Alert(f"Job {job.id} exceeded timeout"))

        # 2. High failure rate
        failure_rate = self.calculate_failure_rate(window='1 hour')
        if failure_rate > 0.1:
            alerts.append(Alert(f"High failure rate: {failure_rate:.1%}"))

        # 3. Queue backlog
        queue_depth = self.job_queue.depth()
        if queue_depth > 1000:
            alerts.append(Alert(f"Large queue backlog: {queue_depth}"))

        return alerts
```

### Interview Discussion Points

1. **How to handle job priority inversion?**
   - Priority aging: boost priority of waiting jobs over time

2. **How to handle resource fragmentation?**
   - Bin packing algorithms, preemption for high-priority jobs

3. **How to ensure exactly-once execution?**
   - Idempotent jobs + database transactions for state changes

---

## 8. Design a WAL Log Enrichment Pipeline

**Type:** System Design

**Problem:** Build a system that captures Write-Ahead Log (WAL) entries from a source database, enriches them with supplementary context, and transmits them to a target database or data warehouse.

### Functional Requirements

1. Capture all DML changes (INSERT, UPDATE, DELETE) from source PostgreSQL
2. Enrich records with data from external services (user details, content metadata)
3. Transform to target schema and deliver to data warehouse
4. Support exactly-once delivery semantics
5. Handle schema evolution gracefully

### Non-Functional Requirements

- **Latency:** < 30 seconds from source commit to target availability
- **Throughput:** 100K changes/second peak
- **Durability:** No data loss even during failures
- **Ordering:** Maintain transaction ordering within each table

### High-Level Architecture

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  PostgreSQL │     │   Debezium  │     │    Kafka    │
│   Source    │────▶│  Connector  │────▶│  (Raw WAL)  │
│             │     │             │     │             │
└─────────────┘     └─────────────┘     └──────┬──────┘
                                               │
                         ┌─────────────────────┴───────────────┐
                         │                                     │
                  ┌──────▼──────┐                       ┌──────▼──────┐
                  │   Flink     │                       │   Schema    │
                  │ Enrichment  │                       │  Registry   │
                  │   Job       │                       │             │
                  └──────┬──────┘                       └─────────────┘
                         │
         ┌───────────────┼───────────────┐
         │               │               │
  ┌──────▼──────┐ ┌──────▼──────┐ ┌──────▼──────┐
  │  User       │ │  Content    │ │  External   │
  │  Service    │ │  Service    │ │  APIs       │
  │  (Lookup)   │ │  (Lookup)   │ │             │
  └─────────────┘ └─────────────┘ └─────────────┘
                         │
                  ┌──────▼──────┐
                  │   Kafka     │
                  │ (Enriched)  │
                  └──────┬──────┘
                         │
                  ┌──────▼──────┐
                  │  Snowflake/ │
                  │  BigQuery   │
                  │  Sink       │
                  └─────────────┘
```

### WAL Capture with Debezium

```yaml
# Debezium PostgreSQL Connector Configuration
{
  "name": "netflix-pg-source",
  "config": {
    "connector.class": "io.debezium.connector.postgresql.PostgresConnector",
    "database.hostname": "source-db.netflix.net",
    "database.port": "5432",
    "database.user": "replicator",
    "database.password": "${secrets:pg-password}",
    "database.dbname": "netflix_prod",
    "database.server.name": "netflix",
    "table.include.list": "public.subscriptions,public.viewing_history",
    "plugin.name": "pgoutput",
    "slot.name": "debezium_slot",
    "publication.name": "dbz_publication",
    "snapshot.mode": "initial",
    "tombstones.on.delete": "true",
    "key.converter": "io.confluent.connect.avro.AvroConverter",
    "value.converter": "io.confluent.connect.avro.AvroConverter"
  }
}
```

### WAL Event Structure

```json
{
  "before": {
    "user_id": "u123",
    "plan_id": "basic",
    "status": "active"
  },
  "after": {
    "user_id": "u123",
    "plan_id": "premium",
    "status": "active"
  },
  "source": {
    "version": "2.4.0",
    "connector": "postgresql",
    "name": "netflix",
    "ts_ms": 1705334400000,
    "db": "netflix_prod",
    "schema": "public",
    "table": "subscriptions",
    "txId": 12345,
    "lsn": 23456789
  },
  "op": "u",
  "ts_ms": 1705334400100
}
```

### Flink Enrichment Job

```python
# Flink SQL for enrichment pipeline
"""
-- Source: Raw WAL events from Kafka
CREATE TABLE raw_subscription_changes (
    before ROW<user_id STRING, plan_id STRING, status STRING>,
    after ROW<user_id STRING, plan_id STRING, status STRING>,
    source ROW<ts_ms BIGINT, txId BIGINT, lsn BIGINT>,
    op STRING,
    ts_ms BIGINT
) WITH (
    'connector' = 'kafka',
    'topic' = 'netflix.public.subscriptions',
    'format' = 'avro-confluent'
);

-- Lookup table: User details (cached, async lookup)
CREATE TABLE user_details (
    user_id STRING,
    email STRING,
    country STRING,
    signup_date DATE,
    PRIMARY KEY (user_id) NOT ENFORCED
) WITH (
    'connector' = 'jdbc',
    'url' = 'jdbc:postgresql://user-service:5432/users',
    'lookup.cache.max-rows' = '100000',
    'lookup.cache.ttl' = '1 hour'
);

-- Enriched output
CREATE TABLE enriched_subscription_changes (
    user_id STRING,
    email STRING,
    country STRING,
    old_plan STRING,
    new_plan STRING,
    change_type STRING,
    event_time TIMESTAMP(3),
    processing_time TIMESTAMP(3)
) WITH (
    'connector' = 'kafka',
    'topic' = 'enriched.subscription.changes',
    'format' = 'avro-confluent'
);

-- Enrichment query with temporal join
INSERT INTO enriched_subscription_changes
SELECT
    s.after.user_id,
    u.email,
    u.country,
    s.before.plan_id AS old_plan,
    s.after.plan_id AS new_plan,
    CASE s.op
        WHEN 'c' THEN 'CREATE'
        WHEN 'u' THEN 'UPDATE'
        WHEN 'd' THEN 'DELETE'
    END AS change_type,
    TO_TIMESTAMP_LTZ(s.source.ts_ms, 3) AS event_time,
    CURRENT_TIMESTAMP AS processing_time
FROM raw_subscription_changes s
LEFT JOIN user_details FOR SYSTEM_TIME AS OF s.proc_time AS u
ON s.after.user_id = u.user_id;
"""
```

### Exactly-Once Delivery

```python
class ExactlyOnceDelivery:
    """Implement exactly-once with Kafka transactions + idempotent writes."""

    def __init__(self):
        self.kafka_producer = KafkaProducer(
            transactional_id="enrichment-pipeline-1",
            enable_idempotence=True
        )
        self.checkpoint_store = RedisCheckpointStore()

    def process_batch(self, records: List[Record]):
        try:
            # Begin Kafka transaction
            self.kafka_producer.begin_transaction()

            for record in records:
                enriched = self.enrich(record)
                self.kafka_producer.send("enriched-topic", enriched)

            # Commit offsets and messages atomically
            self.kafka_producer.send_offsets_to_transaction(
                self.get_current_offsets()
            )
            self.kafka_producer.commit_transaction()

            # Update checkpoint
            self.checkpoint_store.save(records[-1].offset)

        except Exception as e:
            self.kafka_producer.abort_transaction()
            raise
```

### Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| CDC Tool | Debezium | Mature, supports PostgreSQL logical replication |
| Message Format | Avro + Schema Registry | Schema evolution, compact serialization |
| Stream Processor | Flink | Exactly-once, stateful processing, SQL support |
| Enrichment | Async lookup + cache | Avoid blocking on external service calls |
| Target | Kafka → Snowflake | Decouple enrichment from loading |

### Handling Schema Evolution

```python
class SchemaEvolutionHandler:
    def handle_schema_change(self, old_schema, new_schema):
        """Handle schema changes from source database."""

        # 1. Register new schema version
        new_version = self.schema_registry.register(new_schema)

        # 2. Validate compatibility
        if not self.is_backward_compatible(old_schema, new_schema):
            # Alert for manual intervention
            self.alert("Breaking schema change detected")
            return

        # 3. Update Flink job with new schema
        self.flink_job.update_schema(new_schema)

        # 4. Update target table DDL
        ddl = self.generate_alter_table(old_schema, new_schema)
        self.target_db.execute(ddl)
```

### Monitoring & Alerting

```python
metrics = {
    "replication_lag_seconds": Gauge("wal_replication_lag"),
    "events_processed_total": Counter("wal_events_processed"),
    "enrichment_failures": Counter("enrichment_failures"),
    "end_to_end_latency": Histogram("wal_e2e_latency_seconds")
}

# Alert rules
alerts = [
    Alert(
        name="High replication lag",
        condition="wal_replication_lag > 60",
        severity="critical"
    ),
    Alert(
        name="Enrichment failure spike",
        condition="rate(enrichment_failures[5m]) > 0.01",
        severity="warning"
    )
]
```

### Interview Discussion Points

1. **How to handle source database failover?**
   - Use replication slot on replica, auto-reconnect with slot position

2. **How to backfill historical data?**
   - Debezium snapshot mode, or separate batch ETL job

3. **How to handle enrichment service outages?**
   - Dead letter queue for failed enrichments, retry with backoff

---

## 9. Design the Data Model for a Promotion Posting System

**Type:** System Design (Data Modeling)

**Problem:** Create a database schema for Netflix's internal promotion/job posting system where employees can post and apply for internal roles.

### Functional Requirements

1. Managers can create job postings with requirements
2. Employees can view and apply to internal positions
3. Hiring managers can review applications and update status
4. Support for referrals and recommendations
5. Track application lifecycle (submitted → reviewed → interviewed → offered/rejected)

### Entity Relationship Diagram

```
┌─────────────┐       ┌─────────────┐       ┌─────────────┐
│  Employee   │       │   Job       │       │ Application │
├─────────────┤       │  Posting    │       ├─────────────┤
│ id (PK)     │       ├─────────────┤       │ id (PK)     │
│ name        │──1:N──│ id (PK)     │──1:N──│ posting_id  │
│ email       │       │ created_by  │       │ applicant_id│
│ department  │       │ title       │       │ status      │
│ level       │       │ department  │       │ resume_url  │
│ manager_id  │       │ level       │       │ created_at  │
└─────────────┘       │ status      │       └──────┬──────┘
      │               │ created_at  │              │
      │               └─────────────┘              │1:N
      │                     │                      │
      │               ┌─────▼─────┐         ┌──────▼──────┐
      │               │   Job     │         │  Interview  │
      │               │Requirements│        │   Round     │
      │               ├───────────┤         ├─────────────┤
      │               │ id (PK)   │         │ id (PK)     │
      │               │posting_id │         │application_id│
      │               │ skill     │         │interviewer_id│
      │               │ required  │         │ scheduled_at│
      │               └───────────┘         │ feedback    │
      │                                     │ decision    │
      │               ┌─────────────┐       └─────────────┘
      └──────────────▶│  Referral   │
                      ├─────────────┤
                      │ id (PK)     │
                      │ posting_id  │
                      │ referrer_id │
                      │ referee_id  │
                      │ status      │
                      └─────────────┘
```

### Database Schema

```sql
-- Employees (simplified, would join with HR system)
CREATE TABLE employees (
    id UUID PRIMARY KEY,
    email VARCHAR(255) UNIQUE NOT NULL,
    name VARCHAR(255) NOT NULL,
    department VARCHAR(100),
    level VARCHAR(50),  -- IC1, IC2, Manager, Director, VP
    manager_id UUID REFERENCES employees(id),
    hire_date DATE,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Job Postings
CREATE TABLE job_postings (
    id UUID PRIMARY KEY,
    title VARCHAR(255) NOT NULL,
    description TEXT NOT NULL,
    department VARCHAR(100) NOT NULL,
    team VARCHAR(100),
    level VARCHAR(50) NOT NULL,
    location VARCHAR(100) NOT NULL,
    remote_policy VARCHAR(50),  -- onsite, hybrid, remote
    headcount INTEGER DEFAULT 1,
    created_by UUID REFERENCES employees(id) NOT NULL,
    hiring_manager_id UUID REFERENCES employees(id) NOT NULL,
    status VARCHAR(20) DEFAULT 'draft',  -- draft, open, closed, filled, cancelled
    posted_at TIMESTAMPTZ,
    closes_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),

    INDEX idx_postings_status (status, department),
    INDEX idx_postings_manager (hiring_manager_id)
);

-- Job Requirements/Skills
CREATE TABLE job_requirements (
    id UUID PRIMARY KEY,
    posting_id UUID REFERENCES job_postings(id) ON DELETE CASCADE,
    skill VARCHAR(100) NOT NULL,
    proficiency_level VARCHAR(50),  -- beginner, intermediate, expert
    is_required BOOLEAN DEFAULT TRUE,
    years_experience INTEGER,

    UNIQUE(posting_id, skill)
);

-- Applications
CREATE TABLE applications (
    id UUID PRIMARY KEY,
    posting_id UUID REFERENCES job_postings(id) NOT NULL,
    applicant_id UUID REFERENCES employees(id) NOT NULL,
    status VARCHAR(30) DEFAULT 'submitted',
    /*
    Status flow:
    submitted → under_review → phone_screen → onsite →
    offer_pending → offer_extended → accepted/declined
    OR at any point: rejected, withdrawn
    */
    resume_url VARCHAR(500),
    cover_letter TEXT,
    referral_id UUID,  -- If came through referral
    current_stage VARCHAR(50),
    submitted_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),

    UNIQUE(posting_id, applicant_id),  -- One application per job
    INDEX idx_applications_status (posting_id, status)
);

-- Interview Rounds
CREATE TABLE interview_rounds (
    id UUID PRIMARY KEY,
    application_id UUID REFERENCES applications(id) NOT NULL,
    round_type VARCHAR(50) NOT NULL,  -- phone_screen, technical, behavioral, hiring_manager
    interviewer_id UUID REFERENCES employees(id) NOT NULL,
    scheduled_at TIMESTAMPTZ,
    duration_minutes INTEGER DEFAULT 60,
    meeting_link VARCHAR(500),
    status VARCHAR(20) DEFAULT 'scheduled',  -- scheduled, completed, cancelled, no_show
    feedback TEXT,
    decision VARCHAR(20),  -- strong_yes, yes, neutral, no, strong_no
    created_at TIMESTAMPTZ DEFAULT NOW(),

    INDEX idx_interviews_application (application_id),
    INDEX idx_interviews_interviewer (interviewer_id, scheduled_at)
);

-- Referrals
CREATE TABLE referrals (
    id UUID PRIMARY KEY,
    posting_id UUID REFERENCES job_postings(id) NOT NULL,
    referrer_id UUID REFERENCES employees(id) NOT NULL,  -- Employee making referral
    referee_email VARCHAR(255) NOT NULL,  -- Could be external
    referee_name VARCHAR(255) NOT NULL,
    relationship VARCHAR(100),  -- former_colleague, friend, etc.
    referral_note TEXT,
    status VARCHAR(20) DEFAULT 'pending',  -- pending, applied, hired, rejected
    bonus_eligible BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMPTZ DEFAULT NOW(),

    UNIQUE(posting_id, referee_email)
);

-- Application Notes/Comments
CREATE TABLE application_notes (
    id UUID PRIMARY KEY,
    application_id UUID REFERENCES applications(id) NOT NULL,
    author_id UUID REFERENCES employees(id) NOT NULL,
    note TEXT NOT NULL,
    is_private BOOLEAN DEFAULT FALSE,  -- Private = only visible to hiring team
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Audit Log
CREATE TABLE application_status_history (
    id BIGSERIAL PRIMARY KEY,
    application_id UUID REFERENCES applications(id) NOT NULL,
    old_status VARCHAR(30),
    new_status VARCHAR(30) NOT NULL,
    changed_by UUID REFERENCES employees(id),
    reason TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);
```

### Sample Queries

```sql
-- Get all open positions in Engineering
SELECT jp.*, e.name AS hiring_manager_name,
       COUNT(a.id) AS application_count
FROM job_postings jp
JOIN employees e ON jp.hiring_manager_id = e.id
LEFT JOIN applications a ON jp.id = a.posting_id
WHERE jp.status = 'open'
  AND jp.department = 'Engineering'
GROUP BY jp.id, e.name
ORDER BY jp.posted_at DESC;

-- Get application pipeline for a posting
SELECT status, COUNT(*) AS count
FROM applications
WHERE posting_id = 'uuid-here'
GROUP BY status
ORDER BY CASE status
    WHEN 'submitted' THEN 1
    WHEN 'under_review' THEN 2
    WHEN 'phone_screen' THEN 3
    WHEN 'onsite' THEN 4
    WHEN 'offer_pending' THEN 5
    ELSE 6
END;

-- Get interviewer's upcoming interviews
SELECT ir.*, a.id AS application_id, e.name AS candidate_name,
       jp.title AS position_title
FROM interview_rounds ir
JOIN applications a ON ir.application_id = a.id
JOIN employees e ON a.applicant_id = e.id
JOIN job_postings jp ON a.posting_id = jp.id
WHERE ir.interviewer_id = 'interviewer-uuid'
  AND ir.scheduled_at > NOW()
  AND ir.status = 'scheduled'
ORDER BY ir.scheduled_at;

-- Referral conversion rate by department
SELECT
    jp.department,
    COUNT(DISTINCT r.id) AS total_referrals,
    COUNT(DISTINCT CASE WHEN r.status = 'hired' THEN r.id END) AS hired,
    ROUND(100.0 * COUNT(DISTINCT CASE WHEN r.status = 'hired' THEN r.id END) /
          NULLIF(COUNT(DISTINCT r.id), 0), 2) AS conversion_rate
FROM referrals r
JOIN job_postings jp ON r.posting_id = jp.id
GROUP BY jp.department;
```

### Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Application Status | Enum values | Clear state machine, easy reporting |
| Resume Storage | URL reference | Store files in S3, just track URLs |
| Referral Model | Separate table | Track referral source separately from application |
| Audit Trail | Status history table | Compliance, debugging, analytics |
| Soft Delete | is_active flag | Preserve historical data |

### API Design

```python
# Core API endpoints
POST   /api/postings                    # Create job posting
GET    /api/postings                    # List postings (with filters)
GET    /api/postings/{id}               # Get posting details
PATCH  /api/postings/{id}               # Update posting
POST   /api/postings/{id}/publish       # Publish draft posting

POST   /api/postings/{id}/applications  # Submit application
GET    /api/applications/{id}           # Get application details
PATCH  /api/applications/{id}/status    # Update application status

POST   /api/applications/{id}/interviews    # Schedule interview
PATCH  /api/interviews/{id}/feedback        # Submit feedback

POST   /api/referrals                   # Create referral
GET    /api/employees/{id}/referrals    # Get employee's referrals
```

### Interview Discussion Points

1. **How to handle confidentiality?**
   - Role-based access control (RBAC)
   - Hiring team vs general employee visibility
   - Anonymous feedback options

2. **How to prevent bias in hiring?**
   - Structured interview scorecards
   - Blind resume review option
   - Diversity metrics dashboard

3. **How to scale for large organizations?**
   - Partition by department/region
   - Read replicas for reporting
   - Caching for job listings

---

# Anthropic

## 10. Design a 1-to-1 Chat System

**Type:** System Design

**Problem:** Design a chat system that supports only 1-to-1 messaging between users.

### Structure (5 Phases)

- **Phase 1:** Define the Goals (~5 minutes)
- **Phase 2:** Database Schema & Entities (~5 minutes)
- **Phase 3:** How Client and Server Talk (~5 minutes)
- **Phase 4:** System Architecture (~15-25 minutes)
- **Phase 5:** Handling Scale & Challenges (~15-20 minutes)

### Key Areas

- Functional and non-functional requirements
- Data model design
- Real-time communication protocols
- Scalability considerations
- Edge cases and challenges

---

## 11. Inference API System Design

**Type:** System Design

**Problem:** Design a high-concurrency inference API system that can handle massive concurrent requests efficiently.

### Structure

- **Step 1:** Defining the Scope
- **Step 2:** Estimating Scale and Capacity
- **Step 3:** Designing the API
- **Step 4:** Database and Data Structure
- **Step 5:** System Overview
- **Step 6:** Deep Dive into Key Components
- **Step 7:** Finding and Fixing Weak Spots

### Additional Sections

- Problem Requirements
- Sample Solution
- Extra Discussion Points
- Mistakes to Avoid
- How to Pass the Interview
- Practice Questions
- Study Materials

---

## 12. Prompt Playground System Design

**Type:** System Design

**Problem:** Design a prompt engineering playground similar to ChatGPT Playground or Anthropic Console.

### Structure

- **Step 1:** Defining the Requirements
- **Step 2:** Estimating Scale and Costs
- **Step 3:** API Definition
- **Step 4:** Database Schema Design
- **Step 5:** Architecture Overview
- **Step 6:** Deep Dive into Key Components
- **Step 7:** Fixing Performance Issues

### Additional Sections

- The Design Problem
- Additional Design Details
- Comparing Different Approaches
- Interview Advice

---

## 13. Distributed Model Deployment System Design

**Type:** System Design

**Problem:** Design a system that efficiently downloads and distributes a large ML model (e.g., 500GB) from external storage to all GPU workers in a data center cluster.

### Structure

- The Challenge
- Proposed Solution
- Understanding the Requirements
- Mathematical analysis
- API Design
- Data Structure Design
- System Architecture
- Deep Dive analysis
- Bottleneck identification and fixes
- Interview Tips

---

## 14. LLM Request Batching API System Design

**Type:** System Design

**Problem:** Design an HTTP API that exposes a batch processing function for large language model inference.

### Structure

- **The Challenge:** Problem introduction
- **Sample Solution:** Reference implementation
- **Step 1:** Clarifying the Requirements
- **Step 2:** Estimating Scale and Resources
- **Step 3:** API Design
- **Step 4:** Data Storage
- **Step 5:** Basic System Architecture
- **Step 6:** Deep Dive into Components
- **Step 7:** Fixing Potential Problems

---

# Databricks

## 15. Throttling System Design

**Type:** System Design

**Problem:** Design a throttling system that protects both:
- Incoming traffic at the gateway
- Outgoing traffic from API servers to databases, internal services, and third-party APIs

The goal is to prevent overload and cascading failure across the stack.

### Clarifying Questions

- Is the traffic steady or spiky?
- How much is internal vs external?
- What are the downstream capacity limits?
- Should internal traffic get priority?
- Do we need per-user, per-tier, or per-endpoint limits?
- Can limits change dynamically without restart?
- What should clients see when throttled: `429`, queueing, or degraded service?

### Requirements

**Functional Requirements:**
- Enforce incoming rate limits at the gateway
- Enforce outgoing limits toward each dependency
- Support different quotas for free, paid, and internal users
- Isolate tenants so one noisy client cannot starve others
- Degrade gracefully with `429` or fallback responses
- Support dynamic configuration updates

**Non-Functional Requirements:**
- Low latency, ideally under `5 ms` per check
- High availability even if the limiter datastore degrades
- Near-accurate enforcement, not necessarily perfect
- Strong observability and alerting
- Fairness under bursty traffic

### High-Level Design

```text
Client
  -> Gateway / Load Balancer
     -> Incoming Rate Limiter
        -> API Servers
           -> Outgoing Proxy / Sidecar
              -> DB / Internal APIs / 3P APIs

Config Service -> pushes limits to Gateway + Proxy fleet
Redis / local cache -> token state
Metrics pipeline -> Prometheus / dashboards / alerts
```

### Incoming Throttling

**Recommended algorithm: Token Bucket**
- Each principal gets tokens refilled at a steady rate
- Each request consumes a token
- Bursts are allowed up to bucket capacity
- When empty, reject with `429 Too Many Requests`

**Why Token Bucket:**
- Simple
- Supports bursts better than fixed window
- Predictable and cheap to implement

**Enforcement order:**
1. Global system limit
2. Tenant or tier limit
3. User or API key limit
4. Endpoint-specific limit

Reject on the first violated limit.

**Distributed setup:**
- Gateways share bucket state through Redis
- Use Lua scripts for atomic refill-plus-consume
- Maintain a short-lived local cache for ultra-hot keys

### Outgoing Throttling

For downstream dependencies, the limiter should sit in an on-host proxy or sidecar near the API server.

**Controls to combine:**
- Per-dependency token buckets
- Concurrency limits for expensive calls
- Circuit breakers
- Bounded retries with exponential backoff and jitter

**Circuit breaker states:**
- `CLOSED`: Requests flow normally
- `OPEN`: Fail fast because dependency is unhealthy
- `HALF_OPEN`: Allow limited probes to test recovery

This prevents one failing database or third-party service from taking the entire system down.

### Data Model

**ThrottleConfig**
- Principal id
- Tier
- Refill rate
- Burst capacity
- Per-endpoint overrides
- Dependency-specific outgoing limits

**RateLimitState**
- Tokens remaining
- Last refill timestamp
- Optional rolling error counters

**CircuitBreakerState**
- Dependency name
- State
- Recent failure count
- Probe window
- Last state transition time

### Storage Choices

- Config of record: PostgreSQL or another durable store
- Fast counters: Redis
- Local hot cache: in-memory per gateway / sidecar
- Metrics: Prometheus or another time-series system

### Dynamic Configuration

Limits should update without restart.

**Approach:**
- Admins write config to the config service
- Config service persists to durable storage
- Publish change events over pub/sub
- Gateways and sidecars subscribe and refresh local caches

This supports near-real-time rollout of new limits.

### Failure Handling

**Redis unavailable**
- Fall back to local best-effort limits
- Prefer partial protection over no protection
- Alert aggressively because enforcement accuracy is degraded

**Hot key / noisy tenant**
- Shard counters
- Use local admission checks before Redis
- Enforce stricter tenant-level caps

**Thundering herd**
- Include jitter in `Retry-After`
- Spread retries with randomized backoff
- Optionally use client-visible quota headers

### Observability

Track:
- Allowed vs throttled requests
- Bucket exhaustion rate
- Per-tenant reject rate
- Downstream error rates and latency
- Open circuit count
- Retry volume
- Queue depth and concurrency saturation

Alert on:
- Sudden spike in throttles
- Dependency saturation
- Rising tail latency
- Fallback mode activation

### Tradeoffs and Interview Discussion

**Token Bucket vs Sliding Window**
- Token bucket is better for controlled bursts
- Sliding window is more exact but usually more expensive

**Incoming vs Outgoing**
- Incoming protects your fleet from clients
- Outgoing protects dependencies from your own fleet
- You generally need both

**Accuracy vs availability**
- A globally consistent limiter is slower and more fragile
- A slightly approximate limiter with local fallback is usually the better operational choice

### Summary

Use token buckets at the gateway for user-facing rate limits, plus sidecar or proxy-based throttling and circuit breakers for downstream calls. Store fast-changing counters in Redis with local caching, keep configs in a durable config service, push updates dynamically, and design the system to fail gracefully with `429`, jittered retries, and observability around saturation and cascading failure.

---

# Airbnb

## 16. Data Analysis System Design

**Type:** System Design

**Problem:** Design a data analysis system emphasizing high interactivity.

### Approach

1. **Initial Design Phase**: Start with a high-level architecture using traditional Spark processing, while recognizing that this approach may introduce latency issues that don't satisfy requirements.

2. **Advanced Phase**: Explore stream processing solutions, specifically Flink aggregation, to address latency concerns.

### Key Technical Areas

- Integration between Flink and Kafka
- Recovery mechanisms when Flink experiences failures

### Conceptual Implementation

```python
"""
This is a system design problem. Below is a conceptual implementation
demonstrating the key components.
"""

from abc import ABC, abstractmethod
from typing import Any
from collections import defaultdict
import time

class DataProcessor(ABC):
    @abstractmethod
    def process(self, data: Any) -> Any:
        pass

class KafkaConsumer:
    """Simulated Kafka consumer for stream processing"""
    def __init__(self, topic: str, bootstrap_servers: list[str]):
        self.topic = topic
        self.servers = bootstrap_servers
        self.offset = 0

    def consume(self):
        # Simulated message consumption
        pass

    def commit_offset(self, offset: int):
        self.offset = offset

class FlinkAggregator(DataProcessor):
    """Stream processing with Flink-like aggregation"""
    def __init__(self):
        self.state = defaultdict(int)
        self.checkpoints = []

    def process(self, data: dict) -> dict:
        key = data.get('key')
        value = data.get('value', 0)
        self.state[key] += value
        return {'key': key, 'aggregated': self.state[key]}

    def checkpoint(self):
        """Save state for recovery"""
        self.checkpoints.append({
            'state': dict(self.state),
            'timestamp': time.time()
        })

    def recover(self):
        """Recover from last checkpoint"""
        if self.checkpoints:
            last = self.checkpoints[-1]
            self.state = defaultdict(int, last['state'])

class DataAnalysisSystem:
    """
    High-level architecture:
    1. Kafka for data ingestion
    2. Flink for real-time aggregation
    3. Checkpointing for fault tolerance
    """
    def __init__(self):
        self.consumer = KafkaConsumer("events", ["localhost:9092"])
        self.processor = FlinkAggregator()
        self.results_cache = {}

    def run(self):
        """Main processing loop"""
        while True:
            # Consume from Kafka
            events = self.consumer.consume()

            # Process with Flink
            for event in events or []:
                result = self.processor.process(event)
                self.results_cache[result['key']] = result

            # Periodic checkpointing
            self.processor.checkpoint()
            self.consumer.commit_offset(self.consumer.offset)

    def query(self, key: str) -> dict:
        """Low-latency query interface"""
        return self.results_cache.get(key, {})
```

---

## Summary Statistics

| Company | Count | Topics |
|---------|-------|--------|
| Netflix | 9 | Sentiment Analysis, Ad Tech, Recommendations, Billing, ML Infrastructure, Data Pipelines |
| Anthropic | 5 | Chat Systems, LLM Inference, Model Deployment, Batch Processing |
| Databricks | 1 | Rate Limiting & Throttling |
| Airbnb | 1 | Real-time Data Analysis |

**Total: 16 System Design Questions**
