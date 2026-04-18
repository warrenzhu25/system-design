# Netflix Interview Questions

Source: https://www.1point3acres.com/interview/problems/company/netflix

> **Note:** Full problem details require login to 1point3acres.com. This document contains publicly available summaries and structure.

---

## Table of Contents

### System Design (8)
1. [Netflix Sentiment Tracking](#2-netflix-sentiment-tracking)
2. [Design the Data Model for an Ads Demand Platform](#3-design-the-data-model-for-an-ads-demand-platform)
3. [Home Page Video Recommendation](#5-home-page-video-recommendation)
4. [Design an Ads Audience Targeting System](#11-design-an-ads-audience-targeting-system)
5. [Design a Billing System for 300M Subscribers](#12-design-a-billing-system-for-300m-subscribers)
6. [Design an Ads Frequency Cap System](#18-design-an-ads-frequency-cap-system)
7. [ML Job Scheduler](#19-ml-job-scheduler)
8. [Design a WAL Log Enrichment Pipeline](#21-design-a-wal-log-enrichment-pipeline)
9. [Design the Data Model for a Promotion Posting System](#26-design-the-data-model-for-a-promotion-posting-system)

### Coding - Arrays & Strings (12)
10. [Contains Duplicate](#25-contains-duplicate)
11. [Contains Duplicate II](#6-contains-duplicate-ii)
12. [Contains Duplicate III](#14-contains-duplicate-iii)
13. [First Missing Positive](#30-first-missing-positive)
14. [Longest Palindromic Substring](#37-longest-palindromic-substring)
15. [Longest Substring Without Repeating Characters](#15-longest-substring-without-repeating-characters)
16. [Maximum Subarray](#35-maximum-subarray)
17. [Median of Two Sorted Arrays](#34-median-of-two-sorted-arrays)
18. [Meeting Rooms](#17-meeting-rooms)
19. [Number Pairs That Match Target](#29-number-pairs-that-match-target)
20. [Sort by User Preference](#13-sort-by-user-preference)
21. [User Engagement Patterns](#27-user-engagement-patterns)

### Coding - Graph & Search (6)
22. [Number of Islands](#36-number-of-islands)
23. [Parallel Courses](#4-parallel-courses)
24. [Parallel Courses II](#31-parallel-courses-ii)
25. [Reconstruct Itinerary](#23-reconstruct-itinerary)
26. [Movie History Friends](#7-movie-history-friends)
27. [Movie History Friends II](#22-movie-history-friends-ii)

### Coding - Data Structures & Cache (6)
28. [LRU Cache](#32-lru-cache)
29. [Merge K Sorted Lists](#33-merge-k-sorted-lists)
30. [Versioned File System](#8-versioned-file-system)
31. [Auto-Expire Cache](#24-auto-expire-cache)
32. [Music Playlist](#9-music-playlist)
33. [Homepage Title Deduplication](#1-homepage-title-deduplication)

### Coding - Concurrency (2)
34. [Countdown Latch](#20-countdown-latch)
35. [Timer Function](#10-timer-function)

### Coding - ML & Analytics (2)
36. [Error Rate Monitor](#16-error-rate-monitor)
37. [Spam Email Detection](#28-spam-email-detection)

---

## 1. Homepage Title Deduplication

**Type:** Coding Problem

**Problem:** The Netflix homepage organizing structure involves a vertical list of shelves (rows), where each shelf contains a horizontal list of titles (movies or shows). Remove duplicate titles that appear across multiple shelves, keeping only the first occurrence.

**Focus:** HashSet, iteration order preservation

### The Challenge

Given a 2D list representing shelves of movie/show titles on the Netflix homepage, remove duplicate titles across all shelves. When a title appears on multiple shelves, keep only its first occurrence.

### Examples

**Example 1:**
```
Input: shelves = [
  ["Stranger Things", "The Crown", "Wednesday"],
  ["The Crown", "Squid Game", "Wednesday"],
  ["Money Heist", "Stranger Things"]
]
Output: [
  ["Stranger Things", "The Crown", "Wednesday"],
  ["Squid Game"],
  ["Money Heist"]
]
```

### Python Solution

```python
def deduplicateTitles(shelves: list[list[str]]) -> list[list[str]]:
    seen = set()
    result = []

    for shelf in shelves:
        deduped_shelf = []
        for title in shelf:
            if title not in seen:
                seen.add(title)
                deduped_shelf.append(title)
        result.append(deduped_shelf)

    return result
```

### TypeScript Solution

```typescript
function deduplicateTitles(shelves: string[][]): string[][] {
    const seen = new Set<string>();
    const result: string[][] = [];

    for (const shelf of shelves) {
        const dedupedShelf: string[] = [];
        for (const title of shelf) {
            if (!seen.has(title)) {
                seen.add(title);
                dedupedShelf.push(title);
            }
        }
        result.push(dedupedShelf);
    }

    return result;
}
```

### Interview Questions

1. **What if we want to keep the last occurrence instead of first?**
   - Two-pass approach: first pass to record last position of each title, second pass to include only titles at their last position.

2. **How to handle case-insensitive comparison?**
   - Normalize titles to lowercase when checking the set, but preserve original case in output.

3. **What if shelves are very large?**
   - Stream processing approach, or partition by title hash for parallel processing.

### Complexity Analysis

| Metric | Value |
|--------|-------|
| Time | O(n) where n = total titles |
| Space | O(n) for the seen set |

---

## 2. Netflix Sentiment Tracking

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

## 3. Design the Data Model for an Ads Demand Platform

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

## 4. Parallel Courses

**Type:** Coding Problem (Graph)

**Problem:** You are given an integer n, which indicates that there are n courses labeled from 1 to n. Determine course scheduling with prerequisites.

**Focus:** Topological sort, cycle detection, BFS level-order traversal

### The Challenge

You are given an integer `n`. This number tells you there are `n` courses you need to take, labeled from 1 to n.

You are also given a list called `relations`. Inside this list, each item is a pair of numbers: `[prevCourse, nextCourse]`. This represents a rule: you must finish `prevCourse` before you can start `nextCourse`.

**How Semesters Work:** In a single semester, you can take any number of courses. However, you can only take a course if you have already finished all of its prerequisites in a previous semester.

**Your Goal:** Return the minimum number of semesters needed to finish every course. If there is no way to finish all the courses (for example, if the prerequisites form a loop), return -1.

### Examples

**Example 1:**
```
Input: n = 3, relations = [[1,3], [2,3]]
Output: 2
Explanation:
- Semester 1: Take courses 1 and 2 (no prerequisites)
- Semester 2: Take course 3 (1 and 2 are now finished)
Total: 2 semesters
```

**Example 2:**
```
Input: n = 3, relations = [[1,2], [2,3], [3,1]]
Output: -1
Explanation:
Course 1 needs 2. Course 2 needs 3. Course 3 needs 1.
This forms a cycle, so no course can be started.
```

### Constraints

- `1 <= n <= 5000`
- `1 <= relations.length <= 5000`
- `relations[i].length == 2`
- `1 <= prevCourse, nextCourse <= n`
- `prevCourse != nextCourse` (no self-loops)
- All pairs are unique

### Key Insight

This is a **topological sort** problem where we need to find the longest path in the DAG. The minimum number of semesters equals the number of "levels" when processing courses in topological order using BFS (Kahn's algorithm).

### Python Solution

```python
from collections import deque, defaultdict

def minimumSemesters(n: int, relations: list[list[int]]) -> int:
    # Build adjacency list and in-degree count
    graph = defaultdict(list)
    in_degree = [0] * (n + 1)

    for prev_course, next_course in relations:
        graph[prev_course].append(next_course)
        in_degree[next_course] += 1

    # Start with courses that have no prerequisites
    queue = deque()
    for course in range(1, n + 1):
        if in_degree[course] == 0:
            queue.append(course)

    semesters = 0
    courses_taken = 0

    # BFS level by level (each level = one semester)
    while queue:
        semesters += 1
        # Process all courses available this semester
        for _ in range(len(queue)):
            course = queue.popleft()
            courses_taken += 1

            # Unlock dependent courses
            for next_course in graph[course]:
                in_degree[next_course] -= 1
                if in_degree[next_course] == 0:
                    queue.append(next_course)

    # Check if all courses were taken (cycle detection)
    return semesters if courses_taken == n else -1
```

### TypeScript Solution

```typescript
function minimumSemesters(n: number, relations: number[][]): number {
    const graph: Map<number, number[]> = new Map();
    const inDegree: number[] = new Array(n + 1).fill(0);

    // Build graph
    for (const [prev, next] of relations) {
        if (!graph.has(prev)) graph.set(prev, []);
        graph.get(prev)!.push(next);
        inDegree[next]++;
    }

    // Initialize queue with courses having no prerequisites
    const queue: number[] = [];
    for (let course = 1; course <= n; course++) {
        if (inDegree[course] === 0) {
            queue.push(course);
        }
    }

    let semesters = 0;
    let coursesTaken = 0;

    while (queue.length > 0) {
        semesters++;
        const levelSize = queue.length;

        for (let i = 0; i < levelSize; i++) {
            const course = queue.shift()!;
            coursesTaken++;

            for (const nextCourse of graph.get(course) || []) {
                inDegree[nextCourse]--;
                if (inDegree[nextCourse] === 0) {
                    queue.push(nextCourse);
                }
            }
        }
    }

    return coursesTaken === n ? semesters : -1;
}
```

### Walkthrough

For `n = 3, relations = [[1,3], [2,3]]`:

| Step | Queue | Semester | Action |
|------|-------|----------|--------|
| Init | `[1, 2]` | 0 | Courses 1, 2 have in-degree 0 |
| Process | `[]` | 1 | Take 1, 2. Decrement in-degree of 3 twice |
| Process | `[3]` | 1 | Course 3 now has in-degree 0 |
| Process | `[]` | 2 | Take course 3 |

**Result:** 2 semesters

### Interview Questions

1. **Why use BFS instead of DFS?**
   - BFS processes nodes level by level, naturally giving us the minimum number of semesters. DFS would require additional tracking to find the longest path.

2. **How do you detect a cycle?**
   - After BFS completes, if `courses_taken < n`, some courses were never added to the queue, meaning they're stuck in a cycle.

3. **What if we want to limit courses per semester?**
   - This becomes a harder problem (NP-hard). You'd need to prioritize courses that unlock the most dependencies, potentially using a priority queue.

4. **Time complexity?**
   - O(V + E) where V = n courses and E = number of relations. Each course and relation is processed exactly once.

### Complexity Analysis

| Metric | Value |
|--------|-------|
| Time | O(n + relations.length) |
| Space | O(n + relations.length) |

**Note:** LeetCode problem #1136

---

## 5. Home Page Video Recommendation

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

## 6. Contains Duplicate II

**Type:** Coding Problem (Array/HashMap)

**Problem:** Given an integer array `nums` and an integer `k`, return true if there are two distinct indices `i` and `j` in the array such that `nums[i] == nums[j]` and `abs(i - j) <= k`.

**Focus:** Sliding window with HashSet, or HashMap with index tracking

### Examples

**Example 1:**
```
Input: nums = [1,2,3,1], k = 3
Output: true
Explanation: nums[0] == nums[3], and 3 - 0 <= 3
```

**Example 2:**
```
Input: nums = [1,0,1,1], k = 1
Output: true
Explanation: nums[2] == nums[3], and 3 - 2 <= 1
```

**Example 3:**
```
Input: nums = [1,2,3,1,2,3], k = 2
Output: false
```

### Constraints

- `1 <= nums.length <= 10^5`
- `-10^9 <= nums[i] <= 10^9`
- `0 <= k <= 10^5`

### Key Insight

Use a **sliding window** of size k with a HashSet. As you move through the array, maintain only the last k elements in the set. If you find a duplicate within this window, return true.

### Python Solution

```python
def containsNearbyDuplicate(nums: list[int], k: int) -> bool:
    window = set()

    for i, num in enumerate(nums):
        if num in window:
            return True

        window.add(num)

        # Maintain window size of k
        if len(window) > k:
            window.remove(nums[i - k])

    return False
```

### TypeScript Solution

```typescript
function containsNearbyDuplicate(nums: number[], k: number): boolean {
    const window = new Set<number>();

    for (let i = 0; i < nums.length; i++) {
        if (window.has(nums[i])) {
            return true;
        }

        window.add(nums[i]);

        if (window.size > k) {
            window.delete(nums[i - k]);
        }
    }

    return false;
}
```

### Alternative: HashMap Approach

```python
def containsNearbyDuplicateHashMap(nums: list[int], k: int) -> bool:
    last_seen = {}  # value -> last index

    for i, num in enumerate(nums):
        if num in last_seen and i - last_seen[num] <= k:
            return True
        last_seen[num] = i

    return False
```

### Complexity Analysis

| Approach | Time | Space |
|----------|------|-------|
| Sliding Window | O(n) | O(min(n, k)) |
| HashMap | O(n) | O(n) |

**Note:** LeetCode problem #219

---

## 7. Movie History Friends

**Type:** Coding Problem

**Problem:** Two customers are considered "friends" if their recent viewing history contains an exact match of the last k movies. Given customer viewing histories and an integer k, return all pairs of customer IDs that are friends.

**Focus:** HashSet comparison, tuple hashing, pairwise matching

### The Challenge

Given a dictionary mapping customer IDs to their movie viewing history (in order), and an integer `k`, find all pairs of customers whose last `k` movies are exactly the same (same movies in same order).

### Examples

**Example 1:**
```
Input: history = {
  1: ["A", "B", "C", "D", "E"],
  2: ["X", "Y", "C", "D", "E"],
  3: ["P", "Q", "R", "S", "T"]
}, k = 3

Output: [[1, 2]]
Explanation: Customers 1 and 2 both have ["C", "D", "E"] as their last 3 movies.
```

**Example 2:**
```
Input: history = {
  1: ["A", "B"],
  2: ["A", "B"],
  3: ["B", "A"]
}, k = 2

Output: [[1, 2]]
Explanation: Order matters - [A,B] != [B,A]
```

### Constraints

- Customers with fewer than k movies in history cannot be friends with anyone
- Order of movies matters (exact sequence match required)
- Return pairs sorted by customer ID

### Python Solution

```python
def findMovieFriends(history: dict[int, list[str]], k: int) -> list[list[int]]:
    # Group customers by their last k movies (as tuple for hashing)
    last_k_map = {}

    for customer_id, movies in history.items():
        if len(movies) >= k:
            last_k = tuple(movies[-k:])
            if last_k not in last_k_map:
                last_k_map[last_k] = []
            last_k_map[last_k].append(customer_id)

    # Generate all pairs from each group
    result = []
    for customers in last_k_map.values():
        customers.sort()
        for i in range(len(customers)):
            for j in range(i + 1, len(customers)):
                result.append([customers[i], customers[j]])

    result.sort()
    return result
```

### TypeScript Solution

```typescript
function findMovieFriends(
    history: Map<number, string[]>,
    k: number
): number[][] {
    const lastKMap = new Map<string, number[]>();

    for (const [customerId, movies] of history) {
        if (movies.length >= k) {
            const lastK = movies.slice(-k).join('|');
            if (!lastKMap.has(lastK)) {
                lastKMap.set(lastK, []);
            }
            lastKMap.get(lastK)!.push(customerId);
        }
    }

    const result: number[][] = [];

    for (const customers of lastKMap.values()) {
        customers.sort((a, b) => a - b);
        for (let i = 0; i < customers.length; i++) {
            for (let j = i + 1; j < customers.length; j++) {
                result.push([customers[i], customers[j]]);
            }
        }
    }

    return result.sort((a, b) => a[0] - b[0] || a[1] - b[1]);
}
```

### Interview Questions

1. **How to optimize for millions of customers?**
   - Use hash of last-k movies as key, partition by hash, process in parallel.

2. **What if k is very large?**
   - Use rolling hash (like Rabin-Karp) to compute hash incrementally.

3. **How does this relate to recommendation systems?**
   - Similar viewing patterns suggest similar preferences; friends could receive cross-recommendations.

### Complexity Analysis

| Metric | Value |
|--------|-------|
| Time | O(n × k + p) where n = customers, p = pairs |
| Space | O(n × k) for storing last-k sequences |

---

## 8. Versioned File System

**Type:** Coding Problem (Data Structure Design)

**Problem:** Design and implement an in-memory file management system with versioning support. Support creating, reading, updating files, and retrieving previous versions.

**Focus:** HashMap with version history, copy-on-write semantics

### The Challenge

Implement a `VersionedFileSystem` class with the following methods:

- `create(path, content)` - Create a new file
- `read(path, version=None)` - Read file content (latest or specific version)
- `update(path, content)` - Update file, creating a new version
- `history(path)` - Get all versions of a file
- `delete(path)` - Delete a file

### Examples

```
fs = VersionedFileSystem()
fs.create("/docs/readme.txt", "Hello")      # version 1
fs.update("/docs/readme.txt", "Hello World") # version 2
fs.read("/docs/readme.txt")                  # returns "Hello World"
fs.read("/docs/readme.txt", version=1)       # returns "Hello"
fs.history("/docs/readme.txt")               # returns [1, 2]
```

### Python Solution

```python
from collections import defaultdict
from typing import Optional

class VersionedFileSystem:
    def __init__(self):
        # path -> list of (version, content)
        self.files: dict[str, list[tuple[int, str]]] = {}
        self.next_version: dict[str, int] = defaultdict(lambda: 1)

    def create(self, path: str, content: str) -> int:
        """Create a new file. Returns version number."""
        if path in self.files:
            raise FileExistsError(f"File {path} already exists")

        version = self.next_version[path]
        self.files[path] = [(version, content)]
        self.next_version[path] += 1
        return version

    def read(self, path: str, version: Optional[int] = None) -> str:
        """Read file content. If version not specified, read latest."""
        if path not in self.files or not self.files[path]:
            raise FileNotFoundError(f"File {path} not found")

        versions = self.files[path]

        if version is None:
            return versions[-1][1]  # Latest version

        # Binary search for specific version
        for v, content in versions:
            if v == version:
                return content

        raise ValueError(f"Version {version} not found for {path}")

    def update(self, path: str, content: str) -> int:
        """Update file with new content. Returns new version number."""
        if path not in self.files:
            raise FileNotFoundError(f"File {path} not found")

        version = self.next_version[path]
        self.files[path].append((version, content))
        self.next_version[path] += 1
        return version

    def history(self, path: str) -> list[int]:
        """Get all version numbers for a file."""
        if path not in self.files:
            raise FileNotFoundError(f"File {path} not found")

        return [v for v, _ in self.files[path]]

    def delete(self, path: str) -> None:
        """Delete a file and all its versions."""
        if path not in self.files:
            raise FileNotFoundError(f"File {path} not found")

        del self.files[path]
```

### TypeScript Solution

```typescript
class VersionedFileSystem {
    private files: Map<string, Array<[number, string]>> = new Map();
    private nextVersion: Map<string, number> = new Map();

    create(path: string, content: string): number {
        if (this.files.has(path)) {
            throw new Error(`File ${path} already exists`);
        }

        const version = this.getNextVersion(path);
        this.files.set(path, [[version, content]]);
        return version;
    }

    read(path: string, version?: number): string {
        const versions = this.files.get(path);
        if (!versions || versions.length === 0) {
            throw new Error(`File ${path} not found`);
        }

        if (version === undefined) {
            return versions[versions.length - 1][1];
        }

        const entry = versions.find(([v]) => v === version);
        if (!entry) {
            throw new Error(`Version ${version} not found`);
        }
        return entry[1];
    }

    update(path: string, content: string): number {
        const versions = this.files.get(path);
        if (!versions) {
            throw new Error(`File ${path} not found`);
        }

        const version = this.getNextVersion(path);
        versions.push([version, content]);
        return version;
    }

    history(path: string): number[] {
        const versions = this.files.get(path);
        if (!versions) {
            throw new Error(`File ${path} not found`);
        }
        return versions.map(([v]) => v);
    }

    delete(path: string): void {
        if (!this.files.has(path)) {
            throw new Error(`File ${path} not found`);
        }
        this.files.delete(path);
    }

    private getNextVersion(path: string): number {
        const current = this.nextVersion.get(path) || 1;
        this.nextVersion.set(path, current + 1);
        return current;
    }
}
```

### Interview Questions

1. **How would you implement rollback to a previous version?**
   - Add a `rollback(path, version)` method that creates a new version with the content of the specified old version.

2. **How to optimize storage for large files with small changes?**
   - Store diffs instead of full content (delta compression), or use copy-on-write with shared content blocks.

3. **How to make this thread-safe?**
   - Use read-write locks per file path, or implement MVCC (Multi-Version Concurrency Control).

4. **How to handle directory structure?**
   - Use a trie or nested dictionary structure for path hierarchy.

### Complexity Analysis

| Operation | Time | Space |
|-----------|------|-------|
| create | O(1) | O(content size) |
| read | O(versions) or O(1) for latest | O(1) |
| update | O(1) | O(content size) |
| history | O(versions) | O(versions) |
| delete | O(1) | O(1) |

**Total Space:** O(total content across all versions)

---

## 9. Music Playlist

**Type:** Coding Problem (Data Structure)

**Problem:** Design a data structure that tracks music listening history with timestamps. Support adding plays, querying recent plays, and finding the most played songs in a time window.

**Focus:** HashMap + sorted structure, time-based queries

### The Challenge

Implement a `MusicPlaylist` class:

- `play(songId, timestamp)` - Record a song play
- `getRecentPlays(k)` - Get the k most recent song plays
- `getMostPlayed(startTime, endTime)` - Get the most played song in time range

### Examples

```
playlist = MusicPlaylist()
playlist.play("song1", 100)
playlist.play("song2", 200)
playlist.play("song1", 300)
playlist.getRecentPlays(2)           # returns ["song1", "song2"]
playlist.getMostPlayed(0, 400)       # returns "song1" (played twice)
```

### Python Solution

```python
from collections import defaultdict
import heapq

class MusicPlaylist:
    def __init__(self):
        self.plays = []  # [(timestamp, songId)]
        self.song_plays = defaultdict(list)  # songId -> [timestamps]

    def play(self, songId: str, timestamp: int) -> None:
        """Record a song play at given timestamp."""
        self.plays.append((timestamp, songId))
        self.song_plays[songId].append(timestamp)

    def getRecentPlays(self, k: int) -> list[str]:
        """Get the k most recent song plays."""
        # Sort by timestamp descending, take first k
        sorted_plays = sorted(self.plays, key=lambda x: -x[0])
        return [songId for _, songId in sorted_plays[:k]]

    def getMostPlayed(self, startTime: int, endTime: int) -> str:
        """Get most played song in time range [startTime, endTime]."""
        play_count = defaultdict(int)

        for timestamp, songId in self.plays:
            if startTime <= timestamp <= endTime:
                play_count[songId] += 1

        if not play_count:
            return ""

        return max(play_count.keys(), key=lambda x: play_count[x])


# Optimized version with better time complexity
class MusicPlaylistOptimized:
    def __init__(self):
        self.plays = []  # [(timestamp, songId)] - kept sorted by timestamp
        self.song_plays = defaultdict(list)

    def play(self, songId: str, timestamp: int) -> None:
        # Assume timestamps are given in order (common case)
        self.plays.append((timestamp, songId))
        self.song_plays[songId].append(timestamp)

    def getRecentPlays(self, k: int) -> list[str]:
        # O(k) since plays are in order
        return [songId for _, songId in self.plays[-k:][::-1]]

    def getMostPlayed(self, startTime: int, endTime: int) -> str:
        # Binary search for time range
        import bisect
        left = bisect.bisect_left(self.plays, (startTime,))
        right = bisect.bisect_right(self.plays, (endTime + 1,))

        play_count = defaultdict(int)
        for i in range(left, right):
            play_count[self.plays[i][1]] += 1

        if not play_count:
            return ""

        return max(play_count.keys(), key=lambda x: play_count[x])
```

### TypeScript Solution

```typescript
class MusicPlaylist {
    private plays: Array<[number, string]> = [];
    private songPlays: Map<string, number[]> = new Map();

    play(songId: string, timestamp: number): void {
        this.plays.push([timestamp, songId]);
        if (!this.songPlays.has(songId)) {
            this.songPlays.set(songId, []);
        }
        this.songPlays.get(songId)!.push(timestamp);
    }

    getRecentPlays(k: number): string[] {
        return this.plays
            .slice(-k)
            .reverse()
            .map(([_, songId]) => songId);
    }

    getMostPlayed(startTime: number, endTime: number): string {
        const playCount = new Map<string, number>();

        for (const [timestamp, songId] of this.plays) {
            if (timestamp >= startTime && timestamp <= endTime) {
                playCount.set(songId, (playCount.get(songId) || 0) + 1);
            }
        }

        let maxSong = "";
        let maxCount = 0;

        for (const [songId, count] of playCount) {
            if (count > maxCount) {
                maxCount = count;
                maxSong = songId;
            }
        }

        return maxSong;
    }
}
```

### Interview Questions

1. **How to handle ties in getMostPlayed?**
   - Return lexicographically smallest, or most recently played among ties.

2. **How to optimize for real-time top-k queries?**
   - Use a combination of time-bucketed counts and a heap structure.

3. **How to scale for millions of users?**
   - Partition by user ID, use streaming aggregation (e.g., Apache Kafka + Flink).

### Complexity Analysis

| Operation | Basic | Optimized |
|-----------|-------|-----------|
| play | O(1) | O(1) |
| getRecentPlays | O(n log n) | O(k) |
| getMostPlayed | O(n) | O(log n + range size) |

---

## 10. Timer Function

**Type:** Coding Problem

**Problem:** Create a function named `timer` accepting a single parameter—seconds (non-negative integer)—that outputs a human-readable time string representation.

**Focus:** Integer division, modulo operations, string formatting

### The Challenge

Convert seconds into a human-readable format: "X years, X days, X hours, X minutes, X seconds". Only include non-zero units. Handle singular/plural forms correctly.

### Conversion Values

- 1 minute = 60 seconds
- 1 hour = 60 minutes = 3600 seconds
- 1 day = 24 hours = 86400 seconds
- 1 year = 365 days = 31536000 seconds

### Examples

```
timer(0)        → "0 seconds"
timer(1)        → "1 second"
timer(62)       → "1 minute, 2 seconds"
timer(3662)     → "1 hour, 1 minute, 2 seconds"
timer(86400)    → "1 day"
timer(31536000) → "1 year"
timer(31622400) → "1 year, 1 day"
```

### Python Solution

```python
def timer(seconds: int) -> str:
    if seconds == 0:
        return "0 seconds"

    units = [
        (31536000, "year"),
        (86400, "day"),
        (3600, "hour"),
        (60, "minute"),
        (1, "second")
    ]

    parts = []

    for unit_seconds, unit_name in units:
        if seconds >= unit_seconds:
            count = seconds // unit_seconds
            seconds %= unit_seconds

            # Handle singular/plural
            if count == 1:
                parts.append(f"{count} {unit_name}")
            else:
                parts.append(f"{count} {unit_name}s")

    return ", ".join(parts)


# Recursive version
def timer_recursive(seconds: int) -> str:
    if seconds == 0:
        return "0 seconds"

    units = [
        (31536000, "year"),
        (86400, "day"),
        (3600, "hour"),
        (60, "minute"),
        (1, "second")
    ]

    def helper(secs: int, idx: int) -> list[str]:
        if secs == 0 or idx >= len(units):
            return []

        unit_secs, unit_name = units[idx]

        if secs >= unit_secs:
            count = secs // unit_secs
            remaining = secs % unit_secs
            plural = "s" if count > 1 else ""
            return [f"{count} {unit_name}{plural}"] + helper(remaining, idx + 1)
        else:
            return helper(secs, idx + 1)

    return ", ".join(helper(seconds, 0))
```

### TypeScript Solution

```typescript
function timer(seconds: number): string {
    if (seconds === 0) {
        return "0 seconds";
    }

    const units: Array<[number, string]> = [
        [31536000, "year"],
        [86400, "day"],
        [3600, "hour"],
        [60, "minute"],
        [1, "second"]
    ];

    const parts: string[] = [];
    let remaining = seconds;

    for (const [unitSeconds, unitName] of units) {
        if (remaining >= unitSeconds) {
            const count = Math.floor(remaining / unitSeconds);
            remaining %= unitSeconds;

            const plural = count === 1 ? "" : "s";
            parts.push(`${count} ${unitName}${plural}`);
        }
    }

    return parts.join(", ");
}
```

### Interview Questions

1. **How to handle leap years?**
   - Use 365.25 days/year on average, or work with actual calendar dates.

2. **How to localize for different languages?**
   - Use a localization library with plural rules (ICU MessageFormat).

3. **How to parse the string back to seconds?**
   - Regex to extract numbers and units, multiply and sum.

### Complexity Analysis

| Metric | Value |
|--------|-------|
| Time | O(1) - fixed number of units |
| Space | O(1) |

---

## 11. Design an Ads Audience Targeting System

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

## 12. Design a Billing System for 300M Subscribers

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

## 12. Design a Billing System for 300M Subscribers

**Type:** System Design

**Problem:** Architect a billing infrastructure for Netflix capable of processing monthly subscription charges across 300 million users.

**Phases:**
- Phase 1: What We Need to Build
- Phase 2: Database Design
- Phase 3: How Systems Talk
- Phase 4: System Architecture
- Phase 5: Handling Growth and Problems
- Interview Checklist
- Main Takeaways

**Scale:** 300 million subscribers, monthly subscription-based billing

---

## 13. Sort by User Preference

**Type:** Coding Problem

**Problem:** Given a list of shows and user preference data (genre preferences, watch history, ratings), sort the shows to maximize relevance for the user.

**Focus:** Custom sorting, multi-criteria comparison, scoring functions

### The Challenge

Sort a list of shows based on multiple user preference signals:
1. Preferred genres (ranked list)
2. Previously watched shows (boost similar content)
3. User ratings of similar shows
4. Popularity/trending score

### Examples

```
Input:
shows = [
    {"id": 1, "title": "Show A", "genre": "Drama", "popularity": 80},
    {"id": 2, "title": "Show B", "genre": "Comedy", "popularity": 90},
    {"id": 3, "title": "Show C", "genre": "Drama", "popularity": 70}
]
user_prefs = {
    "genre_ranking": ["Drama", "Comedy", "Action"],
    "watched": [4, 5],  # IDs of watched shows
    "ratings": {4: 5, 5: 4}  # show_id -> rating
}

Output: [1, 3, 2]  # Drama shows first (preferred genre), then Comedy
```

### Python Solution

```python
from typing import Any

def sortByUserPreference(
    shows: list[dict[str, Any]],
    user_prefs: dict[str, Any]
) -> list[int]:
    """Sort shows by user preference and return ordered IDs."""

    genre_ranking = {
        genre: i for i, genre in enumerate(user_prefs.get("genre_ranking", []))
    }
    default_genre_rank = len(genre_ranking)

    def calculate_score(show: dict) -> tuple:
        """
        Returns a tuple for sorting (lower is better).
        Tuple: (genre_rank, -popularity, title)
        """
        genre = show.get("genre", "")
        genre_rank = genre_ranking.get(genre, default_genre_rank)
        popularity = show.get("popularity", 0)
        title = show.get("title", "")

        # Negative popularity so higher popularity sorts first
        return (genre_rank, -popularity, title)

    sorted_shows = sorted(shows, key=calculate_score)
    return [show["id"] for show in sorted_shows]


# More sophisticated scoring with weighted factors
def sortByUserPreferenceWeighted(
    shows: list[dict[str, Any]],
    user_prefs: dict[str, Any],
    weights: dict[str, float] = None
) -> list[int]:
    """Sort with weighted scoring across multiple factors."""

    if weights is None:
        weights = {
            "genre": 0.4,
            "popularity": 0.3,
            "recency": 0.2,
            "similarity": 0.1
        }

    genre_ranking = user_prefs.get("genre_ranking", [])
    genre_scores = {g: 1 - (i / len(genre_ranking))
                   for i, g in enumerate(genre_ranking)}

    def calculate_score(show: dict) -> float:
        score = 0.0

        # Genre preference score (0-1)
        genre = show.get("genre", "")
        score += weights["genre"] * genre_scores.get(genre, 0)

        # Popularity score (normalized 0-1)
        popularity = show.get("popularity", 0) / 100
        score += weights["popularity"] * popularity

        # Could add more factors: recency, similarity to watched, etc.

        return score

    sorted_shows = sorted(shows, key=calculate_score, reverse=True)
    return [show["id"] for show in sorted_shows]
```

### TypeScript Solution

```typescript
interface Show {
    id: number;
    title: string;
    genre: string;
    popularity: number;
}

interface UserPrefs {
    genre_ranking: string[];
    watched?: number[];
    ratings?: Record<number, number>;
}

function sortByUserPreference(shows: Show[], userPrefs: UserPrefs): number[] {
    const genreRanking = new Map<string, number>();
    userPrefs.genre_ranking.forEach((genre, index) => {
        genreRanking.set(genre, index);
    });

    const defaultRank = userPrefs.genre_ranking.length;

    const sortedShows = [...shows].sort((a, b) => {
        // Primary: genre preference
        const genreRankA = genreRanking.get(a.genre) ?? defaultRank;
        const genreRankB = genreRanking.get(b.genre) ?? defaultRank;

        if (genreRankA !== genreRankB) {
            return genreRankA - genreRankB;
        }

        // Secondary: popularity (higher first)
        if (a.popularity !== b.popularity) {
            return b.popularity - a.popularity;
        }

        // Tertiary: alphabetical
        return a.title.localeCompare(b.title);
    });

    return sortedShows.map(show => show.id);
}
```

### Interview Questions

1. **How to incorporate collaborative filtering?**
   - Add similarity scores based on what similar users watched; boost shows popular among users with similar taste profiles.

2. **How to handle cold start for new users?**
   - Fall back to popularity-based ranking, or use demographic-based defaults.

3. **How to A/B test different ranking algorithms?**
   - Randomly assign users to cohorts, track engagement metrics (watch time, completion rate).

### Complexity Analysis

| Metric | Value |
|--------|-------|
| Time | O(n log n) for sorting |
| Space | O(n) for sorted copy |

---

## 14. Contains Duplicate III

**Type:** Coding Problem (Array/Sliding Window)

**Problem:** Given an integer array `nums` and two integers `indexDiff` and `valueDiff`, return true if there exist two distinct indices `i` and `j` such that `abs(i - j) <= indexDiff` and `abs(nums[i] - nums[j]) <= valueDiff`.

**Focus:** Bucket sort, sliding window, balanced BST

### Examples

**Example 1:**
```
Input: nums = [1,2,3,1], indexDiff = 3, valueDiff = 0
Output: true
Explanation: nums[0] == nums[3], indices differ by 3, values differ by 0
```

**Example 2:**
```
Input: nums = [1,5,9,1,5,9], indexDiff = 2, valueDiff = 3
Output: false
```

### Constraints

- `2 <= nums.length <= 10^5`
- `-10^9 <= nums[i] <= 10^9`
- `1 <= indexDiff <= nums.length`
- `0 <= valueDiff <= 10^9`

### Key Insight

Use **bucket sort** with bucket size = `valueDiff + 1`. Numbers in the same bucket are within valueDiff. Numbers in adjacent buckets might be within valueDiff. Maintain a sliding window of buckets.

### Python Solution

```python
def containsNearbyAlmostDuplicate(
    nums: list[int],
    indexDiff: int,
    valueDiff: int
) -> bool:
    if valueDiff < 0:
        return False

    buckets = {}
    bucket_size = valueDiff + 1  # Avoid division by zero

    def get_bucket_id(num: int) -> int:
        # Handle negative numbers correctly
        return num // bucket_size

    for i, num in enumerate(nums):
        bucket_id = get_bucket_id(num)

        # Check same bucket
        if bucket_id in buckets:
            return True

        # Check adjacent buckets
        if bucket_id - 1 in buckets and abs(num - buckets[bucket_id - 1]) <= valueDiff:
            return True
        if bucket_id + 1 in buckets and abs(num - buckets[bucket_id + 1]) <= valueDiff:
            return True

        # Add to bucket
        buckets[bucket_id] = num

        # Remove old bucket outside window
        if i >= indexDiff:
            old_bucket_id = get_bucket_id(nums[i - indexDiff])
            del buckets[old_bucket_id]

    return False
```

### TypeScript Solution

```typescript
function containsNearbyAlmostDuplicate(
    nums: number[],
    indexDiff: number,
    valueDiff: number
): boolean {
    if (valueDiff < 0) return false;

    const buckets = new Map<number, number>();
    const bucketSize = valueDiff + 1;

    const getBucketId = (num: number): number => {
        return Math.floor(num / bucketSize);
    };

    for (let i = 0; i < nums.length; i++) {
        const num = nums[i];
        const bucketId = getBucketId(num);

        // Check same bucket
        if (buckets.has(bucketId)) {
            return true;
        }

        // Check adjacent buckets
        if (buckets.has(bucketId - 1) &&
            Math.abs(num - buckets.get(bucketId - 1)!) <= valueDiff) {
            return true;
        }
        if (buckets.has(bucketId + 1) &&
            Math.abs(num - buckets.get(bucketId + 1)!) <= valueDiff) {
            return true;
        }

        // Add current number to bucket
        buckets.set(bucketId, num);

        // Remove element outside window
        if (i >= indexDiff) {
            const oldBucketId = getBucketId(nums[i - indexDiff]);
            buckets.delete(oldBucketId);
        }
    }

    return false;
}
```

### Interview Questions

1. **Why bucket size is valueDiff + 1?**
   - Ensures numbers in same bucket differ by at most valueDiff. If bucket size were valueDiff, boundary cases would fail.

2. **Why check adjacent buckets?**
   - Numbers in adjacent buckets might still be within valueDiff of each other (e.g., bucket boundaries).

3. **Alternative approach?**
   - Use a balanced BST (TreeSet) to find floor/ceiling in O(log k) per operation.

### Complexity Analysis

| Metric | Value |
|--------|-------|
| Time | O(n) |
| Space | O(min(n, indexDiff)) |

**Note:** LeetCode problem #220

---

## 15. Longest Substring Without Repeating Characters

**Type:** Coding Problem (Sliding Window)

**Problem:** Given a string `s`, find the length of the longest substring without repeating characters.

**Focus:** Sliding window, HashSet/HashMap for tracking

### Examples

**Example 1:**
```
Input: s = "abcabcbb"
Output: 3
Explanation: "abc" is the longest substring without repeating characters.
```

**Example 2:**
```
Input: s = "bbbbb"
Output: 1
Explanation: "b" is the longest substring.
```

**Example 3:**
```
Input: s = "pwwkew"
Output: 3
Explanation: "wke" is the answer. Note "pwke" is a subsequence, not substring.
```

### Constraints

- `0 <= s.length <= 5 * 10^4`
- `s` consists of English letters, digits, symbols, and spaces

### Key Insight

Use a **sliding window** with two pointers. Expand the right pointer to include new characters. When a duplicate is found, shrink from the left until the duplicate is removed.

### Python Solution

```python
def lengthOfLongestSubstring(s: str) -> int:
    char_index = {}  # char -> most recent index
    max_length = 0
    left = 0

    for right, char in enumerate(s):
        # If char is in window, move left pointer past its last occurrence
        if char in char_index and char_index[char] >= left:
            left = char_index[char] + 1

        char_index[char] = right
        max_length = max(max_length, right - left + 1)

    return max_length


# Alternative using set
def lengthOfLongestSubstringSet(s: str) -> int:
    char_set = set()
    max_length = 0
    left = 0

    for right in range(len(s)):
        while s[right] in char_set:
            char_set.remove(s[left])
            left += 1

        char_set.add(s[right])
        max_length = max(max_length, right - left + 1)

    return max_length
```

### TypeScript Solution

```typescript
function lengthOfLongestSubstring(s: string): number {
    const charIndex = new Map<string, number>();
    let maxLength = 0;
    let left = 0;

    for (let right = 0; right < s.length; right++) {
        const char = s[right];

        if (charIndex.has(char) && charIndex.get(char)! >= left) {
            left = charIndex.get(char)! + 1;
        }

        charIndex.set(char, right);
        maxLength = Math.max(maxLength, right - left + 1);
    }

    return maxLength;
}
```

### Interview Questions

1. **Why track index instead of just presence?**
   - Allows O(1) jump of left pointer instead of incrementally removing characters.

2. **What if we need the actual substring?**
   - Track `start` and `end` indices of best window, return `s[start:end+1]`.

3. **How to handle Unicode characters?**
   - Same algorithm works; HashMap handles any character type.

### Complexity Analysis

| Metric | Value |
|--------|-------|
| Time | O(n) |
| Space | O(min(n, alphabet_size)) |

**Note:** LeetCode problem #3

---

## 16. Error Rate Monitor

**Type:** Coding Problem (Monitoring)

**Problem:** Design a system that monitors error rates over sliding time windows. Track the ratio of errors to total requests and trigger alerts when thresholds are exceeded.

**Focus:** Sliding window, circular buffer, rate calculation

### The Challenge

Implement an `ErrorRateMonitor` class:

- `record(timestamp, isError)` - Record a request (error or success)
- `getErrorRate(windowSeconds)` - Get error rate in last N seconds
- `isHealthy(threshold)` - Return true if error rate < threshold

### Examples

```
monitor = ErrorRateMonitor()
monitor.record(100, False)  # success at t=100
monitor.record(101, True)   # error at t=101
monitor.record(102, False)  # success at t=102
monitor.record(103, True)   # error at t=103

monitor.getErrorRate(10)    # 2/4 = 0.5 (50% error rate)
monitor.isHealthy(0.6)      # True (0.5 < 0.6)
monitor.isHealthy(0.4)      # False (0.5 >= 0.4)
```

### Python Solution

```python
from collections import deque
from typing import Tuple

class ErrorRateMonitor:
    def __init__(self, max_window: int = 3600):
        """
        Args:
            max_window: Maximum window size to track (default 1 hour)
        """
        self.max_window = max_window
        self.events: deque[Tuple[int, bool]] = deque()  # (timestamp, isError)
        self.current_time = 0

    def record(self, timestamp: int, isError: bool) -> None:
        """Record a request event."""
        self.current_time = max(self.current_time, timestamp)
        self.events.append((timestamp, isError))
        self._cleanup()

    def _cleanup(self) -> None:
        """Remove events outside max window."""
        cutoff = self.current_time - self.max_window
        while self.events and self.events[0][0] < cutoff:
            self.events.popleft()

    def getErrorRate(self, windowSeconds: int) -> float:
        """Get error rate in the last windowSeconds."""
        cutoff = self.current_time - windowSeconds
        total = 0
        errors = 0

        for timestamp, isError in self.events:
            if timestamp >= cutoff:
                total += 1
                if isError:
                    errors += 1

        return errors / total if total > 0 else 0.0

    def isHealthy(self, threshold: float, windowSeconds: int = 60) -> bool:
        """Check if error rate is below threshold."""
        return self.getErrorRate(windowSeconds) < threshold


# Optimized version with bucketed counts
class ErrorRateMonitorOptimized:
    def __init__(self, bucket_size: int = 1, num_buckets: int = 3600):
        self.bucket_size = bucket_size
        self.num_buckets = num_buckets
        self.buckets = [(0, 0)] * num_buckets  # (total, errors) per bucket
        self.current_bucket = 0

    def _get_bucket_index(self, timestamp: int) -> int:
        return (timestamp // self.bucket_size) % self.num_buckets

    def record(self, timestamp: int, isError: bool) -> None:
        idx = self._get_bucket_index(timestamp)
        total, errors = self.buckets[idx]
        self.buckets[idx] = (total + 1, errors + (1 if isError else 0))

    def getErrorRate(self, windowSeconds: int) -> float:
        # Sum buckets in window
        num_buckets = min(windowSeconds // self.bucket_size, self.num_buckets)
        total = errors = 0

        for i in range(num_buckets):
            idx = (self.current_bucket - i) % self.num_buckets
            t, e = self.buckets[idx]
            total += t
            errors += e

        return errors / total if total > 0 else 0.0
```

### TypeScript Solution

```typescript
class ErrorRateMonitor {
    private events: Array<[number, boolean]> = [];
    private currentTime: number = 0;
    private maxWindow: number;

    constructor(maxWindow: number = 3600) {
        this.maxWindow = maxWindow;
    }

    record(timestamp: number, isError: boolean): void {
        this.currentTime = Math.max(this.currentTime, timestamp);
        this.events.push([timestamp, isError]);
        this.cleanup();
    }

    private cleanup(): void {
        const cutoff = this.currentTime - this.maxWindow;
        while (this.events.length > 0 && this.events[0][0] < cutoff) {
            this.events.shift();
        }
    }

    getErrorRate(windowSeconds: number): number {
        const cutoff = this.currentTime - windowSeconds;
        let total = 0;
        let errors = 0;

        for (const [timestamp, isError] of this.events) {
            if (timestamp >= cutoff) {
                total++;
                if (isError) errors++;
            }
        }

        return total > 0 ? errors / total : 0;
    }

    isHealthy(threshold: number, windowSeconds: number = 60): boolean {
        return this.getErrorRate(windowSeconds) < threshold;
    }
}
```

### Interview Questions

1. **How to handle high throughput?**
   - Use bucketed aggregation (per-second or per-minute counts) instead of individual events.

2. **How to support multiple metrics?**
   - Track error codes separately, use labels/tags for different services.

3. **How to implement alerting?**
   - Check thresholds on each record, use hysteresis to prevent alert flapping.

### Complexity Analysis

| Operation | Basic | Optimized (Bucketed) |
|-----------|-------|----------------------|
| record | O(1) amortized | O(1) |
| getErrorRate | O(window events) | O(window buckets) |
| Space | O(max events) | O(num buckets) |

---

## 17. Meeting Rooms

**Type:** Coding Problem (Intervals)

**Problem:** Given an array of meeting time interval objects consisting of start and end times `[[start1,end1],[start2,end2],...]` (start_i < end_i), determine if a person could attend all meetings without any conflicts.

**Focus:** Interval sorting, overlap detection

### Examples

**Example 1:**
```
Input: intervals = [[0,30],[5,10],[15,20]]
Output: false
Explanation: [0,30] overlaps with [5,10] and [15,20]
```

**Example 2:**
```
Input: intervals = [[7,10],[2,4]]
Output: true
Explanation: No overlaps after sorting: [2,4], [7,10]
```

### Constraints

- `0 <= intervals.length <= 10^4`
- `intervals[i].length == 2`
- `0 <= start_i < end_i <= 10^6`

### Key Insight

Sort intervals by start time. Two meetings overlap if the current meeting starts before the previous one ends.

### Python Solution

```python
def canAttendMeetings(intervals: list[list[int]]) -> bool:
    # Sort by start time
    intervals.sort(key=lambda x: x[0])

    for i in range(1, len(intervals)):
        # Check if current meeting starts before previous ends
        if intervals[i][0] < intervals[i - 1][1]:
            return False

    return True
```

### TypeScript Solution

```typescript
function canAttendMeetings(intervals: number[][]): boolean {
    // Sort by start time
    intervals.sort((a, b) => a[0] - b[0]);

    for (let i = 1; i < intervals.length; i++) {
        if (intervals[i][0] < intervals[i - 1][1]) {
            return false;
        }
    }

    return true;
}
```

### Follow-up: Meeting Rooms II (Minimum Rooms)

```python
import heapq

def minMeetingRooms(intervals: list[list[int]]) -> int:
    if not intervals:
        return 0

    intervals.sort(key=lambda x: x[0])

    # Min-heap of end times
    rooms = []
    heapq.heappush(rooms, intervals[0][1])

    for i in range(1, len(intervals)):
        # If earliest ending meeting ends before this one starts, reuse room
        if rooms[0] <= intervals[i][0]:
            heapq.heappop(rooms)

        heapq.heappush(rooms, intervals[i][1])

    return len(rooms)
```

### Interview Questions

1. **What if meetings can be rescheduled?**
   - This becomes an optimization problem (interval scheduling maximization).

2. **How to find which meetings conflict?**
   - Build an interval tree or sweep line algorithm to find all overlapping pairs.

3. **Netflix use case?**
   - Scheduling server maintenance windows, content release slots, user session management.

### Complexity Analysis

| Problem | Time | Space |
|---------|------|-------|
| Meeting Rooms I | O(n log n) | O(1) |
| Meeting Rooms II | O(n log n) | O(n) |

**Note:** LeetCode problems #252 and #253

---

## 18. Design an Ads Frequency Cap System

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

## 19. ML Job Scheduler

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

## 20. Countdown Latch

**Type:** Coding Problem (Concurrency)

**Problem:** Design and implement a thread-safe counter that allows multiple threads to wait until the counter reaches zero.

**Focus:** Thread synchronization, blocking mechanisms, counter management

### The Challenge

You need to build a thread-safe counter. This counter allows several threads to pause and wait until the count drops to zero.

You must create a CountdownLatch class with these methods:

```python
class CountdownLatch:
    def __init__(self, count: int):
        """Start the latch with a specific number (must be >= 0)."""
        pass

    def count_up(self) -> None:
        """Add 1 to the count."""
        pass

    def count_down(self) -> None:
        """
        Subtract 1 from the count.
        If the count hits zero, wake up every thread that is waiting.
        """
        pass

    def wait(self) -> None:
        """
        Stop the current thread here until the count is zero.
        If the count is already zero, do not stop.
        """
        pass
```

### Common Use Cases

- **Parallel Downloads:** Wait for multiple files to finish downloading before combining them.
- **Starting Services:** Wait for all small services to be ready before letting users connect.
- **Batch Jobs:** Wait for a group of items to finish processing.

### Code Example

Here is how you use the latch. We create a latch with a count of 3. Then, we start 3 worker threads. The main thread waits until all workers represent they are done.

```python
import threading
import time

def worker(latch: CountdownLatch, worker_id: int):
    """Act like a worker doing a job."""
    print(f"Worker {worker_id} starting...")
    time.sleep(0.5)  # Pretend to work
    print(f"Worker {worker_id} done!")
    latch.count_down()

# Create latch with count = 3 (we expect 3 workers)
latch = CountdownLatch(3)

# Start 3 worker threads
for i in range(3):
    t = threading.Thread(target=worker, args=(latch, i))
    t.start()

# Main thread pauses here until workers finish
print("Main thread waiting for workers...")
latch.wait()
print("All workers completed!")
```

**Output:**
```
Main thread waiting for workers...
Worker 0 starting...
Worker 1 starting...
Worker 2 starting...
Worker 0 done!
Worker 1 done!
Worker 2 done!
All workers completed!
```

### Python Solution

We use a Condition variable to handle the threads safely.

```python
import threading

class CountdownLatch:
    def __init__(self, count: int):
        if count < 0:
            raise ValueError("Count must be non-negative")

        self._count = count
        self._lock = threading.Lock()
        self._condition = threading.Condition(self._lock)

    def count_up(self) -> None:
        """Add 1 to the count."""
        with self._condition:
            self._count += 1

    def count_down(self) -> None:
        """Subtract 1. If zero, tell everyone to wake up."""
        with self._condition:
            if self._count > 0:
                self._count -= 1
                if self._count == 0:
                    self._condition.notify_all()

    def wait(self) -> None:
        """Block here until count is zero."""
        with self._condition:
            while self._count > 0:
                self._condition.wait()

    def get_count(self) -> int:
        """Get the current number (for checking status)."""
        with self._lock:
            return self._count
```

**Key Implementation Details:**

- **threading.Condition:** This acts like a normal lock, but it also lets threads wait for a signal.
- **notify_all():** This wakes up ALL the sleeping threads once the count hits zero.
- **While loop in wait():** This is a safety check. Sometimes threads wake up without a signal (called spurious wakeups). The loop forces the thread to check the count again.
- **Locking:** We use a lock to make sure the count is accurate when multiple threads change it at the same time.

### TypeScript Solution

Here is how you can do this in JavaScript/TypeScript using Promises.

```typescript
class CountdownLatch {
  private count: number;
  private resolvers: Array<() => void> = [];

  constructor(count: number) {
    if (count < 0) throw new Error("Count must be non-negative");
    this.count = count;
  }

  countUp(): void {
    this.count += 1;
  }

  countDown(): void {
    if (this.count > 0) {
      this.count -= 1;
      if (this.count === 0) {
        this.resolvers.forEach(resolve => resolve());
        this.resolvers = [];
      }
    }
  }

  async wait(): Promise<void> {
    if (this.count === 0) return;

    return new Promise<void>((resolve) => {
      this.resolvers.push(resolve);
    });
  }

  getCount(): number {
    return this.count;
  }
}
```

### Interview Questions

1. **Why use Condition instead of Lock?**
   - A Lock only stops two threads from touching data at the exact same time.
   - A Condition does that too, but it also helps threads coordinate by letting them "sleep" and "wake up" based on a signal.

2. **Why do we need a while loop in wait()?**
   - This handles spurious wakeups. Occasionally, the operating system wakes a thread up for no reason. The loop checks the condition again to ensure the thread only proceeds if the count is actually zero.

3. **What is the difference between notify() and notify_all()?**
   - `notify()` wakes up only ONE thread.
   - `notify_all()` wakes up EVERY waiting thread.
   - We use `notify_all()` because when the count hits zero, everyone who is waiting needs to move forward.

4. **How would you add a timeout?**
   - You would change the `wait()` method to accept a time limit. If the time runs out before the count hits zero, the method returns or throws an error.

5. **How would you make this work across different servers?**
   - This requires a distributed system, like Redis or Zookeeper, to share the count state across the network.

### Performance Analysis

| Operation | Time | Space |
| --- | --- | --- |
| count_up | O(1) | O(1) |
| count_down | O(1) | O(1) |
| wait | O(1) | O(1) |

**Space:** O(W), where W is the number of waiting threads.

---

## 21. Design a WAL Log Enrichment Pipeline

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

## 22. Movie History Friends II

**Type:** Coding Problem

**Problem:** Given a map of customer IDs linked to their viewing history, plus two integers `k` and `m` (with `m <= k`), return all pairs of customer IDs that are "friends" if their last k movies share at least m common titles (order doesn't matter).

**Focus:** Set intersection, pairwise comparison optimization

### The Challenge

Unlike Movie History Friends I (exact sequence match), this version requires at least `m` common movies among the last `k` movies, regardless of order.

### Examples

**Example 1:**
```
Input: history = {
    1: ["A", "B", "C", "D"],
    2: ["X", "D", "B", "A"],
    3: ["P", "Q", "R", "S"]
}, k = 3, m = 2

Output: [[1, 2]]
Explanation:
- Customer 1's last 3: {"B", "C", "D"}
- Customer 2's last 3: {"D", "B", "A"}
- Common: {"B", "D"} = 2 movies >= m
- Customer 3 shares nothing with others
```

**Example 2:**
```
Input: history = {
    1: ["A", "B", "C"],
    2: ["C", "B", "A"],
    3: ["A", "B", "C"]
}, k = 3, m = 3

Output: [[1, 2], [1, 3], [2, 3]]
Explanation: All three share all three movies (order doesn't matter)
```

### Python Solution

```python
def findMovieFriendsII(
    history: dict[int, list[str]],
    k: int,
    m: int
) -> list[list[int]]:
    # Filter customers with at least k movies and get their last k as sets
    customer_sets = {}

    for customer_id, movies in history.items():
        if len(movies) >= k:
            customer_sets[customer_id] = set(movies[-k:])

    # Compare all pairs
    result = []
    customers = sorted(customer_sets.keys())

    for i in range(len(customers)):
        for j in range(i + 1, len(customers)):
            c1, c2 = customers[i], customers[j]
            common = len(customer_sets[c1] & customer_sets[c2])

            if common >= m:
                result.append([c1, c2])

    return result


# Optimized with inverted index for sparse data
def findMovieFriendsIIOptimized(
    history: dict[int, list[str]],
    k: int,
    m: int
) -> list[list[int]]:
    from collections import defaultdict

    # Build customer sets
    customer_sets = {}
    for customer_id, movies in history.items():
        if len(movies) >= k:
            customer_sets[customer_id] = set(movies[-k:])

    # Build inverted index: movie -> set of customers
    movie_to_customers = defaultdict(set)
    for customer_id, movie_set in customer_sets.items():
        for movie in movie_set:
            movie_to_customers[movie].add(customer_id)

    # Count common movies for each pair
    pair_counts = defaultdict(int)
    for movie, customers in movie_to_customers.items():
        customer_list = sorted(customers)
        for i in range(len(customer_list)):
            for j in range(i + 1, len(customer_list)):
                pair_counts[(customer_list[i], customer_list[j])] += 1

    # Filter pairs with at least m common movies
    result = [[c1, c2] for (c1, c2), count in pair_counts.items() if count >= m]
    return sorted(result)
```

### TypeScript Solution

```typescript
function findMovieFriendsII(
    history: Map<number, string[]>,
    k: number,
    m: number
): number[][] {
    const customerSets = new Map<number, Set<string>>();

    // Build sets of last k movies
    for (const [customerId, movies] of history) {
        if (movies.length >= k) {
            customerSets.set(customerId, new Set(movies.slice(-k)));
        }
    }

    const result: number[][] = [];
    const customers = Array.from(customerSets.keys()).sort((a, b) => a - b);

    // Compare all pairs
    for (let i = 0; i < customers.length; i++) {
        for (let j = i + 1; j < customers.length; j++) {
            const set1 = customerSets.get(customers[i])!;
            const set2 = customerSets.get(customers[j])!;

            // Count intersection
            let common = 0;
            for (const movie of set1) {
                if (set2.has(movie)) common++;
            }

            if (common >= m) {
                result.push([customers[i], customers[j]]);
            }
        }
    }

    return result;
}
```

### Interview Questions

1. **How does this differ from Part I?**
   - Part I requires exact sequence match (order matters). Part II requires set overlap (order doesn't matter).

2. **How to optimize for large datasets?**
   - Use MinHash/LSH for approximate similarity, or inverted index to avoid comparing dissimilar customers.

3. **What if m is very close to k?**
   - Most pairs will fail quickly. Early termination when common count can't reach m.

### Complexity Analysis

| Approach | Time | Space |
|----------|------|-------|
| Brute Force | O(n² × k) | O(n × k) |
| Inverted Index | O(n × k + pairs) | O(n × k) |

---

## 23. Reconstruct Itinerary

**Type:** Coding Problem (Graph)

**Problem:** Given a list of airline tickets where `tickets[i] = [from, to]`, reconstruct the itinerary starting from "JFK". You must use all tickets exactly once. If multiple valid itineraries exist, return the lexicographically smallest one.

**Focus:** Eulerian path, Hierholzer's algorithm, DFS with backtracking

### Examples

**Example 1:**
```
Input: tickets = [["MUC","LHR"],["JFK","MUC"],["SFO","SJC"],["LHR","SFO"]]
Output: ["JFK","MUC","LHR","SFO","SJC"]
```

**Example 2:**
```
Input: tickets = [["JFK","SFO"],["JFK","ATL"],["SFO","ATL"],["ATL","JFK"],["ATL","SFO"]]
Output: ["JFK","ATL","JFK","SFO","ATL","SFO"]
Explanation: Another valid itinerary is ["JFK","SFO","ATL","JFK","ATL","SFO"] but it's larger lexicographically.
```

### Constraints

- `1 <= tickets.length <= 300`
- All airports are 3 uppercase letters
- `tickets[i][0] != tickets[i][1]`
- At least one valid itinerary exists

### Key Insight

This is an **Eulerian path** problem. Use Hierholzer's algorithm:
1. Build adjacency list with destinations sorted in reverse order (for efficient pop)
2. DFS from JFK, always taking the smallest available destination
3. Add nodes to result in post-order (after exploring all edges)
4. Reverse the result

### Python Solution

```python
from collections import defaultdict

def findItinerary(tickets: list[list[str]]) -> list[str]:
    # Build adjacency list, sorted in reverse for efficient pop
    graph = defaultdict(list)
    for src, dst in sorted(tickets, reverse=True):
        graph[src].append(dst)

    result = []

    def dfs(airport: str) -> None:
        while graph[airport]:
            next_airport = graph[airport].pop()
            dfs(next_airport)
        result.append(airport)

    dfs("JFK")

    return result[::-1]


# Alternative: iterative with stack
def findItineraryIterative(tickets: list[list[str]]) -> list[str]:
    from collections import defaultdict

    graph = defaultdict(list)
    for src, dst in sorted(tickets, reverse=True):
        graph[src].append(dst)

    stack = ["JFK"]
    result = []

    while stack:
        while graph[stack[-1]]:
            stack.append(graph[stack[-1]].pop())
        result.append(stack.pop())

    return result[::-1]
```

### TypeScript Solution

```typescript
function findItinerary(tickets: string[][]): string[] {
    // Build adjacency list
    const graph = new Map<string, string[]>();

    // Sort tickets and build graph (reverse order for pop efficiency)
    tickets.sort((a, b) => b[1].localeCompare(a[1]));

    for (const [src, dst] of tickets) {
        if (!graph.has(src)) {
            graph.set(src, []);
        }
        graph.get(src)!.push(dst);
    }

    const result: string[] = [];

    function dfs(airport: string): void {
        const destinations = graph.get(airport) || [];
        while (destinations.length > 0) {
            const next = destinations.pop()!;
            dfs(next);
        }
        result.push(airport);
    }

    dfs("JFK");

    return result.reverse();
}
```

### Why Post-Order Works

Building in post-order handles dead-ends correctly:
- Visit all outgoing edges first
- Only add to result when stuck (no more edges)
- This ensures we don't get stuck with unused tickets

### Interview Questions

1. **Why sort in reverse order?**
   - Pop from end of list is O(1). Sorting reverse means smallest is at end, so we always take lexicographically smallest.

2. **What if no valid itinerary exists?**
   - Problem guarantees one exists. Otherwise, check if all nodes have in-degree = out-degree (except start/end for path).

3. **How to handle duplicate tickets?**
   - The algorithm naturally handles duplicates since we pop each ticket once.

### Complexity Analysis

| Metric | Value |
|--------|-------|
| Time | O(E log E) for sorting |
| Space | O(E) for graph and recursion |

**Note:** LeetCode problem #332

---

## 24. Auto-Expire Cache

**Type:** Coding Problem (Data Structure)

**Problem:** Design a key-value cache where entries automatically expire after a TTL (time-to-live). Support get, put with TTL, and efficient cleanup of expired entries.

**Focus:** HashMap with expiration, lazy vs eager cleanup, priority queue for expiration

### The Challenge

Implement an `AutoExpireCache` class:

- `put(key, value, ttl)` - Store key-value with TTL in seconds
- `get(key)` - Get value if exists and not expired
- `cleanup()` - Remove all expired entries
- `size()` - Number of non-expired entries

### Examples

```
cache = AutoExpireCache()
cache.put("a", 1, 10)    # Expires in 10 seconds
cache.put("b", 2, 5)     # Expires in 5 seconds

# At t=0
cache.get("a")           # returns 1
cache.get("b")           # returns 2

# At t=6
cache.get("a")           # returns 1
cache.get("b")           # returns None (expired)
```

### Python Solution

```python
import time
import heapq
from typing import Any, Optional

class AutoExpireCache:
    def __init__(self):
        self.cache: dict[str, tuple[Any, float]] = {}  # key -> (value, expire_time)
        self.expiry_heap: list[tuple[float, str]] = []  # (expire_time, key)

    def put(self, key: str, value: Any, ttl: int) -> None:
        """Store key-value pair with TTL in seconds."""
        expire_time = time.time() + ttl
        self.cache[key] = (value, expire_time)
        heapq.heappush(self.expiry_heap, (expire_time, key))

    def get(self, key: str) -> Optional[Any]:
        """Get value if key exists and not expired."""
        if key not in self.cache:
            return None

        value, expire_time = self.cache[key]

        if time.time() > expire_time:
            # Lazy deletion
            del self.cache[key]
            return None

        return value

    def cleanup(self) -> int:
        """Remove all expired entries. Returns count removed."""
        current_time = time.time()
        removed = 0

        while self.expiry_heap and self.expiry_heap[0][0] < current_time:
            expire_time, key = heapq.heappop(self.expiry_heap)

            # Check if this entry is still valid (not overwritten)
            if key in self.cache:
                _, stored_expire = self.cache[key]
                if stored_expire == expire_time:
                    del self.cache[key]
                    removed += 1

        return removed

    def size(self) -> int:
        """Return count of non-expired entries."""
        self.cleanup()
        return len(self.cache)


# Simpler version without heap (lazy cleanup only)
class AutoExpireCacheSimple:
    def __init__(self):
        self.cache: dict[str, tuple[Any, float]] = {}

    def put(self, key: str, value: Any, ttl: int) -> None:
        expire_time = time.time() + ttl
        self.cache[key] = (value, expire_time)

    def get(self, key: str) -> Optional[Any]:
        if key not in self.cache:
            return None

        value, expire_time = self.cache[key]

        if time.time() > expire_time:
            del self.cache[key]
            return None

        return value

    def cleanup(self) -> int:
        current_time = time.time()
        expired_keys = [
            k for k, (_, exp) in self.cache.items()
            if exp < current_time
        ]
        for key in expired_keys:
            del self.cache[key]
        return len(expired_keys)
```

### TypeScript Solution

```typescript
class AutoExpireCache<T> {
    private cache: Map<string, { value: T; expireTime: number }> = new Map();

    put(key: string, value: T, ttlSeconds: number): void {
        const expireTime = Date.now() + ttlSeconds * 1000;
        this.cache.set(key, { value, expireTime });
    }

    get(key: string): T | null {
        const entry = this.cache.get(key);

        if (!entry) return null;

        if (Date.now() > entry.expireTime) {
            this.cache.delete(key);
            return null;
        }

        return entry.value;
    }

    cleanup(): number {
        const now = Date.now();
        let removed = 0;

        for (const [key, entry] of this.cache) {
            if (now > entry.expireTime) {
                this.cache.delete(key);
                removed++;
            }
        }

        return removed;
    }

    size(): number {
        this.cleanup();
        return this.cache.size;
    }
}
```

### Interview Questions

1. **Lazy vs eager expiration?**
   - Lazy: Check on get(), saves CPU but wastes memory. Eager: Background thread/timer, uses CPU but saves memory.

2. **How to handle high write throughput?**
   - Batch cleanup periodically, use probabilistic cleanup (random sample check).

3. **How to add LRU eviction on top?**
   - Combine with doubly-linked list (like LRU Cache). Evict by either LRU or expiration.

4. **Thread safety?**
   - Use read-write locks, or concurrent hash map with atomic operations.

### Complexity Analysis

| Operation | Simple | With Heap |
|-----------|--------|-----------|
| put | O(1) | O(log n) |
| get | O(1) | O(1) |
| cleanup | O(n) | O(k log n) where k = expired |

---

## 25. Contains Duplicate

**Type:** Coding Problem (Array/HashSet)

**Problem:** Given an integer array `nums`, return true if any value appears more than once in the array, otherwise return false.

**Focus:** HashSet for O(1) lookups, early termination

### Examples

**Example 1:**
```
Input: nums = [1,2,3,1]
Output: true
```

**Example 2:**
```
Input: nums = [1,2,3,4]
Output: false
```

**Example 3:**
```
Input: nums = [1,1,1,3,3,4,3,2,4,2]
Output: true
```

### Constraints

- `1 <= nums.length <= 10^5`
- `-10^9 <= nums[i] <= 10^9`

### Python Solution

```python
def containsDuplicate(nums: list[int]) -> bool:
    seen = set()
    for num in nums:
        if num in seen:
            return True
        seen.add(num)
    return False

# One-liner
def containsDuplicateOneLiner(nums: list[int]) -> bool:
    return len(nums) != len(set(nums))
```

### TypeScript Solution

```typescript
function containsDuplicate(nums: number[]): boolean {
    const seen = new Set<number>();

    for (const num of nums) {
        if (seen.has(num)) {
            return true;
        }
        seen.add(num);
    }

    return false;
}

// One-liner
function containsDuplicateOneLiner(nums: number[]): boolean {
    return nums.length !== new Set(nums).size;
}
```

### Alternative: Sorting

```python
def containsDuplicateSorting(nums: list[int]) -> bool:
    nums.sort()
    for i in range(1, len(nums)):
        if nums[i] == nums[i - 1]:
            return True
    return False
```

### Interview Questions

1. **Trade-offs between approaches?**
   - HashSet: O(n) time, O(n) space
   - Sorting: O(n log n) time, O(1) space (in-place)

2. **What if the array is very large?**
   - External sorting, or probabilistic data structures (Bloom filter) for approximate answer.

### Complexity Analysis

| Approach | Time | Space |
|----------|------|-------|
| HashSet | O(n) | O(n) |
| Sorting | O(n log n) | O(1) |
| One-liner | O(n) | O(n) |

**Note:** LeetCode problem #217

---

## 26. Design the Data Model for a Promotion Posting System

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

## 27. User Engagement Patterns

**Type:** Coding Problem (Analytics)

**Problem:** Given user viewing data (user_id, content_id, watch_duration, timestamp), analyze engagement patterns to identify power users, trending content, and optimal content recommendations.

**Focus:** Data aggregation, grouping, statistical analysis

### The Challenge

Implement functions to analyze user engagement:

1. `getTopUsers(k)` - Get k users with highest total watch time
2. `getTrendingContent(window)` - Get content with most views in time window
3. `getUserEngagementScore(user_id)` - Calculate engagement score for user

### Examples

```
data = [
    {"user": 1, "content": "A", "duration": 30, "ts": 100},
    {"user": 1, "content": "B", "duration": 45, "ts": 200},
    {"user": 2, "content": "A", "duration": 60, "ts": 150},
    {"user": 1, "content": "A", "duration": 30, "ts": 300},
]

analyzer = EngagementAnalyzer(data)
analyzer.getTopUsers(1)           # [1] (total: 105 minutes)
analyzer.getTrendingContent(100, 250)  # "A" (3 views in window)
analyzer.getUserEngagementScore(1)     # High (frequent, diverse viewing)
```

### Python Solution

```python
from collections import defaultdict
from typing import Any
import heapq

class EngagementAnalyzer:
    def __init__(self, events: list[dict[str, Any]]):
        self.events = events
        self._preprocess()

    def _preprocess(self):
        """Build aggregated data structures."""
        self.user_watch_time = defaultdict(int)
        self.user_content = defaultdict(set)
        self.content_views = defaultdict(list)  # content -> [(timestamp, duration)]

        for event in self.events:
            user = event["user"]
            content = event["content"]
            duration = event["duration"]
            ts = event["ts"]

            self.user_watch_time[user] += duration
            self.user_content[user].add(content)
            self.content_views[content].append((ts, duration))

    def getTopUsers(self, k: int) -> list[int]:
        """Get k users with highest total watch time."""
        # Use heap for efficiency
        return heapq.nlargest(
            k,
            self.user_watch_time.keys(),
            key=lambda u: self.user_watch_time[u]
        )

    def getTrendingContent(self, start_ts: int, end_ts: int) -> list[str]:
        """Get content sorted by views in time window."""
        content_view_count = defaultdict(int)

        for content, views in self.content_views.items():
            for ts, _ in views:
                if start_ts <= ts <= end_ts:
                    content_view_count[content] += 1

        return sorted(
            content_view_count.keys(),
            key=lambda c: content_view_count[c],
            reverse=True
        )

    def getUserEngagementScore(self, user_id: int) -> float:
        """
        Calculate engagement score based on:
        - Total watch time (normalized)
        - Content diversity (unique content / total views)
        - Session frequency
        """
        if user_id not in self.user_watch_time:
            return 0.0

        watch_time = self.user_watch_time[user_id]
        unique_content = len(self.user_content[user_id])

        # Count sessions (events) for this user
        session_count = sum(1 for e in self.events if e["user"] == user_id)

        # Simple scoring formula
        max_watch_time = max(self.user_watch_time.values())
        max_diversity = max(len(c) for c in self.user_content.values())

        time_score = watch_time / max_watch_time if max_watch_time > 0 else 0
        diversity_score = unique_content / max_diversity if max_diversity > 0 else 0
        frequency_score = min(session_count / 10, 1.0)  # Cap at 10 sessions

        return (time_score * 0.4 + diversity_score * 0.3 + frequency_score * 0.3)
```

### TypeScript Solution

```typescript
interface ViewEvent {
    user: number;
    content: string;
    duration: number;
    ts: number;
}

class EngagementAnalyzer {
    private userWatchTime: Map<number, number> = new Map();
    private userContent: Map<number, Set<string>> = new Map();
    private contentViews: Map<string, Array<[number, number]>> = new Map();

    constructor(events: ViewEvent[]) {
        this.preprocess(events);
    }

    private preprocess(events: ViewEvent[]): void {
        for (const event of events) {
            // User watch time
            this.userWatchTime.set(
                event.user,
                (this.userWatchTime.get(event.user) || 0) + event.duration
            );

            // User content set
            if (!this.userContent.has(event.user)) {
                this.userContent.set(event.user, new Set());
            }
            this.userContent.get(event.user)!.add(event.content);

            // Content views
            if (!this.contentViews.has(event.content)) {
                this.contentViews.set(event.content, []);
            }
            this.contentViews.get(event.content)!.push([event.ts, event.duration]);
        }
    }

    getTopUsers(k: number): number[] {
        return Array.from(this.userWatchTime.entries())
            .sort((a, b) => b[1] - a[1])
            .slice(0, k)
            .map(([userId]) => userId);
    }

    getTrendingContent(startTs: number, endTs: number): string[] {
        const viewCounts = new Map<string, number>();

        for (const [content, views] of this.contentViews) {
            const count = views.filter(([ts]) => ts >= startTs && ts <= endTs).length;
            if (count > 0) {
                viewCounts.set(content, count);
            }
        }

        return Array.from(viewCounts.entries())
            .sort((a, b) => b[1] - a[1])
            .map(([content]) => content);
    }
}
```

### Interview Questions

1. **How to handle real-time updates?**
   - Streaming aggregation with windowed counts, use Apache Kafka + Flink.

2. **How to detect binge-watching behavior?**
   - Look for consecutive views of same series with short gaps.

3. **How to identify churning users?**
   - Analyze decline in watch time, longer gaps between sessions.

### Complexity Analysis

| Operation | Time | Space |
|-----------|------|-------|
| Preprocess | O(n) | O(n) |
| getTopUsers | O(u log k) | O(k) |
| getTrendingContent | O(v) | O(c) |

---

## 28. Spam Email Detection

**Type:** Coding Problem (ML/Design)

**Problem:** Given a large corpus of email correspondence history, design and implement a system to detect spam emails.

**Sections:**
- Detecting Spam Emails

---

## 29. Number Pairs That Match Target

**Type:** Coding Problem (Array)

**Problem:** Given an array of integers `nums` and an integer `target`, return all pairs of numbers where one of the four arithmetic operations (+, -, *, /) produces exactly the target value.

**Focus:** Pair enumeration, arithmetic operations, handling edge cases (division by zero)

### Examples

**Example 1:**
```
Input: nums = [1, 2, 3, 4, 5], target = 5
Output: [[1,4], [2,3], [1,5], [5,1], [10,2]]
Explanation:
- 1 + 4 = 5 ✓
- 2 + 3 = 5 ✓
- 1 * 5 = 5 ✓
- 5 / 1 = 5 ✓
- 10 / 2 = 5 (if 10 were in array)
```

**Example 2:**
```
Input: nums = [2, 3, 6, 9], target = 3
Output: [[6,3], [9,3], [6,2], [9,6]]
Explanation:
- 6 / 2 = 3 ✓
- 9 / 3 = 3 ✓
- 6 - 3 = 3 ✓
- 9 - 6 = 3 ✓
```

### Constraints

- Pairs can use the same index only if there are duplicates at different indices
- Division must result in an integer (no remainders)
- Avoid division by zero

### Python Solution

```python
def findPairsMatchingTarget(nums: list[int], target: int) -> list[list[int]]:
    result = []
    n = len(nums)
    seen_pairs = set()

    for i in range(n):
        for j in range(n):
            if i == j:
                continue

            a, b = nums[i], nums[j]
            pair_key = (min(a, b), max(a, b))  # For deduplication

            # Check all operations
            operations = []

            # Addition: a + b = target
            if a + b == target:
                operations.append((a, b, '+'))

            # Subtraction: a - b = target
            if a - b == target:
                operations.append((a, b, '-'))

            # Multiplication: a * b = target
            if a * b == target:
                operations.append((a, b, '*'))

            # Division: a / b = target (b != 0, no remainder)
            if b != 0 and a % b == 0 and a // b == target:
                operations.append((a, b, '/'))

            for op_a, op_b, op in operations:
                result_key = (op_a, op_b, op)
                if result_key not in seen_pairs:
                    seen_pairs.add(result_key)
                    result.append([op_a, op_b])

    return result


# Optimized with hash lookups
def findPairsMatchingTargetOptimized(
    nums: list[int],
    target: int
) -> list[tuple[int, int, str]]:
    """Returns pairs with the operation that achieves target."""
    from collections import Counter

    result = []
    num_count = Counter(nums)
    num_set = set(nums)

    for a in num_set:
        # Addition: a + b = target -> b = target - a
        b = target - a
        if b in num_set:
            if a != b or num_count[a] >= 2:
                result.append((a, b, '+'))

        # Subtraction: a - b = target -> b = a - target
        b = a - target
        if b in num_set:
            if a != b or num_count[a] >= 2:
                result.append((a, b, '-'))

        # Multiplication: a * b = target -> b = target / a
        if a != 0 and target % a == 0:
            b = target // a
            if b in num_set:
                if a != b or num_count[a] >= 2:
                    result.append((a, b, '*'))

        # Division: a / b = target -> b = a / target
        if target != 0 and a % target == 0:
            b = a // target
            if b in num_set and b != 0:
                if a != b or num_count[a] >= 2:
                    result.append((a, b, '/'))

    return result
```

### TypeScript Solution

```typescript
function findPairsMatchingTarget(
    nums: number[],
    target: number
): Array<[number, number, string]> {
    const result: Array<[number, number, string]> = [];
    const numSet = new Set(nums);
    const numCount = new Map<number, number>();

    for (const num of nums) {
        numCount.set(num, (numCount.get(num) || 0) + 1);
    }

    const seen = new Set<string>();

    for (const a of numSet) {
        // Addition
        const addB = target - a;
        if (numSet.has(addB) && !seen.has(`${a},${addB},+`)) {
            if (a !== addB || numCount.get(a)! >= 2) {
                result.push([a, addB, '+']);
                seen.add(`${a},${addB},+`);
            }
        }

        // Subtraction
        const subB = a - target;
        if (numSet.has(subB) && !seen.has(`${a},${subB},-`)) {
            if (a !== subB || numCount.get(a)! >= 2) {
                result.push([a, subB, '-']);
                seen.add(`${a},${subB},-`);
            }
        }

        // Multiplication
        if (a !== 0 && target % a === 0) {
            const mulB = target / a;
            if (numSet.has(mulB) && !seen.has(`${a},${mulB},*`)) {
                if (a !== mulB || numCount.get(a)! >= 2) {
                    result.push([a, mulB, '*']);
                    seen.add(`${a},${mulB},*`);
                }
            }
        }

        // Division
        if (target !== 0 && a % target === 0) {
            const divB = a / target;
            if (divB !== 0 && numSet.has(divB) && !seen.has(`${a},${divB},/`)) {
                if (a !== divB || numCount.get(a)! >= 2) {
                    result.push([a, divB, '/']);
                    seen.add(`${a},${divB},/`);
                }
            }
        }
    }

    return result;
}
```

### Interview Questions

1. **How to handle floating point targets?**
   - Use epsilon comparison for division results.

2. **What if we need triplets instead of pairs?**
   - Three-sum style approach with two-pointer after sorting.

3. **How to handle very large arrays?**
   - Hash-based approach is O(n), efficient for large inputs.

### Complexity Analysis

| Approach | Time | Space |
|----------|------|-------|
| Brute Force | O(n²) | O(n²) for results |
| Hash Optimized | O(n) | O(n) |

---

## 30. First Missing Positive

**Type:** Coding Problem (Array)

**Problem:** Given an unsorted integer array `nums`, return the smallest positive integer that does not appear in `nums`.

Your solution must run in **O(n) time** and use **O(1) extra space**.

### Examples

**Example 1:**
```
Input: nums = [1,2,0]
Output: 3
Explanation: The values 1 and 2 are present, so the smallest missing positive is 3.
```

**Example 2:**
```
Input: nums = [3,4,-1,1]
Output: 2
```

**Example 3:**
```
Input: nums = [7,8,9,11,12]
Output: 1
```

### Constraints

- `1 <= nums.length <= 10^5`
- `-2^31 <= nums[i] <= 2^31 - 1`

### Key Insight

For an array of length `n`, the answer must be in the range `[1, n+1]`. If all numbers 1 to n are present, the answer is `n+1`. Otherwise, it's some number in `[1, n]`.

The trick to achieve O(1) space is to use the array itself as a hash table by placing each number in its "correct" position (number `i` at index `i-1`).

### Python Solution

```python
def firstMissingPositive(nums: list[int]) -> int:
    n = len(nums)

    # Step 1: Place each number in its correct position
    # Number i should be at index i-1
    for i in range(n):
        while 1 <= nums[i] <= n and nums[nums[i] - 1] != nums[i]:
            # Swap nums[i] to its correct position
            correct_idx = nums[i] - 1
            nums[i], nums[correct_idx] = nums[correct_idx], nums[i]

    # Step 2: Find the first position where nums[i] != i + 1
    for i in range(n):
        if nums[i] != i + 1:
            return i + 1

    # All positions are correct, answer is n + 1
    return n + 1
```

### TypeScript Solution

```typescript
function firstMissingPositive(nums: number[]): number {
    const n = nums.length;

    // Step 1: Place each number in its correct position
    for (let i = 0; i < n; i++) {
        while (nums[i] >= 1 && nums[i] <= n && nums[nums[i] - 1] !== nums[i]) {
            const correctIdx = nums[i] - 1;
            [nums[i], nums[correctIdx]] = [nums[correctIdx], nums[i]];
        }
    }

    // Step 2: Find first position where nums[i] != i + 1
    for (let i = 0; i < n; i++) {
        if (nums[i] !== i + 1) {
            return i + 1;
        }
    }

    return n + 1;
}
```

### Walkthrough

For `nums = [3, 4, -1, 1]`:

| Step | Array State | Action |
|------|-------------|--------|
| Initial | `[3, 4, -1, 1]` | Start at index 0 |
| Swap | `[-1, 4, 3, 1]` | 3 goes to index 2 |
| Swap | `[-1, 1, 3, 4]` | 4 goes to index 3 |
| Swap | `[1, -1, 3, 4]` | 1 goes to index 0 |
| Scan | `[1, -1, 3, 4]` | Index 1 has -1 ≠ 2 |

**Result:** 2

### Interview Questions

1. **Why does this approach work in O(n) time despite nested loops?**
   - Each element is swapped at most once to its correct position. After being placed correctly, it never moves again. So total swaps across all iterations is at most `n`.

2. **Why check `nums[nums[i] - 1] != nums[i]` in the while condition?**
   - This prevents infinite loops when duplicates exist. If the target position already has the correct value, we stop swapping.

3. **What if we used a HashSet instead?**
   - HashSet would give O(n) time but O(n) space. The in-place swap technique achieves O(1) space.

4. **Can negative numbers or zeros affect the result?**
   - No. We only care about positive integers in range `[1, n]`. Negatives and zeros are ignored during placement.

### Complexity Analysis

| Metric | Value |
|--------|-------|
| Time | O(n) |
| Space | O(1) |

**Note:** Classic LeetCode problem #41

---

## 31. Parallel Courses II

**Type:** Coding Problem (Graph/Bitmask DP)

**Problem:** Given n courses with prerequisites and a limit of k courses per semester, return the minimum number of semesters needed to complete all courses.

**Focus:** Bitmask dynamic programming, subset enumeration, NP-hard optimization

### The Challenge

You are given an integer `n` (courses labeled 1 to n) and an array `relations` where `relations[i] = [prevCourse, nextCourse]` means you must complete `prevCourse` before `nextCourse`.

In one semester, you can take **at most k courses** as long as all prerequisites are completed.

Return the minimum number of semesters needed to take all courses.

### Examples

**Example 1:**
```
Input: n = 4, relations = [[2,1],[3,1],[1,4]], k = 2
Output: 3
Explanation:
- Semester 1: Take courses 2 and 3
- Semester 2: Take course 1
- Semester 3: Take course 4
```

**Example 2:**
```
Input: n = 5, relations = [[2,1],[3,1],[4,1],[1,5]], k = 2
Output: 4
Explanation:
- Semester 1: Take 2, 3
- Semester 2: Take 4 (still need 1 more prereq for course 1)
- Semester 3: Take 1
- Semester 4: Take 5
```

**Example 3:**
```
Input: n = 11, relations = [], k = 2
Output: 6
Explanation: No prerequisites, just ceil(11/2) = 6 semesters.
```

### Constraints

- `1 <= n <= 15`
- `1 <= k <= n`
- `0 <= relations.length <= n * (n - 1) / 2`
- The graph is a **DAG** (no cycles guaranteed)

### Key Insight

The constraint `n <= 15` strongly hints at **bitmask DP**. Unlike Parallel Courses I, greedy doesn't work here because choosing which k courses to take affects future options.

**State:** A bitmask representing which courses have been completed.
**Transition:** For each state, find all available courses (prerequisites satisfied), try all subsets of size ≤ k.

### Python Solution

```python
def minNumberOfSemesters(n: int, relations: list[list[int]], k: int) -> int:
    # prereq[i] = bitmask of prerequisites for course i
    prereq = [0] * n
    for prev, next_course in relations:
        prereq[next_course - 1] |= (1 << (prev - 1))

    # dp[mask] = minimum semesters to complete courses in mask
    dp = [float('inf')] * (1 << n)
    dp[0] = 0

    for mask in range(1 << n):
        if dp[mask] == float('inf'):
            continue

        # Find all available courses (prerequisites satisfied)
        available = 0
        for i in range(n):
            if not (mask & (1 << i)):  # Course i not taken yet
                if (mask & prereq[i]) == prereq[i]:  # All prereqs done
                    available |= (1 << i)

        # Try all subsets of available courses with size <= k
        subset = available
        while subset > 0:
            if bin(subset).count('1') <= k:
                new_mask = mask | subset
                dp[new_mask] = min(dp[new_mask], dp[mask] + 1)
            # Iterate to next subset
            subset = (subset - 1) & available

    return dp[(1 << n) - 1]
```

### TypeScript Solution

```typescript
function minNumberOfSemesters(n: number, relations: number[][], k: number): number {
    // prereq[i] = bitmask of prerequisites for course i
    const prereq: number[] = new Array(n).fill(0);
    for (const [prev, next] of relations) {
        prereq[next - 1] |= (1 << (prev - 1));
    }

    // dp[mask] = minimum semesters to complete courses in mask
    const dp: number[] = new Array(1 << n).fill(Infinity);
    dp[0] = 0;

    for (let mask = 0; mask < (1 << n); mask++) {
        if (dp[mask] === Infinity) continue;

        // Find available courses
        let available = 0;
        for (let i = 0; i < n; i++) {
            if (!(mask & (1 << i))) {  // Not taken
                if ((mask & prereq[i]) === prereq[i]) {  // Prereqs done
                    available |= (1 << i);
                }
            }
        }

        // Try all subsets of size <= k
        let subset = available;
        while (subset > 0) {
            if (countBits(subset) <= k) {
                const newMask = mask | subset;
                dp[newMask] = Math.min(dp[newMask], dp[mask] + 1);
            }
            subset = (subset - 1) & available;
        }
    }

    return dp[(1 << n) - 1];
}

function countBits(n: number): number {
    let count = 0;
    while (n) {
        count += n & 1;
        n >>= 1;
    }
    return count;
}
```

### Walkthrough

For `n = 4, relations = [[2,1],[3,1],[1,4]], k = 2`:

| State (binary) | Completed | Available | Best Subset | Semesters |
|----------------|-----------|-----------|-------------|-----------|
| `0000` | none | 2, 3 | {2,3} | 0 → 1 |
| `0110` | 2, 3 | 1 | {1} | 1 → 2 |
| `0111` | 1, 2, 3 | 4 | {4} | 2 → 3 |
| `1111` | all | - | - | **3** |

### Interview Questions

1. **Why doesn't greedy work for this problem?**
   - With k limit, taking courses that unlock the most dependencies isn't always optimal. You might need to delay certain courses to parallelize better in future semesters.

2. **How does the subset enumeration trick work?**
   - `subset = (subset - 1) & available` generates all non-empty subsets of `available` in decreasing order. This is a standard bitmask technique.

3. **What's the time complexity?**
   - O(3^n) because for each of 2^n states, we enumerate subsets of available courses. The total number of (mask, subset) pairs is bounded by 3^n.

4. **How would you optimize for larger n?**
   - For n > 15, this approach won't work. You'd need heuristics, branch-and-bound, or approximate algorithms. This is NP-hard in general.

5. **Difference from Parallel Courses I?**
   - Part I has no k limit, so BFS level-order traversal works in O(V+E). Part II requires exponential-time DP due to the k constraint.

### Complexity Analysis

| Metric | Value |
|--------|-------|
| Time | O(3^n) |
| Space | O(2^n) |

**Note:** LeetCode problem #1494

---

## 32. LRU Cache

**Type:** Coding Problem (Data Structure Design)

**Problem:** Design a data structure that follows the constraints of a Least Recently Used (LRU) cache. Implement the `LRUCache` class with `get` and `put` operations in O(1) time complexity.

**Focus:** Hash map + doubly linked list, cache eviction policy

### The Challenge

Implement the `LRUCache` class:

- `LRUCache(int capacity)` - Initialize the cache with positive capacity
- `int get(int key)` - Return the value if key exists, otherwise return -1
- `void put(int key, int value)` - Update or insert the value. If capacity is exceeded, evict the least recently used key.

Both operations must run in **O(1)** average time complexity.

### Examples

```
Input:
["LRUCache", "put", "put", "get", "put", "get", "put", "get", "get", "get"]
[[2], [1, 1], [2, 2], [1], [3, 3], [2], [4, 4], [1], [3], [4]]

Output: [null, null, null, 1, null, -1, null, -1, 3, 4]

Explanation:
LRUCache cache = new LRUCache(2);
cache.put(1, 1);    // cache: {1=1}
cache.put(2, 2);    // cache: {1=1, 2=2}
cache.get(1);       // returns 1, cache: {2=2, 1=1}
cache.put(3, 3);    // evicts key 2, cache: {1=1, 3=3}
cache.get(2);       // returns -1 (not found)
cache.put(4, 4);    // evicts key 1, cache: {3=3, 4=4}
cache.get(1);       // returns -1
cache.get(3);       // returns 3
cache.get(4);       // returns 4
```

### Constraints

- `1 <= capacity <= 3000`
- `0 <= key <= 10^4`
- `0 <= value <= 10^5`
- At most `2 * 10^5` calls to `get` and `put`

### Key Insight

Use a **hash map** for O(1) lookups and a **doubly linked list** for O(1) insertion/deletion. The list maintains access order - most recently used at head, least recently used at tail.

### Python Solution

```python
class Node:
    def __init__(self, key: int = 0, val: int = 0):
        self.key = key
        self.val = val
        self.prev = None
        self.next = None

class LRUCache:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache = {}  # key -> Node

        # Dummy head and tail for easier list manipulation
        self.head = Node()
        self.tail = Node()
        self.head.next = self.tail
        self.tail.prev = self.head

    def _remove(self, node: Node) -> None:
        """Remove node from linked list."""
        node.prev.next = node.next
        node.next.prev = node.prev

    def _add_to_head(self, node: Node) -> None:
        """Add node right after head (most recent)."""
        node.next = self.head.next
        node.prev = self.head
        self.head.next.prev = node
        self.head.next = node

    def get(self, key: int) -> int:
        if key not in self.cache:
            return -1

        node = self.cache[key]
        # Move to head (most recently used)
        self._remove(node)
        self._add_to_head(node)
        return node.val

    def put(self, key: int, value: int) -> None:
        if key in self.cache:
            # Update existing node
            node = self.cache[key]
            node.val = value
            self._remove(node)
            self._add_to_head(node)
        else:
            # Create new node
            node = Node(key, value)
            self.cache[key] = node
            self._add_to_head(node)

            # Evict if over capacity
            if len(self.cache) > self.capacity:
                lru = self.tail.prev
                self._remove(lru)
                del self.cache[lru.key]
```

### TypeScript Solution

```typescript
class LRUNode {
    key: number;
    val: number;
    prev: LRUNode | null = null;
    next: LRUNode | null = null;

    constructor(key: number = 0, val: number = 0) {
        this.key = key;
        this.val = val;
    }
}

class LRUCache {
    private capacity: number;
    private cache: Map<number, LRUNode> = new Map();
    private head: LRUNode = new LRUNode();
    private tail: LRUNode = new LRUNode();

    constructor(capacity: number) {
        this.capacity = capacity;
        this.head.next = this.tail;
        this.tail.prev = this.head;
    }

    private remove(node: LRUNode): void {
        node.prev!.next = node.next;
        node.next!.prev = node.prev;
    }

    private addToHead(node: LRUNode): void {
        node.next = this.head.next;
        node.prev = this.head;
        this.head.next!.prev = node;
        this.head.next = node;
    }

    get(key: number): number {
        if (!this.cache.has(key)) return -1;

        const node = this.cache.get(key)!;
        this.remove(node);
        this.addToHead(node);
        return node.val;
    }

    put(key: number, value: number): void {
        if (this.cache.has(key)) {
            const node = this.cache.get(key)!;
            node.val = value;
            this.remove(node);
            this.addToHead(node);
        } else {
            const node = new LRUNode(key, value);
            this.cache.set(key, node);
            this.addToHead(node);

            if (this.cache.size > this.capacity) {
                const lru = this.tail.prev!;
                this.remove(lru);
                this.cache.delete(lru.key);
            }
        }
    }
}
```

### Interview Questions

1. **Why use dummy head and tail nodes?**
   - They eliminate edge cases for empty list or single-element operations. No null checks needed.

2. **Why store the key in the Node?**
   - When evicting the LRU node, we need its key to remove it from the hash map.

3. **How would you implement this thread-safe?**
   - Add a lock around get/put operations, or use a concurrent hash map with fine-grained locking per node.

4. **Netflix use case?**
   - Caching thumbnails, user session data, recently watched titles, or CDN content metadata.

### Complexity Analysis

| Operation | Time | Space |
|-----------|------|-------|
| get | O(1) | O(capacity) |
| put | O(1) | O(capacity) |

**Note:** LeetCode problem #146

---

## 33. Merge K Sorted Lists

**Type:** Coding Problem (Heap/Divide & Conquer)

**Problem:** You are given an array of `k` linked lists, each sorted in ascending order. Merge all the linked lists into one sorted linked list and return it.

**Focus:** Min-heap, divide and conquer, pointer manipulation

### Examples

**Example 1:**
```
Input: lists = [[1,4,5],[1,3,4],[2,6]]
Output: [1,1,2,3,4,4,5,6]
Explanation: Merging the three sorted lists produces one sorted list.
```

**Example 2:**
```
Input: lists = []
Output: []
```

**Example 3:**
```
Input: lists = [[]]
Output: []
```

### Constraints

- `k == lists.length`
- `0 <= k <= 10^4`
- `0 <= lists[i].length <= 500`
- `-10^4 <= lists[i][j] <= 10^4`
- Each `lists[i]` is sorted in ascending order
- Total nodes across all lists ≤ 10^4

### Key Insight

Use a **min-heap** to always get the smallest element among the k list heads. This gives O(N log k) time where N is total elements.

### Python Solution

```python
import heapq
from typing import List, Optional

class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

def mergeKLists(lists: List[Optional[ListNode]]) -> Optional[ListNode]:
    # Min-heap: (value, index, node)
    # Index is used as tiebreaker since nodes aren't comparable
    heap = []

    for i, node in enumerate(lists):
        if node:
            heapq.heappush(heap, (node.val, i, node))

    dummy = ListNode()
    current = dummy

    while heap:
        val, i, node = heapq.heappop(heap)
        current.next = node
        current = current.next

        if node.next:
            heapq.heappush(heap, (node.next.val, i, node.next))

    return dummy.next
```

### TypeScript Solution

```typescript
class ListNode {
    val: number;
    next: ListNode | null;
    constructor(val?: number, next?: ListNode | null) {
        this.val = val ?? 0;
        this.next = next ?? null;
    }
}

function mergeKLists(lists: Array<ListNode | null>): ListNode | null {
    // Filter out null lists
    const heads = lists.filter(node => node !== null) as ListNode[];
    if (heads.length === 0) return null;

    // Use divide and conquer approach
    while (heads.length > 1) {
        const merged: ListNode[] = [];
        for (let i = 0; i < heads.length; i += 2) {
            const l1 = heads[i];
            const l2 = i + 1 < heads.length ? heads[i + 1] : null;
            merged.push(mergeTwoLists(l1, l2));
        }
        heads.length = 0;
        heads.push(...merged);
    }

    return heads[0];
}

function mergeTwoLists(l1: ListNode | null, l2: ListNode | null): ListNode {
    const dummy = new ListNode();
    let current = dummy;

    while (l1 && l2) {
        if (l1.val <= l2.val) {
            current.next = l1;
            l1 = l1.next;
        } else {
            current.next = l2;
            l2 = l2.next;
        }
        current = current.next;
    }

    current.next = l1 || l2;
    return dummy.next!;
}
```

### Interview Questions

1. **Why use a heap instead of merging pairs?**
   - Both work. Heap is O(N log k), divide-and-conquer is also O(N log k). Heap uses O(k) space vs O(1) for iterative merge.

2. **What's the index in the heap tuple for?**
   - Python's heapq compares tuples element by element. If values are equal, it compares the next element. Nodes aren't comparable, so we add an index as a tiebreaker.

3. **Netflix use case?**
   - Merging sorted streams of user activity logs, combining ranked recommendation lists from different algorithms.

### Complexity Analysis

| Metric | Value |
|--------|-------|
| Time | O(N log k) |
| Space | O(k) for heap |

**Note:** LeetCode problem #23

---

## 34. Median of Two Sorted Arrays

**Type:** Coding Problem (Binary Search)

**Problem:** Given two sorted arrays `nums1` and `nums2`, return the median of the two sorted arrays. The overall runtime must be **O(log(m+n))**.

**Focus:** Binary search on partitions, edge case handling

### Examples

**Example 1:**
```
Input: nums1 = [1,3], nums2 = [2]
Output: 2.0
Explanation: Merged = [1,2,3], median = 2.
```

**Example 2:**
```
Input: nums1 = [1,2], nums2 = [3,4]
Output: 2.5
Explanation: Merged = [1,2,3,4], median = (2+3)/2 = 2.5.
```

### Constraints

- `0 <= m, n <= 1000`
- `1 <= m + n <= 2000`
- `-10^6 <= nums1[i], nums2[i] <= 10^6`

### Key Insight

Binary search on the smaller array to find a partition where:
- All elements on the left side ≤ all elements on the right side
- Left side has exactly `(m + n + 1) / 2` elements

### Python Solution

```python
def findMedianSortedArrays(nums1: list[int], nums2: list[int]) -> float:
    # Ensure nums1 is the smaller array
    if len(nums1) > len(nums2):
        nums1, nums2 = nums2, nums1

    m, n = len(nums1), len(nums2)
    half = (m + n + 1) // 2

    left, right = 0, m

    while left <= right:
        i = (left + right) // 2  # Partition index in nums1
        j = half - i              # Partition index in nums2

        # Edge values (use infinity for out of bounds)
        nums1_left = nums1[i - 1] if i > 0 else float('-inf')
        nums1_right = nums1[i] if i < m else float('inf')
        nums2_left = nums2[j - 1] if j > 0 else float('-inf')
        nums2_right = nums2[j] if j < n else float('inf')

        # Check if partition is correct
        if nums1_left <= nums2_right and nums2_left <= nums1_right:
            # Found correct partition
            if (m + n) % 2 == 1:
                return max(nums1_left, nums2_left)
            else:
                return (max(nums1_left, nums2_left) +
                        min(nums1_right, nums2_right)) / 2
        elif nums1_left > nums2_right:
            right = i - 1  # Move partition left in nums1
        else:
            left = i + 1   # Move partition right in nums1

    return 0.0  # Should never reach here
```

### TypeScript Solution

```typescript
function findMedianSortedArrays(nums1: number[], nums2: number[]): number {
    // Ensure nums1 is smaller
    if (nums1.length > nums2.length) {
        [nums1, nums2] = [nums2, nums1];
    }

    const m = nums1.length, n = nums2.length;
    const half = Math.floor((m + n + 1) / 2);

    let left = 0, right = m;

    while (left <= right) {
        const i = Math.floor((left + right) / 2);
        const j = half - i;

        const nums1Left = i > 0 ? nums1[i - 1] : -Infinity;
        const nums1Right = i < m ? nums1[i] : Infinity;
        const nums2Left = j > 0 ? nums2[j - 1] : -Infinity;
        const nums2Right = j < n ? nums2[j] : Infinity;

        if (nums1Left <= nums2Right && nums2Left <= nums1Right) {
            if ((m + n) % 2 === 1) {
                return Math.max(nums1Left, nums2Left);
            }
            return (Math.max(nums1Left, nums2Left) +
                    Math.min(nums1Right, nums2Right)) / 2;
        } else if (nums1Left > nums2Right) {
            right = i - 1;
        } else {
            left = i + 1;
        }
    }

    return 0;
}
```

### Interview Questions

1. **Why binary search on the smaller array?**
   - Reduces the search space. If we search on size m, we do O(log m) operations. Choosing the smaller array minimizes this.

2. **Why use infinity for out-of-bounds?**
   - It ensures comparisons always work correctly at boundaries without special cases.

3. **Netflix use case?**
   - Finding median latency from multiple data centers, computing percentiles for A/B test metrics.

### Complexity Analysis

| Metric | Value |
|--------|-------|
| Time | O(log(min(m, n))) |
| Space | O(1) |

**Note:** LeetCode problem #4

---

## 35. Maximum Subarray

**Type:** Coding Problem (Dynamic Programming)

**Problem:** Given an integer array `nums`, find the subarray with the largest sum and return its sum.

**Focus:** Kadane's algorithm, DP optimization

### Examples

**Example 1:**
```
Input: nums = [-2,1,-3,4,-1,2,1,-5,4]
Output: 6
Explanation: Subarray [4,-1,2,1] has the largest sum 6.
```

**Example 2:**
```
Input: nums = [1]
Output: 1
```

**Example 3:**
```
Input: nums = [5,4,-1,7,8]
Output: 23
Explanation: Entire array has the largest sum.
```

### Constraints

- `1 <= nums.length <= 10^5`
- `-10^4 <= nums[i] <= 10^4`

### Key Insight

**Kadane's Algorithm:** At each position, decide whether to:
1. Extend the previous subarray (add current element)
2. Start a new subarray from current element

The choice depends on whether the previous sum is positive.

### Python Solution

```python
def maxSubArray(nums: list[int]) -> int:
    max_sum = nums[0]
    current_sum = nums[0]

    for i in range(1, len(nums)):
        # Either extend previous subarray or start fresh
        current_sum = max(nums[i], current_sum + nums[i])
        max_sum = max(max_sum, current_sum)

    return max_sum
```

### TypeScript Solution

```typescript
function maxSubArray(nums: number[]): number {
    let maxSum = nums[0];
    let currentSum = nums[0];

    for (let i = 1; i < nums.length; i++) {
        currentSum = Math.max(nums[i], currentSum + nums[i]);
        maxSum = Math.max(maxSum, currentSum);
    }

    return maxSum;
}
```

### Divide and Conquer Alternative

```python
def maxSubArrayDivideConquer(nums: list[int]) -> int:
    def helper(left: int, right: int) -> int:
        if left == right:
            return nums[left]

        mid = (left + right) // 2

        # Max sum crossing the middle
        left_sum = float('-inf')
        curr = 0
        for i in range(mid, left - 1, -1):
            curr += nums[i]
            left_sum = max(left_sum, curr)

        right_sum = float('-inf')
        curr = 0
        for i in range(mid + 1, right + 1):
            curr += nums[i]
            right_sum = max(right_sum, curr)

        cross_sum = left_sum + right_sum

        return max(helper(left, mid), helper(mid + 1, right), cross_sum)

    return helper(0, len(nums) - 1)
```

### Interview Questions

1. **Why is this O(n) instead of O(n²)?**
   - We make a single pass, deciding at each element whether to extend or restart. No nested loops.

2. **How would you return the actual subarray?**
   - Track start/end indices. Update start when we restart, update end when we find a new maximum.

3. **Follow-up: What if the array is circular?**
   - Two cases: max subarray doesn't wrap (use Kadane's) or wraps (total sum minus minimum subarray). Take the maximum.

### Complexity Analysis

| Approach | Time | Space |
|----------|------|-------|
| Kadane's | O(n) | O(1) |
| Divide & Conquer | O(n log n) | O(log n) |

**Note:** LeetCode problem #53

---

## 36. Number of Islands

**Type:** Coding Problem (Graph/DFS)

**Problem:** Given an `m x n` 2D grid of `'1'`s (land) and `'0'`s (water), return the number of islands. An island is surrounded by water and is formed by connecting adjacent lands horizontally or vertically.

**Focus:** DFS/BFS flood fill, connected components

### Examples

**Example 1:**
```
Input: grid = [
  ["1","1","1","1","0"],
  ["1","1","0","1","0"],
  ["1","1","0","0","0"],
  ["0","0","0","0","0"]
]
Output: 1
```

**Example 2:**
```
Input: grid = [
  ["1","1","0","0","0"],
  ["1","1","0","0","0"],
  ["0","0","1","0","0"],
  ["0","0","0","1","1"]
]
Output: 3
```

### Constraints

- `m == grid.length`
- `n == grid[i].length`
- `1 <= m, n <= 300`
- `grid[i][j]` is `'0'` or `'1'`

### Key Insight

Use DFS/BFS to "flood fill" each island. When we find a '1', increment count and mark all connected '1's as visited (change to '0' or use a visited set).

### Python Solution

```python
def numIslands(grid: list[list[str]]) -> int:
    if not grid:
        return 0

    rows, cols = len(grid), len(grid[0])
    count = 0

    def dfs(r: int, c: int) -> None:
        # Boundary check and water check
        if r < 0 or r >= rows or c < 0 or c >= cols or grid[r][c] == '0':
            return

        # Mark as visited (sink the island)
        grid[r][c] = '0'

        # Explore all 4 directions
        dfs(r + 1, c)
        dfs(r - 1, c)
        dfs(r, c + 1)
        dfs(r, c - 1)

    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == '1':
                count += 1
                dfs(r, c)

    return count
```

### TypeScript Solution

```typescript
function numIslands(grid: string[][]): number {
    if (grid.length === 0) return 0;

    const rows = grid.length;
    const cols = grid[0].length;
    let count = 0;

    function dfs(r: number, c: number): void {
        if (r < 0 || r >= rows || c < 0 || c >= cols || grid[r][c] === '0') {
            return;
        }

        grid[r][c] = '0';  // Mark visited

        dfs(r + 1, c);
        dfs(r - 1, c);
        dfs(r, c + 1);
        dfs(r, c - 1);
    }

    for (let r = 0; r < rows; r++) {
        for (let c = 0; c < cols; c++) {
            if (grid[r][c] === '1') {
                count++;
                dfs(r, c);
            }
        }
    }

    return count;
}
```

### BFS Alternative

```python
from collections import deque

def numIslandsBFS(grid: list[list[str]]) -> int:
    if not grid:
        return 0

    rows, cols = len(grid), len(grid[0])
    count = 0

    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == '1':
                count += 1
                queue = deque([(r, c)])
                grid[r][c] = '0'

                while queue:
                    row, col = queue.popleft()
                    for dr, dc in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                        nr, nc = row + dr, col + dc
                        if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] == '1':
                            queue.append((nr, nc))
                            grid[nr][nc] = '0'

    return count
```

### Interview Questions

1. **DFS vs BFS - which is better?**
   - Same time complexity. DFS uses call stack (risk of stack overflow for huge grids), BFS uses explicit queue. BFS is iterative and safer for very large inputs.

2. **How to avoid modifying the input grid?**
   - Use a separate `visited` set of coordinates, but this increases space to O(m×n).

3. **Netflix use case?**
   - Clustering user segments, identifying connected regions in recommendation graphs, analyzing content viewing patterns.

4. **Follow-up: Count the size of the largest island?**
   - Return the count from DFS instead of just marking visited.

### Complexity Analysis

| Metric | Value |
|--------|-------|
| Time | O(m × n) |
| Space | O(m × n) worst case for DFS stack |

**Note:** LeetCode problem #200

---

## 37. Longest Palindromic Substring

**Type:** Coding Problem (String/DP)

**Problem:** Given a string `s`, return the longest palindromic substring in `s`.

**Focus:** Expand around center, dynamic programming

### Examples

**Example 1:**
```
Input: s = "babad"
Output: "bab" (or "aba")
```

**Example 2:**
```
Input: s = "cbbd"
Output: "bb"
```

### Constraints

- `1 <= s.length <= 1000`
- `s` consists of only digits and English letters

### Key Insight

**Expand Around Center:** A palindrome mirrors around its center. There are `2n - 1` possible centers (n single characters + n-1 gaps between characters for even-length palindromes).

### Python Solution

```python
def longestPalindrome(s: str) -> str:
    if len(s) < 2:
        return s

    start, max_len = 0, 1

    def expand_around_center(left: int, right: int) -> int:
        """Expand and return the length of palindrome."""
        while left >= 0 and right < len(s) and s[left] == s[right]:
            left -= 1
            right += 1
        return right - left - 1

    for i in range(len(s)):
        # Odd length palindrome (single center)
        len1 = expand_around_center(i, i)
        # Even length palindrome (gap center)
        len2 = expand_around_center(i, i + 1)

        curr_len = max(len1, len2)
        if curr_len > max_len:
            max_len = curr_len
            start = i - (curr_len - 1) // 2

    return s[start:start + max_len]
```

### TypeScript Solution

```typescript
function longestPalindrome(s: string): string {
    if (s.length < 2) return s;

    let start = 0, maxLen = 1;

    function expandAroundCenter(left: number, right: number): number {
        while (left >= 0 && right < s.length && s[left] === s[right]) {
            left--;
            right++;
        }
        return right - left - 1;
    }

    for (let i = 0; i < s.length; i++) {
        const len1 = expandAroundCenter(i, i);      // Odd
        const len2 = expandAroundCenter(i, i + 1);  // Even

        const currLen = Math.max(len1, len2);
        if (currLen > maxLen) {
            maxLen = currLen;
            start = i - Math.floor((currLen - 1) / 2);
        }
    }

    return s.substring(start, start + maxLen);
}
```

### DP Alternative

```python
def longestPalindromeDP(s: str) -> str:
    n = len(s)
    if n < 2:
        return s

    # dp[i][j] = True if s[i:j+1] is palindrome
    dp = [[False] * n for _ in range(n)]
    start, max_len = 0, 1

    # All single characters are palindromes
    for i in range(n):
        dp[i][i] = True

    # Check for length 2 to n
    for length in range(2, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1

            if length == 2:
                dp[i][j] = (s[i] == s[j])
            else:
                dp[i][j] = (s[i] == s[j] and dp[i + 1][j - 1])

            if dp[i][j] and length > max_len:
                start = i
                max_len = length

    return s[start:start + max_len]
```

### Interview Questions

1. **Why expand around center instead of DP?**
   - Same O(n²) time, but expand uses O(1) space vs O(n²) for DP table. Expand is also more intuitive.

2. **Why 2n-1 centers?**
   - n centers for odd-length palindromes (each character), n-1 centers for even-length (gaps between characters).

3. **What is Manacher's Algorithm?**
   - An O(n) algorithm that uses previously computed palindrome information to skip redundant comparisons. Complex but optimal.

4. **Netflix use case?**
   - Pattern matching in subtitles, detecting repeated sequences in user search queries.

### Complexity Analysis

| Approach | Time | Space |
|----------|------|-------|
| Expand Around Center | O(n²) | O(1) |
| DP | O(n²) | O(n²) |
| Manacher's | O(n) | O(n) |

**Note:** LeetCode problem #5

---

## Summary by Category

| Category | Count | Questions |
|----------|-------|-----------|
| System Design | 9 | Sentiment Tracking, Ads Platform, Video Recommendation, Audience Targeting, Billing (300M), Frequency Cap, ML Scheduler, WAL Pipeline, Promotion System |
| Arrays & Strings | 12 | Contains Duplicate (I, II, III), First Missing Positive, Longest Palindromic Substring, Longest Substring, Maximum Subarray, Median of Two Sorted Arrays, Meeting Rooms, Number Pairs, Sort by Preference, User Engagement |
| Graph & Search | 6 | Number of Islands, Parallel Courses (I, II), Reconstruct Itinerary, Movie History Friends (I, II) |
| Data Structures | 6 | LRU Cache, Merge K Sorted Lists, Versioned File System, Auto-Expire Cache, Music Playlist, Homepage Deduplication |
| Concurrency | 2 | Countdown Latch, Timer Function |
| ML & Analytics | 2 | Error Rate Monitor, Spam Email Detection |

**Total: 37 questions**

---

## Common Patterns at Netflix

1. **Ads/Advertising Systems** - Multiple questions on ad targeting, frequency capping, demand platforms
2. **Content Recommendation** - Homepage recommendations, user preferences, engagement patterns
3. **Data Modeling** - Focus on database schema design for various systems
4. **Scalability** - 300M subscriber billing system shows emphasis on scale
5. **Classic Algorithms** - Contains Duplicate series, Meeting Rooms, Reconstruct Itinerary, Maximum Subarray
6. **Caching** - LRU Cache, Auto-expire cache variations
7. **Concurrency** - Thread-safe implementations (Countdown Latch)
8. **Bitmask DP** - Small constraint problems (Parallel Courses II with n ≤ 15)
9. **Binary Search** - Median of Two Sorted Arrays, partition-based solutions
10. **Heap/Priority Queue** - Merge K Sorted Lists, top-K problems

---

*Last updated: 2026-04-18*
