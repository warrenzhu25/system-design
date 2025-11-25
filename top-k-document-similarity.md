# Top-K Document Similarity System Design

## Table of Contents
1. [Problem Statement](#problem-statement)
2. [Requirements Analysis](#requirements-analysis)
3. [Document Representation Methods](#document-representation-methods)
4. [Similarity Metrics](#similarity-metrics)
5. [Algorithm Approaches](#algorithm-approaches)
6. [System Architecture](#system-architecture)
7. [Scalability & Performance](#scalability--performance)
8. [Implementation Details](#implementation-details)
9. [Trade-offs & Comparisons](#trade-offs--comparisons)
10. [Production Considerations](#production-considerations)

## Problem Statement

**Goal:** Given a query document and a corpus of millions/billions of documents, efficiently find the top-k most similar documents.

**Challenges:**
- **Scale**: Corpus size can be 100M - 10B documents
- **Speed**: Query latency must be < 100ms for real-time applications
- **Accuracy**: Balance between exact and approximate results
- **Memory**: Cannot load all documents in memory simultaneously
- **Updates**: Handle dynamic corpus (new documents, deletions)

**Example Use Cases:**
- Duplicate detection
- Recommendation systems
- Plagiarism detection
- Search engines
- Related articles/documents

## Requirements Analysis

### Functional Requirements
- **Query Interface**: Accept a document (text or ID) and return top-k similar documents
- **Similarity Scoring**: Provide similarity scores for results
- **Ranking**: Results sorted by similarity (descending)
- **Filtering**: Support metadata filters (date, category, author)
- **Real-time Updates**: Index new documents efficiently
- **Batch Queries**: Support batch similarity searches

### Non-Functional Requirements

**Scale Assumptions:**
```
Corpus Size: 100 million documents
Average Document Size: 1 KB (after processing)
Total Storage: ~100 GB raw text
Embedding Dimension: 768 (BERT-base)
Embedding Storage: 100M × 768 × 4 bytes = ~300 GB
Queries per second: 1,000 QPS
Target Latency: P95 < 100ms
Recall@k: > 95% (compared to exact search)
```

**Performance Requirements:**
- **Latency**: P95 < 100ms, P99 < 200ms
- **Throughput**: 1,000+ QPS
- **Recall**: > 95% for top-10 results
- **Availability**: 99.9% uptime
- **Indexing Speed**: Process 10K documents/second

## Document Representation Methods

### 1. Sparse Representations

#### TF-IDF (Term Frequency-Inverse Document Frequency)

**Concept:**
- Represents documents as sparse vectors in vocabulary space
- High weights for distinctive terms, low for common terms

**Formula:**
```
TF-IDF(term, doc) = TF(term, doc) × IDF(term)

where:
  TF(term, doc) = count(term, doc) / total_terms(doc)
  IDF(term) = log(N / DF(term))
  N = total documents
  DF(term) = documents containing term
```

**Implementation:**
```python
from sklearn.feature_extraction.text import TfidfVectorizer

class TFIDFSimilarity:
    def __init__(self, max_features=10000):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words='english',
            ngram_range=(1, 2)
        )

    def fit(self, documents):
        # Build vocabulary and IDF weights
        self.vectors = self.vectorizer.fit_transform(documents)
        # vectors shape: (n_docs, vocab_size)
        # Sparse matrix: only non-zero elements stored

    def find_similar(self, query, k=10):
        # Transform query to TF-IDF vector
        query_vec = self.vectorizer.transform([query])

        # Compute cosine similarity with all documents
        similarities = (self.vectors @ query_vec.T).toarray().flatten()

        # Get top-k indices
        top_k_indices = np.argpartition(similarities, -k)[-k:]
        top_k_indices = top_k_indices[np.argsort(similarities[top_k_indices])][::-1]

        return top_k_indices, similarities[top_k_indices]
```

**Pros:**
- Simple and interpretable
- Works well for keyword-based similarity
- Sparse representation (memory efficient)
- Fast to compute

**Cons:**
- No semantic understanding (synonym problem)
- Vocabulary size can be huge
- Doesn't capture word order or context
- Poor for short documents

**Complexity:**
- Build index: O(N × D) where D = avg document length
- Query: O(N × V) where V = vocabulary size (can be optimized)

#### BM25 (Best Match 25)

**Improved TF-IDF variant:**
```
BM25(term, doc) = IDF(term) × (TF(term, doc) × (k1 + 1)) / (TF(term, doc) + k1 × (1 - b + b × |doc| / avgdl))

where:
  k1 = 1.5 (term frequency saturation)
  b = 0.75 (length normalization)
  avgdl = average document length
```

**Advantages over TF-IDF:**
- Better normalization for document length
- Saturation for term frequency (diminishing returns)
- State-of-the-art for lexical matching

### 2. Dense Representations (Embeddings)

#### Word Embeddings (Word2Vec, GloVe, FastText)

**Approach:**
- Represent each word as dense vector (50-300 dimensions)
- Document vector = average/weighted average of word vectors

**Issues:**
- Loses word order information
- Equal weighting of all words
- Context-independent representations

#### Sentence/Document Embeddings

**Doc2Vec:**
```python
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument

class Doc2VecSimilarity:
    def __init__(self, vector_size=100):
        self.model = Doc2Vec(
            vector_size=vector_size,
            min_count=2,
            epochs=40
        )

    def train(self, documents):
        # Tag documents
        tagged_docs = [
            TaggedDocument(words=doc.split(), tags=[str(i)])
            for i, doc in enumerate(documents)
        ]

        self.model.build_vocab(tagged_docs)
        self.model.train(tagged_docs, total_examples=len(tagged_docs), epochs=40)

        # Store document vectors
        self.doc_vectors = np.array([
            self.model.dv[str(i)] for i in range(len(documents))
        ])

    def find_similar(self, query, k=10):
        # Infer vector for query
        query_vec = self.model.infer_vector(query.split())

        # Compute cosine similarity
        similarities = cosine_similarity([query_vec], self.doc_vectors)[0]

        # Get top-k
        top_k_indices = np.argpartition(similarities, -k)[-k:]
        top_k_indices = top_k_indices[np.argsort(similarities[top_k_indices])][::-1]

        return top_k_indices, similarities[top_k_indices]
```

#### Transformer-based Embeddings (BERT, Sentence-BERT)

**Sentence-BERT (SBERT):**
- Fine-tuned BERT for semantic similarity
- 768-dimensional vectors (base) or 1024 (large)
- Captures semantic meaning and context

**Implementation:**
```python
from sentence_transformers import SentenceTransformer
import numpy as np

class SBERTSimilarity:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        # Load pre-trained model
        # all-MiniLM-L6-v2: 384 dim, fast
        # all-mpnet-base-v2: 768 dim, better quality
        self.model = SentenceTransformer(model_name)

    def encode_corpus(self, documents, batch_size=32):
        # Encode all documents
        self.doc_embeddings = self.model.encode(
            documents,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True  # L2 normalization
        )
        # Shape: (n_docs, embedding_dim)

    def find_similar(self, query, k=10):
        # Encode query
        query_embedding = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        # Compute cosine similarity (dot product if normalized)
        similarities = np.dot(self.doc_embeddings, query_embedding.T).flatten()

        # Get top-k
        top_k_indices = np.argpartition(similarities, -k)[-k:]
        top_k_indices = top_k_indices[np.argsort(similarities[top_k_indices])][::-1]

        return top_k_indices, similarities[top_k_indices]
```

**Pros:**
- Semantic understanding
- Captures context and meaning
- Language agnostic (multilingual models)
- Fixed-size vectors regardless of document length

**Cons:**
- Computationally expensive to generate
- Requires GPU for fast encoding
- Larger storage (dense vectors)
- Black box (hard to interpret)

**Performance:**
- Encoding: ~50-200 docs/sec on CPU, ~1000+ on GPU
- Embedding size: 384-1024 dimensions
- Quality: State-of-the-art for semantic similarity

## Similarity Metrics

### Cosine Similarity

**Formula:**
```
cosine_sim(A, B) = (A · B) / (||A|| × ||B||)
                 = Σ(Ai × Bi) / (sqrt(ΣAi²) × sqrt(ΣBi²))
```

**Properties:**
- Range: [-1, 1] (or [0, 1] for positive vectors)
- Measures angle between vectors
- Invariant to vector magnitude
- Most common for document similarity

**Optimized for Normalized Vectors:**
```python
# If vectors are L2-normalized (||v|| = 1)
# cosine_similarity = dot_product
similarity = np.dot(query_vec, doc_vec)
```

### Euclidean Distance

**Formula:**
```
euclidean_dist(A, B) = sqrt(Σ(Ai - Bi)²)
```

**Properties:**
- Range: [0, ∞)
- Sensitive to magnitude
- Less common for text (cosine preferred)

**Relationship to Cosine (for normalized vectors):**
```
euclidean_dist² = 2 - 2 × cosine_similarity
```

### Jaccard Similarity

**Formula:**
```
jaccard(A, B) = |A ∩ B| / |A ∪ B|
```

**Use case:**
- Set-based similarity
- Fast for deduplication
- MinHash for approximation

## Algorithm Approaches

### Approach 1: Brute Force (Exact Search)

**Algorithm:**
```python
def brute_force_search(query_vec, doc_vectors, k):
    """
    Exact top-k search using brute force
    """
    # Compute similarity with all documents
    similarities = cosine_similarity([query_vec], doc_vectors)[0]

    # Get top-k indices
    top_k_indices = np.argpartition(similarities, -k)[-k:]
    top_k_indices = top_k_indices[np.argsort(similarities[top_k_indices])][::-1]

    return top_k_indices, similarities[top_k_indices]
```

**Complexity:**
- Time: O(N × D) where N = documents, D = dimensions
- Space: O(N × D)

**When to use:**
- Small corpus (< 100K documents)
- High accuracy requirement (100% recall)
- Offline batch processing

**Optimizations:**
```python
import numpy as np

# Use matrix operations (BLAS-optimized)
similarities = np.dot(doc_vectors, query_vec.T)

# Use float16 for 2x memory reduction (slight accuracy loss)
doc_vectors = doc_vectors.astype(np.float16)

# Use GPUs for massive parallelization
import cupy as cp
similarities = cp.dot(doc_vectors_gpu, query_vec_gpu.T)
```

**Scalability Limits:**
- 100M docs × 768 dim × 4 bytes = 300 GB memory
- Query latency: ~seconds for 100M documents
- Not suitable for real-time at billion-scale

### Approach 2: Inverted Index (for Sparse Vectors)

**Concept:**
- For TF-IDF or BM25 representations
- Index: term → list of (doc_id, weight) pairs
- Only compute similarity for documents sharing terms

**Algorithm:**
```python
class InvertedIndex:
    def __init__(self):
        self.index = {}  # term -> [(doc_id, weight), ...]
        self.doc_norms = {}  # doc_id -> L2 norm

    def build(self, documents, vectorizer):
        vectors = vectorizer.fit_transform(documents)

        # Build inverted index
        for doc_id in range(vectors.shape[0]):
            doc_vec = vectors[doc_id]
            self.doc_norms[doc_id] = np.linalg.norm(doc_vec.toarray())

            # Get non-zero terms
            for term_id in doc_vec.nonzero()[1]:
                weight = doc_vec[0, term_id]
                if term_id not in self.index:
                    self.index[term_id] = []
                self.index[term_id].append((doc_id, weight))

    def search(self, query_vec, k=10):
        # Accumulator for partial dot products
        scores = {}
        query_norm = np.linalg.norm(query_vec)

        # Only consider documents with overlapping terms
        for term_id in query_vec.nonzero()[1]:
            query_weight = query_vec[0, term_id]

            if term_id in self.index:
                for doc_id, doc_weight in self.index[term_id]:
                    if doc_id not in scores:
                        scores[doc_id] = 0
                    scores[doc_id] += query_weight * doc_weight

        # Normalize by document norms (cosine similarity)
        for doc_id in scores:
            scores[doc_id] /= (query_norm * self.doc_norms[doc_id])

        # Get top-k
        top_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]
        return [doc_id for doc_id, _ in top_docs], [score for _, score in top_docs]
```

**Complexity:**
- Build: O(N × D)
- Query: O(Q × P × k) where Q = query terms, P = avg posting list size
- Much faster than brute force when documents share few terms

**Optimizations:**
- **WAND (Weak AND)**: Skip low-scoring documents early
- **Block-Max Index**: Store max scores per block for pruning
- **Impact-ordered postings**: Order by weight for early termination

**When to use:**
- Sparse representations (TF-IDF, BM25)
- Keyword-based similarity
- Large vocabulary with few overlaps

### Approach 3: Locality Sensitive Hashing (LSH)

**Concept:**
- Hash similar vectors to the same buckets
- Probability of collision proportional to similarity
- Query only documents in same/nearby buckets

**Random Projection LSH (for Cosine Similarity):**

**Algorithm:**
```python
import numpy as np
from collections import defaultdict

class LSH:
    def __init__(self, num_tables=10, hash_size=10, input_dim=768):
        self.num_tables = num_tables
        self.hash_size = hash_size

        # Random hyperplanes for hashing
        self.hyperplanes = [
            np.random.randn(hash_size, input_dim)
            for _ in range(num_tables)
        ]

        # Hash tables: hash -> [doc_ids]
        self.tables = [defaultdict(list) for _ in range(num_tables)]
        self.doc_vectors = None

    def _hash(self, vec, table_id):
        # Project onto random hyperplanes
        # Hash bit = 1 if projection > 0, else 0
        projections = np.dot(self.hyperplanes[table_id], vec)
        hash_bits = (projections > 0).astype(int)

        # Convert binary to integer hash
        hash_val = int(''.join(map(str, hash_bits)), 2)
        return hash_val

    def index(self, doc_vectors):
        self.doc_vectors = doc_vectors

        for doc_id, vec in enumerate(doc_vectors):
            for table_id in range(self.num_tables):
                hash_val = self._hash(vec, table_id)
                self.tables[table_id][hash_val].append(doc_id)

    def query(self, query_vec, k=10):
        # Collect candidate documents from all tables
        candidates = set()

        for table_id in range(self.num_tables):
            hash_val = self._hash(query_vec, table_id)
            if hash_val in self.tables[table_id]:
                candidates.update(self.tables[table_id][hash_val])

        # Compute exact similarity for candidates only
        candidate_vecs = self.doc_vectors[list(candidates)]
        similarities = np.dot(candidate_vecs, query_vec)

        # Get top-k from candidates
        top_k_indices = np.argpartition(similarities, -k)[-k:]
        top_k_indices = top_k_indices[np.argsort(similarities[top_k_indices])][::-1]

        candidate_ids = list(candidates)
        return [candidate_ids[i] for i in top_k_indices], similarities[top_k_indices]
```

**MinHash (for Jaccard Similarity):**

**Algorithm:**
```python
class MinHashLSH:
    def __init__(self, num_perm=128):
        self.num_perm = num_perm
        self.hash_tables = defaultdict(list)

    def _minhash(self, tokens):
        # Create MinHash signature
        signature = []
        for i in range(self.num_perm):
            # Hash each token with permutation i
            min_hash = min(hash((token, i)) for token in tokens)
            signature.append(min_hash)
        return signature

    def index(self, documents):
        for doc_id, doc in enumerate(documents):
            tokens = set(doc.split())
            signature = self._minhash(tokens)

            # Band-based hashing (LSH)
            bands = 8
            rows_per_band = self.num_perm // bands

            for band_id in range(bands):
                start = band_id * rows_per_band
                end = start + rows_per_band
                band_sig = tuple(signature[start:end])

                self.hash_tables[(band_id, band_sig)].append(doc_id)

    def query(self, query, k=10):
        tokens = set(query.split())
        signature = self._minhash(tokens)

        # Find candidates
        candidates = set()
        bands = 8
        rows_per_band = self.num_perm // bands

        for band_id in range(bands):
            start = band_id * rows_per_band
            end = start + rows_per_band
            band_sig = tuple(signature[start:end])

            if (band_id, band_sig) in self.hash_tables:
                candidates.update(self.hash_tables[(band_id, band_sig)])

        # Compute exact Jaccard for candidates
        # ... (implementation details)

        return list(candidates)[:k]
```

**Properties:**
- **Probability of collision**: P(h(x) = h(y)) = similarity(x, y)
- **Parameters**:
  - More hash tables → higher recall, more candidates
  - Larger hash size → more precise buckets, fewer candidates

**Pros:**
- Sub-linear query time: O(L × K + C × D) where C = candidates << N
- Memory efficient (only store hashes)
- Tunable accuracy/speed trade-off

**Cons:**
- Probabilistic (may miss true top-k)
- Parameter tuning required
- Less effective in very high dimensions

### Approach 4: Hierarchical Navigable Small World (HNSW)

**Concept:**
- Graph-based approximate nearest neighbor search
- Multi-layer skip-list-like structure
- Navigate through layers from coarse to fine

**Architecture:**
```
Layer 2: [Entry] → [A] → [D] (sparse, long edges)
            ↓
Layer 1: [Entry] → [A] → [B] → [D] → [F] (medium density)
            ↓
Layer 0: [Entry] → [A] → [B] → [C] → [D] → [E] → [F] (dense, all nodes)
```

**Algorithm:**
```python
import hnswlib

class HNSWIndex:
    def __init__(self, dim=768, max_elements=1000000):
        self.dim = dim
        self.max_elements = max_elements

        # Initialize HNSW index
        self.index = hnswlib.Index(space='cosine', dim=dim)
        self.index.init_index(
            max_elements=max_elements,
            ef_construction=200,  # Controls build quality
            M=16  # Number of connections per node
        )

    def build(self, doc_vectors):
        # Add vectors to index
        self.index.add_items(doc_vectors, ids=np.arange(len(doc_vectors)))

        # Set query time parameters
        self.index.set_ef(50)  # Controls query quality

    def query(self, query_vec, k=10):
        # Search for k nearest neighbors
        labels, distances = self.index.knn_query(query_vec, k=k)

        # Convert distances to similarities (for cosine)
        similarities = 1 - distances[0]

        return labels[0], similarities
```

**Parameters:**
- **M**: Number of connections per node (typical: 12-48)
  - Higher M → better recall, more memory
- **ef_construction**: Size of candidate list during build (typical: 100-500)
  - Higher → better quality, slower build
- **ef**: Size of candidate list during search (typical: k to 10×k)
  - Higher → better recall, slower search

**Complexity:**
- Build: O(N × log(N) × M × D)
- Query: O(log(N) × M × D)
- Memory: O(N × M × D)

**Pros:**
- Excellent recall/speed trade-off (95%+ recall with 10-100x speedup)
- Logarithmic search complexity
- Supports dynamic updates (add/delete)
- State-of-the-art performance

**Cons:**
- Higher memory usage than LSH
- Build time can be significant
- Not as good for very high-dimensional data (>2000 dims)

### Approach 5: Product Quantization (PQ)

**Concept:**
- Compress vectors by quantizing subspaces
- Divide vector into m subvectors
- Cluster each subspace independently
- Represent vector as cluster IDs (compression)

**Algorithm:**
```python
import faiss

class ProductQuantization:
    def __init__(self, dim=768, m=8, nbits=8):
        # m = number of subquantizers
        # nbits = bits per subquantizer (2^8 = 256 clusters each)
        self.dim = dim
        self.m = m
        self.nbits = nbits

        # Create IVF-PQ index
        # IVF = Inverted File (coarse quantization)
        # PQ = Product Quantization (fine quantization)
        nlist = 1024  # Number of Voronoi cells

        quantizer = faiss.IndexFlatL2(dim)
        self.index = faiss.IndexIVFPQ(
            quantizer,
            dim,
            nlist,
            m,
            nbits
        )

    def train_and_build(self, doc_vectors):
        # Train the index (learn quantizers)
        self.index.train(doc_vectors)

        # Add vectors
        self.index.add(doc_vectors)

    def query(self, query_vec, k=10):
        # Set number of cells to visit (speed/accuracy trade-off)
        self.index.nprobe = 10

        # Search
        distances, indices = self.index.search(query_vec.reshape(1, -1), k)

        return indices[0], 1 / (1 + distances[0])  # Convert to similarity
```

**Compression:**
- Original: N × D × 4 bytes (float32)
- PQ: N × m × nbits / 8 bytes
- Example: 768 dims → 8 × 8 bits = 64 bytes (12x compression)

**Pros:**
- Massive compression (10-100x)
- Fast search (asymmetric distance computation)
- Scales to billions of vectors

**Cons:**
- Lossy compression (reduced accuracy)
- Training required (k-means on subspaces)
- Complex implementation

### Approach 6: ScaNN (Scalable Nearest Neighbors)

**Google's state-of-the-art ANN algorithm:**

```python
import scann

class ScaNNIndex:
    def __init__(self, num_leaves=1000, num_leaves_to_search=100):
        self.num_leaves = num_leaves
        self.num_leaves_to_search = num_leaves_to_search

    def build(self, doc_vectors):
        # Build ScaNN index
        self.searcher = scann.scann_ops_pybind.builder(
            doc_vectors,
            k=10,
            distance_measure="dot_product"
        ).tree(
            num_leaves=self.num_leaves,
            num_leaves_to_search=self.num_leaves_to_search,
            training_sample_size=250000
        ).score_ah(
            dimensions_per_block=2,
            anisotropic_quantization_threshold=0.2
        ).reorder(
            reordering_num_neighbors=100
        ).build()

    def query(self, query_vec, k=10):
        indices, distances = self.searcher.search(query_vec, final_num_neighbors=k)
        return indices, distances
```

**Key Features:**
- **Tree-based partitioning**: Partition space into clusters
- **Quantization**: Compress vectors for fast distance computation
- **Re-ranking**: Refine results using original vectors
- **Performance**: State-of-the-art recall/latency trade-off

## System Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      API Gateway / Load Balancer             │
└────────────────┬───────────────────────────────────────────┘
                 │
    ┌────────────┴─────────────┐
    │                          │
┌───▼─────────┐         ┌──────▼────────┐
│   Query     │         │   Indexing    │
│   Service   │         │   Pipeline    │
└───┬─────────┘         └──────┬────────┘
    │                          │
    │    ┌─────────────────────┴──────────────┐
    │    │                                    │
┌───▼────▼────┐    ┌──────────────┐    ┌─────▼────────┐
│  Vector     │    │  Metadata    │    │  Document    │
│  Index      │◄───│  Database    │◄───│  Processor   │
│ (HNSW/FAISS)│    │ (PostgreSQL) │    │  (Encoding)  │
└─────────────┘    └──────────────┘    └──────────────┘
      │
      │
┌─────▼─────────┐
│  Re-ranking   │
│   Service     │
└───────────────┘
```

### Component Details

#### 1. Document Processor / Indexing Pipeline

**Responsibilities:**
- Ingest new documents
- Generate embeddings
- Update vector index
- Update metadata database

**Pipeline:**
```
Document → Preprocessing → Embedding → Index Update → Metadata Update
```

**Implementation:**
```python
class IndexingPipeline:
    def __init__(self, encoder, vector_index, metadata_db):
        self.encoder = encoder
        self.vector_index = vector_index
        self.metadata_db = metadata_db

    def process_document(self, doc_id, text, metadata):
        # 1. Preprocess
        cleaned_text = self.preprocess(text)

        # 2. Generate embedding
        embedding = self.encoder.encode(cleaned_text)

        # 3. Add to vector index
        self.vector_index.add([embedding], [doc_id])

        # 4. Store metadata
        self.metadata_db.insert({
            'doc_id': doc_id,
            'text_preview': text[:500],
            'metadata': metadata,
            'indexed_at': datetime.now()
        })

    def batch_process(self, documents, batch_size=100):
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i+batch_size]

            # Batch encode (efficient)
            embeddings = self.encoder.encode([d['text'] for d in batch])

            # Batch insert
            doc_ids = [d['id'] for d in batch]
            self.vector_index.add(embeddings, doc_ids)

            # Update metadata
            self.metadata_db.batch_insert([
                {'doc_id': d['id'], 'metadata': d['metadata']}
                for d in batch
            ])
```

#### 2. Query Service

**Responsibilities:**
- Accept query requests
- Generate query embedding
- Search vector index
- Apply filters
- Re-rank results
- Return top-k documents

**API Design:**
```python
from fastapi import FastAPI, Query
from pydantic import BaseModel
from typing import List, Optional, Dict

app = FastAPI()

class SimilarityQuery(BaseModel):
    query: str
    k: int = 10
    filters: Optional[Dict] = None
    rerank: bool = True

class SimilarityResult(BaseModel):
    doc_id: str
    score: float
    text_preview: str
    metadata: Dict

@app.post("/similarity/search")
async def search_similar(request: SimilarityQuery) -> List[SimilarityResult]:
    # 1. Encode query
    query_embedding = encoder.encode(request.query)

    # 2. Search vector index (get more for filtering)
    candidate_ids, scores = vector_index.query(
        query_embedding,
        k=request.k * 3 if request.filters else request.k
    )

    # 3. Get metadata
    candidates = metadata_db.get_batch(candidate_ids)

    # 4. Apply filters
    if request.filters:
        candidates = apply_filters(candidates, request.filters)

    # 5. Re-rank (optional)
    if request.rerank:
        candidates = reranker.rerank(request.query, candidates)

    # 6. Return top-k
    results = candidates[:request.k]

    return [
        SimilarityResult(
            doc_id=c['doc_id'],
            score=c['score'],
            text_preview=c['text_preview'],
            metadata=c['metadata']
        )
        for c in results
    ]
```

**Query Flow:**
```
Request → Encode → Vector Search → Filter → Re-rank → Response
           (5ms)      (20-50ms)     (5ms)    (10ms)    (total: 40-70ms)
```

#### 3. Vector Index Service

**Sharding Strategy:**

For billion-scale corpus, shard the index:

```python
class ShardedVectorIndex:
    def __init__(self, num_shards=10, index_type='hnsw'):
        self.num_shards = num_shards
        self.shards = [
            self.create_index(index_type)
            for _ in range(num_shards)
        ]

    def _get_shard_id(self, doc_id):
        return hash(doc_id) % self.num_shards

    def add(self, embeddings, doc_ids):
        # Route to shards
        shard_batches = defaultdict(list)
        for emb, doc_id in zip(embeddings, doc_ids):
            shard_id = self._get_shard_id(doc_id)
            shard_batches[shard_id].append((emb, doc_id))

        # Add to each shard
        for shard_id, batch in shard_batches.items():
            embs = [b[0] for b in batch]
            ids = [b[1] for b in batch]
            self.shards[shard_id].add(embs, ids)

    def query(self, query_vec, k=10):
        # Query all shards in parallel
        results = []

        with ThreadPoolExecutor(max_workers=self.num_shards) as executor:
            futures = [
                executor.submit(shard.query, query_vec, k)
                for shard in self.shards
            ]

            for future in futures:
                indices, scores = future.result()
                results.extend(zip(indices, scores))

        # Merge and get global top-k
        results.sort(key=lambda x: x[1], reverse=True)
        top_k = results[:k]

        return [r[0] for r in top_k], [r[1] for r in top_k]
```

**Replication:**
```
Primary Shard 0  →  Replica 0a, 0b
Primary Shard 1  →  Replica 1a, 1b
...
Primary Shard 9  →  Replica 9a, 9b
```

#### 4. Re-ranking Service

**Two-stage retrieval:**
1. **Stage 1**: Fast approximate search (retrieve 100-1000 candidates)
2. **Stage 2**: Expensive re-ranking (score top 100, return top 10)

**Re-ranking Methods:**

**A. Cross-encoder:**
```python
from sentence_transformers import CrossEncoder

class Reranker:
    def __init__(self):
        self.model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

    def rerank(self, query, candidates, top_k=10):
        # Create query-document pairs
        pairs = [(query, c['text']) for c in candidates]

        # Score all pairs
        scores = self.model.predict(pairs)

        # Sort by scores
        scored = list(zip(candidates, scores))
        scored.sort(key=lambda x: x[1], reverse=True)

        # Return top-k with updated scores
        for i, (candidate, score) in enumerate(scored[:top_k]):
            candidate['score'] = float(score)

        return [c for c, _ in scored[:top_k]]
```

**B. Feature-based re-ranking:**
```python
class FeatureReranker:
    def rerank(self, query, candidates):
        for candidate in candidates:
            # Combine multiple signals
            candidate['final_score'] = (
                0.6 * candidate['semantic_score'] +  # From vector search
                0.2 * self.bm25_score(query, candidate['text']) +  # Lexical
                0.1 * candidate['popularity_score'] +  # User engagement
                0.1 * self.recency_score(candidate['date'])  # Freshness
            )

        candidates.sort(key=lambda x: x['final_score'], reverse=True)
        return candidates
```

#### 5. Metadata Database

**Schema:**
```sql
CREATE TABLE documents (
    doc_id VARCHAR(255) PRIMARY KEY,
    title TEXT,
    text_preview TEXT,  -- First 500 chars
    category VARCHAR(100),
    author VARCHAR(255),
    created_at TIMESTAMP,
    updated_at TIMESTAMP,
    popularity_score FLOAT,
    metadata JSONB,
    INDEX idx_category (category),
    INDEX idx_author (author),
    INDEX idx_created_at (created_at)
);

CREATE TABLE embeddings (
    doc_id VARCHAR(255) PRIMARY KEY,
    shard_id INT,
    embedding_model VARCHAR(100),
    embedding_version INT,
    indexed_at TIMESTAMP,
    FOREIGN KEY (doc_id) REFERENCES documents(doc_id)
);
```

**Purpose:**
- Store document metadata
- Support filtering (category, date range, author)
- Track indexing status
- Analytics and monitoring

## Scalability & Performance

### Horizontal Scaling

**Shard by Vector Space:**
```
100M documents → 10 shards of 10M each

Query process:
1. Fan out query to all 10 shards (parallel)
2. Each shard returns top-k candidates
3. Merge results and select global top-k
```

**Benefits:**
- Linear scaling with number of shards
- Parallel query processing
- Fault tolerance (replicate each shard)

**Challenges:**
- Need to query all shards (can't route to specific shard)
- Merge overhead
- Load balancing

### Caching

**Multi-level cache:**

```python
class CachedSearchService:
    def __init__(self, vector_index, cache):
        self.vector_index = vector_index
        self.cache = cache  # Redis

    def query(self, query_text, k=10):
        # Generate cache key
        cache_key = f"search:{hash(query_text)}:{k}"

        # Check cache
        cached = self.cache.get(cache_key)
        if cached:
            return cached

        # Compute
        results = self._compute_search(query_text, k)

        # Cache results (TTL: 1 hour)
        self.cache.setex(cache_key, 3600, results)

        return results
```

**What to cache:**
- Query results (for popular queries)
- Document embeddings (avoid re-encoding)
- Metadata lookups
- Pre-computed similarities for trending documents

**Cache hit rate:**
- Popular queries: ~30-50% hit rate
- Latency reduction: 90%+ for cache hits

### Load Balancing

**Strategy:**
```
Client → Load Balancer → Query Service Pool (10 instances)
                           ↓
                    Shard Pool (100 shards, 3 replicas each)
```

**Replica selection:**
- Round-robin across replicas
- Least-loaded replica
- Geographic proximity

### Batch Processing

**Batch queries:**
```python
def batch_query(queries, k=10):
    # Encode all queries in one batch (GPU efficient)
    query_embeddings = encoder.encode(queries, batch_size=32)

    # Search for all queries
    all_results = []
    for query_emb in query_embeddings:
        results = vector_index.query(query_emb, k)
        all_results.append(results)

    return all_results
```

**Benefits:**
- GPU utilization: batch encoding ~10x faster
- Amortize overhead
- Better throughput

### Performance Benchmarks

**Configuration:**
- Corpus: 100M documents
- Embeddings: 768-dim (SBERT)
- Algorithm: HNSW
- Hardware: 32-core CPU, 256GB RAM

**Results:**
```
Metric                  | Brute Force | HNSW     | FAISS IVF-PQ
------------------------|-------------|----------|-------------
Query Latency (P95)     | 5000ms      | 50ms     | 20ms
Throughput (QPS)        | 0.2         | 1000     | 2500
Recall@10               | 100%        | 98%      | 95%
Memory Usage            | 300GB       | 350GB    | 50GB
Build Time              | 1 hour      | 4 hours  | 6 hours
```

## Implementation Details

### Complete Production System

**Directory Structure:**
```
similarity-search/
├── api/
│   ├── server.py           # FastAPI application
│   └── models.py           # Request/response models
├── indexing/
│   ├── pipeline.py         # Indexing pipeline
│   ├── encoder.py          # Document encoder
│   └── preprocessor.py     # Text preprocessing
├── search/
│   ├── vector_index.py     # Vector index abstraction
│   ├── hnsw_index.py       # HNSW implementation
│   ├── faiss_index.py      # FAISS implementation
│   └── reranker.py         # Re-ranking service
├── storage/
│   ├── metadata_db.py      # Metadata database
│   └── cache.py            # Redis cache
├── config/
│   └── config.yaml         # Configuration
└── deploy/
    ├── docker-compose.yaml
    └── kubernetes/
```

**Configuration:**
```yaml
# config.yaml
model:
  name: "sentence-transformers/all-MiniLM-L6-v2"
  dimension: 384
  device: "cuda"  # or "cpu"

index:
  type: "hnsw"  # or "faiss_ivfpq", "lsh"
  num_shards: 10
  hnsw:
    M: 16
    ef_construction: 200
    ef_search: 50
  faiss:
    nlist: 1024
    m: 8
    nbits: 8

search:
  default_k: 10
  max_k: 100
  enable_reranking: true
  candidate_multiplier: 3

cache:
  enabled: true
  ttl: 3600
  max_size: "10GB"

database:
  host: "localhost"
  port: 5432
  name: "similarity_search"
```

### Monitoring & Metrics

**Key Metrics:**
```python
# Prometheus metrics
from prometheus_client import Counter, Histogram, Gauge

queries_total = Counter('queries_total', 'Total queries')
query_latency = Histogram('query_latency_seconds', 'Query latency')
cache_hits = Counter('cache_hits_total', 'Cache hits')
recall_at_k = Gauge('recall_at_k', 'Recall@k metric')
index_size = Gauge('index_size_documents', 'Number of indexed documents')

@app.post("/search")
async def search(request):
    queries_total.inc()

    start = time.time()
    results = await search_service.search(request)
    query_latency.observe(time.time() - start)

    return results
```

**Dashboards:**
- Query latency (P50, P95, P99)
- Throughput (QPS)
- Cache hit rate
- Index size and growth
- Error rate
- Resource utilization

## Trade-offs & Comparisons

### Algorithm Comparison

| Algorithm    | Recall@10 | Latency | Memory | Build Time | Best For |
|-------------|-----------|---------|--------|------------|----------|
| Brute Force | 100%      | 5000ms  | High   | Fast       | < 100K docs, offline |
| Inverted Index | 100% | 50ms | Low | Fast | Sparse vectors (TF-IDF) |
| LSH | 80-90% | 10ms | Low | Fast | Quick deduplication |
| HNSW | 95-99% | 20-50ms | High | Slow | Best all-around |
| FAISS IVF-PQ | 90-95% | 5-20ms | Low | Medium | Billion-scale |
| ScaNN | 96-99% | 10-30ms | Medium | Medium | State-of-the-art |

### Embedding Model Comparison

| Model | Dimension | Quality | Speed (CPU) | Use Case |
|-------|-----------|---------|-------------|----------|
| TF-IDF | 10K-100K | Low | Very Fast | Keyword matching |
| Word2Vec avg | 300 | Medium | Fast | Simple semantic |
| Doc2Vec | 100-300 | Medium | Fast | Legacy systems |
| SBERT Mini | 384 | High | Medium | Balanced |
| SBERT Base | 768 | Very High | Slow | Best quality |
| OpenAI Ada-002 | 1536 | Very High | API call | Production (API) |

### Accuracy vs Speed Trade-off

```
                  Accuracy (Recall)
                        ↑
                        │
                   100% │  Brute Force
                        │    ×
                        │
                    98% │         HNSW (ef=200)
                        │            ×
                        │
                    95% │                 FAISS IVF-PQ
                        │                    ×
                        │
                    90% │                         LSH
                        │                            ×
                        │
                    80% │
                        └─────────────────────────────────────→
                          10ms      50ms     100ms    500ms   Latency
```

### When to Use Each Approach

**Brute Force:**
- Corpus < 100K documents
- Accuracy is critical (100% recall required)
- Offline batch processing
- Ground truth for evaluation

**Inverted Index:**
- Using TF-IDF or BM25 (sparse vectors)
- Keyword-based similarity
- Large vocabulary with low term overlap
- Interpretability required

**LSH:**
- Need very fast queries (< 10ms)
- Can tolerate lower recall (80-90%)
- Limited memory budget
- Quick deduplication tasks

**HNSW:**
- Best general-purpose choice
- Corpus up to 100M documents
- Need high recall (95-99%)
- Sufficient memory available
- Support for dynamic updates

**FAISS IVF-PQ:**
- Billion-scale corpus
- Memory constraints
- Can tolerate slight accuracy loss
- Batch-oriented workload

**ScaNN:**
- State-of-the-art performance needed
- Have engineering resources for complex setup
- Large-scale production system
- Google Cloud environment

## Production Considerations

### Data Privacy & Security

**Embedding Security:**
- Embeddings can leak information about original text
- Store embeddings separately from raw text
- Encrypt embeddings at rest
- Access control for sensitive documents

**Query Privacy:**
- Log sanitization (remove PII from query logs)
- Differential privacy for analytics
- Rate limiting per user

### Cost Optimization

**Compute Costs:**
- **Embedding generation**: Use GPU for batch processing
  - 1M documents × $0.0001/doc (GPU time) = $100
- **Index building**: One-time cost, can use spot instances
- **Query serving**: Optimize for CPU efficiency
  - 1M queries × $0.00001/query = $10/month

**Storage Costs:**
- **Embeddings**: 100M docs × 768 dims × 4 bytes = 300GB
  - S3: $7/month, EBS SSD: $30/month
- **Compressed (PQ)**: 100M docs × 64 bytes = 6GB
  - S3: $0.15/month

**Cost-Performance Trade-offs:**
```
Configuration         | Cost/month | Latency | Recall
----------------------|------------|---------|--------
GPU + Brute Force     | $1000      | 500ms   | 100%
CPU + HNSW (large)    | $200       | 50ms    | 98%
CPU + FAISS IVF-PQ    | $50        | 20ms    | 95%
Serverless + LSH      | $20        | 15ms    | 85%
```

### Evaluation & Testing

**Offline Metrics:**
```python
def evaluate_recall(ground_truth, predictions, k=10):
    """
    ground_truth: list of relevant doc IDs
    predictions: list of retrieved doc IDs
    """
    relevant = set(ground_truth[:k])
    retrieved = set(predictions[:k])

    recall = len(relevant & retrieved) / len(relevant)
    return recall

def evaluate_mrr(ground_truth, predictions):
    """Mean Reciprocal Rank"""
    for i, pred in enumerate(predictions):
        if pred in ground_truth:
            return 1.0 / (i + 1)
    return 0.0

def evaluate_ndcg(ground_truth, predictions, k=10):
    """Normalized Discounted Cumulative Gain"""
    dcg = sum(
        (1 if predictions[i] in ground_truth else 0) / np.log2(i + 2)
        for i in range(k)
    )

    idcg = sum(1 / np.log2(i + 2) for i in range(min(k, len(ground_truth))))

    return dcg / idcg if idcg > 0 else 0
```

**Test Dataset:**
- **MS MARCO**: 8.8M passages, 0.5M queries
- **BEIR**: 18 diverse datasets for evaluation
- **Custom**: Domain-specific test sets

**A/B Testing:**
- Measure click-through rate (CTR)
- User engagement time
- Relevance ratings
- Task completion rate

### Dynamic Updates

**Handling New Documents:**

**Approach 1: Batch Updates**
```python
class BatchIndexUpdater:
    def __init__(self, update_interval=3600):  # 1 hour
        self.buffer = []
        self.update_interval = update_interval

    def add_document(self, doc):
        self.buffer.append(doc)

    def periodic_update(self):
        if len(self.buffer) > 0:
            # Encode batch
            embeddings = encoder.encode([d['text'] for d in self.buffer])

            # Add to index
            doc_ids = [d['id'] for d in self.buffer]
            vector_index.add(embeddings, doc_ids)

            # Clear buffer
            self.buffer = []
```

**Approach 2: Online Updates**
```python
# HNSW supports online updates
def add_document_online(doc_id, text):
    embedding = encoder.encode(text)
    hnsw_index.add_items(embedding, [doc_id])
```

**Approach 3: Dual Index (for immutable indices like FAISS)**
```
Main Index (rebuilt daily)  ← 99% of corpus
                +
Delta Index (real-time)     ← New documents

Query both and merge results
```

### Handling Deletions

**Soft Delete:**
```python
# Mark as deleted in metadata
metadata_db.update(doc_id, {'deleted': True})

# Filter at query time
def query_with_filter(query_vec, k):
    candidates = vector_index.query(query_vec, k * 2)

    # Filter deleted
    active = [
        c for c in candidates
        if not metadata_db.is_deleted(c['doc_id'])
    ]

    return active[:k]
```

**Hard Delete:**
```python
# Rebuild index periodically without deleted docs
def rebuild_index():
    active_docs = metadata_db.get_active_documents()
    embeddings = [load_embedding(d['doc_id']) for d in active_docs]

    new_index = build_index(embeddings)

    # Atomic swap
    swap_index(new_index)
```

### Disaster Recovery

**Backup Strategy:**
```
Daily:
  - Metadata database dump
  - Document embeddings snapshot

Weekly:
  - Full index snapshot
  - Configuration backup

On-demand:
  - Before major updates
  - Model changes
```

**Recovery Procedures:**
```
Scenario 1: Index corruption
  → Load from latest snapshot (< 1 day old)
  → Replay incremental updates from database log

Scenario 2: Total system failure
  → Restore from backup (24-hour RPO)
  → Rebuild index from embeddings (4-6 hours)

Scenario 3: Model change
  → Keep old model running
  → Build new index with new embeddings
  → A/B test before full migration
```

## Conclusion

### Recommended Architecture for Different Scales

**Small Scale (< 1M documents):**
```
Encoding: SBERT (all-MiniLM-L6-v2)
Algorithm: HNSW (hnswlib)
Storage: Single server, PostgreSQL
Deployment: Single container
Cost: ~$50/month
Latency: 10-20ms
```

**Medium Scale (1M - 100M documents):**
```
Encoding: SBERT or sentence transformers
Algorithm: HNSW with sharding (10 shards)
Storage: Sharded PostgreSQL + Redis cache
Deployment: Kubernetes (10 query pods, 10 shard pods)
Cost: ~$500/month
Latency: 20-50ms
```

**Large Scale (100M - 1B+ documents):**
```
Encoding: Optimized sentence transformers
Algorithm: FAISS IVF-PQ or ScaNN
Storage: Distributed (Cassandra/DynamoDB) + S3
Deployment: Kubernetes cluster (100+ pods)
Re-ranking: Two-stage retrieval
Cost: ~$5000/month
Latency: 50-100ms
```

### Key Takeaways

1. **Representation matters**: Dense embeddings (SBERT) generally outperform sparse (TF-IDF) for semantic similarity

2. **Algorithm choice depends on scale**:
   - < 100K: Brute force or simple HNSW
   - 100K - 10M: HNSW
   - 10M - 1B+: FAISS IVF-PQ or ScaNN

3. **Trade-offs are inevitable**:
   - Accuracy ↔ Speed
   - Memory ↔ Compression
   - Build time ↔ Query time

4. **System design is crucial**:
   - Sharding for scale
   - Caching for performance
   - Re-ranking for quality
   - Monitoring for reliability

5. **Production requires more than just algorithms**:
   - Dynamic updates
   - Fault tolerance
   - Cost optimization
   - A/B testing
   - Monitoring and alerting

This design provides a comprehensive foundation for building a production-ready, scalable document similarity search system capable of handling billions of documents with sub-100ms latency.
