# Qdrant Implementation Guide for Hierarchical JSON

## Table of Contents

- [Why Qdrant for Hierarchical JSON](#why-qdrant-for-hierarchical-json)
- [Qdrant vs Alternatives](#qdrant-vs-alternatives)
- [When to Choose Qdrant](#when-to-choose-qdrant)
- [Collection Setup](#collection-setup)
- [Ingestion Patterns](#ingestion-patterns)
- [Query Patterns](#query-patterns)
- [Optimizations](#optimizations)
- [Performance Benchmarks](#performance-benchmarks)

---

## Why Qdrant for Hierarchical JSON

Qdrant is particularly well-suited for the hierarchical JSON to vector pipeline due to several architectural features:

### Strengths for This Use Case

**1. Rich Payload Filtering**

Qdrant's payload (metadata) supports nested JSON natively with complex filter operations:

```python
# Qdrant can filter on nested fields and arrays efficiently
client.search(
    collection_name="orders",
    query_vector=embedding,
    query_filter={
        "must": [
            {"key": "vendors", "match": {"any": ["Apple"]}},
            {"key": "total_price", "range": {"gte": 500}},
            {"key": "shipping_address.province", "match": "Ontario"}
        ]
    }
)
```

**Why this matters**: The framework relies heavily on metadata filtering before semantic search.

**2. Payload Indexing**

Automatic indexing of payload fields for fast filtering:

```python
client.create_payload_index(
    collection_name="orders",
    field_name="vendors",
    field_schema=PayloadSchemaType.KEYWORD  # Fast array contains
)
```

**Why this matters**: Enables sub-20ms filtering on metadata before vector search.

**3. Quantization**

Built-in scalar and product quantization reduces memory by 4-16x without significant accuracy loss:

```python
client.update_collection(
    collection_name="orders",
    quantization_config={
        "scalar": {
            "type": "int8",
            "quantile": 0.99,
            "always_ram": True
        }
    }
)
```

**Why this matters**:
- 1M orders with full docs: 63 GB → 58.5 GB (vectors: 6 GB → 1.5 GB)
- Reference architecture: 13 GB → 8.5 GB
- Helps offset storage overhead from flattening

**4. Collection Aliases**

Easy blue-green deployments and multi-version support:

```python
# Zero-downtime schema migrations
client.update_collection_aliases(
    change_aliases_operations=[
        {"delete_alias": {"alias_name": "orders_prod"}},
        {"create_alias": {"alias_name": "orders_prod", "collection_name": "orders_v2"}}
    ]
)
```

**Why this matters**: Useful for schema evolution as access patterns change.

**5. Sparse Vectors**

Support for hybrid dense+sparse vectors (enables keyword + semantic search in single query):

```python
results = client.search(
    collection_name="orders",
    query_vector=dense_embedding,
    sparse_query_vector=SparseVector(
        indices=[123, 456, 789],  # From BM25 or SPLADE
        values=[0.5, 0.3, 0.2]
    )
)
```

**Why this matters**: True hybrid search without multiple queries.

**6. Open Source**

Self-hostable with no vendor lock-in:
- Docker/Kubernetes deployment
- Full control over data
- No usage-based pricing surprises
- Critical for healthcare/compliance requirements

### Limitations to Consider

**1. Limited ML Model Integrations**

Qdrant has built-in rescoring and fusion (RRF), but doesn't integrate reranking models like Cohere natively:

```python
# Need to handle reranking externally
results = client.search(...)

# Call external reranking API
import cohere
co = cohere.Client(api_key="...")
reranked = co.rerank(
    query=query_text,
    documents=[r.payload["content"] for r in results],
    top_n=10
)
```

**Compare to**: Weaviate has native reranker modules.

**2. Scaling Complexity**

Horizontal scaling requires manual sharding configuration:

```python
# Must explicitly configure sharding
client.create_collection(
    collection_name="orders",
    shard_number=4,  # Manual configuration
    replication_factor=2
)
```

**Compare to**: Pinecone handles scaling automatically.

**3. Learning Curve**

More configuration options = steeper learning curve:
- Collection configuration
- Index management
- Sharding strategy
- Quantization tuning

**Compare to**: Pinecone abstracts most of this away.

**4. Ecosystem**

Smaller ecosystem vs Pinecone:
- Fewer integrations
- Smaller community
- Less documentation/examples

But: Growing rapidly, active development.

---

## Qdrant vs Alternatives

| Feature | Qdrant | Pinecone | Weaviate | pgvector | Milvus |
|---------|--------|----------|----------|----------|--------|
| **Nested payload filtering** | ✅ Native | ✅ JSON support | ✅ Native | ⚠️ JSONB (slower) | ✅ Native |
| **Array field indexing** | ✅ Yes | ✅ Yes | ✅ Yes | ❌ No | ✅ Yes |
| **Quantization** | ✅ Built-in | ✅ Built-in | ⚠️ Limited | ❌ No | ✅ Built-in |
| **Sparse vectors** | ✅ Native | ❌ No | ✅ Via modules | ❌ No | ⚠️ Experimental |
| **Reranking** | ⚠️ Rescoring/fusion | ❌ External | ✅ Model modules | ❌ External | ❌ External |
| **Managed hosting** | ✅ Qdrant Cloud | ✅ Yes | ✅ Yes | ⚠️ Via providers | ⚠️ Zilliz |
| **Self-hosted** | ✅ Docker/K8s | ❌ No | ✅ Docker/K8s | ✅ Postgres ext | ✅ Docker/K8s |
| **Horizontal scaling** | ⚠️ Manual sharding | ✅ Automatic | ✅ Automatic | ⚠️ Postgres limits | ✅ Automatic |
| **Cost (1M vectors)** | ~$50-100/mo | ~$70-150/mo | ~$60-120/mo | ~$30-80/mo | ~$40-100/mo |

### Detailed Comparison

**Qdrant**:
- ✅ Excellent payload filtering (core requirement)
- ✅ Cost-effective
- ✅ Self-hosting option
- ❌ Manual scaling
- ❌ Smaller ecosystem

**Pinecone**:
- ✅ Zero-ops managed service
- ✅ Automatic scaling
- ✅ Large ecosystem
- ❌ No self-hosting
- ❌ Higher cost
- ⚠️ Payload filtering less flexible

**Weaviate**:
- ✅ Native reranking
- ✅ GraphQL queries
- ✅ Modular architecture
- ⚠️ More complex setup
- ⚠️ Slightly slower payload queries

**pgvector**:
- ✅ SQL interface (familiar)
- ✅ Low cost
- ❌ Poor array support
- ❌ Limited to Postgres scale
- ❌ Slower for metadata filtering

**Milvus**:
- ✅ Billion-scale vectors
- ✅ Enterprise features
- ⚠️ Complex Kubernetes setup
- ⚠️ Heavier resource requirements

---

## When to Choose Qdrant

### Choose Qdrant if:

✅ **Need rich metadata filtering** (this pipeline's core requirement)
- 60%+ of queries use metadata filters
- Array field filtering is common
- Range queries on numeric fields

✅ **Want self-hosting option**
- Healthcare compliance (HIPAA)
- Data sovereignty requirements
- Full control over infrastructure

✅ **Budget-conscious**
- Open source + cost-effective cloud
- ~$50-100/month for 1M vectors
- No per-query pricing

✅ **Need quantization**
- Large-scale deployments (>10M vectors)
- Memory optimization critical
- 4x memory savings important

✅ **Want sparse vector support**
- Hybrid search (semantic + keyword)
- BM25 + dense embeddings
- Single query for both

### Choose Alternatives if:

**→ Pinecone**:
- Want zero-ops managed service
- Automatic scaling is critical
- Budget for premium pricing
- Need largest ecosystem

**→ Weaviate**:
- Need built-in reranking models
- GraphQL queries preferred
- Modular ML pipeline
- Strong semantic search focus

**→ pgvector**:
- Already using Postgres heavily
- Prefer SQL interface
- Small scale (<1M vectors)
- Simple use case

**→ Milvus**:
- Enterprise Kubernetes environment
- Need billion-scale vectors
- Complex multi-tenancy
- High-performance requirements

---

## Collection Setup

### Basic Collection (Selective Flattening)

```python
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PayloadSchemaType

client = QdrantClient(url="http://localhost:6333")

# Create collection for Shopify orders
client.create_collection(
    collection_name="shopify_orders",
    vectors_config=VectorParams(
        size=1536,  # OpenAI text-embedding-3-small
        distance=Distance.COSINE
    ),
    # Enable quantization for memory efficiency
    quantization_config={
        "scalar": {
            "type": "int8",
            "quantile": 0.99,
            "always_ram": True
        }
    }
)
```

### Create Payload Indexes

Index all fields you'll filter on:

```python
# Core identifiers
client.create_payload_index(
    collection_name="shopify_orders",
    field_name="order_id",
    field_schema=PayloadSchemaType.INTEGER
)

client.create_payload_index(
    collection_name="shopify_orders",
    field_name="customer_id",
    field_schema=PayloadSchemaType.INTEGER
)

# Geographic filtering
client.create_payload_index(
    collection_name="shopify_orders",
    field_name="shipping_province",
    field_schema=PayloadSchemaType.KEYWORD
)

client.create_payload_index(
    collection_name="shopify_orders",
    field_name="shipping_country",
    field_schema=PayloadSchemaType.KEYWORD
)

# Array fields (extracted from line_items)
client.create_payload_index(
    collection_name="shopify_orders",
    field_name="vendors",
    field_schema=PayloadSchemaType.KEYWORD  # Enables $contains
)

client.create_payload_index(
    collection_name="shopify_orders",
    field_name="skus",
    field_schema=PayloadSchemaType.KEYWORD
)

# Numeric fields
client.create_payload_index(
    collection_name="shopify_orders",
    field_name="total_price",
    field_schema=PayloadSchemaType.FLOAT
)

# Temporal fields
client.create_payload_index(
    collection_name="shopify_orders",
    field_name="created_at",
    field_schema=PayloadSchemaType.DATETIME
)
```

### Multi-Collection Setup (Hybrid Architecture)

For healthcare FHIR example with different architectures per resource type:

```python
# Collection 1: Patients (hierarchical with documents)
client.create_collection(
    collection_name="fhir_patients",
    vectors_config=VectorParams(size=3072, distance=Distance.COSINE),
    quantization_config={"scalar": {"type": "int8"}}
)

client.create_payload_index("fhir_patients", "patient_id", PayloadSchemaType.KEYWORD)
client.create_payload_index("fhir_patients", "mrn", PayloadSchemaType.KEYWORD)
client.create_payload_index("fhir_patients", "age", PayloadSchemaType.INTEGER)
client.create_payload_index("fhir_patients", "gender", PayloadSchemaType.KEYWORD)

# Collection 2: Observations (flattened without documents)
client.create_collection(
    collection_name="fhir_observations",
    vectors_config=VectorParams(size=3072, distance=Distance.COSINE),
    quantization_config={"scalar": {"type": "int8"}},
    shard_number=4  # Shard for 20M observations
)

client.create_payload_index("fhir_observations", "patient_id", PayloadSchemaType.KEYWORD)
client.create_payload_index("fhir_observations", "observation_code", PayloadSchemaType.KEYWORD)
client.create_payload_index("fhir_observations", "value", PayloadSchemaType.FLOAT)  # Critical!
client.create_payload_index("fhir_observations", "effective_date", PayloadSchemaType.DATETIME)
```

---

## Ingestion Patterns

### Selective Flattening (Reference Architecture)

**Recommended for most use cases:**

```python
from qdrant_client.models import PointStruct
import hashlib

def ingest_shopify_order(order_json, embedding_fn):
    """
    Ingest Shopify order using selective flattening + reference architecture.
    Does NOT store full document (saves 5x storage).
    """

    # Generate embedding of full order context
    content = generate_order_summary(order_json)
    embedding = embedding_fn(content)

    # Extract vendors from line items array
    vendors = list(set([item['vendor'] for item in order_json['line_items']]))
    skus = [item['sku'] for item in order_json['line_items']]
    product_ids = [item['product_id'] for item in order_json['line_items']]

    # Create point (NO full document)
    point = PointStruct(
        id=order_json['id'],  # Use order_id as point ID
        vector=embedding,
        payload={
            # Core identifiers
            "order_id": order_json['id'],
            "order_number": order_json['order_number'],
            "customer_id": order_json['customer']['id'],
            "customer_email": order_json['email'],

            # Geographic (flattened from nested address)
            "shipping_province": order_json['shipping_address']['province'],
            "shipping_country": order_json['shipping_address']['country'],
            "shipping_city": order_json['shipping_address']['city'],

            # Extracted from arrays
            "vendors": vendors,
            "skus": skus,
            "product_ids": product_ids,

            # Numeric fields
            "total_price": float(order_json['total_price']),
            "currency": order_json['currency'],

            # Temporal
            "created_at": order_json['created_at'],

            # Status flags
            "financial_status": order_json['financial_status'],
            "fulfillment_status": order_json['fulfillment_status'],

            # Content for LLM
            "content": content,

            # Reference to full document in source DB
            "source_db": "postgres",
            "source_table": "orders",
            "source_id": order_json['id']

            # NO "document" field - saves 50 KB per order!
        }
    )

    client.upsert(collection_name="shopify_orders", points=[point])

def generate_order_summary(order_json):
    """Generate human-readable summary for LLM."""
    customer_name = f"{order_json['customer']['first_name']} {order_json['customer']['last_name']}"

    products = [f"{item['title']} from {item['vendor']}"
                for item in order_json['line_items']]

    return (
        f"Order #{order_json['order_number']} (ID: {order_json['id']}) "
        f"placed on {order_json['created_at']} for {customer_name} "
        f"({order_json['email']}). "
        f"Products: {', '.join(products)}. "
        f"Shipped to {order_json['shipping_address']['city']}, "
        f"{order_json['shipping_address']['province']}, "
        f"{order_json['shipping_address']['country']}. "
        f"Total: ${order_json['total_price']} {order_json['currency']}. "
        f"Status: {order_json['financial_status']}, "
        f"{order_json['fulfillment_status']}."
    )
```

### Batch Ingestion

For high-throughput ingestion:

```python
def batch_ingest_orders(orders_batch, embedding_fn, batch_size=100):
    """
    Ingest orders in batches for better performance.

    Performance: 500-1000 orders/sec single node
    """
    points = []

    for order_json in orders_batch:
        content = generate_order_summary(order_json)
        embedding = embedding_fn(content)

        # ... extract fields as above ...

        point = PointStruct(id=order_json['id'], vector=embedding, payload=payload)
        points.append(point)

        # Upsert in batches
        if len(points) >= batch_size:
            client.upsert(collection_name="shopify_orders", points=points)
            points = []

    # Upsert remaining
    if points:
        client.upsert(collection_name="shopify_orders", points=points)
```

### Pure Flattening (With Cross-References)

For cases where you need entity-level chunks:

```python
def ingest_order_flattened(order_json, embedding_fn):
    """
    Pure flattening: Create separate chunks for order, line items, fulfillments.
    """
    points = []
    order_id = order_json['id']

    # 1. Order summary chunk
    order_content = generate_order_summary(order_json)
    order_point = PointStruct(
        id=f"order_{order_id}",
        vector=embedding_fn(order_content),
        payload={
            "chunk_type": "order_summary",
            "order_id": order_id,
            "child_chunk_ids": [],  # Populate below
            "content": order_content,
            # ... other metadata ...
        }
    )
    points.append(order_point)

    # 2. Line item chunks
    for idx, item in enumerate(order_json['line_items']):
        item_content = (
            f"Order #{order_json['order_number']}: "
            f"{item['title']} - {item['variant_title']} from {item['vendor']}. "
            f"SKU: {item['sku']}, Price: ${item['price']}, Qty: {item['quantity']}"
        )

        item_id = f"order_{order_id}_item_{item['id']}"
        item_point = PointStruct(
            id=item_id,
            vector=embedding_fn(item_content),
            payload={
                "chunk_type": "line_item",
                "order_id": order_id,
                "parent_chunk_id": f"order_{order_id}",
                "line_item_id": item['id'],
                "vendor": item['vendor'],
                "sku": item['sku'],
                "content": item_content,
            }
        )
        points.append(item_point)
        order_point.payload["child_chunk_ids"].append(item_id)

    # 3. Fulfillment chunks
    for fulfillment in order_json.get('fulfillments', []):
        fulfillment_content = (
            f"Fulfillment for order #{order_json['order_number']}: "
            f"Status {fulfillment['status']}, "
            f"Tracking: {fulfillment.get('tracking_company', 'N/A')}"
        )

        fulfillment_id = f"order_{order_id}_fulfillment_{fulfillment['id']}"
        fulfillment_point = PointStruct(
            id=fulfillment_id,
            vector=embedding_fn(fulfillment_content),
            payload={
                "chunk_type": "fulfillment",
                "order_id": order_id,
                "parent_chunk_id": f"order_{order_id}",
                "fulfillment_id": fulfillment['id'],
                "status": fulfillment['status'],
                "content": fulfillment_content,
            }
        )
        points.append(fulfillment_point)
        order_point.payload["child_chunk_ids"].append(fulfillment_id)

    # Batch upsert all chunks
    client.upsert(collection_name="shopify_orders", points=points)
```

---

## Query Patterns

### Pattern 1: Metadata Filtering + Vector Search

**Most common pattern (60% of queries):**

```python
from qdrant_client.models import Filter, FieldCondition, MatchAny, Range

def find_apple_products_ontario(query_text="Apple products"):
    """
    Query: "Find Apple products shipped to Ontario"

    Performance: 30-80ms (depends on filter selectivity)
    """
    results = client.search(
        collection_name="shopify_orders",
        query_vector=embed(query_text),
        query_filter=Filter(
            must=[
                FieldCondition(
                    key="vendors",
                    match=MatchAny(any=["Apple"])
                ),
                FieldCondition(
                    key="shipping_province",
                    match="Ontario"
                )
            ]
        ),
        limit=50,
        with_payload=True
    )

    return results
```

### Pattern 2: Pure Metadata Filtering (No Vector Search)

**For analytics queries:**

```python
def get_orders_by_customer(customer_id):
    """
    Query: "Get all orders for customer"

    Performance: 5-20ms (no vector search needed)
    """
    results = client.scroll(
        collection_name="shopify_orders",
        scroll_filter=Filter(
            must=[
                FieldCondition(key="customer_id", match=customer_id)
            ]
        ),
        limit=1000,
        with_payload=True,
        with_vectors=False  # Don't fetch vectors (faster)
    )

    return results[0]  # Returns points, not search results

def revenue_by_vendor_this_month():
    """
    Query: "Revenue analytics by vendor"

    Performance: 10-50ms depending on result set size
    """
    from datetime import datetime, timedelta
    start_date = (datetime.now() - timedelta(days=30)).isoformat()

    orders = client.scroll(
        collection_name="shopify_orders",
        scroll_filter=Filter(
            must=[
                FieldCondition(
                    key="created_at",
                    range=Range(gte=start_date)
                )
            ]
        ),
        limit=10000,
        with_payload=["vendors", "total_price"],
        with_vectors=False
    )[0]

    # Aggregate client-side
    from collections import defaultdict
    revenue = defaultdict(float)
    for order in orders:
        price = order.payload['total_price']
        vendor_count = len(order.payload['vendors'])
        for vendor in order.payload['vendors']:
            revenue[vendor] += price / vendor_count

    return dict(revenue)
```

### Pattern 3: Range Queries

**For numeric or temporal filtering:**

```python
def find_high_value_orders(min_price=500, state="CA"):
    """
    Query: "Orders over $500 in California"

    Performance: 20-60ms
    """
    results = client.scroll(
        collection_name="shopify_orders",
        scroll_filter=Filter(
            must=[
                FieldCondition(
                    key="total_price",
                    range=Range(gte=min_price)
                ),
                FieldCondition(
                    key="shipping_province",
                    match=state
                )
            ]
        ),
        limit=1000
    )[0]

    return results
```

### Pattern 4: Hybrid Search (Dense + Sparse)

**For best semantic + keyword matching:**

```python
from qdrant_client.models import SparseVector

def hybrid_search(query_text, sparse_indices, sparse_values):
    """
    Query: Semantic search + BM25 keyword matching

    Performance: 40-100ms
    """
    dense_embedding = embed(query_text)

    results = client.search(
        collection_name="shopify_orders",
        query_vector=dense_embedding,
        sparse_query_vector=SparseVector(
            indices=sparse_indices,  # From BM25
            values=sparse_values
        ),
        limit=20
    )

    return results
```

### Pattern 5: Fetch Full Document from Source

**When you need complete JSON:**

```python
def search_and_fetch_full_docs(query, filters):
    """
    Query: Search in Qdrant, fetch full docs from source DB

    Total latency: 30-80ms (Qdrant) + 10-30ms (source DB)
    """
    # Step 1: Search in Qdrant
    results = client.search(
        collection_name="shopify_orders",
        query_vector=embed(query),
        query_filter=filters,
        limit=50
    )

    # Step 2: Extract source IDs
    source_ids = [r.payload["source_id"] for r in results]

    # Step 3: Fetch from source DB (PostgreSQL example)
    import psycopg2
    conn = psycopg2.connect("dbname=shopify user=postgres")
    cur = conn.cursor()

    cur.execute(
        "SELECT * FROM orders WHERE id = ANY(%s)",
        (source_ids,)
    )
    full_orders = cur.fetchall()

    conn.close()
    return full_orders
```

### Pattern 6: Cross-Reference Navigation (Pure Flattening)

**For traversing chunk relationships:**

```python
def get_order_with_details(order_id):
    """
    Get order summary + all related chunks

    Performance: 10-30ms
    """
    # Step 1: Get parent chunk
    order = client.retrieve(
        collection_name="shopify_orders",
        ids=[f"order_{order_id}"]
    )[0]

    # Step 2: Get all children
    child_ids = order.payload.get("child_chunk_ids", [])
    if child_ids:
        children = client.retrieve(
            collection_name="shopify_orders",
            ids=child_ids
        )
    else:
        children = []

    # Step 3: Organize by type
    return {
        "order": order,
        "line_items": [c for c in children if c.payload["chunk_type"] == "line_item"],
        "fulfillments": [c for c in children if c.payload["chunk_type"] == "fulfillment"]
    }

def find_failed_apple_shipments():
    """
    Complex query: Failed fulfillments of Apple products

    Performance: 50-150ms (multi-step)
    """
    # Step 1: Find failed fulfillments
    failed = client.scroll(
        collection_name="shopify_orders",
        scroll_filter=Filter(
            must=[
                FieldCondition(key="chunk_type", match="fulfillment"),
                FieldCondition(key="status", match="failure")
            ]
        ),
        limit=1000
    )[0]

    # Step 2: Get parent order IDs
    order_ids = list(set([f.payload["order_id"] for f in failed]))

    # Step 3: Filter for Apple products
    orders = client.scroll(
        collection_name="shopify_orders",
        scroll_filter=Filter(
            must=[
                FieldCondition(key="order_id", match=MatchAny(any=order_ids)),
                FieldCondition(key="vendors", match=MatchAny(any=["Apple"]))
            ]
        )
    )[0]

    return orders
```

---

## Optimizations

### 1. Quantization for Memory Savings

**Reduces memory by 4x with minimal accuracy loss:**

```python
# Enable quantization
client.update_collection(
    collection_name="shopify_orders",
    quantization_config={
        "scalar": {
            "type": "int8",  # 4x compression
            "quantile": 0.99,
            "always_ram": True  # Keep quantized vectors in RAM
        }
    }
)
```

**Impact**:
- 1M orders (reference architecture): 13 GB → 8.5 GB
- 1M orders (with full docs): 63 GB → 58.5 GB
- Query latency impact: <5% slower
- Accuracy impact: <2% degradation

### 2. Collection Aliases for Zero-Downtime Migrations

**Safe schema evolution:**

```python
# Step 1: Create new collection with updated schema
client.create_collection("shopify_orders_v2", ...)

# Step 2: Reindex data (background process)
# ... migration logic ...

# Step 3: Switch alias atomically (zero downtime)
client.update_collection_aliases(
    change_aliases_operations=[
        {"delete_alias": {"alias_name": "shopify_orders_prod"}},
        {"create_alias": {
            "alias_name": "shopify_orders_prod",
            "collection_name": "shopify_orders_v2"
        }}
    ]
)

# Step 4: Delete old collection after validation
client.delete_collection("shopify_orders_v1")
```

### 3. Sharding for Horizontal Scaling

**For >10M vectors:**

```python
client.create_collection(
    collection_name="fhir_observations",  # 20M observations
    vectors_config=VectorParams(size=3072, distance=Distance.COSINE),
    shard_number=4,  # Distribute across 4 shards
    replication_factor=2  # 2x replication for HA
)
```

**Shard sizing guidelines**:
- <1M vectors: 1 shard
- 1-10M vectors: 2-4 shards
- 10-50M vectors: 4-8 shards
- >50M vectors: 8-16 shards

### 4. Optimize Payload Size

**Keep payloads lean:**

```python
# BAD: Storing large text in payload
payload = {
    "content": "Very long text..." * 1000,  # 100 KB
    "full_document": {...}  # 50 KB
}

# GOOD: Keep only searchable content
payload = {
    "content": "Order #12345 summary...",  # 2 KB
    "source_id": 12345  # Reference only
}
```

### 5. Batch Retrievals

**Fetch multiple points efficiently:**

```python
# BAD: Individual fetches (N queries)
orders = []
for order_id in order_ids:
    order = client.retrieve(collection_name="orders", ids=[order_id])
    orders.append(order[0])

# GOOD: Batch fetch (1 query)
orders = client.retrieve(
    collection_name="orders",
    ids=order_ids  # Up to 1000 IDs
)
```

---

## Performance Benchmarks

### Shopify Use Case (1M orders, selective flattening)

#### Storage Comparison

| Configuration | Per Order | 1M Orders | Monthly Cost | Notes |
|--------------|-----------|-----------|--------------|-------|
| Full documents (raw) | 63 KB | 63 GB | ~$150-200 | Vectors + metadata + content + document |
| Full documents (quantized) | 58.5 KB | 58.5 GB | ~$140-190 | Vectors compressed 4x (6KB→1.5KB) |
| **Reference architecture (raw)** ⭐ | 13 KB | 13 GB | ~$40-60 | No document field, source_id only |
| **Reference architecture (quantized)** | 8.5 KB | 8.5 GB | ~$25-40 | **Recommended: 7.4x cheaper than full docs** |

#### Query Latency

| Query Type | Latency | Notes |
|------------|---------|-------|
| Metadata filter only | 5-20ms | Pure filtering, no vector search |
| Vector search + filter | 30-80ms | Depends on filter selectivity |
| Pure vector search | 20-50ms | No filtering |
| Fetch from source DB | +10-30ms | If full document needed |
| Batch retrieval (100 docs) | 15-40ms | Using `retrieve()` |

#### Ingestion Throughput

| Configuration | Throughput | Notes |
|---------------|------------|-------|
| Single node, single thread | 500-1000 orders/sec | Basic ingestion |
| Single node, batched (100) | 1500-2500 orders/sec | With batch upserts |
| 4 nodes, parallel | 4000-8000 orders/sec | Distributed ingestion |

**Total time for 1M orders**: 7-30 minutes depending on parallelization

---

### FHIR Use Case (100K patients, 20M observations, hybrid)

#### Storage Breakdown

| Resource Type | Architecture | Per Item | Total | Notes |
|--------------|--------------|----------|-------|-------|
| Patients (100K) | Hierarchical w/ docs | 64 KB | 6.4 GB | Full FHIR Patient resources |
| Observations (20M) | Flattened w/o docs | 10 KB | 200 GB | Metadata + content only |
| **Total (raw)** | Hybrid | - | **206.4 GB** | Mixed approach |
| **Total (quantized)** | Hybrid | - | **156 GB** | Vectors: ~50 GB savings |

**With reference architecture for observations**:
- Observations: 20M × 4 KB = 80 GB
- Total (quantized): **82 GB** (2.5x cheaper)

#### Query Latency

| Query Type | Latency | Notes |
|------------|---------|-------|
| Lab value filter (precise) | 10-30ms | `value >= 126` with index |
| Patient record lookup | 5-15ms | Direct ID retrieval |
| Complex cohort query | 100-300ms | Multiple filters + aggregation |
| Semantic similarity | 80-200ms | Vector search on observations |
| Cross-resource join | 50-150ms | Patient → encounters → observations |

#### Ingestion Throughput

| Configuration | Throughput | Notes |
|---------------|------------|-------|
| Single node | 1000-2000 obs/sec | For observations |
| With 4 shards | 4000-6000 obs/sec | Distributed writes |
| Patients | 500-1000 patients/sec | Smaller volume |

**Total time for 100K patients + 20M observations**: 2-6 hours

---

### Real-World Performance Tips

**1. Filter selectivity matters**:
```python
# Fast (high selectivity): Returns 100 results from 1M
filter={"shipping_province": "Ontario"}  # 20ms

# Slow (low selectivity): Returns 500K results from 1M
filter={"currency": "USD"}  # 500ms
```

**2. Limit result size**:
```python
# Use scroll for large result sets
results, offset = client.scroll(
    collection_name="orders",
    scroll_filter=filter,
    limit=1000,  # Page size
    offset=offset  # Continue from previous page
)
```

**3. Use `with_vectors=False` when you don't need embeddings**:
```python
# Faster payload-only retrieval
results = client.scroll(
    collection_name="orders",
    scroll_filter=filter,
    with_vectors=False  # 20-30% faster
)
```

---

### HIPAA Compliance (Healthcare)

For FHIR healthcare use case:

**Requirements**:
1. ✅ Self-hosted (no cloud service)
2. ✅ Encrypted at rest (volume encryption)
3. ✅ Encrypted in transit (TLS)
4. ✅ Access controls (authentication + RBAC)
5. ✅ Audit logging (middleware layer)
6. ✅ Data retention policies (automated cleanup)

**Audit logging implementation**:

```python
import logging
from datetime import datetime

def audit_search(user_id, query, filters):
    """Log all searches for HIPAA compliance."""

    # Perform search
    results = client.search(
        collection_name="fhir_observations",
        query_vector=query,
        query_filter=filters
    )

    # Log access
    audit_log = {
        "timestamp": datetime.now().isoformat(),
        "user_id": user_id,
        "action": "search",
        "collection": "fhir_observations",
        "filters": str(filters),
        "result_count": len(results),
        "patient_ids": list(set([r.payload["patient_id"] for r in results]))
    }

    logging.info(f"AUDIT: {audit_log}")

    # Store in audit database
    audit_db.insert(audit_log)

    return results
```

### Scaling Strategy

**Vertical scaling** (single node):
- Up to 10M vectors: 16 GB RAM, 4 CPU cores
- 10-50M vectors: 32 GB RAM, 8 CPU cores
- 50-100M vectors: 64 GB RAM, 16 CPU cores

**Horizontal scaling** (sharding):
```python
# Determine shard count
vectors_count = 20_000_000  # 20M observations
shard_size_target = 5_000_000  # 5M per shard
shard_number = math.ceil(vectors_count / shard_size_target)  # 4 shards

client.create_collection(
    collection_name="fhir_observations",
    shard_number=shard_number,
    replication_factor=2  # For HA
)
```

### Cost Optimization

**1. Use quantization** (4x memory savings):
```python
quantization_config={"scalar": {"type": "int8"}}
```

**2. Don't store full documents** (5x storage savings):
```python
payload={
    # ... metadata only ...
    "source_id": order_id  # Reference instead
}
```

**3. Use reference architecture** (0.2x storage vs 1.0x):
- 1M orders: $40/month vs $200/month
- Savings: $160/month = $1920/year

**4. Right-size shards**:
- Don't over-shard (overhead)
- Target 5-10M vectors per shard

**5. Use Qdrant Cloud vs self-hosted**:
- Small scale (<1M): Cloud often cheaper (no ops)
- Large scale (>10M): Self-hosted cheaper (economies of scale)

---

## Troubleshooting

### Common Issues

**1. Slow queries despite indexes**

Problem: Queries taking >500ms

Solution:
```python
# Check if indexes exist
info = client.get_collection("orders")
print(info.payload_schema)  # Should show indexed fields

# Create missing indexes
client.create_payload_index(
    collection_name="orders",
    field_name="vendors",
    field_schema=PayloadSchemaType.KEYWORD
)
```

**2. Out of memory errors**

Problem: Qdrant crashing with OOM

Solution:
```python
# Enable quantization
client.update_collection(
    collection_name="orders",
    quantization_config={"scalar": {"type": "int8"}}
)

# Or reduce vectors in RAM
# Use disk-based storage for vectors
```

**3. Ingestion too slow**

Problem: <100 points/sec ingestion rate

Solution:
```python
# Use batching
points = []
for order in orders:
    points.append(create_point(order))
    if len(points) >= 100:
        client.upsert(collection_name="orders", points=points)
        points = []

# Use parallel workers
from concurrent.futures import ThreadPoolExecutor
with ThreadPoolExecutor(max_workers=4) as executor:
    executor.map(ingest_batch, order_batches)
```

**4. High latency on filtered queries**

Problem: Queries with filters taking >200ms

Solution:
```python
# Index all filtered fields
for field in ["vendors", "shipping_province", "total_price"]:
    client.create_payload_index(collection_name="orders", field_name=field)

# Increase filter selectivity (filter more aggressively)
filter=Filter(
    must=[
        # High selectivity filters first
        FieldCondition(key="order_id", match=12345),
        FieldCondition(key="customer_id", match=67890),
    ]
)
```

---

## Comparison: When NOT to Use Qdrant

### Use Pinecone Instead If:

- ✅ Want zero-ops managed service
- ✅ Need automatic scaling
- ✅ Don't want to manage infrastructure
- ✅ Budget allows premium pricing
- ❌ Don't need self-hosting

### Use Weaviate Instead If:

- ✅ Need native reranking models
- ✅ Prefer GraphQL API
- ✅ Want modular ML pipeline
- ❌ Payload filtering is secondary

### Use pgvector Instead If:

- ✅ Already heavily invested in Postgres
- ✅ Small scale (<1M vectors)
- ✅ Simple use case
- ✅ Prefer SQL interface
- ❌ Don't need array field indexing

### Use Milvus Instead If:

- ✅ Billion-scale vectors
- ✅ Enterprise Kubernetes setup
- ✅ Complex multi-tenancy
- ❌ Don't need Qdrant's simplicity

---

## Summary

### Qdrant Excels For This Pipeline Because:

1. ✅ **Rich payload filtering** - Core requirement for metadata-first queries
2. ✅ **Array field indexing** - Extracted vendors, SKUs, product IDs
3. ✅ **Cost-effective** - ~$50/month for 1M orders (reference architecture)
4. ✅ **Self-hosting** - Critical for healthcare/compliance
5. ✅ **Quantization** - 4x memory savings helps offset storage overhead
6. ✅ **Sparse vectors** - True hybrid search (semantic + keyword)

### Trade-offs:

1. ⚠️ **Manual scaling** - Need to configure sharding
2. ⚠️ **Learning curve** - More complex than Pinecone
3. ⚠️ **Smaller ecosystem** - Fewer integrations vs Pinecone
4. ⚠️ **External reranking** - Need to integrate Cohere/Jina APIs

### Bottom Line:

**Qdrant is an excellent choice for hierarchical JSON pipelines** due to strong payload filtering and cost-effectiveness. You trade some operational simplicity for better control and lower costs.

**Recommended configuration**:
- Architecture: Selective flattening + reference architecture
- Storage: 0.2x (13 KB per order)
- Quantization: Enabled (int8)
- Sharding: 4-8 shards for >10M vectors
- Cost: ~$40-60/month for 1M orders

**Next steps**:
1. Start with single-node deployment
2. Use reference architecture (don't store full docs)
3. Enable quantization
4. Index all filtered fields
5. Scale horizontally when >10M vectors

---

## Code Repository

For complete working examples, see:
- `examples/shopify_qdrant.py` - E-commerce implementation
---

## Additional Resources

- **Qdrant Documentation**: https://qdrant.tech/documentation/
- **Qdrant GitHub**: https://github.com/qdrant/qdrant
- **Qdrant Cloud**: https://cloud.qdrant.io/
- **Community Discord**: https://qdrant.to/discord
- **Performance Tips**: https://qdrant.tech/documentation/guides/optimize/
