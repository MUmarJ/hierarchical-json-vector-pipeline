# Examples: Hierarchical JSON to Vector DB

This directory contains working implementations of the hierarchical JSON to vector DB pipeline for real-world use cases.

## 📁 Files

- **`shopify_qdrant.py`** - E-commerce orders (Shopify) with selective flattening + reference architecture
- **`fhir_qdrant.py`** - Healthcare records (FHIR) with hybrid architecture
- **`requirements.txt`** - Python dependencies
- **`.env.example`** - Environment variables template

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Setup Environment

```bash
cp .env.example .env
# Edit .env with your API keys
```

Required environment variables:
```bash
QDRANT_URL=http://localhost:6333
OPENAI_API_KEY=your_openai_key_here
POSTGRES_DSN=dbname=shopify user=postgres  # Only for Shopify example
```

### 3. Start Qdrant

```bash
docker run -p 6333:6333 -p 6334:6334 \
    -v $(pwd)/qdrant_storage:/qdrant/storage \
    qdrant/qdrant
```

---

## 📦 Example 1: Shopify Orders

**Architecture**: Selective Flattening + Reference Architecture

**Storage**: 13 KB per order (5x cheaper than storing full documents)

### Setup

```bash
# Create collections and indexes
python shopify_qdrant.py --mode setup
```

### Ingest Sample Data

```bash
# Ingest sample orders (uses built-in sample data)
python shopify_qdrant.py --mode ingest
```

Or ingest from PostgreSQL:
```python
# In your code:
import psycopg2
conn = psycopg2.connect("dbname=shopify user=postgres")
cur = conn.cursor()
cur.execute("SELECT data FROM orders LIMIT 1000")
orders = [json.loads(row[0]) for row in cur.fetchall()]

pipeline = ShopifyVectorPipeline()
pipeline.ingest_batch(orders)
```

### Query Examples

**Basic semantic search:**
```bash
python shopify_qdrant.py --mode query \
    --query "orders with shipping delays"
```

**Filtered search:**
```bash
# Find Apple products in Ontario
python shopify_qdrant.py --mode query \
    --query "Apple products" \
    --province "Ontario"
```

**Vendor-specific:**
```bash
python shopify_qdrant.py --mode query \
    --query "electronics orders" \
    --vendor "Apple"
```

### Query Patterns in Code

```python
from shopify_qdrant import ShopifyVectorPipeline

pipeline = ShopifyVectorPipeline()

# 1. Metadata filtering (fast, 5-20ms)
results = pipeline.search_orders(
    query="",  # No semantic search
    filters={
        "vendors": ["Apple"],
        "shipping_province": "Ontario",
        "total_price": {"gte": 100}
    }
)

# 2. Semantic + metadata (30-80ms)
results = pipeline.search_orders(
    query="shipping complaints delays",
    filters={"shipping_country": "Canada"}
)

# 3. Get full document from source DB
order_with_full_doc = pipeline.get_order_with_full_document(order_id=12345)
# Returns: {"metadata": {...}, "full_document": {...}}
```

### Expected Performance

| Query Type | Latency | Notes |
|------------|---------|-------|
| Metadata filter only | 5-20ms | Pure filtering |
| Semantic + filter | 30-80ms | With vector search |
| Fetch full doc from source | +10-30ms | PostgreSQL lookup |

---

## 🏥 Example 2: FHIR Healthcare

**Architecture**: Hybrid (Hierarchical patients + Flattened observations)

**Storage**: 
- Patients: 64 KB each (with full FHIR resources)
- Observations: 4 KB each (metadata only, no full resources)

### Setup

```bash
# Create collections and indexes
python fhir_qdrant.py --mode setup
```

### Ingest Sample Data

```bash
# Ingest sample patient
python fhir_qdrant.py --mode ingest \
    --resource-type Patient

# Ingest sample observations
python fhir_qdrant.py --mode ingest \
    --resource-type Observation
```

Or from JSON files:
```bash
python fhir_qdrant.py --mode ingest \
    --resource-type Patient \
    --file patients.json

python fhir_qdrant.py --mode ingest \
    --resource-type Observation \
    --file observations.json
```

### Query Examples

**Clinical query: Find diabetic patients**
```bash
# Find patients with glucose > 126 mg/dL (diabetes threshold)
python fhir_qdrant.py --mode query \
    --obs-code "2339-0" \
    --min-value 126
```

**Semantic search: Similar cases**
```bash
python fhir_qdrant.py --mode query \
    --query "chest pain elevated blood pressure high glucose"
```

### Query Patterns in Code

```python
from fhir_qdrant import FHIRVectorPipeline

pipeline = FHIRVectorPipeline()

# 1. Lab value filtering (precise, 10-30ms)
diabetic_patients = pipeline.find_patients_with_lab_values(
    observation_code="2339-0",  # Glucose LOINC code
    min_value=126  # Diabetic threshold
)

# 2. Cohort identification for research
cohort = pipeline.identify_cohort(
    observation_code="4548-4",  # HbA1c LOINC code
    min_value=6.5,
    time_window_days=365
)
print(f"Cohort size: {cohort['total_patients']} patients")

# 3. Clinical decision support (semantic)
similar_cases = pipeline.search_similar_cases(
    query="chest pain elevated BP high glucose",
    patient_age_range=(50, 70),
    limit=20
)

# 4. Patient timeline
timeline = pipeline.get_patient_timeline(patient_id="patient-67890")
print(f"Patient has {timeline['observation_count']} observations")
```

### Expected Performance

| Query Type | Latency | Notes |
|------------|---------|-------|
| Lab value filter | 10-30ms | Numeric range on indexed field |
| Patient lookup | 5-15ms | Direct ID retrieval |
| Cohort query | 100-300ms | Complex filtering + aggregation |
| Similar cases | 80-200ms | Semantic search |

---

## 📊 Storage Comparison

### Shopify (1M orders)

| Approach | Per Order | Total | Monthly Cost |
|----------|-----------|-------|--------------|
| **Full documents** | 63 KB | 63 GB | ~$150-200 |
| **Reference architecture** ⭐ | 13 KB | 13 GB | ~$40-60 |
| **Savings** | **5x smaller** | **50 GB** | **$100-140** |

### FHIR (100K patients + 20M observations)

| Approach | Storage | Notes |
|----------|---------|-------|
| **Full documents (all)** | 406 GB | If storing all FHIR resources |
| **Hybrid (recommended)** ⭐ | 86 GB | Patients: full, Observations: metadata only |
| **Savings** | **5x smaller** | Massive savings on observations |

---

## 🔧 Customization Guide

### Shopify: Adding Custom Fields

Edit `extract_metadata()` to add your fields:

```python
def extract_metadata(self, order: Dict[str, Any]) -> Dict[str, Any]:
    metadata = {
        # ... existing fields ...
        
        # Add custom fields
        "custom_field": order.get('custom_field'),
        "shipping_method": order['shipping_lines'][0]['code'] if order.get('shipping_lines') else None,
    }
    return metadata
```

Then add indexes:
```python
client.create_payload_index(
    collection_name="shopify_orders",
    field_name="custom_field",
    field_schema=PayloadSchemaType.KEYWORD
)
```

### FHIR: Adding More Resource Types

Add a new resource type (e.g., Condition):

```python
def ingest_condition(self, condition_resource: Dict[str, Any]):
    condition_id = condition_resource['id']
    
    # Resolve patient reference
    patient_ref = condition_resource['subject']['reference']
    patient_id = self.resolve_fhir_reference(patient_ref)
    
    # Extract condition code (SNOMED, ICD-10)
    code_obj = condition_resource.get('code', {})
    coding = code_obj.get('coding', [{}])[0]
    condition_code = coding.get('code')
    condition_display = coding.get('display')
    
    # Generate content
    content = f"{condition_display} diagnosed for patient {patient_id}"
    embedding = self.generate_embedding(content)
    
    # Create point
    point = PointStruct(
        id=str(uuid.uuid4()),
        vector=embedding,
        payload={
            "resource_type": "Condition",
            "condition_id": condition_id,
            "patient_id": patient_id,
            "condition_code": condition_code,
            "condition_display": condition_display,
            "content": content
        }
    )
    
    self.client.upsert(collection_name="fhir_conditions", points=[point])
```

---

## 🐛 Troubleshooting

### "Connection refused" error

**Problem**: Can't connect to Qdrant

**Solution**: Make sure Qdrant is running:
```bash
docker ps | grep qdrant
# If not running:
docker run -p 6333:6333 qdrant/qdrant
```

### "Unauthorized" error with OpenAI

**Problem**: Invalid OpenAI API key

**Solution**: Check your `.env` file:
```bash
cat .env | grep OPENAI_API_KEY
# Make sure key is correct
```

### Slow ingestion

**Problem**: Ingestion taking too long

**Solution**: Use batching:
```python
# Instead of:
for order in orders:
    pipeline.ingest_order(order)

# Use:
pipeline.ingest_batch(orders, batch_size=100)
```

### Out of memory

**Problem**: Qdrant running out of memory

**Solution**: Enable quantization (already in examples) or increase Docker memory:
```bash
docker run -p 6333:6333 \
    --memory=4g \
    qdrant/qdrant
```

---

## 📚 Further Reading

- **Main Design Doc**: `../hierarchical-json-vector-pipeline.md` - Architecture decision framework
- **Qdrant Guide**: `../qdrant-implementation.md` - Detailed Qdrant implementation
- **Shopify Implementation**: `../shopify-implementation-example.md` - Full Shopify guide
- **FHIR Implementation**: `../fhir-healthcare-implementation.md` - Full FHIR guide

---

## 🤝 Contributing

Found a bug or want to add an example? PRs welcome!

Common additions:
- Add MongoDB as source DB option
- Add more FHIR resource types (Medication, Procedure, etc.)
- Add batch processing for large datasets
- Add monitoring/metrics collection

---

## 📄 License

MIT License - feel free to use in your projects!
