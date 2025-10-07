"""
Shopify Orders to Qdrant Vector DB
===================================

Implementation of selective flattening + reference architecture for e-commerce orders.

Architecture:
- Selective flattening (extract key fields to metadata)
- Reference architecture (no full documents, source_id only)
- Storage: 13 KB per order (5x cheaper than storing full docs)

Requirements:
    pip install qdrant-client openai psycopg2-binary python-dotenv

Usage:
    python shopify_qdrant.py --mode ingest --source postgres
    python shopify_qdrant.py --mode query --query "Apple products in Ontario"
"""

import os
import json
from typing import List, Dict, Any, Optional
from datetime import datetime
from dataclasses import dataclass

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct,
    Filter, FieldCondition, MatchAny, Range,
    PayloadSchemaType
)
import openai
import psycopg2
from dotenv import load_dotenv

load_dotenv()

# Configuration
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
POSTGRES_DSN = os.getenv("POSTGRES_DSN", "dbname=shopify user=postgres")

openai.api_key = OPENAI_API_KEY


@dataclass
class ShopifyOrder:
    """Shopify order data structure"""
    id: int
    order_number: int
    email: str
    created_at: str
    total_price: float
    currency: str
    financial_status: str
    fulfillment_status: str
    customer: Dict[str, Any]
    shipping_address: Dict[str, Any]
    billing_address: Dict[str, Any]
    line_items: List[Dict[str, Any]]
    fulfillments: List[Dict[str, Any]]
    discount_codes: List[Dict[str, Any]]
    tags: str


class ShopifyVectorPipeline:
    """Pipeline for ingesting Shopify orders into Qdrant"""
    
    def __init__(self, qdrant_url: str = QDRANT_URL):
        self.client = QdrantClient(url=qdrant_url)
        self.collection_name = "shopify_orders"
        self.embedding_model = "text-embedding-3-small"
        self.embedding_dim = 1536
    
    def setup_collection(self):
        """Create Qdrant collection with proper indexes"""
        
        # Check if collection exists
        collections = self.client.get_collections().collections
        if any(c.name == self.collection_name for c in collections):
            print(f"Collection '{self.collection_name}' already exists")
            return
        
        print(f"Creating collection '{self.collection_name}'...")
        
        # Create collection
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(
                size=self.embedding_dim,
                distance=Distance.COSINE
            ),
            # Enable quantization for 4x memory savings
            quantization_config={
                "scalar": {
                    "type": "int8",
                    "quantile": 0.99,
                    "always_ram": True
                }
            }
        )
        
        # Create payload indexes for common filters
        indexes = {
            "order_id": PayloadSchemaType.INTEGER,
            "order_number": PayloadSchemaType.INTEGER,
            "customer_id": PayloadSchemaType.INTEGER,
            "shipping_province": PayloadSchemaType.KEYWORD,
            "shipping_country": PayloadSchemaType.KEYWORD,
            "vendors": PayloadSchemaType.KEYWORD,  # Array field
            "skus": PayloadSchemaType.KEYWORD,
            "total_price": PayloadSchemaType.FLOAT,
            "created_at": PayloadSchemaType.DATETIME,
            "financial_status": PayloadSchemaType.KEYWORD,
            "fulfillment_status": PayloadSchemaType.KEYWORD,
        }
        
        for field_name, field_schema in indexes.items():
            print(f"  Creating index on '{field_name}'...")
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name=field_name,
                field_schema=field_schema
            )
        
        print("✓ Collection setup complete")
    
    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding using OpenAI"""
        response = openai.embeddings.create(
            model=self.embedding_model,
            input=text
        )
        return response.data[0].embedding
    
    def generate_order_summary(self, order: Dict[str, Any]) -> str:
        """Generate human-readable summary for LLM context"""
        
        customer_name = f"{order['customer']['first_name']} {order['customer']['last_name']}"
        
        # Extract product info
        products = [
            f"{item['title']} ({item.get('variant_title', 'default')}) from {item['vendor']}"
            for item in order['line_items']
        ]
        
        # Build summary
        summary = (
            f"Order #{order['order_number']} (ID: {order['id']}) "
            f"placed on {order['created_at'][:10]} for {customer_name} "
            f"({order['email']}). "
            f"Products: {', '.join(products)}. "
            f"Shipped to {order['shipping_address']['city']}, "
            f"{order['shipping_address']['province']}, "
            f"{order['shipping_address']['country']}. "
            f"Total: ${order['total_price']} {order['currency']}. "
            f"Payment status: {order['financial_status']}. "
            f"Fulfillment status: {order['fulfillment_status']}."
        )
        
        # Add tags if present
        if order.get('tags'):
            summary += f" Tags: {order['tags']}."
        
        return summary
    
    def extract_metadata(self, order: Dict[str, Any]) -> Dict[str, Any]:
        """Extract metadata for filtering (selective flattening)"""
        
        # Extract vendors from line items array
        vendors = list(set([item['vendor'] for item in order['line_items']]))
        skus = [item['sku'] for item in order['line_items'] if item.get('sku')]
        product_ids = [item['product_id'] for item in order['line_items']]
        
        # Extract fulfillment statuses
        fulfillment_statuses = [f['status'] for f in order.get('fulfillments', [])]
        
        # Extract discount codes
        discount_codes = [d['code'] for d in order.get('discount_codes', []) if d.get('code')]
        
        metadata = {
            # Core identifiers
            "order_id": order['id'],
            "order_number": order['order_number'],
            "customer_id": order['customer']['id'],
            "customer_email": order['email'],
            
            # Geographic (flattened from nested address)
            "shipping_province": order['shipping_address']['province'],
            "shipping_country": order['shipping_address']['country'],
            "shipping_city": order['shipping_address']['city'],
            "billing_province": order['billing_address']['province'],
            
            # Extracted from arrays
            "vendors": vendors,
            "skus": skus,
            "product_ids": product_ids,
            "fulfillment_statuses": fulfillment_statuses,
            "discount_codes": discount_codes,
            
            # Numeric fields
            "total_price": float(order['total_price']),
            "currency": order['currency'],
            
            # Temporal
            "created_at": order['created_at'],
            "updated_at": order.get('updated_at'),
            
            # Status flags
            "financial_status": order['financial_status'],
            "fulfillment_status": order['fulfillment_status'],
            "has_discount": len(discount_codes) > 0,
            "item_count": len(order['line_items']),
            
            # Tags
            "tags": order.get('tags', '').split(',') if order.get('tags') else [],
            
            # Reference to full document in source DB (not stored here!)
            "source_db": "postgres",
            "source_table": "orders",
            "source_id": order['id']
        }
        
        return metadata
    
    def ingest_order(self, order: Dict[str, Any]):
        """Ingest single order into Qdrant"""
        
        # Generate content summary
        content = self.generate_order_summary(order)
        
        # Generate embedding
        embedding = self.generate_embedding(content)
        
        # Extract metadata
        metadata = self.extract_metadata(order)
        metadata['content'] = content
        
        # Create point (NO full document - reference architecture!)
        point = PointStruct(
            id=order['id'],
            vector=embedding,
            payload=metadata
        )
        
        # Upsert to Qdrant
        self.client.upsert(
            collection_name=self.collection_name,
            points=[point]
        )
    
    def ingest_batch(self, orders: List[Dict[str, Any]], batch_size: int = 100):
        """Ingest multiple orders in batches for better performance"""
        
        total = len(orders)
        print(f"Ingesting {total} orders in batches of {batch_size}...")
        
        for i in range(0, total, batch_size):
            batch = orders[i:i+batch_size]
            points = []
            
            for order in batch:
                content = self.generate_order_summary(order)
                embedding = self.generate_embedding(content)
                metadata = self.extract_metadata(order)
                metadata['content'] = content
                
                point = PointStruct(
                    id=order['id'],
                    vector=embedding,
                    payload=metadata
                )
                points.append(point)
            
            self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )
            
            print(f"  Ingested {min(i+batch_size, total)}/{total} orders")
        
        print("✓ Batch ingestion complete")
    
    def search_orders(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Search orders with optional metadata filtering"""
        
        # Generate query embedding
        query_embedding = self.generate_embedding(query)
        
        # Build filter if provided
        query_filter = None
        if filters:
            conditions = []
            for key, value in filters.items():
                if isinstance(value, list):
                    conditions.append(
                        FieldCondition(key=key, match=MatchAny(any=value))
                    )
                elif isinstance(value, dict) and 'gte' in value:
                    conditions.append(
                        FieldCondition(key=key, range=Range(gte=value['gte']))
                    )
                elif isinstance(value, dict) and 'lte' in value:
                    conditions.append(
                        FieldCondition(key=key, range=Range(lte=value['lte']))
                    )
                else:
                    conditions.append(
                        FieldCondition(key=key, match=value)
                    )
            
            if conditions:
                query_filter = Filter(must=conditions)
        
        # Search
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            query_filter=query_filter,
            limit=limit,
            with_payload=True
        )
        
        return [
            {
                "order_id": r.payload['order_id'],
                "order_number": r.payload['order_number'],
                "score": r.score,
                "content": r.payload['content'],
                "metadata": {k: v for k, v in r.payload.items() if k != 'content'}
            }
            for r in results
        ]
    
    def get_order_with_full_document(self, order_id: int) -> Dict[str, Any]:
        """
        Get order from Qdrant + fetch full document from source DB.
        
        This demonstrates the reference architecture pattern.
        """
        
        # Step 1: Get order from Qdrant (metadata only)
        result = self.client.retrieve(
            collection_name=self.collection_name,
            ids=[order_id]
        )
        
        if not result:
            return None
        
        order_metadata = result[0].payload
        
        # Step 2: Fetch full document from source DB
        conn = psycopg2.connect(POSTGRES_DSN)
        cur = conn.cursor()
        cur.execute("SELECT data FROM orders WHERE id = %s", (order_id,))
        row = cur.fetchone()
        conn.close()
        
        if row:
            full_document = json.loads(row[0])
        else:
            full_document = None
        
        return {
            "metadata": order_metadata,
            "full_document": full_document
        }


def load_sample_orders() -> List[Dict[str, Any]]:
    """Load sample Shopify orders for testing"""
    
    # In production, load from your actual Shopify API or database
    sample_order = {
        "id": 450789469,
        "order_number": 1001,
        "email": "bob.norman@mail.example.com",
        "created_at": "2008-01-10T11:00:00-05:00",
        "updated_at": "2012-08-24T14:02:15-04:00",
        "total_price": "409.94",
        "currency": "USD",
        "financial_status": "authorized",
        "fulfillment_status": "partial",
        "tags": "imported,vip",
        "customer": {
            "id": 207119551,
            "first_name": "Bob",
            "last_name": "Norman",
            "email": "bob.norman@mail.example.com"
        },
        "shipping_address": {
            "city": "Ottawa",
            "province": "Ontario",
            "country": "Canada"
        },
        "billing_address": {
            "city": "Drayton Valley",
            "province": "Alberta",
            "country": "Canada"
        },
        "line_items": [
            {
                "id": 669751112,
                "product_id": 7513594,
                "variant_id": 4264112,
                "title": "IPod Nano",
                "variant_title": "Pink",
                "sku": "IPOD-342-N",
                "vendor": "Apple",
                "price": "199.99",
                "quantity": 1
            }
        ],
        "fulfillments": [
            {
                "id": 255858046,
                "status": "failure",
                "tracking_company": "USPS"
            }
        ],
        "discount_codes": [
            {"code": "SPRING30"}
        ]
    }
    
    return [sample_order]


def main():
    """Main entry point"""
    
    import argparse
    
    parser = argparse.ArgumentParser(description="Shopify to Qdrant pipeline")
    parser.add_argument("--mode", choices=["setup", "ingest", "query"], required=True)
    parser.add_argument("--query", type=str, help="Search query")
    parser.add_argument("--vendor", type=str, help="Filter by vendor")
    parser.add_argument("--province", type=str, help="Filter by province")
    
    args = parser.parse_args()
    
    pipeline = ShopifyVectorPipeline()
    
    if args.mode == "setup":
        pipeline.setup_collection()
    
    elif args.mode == "ingest":
        orders = load_sample_orders()
        pipeline.ingest_batch(orders)
    
    elif args.mode == "query":
        if not args.query:
            print("Error: --query required for query mode")
            return
        
        # Build filters
        filters = {}
        if args.vendor:
            filters['vendors'] = [args.vendor]
        if args.province:
            filters['shipping_province'] = args.province
        
        # Search
        results = pipeline.search_orders(args.query, filters=filters)
        
        print(f"\nFound {len(results)} results:\n")
        for i, result in enumerate(results, 1):
            print(f"{i}. Order #{result['order_number']} (score: {result['score']:.3f})")
            print(f"   {result['content'][:200]}...")
            print()


if __name__ == "__main__":
    main()
