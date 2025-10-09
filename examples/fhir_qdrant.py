"""
FHIR Healthcare Records to Qdrant Vector DB
============================================

Implementation of hybrid architecture for FHIR clinical data:
- Patients: Hierarchical (with full FHIR resources)
- Observations: Pure flattening (metadata only, no full documents)

Architecture:
- Patients: 100K × 64 KB = 6.4 GB (full FHIR resources)
- Observations: 20M × 4 KB = 80 GB (metadata only, reference architecture)
- Total: ~86 GB (vs 400+ GB if storing all full documents)

Requirements:
    pip install qdrant-client openai fhir.resources python-dotenv

Usage:
    python fhir_qdrant.py --mode setup
    python fhir_qdrant.py --mode ingest --resource-type Patient --file patients.json
    python fhir_qdrant.py --mode ingest --resource-type Observation --file observations.json
    python fhir_qdrant.py --mode query --query "high glucose diabetic patients"
"""

import os
import json
import uuid
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct,
    Filter, FieldCondition, MatchAny, Range,
    PayloadSchemaType
)
import openai
from dotenv import load_dotenv

load_dotenv()

# Configuration
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

openai.api_key = OPENAI_API_KEY


class FHIRVectorPipeline:
    """
    Pipeline for ingesting FHIR resources into Qdrant.
    
    Uses hybrid architecture:
    - Patients: Hierarchical (full FHIR resource stored)
    - Observations: Flattened (metadata only, no full resource)
    """
    
    def __init__(self, qdrant_url: str = QDRANT_URL):
        self.client = QdrantClient(url=qdrant_url)
        self.embedding_model = "text-embedding-3-large"  # Better for medical terminology
        self.embedding_dim = 3072
        
        # Collection names
        self.patients_collection = "fhir_patients"
        self.observations_collection = "fhir_observations"
        self.encounters_collection = "fhir_encounters"
    
    def setup_collections(self):
        """Create Qdrant collections for FHIR resources"""
        
        print("Setting up FHIR collections...")
        
        # === COLLECTION 1: Patients (Hierarchical with documents) ===
        print(f"\n1. Creating '{self.patients_collection}' collection...")
        
        if not self._collection_exists(self.patients_collection):
            self.client.create_collection(
                collection_name=self.patients_collection,
                vectors_config=VectorParams(
                    size=self.embedding_dim,
                    distance=Distance.COSINE
                ),
                quantization_config={"scalar": {"type": "int8"}}
            )
            
            # Patient indexes
            patient_indexes = {
                "patient_id": PayloadSchemaType.KEYWORD,
                "mrn": PayloadSchemaType.KEYWORD,  # Medical Record Number
                "age": PayloadSchemaType.INTEGER,
                "gender": PayloadSchemaType.KEYWORD,
                "state": PayloadSchemaType.KEYWORD,
                "city": PayloadSchemaType.KEYWORD,
            }
            
            for field, schema in patient_indexes.items():
                self.client.create_payload_index(
                    collection_name=self.patients_collection,
                    field_name=field,
                    field_schema=schema
                )
            
            print("  ✓ Patient collection created")
        else:
            print("  ℹ Patient collection already exists")
        
        # === COLLECTION 2: Observations (Flattened without documents) ===
        print(f"\n2. Creating '{self.observations_collection}' collection...")
        
        if not self._collection_exists(self.observations_collection):
            self.client.create_collection(
                collection_name=self.observations_collection,
                vectors_config=VectorParams(
                    size=self.embedding_dim,
                    distance=Distance.COSINE
                ),
                quantization_config={"scalar": {"type": "int8"}},
                shard_number=4  # Shard for 20M observations
            )
            
            # Observation indexes (critical for clinical queries!)
            observation_indexes = {
                "observation_id": PayloadSchemaType.KEYWORD,
                "patient_id": PayloadSchemaType.KEYWORD,
                "encounter_id": PayloadSchemaType.KEYWORD,
                "observation_code": PayloadSchemaType.KEYWORD,  # LOINC code
                "category": PayloadSchemaType.KEYWORD,
                "value": PayloadSchemaType.FLOAT,  # Critical for lab value filtering!
                "unit": PayloadSchemaType.KEYWORD,
                "interpretation": PayloadSchemaType.KEYWORD,
                "effective_date": PayloadSchemaType.DATETIME,
                "is_abnormal": PayloadSchemaType.KEYWORD,
            }
            
            for field, schema in observation_indexes.items():
                self.client.create_payload_index(
                    collection_name=self.observations_collection,
                    field_name=field,
                    field_schema=schema
                )
            
            print("  ✓ Observation collection created with 4 shards")
        else:
            print("  ℹ Observation collection already exists")
        
        print("\n✓ All collections setup complete")
    
    def _collection_exists(self, collection_name: str) -> bool:
        """Check if collection exists"""
        collections = self.client.get_collections().collections
        return any(c.name == collection_name for c in collections)
    
    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding using OpenAI"""
        response = openai.embeddings.create(
            model=self.embedding_model,
            input=text
        )
        return response.data[0].embedding
    
    def calculate_age(self, birth_date: str) -> int:
        """Calculate age from birth date"""
        birth = datetime.fromisoformat(birth_date.replace('Z', '+00:00'))
        today = datetime.now()
        return today.year - birth.year - ((today.month, today.day) < (birth.month, birth.day))
    
    def resolve_fhir_reference(self, reference: str) -> str:
        """
        Resolve FHIR reference to ID.
        Example: "Patient/patient-67890" -> "patient-67890"
        """
        if '/' in reference:
            return reference.split('/')[-1]
        return reference
    
    # === PATIENT INGESTION (Hierarchical with full document) ===
    
    def ingest_patient(self, patient_resource: Dict[str, Any]):
        """
        Ingest FHIR Patient resource.
        Stores FULL FHIR resource (hierarchical approach).
        """
        
        patient_id = patient_resource['id']
        
        # Extract name
        name_obj = patient_resource.get('name', [{}])[0]
        given_names = name_obj.get('given', [])
        family_name = name_obj.get('family', '')
        full_name = f"{' '.join(given_names)} {family_name}".strip()
        
        # Extract MRN (Medical Record Number)
        mrn = None
        for identifier in patient_resource.get('identifier', []):
            if 'mrn' in identifier.get('system', '').lower():
                mrn = identifier.get('value')
                break
        
        # Extract address
        address = patient_resource.get('address', [{}])[0]
        state = address.get('state', '')
        city = address.get('city', '')
        
        # Calculate age
        birth_date = patient_resource.get('birthDate', '')
        age = self.calculate_age(birth_date) if birth_date else None
        
        # Generate content summary
        content = (
            f"Patient {full_name} "
            f"(MRN: {mrn or 'N/A'}), "
            f"{patient_resource.get('gender', 'unknown')} gender, "
            f"born {birth_date} (age {age}). "
            f"Lives in {city}, {state}."
        )
        
        # Generate embedding
        embedding = self.generate_embedding(content)
        
        # Create point (WITH full FHIR document for patients)
        point = PointStruct(
            id=str(uuid.uuid4()),  # Random UUID
            vector=embedding,
            payload={
                "resource_type": "Patient",
                "patient_id": patient_id,
                "mrn": mrn,
                "full_name": full_name,
                "gender": patient_resource.get('gender'),
                "birth_date": birth_date,
                "age": age,
                "state": state,
                "city": city,
                "content": content,
                
                # Store FULL FHIR resource (only ~100K patients, manageable)
                "fhir_resource": patient_resource
            }
        )
        
        self.client.upsert(
            collection_name=self.patients_collection,
            points=[point]
        )
    
    # === OBSERVATION INGESTION (Flattened without full document) ===
    
    def ingest_observation(self, observation_resource: Dict[str, Any]):
        """
        Ingest FHIR Observation resource.
        Does NOT store full FHIR resource (flattened approach for scale).
        """
        
        observation_id = observation_resource['id']
        
        # Resolve references
        patient_ref = observation_resource.get('subject', {}).get('reference', '')
        patient_id = self.resolve_fhir_reference(patient_ref)
        
        encounter_ref = observation_resource.get('encounter', {}).get('reference', '')
        encounter_id = self.resolve_fhir_reference(encounter_ref) if encounter_ref else None
        
        # Extract observation code (LOINC)
        code_obj = observation_resource.get('code', {})
        coding = code_obj.get('coding', [{}])[0]
        obs_code = coding.get('code', '')
        obs_display = coding.get('display', '')
        
        # Extract category
        category_obj = observation_resource.get('category', [{}])[0]
        category_coding = category_obj.get('coding', [{}])[0]
        category = category_coding.get('code', 'unknown')
        
        # Extract value
        value_quantity = observation_resource.get('valueQuantity', {})
        value = value_quantity.get('value')
        unit = value_quantity.get('unit', '')
        
        # Extract interpretation (High/Low/Normal)
        interpretation_obj = observation_resource.get('interpretation', [{}])[0]
        interpretation_coding = interpretation_obj.get('coding', [{}])[0]
        interpretation = interpretation_coding.get('code', 'N')
        
        is_abnormal = interpretation in ['H', 'L', 'A', 'AA', 'HH', 'LL']
        
        # Effective date
        effective_date = observation_resource.get('effectiveDateTime', '')
        
        # Generate content summary
        content = (
            f"{obs_display}: {value} {unit} "
            f"({'High' if interpretation == 'H' else 'Low' if interpretation == 'L' else 'Normal'}) "
            f"for patient {patient_id} on {effective_date[:10]}"
        )
        
        # Add reference range if available
        ref_range = observation_resource.get('referenceRange', [{}])[0]
        if ref_range:
            ref_low = ref_range.get('low', {}).get('value')
            ref_high = ref_range.get('high', {}).get('value')
            if ref_low and ref_high:
                content += f". Reference range: {ref_low}-{ref_high} {unit}"
        
        # Generate embedding
        embedding = self.generate_embedding(content)
        
        # Create point (WITHOUT full FHIR resource - saves massive space!)
        point = PointStruct(
            id=str(uuid.uuid4()),
            vector=embedding,
            payload={
                "resource_type": "Observation",
                "observation_id": observation_id,
                "patient_id": patient_id,
                "encounter_id": encounter_id,
                "observation_code": obs_code,
                "observation_display": obs_display,
                "category": category,
                "value": float(value) if value is not None else None,
                "unit": unit,
                "interpretation": interpretation,
                "is_abnormal": is_abnormal,
                "effective_date": effective_date,
                "content": content,
                
                # NO full FHIR resource stored (reference architecture)
                # For 20M observations, this saves ~300 GB!
            }
        )
        
        self.client.upsert(
            collection_name=self.observations_collection,
            points=[point]
        )
    
    # === QUERY METHODS ===
    
    def find_patients_with_lab_values(
        self,
        observation_code: str,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Clinical query: Find patients with specific lab values.
        
        Example: Find diabetic patients (glucose > 126 mg/dL)
        """
        
        # Build filter
        conditions = [
            FieldCondition(key="observation_code", match=observation_code)
        ]
        
        if min_value is not None:
            conditions.append(
                FieldCondition(key="value", range=Range(gte=min_value))
            )
        
        if max_value is not None:
            conditions.append(
                FieldCondition(key="value", range=Range(lte=max_value))
            )
        
        # Query observations
        observations = self.client.scroll(
            collection_name=self.observations_collection,
            scroll_filter=Filter(must=conditions),
            limit=10000,
            with_payload=True,
            with_vectors=False  # Don't need vectors for this query
        )[0]
        
        # Get unique patient IDs
        patient_ids = list(set([obs.payload['patient_id'] for obs in observations]))
        
        # Fetch patient records
        patients = self.client.scroll(
            collection_name=self.patients_collection,
            scroll_filter=Filter(
                must=[
                    FieldCondition(key="patient_id", match=MatchAny(any=patient_ids))
                ]
            ),
            limit=len(patient_ids),
            with_payload=True
        )[0]
        
        return [
            {
                "patient": p.payload,
                "matching_observations": [
                    obs.payload for obs in observations 
                    if obs.payload['patient_id'] == p.payload['patient_id']
                ]
            }
            for p in patients
        ]
    
    def search_similar_cases(
        self,
        query: str,
        patient_age_range: Optional[tuple] = None,
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Clinical decision support: Find similar patient cases.
        
        Example: "chest pain elevated blood pressure high glucose"
        """
        
        # Generate query embedding
        query_embedding = self.generate_embedding(query)
        
        # Build filter
        conditions = []
        if patient_age_range:
            min_age, max_age = patient_age_range
            conditions.append(
                FieldCondition(key="age", range=Range(gte=min_age, lte=max_age))
            )
        
        query_filter = Filter(must=conditions) if conditions else None
        
        # Search observations (semantic search)
        results = self.client.search(
            collection_name=self.observations_collection,
            query_vector=query_embedding,
            query_filter=query_filter,
            limit=limit,
            with_payload=True
        )
        
        return [
            {
                "observation": r.payload,
                "score": r.score,
                "patient_id": r.payload['patient_id']
            }
            for r in results
        ]
    
    def get_patient_timeline(self, patient_id: str) -> Dict[str, Any]:
        """
        Get complete patient timeline (all observations).
        """
        
        # Get patient record
        patient = self.client.scroll(
            collection_name=self.patients_collection,
            scroll_filter=Filter(
                must=[FieldCondition(key="patient_id", match=patient_id)]
            ),
            limit=1
        )[0]
        
        if not patient:
            return None
        
        # Get all observations for this patient
        observations = self.client.scroll(
            collection_name=self.observations_collection,
            scroll_filter=Filter(
                must=[FieldCondition(key="patient_id", match=patient_id)]
            ),
            limit=10000,
            with_payload=True,
            with_vectors=False
        )[0]
        
        # Sort by date
        sorted_obs = sorted(
            observations,
            key=lambda x: x.payload.get('effective_date', ''),
            reverse=True
        )
        
        return {
            "patient": patient[0].payload,
            "observation_count": len(sorted_obs),
            "observations": [obs.payload for obs in sorted_obs]
        }
    
    def identify_cohort(
        self,
        observation_code: str,
        min_value: float,
        time_window_days: int = 365
    ) -> Dict[str, Any]:
        """
        Research query: Identify patient cohort for clinical trial.
        
        Example: HbA1c > 6.5% in past year (diabetic cohort)
        """
        
        # Calculate date range
        start_date = (datetime.now() - timedelta(days=time_window_days)).isoformat()
        
        # Query observations
        observations = self.client.scroll(
            collection_name=self.observations_collection,
            scroll_filter=Filter(
                must=[
                    FieldCondition(key="observation_code", match=observation_code),
                    FieldCondition(key="value", range=Range(gte=min_value)),
                    FieldCondition(key="effective_date", range=Range(gte=start_date))
                ]
            ),
            limit=50000,
            with_payload=["patient_id", "value", "effective_date"],
            with_vectors=False
        )[0]
        
        # Get unique patients
        patient_ids = list(set([obs.payload['patient_id'] for obs in observations]))
        
        # Get demographics
        patients = self.client.scroll(
            collection_name=self.patients_collection,
            scroll_filter=Filter(
                must=[
                    FieldCondition(key="patient_id", match=MatchAny(any=patient_ids))
                ]
            ),
            limit=len(patient_ids),
            with_payload=["patient_id", "age", "gender", "state"]
        )[0]
        
        # Calculate cohort statistics
        ages = [p.payload.get('age', 0) for p in patients if p.payload.get('age')]
        
        cohort_stats = {
            "total_patients": len(patient_ids),
            "observation_code": observation_code,
            "min_value": min_value,
            "time_window_days": time_window_days,
            "demographics": {
                "age_mean": sum(ages) / len(ages) if ages else 0,
                "age_min": min(ages) if ages else 0,
                "age_max": max(ages) if ages else 0,
                "gender_distribution": self._count_by_field(patients, 'gender'),
                "state_distribution": self._count_by_field(patients, 'state')
            },
            "patient_ids": patient_ids
        }
        
        return cohort_stats
    
    def _count_by_field(self, points, field: str) -> Dict[str, int]:
        """Helper to count occurrences by field"""
        from collections import Counter
        values = [p.payload.get(field) for p in points if p.payload.get(field)]
        return dict(Counter(values))


def load_sample_fhir_data() -> Dict[str, List[Dict[str, Any]]]:
    """Load sample FHIR resources for testing"""
    
    patient = {
        "resourceType": "Patient",
        "id": "patient-67890",
        "identifier": [
            {
                "system": "http://hospital.org/mrn",
                "value": "MRN-123456"
            }
        ],
        "name": [
            {
                "family": "Smith",
                "given": ["John", "Robert"]
            }
        ],
        "gender": "male",
        "birthDate": "1965-03-15",
        "address": [
            {
                "city": "Springfield",
                "state": "IL",
                "country": "USA"
            }
        ]
    }
    
    observation_glucose = {
        "resourceType": "Observation",
        "id": "obs-glucose-001",
        "status": "final",
        "category": [
            {
                "coding": [
                    {
                        "system": "http://terminology.hl7.org/CodeSystem/observation-category",
                        "code": "laboratory",
                        "display": "Laboratory"
                    }
                ]
            }
        ],
        "code": {
            "coding": [
                {
                    "system": "http://loinc.org",
                    "code": "2339-0",
                    "display": "Glucose [Mass/volume] in Blood"
                }
            ]
        },
        "subject": {
            "reference": "Patient/patient-67890"
        },
        "effectiveDateTime": "2024-03-10T15:00:00Z",
        "valueQuantity": {
            "value": 145,
            "unit": "mg/dL",
            "system": "http://unitsofmeasure.org",
            "code": "mg/dL"
        },
        "interpretation": [
            {
                "coding": [
                    {
                        "system": "http://terminology.hl7.org/CodeSystem/v3-ObservationInterpretation",
                        "code": "H",
                        "display": "High"
                    }
                ]
            }
        ],
        "referenceRange": [
            {
                "low": {"value": 70, "unit": "mg/dL"},
                "high": {"value": 100, "unit": "mg/dL"},
                "text": "Normal fasting glucose"
            }
        ]
    }
    
    observation_bp = {
        "resourceType": "Observation",
        "id": "obs-bp-001",
        "status": "final",
        "category": [
            {
                "coding": [
                    {
                        "system": "http://terminology.hl7.org/CodeSystem/observation-category",
                        "code": "vital-signs",
                        "display": "Vital Signs"
                    }
                ]
            }
        ],
        "code": {
            "coding": [
                {
                    "system": "http://loinc.org",
                    "code": "85354-9",
                    "display": "Blood pressure panel"
                }
            ]
        },
        "subject": {
            "reference": "Patient/patient-67890"
        },
        "effectiveDateTime": "2024-03-10T14:35:00Z",
        "valueQuantity": {
            "value": 165,
            "unit": "mmHg"
        },
        "interpretation": [
            {
                "coding": [
                    {
                        "code": "H",
                        "display": "High"
                    }
                ]
            }
        ]
    }
    
    return {
        "patients": [patient],
        "observations": [observation_glucose, observation_bp]
    }


def main():
    """Main entry point"""
    
    import argparse
    
    parser = argparse.ArgumentParser(description="FHIR to Qdrant pipeline")
    parser.add_argument("--mode", choices=["setup", "ingest", "query"], required=True)
    parser.add_argument("--resource-type", choices=["Patient", "Observation"])
    parser.add_argument("--file", type=str, help="JSON file with FHIR resources")
    parser.add_argument("--query", type=str, help="Search query")
    parser.add_argument("--obs-code", type=str, help="LOINC observation code")
    parser.add_argument("--min-value", type=float, help="Minimum observation value")
    
    args = parser.parse_args()
    
    pipeline = FHIRVectorPipeline()
    
    if args.mode == "setup":
        pipeline.setup_collections()
    
    elif args.mode == "ingest":
        if not args.resource_type:
            print("Error: --resource-type required for ingest mode")
            return
        
        # Load sample data or from file
        if args.file:
            with open(args.file, 'r') as f:
                resources = json.load(f)
        else:
            print("Using sample data...")
            sample_data = load_sample_fhir_data()
            resources = sample_data.get(args.resource_type.lower() + 's', [])
        
        # Ingest based on resource type
        if args.resource_type == "Patient":
            for patient in resources:
                pipeline.ingest_patient(patient)
                print(f"  ✓ Ingested patient {patient['id']}")
        
        elif args.resource_type == "Observation":
            for observation in resources:
                pipeline.ingest_observation(observation)
                print(f"  ✓ Ingested observation {observation['id']}")
        
        print(f"\n✓ Ingested {len(resources)} {args.resource_type} resources")
    
    elif args.mode == "query":
        if args.obs_code and args.min_value:
            # Lab value query
            print(f"\nFinding patients with {args.obs_code} >= {args.min_value}...")
            results = pipeline.find_patients_with_lab_values(
                observation_code=args.obs_code,
                min_value=args.min_value
            )
            
            print(f"\nFound {len(results)} patients:\n")
            for i, result in enumerate(results[:10], 1):
                patient = result['patient']
                obs_count = len(result['matching_observations'])
                print(f"{i}. Patient {patient.get('full_name')} (MRN: {patient.get('mrn')})")
                print(f"   Age: {patient.get('age')}, Gender: {patient.get('gender')}")
                print(f"   Matching observations: {obs_count}")
                print()
        
        elif args.query:
            # Semantic search
            print(f"\nSearching for similar cases: '{args.query}'...")
            results = pipeline.search_similar_cases(args.query)
            
            print(f"\nFound {len(results)} similar cases:\n")
            for i, result in enumerate(results[:10], 1):
                obs = result['observation']
                print(f"{i}. Score: {result['score']:.3f}")
                print(f"   {obs.get('content')}")
                print()
        
        else:
            print("Error: Provide either --obs-code + --min-value OR --query")


if __name__ == "__main__":
    main()
