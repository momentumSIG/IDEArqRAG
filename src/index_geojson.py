#!/usr/bin/env python3
"""
Index GeoJSON files into existing Weaviate collections.
Run AFTER the PDF indexing notebook completes.

Usage:
    cd /home/raglinux/RAG
    python src/index_geojson.py
"""
import os
import sys
import json
import gc
import time
import logging
from pathlib import Path
from typing import List, Dict, Any

from dotenv import load_dotenv
load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import EMBEDDINGS, collection_name, GEOJSON_DIR, INDEX_PROPERTIES_FULL

import torch
import weaviate
from langchain_weaviate import WeaviateVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer


# E5 Custom Embedding para evitar errores de memoria
class E5InstructEmbeddings(Embeddings):
    """Custom embedding class for E5 Instruct models.
    
    E5 requires special prefixes:
    - "passage: " for documents during indexing
    - "query: " for queries during retrieval
    """
    
    def __init__(self, model_name="intfloat/multilingual-e5-large-instruct", device=None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model = SentenceTransformer(model_name, device=self.device)
    
    def embed_documents(self, texts):
        prefixed = [f"passage: {t}" for t in texts]
        return self.model.encode(prefixed, device=self.device).tolist()
    
    def embed_query(self, text):
        prefixed = f"query: {text}"
        return self.model.encode([prefixed], device=self.device)[0].tolist()


# Helper Functions para evitar errores de memoria
def safe_empty_cache():
    """Clean GPU memory to prevent OOM errors."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_geojson(geojson_path: str) -> List[Document]:
    """Load c14_v2.geojson as Documents.
    
    Args:
        geojson_path: Path to the GeoJSON file (can be file or directory)
    
    Returns:
        List of Document objects with yacimiento data
    """
    # Handle both file path and directory path
    path = Path(geojson_path)
    if path.is_dir():
        geo_path = path / "c14_v2.geojson"
    else:
        geo_path = path
    
    if not geo_path.exists():
        print(f"ERROR: {geo_path} not found")
        return []

    print(f"Loading {geo_path.name}...")
    all_docs = []
    try:
        with open(geo_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        features = data.get('features', [])
        for idx, feat in enumerate(features):
            geom = feat.get('geometry') or {}
            if geom.get('type') != 'Point':
                continue
            coords = geom.get('coordinates') or []
            if len(coords) < 2:
                continue

            props = feat.get('properties') or {}
            lon, lat = float(coords[0]), float(coords[1])

            text_parts = []
            nombre = props.get('yacimiento', props.get('name', 'Desconocido'))
            text_parts.append(f"Yacimiento: {nombre}")
            if props.get('unidad_territorial'):
                text_parts.append(f"Ubicacion: {props['unidad_territorial']}")
            if props.get('tipologia_crono'):
                text_parts.append(f"Tipologia: {props['tipologia_crono']}")
            if props.get('descripcion'):
                desc = str(props['descripcion'])[:500]
                text_parts.append(f"Descripcion: {desc}")
            if props.get('dataciones_c_14'):
                text_parts.append(f"Dataciones C14: {props['dataciones_c_14']}")

            doc = Document(
                page_content="\n".join(text_parts),
                metadata={
                    'source': str(geo_path),
                    'filename': geo_path.name,
                    'doc_type': 'yacimiento',
                    'doc_index': idx,
                    'lat': lat,
                    'lon': lon,
                    'yacimiento_id': props.get('yacimiento_id', props.get('id')),
                    'yacimiento_nombre': nombre,
                    'unidad_territorial': props.get('unidad_territorial', ''),
                    'tipologia_crono': props.get('tipologia_crono', ''),
                    'title': nombre,
                    'language': 'es',
                    'chunking_method': 'geojson_1doc_1feature',
                }
            )
            all_docs.append(doc)
    except Exception as e:
        print(f"  Error loading {geo_path.name}: {e}")

    print(f"Loaded {len(all_docs)} yacimiento documents from {geo_path.name}")
    return all_docs


def load_embedding_model(emb_cfg: Dict[str, Any]):
    """Load the appropriate embedding model based on configuration.
    
    Handles:
    - E5InstructEmbeddings for E5 models (custom class with prefixes)
    - HuggingFaceEmbeddings with trust_remote_code for GTE
    - Standard HuggingFaceEmbeddings for other models
    - CUDA -> CPU fallback on OOM
    
    Returns:
        Tuple of (embedding_model, device_used)
    """
    model_name = emb_cfg["model_name"]
    model_class = emb_cfg.get("model_class", "HuggingFaceEmbeddings")
    
    # E5 requires custom class with passage/query prefixes
    if model_class == "E5InstructEmbeddings":
        try:
            emb = E5InstructEmbeddings(model_name=model_name, device="cuda")
            return emb, "cuda"
        except Exception as e:
            print(f"  [E5 CUDA failed: {e}] Falling back to CPU...")
            emb = E5InstructEmbeddings(model_name=model_name, device="cpu")
            return emb, "cpu"
    
    # Standard HuggingFaceEmbeddings
    model_kwargs = {"device": "cuda"}
    encode_kwargs = {"device": "cuda"}
    
    # GTE requires trust_remote_code=True
    if emb_cfg.get("trust_remote_code", False):
        model_kwargs["trust_remote_code"] = True
    
    try:
        emb = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs,
        )
        return emb, "cuda"
    except Exception as e:
        print(f"  [CUDA failed: {e}] Falling back to CPU...")
        model_kwargs["device"] = "cpu"
        encode_kwargs["device"] = "cpu"
        emb = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs,
        )
        return emb, "cpu"


def index_geojson_to_collection(emb_key: str, emb_cfg: Dict[str, Any], 
                                 geojson_docs: List[Document], w_client) -> Dict[str, Any]:
    """Index GeoJSON docs into an existing collection (created by PDF indexing).
    
    Features:
    - Conditional embedding model loading (E5, GTE, standard)
    - Batch size 5 to avoid OOM
    - Error counter and retry mechanism
    - Full schema attributes from INDEX_PROPERTIES_FULL
    - Detailed reporting
    
    Returns:
        Dictionary with indexing statistics
    """
    coll_name = collection_name(emb_key)
    print(f"\n{'='*60}")
    print(f"Embedding: {emb_key} -> {coll_name}")
    print(f"GeoJSON docs to add: {len(geojson_docs)}")
    print(f"{'='*60}")

    if not w_client.collections.exists(coll_name):
        print(f"  ERROR: Collection '{coll_name}' does not exist.")
        print(f"  Run the PDF indexing notebook first.")
        return {"indexed": 0, "errors": len(geojson_docs), "object_count": 0}

    # Load embedding model with appropriate class
    emb, device = load_embedding_model(emb_cfg)
    print(f"  Model loaded on {device}: {emb_cfg['model_name']}")

    # Create vector store with FULL schema attributes
    vs = WeaviateVectorStore(
        client=w_client,
        index_name=coll_name,
        text_key="content",
        embedding=emb,
        attributes=[p[0] for p in INDEX_PROPERTIES_FULL if p[0] != "content"],
    )

    # Add docs in batches with retry mechanism
    batch_size = 5
    indexed = 0
    errors = 0
    t0 = time.time()

    for i in range(0, len(geojson_docs), batch_size):
        batch = geojson_docs[i:i + batch_size]
        try:
            vs.add_documents(batch)
            indexed += len(batch)
            
            if indexed % 50 == 0:
                elapsed = time.time() - t0
                rate = indexed / elapsed if elapsed > 0 else 0
                print(f"  [{indexed}/{len(geojson_docs)}] {rate:.1f} docs/s")
                
        except Exception as e:
            errors += len(batch)
            print(f"  Error at batch {i}: {e}")
            safe_empty_cache()
            
            # Retry one by one
            print(f"  Retrying {len(batch)} docs individually...")
            for doc in batch:
                try:
                    vs.add_documents([doc])
                    indexed += 1
                    errors -= 1  # One succeeded
                except Exception as e2:
                    print(f"    Failed: {e2}")

    elapsed = time.time() - t0

    # Verify total objects in collection
    coll = w_client.collections.get(coll_name)
    agg = coll.aggregate.over_all(total_count=True)
    obj_count = agg.total_count or 0

    print(f"\n  Results:")
    print(f"    Indexed: {indexed}")
    print(f"    Errors: {errors}")
    print(f"    Time: {elapsed:.1f}s ({indexed/elapsed:.1f} docs/s)")
    print(f"    Total objects in collection: {obj_count}")
    
    return {
        "indexed": indexed,
        "errors": errors,
        "object_count": obj_count,
        "time_s": elapsed,
        "device": device,
    }


def main():
    print("\n" + "="*60)
    print("RAG-IDEArq â GeoJSON Indexer (Improved)")
    print("="*60)
    print(f"Batch size: 5 (reduced to avoid OOM)")
    print(f"Retry mechanism: enabled")
    print(f"Full schema: {len(INDEX_PROPERTIES_FULL)} properties")
    print("="*60)

    # Connect to Weaviate
    w_client = weaviate.connect_to_local(
        host="localhost",
        port=8080,
        grpc_port=50051,
    )
    print(f"\nWeaviate: {w_client.is_ready()}")
    print(f"Existing collections: {list(w_client.collections.list_all().keys())}")

    # Load GeoJSON
    geojson_docs = load_geojson(GEOJSON_DIR)
    if not geojson_docs:
        print("ERROR: No GeoJSON docs loaded.")
        w_client.close()
        return

    # Index to each existing embedding collection
    results = []
    for emb_key, emb_cfg in EMBEDDINGS.items():
        result = index_geojson_to_collection(emb_key, emb_cfg, geojson_docs, w_client)
        results.append((emb_key, result))
        safe_empty_cache()

    # Final summary
    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)
    for emb_key, result in results:
        coll_name = collection_name(emb_key)
        status = "â OK" if result["errors"] == 0 else "â ï¸  WARN"
        print(f"{status} {emb_key:30s} â {coll_name}")
        print(f"     Indexed: {result['indexed']}, Errors: {result['errors']}, "
              f"Total: {result['object_count']}, Time: {result['time_s']:.1f}s")

    w_client.close()
    print("\nTerminado")


if __name__ == "__main__":
    main()