#!/usr/bin/env python3
"""
Reindex GTE and E5 embeddings only.
Self-contained script - no dependencies on notebook cells.
"""
import os
import sys
import re
import time
import gc
from pathlib import Path
from typing import List, Dict, Any, Optional

# Setup paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env", override=True)

from src.config import (
    CHUNK_SIZE, CHUNK_OVERLAP, SEPARATORS,
    MIN_CHUNK_LENGTH, MAX_CHUNK_LENGTH, MAX_DIGIT_RATIO,
    WEAVIATE_URL, EMBEDDINGS, collection_name, INDEX_PROPERTIES_FULL,
    INGESTA_DIR,
)

import torch
import weaviate
from weaviate.classes.config import Configure, DataType, Property
from langchain_weaviate import WeaviateVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document
from langdetect import detect
from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer


# ── E5 Custom Embedding ──────────────────────────────────────────────────────
class E5InstructEmbeddings(Embeddings):
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


# ── Helper Functions ─────────────────────────────────────────────────────────
def safe_empty_cache():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def extract_year(text: str) -> Optional[int]:
    match = re.search(r'\b(19|20)\d{2}\b', text[:3000])
    return int(match.group()) if match else None

def extract_doi(text: str) -> Optional[str]:
    match = re.search(r'10\.\d{4,9}/[-._;()/:A-Z0-9]+', text[:3000], re.IGNORECASE)
    return match.group() if match else None

def detect_language(text: str) -> str:
    try:
        return detect(text[:1000])
    except Exception:
        return "unknown"

PERIODOS = {
    "paleolitico": ["paleolítico", "paleolithic", "upper paleolithic"],
    "mesolitico": ["mesolítico", "mesolithic"],
    "neolitico": ["neolítico", "neolithic", "neolitización"],
    "calcolitico": ["calcolítico", "chalcolithic", "cobre", "edad del cobre"],
    "bronce": ["bronce", "bronze age", "bronce final"],
    "hierro": ["hierro", "iron age", "edad del hierro"],
}

REGIONES = [
    "andalucia", "andalucía", "extremadura", "castilla", "león", "leon",
    "portugal", "catalunya", "cataluña", "aragon", "aragón", "galicia",
    "asturias", "cantabria", "valencia", "murcia", "baleares", "balears",
    "navarra", "euskadi", "rioja", "mancha",
]

def extract_periodo(text: str) -> Optional[str]:
    lower = text[:3000].lower()
    for periodo, kws in PERIODOS.items():
        if any(k in lower for k in kws):
            return periodo
    return None

def extract_region(text: str) -> Optional[str]:
    lower = text[:3000].lower()
    for r in REGIONES:
        if r in lower:
            return r
    return None

def extract_authors_heuristic(text: str) -> str:
    lines = text[:2000].split('\n')
    for line in lines[:10]:
        line = line.strip()
        if 10 < len(line) < 100 and not line.startswith(('http', 'doi', '10.')):
            if re.match(r'^[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+(\s+[,y&]\s+[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+)+', line):
                return line
    return ""

def extract_metadata(pdf_path: Path, page_text: str, fitz_doc) -> Dict[str, Any]:
    meta = fitz_doc.metadata if hasattr(fitz_doc, 'metadata') else {}
    doc_title = meta.get('title', '') or pdf_path.stem

    return {
        'title': doc_title,
        'year': extract_year(page_text),
        'doi': extract_doi(page_text),
        'authors': extract_authors_heuristic(page_text),
        'language': detect_language(page_text),
        'periodo': extract_periodo(page_text),
        'region': extract_region(page_text),
    }


# ── PDF Loading ──────────────────────────────────────────────────────────────
def load_pdfs(ingesta_dir: Path) -> List[Document]:
    pdf_files = sorted(ingesta_dir.glob("*.pdf"))
    print(f"Found {len(pdf_files)} PDFs in {ingesta_dir}")

    all_docs = []
    failed = []

    for i, pdf_file in enumerate(pdf_files):
        try:
            loader = PyMuPDFLoader(str(pdf_file))
            fitz_docs = loader.load()

            combined_content = "\n\n".join([d.page_content for d in fitz_docs])
            meta = extract_metadata(pdf_file, combined_content[:3000], fitz_docs[0] if fitz_docs else None)

            doc = Document(
                page_content=combined_content,
                metadata={
                    'source': str(pdf_file),
                    'filename': pdf_file.name,
                    'total_pages': len(fitz_docs),
                    'file_type': 'pdf',
                    'doc_type': 'pdf',
                    'doc_index': i,
                    **meta,
                }
            )
            all_docs.append(doc)

            if (i + 1) % 50 == 0:
                print(f"  Loaded {i + 1}/{len(pdf_files)} PDFs")

        except Exception as e:
            failed.append(pdf_file.name)
            print(f"  Error loading {pdf_file.name}: {e}")

    print(f"Loaded {len(all_docs)} PDFs, {len(failed)} failed")
    return all_docs


# ── Chunking ─────────────────────────────────────────────────────────────────
def is_valid_chunk(text: str) -> bool:
    if len(text) < MIN_CHUNK_LENGTH:
        return False
    if len(text) > MAX_CHUNK_LENGTH:
        return False
    digit_ratio = sum(c.isdigit() for c in text) / max(len(text), 1)
    if digit_ratio > MAX_DIGIT_RATIO:
        return False
    return True

def chunk_documents(docs: List[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=SEPARATORS,
        add_start_index=True,
    )

    all_chunks = []
    total_raw = 0
    total_valid = 0

    for doc_idx, doc in enumerate(docs):
        raw_chunks = splitter.split_documents([doc])
        total_raw += len(raw_chunks)

        valid_chunks = []
        for i, chunk in enumerate(raw_chunks):
            if is_valid_chunk(chunk.page_content):
                chunk.metadata.update({
                    'chunk_index': i,
                    'total_chunks_from_doc': len(raw_chunks),
                    'chunking_method': f'recursive_{CHUNK_SIZE}_{CHUNK_OVERLAP}',
                })
                valid_chunks.append(chunk)

        total_valid += len(valid_chunks)
        all_chunks.extend(valid_chunks)

    print(f"Chunks: {total_raw} raw → {total_valid} valid ({100*total_valid/max(total_raw,1):.1f}%)")
    return all_chunks


# ── Collection Management ────────────────────────────────────────────────────
DATA_TYPE_MAP = {
    "TEXT": DataType.TEXT,
    "INT": DataType.INT,
    "NUMBER": DataType.NUMBER,
    "BOOL": DataType.BOOL,
}

def create_collection(w_client, name: str) -> None:
    if w_client.collections.exists(name):
        print(f"  Deleting existing collection: {name}")
        w_client.collections.delete(name)

    properties = [
        Property(name=pname, data_type=DATA_TYPE_MAP[ptype])
        for pname, ptype in INDEX_PROPERTIES_FULL
    ]

    w_client.collections.create(
        name=name,
        properties=properties,
        vector_index_config=Configure.VectorIndex.hnsw(),
        vectorizer_config=Configure.Vectorizer.none(),
    )
    print(f"  Created collection: {name} with {len(properties)} properties")


# ── Safe Embedding Init ──────────────────────────────────────────────────────
def safe_init_embeddings(model_name: str, trust_remote_code: bool = False):
    try:
        emb = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": "cuda"},
            encode_kwargs={"device": "cuda"},
        )
        return emb, "cuda"
    except RuntimeError as e:
        print(f"  [OOM on CUDA] Falling back to CPU: {e}")
        safe_empty_cache()
        emb = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"device": "cpu"},
        )
        return emb, "cpu"
    except Exception as e:
        print(f"  [Error] Using CPU: {e}")
        emb = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"device": "cpu"},
        )
        return emb, "cpu"


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    print("\n" + "="*60)
    print("RAG-IDEArq — Reindex GTE + E5")
    print("="*60)
    print(f"Chunking: {CHUNK_SIZE}/{CHUNK_OVERLAP}")
    print(f"Batch size: 5 (reduced to avoid OOM)")
    print(f"Embeddings to reindex: gte-multilingual-base, e5-large-instruct")
    print("="*60)

    # Connect to Weaviate
    host = WEAVIATE_URL.replace("http://", "").replace("https://", "").split(":")[0]
    w_client = weaviate.connect_to_local(host=host, port=8080, grpc_port=50051)
    print(f"\nWeaviate connected: {w_client.is_ready()}")
    print(f"Existing collections: {list(w_client.collections.list_all().keys())}")

    # Load PDFs
    print("\n[1/4] Loading PDFs...")
    docs = load_pdfs(Path(INGESTA_DIR))
    if not docs:
        print("ERROR: No PDFs loaded.")
        sys.exit(1)

    # Chunk documents
    print("\n[2/4] Chunking documents...")
    chunks = chunk_documents(docs)
    if not chunks:
        print("ERROR: No valid chunks.")
        sys.exit(1)

    # Check which embeddings need reindexing
    print("\n[3/4] Checking existing collections...")
    embeddings_to_index = []
    for emb_key in ["gte-multilingual-base", "e5-large-instruct"]:
        coll_name = collection_name(emb_key)
        if w_client.collections.exists(coll_name):
            coll = w_client.collections.get(coll_name)
            agg = coll.aggregate.over_all(total_count=True)
            obj_count = agg.total_count or 0
            if obj_count > 0:
                print(f"✅ {emb_key}: {obj_count} objects already indexed. Skipping.")
            else:
                print(f"⚠️  {emb_key}: Collection exists but empty. Will reindex.")
                embeddings_to_index.append(emb_key)
        else:
            print(f"❌ {emb_key}: Collection doesn't exist. Will create and index.")
            embeddings_to_index.append(emb_key)

    if not embeddings_to_index:
        print("\n✅ All embeddings already indexed. Nothing to do.")
        w_client.close()
        return

    print(f"\nWill index: {embeddings_to_index}")

    # Index each embedding
    print("\n[4/4] Indexing...")
    results = []
    for emb_key in embeddings_to_index:
        emb_cfg = EMBEDDINGS[emb_key]
        coll_name = collection_name(emb_key)
        
        print(f"\n{'='*60}")
        print(f"Indexing: {emb_key}")
        print(f"Collection: {coll_name}")
        print(f"Chunks to index: {len(chunks)}")
        print(f"{'='*60}")
        
        create_collection(w_client, coll_name)
        
        if emb_cfg["model_class"] == "E5InstructEmbeddings":
            emb = E5InstructEmbeddings(model_name=emb_cfg["model_name"])
            device = emb.device
        else:
            emb, device = safe_init_embeddings(
                emb_cfg["model_name"],
                emb_cfg.get("trust_remote_code", False)
            )
        print(f"  Model loaded on {device}: {emb_cfg['model_name']}")
        
        vs = WeaviateVectorStore(
            client=w_client,
            index_name=coll_name,
            text_key="content",
            embedding=emb,
            attributes=[p[0] for p in INDEX_PROPERTIES_FULL if p[0] != "content"],
        )
        
        batch_size = 5
        indexed = 0
        errors = 0
        t0 = time.time()
        
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            try:
                vs.add_documents(batch)
                indexed += len(batch)
                
                if indexed % 50 == 0:
                    elapsed = time.time() - t0
                    rate = indexed / elapsed if elapsed > 0 else 0
                    print(f"  [{indexed}/{len(chunks)}] {rate:.1f} chunks/s")
                    
            except Exception as e:
                errors += len(batch)
                print(f"  Error at chunk {i}: {e}")
                safe_empty_cache()
                
                print(f"  Retrying with batch_size=1...")
                for chunk in batch:
                    try:
                        vs.add_documents([chunk])
                        indexed += 1
                    except Exception as e2:
                        print(f"    Failed: {e2}")
        
        coll = w_client.collections.get(coll_name)
        agg = coll.aggregate.over_all(total_count=True)
        obj_count = agg.total_count or 0
        
        elapsed = time.time() - t0
        print(f"\n  Results: {indexed} indexed, {errors} errors, {obj_count} objects in Weaviate")
        print(f"  Time: {elapsed:.1f}s ({indexed/elapsed:.1f} chunks/s)")
        
        results.append({
            "embedding": emb_key,
            "collection": coll_name,
            "indexed": indexed,
            "errors": errors,
            "object_count": obj_count,
            "time_s": elapsed,
            "device": device,
        })
        
        safe_empty_cache()

    # Report
    print("\n" + "="*60)
    print("Final Report")
    print("="*60)
    for r in results:
        status = "OK" if r["errors"] == 0 and r["indexed"] == r["object_count"] else "WARN"
        print(f"  [{status}] {r['embedding']} → {r['collection']}")
        print(f"         Indexed: {r['indexed']}, Objects: {r['object_count']}, Errors: {r['errors']}")
        print(f"         Device: {r['device']}, Time: {r['time_s']:.1f}s")

    w_client.close()
    print("\nDone!")


if __name__ == "__main__":
    main()
