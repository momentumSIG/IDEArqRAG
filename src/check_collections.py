import sys
from pathlib import Path

# Fix para Jupyter
try:
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
except NameError:
    PROJECT_ROOT = Path.cwd().parent.parent

sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env", override=True)

import weaviate
from src.config import WEAVIATE_URL, EMBEDDINGS, collection_name

# Conectar a Weaviate
host = WEAVIATE_URL.replace("http://", "").replace("https://", "").split(":")[0]
w_client = weaviate.connect_to_local(host=host, port=8080, grpc_port=50051)

print("="*60)
print("Colecciones en Weaviate")
print("="*60)
print(f"Server: {host}:8080")
print(f"Ready: {w_client.is_ready()}\n")

collections = w_client.collections.list_all()
if not collections:
    print("No hay colecciones indexadas.")
else:
    print(f"{'Colección':<40} {'Objetos':>10}")
    print("-"*60)
    for name in sorted(collections.keys()):
        coll = w_client.collections.get(name)
        agg = coll.aggregate.over_all(total_count=True)
        obj_count = agg.total_count or 0
        print(f"{name:<40} {obj_count:>10}")
    
    # Mostrar colecciones esperadas
    print("\n" + "="*60)
    print("Colecciones esperadas (según config.py):")
    print("="*60)
    for emb_key in EMBEDDINGS.keys():
        expected = collection_name(emb_key)
        exists = "✓" if expected in collections else "✗"
        print(f"{exists} {expected}")

w_client.close()