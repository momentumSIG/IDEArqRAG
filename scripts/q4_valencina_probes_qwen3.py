"""Re-run the Q4 (Valencina) probes and dump the LLM's literal answers durably.

Read-only w.r.t. the repo except for the two output files under data/results/.
"""
import json
import os
import time
import urllib.error
import urllib.request

import pdfplumber

REPO = "/mnt/netapp1/Store_CSIC/home/csic/usi/eag/idearq-rag"
PDF = os.path.join(REPO, "data", "ingesta", "1135_Poblado_2021.pdf")
OUT_JSON = os.path.join(REPO, "data", "results", "q4_valencina_qwen3-4b_probes_20261015.json")

OLLAMA = "http://127.0.0.1:11436"
MODEL = "hf.co/unsloth/Qwen3-4B-Instruct-2507-GGUF:Q8_0"
OPTIONS = {"temperature": 0, "num_predict": 3000, "num_ctx": 16384, "num_thread": 16}
CALL_TIMEOUT = 1800  # seconds, per model call

QUESTION = ("¿Es correcta la datación -1377±23 para Valencina Cerro de la Cabeza "
            "Ladera Sur como la más reciente del área?")
P2_INSTRUCTION = ("Enumera las filas con BP < 300 y di la menor.")

GOLD = ("No, es errónea. La datación correcta más reciente es: Valencina, "
        "Cerro de la Cabeza, Ladera Sur, 175 ± 20.")


# ── 1. Extract the table (pages 31-32) with pdfplumber ───────────────────────
def extract_tables():
    pages = {}
    with pdfplumber.open(PDF) as pdf:
        for pno in (30, 31):  # 0-indexed -> PDF pages 31 and 32
            page = pdf.pages[pno]
            tables = page.extract_tables()
            pages[pno + 1] = tables
    return pages


def render_row(row):
    cells = []
    for c in row:
        if c is None:
            c = ""
        cells.append(" ".join(str(c).split()))
    return " | ".join(cells)


def build_context(tables):
    parts = []
    for pno in sorted(tables):
        for table in tables[pno]:
            for row in table:
                parts.append(render_row(row))
    return "\n".join(parts)


def bp_under_300(tables):
    out = []
    for pno in sorted(tables):
        for table in tables[pno]:
            for row in table:
                bp_raw = (row[1] or "").strip() if len(row) > 1 else ""
                try:
                    bp = int(bp_raw)
                except ValueError:
                    continue
                if bp < 300:
                    out.append({
                        "page": pno,
                        "bp": bp,
                        "plusminus": (row[2] or "").strip() if len(row) > 2 else "",
                        "yacimiento": " ".join((row[0] or "").split()),
                        "lab_material": " ".join((row[7] or "").split()) if len(row) > 7 else "",
                        "raw_row": [("" if c is None else c) for c in row],
                    })
    return out


# ── 2. Call the model ────────────────────────────────────────────────────────
def call_model(prompt):
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "options": OPTIONS,
    }
    req = urllib.request.Request(
        OLLAMA + "/api/generate",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=CALL_TIMEOUT) as r:
            body = json.loads(r.read().decode())
        wall = time.time() - t0
        return {
            "status": "ok",
            "timed_out": False,
            "wall_seconds": round(wall, 2),
            "response": body.get("response"),
            "done": body.get("done"),
            "done_reason": body.get("done_reason"),
            "eval_count": body.get("eval_count"),
            "prompt_eval_count": body.get("prompt_eval_count"),
            "total_duration_ns": body.get("total_duration"),
            "load_duration_ns": body.get("load_duration"),
            "prompt_eval_duration_ns": body.get("prompt_eval_duration"),
            "eval_duration_ns": body.get("eval_duration"),
        }
    except urllib.error.HTTPError as e:
        return {"status": "http_error", "timed_out": False,
                "wall_seconds": round(time.time() - t0, 2),
                "http_code": e.code, "error_body": e.read().decode()[:2000]}
    except Exception as e:  # socket timeout, connection reset, ...
        return {"status": "error", "timed_out": isinstance(e, TimeoutError),
                "wall_seconds": round(time.time() - t0, 2),
                "error": f"{type(e).__name__}: {e}"}


def save(doc):
    tmp = OUT_JSON + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, indent=2)
    os.replace(tmp, OUT_JSON)


def main():
    print("extracting tables from pages 31-32 ...", flush=True)
    tables = extract_tables()
    context = build_context(tables)
    low = bp_under_300(tables)
    has_175 = any(r["bp"] == 175 and r["plusminus"] == "20" for r in low)

    print(f"context chars={len(context)} rows={context.count(chr(10)) + 1}", flush=True)
    print("BP<300 rows:", json.dumps(low, ensure_ascii=False), flush=True)

    doc = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "question": QUESTION,
        "gold_answer": GOLD,
        "gold_sources": "geojson; 1135_Poblado_2021.pdf",
        "pdf": "data/ingesta/1135_Poblado_2021.pdf",
        "pdf_pages": "31-32",
        "endpoint": OLLAMA + "/api/generate",
        "model": MODEL,
        "options": OPTIONS,
        "call_timeout_seconds": CALL_TIMEOUT,
        "extraction_tool": "pdfplumber 0.11.10 (page.extract_tables)",
        "table_text": context,
        "bp_under_300_rows": low,
        "table_contains_175_20": has_175,
        "probes": [],
        "skipped": [],
    }

    p1_prompt = context + "\n\n" + QUESTION
    p1 = {
        "probe": "P1",
        "description": "solo la tabla (pregunta original, sin instrucciones extra)",
        "instruction": QUESTION,
        "prompt": p1_prompt,
        "context": context,
    }
    print("=== P1 ===", flush=True)
    p1.update(call_model(p1_prompt))
    doc["probes"].append(p1)
    save(doc)
    print("P1 done_reason:", p1.get("done_reason"), "status:", p1.get("status"),
          "eval_count:", p1.get("eval_count"), "wall:", p1.get("wall_seconds"), flush=True)

    p2_prompt = context + "\n\n" + P2_INSTRUCTION
    p2 = {
        "probe": "P2",
        "description": "lectura dirigida (filas BP < 300)",
        "instruction": P2_INSTRUCTION,
        "prompt": p2_prompt,
        "context": context,
    }
    print("=== P2 ===", flush=True)
    p2.update(call_model(p2_prompt))
    doc["probes"].append(p2)
    save(doc)
    print("P2 done_reason:", p2.get("done_reason"), "status:", p2.get("status"),
          "eval_count:", p2.get("eval_count"), "wall:", p2.get("wall_seconds"), flush=True)

    print("WROTE", OUT_JSON, os.path.getsize(OUT_JSON), flush=True)


if __name__ == "__main__":
    main()
