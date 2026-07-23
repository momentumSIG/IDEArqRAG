#!/usr/bin/env python3
"""
Full RAG evaluation script for RAG-IDEArq.
Runs all combinations of embeddings, LLMs, prompts, and temperatures.
Saves results to data/results/eval_v3_<timestamp>.csv

Usage:
    cd /home/raglinux/RAG
    source ../env_rag/bin/activate
    python scripts/run_full_evaluation.py
"""
import os
import sys
import time
import itertools
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

# Setup paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env", override=True)

from src.config import (
    EMBEDDINGS, LLMS, LLM_TEMPERATURES, RERANKING_CONFIG,
    collection_name, RESULTS_DIR,
)
from src.langfuse_monitor import get_callback, get_prompt, score_trace, update_trace_output
from src.reranker import rerank_documents
from data.eval_questions import get_eval_items

import weaviate
from langchain_weaviate import WeaviateVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from ragas import EvaluationDataset, evaluate
from ragas.metrics import Faithfulness, ContextPrecision, ContextRecall, AnswerCorrectness
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.dataset_schema import SingleTurnSample
from ragas.run_config import RunConfig
from langchain_mistralai import ChatMistralAI

import pandas as pd
import warnings
warnings.filterwarnings('ignore')


def main():
    print("="*60)
    print("RAG-IDEArq — Full Evaluation")
    print("="*60)
    print(f"Embeddings: {list(EMBEDDINGS.keys())}")
    print(f"LLMs: {list(LLMS.keys())}")
    print(f"Temperatures: {LLM_TEMPERATURES}")
    print(f"Reranking enabled: {RERANKING_CONFIG['enabled']}")
    print(f"Rerank model: {RERANKING_CONFIG['model']}")
    print()

    # Load dataset
    eval_items = get_eval_items()
    print(f"Dataset: {len(eval_items)} items")

    # Connect to Weaviate
    print("\nConnecting to Weaviate...")
    w_client = weaviate.connect_to_local(host='localhost', port=8080, grpc_port=50051)
    print(f"Weaviate: {w_client.is_ready()}")
    print(f"Collections: {list(w_client.collections.list_all().keys())}")

    # Load embedding models
    print("\nLoading embedding models...")
    embedding_models = {}
    for emb_key, emb_cfg in EMBEDDINGS.items():
        try:
            emb = HuggingFaceEmbeddings(
                model_name=emb_cfg["model_name"],
                model_kwargs={"device": "cuda"},
                encode_kwargs={"device": "cuda"},
            )
            embedding_models[emb_key] = emb
            print(f"  [OK] {emb_key} on CUDA")
        except Exception as e:
            if emb_cfg.get("trust_remote_code", False):
                try:
                    emb = HuggingFaceEmbeddings(
                        model_name=emb_cfg["model_name"],
                        model_kwargs={"device": "cuda", "trust_remote_code": True},
                        encode_kwargs={"device": "cuda"},
                    )
                    embedding_models[emb_key] = emb
                    print(f"  [OK] {emb_key} on CUDA (trust_remote_code)")
                except Exception as e2:
                    emb = HuggingFaceEmbeddings(
                        model_name=emb_cfg["model_name"],
                        model_kwargs={"device": "cpu", "trust_remote_code": True},
                        encode_kwargs={"device": "cpu"},
                    )
                    embedding_models[emb_key] = emb
                    print(f"  [CPU] {emb_key} (trust_remote_code)")
            else:
                emb = HuggingFaceEmbeddings(
                    model_name=emb_cfg["model_name"],
                    model_kwargs={"device": "cpu"},
                    encode_kwargs={"device": "cpu"},
                )
                embedding_models[emb_key] = emb
                print(f"  [CPU] {emb_key}")

    # Load LLM models
    print("\nLoading LLM models...")
    llm_models = {}
    for llm_name, llm_model in LLMS.items():
        for temp in LLM_TEMPERATURES:
            key = f"{llm_name}_t{temp}"
            llm_models[key] = OllamaLLM(model=llm_model, temperature=temp)
            print(f"  [OK] {key}")

    # RAGAS setup
    print("\nSetting up RAGAS...")
    mistral_judge = ChatMistralAI(
        model="mistral-small-latest", temperature=0.1, max_retries=3, timeout=180,
    )
    ragas_llm = LangchainLLMWrapper(mistral_judge)
    ragas_embeddings = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    )
    ragas_metrics = [
        Faithfulness(llm=ragas_llm),
        ContextPrecision(llm=ragas_llm),
        ContextRecall(llm=ragas_llm),
        AnswerCorrectness(llm=ragas_llm, embeddings=ragas_embeddings),
    ]
    ragas_run_config = RunConfig(
        max_workers=1, max_retries=10, max_wait=120, timeout=900,
    )
    print("RAGAS metrics configured")

    # Build RAG chain function
    def build_rag_chain(emb_key, llm_name, prompt_key):
        coll_name = collection_name(emb_key)
        if not w_client.collections.exists(coll_name):
            raise ValueError(f"Collection '{coll_name}' does not exist.")

        vs = WeaviateVectorStore(
            client=w_client,
            index_name=coll_name,
            text_key="content",
            embedding=embedding_models[emb_key],
            attributes=["filename", "source", "chunk_index", "year", "language",
                        "doi", "authors", "periodo", "region"],
        )

        k = RERANKING_CONFIG["k_retrieval"] if RERANKING_CONFIG["enabled"] else RERANKING_CONFIG["top_n"]
        retriever = vs.as_retriever(search_type="similarity", search_kwargs={"k": k})
        
        # Use local prompt template (Langfuse prompts have API issues)
        prompt_template = ChatPromptTemplate.from_template(
            """Eres un asistente experto en arqueología, historia y sistemas de información geográfica. Responde la pregunta usando los contextos proporcionados, que pueden estar en español, inglés, francés, catalán o portugués.
Sintetiza información de todos los contextos relevantes independientemente de su idioma. Si encuentras información relevante en cualquier idioma, úsala para construir tu respuesta en el mismo idioma en el que se realiza la pregunta.

Contexto: {context}

Pregunta: {question}

Instrucciones:
- Si encuentras información parcial en el contexto, intégrala en la respuesta aunque no sea completa.
- Si no hay absolutamente nada relevante, responde claramente: "No hay información suficiente en el contexto".
- NO INVENTES NI ALUCINES INFORMACIÓN.
- Cuando sea posible, cita explícitamente los puntos clave del contexto (ej. autores, años, títulos de publicaciones, yacimientos, cronologías, dataciones, coordenadas).
- Haz cálculos utilizando la distancia euclidiana, después transforma los grados a kilómetros y proporciona los nombres de los yacimientos.
- Responde siempre de forma clara, estructurada y útil para un investigador.

Basándote en la información anterior y en tu conocimiento sobre arqueología y sistemas de información geográfica, responde: 

Respuesta:"""
        )

        llm = llm_models[f"{llm_name}_t0.3"]  # Will be overridden

        return retriever, prompt_template, llm

    def run_rag_with_rerank(retriever, prompt_template, llm, question, callback_info):
        docs = retriever.invoke(question)

        if RERANKING_CONFIG["enabled"] and len(docs) > RERANKING_CONFIG["top_n"]:
            ranked = rerank_documents(
                question, docs,
                top_n=RERANKING_CONFIG["top_n"],
                reranker_model=RERANKING_CONFIG["model"]
            )
            selected_docs = [r["doc"] for r in ranked]
        else:
            selected_docs = docs[:RERANKING_CONFIG["top_n"]]

        context = "\n\n".join([d.page_content for d in selected_docs])
        if hasattr(prompt_template, 'format'):
            prompt_text = prompt_template.format(context=context, question=question)
        else:
            prompt_text = prompt_template.compile(context=context, question=question)

        answer = llm.invoke(prompt_text)
        return answer, selected_docs

    # Experiment grid
    combos = list(itertools.product(
        EMBEDDINGS.keys(),
        LLMS.keys(),
        ["zero_shot", "one_shot", "few_shot"],
        LLM_TEMPERATURES,
    ))

    print(f"\n{'='*60}")
    print(f"Total combos: {len(combos)}")
    print(f"Questions per combo: {len(eval_items)}")
    print(f"Total evaluations: {len(combos) * len(eval_items)}")
    print(f"{'='*60}\n")

    all_results = []
    total_start = time.time()
    combo_count = 0

    for emb_key, llm_name, prompt_key, temperature in combos:
        combo_start = time.time()
        combo_count += 1
        llm_key = f"{llm_name}_t{temperature}"
        combo_label = f"{emb_key}|{llm_name}|{prompt_key}|t{temperature}"

        print(f"\n{'='*60}")
        print(f"Combo {combo_count}/{len(combos)}: {combo_label}")
        print(f"{'='*60}")

        try:
            retriever, prompt_template, _ = build_rag_chain(emb_key, llm_name, prompt_key)
            llm = llm_models[llm_key]
        except ValueError as e:
            print(f"  SKIP: {e}")
            continue

        ragas_samples = []

        for idx, item in enumerate(eval_items):
            question = item["input"]["question"]
            ground_truth = item["expected_output"]["ground_truth"]
            source = item["expected_output"]["source"]
            tipo = item["input"]["tipo"]
            idioma = item["input"]["idioma"]
            n_articulos = item["input"]["n_articulos"]

            print(f"  Q{idx+1}/{len(eval_items)} [{tipo}/{idioma}]: {question[:60]}...")

            # Langfuse trace (created after generation to include input+output)
            trace_id = None

            t0 = time.time()
            try:
                answer, docs = run_rag_with_rerank(
                    retriever, prompt_template, llm, question, {}
                )
                latency = time.time() - t0

                # Create Langfuse trace with input+output
                callback_info = get_callback(
                    session_id=combo_label,
                    tags=[emb_key, llm_name, prompt_key, f"t{temperature}", tipo, idioma],
                    input_data={"question": question},
                    output_data={"answer": answer[:500]},
                )
                trace_id = callback_info.get("trace_id")

                # Build RAGAS sample
                contexts = [d.page_content[:800] for d in docs[:5]]
                sample = SingleTurnSample(
                    user_input=question,
                    response=answer,
                    reference=ground_truth,
                    retrieved_contexts=contexts,
                )
                ragas_samples.append(sample)

                # Store result
                all_results.append({
                    "combo": combo_label,
                    "embedding": emb_key,
                    "llm": llm_name,
                    "prompt": prompt_key,
                    "temperature": temperature,
                    "use_rerank": RERANKING_CONFIG["enabled"],
                    "question": question,
                    "answer": answer[:500] if answer else "",
                    "ground_truth": ground_truth[:500],
                    "source": source,
                    "tipo": tipo,
                    "idioma": idioma,
                    "n_articulos": n_articulos,
                    "n_docs_retrieved": len(docs),
                    "latency_s": latency,
                    "faithfulness": 0,  # Will be filled by RAGAS
                    "context_precision": 0,
                    "context_recall": 0,
                    "answer_correctness": 0,
                })

                print(f"    OK ({latency:.1f}s, {len(docs)} docs)")

            except Exception as e:
                print(f"    ERROR: {e}")
                all_results.append({
                    "combo": combo_label,
                    "embedding": emb_key,
                    "llm": llm_name,
                    "prompt": prompt_key,
                    "temperature": temperature,
                    "use_rerank": RERANKING_CONFIG["enabled"],
                    "question": question,
                    "answer": f"Error: {e}",
                    "ground_truth": ground_truth[:500],
                    "source": source,
                    "tipo": tipo,
                    "idioma": idioma,
                    "n_articulos": n_articulos,
                    "n_docs_retrieved": 0,
                    "latency_s": 0,
                    "faithfulness": 0,
                    "context_precision": 0,
                    "context_recall": 0,
                    "answer_correctness": 0,
                })

        # Run RAGAS for this combo
        if ragas_samples:
            print(f"\n  Running RAGAS ({len(ragas_samples)} samples)...")
            eval_ds = EvaluationDataset(samples=ragas_samples)

            for attempt in range(5):
                try:
                    ragas_result = evaluate(
                        dataset=eval_ds,
                        metrics=ragas_metrics,
                        llm=ragas_llm,
                        embeddings=ragas_embeddings,
                        run_config=ragas_run_config,
                    )
                    break
                except Exception as e:
                    if "429" in str(e) or "rate_limit" in str(e).lower():
                        wait = 30 * (2 ** attempt)
                        print(f"  [Rate limit] Waiting {wait}s ({attempt+1}/5)...")
                        time.sleep(wait)
                    else:
                        print(f"  [RAGAS Error] {e}")
                        ragas_result = None
                        break

            if ragas_result:
                # RAGAS v0.4.3: result.scores is a list of dicts
                scores_list = ragas_result.scores
                print(f"  RAGAS results:")
                for i, score_dict in enumerate(scores_list):
                    for metric_name, metric_value in score_dict.items():
                        print(f"    Q{i+1} {metric_name}: {metric_value:.4f}")
                        # Update result
                        if i < len(all_results) and all_results[-len(scores_list) + i].get("combo") == combo_label:
                            result_idx = len(all_results) - len(scores_list) + i
                            all_results[result_idx][metric_name] = metric_value
                        # Score in Langfuse
                        try:
                            score_trace(trace_id=trace_id, name=metric_name, value=metric_value)
                        except Exception:
                            pass

        combo_elapsed = time.time() - combo_start
        print(f"\n  Combo time: {combo_elapsed:.1f}s")
        print(f"  Waiting 60s before next combo...")
        time.sleep(60)

    # Save results
    total_elapsed = time.time() - total_start
    print(f"\n{'='*60}")
    print(f"Total time: {total_elapsed:.1f}s ({total_elapsed/3600:.1f}h)")
    print(f"Results: {len(all_results)}")
    print(f"{'='*60}")

    df = pd.DataFrame(all_results)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    rerank_tag = "rerank" if RERANKING_CONFIG["enabled"] else "baseline"
    output_path = Path(RESULTS_DIR) / f"eval_v3_{rerank_tag}_{timestamp}.csv"
    Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    if 'faithfulness' in df.columns:
        print("\nBy embedding:")
        print(df.groupby("embedding")[["faithfulness", "context_precision", "context_recall", "answer_correctness"]].mean())
        print("\nBy LLM:")
        print(df.groupby("llm")[["faithfulness", "context_precision", "context_recall", "answer_correctness"]].mean())
        print("\nBy type:")
        print(df.groupby("tipo")[["faithfulness", "context_precision", "context_recall", "answer_correctness"]].mean())
        print("\nBy language:")
        print(df.groupby("idioma")[["faithfulness", "context_precision", "context_recall", "answer_correctness"]].mean())

    w_client.close()
    print("\n✅ Evaluation completed!")


if __name__ == "__main__":
    main()
