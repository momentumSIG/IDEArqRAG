import sys
import pickle
from pathlib import Path
import pandas as pd
from gensim import corpora
from gensim.models import LdaModel

sys.path.insert(0, str(Path(__file__).parent))
from utils import load_config, setup_logger, check_ram_usage, log_memory_usage


def main():
    config = load_config()
    logger = setup_logger("08_assign")
    
    data_dir = Path(config['paths']['data'])
    models_dir = Path(config['paths']['models'])
    outputs_dir = Path(config['paths']['outputs'])
    outputs_dir.mkdir(parents=True, exist_ok=True)
    
    memory_log = data_dir / "memory_usage.csv"
    log_memory_usage(memory_log, "start_08_assign", logger)
    
    coherence_path = data_dir / "coherence_scores.csv"
    if not coherence_path.exists():
        logger.error(f"No se encontró {coherence_path}. Ejecuta 07_train_lda.py primero.")
        sys.exit(1)
    
    df_coherence = pd.read_csv(coherence_path)
    df_coherence = df_coherence.dropna(subset=['coherence_cv'])
    best_k = int(df_coherence.loc[df_coherence['coherence_cv'].idxmax(), 'K'])
    logger.info(f"Mejor K={best_k} seleccionado por coherence")
    
    best_model_dir = models_dir / f"lda_K{best_k}_BEST"
    if not best_model_dir.exists():
        best_model_dir = models_dir / f"lda_K{best_k}"
    
    if not best_model_dir.exists():
        logger.error(f"No se encontró modelo para K={best_k}")
        sys.exit(1)
    
    logger.info(f"Cargando modelo desde {best_model_dir}")
    lda = LdaModel.load(str(best_model_dir / "model.gensim"))
    
    dict_path = data_dir / "06_dictionary.gensim"
    dictionary = corpora.Dictionary.load(str(dict_path))
    
    lemmatized_path = data_dir / "05_lemmatized.parquet"
    df = pd.read_parquet(lemmatized_path)
    logger.info(f"Cargados {len(df)} documentos")
    
    corpus_path = data_dir / "06_corpus.pkl"
    with open(corpus_path, 'rb') as f:
        corpus = pickle.load(f)
    
    topic_assignments = []
    
    for i, (doc_bow, row) in enumerate(zip(corpus, df.iterrows())):
        filename = row[1]['filename']
        
        topic_dist = lda.get_document_topics(doc_bow, minimum_probability=0.0)
        
        topic_dict = {f"topic_{t}": p for t, p in topic_dist}
        
        dominant_topic = max(topic_dist, key=lambda x: x[1])
        topic_id = dominant_topic[0]
        topic_weight = dominant_topic[1]
        
        topic_assignments.append({
            'filename': filename,
            'topic_dominant': topic_id,
            'topic_dominant_weight': topic_weight,
            'topic_distribution': str(topic_dict)
        })
        
        if (i + 1) % 50 == 0:
            logger.info(f"[{i+1}/{len(corpus)}] Asignando tópicos...")
            check_ram_usage(logger, config['ram']['cap_gb'], config['ram']['warning_gb'])
            log_memory_usage(memory_log, f"assign_progress_{i+1}", logger)
    
    df_topics = pd.DataFrame(topic_assignments)
    
    df_final = df.merge(df_topics, on='filename', how='left')
    
    output_path = outputs_dir / "papers_topics.csv"
    df_final.to_csv(output_path, index=False, encoding='utf-8')
    logger.info(f"Guardado en {output_path}")
    
    log_memory_usage(memory_log, "end_08_assign", logger)
    
    topic_counts = df_final['topic_dominant'].value_counts().sort_index()
    logger.info(f"Distribución de documentos por tópico dominante:\n{topic_counts}")


if __name__ == "__main__":
    main()
