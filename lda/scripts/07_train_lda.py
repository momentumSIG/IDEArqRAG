import sys
import pickle
import gc
from pathlib import Path
import pandas as pd
from gensim import corpora
from gensim.models import LdaModel, CoherenceModel

sys.path.insert(0, str(Path(__file__).parent))
from utils import load_config, setup_logger, check_ram_usage, log_memory_usage


def train_lda_for_k(corpus, dictionary, texts, k, config, logger):
    lda_config = config['lda']
    
    logger.info(f"Entrenando LDA con K={k}...")
    
    lda = LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=k,
        passes=lda_config['passes'],
        iterations=lda_config['iterations'],
        chunksize=lda_config['chunksize'],
        random_state=lda_config['random_state'],
        alpha='auto',
        eta='auto'
    )
    
    logger.info(f"Calculando coherence para K={k}...")
    coherence_model = CoherenceModel(
        model=lda,
        texts=texts,
        dictionary=dictionary,
        coherence=lda_config['evaluation']['coherence']
    )
    coherence_score = coherence_model.get_coherence()
    
    logger.info(f"K={k}: coherence={coherence_score:.4f}")
    
    return lda, coherence_score


def main():
    config = load_config()
    logger = setup_logger("07_train_lda")
    
    data_dir = Path(config['paths']['data'])
    models_dir = Path(config['paths']['models'])
    models_dir.mkdir(parents=True, exist_ok=True)
    
    memory_log = data_dir / "memory_usage.csv"
    log_memory_usage(memory_log, "start_07_train_lda", logger)
    
    dict_path = data_dir / "06_dictionary.gensim"
    corpus_path = data_dir / "06_corpus.pkl"
    lemmatized_path = data_dir / "05_lemmatized.parquet"
    
    if not dict_path.exists() or not corpus_path.exists():
        logger.error("No se encontraron corpus o diccionario. Ejecuta 06_build_corpus.py primero.")
        sys.exit(1)
    
    logger.info("Cargando diccionario y corpus...")
    dictionary = corpora.Dictionary.load(str(dict_path))
    
    with open(corpus_path, 'rb') as f:
        corpus = pickle.load(f)
    
    df = pd.read_parquet(lemmatized_path)
    texts = df['tokens'].tolist()
    
    logger.info(f"Corpus cargado: {len(corpus)} documentos, {len(dictionary)} términos")
    
    topic_range = config['lda']['topic_range']
    logger.info(f"Grid search sobre K={topic_range}")
    
    results = []
    best_k = None
    best_coherence = -1
    best_model = None
    
    for k in topic_range:
        logger.info(f"\n{'='*60}")
        logger.info(f"Entrenando para K={k}")
        logger.info(f"{'='*60}")
        
        try:
            model, coherence = train_lda_for_k(corpus, dictionary, texts, k, config, logger)
            
            model_dir = models_dir / f"lda_K{k}"
            model_dir.mkdir(parents=True, exist_ok=True)
            model.save(str(model_dir / "model.gensim"))
            
            with open(model_dir / "coherence.txt", 'w') as f:
                f.write(f"{coherence:.6f}\n")
            
            logger.info(f"Modelo K={k} guardado en {model_dir}")
            
            results.append({
                'K': k,
                'coherence_cv': coherence
            })
            
            if coherence > best_coherence:
                best_coherence = coherence
                best_k = k
                best_model = model
            
            del model
            gc.collect()
            
            check_ram_usage(logger, config['ram']['cap_gb'], config['ram']['warning_gb'])
            log_memory_usage(memory_log, f"train_lda_K{k}", logger)
            
        except Exception as e:
            logger.error(f"Error entrenando K={k}: {e}")
            results.append({
                'K': k,
                'coherence_cv': None
            })
    
    df_results = pd.DataFrame(results)
    results_path = data_dir / "coherence_scores.csv"
    df_results.to_csv(results_path, index=False)
    logger.info(f"Resultados de coherence guardados en {results_path}")
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Mejor K={best_k} con coherence={best_coherence:.4f}")
    logger.info(f"{'='*60}")
    
    if best_model:
        best_model_dir = models_dir / f"lda_K{best_k}_BEST"
        best_model_dir.mkdir(parents=True, exist_ok=True)
        best_model.save(str(best_model_dir / "model.gensim"))
        logger.info(f"Mejor modelo guardado en {best_model_dir}")
    
    log_memory_usage(memory_log, "end_07_train_lda", logger)


if __name__ == "__main__":
    main()
