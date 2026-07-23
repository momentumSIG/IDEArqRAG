import sys
import pickle
from pathlib import Path
import pandas as pd
from gensim import corpora

sys.path.insert(0, str(Path(__file__).parent))
from utils import load_config, setup_logger, check_ram_usage, log_memory_usage


def main():
    config = load_config()
    logger = setup_logger("06_build_corpus")
    
    data_dir = Path(config['paths']['data'])
    
    memory_log = data_dir / "memory_usage.csv"
    log_memory_usage(memory_log, "start_06_build_corpus", logger)
    
    lemmatized_path = data_dir / "05_lemmatized.parquet"
    if not lemmatized_path.exists():
        logger.error(f"No se encontró {lemmatized_path}. Ejecuta 05_lemmatize.py primero.")
        sys.exit(1)
    
    df = pd.read_parquet(lemmatized_path)
    logger.info(f"Cargados {len(df)} documentos de {lemmatized_path}")
    
    texts = df['tokens'].tolist()
    
    logger.info("Creando diccionario...")
    dictionary = corpora.Dictionary(texts)
    
    no_below = config['lda']['dictionary']['no_below']
    no_above = config['lda']['dictionary']['no_above']
    keep_n = config['lda']['dictionary']['keep_n']
    
    logger.info(f"Filtrando: no_below={no_below}, no_above={no_above}, keep_n={keep_n}")
    dictionary.filter_extremes(no_below=no_below, no_above=no_above, keep_n=keep_n)
    
    logger.info(f"Tamaño del diccionario después de filtrar: {len(dictionary)}")
    
    logger.info("Creando corpus BoW...")
    corpus = [dictionary.doc2bow(text) for text in texts]
    
    dict_path = data_dir / "06_dictionary.gensim"
    dictionary.save(str(dict_path))
    logger.info(f"Diccionario guardado en {dict_path}")
    
    corpus_path = data_dir / "06_corpus.pkl"
    with open(corpus_path, 'wb') as f:
        pickle.dump(corpus, f)
    logger.info(f"Corpus guardado en {corpus_path}")
    
    log_memory_usage(memory_log, "end_06_build_corpus", logger)
    
    corpus_stats = {
        'num_documents': len(corpus),
        'vocab_size': len(dictionary),
        'avg_tokens_per_doc': sum(len(doc) for doc in corpus) / len(corpus) if corpus else 0,
        'total_tokens': sum(sum(count for _, count in doc) for doc in corpus)
    }
    logger.info(f"Estadísticas del corpus: {corpus_stats}")


if __name__ == "__main__":
    main()
