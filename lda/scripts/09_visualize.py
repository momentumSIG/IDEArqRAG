import sys
import pickle
from pathlib import Path
import pandas as pd
from gensim import corpora
from gensim.models import LdaModel
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis

sys.path.insert(0, str(Path(__file__).parent))
from utils import load_config, setup_logger, check_ram_usage, log_memory_usage


def main():
    config = load_config()
    logger = setup_logger("09_visualize")
    
    data_dir = Path(config['paths']['data'])
    models_dir = Path(config['paths']['models'])
    outputs_dir = Path(config['paths']['outputs'])
    outputs_dir.mkdir(parents=True, exist_ok=True)
    
    memory_log = data_dir / "memory_usage.csv"
    log_memory_usage(memory_log, "start_09_visualize", logger)
    
    coherence_path = data_dir / "coherence_scores.csv"
    df_coherence = pd.read_csv(coherence_path)
    df_coherence = df_coherence.dropna(subset=['coherence_cv'])
    best_k = int(df_coherence.loc[df_coherence['coherence_cv'].idxmax(), 'K'])
    logger.info(f"Mejor K={best_k}")
    
    best_model_dir = models_dir / f"lda_K{best_k}_BEST"
    if not best_model_dir.exists():
        best_model_dir = models_dir / f"lda_K{best_k}"
    
    lda = LdaModel.load(str(best_model_dir / "model.gensim"))
    dictionary = corpora.Dictionary.load(str(data_dir / "06_dictionary.gensim"))
    
    with open(data_dir / "06_corpus.pkl", 'rb') as f:
        corpus = pickle.load(f)
    
    logger.info("Generando visualización pyLDAvis...")
    vis_data = gensimvis.prepare(lda, corpus, dictionary)
    
    html_path = outputs_dir / f"pyldavis_K{best_k}.html"
    pyLDAvis.save_html(vis_data, str(html_path))
    logger.info(f"Visualización guardada en {html_path}")
    
    log_memory_usage(memory_log, "pyldavis_done", logger)
    
    logger.info("Generando top-15 papers por tópico...")
    
    outputs_path = outputs_dir / "papers_topics.csv"
    df_papers = pd.read_csv(outputs_path)
    
    top_n = config['export']['top_n_papers_per_topic']
    
    all_top_papers = []
    
    for topic_id in sorted(df_papers['topic_dominant'].unique()):
        topic_papers = df_papers[df_papers['topic_dominant'] == topic_id].copy()
        topic_papers = topic_papers.sort_values('topic_dominant_weight', ascending=False)
        top_papers = topic_papers.head(top_n)
        
        for rank, (_, row) in enumerate(top_papers.iterrows(), 1):
            all_top_papers.append({
                'topic_id': topic_id,
                'rank': rank,
                'filename': row['filename'],
                'title': row.get('title', ''),
                'author': row.get('author', ''),
                'doi': row.get('doi', ''),
                'weight': row['topic_dominant_weight']
            })
    
    df_top = pd.DataFrame(all_top_papers)
    top_path = outputs_dir / "top15_papers_per_topic.csv"
    df_top.to_csv(top_path, index=False, encoding='utf-8')
    logger.info(f"Top-15 papers guardado en {top_path}")
    
    logger.info("Generando keywords por tópico...")
    
    topic_keywords = []
    for topic_id in range(lda.num_topics):
        terms = lda.show_topic(topic_id, topn=20)
        keywords = [term for term, _ in terms]
        
        topic_keywords.append({
            'topic_id': topic_id,
            'top_words': ', '.join(keywords),
            'top_words_list': keywords
        })
    
    df_keywords = pd.DataFrame(topic_keywords)
    keywords_path = outputs_dir / "topics_keywords.csv"
    df_keywords[['topic_id', 'top_words']].to_csv(keywords_path, index=False, encoding='utf-8')
    logger.info(f"Keywords guardadas en {keywords_path}")
    
    log_memory_usage(memory_log, "end_09_visualize", logger)
    logger.info("Visualización completada")


if __name__ == "__main__":
    main()
