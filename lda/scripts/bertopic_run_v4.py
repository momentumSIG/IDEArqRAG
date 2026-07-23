import sys
from pathlib import Path
import pandas as pd
import json
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
import umap

sys.path.insert(0, str(Path(__file__).parent))
from utils import load_config, setup_logger, log_memory_usage


def main():
    config = load_config()
    logger = setup_logger("bertopic_v4")
    
    data_dir = Path(config['paths']['data'])
    outputs_dir = Path(config['paths']['outputs'])
    models_dir = Path(config['paths']['models'])
    
    memory_log = data_dir / "memory_usage.csv"
    log_memory_usage(memory_log, "start_bertopic_v4", logger)
    
    bertopic_path = data_dir / "04_cleaned_bertopic.parquet"
    df = pd.read_parquet(bertopic_path)
    logger.info(f"Cargados {len(df)} documentos")
    
    docs = df['text_bertopic'].apply(lambda x: x[:8000]).tolist()
    logger.info(f"Textos truncados a 8000 chars, longitud media: {sum(len(d) for d in docs)/len(docs):.0f}")
    
    logger.info("Cargando modelo de embeddings multilingüe...")
    embedding_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    
    log_memory_usage(memory_log, "model_loaded_v4", logger)
    
    umap_model = umap.UMAP(
        n_neighbors=15,
        n_components=10,
        min_dist=0.0,
        metric='cosine',
        random_state=42
    )
    
    logger.info("Entrenando BERTopic v4 (texto completo)...")
    topic_model = BERTopic(
        embedding_model=embedding_model,
        umap_model=umap_model,
        language="multilingual",
        nr_topics=15,
        calculate_probabilities=True,
        verbose=True,
        min_topic_size=10
    )
    
    topics, probs = topic_model.fit_transform(docs)
    
    n_topics = len(set(topics)) - (1 if -1 in topics else 0)
    n_outliers = topics.count(-1)
    logger.info(f"Tópicos encontrados: {n_topics}")
    logger.info(f"Documentos sin tópico (outliers): {n_outliers}")
    
    model_dir = models_dir / "bertopic_v4"
    model_dir.mkdir(parents=True, exist_ok=True)
    topic_model.save(str(model_dir / "bertopic_model"), serialization="safetensors", save_ctfidf=True, save_embedding_model=True)
    logger.info(f"Modelo guardado en {model_dir}")
    
    topic_info = topic_model.get_topic_info()
    logger.info(f"\nInformación de tópicos:\n{topic_info.to_string()}")
    
    topic_info_path = outputs_dir / "bertopic_v4_topic_info.csv"
    topic_info.to_csv(topic_info_path, index=False, encoding='utf-8')
    
    topics_keywords = []
    for topic_id in topic_info['Topic'].tolist():
        if topic_id == -1:
            continue
        words = topic_model.get_topic(topic_id)
        keywords = [word for word, score in words[:20]]
        topics_keywords.append({
            'topic_id': topic_id,
            'num_docs': len([t for t in topics if t == topic_id]),
            'top_words': ', '.join(keywords),
            'keywords': keywords
        })
    
    df_topics = pd.DataFrame(topics_keywords)
    keywords_path = outputs_dir / "bertopic_v4_topics_keywords.csv"
    df_topics[['topic_id', 'num_docs', 'top_words']].to_csv(keywords_path, index=False, encoding='utf-8')
    logger.info(f"\nTópicos BERTopic v4:\n{df_topics[['topic_id', 'num_docs', 'top_words']].to_string()}")
    
    df_result = df.copy()
    df_result['bertopic_topic'] = topics
    df_result['bertopic_prob'] = probs.max(axis=1) if hasattr(probs, 'max') else probs
    
    papers_path = outputs_dir / "bertopic_v4_papers_topics.csv"
    df_result[['filename', 'author', 'title', 'doi', 'language', 'bertopic_topic', 'bertopic_prob']].to_csv(papers_path, index=False, encoding='utf-8')
    
    graph_data = []
    for t in topics_keywords:
        topic_id = t['topic_id']
        topic_papers = df_result[df_result['bertopic_topic'] == topic_id]
        papers_list = []
        for _, row in topic_papers.iterrows():
            papers_list.append({
                'filename': row['filename'],
                'title': row.get('title', ''),
                'author': row.get('author', ''),
                'doi': row.get('doi', ''),
                'language': row.get('language', ''),
                'weight': float(row['bertopic_prob'])
            })
        graph_data.append({
            'id': f"BTv4_{topic_id:02d}",
            'topic_id': int(topic_id),
            'name': f"BERTopic_v4_{topic_id}",
            'keywords': t['keywords'],
            'num_papers': len(papers_list),
            'papers': papers_list
        })
    
    graph_path = outputs_dir / "bertopic_v4_topics_for_graph.json"
    with open(graph_path, 'w', encoding='utf-8') as f:
        json.dump(graph_data, f, ensure_ascii=False, indent=2)
    
    log_memory_usage(memory_log, "end_bertopic_v4", logger)
    logger.info("BERTopic v4 completado")


if __name__ == "__main__":
    main()
