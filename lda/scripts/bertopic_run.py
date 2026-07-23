import sys
from pathlib import Path
import pandas as pd
import json
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer

sys.path.insert(0, str(Path(__file__).parent))
from utils import load_config, setup_logger, check_ram_usage, log_memory_usage


def main():
    config = load_config()
    logger = setup_logger("bertopic")
    
    data_dir = Path(config['paths']['data'])
    outputs_dir = Path(config['paths']['outputs'])
    models_dir = Path(config['paths']['models'])
    
    memory_log = data_dir / "memory_usage.csv"
    log_memory_usage(memory_log, "start_bertopic", logger)
    
    lemmatized_path = data_dir / "05_lemmatized.parquet"
    df = pd.read_parquet(lemmatized_path)
    logger.info(f"Cargados {len(df)} documentos")
    
    docs = [" ".join(tokens) for tokens in df['tokens']]
    
    logger.info("Cargando modelo de embeddings multilingüe: paraphrase-multilingual-MiniLM-L12-v2")
    embedding_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    
    log_memory_usage(memory_log, "model_loaded", logger)
    
    logger.info("Entrenando BERTopic...")
    topic_model = BERTopic(
        embedding_model=embedding_model,
        language="multilingual",
        nr_topics="auto",
        calculate_probabilities=True,
        verbose=True
    )
    
    topics, probs = topic_model.fit_transform(docs)
    
    logger.info(f"Tópicos encontrados: {len(set(topics)) - (1 if -1 in topics else 0)}")
    logger.info(f"Documentos sin tópico (outliers): {topics.count(-1)}")
    
    model_dir = models_dir / "bertopic"
    model_dir.mkdir(parents=True, exist_ok=True)
    topic_model.save(str(model_dir / "bertopic_model"), serialization="safetensors", save_ctfidf=True, save_embedding_model=True)
    logger.info(f"Modelo guardado en {model_dir}")
    
    topic_info = topic_model.get_topic_info()
    logger.info(f"\nInformación de tópicos:\n{topic_info}")
    
    topic_info_path = outputs_dir / "bertopic_topic_info.csv"
    topic_info.to_csv(topic_info_path, index=False, encoding='utf-8')
    logger.info(f"Topic info guardado en {topic_info_path}")
    
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
    keywords_path = outputs_dir / "bertopic_topics_keywords.csv"
    df_topics[['topic_id', 'num_docs', 'top_words']].to_csv(keywords_path, index=False, encoding='utf-8')
    logger.info(f"Keywords guardadas en {keywords_path}")
    
    df['bertopic_topic'] = topics
    df['bertopic_prob'] = probs.max(axis=1) if hasattr(probs, 'max') else probs
    
    papers_path = outputs_dir / "bertopic_papers_topics.csv"
    df[['filename', 'author', 'title', 'doi', 'language', 'bertopic_topic', 'bertopic_prob']].to_csv(papers_path, index=False, encoding='utf-8')
    logger.info(f"Papers con tópicos guardados en {papers_path}")
    
    graph_data = []
    for t in topics_keywords:
        topic_id = t['topic_id']
        topic_papers = df[df['bertopic_topic'] == topic_id]
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
            'id': f"BT{topic_id:02d}",
            'topic_id': int(topic_id),
            'name': f"BERTopic_{topic_id}",
            'keywords': t['keywords'],
            'num_papers': len(papers_list),
            'papers': papers_list
        })
    
    graph_path = outputs_dir / "bertopic_topics_for_graph.json"
    with open(graph_path, 'w', encoding='utf-8') as f:
        json.dump(graph_data, f, ensure_ascii=False, indent=2)
    logger.info(f"JSON para grafo guardado en {graph_path}")
    
    log_memory_usage(memory_log, "end_bertopic", logger)
    logger.info("BERTopic completado")


if __name__ == "__main__":
    main()
