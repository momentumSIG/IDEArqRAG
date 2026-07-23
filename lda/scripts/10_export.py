import sys
import json
from pathlib import Path
import pandas as pd
from deep_translator import GoogleTranslator
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from utils import load_config, setup_logger, check_ram_usage, log_memory_usage


def translate_keywords_to_en(keywords_list, source_lang, translator, logger):
    lang_map = {
        'es': 'es',
        'fr': 'fr',
        'pt': 'pt',
        'ca': 'ca',
        'en': 'en'
    }
    
    source = lang_map.get(source_lang, 'auto')
    
    translated = []
    for keyword in keywords_list:
        try:
            if source == 'en':
                translated.append(keyword)
            else:
                result = translator.translate(keyword)
                translated.append(result if result else keyword)
        except Exception as e:
            logger.warning(f"Error traduciendo '{keyword}' de {source_lang}: {e}")
            translated.append(keyword)
    
    return translated


def main():
    config = load_config()
    logger = setup_logger("10_export")
    
    data_dir = Path(config['paths']['data'])
    outputs_dir = Path(config['paths']['outputs'])
    outputs_dir.mkdir(parents=True, exist_ok=True)
    
    memory_log = data_dir / "memory_usage.csv"
    log_memory_usage(memory_log, "start_10_export", logger)
    
    papers_path = outputs_dir / "papers_topics.csv"
    keywords_path = outputs_dir / "topics_keywords.csv"
    
    if not papers_path.exists() or not keywords_path.exists():
        logger.error("No se encontraron archivos de topics. Ejecuta 08_assign.py y 09_visualize.py primero.")
        sys.exit(1)
    
    df_papers = pd.read_csv(papers_path)
    df_keywords = pd.read_csv(keywords_path)
    
    logger.info("Generando JSON para grafo...")
    
    graph_data = []
    
    for _, topic_row in df_keywords.iterrows():
        topic_id = topic_row['topic_id']
        keywords = topic_row['top_words'].split(', ')
        
        topic_papers = df_papers[df_papers['topic_dominant'] == topic_id]
        
        papers_list = []
        for _, paper_row in topic_papers.iterrows():
            papers_list.append({
                'filename': paper_row['filename'],
                'title': paper_row.get('title', ''),
                'author': paper_row.get('author', ''),
                'doi': paper_row.get('doi', ''),
                'language': paper_row.get('language', ''),
                'weight': float(paper_row['topic_dominant_weight'])
            })
        
        graph_data.append({
            'id': f"T{topic_id:02d}",
            'topic_id': int(topic_id),
            'name': f"Topic_{topic_id:02d}",
            'keywords': keywords,
            'num_papers': len(papers_list),
            'papers': papers_list
        })
    
    graph_path = outputs_dir / "topics_for_graph.json"
    with open(graph_path, 'w', encoding='utf-8') as f:
        json.dump(graph_data, f, ensure_ascii=False, indent=2)
    logger.info(f"JSON para grafo guardado en {graph_path}")
    
    log_memory_usage(memory_log, "graph_json_done", logger)
    
    logger.info("Iniciando post-proceso de alineación de tópicos...")
    
    translator = GoogleTranslator(source='auto', target='en')
    
    alignment_data = []
    
    for _, topic_row in df_keywords.iterrows():
        topic_id = topic_row['topic_id']
        keywords_str = topic_row['top_words']
        keywords = keywords_str.split(', ')
        
        topic_papers = df_papers[df_papers['topic_dominant'] == topic_id]
        
        if len(topic_papers) == 0:
            continue
        
        lang_counts = topic_papers['language'].value_counts()
        dominant_lang = lang_counts.index[0] if len(lang_counts) > 0 else 'unknown'
        
        logger.info(f"Tópico {topic_id}: idioma dominante={dominant_lang}")
        
        keywords_translated = translate_keywords_to_en(keywords, dominant_lang, translator, logger)
        
        alignment_data.append({
            'topic_id': int(topic_id),
            'dominant_language': dominant_lang,
            'original_keywords': keywords_str,
            'translated_keywords_en': ', '.join(keywords_translated),
            'num_papers': len(topic_papers)
        })
    
    df_alignment = pd.DataFrame(alignment_data)
    
    if len(df_alignment) > 1:
        logger.info("Calculando similitudes entre tópicos...")
        
        tfidf = TfidfVectorizer()
        tfidf_matrix = tfidf.fit_transform(df_alignment['translated_keywords_en'])
        
        similarity_matrix = cosine_similarity(tfidf_matrix)
        
        threshold = config['alignment']['similarity_threshold']
        
        merged_groups = []
        assigned = set()
        
        for i in range(len(df_alignment)):
            if i in assigned:
                continue
            
            group = [i]
            for j in range(i + 1, len(df_alignment)):
                if j not in assigned and similarity_matrix[i, j] > threshold:
                    group.append(j)
                    assigned.add(j)
            
            merged_groups.append(group)
            assigned.add(i)
        
        group_mapping = {}
        for group_idx, group in enumerate(merged_groups):
            for topic_idx in group:
                group_mapping[topic_idx] = group_idx
        
        df_alignment['merged_group'] = df_alignment.index.map(lambda x: group_mapping.get(x, -1))
        
        logger.info(f"Grupos de fusión sugeridos: {len(merged_groups)}")
        for group_idx, group in enumerate(merged_groups):
            if len(group) > 1:
                topic_ids = [df_alignment.iloc[i]['topic_id'] for i in group]
                logger.info(f"Grupo {group_idx}: tópicos {topic_ids}")
    
    alignment_path = outputs_dir / "topics_alignment.csv"
    df_alignment.to_csv(alignment_path, index=False, encoding='utf-8')
    logger.info(f"Alineación guardada en {alignment_path}")
    
    log_memory_usage(memory_log, "end_10_export", logger)
    logger.info("Exportación y alineación completadas")


if __name__ == "__main__":
    main()
