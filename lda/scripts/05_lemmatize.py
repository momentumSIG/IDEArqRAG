import sys
from pathlib import Path
import pandas as pd
import spacy
from nltk.stem import SnowballStemmer

sys.path.insert(0, str(Path(__file__).parent))
from utils import load_config, setup_logger, check_ram_usage, log_memory_usage


def load_spacy_model(lang, config, logger):
    model_map = config['languages']['spaCy_models']
    model_name = model_map.get(lang)
    
    if model_name:
        try:
            logger.info(f"Cargando modelo spaCy: {model_name}")
            return spacy.load(model_name)
        except Exception as e:
            logger.error(f"Error cargando {model_name}: {e}")
            return None
    return None


def get_stopwords(lang):
    base_stopwords = {
        'en': {'archaeological', 'archaeology', 'study', 'research', 'analysis', 'data', 
               'results', 'paper', 'article', 'figure', 'table', 'section', 'introduction', 
               'conclusion', 'method', 'methodology', 'discussion', 'based', 'used', 'using', 
               'also', 'however', 'fig', 'pp', 'et', 'al'},
        'es': {'arqueológico', 'arqueología', 'estudio', 'investigación', 'análisis', 'datos',
               'resultados', 'artículo', 'figura', 'tabla', 'sección', 'introducción', 'conclusión',
               'método', 'metodología', 'discusión', 'basado', 'usado', 'usando', 'también',
               'embargo', 'fig', 'pp', 'et', 'al'},
        'fr': {'archéologique', 'archéologie', 'étude', 'recherche', 'analyse', 'données',
               'résultats', 'article', 'figure', 'tableau', 'section', 'introduction', 'conclusion',
               'méthode', 'méthodologie', 'discussion', 'basé', 'utilisé', 'utilisant', 'aussi',
               'cependant', 'fig', 'pp', 'et', 'al'},
        'pt': {'arqueológico', 'arqueologia', 'estudo', 'pesquisa', 'análise', 'dados',
               'resultados', 'artigo', 'figura', 'tabela', 'seção', 'introdução', 'conclusão',
               'método', 'metodologia', 'discussão', 'baseado', 'usado', 'usando', 'também',
               'entanto', 'fig', 'pp', 'et', 'al'},
        'ca': {'arqueològic', 'arqueologia', 'estudi', 'recerca', 'anàlisi', 'dades',
               'resultats', 'article', 'figura', 'taula', 'secció', 'introducció', 'conclusió',
               'mètode', 'metodologia', 'discussió', 'basat', 'usat', 'usant', 'també', 'però',
               'fig', 'pp', 'et', 'al'}
    }
    
    return base_stopwords.get(lang, set())


def lemmatize_text(text, lang, nlp, stemmer, custom_stopwords, logger):
    if not text or len(text.strip()) < 10:
        return []
    
    try:
        if nlp:
            doc = nlp(text[:1000000])
            
            tokens = []
            for token in doc:
                if (not token.is_stop and 
                    not token.is_punct and 
                    token.is_alpha and
                    len(token.text) > 2 and
                    token.lemma_.lower() not in custom_stopwords):
                    tokens.append(token.lemma_.lower())
            
            return tokens
        elif stemmer:
            words = text.lower().split()
            tokens = []
            for word in words:
                if len(word) > 2 and word.isalpha() and word not in custom_stopwords:
                    stemmed = stemmer.stem(word)
                    tokens.append(stemmed)
            return tokens
    except Exception as e:
        logger.error(f"Error lematizando: {e}")
        return []
    
    return []


def main():
    config = load_config()
    logger = setup_logger("05_lemmatize")
    
    data_dir = Path(config['paths']['data'])
    supported_langs = config['languages']['supported']
    
    memory_log = data_dir / "memory_usage.csv"
    log_memory_usage(memory_log, "start_05_lemmatize", logger)
    
    cleaned_path = data_dir / "04_cleaned.parquet"
    if not cleaned_path.exists():
        logger.error(f"No se encontró {cleaned_path}. Ejecuta 04_clean.py primero.")
        sys.exit(1)
    
    df = pd.read_parquet(cleaned_path)
    logger.info(f"Cargados {len(df)} documentos de {cleaned_path}")
    
    spacy_models = {}
    stemmers = {}
    
    for lang in supported_langs:
        nlp = load_spacy_model(lang, config, logger)
        if nlp:
            spacy_models[lang] = nlp
        else:
            logger.warning(f"No hay modelo spaCy para {lang}, usando NLTK stemmer")
            try:
                stemmers[lang] = SnowballStemmer(lang if lang != 'ca' else 'spanish')
            except Exception as e:
                logger.error(f"Error creando stemmer para {lang}: {e}")
                stemmers[lang] = None
    
    lemmatized_texts = []
    
    for i, row in df.iterrows():
        filename = row['filename']
        text = row['text_cleaned']
        lang = row['language']
        
        logger.info(f"[{i+1}/{len(df)}] Lematizando ({lang}): {filename}")
        
        custom_stopwords = get_stopwords(lang)
        
        nlp = spacy_models.get(lang)
        stemmer = stemmers.get(lang)
        
        tokens = lemmatize_text(text, lang, nlp, stemmer, custom_stopwords, logger)
        lemmatized_texts.append(tokens)
        
        if (i + 1) % 20 == 0:
            check_ram_usage(logger, config['ram']['cap_gb'], config['ram']['warning_gb'])
            log_memory_usage(memory_log, f"lemmatize_progress_{i+1}", logger)
    
    df['tokens'] = lemmatized_texts
    
    output_path = data_dir / "05_lemmatized.parquet"
    df.to_parquet(output_path, engine='pyarrow', compression='snappy')
    logger.info(f"Guardado en {output_path}")
    
    log_memory_usage(memory_log, "end_05_lemmatize", logger)
    
    avg_tokens = df['tokens'].apply(len).mean()
    logger.info(f"Tokens promedio por documento: {avg_tokens:.0f}")


if __name__ == "__main__":
    main()
