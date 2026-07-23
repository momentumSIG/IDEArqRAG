import sys
from pathlib import Path
import pandas as pd
import pymupdf4llm
import fitz
import spacy
from nltk.stem import SnowballStemmer
from gensim import corpora
from gensim.models import LdaModel, CoherenceModel
from langdetect import detect, LangDetectException
import re

sys.path.insert(0, str(Path(__file__).parent))
from utils import load_config, setup_logger


def clean_text(text):
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'\b(?:References|Bibliography|Bibliografía|Bibliographie|Bibliografia|Referències)\b.*$', 
                  '', text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r'\|.*?\|', ' ', text)
    text = re.sub(r'\-{3,}', ' ', text)
    text = re.sub(r'\={3,}', ' ', text)
    text = re.sub(r'\b\d{4,}\b', '', text)
    text = re.sub(r'\b\d+\.\d+\b', '', text)
    text = re.sub(r'\b\d{1,3}\b', '', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def extract_doi(text):
    doi_pattern = r'\b(10\.\d{4,9}/[-._;()/:A-Z0-9]+)\b'
    matches = re.findall(doi_pattern, text, re.IGNORECASE)
    if matches:
        return matches[0].rstrip('.')
    return None


def detect_language(text, sample_size=2000):
    sample = text[:sample_size]
    try:
        return detect(sample)
    except LangDetectException:
        return "unknown"


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


def main():
    config = load_config()
    logger = setup_logger("smoke_test")
    
    ingesta_dir = Path(config['paths']['ingesta'])
    data_dir = Path(config['paths']['data'])
    data_dir.mkdir(parents=True, exist_ok=True)
    
    pdf_files = sorted(ingesta_dir.glob("*.pdf"))[:5]
    logger.info(f"Smoke test con {len(pdf_files)} PDFs")
    
    results = []
    
    for i, pdf_path in enumerate(pdf_files, 1):
        logger.info(f"\n[{i}/{len(pdf_files)}] {pdf_path.name}")
        
        try:
            text = pymupdf4llm.to_markdown(pdf_path)
            logger.info(f"  Parseado: {len(text)} chars")
        except Exception as e:
            logger.error(f"  Error parsing: {e}")
            continue
        
        try:
            doc = fitz.open(pdf_path)
            pdf_meta = doc.metadata
            doc.close()
            
            title = pdf_meta.get('title', '')
            author = pdf_meta.get('author', '')
            doi = extract_doi(text)
            
            logger.info(f"  Título: {title[:50] if title else 'N/A'}...")
            logger.info(f"  Autor: {author[:50] if author else 'N/A'}...")
            logger.info(f"  DOI: {doi}")
        except Exception as e:
            logger.error(f"  Error metadatos: {e}")
            title, author, doi = '', '', None
        
        lang = detect_language(text)
        logger.info(f"  Idioma detectado: {lang}")
        
        cleaned = clean_text(text)
        logger.info(f"  Texto limpio: {len(cleaned)} chars")
        
        try:
            model_name = config['languages']['spaCy_models'].get(lang)
            if model_name:
                nlp = spacy.load(model_name)
                custom_stopwords = get_stopwords(lang)
                doc = nlp(cleaned[:100000])
                tokens = [
                    token.lemma_.lower()
                    for token in doc
                    if not token.is_stop
                    and not token.is_punct
                    and token.is_alpha
                    and len(token.text) > 2
                    and token.lemma_.lower() not in custom_stopwords
                ]
            else:
                stemmer = SnowballStemmer('spanish')
                words = cleaned.lower().split()
                tokens = [stemmer.stem(w) for w in words if len(w) > 2 and w.isalpha()]
            
            logger.info(f"  Tokens: {len(tokens)}")
        except Exception as e:
            logger.error(f"  Error lematización: {e}")
            tokens = []
        
        results.append({
            'filename': pdf_path.name,
            'title': title,
            'author': author,
            'doi': doi,
            'language': lang,
            'text_length': len(text),
            'tokens_count': len(tokens),
            'tokens': tokens
        })
    
    logger.info("\n" + "="*60)
    logger.info("RESULTADOS SMOKE TEST")
    logger.info("="*60)
    
    for r in results:
        logger.info(f"\n{r['filename']}:")
        logger.info(f"  Idioma: {r['language']}")
        logger.info(f"  Tokens: {r['tokens_count']}")
        if r['tokens']:
            logger.info(f"  Sample tokens: {r['tokens'][:10]}")
    
    df = pd.DataFrame([{
        'filename': r['filename'],
        'title': r['title'],
        'author': r['author'],
        'doi': r['doi'],
        'language': r['language'],
        'tokens': r['tokens']
    } for r in results])
    
    df.to_parquet(data_dir / "smoke_test.parquet")
    logger.info(f"\nResultados guardados en {data_dir / 'smoke_test.parquet'}")
    
    logger.info("\nSmoke test completado exitosamente")


if __name__ == "__main__":
    main()
