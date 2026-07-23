import sys
import re
from pathlib import Path
import pandas as pd
import fitz

sys.path.insert(0, str(Path(__file__).parent))
from utils import load_config, setup_logger, check_ram_usage, log_memory_usage, extract_doi


def extract_metadata_from_pdf(pdf_path, text, logger):
    metadata = {
        'filename': pdf_path.name,
        'author': None,
        'title': None,
        'doi': None
    }
    
    try:
        doc = fitz.open(pdf_path)
        pdf_meta = doc.metadata
        doc.close()
        
        if pdf_meta.get('author'):
            metadata['author'] = pdf_meta['author'].strip()
        
        if pdf_meta.get('title'):
            metadata['title'] = pdf_meta['title'].strip()
        
        metadata['doi'] = extract_doi(text)
        
        if not metadata['title']:
            metadata['title'] = extract_title_from_text(text)
        
        if not metadata['author']:
            metadata['author'] = extract_author_from_text(text)
        
    except Exception as e:
        logger.error(f"Error extrayendo metadatos de {pdf_path.name}: {e}")
    
    return metadata


def extract_title_from_text(text):
    lines = text.split('\n')
    
    for line in lines[:10]:
        line = line.strip()
        if not line:
            continue
        
        if len(line) > 20 and len(line) < 300:
            if not re.match(r'^(abstract|resumen|résumé|resum|zusammenfassung)', line.lower()):
                if not re.match(r'^(doi|http|www)', line.lower()):
                    return line
    
    return None


def extract_author_from_text(text):
    lines = text.split('\n')
    
    for i, line in enumerate(lines[:15]):
        line = line.strip()
        
        if re.search(r'\b(author|autor|auteure?|autore|autors?)\b', line, re.IGNORECASE):
            if ':' in line:
                parts = line.split(':', 1)
                if len(parts) > 1:
                    author = parts[1].strip()
                    if len(author) > 3 and len(author) < 200:
                        return author
            elif i + 1 < len(lines):
                next_line = lines[i + 1].strip()
                if len(next_line) > 3 and len(next_line) < 200:
                    return next_line
    
    return None


def main():
    config = load_config()
    logger = setup_logger("02_extract_meta")
    
    data_dir = Path(config['paths']['data'])
    ingesta_dir = Path(config['paths']['ingesta'])
    
    memory_log = data_dir / "memory_usage.csv"
    log_memory_usage(memory_log, "start_02_extract_meta", logger)
    
    parsed_path = data_dir / "01_parsed.parquet"
    if not parsed_path.exists():
        logger.error(f"No se encontró {parsed_path}. Ejecuta 01_parse.py primero.")
        sys.exit(1)
    
    df = pd.read_parquet(parsed_path)
    logger.info(f"Cargados {len(df)} documentos de {parsed_path}")
    
    results = []
    
    for i, row in df.iterrows():
        filename = row['filename']
        text = row['text']
        
        pdf_path = ingesta_dir / filename
        if not pdf_path.exists():
            logger.warning(f"PDF no encontrado: {filename}")
            results.append({
                'filename': filename,
                'author': None,
                'title': None,
                'doi': None
            })
            continue
        
        logger.info(f"[{i+1}/{len(df)}] Extrayendo metadatos: {filename}")
        
        meta = extract_metadata_from_pdf(pdf_path, text, logger)
        results.append(meta)
        
        if (i + 1) % 50 == 0:
            check_ram_usage(logger, config['ram']['cap_gb'], config['ram']['warning_gb'])
            log_memory_usage(memory_log, f"extract_meta_progress_{i+1}", logger)
    
    df_meta = pd.DataFrame(results)
    
    df_merged = df.merge(df_meta, on='filename', how='left')
    
    output_path = data_dir / "02_meta.parquet"
    df_merged.to_parquet(output_path, engine='pyarrow', compression='snappy')
    logger.info(f"Guardado en {output_path}")
    
    log_memory_usage(memory_log, "end_02_extract_meta", logger)
    
    stats = {
        'con_autor': df_meta['author'].notna().sum(),
        'con_titulo': df_meta['title'].notna().sum(),
        'con_doi': df_meta['doi'].notna().sum()
    }
    logger.info(f"Estadísticas: {stats}")


if __name__ == "__main__":
    main()
