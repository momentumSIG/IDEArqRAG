import sys
import re
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from utils import load_config, setup_logger, check_ram_usage, log_memory_usage


def clean_text(text, logger=None):
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
    
    text = text.strip()
    
    return text


def main():
    config = load_config()
    logger = setup_logger("04_clean")
    
    data_dir = Path(config['paths']['data'])
    
    memory_log = data_dir / "memory_usage.csv"
    log_memory_usage(memory_log, "start_04_clean", logger)
    
    lang_path = data_dir / "03_lang.parquet"
    if not lang_path.exists():
        logger.error(f"No se encontró {lang_path}. Ejecuta 03_detect_lang.py primero.")
        sys.exit(1)
    
    df = pd.read_parquet(lang_path)
    logger.info(f"Cargados {len(df)} documentos de {lang_path}")
    
    cleaned_texts = []
    
    for i, row in df.iterrows():
        filename = row['filename']
        text = row['text']
        
        logger.info(f"[{i+1}/{len(df)}] Limpiando: {filename}")
        
        cleaned = clean_text(text, logger)
        cleaned_texts.append(cleaned)
        
        if (i + 1) % 50 == 0:
            check_ram_usage(logger, config['ram']['cap_gb'], config['ram']['warning_gb'])
            log_memory_usage(memory_log, f"clean_progress_{i+1}", logger)
    
    df['text_cleaned'] = cleaned_texts
    
    output_path = data_dir / "04_cleaned.parquet"
    df.to_parquet(output_path, engine='pyarrow', compression='snappy')
    logger.info(f"Guardado en {output_path}")
    
    log_memory_usage(memory_log, "end_04_clean", logger)
    
    avg_length = df['text_cleaned'].str.len().mean()
    logger.info(f"Longitud promedio después de limpieza: {avg_length:.0f} caracteres")


if __name__ == "__main__":
    main()
