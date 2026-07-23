import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from utils import load_config, setup_logger, check_ram_usage, log_memory_usage, detect_language


def main():
    config = load_config()
    logger = setup_logger("03_detect_lang")
    
    data_dir = Path(config['paths']['data'])
    supported_langs = config['languages']['supported']
    
    memory_log = data_dir / "memory_usage.csv"
    log_memory_usage(memory_log, "start_03_detect_lang", logger)
    
    meta_path = data_dir / "02_meta.parquet"
    if not meta_path.exists():
        logger.error(f"No se encontró {meta_path}. Ejecuta 02_extract_meta.py primero.")
        sys.exit(1)
    
    df = pd.read_parquet(meta_path)
    logger.info(f"Cargados {len(df)} documentos de {meta_path}")
    
    languages = []
    
    for i, row in df.iterrows():
        filename = row['filename']
        text = row['text']
        
        logger.info(f"[{i+1}/{len(df)}] Detectando idioma: {filename}")
        
        lang = detect_language(text)
        
        if lang not in supported_langs:
            logger.warning(f"Idioma '{lang}' no soportado para {filename}, marcando como 'unknown'")
            lang = 'unknown'
        
        languages.append(lang)
        
        if (i + 1) % 50 == 0:
            check_ram_usage(logger, config['ram']['cap_gb'], config['ram']['warning_gb'])
            log_memory_usage(memory_log, f"detect_lang_progress_{i+1}", logger)
    
    df['language'] = languages
    
    output_path = data_dir / "03_lang.parquet"
    df.to_parquet(output_path, engine='pyarrow', compression='snappy')
    logger.info(f"Guardado en {output_path}")
    
    log_memory_usage(memory_log, "end_03_detect_lang", logger)
    
    lang_counts = df['language'].value_counts()
    logger.info(f"Distribución de idiomas:\n{lang_counts}")


if __name__ == "__main__":
    main()
