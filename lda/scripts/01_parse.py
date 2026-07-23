import sys
from pathlib import Path
import pandas as pd
import pymupdf4llm
import pickle

sys.path.insert(0, str(Path(__file__).parent))
from utils import load_config, setup_logger, check_ram_usage, log_memory_usage, get_pdf_files


def parse_pdf(pdf_path, logger):
    try:
        md_text = pymupdf4llm.to_markdown(pdf_path)
        return md_text
    except Exception as e:
        logger.error(f"Error parsing {pdf_path.name}: {e}")
        return None


def main():
    config = load_config()
    logger = setup_logger("01_parse")
    
    ingesta_dir = config['paths']['ingesta']
    data_dir = Path(config['paths']['data'])
    data_dir.mkdir(parents=True, exist_ok=True)
    
    memory_log = data_dir / "memory_usage.csv"
    if not memory_log.exists():
        with open(memory_log, 'w', encoding='utf-8') as f:
            f.write("timestamp,phase,ram_gb,ram_percent\n")
    
    log_memory_usage(memory_log, "start_01_parse", logger)
    
    pdf_files = get_pdf_files(ingesta_dir)
    logger.info(f"Encontrados {len(pdf_files)} PDFs en {ingesta_dir}")
    
    checkpoint_path = data_dir / "01_parse_checkpoint.pkl"
    if checkpoint_path.exists():
        with open(checkpoint_path, 'rb') as f:
            checkpoint = pickle.load(f)
        results = checkpoint['results']
        failed = checkpoint['failed']
        processed_files = set(r['filename'] for r in results) | set(f['filename'] for f in failed)
        logger.info(f"Retomando desde checkpoint: {len(processed_files)} archivos ya procesados")
    else:
        results = []
        failed = []
        processed_files = set()
    
    remaining_files = [f for f in pdf_files if f.name not in processed_files]
    logger.info(f"Restan {len(remaining_files)} PDFs por procesar")
    
    for i, pdf_path in enumerate(remaining_files, 1):
        logger.info(f"[{len(processed_files) + i}/{len(pdf_files)}] Procesando: {pdf_path.name}")
        
        text = parse_pdf(pdf_path, logger)
        
        if text and len(text.strip()) >= config['parsing']['min_text_length']:
            results.append({
                'filename': pdf_path.name,
                'text': text,
                'text_length': len(text)
            })
        else:
            failed.append({
                'filename': pdf_path.name,
                'reason': 'texto_insuficiente' if text else 'error_parsing'
            })
            logger.warning(f"Descartado {pdf_path.name}: texto insuficiente o error")
        
        if i % 10 == 0:
            check_ram_usage(logger, config['ram']['cap_gb'], config['ram']['warning_gb'])
            log_memory_usage(memory_log, f"parse_progress_{len(processed_files) + i}", logger)
        
        if i % 50 == 0:
            with open(checkpoint_path, 'wb') as f:
                pickle.dump({'results': results, 'failed': failed}, f)
            logger.info(f"Checkpoint guardado en {i}/{len(remaining_files)}")
    
    df = pd.DataFrame(results)
    output_path = data_dir / "01_parsed.parquet"
    df.to_parquet(output_path, engine='pyarrow', compression='snappy')
    logger.info(f"Guardado {len(df)} documentos en {output_path}")
    
    if failed:
        df_failed = pd.DataFrame(failed)
        failed_path = data_dir / "01_failed.csv"
        df_failed.to_csv(failed_path, index=False, encoding='utf-8')
        logger.warning(f"Guardados {len(failed)} documentos fallidos en {failed_path}")
    
    if checkpoint_path.exists():
        checkpoint_path.unlink()
        logger.info("Checkpoint eliminado")
    
    log_memory_usage(memory_log, "end_01_parse", logger)
    logger.info(f"Completado: {len(results)} exitosos, {len(failed)} fallidos")


if __name__ == "__main__":
    main()
