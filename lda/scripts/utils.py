import yaml
import logging
import psutil
import time
from pathlib import Path
from datetime import datetime


def load_config(config_path=None):
    if config_path is None:
        config_path = Path(__file__).parent.parent / "config.yaml"
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def setup_logger(script_name, log_dir=None):
    if log_dir is None:
        config = load_config()
        log_dir = Path(config['paths']['logs'])
    else:
        log_dir = Path(log_dir)
    
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{script_name}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(script_name)


def check_ram_usage(logger=None, cap_gb=16, warning_gb=14):
    ram = psutil.virtual_memory()
    used_gb = ram.used / (1024**3)
    percent = ram.percent
    
    if used_gb > cap_gb:
        msg = f"RAM crítica: {used_gb:.1f}GB ({percent}%) - supera límite {cap_gb}GB"
        if logger:
            logger.error(msg)
        raise MemoryError(msg)
    elif used_gb > warning_gb:
        msg = f"RAM alta: {used_gb:.1f}GB ({percent}%) - supera advertencia {warning_gb}GB"
        if logger:
            logger.warning(msg)
    
    return used_gb


def log_memory_usage(log_file, phase, logger=None):
    ram = psutil.virtual_memory()
    used_gb = ram.used / (1024**3)
    percent = ram.percent
    
    log_file = Path(log_file)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(f"{datetime.now().isoformat()},{phase},{used_gb:.2f},{percent}\n")
    
    if logger:
        logger.info(f"RAM: {used_gb:.1f}GB ({percent}%) en fase: {phase}")
    
    return used_gb


def get_pdf_files(ingesta_dir):
    ingesta_path = Path(ingesta_dir)
    pdf_files = sorted([
        f for f in ingesta_path.glob("*.pdf")
        if not f.name.endswith(":Zone.Identifier")
    ])
    return pdf_files


def extract_doi(text):
    import re
    doi_pattern = r'\b(10\.\d{4,9}/[-._;()/:A-Z0-9]+)\b'
    matches = re.findall(doi_pattern, text, re.IGNORECASE)
    if matches:
        doi = matches[0].rstrip('.')
        return doi
    return None


def detect_language(text, sample_size=2000):
    from langdetect import detect, LangDetectException
    sample = text[:sample_size]
    try:
        lang = detect(sample)
        return lang
    except LangDetectException:
        return "unknown"
