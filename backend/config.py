import logging

SIMILARITY_THRESHOLD = 0.55
CONFIDENCE_THRESHOLD = 0.10
MIN_RECORD_INTERVAL = 5
MAX_UPLOAD_SIZE = 50 * 1024 * 1024

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def get_logger(name):
    return logging.getLogger(name)