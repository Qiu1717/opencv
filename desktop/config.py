import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE_DIR = PROJECT_ROOT

MODEL_DIR = os.path.join(PROJECT_ROOT, 'models')
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
LOG_DIR = os.path.join(BASE_DIR, 'logs')

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

FACE_DETECT_MODEL = os.path.join(MODEL_DIR, 'res10_300x300_ssd_iter_140000_fp16.caffemodel')
FACE_DETECT_PROTO = os.path.join(MODEL_DIR, 'deploy.prototxt.txt')

RECognition_MODEL = os.path.join(MODEL_DIR, 'face_recognition.onnx')

CONFIDENCE_THRESHOLD = 0.5
SIMILARITY_THRESHOLD = 0.6
FPS_TARGET = 15

DB_PATH = os.path.join(DATA_DIR, 'face_db.sqlite')
FEATURE_DB_PATH = os.path.join(DATA_DIR, 'features.npy')