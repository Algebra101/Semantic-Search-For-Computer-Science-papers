from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent  # project root
DATA_DIR = BASE_DIR / "data"                # root/data/ 
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"

RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

RAW_ARXIV_FILE = os.getenv("RAW_ARXIV_FILE", "data/raw/arxiv-metadata-oai-snapshot.json")
DATA_FILE = Path(os.getenv("DATA_FILE", str(PROCESSED_DIR / "projects.csv")))

TOP_K = int(os.getenv("TOP_K"))
SEMANTIC_WEIGHT = float(os.getenv("SEMANTIC_WEIGHT"))
LEXICAL_WEIGHT = float(os.getenv("LEXICAL_WEIGHT"))
VECTOR_SIZE = int(os.getenv("VECTOR_SIZE"))
WINDOW = int(os.getenv("WINDOW"))
MIN_COUNT = int(os.getenv("MIN_COUNT"))
EPOCHS = int(os.getenv("EPOCHS"))
MAX_RECORDS = int(os.getenv("MAX_RECORDS"))

W2V_MODEL_PATH = MODELS_DIR / "word2vec.model"
TFIDF_VECTORIZER_PATH = MODELS_DIR / "tfidf_vectorizer.pkl"
PROCESSED_DATA_PATH = PROCESSED_DIR / "processed_projects.csv"