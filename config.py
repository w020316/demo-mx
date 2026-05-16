import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "pylibs"))

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "docs")
CHROMA_DIR = os.path.join(PROJECT_ROOT, "chroma_db")
CHROMA_CACHE_DIR = os.path.join(PROJECT_ROOT, ".cache", "chroma")
FEEDBACK_DIR = os.path.join(PROJECT_ROOT, "feedback")

MAX_FILE_SIZE = 100 * 1024 * 1024

DEFAULT_CHUNK_SIZE = 500
DEFAULT_CHUNK_OVERLAP = 50
DEFAULT_K = 3
DEFAULT_SEARCH_TYPE = "similarity"
DEFAULT_PROMPT_MODE = "anti_hallucination"

CHAT_FALLBACK_THRESHOLD = 0.52
CHAT_DOMAIN_THRESHOLD = 0.55

RATE_LIMIT_SECONDS = 60
RATE_LIMIT_PER_SESSION = 20
MAX_QUESTION_LENGTH = 500
MAX_ANSWER_LENGTH = 1000
MAX_RATINGS_ENTRIES = 10000
DEDUP_THRESHOLD = 0.65
WILSON_Z = 1.959964

os.environ.setdefault("CHROMA_CACHE_DIR", CHROMA_CACHE_DIR)
