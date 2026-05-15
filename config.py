import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "pylibs"))

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "docs")
CHROMA_DIR = os.path.join(PROJECT_ROOT, "chroma_db")
CHROMA_CACHE_DIR = os.path.join(PROJECT_ROOT, ".cache", "chroma")

os.environ.setdefault("CHROMA_CACHE_DIR", CHROMA_CACHE_DIR)
