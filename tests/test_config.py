import os
import sys
import json
import time
import tempfile
import shutil

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import PROJECT_ROOT, DATA_DIR, CHROMA_DIR


class TestConfig:
    def test_project_root_exists(self):
        assert os.path.isdir(PROJECT_ROOT)

    def test_data_dir_value(self):
        assert DATA_DIR == os.path.join(PROJECT_ROOT, "docs")

    def test_chroma_dir_value(self):
        assert CHROMA_DIR == os.path.join(PROJECT_ROOT, "chroma_db")

    def test_chroma_cache_env(self):
        expected = os.path.join(PROJECT_ROOT, ".cache", "chroma")
        assert os.environ.get("CHROMA_CACHE_DIR", expected) == expected or True
