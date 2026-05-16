import os
import sys
import tempfile
import shutil

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


class TestIngestCleanText:
    def test_clean_multiple_spaces(self):
        from ingest import clean_text
        assert clean_text("hello    world") == "hello world"

    def test_clean_tabs(self):
        from ingest import clean_text
        assert clean_text("hello\t\tworld") == "hello world"

    def test_clean_multiple_newlines(self):
        from ingest import clean_text
        assert clean_text("hello\n\n\n\nworld") == "hello\n\nworld"

    def test_clean_trailing_whitespace(self):
        from ingest import clean_text
        assert clean_text("  hello  ") == "hello"

    def test_clean_empty_string(self):
        from ingest import clean_text
        assert clean_text("") == ""

    def test_clean_only_whitespace(self):
        from ingest import clean_text
        assert clean_text("   \n\n   ") == ""

    def test_clean_mixed_whitespace(self):
        from ingest import clean_text
        result = clean_text("  hello  \n\n\n  world  ")
        assert "hello" in result
        assert "world" in result


class TestIngestDetectEncoding:
    def test_utf8_file(self, tmp_path):
        from ingest import _detect_encoding
        f = tmp_path / "test.txt"
        f.write_text("hello world", encoding="utf-8")
        assert _detect_encoding(str(f)) == "utf-8"

    def test_gbk_file(self, tmp_path):
        from ingest import _detect_encoding
        f = tmp_path / "test_gbk.txt"
        f.write_text("你好世界", encoding="gbk")
        result = _detect_encoding(str(f))
        assert result in ("gbk", "gb18030", "utf-8")

    def test_nonexistent_file(self):
        from ingest import _detect_encoding
        result = _detect_encoding("/nonexistent/file.txt")
        assert result == "utf-8"

    def test_empty_file(self, tmp_path):
        from ingest import _detect_encoding
        f = tmp_path / "empty.txt"
        f.write_bytes(b"")
        result = _detect_encoding(str(f))
        assert isinstance(result, str)


class TestIngestComputeFileHash:
    def test_hash_consistency(self, tmp_path):
        from ingest import compute_file_hash
        f = tmp_path / "test.txt"
        f.write_text("hello world", encoding="utf-8")
        h1 = compute_file_hash(str(f))
        h2 = compute_file_hash(str(f))
        assert h1 == h2

    def test_hash_different_content(self, tmp_path):
        from ingest import compute_file_hash
        f1 = tmp_path / "test1.txt"
        f2 = tmp_path / "test2.txt"
        f1.write_text("hello", encoding="utf-8")
        f2.write_text("world", encoding="utf-8")
        h1 = compute_file_hash(str(f1))
        h2 = compute_file_hash(str(f2))
        assert h1 != h2

    def test_hash_sha256_length(self, tmp_path):
        from ingest import compute_file_hash
        f = tmp_path / "test.txt"
        f.write_text("hello", encoding="utf-8")
        h = compute_file_hash(str(f))
        assert len(h) == 64

    def test_hash_nonexistent_file(self):
        from ingest import compute_file_hash
        h = compute_file_hash("/nonexistent/file.txt")
        assert h == ""


class TestIngestHashPersistence:
    def test_save_and_load_hashes(self, tmp_path):
        from ingest import save_hashes, get_existing_hashes
        hashes = {"file1.pdf": "abc123", "file2.txt": "def456"}
        save_hashes(str(tmp_path), hashes)
        loaded = get_existing_hashes(str(tmp_path))
        assert loaded == hashes

    def test_load_nonexistent_hashes(self, tmp_path):
        from ingest import get_existing_hashes
        hashes = get_existing_hashes(str(tmp_path / "nonexistent"))
        assert hashes == {}


class TestIngestSplitDocuments:
    def test_split_empty_list(self):
        from ingest import split_documents
        result = split_documents([])
        assert result == []

    def test_split_custom_params(self):
        from unittest.mock import MagicMock
        from ingest import split_documents
        doc = MagicMock()
        doc.page_content = "这是一段测试文本。" * 100
        doc.metadata = {"source_file": "test.txt"}
        result = split_documents([doc], chunk_size=200, chunk_overlap=20)
        assert len(result) > 0


class TestIngestLoadDocuments:
    def test_load_from_empty_dir(self, tmp_path):
        from ingest import load_documents
        docs = load_documents(str(tmp_path))
        assert docs == []

    def test_load_from_nonexistent_dir(self, tmp_path):
        from ingest import load_documents
        docs = load_documents(str(tmp_path / "nonexistent"))
        assert docs == []

    def test_load_txt_file(self, tmp_path):
        from ingest import load_documents
        f = tmp_path / "test.txt"
        f.write_text("This is a test document.", encoding="utf-8")
        docs = load_documents(str(tmp_path))
        assert len(docs) >= 1
        assert docs[0].metadata["file_type"] == "txt"
        assert docs[0].metadata["source_file"] == "test.txt"

    def test_load_md_file(self, tmp_path):
        from ingest import load_documents
        f = tmp_path / "test.md"
        f.write_text("# Test Markdown\n\nSome content.", encoding="utf-8")
        docs = load_documents(str(tmp_path))
        assert len(docs) >= 1
        assert docs[0].metadata["file_type"] == "md"

    def test_load_skips_non_document_files(self, tmp_path):
        from ingest import load_documents
        f = tmp_path / "test.exe"
        f.write_bytes(b"\x00\x01\x02")
        docs = load_documents(str(tmp_path))
        assert len(docs) == 0

    def test_load_skips_oversized_file(self, tmp_path):
        from ingest import load_documents, MAX_FILE_SIZE
        f = tmp_path / "big.txt"
        f.write_text("x" * 100, encoding="utf-8")
        import ingest
        original_max = ingest.MAX_FILE_SIZE
        ingest.MAX_FILE_SIZE = 50
        try:
            docs = ingest.load_documents(str(tmp_path))
            assert len(docs) == 0
        finally:
            ingest.MAX_FILE_SIZE = original_max
