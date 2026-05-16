import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


class TestEmbeddings:
    def test_chroma_default_embeddings_instantiation(self):
        from embeddings import ChromaDefaultEmbeddings
        emb = ChromaDefaultEmbeddings()
        assert emb is not None

    def test_embed_query(self):
        from embeddings import ChromaDefaultEmbeddings
        emb = ChromaDefaultEmbeddings()
        result = emb.embed_query("test query")
        assert isinstance(result, list)
        assert len(result) > 0
        assert all(isinstance(v, float) for v in result)

    def test_embed_documents(self):
        from embeddings import ChromaDefaultEmbeddings
        emb = ChromaDefaultEmbeddings()
        result = emb.embed_documents(["test doc 1", "test doc 2"])
        assert isinstance(result, list)
        assert len(result) == 2
        assert all(len(vec) > 0 for vec in result)

    def test_embed_documents_empty(self):
        from embeddings import ChromaDefaultEmbeddings
        emb = ChromaDefaultEmbeddings()
        result = emb.embed_documents([])
        assert isinstance(result, list)
        assert len(result) == 0

    def test_embed_query_returns_floats(self):
        from embeddings import ChromaDefaultEmbeddings
        emb = ChromaDefaultEmbeddings()
        result = emb.embed_query("hello world")
        for v in result:
            assert isinstance(v, float)
