import threading
from langchain_core.embeddings import Embeddings


class ChromaDefaultEmbeddings(Embeddings):
    def __init__(self):
        from chromadb.utils import embedding_functions
        self._ef = embedding_functions.DefaultEmbeddingFunction()

    def embed_documents(self, texts):
        if not texts:
            return []
        result = self._ef(texts)
        return [[float(v) for v in row] for row in result]

    def embed_query(self, text):
        result = self._ef([text])
        return [float(v) for v in result[0]]


_instance = None
_lock = threading.Lock()


def get_embeddings():
    global _instance
    if _instance is None:
        with _lock:
            if _instance is None:
                _instance = ChromaDefaultEmbeddings()
    return _instance
