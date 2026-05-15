from langchain_core.embeddings import Embeddings


class ChromaDefaultEmbeddings(Embeddings):
    def __init__(self):
        from chromadb.utils import embedding_functions
        self._ef = embedding_functions.DefaultEmbeddingFunction()

    def embed_documents(self, texts):
        result = self._ef(texts)
        return [[float(v) for v in row] for row in result]

    def embed_query(self, text):
        result = self._ef([text])
        return [float(v) for v in result[0]]
