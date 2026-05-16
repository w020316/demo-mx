import os
import sys
import threading
import warnings
import logging

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "pylibs"))

from langchain_community.vectorstores import Chroma
from config import CHROMA_DIR
from embeddings import get_embeddings

warnings.filterwarnings("ignore", category=DeprecationWarning, module="langchain_community")

logger = logging.getLogger(__name__)

_instance = None
_lock = threading.Lock()


def get_vectorstore():
    global _instance
    if _instance is None:
        with _lock:
            if _instance is None:
                _instance = Chroma(
                    persist_directory=CHROMA_DIR,
                    embedding_function=get_embeddings(),
                )
    return _instance


def reset_vectorstore():
    global _instance
    with _lock:
        _instance = None


def get_vectorstore_info():
    try:
        vs = get_vectorstore()
        count = vs._collection.count()
        metadatas = vs._collection.get(include=["metadatas"])["metadatas"]
        files = set()
        for m in metadatas:
            if m and "source_file" in m:
                files.add(m["source_file"])
        return {"count": count, "files": sorted(files)}
    except Exception as e:
        logger.warning(f"获取向量库信息失败: {e}")
        return {"count": 0, "files": []}


def get_retriever(k=3, search_type="similarity", fetch_k=None, lambda_mult=0.5):
    vectorstore = get_vectorstore()
    if search_type == "mmr":
        return vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": k,
                "fetch_k": fetch_k or k * 5,
                "lambda_mult": lambda_mult,
            },
        )
    return vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k},
    )


def search_with_scores(question, k=3, search_type="similarity", fetch_k=None, lambda_mult=0.5):
    vectorstore = get_vectorstore()
    try:
        if search_type == "mmr":
            raw_results = vectorstore.max_marginal_relevance_search_with_score(
                question, k=k, fetch_k=fetch_k or k * 5, lambda_mult=lambda_mult,
            )
        else:
            raw_results = vectorstore.similarity_search_with_score(question, k=k)
        docs = []
        scores = []
        for doc, distance in raw_results:
            docs.append(doc)
            similarity = 1.0 / (1.0 + distance)
            scores.append(similarity)
        return docs, scores
    except Exception as e:
        logger.warning(f"向量检索失败: {e}")
        return [], []


def add_documents_to_vectorstore(chunks, persist_directory=None):
    if persist_directory is None:
        persist_directory = CHROMA_DIR
    embeddings = get_embeddings()
    vectorstore = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings,
    )
    vectorstore.add_documents(chunks)
    count = vectorstore._collection.count()
    reset_vectorstore()
    return count


def create_vectorstore_from_documents(chunks, persist_directory=None):
    if persist_directory is None:
        persist_directory = CHROMA_DIR
    embeddings = get_embeddings()
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_directory,
    )
    count = vectorstore._collection.count()
    reset_vectorstore()
    return count
