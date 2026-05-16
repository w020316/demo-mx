import os
import sys
import re
import argparse
import hashlib
import logging

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "pylibs"))

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from config import (
    DATA_DIR, CHROMA_DIR, PROJECT_ROOT,
    MAX_FILE_SIZE, DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP,
)
from embeddings import get_embeddings
from vectorstore import add_documents_to_vectorstore, create_vectorstore_from_documents
from llm import verify_api
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

load_dotenv(os.path.join(PROJECT_ROOT, ".env"))


def clean_text(text):
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def _detect_encoding(filepath):
    encodings = ["utf-8", "gbk", "gb2312", "gb18030", "latin-1"]
    try:
        file_size = os.path.getsize(filepath)
        read_size = min(8192, file_size) if file_size > 0 else 8192
        with open(filepath, "rb") as f:
            raw = f.read(read_size)
    except OSError as e:
        logger.warning(f"读取文件失败: {filepath}, 错误: {e}")
        return "utf-8"
    for enc in encodings:
        try:
            raw.decode(enc)
            return enc
        except (UnicodeDecodeError, UnicodeError):
            continue
    return "utf-8"


def load_documents(data_dir=None):
    if data_dir is None:
        data_dir = DATA_DIR
    documents = []
    if not os.path.exists(data_dir):
        print(f"数据目录不存在: {data_dir}")
        os.makedirs(data_dir, exist_ok=True)
        print(f"已创建数据目录: {data_dir}，请放入 PDF/TXT/MD 文件后重新运行")
        return documents

    for filename in sorted(os.listdir(data_dir)):
        filepath = os.path.join(data_dir, filename)
        if not os.path.isfile(filepath):
            continue
        try:
            file_size = os.path.getsize(filepath)
            if file_size > MAX_FILE_SIZE:
                print(f"  跳过过大文件: {filename} ({file_size / 1024 / 1024:.1f}MB)")
                continue
        except OSError:
            continue
        ext = filename.lower().rsplit('.', 1)[-1]
        try:
            if ext == "pdf":
                print(f"  加载PDF: {filename}")
                loader = PyPDFLoader(filepath)
                docs = loader.load()
                for i, doc in enumerate(docs):
                    doc.page_content = clean_text(doc.page_content)
                    doc.metadata["source_file"] = filename
                    doc.metadata["page"] = i
                    doc.metadata["file_type"] = "pdf"
                documents.extend(docs)
                print(f"    -> {len(docs)} 页")
            elif ext == "txt":
                txt_encoding = _detect_encoding(filepath)
                print(f"  加载TXT: {filename} (编码: {txt_encoding})")
                loader = TextLoader(filepath, encoding=txt_encoding)
                docs = loader.load()
                for doc in docs:
                    doc.page_content = clean_text(doc.page_content)
                    doc.metadata["source_file"] = filename
                    doc.metadata["page"] = 0
                    doc.metadata["file_type"] = "txt"
                    doc.metadata["encoding"] = txt_encoding
                documents.extend(docs)
                print(f"    -> {len(docs)} 个文档")
            elif ext == "md":
                md_encoding = _detect_encoding(filepath)
                print(f"  加载MD: {filename} (编码: {md_encoding})")
                with open(filepath, "r", encoding=md_encoding) as f:
                    content = f.read()
                content = clean_text(content)
                doc = Document(
                    page_content=content,
                    metadata={"source_file": filename, "page": 0, "file_type": "md", "encoding": md_encoding},
                )
                documents.append(doc)
                print(f"    -> 1 个文档")
        except Exception as e:
            print(f"    -> 加载失败: {e}")

    print(f"共加载 {len(documents)} 个文档片段")
    return documents


def compute_file_hash(filepath):
    h = hashlib.sha256()
    try:
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
    except OSError:
        return ""
    return h.hexdigest()


def _get_hashes_path(chroma_dir):
    return os.path.join(chroma_dir, ".file_hashes")


def get_existing_hashes(chroma_dir):
    hash_file = _get_hashes_path(chroma_dir)
    hashes = {}
    if os.path.exists(hash_file):
        with open(hash_file, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) == 2:
                    hashes[parts[0]] = parts[1]
    return hashes


def save_hashes(chroma_dir, hashes):
    os.makedirs(chroma_dir, exist_ok=True)
    hash_file = _get_hashes_path(chroma_dir)
    with open(hash_file, "w", encoding="utf-8") as f:
        for fname, fhash in hashes.items():
            f.write(f"{fname}\t{fhash}\n")


def split_documents(documents, chunk_size=DEFAULT_CHUNK_SIZE, chunk_overlap=DEFAULT_CHUNK_OVERLAP):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", "。", "！", "？", ".", "!", "?", " ", ""],
    )
    chunks = text_splitter.split_documents(documents)
    print(f"分块参数: chunk_size={chunk_size}, chunk_overlap={chunk_overlap}")
    print(f"分块结果: 共 {len(chunks)} 个文本块")
    for i, chunk in enumerate(chunks[:3]):
        preview = chunk.page_content[:80].replace("\n", " ")
        print(f"  块{i+1} (长度{len(chunk.page_content)}): {preview}...")
    if len(chunks) > 3:
        print(f"  ... 还有 {len(chunks)-3} 个文本块")
    return chunks


def main():
    parser = argparse.ArgumentParser(description="MyLibrary RAG 文档向量化工具")
    parser.add_argument("--chunk_size", type=int, default=DEFAULT_CHUNK_SIZE, help="文本块大小")
    parser.add_argument("--chunk_overlap", type=int, default=DEFAULT_CHUNK_OVERLAP, help="文本块重叠大小")
    parser.add_argument("--data_dir", type=str, default=None, help="文档目录路径")
    parser.add_argument("--incremental", action="store_true", help="增量更新模式")
    parser.add_argument("--verify-api", action="store_true", help="验证API连通性")
    args = parser.parse_args()

    print("=" * 60)
    print("  MyLibrary RAG - 文档向量化工具")
    print("=" * 60)

    if args.verify_api:
        success = verify_api()
        print()
        if not success:
            print("API 验证未通过，请修复 .env 配置后重试。")
        return

    data_dir = args.data_dir or DATA_DIR

    if args.incremental:
        existing_hashes = get_existing_hashes(CHROMA_DIR)
        current_hashes = {}
        new_files = []

        if not os.path.exists(data_dir):
            print(f"数据目录不存在: {data_dir}")
            return

        for filename in sorted(os.listdir(data_dir)):
            filepath = os.path.join(data_dir, filename)
            if not os.path.isfile(filepath):
                continue
            ext = filename.lower().rsplit('.', 1)[-1]
            if ext not in ("pdf", "txt", "md"):
                continue
            file_hash = compute_file_hash(filepath)
            current_hashes[filename] = file_hash
            if filename not in existing_hashes or existing_hashes[filename] != file_hash:
                new_files.append(filename)

        if not new_files:
            print("没有新增或修改的文件，无需更新。")
            return

        print(f"发现 {len(new_files)} 个新增/修改文件: {new_files}")
        documents = load_documents(data_dir)
        if not documents:
            return

        documents = [d for d in documents if d.metadata.get("source_file") in new_files]
        if not documents:
            return

        chunks = split_documents(documents)
        print("正在增量更新向量数据库...")
        count = add_documents_to_vectorstore(chunks)
        print(f"增量更新完成，当前共 {count} 个向量")
        save_hashes(CHROMA_DIR, current_hashes)
    else:
        documents = load_documents(data_dir)
        if not documents:
            print("未找到任何文档，请在 docs 目录中放入 PDF/TXT/MD 文件")
            return

        chunks = split_documents(documents, chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)
        print("正在创建向量数据库（本地嵌入模型 all-MiniLM-L6-v2）...")
        count = create_vectorstore_from_documents(chunks)
        print(f"向量数据库已保存至: {CHROMA_DIR}")
        print(f"共存储 {count} 个向量")

        current_hashes = {}
        for filename in sorted(os.listdir(data_dir)):
            filepath = os.path.join(data_dir, filename)
            if os.path.isfile(filepath):
                ext = filename.lower().rsplit('.', 1)[-1]
                if ext in ("pdf", "txt", "md"):
                    current_hashes[filename] = compute_file_hash(filepath)
        save_hashes(CHROMA_DIR, current_hashes)

    print("\n向量化完成！运行以下命令启动问答界面：")
    print("  python run.py")


if __name__ == "__main__":
    main()
