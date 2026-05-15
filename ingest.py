import os
import re
import argparse
import hashlib
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from config import DATA_DIR, CHROMA_DIR, PROJECT_ROOT
from embeddings import ChromaDefaultEmbeddings
from dotenv import load_dotenv

load_dotenv(os.path.join(PROJECT_ROOT, ".env"))


def clean_text(text):
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def _detect_encoding(filepath):
    encodings = ["utf-8", "gbk", "gb2312", "gb18030", "latin-1"]
    with open(filepath, "rb") as f:
        raw = f.read(8192)
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
    h = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
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


def split_documents(documents, chunk_size=500, chunk_overlap=50):
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


def create_vectorstore(chunks, persist_directory=None):
    if persist_directory is None:
        persist_directory = CHROMA_DIR
    embeddings = ChromaDefaultEmbeddings()
    print("正在创建向量数据库（本地嵌入模型 all-MiniLM-L6-v2）...")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_directory,
    )
    print(f"向量数据库已保存至: {persist_directory}")
    print(f"共存储 {vectorstore._collection.count()} 个向量")
    return vectorstore


def incremental_update(data_dir=None, persist_directory=None):
    if data_dir is None:
        data_dir = DATA_DIR
    if persist_directory is None:
        persist_directory = CHROMA_DIR

    existing_hashes = get_existing_hashes(persist_directory)
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
    embeddings = ChromaDefaultEmbeddings()
    print("正在增量更新向量数据库...")
    vectorstore = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings,
    )
    vectorstore.add_documents(chunks)
    print(f"增量更新完成，当前共 {vectorstore._collection.count()} 个向量")
    save_hashes(persist_directory, current_hashes)


def verify_api():
    api_key = os.getenv("OPENAI_API_KEY", "")
    base_url = os.getenv("OPENAI_BASE_URL", "")

    print("-" * 50)
    print("  环境配置与API验证")
    print("-" * 50)

    if not api_key:
        print("  ❌ OPENAI_API_KEY 未设置，请检查 .env 文件")
        return False
    if not base_url:
        print("  ❌ OPENAI_BASE_URL 未设置，请检查 .env 文件")
        return False

    masked_key = api_key[:8] + "..." + api_key[-4:] if len(api_key) > 12 else "***"
    print(f"  ✅ API Key: {masked_key}")
    print(f"  ✅ Base URL: {base_url}")
    print(f"  ✅ Model: {os.getenv('MODEL_NAME', '未设置')}")

    try:
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(
            model=os.getenv("MODEL_NAME", "deepseek-chat"),
            temperature=0,
            openai_api_key=api_key,
            openai_api_base=base_url,
            max_tokens=10,
        )
        response = llm.invoke("说'OK'")
        print(f"  ✅ API 连接成功！模型响应: {response.content.strip()}")
        print("\n  💡 思考与分析要点：")
        print("     · API Key 存放于 .env 的必要性：避免密钥泄露到代码仓库（.gitignore已排除）")
        print(f"     · BASE_URL 必须包含 /v1 后缀，否则会报错（当前值: {base_url}）")
        return True
    except Exception as e:
        error_str = str(e)
        print(f"  ❌ API 连接失败: {error_str}")

        if "401" in error_str or "Unauthorized" in error_str:
            print("     → 原因: API Key 无效或已过期")
        elif "402" in error_str or "Insufficient" in error_str:
            print("     → 原因: 账户余额不足，请充值")
        elif "404" in error_str or "Not Found" in error_str:
            print("     → 原因: BASE_URL 路径错误，检查是否包含 /v1 后缀")
        elif "Connection" in error_str or "timeout" in error_str.lower():
            print("     → 原因: 网络连接失败或服务不可达")

        print("\n  💡 排查建议：删除 BASE_URL 末尾的 /v1 后运行，观察报错变化以定位问题")
        return False


def main():
    parser = argparse.ArgumentParser(description="MyLibrary RAG 文档向量化工具")
    parser.add_argument("--chunk_size", type=int, default=500, help="文本块大小 (默认: 500)")
    parser.add_argument("--chunk_overlap", type=int, default=50, help="文本块重叠大小 (默认: 50)")
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
        incremental_update(data_dir=data_dir)
    else:
        documents = load_documents(data_dir)
        if not documents:
            print("未找到任何文档，请在 docs 目录中放入 PDF/TXT/MD 文件")
            return

        chunks = split_documents(documents, chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)
        vectorstore = create_vectorstore(chunks)

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
