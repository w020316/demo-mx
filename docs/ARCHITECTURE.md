# MyLibrary RAG - 系统架构与 API 参考

## 目录

1. [架构概述](#1-架构概述)
2. [模块设计](#2-模块设计)
3. [数据流](#3-数据流)
4. [核心算法](#4-核心算法)
5. [API 接口参考](#5-api-接口参考)
6. [扩展指南](#6-扩展指南)

---

## 1. 架构概述

### 1.1 整体架构

MyLibrary RAG 采用**分层架构**设计，各层职责清晰，便于维护和扩展：

```
┌─────────────────────────────────────────────────────────────┐
│                    表现层 (Presentation)                     │
│   Streamlit Web UI / CLI Interface / Future REST API        │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│                    业务层 (Business)                         │
│   QA Engine / Conversation Manager / Feedback Engine        │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│                    服务层 (Service)                          │
│   Vector Store / LLM Service / Embedding Service            │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│                    数据层 (Data)                              │
│   ChromaDB / JSON Files / Document Files                    │
└─────────────────────────────────────────────────────────────┘
```

### 1.2 设计原则

| 原则 | 实现方式 |
|------|---------|
| **单一职责** | 每个模块只负责一个功能域 |
| **依赖倒置** | 上层不直接依赖具体实现，通过工厂函数获取实例 |
| **懒加载** | 重量级模块延迟初始化，优化启动速度 |
| **线程安全** | 单例模式 + 线程锁保护共享资源 |
| **原子操作** | 文件写入使用 tempfile + os.replace 保证一致性 |

---

## 2. 模块设计

### 2.1 模块依赖关系

```
app.py ──→ qa_chain.py ──→ vectorstore.py ──→ embeddings.py
   │              │                │
   │              ├────────→ llm.py
   │              │
   └─────→ feedback_engine.py

ingest.py ──→ vectorstore.py ──→ embeddings.py
                 ↑
config.py ←──────┘ (所有模块的基础配置)
```

### 2.2 核心模块说明

#### config.py - 配置中心

```python
# 核心常量
PROJECT_ROOT      # 项目根目录
DATA_DIR          # 文档存储路径 docs/
CHROMA_DIR        # 向量数据库路径 chroma_db/
FEEDBACK_DIR      # 反馈数据路径 feedback/

# 默认参数
DEFAULT_CHUNK_SIZE = 500       # 文本块大小
DEFAULT_CHUNK_OVERLAP = 50    # 文本块重叠
DEFAULT_K = 3                  # 默认检索数量
CHAT_FALLBACK_THRESHOLD = 0.52 # RAG↔Chat 切换阈值
```

#### embeddings.py - 嵌入模型管理

```python
class ChromaDefaultEmbeddings:
    """ChromaDB 内置的 ONNX 嵌入模型封装"""

    def __init__(self):
        self._model = ...  # all-MiniLM-L6-v2

    def embed_documents(self, texts: List[str]) -> List[List[float]]
    def embed_query(self, text: str) -> List[float]

# 工厂函数（线程安全单例）
def get_embeddings() -> ChromaDefaultEmbeddings:
    # 全局唯一实例，避免重复加载模型
```

#### vectorstore.py - 向量数据库

```python
class VectorStoreManager:
    """ChromaDB 向量数据库封装"""

    def get_vectorstore() -> Chroma:
        # 单例模式，全局唯一连接

    def get_retriever(k, search_type, fetch_k, lambda_mult) -> BaseRetriever:
        # 创建检索器（支持 similarity 和 mmr）

    def search_with_scores(question, k, search_type) -> Tuple[List[Document], List[float]]:
        # 带相似度分数的检索

    def add_documents_to_vectorstore(chunks) -> int:
        # 增量添加文档

    def create_vectorstore_from_documents(chunks) -> int:
        # 从文档创建新向量库
```

#### llm.py - 大语言模型接口

```python
class LLMService:
    """DeepSeek API 封装"""

    def get_llm(temperature=0) -> ChatOpenAI:
        # 创建 LLM 实例（支持自定义 base_url）

    def chat_directly(question, chat_history=None) -> str:
        # 直接对话（不走 RAG）

    def condense_question(llm, question, chat_history) -> str:
        # 多轮对话问题压缩

    def verify_api() -> bool:
        # API 连接验证
```

#### qa_chain.py - 问答引擎

```python
class QAEngine:
    """RAG 问答核心引擎"""

    def _looks_like_chat_question(question) -> bool:
        # 启发式聊天检测（正则 + 长度 + 关键词）

    def ask_question(question, k, prompt_mode, ...) -> Tuple[str, List[dict], str]:
        # 单轮问答主流程
        # 返回：(回答, 来源列表, 模式[rag/chat/reject])

    class ConversationManager:
        """多轮对话管理器"""
        def ask(self, question) -> Tuple[str, List[dict], str]:
            # 支持上下文的多轮问答
        def reset(self):
            # 清空对话历史
```

#### feedback_engine.py - 反馈引擎

```python
class FeedbackEngine:
    """自进化反馈系统"""

    def record_rating(question, answer, score, params, reason) -> dict:
        # 记录用户反馈（含防滥用检查）

    def check_abuse(question, session_count) -> Tuple[bool, str]:
        # 防滥用检测（频率限制 + 去重）

    def get_stats() -> dict:
        # 统计分析（按参数分组计算好评率）

    def get_recommendation() -> dict:
        # Wilson 区间评分 + 参数推荐

    def export_qa_pairs_to_docs() -> int:
        # 导出高分问答对到 Markdown
```

#### ingest.py - 文档摄入管线

```python
class IngestPipeline:
    """文档向量化管线"""

    def load_documents(data_dir) -> List[Document]:
        # 加载 PDF/TXT/MD 文档（自动编码检测）

    def split_documents(documents, chunk_size, chunk_overlap) -> List[Document]:
        # 文本分块（RecursiveCharacterTextSplitter）

    def main():
        # CLI 入口：加载 → 分块 → 向量化 → 存储
```

---

## 3. 数据流

### 3.1 文档摄入流程

```
docs/ 目录
    │
    ▼
┌─────────────┐     ┌──────────────┐     ┌─────────────────┐
│  文件扫描    │ ──▶ │  格式识别     │ ──▶ │  编码检测       │
│  *.pdf/txt/md│     │ PDF/TXT/MD   │     │ UTF-8/GBK/...   │
└─────────────┘     └──────────────┘     └────────┬────────┘
                                                │
    ┌─────────────────────────────────────────────▼──────────┐
    │                  文本预处理                              │
    │  clean_text(): 去除多余空白、规范化换行                   │
    └─────────────────────────────┬──────────────────────────┘
                                  │
    ┌─────────────────────────────▼──────────────────────────┐
    │                  文本分块                                │
    │  RecursiveCharacterTextSplitter                        │
    │  chunk_size=500, overlap=50                             │
    └─────────────────────────────┬──────────────────────────┘
                                  │
    ┌─────────────────────────────▼──────────────────────────┐
    │                  向量化                                 │
    │  all-MiniLM-L6-v2 → 384 维浮点向量                      │
    └─────────────────────────────┬──────────────────────────┘
                                  │
                                  ▼
                       ┌─────────────────┐
                       │   ChromaDB 存储  │
                       │  SQLite + HNSW   │
                       └─────────────────┘
```

### 3.2 问答流程

```
用户输入问题
      │
      ▼
┌─────────────────┐
│ 聊天检测         │ ◄── _looks_like_chat_question()
│ 正则匹配/长度/关键词│
└────┬─────┬──────┘
     │     │
  是聊天  否(继续)
     │     │
     ▼     ▼
┌─────────┐  ┌─────────────────┐
│自由对话  │  │  向量检索        │
│chat_    │  │ search_with_     │
│directly │  │ scores()         │
└─────────┘  └────────┬────────┘
                      │
               ┌──────▼──────┐
               │ 相似度判断   │
               │ < 0.52?     │
               └──────┬──────┘
                  是  │  否
                  ▼   │  ▼
           ┌────────┐ │ ┌────────────┐
           │自由对话 │ │ │ RAG 问答链  │
           └────────┘ │ │ _rag_query()│
                      │ └─────┬──────┘
                      │       │
                      │       ▼
                      │  ┌─────────────────┐
                      │  │  空回答检测      │
                      │  │ _is_rag_empty?  │
                      │  └───────┬─────────┘
                      │      是  │  否
                      │      ▼   │  ▼
                      │  ┌──────┴──────┐
                      │  │ 返回最终结果  │
                      │  │ (answer,     │
                      │  │  sources,    │
                      │  │  mode)       │
                      │  └─────────────┘
                      └──────────────►
```

### 3.3 反馈流程

```
用户点击 👍/👎
      │
      ▼
┌─────────────────┐
│  防滥用检查      │ ◄── check_abuse()
│  频率限制/去重   │
└────────┬────────┘
         │ 通过
         ▼
┌─────────────────┐
│  记录反馈        │ ◄── record_rating()
│  JSON 原子写入   │
└────────┬────────┘
         │
    ┌────┴────┐
    │         │
  👍 好评   👎 差评
    │         │
    ▼         ▼
┌──────┐  ┌──────────┐
│保存QA对│  │ 记录原因  │
│去重处理│  │ (可选)    │
└──────┘  └──────────┘
```

---

## 4. 核心算法

### 4.1 智能对话路由算法

```python
def _looks_like_chat_question(question: str) -> bool:
    """
    三层判断机制：
    1. 正则启发式规则（零延迟）
    2. 短文本兜底（≤6字且无领域关键词）
    """
    q = question.strip()

    # 第一层：正则匹配常见闲聊模式
    CHAT_PATTERNS = [
        r"^(你好|您好|嗨|hi|hello)...",  # 问候语
        r"^(你是谁|你能做什...)",           # 自我介绍
        r"^(谢谢|再见)...",                 # 结束语
        r"^(天气|讲个笑话...)",             # 闲聊
    ]
    for pattern in CHAT_PATTERNS:
        if re.search(pattern, q, re.IGNORECASE):
            return True

    # 第二层：短问题兜底
    if len(q) <= 6 and not any(kw in q for kw in DOMAIN_KEYWORDS):
        return True

    return False
```

### 4.2 Wilson 区间评分算法

```python
def _wilson_score(positive: int, total: int) -> float:
    """
    Wilson Score Interval with Continuity Correction

    用于解决小样本下的评分偏差问题：
    - 当样本少时，置信区间宽，得分向中间收敛
    - 当样本多时，置信区间窄，得分接近真实比例

    参数：
    - positive: 好评数
    - total: 总反馈数
    - z: Z值（默认 1.96，对应95%置信度）
    """
    if total == 0:
        return 0.0

    p = (positive + 1) / (total + 2)  # Laplace 平滑
    n = total + 2
    z = 1.959964

    denom = 1 + z * z / n
    center = p + z * z / (2 * n)
    spread = z * sqrt((p * (1 - p) + z * z / (4 * n)) / n)

    return max(0.0, min(1.0, (center - spread) / denom))
```

### 4.3 问题压缩算法

```python
def condense_question(llm, question: str, chat_history: List[Message]) -> str:
    """
    将历史对话 + 新问题压缩为独立问题

    用于多轮对话场景，确保每次检索都能获得完整语义。
    """
    if not chat_history:
        return question

    # 构建对话历史文本
    history_text = ""
    for msg in chat_history[-10:]:  # 只取最近10条
        if isinstance(msg, HumanMessage):
            history_text += f"用户: {msg.content}\n"
        elif isinstance(msg, AIMessage):
            history_text += f"助手: {msg.content}\n"

    # 调用 LLM 压缩
    prompt = f"""根据以下对话历史和最新问题，
    将问题重写为一个独立的、完整的问题。

    对话历史：
    {history_text}

    最新问题：{question}

    独立问题："""

    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content.strip() or question
```

### 4.4 MMR 多样性检索

```python
# MMR (Maximal Marginal Relevance) 公式：

# MMR(D_i) = λ · Sim(D_i, Q) - (1-λ) · max(Sim(D_i, D_j))

# 参数说明：
# - D_i: 候选文档 i
# - Q: 用户查询
# - λ (lambda_mult): 相关性-多样性平衡因子
#   - λ=1.0: 纯相关性（等同于 similarity 检索）
#   - λ=0.0: 纯多样性（可能偏离主题）
#   - λ=0.5: 平衡推荐值

# 使用示例：
retriever = get_retriever(
    k=3,              # 返回数量
    search_type="mmr",
    fetch_k=15,        # 候选池大小（越大多样性越高）
    lambda_mult=0.5,   # 平衡因子
)
```

---

## 5. API 接口参考

### 5.1 qa_chain 模块

#### `ask_question()` 函数

```python
def ask_question(
    question: str,
    k: int = 3,
    prompt_mode: str = "anti_hallucination",
    search_type: str = "similarity",
    similarity_threshold: Optional[float] = None,
    fetch_k: Optional[int] = None,
    lambda_mult: float = 0.5,
) -> Tuple[str, List[Dict[str, Any]], str]
```

**返回值**:

| 字段 | 类型 | 说明 |
|------|------|------|
| `answer` | `str` | 生成的回答 |
| `sources` | `List[Dict]` | 来源文档列表 |
| `mode` | `str` | 回答模式 (`rag`/`chat`/`reject`) |

**sources 结构**:

```python
{
    "content": "原始文档内容片段...",
    "source_file": "文件名.pdf",
    "page": 0,           # 页码（从0开始）
    "file_type": "pdf",  # 文件类型
}
```

#### `ConversationManager` 类

```python
class ConversationManager:
    def __init__(
        self,
        k: int = 3,
        prompt_mode: str = "anti_hallucination",
        search_type: str = "similarity",
        similarity_threshold: Optional[float] = None,
        memory_window: Optional[int] = None,
        fetch_k: Optional[int] = None,
        lambda_mult: float = 0.5,
    )

    def ask(self, question: str) -> Tuple[str, List[Dict], str]: ...
    def reset(self) -> None: ...
```

### 5.2 vectorstore 模块

```python
def get_vectorstore_info() -> Dict[str, Any]:
    """返回 {"count": int, "files": List[str]}"""

def search_with_scores(
    question: str,
    k: int = 3,
    search_type: str = "similarity",
    fetch_k: Optional[int] = None,
    lambda_mult: float = 0.5,
) -> Tuple[List[Document], List[float]]:
    """返回 (文档列表, 相似度分数列表)"""

def add_documents_to_vectorstore(chunks: List[Document]) -> int:
    """增量添加文档，返回总向量数"""

def create_vectorstore_from_documents(chunks: List[Document]) -> int:
    """创建新向量库，返回总向量数"""
```

### 5.3 feedback_engine 模块

```python
def record_rating(
    question: str,
    answer: str,
    score: int,  # 1=好评, -1=差评
    params: Optional[Dict] = None,
    reason: Optional[str] = None,
) -> Dict: ...

def check_abuse(
    question: str,
    session_ratings_count: int,
) -> Tuple[bool, str]:  # (是否允许, 错误信息)

def get_stats() -> Dict:
    """返回完整统计数据"""

def get_recommendation(stats: Optional[Dict] = None) -> Dict:
    """返回参数推荐和洞察"""

def export_qa_pairs_to_docs() -> int:
    """导出高分QA对，返回导出数量"""
```

### 5.4 提示词模板

```python
PROMPT_TEMPLATES = {
    "anti_hallucination": "...",  # 抑制幻觉模式
    "simple": "...",              # 简洁模式
    "academic": "...",            # 学术风格
    "cross_doc": "...",           # 跨文档对比
}
```

---

## 6. 扩展指南

### 6.1 添加新的 LLM 提供商

只需修改 `.env` 配置即可支持任何 OpenAI 兼容的 LLM 服务。

如需添加特殊适配，在 `llm.py` 中扩展：

```python
def get_llm(temperature=0):
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")

    # 可在此处添加特殊逻辑
    if "custom-provider" in base_url:
        return CustomLLM(...)

    return ChatOpenAI(...)
```

### 6.2 添加新的提示词模板

在 `qa_chain.py` 中添加：

```python
# 1. 定义新模板
CUSTOM_TEMPLATE = """你的提示词内容...
{context}
问题：{question}
回答："""

# 2. 注册到字典
PROMPT_TEMPLATES["custom_mode"] = CUSTOM_TEMPLATE
PROMPT_LABELS["custom_mode"] = "🎨 自定义模式"
```

### 6.3 添加新的文档格式支持

在 `ingest.py` 的 `load_documents()` 函数中添加：

```python
elif ext == "docx":
    from langchain_community.document_loaders import Docx2txtLoader
    loader = Docx2txtLoader(filepath)
    docs = loader.load()
    ...
elif ext == "html":
    from langchain_community.document_loaders import UnstructuredHTMLLoader
    loader = UnstructuredHTMLLoader(filepath)
    docs = loader.load()
    ...
```

### 6.4 自定义嵌入模型

修改 `embeddings.py`：

```python
from langchain_openai import OpenAIEmbeddings

class CustomEmbeddings:
    def __init__(self):
        self._model = OpenAIEmbeddings(model="text-embedding-3-small")

    def embed_documents(self, texts):
        return self._model.embed_documents(texts)

    def embed_query(self, text):
        return self._model.embed_query(text)
```

---

*最后更新时间：2026-05-16*
