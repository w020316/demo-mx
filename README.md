# MyLibrary RAG - 智能文档问答系统

<p align="center">
  <strong>基于 RAG（检索增强生成）技术的本地知识库问答系统</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.12-blue.svg" alt="Python 3.12" />
  <img src="https://img.shields.io/badge/LangChain-1.3+-green.svg" alt="LangChain" />
  <img src="https://img.shields.io/badge/Streamlit-1.57+-red.svg" alt="Streamlit" />
  <img src="https://img.shields.io/badge/ChromaDB-1.5+-orange.svg" alt="ChromaDB" />
  <img src="https://img.shields.io/badge/DeepSeek-API-yellow.svg" alt="DeepSeek API" />
</p>

---

## 项目简介

MyLibrary RAG 是一个基于 **RAG（Retrieval-Augmented Generation，检索增强生成）** 技术构建的本地知识库问答系统。用户上传技术文档后，系统自动完成向量化入库，支持通过自然语言提问获取精准答案，并附带来源追溯。

### 核心特性

| 特性 | 描述 |
|------|------|
| **多格式文档支持** | PDF、TXT、Markdown 全格式支持，中文编码自动检测 |
| **智能对话路由** | 启发式规则 + 相似度阈值双重判断，自动切换 RAG/自由对话模式 |
| **自进化反馈引擎** | Wilson 区间统计评分，基于历史反馈的参数推荐系统 |
| **多轮对话记忆** | 问题压缩 + 记忆窗口管理，支持上下文连贯的多轮交互 |
| **MMR 多样性检索** | 兼顾相关性与多样性的高级检索策略 |
| **自定义提示词** | 4 种专业提示词模板：抑制幻觉 / 简洁 / 学术 / 跨文档对比 |
| **相似度拒答** | 可配置阈值，低相关度问题智能拒答 |

## 技术架构

```
┌─────────────────────────────────────────────────────────────┐
│                      用户交互层                              │
│   ┌──────────────┐              ┌──────────────┐           │
│   │  Streamlit   │              │    CLI       │           │
│   │  Web 界面    │              │  query.py    │           │
│   └──────┬───────┘              └──────┬───────┘           │
└──────────┼─────────────────────────────┼───────────────────┘
           │                             │
┌──────────┼─────────────────────────────┼───────────────────┐
           ▼                             ▼                   │
│  ┌──────────────────────────────────────────────────┐     │
│  │              核心业务层 (qa_chain.py)             │     │
│  │                                                  │     │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────────┐  │     │
│  │  │ 智能路由 │→ │ RAG问答链 │  │ 多轮对话管理器│  │     │
│  │  │_looks_  │  │ask_quest │  │Conversation  │  │     │
│  │  │like_chat│  │  ion()   │  │Manager.ask() │  │     │
│  │  └──────────┘  └────┬─────┘  └──────┬───────┘  │     │
│  │                      │               │          │     │
│  │  ┌───────────────────┼───────────────┼──────┐  │     │
│  │  │                   ▼               ▼      │  │     │
│  │  │  ┌──────────┐  ┌──────────┐  ┌─────────┐│  │     │
│  │  │  │ChromaDB  │  │DeepSeek  │  │问题压缩  ││  │     │
│  │  │  │向量检索   │  │API生成   │  │condense  ││  │     │
│  │  │  └────┬─────┘  └────┬─────┘  └─────────┘│  │     │
│  │  └───────┼─────────────┼───────────────────┘  │     │
│  └──────────┼─────────────┼──────────────────────┘     │
│             │             │                            │
├─────────────┼─────────────┼────────────────────────────┤
│             ▼             ▼                            │
│  ┌────────────────┐  ┌────────────────┐               │
│  │  数据存储层     │  │  外部服务      │               │
│  │                │  │                │               │
│  │  ChromaDB      │  │  DeepSeek API  │               │
│  │  (向量数据库)   │  │  LLM 服务      │               │
│  │                │  │                │               │
│  └────────────────┘  └────────────────┘               │
│                                                       │
│  ┌───────────────────────────────────────────────┐   │
│  │         数据摄入层 (ingest.py)                 │   │
│  │                                               │   │
│  │  docs/ → 加载 → 编码检测 → 分块 → 向量化 → 存储  │   │
│  └───────────────────────────────────────────────┘   │
│                                                       │
│  ┌───────────────────────────────────────────────┐   │
│  │      自进化层 (feedback_engine.py)              │   │
│  │                                               │   │
│  │  👍/👎 → 防滥用 → 统计分析 → 参数推荐 → 导出     │   │
│  └───────────────────────────────────────────────┘   │
└───────────────────────────────────────────────────────┘
```

## 技术栈

| 组件 | 技术选型 | 版本要求 | 说明 |
|------|---------|---------|------|
| 文档加载 | PyPDFLoader / TextLoader | pypdf>=5.0 | PDF/TXT/MD 多格式支持 |
| 文本分割 | RecursiveCharacterTextSplitter | langchain-text-splitters | 中英文语义边界分割 |
| 嵌入模型 | all-MiniLM-L6-v2 (ONNX) | chromadb>=1.5 | 本地运行，无需 GPU |
| 向量数据库 | ChromaDB | >=1.5.0 | SQLite + HNSW 索引 |
| 大语言模型 | DeepSeek API | langchain-openai | ChatOpenAI 兼容接口 |
| 前端框架 | Streamlit | >=1.57.0 | Web 可视化交互界面 |
| 反馈引擎 | JSON 本地存储 | Python 内置 | Wilson 区间统计 |

## 快速开始

### 前置条件

- **操作系统**: Windows 10/11 (推荐), Linux, macOS
- **Python**: 3.10+ (推荐 3.12)
- **API Key**: DeepSeek API Key ([获取地址](https://platform.deepseek.com/api_keys))

### 一键安装 (Windows)

```bash
# 方式一：使用安装向导（推荐）
setup.bat

# 方式二：手动安装
# 1. 安装依赖
pip install --no-cache-dir --target pylibs -r requirements.txt

# 2. 配置环境变量
cp .env.example .env
# 编辑 .env 填入您的 API Key

# 3. 导入文档
python ingest.py

# 4. 启动应用
start.bat
```

### Linux/macOS 安装

```bash
# 克隆项目
git clone https://github.com/w020316/demo-mx.git mylibrary-rag
cd mylibrary-rag

# 创建虚拟环境（推荐）
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt

# 配置环境变量
cp .env.example .env
nano .env  # 填入您的 API Key

# 导入文档到向量数据库
python ingest.py

# 启动 Web 应用
streamlit run app.py
```

## 使用指南

### Web 界面操作

启动后浏览器自动打开 `http://localhost:8501`：

1. **左侧边栏**: 调整检索参数（k值、检索策略、提示词模式等）
2. **主区域**: 输入问题进行问答
3. **底部按钮**: 对回答进行 👍/👎 反馈评分
4. **侧边栏仪表盘**: 查看统计分析和参数推荐

### CLI 命令行问答

```bash
python query.py

# 支持命令:
# /k <数字>        设置检索数量
# /mode <模式>     设置提示词模式
# /search <类型>   设置检索策略(similarity/mmr)
# /threshold <值>  设置拒答阈值(0=关闭)
# /window <数字>   设置记忆窗口(0=不限制)
# quit 或 exit     退出
```

### 文档导入

```bash
# 默认导入（chunk_size=500）
python ingest.py

# 自定义分块大小
python ingest.py --chunk_size 200

# 增量更新（仅处理新增/修改文件）
python ingest.py --incremental

# 自定义文档目录
python ingest.py --data_dir /path/to/docs
```

## 配置说明

### 环境变量 (.env)

```bash
# 必填：API 密钥（从 DeepSeek 平台获取）
OPENAI_API_KEY=sk-your-api-key-here

# 必填：API 基础 URL
OPENAI_BASE_URL=https://api.deepseek.com/v1

# 可选：模型名称（默认 deepseek-chat）
MODEL_NAME=deepseek-chat
```

### Streamlit 配置 (.streamlit/config.toml)

```toml
[global]
developmentMode = false

[server]
enableCORS = true
enableXsrfProtection = false
fileWatcherType = "none"
headless = true

[browser]
gatherUsageStats = false
```

### 关键参数调优

| 参数 | 默认值 | 推荐范围 | 说明 |
|------|-------|---------|------|
| k (检索数量) | 3 | 1-7 | 返回的文档片段数 |
| chunk_size | 500 | 200-800 | 文本块大小 |
| chunk_overlap | 50 | 20-100 | 文本块重叠长度 |
| CHAT_FALLBACK_THRESHOLD | 0.52 | 0.45-0.60 | RAG↔Chat 切换阈值 |
| similarity_threshold | 0.0 | 0.0-0.7 | 拒答阈值（0=关闭） |
| lambda_mult | 0.5 | 0.0-1.0 | MMR 多样性平衡因子 |

## 项目结构

```
mylibrary-rag/
├── app.py                  # Streamlit Web 主界面
├── qa_chain.py             # QA 核心逻辑（问答链/对话管理）
├── feedback_engine.py      # 自进化反馈引擎（Wilson评分/推荐）
├── ingest.py               # 文档向量化工具（加载/分块/存储）
├── vectorstore.py          # 向量数据库集成（ChromaDB封装）
├── llm.py                  # LLM接口（DeepSeek API封装）
├── embeddings.py            # ONNX嵌入模型（单例管理）
├── config.py               # 全局配置（路径常量/默认参数）
├── query.py                # CLI 问答入口
├── run.py                  # Streamlit 启动入口
├── start.bat               # Windows 一键启动脚本
├── setup.bat               # Windows 安装向导
├── ingest.bat              # Windows 入库脚本
├── requirements.txt        # Python 依赖清单
├── .env.example            # 环境变量示例
├── .gitignore              # Git 忽略规则
├── .streamlit/config.toml  # Streamlit 配置
├── docs/                   # 知识库文档
│   ├── Python与RAG技术介绍.txt
│   ├── LangChain框架入门指南.txt
│   ├── Streamlit快速开发指南.md
│   └── 用户反馈问答对.md    # 自动生成的高分QA对
├── tests/                  # 单元测试
│   ├── test_config.py
│   ├── test_embeddings.py
│   ├── test_qa_chain.py
│   ├── test_feedback_engine.py
│   ├── test_ingest.py
│   └── test_integration.py
├── chroma_db/              # 向量数据库（运行时生成）
└── feedback/               # 反馈数据（运行时生成）
```

## API 参考

### 核心模块接口

#### qa_chain.py - 问答引擎

```python
from qa_chain import ask_question, ConversationManager

# 单轮问答
answer, sources, mode = ask_question(
    question="RAG技术的核心原理是什么？",
    k=3,
    prompt_mode="anti_hallucination",
    search_type="similarity",
    similarity_threshold=None,  # None 表示不启用拒答
)

# 多轮对话
conv = ConversationManager(
    k=3,
    prompt_mode="anti_hallucination",
    memory_window=5,  # 保留最近5轮对话
)
answer, sources, mode = conv.ask("LangChain有哪些主要组件？")
answer, sources, mode = conv.ask("它的优势是什么？")  # 自动关联上文
```

#### llm.py - LLM 接口

```python
from llm import get_llm, chat_directly, condense_question, verify_api

# 获取 LLM 实例
llm = get_llm(temperature=0)

# 直接对话（不走 RAG）
response = chat_directly("你好")

# 问题压缩（用于多轮对话）
condensed = condense_question(llm, "它有什么特点？", chat_history_messages)

# API 连接验证
success = verify_api()  # 返回 True/False
```

#### vectorstore.py - 向量数据库

```python
from vectorstore import (
    get_vectorstore, get_retriever, search_with_scores,
    add_documents_to_vectorstore, create_vectorstore_from_documents,
    get_vectorstore_info
)

# 获取向量库信息
info = get_vectorstore_info()
# {"count": 47, "files": ["file1.pdf", "file2.txt"]}

# 带分数的检索
docs, scores = search_with_scores(
    "问题内容",
    k=3,
    search_type="similarity",  # or "mmr"
    fetch_k=15,  # MMR 候选数量
    lambda_mult=0.5,  # MMR 多样性平衡
)
```

#### feedback_engine.py - 反馈引擎

```python
from feedback_engine import (
    record_rating, get_stats, get_recommendation,
    check_abuse, export_qa_pairs_to_docs,
)

# 记录用户反馈
record_rating(
    question="问题",
    answer="回答",
    score=1,  # 1=好评, -1=差评
    params={"k": 3, "prompt_mode": "anti_hallucination"},
    reason="inaccurate",  # 差评原因（可选）
)

# 获取统计数据
stats = get_stats()
# {"total": 100, "positive": 85, "health_score": 78, ...}

# 获取参数推荐
rec = get_recommendation()
# {"recommendations": [...], "insights": [...]}
```

## 测试

```bash
# 运行所有测试
python -m pytest tests/ -v

# 运行特定模块测试
python -m pytest tests/test_qa_chain.py -v
python -m pytest tests/test_feedback_engine.py -v

# 运行集成测试
python -m pytest tests/test_integration.py -v
```

## 常见问题

### Q: 页面显示 404 Not Found？

确保在启动前设置环境变量：
```bash
set STREAMLIT_GLOBAL_DEVELOPMENTMODE=false
streamlit run app.py
```
或在 `start.bat` 中已包含此设置。

### Q: API 认证失败 (401)？

检查 `.env` 文件中的 `OPENAI_API_KEY` 是否正确：
1. 从 [DeepSeek 平台](https://platform.deepseek.com/api_keys) 复制完整的 API Key
2. 确保 Key 以 `sk-` 开头
3. 检查账户余额是否充足

### Q: 中文文档乱码？

系统已内置编码自动检测（UTF-8/GBK/GB2312/GB18030）。如仍有问题：
- 手动指定编码：修改 `ingest.py` 中的 `_detect_encoding()` 函数优先级
- 将文档转换为 UTF-8 编码后重新导入

### Q: 回答质量不理想？

尝试以下优化：
1. 调整 k 值（增大获取更多上下文）
2. 切换提示词模式（学术模式更严谨）
3. 使用 MMR 检索策略提高多样性
4. 检查文档是否覆盖了问题领域

## 开发路线图

- [x] 基础 RAG 问答功能
- [x] 多格式文档支持
- [x] 智能对话路由
- [x] 自进化反馈引擎
- [x] MMR 多样性检索
- [x] 多轮对话记忆
- [ ] 支持更多 LLM 提供商（OpenAI/Claude/Qwen 等）
- [ ] 文档版本管理与增量同步
- [ ] 用户权限与多租户支持
- [ ] Docker 容器化部署
- [ ] REST API 服务化

## 许可证

MIT License

## 致谢

- [LangChain](https://github.com/langchain-ai/langchain) - LLM 应用开发框架
- [ChromaDB](https://github.com/chroma-core/chroma) - 开源向量数据库
- [Streamlit](https://github.com/streamlit/streamlit) - 数据应用框架
- [DeepSeek](https://deepseek.com/) - 大语言模型服务
- [sentence-transformers](https://github.com/UKPLab/sentence-transformers) - 嵌入模型
