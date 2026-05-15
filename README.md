# MyLibrary RAG - 智能文档问答系统

基于本地文档的 RAG（检索增强生成）智能问答系统，支持 PDF/TXT/MD 多格式文档导入，具备命令行和 Web 两种交互界面，集成自进化知识库机制。

## 项目目录结构

```
MyLibrary-RAG/
├── docs/                # 存放待处理的文档（PDF/TXT/MD）
├── chroma_db/           # 向量数据库持久化目录（运行 ingest.py 后自动生成）
├── feedback/            # 反馈数据目录（运行时自动生成）
├── ingest.py            # 文档摄入脚本：加载文档 → 文本分块 → 向量化 → 存入 ChromaDB
├── query.py             # 命令行问答脚本：加载向量库 → 检索 → 调用 LLM → 输出答案
├── app.py               # Streamlit Web 界面：提供可视化问答交互
├── qa_chain.py          # 问答链核心逻辑（检索、记忆、Prompt模板、拒答机制）
├── feedback_engine.py   # 自进化反馈引擎（评分记录、统计分析、参数推荐、问答对导出）
├── config.py            # 共享配置（路径常量、环境变量）
├── embeddings.py        # 共享嵌入模型封装
├── run.py               # Streamlit 启动入口
├── start.bat            # Windows 一键启动脚本
├── ingest.bat           # Windows 文档入库脚本
├── requirements.txt     # Python 依赖清单
├── .env                 # 环境变量配置文件（API_KEY、BASE_URL、MODEL_NAME）
├── .env.example         # 环境变量示例文件
├── .streamlit/config.toml  # Streamlit 配置
└── README.md            # 项目说明文档
```

## 技术栈

| 组件 | 技术选型 | 说明 |
|------|---------|------|
| 文档加载 | PyPDFLoader / TextLoader | 支持 PDF、TXT、MD 格式，自动检测中文编码 |
| 文本分割 | RecursiveCharacterTextSplitter | 中英文语义边界分割，可调 chunk_size/overlap |
| 嵌入模型 | ChromaDB ONNX (all-MiniLM-L6-v2) | 本地运行，无需 GPU |
| 向量数据库 | ChromaDB | 持久化存储，支持相似度检索与 MMR 多样性检索 |
| 大模型 | DeepSeek API (deepseek-chat) | 通过 ChatOpenAI 兼容接口调用 |
| 前端界面 | Streamlit | 支持 st.chat_message、st.session_state、懒加载优化 |
| 反馈引擎 | JSON 本地存储 | 评分记录、防滥用、自进化参数推荐 |

## 快速开始

### 1. 环境准备

```bash
# 安装 Python 依赖到项目目录
pip install --no-cache-dir --target pylibs -r requirements.txt
```

### 2. 配置 API Key

复制 `.env.example` 为 `.env`，填入你的 API 配置：

```bash
cp .env.example .env
```

编辑 `.env`：

```
OPENAI_API_KEY=your_api_key_here
OPENAI_BASE_URL=https://api.deepseek.com/v1
MODEL_NAME=deepseek-chat
```

验证 API 连通性：

```bash
python ingest.py --verify-api
```

### 3. 导入文档

将 PDF/TXT/MD 文件放入 `docs/` 目录，然后运行：

```bash
python ingest.py
# 或使用增量更新模式
python ingest.py --incremental
# 自定义分块参数
python ingest.py --chunk_size 500 --chunk_overlap 50
```

### 4. 启动问答

**命令行模式：**

```bash
python query.py
```

**Web 界面模式：**

```bash
# 方式一：直接运行启动脚本
start.bat

# 方式二：手动启动
python run.py
# 浏览器自动打开 http://localhost:8501
```

## 功能特性

### 基础功能（Part 1）

- **多格式文档加载**：PDF、TXT、Markdown，自动检测中文编码（GBK/GB2312/GB18030）
- **文本清洗与分块**：自动移除多余空格和换行符，基于语义边界的递归字符分割
- **API 安全配置**：API Key 通过 .env 管理，已加入 .gitignore
- **环境验证**：`--verify-api` 参数验证 API 连通性并给出排查建议
- **命令行与 Web 双界面**：query.py 命令行交互 + Streamlit Web 可视化界面
- **引用溯源**：每条回答附带来源文件、页码和原文引用

### 进阶功能（Part 2）

| 编号 | 功能 | 说明 |
|------|------|------|
| A1 | 多格式支持 | PDF/TXT/MD 三种格式，中文编码自动检测，统一元数据 |
| A2 | 自定义助手风格 | 4种提示词模式：抑制幻觉 / 简单模式 / 学术风格 / 跨文档对比 |
| A3 | 拒答阈值 | 基于相似度分数的拒答机制，低于阈值时拒绝回答并给出建议 |
| B1 | MMR 多样性检索 | 支持 fetch_k 和 lambda_mult 参数调节，兼顾相关性与多样性 |
| B3 | 多轮对话记忆窗口 | 可设置记忆轮数上限，防止上下文过长导致性能下降 |

### 大师挑战（M5 - 自进化知识库）

- **用户反馈收集**：👍/👎 评分按钮，记录问题、回答、评分及当前参数
- **防滥用机制**：单会话最多 20 次评分，同一问题 60 秒内不可重复评分
- **统计分析**：按 k 值/阈值/检索策略/提示词模式分组统计好评率
- **自进化推荐**：基于历史反馈数据推荐最优参数组合
- **高分问答对导出**：将好评问答对导出为文档，支持增量入库实现知识库自增长

## 命令行交互指令

在 `query.py` 交互模式中可用以下指令：

| 指令 | 说明 | 示例 |
|------|------|------|
| `/k <数字>` | 设置检索数量 k | `/k 5` |
| `/mode <模式>` | 切换提示词模式 | `/mode academic` |
| `/search <类型>` | 切换检索策略 | `/search mmr` |
| `/threshold <值>` | 设置拒答阈值 | `/threshold 0.3` |
| `/window <轮数>` | 设置记忆窗口 | `/window 5` |
| `quit` / `exit` | 退出 | — |

可用提示词模式：`anti_hallucination`、`simple`、`academic`、`cross_doc`

## Web 界面功能

- **侧边栏参数调节**：k 值滑块、检索策略选择、提示词模式（中文标签）、多轮记忆开关
- **进阶参数面板**：拒答阈值(A3)、MMR 参数(B1)、记忆窗口(B3)
- **数据状态面板**：向量片段数、反馈统计、已加载文档列表
- **自进化分析面板**：AI 参数推荐、高分问答对计数、一键导出
- **来源溯源**：折叠面板展示来源文档，含彩色类型徽章和相似度分数
- **反馈按钮**：每条回答下方 👍/👎 评分
- **快捷问题**：欢迎页建议问题按钮
- **空库提示**：向量库为空时显示入库指引

## 验收标准

| 验收项 | 标准 | 状态 |
|--------|------|------|
| 向量化成功 | 执行 `python ingest.py` 后 chroma_db 目录生成且包含数据文件 | ✅ |
| 命令行问答 | 执行 `python query.py` 后进入交互模式，可准确回答文档内问题 | ✅ |
| Web 界面可用 | 执行 `python run.py` 后浏览器打开页面，可正常问答 | ✅ |
| 参数可调 | Web 侧边栏提供 k 值滑块控件，调整后回答内容随之变化 | ✅ |
| API 安全 | .env 未提交版本控制，API Key 通过环境变量读取 | ✅ |
| 多格式支持(A1) | PDF/TXT/MD 均可加载，中文编码自动检测 | ✅ |
| 提示词模式(A2) | 4种模式可切换，回答风格随模式变化 | ✅ |
| 拒答阈值(A3) | 设置阈值后，低相似度问题返回拒答提示 | ✅ |
| MMR检索(B1) | 切换为MMR策略后，检索结果兼顾多样性与相关性 | ✅ |
| 记忆窗口(B3) | 设置窗口后，超出轮数的对话历史自动裁剪 | ✅ |
| 自进化知识库(M5) | 反馈收集→统计分析→参数推荐→问答对导出→增量入库 | ✅ |

## 注意事项

- `.env` 文件包含敏感信息，已加入 `.gitignore`，请勿提交至版本控制
- 嵌入模型首次运行时会自动下载 ONNX 模型到 `.cache/chroma/` 目录
- DeepSeek API 需要网络连接和有效余额才能正常调用
- Streamlit 文件监视器已禁用（`fileWatcherType = "none"`），修改代码后需手动刷新浏览器
- 反馈数据存储在 `feedback/` 目录下（JSON 格式），已加入 `.gitignore`
