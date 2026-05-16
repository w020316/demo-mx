# MyLibrary RAG - API 配置与使用指南

## 概述

MyLibrary RAG 使用 **DeepSeek API** 作为大语言模型（LLM）后端，通过 OpenAI 兼容接口进行调用。本指南详细说明如何配置和使用 API 服务。

---

## 目录

1. [获取 API Key](#1-获取-api-key)
2. [配置环境变量](#2-配置环境变量)
3. [API 验证](#3-api-验证)
4. [支持的 LLM 提供商](#4-支持的-llm-提供商)
5. [常见问题排查](#5-常见问题排查)

---

## 1. 获取 API Key

### DeepSeek API（默认）

1. 访问 [DeepSeek 开放平台](https://platform.deepseek.com/)
2. 注册/登录账户
3. 进入 **API Keys** 管理页面
4. 点击 **Create new key** 创建新的 API Key
5. 复制生成的 Key（格式：`sk-xxxxxxxxxxxxxxxx`）

> ⚠️ **安全提示**：API Key 仅在创建时显示一次，请妥善保管。如泄露请立即在平台删除并重新生成。

### 其他兼容提供商

本项目支持任何提供 OpenAI 兼容接口的 LLM 服务：

| 提供商 | Base URL | 获取地址 |
|-------|---------|---------|
| DeepSeek | `https://api.deepseek.com/v1` | https://platform.deepseek.com/api_keys |
| OpenAI | `https://api.openai.com/v1` | https://platform.openai.com/api-keys |
| Azure OpenAI | `https://{resource}.openai.azure.com/openai/deployments/{deployment}` | https://portal.azure.com |
| 通义千问 (Qwen) | `https://dashscope.aliyuncs.com/compatible-mode/v1` | https://dashscope.console.aliyun.com |
| 智谱 AI (GLM) | `https://open.bigmodel.cn/api/paas/v4` | https://open.bigmodel.cn/usercenter/apikeys |
| Moonshot (Kimi) | `https://api.moonshot.cn/v1` | https://platform.moonshot.cn/console/api-keys |

---

## 2. 配置环境变量

### 方法一：使用 .env 文件（推荐）

在项目根目录创建 `.env` 文件：

```bash
# 从示例文件复制
cp .env.example .env

# 编辑填入您的配置
```

`.env` 文件内容：

```ini
# ============================================
# MyLibrary RAG - API 配置
# ============================================

# 必填：API 密钥
OPENAI_API_KEY=sk-your-api-key-here

# 必填：API 基础 URL（根据选择的 LLM 提供商修改）
OPENAI_BASE_URL=https://api.deepseek.com/v1

# 可选：模型名称
MODEL_NAME=deepseek-chat

# 其他可选配置：
# OPENAI_ORG_ID=org-xxxxx          # 组织 ID（OpenAI 需要）
# OPENAI_PROXY=http://127.0.0.1:7890  # HTTP 代理
```

### 方法二：系统环境变量

**Windows (PowerShell)**:
```powershell
$env:OPENAI_API_KEY = "sk-your-api-key-here"
$env:OPENAI_BASE_URL = "https://api.deepseek.com/v1"
$env:MODEL_NAME = "deepseek-chat"
```

**Windows (CMD)**:
```cmd
set OPENAI_API_KEY=sk-your-api-key-here
set OPENAI_BASE_URL=https://api.deepseek.com/v1
set MODEL_NAME=deepseek-chat
```

**Linux/macOS (Bash)**:
```bash
export OPENAI_API_KEY="sk-your-api-key-here"
export OPENAI_BASE_URL="https://api.deepseek.com/v1"
export MODEL_NAME="deepseek-chat"
```

### 方法三：通过安装向导配置（Windows）

双击运行 `setup.bat`，向导会自动引导您完成配置。

---

## 3. API 验证

配置完成后，验证 API 连接是否正常：

```bash
python ingest.py --verify-api
```

预期输出：

```
--------------------------------------------------
  环境配置与API验证
--------------------------------------------------
  ✅ API Key: sk-xxxx...xxxx
  ✅ Base URL: https://api.deepseek.com/v1
  ✅ Model: deepseek-chat
  ✅ API 连接成功！模型响应: OK
```

如果出现错误，请参考[常见问题排查](#5-常见问题排查)。

---

## 4. 支持的 LLM 提供商

### DeepSeek（推荐）

```ini
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxx
OPENAI_BASE_URL=https://api.deepseek.com/v1
MODEL_NAME=deepseek-chat        # 或 deepseek-reasoner（推理模型）
```

**优势**：性价比高、中文能力强、API 响应快

### OpenAI GPT 系列

```ini
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxx
OPENAI_BASE_URL=https://api.openai.com/v1
MODEL_NAME=gpt-4o               # 或 gpt-4o-mini, gpt-3.5-turbo
```

**注意**：需要有效的 OpenAI API Key 和充足余额

### 通义千问 Qwen

```ini
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxx
OPENAI_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
MODEL_NAME=qwen-plus             # 或 qwen-max, qwen-turbo
```

### 智谱 GLM

```ini
OPENAI_API_KEY=xxxxxxxxxxxxxxxx.xxxxxxxx
OPENAI_BASE_URL=https://open.bigmodel.cn/api/paas/v4
MODEL_NAME=glm-4                 # 或 glm-4-flash, glm-3-turbo
```

### Kimi / Moonshot

```ini
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxx
OPENAI_BASE_URL=https://api.moonshot.cn/v1
MODEL_NAME=moonshot-v1-8k        # 或 moonshot-v1-32k, moonshot-v1-128k
```

---

## 5. 常见问题排查

### 错误码对照表

| HTTP 状态码 | 错误信息 | 原因 | 解决方案 |
|------------|---------|------|---------|
| 401 | Authentication Fails | API Key 无效或过期 | 检查 Key 是否正确复制 |
| 402 | Insufficient Balance | 账户余额不足 | 充值账户 |
| 404 | Not Found | Base URL 路径错误 | 确保 URL 以 `/v1` 结尾 |
| 429 | Rate Limit Exceeded | 请求频率超限 | 降低请求频率或升级套餐 |
| 500 | Internal Server Error | 服务端错误 | 稍后重试 |
| Connection Error | 连接失败 | 网络问题或代理设置 | 检查网络/配置代理 |

### 问题诊断步骤

#### Step 1: 验证网络连通性

```bash
# 测试 DeepSeek API 是否可达
curl https://api.deepseek.com/v1/models -H "Authorization: Bearer sk-your-key"

# 或使用 Python
python -c "import urllib.request; r = urllib.request.urlopen('https://api.deepseek.com'); print(r.status)"
```

#### Step 2: 检查 API Key 格式

- DeepSeek Key 格式：`sk-` + 32位十六进制字符
- 总长度应为 35 个字符
- 示例：`sk-a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6`

#### Step 3: 检查代理设置（如有）

如果在需要代理的网络环境中使用：

```ini
# 在 .env 中添加
OPENAI_PROXY=http://127.0.0.1:7890
```

或在代码中设置：

```python
import os
os.environ["HTTP_PROXY"] = "http://127.0.0.1:7890"
os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7890"
```

#### Step 4: 验证 Python 依赖

```bash
pip show langchain-openai
pip show openai
```

确保已安装正确版本的依赖包。

---

## 附录：API 调用示例

### 直接调用 DeepSeek API（不经过 RAG）

```python
from llm import chat_directly, get_llm

# 方式一：直接对话
response = chat_directly("你好，介绍一下你自己")
print(response)

# 方式二：自定义 LLM 实例
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model="deepseek-chat",
    temperature=0.7,
    streaming=True,
)

response = llm.invoke("用Python写一个Hello World")
print(response.content)
```

### 自定义 Prompt 模板

```python
from qa_chain import PROMPT_TEMPLATES

# 查看所有可用模板
for name, template in PROMPT_TEMPLATES.items():
    print(f"{name}: {template[:50]}...")
```

---

*最后更新时间：2026-05-16*
