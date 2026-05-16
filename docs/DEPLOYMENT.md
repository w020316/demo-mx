# MyLibrary RAG - 部署指南

## 目录

1. [系统要求](#1-系统要求)
2. [Windows 部署](#2-windows-部署)
3. [Linux 部署](#3-linux-部署)
4. [macOS 部署](#4-macos-部署)
5. [Docker 部署（实验性）](#5-docker-部署实验性)
6. [生产环境配置](#6-生产环境配置)
7. [性能优化建议](#7-性能优化建议)
8. [故障排除](#8-故障排除)

---

## 1. 系统要求

### 最低配置

| 项目 | 最低要求 | 推荐配置 |
|------|---------|---------|
| 操作系统 | Windows 10 / Ubuntu 20.04 / macOS 12 | Windows 11 / Ubuntu 22.04 / macOS 14 |
| CPU | 2 核 | 4 核+ |
| 内存 | 4 GB RAM | 8 GB+ |
| 磁盘空间 | 500 MB（不含文档） | 2 GB+ |
| Python | 3.10+ | 3.12 |
| 网络 | 可访问 DeepSeek API | 稳定宽带连接 |

### Python 版本兼容性

| Python 版本 | 支持状态 | 备注 |
|------------|---------|------|
| 3.10 | ✅ 支持 | 最低版本 |
| 3.11 | ✅ 支持 | 推荐 |
| 3.12 | ✅ **推荐** | 主要测试版本 |
| 3.13 | ⚠️ 实验性 | 未完全测试 |
| < 3.10 | ❌ 不支持 | 请升级 |

---

## 2. Windows 部署

### 方式一：一键安装向导（推荐新手）

```bash
# 双击运行 setup.bat，按提示完成：
# 1. 检测 Python 环境
# 2. 引导配置 API Key
# 3. 自动安装依赖
# 4. 导入示例文档
```

### 方式二：手动安装

#### Step 1: 安装 Python

1. 访问 https://www.python.org/downloads/
2. 下载 Python 3.12 安装包
3. 运行安装程序，**务必勾选 "Add Python to PATH"**
4. 验证安装：

```bash
python --version
# 应输出：Python 3.12.x
```

#### Step 2: 克隆/下载项目

```bash
git clone https://github.com/w020316/demo-mx.git mylibrary-rag
cd mylibrary-rag
```

#### Step 3: 安装依赖

```bash
pip install --no-cache-dir --target pylibs -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com
```

> 使用 `--target pylibs` 将依赖安装到项目目录，避免污染系统 Python 环境。

#### Step 4: 配置 API Key

```bash
copy .env.example .env
notepad .env
# 填入您的 API Key
```

#### Step 5: 导入文档

```bash
python ingest.py
```

#### Step 6: 启动应用

```bash
start.bat
# 或手动启动
set STREAMLIT_GLOBAL_DEVELOPMENTMODE=false
streamlit run app.py
```

浏览器自动打开 `http://localhost:8501`

---

## 3. Linux 部署

### Ubuntu/Debian

```bash
# 更新系统
sudo apt update && sudo apt upgrade -y

# 安装 Python 和 pip
sudo apt install python3 python3-pip python3-venv git -y

# 克隆项目
git clone https://github.com/w020316/demo-mx.git mylibrary-rag
cd mylibrary-rag

# 创建虚拟环境
python3 -m venv venv
source venv/bin/activate

# 安装依赖
pip install --no-cache-dir -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple/

# 配置环境变量
cp .env.example .env
nano .env  # 填入 API Key

# 导入文档
python ingest.py

# 启动应用
export STREAMLIT_GLOBAL_DEVELOPMENTMODE=false
streamlit run app.py --server.address 0.0.0.0 --server.port 8501
```

### CentOS/RHEL/Fedora

```bash
# 安装依赖
sudo dnf install python3 python3-pip git -y

# 后续步骤同上
```

### 后台运行 (systemd 服务)

创建服务文件 `/etc/systemd/system/mylibrary-rag.service`：

```ini
[Unit]
Description=MyLibrary RAG Service
After=network.target

[Service]
Type=simple
User=your_username
WorkingDirectory=/path/to/mylibrary-rag
Environment="PATH=/path/to/mylibrary-rag/venv/bin"
Environment="STREAMLIT_GLOBAL_DEVELOPMENTMODE=false"
ExecStart=/path/to/mylibrary-rag/venv/bin/streamlit run app.py --server.address 0.0.0.0 --server.port 8501
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

启用并启动服务：

```bash
sudo systemctl daemon-reload
sudo systemctl enable mylibrary-rag
sudo systemctl start mylibrary-rag
sudo systemctl status mylibrary-rag
```

---

## 4. macOS 部署

```bash
# 安装 Homebrew（如果未安装）
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# 安装 Python
brew install python@3.12

# 克隆项目
git clone https://github.com/w020316/demo-mx.git mylibrary-rag
cd mylibrary-rag

# 创建虚拟环境
python3.12 -m venv venv
source venv/bin/activate

# 安装依赖
pip install --no-cache-dir -r requirements.txt

# 配置并启动（同 Linux）
cp .env.example .env
open -e .env  # 用文本编辑器打开
python ingest.py
streamlit run app.py
```

---

## 5. Docker 部署（实验性）

> ⚠️ Docker 部署目前为实验性功能，可能需要额外调整。

创建 `Dockerfile`：

```dockerfile
FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV STREAMLIT_GLOBAL_DEVELOPMENTMODE=false
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.address", "0.0.0.0"]
```

构建和运行：

```bash
docker build -t mylibrary-rag .
docker run -d \
  -p 8501:8501 \
  -v $(pwd)/docs:/app/docs \
  -v $(pwd)/chroma_db:/app/chroma_db \
  -e OPENAI_API_KEY=sk-your-key \
  -e OPENAI_BASE_URL=https://api.deepseek.com/v1 \
  --name mylibrary-rag \
  mylibrary-rag
```

---

## 6. 生产环境配置

### Streamlit 生产配置 (.streamlit/config.toml)

```toml
[global]
developmentMode = false

[server]
enableCORS = true
enableXsrfProtection = false
fileWatcherType = "none"
headless = true
maxUploadSize = 200
maxMessageSize = 100

[browser]
gatherUsageStats = false
serverPort = 8501

[logger]
level = "info"

[client]
showErrorDetails = false
toolbarMode = "minimal"
```

### Nginx 反向代理配置

```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://127.0.0.1:8501;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # WebSocket 超时设置
        proxy_read_timeout 86400;
    }
}
```

### SSL/TLS 配置（使用 Let's Encrypt）

```bash
# 安装 Certbot
sudo apt install certbot python3-certbot-nginx -y

# 获取证书
sudo certbot --nginx -d your-domain.com

# 自动续期
sudo certbot renew --dry-run
```

---

## 7. 性能优化建议

### 内存优化

```python
# config.py 中可调整的内存相关参数
MAX_FILE_SIZE = 100 * 1024 * 1024  # 限制文件大小 100MB
CHROMA_CACHE_DIR = ".cache/chroma"  # ChromaDB 缓存目录
```

### 启动速度优化

- **懒加载**：`app.py` 已实现模块懒加载，首次使用时才导入重量级依赖
- **禁用文件监视器**：`fileWatcherType = "none"` 避免扫描大量文件
- **缓存嵌入模型**：`embeddings.py` 使用单例模式，避免重复加载

### 检索性能优化

| 参数 | 低延迟配置 | 高质量配置 |
|------|-----------|-----------|
| k | 2-3 | 5-7 |
| chunk_size | 300-400 | 600-800 |
| search_type | similarity | mmr |
| fetch_k (MMR) | 10 | 25 |

---

## 8. 故障排除

### 启动失败

```bash
# 查看详细日志
streamlit run app.py --logger.level debug > streamlit.log 2>&1
cat streamlit.log
```

### 常见问题

| 问题 | 可能原因 | 解决方案 |
|------|---------|---------|
| 端口被占用 | 其他进程占用 8501 | `netstat -ano \| findstr 8501` 找到进程并终止 |
| 依赖安装失败 | 网络问题或权限不足 | 使用国内镜像源或 `sudo` |
| 向量数据库损坏 | 异常关闭导致 | 删除 `chroma_db/` 目录重新导入 |
| 页面空白 | JavaScript 加载失败 | 清除浏览器缓存，检查控制台错误 |

### 日志位置

| 日志类型 | 位置 |
|---------|------|
| Streamlit 日志 | `.streamlit/logs/` 或终端输出 |
| 应用日志 | 终端标准输出 |
| 错误日志 | 浏览器开发者工具 Console |

---

*最后更新时间：2026-05-16*
