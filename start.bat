@echo off
chcp 65001 >nul
cd /d "%~dp0"

echo ============================================================
echo   MyLibrary RAG 智能问答系统 - 启动脚本
echo ============================================================
echo.

if not exist ".env" (
    echo [错误] 未找到 .env 文件！
    echo 请复制 .env.example 为 .env 并填入您的 API Key
    echo.
    echo   copy .env.example .env
    echo   然后编辑 .env 填入 OPENAI_API_KEY 和 OPENAI_BASE_URL
    pause
    exit /b 1
)

echo [1/2] 检查依赖...
D:\dev-tools\Python312\python.exe -c "import sys;sys.path.insert(0,r'%~dp0pylibs');import langchain, chromadb, streamlit" 2>nul
if errorlevel 1 (
    echo 正在安装依赖到 %~dp0pylibs ...
    set PIP_CONFIG_FILE=NUL
    D:\dev-tools\Python312\python.exe -m pip install --no-cache-dir --target "%~dp0pylibs" -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com
)

echo.
echo [2/2] 启动 Streamlit 应用...
echo.
D:\dev-tools\Python312\python.exe run.py
