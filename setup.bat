@echo off
chcp 65001 >nul
cd /d "%~dp0"

echo ============================================================
echo   MyLibrary RAG - 首次安装向导
echo ============================================================
echo.

where python >nul 2>&1
if errorlevel 1 (
    echo [错误] 未找到 Python！
    echo.
    echo 请先安装 Python 3.10 或更高版本：
    echo   1. 访问 https://www.python.org/downloads/
    echo   2. 下载并运行安装程序
    echo   3. 安装时务必勾选 "Add Python to PATH"
    echo   4. 安装完成后重新运行此脚本
    echo.
    pause
    exit /b 1
)

for /f "tokens=*" %%i in ('where python') do set PYTHON=%%i
echo [✓] Python 已找到: %PYTHON%

echo.
echo ---- 第 1 步: 配置 API Key ----
if not exist ".env" (
    copy .env.example .env >nul
    echo [!] 已从 .env.example 创建 .env 文件
    echo.
    echo     请编辑 .env 文件，填入您的 DeepSeek API Key：
    echo.
    echo     OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxx
    echo     OPENAI_BASE_URL=https://api.deepseek.com/v1
    echo     MODEL_NAME=deepseek-chat
    echo.
    echo     获取 API Key: https://platform.deepseek.com/api_keys
    echo.
    notepad .env
    echo.
    echo [?] 已完成 .env 配置？
    pause
) else (
    echo [✓] .env 文件已存在
)

echo.
echo ---- 第 2 步: 安装 Python 依赖 ----
echo     依赖将安装到项目目录下的 pylibs 文件夹（不影响系统 Python）
echo.
set PIP_CONFIG_FILE=NUL
%PYTHON% -m pip install --no-cache-dir --target "%~dp0pylibs" -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com
if errorlevel 1 (
    echo.
    echo [错误] 依赖安装失败，请检查网络连接后重试
    pause
    exit /b 1
)
echo [✓] 依赖安装完成

echo.
echo ---- 第 3 步: 导入文档到向量数据库 ----
if exist "chroma_db\chroma.sqlite3" (
    echo [✓] 向量数据库已存在，跳过导入
    echo     如需重新导入，请运行: ingest.bat
) else (
    echo     正在将 docs 目录中的文档向量化...
    echo.
    %PYTHON% ingest.py
    if errorlevel 1 (
        echo.
        echo [!] 文档导入失败，请确认 docs 目录中有 PDF/TXT/MD 文件
        echo     稍后可手动运行: ingest.bat
    )
)

echo.
echo ============================================================
echo   安装完成！
echo.
echo   启动方式: 双击 start.bat
echo   导入文档: 双击 ingest.bat
echo   命令行问答: python query.py
echo.
echo   浏览器将自动打开 http://localhost:8501
echo ============================================================
echo.
pause
