@echo off
chcp 65001 >nul
cd /d "%~dp0"

echo ============================================================
echo   MyLibrary RAG - 文档向量化工具
echo ============================================================
echo.

where python >nul 2>&1
if errorlevel 1 (
    echo [错误] 未找到 Python！请先安装 Python 3.10+ 并添加到 PATH
    pause
    exit /b 1
)

for /f "tokens=*" %%i in ('where python') do set PYTHON=%%i

echo 用法:
echo   ingest.bat                            默认参数 (chunk_size=500, overlap=50)
echo   ingest.bat --chunk_size 200           设置chunk_size为200
echo   ingest.bat --incremental              增量更新模式
echo.

%PYTHON% ingest.py %*
