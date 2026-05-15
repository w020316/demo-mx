@echo off
chcp 65001 >nul
cd /d "%~dp0"

echo ============================================================
echo   MyLibrary RAG - 文档向量化工具
echo ============================================================
echo.
echo 用法:
echo   ingest.bat                            默认参数 (chunk_size=500, overlap=50)
echo   ingest.bat --chunk_size 200           设置chunk_size为200
echo   ingest.bat --incremental              增量更新模式
echo.

D:\dev-tools\Python312\python.exe ingest.py %*
