@echo off
chcp 65001 >nul
cd /d "%~dp0"
title 知识库 RAG 服务（关闭本窗口即停止服务）

if not exist ".env" (
  echo [警告] 未找到 .env 文件，Embedding 与 LLM 可能无法调用。
  echo         请参考 .env.example 配置后重试。
  echo.
)

netstat -ano | findstr "LISTENING" | findstr ":8000 " >nul
if not errorlevel 1 (
  echo [提示] 8000 端口已被占用，服务可能已经在运行。
  echo         请直接访问：http://127.0.0.1:8000
  echo         如需重启：先结束占用 8000 端口的进程，再运行本脚本。
  echo.
  pause
  exit /b 1
)

echo 正在启动知识库 RAG 服务...
echo 访问地址：http://127.0.0.1:8000
echo 停止服务：在本窗口按 Ctrl+C，或直接关闭本窗口
echo ============================================================
python -m uvicorn app:app --host 127.0.0.1 --port 8000

echo.
echo 服务已退出（退出码 %errorlevel%）
pause