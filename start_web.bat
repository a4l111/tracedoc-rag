@echo off
setlocal
cd /d "%~dp0"

where python >nul 2>nul
if errorlevel 1 (
  echo [TraceDoc RAG] Python was not found on PATH.
  echo Install Python 3.10 or later and follow README.md first.
  exit /b 1
)

if not exist "app.py" (
  echo [TraceDoc RAG] app.py is missing from %CD%.
  exit /b 1
)

python app.py serve %*
exit /b %ERRORLEVEL%
