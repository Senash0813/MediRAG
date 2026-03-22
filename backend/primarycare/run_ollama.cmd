@echo off
setlocal EnableExtensions EnableDelayedExpansion

cd /d "%~dp0"

if exist ".venv\Scripts\activate.bat" (
	call ".venv\Scripts\activate.bat"
)

if not exist ".env.ollama" (
	echo [ERROR] .env.ollama not found in project root.
	exit /b 1
)

for /f "usebackq tokens=1,* delims==" %%A in (".env.ollama") do (
	set "key=%%~A"
	set "val=%%~B"
	if not "!key!"=="" if /i not "!key:~0,1!"=="#" set "!key!=!val!"
)

if /i "%S2_API_KEY%"=="your_s2_api_key_here" set "S2_API_KEY="
if "%S2_API_KEY%"=="" (
	set /p S2_API_KEY=Enter S2_API_KEY: 
)

echo Starting API with Ollama provider...
echo MEDIRAG_LLM_PROVIDER=%MEDIRAG_LLM_PROVIDER%
echo MEDIRAG_OLLAMA_BASE_URL=%MEDIRAG_OLLAMA_BASE_URL%
echo MEDIRAG_OLLAMA_MODEL=%MEDIRAG_OLLAMA_MODEL%

uvicorn main:app --host 0.0.0.0 --port 8003
