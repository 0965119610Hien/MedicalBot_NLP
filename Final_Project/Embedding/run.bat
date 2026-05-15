@echo off
setlocal
cd /d "%~dp0"

echo.
echo ===============================================
echo  Medical AI Chatbot - Streamlit
echo  Trợ lý Y tế AI
echo ===============================================
echo.
echo Khởi động ứng dụng...
echo URL: http://localhost:8501
echo.

REM Check for API key
if "%GOOGLE_API_KEY%"=="" (
    if not exist ".streamlit\secrets.toml" (
        echo ⚠️  CẢNH BÁO: Chưa cấu hình GOOGLE_API_KEY
        echo.
        echo Hãy tạo file .streamlit/secrets.toml với:
        echo   GOOGLE_API_KEY = "YOUR_KEY_HERE"
        echo.
        pause
    )
)

REM Run Streamlit
"..\.venv\Scripts\python.exe" -m streamlit run app.py

pause
