@echo off
chcp 65001 > nul
title MppleAI Chatbot

echo ========================================
echo    MppleAI - Chatbot
echo ========================================
echo.

echo [*] Activating conda environment...
call conda activate aichat

echo [*] Checking PyTorch...
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')" 2>nul
if errorlevel 1 (
    echo [*] Installing PyTorch...
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 -i https://mirrors.aliyun.com/simple
)

echo [OK] Environment ready
echo.

echo Choose mode:
echo   1 - Train GUI (Tkinter)
echo   2 - Chat GUI (Tkinter)
echo   3 - Train in terminal
echo   4 - Chat in terminal
echo   5 - Train + Chat GUI together
echo.

set /p choice="Choice (1/2/3/4/5): "

if "%choice%"=="1" (
    echo [*] Starting training GUI...
    python train_gui.py
) else if "%choice%"=="2" (
    echo [*] Starting chat GUI...
    python chat_gui.py
) else if "%choice%"=="3" (
    echo [*] Starting terminal training...
    python train_infinite.py
) else if "%choice%"=="4" (
    echo [*] Starting terminal chat...
    python main.py
) else if "%choice%"=="5" (
    echo [*] Starting training GUI...
    start cmd /k "conda activate aichat && python train_gui.py"
    timeout /t 2 /nobreak > nul
    echo [*] Starting chat GUI...
    python chat_gui.py
) else (
    echo Invalid choice
)

pause
