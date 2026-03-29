@echo off
chcp 65001 > nul
title MppleAI Chatbot

echo ========================================
echo    MppleAI - RNN Chatbot
echo ========================================
echo.

REM 检查虚拟环境是否存在
if not exist "venv\" (
    echo [*] 首次运行，正在创建虚拟环境...
    python -m venv venv
    echo [√] 虚拟环境创建完成
)

REM 激活虚拟环境
call venv\Scripts\activate.bat

REM 安装依赖
echo [*] 检查依赖...
pip install numpy -q

REM 检查是否有 GPU 版本
python -c "import cupy" > nul 2>&1
if errorlevel 1 (
    echo [*] 安装 CPU 版本...
    pip install numpy -q
)

echo [√] 环境准备完成
echo.

REM 运行程序
python main.py

pause