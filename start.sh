#!/bin/bash

echo "========================================"
echo "   MppleAI - RNN Chatbot"
echo "========================================"
echo ""

# 检查 Python
if ! command -v python3 &> /dev/null; then
    echo "[错误] 未检测到 Python3"
    exit 1
fi

# 检查依赖
pip3 show numpy &> /dev/null
if [ $? -ne 0 ]; then
    echo "[*] 安装 numpy..."
    pip3 install numpy
fi

echo "[√] 环境准备完成"
echo ""

# 运行程序
python3 main.py

read -p "按回车键退出..."