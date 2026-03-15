#!/bin/bash

echo "=========================================="
echo "  ChatMed-VHI Environment Setup"
echo "=========================================="

# 检查Python
if ! command -v python &> /dev/null; then
    echo "Error: Python not found!"
    exit 1
fi

echo "Python version: $(python --version)"

# 创建虚拟环境
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python -m venv venv
else
    echo "Virtual environment already exists."
fi

# 激活虚拟环境
echo "Activating virtual environment..."
source venv/bin/activate

# 升级pip
echo "Upgrading pip..."
pip install --upgrade pip

# 安装依赖
if [ -f "requirements.txt" ]; then
    echo "Installing dependencies..."
    pip install -r requirements.txt
else
    echo "Warning: requirements.txt not found!"
fi

echo ""
echo "=========================================="
echo "  Setup Complete!"
echo "  Activate: source venv/bin/activate"
echo "=========================================="
