#!/bin/bash

# SenseVoice MLX API 启动脚本

# 设置颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 默认配置
HOST="${SENSEVOICE_MLX_HOST:-0.0.0.0}"
PORT="${SENSEVOICE_MLX_PORT:-8001}"
LOG_LEVEL="${SENSEVOICE_MLX_LOG_LEVEL:-INFO}"

# 打印启动信息
echo -e "${BLUE}========================================${NC}"
echo -e "${GREEN}🚀 SenseVoice MLX API Server${NC}"
echo -e "${BLUE}========================================${NC}"
echo -e "${YELLOW}配置信息:${NC}"
echo -e "  主机: ${HOST}"
echo -e "  端口: ${PORT}"
echo -e "  日志级别: ${LOG_LEVEL}"
echo -e "${BLUE}========================================${NC}"

# 检查虚拟环境
if [ ! -d "venv" ]; then
    echo -e "${RED}❌ 虚拟环境不存在，请先创建虚拟环境${NC}"
    echo -e "${YELLOW}运行: python3 -m venv venv${NC}"
    exit 1
fi

# 激活虚拟环境
echo -e "${YELLOW}激活虚拟环境...${NC}"
source venv/bin/activate

# 检查依赖
echo -e "${YELLOW}检查依赖...${NC}"
python -c "import mlx" 2>/dev/null
if [ $? -ne 0 ]; then
    echo -e "${RED}❌ MLX 未安装${NC}"
    echo -e "${YELLOW}运行: pip install mlx${NC}"
    exit 1
fi

python -c "import fastapi" 2>/dev/null
if [ $? -ne 0 ]; then
    echo -e "${RED}❌ FastAPI 未安装${NC}"
    echo -e "${YELLOW}运行: pip install fastapi uvicorn[standard]${NC}"
    exit 1
fi

# 检查模型文件
MODEL_PATH="${SENSEVOICE_MLX_MODEL_PATH:-/Users/taylor/Documents/code/SenseVoice/model/model_mlx.safetensors}"
if [ ! -f "$MODEL_PATH" ]; then
    echo -e "${RED}❌ 模型文件不存在: $MODEL_PATH${NC}"
    echo -e "${YELLOW}请先运行转换脚本: python convert_mlx_weights.py${NC}"
    exit 1
fi

echo -e "${GREEN}✅ 所有检查通过${NC}"

# 启动服务器
echo -e "${BLUE}========================================${NC}"
echo -e "${GREEN}🎯 启动 MLX API 服务器...${NC}"
echo -e "${BLUE}========================================${NC}"
echo -e "${YELLOW}API 文档: http://${HOST}:${PORT}/docs${NC}"
echo -e "${YELLOW}健康检查: http://${HOST}:${PORT}/health${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# 运行服务器
python openai_whisper_api_mlx.py