#!/bin/bash

# SenseVoice MLX API 后台启动脚本
# 功能：启动/停止/重启 MLX API 服务

# API 配置
API_SCRIPT="openai_whisper_api_mlx.py"
PID_FILE="api_mlx.pid"
LOG_FILE="api_mlx.log"

# 默认配置
HOST="${SENSEVOICE_MLX_HOST:-127.0.0.1}"
PORT="${SENSEVOICE_MLX_PORT:-6209}"
LOG_LEVEL="${SENSEVOICE_MLX_LOG_LEVEL:-INFO}"

# 设置颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# 打印带颜色的信息
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 显示使用帮助
show_help() {
    echo -e "${CYAN}========================================${NC}"
    echo -e "${GREEN}🚀 SenseVoice MLX API 管理脚本${NC}"
    echo -e "${CYAN}========================================${NC}"
    echo ""
    echo "使用方法："
    echo "  $0 [命令]"
    echo ""
    echo "可用命令："
    echo "  start       - 启动API服务（后台运行）"
    echo "  stop        - 停止API服务"
    echo "  restart     - 重启API服务"
    echo "  status      - 查看服务状态"
    echo "  logs        - 查看服务日志"
    echo "  follow      - 实时跟踪日志"
    echo "  test        - 测试API服务"
    echo "  foreground  - 前台运行（调试模式）"
    echo "  help        - 显示此帮助信息"
    echo ""
    echo "环境变量："
    echo "  SENSEVOICE_MLX_HOST    - 监听地址 (默认: 127.0.0.1)"
    echo "  SENSEVOICE_MLX_PORT    - 监听端口 (默认: 6209)"
    echo "  SENSEVOICE_MLX_LOG_LEVEL - 日志级别 (默认: INFO)"
    echo "  SENSEVOICE_ENABLE_PUNCTUATION - 启用标点恢复 (默认: true)"
    echo ""
    echo -e "${CYAN}========================================${NC}"
}

# 检测虚拟环境
detect_venv() {
    # 检查当前是否在虚拟环境中
    if [ -n "$VIRTUAL_ENV" ]; then
        print_info "当前已在虚拟环境中: $VIRTUAL_ENV"
        PYTHON_CMD="python"
        return 0
    fi
    
    # 检查常见的虚拟环境位置
    VENV_PATHS=("venv" "env" ".venv" ".env" "ENV" "python-env")
    
    for venv_path in "${VENV_PATHS[@]}"; do
        if [ -d "$venv_path" ]; then
            if [ -f "$venv_path/bin/activate" ]; then
                print_info "发现虚拟环境: $venv_path"
                source "$venv_path/bin/activate"
                PYTHON_CMD="python"
                return 0
            fi
        fi
    done
    
    # 如果没有找到虚拟环境，使用系统Python
    print_warning "未发现虚拟环境，使用系统Python"
    PYTHON_CMD="python3"
}

# 检查依赖
check_dependencies() {
    detect_venv
    
    # 检查 MLX
    $PYTHON_CMD -c "import mlx" 2>/dev/null
    if [ $? -ne 0 ]; then
        print_error "MLX 未安装"
        echo "请运行: pip install mlx"
        return 1
    fi
    
    # 检查 FastAPI
    $PYTHON_CMD -c "import fastapi" 2>/dev/null
    if [ $? -ne 0 ]; then
        print_error "FastAPI 未安装"
        echo "请运行: pip install fastapi uvicorn[standard]"
        return 1
    fi
    
    # 检查模型文件
    MODEL_PATH="${SENSEVOICE_MLX_MODEL_PATH:-/Users/taylor/Documents/code/SenseVoice/model/model_mlx.safetensors}"
    if [ ! -f "$MODEL_PATH" ]; then
        print_error "模型文件不存在: $MODEL_PATH"
        echo "请先运行转换脚本: python convert_mlx_weights.py"
        return 1
    fi
    
    return 0
}

# 检查进程是否运行
is_running() {
    if [ -f "$PID_FILE" ]; then
        local pid=$(cat "$PID_FILE")
        if ps -p $pid > /dev/null 2>&1; then
            # 验证这个PID确实是我们的API脚本
            if ps -p $pid -o command= | grep -q "$API_SCRIPT"; then
                return 0  # 进程在运行
            fi
        fi
        # PID文件存在但进程不在运行，删除无效的PID文件
        rm -f "$PID_FILE"
    fi
    
    # 通过进程名查找
    local running_pid=$(pgrep -f "$API_SCRIPT" | head -1)
    if [ -n "$running_pid" ]; then
        echo "$running_pid" > "$PID_FILE"
        return 0  # 进程在运行
    fi
    
    return 1  # 进程不在运行
}

# 获取进程PID
get_pid() {
    if [ -f "$PID_FILE" ]; then
        cat "$PID_FILE"
    else
        pgrep -f "$API_SCRIPT" | head -1
    fi
}

# 启动API服务（后台）
start_api() {
    if is_running; then
        print_warning "API服务已经在运行中 (PID: $(get_pid))"
        return 1
    fi
    
    print_info "正在启动 MLX API 服务..."
    
    # 检查依赖
    if ! check_dependencies; then
        print_error "依赖检查失败，无法启动服务"
        return 1
    fi
    
    # 后台启动服务
    detect_venv
    nohup $PYTHON_CMD -u "$API_SCRIPT" > "$LOG_FILE" 2>&1 &
    local pid=$!
    echo $pid > "$PID_FILE"
    
    # 等待服务启动
    print_info "等待服务启动..."
    sleep 3
    
    # 检查服务是否成功启动
    if is_running; then
        print_success "MLX API 服务启动成功 (PID: $pid)"
        print_info "API 文档: http://${HOST}:${PORT}/docs"
        print_info "健康检查: http://${HOST}:${PORT}/health"
        print_info "日志文件: $LOG_FILE"
        return 0
    else
        print_error "服务启动失败，请查看日志: $LOG_FILE"
        return 1
    fi
}

# 停止API服务
stop_api() {
    if ! is_running; then
        print_warning "API服务未运行"
        return 1
    fi
    
    local pid=$(get_pid)
    print_info "正在停止 API 服务 (PID: $pid)..."
    
    kill $pid 2>/dev/null
    sleep 2
    
    # 如果进程仍在运行，强制终止
    if ps -p $pid > /dev/null 2>&1; then
        print_warning "正在强制终止进程..."
        kill -9 $pid 2>/dev/null
    fi
    
    rm -f "$PID_FILE"
    print_success "API 服务已停止"
    return 0
}

# 重启API服务
restart_api() {
    print_info "正在重启 API 服务..."
    stop_api
    sleep 2
    start_api
}

# 查看服务状态
check_status() {
    if is_running; then
        local pid=$(get_pid)
        print_success "API 服务正在运行 (PID: $pid)"
        
        # 尝试调用健康检查端点
        if command -v curl &> /dev/null; then
            print_info "正在检查健康状态..."
            local health_response=$(curl -s "http://${HOST}:${PORT}/health" 2>/dev/null)
            if [ $? -eq 0 ]; then
                echo "$health_response" | $PYTHON_CMD -m json.tool 2>/dev/null || echo "$health_response"
            else
                print_warning "无法连接到健康检查端点"
            fi
        fi
        
        # 显示内存使用
        if command -v ps &> /dev/null; then
            local mem_usage=$(ps -o rss= -p $pid | awk '{printf "%.1f", $1/1024}')
            print_info "内存使用: ${mem_usage} MB"
        fi
        
        return 0
    else
        print_warning "API 服务未运行"
        return 1
    fi
}

# 查看日志
view_logs() {
    if [ ! -f "$LOG_FILE" ]; then
        print_warning "日志文件不存在: $LOG_FILE"
        return 1
    fi
    
    print_info "显示最后 50 行日志:"
    tail -n 50 "$LOG_FILE"
}

# 实时跟踪日志
follow_logs() {
    if [ ! -f "$LOG_FILE" ]; then
        print_warning "日志文件不存在: $LOG_FILE"
        return 1
    fi
    
    print_info "实时跟踪日志 (Ctrl+C 退出):"
    tail -f "$LOG_FILE"
}

# 前台运行（调试模式）
run_foreground() {
    if is_running; then
        print_warning "API服务已经在运行中，请先停止服务"
        return 1
    fi
    
    print_info "前台运行模式（Ctrl+C 退出）"
    
    # 检查依赖
    if ! check_dependencies; then
        print_error "依赖检查失败"
        return 1
    fi
    
    # 显示配置
    echo -e "${CYAN}========================================${NC}"
    echo -e "配置信息:"
    echo -e "  主机: ${HOST}"
    echo -e "  端口: ${PORT}"
    echo -e "  日志级别: ${LOG_LEVEL}"
    echo -e "${CYAN}========================================${NC}"
    
    # 前台运行
    detect_venv
    $PYTHON_CMD "$API_SCRIPT"
}

# 测试API
test_api() {
    if ! is_running; then
        print_warning "API服务未运行，正在启动..."
        start_api
        sleep 3
    fi
    
    print_info "运行 API 测试..."
    detect_venv
    
    if [ -f "test_mlx_api.py" ]; then
        $PYTHON_CMD test_mlx_api.py "http://${HOST}:${PORT}"
    else
        print_warning "测试脚本不存在: test_mlx_api.py"
        print_info "尝试基本连接测试..."
        curl -s "http://${HOST}:${PORT}/health" | $PYTHON_CMD -m json.tool
    fi
}

# 主程序
main() {
    case "$1" in
        start)
            start_api
            ;;
        stop)
            stop_api
            ;;
        restart)
            restart_api
            ;;
        status)
            check_status
            ;;
        logs)
            view_logs
            ;;
        follow)
            follow_logs
            ;;
        test)
            test_api
            ;;
        foreground|debug)
            run_foreground
            ;;
        help|--help|-h)
            show_help
            ;;
        *)
            if [ -z "$1" ]; then
                # 默认行为：显示帮助
                show_help
            else
                print_error "未知命令: $1"
                show_help
                exit 1
            fi
            ;;
    esac
}

# 执行主程序
main "$@"