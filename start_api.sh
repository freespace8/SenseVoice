#!/bin/bash

# SenseVoice API 后台启动脚本
# 功能：启动/重启 OpenAI 兼容 API 服务

API_SCRIPT="openai_whisper_compatible_api.py"
PID_FILE="api.pid"
LOG_FILE="api.log"
HOST="0.0.0.0"
PORT=8000

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
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

# 检查脚本是否存在
check_api_script() {
    if [ ! -f "$API_SCRIPT" ]; then
        print_error "API脚本 $API_SCRIPT 不存在！"
        exit 1
    fi
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

# 检查进程是否运行（增强版：处理PID文件丢失的情况）
is_running() {
    local running_pid=""
    
    # 方法1：检查PID文件
    if [ -f "$PID_FILE" ]; then
        local pid=$(cat "$PID_FILE")
        if ps -p $pid > /dev/null 2>&1; then
            # 验证这个PID确实是我们的API脚本
            if ps -p $pid -o command= | grep -q "$API_SCRIPT"; then
                return 0  # 进程在运行且是正确的脚本
            else
                # PID被其他进程复用，删除无效的PID文件
                rm -f "$PID_FILE"
            fi
        else
            # PID文件存在但进程不在运行，删除无效的PID文件
            rm -f "$PID_FILE"
        fi
    fi
    
    # 方法2：如果PID文件不存在或无效，通过进程名查找
    running_pid=$(pgrep -f "$API_SCRIPT" | head -1)
    if [ -n "$running_pid" ]; then
        # 找到了运行的进程，重建PID文件
        echo "$running_pid" > "$PID_FILE"
        print_warning "检测到API服务正在运行但PID文件丢失，已重建PID文件 (PID: $running_pid)"
        return 0  # 进程在运行
    fi
    
    # 方法3：检查端口占用（作为最后的验证）
    local port_pid=$(lsof -ti:$PORT 2>/dev/null)
    if [ -n "$port_pid" ]; then
        # 验证占用端口的进程是否是我们的API脚本
        if ps -p $port_pid -o command= | grep -q "$API_SCRIPT"; then
            echo "$port_pid" > "$PID_FILE"
            print_warning "检测到API服务正在运行但PID文件丢失，已重建PID文件 (PID: $port_pid)"
            return 0  # 进程在运行
        fi
    fi
    
    return 1  # 确认进程不在运行
}

# 获取进程PID（增强版：自动恢复丢失的PID文件）
get_pid() {
    # 首先检查PID文件是否存在且有效
    if [ -f "$PID_FILE" ]; then
        local pid=$(cat "$PID_FILE")
        # 验证PID对应的进程是否存在且是正确的脚本
        if ps -p $pid > /dev/null 2>&1 && ps -p $pid -o command= | grep -q "$API_SCRIPT"; then
            echo "$pid"
            return
        else
            # PID文件无效，删除它
            rm -f "$PID_FILE"
        fi
    fi
    
    # 如果PID文件不存在或无效，尝试通过进程名查找
    local running_pid=$(pgrep -f "$API_SCRIPT" | head -1)
    if [ -n "$running_pid" ]; then
        # 重建PID文件
        echo "$running_pid" > "$PID_FILE"
        echo "$running_pid"
    else
        echo ""
    fi
}

# 停止API服务
stop_api() {
    print_info "正在停止API服务..."
    
    # 首先尝试从PID文件停止
    if [ -f "$PID_FILE" ]; then
        local pid=$(cat "$PID_FILE")
        if ps -p $pid > /dev/null 2>&1; then
            print_info "正在终止进程 $pid..."
            kill $pid
            
            # 等待进程结束
            local count=0
            while ps -p $pid > /dev/null 2>&1 && [ $count -lt 10 ]; do
                sleep 1
                count=$((count + 1))
            done
            
            # 如果进程仍在运行，强制杀死
            if ps -p $pid > /dev/null 2>&1; then
                print_warning "进程未正常结束，强制终止..."
                kill -9 $pid
            fi
            
            print_success "进程 $pid 已终止"
        fi
        rm -f "$PID_FILE"
    fi
    
    # 查找并停止可能遗留的进程
    local pids=$(pgrep -f "$API_SCRIPT")
    if [ -n "$pids" ]; then
        print_info "发现遗留进程，正在清理..."
        echo "$pids" | xargs kill -9 2>/dev/null || true
        print_success "遗留进程已清理"
    fi
    
    # 检查端口占用
    local port_pid=$(lsof -ti:$PORT 2>/dev/null)
    if [ -n "$port_pid" ]; then
        print_warning "端口 $PORT 被进程 $port_pid 占用，正在释放..."
        kill -9 $port_pid 2>/dev/null || true
    fi
}

# 启动API服务
start_api() {
    print_info "正在启动API服务..."
    
    # 检查虚拟环境
    detect_venv
    
    # 检查Python环境和依赖
    if ! $PYTHON_CMD -c "import fastapi, funasr, torch, torchaudio" 2>/dev/null; then
        print_error "Python环境缺少必要依赖，请检查 fastapi, funasr, torch, torchaudio 是否已安装"
        exit 1
    fi
    
    # 启动服务
    print_info "使用Python命令: $PYTHON_CMD"
    print_info "启动服务在 http://$HOST:$PORT"
    
    nohup $PYTHON_CMD "$API_SCRIPT" > "$LOG_FILE" 2>&1 &
    local pid=$!
    echo $pid > "$PID_FILE"
    
    # 等待服务启动
    sleep 3
    
    # 检查服务是否成功启动
    if ps -p $pid > /dev/null 2>&1; then
        print_success "API服务已启动 (PID: $pid)"
        print_info "日志文件: $LOG_FILE"
        print_info "访问地址: http://$HOST:$PORT"
        print_info "API文档: http://$HOST:$PORT/docs"
        
        # 测试服务是否响应
        sleep 2
        if curl -s "http://localhost:$PORT/health" > /dev/null 2>&1; then
            print_success "服务健康检查通过"
        else
            print_warning "服务可能还在初始化中，请稍后检查"
        fi
    else
        print_error "API服务启动失败，请检查日志: $LOG_FILE"
        rm -f "$PID_FILE"
        exit 1
    fi
}

# 显示服务状态
status_api() {
    if is_running; then
        local pid=$(get_pid)
        print_success "API服务正在运行 (PID: $pid)"
        print_info "访问地址: http://$HOST:$PORT"
        print_info "日志文件: $LOG_FILE"
        
        # 显示内存和CPU使用情况
        if command -v ps > /dev/null; then
            local stats=$(ps -p $pid -o pid,pcpu,pmem,etime --no-headers 2>/dev/null)
            if [ -n "$stats" ]; then
                print_info "资源使用: $stats"
            fi
        fi
    else
        print_warning "API服务未运行"
    fi
}

# 重启服务
restart_api() {
    print_info "重启API服务..."
    stop_api
    sleep 2
    start_api
}

# 显示日志
show_logs() {
    if [ -f "$LOG_FILE" ]; then
        if [ "$1" = "-f" ]; then
            print_info "实时显示日志 (Ctrl+C 退出):"
            tail -f "$LOG_FILE"
        else
            print_info "显示最近的日志:"
            tail -n 50 "$LOG_FILE"
        fi
    else
        print_warning "日志文件不存在: $LOG_FILE"
    fi
}

# 显示帮助信息
show_help() {
    echo "SenseVoice API 管理脚本"
    echo ""
    echo "用法: $0 {start|stop|restart|status|logs|help}"
    echo ""
    echo "命令说明:"
    echo "  start     启动API服务"
    echo "  stop      停止API服务"
    echo "  restart   重启API服务"
    echo "  status    显示服务状态"
    echo "  logs      显示日志"
    echo "  logs -f   实时显示日志"
    echo "  help      显示帮助信息"
    echo ""
    echo "服务信息:"
    echo "  端口: $PORT"
    echo "  主机: $HOST"
    echo "  API文档: http://$HOST:$PORT/docs"
    echo "  健康检查: http://$HOST:$PORT/health"
}

# 主函数
main() {
    # 检查API脚本是否存在
    check_api_script
    
    case "$1" in
        start)
            if is_running; then
                print_warning "API服务已在运行 (PID: $(get_pid))"
                print_info "如需重启，请使用: $0 restart"
            else
                start_api
            fi
            ;;
        stop)
            if is_running; then
                stop_api
                print_success "API服务已停止"
            else
                print_warning "API服务未运行"
            fi
            ;;
        restart)
            restart_api
            ;;
        status)
            status_api
            ;;
        logs)
            show_logs "$2"
            ;;
        help|--help|-h)
            show_help
            ;;
        "")
            # 默认行为：如果服务在运行则重启，否则启动
            if is_running; then
                print_info "检测到服务正在运行，执行重启..."
                restart_api
            else
                start_api
            fi
            ;;
        *)
            print_error "未知命令: $1"
            show_help
            exit 1
            ;;
    esac
}

# 执行主函数
main "$@"