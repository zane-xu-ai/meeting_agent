#!/usr/bin/env bash
set -euo pipefail

cd -P -- "$(dirname -- "${BASH_SOURCE[0]}")"
export PATH="$HOME/.local/bin:$PATH"
unit=meeting-agent.service
health_url=http://127.0.0.1:19090/api/health

service() {
    systemctl --user "$@"
}

show_health() {
    local result
    result=$(curl --noproxy '*' --fail --silent --max-time 1 "$health_url") || return 1
    printf '健康检查 %s\n' "$result"
}

start_service() {
    if service is-active --quiet "$unit"; then
        service status --no-pager "$unit"
        show_health
        return
    fi
    if [[ "$PWD" != "$(cd "$HOME" && pwd -P)/projects/meeting_agent" ]]; then
        echo '请将项目部署到 ~/projects/meeting_agent，再启动 systemd 服务。' >&2
        return 1
    fi
    for tool in uv ffmpeg ffprobe curl ss; do
        if ! command -v "$tool" >/dev/null; then
            printf '缺少命令 %s，请按照 README 安装。\n' "$tool" >&2
            return 1
        fi
    done
    if [[ -n "$(ss -H -ltn 'sport = :19090')" ]]; then
        echo 'TCP 19090 已被占用。' >&2
        return 1
    fi

    echo '使用 uv 同步锁定的 Python 环境和依赖'
    uv sync --locked
    local unit_path="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user/$unit"
    if ! cmp -s "deploy/$unit" "$unit_path"; then
        mkdir -p -- "$(dirname -- "$unit_path")"
        install -m 644 "deploy/$unit" "$unit_path"
        service daemon-reload
    fi

    echo "正在启动 ${unit}，等待 HTTP 健康检查"
    local since
    since=$(date '+%Y-%m-%d %H:%M:%S')
    journalctl --user -u "$unit" --since "$since" --no-pager --follow 9>&- &
    log_pid=$!
    trap '{ kill "$log_pid"; wait "$log_pid"; } 2>/dev/null || true' EXIT
    trap 'exit 130' INT
    trap 'exit 143' TERM

    if service start "$unit"; then
        local deadline=$((SECONDS + 30))
        while (( SECONDS < deadline )); do
            if service is-active --quiet "$unit" && show_health; then
                service enable "$unit"
                service status --no-pager --lines=0 "$unit"
                echo 'Meeting Agent 已就绪 http://127.0.0.1:19090/'
                return 0
            fi
            if service is-failed --quiet "$unit"; then
                break
            fi
            sleep 1
        done
    fi
    service status --no-pager "$unit" >&2 || true
    service stop "$unit" || true
    echo '启动失败，可用 ./start.sh logs 查看日志。' >&2
    return 1
}

case "${1:-start}" in
    start|restart|stop)
        exec 9>"${XDG_RUNTIME_DIR:?请在 Ubuntu 上以部署用户登录}/meeting-agent.deploy.lock"
        flock 9
        case "${1:-start}" in
            start) start_service ;;
            restart) service stop "$unit"; start_service ;;
            stop) service stop "$unit"; echo 'Meeting Agent 已停止' ;;
        esac
        ;;
    status)
        service status --no-pager "$unit"
        show_health
        ;;
    logs)
        shift
        exec journalctl --user -u "$unit" --no-pager -n 60 "$@"
        ;;
    *)
        echo '用法 ./start.sh [start|restart|stop|status|logs [--follow]]' >&2
        exit 2
        ;;
esac
