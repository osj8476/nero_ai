#!/bin/bash
# scripts/run_cluster.sh
# 사용법:  bash scripts/run_cluster.sh start 10
#         bash scripts/run_cluster.sh stop
#         bash scripts/run_cluster.sh status
#
# N개의 moondream_server 인스턴스를 포트 8000~(8000+N-1)에 띄움.
# 각 서버 로그는 ~/.nero_ai_logs/moondream_<PORT>.log 에 저장.

set -e

LOG_DIR="$HOME/.nero_ai_logs"
PID_DIR="$HOME/.nero_ai_pids"
BASE_PORT=8000

mkdir -p "$LOG_DIR" "$PID_DIR"

action="${1:-status}"
n="${2:-10}"

case "$action" in
  start)
    echo "Launching $n moondream servers starting at :$BASE_PORT"
    for i in $(seq 0 $((n-1))); do
      port=$((BASE_PORT + i))
      pid_file="$PID_DIR/moondream_$port.pid"
      log_file="$LOG_DIR/moondream_$port.log"

      if [ -f "$pid_file" ] && kill -0 "$(cat $pid_file)" 2>/dev/null; then
        echo "  :$port already running (pid $(cat $pid_file))"
        continue
      fi

      nohup python3 -m nero_ai.moondream_server --port "$port" \
        > "$log_file" 2>&1 &
      echo $! > "$pid_file"
      echo "  :$port started (pid $!, log: $log_file)"
    done
    echo ""
    echo "모델 로딩 대기 (약 30~60초). 헬스체크:"
    echo "  curl http://127.0.0.1:$BASE_PORT/health"
    ;;

  stop)
    echo "Stopping all moondream servers..."
    for pid_file in "$PID_DIR"/moondream_*.pid; do
      [ -e "$pid_file" ] || continue
      pid=$(cat "$pid_file")
      if kill -0 "$pid" 2>/dev/null; then
        kill "$pid"
        echo "  killed pid $pid"
      fi
      rm -f "$pid_file"
    done
    ;;

  status)
    echo "moondream servers:"
    any=0
    for pid_file in "$PID_DIR"/moondream_*.pid; do
      [ -e "$pid_file" ] || continue
      any=1
      port=$(basename "$pid_file" .pid | sed 's/moondream_//')
      pid=$(cat "$pid_file")
      if kill -0 "$pid" 2>/dev/null; then
        # 헬스체크 (응답 빠르면 모델 로드 완료)
        if curl -fs -m 1 "http://127.0.0.1:$port/health" >/dev/null 2>&1; then
          echo "  :$port  ready    (pid $pid)"
        else
          echo "  :$port  loading  (pid $pid)"
        fi
      else
        echo "  :$port  dead     (stale pid file)"
      fi
    done
    [ "$any" -eq 0 ] && echo "  (none running)"
    ;;

  *)
    echo "Usage: $0 {start <N>|stop|status}"
    exit 1
    ;;
esac
