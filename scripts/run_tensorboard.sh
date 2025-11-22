#!/usr/bin/env bash
# Launch TensorBoard for the most recent log run (default: ./logs).
set -euo pipefail

LOG_ROOT=${1:-logs}
PORT=${2:-6006}

# pick latest subdir if user didn't specify a concrete logdir
if [ -d "${LOG_ROOT}" ] && [ $# -eq 0 ]; then
  # list dirs by mtime, pick newest
  LATEST_DIR=$(find "${LOG_ROOT}" -mindepth 1 -maxdepth 1 -type d -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -n1 | cut -d' ' -f2-)
  if [ -n "${LATEST_DIR}" ]; then
    LOGDIR="${LATEST_DIR}"
  else
    LOGDIR="${LOG_ROOT}"
  fi
else
  LOGDIR="${LOG_ROOT}"
fi

if [ ! -d "${LOGDIR}" ]; then
  echo "Log directory not found: ${LOGDIR}" >&2
  exit 1
fi

echo "Starting TensorBoard..."
echo "  logdir : ${LOGDIR}"
echo "  port   : ${PORT}"

echo "Access URLs:"
echo "  localhost : http://localhost:${PORT}"

exec tensorboard --logdir "${LOGDIR}" --port "${PORT}" --bind_all
