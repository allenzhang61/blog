#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-AI/chapter-14/mac-server/localai}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8080}"

mkdir -p "$ROOT/models" "$ROOT/backends"

# The shell environment may contain DEBUG=release from other tooling; LocalAI
# expects DEBUG to be a boolean, so unset it for this command.
exec env -u DEBUG local-ai run \
  --address "$HOST:$PORT" \
  --models-path "$ROOT/models" \
  --backends-path "$ROOT/backends"
