#!/usr/bin/env bash
# run_public_demo.sh - serve the Ekegusii demo from this machine on a public URL.
#
#   bash serve/run_public_demo.sh
#
# Starts uvicorn on localhost and opens a Cloudflare quick tunnel to it, which
# gives back a public https://<random>.trycloudflare.com address. Nothing is
# installed system-wide and no Cloudflare account is needed.
#
# WHAT THIS IS NOT: permanent. A quick tunnel's hostname is random and changes
# every time this script runs, and the URL dies when this process or the node
# does. Use it for a live demo. For a link that has to keep working - a link in
# a paper - it needs a host that outlives the session.
#
# Loads models from artifacts/ when they are present, so on the training node it
# starts in seconds and needs no Hugging Face token at all.

set -euo pipefail

PORT="${PORT:-8000}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$HERE")"
cd "$HERE"

# ---- models: prefer the local checkpoints over downloading them again -------
LOCAL_MIXED="$ROOT/artifacts/nllb600m-mixed-control"
if [ -d "$LOCAL_MIXED" ]; then
  export HF_MIXED="$LOCAL_MIXED"
  echo "using local weights: $LOCAL_MIXED"
else
  echo "no local checkpoint at $LOCAL_MIXED - will download from the Hub."
  echo "that needs HF_TOKEN in the environment, because the repo is private."
fi

export ENABLED_SYSTEMS="${ENABLED_SYSTEMS:-mixed}"
export MAX_RESIDENT="${MAX_RESIDENT:-1}"
export PRELOAD="${PRELOAD:-1}"
export RATE_LIMIT_PER_MIN="${RATE_LIMIT_PER_MIN:-30}"

# ---- cloudflared: a single static binary, no install, no root --------------
CF="$HERE/.cloudflared"
if [ ! -x "$CF" ]; then
  echo "downloading cloudflared ..."
  ARCH="$(uname -m)"
  case "$ARCH" in
    x86_64)  SUFFIX=amd64 ;;
    aarch64|arm64) SUFFIX=arm64 ;;
    *) echo "unsupported architecture: $ARCH"; exit 1 ;;
  esac
  curl -fsSL -o "$CF" \
    "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-${SUFFIX}"
  chmod +x "$CF"
fi

cleanup() {
  echo
  echo "shutting down ..."
  [ -n "${TUNNEL_PID:-}" ] && kill "$TUNNEL_PID" 2>/dev/null || true
  [ -n "${API_PID:-}" ] && kill "$API_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

# ---- API -------------------------------------------------------------------
echo "starting the API on 127.0.0.1:$PORT ..."
python -m uvicorn app:app --host 127.0.0.1 --port "$PORT" > "$HERE/api.log" 2>&1 &
API_PID=$!

for i in $(seq 1 180); do
  if curl -fsS "http://127.0.0.1:$PORT/api/health" > /dev/null 2>&1; then break; fi
  if ! kill -0 "$API_PID" 2>/dev/null; then
    echo "the API died on startup. Last lines of serve/api.log:"; tail -30 "$HERE/api.log"; exit 1
  fi
  sleep 1
done
echo "API is up (loading the model can take another minute; watch serve/api.log)"

# ---- tunnel ----------------------------------------------------------------
echo "opening the tunnel ..."
# --protocol http2 avoids the QUIC/UDP 7844 path, which is blocked on a lot of
# university and cloud networks and fails with a confusing timeout.
"$CF" tunnel --no-autoupdate --protocol http2 \
      --url "http://127.0.0.1:$PORT" > "$HERE/tunnel.log" 2>&1 &
TUNNEL_PID=$!

URL=""
for i in $(seq 1 60); do
  URL="$(grep -oE 'https://[a-z0-9-]+\.trycloudflare\.com' "$HERE/tunnel.log" | head -1 || true)"
  [ -n "$URL" ] && break
  if ! kill -0 "$TUNNEL_PID" 2>/dev/null; then
    echo "the tunnel died. Last lines of serve/tunnel.log:"; tail -30 "$HERE/tunnel.log"; exit 1
  fi
  sleep 1
done

if [ -z "$URL" ]; then
  echo "no URL appeared in 60s. serve/tunnel.log:"; tail -30 "$HERE/tunnel.log"; exit 1
fi

echo "$URL" > "$HERE/PUBLIC_URL.txt"
echo
echo "======================================================================"
echo "  $URL"
echo "======================================================================"
echo "  also written to serve/PUBLIC_URL.txt"
echo
echo "  This address is temporary. It changes every time this script runs and"
echo "  stops working when this process or this node stops."
echo
echo "  Ctrl-C to stop. Logs: serve/api.log, serve/tunnel.log"
echo

wait "$API_PID"
