#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

if [[ ! -f .env ]]; then
    echo "Missing .env file. Copy .env.example to .env and configure it first." >&2
    exit 1
fi

set -a
source .env
set +a

if ! command -v docker >/dev/null 2>&1; then
    echo "Docker is required to run SearXNG. Install Docker Desktop and try again." >&2
    exit 1
fi

if ! command -v ollama >/dev/null 2>&1; then
    echo "Ollama is required. Install it from https://ollama.com/download and try again." >&2
    exit 1
fi

if ! docker info >/dev/null 2>&1; then
    echo "Docker is not running. Start Docker Desktop and try again." >&2
    exit 1
fi

OLLAMA_PID=""
if ! curl --silent --fail http://localhost:11434/api/tags >/dev/null; then
    ollama serve >/dev/null 2>&1 &
    OLLAMA_PID=$!
    for _ in {1..30}; do
        curl --silent --fail http://localhost:11434/api/tags >/dev/null && break
        sleep 1
    done
    if ! curl --silent --fail http://localhost:11434/api/tags >/dev/null; then
        echo "Ollama did not start at http://localhost:11434." >&2
        exit 1
    fi
fi

cat > .searxng-settings.yml <<'EOF'
use_default_settings: true
server:
  secret_key: "open-deep-research-local-dev-key"
  limiter: false
search:
  formats:
    - html
    - json
EOF

docker run -d --rm --name open-deep-research-searxng \
    -p 8080:8080 \
    -e SEARXNG_SETTINGS_PATH=/etc/searxng/settings.yml \
    -v "$PWD/.searxng-settings.yml:/etc/searxng/settings.yml:ro" \
    searxng/searxng:latest >/dev/null

cleanup() {
    trap - EXIT INT TERM
    docker stop open-deep-research-searxng >/dev/null 2>&1 || true
    rm -rf .searxng-settings.yml .langgraph_api
    if [[ -n "$OLLAMA_PID" ]]; then
        kill "$OLLAMA_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

OLLAMA_MODEL="${OLLAMA_MODEL:-ollama:gemma4:latest}"
OLLAMA_MODEL_NAME="${OLLAMA_MODEL#ollama:}"

export SEARXNG_URL="${SEARXNG_URL:-http://localhost:8080}"
export SEARCH_API="${SEARCH_API:-searxng}"
export MAX_CONCURRENT_RESEARCH_UNITS="${MAX_CONCURRENT_RESEARCH_UNITS:-1}"
export SUMMARIZATION_MODEL="${SUMMARIZATION_MODEL:-$OLLAMA_MODEL}"
export RESEARCH_MODEL="${RESEARCH_MODEL:-$OLLAMA_MODEL}"
export COMPRESSION_MODEL="${COMPRESSION_MODEL:-$OLLAMA_MODEL}"
export FINAL_REPORT_MODEL="${FINAL_REPORT_MODEL:-$OLLAMA_MODEL}"

if ! ollama list | grep -Fq "$OLLAMA_MODEL_NAME"; then
    ollama pull "$OLLAMA_MODEL_NAME"
fi

uvx --refresh --from "langgraph-cli[inmem]" --with-editable . --python 3.11 \
    langgraph dev --config langgraph.local.json --allow-blocking
