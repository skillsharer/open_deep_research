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
SEARXNG_URL="${SEARXNG_URL:-http://localhost:8080}"
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

if [[ ! -f searxng/settings.yml ]]; then
    echo "Missing searxng/settings.yml." >&2
    exit 1
fi

docker run -d --rm --name open-deep-research-searxng \
    -p 8080:8080 \
    -e SEARXNG_SETTINGS_PATH=/etc/searxng/settings.yml \
    -v "$PWD/searxng/settings.yml:/etc/searxng/settings.yml:ro" \
    searxng/searxng:latest >/dev/null

for _ in {1..60}; do
    if curl --silent --fail \
        "$SEARXNG_URL/search?q=python&format=json" >/dev/null; then
        break
    fi
    sleep 1
done
if ! curl --silent --fail \
    "$SEARXNG_URL/search?q=python&format=json" >/dev/null; then
    echo "SearXNG did not start at $SEARXNG_URL." >&2
    exit 1
fi

cleanup() {
    trap - EXIT INT TERM
    docker stop open-deep-research-searxng >/dev/null 2>&1 || true
    if [[ -n "$OLLAMA_PID" ]]; then
        kill "$OLLAMA_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

OLLAMA_MODEL="${OLLAMA_MODEL:-ollama:gemma4:latest}"
OLLAMA_MODEL_NAME="${OLLAMA_MODEL#ollama:}"
EMBEDDING_MODEL="${EMBEDDING_MODEL:-nomic-embed-text}"
EMBEDDING_MODEL_NAME="${EMBEDDING_MODEL#ollama:}"

export SEARXNG_URL
export LANGSMITH_TRACING=false
export LANGCHAIN_TRACING_V2=false
export SEARCH_API="${SEARCH_API:-searxng}"
export MAX_CONCURRENT_RESEARCH_UNITS="${MAX_CONCURRENT_RESEARCH_UNITS:-1}"
export SUMMARIZATION_MODEL="${SUMMARIZATION_MODEL:-$OLLAMA_MODEL}"
export RESEARCH_MODEL="${RESEARCH_MODEL:-$OLLAMA_MODEL}"
export COMPRESSION_MODEL="${COMPRESSION_MODEL:-$OLLAMA_MODEL}"
export FINAL_REPORT_MODEL="${FINAL_REPORT_MODEL:-$OLLAMA_MODEL}"
export EMBEDDING_MODEL="$EMBEDDING_MODEL"

if ! ollama list | grep -Fq "$OLLAMA_MODEL_NAME"; then
    ollama pull "$OLLAMA_MODEL_NAME"
fi
if ! ollama list | grep -Fq "$EMBEDDING_MODEL_NAME"; then
    ollama pull "$EMBEDDING_MODEL_NAME"
fi

if [[ "${1:-}" == "benchmark" ]]; then
    uv run python tests/run_benchmark.py --limit "${BENCHMARK_LIMIT:-3}"
    uv run python tests/evaluate_benchmark.py
    exit 0
fi

uv run streamlit run app.py --browser.gatherUsageStats false
