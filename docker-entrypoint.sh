#!/bin/sh
set -e

# Start the Ollama daemon in the background so the embedded model is available.
ollama serve &

# Wait until the Ollama API is responsive before accepting requests.
# Exit with an error if Ollama does not start within MAX_WAIT seconds.
echo "Waiting for Ollama to start..."
MAX_WAIT=60
i=0
until ollama list > /dev/null 2>&1; do
    sleep 1
    i=$((i + 1))
    if [ "$i" -ge "$MAX_WAIT" ]; then
        echo "ERROR: Ollama failed to start within ${MAX_WAIT}s" >&2
        exit 1
    fi
done
echo "Ollama is ready."

# Hand off to the FastAPI application.
exec fastapi run app/main.py --host 0.0.0.0 --port 8000
