#!/bin/sh
set -e

# Start the Ollama daemon in the background so the embedded model is available.
ollama serve &

# Wait until the Ollama API is responsive before accepting requests.
echo "Waiting for Ollama to start..."
until ollama list > /dev/null 2>&1; do
    sleep 1
done
echo "Ollama is ready."

# Hand off to the FastAPI application.
exec fastapi run app/main.py --host 0.0.0.0 --port 8000
