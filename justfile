venv := ".venv"
python := venv / "bin/python"
pip := venv / "bin/pip"

# Create venv and install dependencies
setup:
    python3 -m venv {{venv}}
    {{pip}} install --upgrade pip
    {{pip}} install -e '.[dev]'

# Run the dev server
run:
    {{python}} -m fastapi dev app/main.py

# Format and lint
lint:
    {{venv}}/bin/ruff check --fix app/ tests/
    {{venv}}/bin/ruff format app/ tests/

# Run tests
test:
    {{python}} -m pytest tests/ -v

# Run unit tests only
test-unit:
    {{python}} -m pytest tests/unit/ -v

# Run integration tests only
test-integration:
    {{python}} -m pytest tests/integration/ -v
