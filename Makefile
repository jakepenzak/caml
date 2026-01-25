.PHONY: help venv docs docs-refresh tests coverage marimo

help:
	@printf "Usage: make <target> \n\n"
	@printf "Targets:\n"
	@printf "   venv            Create/sync virtual environment and install hooks.\n"
	@printf "   docs            Build and preview project documentation.\n"
	@printf "   docs-refresh    Build and preview project documentation, re-rendering all files.\n"
	@printf "   tests           Run tests with pytest.\n"
	@printf "   coverage        Load coverage report in browser.\n"
	@printf "   marimo          Launch marimo notebooks for development.\n"
	@printf "   help            Show this help message.\n"

venv:
	@echo "Syncing venv based on lock file..."
	@uv sync --all-extras --all-groups --frozen
	@uv run pre-commit install

docs:
	@echo "Building documentation..."
	@uv run quartodoc build --config docs/_quarto.yml && uv run quarto preview docs/

docs-refresh:
	@echo "Building documentation..."
	@uv run quartodoc build --config docs/_quarto.yml && uv run quarto preview docs/ --render all

tests:
	@echo "Running tests with pytest..."
	@uv run pytest tests/

coverage:
	@echo "Loading coverage report in browser..."
	@uv run python -m http.server -d tests/reports/htmlcov

marimo:
	@echo "Launching marimo notebook server..."
	@uv run marimo edit
