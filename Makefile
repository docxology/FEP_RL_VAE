.PHONY: install install-dev test test-cov lint format clean help

help:
	@echo "Available targets:"
	@echo "  install      - Install package in production mode"
	@echo "  install-dev  - Install package with dev dependencies"
	@echo "  test         - Run test suite"
	@echo "  test-cov     - Run tests with coverage report"
	@echo "  lint         - Run linters (flake8, mypy)"
	@echo "  format       - Format code (black, isort)"
	@echo "  clean        - Remove build artifacts and cache"

install:
	uv pip install -e .

install-dev:
	uv pip install -e ".[dev]"

test:
	uv run pytest

test-cov:
	uv run pytest --cov=fep_rl_vae --cov-report=html --cov-report=term

lint:
	uv run flake8 src/ tests/ examples/
	uv run mypy src/

format:
	uv run black src/ tests/ examples/
	uv run isort src/ tests/ examples/

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf .pytest_cache
	rm -rf .coverage
	rm -rf htmlcov/
	find . -type d -name __pycache__ -exec rm -r {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
