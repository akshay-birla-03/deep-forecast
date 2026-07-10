.PHONY: install test lint train clean
install:
	pip install -e ".[dev]"
test:
	pytest -q
lint:
	ruff check src tests
train:
	deepforecast --series 30 --length 240 --epochs 12
clean:
	rm -rf build dist *.egg-info .pytest_cache .ruff_cache
	find . -type d -name __pycache__ -exec rm -rf {} +
