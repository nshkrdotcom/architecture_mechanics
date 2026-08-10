.PHONY: test lint gpu-check

test:
	uv run pytest tests -q

lint:
	uv run ruff check src tests

gpu-check:
	uv run python -m architecture_mechanics.device
