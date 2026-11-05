PROJECT_NAME = vietnamese-sign-language-translation
SRC_DIRS = src serving

## Install Python Dependencies (uv-managed project)
.PHONY: requirements
requirements:
	uv sync

## Delete all compiled Python files
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

## Lint using flake8, isort and black (use `make format` to do formatting)
.PHONY: lint
lint:
	uvx flake8 $(SRC_DIRS)
	uvx isort --check --diff --profile black $(SRC_DIRS)
	uvx black --check --config pyproject.toml $(SRC_DIRS)

## Format source code with black
.PHONY: format
format:
	uvx black --config pyproject.toml $(SRC_DIRS)

.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys; \
lines = '\n'.join([line for line in sys.stdin]); \
matches = re.findall(r'\n## (.*)\n[\s\S]+?\n([a-zA-Z_-]+):', lines); \
print('Available rules:\n'); \
print('\n'.join(['{:25}{}'.format(*reversed(match)) for match in matches]))
endef
export PRINT_HELP_PYSCRIPT

help:
	@python3 -c "${PRINT_HELP_PYSCRIPT}" < $(MAKEFILE_LIST)
