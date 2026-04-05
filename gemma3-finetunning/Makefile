.PHONY: setup login convert train fuse upload test test-unit all clean

setup:
	uv sync

login:
	uv run python cli.py login

convert:
	uv run python cli.py convert

train:
	uv run python cli.py train

fuse:
	uv run python cli.py fuse

upload:
	uv run python cli.py upload

test:
	uv run python cli.py test

test-unit:
	uv run pytest tests/ -v

all: setup login convert train fuse test

clean:
	rm -rf gemma3f-mlx/ adaptersf_gemma3/ new_functiongemma3/
