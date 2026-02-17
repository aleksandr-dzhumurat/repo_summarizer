CURRENT_DIR = $(shell pwd)
include .env
export

run:
	DATA_DIR=${CURRENT_DIR}/data PYTHONPATH=${CURRENT_DIR} uv run src/main.py