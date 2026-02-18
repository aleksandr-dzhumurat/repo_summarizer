CURRENT_DIR = $(shell pwd)
include .env
export

prepare-dirs:
	mkdir -p ${CURRENT_DIR}/data

run:
	DATA_DIR=${CURRENT_DIR}/data PYTHONPATH=${CURRENT_DIR} python3 scripts/main.py $(REPO)

serve:
	DATA_DIR=${CURRENT_DIR}/data PYTHONPATH=${CURRENT_DIR} uv run uvicorn src.app:app --reload --host 0.0.0.0 --port 8000

test-api:
	DATA_DIR=${CURRENT_DIR}/data PYTHONPATH=${CURRENT_DIR} python3 scripts/test_api.py