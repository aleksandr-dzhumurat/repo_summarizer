# GitHub Repository Summarizer

A tool that clones GitHub repositories, analyzes their structure, and generates LLM-based summaries.

## Setup

### 1. Environment Configuration

Create a `.env` file with required configuration:

```bash
NEBIUS_API_KEY=your_api_key_here
MAX_PROMPT_TOKENS=6000
```

### 2. Create Data Directory

```bash
make prepare-dirs
```

This creates the `data/` directory where cloned repositories will be stored.

### 3. Install Dependencies

```bash
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

## Usage

Start the FastAPI server:

```bash
make serve
```

The server runs on `http://0.0.0.0:8000`


#### API Endpoints

**POST /summarize**
- Request: `{"github_url": "https://github.com/owner/repo"}`
- Response: Returns task_id, status, summary, and statistics
- Analyzes the repository and returns results

**GET /health**
- Returns server health status

#### Test the API

```bash
make test-api
```

This runs a smoke test that checks the health endpoint and submits a summarization request.

## Makefile Commands

| Command | Description |
|---------|-------------|
| `make prepare-dirs` | Create the data directory |
| `make run REPO=<url>` | Run CLI summarizer on a repository |
| `make serve` | Start the FastAPI server with auto-reload |
| `make test-api` | Run API smoke tests |

## Project Structure

- `src/app.py` - FastAPI application
- `src/utils.py` - Utility functions (logging, cloning)
- `src/crawler/` - Repository indexing and skeleton generation
- `src/llm/` - LLM adapter and prompts
- `scripts/main.py` - CLI entry point
- `scripts/test_api.py` - API testing script
- `data/` - Cloned repositories storage

## Workflow

On each request
- process Clone the repository to `data/`
- Index all files
- Generate a code skeleton
- Create an LLM-based summary

test separately
```shell
DATA_DIR=./data PYTHONPATH=. python3 scripts/generate_skeleton.py https://github.com/microsoft/agent-lightning
```


## Features

- Clone and analyze GitHub repositories
- Generate code structure skeletons
- Create AI-powered summaries using DeepSeek LLM
- FastAPI REST API for programmatic access
- CLI script for direct execution

## Why DeepSeek?

DeepSeek-V3.2 was chosen for its excellent performance on code understanding tasks and cost-effectiveness compared to other frontier models.

## Requirements

- Python 3.8+
- Git (for cloning repositories)
- uv package manager
- Valid NEBIUS_API_KEY for LLM access
