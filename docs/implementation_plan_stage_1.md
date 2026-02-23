# Implementation Plan Stage 1: Multi-Task Queue with SQLite

This document contains all components related to background task processing, SQLite queue management, and async worker orchestration extracted from the original implementation plan.

---

## Architecture Overview (Multi-Task)

```
Client → POST /summarize → FastAPI → SQLite (tasks) → Background Worker
Client → GET /info/{uuid} ←─────────────────────────────────────────────┘

Background Worker Pipeline:
  clone_repo → repo_observer → ast_extractor → pyan3 call graph
                    ↓                ↓               ↓
               priority map     module skeletons   call edges
                    └──────────────┴───────────────┘
                                   ↓
                             networkx graph
                                   ↓
                            llm summarizer
                                   ↓
                          update tasks table
```

---

## Project Structure (Task Management Components)

```
gh-summarizer/
├── src/
│   ├── main.py               # FastAPI app with lifespan, task queue routes
│   ├── database.py           # SQLite + aiosqlite, CRUD helpers for tasks
│   ├── models.py             # Task-related Pydantic schemas
│   ├── worker.py             # Background task orchestrator
│   └── ...
├── requirements.txt
├── .env
└── data/
    └── tasks.db              # SQLite database
```

---

## Phase 1 — Database Layer (`database.py`)

### SQLite Schema

```sql
CREATE TABLE IF NOT EXISTS tasks (
    task_id     TEXT PRIMARY KEY,
    task_status TEXT NOT NULL DEFAULT 'pending',
    task_meta   TEXT NOT NULL
);
```

### Task Metadata JSON Structure

`task_meta` JSON shape evolves as the pipeline runs:

```python
# pending state
{ "url": "https://github.com/owner/repo" }

# running state
{ 
  "url": "https://github.com/owner/repo",
  "started_at": "2024-01-15T10:30:00Z",
  "stage": "cloning|observing|extracting|summarizing"
}

# finished state
{ 
  "url": "https://github.com/owner/repo",
  "started_at": "2024-01-15T10:30:00Z",
  "finished_at": "2024-01-15T10:35:00Z",
  "summary": {...},
  "stats": {...},
  "dependency_graph": {...}
}

# failed state
{ 
  "url": "https://github.com/owner/repo",
  "error": "Clone failed: repository not found",
  "stage": "cloning"
}
```

### Database Operations

```python
import aiosqlite
import json
from pathlib import Path

DB_PATH = Path("data/tasks.db")

async def init_db():
    """Initialize SQLite database with tasks table."""
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("""
            CREATE TABLE IF NOT EXISTS tasks (
                task_id     TEXT PRIMARY KEY,
                task_status TEXT NOT NULL DEFAULT 'pending',
                task_meta   TEXT NOT NULL
            )
        """)
        await db.commit()

async def create_task(task_id: str, repo_url: str) -> None:
    """Insert a new pending task."""
    meta = {"url": repo_url}
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "INSERT INTO tasks (task_id, task_status, task_meta) VALUES (?, ?, ?)",
            (task_id, "pending", json.dumps(meta))
        )
        await db.commit()

async def get_task(task_id: str) -> dict | None:
    """Retrieve task by ID."""
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT task_id, task_status, task_meta FROM tasks WHERE task_id = ?",
            (task_id,)
        ) as cursor:
            row = await cursor.fetchone()
            if row:
                return {
                    "task_id": row["task_id"],
                    "task_status": row["task_status"],
                    "task_meta": json.loads(row["task_meta"])
                }
    return None

async def update_task(task_id: str, status: str, meta: dict) -> None:
    """Update task status and metadata."""
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "UPDATE tasks SET task_status = ?, task_meta = ? WHERE task_id = ?",
            (status, json.dumps(meta), task_id)
        )
        await db.commit()

async def list_tasks(limit: int = 100) -> list[dict]:
    """List recent tasks."""
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT task_id, task_status, task_meta FROM tasks ORDER BY rowid DESC LIMIT ?",
            (limit,)
        ) as cursor:
            rows = await cursor.fetchall()
            return [
                {
                    "task_id": row["task_id"],
                    "task_status": row["task_status"],
                    "task_meta": json.loads(row["task_meta"])
                }
                for row in rows
            ]
```

---

## Phase 2 — Task Models (`models.py`)

### API Request/Response Models

```python
from pydantic import BaseModel, HttpUrl, field_validator

class SummarizeRequest(BaseModel):
    repo_url: HttpUrl

    @field_validator("repo_url")
    def must_be_github(cls, v):
        if "github.com" not in str(v):
            raise ValueError("Only GitHub URLs are supported")
        return v

class SummarizeResponse(BaseModel):
    task_id: str
    status: str
    message: str

class InfoResponse(BaseModel):
    task_id: str
    task_status: str
    task_meta: dict
```

---

## Phase 3 — API Endpoints (`main.py`)

### FastAPI Application with Task Queue

```python
from fastapi import FastAPI, HTTPException, BackgroundTasks
from contextlib import asynccontextmanager
import uuid

from .database import init_db, create_task, get_task
from .models import SummarizeRequest, SummarizeResponse, InfoResponse
from .worker import run_pipeline

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize database on startup."""
    await init_db()
    yield

app = FastAPI(
    title="GitHub Repository Summarizer",
    description="Analyze GitHub repositories with background task processing",
    version="0.1.0",
    lifespan=lifespan
)

@app.post("/summarize", response_model=SummarizeResponse)
async def summarize_repo(
    request: SummarizeRequest,
    background_tasks: BackgroundTasks
) -> SummarizeResponse:
    """
    Queue a repository summarization task.
    
    Returns immediately with a task_id. Use GET /info/{task_id} to check status.
    """
    repo_url = str(request.repo_url)
    task_id = str(uuid.uuid4())
    
    # Create task in database
    await create_task(task_id, repo_url)
    
    # Queue background worker
    background_tasks.add_task(run_pipeline, task_id, repo_url)
    
    return SummarizeResponse(
        task_id=task_id,
        status="pending",
        message=f"Task queued for {repo_url}. Use /info/{task_id} to check progress."
    )

@app.get("/info/{task_id}", response_model=InfoResponse)
async def get_task_info(task_id: str) -> InfoResponse:
    """
    Retrieve task status and results.
    
    The task_meta field contains stage information during execution:
    - pending: { "url": "..." }
    - running: { "url": "...", "stage": "cloning|observing|extracting|summarizing", "started_at": "..." }
    - finished: { "url": "...", "summary": {...}, "stats": {...}, "finished_at": "..." }
    - failed: { "url": "...", "error": "...", "stage": "..." }
    """
    task = await get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    
    return InfoResponse(
        task_id=task["task_id"],
        task_status=task["task_status"],
        task_meta=task["task_meta"]
    )

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok"}
```

### Stage Progress Tracking

The polling endpoint exposes the current stage so clients can show progress:

```json
{
  "task_id": "abc-123",
  "task_status": "running",
  "task_meta": {
    "url": "https://github.com/owner/repo",
    "stage": "extracting",
    "started_at": "2024-01-15T10:30:00Z"
  }
}
```

---

## Phase 4 — Worker Orchestrator (`worker.py`)

### Background Task Pipeline

```python
import asyncio
import shutil
from datetime import datetime
from pathlib import Path

from .database import update_task
from .cloner import clone_repo, preflight_size_check
from .repo_observer import observe_repo
from .ast_extractor import extract_file, find_entry_points, compute_stats
from .call_graph import build_call_graph, extract_graph_signals, build_import_graph, find_architectural_layers
from .llm_summarizer import summarize_docs, summarize_code, merge_summaries

async def run_pipeline(task_id: str, repo_url: str):
    """
    Execute the full analysis pipeline as a background task.
    
    Updates the database at each stage to provide progress tracking.
    Cleans up cloned repository on completion or failure.
    """
    repo_path = None
    
    async def update_stage(stage: str):
        """Helper to update task metadata with current stage."""
        await update_task(task_id, "running", {
            "url": repo_url,
            "stage": stage,
            "started_at": datetime.utcnow().isoformat()
        })
    
    try:
        # Stage 1: Preflight check
        await update_stage("preflight")
        await preflight_size_check(repo_url, max_mb=500)
        
        # Stage 2: Clone repository
        await update_stage("cloning")
        repo_path = await clone_repo(repo_url, timeout=180)
        
        # Stage 3: Observe repository structure
        await update_stage("observing")
        observer_result = await observe_repo(str(repo_path))
        
        # Stage 4: Extract AST skeletons
        await update_stage("extracting")
        skeletons = []
        for file_path in observer_result.high_importance:
            skeleton = extract_file(repo_path / file_path, repo_path)
            skeletons.append(skeleton)
        
        # Attach importance scores from observer
        score_map = {
            s.entry.rel_path: s.importance_score 
            for s in observer_result.scored_tree
        }
        for sk in skeletons:
            sk.importance_score = score_map.get(sk.file, 5)
        
        entry_points = find_entry_points(repo_path)
        stats = compute_stats(skeletons, entry_points)
        
        # Stage 5: Build call graph
        await update_stage("call_graph")
        py_files = [str(repo_path / sk.file) for sk in skeletons]
        call_graph = build_call_graph(py_files)
        graph_signals = extract_graph_signals(call_graph)
        import_graph = build_import_graph(skeletons)
        layer_analysis = find_architectural_layers(import_graph)
        
        # Stage 6: LLM summarization
        await update_stage("summarizing")
        readme_text = _read_readme(repo_path)
        doc_summary = await summarize_docs(readme_text)
        code_summary = await summarize_code(skeletons, graph_signals, layer_analysis, stats)
        final_summary = await merge_summaries(repo_url, doc_summary, code_summary, stats)
        
        # Stage 7: Mark as finished
        await update_task(task_id, "finished", {
            "url": repo_url,
            "started_at": datetime.utcnow().isoformat(),
            "finished_at": datetime.utcnow().isoformat(),
            "summary": final_summary,
            "stats": stats.dict(),
            "graph_signals": graph_signals,
            "high_importance_files": observer_result.high_importance,
        })
        
    except Exception as e:
        # Mark as failed with error details
        await update_task(task_id, "failed", {
            "url": repo_url,
            "error": str(e),
            "stage": "unknown"  # Updated by update_stage if error occurs mid-stage
        })
    
    finally:
        # Cleanup: remove cloned repository
        if repo_path and repo_path.exists():
            shutil.rmtree(repo_path, ignore_errors=True)

def _read_readme(repo_path: Path) -> str:
    """Find and read README file."""
    for name in ["README.md", "readme.md", "README", "README.rst"]:
        readme = repo_path / name
        if readme.exists():
            return readme.read_text(encoding="utf-8", errors="ignore")
    return ""
```

### Error Handling and Cleanup

- **Transient failures**: If a stage fails (e.g., network timeout during clone), the task is marked `failed` with error details
- **Resource cleanup**: The `finally` block ensures cloned repositories are always removed
- **Progress tracking**: Each stage update allows clients to monitor progress via polling

---

## Dependencies for Multi-Task System

```
fastapi>=0.95.0
uvicorn>=0.21.0
aiosqlite>=0.19.0
pydantic>=2.0.0
python-dotenv>=1.0.0
httpx>=0.24.0        # for GitHub API preflight check
```

---

## Database Lifecycle

### Initialization

```python
# On application startup (in lifespan context manager)
await init_db()
```

### Task Creation Flow

```
1. Client POSTs to /summarize
2. Server generates UUID task_id
3. create_task() inserts row with status="pending"
4. BackgroundTasks queues run_pipeline()
5. Server returns task_id immediately (202 Accepted)
```

### Task Polling Flow

```
1. Client GETs /info/{task_id}
2. get_task() queries SQLite
3. Returns current status and stage
4. Client repeats until status="finished" or "failed"
```

### Status Transitions

```
pending → running (stage: preflight) → running (stage: cloning) → 
running (stage: observing) → running (stage: extracting) → 
running (stage: call_graph) → running (stage: summarizing) → 
finished

OR

pending → running → failed (with error message)
```

---

## Testing the Queue System

### Manual Testing

```bash
# Start server
uvicorn src.main:app --reload

# Queue a task
curl -X POST http://localhost:8000/summarize \
  -H "Content-Type: application/json" \
  -d '{"repo_url": "https://github.com/psf/requests"}'

# Response: {"task_id": "abc-123", "status": "pending", "message": "..."}

# Poll for status
curl http://localhost:8000/info/abc-123

# Response during execution:
# {"task_id": "abc-123", "task_status": "running", "task_meta": {"stage": "extracting", ...}}

# Response when complete:
# {"task_id": "abc-123", "task_status": "finished", "task_meta": {"summary": {...}, ...}}
```

### Load Testing

The SQLite queue can handle multiple concurrent requests. FastAPI's BackgroundTasks will process them sequentially or in parallel depending on your uvicorn worker configuration:

```bash
# Single worker (sequential processing)
uvicorn src.main:app --workers 1

# Multiple workers (parallel processing, requires separate worker process management)
# Use Celery or similar for true distributed task queue if needed
```

---

## Future Enhancements

### Priority Queue

Add a `priority` column to sort tasks:

```sql
ALTER TABLE tasks ADD COLUMN priority INTEGER DEFAULT 5;
```

### Task Expiration

Add a `created_at` timestamp and periodic cleanup:

```python
async def cleanup_old_tasks(days: int = 7):
    cutoff = datetime.utcnow() - timedelta(days=days)
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "DELETE FROM tasks WHERE created_at < ?",
            (cutoff.isoformat(),)
        )
        await db.commit()
```

### Retry Logic

Track retry attempts in metadata:

```python
{
  "url": "...",
  "error": "Timeout",
  "retry_count": 2,
  "max_retries": 3
}
```

### Distributed Queue

For production scale, migrate from SQLite to Redis + Celery or RabbitMQ for true distributed task processing across multiple worker processes.

---

## Summary

This stage implements a complete async task queue system using:

- **SQLite** for persistent task storage
- **aiosqlite** for async database operations
- **FastAPI BackgroundTasks** for worker orchestration
- **Polling-based status tracking** with stage visibility
- **Graceful error handling** and resource cleanup

The system supports immediate response times (202 Accepted), progress tracking, and scales to handle multiple concurrent analysis requests.
