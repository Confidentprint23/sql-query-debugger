"""
server/app.py — FastAPI server for the SQL Query Debugger OpenEnv environment.

Endpoints:
  POST /reset          — start a new episode
  POST /step           — submit one SQL action
  GET  /state          — return full internal state
  GET  /health         — liveness probe
  GET  /tasks          — list available tasks and metadata
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Adjust import path when running from repo root vs inside server/
try:
    from server.environment import SQLDebugEnvironment, TASKS, _SCHEMA_DDL
except ModuleNotFoundError:
    from environment import SQLDebugEnvironment, TASKS, _SCHEMA_DDL  # type: ignore

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="SQL Query Debugger — OpenEnv",
    description=(
        "An OpenEnv-compliant environment where an AI agent debugs broken SQL queries "
        "against a real SQLite database. Three tasks: easy → medium → hard."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Single global environment instance (one session at a time)
_env: SQLDebugEnvironment = SQLDebugEnvironment()

# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class ResetRequest(BaseModel):
    task: Optional[str] = "easy"

class ActionPayload(BaseModel):
    query: str

class StepRequest(BaseModel):
    action: ActionPayload

# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/reset")
def reset(req: ResetRequest = ResetRequest()) -> Dict[str, Any]:
    """Reset the environment for the given task and return the initial observation."""
    task = (req.task or "easy").lower()
    if task not in TASKS:
        raise HTTPException(status_code=400, detail=f"Unknown task '{task}'. Valid: {list(TASKS)}")
    result = _env.reset(task=task)
    return result


@app.post("/step")
def step(req: StepRequest) -> Dict[str, Any]:
    """Submit an SQL query action and receive (observation, reward, done, info)."""
    try:
        result = _env.step(req.action.query)
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    return result


@app.get("/state")
def state() -> Dict[str, Any]:
    """Return the full internal environment state."""
    return _env.state()


@app.get("/tasks")
def list_tasks() -> Dict[str, Any]:
    """List all available tasks with metadata (no expected_rows exposed)."""
    return {
        task_name: {
            "task_id": cfg["task_id"],
            "description": cfg["description"],
            "broken_query": cfg["broken_query"],
            "schema": _SCHEMA_DDL,
        }
        for task_name, cfg in TASKS.items()
    }


# ---------------------------------------------------------------------------
# Dev entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "7860"))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)