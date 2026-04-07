"""
models.py — Typed Pydantic models for the SQL Query Debugger environment.
Action, Observation, State used by both client and server.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Action
# ---------------------------------------------------------------------------

class SQLDebugAction(BaseModel):
    """
    The agent submits a fixed SQL query string.
    The environment executes it against the task database and grades the result.
    """
    query: str = Field(..., description="The SQL query the agent wants to execute and submit as its answer.")


# ---------------------------------------------------------------------------
# Observation
# ---------------------------------------------------------------------------

class SQLDebugObservation(BaseModel):
    """
    What the agent sees after each step.
    """
    task_id: str = Field(..., description="Unique identifier for the current task.")
    task_description: str = Field(..., description="Natural-language description of what the query should return.")
    broken_query: str = Field(..., description="The original broken SQL query the agent must debug.")
    schema_info: str = Field(..., description="DDL / CREATE TABLE statements describing the available tables.")
    last_query: Optional[str] = Field(None, description="The query submitted in the previous step (None on first step).")
    last_execution_result: Optional[str] = Field(
        None,
        description="String representation of rows returned (or error message) from the last submitted query.",
    )
    last_reward: float = Field(0.0, description="Reward received for the previous step.")
    step_number: int = Field(1, description="Current step index (1-based).")
    done: bool = Field(False, description="Whether the episode has ended.")


# ---------------------------------------------------------------------------
# State  (full internal state, returned by state())
# ---------------------------------------------------------------------------

class SQLDebugState(BaseModel):
    """
    Full internal environment state (superset of Observation).
    """
    task_id: str
    task_description: str
    broken_query: str
    schema_info: str
    expected_rows: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Gold-standard rows the correct query must return.",
    )
    current_step: int = 0
    max_steps: int = 8
    done: bool = False
    cumulative_reward: float = 0.0
    last_query: Optional[str] = None
    last_execution_result: Optional[str] = None
    last_reward: float = 0.0