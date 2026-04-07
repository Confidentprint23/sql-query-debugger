"""
client.py — HTTP client for the SQL Query Debugger OpenEnv environment.

Usage:
    from client import SQLDebugEnv, SQLDebugAction

    env = SQLDebugEnv(base_url="http://localhost:7860", task="easy")
    obs = env.reset()
    result = env.step(SQLDebugAction(query="SELECT ..."))
    state = env.state()
    env.close()
"""

from __future__ import annotations

import httpx
from typing import Any, Dict, Optional

from models import SQLDebugAction, SQLDebugObservation, SQLDebugState


# ---------------------------------------------------------------------------
# StepResult — thin wrapper returned by step()
# ---------------------------------------------------------------------------

class StepResult:
    def __init__(self, raw: Dict[str, Any]):
        self.observation: SQLDebugObservation = SQLDebugObservation(**raw["observation"])
        self.reward: float = raw["reward"]
        self.done: bool = raw["done"]
        self.info: Dict[str, Any] = raw.get("info", {})

    def __repr__(self) -> str:
        return (
            f"StepResult(reward={self.reward:.3f}, done={self.done}, "
            f"step={self.observation.step_number})"
        )


# ---------------------------------------------------------------------------
# SQLDebugEnv
# ---------------------------------------------------------------------------

class SQLDebugEnv:
    """
    Synchronous HTTP client wrapping the FastAPI server.

    Args:
        base_url: Root URL of the running server (e.g. "http://localhost:7860").
        task:     Task difficulty — "easy" | "medium" | "hard".
        timeout:  HTTP timeout in seconds.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:7860",
        task: str = "easy",
        timeout: float = 30.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.task = task
        self._client = httpx.Client(timeout=timeout)

    # ------------------------------------------------------------------
    # OpenEnv interface
    # ------------------------------------------------------------------

    def reset(self) -> SQLDebugObservation:
        """Reset the environment and return the initial observation."""
        resp = self._client.post(f"{self.base_url}/reset", json={"task": self.task})
        resp.raise_for_status()
        return SQLDebugObservation(**resp.json()["observation"])

    def step(self, action: SQLDebugAction) -> StepResult:
        """Submit an action and return (observation, reward, done, info)."""
        resp = self._client.post(
            f"{self.base_url}/step",
            json={"action": action.model_dump()},
        )
        resp.raise_for_status()
        return StepResult(resp.json())

    def state(self) -> SQLDebugState:
        """Return the full internal state of the environment."""
        resp = self._client.get(f"{self.base_url}/state")
        resp.raise_for_status()
        return SQLDebugState(**resp.json())

    def close(self) -> None:
        """Close the HTTP client."""
        self._client.close()

    # ------------------------------------------------------------------
    # Context manager support
    # ------------------------------------------------------------------

    def __enter__(self) -> "SQLDebugEnv":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()