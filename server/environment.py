"""
server/environment.py — Core SQL Query Debugger environment logic.

Three tasks of increasing difficulty:
  easy   – Fix a simple syntax error (missing column alias / wrong column name).
  medium – Fix a JOIN with a wrong condition + filter logic bug.
  hard   – Fix a correlated subquery with aggregation and HAVING clause errors.

Reward is shaped continuously, strictly in (0.01, 0.99) — never 0.0 or 1.0.
"""

from __future__ import annotations

import sqlite3
import textwrap
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

_SCHEMA_DDL = textwrap.dedent(
    """
    CREATE TABLE employees (
        emp_id   INTEGER PRIMARY KEY,
        name     TEXT NOT NULL,
        dept_id  INTEGER NOT NULL,
        salary   REAL NOT NULL,
        hire_year INTEGER NOT NULL
    );

    CREATE TABLE departments (
        dept_id   INTEGER PRIMARY KEY,
        dept_name TEXT NOT NULL,
        budget    REAL NOT NULL
    );

    CREATE TABLE projects (
        proj_id   INTEGER PRIMARY KEY,
        proj_name TEXT NOT NULL,
        dept_id   INTEGER NOT NULL,
        cost      REAL NOT NULL
    );
    """
).strip()

_SEED_SQL = textwrap.dedent(
    """
    INSERT INTO departments VALUES
        (1,'Engineering',500000),
        (2,'Marketing',200000),
        (3,'HR',100000);

    INSERT INTO employees VALUES
        (1,'Alice',1,95000,2018),
        (2,'Bob',1,85000,2019),
        (3,'Carol',2,72000,2020),
        (4,'Dave',2,68000,2021),
        (5,'Eve',3,60000,2022),
        (6,'Frank',1,110000,2017),
        (7,'Grace',3,55000,2023);

    INSERT INTO projects VALUES
        (1,'Alpha',1,120000),
        (2,'Beta',1,80000),
        (3,'Gamma',2,50000),
        (4,'Delta',3,30000),
        (5,'Epsilon',1,200000);
    """
).strip()


TASKS: Dict[str, Dict[str, Any]] = {
    "easy": {
        "task_id": "easy_fix_column_name",
        "description": (
            "Return the name and salary of every employee in department 1 (Engineering), "
            "ordered by salary descending. The query has a typo in the column name."
        ),
        "broken_query": textwrap.dedent(
            """
            SELECT name, salry
            FROM employees
            WHERE dept_id = 1
            ORDER BY salary DESC;
            """
        ).strip(),
        "expected_rows": [
            {"name": "Frank", "salary": 110000.0},
            {"name": "Alice", "salary": 95000.0},
            {"name": "Bob",   "salary": 85000.0},
        ],
    },
    "medium": {
        "task_id": "medium_fix_join",
        "description": (
            "Return each department name together with the total salary of its employees. "
            "Only include departments whose total salary exceeds 100 000. "
            "The query joins on the wrong column and the HAVING threshold is wrong."
        ),
        "broken_query": textwrap.dedent(
            """
            SELECT d.dept_name, SUM(e.salary) AS total_salary
            FROM employees e
            JOIN departments d ON e.emp_id = d.dept_id
            GROUP BY d.dept_name
            HAVING SUM(e.salary) > 500000;
            """
        ).strip(),
        "expected_rows": [
            {"dept_name": "Engineering", "total_salary": 290000.0},
            {"dept_name": "HR",          "total_salary": 115000.0},
            {"dept_name": "Marketing",   "total_salary": 140000.0},
        ],
    },
    "hard": {
        "task_id": "hard_fix_subquery",
        "description": (
            "Return the name and salary of employees who earn MORE than the average salary "
            "of their own department. The subquery references the wrong table alias and "
            "the comparison operator is reversed."
        ),
        "broken_query": textwrap.dedent(
            """
            SELECT name, salary
            FROM employees e1
            WHERE salary < (
                SELECT AVG(salary)
                FROM employees e2
                WHERE e2.dept_id = e1.emp_id
            )
            ORDER BY salary DESC;
            """
        ).strip(),
        "expected_rows": [
            {"name": "Frank", "salary": 110000.0},
            {"name": "Carol", "salary": 72000.0},
            {"name": "Eve",   "salary": 60000.0},
        ],
    },
}

MAX_STEPS = 8


def _build_db() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript(_SCHEMA_DDL)
    conn.executescript(_SEED_SQL)
    conn.commit()
    return conn


def _execute(conn: sqlite3.Connection, query: str) -> Tuple[Optional[List[Dict]], Optional[str]]:
    try:
        cur = conn.execute(query)
        rows = [dict(r) for r in cur.fetchall()]
        return rows, None
    except Exception as exc:
        return None, str(exc)


def _rows_match(got: List[Dict], expected: List[Dict]) -> float:
    """Return correctness in (0.02, 0.98) — never exactly 0.0 or 1.0."""
    if not expected:
        return 0.98 if not got else 0.02

    def norm(row: Dict) -> Dict:
        return {
            k.lower(): (round(float(v), 0) if isinstance(v, (float, int)) else v)
            for k, v in row.items()
        }

    got_norm = [norm(r) for r in got]
    exp_norm = [norm(r) for r in expected]

    if got_norm == exp_norm:
        return 0.98

    if sorted(str(r) for r in got_norm) == sorted(str(r) for r in exp_norm):
        return 0.70

    got_set = {str(r) for r in got_norm}
    exp_set = {str(r) for r in exp_norm}
    overlap = len(got_set & exp_set) / len(exp_set)
    if overlap >= 0.5:
        return 0.40
    if overlap > 0.0:
        return 0.20
    return 0.02


class SQLDebugEnvironment:

    def __init__(self) -> None:
        self._task_cfg: Dict[str, Any] = {}
        self._conn: Optional[sqlite3.Connection] = None
        self._step: int = 0
        self._done: bool = False
        self._cumulative_reward: float = 0.0
        self._last_query: Optional[str] = None
        self._last_result: Optional[str] = None
        self._last_reward: float = 0.0

    def reset(self, task: str = "easy") -> Dict[str, Any]:
        if task not in TASKS:
            raise ValueError(f"Unknown task '{task}'. Choose from: {list(TASKS)}")
        self._task_cfg = deepcopy(TASKS[task])
        self._conn = _build_db()
        self._step = 0
        self._done = False
        self._cumulative_reward = 0.0
        self._last_query = None
        self._last_result = None
        self._last_reward = 0.0
        return {"observation": self._make_obs()}

    def step(self, query: str) -> Dict[str, Any]:
        if self._done:
            raise RuntimeError("Episode is done. Call reset() first.")

        self._step += 1
        rows, error = _execute(self._conn, query)  # type: ignore[arg-type]

        if error:
            correctness = 0.0
            result_str = f"ERROR: {error}"
            reward = 0.01
        else:
            result_str = str(rows)
            correctness = _rows_match(rows, self._task_cfg["expected_rows"])  # type: ignore[arg-type]
            reward = 0.05 + 0.90 * correctness
            reward = min(reward, 0.99)
            reward = max(reward, 0.01)

        self._last_query = query
        self._last_result = result_str
        self._last_reward = round(reward, 4)
        self._cumulative_reward += self._last_reward

        done = (correctness >= 0.98) or (self._step >= MAX_STEPS)
        self._done = done

        return {
            "observation": self._make_obs(),
            "reward": self._last_reward,
            "done": done,
            "info": {
                "correctness": correctness,
                "error": error,
                "rows_returned": len(rows) if rows is not None else 0,
            },
        }

    def state(self) -> Dict[str, Any]:
        return {
            "task_id": self._task_cfg.get("task_id", ""),
            "task_description": self._task_cfg.get("description", ""),
            "broken_query": self._task_cfg.get("broken_query", ""),
            "schema_info": _SCHEMA_DDL,
            "expected_rows": self._task_cfg.get("expected_rows", []),
            "current_step": self._step,
            "max_steps": MAX_STEPS,
            "done": self._done,
            "cumulative_reward": round(self._cumulative_reward, 4),
            "last_query": self._last_query,
            "last_execution_result": self._last_result,
            "last_reward": self._last_reward,
        }

    def _make_obs(self) -> Dict[str, Any]:
        return {
            "task_id": self._task_cfg.get("task_id", ""),
            "task_description": self._task_cfg.get("description", ""),
            "broken_query": self._task_cfg.get("broken_query", ""),
            "schema_info": _SCHEMA_DDL,
            "last_query": self._last_query,
            "last_execution_result": self._last_result,
            "last_reward": self._last_reward,
            "step_number": self._step + 1,
            "done": self._done,
        }


def grade_task(task: str, final_query: str) -> float:
    if task not in TASKS:
        raise ValueError(f"Unknown task: {task}")
    conn = _build_db()
    rows, error = _execute(conn, final_query)
    conn.close()
    if error or rows is None:
        return 0.02
    return _rows_match(rows, TASKS[task]["expected_rows"])
