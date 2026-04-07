---
title: SQL Query Debugger
emoji: 🛠️
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
tags:
  - openenv
---
# SQL Query Debugger — OpenEnv Environment

An **OpenEnv-compliant** reinforcement-learning environment where an AI agent must
diagnose and fix broken SQL queries against a real in-memory SQLite database.

This simulates a task data engineers perform daily: given a failing or incorrect
query and a database schema, produce the correct query.

---

## Environment Description & Motivation

SQL debugging is a ubiquitous real-world task. Developers and analysts spend
significant time fixing broken queries — wrong column names, bad JOIN conditions,
incorrect aggregation filters, and flawed subqueries. This environment lets agents
practice and be evaluated on that exact skill with clear, deterministic grading.

---

## Action Space

| Field   | Type   | Description                                        |
|---------|--------|----------------------------------------------------|
| `query` | string | A SQL `SELECT` statement submitted as the fix attempt |

```json
{ "action": { "query": "SELECT name, salary FROM employees WHERE dept_id = 1 ORDER BY salary DESC;" } }
```

---

## Observation Space

| Field                    | Type    | Description                                              |
|--------------------------|---------|----------------------------------------------------------|
| `task_id`                | string  | Unique task identifier                                   |
| `task_description`       | string  | What the correct query must return                       |
| `broken_query`           | string  | The original buggy SQL to be fixed                       |
| `schema_info`            | string  | `CREATE TABLE` DDL for all available tables              |
| `last_query`             | string? | The query submitted in the previous step                 |
| `last_execution_result`  | string? | Rows returned or error message from the last query       |
| `last_reward`            | float   | Reward received for the previous step                    |
| `step_number`            | int     | Current step (1-based)                                   |
| `done`                   | bool    | Whether the episode has ended                            |

---

## Reward Function

| Condition                               | Reward         |
|-----------------------------------------|----------------|
| Query fails to execute (SQL error)      | −0.05          |
| Query runs but returns wrong rows       | +0.05 to +0.71 |
| Query runs, ≥50 % rows correct          | +0.43          |
| Query runs, same rows wrong order       | +0.72          |
| Query returns exactly correct rows      | **+1.00**      |

The episode ends immediately on an exact match or after **8 steps**.

---

## Tasks

### easy — Fix Column Name Typo
**Difficulty:** Easy  
**Description:** The query has a typo (`salry` instead of `salary`). Fix it.  
**Database:** `employees` table only  
**Expected baseline score:** ~0.90

### medium — Fix JOIN Column + HAVING Threshold
**Difficulty:** Medium  
**Description:** The JOIN references `emp_id` instead of `dept_id`, and the HAVING threshold is wrong.  
**Database:** `employees JOIN departments`  
**Expected baseline score:** ~0.60

### hard — Fix Correlated Subquery
**Difficulty:** Hard  
**Description:** The subquery references the wrong alias (`e1.emp_id` instead of `e1.dept_id`) and the comparison operator is reversed (`<` instead of `>`).  
**Database:** Self-join on `employees`  
**Expected baseline score:** ~0.30

---

## Database Schema

```sql
CREATE TABLE employees (
    emp_id    INTEGER PRIMARY KEY,
    name      TEXT NOT NULL,
    dept_id   INTEGER NOT NULL,
    salary    REAL NOT NULL,
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
```

---

## Setup & Usage

### Option A — Google Colab (recommended for inference)

See **`colab_setup.ipynb`** or run the cells below:

```python
# 1. Install dependencies
!pip install fastapi uvicorn httpx pydantic openai nest_asyncio

# 2. Clone the repo
!git clone https://huggingface.co/spaces/<your-space>/sql-query-debugger
%cd sql-query-debugger

# 3. Start the server in the background
import subprocess, time
proc = subprocess.Popen(["python", "-m", "uvicorn", "server.app:app",
                         "--host", "0.0.0.0", "--port", "7860"])
time.sleep(3)

# 4. Set env vars and run inference
import os
os.environ["API_BASE_URL"] = "https://router.huggingface.co/v1"
os.environ["MODEL_NAME"]   = "Qwen/Qwen2.5-72B-Instruct"
os.environ["HF_TOKEN"]     = "hf_..."
os.environ["SERVER_URL"]   = "http://localhost:7860"
os.environ["MY_ENV_TASK"]  = "easy"   # or medium / hard / all

!python inference.py
```

### Option B — Local Docker

```bash
# Build
docker build -f server/Dockerfile -t sql-debug-env .

# Run
docker run -p 7860:7860 sql-debug-env

# In another terminal — run inference
export HF_TOKEN=hf_...
export SERVER_URL=http://localhost:7860
python inference.py
```

### Option C — Direct Python (no Docker)

```bash
pip install -r requirements.txt

# Start server
uvicorn server.app:app --host 0.0.0.0 --port 7860

# Run inference (separate terminal)
export HF_TOKEN=hf_...
python inference.py
```

---

## API Reference

| Method | Endpoint  | Body                         | Description                  |
|--------|-----------|------------------------------|------------------------------|
| POST   | `/reset`  | `{"task": "easy"}`           | Start a new episode          |
| POST   | `/step`   | `{"action": {"query": "…"}}` | Submit a SQL fix attempt     |
| GET    | `/state`  | —                            | Full internal environment state |
| GET    | `/tasks`  | —                            | List all tasks + metadata    |
| GET    | `/health` | —                            | Liveness check               |

---

## Baseline Scores

| Task   | Model                       | Score | Steps |
|--------|-----------------------------|-------|-------|
| easy   | Qwen2.5-72B-Instruct        | 0.95  | 1     |
| medium | Qwen2.5-72B-Instruct        | 0.62  | 3     |
| hard   | Qwen2.5-72B-Instruct        | 0.28  | 8     |

---

## Validation

Run the pre-submission validator:

```bash
chmod +x validate-submission.sh
./validate-submission.sh https://<your-space>.hf.space .
```

All three checks (HF Space live, Docker build, openenv validate) must pass.

---

## License

Apache 2.0