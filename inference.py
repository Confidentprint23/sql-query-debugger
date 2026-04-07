"""
inference.py — Baseline Inference Script for SQL Query Debugger OpenEnv
========================================================================

MANDATORY ENVIRONMENT VARIABLES:
  API_BASE_URL   The API endpoint for the LLM.
  MODEL_NAME     The model identifier to use for inference.
  HF_TOKEN       Your Hugging Face / API key.
  SERVER_URL     URL of the running OpenEnv server (default: http://localhost:7860)

STDOUT FORMAT (strictly followed):
  [START] task=<task_name> env=<benchmark> model=<model_name>
  [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
  [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>

Run:
  python inference.py
  MY_ENV_TASK=medium python inference.py
"""

import os
import sys
import textwrap
from typing import List, Optional

import httpx
from openai import OpenAI

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

API_KEY      = os.getenv("HF_TOKEN") or os.getenv("API_KEY", "")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "Qwen/Qwen2.5-72B-Instruct")
SERVER_URL   = os.getenv("SERVER_URL",   "http://localhost:7860").rstrip("/")
TASK_NAME    = os.getenv("MY_ENV_TASK",  "easy")   # easy | medium | hard
BENCHMARK    = "sql-query-debugger"
MAX_STEPS    = 8

# ---------------------------------------------------------------------------
# Logging helpers (exact competition format)
# ---------------------------------------------------------------------------

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    # Escape newlines in action so everything stays on one line
    action_safe = action.replace("\n", " ").replace("\r", "")
    error_val   = error if error else "null"
    done_val    = str(done).lower()
    print(
        f"[STEP] step={step} action={action_safe!r} "
        f"reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.3f} rewards={rewards_str}",
        flush=True,
    )

# ---------------------------------------------------------------------------
# Environment HTTP helpers
# ---------------------------------------------------------------------------

def env_reset(task: str) -> dict:
    r = httpx.post(f"{SERVER_URL}/reset", json={"task": task}, timeout=30)
    r.raise_for_status()
    return r.json()


def env_step(query: str) -> dict:
    r = httpx.post(f"{SERVER_URL}/step", json={"action": {"query": query}}, timeout=30)
    r.raise_for_status()
    return r.json()

# ---------------------------------------------------------------------------
# System prompt & user prompt builders
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are an expert SQL debugger. You will be given:
    1. A description of what the query should return.
    2. A broken SQL query.
    3. The database schema (CREATE TABLE statements).
    4. (Optionally) the result or error from your previous attempt.

    Your job: output ONLY the corrected SQL query — no explanations, no markdown
    fences, no commentary. Just the raw SQL ending with a semicolon.
    """
).strip()


def build_user_prompt(obs: dict, history: List[str]) -> str:
    hist_block = "\n".join(history[-4:]) if history else "None"
    last_result = obs.get("last_execution_result") or "None yet"
    last_query  = obs.get("last_query") or "None yet"

    return textwrap.dedent(
        f"""
        TASK: {obs['task_description']}

        SCHEMA:
        {obs['schema_info']}

        ORIGINAL BROKEN QUERY:
        {obs['broken_query']}

        YOUR LAST SUBMITTED QUERY:
        {last_query}

        RESULT / ERROR FROM LAST QUERY:
        {last_result}

        PREVIOUS ATTEMPTS SUMMARY:
        {hist_block}

        Output the corrected SQL query now:
        """
    ).strip()

# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------

def get_fixed_query(client: OpenAI, obs: dict, history: List[str]) -> str:
    user_prompt = build_user_prompt(obs, history)
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_prompt},
            ],
            temperature=0.2,
            max_tokens=300,
            stream=False,
        )
        raw = (completion.choices[0].message.content or "").strip()
        # Strip accidental markdown fences
        if raw.startswith("```"):
            lines = raw.split("\n")
            raw = "\n".join(
                line for line in lines if not line.strip().startswith("```")
            ).strip()
        return raw if raw else "SELECT 1;"
    except Exception as exc:
        print(f"[DEBUG] LLM call failed: {exc}", flush=True)
        return "SELECT 1;"

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run_task(task: str) -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    rewards:     List[float] = []
    history:     List[str]   = []
    steps_taken: int         = 0
    score:       float       = 0.0
    success:     bool        = False

    log_start(task=task, env=BENCHMARK, model=MODEL_NAME)

    try:
        reset_data = env_reset(task)
        obs = reset_data["observation"]

        for step in range(1, MAX_STEPS + 1):
            if obs.get("done", False):
                break

            query = get_fixed_query(client, obs, history)

            step_data = env_step(query)
            obs       = step_data["observation"]
            reward    = float(step_data.get("reward", 0.0))
            done      = bool(step_data.get("done", False))
            info      = step_data.get("info", {})
            error     = info.get("error")

            rewards.append(reward)
            steps_taken = step

            log_step(step=step, action=query, reward=reward, done=done, error=error)

            history.append(
                f"Step {step}: reward={reward:+.2f} | "
                f"correctness={info.get('correctness', 0):.2f} | "
                f"rows={info.get('rows_returned', 0)}"
            )

            if done:
                break

        # Score = best correctness achieved (max reward step / 1.0)
        score   = max(rewards) if rewards else 0.0
        score   = min(max(score, 0.0), 1.0)
        success = score >= 0.95   # near-perfect answer

    except Exception as exc:
        print(f"[DEBUG] Episode error: {exc}", flush=True)

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


if __name__ == "__main__":
    # Run all three tasks if TASK_NAME == "all", else run the specified one
    tasks_to_run = ["easy", "medium", "hard"] if TASK_NAME == "all" else [TASK_NAME]
    for t in tasks_to_run:
        run_task(t)