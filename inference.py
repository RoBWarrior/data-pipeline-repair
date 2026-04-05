import os
import json
import requests
from openai import OpenAI

# ── Mandatory env vars ────────────────────────────────────────────────
API_BASE_URL  = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME    = os.getenv("MODEL_NAME",   "meta-llama/Llama-3.3-70B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")
ENV_BASE_URL = os.getenv("ENV_BASE_URL", "https://robwarrior-data-pipeline-repair.hf.space")

MAX_STEPS = 15
TASKS = [
    "fix_basic_pipeline",
    "fix_schema_drift",
    "fix_multi_table_join"
]

SYSTEM_PROMPT = """You are a data engineering agent. 
You fix broken datasets by issuing commands.

Available commands:
- cast_column:      {"column": "x", "dtype": "int/float/str", "strip": "optional_suffix"}
- fill_nulls:       {"column": "x", "strategy": "mean/median/mode/ffill/zero"} or {"column": "x", "value": "literal"}
- drop_duplicates:  {"subset": ["col1"]}
- rename_column:    {"old": "old_name", "new": "new_name"}
- fix_dates:        {"column": "x"}
- strip_column:     {"column": "x"}
- uppercase_column: {"column": "x"}
- replace_values:   {"column": "x", "mapping": {"N/A": null, "none": null}}
- strip_chars:      {"column": "x", "char": "$"}
- join_tables:      {"on": "product_code", "how": "inner"}
- apply_to_secondary: {"command": "strip_column", "parameters": {"column": "product_code"}}
- done:             {} (call when all errors are fixed)

Reply with ONLY a JSON object like:
{"command": "fill_nulls", "parameters": {"column": "salary", "strategy": "median"}}

No explanation. No markdown. Just the JSON."""


def log_start(task, model):
    print(f"[START] task={task} env=data-pipeline-repair model={model}", flush=True)

def log_step(step, action, reward, done, error=None):
    error_val = error if error else "null"
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={error_val}", flush=True)

def log_end(success, steps, rewards):
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} rewards={rewards_str}", flush=True)


def reset_env(task_id: str) -> dict:
    r = requests.post(f"{ENV_BASE_URL}/reset", json={"task_id": task_id})
    return r.json()

def step_env(command: str, parameters: dict) -> dict:
    r = requests.post(f"{ENV_BASE_URL}/step", json={"command": command, "parameters": parameters})
    return r.json()


def build_prompt(obs: dict) -> str:
    errors = obs.get("errors", [])
    dtypes = obs.get("dtypes", {})
    nulls  = obs.get("null_counts", {})
    cols   = obs.get("columns", [])
    sample = obs.get("current_data_sample", [])[:3]
    score  = obs.get("score_so_far", 0)
    step   = obs.get("step_number", 0)

    return f"""Step {step} | Score so far: {score}

GOAL:
{obs.get('goal', '')}

CURRENT ERRORS:
{chr(10).join(f'- {e}' for e in errors)}

COLUMNS: {cols}
DTYPES: {dtypes}
NULL COUNTS: {nulls}
SAMPLE ROWS: {json.dumps(sample, default=str)}

What is your next command to fix one of the errors above?
Reply with ONLY valid JSON."""


def run_task(client: OpenAI, task_id: str) -> float:
    log_start(task_id, MODEL_NAME)

    obs = reset_env(task_id)
    rewards = []
    steps_taken = 0
    success = False
    history = []

    try:
        for step in range(1, MAX_STEPS + 1):
            if obs.get("done", False):
                break

            user_prompt = build_prompt(obs)
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                *history[-6:],  # last 3 turns for context
                {"role": "user", "content": user_prompt}
            ]

            # Call LLM
            try:
                completion = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=messages,
                    temperature=0.1,
                    max_tokens=150,
                )
                raw = completion.choices[0].message.content.strip()
            except Exception as e:
                raw = '{"command": "done", "parameters": {}}'

            # Parse action
            try:
                # Strip markdown if model wraps in ```
                if "```" in raw:
                    raw = raw.split("```")[1].replace("json", "").strip()
                action = json.loads(raw)
                command    = action.get("command", "done")
                parameters = action.get("parameters", {})
            except:
                command, parameters = "done", {}

            action_str = f"{command}({json.dumps(parameters)})"

            # Step environment
            result = step_env(command, parameters)
            obs        = result.get("observation", {})
            reward     = result.get("reward", 0.0)
            done       = result.get("done", False)
            error      = None

            rewards.append(reward)
            steps_taken = step

            log_step(step, action_str, reward, done, error)

            history.append({"role": "assistant", "content": raw})
            history.append({"role": "user",      "content": f"reward={reward:.2f} done={done}"})

            if done:
                final_score = obs.get("score_so_far", 0.0)
                success = final_score >= 0.7
                break

    finally:
        log_end(success, steps_taken, rewards)

    return obs.get("score_so_far", 0.0)


def main():
    client = OpenAI(
        base_url=API_BASE_URL,
        api_key=HF_TOKEN or "dummy"
    )

    print("=" * 50)
    print("Data Pipeline Repair — Baseline Inference")
    print("=" * 50)

    all_scores = []
    for task_id in TASKS:
        print(f"\n--- Running task: {task_id} ---")
        score = run_task(client, task_id)
        all_scores.append(score)
        print(f"Final score: {score:.3f}")

    print("\n" + "=" * 50)
    print(f"Average score: {sum(all_scores)/len(all_scores):.3f}")
    print(f"Scores: {all_scores}")
    print("=" * 50)


if __name__ == "__main__":
    main()