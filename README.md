# Data Pipeline Repair Environment

An OpenEnv-compatible environment where AI agents learn to fix broken data pipelines.

## What it does
Real data engineers spend hours fixing broken datasets every week.
This environment simulates that task — giving an agent a broken dataset
and asking it to fix schema errors, missing values, duplicates, and type mismatches.

## Tasks

| Task | Difficulty | Description |
|------|-----------|-------------|
| fix_basic_pipeline | Easy | Fix age column types + fill nulls in employee dataset |
| fix_schema_drift | Medium | Fix dates, revenue types, duplicates, rename columns in sales dataset |
| fix_multi_table_join | Hard | Fix and join two broken tables with multiple issues |

## Action Space

| Command | Parameters | Description |
|---------|-----------|-------------|
| cast_column | column, dtype, strip | Fix column data type |
| fill_nulls | column, strategy/value | Fill missing values |
| drop_duplicates | subset | Remove duplicate rows |
| rename_column | old, new | Rename a column |
| fix_dates | column | Standardize date format |
| strip_column | column | Strip whitespace |
| uppercase_column | column | Uppercase string values |
| replace_values | column, mapping | Replace specific values |
| strip_chars | column, char | Strip characters like $ |
| join_tables | on, how | Join orders + products (hard task) |
| apply_to_secondary | command, parameters | Apply action to second table (hard task) |
| done | {} | Signal task complete |

## Observation Space

Each step returns:
- `goal` — task description
- `current_data_sample` — first 5 rows
- `columns`, `dtypes`, `null_counts` — schema info
- `errors` — list of current issues to fix
- `score_so_far` — current score 0.0–1.0
- `duplicate_count`, `total_rows`, `step_number`

## Reward Function
- Positive reward for each improvement in score
- Small step penalty (-0.01) to encourage efficiency
- Repetition penalty (-0.05) to discourage loops
- Episode ends when score >= 0.99, agent calls done, or max 20 steps reached

## Setup
```bash
pip install -r requirements.txt
uvicorn server.app:app --host 0.0.0.0 --port 7860
```

## Docker
```bash
docker build -t data-pipeline-repair .
docker run -p 7860:7860 data-pipeline-repair
```

## Baseline Scores

| Task | Baseline Score |
|------|---------------|
| fix_basic_pipeline | 0.200 |
| fix_schema_drift | 0.160 |
| fix_multi_table_join | 0.422 |
| Average | 0.261 |

## Environment Variables

| Variable | Description |
|----------|-------------|
| API_BASE_URL | LLM API endpoint |
| MODEL_NAME | Model identifier |
| HF_TOKEN | HuggingFace API key |