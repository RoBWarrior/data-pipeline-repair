from pydantic import BaseModel
from typing import Any, Dict, List, Optional


class PipelineAction(BaseModel):
    """What the agent can do to fix the pipeline."""
    command: str  
    # Available commands:
    # "cast_column"     → fix column data type
    # "fill_nulls"      → fill missing values
    # "drop_duplicates" → remove duplicate rows
    # "rename_column"   → rename a column
    # "fix_dates"       → standardize date format
    # "drop_column"     → drop an unwanted column
    # "filter_rows"     → remove rows matching condition
    # "done"            → agent signals task complete

    parameters: Dict[str, Any] = {}
    # Examples:
    # cast_column:     {"column": "age", "dtype": "int"}
    # fill_nulls:      {"column": "price", "strategy": "mean"}
    # drop_duplicates: {"subset": ["order_id"]}
    # rename_column:   {"old": "nm", "new": "name"}
    # fix_dates:       {"column": "date", "format": "%d/%m/%Y"}
    # filter_rows:     {"column": "age", "condition": "gt", "value": 0}


class PipelineObservation(BaseModel):
    """What the agent sees after each step."""
    task_id: str
    goal: str                          # what needs to be fixed
    current_data_sample: List[Dict]    # first 5 rows of current data
    columns: List[str]                 # column names
    dtypes: Dict[str, str]             # column -> dtype
    null_counts: Dict[str, int]        # column -> null count
    total_rows: int
    duplicate_count: int
    errors: List[str]                  # list of current issues
    step_number: int
    max_steps: int
    score_so_far: float                # current score 0.0-1.0
    done: bool


class PipelineState(BaseModel):
    """Internal state — returned by /state endpoint."""
    task_id: str
    step_number: int
    max_steps: int
    total_rows: int
    valid_rows: int
    score: float
    done: bool
    actions_taken: List[str]