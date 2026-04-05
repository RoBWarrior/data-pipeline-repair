import pandas as pd
import numpy as np
from typing import Optional, Tuple, Any, Dict

from models import (
    PipelineAction, PipelineObservation, PipelineState
)
from server.tasks import (
    generate_easy_dataset, grade_easy, get_easy_errors, EASY_TASK_ID, EASY_GOAL,
    generate_medium_dataset, grade_medium, get_medium_errors, MEDIUM_TASK_ID, MEDIUM_GOAL,
    generate_hard_dataset, grade_hard, get_hard_errors, HARD_TASK_ID, HARD_GOAL
)

MAX_STEPS = 20

class DataPipelineEnvironment:

    def __init__(self):
        self.task_id: str = EASY_TASK_ID
        self.df: Optional[pd.DataFrame] = None
        self.df2: Optional[pd.DataFrame] = None  # only for hard task
        self.step_number: int = 0
        self.actions_taken: list = []
        self.done: bool = False
        self.score: float = 0.0

    # ------------------------------------------------------------------ #
    #  RESET                                                               #
    # ------------------------------------------------------------------ #
    def reset(self, task_id: str = EASY_TASK_ID) -> PipelineObservation:
        self.task_id = task_id
        self.step_number = 0
        self.actions_taken = []
        self.done = False
        self.score = 0.0
        self.df2 = None

        if task_id == EASY_TASK_ID:
            self.df = generate_easy_dataset()
        elif task_id == MEDIUM_TASK_ID:
            self.df = generate_medium_dataset()
        elif task_id == HARD_TASK_ID:
            self.df, self.df2 = generate_hard_dataset()
        else:
            self.df = generate_easy_dataset()

        return self._observe()

    # ------------------------------------------------------------------ #
    #  STEP                                                                #
    # ------------------------------------------------------------------ #
    def step(self, action: PipelineAction) -> Tuple[PipelineObservation, float, bool]:
        if self.done:
            return self._observe(), self.score, True

        self.step_number += 1
        self.actions_taken.append(action.command)

        reward = 0.0
        score_before = self._current_score()

        try:
            self._apply_action(action)
        except Exception as e:
            # Bad action — small penalty, keep going
            pass

        score_after = self._current_score()

        # Reward = improvement in score
        reward = score_after - score_before

        # Penalize repeated same action (loop detection)
        if len(self.actions_taken) >= 3:
            last3 = self.actions_taken[-3:]
            if last3[0] == last3[1] == last3[2]:
                reward -= 0.05

        # Penalize step cost (encourages efficiency)
        reward -= 0.01

        self.score = score_after

        # Check done conditions
        if action.command == "done":
            self.done = True
        elif self.step_number >= MAX_STEPS:
            self.done = True
        elif score_after >= 0.99:
            self.done = True

        reward = round(max(-1.0, min(1.0, reward)), 4)
        return self._observe(), reward, self.done

    # ------------------------------------------------------------------ #
    #  STATE                                                               #
    # ------------------------------------------------------------------ #
    def state(self) -> PipelineState:
        valid_rows = self._count_valid_rows()
        return PipelineState(
            task_id=self.task_id,
            step_number=self.step_number,
            max_steps=MAX_STEPS,
            total_rows=len(self.df) if self.df is not None else 0,
            valid_rows=valid_rows,
            score=self.score,
            done=self.done,
            actions_taken=self.actions_taken
        )

    # ------------------------------------------------------------------ #
    #  APPLY ACTION                                                        #
    # ------------------------------------------------------------------ #
    def _apply_action(self, action: PipelineAction):
        cmd = action.command.lower().strip()
        p = action.parameters

        if cmd == "cast_column":
            col = p.get("column")
            dtype = p.get("dtype", "float")
            strip_str = p.get("strip", "")
            if strip_str:
                self.df[col] = self.df[col].astype(str).str.replace(strip_str, "", regex=False)
            if dtype == "int":
                self.df[col] = pd.to_numeric(self.df[col], errors="coerce").astype("Int64")
            elif dtype == "float":
                self.df[col] = pd.to_numeric(self.df[col], errors="coerce")
            elif dtype == "str":
                self.df[col] = self.df[col].astype(str)

        elif cmd == "fill_nulls":
            col = p.get("column")
            strategy = p.get("strategy", "mean")
            value = p.get("value", None)
            if value is not None:
                self.df[col] = self.df[col].fillna(value)
            elif strategy == "mean":
                self.df[col] = self.df[col].fillna(self.df[col].mean())
            elif strategy == "median":
                self.df[col] = self.df[col].fillna(self.df[col].median())
            elif strategy == "mode":
                self.df[col] = self.df[col].fillna(self.df[col].mode()[0])
            elif strategy == "ffill":
                self.df[col] = self.df[col].ffill()
            elif strategy == "zero":
                self.df[col] = self.df[col].fillna(0)

        elif cmd == "drop_duplicates":
            subset = p.get("subset", None)
            keep = p.get("keep", "first")
            self.df = self.df.drop_duplicates(subset=subset, keep=keep).reset_index(drop=True)

        elif cmd == "rename_column":
            old = p.get("old")
            new = p.get("new")
            self.df = self.df.rename(columns={old: new})

        elif cmd == "fix_dates":
            col = p.get("column")
            # Handle integer dates like 20230115
            if pd.api.types.is_integer_dtype(self.df[col]):
                self.df[col] = pd.to_datetime(self.df[col].astype(str), format="%Y%m%d", errors="coerce")
            else:
                self.df[col] = pd.to_datetime(self.df[col], infer_datetime_format=True, errors="coerce")
            self.df[col] = self.df[col].dt.strftime("%Y-%m-%d")

        elif cmd == "strip_column":
            col = p.get("column")
            self.df[col] = self.df[col].astype(str).str.strip()

        elif cmd == "uppercase_column":
            col = p.get("column")
            self.df[col] = self.df[col].astype(str).str.upper()

        elif cmd == "replace_values":
            col = p.get("column")
            mapping = p.get("mapping", {})
            self.df[col] = self.df[col].replace(mapping)
            # Also cast to numeric if possible after replacement
            self.df[col] = pd.to_numeric(self.df[col], errors="coerce")

        elif cmd == "strip_chars":
            col = p.get("column")
            char = p.get("char", "$")
            self.df[col] = self.df[col].astype(str).str.replace(char, "", regex=False)
            self.df[col] = pd.to_numeric(self.df[col], errors="coerce")

        elif cmd == "join_tables":
            # Hard task only — join orders + products
            if self.df2 is not None:
                on = p.get("on", "product_code")
                how = p.get("how", "inner")
                self.df = pd.merge(self.df, self.df2, on=on, how=how)

        elif cmd == "apply_to_secondary":
            # Hard task — apply action to df2 (products table)
            sub_cmd = p.get("command")
            sub_params = p.get("parameters", {})
            sub_action = PipelineAction(command=sub_cmd, parameters=sub_params)
            # Temporarily swap
            self.df, self.df2 = self.df2, self.df
            self._apply_action(sub_action)
            self.df, self.df2 = self.df2, self.df

        elif cmd == "done":
            pass  # handled in step()

    # ------------------------------------------------------------------ #
    #  HELPERS                                                             #
    # ------------------------------------------------------------------ #
    def _current_score(self) -> float:
        if self.df is None:
            return 0.0
        if self.task_id == EASY_TASK_ID:
            return grade_easy(self.df)
        elif self.task_id == MEDIUM_TASK_ID:
            return grade_medium(self.df)
        elif self.task_id == HARD_TASK_ID:
            return grade_hard(self.df, self.df2 if self.df2 is not None else pd.DataFrame())
        return 0.0

    def _count_valid_rows(self) -> int:
        if self.df is None:
            return 0
        return int(self.df.dropna().shape[0])

    def _observe(self) -> PipelineObservation:
        if self.df is None:
            return PipelineObservation(
                task_id=self.task_id,
                goal="",
                current_data_sample=[],
                columns=[],
                dtypes={},
                null_counts={},
                total_rows=0,
                duplicate_count=0,
                errors=[],
                step_number=self.step_number,
                max_steps=MAX_STEPS,
                score_so_far=0.0,
                done=self.done
            )

        goal = {
            EASY_TASK_ID: EASY_GOAL,
            MEDIUM_TASK_ID: MEDIUM_GOAL,
            HARD_TASK_ID: HARD_GOAL
        }.get(self.task_id, "")

        # Safe sample
        sample = self.df.head(5).copy()
        sample = sample.where(pd.notnull(sample), None)
        sample_records = sample.astype(object).to_dict(orient="records")

        null_counts = self.df.isnull().sum().to_dict()
        null_counts = {k: int(v) for k, v in null_counts.items()}

        dtypes = {col: str(dtype) for col, dtype in self.df.dtypes.items()}

        dup_count = int(self.df.duplicated().sum())

        if self.task_id == EASY_TASK_ID:
            errors = get_easy_errors(self.df)
        elif self.task_id == MEDIUM_TASK_ID:
            errors = get_medium_errors(self.df)
        elif self.task_id == HARD_TASK_ID:
            errors = get_hard_errors(self.df, self.df2 if self.df2 is not None else pd.DataFrame())
        else:
            errors = []

        return PipelineObservation(
            task_id=self.task_id,
            goal=goal,
            current_data_sample=sample_records,
            columns=list(self.df.columns),
            dtypes=dtypes,
            null_counts=null_counts,
            total_rows=len(self.df),
            duplicate_count=dup_count,
            errors=errors,
            step_number=self.step_number,
            max_steps=MAX_STEPS,
            score_so_far=self._current_score(),
            done=self.done
        )