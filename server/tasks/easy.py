import pandas as pd
import numpy as np
import math

EASY_TASK_ID = "fix_basic_pipeline"
EASY_GOAL = """Fix this broken employee dataset:
1. Column 'age' has values like '25yrs', '30yrs' — strip 'yrs' and cast to int
2. Column 'salary' has ~20% null values — fill with median
3. Column 'email' has some nulls — fill with 'unknown@company.com'
"""

def generate_easy_dataset() -> pd.DataFrame:
    np.random.seed(42)
    n = 50
    ages = [f"{a}yrs" for a in np.random.randint(22, 60, n)]
    salaries = np.random.randint(30000, 120000, n).astype(float)
    emails = [f"user{i}@company.com" for i in range(n)]
    names = [f"Employee_{i}" for i in range(n)]
    departments = np.random.choice(["HR", "Engineering", "Sales", "Finance"], n)
    null_salary_idx = np.random.choice(n, size=10, replace=False)
    for idx in null_salary_idx:
        salaries[idx] = np.nan
    null_email_idx = np.random.choice(n, size=5, replace=False)
    for idx in null_email_idx:
        emails[idx] = None
    return pd.DataFrame({
        "name": names, "age": ages,
        "salary": salaries, "email": emails,
        "department": departments
    })

def grade_easy(df: pd.DataFrame) -> float:
    try:
        n = max(len(df), 1)
        components = []

        # age numeric: 0.1 to 0.4
        try:
            if pd.api.types.is_numeric_dtype(df["age"]):
                components.append(0.4)
            else:
                components.append(0.1)
        except:
            components.append(0.1)

        # salary nulls: 0.1 to 0.35
        try:
            nulls = df["salary"].isnull().sum()
            ratio = 1.0 - (nulls / n)
            components.append(0.1 + 0.25 * ratio)
        except:
            components.append(0.1)

        # email nulls: 0.1 to 0.25
        try:
            nulls = df["email"].isnull().sum()
            ratio = 1.0 - (nulls / n)
            components.append(0.1 + 0.15 * ratio)
        except:
            components.append(0.1)

        score = sum(components)
        # mathematically impossible to be 0 or 1
        # min = 0.1+0.1+0.1 = 0.3
        # max = 0.4+0.35+0.25 = 1.0 → cap at 0.95
        return round(max(0.1, min(0.95, score)), 4)
    except:
        return 0.5

def get_errors(df: pd.DataFrame) -> list:
    errors = []
    try:
        if not pd.api.types.is_numeric_dtype(df["age"]):
            errors.append("'age' column is not numeric — contains strings like '25yrs'")
    except:
        errors.append("'age' column missing or broken")
    try:
        nulls = df["salary"].isnull().sum()
        if nulls > 0:
            errors.append(f"'salary' has {nulls} null values — fill with median")
    except:
        errors.append("'salary' column missing or broken")
    try:
        nulls = df["email"].isnull().sum()
        if nulls > 0:
            errors.append(f"'email' has {nulls} null values — fill with default")
    except:
        errors.append("'email' column missing or broken")
    if not errors:
        errors.append("No errors found! Call 'done' to finish.")
    return errors