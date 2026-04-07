import pandas as pd
import numpy as np
from io import StringIO

EASY_TASK_ID = "fix_basic_pipeline"

EASY_GOAL = """Fix this broken employee dataset:
1. Column 'age' has values like '25yrs', '30yrs' — strip 'yrs' and cast to int
2. Column 'salary' has ~20% null values — fill with median
3. Column 'email' has some nulls — fill with 'unknown@company.com'
Goal: All rows valid, correct dtypes, no critical nulls.
"""

def generate_easy_dataset() -> pd.DataFrame:
    """Generate a broken dataset for the easy task."""
    np.random.seed(42)
    n = 50

    ages = [f"{a}yrs" for a in np.random.randint(22, 60, n)]
    salaries = np.random.randint(30000, 120000, n).astype(float)
    emails = [f"user{i}@company.com" for i in range(n)]
    names = [f"Employee_{i}" for i in range(n)]
    departments = np.random.choice(["HR", "Engineering", "Sales", "Finance"], n)

    # Inject nulls in salary (~20%)
    null_salary_idx = np.random.choice(n, size=10, replace=False)
    for idx in null_salary_idx:
        salaries[idx] = np.nan

    # Inject nulls in email (~10%)
    null_email_idx = np.random.choice(n, size=5, replace=False)
    for idx in null_email_idx:
        emails[idx] = None

    df = pd.DataFrame({
        "name": names,
        "age": ages,           # broken: "25yrs" instead of 25
        "salary": salaries,    # broken: has nulls
        "email": emails,       # broken: has nulls
        "department": departments
    })

    return df


def grade_easy(df: pd.DataFrame) -> float:
    """
    Grade the fixed dataset. Returns score 0.0 - 1.0.
    
    Scoring:
    - 0.4 → age column is int dtype with no nulls
    - 0.3 → salary has no nulls
    - 0.2 → email has no nulls
    - 0.1 → all row count preserved (no accidental deletions)
    """
    score = 0.0
    total_rows = 50

    # Check 1: age is numeric and clean (0.4)
    try:
        if pd.api.types.is_numeric_dtype(df["age"]):
            null_ages = df["age"].isnull().sum()
            if null_ages == 0:
                score += 0.4
            else:
                score += 0.2  # partial — right type but still has nulls
        else:
            # Maybe they stripped 'yrs' but didn't cast — partial credit
            try:
                df["age"].str.replace("yrs", "").astype(int)
                score += 0.1
            except:
                pass
    except:
        pass

    # Check 2: salary has no nulls (0.3)
    try:
        if df["salary"].isnull().sum() == 0:
            score += 0.3
        else:
            # Partial credit for reducing nulls
            remaining = df["salary"].isnull().sum()
            reduction = max(0, 10 - remaining) / 10
            score += 0.15 * reduction
    except:
        pass

    # Check 3: email has no nulls (0.2)
    try:
        if df["email"].isnull().sum() == 0:
            score += 0.2
        else:
            remaining = df["email"].isnull().sum()
            reduction = max(0, 5 - remaining) / 5
            score += 0.1 * reduction
    except:
        pass

    # Check 4: row count preserved (0.1)
    try:
        if len(df) >= total_rows:
            score += 0.1
        elif len(df) >= total_rows * 0.9:
            score += 0.05
    except:
        pass

    # Ensure score is strictly between 0 and 1
    score = max(0.01, min(0.99, score))
    return round(score, 4)


def get_errors(df: pd.DataFrame) -> list:
    """Return list of current issues in the dataframe."""
    errors = []

    try:
        if not pd.api.types.is_numeric_dtype(df["age"]):
            errors.append("'age' column is not numeric — contains strings like '25yrs'")
        elif df["age"].isnull().sum() > 0:
            errors.append(f"'age' has {df['age'].isnull().sum()} null values")
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