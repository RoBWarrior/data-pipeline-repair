import pandas as pd
import numpy as np
from io import StringIO
import math

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

def safe_score(x):
    if x is None or not math.isfinite(x):
        return 0.5
    x = float(x)
    return max(1e-6, min(1 - 1e-6, x))



def grade_easy(df: pd.DataFrame) -> float:
    try:
        total = max(len(df), 1)
        score = 0.1  # base score — never starts at 0

        # age numeric: +0.30
        try:
            if pd.api.types.is_numeric_dtype(df["age"]):
                score += 0.30
        except: pass

        # salary nulls: +0.30 proportional
        try:
            nulls = df["salary"].isnull().sum()
            score += 0.30 * (1 - nulls / total)
        except: pass

        # email nulls: +0.20 proportional
        try:
            nulls = df["email"].isnull().sum()
            score += 0.20 * (1 - nulls / total)
        except: pass

        # cap strictly between 0 and 1
        return round(max(0.05, min(0.95, score)), 4)
    except:
        return 0.5

def sanitize_score(x):
    try:
        x = float(x)
    except Exception:
        return 0.5
    if not math.isfinite(x):
        return 0.5
    return max(1e-6, min(1 - 1e-6, x))

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