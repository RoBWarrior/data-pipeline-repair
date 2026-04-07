import pandas as pd
import numpy as np

MEDIUM_TASK_ID = "fix_schema_drift"

MEDIUM_GOAL = """Fix this broken sales dataset with multiple issues:
1. Column 'order_date' has mixed formats ('2023/01/15' and '15-01-2023') — standardize to YYYY-MM-DD
2. Column 'revenue' stored as string with '$' signs ('$1200.50') — cast to float
3. Duplicate order_ids exist — remove duplicates keeping first occurrence
4. Column 'region' has nulls (~15%) — fill with 'Unknown'
5. Column named 'cust_nm' should be renamed to 'customer_name'
Goal: Clean schema, no duplicates, correct types, no critical nulls.
"""

def generate_medium_dataset() -> pd.DataFrame:
    """Generate a broken sales dataset for medium task."""
    np.random.seed(123)
    n = 80

    # Mixed date formats
    dates_fmt1 = [f"2023/{np.random.randint(1,13):02d}/{np.random.randint(1,28):02d}" 
                  for _ in range(50)]
    dates_fmt2 = [f"{np.random.randint(1,28):02d}-{np.random.randint(1,13):02d}-2023" 
                  for _ in range(30)]
    all_dates = dates_fmt1 + dates_fmt2
    np.random.shuffle(all_dates)

    # Revenue as strings with $ sign
    revenues = [f"${np.random.uniform(100, 5000):.2f}" for _ in range(n)]

    # Order IDs with duplicates (~10 duplicates)
    order_ids = list(range(1000, 1000 + n))
    dup_indices = np.random.choice(n, size=10, replace=False)
    for idx in dup_indices:
        order_ids[idx] = order_ids[np.random.randint(0, idx + 1)]

    # Region with nulls
    regions = np.random.choice(["North", "South", "East", "West"], n).tolist()
    null_region_idx = np.random.choice(n, size=12, replace=False)
    for idx in null_region_idx:
        regions[idx] = None

    customers = [f"Customer_{i}" for i in range(n)]

    df = pd.DataFrame({
        "order_id": order_ids,
        "cust_nm": customers,        # wrong column name
        "order_date": all_dates,     # mixed formats
        "revenue": revenues,         # string with $
        "region": regions            # has nulls
    })

    return df


def grade_medium(df: pd.DataFrame) -> float:
    """
    Grade the fixed dataset. Returns score 0.0 - 1.0.
    
    Scoring:
    - 0.25 → dates standardized to YYYY-MM-DD format
    - 0.25 → revenue is float dtype
    - 0.2  → no duplicate order_ids
    - 0.15 → no nulls in region
    - 0.15 → column renamed to customer_name
    """
    score = 0.0

    # Check 1: dates standardized (0.25)
    try:
        parsed = pd.to_datetime(df["order_date"], format="%Y-%m-%d", errors="coerce")
        valid_dates = parsed.notna().sum()
        ratio = valid_dates / len(df)
        score += 0.25 * ratio
    except:
        pass

    # Check 2: revenue is float (0.25)
    try:
        if pd.api.types.is_float_dtype(df["revenue"]) or pd.api.types.is_integer_dtype(df["revenue"]):
            score += 0.25
        else:
            # Try to see if values are numeric strings without $
            test = df["revenue"].astype(str).str.replace("$", "").str.strip()
            pd.to_numeric(test, errors="raise")
            score += 0.1  # partial — removed $ but not cast
    except:
        pass

    # Check 3: no duplicate order_ids (0.2)
    try:
        dup_count = df["order_id"].duplicated().sum()
        if dup_count == 0:
            score += 0.2
        else:
            original_dups = 10
            fixed = max(0, original_dups - dup_count)
            score += 0.2 * (fixed / original_dups)
    except:
        pass

    # Check 4: no nulls in region (0.15)
    try:
        nulls = df["region"].isnull().sum()
        if nulls == 0:
            score += 0.15
        else:
            fixed = max(0, 12 - nulls)
            score += 0.15 * (fixed / 12)
    except:
        pass

    # Check 5: column renamed (0.15)
    try:
        if "customer_name" in df.columns:
            score += 0.15
        elif "cust_nm" not in df.columns:
            score += 0.05  # renamed to something, just not exact name
    except:
        pass

    score = max(0.01, min(0.99, score))
    return round(score, 4)


def get_errors(df: pd.DataFrame) -> list:
    errors = []

    try:
        parsed = pd.to_datetime(df["order_date"], format="%Y-%m-%d", errors="coerce")
        bad = parsed.isna().sum()
        if bad > 0:
            errors.append(f"'order_date' has {bad} rows with non-standard format")
    except:
        errors.append("'order_date' column broken")

    try:
        if not (pd.api.types.is_float_dtype(df["revenue"]) or 
                pd.api.types.is_integer_dtype(df["revenue"])):
            errors.append("'revenue' is not numeric — contains '$' signs")
    except:
        errors.append("'revenue' column broken")

    try:
        dups = df["order_id"].duplicated().sum()
        if dups > 0:
            errors.append(f"{dups} duplicate order_ids found")
    except:
        errors.append("'order_id' column broken")

    try:
        nulls = df["region"].isnull().sum()
        if nulls > 0:
            errors.append(f"'region' has {nulls} null values")
    except:
        errors.append("'region' column broken")

    if "cust_nm" in df.columns:
        errors.append("Column 'cust_nm' should be renamed to 'customer_name'")

    if not errors:
        errors.append("No errors found! Call 'done' to finish.")

    return errors