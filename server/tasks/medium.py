import pandas as pd
import numpy as np

MEDIUM_TASK_ID = "fix_schema_drift"
MEDIUM_GOAL = """Fix this broken sales dataset:
1. 'order_date' has mixed formats — standardize to YYYY-MM-DD
2. 'revenue' stored as string with '$' — cast to float
3. Duplicate order_ids exist — remove duplicates
4. 'region' has nulls — fill with 'Unknown'
5. 'cust_nm' should be renamed to 'customer_name'
"""

def generate_medium_dataset() -> pd.DataFrame:
    np.random.seed(123)
    n = 80
    dates_fmt1 = [f"2023/{np.random.randint(1,13):02d}/{np.random.randint(1,28):02d}" for _ in range(50)]
    dates_fmt2 = [f"{np.random.randint(1,28):02d}-{np.random.randint(1,13):02d}-2023" for _ in range(30)]
    all_dates = dates_fmt1 + dates_fmt2
    np.random.shuffle(all_dates)
    revenues = [f"${np.random.uniform(100, 5000):.2f}" for _ in range(n)]
    order_ids = list(range(1000, 1000 + n))
    dup_indices = np.random.choice(n, size=10, replace=False)
    for idx in dup_indices:
        order_ids[idx] = order_ids[np.random.randint(0, max(idx,1))]
    regions = np.random.choice(["North", "South", "East", "West"], n).tolist()
    null_region_idx = np.random.choice(n, size=12, replace=False)
    for idx in null_region_idx:
        regions[idx] = None
    return pd.DataFrame({
        "order_id": order_ids,
        "cust_nm": [f"Customer_{i}" for i in range(n)],
        "order_date": all_dates,
        "revenue": revenues,
        "region": regions
    })

def grade_medium(df: pd.DataFrame) -> float:
    try:
        n = max(len(df), 1)
        components = []

        # dates: 0.1 to 0.35
        try:
            parsed = pd.to_datetime(df["order_date"], format="%Y-%m-%d", errors="coerce")
            ratio = parsed.notna().sum() / n
            components.append(0.1 + 0.25 * ratio)
        except:
            components.append(0.1)

        # revenue float: 0.1 to 0.3
        try:
            if pd.api.types.is_float_dtype(df["revenue"]):
                components.append(0.3)
            else:
                numeric = pd.to_numeric(
                    df["revenue"].astype(str).str.replace("$","",regex=False),
                    errors="coerce"
                )
                ratio = numeric.notna().sum() / n
                components.append(0.1 + 0.2 * ratio)
        except:
            components.append(0.1)

        # no duplicates: 0.1 to 0.25
        try:
            dup_ratio = df["order_id"].duplicated().sum() / n
            components.append(0.1 + 0.15 * (1 - dup_ratio))
        except:
            components.append(0.1)

        # region no nulls: 0.05 to 0.1
        try:
            null_ratio = df["region"].isnull().sum() / n
            components.append(0.05 + 0.05 * (1 - null_ratio))
        except:
            components.append(0.05)

        score = sum(components)
        # min = 0.1+0.1+0.1+0.05 = 0.35
        # max = 0.35+0.3+0.25+0.1 = 1.0 → cap at 0.95
        return round(max(0.1, min(0.95, score)), 4)
    except:
        return 0.5

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
        if not pd.api.types.is_float_dtype(df["revenue"]):
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