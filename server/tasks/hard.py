import pandas as pd
import numpy as np
import math

HARD_TASK_ID = "fix_multi_table_join"

HARD_GOAL = """Fix and join two broken datasets (orders + products):
ORDERS table issues:
1. 'amount' column has nulls (~25%) — fill with median
2. 'product_code' has inconsistent casing ('ABC123' vs 'abc123') — uppercase all
3. 'order_date' stored as int (20230115) — convert to proper date YYYY-MM-DD
PRODUCTS table issues:
4. 'product_code' has extra whitespace ('ABC123 ') — strip whitespace
5. 'price' has string values ('N/A', 'none') instead of nulls — replace with median
After fixing both: produce a valid joined dataset on 'product_code'.
Goal: Joined dataframe with no nulls in key columns, correct types, 
      at least 80% of orders successfully joined.
"""

def generate_hard_dataset() -> tuple:
    """Generate two broken datasets that need to be fixed and joined."""
    np.random.seed(999)

    # Product codes
    codes = [f"PROD{i:03d}" for i in range(1, 21)]  # 20 products

    # ORDERS table
    n_orders = 100
    order_codes = np.random.choice(codes, n_orders).tolist()

    # Inject inconsistent casing in ~30% of order product codes
    for i in range(n_orders):
        if np.random.random() < 0.3:
            order_codes[i] = order_codes[i].lower()

    amounts = np.random.uniform(50, 500, n_orders).tolist()
    # Inject nulls in amount (~25%)
    null_idx = np.random.choice(n_orders, size=25, replace=False)
    for idx in null_idx:
        amounts[idx] = None

    # Dates stored as int
    dates_as_int = []
    for _ in range(n_orders):
        m = np.random.randint(1, 13)
        d = np.random.randint(1, 28)
        dates_as_int.append(int(f"2023{m:02d}{d:02d}"))

    orders_df = pd.DataFrame({
        "order_id": range(1, n_orders + 1),
        "product_code": order_codes,
        "amount": amounts,
        "order_date": dates_as_int
    })

    # PRODUCTS table
    product_codes_with_spaces = [c + (" " if np.random.random() < 0.5 else "") 
                                  for c in codes]
    prices = np.random.uniform(10, 200, len(codes)).tolist()

    # Inject 'N/A' and 'none' in prices (~20%)
    for i in range(len(codes)):
        if np.random.random() < 0.2:
            prices[i] = np.random.choice(["N/A", "none", "null"])

    products_df = pd.DataFrame({
        "product_code": product_codes_with_spaces,
        "product_name": [f"Product_{c}" for c in codes],
        "price": prices,
        "category": np.random.choice(["Electronics", "Clothing", "Food", "Tools"], len(codes))
    })

    return orders_df, products_df

def safe_score(x):
    if x is None or not math.isfinite(x):
        return 0.5
    x = float(x)
    return max(1e-6, min(1 - 1e-6, x))


def grade_hard(orders_df: pd.DataFrame,
               products_df: pd.DataFrame,
               joined_df: pd.DataFrame = None) -> float:
    try:
        score = 0.1  # base score

        o_total = max(len(orders_df), 1)
        p_total = max(len(products_df), 1)

        # orders.amount no nulls: +0.15
        try:
            score += 0.15 * (1 - orders_df["amount"].isnull().sum() / o_total)
        except: pass

        # orders.product_code uppercase: +0.15
        try:
            codes = orders_df["product_code"].dropna()
            score += 0.15 * (codes == codes.str.upper()).mean()
        except: pass

        # orders.order_date is date not int: +0.15
        try:
            if not pd.api.types.is_integer_dtype(orders_df["order_date"]):
                parsed = pd.to_datetime(orders_df["order_date"], errors="coerce")
                score += 0.15 * parsed.notna().mean()
        except: pass

        # products.product_code stripped: +0.10
        try:
            codes = products_df["product_code"]
            score += 0.10 * (codes == codes.str.strip()).mean()
        except: pass

        # products.price numeric: +0.10
        try:
            if pd.api.types.is_numeric_dtype(products_df["price"]):
                score += 0.10
            else:
                numeric = pd.to_numeric(products_df["price"], errors="coerce")
                score += 0.10 * numeric.notna().mean()
        except: pass

        # join success: +0.25
        try:
            test_join = pd.merge(orders_df, products_df, on="product_code", how="inner")
            join_ratio = len(test_join) / 100
            score += 0.25 * min(join_ratio, 0.95)
        except: pass

        return round(max(0.05, min(0.95, score)), 4)
    except:
        return 0.5


def get_errors(orders_df: pd.DataFrame, products_df: pd.DataFrame) -> list:
    errors = []

    try:
        nulls = orders_df["amount"].isnull().sum()
        if nulls > 0:
            errors.append(f"orders.'amount' has {nulls} null values")
    except:
        errors.append("orders.'amount' column broken")

    try:
        codes = orders_df["product_code"].dropna()
        bad = (codes != codes.str.upper()).sum()
        if bad > 0:
            errors.append(f"orders.'product_code' has {bad} lowercase entries")
    except:
        pass

    try:
        if pd.api.types.is_integer_dtype(orders_df["order_date"]):
            errors.append("orders.'order_date' is stored as int — convert to date")
    except:
        pass

    try:
        codes = products_df["product_code"]
        bad = (codes != codes.str.strip()).sum()
        if bad > 0:
            errors.append(f"products.'product_code' has {bad} entries with whitespace")
    except:
        pass

    try:
        if not pd.api.types.is_numeric_dtype(products_df["price"]):
            bad_vals = products_df["price"].isin(["N/A", "none", "null"]).sum()
            errors.append(f"products.'price' has {bad_vals} non-numeric values (N/A, none)")
    except:
        pass

    try:
        test_join = pd.merge(orders_df, products_df, on="product_code", how="inner")
        join_pct = len(test_join) / 100 * 100
        if join_pct < 80:
            errors.append(f"Only {join_pct:.0f}% of orders can be joined — fix product_codes first")
    except:
        errors.append("Cannot join tables yet — fix product_codes in both tables")

    if not errors:
        errors.append("No errors found! Call 'done' to finish.")

    return errors