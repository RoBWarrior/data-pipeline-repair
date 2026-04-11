import pandas as pd
import numpy as np

HARD_TASK_ID = "fix_multi_table_join"
HARD_GOAL = """Fix and join two broken datasets (orders + products):
ORDERS: fix amount nulls, uppercase product_code, convert date from int
PRODUCTS: strip whitespace from product_code, fix price N/A values
Then join both tables on product_code.
"""

def generate_hard_dataset() -> tuple:
    np.random.seed(999)
    codes = [f"PROD{i:03d}" for i in range(1, 21)]
    n_orders = 100
    order_codes = np.random.choice(codes, n_orders).tolist()
    for i in range(n_orders):
        if np.random.random() < 0.3:
            order_codes[i] = order_codes[i].lower()
    amounts = np.random.uniform(50, 500, n_orders).tolist()
    null_idx = np.random.choice(n_orders, size=25, replace=False)
    for idx in null_idx:
        amounts[idx] = None
    dates_as_int = [int(f"2023{np.random.randint(1,13):02d}{np.random.randint(1,28):02d}") for _ in range(n_orders)]
    orders_df = pd.DataFrame({
        "order_id": range(1, n_orders + 1),
        "product_code": order_codes,
        "amount": amounts,
        "order_date": dates_as_int
    })
    product_codes_with_spaces = [c + (" " if np.random.random() < 0.5 else "") for c in codes]
    prices = np.random.uniform(10, 200, len(codes)).tolist()
    for i in range(len(codes)):
        if np.random.random() < 0.2:
            prices[i] = np.random.choice(["N/A", "none", "null"])
    products_df = pd.DataFrame({
        "product_code": product_codes_with_spaces,
        "product_name": [f"Product_{c}" for c in codes],
        "price": prices,
        "category": np.random.choice(["Electronics","Clothing","Food","Tools"], len(codes))
    })
    return orders_df, products_df

def grade_hard(orders_df: pd.DataFrame,
               products_df: pd.DataFrame,
               joined_df: pd.DataFrame = None) -> float:
    try:
        o_total = max(len(orders_df), 1)
        p_total = max(len(products_df), 1)
        components = []

        # amount no nulls: 0.1 to 0.25
        try:
            null_ratio = orders_df["amount"].isnull().sum() / o_total
            components.append(0.1 + 0.15 * (1 - null_ratio))
        except:
            components.append(0.1)

        # product_code uppercase: 0.1 to 0.2
        try:
            codes = orders_df["product_code"].dropna().astype(str)
            upper_ratio = (codes == codes.str.upper()).mean()
            components.append(0.1 + 0.1 * upper_ratio)
        except:
            components.append(0.1)

        # order_date not int: 0.1 to 0.2
        try:
            if not pd.api.types.is_integer_dtype(orders_df["order_date"]):
                parsed = pd.to_datetime(orders_df["order_date"], errors="coerce")
                components.append(0.1 + 0.1 * parsed.notna().mean())
            else:
                components.append(0.1)
        except:
            components.append(0.1)

        # products price numeric: 0.1 to 0.15
        try:
            if pd.api.types.is_numeric_dtype(products_df["price"]):
                components.append(0.15)
            else:
                numeric = pd.to_numeric(products_df["price"], errors="coerce")
                components.append(0.1 + 0.05 * numeric.notna().mean())
        except:
            components.append(0.1)

        # join success: 0.1 to 0.2
        try:
            test_join = pd.merge(orders_df, products_df, on="product_code", how="inner")
            join_ratio = min(len(test_join) / 100, 1.0)
            components.append(0.1 + 0.1 * join_ratio)
        except:
            components.append(0.1)

        score = sum(components)
        # min = 0.1*5 = 0.5
        # max = 0.25+0.2+0.2+0.15+0.2 = 1.0 → cap at 0.95
        return round(max(0.1, min(0.95, score)), 4)
    except:
        return 0.5

def get_errors(orders_df: pd.DataFrame, products_df: pd.DataFrame) -> list:
    errors = []
    try:
        nulls = orders_df["amount"].isnull().sum()
        if nulls > 0:
            errors.append(f"orders.'amount' has {nulls} null values")
    except:
        pass
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
        if not pd.api.types.is_numeric_dtype(products_df["price"]):
            errors.append("products.'price' has non-numeric values")
    except:
        pass
    if not errors:
        errors.append("No errors found! Call 'done' to finish.")
    return errors