import os
from typing import List, Optional
import pandas as pd


def load_and_merge_olist_data(data_dir: str = "Data/Raw") -> pd.DataFrame:
    """
    Loads and merges the 4 core Olist datasets:
    - olist_customers_dataset.csv
    - olist_orders_dataset.csv
    - olist_order_items_dataset.csv
    - olist_order_payments_dataset.csv
    """
    print(f"Loading datasets from: {os.path.abspath(data_dir)}")
    
    df_customers = pd.read_csv(os.path.join(data_dir, "olist_customers_dataset.csv"))
    df_orders = pd.read_csv(os.path.join(data_dir, "olist_orders_dataset.csv"))
    df_order_items = pd.read_csv(os.path.join(data_dir, "olist_order_items_dataset.csv"))
    df_order_payments = pd.read_csv(os.path.join(data_dir, "olist_order_payments_dataset.csv"))

    # Step-by-step merge
    df_step1 = pd.merge(df_orders, df_customers, on="customer_id", how="inner")
    df_step2 = pd.merge(df_step1, df_order_items, on="order_id", how="inner")
    df_master = pd.merge(df_step2, df_order_payments, on="order_id", how="inner")

    return df_master


def clean_olist_master(
    df: pd.DataFrame,
    date_columns: Optional[List[str]] = None,
    missing_threshold: float = 0.40
) -> pd.DataFrame:
    """
    Cleans df_master:
    1) Filters for order_status == 'delivered'
    2) Converts date columns to pd.to_datetime
    3) Drops columns with > 40% missing values
    4) Drops duplicate rows
    """
    if date_columns is None:
        date_columns = [
            "order_purchase_timestamp",
            "order_approved_at",
            "order_delivered_carrier_date",
            "order_delivered_customer_date",
            "order_estimated_delivery_date",
            "shipping_limit_date"
        ]

    # 1. Filter delivered orders
    df_cleaned = df[df["order_status"] == "delivered"].copy()

    # 2. Datetime conversions
    for col in date_columns:
        if col in df_cleaned.columns:
            df_cleaned[col] = pd.to_datetime(df_cleaned[col])

    # 3. Drop columns > 40% missing
    high_missing_cols = df_cleaned.columns[df_cleaned.isnull().mean() > missing_threshold]
    if len(high_missing_cols) > 0:
        print(f"Dropping columns with >{missing_threshold*100:.0f}% missing values: {list(high_missing_cols)}")
        df_cleaned = df_cleaned.drop(columns=high_missing_cols)

    # 4. Drop duplicates
    df_cleaned = df_cleaned.drop_duplicates()

    return df_cleaned


if __name__ == "__main__":
    df_raw_master = load_and_merge_olist_data()
    print(f"Raw df_master shape: {df_raw_master.shape}")

    df_cleaned_master = clean_olist_master(df_raw_master)
    print(f"\nCleaned df_master shape: {df_cleaned_master.shape}")
    print("\nMissing values per column:")
    print(df_cleaned_master.isnull().sum())
