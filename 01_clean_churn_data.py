"""
Customer Churn Prediction - Data Cleaning
=====================================================================
Dataset: IBM Telco Customer Churn (7,043 customers)
Input:  telco.csv (raw)
Output: telco_cleaned.csv
"""

import pandas as pd

RAW_PATH = "telco.csv"
OUT_PATH = "telco_cleaned.csv"

df = pd.read_csv(RAW_PATH)
print(f"Loaded {len(df):,} rows, {df.shape[1]} columns")

# ---------------------------------------------------------
# Fix TotalCharges: blank strings -> numeric.
# All 11 blanks correspond to tenure=0 (brand new customers,
# not yet billed) -> set TotalCharges to 0.
# ---------------------------------------------------------
df["TotalCharges"] = df["TotalCharges"].replace(" ", pd.NA)
blank_mask = df["TotalCharges"].isna()
print(f"Rows with blank TotalCharges (tenure=0 new customers): {blank_mask.sum()}")
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df.loc[blank_mask, "TotalCharges"] = 0.0

# ---------------------------------------------------------
# Standardize SeniorCitizen to Yes/No for consistency with other flags
# ---------------------------------------------------------
df["SeniorCitizen"] = df["SeniorCitizen"].map({0: "No", 1: "Yes"})

# ---------------------------------------------------------
# Add a tenure bucket for segmentation (used in SQL/EDA step)
# ---------------------------------------------------------
def tenure_bucket(t):
    if t <= 12:
        return "0-1 Year"
    elif t <= 24:
        return "1-2 Years"
    elif t <= 48:
        return "2-4 Years"
    else:
        return "4+ Years"

df["Tenure Bucket"] = df["tenure"].apply(tenure_bucket)

# ---------------------------------------------------------
# Validation
# ---------------------------------------------------------
assert df["customerID"].duplicated().sum() == 0, "Duplicate customers found!"
assert df["TotalCharges"].isnull().sum() == 0, "Nulls remain in TotalCharges!"
assert df.isnull().sum().sum() == 0, "Nulls remain somewhere!"

print("Validation passed: no duplicates, no nulls.")
print(f"\nChurn rate: {(df['Churn']=='Yes').mean():.2%}")
print(f"Tenure buckets:\n{df['Tenure Bucket'].value_counts()}")

df.to_csv(OUT_PATH, index=False)
print(f"\nCleaned file saved to: {OUT_PATH}")
print(f"Final shape: {df.shape}")
