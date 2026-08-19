"""
Customer Churn Analysis - Run SQL Segmentation Queries
=====================================================================
Loads telco_cleaned.csv into a local SQLite database and runs all
segmentation queries, printing results to the console and saving
them to sql_results.txt
"""

import pandas as pd
import sqlite3

CSV_PATH = "telco_cleaned.csv"
DB_PATH = "churn.db"
OUT_PATH = "sql_results.txt"

# Load cleaned data into SQLite
df = pd.read_csv(CSV_PATH)
conn = sqlite3.connect(DB_PATH)
df.to_sql("customers", conn, if_exists="replace", index=False)
print(f"Loaded {len(df):,} rows into SQLite database ({DB_PATH})")

queries = {
    "Churn Rate by Contract Type": '''
        SELECT Contract, COUNT(*) AS total_customers,
            SUM(CASE WHEN Churn='Yes' THEN 1 ELSE 0 END) AS churned,
            ROUND(SUM(CASE WHEN Churn='Yes' THEN 1 ELSE 0 END)*100.0/COUNT(*), 2) AS churn_rate_pct
        FROM customers GROUP BY Contract ORDER BY churn_rate_pct DESC;
    ''',
    "Churn Rate by Tenure Bucket": '''
        SELECT "Tenure Bucket" AS tenure_bucket, COUNT(*) AS total_customers,
            ROUND(SUM(CASE WHEN Churn='Yes' THEN 1 ELSE 0 END)*100.0/COUNT(*), 2) AS churn_rate_pct
        FROM customers GROUP BY "Tenure Bucket" ORDER BY churn_rate_pct DESC;
    ''',
    "Churn Rate by Payment Method": '''
        SELECT PaymentMethod, COUNT(*) AS total_customers,
            ROUND(SUM(CASE WHEN Churn='Yes' THEN 1 ELSE 0 END)*100.0/COUNT(*), 2) AS churn_rate_pct
        FROM customers GROUP BY PaymentMethod ORDER BY churn_rate_pct DESC;
    ''',
    "Churn Rate by Internet Service": '''
        SELECT InternetService, COUNT(*) AS total_customers,
            ROUND(SUM(CASE WHEN Churn='Yes' THEN 1 ELSE 0 END)*100.0/COUNT(*), 2) AS churn_rate_pct
        FROM customers GROUP BY InternetService ORDER BY churn_rate_pct DESC;
    ''',
    "Highest-Risk Segment (Contract x Tenure)": '''
        SELECT Contract, "Tenure Bucket" AS tenure_bucket, COUNT(*) AS total_customers,
            ROUND(SUM(CASE WHEN Churn='Yes' THEN 1 ELSE 0 END)*100.0/COUNT(*), 2) AS churn_rate_pct
        FROM customers GROUP BY Contract, "Tenure Bucket"
        HAVING total_customers >= 30 ORDER BY churn_rate_pct DESC LIMIT 5;
    ''',
}

output_lines = []
for title, query in queries.items():
    print(f"\n=== {title} ===")
    result = pd.read_sql(query, conn)
    print(result.to_string(index=False))
    output_lines.append(f"=== {title} ===\n{result.to_string(index=False)}\n")

conn.close()

with open(OUT_PATH, "w") as f:
    f.write("\n".join(output_lines))

print(f"\nAll query results saved to: {OUT_PATH}")
