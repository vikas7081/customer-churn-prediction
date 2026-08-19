"""
Customer Churn Prediction - Logistic Regression Model
=====================================================================
Trains and evaluates a logistic regression model to predict customer
churn, using Scikit-learn. Reports honest accuracy/precision/recall
(not a target number pulled out of thin air).
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)

df = pd.read_csv("telco_cleaned.csv")

# ---------------------------------------------------------
# Feature preparation
# ---------------------------------------------------------
target = (df["Churn"] == "Yes").astype(int)

drop_cols = ["customerID", "Churn", "Tenure Bucket"]
features = df.drop(columns=drop_cols)

# One-hot encode all categorical columns
features_encoded = pd.get_dummies(features, drop_first=True)

print(f"Features used: {features_encoded.shape[1]}")

# ---------------------------------------------------------
# Train/test split (stratified to preserve churn ratio)
# ---------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    features_encoded, target, test_size=0.2, random_state=42, stratify=target
)

# Scale numeric features (helps logistic regression converge + perform better)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------------------------------------------------------
# Train logistic regression
# ---------------------------------------------------------
model = LogisticRegression(max_iter=2000, class_weight=None, random_state=42)
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)
y_proba = model.predict_proba(X_test_scaled)[:, 1]

# ---------------------------------------------------------
# Evaluation
# ---------------------------------------------------------
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_proba)
cm = confusion_matrix(y_test, y_pred)

# Baseline: always predict majority class (no churn)
baseline_acc = (y_test == 0).mean()

print("\n" + "=" * 60)
print("MODEL EVALUATION")
print("=" * 60)
print(f"Baseline accuracy (always predict 'No Churn'): {baseline_acc:.2%}")
print(f"Model accuracy:                                {acc:.2%}")
print(f"Precision (churn class):                       {prec:.2%}")
print(f"Recall (churn class):                           {rec:.2%}")
print(f"F1 score (churn class):                         {f1:.2%}")
print(f"ROC-AUC:                                         {auc:.3f}")
print(f"\nConfusion Matrix:")
print(f"                 Predicted No   Predicted Yes")
print(f"Actual No        {cm[0][0]:>10}      {cm[0][1]:>10}")
print(f"Actual Yes       {cm[1][0]:>10}      {cm[1][1]:>10}")

# ---------------------------------------------------------
# Top predictive features (by absolute coefficient weight)
# ---------------------------------------------------------
coef_df = pd.DataFrame({
    "feature": features_encoded.columns,
    "coefficient": model.coef_[0]
})
coef_df["abs_coef"] = coef_df["coefficient"].abs()
coef_df = coef_df.sort_values("abs_coef", ascending=False)

print("\nTop 10 features driving churn (by model weight):")
print(coef_df.head(10)[["feature", "coefficient"]].to_string(index=False))

# Save evaluation summary
with open("model_evaluation.txt", "w") as f:
    f.write("CUSTOMER CHURN PREDICTION - MODEL EVALUATION SUMMARY\n")
    f.write("=" * 60 + "\n")
    f.write(f"Model: Logistic Regression (Scikit-learn)\n")
    f.write(f"Train/Test split: 80/20, stratified\n")
    f.write(f"Test set size: {len(y_test)} customers\n\n")
    f.write(f"Baseline accuracy (majority class): {baseline_acc:.2%}\n")
    f.write(f"Model accuracy:                     {acc:.2%}\n")
    f.write(f"Precision (churn class):            {prec:.2%}\n")
    f.write(f"Recall (churn class):               {rec:.2%}\n")
    f.write(f"F1 score (churn class):             {f1:.2%}\n")
    f.write(f"ROC-AUC:                            {auc:.3f}\n\n")
    f.write("Top 10 predictive features:\n")
    for _, row in coef_df.head(10).iterrows():
        direction = "increases" if row["coefficient"] > 0 else "decreases"
        f.write(f"  - {row['feature']}: {direction} churn likelihood (coef={row['coefficient']:+.3f})\n")

print("\nEvaluation summary saved to model_evaluation.txt")
