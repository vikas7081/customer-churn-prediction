"""
Customer Churn Prediction - Generate Charts
=====================================================================
Creates two visualization files:
- churn_segmentation.png (churn rate by contract type & tenure)
- model_results.png (confusion matrix & top predictive features)
"""

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

df = pd.read_csv("telco_cleaned.csv")

# ===========================================================
# CHART 1: Segmentation (Contract type & Tenure vs Churn)
# ===========================================================
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

contract_churn = df.groupby("Contract").apply(
    lambda x: (x["Churn"] == "Yes").mean() * 100, include_groups=False
).sort_values(ascending=False)
colors1 = ["#c0392b" if v > 30 else "#e67e22" if v > 10 else "#27ae60" for v in contract_churn.values]
axes[0].bar(contract_churn.index, contract_churn.values, color=colors1)
axes[0].set_title("Churn Rate by Contract Type", fontsize=13, fontweight="bold")
axes[0].set_ylabel("Churn Rate (%)")
for i, v in enumerate(contract_churn.values):
    axes[0].text(i, v + 1, f"{v:.1f}%", ha="center", fontweight="bold")

order = ["0-1 Year", "1-2 Years", "2-4 Years", "4+ Years"]
tenure_churn = df.groupby("Tenure Bucket").apply(
    lambda x: (x["Churn"] == "Yes").mean() * 100, include_groups=False
).reindex(order)
colors2 = ["#c0392b" if v > 40 else "#e67e22" if v > 20 else "#27ae60" for v in tenure_churn.values]
axes[1].bar(tenure_churn.index, tenure_churn.values, color=colors2)
axes[1].set_title("Churn Rate by Customer Tenure", fontsize=13, fontweight="bold")
axes[1].set_ylabel("Churn Rate (%)")
for i, v in enumerate(tenure_churn.values):
    axes[1].text(i, v + 1, f"{v:.1f}%", ha="center", fontweight="bold")

plt.tight_layout()
plt.savefig("churn_segmentation.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved churn_segmentation.png")

# ===========================================================
# CHART 2: Model results (confusion matrix & feature importance)
# ===========================================================
target = (df["Churn"] == "Yes").astype(int)
features = df.drop(columns=["customerID", "Churn", "Tenure Bucket"])
features_encoded = pd.get_dummies(features, drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(
    features_encoded, target, test_size=0.2, random_state=42, stratify=target
)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)
model = LogisticRegression(max_iter=2000, random_state=42)
model.fit(X_train_s, y_train)
y_pred = model.predict(X_test_s)
cm = confusion_matrix(y_test, y_pred)

coef_df = pd.DataFrame({"feature": features_encoded.columns, "coefficient": model.coef_[0]})
coef_df["abs_coef"] = coef_df["coefficient"].abs()
top10 = coef_df.sort_values("abs_coef", ascending=False).head(10).sort_values("coefficient")

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

im = axes[0].imshow(cm, cmap="Blues")
axes[0].set_xticks([0, 1]); axes[0].set_xticklabels(["No Churn", "Churn"])
axes[0].set_yticks([0, 1]); axes[0].set_yticklabels(["No Churn", "Churn"])
axes[0].set_xlabel("Predicted"); axes[0].set_ylabel("Actual")
axes[0].set_title("Confusion Matrix (Test Set)", fontsize=13, fontweight="bold")
for i in range(2):
    for j in range(2):
        axes[0].text(j, i, str(cm[i][j]), ha="center", va="center", fontsize=16, fontweight="bold",
                     color="white" if cm[i][j] > cm.max() / 2 else "black")

colors = ["#c0392b" if c > 0 else "#27ae60" for c in top10["coefficient"]]
axes[1].barh(top10["feature"], top10["coefficient"], color=colors)
axes[1].set_title("Top 10 Churn Predictors (Logistic Regression Weights)", fontsize=12, fontweight="bold")
axes[1].set_xlabel("Coefficient (red = increases churn, green = decreases churn)")
axes[1].axvline(0, color="black", linewidth=0.8)

plt.tight_layout()
plt.savefig("model_results.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved model_results.png")
