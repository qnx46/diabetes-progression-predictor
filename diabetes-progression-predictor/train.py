"""
Diabetes Progression Predictor
-------------------------------
A regression project predicting disease progression in diabetes patients
one year after baseline, using 10 baseline health measurements.

Dataset: sklearn's built-in `load_diabetes` (originally from Efron et al., 2004,
"Least Angle Regression", Annals of Statistics). 442 patients, 10 features.

Models compared: Linear Regression vs Random Forest Regressor.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

RANDOM_STATE = 42

# ---------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------
data = load_diabetes(as_frame=True)
df = data.frame  # includes features + target column
print("Dataset shape:", df.shape)
print("\nFirst 5 rows:\n", df.head())
print("\nSummary stats:\n", df.describe())

# ---------------------------------------------------------------
# 2. Exploratory Data Analysis
# ---------------------------------------------------------------
# Correlation of each feature with the target
correlations = df.corr()["target"].drop("target").sort_values(ascending=False)
print("\nFeature correlation with target (disease progression):\n", correlations)

plt.figure(figsize=(8, 5))
correlations.plot(kind="barh", color="#1F3864")
plt.title("Feature Correlation with Disease Progression")
plt.xlabel("Correlation coefficient")
plt.tight_layout()
plt.savefig("plots/feature_correlation.png", dpi=120)
plt.close()

plt.figure(figsize=(6, 4))
plt.hist(df["target"], bins=30, color="#1F3864", edgecolor="white")
plt.title("Distribution of Target (Disease Progression Score)")
plt.xlabel("Progression score")
plt.ylabel("Number of patients")
plt.tight_layout()
plt.savefig("plots/target_distribution.png", dpi=120)
plt.close()

# ---------------------------------------------------------------
# 3. Train/test split
# ---------------------------------------------------------------
X = df.drop(columns=["target"])
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE
)
print(f"\nTrain size: {X_train.shape[0]} | Test size: {X_test.shape[0]}")

# ---------------------------------------------------------------
# 4. Model 1 — Linear Regression (baseline)
# ---------------------------------------------------------------
lr = LinearRegression()
lr.fit(X_train, y_train)
lr_preds = lr.predict(X_test)

lr_rmse = np.sqrt(mean_squared_error(y_test, lr_preds))
lr_mae = mean_absolute_error(y_test, lr_preds)
lr_r2 = r2_score(y_test, lr_preds)

# ---------------------------------------------------------------
# 5. Model 2 — Random Forest Regressor
# ---------------------------------------------------------------
rf = RandomForestRegressor(n_estimators=200, random_state=RANDOM_STATE)
rf.fit(X_train, y_train)
rf_preds = rf.predict(X_test)

rf_rmse = np.sqrt(mean_squared_error(y_test, rf_preds))
rf_mae = mean_absolute_error(y_test, rf_preds)
rf_r2 = r2_score(y_test, rf_preds)

# ---------------------------------------------------------------
# 6. Compare results
# ---------------------------------------------------------------
results = pd.DataFrame({
    "Model": ["Linear Regression", "Random Forest"],
    "RMSE": [lr_rmse, rf_rmse],
    "MAE": [lr_mae, rf_mae],
    "R2 Score": [lr_r2, rf_r2],
})
print("\nModel comparison:\n", results.to_string(index=False))
results.to_csv("results.csv", index=False)

# Plot: predicted vs actual for the better model
best_preds = rf_preds if rf_r2 > lr_r2 else lr_preds
best_name = "Random Forest" if rf_r2 > lr_r2 else "Linear Regression"

plt.figure(figsize=(6, 6))
plt.scatter(y_test, best_preds, alpha=0.6, color="#1F3864")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--", lw=2)
plt.xlabel("Actual Progression Score")
plt.ylabel("Predicted Progression Score")
plt.title(f"Predicted vs Actual — {best_name}")
plt.tight_layout()
plt.savefig("plots/predicted_vs_actual.png", dpi=120)
plt.close()

# Feature importance (Random Forest)
importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
plt.figure(figsize=(8, 5))
importances.plot(kind="barh", color="#1F3864")
plt.title("Random Forest — Feature Importance")
plt.xlabel("Importance")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig("plots/feature_importance.png", dpi=120)
plt.close()

print("\nBest model:", best_name)
print("Plots saved to ./plots/")
print("Results saved to results.csv")
