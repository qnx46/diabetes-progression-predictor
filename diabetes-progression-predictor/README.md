# Diabetes Progression Predictor

A regression project that predicts disease progression in diabetes patients
one year after baseline, using 10 routine health measurements (age, sex, BMI,
blood pressure, and six blood serum values).

## Problem

Given a patient's baseline health metrics, can we predict how their diabetes
will progress over the following year? This is a classic supervised
regression problem: continuous numeric features in, a continuous numeric
target out.

## Dataset

- **Source:** scikit-learn's built-in `load_diabetes` dataset (originally
  published in Efron, Hastie, Johnstone & Tibshirani, *"Least Angle
  Regression"*, Annals of Statistics, 2004).
- **Size:** 442 patients, 10 features, 1 target.
- **Features:** age, sex, bmi, bp (blood pressure), and s1–s6 (six blood
  serum measurements). All features are pre-normalized (mean-centered,
  scaled by standard deviation) in the original dataset.
- **Target:** a quantitative measure of disease progression one year after
  baseline (higher = more progression).

## Approach

1. **Exploratory Data Analysis** — checked feature distributions and
   correlation of each feature with the target.
2. **Train/test split** — 80/20 split, random_state fixed for reproducibility.
3. **Model 1: Linear Regression** — simple, interpretable baseline.
4. **Model 2: Random Forest Regressor** (200 trees) — captures non-linear
   relationships, used for comparison and feature importance.
5. **Evaluation** — RMSE, MAE, and R² on the held-out test set.

## Results

| Model              | RMSE  | MAE   | R² Score |
|--------------------|-------|-------|----------|
| Linear Regression  | 53.85 | 42.79 | 0.453    |
| Random Forest      | 54.46 | 44.28 | 0.440    |

**Linear Regression slightly outperformed Random Forest** on this dataset.
This makes sense: with only 442 samples and 10 features, there isn't enough
data for the Random Forest to learn complex non-linear patterns without
overfitting — a good reminder that more complex models aren't always better,
especially on small datasets.

`bmi` and `s5` (a blood serum measurement) had the strongest correlation with
disease progression, which lines up with known clinical understanding of
diabetes risk factors.

## What I'd improve with more time

- Try regularized linear models (Ridge/Lasso) to see if they reduce
  variance further.
- Use k-fold cross-validation instead of a single train/test split, since
  442 samples is small enough that the split itself introduces variance.
- Engineer interaction features (e.g., bmi × bp) to help the linear model
  capture combined effects.

## Project structure

```
├── train.py              # main script: EDA, training, evaluation
├── notebook.ipynb         # same workflow in notebook form
├── results.csv            # model comparison metrics
├── plots/
│   ├── feature_correlation.png
│   ├── target_distribution.png
│   ├── predicted_vs_actual.png
│   └── feature_importance.png
└── requirements.txt
```

## How to run

```bash
pip install -r requirements.txt
python train.py
```

## Tech stack

Python, pandas, NumPy, scikit-learn, Matplotlib
