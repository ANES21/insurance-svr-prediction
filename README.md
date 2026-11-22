# Insurance Charges Prediction with SVR

This project predicts medical insurance costs (`charges`) using **Support Vector Regression (SVR)** on the `insurance.csv` dataset[cite: 1, 2].

## Dataset Overview
- **Source File:** `insurance.csv`[cite: 1, 2]
- **Size:** 1,338 rows and 7 columns (`age`, `sex`, `bmi`, `children`, `smoker`, `region`, `charges`)[cite: 1, 2]
- **Target Variable:** `charges` (medical insurance costs)[cite: 1, 2]

## Methodology & Preprocessing
1. Split the dataset into **80% training** and **20% testing** sets (`random_state=42`)[cite: 1, 2].
2. Apply one-hot encoding (`pd.get_dummies(..., drop_first=True)`) after splitting to avoid data leakage, and align test columns with training columns[cite: 1, 2].
3. Standardize features and the target variable using `StandardScaler`[cite: 1, 2].
4. Train an `SVR` model (`kernel='rbf'`, `C=100`, `gamma=0.1`, `epsilon=0.1`) and inverse-transform predictions back to the original scale[cite: 1, 2].
5. Evaluate predictions on the test set and visualize actual vs. predicted charges using `matplotlib` and `seaborn`[cite: 1, 2].

## Model Evaluation Results (Test Set)
- **MAE (Mean Absolute Error):** `2617.75`[cite: 1, 2]
- **R² Score:** `0.85`[cite: 1, 2]

## Project Structure
```text
insurance-svr-prediction/
├── snv_2.ipynb
├── insurance.csv
└── README.md
