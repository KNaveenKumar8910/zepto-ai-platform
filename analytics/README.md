# Module 2 — Analytics Pipeline

## Part A: Profiling & Missing Data Handling
- **Missing Value Handling Strategy**:
  - `embarked` / `embark_town` (0.22% missing, <5% threshold): Dropped the affected rows.
  - `age` (19.87% missing, 5%–30% threshold): Imputed using median within the modeling pipeline.
  - `deck` (77.22% missing, >30% threshold): Dropped column completely to avoid introducing synthetic noise.
- **Univariate Analysis & Outliers**:
  - `age`: 8 outliers identified using the IQR rule.
  - `fare`: 114 outliers identified using the IQR rule.
  - `fare` Skewness: Strongly right-skewed, verified by the central tendency ordering: Mean (32.10) > Median (14.45) > Mode (8.05).
- **Bivariate Survival Rates**:
  - By Sex: Female = 74.04%, Male = 18.89%
  - By Pclass: 1st = 62.62%, 2nd = 47.28%, 3rd = 24.24%
  - By Sex & Pclass:
    - Female: Class 1 = 96.74%, Class 2 = 92.11%, Class 3 = 50.00%
    - Male: Class 1 = 36.89%, Class 2 = 15.74%, Class 3 = 13.54%
- **Correlations (6x6 Matrix)**:
  - Restricted to: `survived`, `pclass`, `age`, `sibsp`, `parch`, `fare` (excluding redundant boolean flags `adult_male` and `alone`).
  - Top 2 off-diagonal correlations: `pclass` vs `fare` (-0.55) and `pclass` vs `age` (-0.37).

## Part B: Predictive Modeling & Imbalance Handling
- **Stratified Split**: Applied `stratify=y` on survived (maintaining the ~38.4% survival class ratio in both training and testing folds).
- **Preprocessing**: Leak-free `ColumnTransformer` (median imputer + `StandardScaler` for numeric, most-frequent imputer + `OneHotEncoder` for categorical) fit strictly on train split and transformed on test split.
- **Imbalance Handling Evaluation**:
  - Baseline Random Forest F1: 0.7458
  - SMOTE (Train fold only): F1 = 0.7299
- **Hyperparameter Tuning & Out-of-Bag (OOB)**:
  - Model: `RandomForestClassifier(n_estimators=100, max_depth=5, oob_score=True, random_state=42)`
  - OOB Score achieved: **0.8143**
- **Regression Side-Task (Fare Prediction)**:
  - Linear regression on fare shows clear heteroscedasticity: residual variance fans out significantly as predicted fare increases.

## Model Comparison Table

| Model | Accuracy | Precision | Recall | F1 Score | AUC | MAE | RMSE | R² |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression** | 0.8146 | 0.7966 | 0.6912 | 0.7402 | **0.8596** | — | — | — |
| **Decision Tree** | 0.8034 | 0.8000 | 0.6471 | 0.7154 | 0.8481 | — | — | — |
| **Random Forest** | **0.8315** | **0.8800** | 0.6471 | **0.7458** | 0.8398 | — | — | — |
| **Linear Regression (Fare)** | — | — | — | — | — | 19.14 | 31.11 | 0.37 |

## Deployment Recommendation
Deploy the **Random Forest** classifier. It achieved the highest overall Accuracy (0.8315), the highest Precision (0.8800), and the best F1 Score (0.7458), along with an OOB Score of 0.8143, demonstrating strong generalization without overfitting to the majority non-survivor class.
