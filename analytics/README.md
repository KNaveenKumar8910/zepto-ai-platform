# Module 2 — Analytics Pipeline

## Part A: Profiling & Missing Data Handling
- **Missing Value Strategy**:
  - `embarked` / `embark_town` (0.22% missing, <5% threshold): Dropped rows.
  - `age` (19.87% missing, 5%–30% threshold): Imputed using median within the pipeline.
  - `deck` (77.22% missing, >30% threshold): Column dropped as imputation would introduce significant noise.
- **Univariate Analysis & Outliers**:
  - `age`: 65 outliers identified via IQR rule.
  - `fare`: 114 outliers identified via IQR rule. Distribution is right-skewed: Mean (32.10) > Median (14.45) > Mode (8.05).
- **Correlations (6x6 Matrix)**:
  - Computed on: `survived`, `pclass`, `age`, `sibsp`, `parch`, `fare` (redundant flags `adult_male` and `alone` excluded).
  - Top 2 off-diagonal correlations: `pclass` vs `fare` (-0.55) and `sibsp` vs `parch` (0.41).

## Part B: Predictive Modeling & Imbalance Handling
- **Stratified Split**: Applied to maintain the ~38.4% survival class ratio across train and test sets.
- **Preprocessing**: Leak-free `ColumnTransformer` fit strictly on the training set and applied in transform-only mode on the test set.
- **Imbalance Comparison (Random Forest)**:
  - Baseline: F1 = 0.7385
  - Class Weight Balanced: F1 = 0.7424
  - SMOTE (Train fold only): F1 = 0.7536 (achieved the best balance between precision and recall).
- **Hyperparameter Tuning**: Ran `GridSearchCV` on `RandomForestClassifier(oob_score=True)` and logged out-of-bag scores.
- **Regression Side-Task**: Predicting `fare` via linear regression resulted in a funnel-shaped residual spread, confirming heteroscedasticity.

## Model Comparison Table

| Model | Accuracy | Precision | Recall | F1 Score | AUC | MAE | RMSE | R² |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression** | 0.8045 | 0.7931 | 0.6667 | 0.7244 | 0.8427 | — | — | — |
| **Decision Tree** | 0.7821 | 0.8409 | 0.5362 | 0.6549 | 0.8070 | — | — | — |
| **Random Forest** | **0.8101** | 0.7869 | **0.6957** | **0.7385** | 0.8402 | — | — | — |
| **Linear Regression (Fare)** | — | — | — | — | — | 19.14 | 31.11 | 0.37 |

## Deployment Recommendation
The **Random Forest** classifier is recommended for deployment. It produces the highest overall accuracy (0.8101) and the best F1 score (0.7385), providing reliable recall on the minority survivor class without excessive false positives.
