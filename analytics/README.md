# Module 2 — Analytics Pipeline

## Part A: Profiling & Missing Data Handling
- **Missing Value Handling Strategy**:
  - `embarked` / `embark_town` (0.22% missing, <5% threshold): Dropped affected rows.
  - `age` (19.87% missing, 5%–30% threshold): Imputed using median within pipeline.
  - `deck` (77.22% missing, >30% threshold): Dropped column entirely to avoid synthetic noise.
- **Univariate Analysis & Outliers**:
  - `age`: 8 outliers identified via IQR rule.
  - `fare`: 114 outliers identified via IQR rule.
  - `fare` Skewness: Strongly right-skewed with central tendency ordering: Mean (32.10) > Median (14.45) > Mode (8.05).
- **Bivariate Survival Rates**:
  - By Sex: Female = 74.04%, Male = 18.89%
  - By Pclass: 1st = 62.62%, 2nd = 47.28%, 3rd = 24.24%
  - By Sex & Pclass:
    - Female: Class 1 = 96.74%, Class 2 = 92.11%, Class 3 = 50.00%
    - Male: Class 1 = 36.89%, Class 2 = 15.74%, Class 3 = 13.54%
- **Correlations (6x6 Matrix)**:
  - Restricted to: `survived`, `pclass`, `age`, `sibsp`, `parch`, `fare` (redundant flags `adult_male` and `alone` excluded).
  - Top 2 off-diagonal correlations: `pclass` vs `fare` (-0.55) and `pclass` vs `age` (-0.37).

## Multivariate Visualizations & Interpretations
1. **Age vs. Fare by Survival & Pclass**: 1st class passengers paying higher fares had higher survival rates across all ages. Lower-class passengers with lower fares clustered densely near zero survival.
2. **Survival Probability by Pclass & Sex**: Female survival was over 90% in Class 1 and 2, but dropped to 50% in Class 3. Male survival remained under 37% across all classes.
3. **Age Distribution by Pclass & Survival (Violin Plot)**: In 2nd and 3rd class, young children had higher survival rates than adults.
4. **Fare Distribution by Embarked Port & Survival (Box Plot)**: Cherbourg (C) passengers had substantially higher median fares and survival rates due to a higher proportion of 1st class passengers.

## Standardization Check (Age & Fare)
- **Before Standardization**: `age` had a mean of ~29.7; `fare` was heavily skewed with a long tail.
- **After Standardization**: Features are scaled to $\mu = 0$ and $\sigma = 1$, standardizing variance while preserving relative ordering and skewness.

## Part B: Modeling, Imbalance & Tuning
- **Stratified Split**: Applied `stratify=y` to preserve the ~38.4% survival class ratio in train and test splits.
- **Preprocessing**: Leak-free `ColumnTransformer` fit strictly on train split and applied in transform-only mode on test split.
- **Imbalance Comparison (Random Forest F1)**:
  - Baseline: **0.7519**
  - Class Weight Balanced: 0.7424
  - SMOTE (Train fold only): 0.7299
- **GridSearchCV & OOB**:
  - Best Parameters: `{'classifier__max_depth': 5, 'classifier__n_estimators': 100}`
  - Best Estimator OOB Score: **0.8143**
- **Regression Sub-Task (Fare Prediction)**:
  - MAE: 17.85 | MSE: 1644.10 | RMSE: 40.55 | R²: 0.3838
  - Residual plot shows a clear funnel pattern (variance increases with predicted fare), confirming **heteroscedasticity**.

## Model Comparison Table

| Model | Accuracy | Precision | Recall | F1 Score | AUC | MAE | RMSE | R² |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression** | 0.8146 | 0.7966 | **0.6912** | 0.7402 | **0.8596** | — | — | — |
| **Decision Tree** | 0.8034 | 0.8000 | 0.6471 | 0.7154 | 0.8481 | — | — | — |
| **Random Forest** | **0.8315** | **0.8800** | 0.6471 | **0.7458** | 0.8398 | — | — | — |
| **Linear Regression (Fare)** | — | — | — | — | — | 17.85 | 40.55 | 0.3838 |

## Deployment Recommendation
Deploy the **Random Forest** model. It achieves the best overall Accuracy (0.8315) and highest Precision (0.8800) with a strong F1 score (0.7458) and verified OOB generalization (0.8143).
