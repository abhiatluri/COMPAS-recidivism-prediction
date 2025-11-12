# COMPAS Recidivism Prediction

A machine learning project that develops supervised classification models to predict recidivism risk using the publicly available COMPAS dataset from Broward County, Florida. This project trains logistic regression and XGBoost models and compares their performance against the original COMPAS risk assessment scores.

## Project Overview

This project implements two machine learning models (logistic regression and XGBoost) to predict recidivism and evaluates their performance against the original COMPAS system. The goal is to achieve better predictive accuracy and fairness compared to the COMPAS baseline.

## Dataset

The project uses the COMPAS dataset from Broward County, Florida, stored in a SQLite database (`compas.db`). The dataset contains information about individuals who were assessed using the COMPAS risk assessment tool.

**Dataset Statistics:**
- Total individuals: 11,038 (after filtering invalid target values)
- Target variable: `is_recid` (binary: 0 = no recidivism, 1 = recidivism)
- Demographic groups: African-American, Caucasian, Hispanic, Other (Asian and Native American combined into Other)
- Baseline: Original COMPAS `decile_score` (1-10 risk scores)

## Data Pipeline

The data preprocessing pipeline is implemented in `preprocessing.py` and consists of four main steps:

### 1. Data Extraction

The `extract_data()` function loads all columns from the `people` table in the SQLite database. It filters out records where `is_recid = -1` (invalid/missing target values) to ensure we only work with valid recidivism outcomes.

### 2. Data Cleaning

The `clean_data()` function handles data quality issues:
- Converts null race values to "Other" category
- Removes duplicate records (rows with identical values across all columns except ID)
- Reports the number of duplicates found and removed

The function uses all available columns to identify duplicates, which helps distinguish between truly duplicate records and records that happen to share some common features.

### 3. Feature Engineering

The `engineer_features()` function selects the features used for modeling:
- Demographics: `race`, `sex`, `age`
- Criminal history: `priors_count`, `juv_fel_count`, `juv_misd_count`, `juv_other_count`
- Target variable: `is_recid`
- Baseline comparison: `decile_score` (original COMPAS scores)

Other columns (names, dates, case numbers, etc.) are dropped as they are not needed for prediction.

### 4. Data Splitting

The `split_data()` function splits the data into training (70%), validation (15%), and test (15%) sets. The split is stratified by the target variable to maintain balanced class distributions across splits.

A special consideration is made for records with missing `decile_score` values. Since COMPAS comparison requires decile scores, records without them are placed only in the training set. The remaining records with decile scores are split proportionally to achieve the exact 70-15-15 distribution.

The function returns feature matrices (X) and target vectors (y) for each split, plus decile score arrays for validation and test sets (used for COMPAS comparison).

## Model Training

Model training is implemented in `compas_predictions.py`. The script trains two models and compares them to the COMPAS baseline.

### Data Preparation

Before training, categorical variables (`race` and `sex`) are one-hot encoded since XGBoost requires numeric input. The encoded features are aligned across train, validation, and test sets to ensure consistent column structures.

### XGBoost Model

The XGBoost classifier is trained with the following hyperparameters:
- `n_estimators`: 1000 (number of trees)
- `max_depth`: 6 (maximum tree depth)
- `learning_rate`: 0.1 (step size shrinkage)
- `early_stopping_rounds`: 50 (stops training if validation performance doesn't improve)

The model uses early stopping based on validation set performance to prevent overfitting. Training progress is printed every 100 trees.

### Logistic Regression Model

The logistic regression model uses:
- `C`: 1.0 (regularization strength, inverse of L2 regularization)
- `max_iter`: 1000 (maximum iterations)
- `solver`: 'lbfgs' (optimization algorithm)

Features are standardized using `StandardScaler` before training, which is important for logistic regression convergence and performance.

## Evaluation

Both models are evaluated on the test set using accuracy and AUC-ROC metrics. The COMPAS baseline is also evaluated for comparison.

### COMPAS Baseline

The original COMPAS `decile_score` (ranging from 1 to 10) is converted to binary predictions using a threshold of 5. Scores of 5 or higher are treated as predicting recidivism. The decile scores are also normalized to a 0-1 range for AUC calculation.

### Performance Comparison

The script compares:
- XGBoost vs COMPAS baseline (accuracy and AUC)
- Logistic Regression vs COMPAS baseline (accuracy and AUC)
- XGBoost vs Logistic Regression (accuracy and AUC)

Results are printed showing the performance of each model and the differences between them. The script indicates if either model achieves the target of +5% higher AUC compared to COMPAS.

## Project Structure

```
COMPAS/
├── README.md                 # This file
├── preprocessing.py          # Data pipeline functions
├── compas_predictions.py     # Model training and evaluation
├── compas.db                 # SQLite database
└── .gitignore               # Git ignore file
```

## Dependencies

- Python 3.8+
- pandas: Data manipulation and analysis
- numpy: Numerical computing
- scikit-learn: Machine learning utilities (LogisticRegression, StandardScaler, train_test_split, metrics)
- xgboost: Gradient boosting classifier
- sqlite3: Database access (built-in Python module)

## Usage

To run the complete pipeline and train models:

```bash
python compas_predictions.py
```

This will:
1. Load and preprocess the data
2. Train both XGBoost and Logistic Regression models
3. Evaluate performance on test set
4. Compare results to COMPAS baseline
5. Print detailed performance metrics

To run just the data pipeline:

```bash
python preprocessing.py
```

This executes the preprocessing steps and prints information about the data processing.

## Implementation Details

### Data Pipeline Function

The `run_pipeline()` function in `preprocessing.py` executes the complete data preprocessing pipeline and returns the train/validation/test splits along with decile score arrays for comparison. This function can be imported and used by other scripts.

### Model Training Process

The training script:
1. Imports the data pipeline function to get processed data
2. One-hot encodes categorical variables
3. Trains XGBoost model (with early stopping)
4. Trains Logistic Regression model (with feature scaling)
5. Makes predictions on all sets
6. Calculates accuracy and AUC metrics
7. Compares all models and prints results

### Reproducibility

Random seeds are set (`random_state=42`) in both the data splitting and model training to ensure reproducible results across runs.

## Dataset Limitations

The COMPAS dataset has several limitations that should be considered:
- Historical data that may reflect past biases in the criminal justice system
- Recidivism is defined based on rearrest within a specific time period, which may not capture all relevant outcomes
- Some records have missing values, particularly for `decile_score`
- Data is from a specific geographic area (Broward County) and time period, limiting generalizability
- The dataset does not include information about interventions, rehabilitation programs, or other factors that might influence recidivism

## Ethical Considerations

This project is designed for research and educational purposes. Predictive models for recidivism risk assessment raise important ethical concerns:

- Models trained on historical data may perpetuate existing biases in the criminal justice system
- Predictive accuracy does not necessarily mean the model is fair or appropriate for use in decision-making
- Demographic features in the model could lead to discriminatory outcomes if not carefully managed
- Models should not be used as the sole basis for decisions affecting individuals' lives

The models in this project are evaluated for predictive performance, but fairness analysis (such as false-positive rate disparities by demographic group) is not yet implemented. Any real-world application would require thorough fairness auditing and consideration of ethical implications.
