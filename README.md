# COMPAS Recidivism Prediction

A machine learning project that develops supervised classification models to predict recidivism risk using the publicly available COMPAS dataset from Broward County, Florida. This project trains logistic regression and XGBoost models and compares their performance against the original COMPAS risk assessment scores.

## Dataset

The project uses the COMPAS dataset from Broward County, Florida, stored in a SQLite database (`compas.db`). The dataset contains 11,038 individuals after filtering invalid target values. The target variable is `is_recid` (binary: 0 = no recidivism, 1 = recidivism). Demographic groups include African-American, Caucasian, Hispanic, and Other (Asian and Native American combined). The original COMPAS `decile_score` (1-10 risk scores) serves as the baseline for comparison.

## Data Pipeline

The data preprocessing pipeline (`preprocessing.py`) consists of four steps:

1. **Data Extraction**: Loads all columns from the `people` table, filtering out records where `is_recid = -1`
2. **Data Cleaning**: Converts null race values to "Other" and removes duplicate records
3. **Feature Engineering**: Selects features for modeling (demographics, criminal history, target variable, and decile_score)
4. **Data Splitting**: Splits data into training (70%), validation (15%), and test (15%) sets with stratification. Records with missing `decile_score` are placed only in the training set.

## Models

Two models are trained in `compas_predictions.py`:

**XGBoost**: Trained with 1000 trees, max depth 6, learning rate 0.1, and early stopping (50 rounds). Categorical variables are one-hot encoded before training.

**Logistic Regression**: Trained with C=1.0, max_iter=1000, using the 'lbfgs' solver. Features are standardized using StandardScaler.

## Evaluation

Both models are evaluated on the test set using accuracy and AUC-ROC metrics. The COMPAS baseline is converted to binary predictions using a threshold of 5 (decile scores ≥ 5 predict recidivism). The script compares XGBoost vs COMPAS, Logistic Regression vs COMPAS, and XGBoost vs Logistic Regression, indicating if either model achieves +5% higher AUC compared to COMPAS.

## Usage

To run the complete pipeline and see results:

```bash
python compas_predictions.py
```

To run just the data pipeline:

```bash
python preprocessing.py
```

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
- pandas, numpy, scikit-learn, xgboost
- sqlite3 (built-in)

## Results

The final models have been trained and evaluated. Results are printed when running `compas_predictions.py`, showing accuracy and AUC metrics for both models compared to the COMPAS baseline.

## Limitations and Ethical Considerations

The COMPAS dataset has limitations: it reflects historical biases, uses a specific recidivism definition, has missing values, and is limited to Broward County. This project is for research and educational purposes. Predictive models for recidivism raise ethical concerns about perpetuating biases and should not be used as the sole basis for decisions affecting individuals' lives.
