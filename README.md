# COMPAS Recidivism Prediction

This project develops supervised classification models to predict recidivism risk using the publicly available COMPAS dataset from Broward County, Florida. The goal is to build models that outperform the original COMPAS system in both accuracy and fairness.

## What This Project Does

The project trains two machine learning models (Logistic Regression and XGBoost) to predict whether someone will recidivate, then compares their performance against the original COMPAS risk assessment scores. The system includes a complete data pipeline that extracts, cleans, and prepares the data, then trains and evaluates the models.

## Dataset

The dataset comes from a SQLite database containing information about 11,038 people from Broward County who were assessed using the COMPAS system. The data includes:

- Demographics: race, sex, age
- Criminal history: prior convictions, juvenile offenses
- Outcomes: whether each person recidivated (is_recid)
- COMPAS scores: original risk assessment scores (decile_score) for comparison

The dataset has been preprocessed to combine small demographic groups (Asian and Native American were merged into "Other") to ensure sufficient sample sizes for analysis.

## Data Pipeline

The data pipeline is implemented in `preprocessing.py` and consists of four main steps:

**1. Data Extraction**
- Loads all columns from the SQLite database
- Filters out records with invalid target values (is_recid = -1)
- Returns a pandas DataFrame with all available data

**2. Data Cleaning**
- Handles missing race values by converting them to "Other"
- Removes duplicate records (rows with identical data except for ID)
- Reports how many duplicates were found and removed

**3. Feature Engineering**
- Selects relevant features for modeling: race, sex, age, priors_count, and juvenile offense counts
- Keeps the target variable (is_recid) and COMPAS decile_score for comparison
- Drops unnecessary columns like names, dates, and case numbers

**4. Data Splitting**
- Splits data into 70% training, 15% validation, and 15% test sets
- Uses stratified splitting to maintain class balance across splits
- Handles missing decile_score values by putting them only in the training set (since they can't be used for COMPAS comparison)
- Returns separate X (features) and y (target) arrays for each split, plus decile_score arrays for comparison

The pipeline can be run using the `run_pipeline()` function, which executes all steps and returns the prepared data.

## Models

**Logistic Regression**
- A simple, interpretable linear model
- Uses L2 regularization (C=1.0)
- Features are standardized before training
- Fast to train (typically completes in seconds)

**XGBoost**
- A gradient boosting ensemble model
- Uses 1000 trees with early stopping
- Hyperparameters: max_depth=6, learning_rate=0.1
- Takes longer to train (typically 2-5 minutes) but often achieves better performance

Both models use one-hot encoding for categorical variables (race and sex) since XGBoost requires numeric input and Logistic Regression works better with encoded categoricals.

## Evaluation

The models are evaluated using two main metrics:

**Accuracy**: The percentage of correct predictions (0 or 1 for recidivism)

**AUC-ROC**: Area Under the ROC Curve, which measures how well the model can distinguish between people who recidivate and those who don't. This is generally considered a better metric than accuracy for imbalanced datasets.

The evaluation compares:
- XGBoost performance vs COMPAS baseline
- Logistic Regression performance vs COMPAS baseline  
- XGBoost vs Logistic Regression to see which performs better

The COMPAS baseline is created by converting the original decile scores (1-10) to binary predictions using a threshold of 5 (decile >= 5 predicts recidivism).

## Current Implementation

The main script is `compas_predictions.py`, which:

1. Loads and preprocesses the data using the pipeline
2. Encodes categorical variables using one-hot encoding
3. Trains both XGBoost and Logistic Regression models
4. Makes predictions on the test set
5. Calculates accuracy and AUC for both models
6. Compares both models to the COMPAS baseline
7. Prints a detailed comparison report

The script shows training progress for XGBoost and reports when training is complete. It then displays performance metrics for all three approaches (XGBoost, Logistic Regression, and COMPAS) side by side.

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
- pandas: Data manipulation
- numpy: Numerical computing
- scikit-learn: Machine learning utilities and metrics
- xgboost: Gradient boosting classifier
- sqlite3: Database access (built into Python)

## Usage

To run the complete pipeline and train the models:

```bash
python compas_predictions.py
```

This will:
1. Extract and clean the data
2. Train both models
3. Evaluate performance
4. Compare results to COMPAS baseline

The output shows detailed performance metrics and comparisons for all models.

## Dataset Limitations

This dataset has several important limitations:

- Historical data that may reflect past biases in the criminal justice system
- Recidivism is defined in a specific way that may not capture all relevant outcomes
- Some data fields have missing values
- Data is from a specific time period and geographic location (Broward County)
- The dataset doesn't include information about interventions, rehabilitation programs, or other factors that might affect recidivism

These limitations should be considered when interpreting model results.

## Ethical Considerations

This project is designed to evaluate and potentially improve upon existing risk assessment tools, but it's important to recognize:

- Predictive models should not be the sole basis for decisions affecting people's lives
- Models trained on historical data may perpetuate existing biases
- The goal is to build fairer models, but fairness is complex and context-dependent
- Any use of these models in practice would require careful consideration of ethical implications and human oversight

The project focuses on transparency by comparing model performance and documenting the process, but does not claim to solve all fairness issues in algorithmic risk assessment.
