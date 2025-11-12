# COMPAS Recidivism Prediction with Fairness Analysis

A machine learning project that develops supervised classification models to predict recidivism risk using the publicly available COMPAS dataset from Broward County. This project focuses on building fair, transparent, and accurate models while addressing demographic bias in algorithmic risk assessment.

## Project Overview

This project implements logistic regression and XGBoost models to predict recidivism, achieving improved performance metrics and reduced false-positive rate disparities compared to the original COMPAS outcomes. The system includes a reproducible evaluation pipeline with fairness metrics, interactive visualizations for demographic bias analysis, and comprehensive documentation of dataset limitations and ethical considerations.

## System Architecture

### High-Level Design

```
┌─────────────────────────────────────────────────────────────────┐
│                    Data Pipeline Layer                          │
├─────────────────────────────────────────────────────────────────┤
│  SQLite Database → Data Extraction → Validation → Cleaning    │
│  → Feature Engineering → Train/Val/Test Split                   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Model Training Layer                         │
├─────────────────────────────────────────────────────────────────┤
│  Logistic Regression  │  XGBoost Classifier                    │
│  Hyperparameter Tuning │  Cross-Validation                     │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Evaluation & Fairness Layer                 │
├─────────────────────────────────────────────────────────────────┤
│  Performance Metrics  │  Fairness Metrics                      │
│  (AUC, Accuracy, etc.) │  (FPR, FNR, Demographic Parity)       │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Visualization & Reporting Layer              │
├─────────────────────────────────────────────────────────────────┤
│  Interactive Dashboards │  Bias Analysis Charts                │
│  Model Comparison Plots │  Fairness Metric Reports             │
└─────────────────────────────────────────────────────────────────┘
```

## Data Pipeline

### 1. Data Extraction

**Source**: SQLite database (`compas.db`) containing:
- `people` table: 11,757 individuals with demographics, prior offenses, and recidivism outcomes
- `compas` table: 37,578 COMPAS assessment records with risk scores
- `casearrest`, `charge`, `jailhistory`, `prisonhistory`: Additional case and criminal history data

**Extraction Process**:
- SQL queries to join relevant tables
- Filter for valid records (non-null outcomes, complete demographic information)
- Extract features and target variables

### 2. Data Validation

**Quality Checks**:
- Missing value detection and reporting
- Data type validation
- Range checks for numerical features
- Consistency checks (e.g., age vs. date of birth)
- Duplicate record identification

**Demographic Group Validation**:
- Verify race distribution (African-American, Caucasian, Hispanic, Other)
- Check sample sizes for each group
- Validate recidivism outcome distribution

### 3. Data Cleaning

**Preprocessing Steps**:
- Handle missing values (imputation or exclusion based on feature importance)
- Remove outliers using statistical methods (IQR, Z-score)
- Standardize date formats
- Normalize categorical variables
- Address data inconsistencies

**Demographic Group Consolidation**:
- Combine Asian and Native American into "Other" category for sufficient sample sizes
- Final groups: African-American, Caucasian, Hispanic, Other

### 4. Feature Engineering

**Demographic Features**:
- `race`: Categorical (one-hot encoded)
- `sex`: Binary categorical
- `age`: Continuous (also create age categories)
- `age_cat`: Categorical age groups

**Criminal History Features**:
- `priors_count`: Total prior convictions
- `juv_fel_count`: Juvenile felony count
- `juv_misd_count`: Juvenile misdemeanor count
- `juv_other_count`: Other juvenile offenses

**Temporal Features**:
- `days_b_screening_arrest`: Days between screening and arrest
- `c_days_from_compas`: Days from COMPAS assessment to case
- Age at time of assessment

**COMPAS Score Features** (for comparison):
- `decile_score`: Original COMPAS decile score (1-10)
- `score_text`: Risk level category (Low/Medium/High)

**Derived Features**:
- Total juvenile offenses
- Prior offense rate (priors per year of age)
- Time-based risk indicators

### 5. Data Splitting

**Stratified Split Strategy**:
- Train: 70% of data
- Validation: 15% of data
- Test: 15% of data

**Stratification**:
- Stratify by both target variable (`is_recid`) and race to ensure:
  - Balanced recidivism rates across splits
  - Representative demographic distribution in each split
  - Fair evaluation across demographic groups

## Model Architecture

### Model Comparison Strategy

**Baseline: Original COMPAS Model**
- The original COMPAS risk assessment scores from the dataset serve as the baseline
- These are the actual scores used in the U.S. justice system
- Goal: Beat the original COMPAS model in both effectiveness and fairness

**Our Models:**
This project trains two new model types to outperform the original COMPAS:

**Logistic Regression** serves as:
- An interpretable alternative model for understanding feature contributions
- A fast, simple model that provides probabilistic outputs
- A baseline for comparing our new models against the original COMPAS
- A model that can be easily explained to stakeholders

**XGBoost Classifier** serves as:
- A high-performance model to maximize predictive accuracy
- A model that captures non-linear relationships and feature interactions
- The primary candidate for achieving the target +5% AUC improvement over original COMPAS
- A state-of-the-art approach for tabular data classification

**Cross-Validation** is a technique (not a model) used during training of both new models to:
- Tune hyperparameters robustly without overfitting
- Get reliable performance estimates before final evaluation
- Select the best model configuration
- Ensure generalization to unseen data

**Three-Way Comparison:**
All three approaches are evaluated and compared:
1. **Original COMPAS** (baseline from dataset)
2. **Logistic Regression** (our interpretable model)
3. **XGBoost** (our high-performance model)

The goal is to demonstrate that our new models (especially XGBoost) achieve:
- +5% higher AUC than original COMPAS
- 12% reduction in false-positive rate disparities compared to original COMPAS

### Logistic Regression

**Model Type**: Binary classification with L2 regularization

**Hyperparameters**:
- `C`: Inverse regularization strength (tuned via grid search)
- `penalty`: L2 regularization
- `solver`: 'lbfgs' or 'liblinear'
- `max_iter`: Maximum iterations for convergence

**Training Process**:
- Standardize features using StandardScaler
- Grid search cross-validation for hyperparameter tuning
- 5-fold cross-validation on training set
- Fit on full training set with best parameters

**Advantages**:
- Interpretable coefficients
- Fast training and prediction
- Baseline model for comparison
- Probabilistic outputs

### XGBoost Classifier

**Model Type**: Gradient boosting ensemble

**Key Hyperparameters**:
- `n_estimators`: Number of boosting rounds (100-500)
- `max_depth`: Maximum tree depth (3-7)
- `learning_rate`: Step size shrinkage (0.01-0.3)
- `subsample`: Row sampling ratio (0.8-1.0)
- `colsample_bytree`: Column sampling ratio (0.8-1.0)
- `min_child_weight`: Minimum sum of instance weight in child
- `gamma`: Minimum loss reduction for split
- `reg_alpha`: L1 regularization
- `reg_lambda`: L2 regularization
- `scale_pos_weight`: Balance positive/negative class weights

**Training Process**:
- Randomized search or Bayesian optimization for hyperparameter tuning
- Early stopping based on validation set performance
- 5-fold cross-validation for robust evaluation
- Feature importance analysis

**Advantages**:
- High predictive performance
- Handles non-linear relationships
- Feature interaction modeling
- Built-in regularization
- Handles missing values

**Fairness Considerations**:
- Monitor feature importance for demographic features
- Use `scale_pos_weight` to address class imbalance
- Regularization to prevent overfitting to training demographics

## Evaluation Pipeline

### Performance Metrics

**Classification Metrics**:
- Area Under ROC Curve (AUC-ROC)
- Accuracy
- Precision
- Recall (Sensitivity)
- F1-Score
- Specificity

**Threshold Selection**:
- Optimize threshold based on validation set
- Consider fairness-constrained threshold selection
- Compare default 0.5 threshold vs. optimized threshold

### Fairness Metrics

**Demographic Parity**:
- Equal positive prediction rates across groups
- Measured as difference in positive prediction rates

**Equalized Odds**:
- Equal true positive rates (TPR) across groups
- Equal false positive rates (FPR) across groups

**Calibration**:
- Equal positive predictive value (PPV) across groups
- Equal negative predictive value (NPV) across groups

**Disparate Impact**:
- Ratio of positive prediction rates between groups
- Target: 0.8-1.25 range (80% rule)

**Group-Specific Metrics**:
- FPR by race: False positive rate for each demographic group
- FNR by race: False negative rate for each demographic group
- Precision by race: Positive predictive value by group
- Recall by race: True positive rate by group

**Target Improvements**:
- +5% higher AUC compared to original COMPAS outcomes
- 12% reduction in false-positive rate disparities

### Model Comparison

**Three-Model Evaluation**:
- **Original COMPAS**: Use decile scores and risk categories from the dataset as baseline predictions
- **Logistic Regression**: Our trained interpretable model
- **XGBoost**: Our trained high-performance model

**Comparison Metrics**:
- Side-by-side performance comparison (AUC, accuracy, precision, recall)
- Fairness metric comparison (FPR disparities, equalized odds, calibration)
- Statistical significance testing between models
- Trade-off analysis (accuracy vs. fairness for each model)
- Direct comparison showing improvement over original COMPAS outcomes

**Success Criteria**:
- At least one of our models (Logistic Regression or XGBoost) achieves +5% higher AUC than original COMPAS
- At least one of our models achieves 12% reduction in false-positive rate disparities compared to original COMPAS

## Visualization Components

### Interactive Dashboards

**Demographic Bias Analysis**:
- FPR/FNR by race group
- Prediction distribution by demographic
- ROC curves by demographic group
- Calibration plots by group

**Model Performance Comparison**:
- ROC curves for all models
- Precision-Recall curves
- Confusion matrices by demographic group
- Feature importance plots

**Fairness Metrics Visualization**:
- Bar charts comparing fairness metrics across groups
- Heatmaps of metric differences
- Disparate impact ratios
- Fairness-accuracy trade-off curves

**Tools**:
- Plotly for interactive visualizations
- Matplotlib/Seaborn for static plots
- Dashboard framework (Streamlit or Dash)

## Reproducibility

### Version Control
- Git repository with commit history
- Requirements file for dependencies
- Configuration files for hyperparameters
- Random seed setting for reproducibility

### Documentation
- Code comments and docstrings
- Methodology documentation
- Dataset documentation
- Ethical considerations section

### Experiment Tracking
- Model versioning
- Hyperparameter logging
- Performance metric tracking
- Fairness metric tracking

## Dataset Information

### Source
- COMPAS dataset from Broward County, Florida
- Publicly available dataset used in U.S. justice system
- Contains recidivism outcomes and COMPAS risk scores

### Dataset Statistics
- Total individuals: 11,757
- COMPAS assessments: 37,578
- Demographic distribution:
  - African-American: 5,813 (49.4%)
  - Caucasian: 4,085 (34.8%)
  - Hispanic: 1,100 (9.4%)
  - Other: 759 (6.5%)

### Limitations
- Historical data with potential selection bias
- Recidivism definition limitations
- Missing data in some fields
- Temporal constraints (data from specific time period)
- Geographic limitations (Broward County only)

## Ethical Considerations

### Algorithmic Fairness
- Acknowledgment of historical biases in criminal justice data
- Focus on reducing false-positive disparities
- Transparency in model decisions
- Regular fairness audits

### Limitations and Caveats
- Models should not be used as sole decision-making tool
- Context-specific limitations of predictive models
- Potential for perpetuating existing biases
- Need for human oversight in deployment

### Responsible Use
- Documentation of model limitations
- Clear communication of uncertainty
- Regular model monitoring and updates
- Consideration of societal impact

## Project Structure

```
COMPAS/
├── README.md                 # This file
├── compas_predictions.py     # Main analysis script
├── compas.db                 # SQLite database
├── requirements.txt          # Python dependencies
├── config.yaml              # Configuration file (optional)
├── data/                    # Processed data (if needed)
├── models/                  # Saved model files
├── results/                 # Evaluation results
├── visualizations/          # Generated plots and dashboards
└── notebooks/               # Jupyter notebooks for exploration
```

## Dependencies

- Python 3.8+
- pandas: Data manipulation
- numpy: Numerical computing
- scikit-learn: Machine learning utilities
- xgboost: Gradient boosting classifier
- sqlite3: Database access
- matplotlib/seaborn: Static visualizations
- plotly: Interactive visualizations
- streamlit/dash: Dashboard framework (optional)

## Usage

(To be completed with actual usage instructions once code is implemented)

## Results

(To be completed with actual results once models are trained and evaluated)

## License

(To be specified)

## References

- COMPAS dataset source
- Relevant research papers on algorithmic fairness
- COMPAS-related studies and critiques
