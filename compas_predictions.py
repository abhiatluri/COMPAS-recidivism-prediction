from preprocessing import run_pipeline
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score

# Get data (including decile_score for COMPAS comparison)
X_train, X_val, X_test, y_train, y_val, y_test, decile_val, decile_test = run_pipeline()

# Encode categorical variables (race, sex) using one-hot encoding
# XGBoost requires numeric or category types, not object (string) types
categorical_cols = ['race', 'sex']

# One-hot encode categorical variables
X_train_encoded = pd.get_dummies(X_train, columns=categorical_cols, prefix=categorical_cols)
X_val_encoded = pd.get_dummies(X_val, columns=categorical_cols, prefix=categorical_cols)
X_test_encoded = pd.get_dummies(X_test, columns=categorical_cols, prefix=categorical_cols)

# Ensure all sets have the same columns (in case some categories are missing in val/test)
# Align columns to match training set
X_val_encoded = X_val_encoded.reindex(columns=X_train_encoded.columns, fill_value=0)
X_test_encoded = X_test_encoded.reindex(columns=X_train_encoded.columns, fill_value=0)

# Drop 'id' column if present (not needed for training)
if 'id' in X_train_encoded.columns:
    X_train_encoded = X_train_encoded.drop(columns=['id'])
    X_val_encoded = X_val_encoded.drop(columns=['id'])
    X_test_encoded = X_test_encoded.drop(columns=['id'])

# ============================================================================
# TRAIN XGBOOST MODEL
# ============================================================================
print("\n" + "=" * 60)
print("TRAINING XGBOOST MODEL")
print("=" * 60)
print("Hyperparameters:")
print(f"  n_estimators (trees): 1000")
print(f"  max_depth: 6")
print(f"  learning_rate: 0.1")
print("This may take a few minutes...\n")

xgb_model = XGBClassifier(
    n_estimators=1000,      # Number of trees (increased for better performance)
    max_depth=6,            # Tree depth
    learning_rate=0.1,       # Learning rate (lower = more careful learning)
    verbosity=1,            # Show progress
    random_state=42,        # For reproducibility
    early_stopping_rounds=50  # Stop if no improvement for 50 rounds
)

# Train with early stopping on validation set
xgb_model.fit(
    X_train_encoded, y_train,
    eval_set=[(X_val_encoded, y_val)],
    verbose=100  # Print every 100 trees
)
print("✓ XGBoost training complete!\n")

# ============================================================================
# TRAIN LOGISTIC REGRESSION MODEL
# ============================================================================
print("=" * 60)
print("TRAINING LOGISTIC REGRESSION MODEL")
print("=" * 60)
print("Hyperparameters:")
print(f"  C (regularization): 1.0")
print(f"  max_iter: 1000")
print("Training...\n")

# Standardize features for logistic regression (important for convergence)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_encoded)
X_val_scaled = scaler.transform(X_val_encoded)
X_test_scaled = scaler.transform(X_test_encoded)

lr_model = LogisticRegression(
    C=1.0,                  # Regularization strength
    max_iter=1000,          # Maximum iterations
    random_state=42,        # For reproducibility
    solver='lbfgs'          # Good solver for this problem
)

lr_model.fit(X_train_scaled, y_train)
print("✓ Logistic Regression training complete!\n")

# Make predictions for both models
print("Making predictions...")

# XGBoost predictions
xgb_pred_train = xgb_model.predict(X_train_encoded)
xgb_pred_val = xgb_model.predict(X_val_encoded)
xgb_pred_test = xgb_model.predict(X_test_encoded)
xgb_pred_proba_test = xgb_model.predict_proba(X_test_encoded)[:, 1]

# Logistic Regression predictions
lr_pred_train = lr_model.predict(X_train_scaled)
lr_pred_val = lr_model.predict(X_val_scaled)
lr_pred_test = lr_model.predict(X_test_scaled)
lr_pred_proba_test = lr_model.predict_proba(X_test_scaled)[:, 1]

# Calculate XGBoost metrics
xgb_train_acc = accuracy_score(y_train, xgb_pred_train)
xgb_val_acc = accuracy_score(y_val, xgb_pred_val)
xgb_test_acc = accuracy_score(y_test, xgb_pred_test)
xgb_test_auc = roc_auc_score(y_test, xgb_pred_proba_test)

# Calculate Logistic Regression metrics
lr_train_acc = accuracy_score(y_train, lr_pred_train)
lr_val_acc = accuracy_score(y_val, lr_pred_val)
lr_test_acc = accuracy_score(y_test, lr_pred_test)
lr_test_auc = roc_auc_score(y_test, lr_pred_proba_test)

print("=" * 60)
print("XGBOOST MODEL PERFORMANCE")
print("=" * 60)
print(f"Training Accuracy:   {xgb_train_acc:.4f} ({xgb_train_acc*100:.2f}%)")
print(f"Validation Accuracy: {xgb_val_acc:.4f} ({xgb_val_acc*100:.2f}%)")
print(f"Test Accuracy:       {xgb_test_acc:.4f} ({xgb_test_acc*100:.2f}%)")
print(f"Test AUC-ROC:        {xgb_test_auc:.4f}")
print()

print("=" * 60)
print("LOGISTIC REGRESSION MODEL PERFORMANCE")
print("=" * 60)
print(f"Training Accuracy:   {lr_train_acc:.4f} ({lr_train_acc*100:.2f}%)")
print(f"Validation Accuracy: {lr_val_acc:.4f} ({lr_val_acc*100:.2f}%)")
print(f"Test Accuracy:       {lr_test_acc:.4f} ({lr_test_acc*100:.2f}%)")
print(f"Test AUC-ROC:        {lr_test_auc:.4f}")
print()

# COMPAS Baseline Comparison
# Convert COMPAS decile_score (1-10) to binary predictions
# Higher decile = higher risk = predict recidivism
# Use threshold: decile >= 5 means predict recidivism (1), else 0
compas_threshold = 5
compas_pred_test = (decile_test >= compas_threshold).astype(int)

# Calculate COMPAS accuracy
compas_accuracy = accuracy_score(y_test, compas_pred_test)

# Calculate COMPAS AUC (convert decile to probability-like score)
# Normalize decile (1-10) to 0-1 range for AUC calculation
decile_normalized = (decile_test - 1) / 9  # Maps 1->0, 10->1
compas_auc = roc_auc_score(y_test, decile_normalized)

print("=" * 60)
print("COMPAS BASELINE PERFORMANCE")
print("=" * 60)
print(f"Test Accuracy:       {compas_accuracy:.4f} ({compas_accuracy*100:.2f}%)")
print(f"Test AUC-ROC:        {compas_auc:.4f}")
print()

# Comparison: All models vs COMPAS
print("=" * 60)
print("COMPARISON: All Models vs COMPAS Baseline")
print("=" * 60)

# XGBoost vs COMPAS
xgb_acc_diff = xgb_test_acc - compas_accuracy
xgb_auc_diff = xgb_test_auc - compas_auc

print("\nXGBoost vs COMPAS:")
print(f"  Accuracy: {xgb_test_acc:.4f} vs {compas_accuracy:.4f} ({xgb_acc_diff:+.4f}, {xgb_acc_diff*100:+.2f}%)")
print(f"  AUC:      {xgb_test_auc:.4f} vs {compas_auc:.4f} ({xgb_auc_diff:+.4f}, {xgb_auc_diff*100:+.2f}%)")
if xgb_auc_diff >= 0.05:
    print(f"  ✓ XGBoost achieved +5% higher AUC target!")

# Logistic Regression vs COMPAS
lr_acc_diff = lr_test_acc - compas_accuracy
lr_auc_diff = lr_test_auc - compas_auc

print("\nLogistic Regression vs COMPAS:")
print(f"  Accuracy: {lr_test_acc:.4f} vs {compas_accuracy:.4f} ({lr_acc_diff:+.4f}, {lr_acc_diff*100:+.2f}%)")
print(f"  AUC:      {lr_test_auc:.4f} vs {compas_auc:.4f} ({lr_auc_diff:+.4f}, {lr_auc_diff*100:+.2f}%)")
if lr_auc_diff >= 0.05:
    print(f"  ✓ Logistic Regression achieved +5% higher AUC target!")

# XGBoost vs Logistic Regression
xgb_vs_lr_acc = xgb_test_acc - lr_test_acc
xgb_vs_lr_auc = xgb_test_auc - lr_test_auc

print("\nXGBoost vs Logistic Regression:")
print(f"  Accuracy: {xgb_test_acc:.4f} vs {lr_test_acc:.4f} ({xgb_vs_lr_acc:+.4f}, {xgb_vs_lr_acc*100:+.2f}%)")
print(f"  AUC:      {xgb_test_auc:.4f} vs {lr_test_auc:.4f} ({xgb_vs_lr_auc:+.4f}, {xgb_vs_lr_auc*100:+.2f}%)")
if xgb_test_acc > lr_test_acc:
    print(f"  → XGBoost performs better")
elif lr_test_acc > xgb_test_acc:
    print(f"  → Logistic Regression performs better")
else:
    print(f"  → Models perform similarly")

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"Best Model Accuracy: {max(xgb_test_acc, lr_test_acc, compas_accuracy):.4f} ({max(xgb_test_acc, lr_test_acc, compas_accuracy)*100:.2f}%)")
print(f"Best Model AUC:      {max(xgb_test_auc, lr_test_auc, compas_auc):.4f}")
print("=" * 60)