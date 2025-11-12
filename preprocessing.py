import pandas as pd
import numpy as np
import sqlite3
from sklearn.model_selection import train_test_split

def extract_data():
    # Load ALL columns from database
    # We'll select which features to use later in engineer_features
    # This gives clean_data more columns to differentiate duplicates
    conn = sqlite3.connect('/Users/abhiramatluri/Documents/COMPAS/compas.db')
    query = """
    SELECT * FROM people WHERE is_recid != -1
    """
    df = pd.read_sql_query(query, conn)
    print(f"Number of people: {len(df)}")
    print(f"Number of columns: {len(df.columns)}")
    conn.close()
    return df


def clean_data(data):
    # Fix problems
    print(f"\nStarting clean_data with {len(data)} people")
    
    # Find and set null race values to "Other"
    null_race_count = data['race'].isnull().sum()
    if null_race_count > 0:
        print(f"Found {null_race_count} people with null race. Changing to 'Other'.")
        data['race'] = data['race'].fillna('Other')
    else:
        print("No null race values found.")
    
    # Delete all duplicates
    # If IDs are different, but all other values are the same, they are duplicates
    # Check all columns except 'id'
    other_cols = [col for col in data.columns if col != 'id']
    before_dup = len(data)
    duplicates = data.duplicated(subset=other_cols)
    if duplicates.any():
        num_duplicates = duplicates.sum()
        print(f"Found {num_duplicates} duplicate rows. Dropping duplicates.")
        data = data.drop_duplicates(subset=other_cols, keep='first')
        after_dup = len(data)
        print(f"Dropped {before_dup - after_dup} duplicate rows. {after_dup} people remaining.")
    else:
        print("No duplicates found.")
    
    print(f"clean_data complete: {len(data)} people remaining\n")
    return data

def engineer_features(data):
    # Select only the features we need for modeling
    # Drop columns we don't need (like names, dates, case numbers, etc.)
    # Keep: demographics, criminal history, target, and decile_score for comparison
    
    features_to_keep = [
        'id',  # Keep for reference (will drop in split_data)
        'race',
        'sex', 
        'age',
        'priors_count',
        'juv_fel_count',
        'juv_misd_count',
        'juv_other_count',
        'is_recid',  # Target variable
        'decile_score'  # For COMPAS comparison
    ]
    
    # Only keep columns that exist in the data
    available_features = [col for col in features_to_keep if col in data.columns]
    data = data[available_features]
    
    print(f"\nengineer_features: Selected {len(available_features)} features for modeling")
    print(f"Features: {', '.join(available_features)}")
    
    return data

def split_data(data):

    # Split data into train (70%), validation (15%), and test (15%) sets
    # NULL decile_score values can ONLY be used in training set (can't compare to COMPAS without it)
    # This ensures exactly 70-15-15 split while handling missing decile_score
    
    # 1. Separate X (features), y (target), and decile_score (for comparison)
    X = data.drop(columns=['is_recid', 'decile_score'])
    y = data['is_recid']
    decile = data['decile_score']
    
    # 2. Separate by decile_score presence
    # Rows with decile_score can go to any set (need for comparison)
    # Rows without decile_score must go to training only
    has_decile_mask = decile.notna()
    missing_decile_mask = decile.isna()
    
    X_has_decile = X[has_decile_mask]
    y_has_decile = y[has_decile_mask]
    decile_has = decile[has_decile_mask]
    
    X_missing_decile = X[missing_decile_mask]
    y_missing_decile = y[missing_decile_mask]
    
    # 3. Calculate target sizes for exactly 70-15-15 split
    total = len(data)
    target_train = int(0.70 * total)
    target_val = int(0.15 * total)
    target_test = int(0.15 * total)
    
    # 4. Calculate how much to take from has_decile data
    # All missing_decile goes to training, so subtract that from target
    train_from_has = target_train - len(X_missing_decile)
    
    # 5. Split has_decile data to fill remaining train, plus all val and test
    # First, separate test set (15% of total)
    # Use proportion instead of absolute number for stratification to work properly
    test_proportion = target_test / len(X_has_decile)
    X_temp, X_test, y_temp, y_test, decile_temp, decile_test = train_test_split(
        X_has_decile, y_has_decile, decile_has,
        test_size=test_proportion,
        stratify=y_has_decile,  # Maintain class balance
        random_state=42
    )
    
    # Then split remaining into train/val
    # Use proportion: we want target_val from remaining data
    val_proportion = target_val / len(X_temp)
    X_train_has, X_val, y_train_has, y_val, decile_train_has, decile_val = train_test_split(
        X_temp, y_temp, decile_temp,
        test_size=val_proportion,
        stratify=y_temp,  # Maintain class balance
        random_state=42
    )
    
    # 6. Combine: add missing decile rows to training set
    # This gives us exactly 70% training (train_from_has + missing_decile)
    X_train = pd.concat([X_train_has, X_missing_decile], ignore_index=True)
    y_train = pd.concat([y_train_has, y_missing_decile], ignore_index=True)
    
    # 7. Get decile_score arrays for COMPAS comparison (only for val and test)
    # Training set has missing decile_score, so we only return for val/test
    decile_val_array = decile_val.values
    decile_test_array = decile_test.values
    
    # Print final split sizes for verification
    print(f"\nFinal split sizes:")
    print(f"  Training: {len(X_train)} ({100*len(X_train)/total:.1f}%)")
    print(f"  Validation: {len(X_val)} ({100*len(X_val)/total:.1f}%)")
    print(f"  Test: {len(X_test)} ({100*len(X_test)/total:.1f}%)")
    
    return X_train, X_val, X_test, y_train, y_val, y_test, decile_val_array, decile_test_array


def run_pipeline():
    # Run the full preprocessing pipeline
    raw = extract_data()
    cleaned = clean_data(raw)
    features = engineer_features(cleaned)
    X_train, X_val, X_test, y_train, y_val, y_test, decile_val, decile_test = split_data(features)
    return X_train, X_val, X_test, y_train, y_val, y_test, decile_val, decile_test


# Only run pipeline if this file is executed directly (not when imported)
if __name__ == "__main__":
    X_train, X_val, X_test, y_train, y_val, y_test, decile_val, decile_test = run_pipeline()