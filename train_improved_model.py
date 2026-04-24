"""
IMPROVED DISEASE PREDICTION MODEL TRAINING PIPELINE
======================================================
Implements best practices in machine learning:
- XGBoost: More powerful than RandomForest
- GridSearchCV: Hyperparameter tuning with cross-validation
- StratifiedKFold: Prevents class imbalance issues
- CalibratedClassifierCV: Ensures realistic probability estimates
- Cross-validation: Robust evaluation across folds
- Feature engineering: Create meaningful interactions
- Detailed metrics: Classification report, confusion matrix
- Overfitting prevention: Max depth, min samples constraints
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import warnings
import logging
from pathlib import Path

# Scikit-learn imports
from sklearn.model_selection import (
    train_test_split, 
    GridSearchCV, 
    cross_val_score,
    KFold,
    StratifiedKFold,
    cross_validate
)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, 
    classification_report, 
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    auc,
    f1_score
)
from sklearn.calibration import CalibratedClassifierCV

# XGBoost - More powerful than RandomForest
from xgboost import XGBClassifier

# Imbalanced learning - Handle class imbalance
from imblearn.over_sampling import SMOTE

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)

# ============================================================================
# PATH DEFINITIONS
# ============================================================================

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DATA_PATH = os.getenv(
    "DATA_PATH",
    os.path.join(BASE_DIR, "healthcare-chatbot", "Data")
)
DATA_TRAIN_PATH = os.path.join(BASE_DATA_PATH, "Train")
PROGRESSION_PATH = os.path.join(BASE_DATA_PATH, "Progression", "chronic_disease_progression.csv")
SUPPORT_PATH = os.path.join(BASE_DATA_PATH, "Support")

logging.info(f"Working directory: {os.getcwd()}")
logging.info(f"Data path: {os.path.abspath(DATA_TRAIN_PATH)}")

# ============================================================================
# STEP 1: LOAD DATA
# ============================================================================

def load_and_clean_datasets():
    """
    Load and clean all training datasets from DATA_TRAIN_PATH.
    Handles heterogeneous datasets by standardizing column names and aligning schemas.
    Concatenates multiple CSV files, removes unnamed columns, validates target column.
    Returns: X (features), y (target), column names
    """
    logging.info("=" * 70)
    logging.info("LOADING AND CLEANING DATASETS")
    logging.info("=" * 70)
    
    datasets = []
    loaded_files = 0
    all_columns = set()
    
    # Get all CSV files in DATA_TRAIN_PATH
    if not os.path.exists(DATA_TRAIN_PATH):
        logging.error(f"Data train path does not exist: {DATA_TRAIN_PATH}")
        raise FileNotFoundError(f"Data train path not found: {DATA_TRAIN_PATH}")
    
    for filename in os.listdir(DATA_TRAIN_PATH):
        if filename.endswith('.csv'):
            file_path = os.path.join(DATA_TRAIN_PATH, filename)
            if not os.path.exists(file_path):
                logging.warning(f"File not found: {file_path}")
                continue
            try:
                df = pd.read_csv(file_path)
                # Remove unnamed columns
                df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
                
                # Standardize column names
                df.columns = df.columns.str.strip().str.lower()
                
                # Check and standardize target column
                if 'prognosis' in df.columns:
                    df.rename(columns={'prognosis': 'disease'}, inplace=True)
                elif 'disease' not in df.columns:
                    logging.warning(f"No target column ('prognosis' or 'disease') found in {filename}, skipping")
                    continue
                
                # Remove non-symptom columns
                exclude_keywords = ['_id', 'patient', 'hospital', 'date', 'name']
                df = df[
                    [
                        col for col in df.columns
                        if col == 'disease' or not any(k in col.lower() for k in exclude_keywords)
                    ]
                ]
                
                # Collect all feature columns (exclude target)
                all_columns.update([col for col in df.columns if col != 'disease'])
                
                datasets.append(df)
                loaded_files += 1
                logging.info(f"✓ Loaded {filename}: {df.shape}")
            except Exception as e:
                logging.warning(f"Error loading {filename}: {e}")
                continue
    
    if len(datasets) == 0:
        raise ValueError("No valid datasets loaded. Check column names and DATA_TRAIN_PATH.")
    
    # Align datasets by adding missing columns with 0
    aligned_datasets = []
    for df in datasets:
        for col in all_columns:
            if col not in df.columns:
                df[col] = 0
        aligned_datasets.append(df)
    
    # Concatenate all aligned datasets
    combined_df = pd.concat(aligned_datasets, ignore_index=True)
    
    # Remove duplicate columns
    combined_df = combined_df.loc[:, ~combined_df.columns.duplicated()]
    
    # Separate features and target
    X = combined_df.drop('disease', axis=1)
    y = combined_df['disease']
    
    # Ensure all features are numeric
    X = X.apply(pd.to_numeric, errors='coerce').fillna(0)
    
    # Ensure stable feature order
    feature_names = sorted(X.columns.tolist())
    X = X[feature_names]
    
    logging.info(f"✓ Combined dataset loaded successfully")
    logging.info(f"  - Total files loaded: {loaded_files}")
    logging.info(f"  - Combined dataset shape: {X.shape}")
    logging.info(f"  - Number of features (symptoms): {X.shape[1]}")
    logging.info(f"  - Number of samples: {X.shape[0]}")
    logging.info(f"  - Number of unique diseases: {y.nunique()}")
    logging.info(f"  - Class distribution:\n{y.value_counts()}\n")
    
    return X, y, feature_names


def load_data():
    """
    Load classification dataset using load_and_clean_datasets().
    Returns: X (features), y (target), column names
    """
    logging.info("=" * 70)
    logging.info("STEP 1: LOADING DATA")
    logging.info("=" * 70)
    
    return load_and_clean_datasets()


# ============================================================================
# STEP 2: HANDLE CLASS IMBALANCE WITH SMOTE
# ============================================================================

def handle_class_imbalance(X_train, y_train):
    """
    Apply SMOTE (Synthetic Minority Over-sampling Technique) to handle
    class imbalance by generating synthetic samples for minority classes.
    
    Returns: Balanced X_train, y_train
    """
    logging.info("\n" + "=" * 70)
    logging.info("STEP 2: HANDLING CLASS IMBALANCE WITH SMOTE")
    logging.info("=" * 70)
    
    # Calculate initial class distribution
    unique, counts = np.unique(y_train, return_counts=True)
    logging.info(f"Before SMOTE:")
    logging.info(f"  Class distribution: {dict(zip(unique, counts))}")
    
    # Apply SMOTE only if there's class imbalance
    if len(np.unique(y_train)) > 1 and counts.std() > counts.mean() * 0.3:
        # Calculate minimum class size to determine safe k_neighbors
        min_class_size = min(counts)
        k_neighbors = max(1, min(3, min_class_size - 1))
        
        smote = SMOTE(random_state=42, k_neighbors=k_neighbors)
        try:
            X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
            logging.info(f"\nAfter SMOTE:")
            unique_after, counts_after = np.unique(y_train_balanced, return_counts=True)
            logging.info(f"  Class distribution: {dict(zip(unique_after, counts_after))}")
            logging.info(f"  New training set size: {X_train_balanced.shape[0]} (was {X_train.shape[0]})\n")
            return X_train_balanced, y_train_balanced
        except Exception as e:
            logging.warning(f"⚠ SMOTE failed: {e}. Using original data.\n")
            return X_train, y_train
    else:
        logging.info(f"  ✓ No significant class imbalance detected. Using original data.\n")
        return X_train, y_train


# ============================================================================
# STEP 3: FEATURE ENGINEERING
# ============================================================================

def engineer_features(X):
    """
    Create interaction features to capture disease symptom interactions.
    Example: fever + cough -> new feature 'fever_and_cough'
    
    This helps the model learn complex symptom relationships.
    """
    logging.info("\n" + "=" * 70)
    logging.info("STEP 3: FEATURE ENGINEERING")
    logging.info("=" * 70)
    
    X_engineered = X.copy()
    n_original = X.shape[1]
    
    # Get top symptoms by variance
    top_symptoms = X.var().nlargest(10).index.tolist()
    
    # Create interaction features (limited to prevent dimensionality explosion)
    interaction_count = 0
    for i in range(len(top_symptoms)):
        for j in range(i + 1, min(i + 3, len(top_symptoms))):  # Limit interactions
            symptom1 = top_symptoms[i]
            symptom2 = top_symptoms[j]
            # Create AND interaction: new feature is 1 when both symptoms present
            feature_name = f"{symptom1}_and_{symptom2}"
            X_engineered[feature_name] = (X[symptom1] * X[symptom2]).astype(int)
            interaction_count += 1
    
    logging.info(f"✓ Features engineered successfully")
    logging.info(f"  - Original features: {n_original}")
    logging.info(f"  - Interaction features created: {interaction_count}")
    logging.info(f"  - Total features: {X_engineered.shape[1]}\n")
    
    return X_engineered


# ============================================================================
# STEP 4: PREPROCESS DATA
# ============================================================================

def preprocess_data(X, y):
    """
    Filter rare classes first, then encode labels once and prepare
    the train/test split with stratification.
    """
    logging.info("\n" + "=" * 70)
    logging.info("STEP 4: PREPROCESSING DATA")
    logging.info("=" * 70)
    
    # Filter rare classes before encoding so labels remain contiguous
    class_counts = y.value_counts()
    valid_classes = class_counts[class_counts >= 5].index
    mask = y.isin(valid_classes)
    X = X[mask]
    y = y[mask]
    logging.info(f"  - Removed {len(class_counts) - len(valid_classes)} rare classes (with <2 samples)")
    
    # Reset indices to ensure contiguous indexing
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)
    
    # Encode filtered labels once; no remapping is needed later
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # Apply feature engineering
    X_engineered = engineer_features(X)
    
    # Train-test split with stratification (preserves class distribution)
    X_train, X_test, y_train, y_test = train_test_split(
        X_engineered, 
        y_encoded,
        test_size=0.2,  # 80% train, 20% test
        random_state=42,
        stratify=y_encoded  # ← Important: maintains class distribution
    )
    
    logging.info(f"✓ Data preprocessing complete")
    logging.info(f"  - Training set: {X_train.shape}")
    logging.info(f"  - Test set: {X_test.shape}")
    logging.info(f"  - Total features: {X_train.shape[1]}")
    logging.info(f"  - Label encoder classes: {le.classes_}\n")
    
    num_classes = len(le.classes_)
    return X_train, X_test, y_train, y_test, le, num_classes


# ============================================================================
# STEP 5: HYPERPARAMETER TUNING WITH GRIDSEARCHCV
# ============================================================================

def tune_hyperparameters(X_train, y_train, num_classes):
    """
    Use GridSearchCV to find optimal hyperparameters for XGBoost.
    Searches across multiple parameter combinations using cross-validation.
    """
    logging.info("\n" + "=" * 70)
    logging.info("STEP 5: HYPERPARAMETER TUNING WITH GRIDSEARCHCV")
    logging.info("=" * 70)
    logging.info("Searching for optimal XGBoost hyperparameters...")
    logging.info("This may take a minute...\n")
    
    # Fast debug mode flag
    FAST_MODE = True
    
    # Define XGBoost base model with overfitting prevention constraints
    xgb_base = XGBClassifier(
        random_state=42,
        n_jobs=-1,  # Use all CPUs
        tree_method='hist',  # Faster training
        device='cpu',
        objective='multi:softprob',  # Multi-class classification
        num_class=num_classes,  # Explicitly define total classes across folds
        eval_metric='mlogloss',
        use_label_encoder=False
    )
    
    # Parameter grid - optimized for speed and performance
    if FAST_MODE:
        param_grid = {
            'n_estimators': [100],                   # Single value for fast testing
            'max_depth': [4],                        # Single value for fast testing
            'learning_rate': [0.1]                   # Single value for fast testing
        }
        cv_folds = 2  # Reduced CV folds for speed
        logging.info("FAST_MODE enabled: Using minimal parameter grid and 2-fold CV\n")
    else:
        param_grid = {
            'n_estimators': [100, 200],              # Number of boosting rounds
            'max_depth': [4, 6],                     # Prevent overfitting
            'learning_rate': [0.05, 0.1]             # Learning rate (eta)
        }
        cv_folds = 3  # Full CV folds
        logging.info("FULL_MODE: Using complete parameter grid and 3-fold CV\n")
    
    cv = StratifiedKFold(
        n_splits=cv_folds,
        shuffle=True,
        random_state=42
    )

    # GridSearchCV with stratified folds to preserve class balance
    grid_search = GridSearchCV(
        estimator=xgb_base,
        param_grid=param_grid,
        cv=cv,
        scoring='accuracy',
        n_jobs=-1,  # Use all available cores
        verbose=1,  # Show progress
        error_score='raise'
    )
    
    # Fit grid search
    grid_search.fit(X_train, y_train)
    
    logging.info(f"\n✓ Hyperparameter tuning complete")
    logging.info(f"  - Best parameters: {grid_search.best_params_}")
    logging.info(f"  - Best CV accuracy: {grid_search.best_score_:.4f}")
    logging.info(f"  - Total combinations tested: {len(grid_search.cv_results_['params'])}\n")
    
    return grid_search.best_estimator_


# ============================================================================
# STEP 6: CROSS-VALIDATION EVALUATION
# ============================================================================

def evaluate_with_cross_validation(model, X_train, y_train):
    """
    Evaluate model using cross-validation to ensure robustness.
    Returns mean accuracy and standard deviation across folds.
    """
    logging.info("\n" + "=" * 70)
    logging.info("STEP 6: CROSS-VALIDATION EVALUATION")
    logging.info("=" * 70)
    
    # Stratified K-Fold ensures class balance in each fold
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Multiple scoring metrics
    scoring = {
        'accuracy': 'accuracy',
        'f1_weighted': 'f1_weighted',
        'f1_macro': 'f1_macro'
    }
    
    cv_results = cross_validate(
        model, 
        X_train, 
        y_train, 
        cv=skf,
        scoring=scoring,
        return_train_score=True
    )
    
    # Print results for each fold
    logging.info("Cross-validation results (5-fold Stratified):\n")
    for fold_idx in range(5):
        logging.info(f"Fold {fold_idx + 1}:")
        logging.info(f"  Train Accuracy: {cv_results['train_accuracy'][fold_idx]:.4f}")
        logging.info(f"  Test Accuracy:  {cv_results['test_accuracy'][fold_idx]:.4f}")
        logging.info(f"  Test F1 (weighted): {cv_results['test_f1_weighted'][fold_idx]:.4f}")
    
    # Summary statistics
    logging.info(f"\nSummary Statistics:")
    logging.info(f"  Mean CV Accuracy: {cv_results['test_accuracy'].mean():.4f} "
          f"(±{cv_results['test_accuracy'].std():.4f})")
    logging.info(f"  Mean CV F1 Score: {cv_results['test_f1_weighted'].mean():.4f} "
          f"(±{cv_results['test_f1_weighted'].std():.4f})\n")
    
    return cv_results


# ============================================================================
# STEP 7: PROBABILITY CALIBRATION
# ============================================================================

def calibrate_model(model, X_train, y_train):
    """
    Calibrate model probabilities using CalibratedClassifierCV.
    Ensures predicted probabilities are realistic (e.g., P(y=1) ≈ fraction of positives).
    
    This is critical for disease prediction where probability thresholds matter.
    """
    logging.info("\n" + "=" * 70)
    logging.info("STEP 7: PROBABILITY CALIBRATION")
    logging.info("=" * 70)
    
    # CalibratedClassifierCV wraps the model and calibrates probabilities
    calibrated_model = CalibratedClassifierCV(
        estimator=model,
        method='sigmoid',  # Sigmoid calibration - works well for most classifiers
        cv=5  # 5-fold cross-validation for calibration
    )
    
    calibrated_model.fit(X_train, y_train)
    
    logging.info(f"✓ Model calibrated successfully")
    logging.info(f"  - Calibration method: sigmoid")
    logging.info(f"  - CV folds for calibration: 5")
    logging.info(f"  - Ensures realistic probability estimates\n")
    
    return calibrated_model


# ============================================================================
# STEP 8: COMPREHENSIVE EVALUATION
# ============================================================================

def evaluate_model(model, X_train, y_train, X_test, y_test, le, model_name="Model"):
    """
    Comprehensive evaluation using multiple metrics.
    Returns detailed performance metrics.
    """
    logging.info("\n" + "=" * 70)
    logging.info(f"STEP 8: COMPREHENSIVE EVALUATION - {model_name}")
    logging.info("=" * 70)
    
    # Predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # Probabilities (for calibrated models)
    y_pred_proba_train = model.predict_proba(X_train)
    y_pred_proba_test = model.predict_proba(X_test)
    
    # Accuracy
    train_accuracy = accuracy_score(y_train, y_pred_train)
    test_accuracy = accuracy_score(y_test, y_pred_test)
    
    logging.info(f"\n>>> ACCURACY METRICS")
    logging.info(f"  Training Accuracy: {train_accuracy:.4f}")
    logging.info(f"  Test Accuracy:     {test_accuracy:.4f}")
    
    # Check for overfitting
    overfitting_gap = train_accuracy - test_accuracy
    if overfitting_gap > 0.1:
        logging.warning(f"  ⚠ Warning: Possible overfitting detected (gap: {overfitting_gap:.4f})")
    else:
        logging.info(f"  ✓ Generalization gap is acceptable ({overfitting_gap:.4f})")
    
    # F1 Score (handles class imbalance better than accuracy)
    f1_weighted = f1_score(y_test, y_pred_test, average='weighted')
    f1_macro = f1_score(y_test, y_pred_test, average='macro')
    
    logging.info(f"\n>>> F1 SCORES (Better for imbalanced data)")
    logging.info(f"  F1 Score (Weighted): {f1_weighted:.4f}")
    logging.info(f"  F1 Score (Macro):    {f1_macro:.4f}")
    
    # Classification Report (per-class metrics)
    logging.info(f"\n>>> CLASSIFICATION REPORT (Detailed per-class metrics)")
    report = classification_report(
        y_test, 
        y_pred_test,
        target_names=le.classes_,
        digits=4
    )
    logging.info(f"\n{report}")
    
    # Confusion Matrix
    logging.info(f"\n>>> CONFUSION MATRIX")
    cm = confusion_matrix(y_test, y_pred_test)
    logging.info(f"{cm}")
    
    # Plot confusion matrix
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=le.classes_,
        yticklabels=le.classes_,
        cbar_kws={'label': 'Count'}
    )
    plt.title(f'{model_name} - Confusion Matrix')
    plt.xlabel('Predicted Disease')
    plt.ylabel('Actual Disease')
    plt.tight_layout()
    plt.savefig('confusion_matrix_improved.png', dpi=300, bbox_inches='tight')
    logging.info(f"\n  ✓ Confusion matrix saved as 'confusion_matrix_improved.png'")
    plt.close()
    
    # ROC-AUC for multi-class (one-vs-rest)
    if len(le.classes_) > 2:
        try:
            roc_auc = roc_auc_score(y_test, y_pred_proba_test, multi_class='ovr', average='weighted')
            logging.info(f"\n>>> ROC-AUC SCORE (One-vs-Rest)")
            logging.info(f"  ROC-AUC (Weighted): {roc_auc:.4f}")
        except Exception as e:
            logging.warning(f"\n>>> ROC-AUC SCORE: Could not compute ({e})")
    
    # Probability calibration check
    logging.info(f"\n>>> PROBABILITY CALIBRATION CHECK")
    avg_proba = y_pred_proba_test.max(axis=1).mean()
    logging.info(f"  Average max predicted probability: {avg_proba:.4f}")
    logging.info(f"  (Closer to test accuracy {test_accuracy:.4f} = better calibration)")
    
    logging.info("")
    return {
        'train_accuracy': train_accuracy,
        'test_accuracy': test_accuracy,
        'f1_weighted': f1_weighted,
        'f1_macro': f1_macro,
        'confusion_matrix': cm
    }


# ============================================================================
# STEP 9: SAVE MODELS
# ============================================================================

def save_models(model, le, feature_names, output_dir='.'):
    """
    Save the trained model, label encoder, and feature names for inference.
    """
    logging.info("\n" + "=" * 70)
    logging.info("STEP 9: SAVING MODELS")
    logging.info("=" * 70)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Save calibrated model
    model_path = output_dir / 'model_improved.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    logging.info(f"✓ Model saved: {model_path}")
    
    # Save label encoder
    le_path = output_dir / 'le_classification_improved.pkl'
    with open(le_path, 'wb') as f:
        pickle.dump(le, f)
    logging.info(f"✓ Label encoder saved: {le_path}")
    
    # Save feature names (important for inference)
    features_path = output_dir / 'feature_names_improved.pkl'
    with open(features_path, 'wb') as f:
        pickle.dump(feature_names, f)
    logging.info(f"✓ Feature names saved: {features_path}")
    
    logging.info("")


# ============================================================================
# STEP 10: MAIN TRAINING PIPELINE
# ============================================================================

def main():
    """
    Main training pipeline orchestrating all steps.
    """
    logging.info("\n")
    logging.info("#" * 70)
    logging.info("# IMPROVED DISEASE PREDICTION MODEL TRAINING PIPELINE")
    logging.info("#" * 70)
    logging.info("# Using Best Practices:")
    logging.info("#  • XGBoost: More powerful gradient boosting")
    logging.info("#  • GridSearchCV: Systematic hyperparameter tuning")
    logging.info("#  • Cross-validation: Robust evaluation")
    logging.info("#  • SMOTE: Handle class imbalance")
    logging.info("#  • Feature Engineering: Create meaningful interactions")
    logging.info("#  • Calibration: Realistic probability estimates")
    logging.info("#  • Comprehensive Metrics: Not just accuracy")
    logging.info("#" * 70 + "\n")
    
    # Step 1: Load data
    X, y, feature_names = load_data()
    
    # Step 4: Preprocess
    X_train, X_test, y_train, y_test, le, num_classes = preprocess_data(X, y)
    
    # Step 2: Handle class imbalance
    X_train_balanced, y_train_balanced = handle_class_imbalance(X_train, y_train)
    
    # Step 5: Hyperparameter tuning
    logging.info("Training base XGBoost model for hyperparameter tuning...")
    best_model = tune_hyperparameters(X_train_balanced, y_train_balanced, num_classes)
    
    # Step 6: Cross-validation
    cv_results = evaluate_with_cross_validation(best_model, X_train_balanced, y_train_balanced)
    
    # Step 7: Calibrate model
    calibrated_model = calibrate_model(best_model, X_train_balanced, y_train_balanced)
    
    # Step 8: Evaluate on test set
    metrics = evaluate_model(
        calibrated_model, 
        X_train_balanced, 
        y_train_balanced, 
        X_test, 
        y_test, 
        le,
        model_name="Calibrated XGBoost Classifier"
    )
    
    # Step 9: Save models
    save_models(calibrated_model, le, feature_names)
    
    # Final summary
    logging.info("\n" + "=" * 70)
    logging.info("TRAINING PIPELINE COMPLETE!")
    logging.info("=" * 70)
    logging.info(f"\n✓ Model successfully trained and saved")
    logging.info(f"  - Test Accuracy: {metrics['test_accuracy']:.4f}")
    logging.info(f"  - F1 Score (Weighted): {metrics['f1_weighted']:.4f}")
    logging.info(f"  - Model type: Calibrated XGBoost")
    logging.info(f"\n✓ Files saved:")
    logging.info(f"  - model_improved.pkl (trained model)")
    logging.info(f"  - le_classification_improved.pkl (label encoder)")
    logging.info(f"  - feature_names_improved.pkl (feature names)")
    logging.info(f"  - confusion_matrix_improved.png (evaluation visualization)\n")


if __name__ == "__main__":
    main()
