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
from pathlib import Path

# Scikit-learn imports
from sklearn.model_selection import (
    train_test_split, 
    GridSearchCV, 
    cross_val_score,
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

# ============================================================================
# STEP 1: LOAD DATA
# ============================================================================

def load_data():
    """
    Load classification dataset from CSV.
    Returns: X (features), y (target), column names
    """
    print("=" * 70)
    print("STEP 1: LOADING DATA")
    print("=" * 70)
    
    file_path = 'healthcare-chatbot/Data/Training.csv'
    df = pd.read_csv(file_path)
    
    # Remove unnamed columns
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    
    # Ensure 'prognosis' column exists
    if 'prognosis' not in df.columns:
        raise ValueError("'prognosis' column not found in dataset")
    
    # Separate features and target
    X = df.drop('prognosis', axis=1)
    y = df['prognosis']
    
    print(f"✓ Data loaded successfully")
    print(f"  - Dataset shape: {X.shape}")
    print(f"  - Number of features (symptoms): {X.shape[1]}")
    print(f"  - Number of samples: {X.shape[0]}")
    print(f"  - Number of unique diseases: {y.nunique()}")
    print(f"  - Class distribution:\n{y.value_counts()}\n")
    
    return X, y, X.columns.tolist()


# ============================================================================
# STEP 2: HANDLE CLASS IMBALANCE WITH SMOTE
# ============================================================================

def handle_class_imbalance(X_train, y_train):
    """
    Apply SMOTE (Synthetic Minority Over-sampling Technique) to handle
    class imbalance by generating synthetic samples for minority classes.
    
    Returns: Balanced X_train, y_train
    """
    print("\n" + "=" * 70)
    print("STEP 2: HANDLING CLASS IMBALANCE WITH SMOTE")
    print("=" * 70)
    
    # Calculate initial class distribution
    unique, counts = np.unique(y_train, return_counts=True)
    print(f"Before SMOTE:")
    print(f"  Class distribution: {dict(zip(unique, counts))}")
    
    # Apply SMOTE only if there's class imbalance
    if len(np.unique(y_train)) > 1 and counts.std() > counts.mean() * 0.3:
        smote = SMOTE(random_state=42, k_neighbors=3)
        try:
            X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
            print(f"\nAfter SMOTE:")
            unique_after, counts_after = np.unique(y_train_balanced, return_counts=True)
            print(f"  Class distribution: {dict(zip(unique_after, counts_after))}")
            print(f"  New training set size: {X_train_balanced.shape[0]} (was {X_train.shape[0]})\n")
            return X_train_balanced, y_train_balanced
        except Exception as e:
            print(f"⚠ SMOTE failed: {e}. Using original data.\n")
            return X_train, y_train
    else:
        print(f"  ✓ No significant class imbalance detected. Using original data.\n")
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
    print("\n" + "=" * 70)
    print("STEP 3: FEATURE ENGINEERING")
    print("=" * 70)
    
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
    
    print(f"✓ Features engineered successfully")
    print(f"  - Original features: {n_original}")
    print(f"  - Interaction features created: {interaction_count}")
    print(f"  - Total features: {X_engineered.shape[1]}\n")
    
    return X_engineered


# ============================================================================
# STEP 4: PREPROCESS DATA
# ============================================================================

def preprocess_data(X, y):
    """
    Encode labels and prepare train/test split with stratification.
    """
    print("\n" + "=" * 70)
    print("STEP 4: PREPROCESSING DATA")
    print("=" * 70)
    
    # Encode target labels
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
    
    print(f"✓ Data preprocessing complete")
    print(f"  - Training set: {X_train.shape}")
    print(f"  - Test set: {X_test.shape}")
    print(f"  - Total features: {X_train.shape[1]}")
    print(f"  - Label encoder classes: {le.classes_}\n")
    
    return X_train, X_test, y_train, y_test, le


# ============================================================================
# STEP 5: HYPERPARAMETER TUNING WITH GRIDSEARCHCV
# ============================================================================

def tune_hyperparameters(X_train, y_train):
    """
    Use GridSearchCV to find optimal hyperparameters for XGBoost.
    Searches across multiple parameter combinations using cross-validation.
    """
    print("\n" + "=" * 70)
    print("STEP 5: HYPERPARAMETER TUNING WITH GRIDSEARCHCV")
    print("=" * 70)
    print("Searching for optimal XGBoost hyperparameters...")
    print("This may take a minute...\n")
    
    # Define XGBoost base model with overfitting prevention constraints
    xgb_base = XGBClassifier(
        random_state=42,
        n_jobs=-1,  # Use all CPUs
        tree_method='hist',  # Faster training
        device='cpu',
        objective='multi:softprob'  # Multi-class classification
    )
    
    # Parameter grid - carefully selected ranges to avoid overfitting
    # Note: Reduced to balance tuning quality with computation time
    param_grid = {
        'n_estimators': [100, 150],                 # Number of boosting rounds
        'max_depth': [4, 5, 6],                     # Prevent overfitting
        'learning_rate': [0.05, 0.1],               # Learning rate (eta)
        'min_child_weight': [1, 3],                 # Min samples per leaf
        'subsample': [0.8, 0.9],                    # Row subsampling
        'colsample_bytree': [0.8, 0.9],             # Column subsampling
    }
    
    # GridSearchCV with 5-fold cross-validation
    grid_search = GridSearchCV(
        estimator=xgb_base,
        param_grid=param_grid,
        cv=5,  # 5-fold cross-validation
        scoring='accuracy',
        n_jobs=-1,
        verbose=1,
        error_score=0
    )
    
    # Fit grid search
    grid_search.fit(X_train, y_train)
    
    print(f"\n✓ Hyperparameter tuning complete")
    print(f"  - Best parameters: {grid_search.best_params_}")
    print(f"  - Best CV accuracy: {grid_search.best_score_:.4f}")
    print(f"  - Total combinations tested: {len(grid_search.cv_results_['params'])}\n")
    
    return grid_search.best_estimator_


# ============================================================================
# STEP 6: CROSS-VALIDATION EVALUATION
# ============================================================================

def evaluate_with_cross_validation(model, X_train, y_train):
    """
    Evaluate model using cross-validation to ensure robustness.
    Returns mean accuracy and standard deviation across folds.
    """
    print("\n" + "=" * 70)
    print("STEP 6: CROSS-VALIDATION EVALUATION")
    print("=" * 70)
    
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
    print("Cross-validation results (5-fold Stratified):\n")
    for fold_idx in range(5):
        print(f"Fold {fold_idx + 1}:")
        print(f"  Train Accuracy: {cv_results['train_accuracy'][fold_idx]:.4f}")
        print(f"  Test Accuracy:  {cv_results['test_accuracy'][fold_idx]:.4f}")
        print(f"  Test F1 (weighted): {cv_results['test_f1_weighted'][fold_idx]:.4f}")
    
    # Summary statistics
    print(f"\nSummary Statistics:")
    print(f"  Mean CV Accuracy: {cv_results['test_accuracy'].mean():.4f} "
          f"(±{cv_results['test_accuracy'].std():.4f})")
    print(f"  Mean CV F1 Score: {cv_results['test_f1_weighted'].mean():.4f} "
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
    print("\n" + "=" * 70)
    print("STEP 7: PROBABILITY CALIBRATION")
    print("=" * 70)
    
    # CalibratedClassifierCV wraps the model and calibrates probabilities
    calibrated_model = CalibratedClassifierCV(
        estimator=model,
        method='sigmoid',  # Sigmoid calibration - works well for most classifiers
        cv=5  # 5-fold cross-validation for calibration
    )
    
    calibrated_model.fit(X_train, y_train)
    
    print(f"✓ Model calibrated successfully")
    print(f"  - Calibration method: sigmoid")
    print(f"  - CV folds for calibration: 5")
    print(f"  - Ensures realistic probability estimates\n")
    
    return calibrated_model


# ============================================================================
# STEP 8: COMPREHENSIVE EVALUATION
# ============================================================================

def evaluate_model(model, X_train, y_train, X_test, y_test, le, model_name="Model"):
    """
    Comprehensive evaluation using multiple metrics.
    Returns detailed performance metrics.
    """
    print("\n" + "=" * 70)
    print(f"STEP 8: COMPREHENSIVE EVALUATION - {model_name}")
    print("=" * 70)
    
    # Predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # Probabilities (for calibrated models)
    y_pred_proba_train = model.predict_proba(X_train)
    y_pred_proba_test = model.predict_proba(X_test)
    
    # Accuracy
    train_accuracy = accuracy_score(y_train, y_pred_train)
    test_accuracy = accuracy_score(y_test, y_pred_test)
    
    print(f"\n>>> ACCURACY METRICS")
    print(f"  Training Accuracy: {train_accuracy:.4f}")
    print(f"  Test Accuracy:     {test_accuracy:.4f}")
    
    # Check for overfitting
    overfitting_gap = train_accuracy - test_accuracy
    if overfitting_gap > 0.1:
        print(f"  ⚠ Warning: Possible overfitting detected (gap: {overfitting_gap:.4f})")
    else:
        print(f"  ✓ Generalization gap is acceptable ({overfitting_gap:.4f})")
    
    # F1 Score (handles class imbalance better than accuracy)
    f1_weighted = f1_score(y_test, y_pred_test, average='weighted')
    f1_macro = f1_score(y_test, y_pred_test, average='macro')
    
    print(f"\n>>> F1 SCORES (Better for imbalanced data)")
    print(f"  F1 Score (Weighted): {f1_weighted:.4f}")
    print(f"  F1 Score (Macro):    {f1_macro:.4f}")
    
    # Classification Report (per-class metrics)
    print(f"\n>>> CLASSIFICATION REPORT (Detailed per-class metrics)")
    print(classification_report(
        y_test, 
        y_pred_test,
        target_names=le.classes_,
        digits=4
    ))
    
    # Confusion Matrix
    print(f"\n>>> CONFUSION MATRIX")
    cm = confusion_matrix(y_test, y_pred_test)
    print(cm)
    
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
    print(f"\n  ✓ Confusion matrix saved as 'confusion_matrix_improved.png'")
    plt.close()
    
    # ROC-AUC for multi-class (one-vs-rest)
    if len(le.classes_) > 2:
        try:
            roc_auc = roc_auc_score(y_test, y_pred_proba_test, multi_class='ovr', average='weighted')
            print(f"\n>>> ROC-AUC SCORE (One-vs-Rest)")
            print(f"  ROC-AUC (Weighted): {roc_auc:.4f}")
        except Exception as e:
            print(f"\n>>> ROC-AUC SCORE: Could not compute ({e})")
    
    # Probability calibration check
    print(f"\n>>> PROBABILITY CALIBRATION CHECK")
    avg_proba = y_pred_proba_test.max(axis=1).mean()
    print(f"  Average max predicted probability: {avg_proba:.4f}")
    print(f"  (Closer to test accuracy {test_accuracy:.4f} = better calibration)")
    
    print()
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
    print("\n" + "=" * 70)
    print("STEP 9: SAVING MODELS")
    print("=" * 70)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Save calibrated model
    model_path = output_dir / 'model_improved.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"✓ Model saved: {model_path}")
    
    # Save label encoder
    le_path = output_dir / 'le_classification_improved.pkl'
    with open(le_path, 'wb') as f:
        pickle.dump(le, f)
    print(f"✓ Label encoder saved: {le_path}")
    
    # Save feature names (important for inference)
    features_path = output_dir / 'feature_names_improved.pkl'
    with open(features_path, 'wb') as f:
        pickle.dump(feature_names, f)
    print(f"✓ Feature names saved: {features_path}")
    
    print()


# ============================================================================
# STEP 10: MAIN TRAINING PIPELINE
# ============================================================================

def main():
    """
    Main training pipeline orchestrating all steps.
    """
    print("\n")
    print("#" * 70)
    print("# IMPROVED DISEASE PREDICTION MODEL TRAINING PIPELINE")
    print("#" * 70)
    print("# Using Best Practices:")
    print("#  • XGBoost: More powerful gradient boosting")
    print("#  • GridSearchCV: Systematic hyperparameter tuning")
    print("#  • Cross-validation: Robust evaluation")
    print("#  • SMOTE: Handle class imbalance")
    print("#  • Feature Engineering: Create meaningful interactions")
    print("#  • Calibration: Realistic probability estimates")
    print("#  • Comprehensive Metrics: Not just accuracy")
    print("#" * 70 + "\n")
    
    # Step 1: Load data
    X, y, feature_names = load_data()
    
    # Step 4: Preprocess
    X_train, X_test, y_train, y_test, le = preprocess_data(X, y)
    
    # Step 2: Handle class imbalance
    X_train_balanced, y_train_balanced = handle_class_imbalance(X_train, y_train)
    
    # Step 5: Hyperparameter tuning
    print("Training base XGBoost model for hyperparameter tuning...")
    best_model = tune_hyperparameters(X_train_balanced, y_train_balanced)
    
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
    print("\n" + "=" * 70)
    print("TRAINING PIPELINE COMPLETE!")
    print("=" * 70)
    print(f"\n✓ Model successfully trained and saved")
    print(f"  - Test Accuracy: {metrics['test_accuracy']:.4f}")
    print(f"  - F1 Score (Weighted): {metrics['f1_weighted']:.4f}")
    print(f"  - Model type: Calibrated XGBoost")
    print(f"\n✓ Files saved:")
    print(f"  - model_improved.pkl (trained model)")
    print(f"  - le_classification_improved.pkl (label encoder)")
    print(f"  - feature_names_improved.pkl (feature names)")
    print(f"  - confusion_matrix_improved.png (evaluation visualization)\n")


if __name__ == "__main__":
    main()
