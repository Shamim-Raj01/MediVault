"""
FAST IMPROVED MODEL TRAINING (Production-Ready)
================================================
Faster version of the improved pipeline for quicker training.
Still implements all best practices:
- XGBoost with optimized parameters
- Cross-validation for robustness
- SMOTE for class imbalance handling
- Feature engineering
- Probability calibration
- Comprehensive evaluation
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import warnings
from pathlib import Path

from sklearn.model_selection import (
    train_test_split, 
    cross_val_score,
    StratifiedKFold,
)
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, 
    classification_report, 
    confusion_matrix,
    f1_score
)
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

warnings.filterwarnings('ignore')

# ============================================================================
# STEP 1: LOAD DATA
# ============================================================================

def load_data():
    """Load classification dataset from CSV."""
    print("=" * 70)
    print("STEP 1: LOADING DATA")
    print("=" * 70)
    
    file_path = 'healthcare-chatbot/Data/Training.csv'
    df = pd.read_csv(file_path)
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    
    X = df.drop('prognosis', axis=1)
    y = df['prognosis']
    
    print(f"✓ Data loaded successfully")
    print(f"  - Dataset shape: {X.shape}")
    print(f"  - Number of features: {X.shape[1]}")
    print(f"  - Number of samples: {X.shape[0]}")
    print(f"  - Number of diseases: {y.nunique()}\n")
    
    return X, y, X.columns.tolist()


# ============================================================================
# STEP 2: FEATURE ENGINEERING
# ============================================================================

def engineer_features(X):
    """Create interaction features."""
    print("=" * 70)
    print("STEP 2: FEATURE ENGINEERING")
    print("=" * 70)
    
    X_engineered = X.copy()
    n_original = X.shape[1]
    
    # Create interaction features from top symptoms
    top_symptoms = X.var().nlargest(10).index.tolist()
    interaction_count = 0
    
    for i in range(len(top_symptoms)):
        for j in range(i + 1, min(i + 3, len(top_symptoms))):
            symptom1 = top_symptoms[i]
            symptom2 = top_symptoms[j]
            feature_name = f"{symptom1}_and_{symptom2}"
            X_engineered[feature_name] = (X[symptom1] * X[symptom2]).astype(int)
            interaction_count += 1
    
    print(f"✓ Features engineered")
    print(f"  - Original features: {n_original}")
    print(f"  - Interaction features: {interaction_count}")
    print(f"  - Total features: {X_engineered.shape[1]}\n")
    
    return X_engineered


# ============================================================================
# STEP 3: PREPROCESS DATA
# ============================================================================

def preprocess_data(X, y):
    """Encode labels and split data."""
    print("=" * 70)
    print("STEP 3: PREPROCESSING & STRATIFIED SPLIT")
    print("=" * 70)
    
    # Feature engineering
    X_engineered = engineer_features(X)
    
    # Encode target
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # Stratified split (preserves class distribution)
    X_train, X_test, y_train, y_test = train_test_split(
        X_engineered, 
        y_encoded,
        test_size=0.2,
        random_state=42,
        stratify=y_encoded
    )
    
    print(f"✓ Data preprocessing complete")
    print(f"  - Training set: {X_train.shape}")
    print(f"  - Test set: {X_test.shape}")
    print(f"  - Stratified split: YES\n")
    
    return X_train, X_test, y_train, y_test, le


# ============================================================================
# STEP 4: HANDLE CLASS IMBALANCE WITH SMOTE
# ============================================================================

def handle_class_imbalance(X_train, y_train):
    """Apply SMOTE to balance training data."""
    print("=" * 70)
    print("STEP 4: CLASS IMBALANCE HANDLING (SMOTE)")
    print("=" * 70)
    
    unique, counts = np.unique(y_train, return_counts=True)
    print(f"Before SMOTE: {dict(zip(unique, counts))}")
    
    if len(np.unique(y_train)) > 1 and counts.std() > counts.mean() * 0.3:
        smote = SMOTE(random_state=42, k_neighbors=3)
        try:
            X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
            unique_after, counts_after = np.unique(y_train_balanced, return_counts=True)
            print(f"After SMOTE: {dict(zip(unique_after, counts_after))}")
            print(f"  New training size: {X_train_balanced.shape[0]}\n")
            return X_train_balanced, y_train_balanced
        except Exception as e:
            print(f"⚠ SMOTE skipped: {e}\n")
            return X_train, y_train
    else:
        print(f"  ✓ Balanced data - using original\n")
        return X_train, y_train


# ============================================================================
# STEP 5: TRAIN OPTIMIZED XGBOOST MODEL
# ============================================================================

def train_model(X_train, y_train):
    """Train XGBoost with well-tuned hyperparameters."""
    print("=" * 70)
    print("STEP 5: TRAINING OPTIMIZED XGBOOST")
    print("=" * 70)
    
    # Pre-tuned optimal hyperparameters (from GridSearch experience)
    model = XGBClassifier(
        n_estimators=150,
        max_depth=5,
        learning_rate=0.1,
        min_child_weight=1,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
        n_jobs=-1,
        tree_method='hist',
        device='cpu',
        objective='multi:softprob'
    )
    
    model.fit(X_train, y_train)
    
    print(f"✓ XGBoost model trained")
    print(f"  - Estimators: 150")
    print(f"  - Max depth: 5")
    print(f"  - Learning rate: 0.1")
    print(f"  - Subsample: 0.9\n")
    
    return model


# ============================================================================
# STEP 6: CROSS-VALIDATION EVALUATION
# ============================================================================

def evaluate_with_cv(model, X_train, y_train):
    """5-fold cross-validation evaluation."""
    print("=" * 70)
    print("STEP 6: CROSS-VALIDATION (5-Fold Stratified)")
    print("=" * 70)
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Calculate CV scores
    cv_accuracy = cross_val_score(model, X_train, y_train, cv=skf, scoring='accuracy')
    cv_f1 = cross_val_score(model, X_train, y_train, cv=skf, scoring='f1_weighted')
    
    print(f"Fold-by-fold results:")
    for i, (acc, f1) in enumerate(zip(cv_accuracy, cv_f1), 1):
        print(f"  Fold {i}: Accuracy={acc:.4f}, F1={f1:.4f}")
    
    print(f"\nMean CV Accuracy: {cv_accuracy.mean():.4f} (±{cv_accuracy.std():.4f})")
    print(f"Mean CV F1 Score: {cv_f1.mean():.4f} (±{cv_f1.std():.4f})\n")
    
    return cv_accuracy, cv_f1


# ============================================================================
# STEP 7: CALIBRATE PROBABILITIES
# ============================================================================

def calibrate_model(model, X_train, y_train):
    """Calibrate model probabilities."""
    print("=" * 70)
    print("STEP 7: PROBABILITY CALIBRATION")
    print("=" * 70)
    
    calibrated_model = CalibratedClassifierCV(
        estimator=model,
        method='sigmoid',
        cv=5
    )
    
    calibrated_model.fit(X_train, y_train)
    
    print(f"✓ Model calibrated with sigmoid method")
    print(f"  - Ensures realistic probability estimates")
    print(f"  - Critical for clinical decision-making\n")
    
    return calibrated_model


# ============================================================================
# STEP 8: COMPREHENSIVE EVALUATION
# ============================================================================

def evaluate_model(model, X_train, y_train, X_test, y_test, le):
    """Comprehensive evaluation."""
    print("=" * 70)
    print("STEP 8: COMPREHENSIVE EVALUATION")
    print("=" * 70)
    
    # Predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # Accuracy
    train_acc = accuracy_score(y_train, y_pred_train)
    test_acc = accuracy_score(y_test, y_pred_test)
    
    print(f"\n>>> ACCURACY")
    print(f"  Training: {train_acc:.4f}")
    print(f"  Testing:  {test_acc:.4f}")
    
    gap = train_acc - test_acc
    if gap > 0.1:
        print(f"  ⚠ Overfitting detected (gap: {gap:.4f})")
    else:
        print(f"  ✓ Good generalization (gap: {gap:.4f})")
    
    # F1 Scores
    f1_weighted = f1_score(y_test, y_pred_test, average='weighted')
    f1_macro = f1_score(y_test, y_pred_test, average='macro')
    
    print(f"\n>>> F1 SCORES")
    print(f"  Weighted: {f1_weighted:.4f}")
    print(f"  Macro:    {f1_macro:.4f}")
    
    # Classification Report
    print(f"\n>>> CLASSIFICATION REPORT")
    print(classification_report(
        y_test, 
        y_pred_test,
        target_names=le.classes_,
        digits=4,
        zero_division=0
    ))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred_test)
    
    print(f"\n>>> CONFUSION MATRIX")
    print(f"  Shape: {cm.shape}")
    print(f"  Correct predictions: {np.trace(cm)}")
    
    # Visualize
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm, 
        annot=False, 
        fmt='d',
        cmap='Blues',
        cbar_kws={'label': 'Count'}
    )
    plt.title('Confusion Matrix - Improved XGBoost Model')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig('confusion_matrix_improved.png', dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved as 'confusion_matrix_improved.png'\n")
    plt.close()
    
    return {
        'train_accuracy': train_acc,
        'test_accuracy': test_acc,
        'f1_weighted': f1_weighted,
        'f1_macro': f1_macro,
        'cm': cm
    }


# ============================================================================
# STEP 9: SAVE MODELS
# ============================================================================

def save_models(model, le, feature_names):
    """Save model and preprocessors."""
    print("=" * 70)
    print("STEP 9: SAVING MODELS")
    print("=" * 70)
    
    # Save model
    with open('model_improved.pkl', 'wb') as f:
        pickle.dump(model, f)
    print(f"✓ Model saved: model_improved.pkl")
    
    # Save label encoder
    with open('le_classification_improved.pkl', 'wb') as f:
        pickle.dump(le, f)
    print(f"✓ Label encoder saved: le_classification_improved.pkl")
    
    # Save feature names
    with open('feature_names_improved.pkl', 'wb') as f:
        pickle.dump(feature_names, f)
    print(f"✓ Feature names saved: feature_names_improved.pkl\n")


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main training pipeline."""
    print("\n")
    print("#" * 70)
    print("# FAST IMPROVED MODEL TRAINING - Production Ready")
    print("#" * 70)
    print("# Best Practices Implemented:")
    print("#  ✓ XGBoost - Powerful gradient boosting")
    print("#  ✓ Stratified K-Fold - Better evaluation")
    print("#  ✓ Feature Engineering - Interaction features")
    print("#  ✓ SMOTE - Class imbalance handling")
    print("#  ✓ Calibration - Realistic probabilities")
    print("#  ✓ Comprehensive Metrics - Not just accuracy")
    print("#  ✓ Cross-validation - Robust assessment")
    print("#" * 70 + "\n")
    
    # 1. Load data
    X, y, feature_names = load_data()
    
    # 3. Preprocess
    X_train, X_test, y_train, y_test, le = preprocess_data(X, y)
    
    # 4. Handle imbalance
    X_train_balanced, y_train_balanced = handle_class_imbalance(X_train, y_train)
    
    # 5. Train model
    model = train_model(X_train_balanced, y_train_balanced)
    
    # 6. Cross-validation
    cv_acc, cv_f1 = evaluate_with_cv(model, X_train_balanced, y_train_balanced)
    
    # 7. Calibrate
    calibrated_model = calibrate_model(model, X_train_balanced, y_train_balanced)
    
    # 8. Evaluate
    metrics = evaluate_model(
        calibrated_model, 
        X_train_balanced, 
        y_train_balanced, 
        X_test, 
        y_test, 
        le
    )
    
    # 9. Save
    save_models(calibrated_model, le, feature_names)
    
    # Summary
    print("=" * 70)
    print("TRAINING COMPLETE!")
    print("=" * 70)
    print(f"\n✓ Model Performance:")
    print(f"  - Test Accuracy:    {metrics['test_accuracy']:.4f}")
    print(f"  - F1 Score:         {metrics['f1_weighted']:.4f}")
    print(f"  - CV Accuracy:      {cv_acc.mean():.4f} ± {cv_acc.std():.4f}")
    print(f"\n✓ Model Type: Calibrated XGBoost Classifier")
    print(f"✓ Feature Count: {len(feature_names)}")
    print(f"✓ Diseases: {le.classes_.shape[0]}\n")


if __name__ == "__main__":
    main()
