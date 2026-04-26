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
import matplotlib
matplotlib.use("Agg")
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
    cross_validate,
    ParameterGrid
)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score, 
    classification_report, 
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    auc,
    f1_score,
    top_k_accuracy_score
)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.utils.class_weight import compute_class_weight

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

EXCLUDED_DATASETS = {"healthcare_dataset.csv"}

SYMPTOM_MAP = {
    "fever": "fever",
    "high_fever": "fever",
    "mild_fever": "fever",
    "skin_rash": "rash",
    "abnormal_appearing_skin": "rash",
    "skin_lesion": "rash",
    "shortness_of_breath": "breathlessness",
    "difficulty_breathing": "breathlessness",
    "difficulty_in_breathing": "breathlessness",
    "nasal_congestion": "congestion",
    "sinus_congestion": "congestion",
    "coryza": "runny_nose",
    "diarrhea": "diarrhoea",
    "painful_urination": "burning_micturition",
    "frequent_urination": "polyuria",
    "yellow_skin": "yellowish_skin",
    "yellowing_skin": "yellowish_skin",
    "jaundice": "yellowish_skin",
    "feeling_ill": "malaise",
    "sharp_abdominal_pain": "abdominal_pain",
    "upper_abdominal_pain": "abdominal_pain",
    "lower_abdominal_pain": "abdominal_pain",
    "belly_pain": "abdominal_pain",
    "stomach_pain": "abdominal_pain",
    "throat_irritation": "sore_throat",
    "patches_in_throat": "sore_throat",
    "watering_from_eyes": "lacrimation",
}

GROUP_MAP = {
    "bronchial_asthma": "asthma",
    "acute_bronchospasm": "asthma",
    "common_cold": "respiratory infection",
    "influenza": "respiratory infection",
    "acute_bronchitis": "respiratory infection",
    "acute_bronchiolitis": "respiratory infection",
    "bronchitis": "respiratory infection",
    "allergy": "allergic disorder",
    "allergic_rhinitis": "allergic disorder",
    "drug_reaction": "allergic disorder",
    "gerd": "acid peptic disease",
    "peptic_ulcer_diseae": "acid peptic disease",
    "peptic_ulcer_disease": "acid peptic disease",
    "urinary_tract_infection": "urinary tract infection",
    "urinary_tract_infection_uti": "urinary tract infection",
    "dimorphic_hemmorhoids_piles": "hemorrhoids",
    "dimorphic_hemorrhoids_piles": "hemorrhoids",
    "osteoarthritis": "arthritis",
    "arthritis_of_the_hip": "arthritis",
    "migraine": "headache disorder",
    "hepatitis_a": "hepatitis",
    "hepatitis_b": "hepatitis",
    "hepatitis_c": "hepatitis",
    "hepatitis_d": "hepatitis",
    "hepatitis_e": "hepatitis",
    "alcoholic_hepatitis": "hepatitis",
    "myocardial_infarction_heart_attack": "heart disease",
    "heart_attack": "heart disease",
    "angina": "heart disease",
    "panic_disorder": "anxiety disorder",
    "anxiety_disorders": "anxiety disorder",
    "depression": "mood disorder",
    "bipolar_disorder": "mood disorder",
    "chicken_pox": "chickenpox",
}

MIN_CLASS_SUPPORT = 20
WEAK_CLASS_F1_THRESHOLD = 0.25
PRIMARY_TOP_K = 3
SECONDARY_TOP_K = 5
INTERACTION_TOP_FEATURES = 50

logging.info(f"Working directory: {os.getcwd()}")
logging.info(f"Data path: {os.path.abspath(DATA_TRAIN_PATH)}")

# ============================================================================
# STEP 1: LOAD DATA
# ============================================================================

def normalize_token(text):
    """Normalize free-text labels into stable snake_case tokens."""
    text = str(text).strip().lower()
    text = text.replace("&", " and ")
    text = text.replace("/", " ")
    text = text.replace("-", " ")
    text = text.replace(",", " ")
    text = text.replace("(", " ").replace(")", " ")
    text = "_".join(text.split())
    return text


def normalize_symptom_name(name):
    normalized = normalize_token(name)
    return SYMPTOM_MAP.get(normalized, normalized)


def normalize_disease_label(label):
    normalized = normalize_token(label)
    grouped = GROUP_MAP.get(normalized, normalized)
    return grouped.replace("_", " ")


def is_binary_series(series):
    values = pd.to_numeric(pd.Series(series), errors='coerce').dropna()
    if values.empty:
        return False
    return set(values.unique()).issubset({0, 1})


def standardize_feature_frame(df):
    """
    Keep only usable symptom columns and normalize synonyms into a shared schema.
    """
    object_map = {
        'yes': 1, 'no': 0,
        'positive': 1, 'negative': 0,
        'present': 1, 'absent': 0
    }
    non_feature_tokens = (
        'patient', 'hospital', 'doctor', 'name', 'date', 'admission', 'discharge',
        'insurance', 'billing', 'room', 'outcome', 'age', 'gender', 'blood_type',
        'blood_pressure', 'cholesterol', 'medication', 'test_results'
    )
    cleaned_features = {}

    for col in df.columns:
        if col == 'disease':
            continue
        if col.startswith('unnamed'):
            continue
        if any(token in col for token in non_feature_tokens):
            continue

        series = df[col]
        if series.dtype == object:
            lowered = series.astype(str).str.strip().str.lower()
            unique_values = set(lowered.replace({'nan': np.nan, 'none': np.nan}).dropna().unique())
            if unique_values and unique_values.issubset(set(object_map)):
                numeric = lowered.map(object_map).fillna(0)
            else:
                continue
        else:
            numeric = pd.to_numeric(series, errors='coerce').fillna(0)

        if not is_binary_series(numeric):
            continue

        normalized_col = normalize_symptom_name(col)
        numeric = (pd.to_numeric(numeric, errors='coerce').fillna(0) > 0).astype(int)
        if normalized_col in cleaned_features:
            cleaned_features[normalized_col] = np.maximum(cleaned_features[normalized_col], numeric)
        else:
            cleaned_features[normalized_col] = numeric

    if not cleaned_features:
        return pd.DataFrame(index=df.index)

    return pd.DataFrame(cleaned_features, index=df.index).fillna(0).astype(int)


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


def load_and_clean_datasets():
    """
    Load and clean all training datasets from DATA_TRAIN_PATH.
    Excludes incompatible datasets, normalizes symptom synonyms, and groups labels.
    """
    logging.info("=" * 70)
    logging.info("LOADING AND CLEANING DATASETS")
    logging.info("=" * 70)

    datasets = []
    loaded_files = 0
    all_columns = set()

    if not os.path.exists(DATA_TRAIN_PATH):
        logging.error(f"Data train path does not exist: {DATA_TRAIN_PATH}")
        raise FileNotFoundError(f"Data train path not found: {DATA_TRAIN_PATH}")

    for filename in sorted(os.listdir(DATA_TRAIN_PATH)):
        if not filename.endswith('.csv'):
            continue
        if filename in EXCLUDED_DATASETS:
            logging.info(f"Skipping excluded dataset: {filename}")
            continue

        file_path = os.path.join(DATA_TRAIN_PATH, filename)
        try:
            df = pd.read_csv(file_path)
            df = df.loc[:, ~df.columns.astype(str).str.contains('^Unnamed')]
            df = df.rename(columns={col: normalize_token(col) for col in df.columns})

            if 'prognosis' in df.columns:
                df.rename(columns={'prognosis': 'disease'}, inplace=True)
            elif 'diseases' in df.columns:
                df.rename(columns={'diseases': 'disease'}, inplace=True)
            elif 'medical_condition' in df.columns:
                df.rename(columns={'medical_condition': 'disease'}, inplace=True)
            elif 'disease' not in df.columns:
                logging.warning(f"No target column found in {filename}, skipping")
                continue

            df['disease'] = df['disease'].astype(str).map(normalize_disease_label)
            feature_df = standardize_feature_frame(df)
            if feature_df.shape[1] == 0:
                logging.warning(f"Skipping {filename}: zero usable symptom features after cleaning")
                continue

            cleaned_df = feature_df.copy()
            cleaned_df['disease'] = df['disease'].values
            all_columns.update(feature_df.columns.tolist())
            datasets.append(cleaned_df)
            loaded_files += 1
            logging.info(
                f"Loaded {filename}: samples={cleaned_df.shape[0]}, usable_features={feature_df.shape[1]}"
            )
        except Exception as e:
            logging.warning(f"Error loading {filename}: {e}")

    if not datasets:
        raise ValueError("No valid datasets loaded. Check column names and DATA_TRAIN_PATH.")

    aligned_datasets = []
    for df in datasets:
        aligned_df = df.copy()
        for col in all_columns:
            if col not in aligned_df.columns:
                aligned_df[col] = 0
        aligned_datasets.append(aligned_df)

    combined_df = pd.concat(aligned_datasets, ignore_index=True)
    combined_df = combined_df.loc[:, ~combined_df.columns.duplicated()]

    X = combined_df.drop('disease', axis=1)
    y = combined_df['disease']
    X = X.apply(pd.to_numeric, errors='coerce').fillna(0).astype(int)

    feature_names = sorted(X.columns.tolist())
    X = X[feature_names]

    logging.info("Combined dataset loaded successfully")
    logging.info(f"  - Total files loaded: {loaded_files}")
    logging.info(f"  - Combined dataset shape: {X.shape}")
    logging.info(f"  - Number of features (symptoms): {X.shape[1]}")
    logging.info(f"  - Number of samples: {X.shape[0]}")
    logging.info(f"  - Number of unique diseases: {y.nunique()}")
    logging.info(f"  - Class distribution:\n{y.value_counts()}\n")

    return X, y, feature_names


def select_features(X_train, X_test, y_train, max_features=120):
    """
    Keep the most informative engineered features to reduce sparsity.
    """
    logging.info("\n" + "=" * 70)
    logging.info("STEP 3B: FEATURE SELECTION")
    logging.info("=" * 70)

    non_zero_columns = X_train.columns[(X_train.sum(axis=0) > 0)]
    X_train_filtered = X_train[non_zero_columns].copy()
    X_test_filtered = X_test.reindex(columns=non_zero_columns, fill_value=0).copy()

    if X_train_filtered.shape[1] <= max_features:
        logging.info(f"Feature count already compact: {X_train_filtered.shape[1]} features kept\n")
        return X_train_filtered, X_test_filtered, X_train_filtered.columns.tolist()

    from sklearn.feature_selection import SelectKBest, chi2

    selector = SelectKBest(score_func=chi2, k=min(max_features, X_train_filtered.shape[1]))
    selector.fit(X_train_filtered, y_train)
    selected_scores = pd.Series(selector.scores_, index=X_train_filtered.columns).fillna(0.0)
    selected_columns = selected_scores.nlargest(
        min(max_features, X_train_filtered.shape[1])
    ).index.tolist()

    logging.info(f"Selected {len(selected_columns)} of {X_train.shape[1]} engineered features\n")
    return (
        X_train_filtered[selected_columns].copy(),
        X_test_filtered[selected_columns].copy(),
        selected_columns
    )


def create_pairwise_interactions(X_train, X_test, top_features, max_pairs=None):
    """
    Create pairwise interaction features for the provided top feature list.
    """
    logging.info("\n" + "=" * 70)
    logging.info("STEP 3C: FEATURE INTERACTIONS")
    logging.info("=" * 70)

    X_train_interactions = X_train.copy()
    X_test_interactions = X_test.copy()
    interaction_count = 0

    for i, feature_a in enumerate(top_features):
        for j in range(i + 1, len(top_features)):
            feature_b = top_features[j]
            interaction_name = f"{feature_a}_x_{feature_b}"
            X_train_interactions[interaction_name] = (
                X_train[feature_a].to_numpy() * X_train[feature_b].to_numpy()
            ).astype(np.int8)
            X_test_interactions[interaction_name] = (
                X_test[feature_a].to_numpy() * X_test[feature_b].to_numpy()
            ).astype(np.int8)
            interaction_count += 1
            if max_pairs is not None and interaction_count >= max_pairs:
                break
        if max_pairs is not None and interaction_count >= max_pairs:
            break

    logging.info(f"Top features used for interactions: {len(top_features)}")
    logging.info(f"Interaction features created: {interaction_count}")
    logging.info(f"Total features after interactions: {X_train_interactions.shape[1]}\n")
    return X_train_interactions, X_test_interactions


def compute_sample_weights(y, reference_y=None):
    """
    Compute balanced sample weights from class weights.

    When reference_y is provided, class weights are estimated from that label
    distribution and then mapped onto y. This keeps the original imbalance
    signal even after optional resampling.
    """
    if reference_y is None:
        reference_y = y

    classes = np.unique(reference_y)
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=reference_y
    )
    weight_map = dict(zip(classes, class_weights))
    sample_weight = np.asarray([weight_map[label] for label in y], dtype=np.float32)

    logging.info("Computed class weights for training")
    logging.info(f"  - Classes weighted: {len(weight_map)}")
    if reference_y is not y:
        logging.info("  - Weight source: original pre-resampling class distribution")
    logging.info(
        "  - Weight range: "
        f"{sample_weight.min():.4f} to {sample_weight.max():.4f}\n"
    )
    return sample_weight, weight_map


def top_k_loss(y_true, y_pred_proba, k=PRIMARY_TOP_K, labels=None):
    """
    Loss counterpart of top-k accuracy for model selection.
    """
    if labels is None:
        labels = np.unique(y_true)
    top_k_acc = top_k_accuracy_score(y_true, y_pred_proba, k=min(k, len(labels)), labels=labels)
    return 1.0 - top_k_acc


def get_feature_importance_estimator(model):
    """
    Return a fitted estimator that exposes feature_importances_ when available.
    """
    if hasattr(model, "feature_importances_"):
        return model

    if hasattr(model, "estimator") and hasattr(model.estimator, "feature_importances_"):
        return model.estimator

    calibrated_models = getattr(model, "calibrated_classifiers_", None)
    if calibrated_models:
        first_calibrator = calibrated_models[0]
        estimator = getattr(first_calibrator, "estimator", None)
        if estimator is not None and hasattr(estimator, "feature_importances_"):
            return estimator

    return None


def preprocess_data(X, y, min_class_samples=MIN_CLASS_SUPPORT):
    """
    Filter rare classes, encode grouped labels, engineer features, and select features.
    """
    logging.info("\n" + "=" * 70)
    logging.info("STEP 4: PREPROCESSING DATA")
    logging.info("=" * 70)

    class_counts = y.value_counts()
    valid_classes = class_counts[class_counts >= min_class_samples].index
    mask = y.isin(valid_classes)
    X = X[mask].reset_index(drop=True)
    y = y[mask].reset_index(drop=True)
    logging.info(
        f"  - Removed {len(class_counts) - len(valid_classes)} rare classes "
        f"(with <{min_class_samples} samples)"
    )
    X_train, X_test, y_train_raw, y_test_raw = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    train_class_counts = y_train_raw.value_counts()
    supported_classes = train_class_counts[train_class_counts >= 2].index
    X_train = X_train[y_train_raw.isin(supported_classes)].reset_index(drop=True)
    y_train_raw = y_train_raw[y_train_raw.isin(supported_classes)].reset_index(drop=True)
    X_test = X_test[y_test_raw.isin(supported_classes)].reset_index(drop=True)
    y_test_raw = y_test_raw[y_test_raw.isin(supported_classes)].reset_index(drop=True)
    logging.info(f"  - Retained {len(supported_classes)} classes with at least 2 train samples")

    if X_train.shape[0] > 25000:
        base_parts = []
        remaining_indices = []
        for disease, idx in y_train_raw.groupby(y_train_raw).groups.items():
            idx = np.array(list(idx))
            keep = min(len(idx), 8)
            base_parts.extend(idx[:keep])
            if len(idx) > keep:
                remaining_indices.extend(idx[keep:])

        base_parts = np.array(sorted(base_parts))
        remaining_budget = max(0, 25000 - len(base_parts))
        if remaining_budget > 0 and remaining_indices:
            X_remaining = X_train.iloc[remaining_indices]
            y_remaining = y_train_raw.iloc[remaining_indices]
            if len(X_remaining) > remaining_budget:
                X_extra, _, y_extra, _ = train_test_split(
                    X_remaining,
                    y_remaining,
                    train_size=remaining_budget,
                    random_state=42,
                    stratify=y_remaining
                )
                X_train = pd.concat([X_train.iloc[base_parts], X_extra], ignore_index=True)
                y_train_raw = pd.concat([y_train_raw.iloc[base_parts], y_extra], ignore_index=True)
            else:
                X_train = pd.concat([X_train.iloc[base_parts], X_remaining], ignore_index=True)
                y_train_raw = pd.concat([y_train_raw.iloc[base_parts], y_remaining], ignore_index=True)
        else:
            X_train = X_train.iloc[base_parts].reset_index(drop=True)
            y_train_raw = y_train_raw.iloc[base_parts].reset_index(drop=True)

        if len(X_train) > 25000:
            X_train, _, y_train_raw, _ = train_test_split(
                X_train,
                y_train_raw,
                train_size=25000,
                random_state=42,
                stratify=y_train_raw
            )
        logging.info("  - Downsampled training split to 25,000 rows for tractable retraining")

    le = LabelEncoder()
    y_train = le.fit_transform(y_train_raw)
    y_test = le.transform(y_test_raw)

    X_train_base, X_test_base, base_feature_names = select_features(
        X_train,
        X_test,
        y_train,
        max_features=max(120, INTERACTION_TOP_FEATURES)
    )
    interaction_features = base_feature_names[:min(INTERACTION_TOP_FEATURES, len(base_feature_names))]
    X_train_enhanced, X_test_enhanced = create_pairwise_interactions(
        X_train_base,
        X_test_base,
        interaction_features
    )
    X_train, X_test, selected_feature_names = select_features(
        X_train_enhanced,
        X_test_enhanced,
        y_train,
        max_features=min(400, X_train_enhanced.shape[1])
    )

    logging.info("Data preprocessing complete")
    logging.info(f"  - Training set: {X_train.shape}")
    logging.info(f"  - Test set: {X_test.shape}")
    logging.info(f"  - Total features: {X_train.shape[1]}")
    logging.info(f"  - Pairwise interaction source features: {len(interaction_features)}")
    logging.info(f"  - Label encoder classes: {le.classes_}\n")

    num_classes = len(le.classes_)
    return X_train, X_test, y_train, y_test, le, num_classes, selected_feature_names


def evaluate_model(model, X_train, y_train, X_test, y_test, le, model_name="Model"):
    """
    Comprehensive evaluation with Accuracy, F1, and Top-3 Accuracy.
    """
    logging.info("\n" + "=" * 70)
    logging.info(f"STEP 8: COMPREHENSIVE EVALUATION - {model_name}")
    logging.info("=" * 70)

    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    y_pred_proba_test = model.predict_proba(X_test)

    train_accuracy = accuracy_score(y_train, y_pred_train)
    test_accuracy = accuracy_score(y_test, y_pred_test)
    top_3_accuracy = top_k_accuracy_score(
        y_test, y_pred_proba_test, k=PRIMARY_TOP_K, labels=np.arange(len(le.classes_))
    )
    top_5_accuracy = top_k_accuracy_score(
        y_test, y_pred_proba_test, k=min(SECONDARY_TOP_K, len(le.classes_)), labels=np.arange(len(le.classes_))
    )
    f1_weighted = f1_score(y_test, y_pred_test, average='weighted')
    f1_macro = f1_score(y_test, y_pred_test, average='macro')

    logging.info("\n>>> ACCURACY METRICS")
    logging.info(f"  Training Accuracy: {train_accuracy:.4f}")
    logging.info(f"  Test Accuracy:     {test_accuracy:.4f}")

    logging.info("\n>>> F1 SCORES")
    logging.info(f"  F1 Score (Weighted): {f1_weighted:.4f}")
    logging.info(f"  F1 Score (Macro):    {f1_macro:.4f}")

    logging.info("\n>>> TOP-K ACCURACY")
    logging.info(f"  Top-{PRIMARY_TOP_K} Accuracy:      {top_3_accuracy:.4f}")
    logging.info(f"  Top-{SECONDARY_TOP_K} Accuracy:      {top_5_accuracy:.4f}")

    report = classification_report(
        y_test,
        y_pred_test,
        target_names=le.classes_,
        digits=4,
        zero_division=0
    )
    report_dict = classification_report(
        y_test,
        y_pred_test,
        target_names=le.classes_,
        output_dict=True,
        zero_division=0
    )
    logging.info(f"\n>>> CLASSIFICATION REPORT\n{report}")

    cm = confusion_matrix(y_test, y_pred_test)
    if len(le.classes_) <= 50:
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
        plt.close()
    else:
        logging.info("Skipping confusion matrix plot because the label space is too large for a readable heatmap.")

    return {
        'train_accuracy': train_accuracy,
        'test_accuracy': test_accuracy,
        'f1_weighted': f1_weighted,
        'f1_macro': f1_macro,
        'top_3_accuracy': top_3_accuracy,
        'top_5_accuracy': top_5_accuracy,
        'confusion_matrix': cm,
        'report_dict': report_dict,
        'y_pred_proba_test': y_pred_proba_test
    }


def identify_weak_classes(report_dict, class_counts, min_support=MIN_CLASS_SUPPORT, min_f1=WEAK_CLASS_F1_THRESHOLD):
    """
    Identify classes to drop based on support or holdout F1.
    """
    weak_classes = []
    for disease, support in class_counts.items():
        if support < min_support:
            weak_classes.append(disease)
            continue

        metrics = report_dict.get(disease)
        if metrics is None:
            weak_classes.append(disease)
            continue

        if metrics.get('f1-score', 0.0) < min_f1:
            weak_classes.append(disease)

    return sorted(set(weak_classes))


def train_and_evaluate_cycle(X, y, cycle_name):
    """
    Run one full train/evaluate cycle on the provided label space.
    """
    logging.info("\n" + "=" * 70)
    logging.info(f"{cycle_name}")
    logging.info("=" * 70)

    X_train, X_test, y_train, y_test, le, num_classes, selected_feature_names = preprocess_data(
        X, y, min_class_samples=MIN_CLASS_SUPPORT
    )
    original_y_train = y_train.copy()
    X_train_balanced, y_train_balanced = handle_class_imbalance(X_train, y_train)
    sample_weight, _ = compute_sample_weights(y_train_balanced, reference_y=original_y_train)

    logging.info("Training base XGBoost model...")
    best_model = tune_hyperparameters(X_train_balanced, y_train_balanced, num_classes, sample_weight)
    calibrated_model = calibrate_model(best_model, X_train_balanced, y_train_balanced, sample_weight)
    metrics = evaluate_model(
        calibrated_model,
        X_train_balanced,
        y_train_balanced,
        X_test,
        y_test,
        le,
        model_name=cycle_name
    )

    return calibrated_model, le, selected_feature_names, metrics, X_train_balanced, y_train_balanced, X_test, y_test


def retrain_with_top_features(X_train, y_train, X_test, y_test, le, base_model, feature_names, top_n=30):
    """
    Retrain on the most important features from the fitted model and compare holdout accuracy.
    """
    logging.info("\n" + "=" * 70)
    logging.info("FEATURE IMPORTANCE RE-TRAINING")
    logging.info("=" * 70)

    importance_estimator = get_feature_importance_estimator(base_model)
    if importance_estimator is None:
        logging.info("Model does not expose feature importances; skipping top-feature retraining.\n")
        return base_model, feature_names, None

    importances = pd.Series(importance_estimator.feature_importances_, index=feature_names)
    top_features = importances.sort_values(ascending=False).head(min(top_n, len(importances))).index.tolist()
    logging.info(f"Top {len(top_features)} features selected for retraining")
    logging.info(f"Top features: {top_features}\n")

    X_train_top = X_train[top_features].copy()
    X_test_top = X_test[top_features].copy()
    sample_weight, _ = compute_sample_weights(y_train)

    retrained_model = tune_hyperparameters(X_train_top, y_train, len(le.classes_), sample_weight)
    retrained_model = calibrate_model(retrained_model, X_train_top, y_train, sample_weight)
    retrained_metrics = evaluate_model(
        retrained_model,
        X_train_top,
        y_train,
        X_test_top,
        y_test,
        le,
        model_name=f"TOP-{len(top_features)} FEATURE XGBOOST"
    )

    return retrained_model, top_features, retrained_metrics


def main():
    """
    Main training pipeline orchestrating the normalized sparse-data training flow.
    """
    logging.info("\n")
    logging.info("#" * 70)
    logging.info("# IMPROVED DISEASE PREDICTION MODEL TRAINING PIPELINE")
    logging.info("#" * 70)

    X, y, _ = load_data()
    initial_class_counts = y.value_counts()
    initial_model, initial_le, initial_feature_names, initial_metrics, initial_X_train, initial_y_train, initial_X_test, initial_y_test = train_and_evaluate_cycle(
        X, y, "INITIAL REDUCED-SPACE XGBOOST"
    )

    weak_classes = identify_weak_classes(
        initial_metrics['report_dict'],
        initial_class_counts,
        min_support=MIN_CLASS_SUPPORT,
        min_f1=WEAK_CLASS_F1_THRESHOLD
    )
    logging.info(
        f"Weak classes flagged for removal: {len(weak_classes)} "
        f"(support < {MIN_CLASS_SUPPORT} or F1 < {WEAK_CLASS_F1_THRESHOLD:.2f})"
    )

    if weak_classes:
        filtered_mask = ~y.isin(weak_classes)
        X_final = X.loc[filtered_mask].reset_index(drop=True)
        y_final = y.loc[filtered_mask].reset_index(drop=True)
        logging.info(
            f"Remaining label space after pruning: {y_final.nunique()} classes / {len(y_final)} samples"
        )
        final_model, final_le, selected_feature_names, metrics, X_train_final, y_train_final, X_test_final, y_test_final = train_and_evaluate_cycle(
            X_final, y_final, "FINAL PRUNED-SPACE XGBOOST"
        )
    else:
        final_model = initial_model
        final_le = initial_le
        selected_feature_names = initial_feature_names
        metrics = initial_metrics
        X_train_final, y_train_final, X_test_final, y_test_final = (
            initial_X_train, initial_y_train, initial_X_test, initial_y_test
        )

    importance_model, top_feature_names, importance_metrics = retrain_with_top_features(
        X_train_final,
        y_train_final,
        X_test_final,
        y_test_final,
        final_le,
        final_model,
        selected_feature_names,
        top_n=INTERACTION_TOP_FEATURES
    )

    logging.info("\n" + "=" * 70)
    logging.info("FEATURE COMPARISON")
    logging.info("=" * 70)
    logging.info(f"  - Baseline Top-3 Accuracy: {metrics['top_3_accuracy']:.4f}")
    if importance_metrics is not None:
        logging.info(
            f"  - Top-{INTERACTION_TOP_FEATURES} Feature Top-3 Accuracy: "
            f"{importance_metrics['top_3_accuracy']:.4f}"
        )
        if importance_metrics['top_3_accuracy'] >= metrics['top_3_accuracy']:
            final_model = importance_model
            selected_feature_names = top_feature_names
            metrics = importance_metrics
            logging.info(f"  - Using Top-{INTERACTION_TOP_FEATURES} feature retrained model as final artifact")
        else:
            logging.info("  - Keeping baseline model because it performed better on Top-3 accuracy")

    save_models(final_model, final_le, selected_feature_names)

    logging.info("\n" + "=" * 70)
    logging.info("TRAINING PIPELINE COMPLETE!")
    logging.info("=" * 70)
    logging.info(f"  - Accuracy: {metrics['test_accuracy']:.4f}")
    logging.info(f"  - F1 Score: {metrics['f1_weighted']:.4f}")
    logging.info(f"  - Top-3 Accuracy: {metrics['top_3_accuracy']:.4f}")
    logging.info(f"  - Top-5 Accuracy: {metrics['top_5_accuracy']:.4f}\n")


def tune_hyperparameters(X_train, y_train, num_classes, sample_weight):
    """
    Select an XGBoost configuration using Top-3 loss on a stratified validation split.
    """
    logging.info("\n" + "=" * 70)
    logging.info("STEP 5: XGBOOST TRAINING")
    logging.info("=" * 70)
    logging.info("Training XGBoost with Top-3 model selection and class-weighted fitting...\n")

    if len(X_train) > 12000:
        X_fit, _, y_fit, _, sample_weight_fit, _ = train_test_split(
            X_train,
            y_train,
            sample_weight,
            train_size=12000,
            random_state=42,
            stratify=y_train
        )
    else:
        X_fit, y_fit, sample_weight_fit = X_train, y_train, sample_weight

    if len(X_fit) >= max(200, num_classes * 2):
        X_model_train, X_val, y_model_train, y_val, weight_model_train, _ = train_test_split(
            X_fit,
            y_fit,
            sample_weight_fit,
            test_size=0.2,
            random_state=42,
            stratify=y_fit
        )
    else:
        X_model_train, y_model_train, weight_model_train = X_fit, y_fit, sample_weight_fit
        X_val, y_val = X_fit, y_fit

    candidate_grid = list(ParameterGrid({
        'n_estimators': [40, 80],
        'max_depth': [4, 6],
        'learning_rate': [0.05, 0.1],
        'subsample': [0.8],
        'colsample_bytree': [0.8]
    }))

    best_model = None
    best_params = None
    best_top3 = -np.inf
    best_loss = np.inf
    labels = np.arange(num_classes)

    for params in candidate_grid:
        candidate = XGBClassifier(
            random_state=42,
            n_jobs=1,
            tree_method='hist',
            device='cpu',
            objective='multi:softprob',
            num_class=num_classes,
            eval_metric='mlogloss',
            use_label_encoder=False,
            **params
        )
        candidate.fit(X_model_train, y_model_train, sample_weight=weight_model_train)
        y_val_proba = candidate.predict_proba(X_val)
        candidate_loss = top_k_loss(y_val, y_val_proba, k=PRIMARY_TOP_K, labels=labels)
        candidate_top3 = 1.0 - candidate_loss

        logging.info(
            f"  Candidate {params} -> Top-{PRIMARY_TOP_K}: {candidate_top3:.4f}, "
            f"loss: {candidate_loss:.4f}"
        )

        if (
            candidate_top3 > best_top3
            or (np.isclose(candidate_top3, best_top3) and candidate_loss < best_loss)
        ):
            best_model = clone(candidate)
            best_params = params
            best_top3 = candidate_top3
            best_loss = candidate_loss

    best_model.fit(X_fit, y_fit, sample_weight=sample_weight_fit)

    logging.info(f"Training rows used: {len(X_fit)}")
    logging.info(f"Feature count used: {X_fit.shape[1]}\n")
    logging.info(f"Best params by Top-{PRIMARY_TOP_K}: {best_params}")
    logging.info(f"Best validation Top-{PRIMARY_TOP_K}: {best_top3:.4f}\n")
    return best_model


def handle_class_imbalance(X_train, y_train):
    """
    Use SMOTE conservatively. Skip it for very large, high-cardinality training sets.
    """
    logging.info("\n" + "=" * 70)
    logging.info("STEP 2: HANDLING CLASS IMBALANCE WITH SMOTE")
    logging.info("=" * 70)

    unique, counts = np.unique(y_train, return_counts=True)
    logging.info("Before SMOTE:")
    logging.info(f"  Class distribution: {dict(zip(unique, counts))}")

    if X_train.shape[0] > 50000 or len(unique) > 60:
        logging.info("  Skipping SMOTE for large multi-class training set to avoid synthetic noise and runtime blow-up.\n")
        return X_train, y_train

    if len(unique) > 1 and counts.std() > counts.mean() * 0.3:
        min_class_size = min(counts)
        k_neighbors = max(1, min(3, min_class_size - 1))
        smote = SMOTE(random_state=42, k_neighbors=k_neighbors)
        try:
            X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
            logging.info("\nAfter SMOTE:")
            unique_after, counts_after = np.unique(y_train_balanced, return_counts=True)
            logging.info(f"  Class distribution: {dict(zip(unique_after, counts_after))}")
            logging.info(f"  New training set size: {X_train_balanced.shape[0]} (was {X_train.shape[0]})\n")
            return X_train_balanced, y_train_balanced
        except Exception as e:
            logging.warning(f"SMOTE failed: {e}. Using original data.\n")

    logging.info("  No significant class imbalance adjustment applied.\n")
    return X_train, y_train


def calibrate_model(model, X_train, y_train, sample_weight):
    """
    Keep calibration lightweight for large multi-class training runs.
    """
    logging.info("\n" + "=" * 70)
    logging.info("STEP 7: PROBABILITY CALIBRATION")
    logging.info("=" * 70)

    if X_train.shape[0] > 30000 or len(np.unique(y_train)) > 80:
        logging.info("Skipping explicit calibration for this large multi-class run; using the fitted XGBoost model directly.\n")
        model.fit(X_train, y_train, sample_weight=sample_weight)
        return model

    calibrated_model = CalibratedClassifierCV(
        estimator=model,
        method='sigmoid',
        cv=2
    )
    calibrated_model.fit(X_train, y_train, sample_weight=sample_weight)
    logging.info("Model calibrated successfully\n")
    return calibrated_model


if __name__ == "__main__":
    main()
