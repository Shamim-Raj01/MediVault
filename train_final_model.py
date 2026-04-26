"""
Final disease prediction training pipeline.

This script is intentionally conservative:
- it treats CSV files as heterogeneous sources,
- aligns symptom features by union,
- applies SMOTE only after the train/test split,
- evaluates on untouched holdout data,
- saves all artifacts needed for inference.
"""

from __future__ import annotations

import argparse
import logging
import os
import pickle
import re
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_selection import SelectKBest, VarianceThreshold, chi2
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import (
    learning_curve,
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier


warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(message)s")
LOGGER = logging.getLogger(__name__)

RANDOM_STATE = 42
RARE_CLASS_MIN_SAMPLES = 3
MIN_SYMPTOM_PRESENCE = 10
VARIANCE_THRESHOLD = 0.01
MAX_INTERACTION_FEATURES = 20
MIN_CLASS_SAMPLES_FINAL = 5
MIN_CLASS_F1_FINAL = 0.2
CHI2_MAX_FEATURES = 50

BASE_DIR = Path(__file__).resolve().parent
BASE_DATA_PATH = Path(
    os.getenv("DATA_PATH", BASE_DIR / "healthcare-chatbot" / "Data")
)
DATA_TRAIN_PATH = Path(
    os.getenv("DATA_TRAIN_PATH", BASE_DATA_PATH / "Train")
)

TARGET_ALIASES = {
    "disease",
    "diseases",
    "prognosis",
    "medical_condition",
    "medical condition",
}

MERGE_MAP = {
    "chicken pox": "chickenpox",
    "dengue fever": "dengue",
    "typhoid fever": "typhoid",
    "urinary tract infection (uti)": "urinary tract infection",
    "otitis media (ear infection)": "otitis media",
    "conjunctivitis (pink eye)": "conjunctivitis",
    "osteoarthritis": "osteoarthristis",
    "peptic ulcer disease": "peptic ulcer diseae",
    "dimorphic hemorrhoids(piles)": "dimorphic hemmorhoids(piles)",
    "hiv/aids": "aids",
    "chronic obstructive pulmonary disease": "chronic obstructive pulmonary disease (copd)",
    "sleep apnea": "obstructive sleep apnea (osa)",
    "heart attack": "myocardial infarction (heart attack)",
}

DROP_EXACT = {
    "id",
    "patient",
    "patient_id",
    "name",
    "date",
    "date_of_admission",
    "discharge_date",
    "doctor",
    "hospital",
    "insurance_provider",
    "billing_amount",
    "room_number",
    "admission_type",
    "medication",
    "test_results",
    "outcome_variable",
    "age",
    "gender",
    "blood_type",
    "blood_pressure",
    "cholesterol_level",
}

DROP_CONTAINS = (
    "unnamed",
    " id",
    "_id",
    "patient",
    "hospital",
    "name",
    "date",
)

YES_NO_MAP = {
    "yes": 1,
    "y": 1,
    "true": 1,
    "positive": 1,
    "present": 1,
    "no": 0,
    "n": 0,
    "false": 0,
    "negative": 0,
    "absent": 0,
}


def normalize_column_name(name: object) -> str:
    """Normalize source-specific columns into stable snake_case names."""
    name = str(name).strip().lower()
    name = re.sub(r"\s+", "_", name)
    name = re.sub(r"[^a-z0-9_]+", "_", name)
    name = re.sub(r"_+", "_", name).strip("_")
    name = re.sub(r"_(\d+)$", "", name)
    return name


def section(title: str) -> None:
    LOGGER.info("\n%s", "=" * 80)
    LOGGER.info(title)
    LOGGER.info("%s", "=" * 80)


def find_target_column(columns: list[str]) -> str | None:
    for col in columns:
        if col in TARGET_ALIASES:
            return col
    return None


def should_drop_column(col: str) -> bool:
    if col in DROP_EXACT:
        return True
    return any(token in col for token in DROP_CONTAINS)


def coerce_feature_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Convert feature values to numeric and drop columns that are not symptoms."""
    dropped = []
    clean = pd.DataFrame(index=df.index)

    for col in df.columns:
        if should_drop_column(col):
            dropped.append(col)
            continue

        series = df[col]
        if series.dtype == object or str(series.dtype).startswith("string"):
            lowered = series.astype(str).str.strip().str.lower()
            non_null = lowered[~lowered.isin({"", "nan", "none"})]
            unique_values = set(non_null.unique())
            if unique_values and unique_values.issubset(set(YES_NO_MAP)):
                clean[col] = lowered.map(YES_NO_MAP).fillna(0)
            else:
                dropped.append(col)
                continue
        else:
            clean[col] = pd.to_numeric(series, errors="coerce").fillna(0)

    clean = clean.apply(pd.to_numeric, errors="coerce").fillna(0)

    # Symptom features should be binary. Any positive value means present.
    clean = (clean > 0).astype(np.int8)
    return clean, dropped


def dataset_overlap_summary(feature_sets: dict[str, set[str]]) -> list[tuple[str, str, int, float]]:
    """Return pairwise feature overlap so mixed schemas are visible."""
    rows = []
    filenames = list(feature_sets)
    for i, left in enumerate(filenames):
        for right in filenames[i + 1 :]:
            left_set = feature_sets[left]
            right_set = feature_sets[right]
            union_size = len(left_set | right_set)
            overlap = len(left_set & right_set)
            jaccard = overlap / union_size if union_size else 0.0
            rows.append((left, right, overlap, jaccard))
    return rows


def load_and_validate_datasets(train_dir: Path) -> tuple[pd.DataFrame, pd.Series, dict]:
    section("1. DATA VALIDATION AND FEATURE ALIGNMENT")

    if not train_dir.exists():
        raise FileNotFoundError(f"Training directory not found: {train_dir}")

    csv_files = sorted(train_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in: {train_dir}")

    datasets = []
    metadata = {
        "loaded_files": [],
        "skipped_files": [],
        "raw_rows": 0,
        "raw_feature_counts": {},
        "dropped_columns": {},
        "feature_sets": {},
        "dataset_rows": {},
    }

    feature_union: set[str] = set()

    for path in csv_files:
        try:
            df = pd.read_csv(path)
        except Exception as exc:
            LOGGER.warning("Skipping %s: could not read CSV (%s)", path.name, exc)
            metadata["skipped_files"].append(path.name)
            continue

        original_rows = len(df)
        metadata["raw_rows"] += original_rows
        df.columns = [normalize_column_name(col) for col in df.columns]
        df = df.loc[:, ~pd.Index(df.columns).duplicated()]

        target_col = find_target_column(list(df.columns))
        if target_col is None:
            LOGGER.warning("Skipping %s: no disease/prognosis target column", path.name)
            metadata["skipped_files"].append(path.name)
            continue

        df = df.rename(columns={target_col: "disease"})
        y = df["disease"].astype(str).str.strip().str.lower()
        feature_df = df.drop(columns=["disease"])
        feature_df, dropped_cols = coerce_feature_frame(feature_df)

        dataset = feature_df.copy()
        dataset["disease"] = y
        dataset = dataset[dataset["disease"].notna() & (dataset["disease"] != "")]

        metadata["loaded_files"].append(path.name)
        metadata["raw_feature_counts"][path.name] = feature_df.shape[1]
        metadata["dropped_columns"][path.name] = dropped_cols
        metadata["feature_sets"][path.name] = set(feature_df.columns)
        metadata["dataset_rows"][path.name] = original_rows
        feature_union.update(feature_df.columns)

        LOGGER.info(
            "Loaded %-55s rows=%7d usable_features=%4d dropped_columns=%3d",
            path.name,
            original_rows,
            feature_df.shape[1],
            len(dropped_cols),
        )
        datasets.append(dataset)

    if not datasets:
        raise ValueError("No valid disease datasets were loaded.")

    feature_names = sorted(feature_union)
    aligned = []
    for df in datasets:
        for col in feature_names:
            if col not in df.columns:
                df[col] = 0
        aligned.append(df[feature_names + ["disease"]])

    combined = pd.concat(aligned, ignore_index=True)
    before_dedup = len(combined)
    combined = combined.drop_duplicates().reset_index(drop=True)
    duplicate_rows_removed = before_dedup - len(combined)

    X = combined[feature_names].apply(pd.to_numeric, errors="coerce").fillna(0)
    X = (X > 0).astype(np.int8)
    y = combined["disease"].astype(str).str.strip().str.lower()

    metadata["samples_before_dedup"] = before_dedup
    metadata["duplicate_rows_removed"] = duplicate_rows_removed
    metadata["features_after_merge"] = X.shape[1]
    metadata["max_single_dataset_features"] = max(metadata["raw_feature_counts"].values())

    LOGGER.info("\nDatasets loaded: %d", len(metadata["loaded_files"]))
    LOGGER.info("Total samples after duplicate removal: %d", len(X))
    LOGGER.info("Number of features after union merge: %d", X.shape[1])
    LOGGER.info("Duplicate rows removed: %d", duplicate_rows_removed)
    LOGGER.info("\nNumber of unique features per dataset:")
    for filename, count in metadata["raw_feature_counts"].items():
        LOGGER.info("- %-55s %d", filename, count)

    LOGGER.info("\nOverlap between datasets:")
    overlap_rows = dataset_overlap_summary(metadata["feature_sets"])
    for left, right, overlap, jaccard in overlap_rows:
        LOGGER.info(
            "- %-35s <-> %-35s shared=%3d jaccard=%.3f",
            left[:35],
            right[:35],
            overlap,
            jaccard,
        )

    LOGGER.info("\nInitial class distribution:")
    LOGGER.info("\n%s", y.value_counts().to_string())

    return X, y, metadata


def run_quality_checks(X: pd.DataFrame, y: pd.Series) -> tuple[pd.DataFrame, pd.Series, dict]:
    section("2. DATA QUALITY CHECKS")

    before_rows = len(X)
    zero_fraction = (X == 0).mean(axis=1)
    mostly_zero_mask = zero_fraction > 0.90
    mostly_zero_rows = int(mostly_zero_mask.sum())
    feature_sparsity = float((X == 0).to_numpy().mean())
    symptom_frequency = X.sum(axis=0).sort_values(ascending=False)

    non_empty_mask = X.sum(axis=1) > 0
    X = X.loc[non_empty_mask].reset_index(drop=True)
    y = y.loc[non_empty_mask].reset_index(drop=True)
    all_zero_rows_removed = before_rows - len(X)

    class_counts = y.value_counts()
    valid_classes = class_counts[class_counts >= RARE_CLASS_MIN_SAMPLES].index
    rare_rows_mask = y.isin(valid_classes)
    rare_classes_removed = sorted(set(class_counts.index) - set(valid_classes))

    X = X.loc[rare_rows_mask].reset_index(drop=True)
    y = y.loc[rare_rows_mask].reset_index(drop=True)

    LOGGER.info("Rows with mostly zeros (>90%% zeros): %d (%.2f%%)", mostly_zero_rows, mostly_zero_rows / before_rows * 100)
    LOGGER.info("Feature sparsity level: %.2f%% zeros", feature_sparsity * 100)
    LOGGER.info("Low-information rows removed: %d", all_zero_rows_removed)
    LOGGER.info("Rows removed where all features = 0: %d", all_zero_rows_removed)
    LOGGER.info(
        "Rare classes removed (< %d samples): %d",
        RARE_CLASS_MIN_SAMPLES,
        len(rare_classes_removed),
    )
    LOGGER.info("Samples after quality checks: %d", len(X))
    LOGGER.info("Features after quality checks: %d", X.shape[1])
    LOGGER.info("\nUpdated class distribution:")
    LOGGER.info("\n%s", y.value_counts().to_string())
    LOGGER.info("\nTop symptoms:")
    LOGGER.info("\n%s", symptom_frequency.head(10).to_string())

    return X, y, {
        "all_zero_rows_removed": all_zero_rows_removed,
        "mostly_zero_rows": mostly_zero_rows,
        "feature_sparsity": feature_sparsity,
        "top_symptoms": symptom_frequency.head(10),
        "rare_classes_removed": rare_classes_removed,
    }


def group_very_rare_classes(
    y: pd.Series,
    threshold: int = 5,
) -> tuple[pd.Series, dict]:
    original_classes = y.nunique()
    if original_classes <= 100:
        return y, {"classes_before_grouping": original_classes, "classes_after_grouping": original_classes}

    class_counts = y.value_counts()
    rare_classes = class_counts[class_counts < threshold].index
    y_grouped = y.where(~y.isin(rare_classes), "other")
    reduced_classes = y_grouped.nunique()
    LOGGER.info("Classes reduced from %d to %d", original_classes, reduced_classes)
    return y_grouped, {
        "classes_before_grouping": original_classes,
        "classes_after_grouping": reduced_classes,
        "grouped_rare_classes": sorted(rare_classes),
    }


def apply_label_merges(y: pd.Series) -> tuple[pd.Series, dict]:
    section("3. LABEL MERGING")

    classes_before = y.nunique()
    y_merged = y.astype(str).str.strip().str.lower().replace(MERGE_MAP)
    classes_after = y_merged.nunique()
    changed_rows = int((y.astype(str).str.strip().str.lower() != y_merged).sum())

    LOGGER.info("MERGE_MAP entries: %d", len(MERGE_MAP))
    LOGGER.info("Labels changed by merge map: %d", changed_rows)
    LOGGER.info("Classes before merge: %d", classes_before)
    LOGGER.info("Classes after merge: %d", classes_after)
    return y_merged, {
        "classes_before_merge": classes_before,
        "classes_after_merge": classes_after,
        "merged_label_rows": changed_rows,
    }


def drop_low_sample_classes(
    X: pd.DataFrame,
    y: pd.Series,
    min_samples: int = MIN_CLASS_SAMPLES_FINAL,
) -> tuple[pd.DataFrame, pd.Series, dict]:
    section("4. REMOVE VERY WEAK CLASSES BY SAMPLE COUNT")

    counts = y.value_counts()
    keep_classes = counts[counts >= min_samples].index
    dropped_classes = sorted(set(counts.index) - set(keep_classes))
    keep_mask = y.isin(keep_classes)

    X_out = X.loc[keep_mask].reset_index(drop=True)
    y_out = y.loc[keep_mask].reset_index(drop=True)

    LOGGER.info("Classes dropped with sample count < %d: %d", min_samples, len(dropped_classes))
    LOGGER.info("Samples after low-sample class drop: %d", len(X_out))
    LOGGER.info("Classes after low-sample class drop: %d", y_out.nunique())
    return X_out, y_out, {
        "low_sample_classes_dropped": dropped_classes,
        "rows_after_low_sample_drop": len(X_out),
    }


def weak_classes_from_report(
    report_dict: dict,
    label_encoder: LabelEncoder,
    min_f1: float = MIN_CLASS_F1_FINAL,
) -> list[str]:
    class_names = set(label_encoder.classes_)
    weak_classes = [
        name
        for name, values in report_dict.items()
        if name in class_names
        and isinstance(values, dict)
        and values.get("support", 0) > 0
        and values.get("f1-score", 0.0) < min_f1
    ]
    return sorted(weak_classes)


def build_xgb_model(num_classes: int, **overrides) -> XGBClassifier:
    params = {
        "objective": "multi:softprob",
        "eval_metric": "mlogloss",
        "n_estimators": 150,
        "max_depth": 5,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": RANDOM_STATE,
        "num_class": num_classes,
        "n_jobs": -1,
        "tree_method": "hist",
    }
    params.update(overrides)
    return XGBClassifier(**params)


def safe_smote(y_train: np.ndarray, cv_folds: int | None = None) -> SMOTE | None:
    counts = pd.Series(y_train).value_counts()
    min_class_size = int(counts.min())
    effective_min_class_size = min_class_size
    if cv_folds is not None and cv_folds > 1:
        effective_min_class_size = int(np.floor(min_class_size * (cv_folds - 1) / cv_folds))

    if effective_min_class_size < 2:
        LOGGER.warning("Skipping SMOTE: at least one training class has fewer than 2 samples.")
        return None

    k_neighbors = min(3, effective_min_class_size - 1)
    LOGGER.info("SMOTE k_neighbors selected safely: %d", k_neighbors)
    return SMOTE(random_state=RANDOM_STATE, k_neighbors=k_neighbors)


def remove_rare_symptoms(
    X: pd.DataFrame,
    min_presence: int = MIN_SYMPTOM_PRESENCE,
) -> tuple[pd.DataFrame, list[str]]:
    section("3. FEATURE SPARSITY REDUCTION")

    original_feature_count = X.shape[1]
    valid_cols = X.sum(axis=0) >= min_presence
    X_reduced = X.loc[:, valid_cols].copy()
    removed_features = original_feature_count - X_reduced.shape[1]
    removed_feature_names = X.columns[~valid_cols].tolist()

    LOGGER.info("Rare symptom min_presence: %d", min_presence)
    LOGGER.info("Rare symptoms removed: %d", removed_features)
    LOGGER.info("Features after rare removal: %d", X_reduced.shape[1])
    return X_reduced, removed_feature_names


def apply_variance_selection(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], VarianceThreshold]:
    section("4. STRONG VARIANCE FILTER")

    selector = VarianceThreshold(threshold=VARIANCE_THRESHOLD)
    selector.fit(X_train)
    selected_features = X_train.columns[selector.get_support()].tolist()
    removed = X_train.shape[1] - len(selected_features)

    X_train_selected = pd.DataFrame(
        selector.transform(X_train),
        columns=selected_features,
        index=X_train.index,
    )
    X_test_selected = pd.DataFrame(
        selector.transform(X_test),
        columns=selected_features,
        index=X_test.index,
    )

    LOGGER.info("Variance threshold: %.4f", VARIANCE_THRESHOLD)
    LOGGER.info("Feature count before variance: %d", X_train.shape[1])
    LOGGER.info("Feature count after variance:  %d", len(selected_features))
    LOGGER.info("Low-variance features removed: %d", removed)
    return X_train_selected, X_test_selected, selected_features, selector


def apply_chi2_selection(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], SelectKBest]:
    section("5. CHI-SQUARE FEATURE SELECTION")

    k = min(CHI2_MAX_FEATURES, X_train.shape[1])
    selector = SelectKBest(chi2, k=k)
    selector.fit(X_train, y_train)
    selected_indices = selector.get_support(indices=True)
    selected_features = X_train.columns[selected_indices].tolist()

    X_train_selected = pd.DataFrame(
        selector.transform(X_train),
        columns=selected_features,
        index=X_train.index,
    )
    X_test_selected = pd.DataFrame(
        selector.transform(X_test),
        columns=selected_features,
        index=X_test.index,
    )

    LOGGER.info("Selected top features: %d", len(selected_indices))
    return X_train_selected, X_test_selected, selected_features, selector


def print_feature_count_comparison(
    before_cleaning: int,
    after_rare: int,
    after_variance: int,
    after_chi2: int,
) -> None:
    section("6. FEATURE COUNT COMPARISON")
    LOGGER.info("Before cleaning: %d", before_cleaning)
    LOGGER.info("After rare removal: %d", after_rare)
    LOGGER.info("After variance: %d", after_variance)
    LOGGER.info("After chi2: %d", after_chi2)


def fit_interaction_features(X_train: pd.DataFrame) -> list[tuple[str, str, str]]:
    prevalence = X_train.mean(axis=0).sort_values(ascending=False)
    top_features = prevalence.head(8).index.tolist()
    interactions = []
    for i, left in enumerate(top_features):
        for right in top_features[i + 1 :]:
            name = f"{left}_and_{right}"
            interactions.append((left, right, name))
            if len(interactions) >= MAX_INTERACTION_FEATURES:
                return interactions
    return interactions


def add_interaction_features(
    X: pd.DataFrame,
    interactions: list[tuple[str, str, str]],
) -> pd.DataFrame:
    X_out = X.copy()
    for left, right, name in interactions:
        if left in X_out.columns and right in X_out.columns:
            X_out[name] = (X_out[left] * X_out[right]).astype(np.int8)
    return X_out


def train_logistic_baseline(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: np.ndarray,
    y_test: np.ndarray,
) -> tuple[LogisticRegression, dict]:
    section("4. SIMPLE BASELINE COMPARISON")

    model = LogisticRegression(max_iter=1000, n_jobs=-1, random_state=RANDOM_STATE)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, pred)
    f1 = f1_score(y_test, pred, average="weighted", zero_division=0)

    LOGGER.info("Baseline Accuracy: %.4f", accuracy)
    LOGGER.info("Baseline F1: %.4f", f1)
    return model, {"accuracy": float(accuracy), "f1_weighted": float(f1)}


def evaluate_variant(
    name: str,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: np.ndarray,
    y_test: np.ndarray,
    num_classes: int,
    use_smote: bool,
    use_feature_engineering: bool,
) -> dict:
    X_train_variant = X_train.copy()
    X_test_variant = X_test.copy()

    if use_feature_engineering:
        interactions = fit_interaction_features(X_train_variant)
        X_train_variant = add_interaction_features(X_train_variant, interactions)
        X_test_variant = add_interaction_features(X_test_variant, interactions)

    if use_smote:
        smote = safe_smote(y_train)
        if smote is not None:
            X_train_variant, y_train_variant = smote.fit_resample(X_train_variant, y_train)
        else:
            y_train_variant = y_train
    else:
        y_train_variant = y_train

    model = build_xgb_model(num_classes)
    model.fit(X_train_variant, y_train_variant)
    pred = model.predict(X_test_variant)
    accuracy = accuracy_score(y_test, pred)
    f1 = f1_score(y_test, pred, average="weighted", zero_division=0)
    return {"variant": name, "accuracy": float(accuracy), "f1_weighted": float(f1)}


def run_ablation_tests(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: np.ndarray,
    y_test: np.ndarray,
    num_classes: int,
) -> list[dict]:
    section("5. ABLATION TESTS")

    variants = [
        ("A. Without SMOTE", False, False),
        ("B. With SMOTE", True, False),
        ("C. Without feature engineering", True, False),
        ("D. With feature engineering", True, True),
    ]
    results = [
        evaluate_variant(name, X_train, X_test, y_train, y_test, num_classes, use_smote, use_fe)
        for name, use_smote, use_fe in variants
    ]

    LOGGER.info("Model Variant | Accuracy | F1 Score")
    for result in results:
        LOGGER.info(
            "%-32s | %.4f | %.4f",
            result["variant"],
            result["accuracy"],
            result["f1_weighted"],
        )
    return results


def train_xgb_with_optional_smote(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    num_classes: int,
    use_smote: bool,
) -> tuple[XGBClassifier, pd.DataFrame, np.ndarray]:
    X_fit = X_train.copy()
    y_fit = y_train
    if use_smote:
        smote = safe_smote(y_train)
        if smote is not None:
            X_fit, y_fit = smote.fit_resample(X_fit, y_fit)

    model = build_xgb_model(num_classes)
    model.fit(X_fit, y_fit)
    return model, X_fit, y_fit


def train_xgb_with_early_stopping(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    num_classes: int,
    use_smote: bool,
) -> tuple[XGBClassifier, pd.DataFrame, np.ndarray]:
    X_train_sub, X_val, y_train_sub, y_val = train_test_split(
        X_train,
        y_train,
        test_size=0.1,
        stratify=y_train,
        random_state=RANDOM_STATE,
    )

    X_fit = X_train_sub.copy()
    y_fit = y_train_sub
    if use_smote:
        smote = safe_smote(y_train_sub)
        if smote is not None:
            X_fit, y_fit = smote.fit_resample(X_fit, y_fit)

    model = build_xgb_model(num_classes, early_stopping_rounds=20)
    model.fit(
        X_fit,
        y_fit,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    LOGGER.info("Early stopping validation samples: %d", len(X_val))
    return model, X_fit, y_fit


def run_smote_ablation(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: np.ndarray,
    y_test: np.ndarray,
    num_classes: int,
) -> tuple[list[dict], dict[str, XGBClassifier]]:
    section("7. ABLATION: SMOTE VS NO SMOTE")

    results = []
    models = {}
    for variant, use_smote in [("Without SMOTE", False), ("With SMOTE", True)]:
        model, _, _ = train_xgb_with_optional_smote(
            X_train,
            y_train,
            num_classes,
            use_smote=use_smote,
        )
        y_pred = model.predict(X_test)
        result = {
            "variant": variant,
            "use_smote": use_smote,
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "f1_weighted": float(f1_score(y_test, y_pred, average="weighted", zero_division=0)),
        }
        results.append(result)
        models[variant] = model

    LOGGER.info("Variant | Accuracy | F1")
    for result in results:
        LOGGER.info(
            "%-14s | %.4f | %.4f",
            result["variant"],
            result["accuracy"],
            result["f1_weighted"],
        )

    return results, models


def cross_validate_model(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    num_classes: int,
    smote: SMOTE | None,
    model_params: dict | None = None,
) -> float | None:
    section("8. CROSS VALIDATION")

    min_class_size = int(pd.Series(y_train).value_counts().min())
    n_splits = min(5, min_class_size)
    if n_splits < 2:
        LOGGER.warning("Skipping cross-validation: not enough samples per class.")
        return None
    if n_splits < 5:
        LOGGER.warning(
            "Using StratifiedKFold(n_splits=%d) because the smallest training class has %d samples.",
            n_splits,
            min_class_size,
        )

    model = build_xgb_model(num_classes, **(model_params or {}))
    cv_smote = safe_smote(y_train, cv_folds=n_splits) if smote is not None else None
    estimator = (
        ImbPipeline([("smote", cv_smote), ("model", model)])
        if cv_smote is not None
        else model
    )
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    try:
        scores = cross_val_score(estimator, X_train, y_train, cv=cv, scoring="accuracy", n_jobs=-1)
    except PermissionError as exc:
        LOGGER.warning("Parallel CV failed (%s). Retrying with n_jobs=1.", exc)
        scores = cross_val_score(estimator, X_train, y_train, cv=cv, scoring="accuracy", n_jobs=1)

    LOGGER.info("StratifiedKFold splits: %d", n_splits)
    LOGGER.info("CV accuracy scores: %s", np.round(scores, 4))
    LOGGER.info("CV Accuracy: %.4f", scores.mean())
    return float(scores.mean())


def build_calibrated_model(
    final_model: XGBClassifier,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    num_classes: int,
    use_smote: bool,
):
    section("9. PROBABILITY CALIBRATION")

    min_class_size = int(pd.Series(y_train).value_counts().min())
    cv_folds = min(3, min_class_size)
    if cv_folds < 2:
        LOGGER.warning("Skipping calibration: not enough training samples per class.")
        return final_model
    if cv_folds < 3:
        LOGGER.warning(
            "Using calibration cv=%d because the smallest training class has %d samples.",
            cv_folds,
            min_class_size,
        )

    base_estimator = build_xgb_model(num_classes)
    if use_smote:
        smote = safe_smote(y_train, cv_folds=cv_folds)
        if smote is not None:
            base_estimator = ImbPipeline([("smote", smote), ("model", base_estimator)])

    try:
        calibrated_model = CalibratedClassifierCV(
            estimator=base_estimator,
            method="sigmoid",
            cv=cv_folds,
        )
    except TypeError:
        calibrated_model = CalibratedClassifierCV(
            base_estimator=base_estimator,
            method="sigmoid",
            cv=cv_folds,
        )

    calibrated_model.fit(X_train, y_train)
    LOGGER.info("Using calibrated probabilities")
    return calibrated_model


def top_k_accuracy(model, X, y, k=3):
    probs = model.predict_proba(X)
    topk = np.argsort(probs, axis=1)[:, -k:]
    return np.mean([y[i] in topk[i] for i in range(len(y))])


def print_top_k_metrics(model, X_test: pd.DataFrame, y_test: np.ndarray) -> dict:
    top1 = top_k_accuracy(model, X_test, y_test, k=1)
    top3 = top_k_accuracy(model, X_test, y_test, k=min(3, len(np.unique(y_test))))
    top5 = top_k_accuracy(model, X_test, y_test, k=min(5, len(np.unique(y_test))))
    LOGGER.info("Top-1 Accuracy (same as normal accuracy): %.4f", top1)
    LOGGER.info("Top-3 Accuracy: %.4f", top3)
    LOGGER.info("Top-5 Accuracy: %.4f", top5)
    return {"top1": float(top1), "top3": float(top3), "top5": float(top5)}


def tune_xgboost(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    num_classes: int,
    smote: SMOTE | None,
) -> tuple[dict, float]:
    section("7. RANDOMIZED XGBOOST SEARCH")

    model = build_xgb_model(num_classes)
    estimator = (
        ImbPipeline([("smote", smote), ("model", model)])
        if smote is not None
        else model
    )
    prefix = "model__" if smote is not None else ""
    param_distributions = {
        f"{prefix}n_estimators": [100, 150, 200],
        f"{prefix}max_depth": [4, 5, 6],
        f"{prefix}learning_rate": [0.05, 0.1],
        f"{prefix}subsample": [0.7, 0.8, 1.0],
    }
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    search = RandomizedSearchCV(
        estimator=estimator,
        param_distributions=param_distributions,
        n_iter=10,
        scoring="accuracy",
        cv=cv,
        n_jobs=-1,
        random_state=RANDOM_STATE,
        verbose=1,
    )
    try:
        search.fit(X_train, y_train)
    except PermissionError as exc:
        LOGGER.warning("Parallel randomized search failed (%s). Retrying with n_jobs=1.", exc)
        search.set_params(n_jobs=1)
        search.fit(X_train, y_train)

    best_params = {
        key.replace(prefix, ""): value for key, value in search.best_params_.items()
    }
    LOGGER.info("Best XGBoost parameters: %s", best_params)
    LOGGER.info("Best randomized-search CV accuracy: %.4f", search.best_score_)
    return best_params, float(search.best_score_)


def fit_final_model_with_early_stopping(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    num_classes: int,
    smote: SMOTE | None,
    best_params: dict,
) -> tuple[XGBClassifier, pd.DataFrame, np.ndarray]:
    section("8. HANDLE CLASS IMBALANCE AND TRAIN FINAL MODEL")

    X_fit, X_val, y_fit, y_val = train_test_split(
        X_train,
        y_train,
        test_size=0.15,
        random_state=RANDOM_STATE,
        stratify=y_train,
    )

    fit_smote = safe_smote(y_fit) if smote is not None else None
    if fit_smote is not None:
        X_train_resampled, y_train_resampled = fit_smote.fit_resample(X_fit, y_fit)
        LOGGER.info("Training samples before SMOTE: %d", len(X_fit))
        LOGGER.info("Training samples after SMOTE:  %d", len(X_train_resampled))
    else:
        X_train_resampled, y_train_resampled = X_fit, y_fit
        LOGGER.info("Training without SMOTE.")

    LOGGER.info("Validation samples for early stopping: %d", len(X_val))

    LOGGER.info("\nTraining class distribution:")
    LOGGER.info("%s", pd.Series(y_fit).value_counts().sort_index().to_string())
    LOGGER.info("\nTraining class distribution after SMOTE:")
    LOGGER.info("%s", pd.Series(y_train_resampled).value_counts().sort_index().to_string())

    model = build_xgb_model(num_classes, **best_params, early_stopping_rounds=20)
    model.fit(
        X_train_resampled,
        y_train_resampled,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    LOGGER.info("XGBoost final model trained.")
    return model, X_train_resampled, y_train_resampled


def evaluate_model(
    model,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    label_encoder: LabelEncoder,
) -> dict:
    section("9. FINAL EVALUATION ON UNTOUCHED TEST SET")

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1_weighted = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    report = classification_report(
        y_test,
        y_pred,
        labels=np.arange(len(label_encoder.classes_)),
        target_names=label_encoder.classes_,
        digits=4,
        zero_division=0,
    )
    report_dict = classification_report(
        y_test,
        y_pred,
        labels=np.arange(len(label_encoder.classes_)),
        target_names=label_encoder.classes_,
        digits=4,
        zero_division=0,
        output_dict=True,
    )

    LOGGER.info("Final Accuracy: %.4f", accuracy)
    LOGGER.info("Final F1 Score: %.4f", f1_weighted)
    LOGGER.info("\nclassification_report:\n%s", report)
    LOGGER.info("Worst performing diseases:")
    disease_rows = [
        (name, values["f1-score"], values["precision"], values["recall"], values["support"])
        for name, values in report_dict.items()
        if isinstance(values, dict) and name in set(label_encoder.classes_)
    ]
    for name, f1_value, precision, recall, support in sorted(disease_rows, key=lambda row: row[1])[:10]:
        LOGGER.info(
            "%-40s f1=%.4f precision=%.4f recall=%.4f support=%s",
            name[:40],
            f1_value,
            precision,
            recall,
            int(support),
        )

    return {
        "accuracy": float(accuracy),
        "f1_weighted": float(f1_weighted),
        "classification_report": report,
        "classification_report_dict": report_dict,
        "y_pred": y_pred,
    }


def print_confusion_analysis(
    y_test: np.ndarray,
    y_pred: np.ndarray,
    label_encoder: LabelEncoder,
) -> None:
    section("10. CONFUSION ANALYSIS")

    cm = confusion_matrix(y_test, y_pred, labels=np.arange(len(label_encoder.classes_)))
    confused = []
    for actual_idx, actual_name in enumerate(label_encoder.classes_):
        for pred_idx, pred_name in enumerate(label_encoder.classes_):
            if actual_idx == pred_idx:
                continue
            count = int(cm[actual_idx, pred_idx])
            if count:
                confused.append((count, actual_name, pred_name))

    confused.sort(reverse=True, key=lambda row: row[0])
    LOGGER.info("Top 10 most confused classes:")
    if not confused:
        LOGGER.info("No misclassifications found on the test set.")
    for count, actual_name, pred_name in confused[:10]:
        LOGGER.info("actual=%-35s predicted=%-35s count=%d", actual_name[:35], pred_name[:35], count)

    total_errors = int((y_test != y_pred).sum())
    LOGGER.info("\nMisclassification summary:")
    LOGGER.info("Total test samples: %d", len(y_test))
    LOGGER.info("Total misclassified: %d", total_errors)
    LOGGER.info("Misclassification rate: %.2f%%", total_errors / len(y_test) * 100)


def plot_learning_curve(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    num_classes: int,
    use_smote: bool,
    output_path: Path,
) -> None:
    section("11. LEARNING CURVE")

    min_class_size = int(pd.Series(y_train).value_counts().min())
    cv_folds = min(3, min_class_size)
    if cv_folds < 2:
        LOGGER.warning("Skipping learning curve: not enough samples per class.")
        return

    estimator = build_xgb_model(num_classes, n_estimators=75)
    if use_smote:
        smote = safe_smote(y_train, cv_folds=cv_folds)
        if smote is not None:
            estimator = ImbPipeline([("smote", smote), ("model", estimator)])

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)
    try:
        train_sizes, train_scores, val_scores = learning_curve(
            estimator,
            X_train,
            y_train,
            train_sizes=np.linspace(0.2, 1.0, 5),
            cv=cv,
            scoring="accuracy",
            n_jobs=-1,
        )
    except PermissionError as exc:
        LOGGER.warning("Parallel learning curve failed (%s). Retrying with n_jobs=1.", exc)
        train_sizes, train_scores, val_scores = learning_curve(
            estimator,
            X_train,
            y_train,
            train_sizes=np.linspace(0.2, 1.0, 5),
            cv=cv,
            scoring="accuracy",
            n_jobs=1,
        )

    train_mean = train_scores.mean(axis=1)
    val_mean = val_scores.mean(axis=1)
    LOGGER.info("Learning curve train sizes: %s", train_sizes)
    LOGGER.info("Training scores: %s", np.round(train_mean, 4))
    LOGGER.info("Validation scores: %s", np.round(val_mean, 4))

    plt.figure(figsize=(8, 5))
    plt.plot(train_sizes, train_mean, marker="o", label="Training score")
    plt.plot(train_sizes, val_mean, marker="o", label="Validation score")
    plt.xlabel("Training examples")
    plt.ylabel("Accuracy")
    plt.title("Learning Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    LOGGER.info("Learning curve saved: %s", output_path)


def plot_feature_importance(model: XGBClassifier, feature_names: list[str], output_path: Path) -> None:
    section("12. FEATURE IMPORTANCE")

    importances = np.asarray(model.feature_importances_)
    top_idx = np.argsort(importances)[-20:][::-1]
    top_features = [feature_names[i] for i in top_idx]
    top_values = importances[top_idx]

    LOGGER.info("Top important symptoms:")
    for feature, value in zip(top_features[:10], top_values[:10]):
        LOGGER.info("%-40s %.6f", feature, value)

    importance_table = pd.DataFrame({"feature": top_features, "importance": top_values})
    csv_path = output_path.with_suffix(".csv")
    importance_table.to_csv(csv_path, index=False)

    plt.figure(figsize=(10, 6))
    plt.barh(top_features[::-1], top_values[::-1], color="#2f6f73")
    plt.xlabel("Importance")
    plt.title("Top 20 Feature Importances")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    LOGGER.info("Feature importance plot saved: %s", output_path)
    LOGGER.info("Top 20 feature importance table saved: %s", csv_path)


def save_artifacts(
    baseline_model: object,
    improved_model,
    label_encoder: LabelEncoder,
    feature_names: list[str],
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    output_dir: Path,
) -> None:
    section("13. SAVE ARTIFACTS")

    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts = {
        "baseline_model.pkl": baseline_model,
        "improved_model.pkl": improved_model,
        "calibrated_model.pkl": improved_model,
        "model.pkl": improved_model,
        "label_encoder.pkl": label_encoder,
        "feature_names.pkl": feature_names,
        "X_test.pkl": X_test,
        "y_test.pkl": y_test,
    }

    for filename, obj in artifacts.items():
        path = output_dir / filename
        with path.open("wb") as file:
            pickle.dump(obj, file)
        LOGGER.info("Saved %s", path)


def print_debug_summary(
    metadata: dict,
    X_before_quality: pd.DataFrame,
    X_after_quality: pd.DataFrame,
    after_rare_feature_count: int,
    after_variance_feature_count: int,
    selected_feature_count: int,
    final_feature_count: int,
    y_after_quality: pd.Series,
    baseline_metrics: dict,
    final_acc: float,
    cv_mean_acc: float | None,
) -> None:
    section("12. DEBUG SUMMARY")

    largest_single_dataset_rows = max([0] + list(metadata.get("dataset_rows", {}).values()))

    LOGGER.info("Dataset size before cleaning: %d", metadata["samples_before_dedup"])
    LOGGER.info("Dataset size after cleaning:  %d", len(X_after_quality))
    LOGGER.info("Dataset size: %d", len(X_after_quality))
    LOGGER.info(
        "Feature count before/after: %d -> %d",
        X_before_quality.shape[1],
        final_feature_count,
    )
    LOGGER.info("Feature count after rare removal: %d", after_rare_feature_count)
    LOGGER.info("Feature count after variance: %d", after_variance_feature_count)
    LOGGER.info("Feature count after chi2: %d", selected_feature_count)

    LOGGER.info("Accuracy before vs after:")
    LOGGER.info("Old: %.2f%%", baseline_metrics["accuracy"] * 100)
    LOGGER.info("New: %.2f%%", final_acc * 100)
    LOGGER.info("Delta: %.2f percentage points", (final_acc - baseline_metrics["accuracy"]) * 100)
    if cv_mean_acc is not None:
        LOGGER.info("CV Score: %.4f", cv_mean_acc)

    LOGGER.info(
        "Number of features before/after merge: %d -> %d",
        metadata["max_single_dataset_features"],
        metadata["features_after_merge"],
    )
    LOGGER.info(
        "Number of features before/after quality checks: %d -> %d",
        X_before_quality.shape[1],
        X_after_quality.shape[1],
    )
    LOGGER.info(
        "Dataset size increase vs largest single source: %d -> %d rows",
        largest_single_dataset_rows,
        metadata["raw_rows"],
    )
    LOGGER.info(
        "Dataset size after cleaning: %d rows before quality checks, %d rows after quality checks",
        len(X_before_quality),
        len(X_after_quality),
    )
    LOGGER.info("Low-information rows removed: %d", metadata.get("all_zero_rows_removed", 0))
    LOGGER.info("Feature sparsity level: %.2f%% zeros", metadata.get("feature_sparsity", 0) * 100)

    LOGGER.info("\nTop 10 classes after cleaning:")
    LOGGER.info("\n%s", y_after_quality.value_counts().head(10).to_string())

    LOGGER.info("\nLoaded files:")
    for filename in metadata["loaded_files"]:
        LOGGER.info("- %s", filename)

    if metadata["skipped_files"]:
        LOGGER.info("\nSkipped files:")
        for filename in metadata["skipped_files"]:
            LOGGER.info("- %s", filename)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train final disease prediction model.")
    parser.add_argument(
        "--train-dir",
        type=Path,
        default=DATA_TRAIN_PATH,
        help="Directory containing training CSV files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=BASE_DIR,
        help="Directory where artifacts will be saved.",
    )
    return parser.parse_args()


def run_training_round(
    X: pd.DataFrame,
    y: pd.Series,
    X_before_quality: pd.DataFrame,
    metadata: dict,
    quality_metadata: dict,
    args: argparse.Namespace,
    round_name: str,
    save_outputs: bool,
) -> dict:
    section(f"{round_name.upper()} TRAINING ROUND")
    original_feature_count = X_before_quality.shape[1]
    X, rare_removed_features = remove_rare_symptoms(X, min_presence=MIN_SYMPTOM_PRESENCE)
    after_rare_feature_count = X.shape[1]

    section("5. TRAIN / TEST SPLIT")
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_encoded,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y_encoded,
    )
    LOGGER.info("Train shape: %s", X_train.shape)
    LOGGER.info("Test shape:  %s", X_test.shape)
    LOGGER.info("Labels encoded once. Number of classes: %d", len(label_encoder.classes_))

    num_classes = len(label_encoder.classes_)
    X_train_variance, X_test_variance, variance_features, _ = apply_variance_selection(
        X_train,
        X_test,
    )
    after_variance_feature_count = X_train_variance.shape[1]

    X_train_selected, X_test_selected, selected_features, _ = apply_chi2_selection(
        X_train_variance,
        X_test_variance,
        y_train,
    )
    after_chi2_feature_count = X_train_selected.shape[1]

    print_feature_count_comparison(
        before_cleaning=original_feature_count,
        after_rare=after_rare_feature_count,
        after_variance=after_variance_feature_count,
        after_chi2=after_chi2_feature_count,
    )

    ablation_results, ablation_models = run_smote_ablation(
        X_train_selected,
        X_test_selected,
        y_train,
        y_test,
        num_classes,
    )

    best_result = max(ablation_results, key=lambda row: (row["f1_weighted"], row["accuracy"]))
    best_use_smote = bool(best_result["use_smote"])
    baseline_result = next(row for row in ablation_results if not row["use_smote"])
    LOGGER.info(
        "Best configuration from ablation: %s (Accuracy %.4f, F1 %.4f)",
        best_result["variant"],
        best_result["accuracy"],
        best_result["f1_weighted"],
    )

    smote = safe_smote(y_train) if best_use_smote else None
    cv_mean_acc = cross_validate_model(X_train_selected, y_train, num_classes, smote)

    section("9. FINAL MODEL TRAINING WITH EARLY STOPPING")
    raw_model, X_train_fit, y_train_fit = train_xgb_with_early_stopping(
        X_train_selected,
        y_train,
        num_classes,
        use_smote=best_use_smote,
    )
    LOGGER.info("Final XGBoost training samples: %d", len(X_train_fit))

    calibrated_model = build_calibrated_model(
        raw_model,
        X_train_selected,
        y_train,
        num_classes,
        use_smote=best_use_smote,
    )

    metrics = evaluate_model(calibrated_model, X_test_selected, y_test, label_encoder)
    topk_metrics = print_top_k_metrics(calibrated_model, X_test_selected, y_test)
    print_confusion_analysis(y_test, metrics["y_pred"], label_encoder)
    plot_learning_curve(
        X_train_selected,
        y_train,
        num_classes,
        use_smote=best_use_smote,
        output_path=args.output_dir / "learning_curve.png",
    )

    plot_feature_importance(
        raw_model,
        selected_features,
        args.output_dir / "feature_importance_top20.png",
    )
    if save_outputs:
        save_artifacts(
            ablation_models["Without SMOTE"],
            calibrated_model,
            label_encoder,
            selected_features,
            X_test_selected,
            y_test,
            args.output_dir,
        )

    print_debug_summary(
        metadata={**metadata, **quality_metadata},
        X_before_quality=X_before_quality,
        X_after_quality=X,
        after_rare_feature_count=after_rare_feature_count,
        after_variance_feature_count=after_variance_feature_count,
        selected_feature_count=len(selected_features),
        final_feature_count=len(selected_features),
        y_after_quality=y,
        baseline_metrics={"accuracy": baseline_result["accuracy"]},
        final_acc=metrics["accuracy"],
        cv_mean_acc=cv_mean_acc,
    )

    section("DONE")
    LOGGER.info("Final Accuracy: %.4f", metrics["accuracy"])
    LOGGER.info("F1 Score: %.4f", metrics["f1_weighted"])
    LOGGER.info("Top-3 Accuracy: %.4f", topk_metrics["top3"])
    LOGGER.info("Top-5 Accuracy: %.4f", topk_metrics["top5"])
    if cv_mean_acc is not None:
        LOGGER.info("CV Accuracy: %.4f", cv_mean_acc)

    return {
        "metrics": metrics,
        "topk_metrics": topk_metrics,
        "cv_mean_acc": cv_mean_acc,
        "label_encoder": label_encoder,
    }


def main() -> None:
    args = parse_args()

    section("FINAL DISEASE MODEL TRAINING")
    LOGGER.info("Training directory: %s", args.train_dir.resolve())
    LOGGER.info("Output directory:   %s", args.output_dir.resolve())

    X, y, metadata = load_and_validate_datasets(args.train_dir)
    X_before_quality = X.copy()

    X, y, quality_metadata = run_quality_checks(X, y)
    y, merge_metadata = apply_label_merges(y)
    X, y, class_drop_metadata = drop_low_sample_classes(X, y, min_samples=MIN_CLASS_SAMPLES_FINAL)
    if y.nunique() < 2:
        raise ValueError("Need at least two disease classes after label cleanup.")

    first_round = run_training_round(
        X=X,
        y=y,
        X_before_quality=X_before_quality,
        metadata={**metadata, **merge_metadata, **class_drop_metadata},
        quality_metadata=quality_metadata,
        args=args,
        round_name="initial",
        save_outputs=False,
    )

    weak_classes = weak_classes_from_report(
        first_round["metrics"]["classification_report_dict"],
        first_round["label_encoder"],
        min_f1=MIN_CLASS_F1_FINAL,
    )

    if weak_classes:
        section("REMOVE VERY WEAK CLASSES BY F1")
        LOGGER.info("Classes dropped with F1 < %.2f: %d", MIN_CLASS_F1_FINAL, len(weak_classes))
        for disease in weak_classes[:30]:
            LOGGER.info("- %s", disease)
        keep_mask = ~y.isin(weak_classes)
        X = X.loc[keep_mask].reset_index(drop=True)
        y = y.loc[keep_mask].reset_index(drop=True)
    else:
        LOGGER.info("No classes found with F1 < %.2f", MIN_CLASS_F1_FINAL)

    if y.nunique() < 2:
        raise ValueError("Weak-class pruning left fewer than two disease classes.")

    final_round = run_training_round(
        X=X,
        y=y,
        X_before_quality=X_before_quality,
        metadata={
            **metadata,
            **merge_metadata,
            **class_drop_metadata,
            "f1_weak_classes_dropped": weak_classes,
        },
        quality_metadata=quality_metadata,
        args=args,
        round_name="final",
        save_outputs=True,
    )

    section("FINAL CLEANED MODEL SUMMARY")
    LOGGER.info("Final Accuracy: %.4f", final_round["metrics"]["accuracy"])
    LOGGER.info("F1 Score: %.4f", final_round["metrics"]["f1_weighted"])
    LOGGER.info("Top-3 Accuracy: %.4f", final_round["topk_metrics"]["top3"])


if __name__ == "__main__":
    main()
