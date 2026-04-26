"""
INFERENCE WRAPPER FOR IMPROVED MODEL
=====================================
This module loads the improved calibrated XGBoost model and provides
inference functions for disease prediction. Compatible with app.py.
"""

import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, List

CONFIDENCE_WARNING_THRESHOLD = 0.4
MIN_SYMPTOMS_REQUIRED = 3
CRITICAL_SYMPTOM_WEIGHTS = {
    "fever": 1.2,
    "breathlessness": 1.5,
    "shortness_of_breath": 1.5,
    "difficulty_breathing": 1.5,
    "difficulty_in_breathing": 1.5,
    "chest_pain": 1.5,
    "sharp_chest_pain": 1.5,
    "chest_tightness": 1.4,
    "irregular_heartbeat": 1.4,
    "palpitations": 1.3,
    "vomiting_blood": 1.6,
    "blood_in_stool": 1.5,
    "blood_in_urine": 1.5,
    "seizures": 1.6,
    "fainting": 1.5,
    "weakness": 1.2,
    "focal_weakness": 1.4,
    "yellowish_skin": 1.3,
}

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

class ImprovedModelInference:
    """
    Wrapper for loading and using the improved calibrated model.
    Handles feature engineering, prediction, and probability calibration.
    """
    
    def __init__(self, model_dir='.'):
        """
        Initialize the inference engine by loading saved models.
        
        Args:
            model_dir: Directory containing model files
        """
        model_dir = Path(model_dir)
        
        # Load model files
        self.model = self._load_pickle(model_dir / 'model_improved.pkl', 'Model')
        self.label_encoder = self._load_pickle(model_dir / 'le_classification_improved.pkl', 'Label Encoder')
        self.feature_names = self._load_pickle(model_dir / 'feature_names_improved.pkl', 'Feature Names')
        
    @staticmethod
    def _load_pickle(path, name):
        """Load pickle file with error handling."""
        try:
            with open(path, 'rb') as f:
                return pickle.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"{name} not found at {path}. Run train_improved_model.py first.")

    @staticmethod
    def normalize_token(text: str) -> str:
        text = str(text).strip().lower()
        text = text.replace("&", " and ")
        text = text.replace("/", " ")
        text = text.replace("-", " ")
        text = text.replace(",", " ")
        text = text.replace("(", " ").replace(")", " ")
        return "_".join(text.split())

    def normalize_symptom_name(self, symptom: str) -> str:
        normalized = self.normalize_token(symptom)
        return SYMPTOM_MAP.get(normalized, normalized)

    @staticmethod
    def build_confidence_warning(confidence: float) -> str | None:
        if confidence < CONFIDENCE_WARNING_THRESHOLD:
            return (
                "Warning: low-confidence prediction. Add more distinguishing symptoms "
                "and consider clinical review."
            )
        return None

    def engineer_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Apply the same feature engineering used during training.
        Creates the interaction features referenced by the saved feature set.
        """
        X_engineered = X.copy()

        interaction_features = [
            name for name in self.feature_names
            if "_and_" in name or "_x_" in name
        ]
        for feature_name in interaction_features:
            if "_and_" in feature_name:
                symptom1, symptom2 = feature_name.split("_and_", 1)
            else:
                symptom1, symptom2 = feature_name.split("_x_", 1)
            left = X_engineered.get(symptom1, pd.Series(0, index=X_engineered.index))
            right = X_engineered.get(symptom2, pd.Series(0, index=X_engineered.index))
            X_engineered[feature_name] = left * right

        return X_engineered
    
    def predict_disease(
        self, 
        symptoms: List[str], 
        all_symptoms: List[str]
    ) -> Tuple[str, float]:
        """
        Return the top-ranked disease from the primary Top-3 output.
        
        Args:
            symptoms: List of symptom names present
            all_symptoms: List of all available symptoms
            
        Returns:
            Tuple: (predicted_disease, confidence_score)
            - predicted_disease: str, name of predicted disease
            - confidence_score: float, probability between 0-1
        """
        if len(symptoms) < MIN_SYMPTOMS_REQUIRED:
            raise ValueError(
                f"At least {MIN_SYMPTOMS_REQUIRED} symptoms are required for prediction."
            )

        input_vector = pd.DataFrame(
            np.zeros((1, len(all_symptoms))),
            columns=[self.normalize_symptom_name(symptom) for symptom in all_symptoms]
        )
        input_vector = input_vector.groupby(level=0, axis=1).max()

        for symptom in symptoms:
            normalized_symptom = self.normalize_symptom_name(symptom)
            if normalized_symptom in input_vector.columns:
                input_vector[normalized_symptom] = CRITICAL_SYMPTOM_WEIGHTS.get(normalized_symptom, 1.0)

        input_engineered = self.engineer_features(input_vector)

        # Handle calibrated models
        if hasattr(self.model, "base_estimator"):
            base_model = self.model.base_estimator
        elif hasattr(self.model, "estimator"):
            base_model = self.model.estimator
        else:
            base_model = self.model

        # Align features
        if hasattr(base_model, "feature_names_in_"):
            input_final = input_engineered.reindex(
                columns=base_model.feature_names_in_,
                fill_value=0
            )
            # Optional debug
            print("Expected features:", len(base_model.feature_names_in_))
            print("Input features:", len(input_final.columns))
        else:
            input_final = input_engineered

        try:
            top_predictions = self.get_top_predictions(symptoms, all_symptoms, top_k=3)
            disease, confidence = top_predictions[0]
            return disease, float(confidence)
        except Exception as e:
            print("Prediction error:", str(e))
            raise ValueError("Prediction failed due to feature mismatch")
    
    def predict_disease_batch(
        self,
        symptoms_list: List[List[str]],
        all_symptoms: List[str]
    ) -> List[Tuple[str, float]]:
        """
        Predict diseases for multiple patients.
        
        Args:
            symptoms_list: List of symptom lists
            all_symptoms: All available symptoms
            
        Returns:
            List of (disease, confidence) tuples
        """
        results = []
        for symptoms in symptoms_list:
            disease, confidence = self.predict_disease(symptoms, all_symptoms)
            results.append((disease, confidence))
        return results
    
    def get_top_predictions(
        self,
        symptoms: List[str],
        all_symptoms: List[str],
        top_k: int = 3
    ) -> List[Tuple[str, float]]:
        """
        Get Top-K disease predictions sorted by probability descending.
        
        Args:
            symptoms: List of symptom names present
            all_symptoms: List of all available symptoms
            top_k: Number of top predictions to return
            
        Returns:
            List of (disease, probability) sorted by probability descending
        """
        # Create feature vector
        if len(symptoms) < MIN_SYMPTOMS_REQUIRED:
            raise ValueError(
                f"At least {MIN_SYMPTOMS_REQUIRED} symptoms are required for prediction."
            )

        input_vector = pd.DataFrame(
            np.zeros((1, len(all_symptoms))),
            columns=[self.normalize_symptom_name(symptom) for symptom in all_symptoms]
        )
        input_vector = input_vector.groupby(level=0, axis=1).max()
        
        for symptom in symptoms:
            normalized_symptom = self.normalize_symptom_name(symptom)
            if normalized_symptom in input_vector.columns:
                input_vector[normalized_symptom] = CRITICAL_SYMPTOM_WEIGHTS.get(normalized_symptom, 1.0)
        
        # Apply feature engineering
        input_engineered = self.engineer_features(input_vector)
        
        # Ensure all trained features
        for feature in self.feature_names:
            if feature not in input_engineered.columns:
                input_engineered[feature] = 0
        
        input_final = input_engineered[self.feature_names]
        
        # Get probabilities
        probabilities = self.model.predict_proba(input_final)[0]
        
        # Get top-K
        top_indices = np.argsort(probabilities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            disease = self.label_encoder.inverse_transform([idx])[0]
            prob = float(probabilities[idx])
            results.append((disease, prob))
        
        return results


# ============================================================================
# INTEGRATION EXAMPLES FOR app.py
# ============================================================================

def load_improved_model(model_dir='.'):
    """
    Load the improved model for use in Streamlit app.
    
    Usage in app.py:
        model_inference = load_improved_model()
        disease, confidence = model_inference.predict_disease(symptoms, all_symptoms)
    """
    return ImprovedModelInference(model_dir)


def predict_with_improved_model(symptoms: List[str], all_symptoms: List[str]) -> dict:
    """
    Standalone function for disease prediction using improved model.
    
    Returns:
        {
            'disease': str,
            'confidence': float,
            'alternatives': list of (disease, prob) tuples
        }
    """
    model = ImprovedModelInference()
    top_predictions = model.get_top_predictions(symptoms, all_symptoms, top_k=3)
    disease, confidence = top_predictions[0]
    warning = model.build_confidence_warning(confidence)
    
    return {
        'top_predictions': top_predictions,
        'disease': disease,
        'confidence': confidence,
        'alternatives': top_predictions[1:],
        'warning': warning
    }


if __name__ == '__main__':
    # Test the inference engine
    print("Loading improved model...")
    try:
        model_inference = ImprovedModelInference()
        print("✓ Model loaded successfully!")
        print(f"Model features: {len(model_inference.feature_names)}")
        print(f"Classes: {len(model_inference.label_encoder.classes_)} diseases")
        print(f"Sample classes: {model_inference.label_encoder.classes_[:5]}")
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        print("Make sure to run train_improved_model.py first.")
