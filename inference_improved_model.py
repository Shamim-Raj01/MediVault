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
    
    def engineer_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Apply the same feature engineering used during training.
        Creates interaction features from top symptoms.
        """
        X_engineered = X.copy()
        
        # Get original feature count
        n_original = X.shape[1]
        
        # Get top symptoms by variance (same as training)
        top_symptoms = X.var().nlargest(10).index.tolist()
        
        # Create interaction features (limited)
        for i in range(len(top_symptoms)):
            for j in range(i + 1, min(i + 3, len(top_symptoms))):
                symptom1 = top_symptoms[i]
                symptom2 = top_symptoms[j]
                feature_name = f"{symptom1}_and_{symptom2}"
                X_engineered[feature_name] = (X[symptom1] * X[symptom2]).astype(int)
        
        return X_engineered
    
    def predict_disease(
        self, 
        symptoms: List[str], 
        all_symptoms: List[str]
    ) -> Tuple[str, float]:
        """
        Predict disease from list of symptoms.
        
        Args:
            symptoms: List of symptom names present
            all_symptoms: List of all available symptoms
            
        Returns:
            Tuple: (predicted_disease, confidence_score)
            - predicted_disease: str, name of predicted disease
            - confidence_score: float, probability between 0-1
        """
        input_vector = pd.DataFrame(
            np.zeros((1, len(all_symptoms))),
            columns=all_symptoms
        )

        for symptom in symptoms:
            if symptom in all_symptoms:
                input_vector[symptom] = 1

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

        # Make prediction with error handling
        try:
            prediction_encoded = self.model.predict(input_final)[0]
            disease = self.label_encoder.inverse_transform([prediction_encoded])[0]
            
            probabilities = self.model.predict_proba(input_final)[0]
            confidence = np.max(probabilities)
            
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
        top_k: int = 5
    ) -> List[Tuple[str, float]]:
        """
        Get top-K disease predictions with probabilities.
        Useful for showing alternative diagnoses.
        
        Args:
            symptoms: List of symptom names present
            all_symptoms: List of all available symptoms
            top_k: Number of top predictions to return
            
        Returns:
            List of (disease, probability) sorted by probability descending
        """
        # Create feature vector
        input_vector = pd.DataFrame(
            np.zeros((1, len(all_symptoms))),
            columns=all_symptoms
        )
        
        for symptom in symptoms:
            if symptom in all_symptoms:
                input_vector[symptom] = 1
        
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
    disease, confidence = model.predict_disease(symptoms, all_symptoms)
    alternatives = model.get_top_predictions(symptoms, all_symptoms, top_k=3)
    
    return {
        'disease': disease,
        'confidence': confidence,
        'alternatives': alternatives
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
