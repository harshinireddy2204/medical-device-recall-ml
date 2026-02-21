"""
ML Module for Recall Likelihood Prediction
Trains classification models to predict if a device will have recalls
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score
import joblib
import warnings
warnings.filterwarnings('ignore')


class RecallPredictor:
    """Machine Learning model to predict recall likelihood for medical devices"""
    
    def __init__(self):
        self.model = None
        self.label_encoders = {}
        self.feature_columns = None
        self.is_trained = False
        
    def prepare_features(self, df):
        """
        Prepare features for ML model
        
        Features:
        - rpss: Risk score
        - device_class: Device classification (1, 2, 3)
        - root_cause_description: Encoded root cause
        - total_adverse_events: Number of adverse events
        - unique_manufacturers: Number of manufacturers (accepts column named unique_manufacturers or unique-manufacturers)
        """
        df_clean = df.copy()
        
        # Normalize manufacturer column name (SQL uses underscore, CSV may use hyphen)
        if 'unique_manufacturers' in df_clean.columns and 'unique-manufacturers' not in df_clean.columns:
            df_clean['unique-manufacturers'] = df_clean['unique_manufacturers']
        elif 'unique-manufacturers' not in df_clean.columns:
            df_clean['unique-manufacturers'] = 0
        
        # Handle missing values
        df_clean['device_class'] = df_clean['device_class'].fillna(0)
        df_clean['total_adverse_events'] = df_clean['total_adverse_events'].fillna(0)
        df_clean['unique-manufacturers'] = df_clean['unique-manufacturers'].fillna(0)
        df_clean['rpss'] = df_clean['rpss'].fillna(0)
        
        # Encode root_cause_description
        if 'root_cause_description' not in self.label_encoders:
            self.label_encoders['root_cause_description'] = LabelEncoder()
            df_clean['root_cause_encoded'] = self.label_encoders['root_cause_description'].fit_transform(
                df_clean['root_cause_description'].fillna('Other').astype(str)
            )
        else:
            # Handle unseen categories
            known_categories = set(self.label_encoders['root_cause_description'].classes_)
            df_clean['root_cause_description'] = df_clean['root_cause_description'].fillna('Other').astype(str)
            df_clean['root_cause_description'] = df_clean['root_cause_description'].apply(
                lambda x: x if x in known_categories else 'Other'
            )
            df_clean['root_cause_encoded'] = self.label_encoders['root_cause_description'].transform(
                df_clean['root_cause_description']
            )
        
        # Create binary target: will have recall (recall_count > 0)
        df_clean['has_recall'] = (df_clean['recall_count'].fillna(0) > 0).astype(int)
        
        # Select features
        features = [
            'rpss',
            'device_class',
            'root_cause_encoded',
            'total_adverse_events',
            'unique-manufacturers'
        ]
        
        # Log transform for skewed features
        df_clean['log_adverse_events'] = np.log1p(df_clean['total_adverse_events'])
        df_clean['log_manufacturers'] = np.log1p(df_clean['unique-manufacturers'])
        
        features_extended = [
            'rpss',
            'device_class',
            'root_cause_encoded',
            'log_adverse_events',
            'log_manufacturers'
        ]
        
        return df_clean[features_extended], df_clean['has_recall']
    
    def train(self, df, test_size=0.2, model_type='random_forest'):
        """
        Train the recall prediction model
        
        Parameters:
        -----------
        df : DataFrame
            Input data with device information
        test_size : float
            Proportion of data for testing
        model_type : str
            'random_forest' or 'gradient_boosting'
        """
        print("Preparing features...")
        X, y = self.prepare_features(df)
        
        # Remove rows with invalid values
        valid_mask = ~(X.isnull().any(axis=1) | np.isinf(X).any(axis=1))
        X = X[valid_mask]
        y = y[valid_mask]
        
        if len(X) == 0:
            raise ValueError("No valid data after cleaning")
        
        self.feature_columns = X.columns.tolist()
        
        print(f"Training on {len(X)} samples...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Select model
        if model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
        elif model_type == 'gradient_boosting':
            self.model = GradientBoostingClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42
            )
        else:
            raise ValueError(f"Unknown model_type: {model_type}")
        
        print(f"Training {model_type}...")
        self.model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        accuracy = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)
        
        print(f"\nModel Performance:")
        print(f"Accuracy: {accuracy:.3f}")
        print(f"AUC-ROC: {auc:.3f}")
        print(f"\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['No Recall', 'Has Recall']))
        
        self.is_trained = True
        return {
            'accuracy': accuracy,
            'auc': auc,
            'confusion_matrix': confusion_matrix(y_test, y_pred)
        }
    
    def predict(self, df):
        """
        Predict recall likelihood for devices
        
        Returns:
        --------
        DataFrame with predictions and probabilities
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        X, _ = self.prepare_features(df)
        
        # Remove invalid rows
        valid_mask = ~(X.isnull().any(axis=1) | np.isinf(X).any(axis=1))
        predictions = pd.Series([0] * len(df), index=df.index)
        probabilities = pd.Series([0.0] * len(df), index=df.index)
        
        if valid_mask.sum() > 0:
            X_valid = X[valid_mask]
            pred_valid = self.model.predict(X_valid)
            proba_valid = self.model.predict_proba(X_valid)[:, 1]
            
            predictions[valid_mask] = pred_valid
            probabilities[valid_mask] = proba_valid
        
        result = df.copy()
        result['predicted_recall'] = predictions
        result['recall_probability'] = probabilities
        
        return result
    
    def get_feature_importance(self):
        """Get feature importance from trained model"""
        if not self.is_trained:
            raise ValueError("Model must be trained first")
        
        importance = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return importance
    
    def save_model(self, filepath):
        """Save trained model and encoders"""
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        model_data = {
            'model': self.model,
            'label_encoders': self.label_encoders,
            'feature_columns': self.feature_columns
        }
        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load trained model and encoders"""
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.label_encoders = model_data['label_encoders']
        self.feature_columns = model_data['feature_columns']
        self.is_trained = True
        print(f"Model loaded from {filepath}")


if __name__ == "__main__":
    # Example usage
    print("Loading data...")
    df = pd.read_csv("../visualization/device_rpss_sample.csv")
    
    print("Initializing predictor...")
    predictor = RecallPredictor()
    
    print("Training model...")
    metrics = predictor.train(df, model_type='random_forest')
    
    print("\nFeature Importance:")
    print(predictor.get_feature_importance())
    
    print("\nMaking predictions on sample...")
    predictions = predictor.predict(df.head(100))
    print(predictions[['PMA_PMN_NUM', 'recall_count', 'predicted_recall', 'recall_probability']].head(10))
