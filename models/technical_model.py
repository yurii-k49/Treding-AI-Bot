# models/technical_model.py
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score
import logging
import joblib
import os

class TechnicalModel:
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger('models.technical')
        self.setup_model()
        
    def setup_model(self):
        """Initialize technical model"""
        try:
            self.model = RandomForestClassifier(
                n_estimators=200,
                max_depth=5,
                min_samples_split=20,
                min_samples_leaf=10,
                max_features='sqrt',
                random_state=42,
                class_weight='balanced'
            )
            self.logger.info("Technical model initialized")
        except Exception as e:
            self.logger.error(f"Error initializing technical model: {str(e)}")
            raise
            
    async def train(self, data):
        """Train the model"""
        try:
            if data is None:
                raise Exception("No training data provided")
                
            # Prepare training data
            X, y = self._prepare_training_data(data)
            
            if X is None or y is None or len(X) == 0:
                raise Exception("Failed to prepare training data")
                
            # Train model
            self.model.fit(X, y)
            self.logger.info("Technical model trained successfully")
            
            # Calculate training metrics
            y_pred = self.model.predict(X)
            accuracy = np.mean(y_pred == y)
            self.logger.info(f"Training accuracy: {accuracy:.4f}")
            
        except Exception as e:
            self.logger.error(f"Error training technical model: {str(e)}")
            raise
            
    def _prepare_training_data(self, df):
        """Prepare data for training"""
        try:
            if df is None:
                return None, None
                
            # Available features
            features = df.columns.tolist()
            features.remove('returns')  # Remove target variable
            
            X = df[features].values
            y = (df['returns'].shift(-1) > 0).astype(int)  # Next day direction
            
            # Remove last row (no future return)
            X = X[:-1]
            y = y[:-1]
            
            # Remove any remaining NaN
            mask = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
            X = X[mask]
            y = y[mask]
            
            return X, y
            
        except Exception as e:
            self.logger.error(f"Error preparing training data: {str(e)}")
            return None, None
            
    async def validate(self, validation_data):
        """Validate model performance"""
        try:
            if validation_data is None:
                return None
                
            # Prepare validation data
            X, y = self._prepare_training_data(validation_data)
            
            if X is None or y is None or len(X) == 0:
                self.logger.error("No validation data after preparation")
                return None
                
            # Get predictions
            y_pred = self.model.predict(X)
            
            # Calculate metrics
            metrics = {
                'accuracy': np.mean(y_pred == y),
                'precision': precision_score(y, y_pred, zero_division=0),
                'recall': recall_score(y, y_pred, zero_division=0),
                'f1': f1_score(y, y_pred, zero_division=0)
            }
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error validating technical model: {str(e)}")
            return None
            
    def save_model(self):
        """Save model to disk"""
        try:
            os.makedirs('data/models', exist_ok=True)
            joblib.dump(self.model, 'data/models/technical_model.joblib')
            self.logger.info("Technical model saved successfully")
        except Exception as e:
            self.logger.error(f"Error saving model: {str(e)}")