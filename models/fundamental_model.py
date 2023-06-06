# models/fundamental_model.py
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import logging
import joblib
import os

class FundamentalModel:
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger('models.fundamental')
        self.setup_models()
        
    def setup_models(self):
        """Initialize models for different aspects"""
        try:
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=10,
                random_state=42
            )
            self.logger.info("Fundamental model initialized")
        except Exception as e:
            self.logger.error(f"Error initializing fundamental model: {str(e)}")
            raise
            
    async def train(self, data):
        """Train the model"""
        try:
            if data is None:
                raise Exception("No training data provided")
                
            # Prepare training data
            X, y = self._prepare_training_data(data)
            
            if X is None or y is None:
                raise Exception("Failed to prepare training data")
                
            # Train model
            self.model.fit(X, y)
            self.logger.info("Fundamental model trained successfully")
            
        except Exception as e:
            self.logger.error(f"Error training fundamental model: {str(e)}")
            raise
            
    def _prepare_training_data(self, data):
        """Prepare data for training"""
        try:
            if data is None:
                return None, None
                
            # Process economic indicators
            economic_data = self._prepare_macro_data(data)
            if economic_data is None:
                return None, None
                
            # Create features
            X = economic_data.values
            
            # Create targets (using simple strategy - if next day's value increases)
            y = (economic_data.diff().shift(-1).iloc[:, 0] > 0).astype(int)
            
            # Remove last row (no future value)
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
            
    def _prepare_macro_data(self, data):
        """Prepare macroeconomic data"""
        try:
            if data is None or 'economic' not in data:
                return None
                
            # Convert economic indicators to DataFrame
            df_list = []
            for indicator, values in data['economic'].items():
                temp_df = pd.DataFrame(values)
                temp_df.set_index('date', inplace=True)
                temp_df.columns = [indicator]
                df_list.append(temp_df)
                
            if not df_list:
                return None
                
            # Combine all indicators
            return pd.concat(df_list, axis=1)
            
        except Exception as e:
            self.logger.error(f"Error preparing macro data: {str(e)}")
            return None
            
    async def predict(self, data):
        """Make predictions"""
        try:
            if data is None:
                return None
                
            # Prepare features
            X = self._prepare_prediction_data(data)
            
            if X is None:
                return None
                
            # Make prediction
            predictions = self.model.predict_proba(X)[:, 1]
            
            return {
                'signal': predictions[-1],  # Latest prediction
                'confidence': np.abs(predictions[-1] - 0.5) * 2  # Convert to 0-1 scale
            }
            
        except Exception as e:
            self.logger.error(f"Error making prediction: {str(e)}")
            return None
            
    def _prepare_prediction_data(self, df):
        """Prepare data for prediction"""
        try:
            return self._prepare_training_data(df)[0]
        except Exception as e:
            self.logger.error(f"Error preparing prediction data: {str(e)}")
            return None
            
    def save_model(self):
        """Save model to disk"""
        try:
            # Create directory if it doesn't exist
            os.makedirs('data/models', exist_ok=True)
            
            # Save model
            joblib.dump(self.model, 'data/models/fundamental_model.joblib')
            self.logger.info("Fundamental model saved successfully")
            
        except Exception as e:
            self.logger.error(f"Error saving model: {str(e)}")