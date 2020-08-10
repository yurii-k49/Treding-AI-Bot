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
            self.logger.info("Starting technical model training...")
            
            if data is None:
                raise Exception("No training data provided")
                
            # Prepare training data
            X, y = self._prepare_training_data(data)
            
            if X is None or y is None:
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
            if df is None or df.empty:
                return None, None
                
            # Calculate technical indicators
            df['returns'] = df['close'].pct_change()
            df['MA5'] = df['close'].rolling(window=5).mean()
            df['MA20'] = df['close'].rolling(window=20).mean()
            df['RSI'] = self._calculate_rsi(df['close'])
            
            # Feature columns
            feature_columns = [
                'returns',
                'MA5',
                'MA20',
                'RSI',
                'tick_volume'
            ]
            
            # Create features and target
            X = df[feature_columns].values
            y = (df['returns'].shift(-1) > 0).astype(int)
            
            # Remove last row (no future return)
            X = X[:-1]
            y = y[:-1]
            
            # Remove NaN
            mask = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
            X = X[mask]
            y = y[mask]
            
            return X, y
            
        except Exception as e:
            self.logger.error(f"Error preparing training data: {str(e)}")
            return None, None
            
    def _calculate_rsi(self, prices, period=14):
        """Calculate RSI technical indicator"""
        try:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi.fillna(50)  # Fill NaN with neutral RSI value
        except Exception as e:
            self.logger.error(f"Error calculating RSI: {str(e)}")
            return pd.Series(50, index=prices.index)  # Return neutral RSI on error
            
    def _prepare_prediction_data(self, df):
        """Prepare data for prediction"""
        try:
            # Calculate same features as in training
            df_copy = df.copy()
            df_copy['returns'] = df_copy['close'].pct_change()
            df_copy['MA5'] = df_copy['close'].rolling(window=5).mean()
            df_copy['MA20'] = df_copy['close'].rolling(window=20).mean()
            df_copy['RSI'] = self._calculate_rsi(df_copy['close'])
            
            feature_columns = [
                'returns',
                'MA5',
                'MA20',
                'RSI',
                'tick_volume'
            ]
            
            X = df_copy[feature_columns].values
            mask = ~np.isnan(X).any(axis=1)
            return X[mask]
            
        except Exception as e:
            self.logger.error(f"Error preparing prediction data: {str(e)}")
            return None
            
    async def predict(self, data):
        """Make predictions with confidence scores"""
        try:
            # First train the model if not already trained
            if not hasattr(self.model, 'classes_'):
                await self.train(data)

            # Prepare prediction data
            X = self._prepare_prediction_data(data)
            if X is None:
                raise Exception("Failed to prepare prediction data")
                
            # Get probability predictions
            probabilities = self.model.predict_proba(X)
            
            # Convert to signals (-1 to 1) and confidence scores
            signals = (probabilities[:, 1] - 0.5) * 2  # Scale to [-1, 1]
            confidence = np.abs(probabilities[:, 1] - 0.5) * 2  # Scale to [0, 1]
            
            return {
                'signal': signals,
                'confidence': confidence
            }
            
        except Exception as e:
            self.logger.error(f"Error making prediction: {str(e)}")
            raise
            
    async def validate(self, validation_data):
        """Validate model performance"""
        try:
            if validation_data is None:
                return None
                
            X, y = self._prepare_training_data(validation_data)
            
            if X is None or y is None:
                return None
                
            y_pred = self.model.predict(X)
            
            return {
                'accuracy': np.mean(y_pred == y),
                'precision': precision_score(y, y_pred, zero_division=0),
                'recall': recall_score(y, y_pred, zero_division=0),
                'f1': f1_score(y, y_pred, zero_division=0)
            }
            
        except Exception as e:
            self.logger.error(f"Error validating model: {str(e)}")
            return None
        
    def save_model(self):
        """Save model to disk"""
        try:
            # Create directory if it doesn't exist
            models_dir = 'data/models'
            os.makedirs(models_dir, exist_ok=True)
            
            # Save model
            model_path = f'{models_dir}/technical_model.joblib'
            joblib.dump(self.model, model_path)
            
            # Save scaler if exists
            if hasattr(self, 'scaler'):
                scaler_path = f'{models_dir}/technical_scaler.joblib'
                joblib.dump(self.scaler, scaler_path)
            
            self.logger.info("Technical model saved successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving technical model: {str(e)}")
            return False