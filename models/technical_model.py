# models/technical_model.py
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import logging
import joblib
import os

class TechnicalModel:
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger('models.technical')
        self.setup_model()
        
    def setup_model(self):
        """Initialize the model based on configuration"""
        try:
            if self.config.DEEP_LEARNING:
                self.model = self._create_deep_model()
                self.logger.info("Deep learning model initialized")
            else:
                self.model = self._create_rf_model()
                self.logger.info("Random Forest model initialized")
                
        except Exception as e:
            self.logger.error(f"Error initializing technical model: {str(e)}")
            raise
            
    def _create_deep_model(self):
        """Create deep learning model"""
        model = Sequential([
            LSTM(100, input_shape=(100, self._get_n_features()), return_sequences=True),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dense(25, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
        
    def _create_rf_model(self):
        """Create random forest model"""
        return RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=10,
            random_state=42
        )
        
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
            if self.config.DEEP_LEARNING:
                self.model.fit(
                    X, y,
                    epochs=10,
                    batch_size=32,
                    validation_split=0.2,
                    verbose=1
                )
            else:
                self.model.fit(X, y)
                
            self.logger.info("Technical model trained successfully")
            
        except Exception as e:
            self.logger.error(f"Error training technical model: {str(e)}")
            raise
            
    def _prepare_training_data(self, df):
        """Prepare data for training"""
        try:
            if df is None:
                return None, None
                
            # Prepare features
            features = [
                'returns', 'log_returns',
                'ma_9', 'ma_20', 'ma_50', 'ma_200',
                'daily_range', 'body_size',
                'volatility_5', 'volatility_10', 'volatility_20',
                'volume_ma', 'volume_std'
            ]
            
            X = df[features].values
            
            # Prepare targets (next day return direction)
            future_returns = df['returns'].shift(-1)
            y = (future_returns > 0).astype(int)
            
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
            if self.config.DEEP_LEARNING:
                predictions = self.model.predict(X)
            else:
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
            
            if self.config.DEEP_LEARNING:
                self.model.save('data/models/technical_model')
            else:
                joblib.dump(self.model, 'data/models/technical_model.joblib')
                
            self.logger.info("Technical model saved successfully")
            
        except Exception as e:
            self.logger.error(f"Error saving model: {str(e)}")
            
    def _get_n_features(self):
        """Get number of input features"""
        return 13  # Number of technical features

    async def validate(self, validation_data):
        """Validate model performance"""
        try:
            if validation_data is None:
                return None
                
            # Prepare validation data
            X, y = self._prepare_training_data(validation_data)
            
            if X is None or y is None:
                return None
                
            # Get predictions
            if self.config.DEEP_LEARNING:
                predictions = self.model.predict(X)
            else:
                predictions = self.model.predict_proba(X)[:, 1]
                
            # Calculate accuracy
            y_pred = (predictions > 0.5).astype(int)
            accuracy = (y_pred == y).mean()
            
            return accuracy
            
        except Exception as e:
            self.logger.error(f"Error validating model: {str(e)}")
            return None