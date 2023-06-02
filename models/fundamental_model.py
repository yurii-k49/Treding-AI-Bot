# models/fundamental_model.py
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
import joblib
import logging

class FundamentalModel:
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger('model.fundamental')
        self.scaler = StandardScaler()
        self.setup_models()

    def setup_models(self):
        """Initialize all fundamental analysis models"""
        try:
            if self.config.DEEP_LEARNING:
                self.macro_model = self._create_deep_macro_model()
                self.company_model = self._create_deep_company_model()
                self.sector_model = self._create_deep_sector_model()
            else:
                self.macro_model = GradientBoostingRegressor(
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=5
                )
                self.company_model = RandomForestRegressor(
                    n_estimators=100,
                    max_depth=10
                )
                self.sector_model = GradientBoostingRegressor(
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=5
                )

            self.logger.info("Fundamental models initialized successfully")
        except Exception as e:
            self.logger.error(f"Error initializing models: {str(e)}")
            raise

    def _create_deep_macro_model(self):
        """Create deep learning model for macroeconomic analysis"""
        model = Sequential([
            LSTM(64, input_shape=(None, 20), return_sequences=True),
            Dropout(0.2),
            LSTM(32, return_sequences=False),
            Dense(16, activation='relu'),
            Dense(1, activation='linear')
        ])
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        return model

    def _create_deep_company_model(self):
        """Create deep learning model for company analysis"""
        model = Sequential([
            Dense(64, input_dim=20, activation='relu'),
            Dropout(0.3),
            Dense(32, activation='relu'),
            Dense(16, activation='relu'),
            Dense(1, activation='linear')
        ])
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        return model

    def _create_deep_sector_model(self):
        """Create deep learning model for sector analysis"""
        model = Sequential([
            Dense(32, input_dim=15, activation='relu'),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(1, activation='linear')
        ])
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        return model

    async def train(self, data):
        """Train all fundamental models"""
        try:
            # Prepare data
            macro_data = self._prepare_macro_data(data.get('macro', {}))
            company_data = self._prepare_company_data(data.get('company', {}))
            sector_data = self._prepare_sector_data(data.get('sector', {}))

            # Train models
            if self.config.DEEP_LEARNING:
                await self._train_deep_models(macro_data, company_data, sector_data)
            else:
                await self._train_traditional_models(macro_data, company_data, sector_data)

            self.logger.info("Fundamental models trained successfully")
        except Exception as e:
            self.logger.error(f"Error training models: {str(e)}")
            raise

    async def predict(self, data):
        """Make predictions using all fundamental models"""
        try:
            # Prepare prediction data
            macro_features = self._prepare_macro_data(data.get('macro', {}))
            company_features = self._prepare_company_data(data.get('company', {}))
            sector_features = self._prepare_sector_data(data.get('sector', {}))

            # Get predictions
            macro_pred = self.macro_model.predict(macro_features)
            company_pred = self.company_model.predict(company_features)
            sector_pred = self.sector_model.predict(sector_features)

            # Combine predictions
            combined_score = self._combine_predictions(
                macro_pred, company_pred, sector_pred
            )

            return {
                'combined_score': combined_score,
                'macro_score': macro_pred[0],
                'company_score': company_pred[0],
                'sector_score': sector_pred[0]
            }
        except Exception as e:
            self.logger.error(f"Error making predictions: {str(e)}")
            return None

    def save_models(self):
        """Save all fundamental models"""
        try:
            model_dir = 'data/models/fundamental'
            os.makedirs(model_dir, exist_ok=True)

            if self.config.DEEP_LEARNING:
                self.macro_model.save(f"{model_dir}/macro_model")
                self.company_model.save(f"{model_dir}/company_model")
                self.sector_model.save(f"{model_dir}/sector_model")
            else:
                joblib.dump(self.macro_model, f"{model_dir}/macro_model.joblib")
                joblib.dump(self.company_model, f"{model_dir}/company_model.joblib")
                joblib.dump(self.sector_model, f"{model_dir}/sector_model.joblib")

            self.logger.info("Fundamental models saved successfully")
        except Exception as e:
            self.logger.error(f"Error saving models: {str(e)}")
            raise