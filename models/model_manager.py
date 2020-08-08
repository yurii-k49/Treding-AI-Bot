# models/model_manager.py
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import logging
import joblib
import os
from datetime import datetime

from .technical_model import TechnicalModel
from .fundamental_model import FundamentalModel
from .sentiment_model import SentimentModel

class ModelManager:
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger('models.manager')
        
        # Modellarni yaratish
        self.technical_model = TechnicalModel(config)
        self.fundamental_model = FundamentalModel(config)
        self.sentiment_model = SentimentModel(config)
        self.market_model = self._create_market_model()
        
        self.scaler = StandardScaler()
        
    def _create_market_model(self):
        """Market model yaratish"""
        return GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )
        
    async def train_all_models(self, data):
        """Train all models"""
        try:
            if data["historical"] is None or data["historical"].empty:
                raise ValueError("Historical data is not available")
                
            results = {}
            
            # Train technical model
            self.logger.info("Training technical model...")
            await self.technical_model.train(data["historical"])
            tech_metrics = await self.technical_model.validate(data["historical"])
            if tech_metrics:
                results["technical"] = tech_metrics
                self.logger.info("Technical Model Training Metrics:")
                for metric, value in tech_metrics.items():
                    self.logger.info(f"- {metric}: {value:.4f}")
            
            # Train market model
            if data.get("market") is not None and not data["market"].empty:
                self.logger.info("Training market model...")
                try:
                    X = data["market"].values[:-1]
                    y = data["historical"]["returns"].values[1:]
                    
                    self.market_model.fit(X, y)
                    predictions = self.market_model.predict(X)
                    
                    market_metrics = {
                        "r2_score": self.market_model.score(X, y),
                        "rmse": float(np.sqrt(mean_squared_error(y, predictions))),
                        "mae": float(mean_absolute_error(y, predictions))
                    }
                    
                    results["market"] = market_metrics
                    self.logger.info("Market Model Training Metrics:")
                    for metric, value in market_metrics.items():
                        self.logger.info(f"- {metric}: {value:.4f}")
                    
                except Exception as e:
                    self.logger.error(f"Error training market model: {str(e)}")
            
            # Save models
            self.save_models()
            
            self.logger.info("All models trained successfully")
            return results
            
        except Exception as e:
            self.logger.error(f"Error training models: {str(e)}")
            raise
    
    async def validate_models(self, validation_data):
        """Model validation"""
        try:
            if not validation_data:
                raise ValueError("No validation data provided")
                
            results = {}
            
            # Technical model validation
            if validation_data.get("historical") is not None:
                self.logger.info("Validating technical model...")
                tech_metrics = await self.technical_model.validate(validation_data["historical"])
                if tech_metrics:
                    results["technical"] = tech_metrics
                    self.logger.info("Technical Model Validation Metrics:")
                    for metric, value in tech_metrics.items():
                        self.logger.info(f"- {metric}: {value:.4f}")
            
            # Market model validation
            if validation_data.get("market") is not None and not validation_data["market"].empty:
                self.logger.info("Validating market model...")
                try:
                    X_val = validation_data["market"].values[:-1]
                    y_val = validation_data["historical"]["returns"].values[1:]
                    
                    predictions = self.market_model.predict(X_val)
                    
                    market_metrics = {
                        "r2_score": float(self.market_model.score(X_val, y_val)),
                        "rmse": float(np.sqrt(mean_squared_error(y_val, predictions))),
                        "mae": float(mean_absolute_error(y_val, predictions))
                    }
                    
                    results["market"] = market_metrics
                    self.logger.info("Market Model Validation Metrics:")
                    for metric, value in market_metrics.items():
                        self.logger.info(f"- {metric}: {value:.4f}")
                    
                except Exception as e:
                    self.logger.error(f"Error validating market model: {str(e)}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error in model validation: {str(e)}")
            raise
            
    def save_models(self):
        """Save all models"""
        try:
            models_dir = 'data/models'
            os.makedirs(models_dir, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Technical model
            self.technical_model.save_model()
            
            # Market model
            if hasattr(self, 'market_model'):
                joblib.dump(self.market_model, 
                          f'{models_dir}/market_model_{timestamp}.joblib')
            
            # Fundamental model
            if self.config.USE_FUNDAMENTAL_ANALYSIS:
                self.fundamental_model.save_model()
            
            # Sentiment model
            self.sentiment_model.save_model()
            
            # Scaler
            joblib.dump(self.scaler, f'{models_dir}/scaler_{timestamp}.joblib')
            
            self.logger.info("All models saved successfully")
            
        except Exception as e:
            self.logger.error(f"Error saving models: {str(e)}")


    def _calculate_market_metrics(self, predictions, validation_data):
        """Calculate detailed market model metrics"""
        try:
            actual = validation_data["historical"]["close"].pct_change().fillna(0)
            rmse = np.sqrt(np.mean((predictions - actual) ** 2))
            mae = np.mean(np.abs(predictions - actual))
            
            # Direction accuracy
            direction_accuracy = np.mean((predictions > 0) == (actual > 0))
            
            return {
                "rmse": rmse,
                "mae": mae,
                "direction_accuracy": direction_accuracy
            }
        except Exception as e:
            self.logger.error(f"Error calculating market metrics: {str(e)}")
            return {}