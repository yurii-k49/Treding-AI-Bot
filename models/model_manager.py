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
from tqdm import tqdm
import asyncio

from .technical_model import TechnicalModel
from .fundamental_model import FundamentalModel
from .sentiment_model import SentimentModel
from .neural_model import NeuralModel

class ModelManager:
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger('models.manager')
        
        # Initialize models
        self.technical_model = TechnicalModel(config)
        self.fundamental_model = FundamentalModel(config)
        self.sentiment_model = SentimentModel(config)
        self.neural_model = NeuralModel(config)
        
        # Initialize market model
        self.market_model = self._create_market_model()
        self.scaler = StandardScaler()
        
    def _create_market_model(self):
        """Create market model with optimized parameters"""
        return GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42
        )

    async def train_all_models(self, data):
        """Train all models with detailed progress tracking"""
        try:
            if data["historical"] is None or data["historical"].empty:
                raise ValueError("Historical data is not available")
                
            results = {}
            total_steps = 4  # Neural, Market, Technical models + Saving
            
            with tqdm(total=total_steps, desc="Overall Progress") as pbar:
                # Neural Model Training
                self.logger.info("\n=== Training Neural Network Model ===")
                pbar.set_description("Training Neural Network")
                neural_success = await self.neural_model.train(data["historical"])
                if neural_success:
                    self.logger.info("✓ Neural model trained successfully")
                pbar.update(1)
                
                # Market Model Training
                if data.get("market") is not None and not data["market"].empty:
                    self.logger.info("\n=== Training Market Model ===")
                    pbar.set_description("Training Market Model")
                    try:
                        self.logger.info("1/5: Preparing market data...")
                        market_data = data["market"].ffill().bfill()
                        historical_data = data["historical"].ffill().bfill()
                        
                        if not market_data.empty and not historical_data.empty:
                            self.logger.info("2/5: Calculating returns...")
                            historical_data['returns'] = historical_data['close'].pct_change()
                            historical_data['returns'] = historical_data['returns'].fillna(0)
                            
                            self.logger.info("3/5: Aligning data...")
                            common_index = market_data.index.intersection(historical_data.index)
                            X = market_data.loc[common_index].values
                            y = historical_data.loc[common_index, 'returns'].values
                            
                            # Remove NaN rows
                            mask = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
                            X = X[mask]
                            y = y[mask]
                            
                            if len(X) > 0 and len(y) > 0:
                                self.logger.info("4/5: Scaling features...")
                                X = self.scaler.fit_transform(X)
                                
                                self.logger.info("5/5: Training market model...")
                                self.market_model.fit(X, y)
                                predictions = self.market_model.predict(X)
                                
                                market_metrics = {
                                    "r2_score": self.market_model.score(X, y),
                                    "rmse": float(np.sqrt(mean_squared_error(y, predictions))),
                                    "mae": float(mean_absolute_error(y, predictions))
                                }
                                
                                results["market"] = market_metrics
                                self.logger.info("\nMarket Model Training Metrics:")
                                for metric, value in market_metrics.items():
                                    self.logger.info(f"- {metric}: {value:.4f}")
                                self.logger.info("✓ Market model training completed")
                            else:
                                self.logger.warning("⚠ No valid samples after data preparation")
                        
                    except Exception as e:
                        self.logger.error(f"✗ Error training market model: {str(e)}")
                pbar.update(1)
                
                # Technical Model Training
                self.logger.info("\n=== Training Technical Model ===")
                pbar.set_description("Training Technical Model")
                await self.technical_model.train(data["historical"])
                tech_metrics = await self.technical_model.validate(data["historical"])
                if tech_metrics:
                    results["technical"] = tech_metrics
                    self.logger.info("\nTechnical Model Metrics:")
                    for metric, value in tech_metrics.items():
                        self.logger.info(f"- {metric}: {value:.4f}")
                    self.logger.info("✓ Technical model training completed")
                pbar.update(1)
                
                # Save Models
                self.logger.info("\n=== Saving Models ===")
                pbar.set_description("Saving Models")
                try:
                    self.save_models()
                    self.logger.info("✓ All models saved successfully")
                except Exception as e:
                    self.logger.error(f"✗ Error saving models: {str(e)}")
                pbar.update(1)
                
                self.logger.info("\n=== Training Complete ===")
                self.logger.info(f"Total models trained: {len(results)}")
                self.logger.info("Models ready for use")
                
            return results
            
        except Exception as e:
            self.logger.error(f"✗ Error training models: {str(e)}")
            raise

    def save_models(self):
        """Save all models"""
        try:
            models_dir = 'data/models'
            os.makedirs(models_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Save market model
            joblib.dump(self.market_model, f'{models_dir}/market_model_{timestamp}.joblib')
            joblib.dump(self.scaler, f'{models_dir}/scaler_{timestamp}.joblib')
            
            # Save technical model
            if hasattr(self.technical_model, 'save_model'):
                self.technical_model.save_model()
                
            # Save neural model if exists
            if hasattr(self.neural_model, 'save_model'):
                asyncio.create_task(self.neural_model.save_model())
            
            self.logger.info("All models saved successfully")
            
        except Exception as e:
            self.logger.error(f"Error saving models: {str(e)}")

    async def load_models(self):
        """Load all saved models"""
        try:
            models_dir = 'data/models'
            os.makedirs(models_dir, exist_ok=True)
            
            # Load latest market model and scaler
            market_models = sorted([f for f in os.listdir(models_dir) if f.startswith('market_model_')])
            if market_models:
                latest_model = market_models[-1]
                self.market_model = joblib.load(f'{models_dir}/{latest_model}')
                
                # Load corresponding scaler
                scaler_file = latest_model.replace('market_model_', 'scaler_')
                if os.path.exists(f'{models_dir}/{scaler_file}'):
                    self.scaler = joblib.load(f'{models_dir}/{scaler_file}')
            
            # Load technical model
            if hasattr(self.technical_model, 'load_model'):
                tech_model_path = f'{models_dir}/technical_model.joblib'
                if os.path.exists(tech_model_path):
                    self.technical_model = joblib.load(tech_model_path)
            
            # Load neural model
            if hasattr(self.neural_model, 'load_model'):
                await self.neural_model.load_model()
            
            self.logger.info("All models loaded successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading models: {str(e)}")
            return False
            
    async def validate_models(self, validation_data):
        """Validate all models with enhanced data handling"""
        try:
            results = {}
            
            # Validate market model
            if validation_data.get("market") is not None and not validation_data["market"].empty:
                try:
                    self.logger.info("\n=== Validating Market Model ===")
                    self.logger.info("1/5: Processing market data...")
                    
                    # Check data availability
                    market_data = validation_data["market"]
                    historical_data = validation_data["historical"]
                    
                    self.logger.info(f"Initial data shapes:")
                    self.logger.info(f"- Market data: {market_data.shape}")
                    self.logger.info(f"- Historical data: {historical_data.shape}")
                    
                    # Ensure all values are numeric
                    self.logger.info("2/5: Converting to numeric...")
                    market_data = market_data.apply(pd.to_numeric, errors='coerce')
                    historical_data['close'] = pd.to_numeric(historical_data['close'], errors='coerce')
                    
                    # Handle NaN values
                    self.logger.info("3/5: Handling missing values...")
                    market_data = market_data.ffill().bfill()
                    historical_data = historical_data.ffill().bfill()
                    
                    if not market_data.empty and not historical_data.empty:
                        # Calculate returns safely
                        self.logger.info("4/5: Calculating returns...")
                        historical_data['returns'] = historical_data['close'].pct_change()
                        historical_data['returns'] = historical_data['returns'].fillna(0)
                        
                        # Align data
                        common_index = market_data.index.intersection(historical_data.index)
                        self.logger.info(f"Common dates found: {len(common_index)}")
                        
                        if len(common_index) > 0:
                            X_val = market_data.loc[common_index].astype(float).values
                            y_val = historical_data.loc[common_index, 'returns'].astype(float).values
                            
                            # Check for NaN values
                            mask = ~np.isnan(X_val).any(axis=1) & ~np.isnan(y_val)
                            X_val = X_val[mask]
                            y_val = y_val[mask]
                            
                            self.logger.info(f"Valid samples after cleaning: {len(X_val)}")
                            
                            if len(X_val) > 0 and len(y_val) > 0:
                                # Scale features
                                self.logger.info("5/5: Making predictions...")
                                try:
                                    X_val = self.scaler.transform(X_val)
                                    predictions = self.market_model.predict(X_val)
                                    
                                    results["market"] = {
                                        "r2_score": float(self.market_model.score(X_val, y_val)),
                                        "rmse": float(np.sqrt(mean_squared_error(y_val, predictions))),
                                        "mae": float(mean_absolute_error(y_val, predictions))
                                    }
                                    
                                    self.logger.info("\nMarket Model Validation Metrics:")
                                    for metric, value in results["market"].items():
                                        self.logger.info(f"- {metric}: {value:.4f}")
                                    
                                except Exception as e:
                                    self.logger.error(f"Error in market model prediction: {str(e)}")
                                    self.logger.debug("Data shapes at prediction:")
                                    self.logger.debug(f"X_val shape: {X_val.shape}")
                                    self.logger.debug(f"y_val shape: {y_val.shape}")
                            else:
                                self.logger.warning("⚠ No valid samples after cleaning")
                                self.logger.debug("Data inspection:")
                                self.logger.debug(f"NaN in X_val: {np.isnan(X_val).sum()}")
                                self.logger.debug(f"NaN in y_val: {np.isnan(y_val).sum()}")
                        else:
                            self.logger.warning("⚠ No common dates found between market and historical data")
                    else:
                        self.logger.warning("⚠ Empty data after cleaning")
                        
                except Exception as e:
                    self.logger.error(f"✗ Error validating market model: {str(e)}")
            
            # Validate technical model
            try:
                self.logger.info("\n=== Validating Technical Model ===")
                tech_metrics = await self.technical_model.validate(validation_data["historical"])
                if tech_metrics:
                    results["technical"] = tech_metrics
                    self.logger.info("Technical Model Validation Metrics:")
                    for metric, value in tech_metrics.items():
                        self.logger.info(f"- {metric}: {value:.4f}")
                    self.logger.info("✓ Technical model validation completed")
            except Exception as e:
                self.logger.error(f"✗ Error validating technical model: {str(e)}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"✗ Error in model validation: {str(e)}")
            return results