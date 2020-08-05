# models/model_manager.py
import logging
import asyncio
from datetime import datetime
from models.technical_model import TechnicalModel
from models.fundamental_model import FundamentalModel
from models.sentiment_model import SentimentModel

class ModelManager:
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger('models.manager')
        self.last_update = None
        self.setup_models()

    def setup_models(self):
        """Initialize all models"""
        try:
            self.technical_model = TechnicalModel(self.config)
            self.fundamental_model = FundamentalModel(self.config)
            self.sentiment_model = SentimentModel(self.config)
            self.logger.info("All models initialized successfully")
        except Exception as e:
            self.logger.error(f"Error initializing models: {str(e)}")
            raise

    async def train_all_models(self, training_data):
        """Train all models with provided data"""
        try:
            self.logger.info("Starting models training...")
            
            # Train models in parallel
            await asyncio.gather(
                self.technical_model.train(training_data['technical']),
                self.fundamental_model.train(training_data['fundamental']),
                self.sentiment_model.train(training_data['sentiment'])
            )
            
            # Save trained models
            self._save_models()
            
            self.last_update = datetime.now()
            self.logger.info("All models trained successfully")
            
        except Exception as e:
            self.logger.error(f"Error training models: {str(e)}")
            raise

    async def update_if_needed(self):
        """Check and update models if necessary"""
        try:
            current_time = datetime.now()
            
            # Check if update is needed
            if (self.last_update is None or 
                (current_time - self.last_update).total_seconds() > 
                self.config.MODEL_UPDATE_INTERVAL * 3600):
                
                self.logger.info("Models update needed")
                await self.train_all_models()
                self.last_update = current_time
                
        except Exception as e:
            self.logger.error(f"Error checking model updates: {str(e)}")

    async def predict(self, data):
        """Get predictions from all models"""
        try:
            # Get predictions in parallel
            technical_pred, fundamental_pred, sentiment_pred = await asyncio.gather(
                self.technical_model.predict(data['technical']),
                self.fundamental_model.predict(data['fundamental']),
                self.sentiment_model.predict(data['sentiment'])
            )
            
            return {
                'technical': technical_pred,
                'fundamental': fundamental_pred,
                'sentiment': sentiment_pred
            }
            
        except Exception as e:
            self.logger.error(f"Error getting predictions: {str(e)}")
            return None

    def _save_models(self):
        """Save all models to disk"""
        try:
            self.technical_model.save_model()
            self.fundamental_model.save_model()
            self.sentiment_model.save_model()
            self.logger.info("All models saved successfully")
        except Exception as e:
            self.logger.error(f"Error saving models: {str(e)}")

    async def validate_models(self, validation_data):
        """Validate all models"""
        try:
            # Validate models in parallel
            technical_score, fundamental_score, sentiment_score = await asyncio.gather(
                self.technical_model.validate(validation_data['technical']),
                self.fundamental_model.validate(validation_data['fundamental']),
                self.sentiment_model.validate(validation_data['sentiment'])
            )
            
            validation_results = {
                'technical_score': technical_score,
                'fundamental_score': fundamental_score,
                'sentiment_score': sentiment_score,
                'combined_score': (technical_score + fundamental_score + sentiment_score) / 3
            }
            
            self.logger.info(f"Validation results: {validation_results}")
            return validation_results
            
        except Exception as e:
            self.logger.error(f"Error validating models: {str(e)}")
            return None