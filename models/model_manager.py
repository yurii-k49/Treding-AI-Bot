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
        self.logger = logging.getLogger('model.manager')
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

    async def update_if_needed(self):
        """Check and update models if necessary"""
        try:
            current_time = datetime.now()
            
            # Check if update is needed
            if (self.last_update is None or 
                (current_time - self.last_update).total_seconds() > 
                self.config.MODEL_UPDATE_INTERVAL * 3600):
                
                await self.update_models()
                self.last_update = current_time
                self.logger.info("Models updated successfully")
                
        except Exception as e:
            self.logger.error(f"Error checking model updates: {str(e)}")

    async def update_models(self):
        """Update all models"""
        try:
            # Get training data
            technical_data = await self._get_technical_training_data()
            fundamental_data = await self._get_fundamental_training_data()
            sentiment_data = await self._get_sentiment_training_data()
            
            # Update models in parallel
            await asyncio.gather(
                self.technical_model.train(technical_data),
                self.fundamental_model.train(fundamental_data),
                self.sentiment_model.train(sentiment_data)
            )
            
            # Save updated models
            self._save_models()
            
        except Exception as e:
            self.logger.error(f"Error updating models: {str(e)}")
            raise

    async def get_predictions(self, market_data):
        """Get predictions from all models"""
        try:
            # Get predictions in parallel
            technical_pred, fundamental_pred, sentiment_pred = await asyncio.gather(
                self.technical_model.predict(market_data.get('technical')),
                self.fundamental_model.predict(market_data.get('fundamental')),
                self.sentiment_model.predict(market_data.get('sentiment'))
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
            self.fundamental_model.save_models()
            self.sentiment_model.save_model()
        except Exception as e:
            self.logger.error(f"Error saving models: {str(e)}")