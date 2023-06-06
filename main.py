# main.py
import asyncio
import logging
import sys
import os
import pandas as pd
from datetime import datetime

from config.settings import Config
from config.logging_config import setup_logging
from models.model_manager import ModelManager
from analysis.analyzer import MainAnalyzer
from trading.strategy import TradingStrategy
from trading.broker import MT5Broker
from utils.data_loader import DataLoader
from utils.preprocessing import DataPreprocessor

# Setup root logger first
root_logger = setup_logging()

class TradingBot:
   def __init__(self):
       """Initialize trading bot"""
       try:
           self.config = Config()
           self.logger = logging.getLogger('trading')
           self.initialize_components()
           self.logger.info("Trading bot initialized successfully")
       except Exception as e:
           self.logger.error(f"Initialization error: {str(e)}")
           raise

   def initialize_components(self):
       """Initialize all components"""
       try:
           # Data components
           self.data_loader = DataLoader(self.config)
           self.preprocessor = DataPreprocessor(self.config)
           
           # AI components
           self.model_manager = ModelManager(self.config)
           self.analyzer = MainAnalyzer(self.config)
           
           # Trading components
           self.strategy = TradingStrategy(self.config)
           self.broker = MT5Broker(self.config)
           
           self.logger.info("All components initialized successfully")
           
       except Exception as e:
           self.logger.error(f"Component initialization error: {str(e)}")
           raise

   async def run(self):
       """Main trading loop"""
       self.logger.info(f"Starting trading bot in {self.config.MODE} mode")
       
       while True:
           try:
               # Get market data
               market_data = await self.data_loader.get_historical_data(
                   self.config.SYMBOL,
                   self.config.TIMEFRAME
               )
               
               if market_data is None:
                   raise Exception("Failed to get market data")
               
               # Preprocess data
               processed_data = self.preprocessor.prepare_technical_features(market_data)
               
               # Get fundamental and sentiment data if enabled
               if self.config.USE_FUNDAMENTAL_ANALYSIS:
                   fundamental_data, sentiment_data = await asyncio.gather(
                       self.data_loader.get_fundamental_data(self.config.SYMBOL),
                       self.data_loader.get_news_data(self.config.SYMBOL)
                   )
                   processed_fundamental = self.preprocessor.prepare_fundamental_features(fundamental_data)
                   processed_sentiment = self.preprocessor.prepare_sentiment_features(sentiment_data)
               else:
                   processed_fundamental = None
                   processed_sentiment = None
               
               # Run analysis with all components
               analysis_results = await self.analyzer.run_analysis({
                   'technical': processed_data,
                   'fundamental': processed_fundamental,
                   'sentiment': processed_sentiment
               })
               
               # Generate trading signals
               signals = self.strategy.generate_signals(analysis_results)
               
               # Execute trades if we have valid signals
               if signals:
                   await self.broker.execute_trades(signals)
               
               # Update models if needed
               await self.model_manager.update_if_needed()
               
               # Wait for next iteration
               await asyncio.sleep(self.config.UPDATE_INTERVAL)
               
           except Exception as e:
               self.logger.error(f"Error in main loop: {str(e)}")
               await asyncio.sleep(60)

   async def train_models(self):
    try:
        self.logger.info("Starting model training")
        
        # Get training data
        training_data = await self.data_loader.get_training_data(
            symbol=self.config.SYMBOL,
            timeframe=self.config.TIMEFRAME,
            bars=10000
        )
        
        if training_data is None:
            raise Exception("Failed to get training data")
            
        self.logger.info("Got training data, preprocessing...")
        
        # Preprocess data
        processed_data = {
            'technical': self.preprocessor.prepare_technical_features(training_data['historical']),
            'fundamental': self.preprocessor.prepare_fundamental_features(training_data['fundamental']),
            'sentiment': self.preprocessor.prepare_sentiment_features(training_data['sentiment'])
        }
        
        # Log shapes of processed data
        for key, data in processed_data.items():
            if isinstance(data, pd.DataFrame):
                self.logger.info(f"{key} data shape: {data.shape}")
        
        # Train models
        await self.model_manager.train_all_models(processed_data)
        
        self.logger.info("Model training completed successfully")
        
    except Exception as e:
        self.logger.error(f"Error in model training: {str(e)}")
        raise

   async def validate_models(self):
       """Validate models on test data"""
       try:
           self.logger.info("Starting model validation")
           
           # Get validation data
           validation_data = await self.data_loader.get_validation_data(
               self.config.SYMBOL,
               self.config.TIMEFRAME
           )
           
           if validation_data is None:
               raise Exception("Failed to get validation data")
           
           # Preprocess validation data
           processed_data = self.preprocessor.prepare_technical_features(validation_data)
           
           # Get validation fundamental and sentiment data
           if self.config.USE_FUNDAMENTAL_ANALYSIS:
               fundamental_data, sentiment_data = await asyncio.gather(
                   self.data_loader.get_validation_fundamental_data(self.config.SYMBOL),
                   self.data_loader.get_validation_sentiment_data(self.config.SYMBOL)
               )
               processed_fundamental = self.preprocessor.prepare_fundamental_features(fundamental_data)
               processed_sentiment = self.preprocessor.prepare_sentiment_features(sentiment_data)
           else:
               processed_fundamental = None
               processed_sentiment = None
           
           # Validate models
           validation_results = await self.model_manager.validate_models({
               'technical': processed_data,
               'fundamental': processed_fundamental,
               'sentiment': processed_sentiment
           })
           
           self.logger.info(f"Model validation results: {validation_results}")
           
           return validation_results
           
       except Exception as e:
           self.logger.error(f"Error in model validation: {str(e)}")
           raise
               
   def cleanup(self):
       """Cleanup resources"""
       try:
           self.broker.cleanup()
           self.logger.info("Cleanup completed successfully")
       except Exception as e:
           self.logger.error(f"Cleanup error: {str(e)}")

if __name__ == "__main__":
   main_logger = logging.getLogger('main')
   
   try:
       # Parse command line arguments
       if len(sys.argv) > 1:
           command = sys.argv[1]
           
           if command == 'learn':
               # Training mode
               main_logger.info("Starting in training mode")
               bot = TradingBot()
               asyncio.run(bot.train_models())
               # After training, validate the models
               validation_results = asyncio.run(bot.validate_models())
               main_logger.info(f"Training completed. Validation results: {validation_results}")
               
           elif command == 'validate':
               # Validation only mode
               main_logger.info("Starting in validation mode")
               bot = TradingBot()
               validation_results = asyncio.run(bot.validate_models())
               main_logger.info(f"Validation results: {validation_results}")
               
           else:
               main_logger.error(f"Unknown command: {command}")
               sys.exit(1)
       else:
           # Normal trading mode
           main_logger.info("Starting in trading mode")
           bot = TradingBot()
           asyncio.run(bot.run())
           
   except KeyboardInterrupt:
       main_logger.info("Bot stopped by user")
       bot.cleanup()
   except Exception as e:
       main_logger.error(f"Bot crashed: {str(e)}")
       if 'bot' in locals():
           bot.cleanup()
       sys.exit(1)