# main.py
import asyncio
import logging
import sys
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

root_logger = setup_logging()

class TradingBot:
    def __init__(self):
        """Initialize trading bot"""
        try:
            self.config = Config()
            self.logger = logging.getLogger("trading")
            self.initialize_components()
            self.logger.info("Trading bot initialized successfully")
        except Exception as e:
            self.logger.error(f"Initialization error: {str(e)}")
            raise

    def initialize_components(self):
        """Initialize all components"""
        try:
            self.data_loader = DataLoader(self.config)
            self.preprocessor = DataPreprocessor(self.config)
            self.model_manager = ModelManager(self.config)
            self.analyzer = MainAnalyzer(self.config)
            self.strategy = TradingStrategy(self.config)
            self.broker = MT5Broker(self.config)
            self.logger.info("All components initialized successfully")
        except Exception as e:
            self.logger.error(f"Component initialization error: {str(e)}")
            raise

    async def train_models(self):
        """Train and validate models"""
        try:
            self.logger.info("Starting model training")

            # Get training data
            training_data = await self.data_loader.get_training_data(
                symbol=self.config.SYMBOL, 
                timeframe=self.data_loader.timeframe,
                bars=10000
            )

            if training_data["historical"] is None or training_data["historical"].empty:
                raise Exception("Failed to get historical training data")
                
            self.logger.info("Got training data, preprocessing...")
            self.logger.info(f"Historical data shape: {training_data['historical'].shape}")

            # Process training data
            processed_data = {
                "historical": self.preprocessor.prepare_technical_features(training_data["historical"]),
                "market": self.preprocessor.prepare_market_features(training_data["market"]) 
                         if "market" in training_data else None
            }

            # Log data shapes
            for key, data in processed_data.items():
                if isinstance(data, pd.DataFrame) and not data.empty:
                    self.logger.info(f"Processed {key} data shape: {data.shape}")

            # Train models
            self.logger.info("Training models...")
            training_results = await self.model_manager.train_all_models(processed_data)

            # Log training results
            if training_results:
                self.logger.info("\n=== Training Results ===")
                if "technical" in training_results:
                    self.logger.info("Technical Model:")
                    for metric, value in training_results["technical"].items():
                        self.logger.info(f"- {metric}: {value:.4f}")
                if "market" in training_results:
                    self.logger.info("Market Model:")
                    for metric, value in training_results["market"].items():
                        self.logger.info(f"- {metric}: {value:.4f}")

            return training_results

        except Exception as e:
            self.logger.error(f"Error in model training: {str(e)}")
            raise

    def cleanup(self):
        """Cleanup resources"""
        try:
            self.broker.cleanup()
            self.logger.info("Cleanup completed successfully")
        except Exception as e:
            self.logger.error(f"Cleanup error: {str(e)}")

if __name__ == "__main__":
    main_logger = logging.getLogger("main")
    
    try:
        if len(sys.argv) > 1 and sys.argv[1] == "learn":
            bot = TradingBot()
            asyncio.run(bot.train_models())
        else:
            main_logger.error("Please specify 'learn' command to start training")
            
    except KeyboardInterrupt:
        main_logger.info("Bot stopped by user")
        if 'bot' in locals():
            bot.cleanup()
    except Exception as e:
        main_logger.error(f"Bot crashed: {str(e)}")
        if 'bot' in locals():
            bot.cleanup()
        sys.exit(1)