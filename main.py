# main.py
import asyncio
import logging
import sys
import os
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
                
                # Get fundamental data if enabled
                if self.config.USE_FUNDAMENTAL_ANALYSIS:
                    fundamental_data = await self.data_loader.get_fundamental_data(
                        self.config.SYMBOL
                    )
                    processed_fundamental = self.preprocessor.prepare_fundamental_features(
                        fundamental_data
                    )
                else:
                    processed_fundamental = None
                
                # Run analysis
                analysis_results = await self.analyzer.run_analysis({
                    'technical': processed_data,
                    'fundamental': processed_fundamental
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
                
    def cleanup(self):
        """Cleanup resources"""
        try:
            self.broker.cleanup()
            self.logger.info("Cleanup completed successfully")
        except Exception as e:
            self.logger.error(f"Cleanup error: {str(e)}")

if __name__ == "__main__":
    # Get root logger for main script
    main_logger = logging.getLogger('main')
    
    try:
        # Create and run bot
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