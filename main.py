# main.py
import asyncio
import logging
from datetime import datetime
import sys
import os

from config.settings import Config
from config.logging_config import setup_logging
from models.model_manager import ModelManager
from analysis.analyzer import MainAnalyzer
from trading.strategy import TradingStrategy
from trading.broker import MT5Broker
from trading.risk_manager import RiskManager
from utils.data_loader import DataLoader
from utils.preprocessing import DataPreprocessor

class TradingBot:
    def __init__(self):
        """Initialize trading bot"""
        self.config = Config()
        self.logger = self._setup_logging()
        self.initialize_components()
        
    def _setup_logging(self):
        """Setup logging configuration"""
        trading_logger, _ = setup_logging()
        return trading_logger
        
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
            self.risk_manager = RiskManager(self.config)
            self.strategy = TradingStrategy(self.config)
            self.broker = MT5Broker(self.config)
            
            self.logger.info("All components initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Initialization error: {str(e)}")
            sys.exit(1)
            
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
                
                # Run analysis
                analysis_results = await self.analyzer.run_analysis({
                    'technical': processed_data,
                    'fundamental': processed_fundamental if self.config.USE_FUNDAMENTAL_ANALYSIS else None
                })
                
                # Generate trading signals
                signals = self.strategy.generate_signals(analysis_results)
                
                # Apply risk management
                account_info = self.broker.get_account_info()
                for signal in signals:
                    if self.risk_manager.check_risk_limits(account_info, signal):
                        # Calculate position size
                        signal['lot_size'] = self.risk_manager.calculate_position_size(
                            signal['strength'],
                            account_info
                        )
                        
                        # Execute trade
                        await self.broker.execute_trades([signal])
                    
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
    bot = TradingBot()
    try:
        asyncio.run(bot.run())
    except KeyboardInterrupt:
        bot.logger.info("Bot stopped by user")
        bot.cleanup()
    except Exception as e:
        bot.logger.error(f"Bot crashed: {str(e)}")
        bot.cleanup()