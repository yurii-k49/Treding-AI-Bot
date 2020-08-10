# main.py

import asyncio
import logging
import sys
from aiohttp import web
import json
import os
import numpy as np
import pandas as pd

from config.settings import Config
from config.logging_config import setup_logging
from models.model_manager import ModelManager
from analysis.analyzer import MainAnalyzer
from trading.strategy import TradingStrategy
from trading.broker import MT5Broker
from utils.data_loader import DataLoader
from utils.preprocessing import DataPreprocessor
from trading_history_generator import generate_trading_history
from trading.backtest import Backtester  # Import Backtester

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
            self.logger.info("Fetching training data...")
            training_data = await self.data_loader.get_training_data(
                symbol=self.config.SYMBOL,
                timeframe=self.data_loader.timeframe,
                bars=10000
            )

            # Validate training data
            if training_data["historical"] is None or training_data["historical"].empty:
                raise Exception("Failed to get historical training data")

            self.logger.info("\nData Overview:")
            self.logger.info(f"Historical data shape: {training_data['historical'].shape}")
            if training_data.get("market") is not None:
                self.logger.info(f"Market data shape: {training_data['market'].shape}")

            # Check for minimal data requirements
            min_samples = 1000
            if len(training_data["historical"]) < min_samples:
                self.logger.warning(f"⚠ Historical data has less than {min_samples} samples")

            # Train models
            self.logger.info("\nTraining models...")
            results = await self.model_manager.train_all_models(training_data)

            # Validate models
            self.logger.info("\nValidating models...")
            validation_data = await self.data_loader.get_validation_data(
                symbol=self.config.SYMBOL,
                timeframe=self.data_loader.timeframe
            )

            if validation_data:
                self.logger.info("Validation data received:")
                self.logger.info(f"- Historical: {validation_data['historical'].shape if validation_data.get('historical') is not None else 'None'}")
                self.logger.info(f"- Market: {validation_data['market'].shape if validation_data.get('market') is not None else 'None'}")

                validation_results = await self.model_manager.validate_models(validation_data)
                self.logger.info("Validation results:", validation_results)
            else:
                self.logger.warning("⚠ No validation data available")

            return results

        except Exception as e:
            self.logger.error(f"Error in model training: {str(e)}")
            raise

    async def handle_index(self, request):
        """Serve index.html"""
        try:
            with open("frontend/index.html") as f:
                return web.Response(text=f.read(), content_type="text/html")
        except Exception as e:
            self.logger.error(f"Error serving index.html: {str(e)}")
            return web.Response(text="Error", status=500)

    async def handle_backtest(self, request):
        """Serve backtest.html"""
        try:
            with open("frontend/backtest.html") as f:
                return web.Response(text=f.read(), content_type="text/html")
        except Exception as e:
            self.logger.error(f"Error serving backtest.html: {str(e)}")
            return web.Response(text="Error", status=500)

    async def handle_backtest_results(self, request):
        """API endpoint for backtest results"""
        try:
            # Get historical data for backtest
            historical_data = await self.data_loader.get_historical_data(
                symbol=self.config.SYMBOL,
                timeframe=self.data_loader.timeframe,
                bars=10000
            )
            
            if historical_data is None or historical_data.empty:
                return web.json_response({"error": "No historical data available"}, status=500)

            # Get predictions from neural model
            predictions = await self.model_manager.neural_model.predict(historical_data)
            if predictions is None:
                return web.json_response({"error": "Failed to get predictions"}, status=500)

            # Run backtest
            backtester = Backtester(self.config)
            results = await backtester.run_backtest(historical_data, predictions)
            
            if results is None:
                return web.json_response({"error": "Backtest failed"}, status=500)

            # Calculate monthly returns
            monthly_returns = backtester.get_monthly_returns()
            monthly_data = [
                {
                    'date': date.strftime('%Y-%m'),
                    'return': float(ret) * 100
                }
                for date, ret in monthly_returns.items()
            ]

            # Prepare response
            response_data = {
                'metrics': results['metrics'],
                'trades': [
                    {
                        'entry_date': t['entry_date'].strftime('%Y-%m-%d %H:%M:%S'),
                        'exit_date': t['exit_date'].strftime('%Y-%m-%d %H:%M:%S'),
                        'type': t['type'],
                        'entry_price': float(t['entry_price']),
                        'exit_price': float(t['exit_price']),
                        'pnl': float(t['pnl']),
                        'balance': float(t['balance'])
                    }
                    for t in results['trades']
                ],
                'equity_curve': [
                    {
                        'date': e['date'].strftime('%Y-%m-%d %H:%M:%S'),
                        'equity': float(e['equity'])
                    }
                    for e in results['equity_curve']
                ],
                'monthly_returns': monthly_data
            }

            return web.json_response(response_data)

        except Exception as e:
            self.logger.error(f"Error serving backtest results: {str(e)}")
            return web.Response(text=str(e), status=500)

    async def handle_trading_history(self, request):
        """API endpoint for trading history"""
        try:
            json_file = "data/history/trading_history.json"

            # Generate trading history if it doesn't exist
            if not os.path.exists(json_file):
                await self.generate_history()

            # Read and validate JSON data
            try:
                with open(json_file, "r") as f:
                    data = json.load(f)

                # Validate data structure
                if not isinstance(data, list):
                    raise ValueError("Data must be a list of trades")

                return web.json_response(data)
            except json.JSONDecodeError as e:
                self.logger.error(f"Invalid JSON in trading history file: {e}")
                return web.json_response({"error": "Invalid trading history data"}, status=500)
            except ValueError as e:
                self.logger.error(f"Invalid data structure: {e}")
                return web.json_response({"error": str(e)}, status=500)

        except Exception as e:
            self.logger.error(f"Error serving trading history: {str(e)}")
            return web.json_response({"error": "Internal server error"}, status=500)

    async def start_web_server(self):
        """Start web server"""
        try:
            # Initialize models
            await self.model_manager.load_models()
            
            # Create application
            app = web.Application()
            
            # Add routes
            app.router.add_get('/', self.handle_index)
            app.router.add_get('/backtest', self.handle_backtest)
            app.router.add_get('/api/trading-history', self.handle_trading_history)
            app.router.add_get('/api/backtest-results', self.handle_backtest_results)
            
            # Add static files
            app.router.add_static('/static', 'frontend/static')
            
            # Setup and start server
            runner = web.AppRunner(app)
            await runner.setup()
            site = web.TCPSite(runner, 'localhost', 8080)
            
            self.logger.info("Starting web server at http://localhost:8080")
            await site.start()
            
            # Keep server running
            while True:
                await asyncio.sleep(1)
                
        except Exception as e:
            self.logger.error(f"Error starting web server: {str(e)}")
            raise

    def cleanup(self):
        """Cleanup resources"""
        try:
            self.broker.cleanup()
            self.logger.info("Cleanup completed successfully")
        except Exception as e:
            self.logger.error(f"Cleanup error: {str(e)}")

async def main(command):
    bot = TradingBot()
    try:
        if command == "learn":
            await bot.train_models()
        elif command == "history":
            await bot.generate_history()
        elif command == "web":
            await bot.start_web_server()
        else:
            print("Invalid command. Use 'learn', 'history', or 'web'")
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        bot.cleanup()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python main.py [learn|history|web]")
        sys.exit(1)
        
    command = sys.argv[1]
    asyncio.run(main(command))
