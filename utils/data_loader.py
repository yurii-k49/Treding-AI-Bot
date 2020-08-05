# utils/data_loader.py
import logging
import pandas as pd
import numpy as np
import MetaTrader5 as mt5
import yfinance as yf
import requests
from datetime import datetime, timedelta
import asyncio
import json

class DataLoader:
    def __init__(self, config):
        """Data loader initialization"""
        self.config = config
        self.logger = logging.getLogger('utils.data_loader')
        self._initialize_mt5()

    def _initialize_mt5(self):
        """Initialize MT5 connection"""
        try:
            if not mt5.initialize():
                self.logger.error("MT5 initialization failed")
                raise Exception("MT5 initialization failed")

            # Login to MT5
            if not mt5.login(
                login=int(self.config.MT5_LOGIN),
                password=self.config.MT5_PASSWORD,
                server=self.config.MT5_SERVER
            ):
                self.logger.error("MT5 login failed")
                raise Exception("MT5 login failed")

            self.logger.info("MT5 connection established successfully")

        except Exception as e:
            self.logger.error(f"MT5 initialization error: {str(e)}")
            raise

    async def get_historical_data(self, symbol, timeframe, bars=1000):
        """Get historical market data from MT5"""
        try:
            # Convert timeframe string to MT5 timeframe
            mt5_timeframe = self._get_mt5_timeframe(timeframe)

            # Get historical data
            rates = mt5.copy_rates_from_pos(symbol, mt5_timeframe, 0, bars)
            if rates is None:
                raise Exception("Failed to get historical data")

            # Convert to DataFrame
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')

            self.logger.info(f"Retrieved {len(df)} bars of historical data")
            return df

        except Exception as e:
            self.logger.error(f"Error loading historical data: {str(e)}")
            return None
            
    async def get_training_data(self, symbol, timeframe, bars=10000):
        try:
            # Get extended historical data for training
            df = await self.get_historical_data(symbol, timeframe, bars)
            if df is None:
                raise Exception("Failed to get training data")
            
            # Get additional data for training if enabled
            training_data = {
                'historical': df,
                'fundamental': None,
                'sentiment': None
            }
            
            if self.config.USE_FUNDAMENTAL_ANALYSIS:
                fundamental_data = await self.get_training_fundamental_data(symbol)
                sentiment_data = await self.get_training_sentiment_data(symbol)
                
                training_data['fundamental'] = fundamental_data
                training_data['sentiment'] = sentiment_data
            
            self.logger.info(f"Got training data: {len(df)} candles for {symbol}")
            return training_data
            
        except Exception as e:
            self.logger.error(f"Error getting training data: {str(e)}")
            return None

    async def get_training_fundamental_data(self, symbol, days=365):
        try:
            start_date = datetime.now() - timedelta(days=days)
            
            data = {
                'economic': await self._get_historical_economic_data(symbol, start_date),
                'financial': await self._get_historical_financial_data(symbol, start_date)
            }
            
            return data
            
        except Exception as e:
            self.logger.error(f"Error getting training fundamental data: {str(e)}")
            return None

    async def get_training_sentiment_data(self, symbol, days=365):
        try:
            start_date = datetime.now() - timedelta(days=days)
            
            data = {
                'news': await self._get_historical_news(symbol, start_date),
                'social': await self._get_historical_social_sentiment(symbol, start_date)
            }
            
            return data
            
        except Exception as e:
            self.logger.error(f"Error getting training sentiment data: {str(e)}")
            return None

    async def _get_historical_economic_data(self, symbol, start_date):
        try:
            return {
                'gdp': self._generate_demo_data(start_date, base=2.5),
                'inflation': self._generate_demo_data(start_date, base=2.0),
                'interest_rate': self._generate_demo_data(start_date, base=1.5)
            }
        except Exception as e:
            self.logger.error(f"Error getting historical economic data: {str(e)}")
            return None

    async def _get_historical_financial_data(self, symbol, start_date):
        try:
            return {
                'pe_ratio': self._generate_demo_data(start_date, base=15),
                'revenue': self._generate_demo_data(start_date, base=1000000),
                'profit': self._generate_demo_data(start_date, base=100000)
            }
        except Exception as e:
            self.logger.error(f"Error getting historical financial data: {str(e)}")
            return None

    async def _get_historical_news(self, symbol, start_date):
        try:
            return [
                {
                    'date': start_date + timedelta(days=x),
                    'title': f'News event {x}',
                    'sentiment': 0.5 + (np.random.random() - 0.5) * 0.5
                }
                for x in range(365)
            ]
        except Exception as e:
            self.logger.error(f"Error getting historical news: {str(e)}")
            return None

    async def _get_historical_social_sentiment(self, symbol, start_date):
        try:
            return [
                {
                    'date': start_date + timedelta(days=x),
                    'sentiment': 0.5 + (np.random.random() - 0.5) * 0.5,
                    'volume': 1000 + np.random.random() * 1000
                }
                for x in range(365)
            ]
        except Exception as e:
            self.logger.error(f"Error getting historical social sentiment: {str(e)}")
            return None

    def _generate_demo_data(self, start_date, base=100):
        return [
            {
                'date': start_date + timedelta(days=x),
                'value': base + (np.random.random() - 0.5) * base * 0.1
            }
            for x in range(365)
        ]

    def _get_mt5_timeframe(self, timeframe_str):
        """Convert string timeframe to MT5 timeframe value"""
        timeframe_map = {
            'M1': mt5.TIMEFRAME_M1,
            'M5': mt5.TIMEFRAME_M5,
            'M15': mt5.TIMEFRAME_M15,
            'M30': mt5.TIMEFRAME_M30,
            'H1': mt5.TIMEFRAME_H1,
            'H4': mt5.TIMEFRAME_H4,
            'D1': mt5.TIMEFRAME_D1,
            'W1': mt5.TIMEFRAME_W1,
            'MN1': mt5.TIMEFRAME_MN1
        }
        return timeframe_map.get(timeframe_str, mt5.TIMEFRAME_H1)

    def cleanup(self):
        """Cleanup resources"""
        try:
            mt5.shutdown()
            self.logger.info("MT5 connection closed")
        except Exception as e:
            self.logger.error(f"Error in cleanup: {str(e)}")