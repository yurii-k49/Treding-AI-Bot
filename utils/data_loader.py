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
       self.cache = {}
       self.last_update = {}

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
           # Check cache first
           cache_key = f"{symbol}_{timeframe}_{bars}"
           if self._is_cache_valid(cache_key):
               return self.cache[cache_key]

           # Convert timeframe string to MT5 timeframe
           mt5_timeframe = self._get_mt5_timeframe(timeframe)

           # Validate symbol
           symbol_info = mt5.symbol_info(symbol)
           if symbol_info is None:
               raise Exception(f"Symbol {symbol} not found")

           # Enable symbol if needed
           if not symbol_info.visible:
               if not mt5.symbol_select(symbol, True):
                   raise Exception(f"Symbol {symbol} selection failed")

           # Get historical data
           rates = mt5.copy_rates_from_pos(symbol, mt5_timeframe, 0, bars)
           if rates is None:
               raise Exception("Failed to get historical data")

           # Convert to DataFrame
           df = pd.DataFrame(rates)
           df['time'] = pd.to_datetime(df['time'], unit='s')

           # Cache the result
           self._update_cache(cache_key, df)

           self.logger.info(f"Retrieved {len(df)} bars of historical data for {symbol}")
           return df

       except Exception as e:
           self.logger.error(f"Error loading historical data: {str(e)}")
           return None

   async def get_fundamental_data(self, symbol):
       """Get fundamental data for symbol"""
       try:
           cache_key = f"fundamental_{symbol}"
           if self._is_cache_valid(cache_key):
               return self.cache[cache_key]

           # Determine symbol type and get appropriate data
           if self._is_forex_pair(symbol):
               data = await self._get_forex_fundamental_data(symbol)
           elif self._is_stock_symbol(symbol):
               data = await self._get_stock_fundamental_data(symbol)
           elif self._is_crypto_symbol(symbol):
               data = await self._get_crypto_fundamental_data(symbol)
           else:
               self.logger.warning(f"Unsupported symbol type: {symbol}")
               return None

           # Cache the result
           self._update_cache(cache_key, data)
           return data

       except Exception as e:
           self.logger.error(f"Error getting fundamental data: {str(e)}")
           return None

   async def get_news_data(self, symbol, days=7):
       """Get news data for symbol"""
       try:
           cache_key = f"news_{symbol}"
           if self._is_cache_valid(cache_key):
               return self.cache[cache_key]

           if self.config.NEWS_API_KEY:
               news = await self._fetch_news_from_api(symbol, days)
           else:
               news = await self._fetch_news_from_mt5(symbol, days)

           self._update_cache(cache_key, news)
           return news

       except Exception as e:
           self.logger.error(f"Error getting news data: {str(e)}")
           return None

   async def _get_forex_fundamental_data(self, symbol):
       """Get forex fundamental data"""
       try:
           base_currency = symbol[:3]
           quote_currency = symbol[3:]

           # Get data in parallel
           calendar_data, interest_rates, trade_balance = await asyncio.gather(
               self._get_economic_calendar([base_currency, quote_currency]),
               self._get_interest_rates([base_currency, quote_currency]),
               self._get_trade_balance([base_currency, quote_currency])
           )

           return {
               'calendar': calendar_data,
               'interest_rates': interest_rates,
               'trade_balance': trade_balance
           }

       except Exception as e:
           self.logger.error(f"Error in forex fundamental data: {str(e)}")
           return None

   async def _get_stock_fundamental_data(self, symbol):
       """Get stock fundamental data"""
       try:
           stock = yf.Ticker(symbol)
           
           # Get basic info
           info = stock.info
           
           # Get financial statements
           financials = {
               'income_stmt': stock.financials.to_dict(),
               'balance_sheet': stock.balance_sheet.to_dict(),
               'cash_flow': stock.cashflow.to_dict()
           }
           
           # Calculate key ratios
           ratios = {
               'pe_ratio': info.get('forwardPE'),
               'pb_ratio': info.get('priceToBook'),
               'profit_margin': info.get('profitMargins'),
               'debt_to_equity': info.get('debtToEquity')
           }

           return {
               'info': info,
               'financials': financials,
               'ratios': ratios
           }

       except Exception as e:
           self.logger.error(f"Error in stock fundamental data: {str(e)}")
           return None

   async def _get_crypto_fundamental_data(self, symbol):
       """Get cryptocurrency fundamental data"""
       try:
           # Basic demo data for crypto
           return {
               'market_cap': 1000000000,
               'volume_24h': 50000000,
               'circulating_supply': 18000000,
               'max_supply': 21000000
           }
       except Exception as e:
           self.logger.error(f"Error in crypto fundamental data: {str(e)}")
           return None

   async def _get_economic_calendar(self, currencies):
       """Get economic calendar data"""
       try:
           # Demo calendar data
           return {
               'events': [
                   {
                       'currency': currencies[0],
                       'event': 'Interest Rate Decision',
                       'importance': 'high',
                       'actual': '1.5%',
                       'forecast': '1.5%',
                       'previous': '1.25%'
                   }
               ]
           }
       except Exception as e:
           self.logger.error(f"Error getting economic calendar: {str(e)}")
           return None

   async def _get_interest_rates(self, currencies):
       """Get interest rates for currencies"""
       try:
           return {currencies[0]: 1.5, currencies[1]: 0.5}
       except Exception as e:
           self.logger.error(f"Error getting interest rates: {str(e)}")
           return None

   async def _get_trade_balance(self, currencies):
       """Get trade balance data"""
       try:
           return {currencies[0]: 100, currencies[1]: -50}
       except Exception as e:
           self.logger.error(f"Error getting trade balance: {str(e)}")
           return None

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

   def _is_forex_pair(self, symbol):
       """Check if symbol is forex pair"""
       return len(symbol) == 6 and symbol.isalpha()

   def _is_stock_symbol(self, symbol):
       """Check if symbol is stock"""
       return '.' not in symbol and len(symbol) < 5

   def _is_crypto_symbol(self, symbol):
       """Check if symbol is crypto"""
       return symbol.endswith('USD') or symbol.endswith('BTC')

   def _is_cache_valid(self, key):
       """Check if cached data is still valid"""
       if key not in self.cache or key not in self.last_update:
           return False
       
       cache_duration = {
           'historical': 300,  # 5 minutes
           'fundamental': 3600,  # 1 hour
           'news': 1800  # 30 minutes
       }
       
       data_type = key.split('_')[0]
       max_age = cache_duration.get(data_type, 300)
       
       age = (datetime.now() - self.last_update[key]).total_seconds()
       return age < max_age

   def _update_cache(self, key, data):
       """Update cache with new data"""
       self.cache[key] = data
       self.last_update[key] = datetime.now()

   def cleanup(self):
       """Cleanup resources"""
       try:
           mt5.shutdown()
           self.logger.info("MT5 connection closed")
       except Exception as e:
           self.logger.error(f"Error in cleanup: {str(e)}")