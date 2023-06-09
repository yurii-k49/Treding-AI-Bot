# utils/data_loader.py

import pandas as pd
import numpy as np
import MetaTrader5 as mt5
import logging
from datetime import datetime, timedelta
import asyncio
from tqdm import tqdm

class DataLoader:
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger('utils.data_loader')
        self.symbol = config.SYMBOL
        
        self.timeframe_map = {
            'M1': mt5.TIMEFRAME_M1,
            'M5': mt5.TIMEFRAME_M5,
            'M15': mt5.TIMEFRAME_M15,
            'M30': mt5.TIMEFRAME_M30,
            'H1': mt5.TIMEFRAME_H1,
            'H4': mt5.TIMEFRAME_H4,
            'D1': mt5.TIMEFRAME_D1,
        }
        self.timeframe = self.timeframe_map.get(config.TIMEFRAME, mt5.TIMEFRAME_H1)

    def _initialize_mt5(self):
        """Initialize MT5 connection"""
        try:
            if not mt5.initialize(
                login=self.config.MT5_LOGIN,
                password=self.config.MT5_PASSWORD,
                server=self.config.MT5_SERVER
            ):
                self.logger.error(f"MT5 initialization failed: {mt5.last_error()}")
                return False
                
            if not mt5.terminal_info().connected:
                self.logger.error("MT5 not connected to server!")
                return False
                
            return True
            
        except Exception as e:
            self.logger.error(f"MT5 initialization error: {str(e)}")
            return False

    async def get_historical_data(self, symbol=None, timeframe=None, bars=1000):
        """Get historical price data"""
        try:
            if not self._initialize_mt5():
                return None
                
            symbol = symbol or self.symbol
            timeframe = timeframe or self.timeframe
            
            self.logger.info(f"Fetching {bars} bars of historical data for {symbol}")
            
            rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, bars)
            if rates is None:
                self.logger.error(f"Failed to get historical data: {mt5.last_error()}")
                return None
                
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df.set_index('time', inplace=True)
            
            self.logger.info(f"Retrieved {len(df)} bars of historical data")
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error getting historical data: {str(e)}")
            return None
        finally:
            mt5.shutdown()

    async def get_market_data(self, symbol=None):
        """Get current market data and info"""
        try:
            if not self._initialize_mt5():
                return None
                
            symbol = symbol or self.symbol
            symbol_info = mt5.symbol_info(symbol)
            
            if symbol_info is None:
                self.logger.error(f"Failed to get symbol info for {symbol}")
                return None
                
            tick = mt5.symbol_info_tick(symbol)
            
            return {
                'symbol': symbol,
                'bid': tick.bid,
                'ask': tick.ask,
                'last': tick.last,
                'spread': tick.ask - tick.bid,
                'volume': tick.volume,
                'volume_real': tick.volume_real,
                'time': datetime.fromtimestamp(tick.time),
                'point': symbol_info.point,
                'digits': symbol_info.digits,
                'spread_float': symbol_info.spread_float,
                'trade_calc_mode': symbol_info.trade_calc_mode,
                'trade_mode': symbol_info.trade_mode,
                'volume_min': symbol_info.volume_min,
                'volume_max': symbol_info.volume_max,
                'volume_step': symbol_info.volume_step,
                'margin_initial': symbol_info.margin_initial,
                'margin_maintenance': symbol_info.margin_maintenance,
                'tick_value': symbol_info.trade_tick_value,
                'tick_size': symbol_info.trade_tick_size
            }
            
        except Exception as e:
            self.logger.error(f"Error getting market data: {str(e)}")
            return None
        finally:
            mt5.shutdown()

    async def get_training_data(self, symbol=None, timeframe=None, bars=10000):
        """Get data for model training"""
        try:
            self.logger.info(f"Getting training data for {bars} bars...")
            
            # Get historical data
            historical_df = await self.get_historical_data(symbol, timeframe, bars)
            if historical_df is None or historical_df.empty:
                raise RuntimeError("Failed to get historical training data")
                
            # Calculate market indicators
            market_indicators = await self._get_market_indicators(historical_df)
            
            # Log data info
            self.logger.info(f"Training data shapes:")
            self.logger.info(f"- Historical: {historical_df.shape}")
            self.logger.info(f"- Market indicators: {market_indicators.shape}")
            
            return {
                "historical": historical_df,
                "market": market_indicators
            }
            
        except Exception as e:
            self.logger.error(f"Error getting training data: {str(e)}")
            raise

    async def get_validation_data(self, symbol=None, timeframe=None, bars=1000):
        """Get validation data"""
        try:
            self.logger.info("Getting validation data...")
            
            # Get historical data
            historical_df = await self.get_historical_data(
                symbol=symbol, 
                timeframe=timeframe, 
                bars=bars
            )
            
            if historical_df is None or historical_df.empty:
                raise RuntimeError("Failed to get historical validation data")
                
            # Calculate market indicators
            market_indicators = await self._get_market_indicators(historical_df)
            
            # Log data shapes
            self.logger.info(f"Validation data shapes:")
            self.logger.info(f"- Historical: {historical_df.shape}")
            self.logger.info(f"- Market indicators: {market_indicators.shape}")
            
            return {
                "historical": historical_df,
                "market": market_indicators
            }
                
        except Exception as e:
            self.logger.error(f"Error getting validation data: {str(e)}")
            return None

    async def _get_market_indicators(self, df):
        """Calculate technical indicators"""
        try:
            self.logger.info("Calculating market indicators...")
            indicators = pd.DataFrame(index=df.index)
            
            # Price based indicators
            indicators['price_sma5'] = df['close'].rolling(window=5).mean()
            indicators['price_sma20'] = df['close'].rolling(window=20).mean()
            indicators['price_sma50'] = df['close'].rolling(window=50).mean()
            
            # Price ratios
            indicators['price_ratio_sma5'] = df['close'] / indicators['price_sma5']
            indicators['price_ratio_sma20'] = df['close'] / indicators['price_sma20']
            
            # Volatility indicators
            indicators['volatility'] = df['close'].pct_change().rolling(window=20).std()
            indicators['atr'] = await self._calculate_atr(df)
            
            # Volume indicators
            indicators['volume_sma20'] = df['tick_volume'].rolling(window=20).mean()
            indicators['volume_ratio'] = df['tick_volume'] / indicators['volume_sma20']
            
            # Momentum indicators
            indicators['rsi'] = await self._calculate_rsi(df['close'])
            indicators['macd'] = await self._calculate_macd(df['close'])
            
            # Price changes
            indicators['returns'] = df['close'].pct_change()
            indicators['log_returns'] = np.log(df['close'] / df['close'].shift(1))
            
            # High-Low range
            indicators['hl_range'] = (df['high'] - df['low']) / df['close']
            
            # Time-based features
            indicators['hour'] = pd.to_datetime(df.index).hour
            indicators['day_of_week'] = pd.to_datetime(df.index).dayofweek
            
            # Fill missing values
            indicators = indicators.ffill().bfill()
            
            # Remove any remaining NaN
            indicators = indicators.fillna(0)
            
            self.logger.info(f"Calculated {len(indicators.columns)} market indicators")
            return indicators
            
        except Exception as e:
            self.logger.error(f"Error calculating market indicators: {str(e)}")
            return pd.DataFrame()

    async def _calculate_atr(self, df, period=14):
        """Calculate Average True Range"""
        try:
            high = df['high']
            low = df['low']
            close = df['close']
            
            tr1 = high - low
            tr2 = abs(high - close.shift())
            tr3 = abs(low - close.shift())
            
            tr = pd.DataFrame({'TR1': tr1, 'TR2': tr2, 'TR3': tr3}).max(axis=1)
            atr = tr.rolling(window=period).mean()
            
            return atr.bfill()
            
        except Exception as e:
            self.logger.error(f"Error calculating ATR: {str(e)}")
            return pd.Series(0, index=df.index)

    async def _calculate_rsi(self, prices, period=14):
        """Calculate Relative Strength Index"""
        try:
            # Calculate price changes
            delta = prices.diff()
            
            # Separate gains and losses
            gains = delta.where(delta > 0, 0)
            losses = -delta.where(delta < 0, 0)
            
            # Calculate average gains and losses
            avg_gains = gains.rolling(window=period).mean()
            avg_losses = losses.rolling(window=period).mean()
            
            # Calculate RS and RSI
            rs = avg_gains / avg_losses
            rsi = 100 - (100 / (1 + rs))
            
            # Fill NaN values with neutral RSI
            rsi = rsi.fillna(50)
            
            return rsi
            
        except Exception as e:
            self.logger.error(f"Error calculating RSI: {str(e)}")
            return pd.Series(50, index=prices.index)

    async def _calculate_macd(self, prices, fast=12, slow=26, signal=9):
        """Calculate MACD"""
        try:
            # Calculate EMAs
            fast_ema = prices.ewm(span=fast, adjust=False).mean()
            slow_ema = prices.ewm(span=slow, adjust=False).mean()
            
            # Calculate MACD line
            macd_line = fast_ema - slow_ema
            
            # Calculate signal line
            signal_line = macd_line.ewm(span=signal, adjust=False).mean()
            
            # Calculate MACD histogram
            macd_hist = macd_line - signal_line
            
            # Fill NaN values
            macd_hist = macd_hist.fillna(0)
            
            return macd_hist
            
        except Exception as e:
            self.logger.error(f"Error calculating MACD: {str(e)}")
            return pd.Series(0, index=prices.index)

    async def get_positions_data(self):
        """Get open positions data"""
        try:
            if not self._initialize_mt5():
                return None
                
            positions = mt5.positions_get()
            if positions is None:
                return pd.DataFrame()
                
            positions_data = []
            for pos in positions:
                positions_data.append({
                    'ticket': pos.ticket,
                    'time': datetime.fromtimestamp(pos.time),
                    'symbol': pos.symbol,
                    'type': pos.type,  # 0=BUY, 1=SELL
                    'volume': pos.volume,
                    'price_open': pos.price_open,
                    'sl': pos.sl,
                    'tp': pos.tp,
                    'price_current': pos.price_current,
                    'profit': pos.profit,
                    'swap': pos.swap,
                    'magic': pos.magic
                })
                
            return pd.DataFrame(positions_data)
            
        except Exception as e:
            self.logger.error(f"Error getting positions data: {str(e)}")
            return pd.DataFrame()
        finally:
            mt5.shutdown()

    async def get_orders_history(self, days=30):
        """Get orders history"""
        try:
            if not self._initialize_mt5():
                return None
                
            from_date = datetime.now() - timedelta(days=days)
            
            # Get trade history
            trades = mt5.history_deals_get(from_date)
            if trades is None:
                return pd.DataFrame()
                
            trades_data = []
            for trade in trades:
                trades_data.append({
                    'ticket': trade.ticket,
                    'time': datetime.fromtimestamp(trade.time),
                    'symbol': trade.symbol,
                    'type': trade.type,
                    'volume': trade.volume,
                    'price': trade.price,
                    'profit': trade.profit,
                    'commission': trade.commission,
                    'swap': trade.swap,
                    'magic': trade.magic,
                    'position_id': trade.position_id
                })
                
            return pd.DataFrame(trades_data)
            
        except Exception as e:
            self.logger.error(f"Error getting orders history: {str(e)}")
            return pd.DataFrame()
        finally:
            mt5.shutdown()