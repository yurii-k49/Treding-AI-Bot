# utils/data_loader.py
import pandas as pd
import numpy as np
import MetaTrader5 as mt5
import logging
from datetime import datetime, timedelta

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
        """MT5 terminalini ishga tushirish"""
        try:
            if not mt5.initialize(
                login=self.config.MT5_LOGIN,
                password=self.config.MT5_PASSWORD,
                server=self.config.MT5_SERVER
            ):
                self.logger.error(f"MT5 ishga tushmadi: {mt5.last_error()}")
                return False
                
            if not mt5.terminal_info().connected:
                self.logger.error("MT5 serverga ulanmagan!")
                return False
                
            return True
        except Exception as e:
            self.logger.error(f"MT5 ishga tushmadi: {str(e)}")
            return False
            
    async def get_historical_data(self, symbol=None, timeframe=None, bars=1000):
        """Tarixiy narx ma'lumotlarini olish"""
        try:
            if not self._initialize_mt5():
                return None
                
            symbol = symbol or self.symbol
            timeframe = timeframe or self.timeframe
            
            rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, bars)
            if rates is None:
                self.logger.error(f"Ma'lumotlarni ololmadik: {mt5.last_error()}")
                return None
                
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            return df
            
        finally:
            mt5.shutdown()
            
    async def get_market_data(self, symbol=None):
        """Joriy bozor ma'lumotlarini olish"""
        try:
            if not self._initialize_mt5():
                return None
                
            symbol = symbol or self.symbol
            symbol_info = mt5.symbol_info(symbol)
            
            if symbol_info is None:
                return None
                
            tick = mt5.symbol_info_tick(symbol)
            
            return {
                'symbol': symbol,
                # Asosiy narx ma'lumotlari
                'bid': tick.bid,
                'ask': tick.ask,
                'last': tick.last,
                'spread': tick.ask - tick.bid,
                'volume': tick.volume,
                'volume_real': tick.volume_real,
                'time': datetime.fromtimestamp(tick.time),
                
                # Symbol ma'lumotlari
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
                'order_mode': symbol_info.order_mode,
                'tick_value': symbol_info.trade_tick_value,
                'tick_size': symbol_info.trade_tick_size
            }
            
        finally:
            mt5.shutdown()
    
            
    async def get_positions_data(self):
        """Ochiq pozitsiyalar ma'lumotlarini olish"""
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
                    'sl': pos.sl,  # stop loss
                    'tp': pos.tp,  # take profit
                    'price_current': pos.price_current,
                    'comment': pos.comment,
                    'profit': pos.profit,
                    'swap': pos.swap,
                    'magic': pos.magic
                })
                
            return pd.DataFrame(positions_data)
            
        finally:
            mt5.shutdown()
            
    async def get_orders_history(self, days=30):
        """Order'lar tarixini olish"""
        try:
            if not self._initialize_mt5():
                return None
                
            from_date = datetime.now() - timedelta(days=days)
            
            # Savdo tarixi
            deals = mt5.history_deals_get(from_date)
            if deals is None:
                return pd.DataFrame()
                
            deals_data = []
            for deal in deals:
                deals_data.append({
                    'ticket': deal.ticket,
                    'time': datetime.fromtimestamp(deal.time),
                    'symbol': deal.symbol,
                    'type': deal.type,  # 0=BUY, 1=SELL
                    'volume': deal.volume,
                    'price': deal.price,
                    'profit': deal.profit,
                    'comment': deal.comment,
                    'commission': deal.commission,
                    'swap': deal.swap,
                    'magic': deal.magic,
                    'position_id': deal.position_id
                })
                
            return pd.DataFrame(deals_data)
            
        finally:
            mt5.shutdown()
        
    async def get_training_data(self, symbol=None, timeframe=None, bars=10000):
        """Model training uchun barcha ma'lumotlarni olish"""
        try:
            self.logger.info(f"{bars} bar uchun ma'lumotlar olinmoqda...")
            
            # Asosiy ma'lumotlar
            historical_df = await self.get_historical_data(symbol, timeframe, bars)
            if historical_df is None or historical_df.empty:
                raise RuntimeError("Tarixiy ma'lumotlarni ololmadik")
                
            # Qo'shimcha ma'lumotlar
            market_data = await self.get_market_data(symbol)
            positions_data = await self.get_positions_data()
            orders_history = await self.get_orders_history()
            
            # Market ma'lumotlarini DataFrame ga o'tkazish
            market_df = pd.DataFrame([market_data]) if market_data else pd.DataFrame()
            
            # Bo'sh DataFrame lar yaratish
            positions_df = positions_data if isinstance(positions_data, pd.DataFrame) else pd.DataFrame()
            orders_df = orders_history if isinstance(orders_history, pd.DataFrame) else pd.DataFrame()
            
            self.logger.info(f"Ma'lumotlar olindi:")
            self.logger.info(f"- Tarixiy ma'lumotlar: {len(historical_df)} qator")
            self.logger.info(f"- Ochiq pozitsiyalar: {len(positions_df)} ta")
            self.logger.info(f"- Orderlar tarixi: {len(orders_df)} ta")
            
            # Market va pozitsiyalar ma'lumotlaridan qo'shimcha indikatorlar yaratish
            market_indicators = self._calculate_market_indicators(historical_df, market_df)
            
            return {
                "historical": historical_df,
                "market": market_indicators,
                "positions": positions_df,
                "orders_history": orders_df
            }
            
        except Exception as e:
            self.logger.error(f"Training ma'lumotlarini olishda xatolik: {str(e)}")
            raise

    def _calculate_market_indicators(self, historical_df, market_df):
        """Bozor ma'lumotlaridan qo'shimcha indikatorlar hisoblash"""
        try:
            if historical_df.empty:
                return pd.DataFrame()
                
            indicators = pd.DataFrame()
            
            # Asosiy indikatorlar
            indicators['volatility_daily'] = historical_df['high'] - historical_df['low']
            indicators['volume_ma'] = historical_df['tick_volume'].rolling(window=20).mean()
            indicators['spread_avg'] = historical_df['spread'].rolling(window=20).mean()
            
            # Narx o'zgarishlari
            indicators['price_change'] = historical_df['close'].pct_change()
            indicators['price_change_ma'] = indicators['price_change'].rolling(window=20).mean()
            
            # Volatillik indikatorlari
            indicators['volatility'] = indicators['price_change'].rolling(window=20).std()
            indicators['volatility_ma'] = indicators['volatility'].rolling(window=20).mean()
            
            # Market ma'lumotlaridan indikatorlar
            if not market_df.empty:
                last_row = market_df.iloc[-1]
                indicators['current_spread'] = last_row.get('spread', 0)
                indicators['volume_current'] = last_row.get('volume', 0)
            
            return indicators
            
        except Exception as e:
            self.logger.error(f"Indikatorlarni hisoblashda xatolik: {str(e)}")
            return pd.DataFrame()
        
    async def get_validation_data(self, symbol=None, timeframe=None, bars=1000):
        """Validatsiya uchun ma'lumotlarni olish"""
        try:
            self.logger.info("Validatsiya ma'lumotlari olinmoqda...")
            
            # Validatsiya uchun oxirgi 1000 ta ma'lumotni olish
            historical_df = await self.get_historical_data(symbol, timeframe, bars)
            if historical_df is None or historical_df.empty:
                raise RuntimeError("Validatsiya uchun tarixiy ma'lumotlarni ololmadik")
            
            # Market ma'lumotlarini olish
            market_data = await self.get_market_data(symbol)
            market_df = pd.DataFrame([market_data]) if market_data else pd.DataFrame()
            
            # Positions va orders ma'lumotlarini olish
            positions_data = await self.get_positions_data()
            orders_history = await self.get_orders_history(days=7)
            
            self.logger.info(f"Validatsiya ma'lumotlari olindi:")
            self.logger.info(f"- Tarixiy ma'lumotlar: {len(historical_df)} qator")
            self.logger.info(f"- Ochiq pozitsiyalar: {len(positions_data)} ta")
            self.logger.info(f"- Orderlar tarixi: {len(orders_history)} ta")
            
            return {
                "historical": historical_df,
                "market": market_df,
                "positions": positions_data,
                "orders_history": orders_history
            }
            
        except Exception as e:
            self.logger.error(f"Validatsiya ma'lumotlarini olishda xatolik: {str(e)}")
            return None

    async def get_validation_fundamental_data(self, symbol=None):
        """Bu metod hozircha bo'sh DataFrame qaytaradi"""
        return pd.DataFrame()

    async def get_validation_sentiment_data(self, symbol=None):
        """Bu metod hozircha bo'sh DataFrame qaytaradi"""
        return pd.DataFrame()