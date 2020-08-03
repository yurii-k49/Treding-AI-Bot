# trading/broker.py
import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime
import time
import logging
import asyncio

class MT5Broker:
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger('trading.broker')
        self._initialize_mt5()
        
    def _initialize_mt5(self):
        """Initialize MT5 connection"""
        if not mt5.initialize():
            raise Exception("MT5 initialization failed")
            
        # Login to account
        if not mt5.login(
            login=int(self.config.MT5_LOGIN),
            password=self.config.MT5_PASSWORD,
            server=self.config.MT5_SERVER
        ):
            raise Exception("MT5 login failed")
            
        self.logger.info(f"MT5 initialized successfully in {self.config.MODE} mode")
        
    async def execute_trades(self, signals):
        """Execute trading signals"""
        try:
            for signal in signals:
                if signal['direction'] != 0:
                    # Check existing positions
                    if self._check_existing_positions(signal['symbol']):
                        continue
                        
                    # Place new order
                    order_result = await self._place_order(
                        symbol=signal['symbol'],
                        order_type=mt5.ORDER_TYPE_BUY if signal['direction'] > 0 else mt5.ORDER_TYPE_SELL,
                        lot_size=signal['lot_size'],
                        stop_loss=signal['stop_loss'],
                        take_profit=signal['take_profit']
                    )
                    
                    if order_result:
                        self.logger.info(f"Order executed successfully: {order_result}")
                    
        except Exception as e:
            self.logger.error(f"Error executing trades: {str(e)}")
            
    async def _place_order(self, symbol, order_type, lot_size, stop_loss, take_profit):
        """Place order in MT5"""
        try:
            # Get symbol info
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None:
                raise Exception(f"Symbol {symbol} not found")
                
            # Calculate price
            point = symbol_info.point
            price = mt5.symbol_info_tick(symbol).ask if order_type == mt5.ORDER_TYPE_BUY else mt5.symbol_info_tick(symbol).bid
            
            # Prepare request
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": float(lot_size),
                "type": order_type,
                "price": price,
                "sl": stop_loss,
                "tp": take_profit,
                "deviation": 20,
                "magic": 234000,
                "comment": "AI Trader",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            
            # Send order
            result = mt5.order_send(request)
            
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                raise Exception(f"Order failed: {result.comment}")
                
            return result.order
            
        except Exception as e:
            self.logger.error(f"Error placing order: {str(e)}")
            return None
            
    def _check_existing_positions(self, symbol):
        """Check if there are existing positions for the symbol"""
        try:
            positions = mt5.positions_get(symbol=symbol)
            return positions is not None and len(positions) > 0
        except Exception as e:
            self.logger.error(f"Error checking positions: {str(e)}")
            return True  # Conservative approach
            
    def get_account_info(self):
        """Get account information"""
        try:
            account_info = mt5.account_info()
            if account_info is None:
                raise Exception("Failed to get account info")
                
            return {
                'balance': account_info.balance,
                'equity': account_info.equity,
                'margin': account_info.margin,
                'free_margin': account_info.margin_free,
                'margin_level': account_info.margin_level,
                'daily_pnl': self._calculate_daily_pnl()
            }
            
        except Exception as e:
            self.logger.error(f"Error getting account info: {str(e)}")
            return None
            
    def _calculate_daily_pnl(self):
        """Calculate daily profit/loss"""
        try:
            today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
            
            # Get today's closed positions
            history_deals = mt5.history_deals_get(today, datetime.now())
            if history_deals is None:
                return 0
                
            # Calculate profit
            profit = sum(deal.profit for deal in history_deals)
            
            # Add floating P/L from open positions
            positions = mt5.positions_get()
            if positions is not None:
                floating_pl = sum(pos.profit for pos in positions)
                profit += floating_pl
                
            return profit
            
        except Exception as e:
            self.logger.error(f"Error calculating daily PnL: {str(e)}")
            return 0
            
    def modify_position(self, position_id, stop_loss=None, take_profit=None):
        """Modify existing position"""
        try:
            position = mt5.positions_get(ticket=position_id)
            if position is None or len(position) == 0:
                raise Exception("Position not found")
                
            position = position[0]
            
            request = {
                "action": mt5.TRADE_ACTION_SLTP,
                "symbol": position.symbol,
                "position": position_id,
                "sl": stop_loss if stop_loss is not None else position.sl,
                "tp": take_profit if take_profit is not None else position.tp
            }
            
            result = mt5.order_send(request)
            
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                raise Exception(f"Modification failed: {result.comment}")
                
            return True
            
        except Exception as e:
            self.logger.error(f"Error modifying position: {str(e)}")
            return False
            
    def close_position(self, position_id):
        """Close specific position"""
        try:
            position = mt5.positions_get(ticket=position_id)
            if position is None or len(position) == 0:
                raise Exception("Position not found")
                
            position = position[0]
            
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": position.symbol,
                "volume": position.volume,
                "type": mt5.ORDER_TYPE_SELL if position.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY,
                "position": position_id,
                "price": mt5.symbol_info_tick(position.symbol).bid if position.type == mt5.ORDER_TYPE_BUY else mt5.symbol_info_tick(position.symbol).ask,
                "deviation": 20,
                "magic": 234000,
                "comment": "Close by AI Trader",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            
            result = mt5.order_send(request)
            
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                raise Exception(f"Close failed: {result.comment}")
                
            return True
            
        except Exception as e:
            self.logger.error(f"Error closing position: {str(e)}")
            return False
            
    def cleanup(self):
        """Cleanup MT5 connection"""
        mt5.shutdown()
        self.logger.info("MT5 connection closed")