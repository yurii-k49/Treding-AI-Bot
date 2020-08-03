# trading/strategy.py
import numpy as np
from datetime import datetime
import logging

class TradingStrategy:
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger('trading.strategy')
        
    def generate_signals(self, analysis):
        """Generate trading signals from analysis"""
        # Barcha tahlillarni birlashtirish
        technical_signal = self._process_technical_signals(analysis['technical'])
        fundamental_signal = self._process_fundamental_signals(analysis['fundamental'])
        sentiment_signal = self._process_sentiment_signals(analysis['sentiment'])
        
        # Signallarni vaznlar bilan birlashtirish
        weights = self.config.SIGNAL_WEIGHTS
        combined_signal = (
            technical_signal * weights['technical'] +
            fundamental_signal * weights['fundamental'] +
            sentiment_signal * weights['sentiment']
        )
        
        return {
            'direction': np.sign(combined_signal),
            'strength': abs(combined_signal),
            'technical': technical_signal,
            'fundamental': fundamental_signal,
            'sentiment': sentiment_signal
        }
    
    def _process_technical_signals(self, technical_analysis):
        """Process technical analysis signals"""
        signals = technical_analysis['signals']
        
        # Trend signals
        trend_signal = signals['trend'][-1]
        macd_signal = signals['macd'][-1]
        rsi_signal = signals['rsi'][-1]
        
        # Pattern signals
        patterns = technical_analysis['patterns']
        pattern_signal = self._evaluate_patterns(patterns)
        
        return np.mean([trend_signal, macd_signal, rsi_signal, pattern_signal])
    
    def _evaluate_patterns(self, patterns):
        """Evaluate chart patterns"""
        pattern_scores = []
        
        if patterns['engulfing']:
            pattern_scores.append(patterns['engulfing'][-1] * 1.2)  # Higher weight
            
        if patterns['doji']:
            pattern_scores.append(patterns['doji'][-1])
            
        if patterns['hammer']:
            pattern_scores.append(patterns['hammer'][-1] * 1.1)
            
        return np.mean(pattern_scores) if pattern_scores else 0

# trading/risk_manager.py
class RiskManager:
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger('trading.risk')
        
    def check_risk_limits(self, account_info, new_position):
        """Check if new position meets risk limits"""
        # Kunlik zarar limiti
        if self._check_daily_loss_limit(account_info):
            return False
            
        # Pozitsiya hajmi limiti
        if not self._check_position_size_limit(new_position):
            return False
            
        # Margin requirements
        if not self._check_margin_requirements(account_info, new_position):
            return False
            
        return True
        
    def calculate_position_size(self, signal_strength, account_info):
        """Calculate optimal position size"""
        base_lot = self.config.LOT_SIZE
        
        # Risk-based lot size
        account_risk = self._calculate_account_risk(account_info)
        risk_lot = base_lot * account_risk
        
        # Signal-based adjustment
        signal_lot = risk_lot * signal_strength
        
        # Apply limits
        final_lot = min(signal_lot, self.config.MAX_POSITION_SIZE)
        
        return final_lot
        
    def calculate_stop_loss(self, entry_price, direction, atr):
        """Calculate stop loss level"""
        atr_multiplier = 1.5
        
        if direction > 0:  # Long position
            stop_loss = entry_price - (atr * atr_multiplier)
        else:  # Short position
            stop_loss = entry_price + (atr * atr_multiplier)
            
        return stop_loss

# trading/broker.py
import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime
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
            
        self.logger.info("MT5 initialized successfully")
        
    async def execute_trades(self, signals):
        """Execute trading signals"""
        for signal in signals:
            try:
                if signal['direction'] != 0:
                    await self._place_order(
                        symbol=signal['symbol'],
                        order_type=mt5.ORDER_TYPE_BUY if signal['direction'] > 0 else mt5.ORDER_TYPE_SELL,
                        lot_size=signal['lot_size'],
                        stop_loss=signal['stop_loss'],
                        take_profit=signal['take_profit']
                    )
            except Exception as e:
                self.logger.error(f"Order execution error: {str(e)}")
                
    async def _place_order(self, symbol, order_type, lot_size, stop_loss, take_profit):
        """Place order in MT5"""
        point = mt5.symbol_info(symbol).point
        price = mt5.symbol_info_tick(symbol).ask if order_type == mt5.ORDER_TYPE_BUY else mt5.symbol_info_tick(symbol).bid
        
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": lot_size,
            "type": order_type,
            "price": price,
            "sl": stop_loss,
            "tp": take_profit,
            "deviation": 20,
            "magic": 234000,
            "comment": "AI trader",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        
        result = mt5.order_send(request)
        
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            raise Exception(f"Order failed: {result.comment}")
        
        self.logger.info(f"Order placed successfully: {result.order}")
        return result.order