# trading_execution.py
from trading_model import TradingModel
import MetaTrader5 as mt5
import numpy as np
from datetime import datetime
import time
from typing import Dict, Optional, Tuple
import random

class TradingExecution(TradingModel):
    def __init__(self, symbol: str = "BTCUSD", timeframe: int = mt5.TIMEFRAME_M5):
        super().__init__(symbol, timeframe)
        self.last_check_time = time.time()
        self.market_check_interval = 60
        
        if not self.initialize_mt5():
            raise Exception("MT5 bilan bog'lanishda xatolik!")
            
    def initialize_mt5(self) -> bool:
        if not mt5.initialize():
            return False
            
        account_info = mt5.account_info()
        if account_info is None:
            return False
            
        symbol_info = mt5.symbol_info(self.symbol)
        if symbol_info is None:
            return False
            
        if not symbol_info.visible:
            if not mt5.symbol_select(self.symbol, True):
                return False
                
        return True
        
    def execute_trade(self, action: int) -> float:
        strategy = list(self.strategies.keys())[action]
        if not self.can_open_new_order(strategy):
            return 0
        
        try:
            rates = mt5.copy_rates_from_pos(self.symbol, self.timeframe, 0, 50)
            if rates is None:
                return 0
                
            rates = np.array(rates)
            atr = self.calculate_atr(rates)[-1]
            tp_points, sl_points = self.get_dynamic_tp_sl(strategy, atr)
            
            account_info = mt5.account_info()
            if account_info is None:
                return 0
                
            balance = account_info.balance
            risk_per_trade = self._get_risk_per_trade(strategy)
            
            volume = round((balance * risk_per_trade) / (sl_points * 0.0001), 3)
            volume = max(0.01, min(volume, 1.0))
            
            price_info = mt5.symbol_info_tick(self.symbol)
            if price_info is None:
                return 0
                
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": self.symbol,
                "volume": volume,
                "type": mt5.ORDER_TYPE_BUY,
                "price": price_info.ask,
                     "sl": price_info.ask - sl_points * 0.0001,
                "tp": price_info.ask + tp_points * 0.0001,
                "comment": f"AI_{strategy}",
                "type_filling": mt5.ORDER_FILLING_IOC,
                "deviation": 20,
            }
            
            result = mt5.order_send(request)
            if result.retcode == mt5.TRADE_RETCODE_DONE:
                self.last_order_times[strategy] = time.time()
                potential_reward = (request['tp'] - request['price']) * volume * 100000
                return potential_reward
            else:
                print(f"Order xatoligi: {result.comment}")
                return 0
                    
        except Exception as e:
            print(f"Order ochishda xatolik: {e}")
            return 0

    def trade(self) -> None:
        print(f"\nTrading rejimi boshlandi - {self.symbol}")
        
        while True:
            try:
                state, _ = self.get_market_state()
                if state is None:
                    time.sleep(1)
                    continue
                
                action = np.argmax(self.model.predict(
                    state.reshape(1, 50, 8), verbose=0)[0]
                )
                
                reward = self.execute_trade(action)
                if reward != 0:
                    strategies = list(self.strategies.keys())
                    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    print(f"\n{current_time}:")
                    print(f"  Strategy: {strategies[action]}")
                    print(f"  Result: {reward}")
                
                if time.time() - self.last_check_time > self.market_check_interval:
                    self.analyze_market_conditions(
                        np.array(mt5.copy_rates_from_pos(self.symbol, self.timeframe, 0, 50))
                    )
                    print("\nMarket holati:")
                    print(f"  Trend kuchi: {self.market_conditions['trend_strength']}")
                    print(f"  Volatillik: {self.market_conditions['volatility']}")
                    print(f"  Sessiya faolligi: {self.market_conditions['session_activity']}")
                    self.last_check_time = time.time()
                
                time.sleep(1)
                
            except Exception as e:
                print(f"Xatolik yuz berdi: {e}")
                time.sleep(5)
                continue

    def can_open_new_order(self, strategy: str) -> bool:
        positions = mt5.positions_get(symbol=self.symbol)
        if positions is None:
            return False
            
        strategy_positions = [pos for pos in positions 
                            if pos.comment.startswith(f"AI_{strategy}")]
                            
        if len(strategy_positions) >= self.strategies[strategy]['max_orders']:
            return False
            
        current_time = time.time()
        if current_time - self.last_order_times[strategy] < self.strategies[strategy]['interval']:
            return False
            
        if time.time() - self.last_check_time > self.market_check_interval:
            rates = mt5.copy_rates_from_pos(self.symbol, self.timeframe, 0, 50)
            if rates is not None:
                self.analyze_market_conditions(np.array(rates))
                self.last_check_time = time.time()
        
        if strategy == 'Scalping':
            return self.market_conditions['volatility'] < 2 and \
                   self.market_conditions['session_activity'] > 0
        elif strategy == 'Breakout':
            return self.market_conditions['volatility'] > 0 and \
                   self.market_conditions['session_activity'] > 1
        elif strategy == 'OrderBlock':
            return abs(self.market_conditions['trend_strength']) == 1 and \
                   self.market_conditions['session_activity'] > 0
                    
        return True

    def _get_risk_per_trade(self, strategy: str) -> float:
        base_risk = 0.005  # 0.5% asosiy risk
        
        trend_factor = 1.2 if abs(self.market_conditions['trend_strength']) == 1 else 0.8
        volatility_factor = {
            2: 0.7,  # Yuqori volatillik
            1: 0.9,  # O'rta volatillik
            0: 0.8   # Past volatillik
        }[self.market_conditions['volatility']]
        
        strategy_factor = {
            'Scalping': 0.8,
            'Breakout': 1.2,
            'OrderBlock': 1.0
        }[strategy]
        
        risk = base_risk * trend_factor * volatility_factor * strategy_factor
        return min(risk, 0.01)  # Maksimal 1% risk