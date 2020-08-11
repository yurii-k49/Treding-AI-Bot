# trading_model.py
from trading_model_base import BaseModel, MarketAnalyzer
import MetaTrader5 as mt5
import numpy as np
from datetime import datetime
import os
import json

class TradingModel(BaseModel):
   def __init__(self, symbol="EURUSD", timeframe=mt5.TIMEFRAME_M5):
       super().__init__()
       self.symbol = symbol
       self.timeframe = timeframe
       self.initial_balance = 10.0
       self.current_episode = 0 
       self.total_episodes = 100000
       self.training_state_file = 'training_state.json'
       self.failed_episodes = []
       
       # Strategy configs
       self.strategies = {
           'Scalping': {
               'interval': 12,
               'max_orders': 3,
               'tp_multiplier': 2.0,
               'sl_multiplier': 1.0,
               'risk_percent': 2.0
           },
           'Breakout': { 
               'interval': 72,
               'max_orders': 2,
               'tp_multiplier': 3.0,
               'sl_multiplier': 1.5,
               'risk_percent': 3.0
           },
           'OrderBlock': {
               'interval': 144,
               'max_orders': 1,
               'tp_multiplier': 2.5,
               'sl_multiplier': 1.2,
               'risk_percent': 4.0
           }
       }
       
       self.last_order_times = {strat: 0 for strat in self.strategies}
       self.market_conditions = {
           'trend_strength': 0,
           'volatility': 0, 
           'session_activity': 0
       }
       
       # Point value for profit calculation
       self.point_value = mt5.symbol_info(symbol).point
       
       if not mt5.initialize():
           raise Exception("MT5 ishga tushmadi!")
           
       self._load_training_state()

   def _save_training_state(self):
       state = {
           'current_episode': self.current_episode,
           'epsilon': self.epsilon,
           'failed_episodes': self.failed_episodes,
           'last_model': f'models/model_episode_{self.current_episode}.keras' 
       }
       with open(self.training_state_file, 'w') as f:
           json.dump(state, f)

   def _load_training_state(self):
       if os.path.exists(self.training_state_file):
           with open(self.training_state_file, 'r') as f:
               state = json.load(f)
               self.current_episode = state['current_episode']
               self.epsilon = state['epsilon']
               self.failed_episodes = state.get('failed_episodes', [])
               if os.path.exists(state.get('last_model')):
                   self.load_model(state['last_model'])

   def analyze_market_conditions(self, rates):
       ema20 = self.calculate_ema(rates['close'], 20)
       ema50 = self.calculate_ema(rates['close'], 50)
       ema200 = self.calculate_ema(rates['close'], 200)
       atr = self.calculate_atr(rates)
       
       self.market_conditions = {
           'trend_strength': MarketAnalyzer.analyze_trend(ema20, ema50, ema200),
           'volatility': MarketAnalyzer.analyze_volatility(atr),
           'session_activity': MarketAnalyzer.analyze_session()
       }

   def get_market_state(self):
       rates = mt5.copy_rates_from_pos(self.symbol, self.timeframe, 0, 50)
       if rates is None:
           return None, 0
           
       rates = np.array(rates)
       return self._process_market_data(rates), self.calculate_atr(rates)[-1]

   def _process_market_data(self, rates):
       processed = np.column_stack((
           rates['open'],
           rates['high'], 
           rates['low'],
           rates['close'],
           self.calculate_ema(rates['close'], 50),
           self.calculate_rsi(rates['close']),
           self.calculate_atr(rates),
           self.calculate_ema(rates['close'], 200)
       ))
       return np.nan_to_num(processed, nan=0.0)

   def _get_risk_per_trade(self, strategy):
       return self.strategies[strategy]['risk_percent'] / 100.0

   def get_dynamic_tp_sl(self, strategy, atr):
       strategy_params = self.strategies[strategy]
       
       # Adjust multipliers based on conditions
       if abs(self.market_conditions['trend_strength']) == 1:
           tp_mult = strategy_params['tp_multiplier'] * 1.2
           sl_mult = strategy_params['sl_multiplier'] * 1.2
       elif self.market_conditions['volatility'] == 2:
           tp_mult = strategy_params['tp_multiplier'] * 0.8 
           sl_mult = strategy_params['sl_multiplier'] * 0.8
       else:
           tp_mult = strategy_params['tp_multiplier']
           sl_mult = strategy_params['sl_multiplier']

       tp_points = int(atr * tp_mult / self.point_value)
       sl_points = int(atr * sl_mult / self.point_value)
       
       return tp_points, sl_points