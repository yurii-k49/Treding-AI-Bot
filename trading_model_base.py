# trading_model_base.py
import numpy as np
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, BatchNormalization
from keras.optimizers import Adam
import MetaTrader5 as mt5
from collections import deque
from datetime import datetime
import multiprocessing as mp
import os

class BaseModel:
   def __init__(self):
       self.cpu_cores = mp.cpu_count() - 1
       self.memory = deque(maxlen=10000 * self.cpu_cores)
       self.gamma = 0.95
       self.epsilon = 1.0 
       self.epsilon_min = 0.01
       self.epsilon_decay = 0.995
       self.batch_size = 64 * self.cpu_cores
       self.model = self._build_model()

   def _build_model(self):
       model = Sequential([
           LSTM(256, input_shape=(50, 8), return_sequences=True),
           BatchNormalization(),
           Dropout(0.2),
           
           LSTM(128, return_sequences=True),
           BatchNormalization(), 
           Dropout(0.2),
           
           LSTM(64),
           BatchNormalization(),
           Dropout(0.2),
           
           Dense(32, activation='relu'),
           BatchNormalization(),
           
           Dense(16, activation='relu'),
           BatchNormalization(),
           
           Dense(3, activation='softmax')
       ])
       
       model.compile(
           optimizer=Adam(learning_rate=0.001),
           loss='categorical_crossentropy',
           metrics=['accuracy']
       )
       return model

   def _get_action(self, state):
       if not isinstance(state, np.ndarray):
           state = np.array(state)
           
       if len(state.shape) == 2:
           state = np.expand_dims(state, axis=0)
           
       if np.random.random() <= self.epsilon:
           action = np.random.randint(0, 3)
       else:
           q_values = self.model.predict(state, verbose=0)
           action = np.argmax(q_values[0])
           
       if self.epsilon > self.epsilon_min:
           self.epsilon *= self.epsilon_decay
           
       return action

   def save_model(self, filename):
       try:
           filename = filename.replace('.h5', '.keras')
           self.model.save(filename)
           print(f"✅ Model saqlandi: {filename}")
       except Exception as e:
           print(f"❌ Model saqlanmadi: {str(e)}")

   def load_model(self, filename):
       try:
           self.model = keras.models.load_model(filename)
           print(f"✅ Model yuklandi: {filename}")
       except Exception as e:
           print(f"❌ Model yuklanmadi: {str(e)}")

   @staticmethod
   def calculate_ema(prices, period):
       alpha = 2.0 / (period + 1)
       ema = np.zeros_like(prices)
       ema[0] = prices[0]
       
       for i in range(1, len(prices)):
           ema[i] = alpha * prices[i] + (1 - alpha) * ema[i-1]
       return ema

   @staticmethod 
   def calculate_rsi(prices, period=14):
       deltas = np.diff(prices)
       seed = deltas[:period+1]
       up = seed[seed >= 0].sum()/period
       down = -seed[seed < 0].sum()/period
       rs = up/down
       rsi = np.zeros_like(prices)
       rsi[period] = 100 - 100/(1+rs)
       
       for i in range(period+1, len(prices)):
           delta = deltas[i-1]
           upval = delta if delta > 0 else 0
           downval = -delta if delta < 0 else 0
               
           up = (up*(period-1) + upval)/period
           down = (down*(period-1) + downval)/period
           rs = up/down
           rsi[i] = 100 - 100/(1+rs)
           
       return rsi

   @staticmethod
   def calculate_atr(rates, period=14):
       high, low, close = rates['high'], rates['low'], rates['close']
       
       tr1 = np.abs(high - low)
       tr2 = np.abs(high - np.roll(close, 1))
       tr3 = np.abs(low - np.roll(close, 1))
       
       tr = np.maximum(tr1, np.maximum(tr2, tr3))
       tr[0] = tr1[0]
       
       atr = np.zeros_like(tr)
       atr[0] = tr[0]
       
       for i in range(1, len(tr)):
           atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
           
       return atr

class MarketAnalyzer:
   @staticmethod
   def analyze_trend(ema20, ema50, ema200):
       if (ema20[-1] > ema50[-1] > ema200[-1]) and (ema20[-2] > ema50[-2] > ema200[-2]):
           return 1
       elif (ema20[-1] < ema50[-1] < ema200[-1]) and (ema20[-2] < ema50[-2] < ema200[-2]):
           return -1
       return 0

   @staticmethod
   def analyze_volatility(atr):
       atr_mean = np.mean(atr)
       atr_std = np.std(atr)
       current_atr = atr[-1]
       
       if current_atr > atr_mean + atr_std:
           return 2
       elif current_atr > atr_mean:
           return 1
       return 0

   @staticmethod
   def analyze_session():
       hour = datetime.now().hour
       if 8 <= hour <= 16:
           return 2  # London/NY
       elif 3 <= hour <= 7 or 17 <= hour <= 22:
           return 1  # Tokyo/Sydney
       return 0