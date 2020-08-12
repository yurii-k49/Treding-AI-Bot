from trading_model import TradingModel
from collections import deque
import MetaTrader5 as mt5
import numpy as np
from datetime import datetime, timedelta
import os
import logging
from typing import Optional, Dict, List, Tuple, Any
import json
import time

class HistoricalTrader(TradingModel):
    def __init__(self, symbol: str = "EURUSD", timeframe: int = mt5.TIMEFRAME_M5):
        super().__init__(symbol, timeframe)
        self.historical_data = None
        self.current_index = 50
        self.initial_balance = 100.0
        self.virtual_balance = self.initial_balance
        self.open_positions: List[Dict] = []
        self.trade_history: List[Dict] = []
        self.logger = self._setup_logger()
        
        # Strategiya parametrlari optimallashtirildi
        self.strategies = {
           'Scalping': {
                'interval': 12,
                'max_orders': 2,
                'tp_multiplier': 2.5,  # Ko'proq foyda
                'sl_multiplier': 1.0,
                'risk_percent': 2.0,  # Risk oshirildi
                'min_volume': 0.05,  # Min lot oshirildi
                'max_volume': 0.2   # Max lot oshirildi
            },
            'Breakout': {
                'interval': 8,
                'max_orders': 2,
                'tp_multiplier': 2.5,  # TP/SL=2.5 (25/10 punkt)
                'sl_multiplier': 1.0,
                'risk_percent': 1.2,
                'min_volume': 0.01,
                'max_volume': 0.1
            },
            'OrderBlock': {
                'interval': 12,
                'max_orders': 2,
                'tp_multiplier': 3.0,  # TP/SL=3 (30/10 punkt)
                'sl_multiplier': 1.0,
                'risk_percent': 1.5,
                'min_volume': 0.01,
                'max_volume': 0.1
            }
        }
        self.last_order_times = {strat: 0 for strat in self.strategies}
        self.market_conditions = {
            'trend_strength': 0,
            'volatility': 0,
            'session_activity': 0,
            'volume_ratio': 0,
            'atr': 0
        }

    @staticmethod
    def _setup_logger() -> logging.Logger:
        """Trading bot uchun logger sozlash"""
        logger = logging.getLogger('HistoricalTrader')
        logger.setLevel(logging.INFO)
        
        if not os.path.exists('logs'):
            os.makedirs('logs')
            
        log_file = f'logs/trading_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.INFO)
        
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        
        return logger

    def load_historical_data(self, days: int = 30) -> bool:
        """Tarixiy ma'lumotlarni yuklash"""
        print("\n=== LOADING HISTORICAL DATA ===")
        
        try:
            if not mt5.initialize():
                self.logger.error("MT5 initialization failed")
                return False
                
            if not mt5.symbol_select(self.symbol, True):
                self.logger.error(f"Failed to select symbol {self.symbol}")
                return False
                
            utc_from = datetime.now() - timedelta(days=days)
            rates = mt5.copy_rates_from(self.symbol, self.timeframe, utc_from, days * 24 * 12)
            
            if rates is None or len(rates) == 0:
                self.logger.error("Failed to load historical data")
                return False
                
            self.historical_data = np.array(rates)
            
            print(f"✅ Loaded: {len(rates)} bars")
            print(f"📅 First date: {datetime.fromtimestamp(rates[0]['time'])}")
            print(f"📅 Last date: {datetime.fromtimestamp(rates[-1]['time'])}")
            
            self.logger.info(f"Successfully loaded {len(rates)} bars of historical data")
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading historical data: {str(e)}")
            return False
        
    def train_on_historical(self, episodes: int = 100) -> None:
        """Trade botni tarixiy ma'lumotlarda o'qitish"""
        if not os.path.exists('models'):
            os.makedirs('models')
            
        print("\n=== STARTING TRAINING ===")
        print(f"📊 Total bars: {len(self.historical_data)}")
        print(f"🔄 Episodes: {episodes}")
        print(f"💰 Initial balance: ${self.initial_balance:,.2f}\n")
        
        # O'qitish parametrlari
        epsilon = 1.0
        epsilon_min = 0.01
        epsilon_decay = 0.995
        memory = deque(maxlen=2000)
        batch_size = 64
        gamma = 0.95
        
        best_profit = float('-inf')
        best_episode = -1
        episodes_without_improvement = 0
        max_episodes_without_improvement = 10
        
        for episode in range(episodes):
            # Episode uchun yangi holat
            self.virtual_balance = self.initial_balance
            self.current_index = 50
            self.open_positions.clear()
            self.trade_history.clear()
            trades_this_episode = 0
            
            print(f"\n=== EPISODE {episode + 1}/{episodes} ===")
            self.logger.info(f"Starting episode {episode + 1}")
            episode_start = time.time()
            
            try:
                while self.current_index < len(self.historical_data) - 50:
                    if self.virtual_balance <= 0:
                        print(f"❌ Margin Call! Balance: ${self.virtual_balance:.2f}")
                        self.logger.warning(f"Margin call at episode {episode + 1}")
                        break
                        
                    if trades_this_episode >= 100:
                        break
                        
                    # Get state
                    window = self.historical_data[self.current_index-50:self.current_index]
                    self.analyze_market_conditions(window)
                    current_state = self._process_market_data(window)
                    current_state = np.expand_dims(current_state, axis=0)
                    
                    # Get action
                    if np.random.random() < epsilon:
                        action = np.random.choice(len(self.strategies))
                    else:
                        q_values = self.model.predict(current_state, verbose=0)[0]
                        action = np.argmax(q_values)
                        
                    # Execute trade and get reward    
                    trade_result = self.execute_trade(action)
                    reward = self._calculate_reward(trade_result)
                    
                    # Get next state
                    next_window = self.historical_data[self.current_index-49:self.current_index+1]
                    next_state = self._process_market_data(next_window)
                    next_state = np.expand_dims(next_state, axis=0)
                    
                    # Store experience
                    memory.append((current_state, action, reward, next_state))
                    
                    # Train on batch
                    if len(memory) >= batch_size:
                        batch = np.random.choice(len(memory), batch_size, replace=False)
                        states = []
                        targets = []
                        
                        for idx in batch:
                            s, a, r, n_s = memory[idx]
                            target = self.model.predict(s, verbose=0)[0]
                            Q_future = np.max(self.model.predict(n_s, verbose=0)[0])
                            target[a] = r + gamma * Q_future
                            states.append(s[0])
                            targets.append(target)
                            
                        states_array = np.array(states)
                        self.model.fit(states_array, np.array(targets), 
                                        epochs=1, verbose=0, batch_size=batch_size)
                    
                    # Check positions
                    position_result = self.check_positions()
                    if position_result != 0:
                        self.logger.info(f"Position closed with profit: ${position_result:.2f}")
                    
                    trades_this_episode = len(self.trade_history)
                    self.current_index += 1
                
                # Episode results
                episode_time = time.time() - episode_start
                episode_profit = self.virtual_balance - self.initial_balance
                win_trades = len([t for t in self.trade_history if t['profit'] > 0])
                total_trades = len(self.trade_history)
                win_rate = (win_trades/total_trades*100) if total_trades > 0 else 0
                
                print(f"\n{'='*50}")
                print(f"📊 EPISODE {episode + 1} RESULTS:")
                print(f"⏱️ Time: {episode_time:.1f}s")
                print(f"💰 Balance: ${self.virtual_balance:.2f}")
                print(f"📈 Profit: ${episode_profit:.2f}")
                print(f"🎯 Trades: {total_trades} (Win Rate: {win_rate:.1f}%)")
                
                # Check improvement
                if episode_profit > best_profit:
                    print("✨ New best profit!")
                    best_profit = episode_profit
                    best_episode = episode + 1
                    episodes_without_improvement = 0
                    self.save_model(f'models/best_model_episode_{episode + 1}.keras')
                else:
                    episodes_without_improvement += 1
                    
                if episodes_without_improvement >= max_episodes_without_improvement:
                    print(f"\n🛑 Stopping - No improvement for {max_episodes_without_improvement} episodes")
                    break
                    
                epsilon = max(epsilon_min, epsilon * epsilon_decay)
                
            except Exception as e:
                self.logger.error(f"Error in episode {episode + 1}: {str(e)}")
                continue
            
        print("\n=== TRAINING COMPLETED ===")
        print(f"Best Model: Episode {best_episode}")
        print(f"Best Profit: ${best_profit:.2f}")

    def _process_market_data(self, window: np.ndarray) -> np.ndarray:
        """Process market data into correct shape for LSTM"""
        processed = np.column_stack((
            window['open'],
            window['high'],
            window['low'],
            window['close'],
            self.calculate_ema(window['close'], 50),
            self.calculate_rsi(window['close']),
            self.calculate_atr(window),
            self.calculate_ema(window['close'], 200)
        ))
        return np.nan_to_num(processed, nan=0.0)  # Returns shape (50, 8)

    def _get_strategy_weights(self) -> np.ndarray:
        """Strategiyalar uchun dinamik vaznlarni hisoblash"""
        volatility = self.market_conditions['volatility']
        trend = abs(self.market_conditions['trend_strength'])
        volume = self.market_conditions['volume_ratio']
        
        weights = {
            'Scalping': 0.5 if volatility < 2 and volume > 1.2 else 0.2,
            'Breakout': 0.3 if volatility >= 2 else 0.2,
            'OrderBlock': 0.2 if trend == 1 else 0.1
        }
        
        # Normalize weights
        values = np.array(list(weights.values()))
        return values / np.sum(values)
        
    def _get_state(self, window: np.ndarray) -> np.ndarray:
        """Market state'ni model input formatiga o'tkazish"""
        processed = np.column_stack((
            window['open'],
            window['high'],
            window['low'],
            window['close'],
            self.calculate_ema(window['close'], 50),
            self.calculate_rsi(window['close']),
            self.calculate_atr(window),
            self.calculate_ema(window['close'], 200)
        ))
        
        # Reshape to match model's expected input shape (None, 50, 8)
        processed = np.nan_to_num(processed, nan=0.0)
        return processed.reshape(1, 50, 8)
    
    def _calculate_reward(self, trade_result: float) -> float:
        """Savdo natijasi uchun reward hisoblash"""
        if trade_result > 0:
            return 1.0  # Foydali savdo
        elif trade_result < 0:
            return -1.0  # Zarali savdo
        return -0.1  # Savdo bo'lmagan holatda
        

    def _get_action(self, state):
        # Market holatini analiz qilish
        volatility = self.market_conditions['volatility']
        trend = abs(self.market_conditions['trend_strength'])
        volume = self.market_conditions['volume_ratio']
        
        # Strategy weights - har bir strategiya uchun vazn
        weights = {
            'Scalping': 0.4 if volatility < 2 and volume > 1.2 else 0.1,
            'Breakout': 0.3 if volatility >= 2 else 0.2,
            'OrderBlock': 0.3 if trend == 1 else 0.2
        }
        
        # Strategy tanlash - weighted random
        strategies = list(self.strategies.keys())
        probs = [weights[s] for s in strategies]
        probs = np.array(probs) / sum(probs)  # Normalize
        
        return np.random.choice(len(strategies), p=probs)

    def _get_current_state(self) -> Optional[np.ndarray]:
        """Joriy market holatini olish"""
        if self.current_index >= len(self.historical_data):
            return None
            
        try:
            window = self.historical_data[self.current_index-50:self.current_index]
            
            closes = window['close']
            processed = np.column_stack((
                window['open'],
                window['high'],
                window['low'],
                closes,
                self.calculate_ema(closes, 50),
                self.calculate_rsi(closes),
                self.calculate_atr(window),
                self.calculate_ema(closes, 200)
            ))
            
            return np.nan_to_num(processed, nan=0.0)
            
        except Exception as e:
            self.logger.error(f"Error getting current state: {str(e)}")
            return None

    def execute_trade(self, action: int) -> float:
        """Savdo bajarish"""
        if self.current_index >= len(self.historical_data):
            return 0
            
        # Strategiyani tanlash - action asosida
        strategies = list(self.strategies.keys())
        if action >= len(strategies):
            self.logger.error(f"Invalid action: {action}")
            return 0
            
        strategy = strategies[action]
        if not self.can_open_new_order(strategy):
            return 0
            
        try:
            window = self.historical_data[self.current_index-50:self.current_index]
            current_price = self.historical_data[self.current_index]['close']
            trade_time = datetime.fromtimestamp(self.historical_data[self.current_index]['time'])
            
            # Texnik indikatorlar
            ema_fast = self.calculate_ema(window['close'], 20)[-1]
            ema_slow = self.calculate_ema(window['close'], 50)[-1]
            rsi = self.calculate_rsi(window['close'])[-1]
            atr = self.calculate_atr(window)[-1]
            
            # Savdo signalini olish
            trade_type = self._get_trade_type(strategy, current_price, rsi, ema_fast, ema_slow, window)
            if not trade_type:
                return 0

            # TP/SL hisoblash
            tp_points, sl_points = self.get_dynamic_tp_sl(strategy, atr)
            
            # Lot hajmini hisoblash
            volume = self.calculate_position_size(strategy, sl_points)
            if volume <= 0:
                return 0
                
            point_value = self.point_value if self.point_value > 0 else 0.0001
            
            # Pozitsiya parametrlari
            position = {
                'ticket': len(self.open_positions) + 1,
                'type': trade_type,
                'volume': volume,
                'price': current_price,
                'sl': current_price - (sl_points * point_value) if trade_type == 'buy' else current_price + (sl_points * point_value),
                'tp': current_price + (tp_points * point_value) if trade_type == 'buy' else current_price - (tp_points * point_value),
                'strategy': strategy,
                'open_time': trade_time,
                'risk_amount': self.virtual_balance * (self.strategies[strategy]['risk_percent'] / 100)
            }
            
            # Savdo ma'lumotlarini chiqarish
            print(f"\n{'='*50}")
            print(f"🔄 NEW ORDER #{position['ticket']} | {trade_time}")
            print(f"🎯 Strategy: {strategy}")
            print(f"📈 Type: {trade_type.upper()}")
            print(f"📦 Volume: {volume:.2f} lot")
            print(f"💰 Entry: {current_price:.5f}")
            print(f"🎯 Take Profit: {position['tp']:.5f} ({tp_points} points)")
            print(f"🛑 Stop Loss: {position['sl']:.5f} ({sl_points} points)")
            print(f"💵 Risk Amount: ${position['risk_amount']:.2f} ({self.strategies[strategy]['risk_percent']}%)")
            print(f"⚖️ Current Balance: ${self.virtual_balance:.2f}")
            print(f"📊 Market Conditions:")
            print(f"   Trend: {self.market_conditions['trend_strength']}")
            print(f"   Volatility: {self.market_conditions['volatility']}")
            print(f"   Session: {self.market_conditions['session_activity']}")
            
            self.open_positions.append(position)
            self.last_order_times[strategy] = self.current_index
            
            expected_profit = (position['tp'] - current_price) * volume * 100000 if trade_type == 'buy' else (current_price - position['tp']) * volume * 100000
            return expected_profit
            
        except Exception as e:
            self.logger.error(f"Error executing trade: {str(e)}")
            return 0
        
    def check_positions(self) -> float:
        """Ochiq pozitsiyalarni tekshirish"""
        if not self.open_positions:
            return 0
            
        total_profit = 0
        current_price = self.historical_data[self.current_index]['close']
        current_time = datetime.fromtimestamp(self.historical_data[self.current_index]['time'])
        positions_to_close = []
        
        for pos in self.open_positions:
            profit = self._calculate_position_profit(pos, current_price)
            close_reason = self._check_close_conditions(pos, current_price)
            
            if close_reason:
                total_profit += profit
                self.virtual_balance += profit
                positions_to_close.append(pos)
                
                roi = (profit / pos['risk_amount']) * 100
                
                print(f"\n{'='*50}")
                print(f"📍 ORDER #{pos['ticket']} CLOSED | {current_time}")
                print(f"📊 Strategy: {pos['strategy']}")
                print(f"📈 Type: {pos['type'].upper()}")
                print(f"📦 Volume: {pos['volume']:.2f} lot")
                print(f"💰 Entry → Exit: {pos['price']:.5f} → {current_price:.5f}")
                print(f"💵 P/L: ${profit:.2f} ({roi:.1f}% ROI)")
                print(f"❗ Reason: {close_reason}")
                print(f"⚖️ New Balance: ${self.virtual_balance:.2f}")
                print(f"⏱️ Duration: {current_time - pos['open_time']}")
                
                self.trade_history.append({
                    **pos,
                    'close_price': current_price,
                    'close_time': current_time,
                    'profit': profit,
                    'roi': roi,
                    'close_reason': close_reason
                })
        
        for pos in positions_to_close:
            self.open_positions.remove(pos)
            
        return total_profit

    def _get_trade_type(self, strategy: str, price: float, rsi: float, 
                       ema_fast: float, ema_slow: float, window: np.ndarray) -> Optional[str]:
        """Savdo signalini aniqlash"""
        try:
            atr = self.calculate_atr(window)[-1]
            volume_avg = np.mean(window['tick_volume'][-20:])
            current_volume = window['tick_volume'][-1]
            momentum = window['close'][-1] - window['close'][-5]
            ema200 = self.calculate_ema(window['close'], 200)[-1]
            
            if strategy == 'Scalping':
                if (rsi < 30 and 
                    ema_fast > ema_slow and
                    price > ema200 and
                    momentum > 0 and
                    current_volume > volume_avg):
                    return 'buy'
                elif (rsi > 70 and 
                      ema_fast < ema_slow and
                      price < ema200 and
                      momentum < 0 and
                      current_volume > volume_avg):
                    return 'sell'
                    
            elif strategy == 'Breakout':
                highs = window['high'][-20:]
                lows = window['low'][-20:]
                resistance = np.max(highs[:-1])
                support = np.min(lows[:-1])
                
                if (price > resistance + atr * 0.5 and
                    current_volume > volume_avg * 1.2 and
                    momentum > 0):
                    return 'buy'
                elif (price < support - atr * 0.5 and
                      current_volume > volume_avg * 1.2 and
                      momentum < 0):
                    return 'sell'
                    
            elif strategy == 'OrderBlock':
                if (price > ema200 and
                    ema_fast > ema_slow and
                    rsi > 40 and rsi < 75 and
                    momentum > 0 and
                    current_volume > volume_avg and
                    self.market_conditions['trend_strength'] == 1):
                    return 'buy'
                elif (price < ema200 and
                      ema_fast < ema_slow and
                      rsi > 25 and rsi < 60 and
                      momentum < 0 and
                      current_volume > volume_avg and
                      self.market_conditions['trend_strength'] == -1):
                    return 'sell'
                    
            return None
            
        except Exception as e:
            self.logger.error(f"Error in trade type calculation: {str(e)}")
            return None
        
    def analyze_market_conditions(self, rates: np.ndarray) -> None:
        """Bozor holatini tahlil qilish"""
        try:
            closes = rates['close']
            highs = rates['high']
            lows = rates['low']
            volumes = rates['tick_volume']

            # Trend tahlili
            ema20 = self.calculate_ema(closes, 20)
            ema50 = self.calculate_ema(closes, 50)
            ema200 = self.calculate_ema(closes, 200)
            atr = self.calculate_atr(rates)
            
            # Kengaytirilgan volatillik
            true_range = np.maximum(
                highs - lows,
                np.maximum(
                    np.abs(highs - np.roll(closes, 1)),
                    np.abs(lows - np.roll(closes, 1))
                )
            )
            atr_standard = np.mean(true_range[-20:])
            current_volatility = true_range[-1] / atr_standard
            
            # Hajm tahlili
            volume_sma = np.mean(volumes[-20:])
            volume_ratio = volumes[-1] / volume_sma
            
            # Sessiya vaqti
            current_hour = datetime.now().hour
            
            # Trend kuchi
            if ema20[-1] > ema50[-1] > ema200[-1] and volume_ratio > 1.1:
                trend = 1  # Kuchli yuqoriga
            elif ema20[-1] < ema50[-1] < ema200[-1] and volume_ratio > 1.1:
                trend = -1  # Kuchli pastga
            else:
                trend = 0  # Aniq trend yo'q
                
            # Volatillik darajasi
            if current_volatility > 1.5:
                volatility = 2  # Yuqori
            elif current_volatility > 1.0:
                volatility = 1  # O'rta
            else:
                volatility = 0  # Past
                
            # Sessiya faolligi
            if 8 <= current_hour <= 16:  # London/NY
                session = 2 if volume_ratio > 1.2 else 1
            elif 3 <= current_hour <= 7 or 17 <= current_hour <= 22:  # Tokyo/Sydney
                session = 1 if volume_ratio > 1.0 else 0
            else:
                session = 0
                
            self.market_conditions = {
                'trend_strength': trend,
                'volatility': volatility,
                'session_activity': session,
                'volume_ratio': volume_ratio,
                'atr': atr[-1]
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing market: {str(e)}")

    def get_dynamic_tp_sl(self, strategy: str, atr: float) -> Tuple[float, float]:
        """Vaziyatga qarab dinamik TP/SL hisoblash"""
        try:
            strategy_params = self.strategies[strategy]
            
            # Market holatini analiz
            trend_strength = abs(self.market_conditions['trend_strength'])
            volatility = self.market_conditions['volatility']
            volume_ratio = self.market_conditions['volume_ratio']
            
            # Bazaviy multiplikatorlar
            base_tp = strategy_params['tp_multiplier']
            base_sl = strategy_params['sl_multiplier']
            
            # Trend kuchiga qarab
            if trend_strength == 1:  # Kuchli trend
                tp_mult = base_tp * 2.0  # Kattaroq target
                sl_mult = base_sl * 0.8  # Yaqinroq stop
            else:  # Trend yo'q
                tp_mult = base_tp * 1.2
                sl_mult = base_sl * 1.0
                
            # Volatillikka qarab
            if volatility == 2:  # Yuqori
                tp_mult *= 1.5
                sl_mult *= 1.2
            elif volatility == 1:  # O'rta
                tp_mult *= 1.2
                sl_mult *= 1.0
            else:  # Past
                tp_mult *= 1.0
                sl_mult *= 0.8
                
            # Hajmga qarab
            volume_factor = min(volume_ratio, 2.0)
            tp_mult *= volume_factor
            sl_mult *= volume_factor
            
            # Risk/Reward nisbatini hisoblash
            tp_points = int(atr * tp_mult)
            sl_points = int(atr * sl_mult)
            risk_reward = tp_points / sl_points if sl_points > 0 else 0
            
            # Minimal masofalar
            min_points = 15  # Minimal 15 punkt
            tp_points = max(tp_points, min_points)
            sl_points = max(sl_points, min_points)
            
            return tp_points, sl_points
            
        except Exception as e:
            self.logger.error(f"Error calculating TP/SL: {str(e)}")
            return 30, 20  # Default qiymatlar
        

    def calculate_position_size(self, strategy: str, sl_points: float) -> float:
        """Vaziyatga qarab pozitsiya hajmini hisoblash"""
        try:
            strategy_params = self.strategies[strategy]
            
            # Bazaviy risk
            base_risk = strategy_params['risk_percent'] / 100
            
            # Market holatiga qarab riskni sozlash
            market = self.market_conditions
            trend_strength = abs(market['trend_strength'])
            volatility = market['volatility']
            volume_ratio = market['volume_ratio']
            session = market['session_activity']
            
            # Trend va sessiyaga qarab riskni o'zgartirish
            if trend_strength == 1 and session > 1:
                risk_mult = 1.5  # 1:1.5 risk/reward
            elif trend_strength == 1 or session > 1:
                risk_mult = 1.2  # 1:1.2 risk/reward
            else:
                risk_mult = 1.0  # 1:1 risk/reward
                
            # Volatillikka qarab
            if volatility == 2:
                risk_mult *= 0.7  # Risk kamaytiriladi
            elif volatility == 1:
                risk_mult *= 0.9
                
            # Hajmga qarab
            if volume_ratio > 1.5:
                risk_mult *= 1.2
                
            # Yakuniy risk
            adjusted_risk = base_risk * risk_mult
            
            # Joriy pozitsiyalarni tekshirish
            open_risk = sum(pos['risk_amount'] for pos in self.open_positions)
            max_risk = self.virtual_balance * 0.1  # Maksimal 10% risk
            
            if open_risk + adjusted_risk > max_risk:
                adjusted_risk = max(0, max_risk - open_risk)
                
            # Lot hajmini hisoblash
            point_value = self.point_value if self.point_value > 0 else 0.0001
            volume = round((self.virtual_balance * adjusted_risk) / (sl_points * point_value * 10), 2)
            
            # Min/max chegaralar
            volume = max(strategy_params['min_volume'], 
                        min(volume, strategy_params['max_volume']))
            
            return volume
            
        except Exception as e:
            self.logger.error(f"Error calculating position size: {str(e)}")
            return 0.01  # Minimal lot

    def _calculate_position_profit(self, position: Dict, current_price: float) -> float:
        """Pozitsiya foydasini hisoblash"""
        if position['type'] == 'buy':
            points = current_price - position['price']
        else:
            points = position['price'] - current_price
            
        return points * position['volume'] * 100000

    def _check_close_conditions(self, position: Dict, current_price: float) -> Optional[str]:
        """Pozitsiyani yopish shartlarini tekshirish"""
        if position['type'] == 'buy':
            if current_price >= position['tp']:
                return 'TP'
            elif current_price <= position['sl']:
                return 'SL'
        else:
            if current_price <= position['tp']:
                return 'TP'
            elif current_price >= position['sl']:
                return 'SL'
        return None
    

    def can_open_new_order(self, strategy: str) -> bool:
        """Yangi order ochish mumkinligini tekshirish"""
        try:
            # Asosiy tekshiruvlar
            strategy_positions = [pos for pos in self.open_positions 
                                if pos['strategy'] == strategy]
            if len(strategy_positions) >= self.strategies[strategy]['max_orders']:
                return False
                
            # Vaqt intervali
            if self.current_index - self.last_order_times[strategy] < self.strategies[strategy]['interval']:
                return False
                
            # Margin tekshiruvi
            total_risk = sum(pos['risk_amount'] for pos in self.open_positions)
            max_risk_percent = 30  # Maksimal risk 30%
            if total_risk >= self.virtual_balance * (max_risk_percent / 100):
                return False
                
            # Strategiyaga xos shartlar
            market = self.market_conditions
            volume_condition = market['volume_ratio'] > 1.0
            
            if strategy == 'Scalping':
                return (market['volatility'] < 2 and  # Past volatillik
                       market['session_activity'] > 0 and  # Faol sessiya
                       volume_condition and  # Yetarli hajm
                       abs(market['trend_strength']) > 0)  # Aniq trend
                       
            elif strategy == 'Breakout':
                return (market['volatility'] > 0 and  # O'rta/yuqori volatillik
                       market['session_activity'] > 1 and  # Juda faol sessiya
                       volume_condition and  # Yetarli hajm
                       market['volume_ratio'] > 1.5)  # Kuchli hajm o'sishi
                       
            elif strategy == 'OrderBlock':
                return (abs(market['trend_strength']) == 1 and  # Kuchli trend
                       market['session_activity'] > 0 and  # Faol sessiya
                       volume_condition and  # Yetarli hajm
                       market['volatility'] < 2)  # Past/o'rta volatillik
                       
            return True
            
        except Exception as e:
            self.logger.error(f"Error checking order conditions: {str(e)}")
            return False
        
    def _print_episode_summary(self, episode: int, best_profit: float, best_episode: int) -> None:
        """Episode yakunini chiqarish"""
        try:
            profit = self.virtual_balance - self.initial_balance
            total_trades = len(self.trade_history)
            
            print(f"\n{'='*50}")
            print(f"📊 EPISODE {episode + 1} SUMMARY")
            print(f"{'='*50}")
            print(f"💰 Final Balance: ${self.virtual_balance:.2f}")
            print(f"📈 Net Profit: ${profit:.2f} ({profit/self.initial_balance*100:.1f}%)")
            
            if total_trades > 0:
                win_trades = len([t for t in self.trade_history if t['profit'] > 0])
                avg_profit = sum(t['profit'] for t in self.trade_history) / total_trades
                avg_roi = sum(t['roi'] for t in self.trade_history) / total_trades
                win_rate = win_trades/total_trades*100
                
                print(f"\n📊 TRADING STATISTICS")
                print(f"{'='*50}")
                print(f"📈 Total Trades: {total_trades}")
                print(f"✅ Profitable: {win_trades}")
                print(f"❌ Loss Making: {total_trades - win_trades}")
                print(f"📊 Win Rate: {win_rate:.1f}%")
                print(f"💵 Average Profit: ${avg_profit:.2f}")
                print(f"📈 Average ROI: {avg_roi:.1f}%")
                
                # Strategy breakdown
                print(f"\n📊 STRATEGY PERFORMANCE")
                print(f"{'='*50}")
                for strategy in self.strategies.keys():
                    strategy_trades = [t for t in self.trade_history if t['strategy'] == strategy]
                    if strategy_trades:
                        s_total = len(strategy_trades)
                        s_wins = len([t for t in strategy_trades if t['profit'] > 0])
                        s_profit = sum(t['profit'] for t in strategy_trades)
                        print(f"\n{strategy}:")
                        print(f"  Trades: {s_total}")
                        print(f"  Win Rate: {(s_wins/s_total*100):.1f}%")
                        print(f"  Net Profit: ${s_profit:.2f}")
                
                if best_profit > float('-inf'):
                    print(f"\n🏆 BEST PERFORMANCE")
                    print(f"{'='*50}")
                    print(f"💰 Best Profit: ${best_profit:.2f}")
                    print(f"📅 Best Episode: {best_episode}")
            else:
                print("\n⚠️ No trades executed in this episode")
            
            # Log ma'lumotlarni yozish
            if total_trades > 0:
                win_trades = len([t for t in self.trade_history if t['profit'] > 0])
                self.logger.info(
                    f"Episode {episode + 1} completed - "
                    f"Balance: ${self.virtual_balance:.2f}, "
                    f"Profit: ${profit:.2f}, "
                    f"Trades: {total_trades}, "
                    f"Win Rate: {win_trades/total_trades*100:.1f}%"
                )
            else:
                self.logger.info(
                    f"Episode {episode + 1} completed - "
                    f"No trades executed, Balance: ${self.virtual_balance:.2f}"
                )
                
        except Exception as e:
            self.logger.error(f"Error printing episode summary: {str(e)}")
            print("\n⚠️ Error printing episode summary")