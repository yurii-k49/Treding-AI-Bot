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
import random

class HistoricalTrader(TradingModel):
    """Trading bot that trains on historical data with optimized performance"""
    
    def __init__(self, symbol: str = "EURUSD", timeframe: int = mt5.TIMEFRAME_M5):
        """Initialize the trader with optimized parameters"""
        super().__init__(symbol, timeframe)
        
        # Core trading parameters
        self.historical_data = None
        self.current_index = 50
        self.initial_balance = 100.0
        self.virtual_balance = self.initial_balance
        self.gamma = 0.95  # Added gamma parameter for Q-learning
        
        # Performance optimization parameters
        self.batch_size = 128  # Increased for faster training
        self.data_window_size = 1000  # Process data in chunks
        self.update_interval = 10  # Market update frequency
        self.market_state_cache = {}  # Cache for market states
        self.precomputed_indicators = {}
        
        # Trading state tracking
        self.open_positions: List[Dict] = []
        self.trade_history: List[Dict] = []
        self.logger = self._setup_logger()
        
        # Strategy configurations with optimized parameters
        self.strategies = {
            'Scalping': {
                'interval': 12,
                'max_orders': 2,
                'tp_multiplier': 2.5,
                'sl_multiplier': 1.0,
                'risk_percent': 2.0,
                'min_volume': 0.05,
                'max_volume': 0.2
            },
            'Breakout': {
                'interval': 8,
                'max_orders': 2,
                'tp_multiplier': 2.5,
                'sl_multiplier': 1.0,
                'risk_percent': 1.2,
                'min_volume': 0.01,
                'max_volume': 0.1
            },
            'OrderBlock': {
                'interval': 12,
                'max_orders': 2,
                'tp_multiplier': 3.0,
                'sl_multiplier': 1.0,
                'risk_percent': 1.5,
                'min_volume': 0.01,
                'max_volume': 0.1
            }
        }
        
        # State tracking
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
        """Configure logging with optimized settings"""
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
        """Load and preprocess historical data efficiently"""
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
            
            # Precompute base indicators after loading data
            self._precompute_base_indicators()
            
            print(f"✅ Loaded: {len(rates)} bars")
            print(f"📅 First date: {datetime.fromtimestamp(rates[0]['time'])}")
            print(f"📅 Last date: {datetime.fromtimestamp(rates[-1]['time'])}")
            
            self.logger.info(f"Successfully loaded {len(rates)} bars of historical data")
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading historical data: {str(e)}")
            return False

    def _precompute_base_indicators(self) -> None:
        """Pre-compute all base technical indicators for faster access"""
        print("\nPre-computing base technical indicators...")
        try:
            closes = self.historical_data['close']
            highs = self.historical_data['high']
            lows = self.historical_data['low']
            volumes = self.historical_data['tick_volume']
            
            # Initialize arrays
            n = len(closes)
            highs_20 = np.zeros(n)
            lows_20 = np.zeros(n)
            volume_sma = np.zeros(n)
            momentum = np.zeros(n)
            
            # Calculate rolling indicators
            window = 20
            for i in range(window, n):
                highs_20[i] = np.max(highs[i-window:i])
                lows_20[i] = np.min(lows[i-window:i])
                volume_sma[i] = np.mean(volumes[i-window:i])
            
            # Fill initial values
            highs_20[:window] = highs_20[window]
            lows_20[:window] = lows_20[window]
            volume_sma[:window] = volume_sma[window]
            
            # Calculate momentum (5-period)
            for i in range(5, n):
                momentum[i] = closes[i] - closes[i-5]
            
            # Store all indicators
            self.precomputed_indicators = {
                'ema20': self.calculate_ema(closes, 20),
                'ema50': self.calculate_ema(closes, 50),
                'ema200': self.calculate_ema(closes, 200),
                'rsi': self.calculate_rsi(closes),
                'atr': self.calculate_atr(self.historical_data),
                'highs_20': highs_20,
                'lows_20': lows_20,
                'volume_sma': volume_sma,
                'momentum': momentum
            }
            
            print("✅ Base indicators pre-computed successfully")
            
        except Exception as e:
            self.logger.error(f"Error pre-computing indicators: {str(e)}")
            raise

    def _update_market_conditions(self) -> None:
        """Update market conditions based on current market state"""
        try:
            # Get current window for analysis
            window_start = max(0, self.current_index - 50)
            window = self.historical_data[window_start:self.current_index]
            
            # Get indicator values at current index
            current_idx = self.current_index
            ema20 = self.precomputed_indicators['ema20'][current_idx]
            ema50 = self.precomputed_indicators['ema50'][current_idx]
            ema200 = self.precomputed_indicators['ema200'][current_idx]
            rsi = self.precomputed_indicators['rsi'][current_idx]
            atr = self.precomputed_indicators['atr'][current_idx]
            
            # Calculate volume metrics
            current_volume = window[-1]['tick_volume']
            avg_volume = np.mean(window['tick_volume'])
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
            
            # Analyze trend strength
            if ema20 > ema50 > ema200 and volume_ratio > 1.2:
                trend_strength = 1  # Strong uptrend
            elif ema20 < ema50 < ema200 and volume_ratio > 1.2:
                trend_strength = -1  # Strong downtrend
            else:
                trend_strength = 0  # No clear trend
            
            # Analyze volatility
            atr_mean = np.mean(self.precomputed_indicators['atr'][window_start:current_idx])
            atr_std = np.std(self.precomputed_indicators['atr'][window_start:current_idx])
            
            if atr > atr_mean + 2 * atr_std:
                volatility = 2  # High volatility
            elif atr > atr_mean + atr_std:
                volatility = 1  # Medium volatility
            else:
                volatility = 0  # Low volatility
            
            # Determine session activity based on volatility and volume
            if volatility >= 1 and volume_ratio > 1.5:
                session_activity = 2  # High activity
            elif volatility >= 1 or volume_ratio > 1.2:
                session_activity = 1  # Medium activity
            else:
                session_activity = 0  # Low activity
            
            # Update market conditions
            self.market_conditions = {
                'trend_strength': trend_strength,
                'volatility': volatility,
                'session_activity': session_activity,
                'volume_ratio': volume_ratio,
                'atr': atr,
                'rsi': rsi
            }
            
        except Exception as e:
            self.logger.error(f"Error updating market conditions: {str(e)}")
            # Set default values in case of error
            self.market_conditions = {
                'trend_strength': 0,
                'volatility': 0,
                'session_activity': 0,
                'volume_ratio': 1.0,
                'atr': 0.0,
                'rsi': 50.0
            }

    def train_on_historical(self, episodes: int = 100) -> None:
        """Main training loop with strategy optimization"""
        if not os.path.exists('models'):
            os.makedirs('models')
                
        print("\n=== STARTING TRAINING ===")
        print(f"📊 Total bars: {len(self.historical_data)}")
        print(f"🔄 Episodes: {episodes}")
        print(f"💰 Initial balance: ${self.initial_balance:,.2f}\n")
        
        # Pre-compute indicators once for all episodes
        self._precompute_base_indicators()
        
        # Training parameters
        epsilon = 1.0
        epsilon_min = 0.01
        epsilon_decay = 0.995
        batch_size = self.batch_size
        gamma = self.gamma
        memory = deque(maxlen=2000)
        
        # Performance tracking
        best_profit = float('-inf')
        best_episode = -1
        episodes_without_improvement = 0
        max_episodes_without_improvement = 10
        
        # Strategy performance history
        strategy_history = {
            strategy: {
                'win_rates': [],
                'profits': []
            } for strategy in self.strategies.keys()
        }
        
        for episode in range(episodes):
            self.logger.info(f"Starting episode {episode + 1}")
            
            # Run episode and get stats
            episode_stats = self._run_training_episode(
                episode, epsilon, memory, batch_size, gamma
            )
            
            if episode_stats:
                # Update strategy history
                episode_profit = self.virtual_balance - self.initial_balance
                
                for strategy, stats in episode_stats.items():
                    if stats['trades'] > 0:
                        win_rate = (stats['wins'] / stats['trades']) * 100
                        strategy_history[strategy]['win_rates'].append(win_rate)
                        strategy_history[strategy]['profits'].append(stats['profit'])
                
                # Update best model if improved
                if episode_profit > best_profit:
                    best_profit = episode_profit
                    best_episode = episode + 1
                    episodes_without_improvement = 0
                    
                    # Save best model
                    model_path = f'models/best_model_episode_{episode + 1}.keras'
                    self.save_model(model_path)
                    print(f"\n✨ New best profit! Model saved: {model_path}")
                    
                    # Adjust strategy parameters based on performance
                    self._optimize_strategy_parameters(strategy_history)
                else:
                    episodes_without_improvement += 1
                
                # Early stopping check
                if episodes_without_improvement >= max_episodes_without_improvement:
                    print("\n⚠️ Stopping early due to no improvement")
                    break
                
                # Update epsilon for exploration
                epsilon = max(epsilon_min, epsilon * epsilon_decay)
                
                # Log episode results
                trades = len(self.trade_history)
                if trades > 0:
                    win_trades = len([t for t in self.trade_history if t['profit'] > 0])
                    win_rate = (win_trades/trades) * 100
                    self.logger.info(
                        f"Episode {episode + 1} completed - "
                        f"Balance: ${self.virtual_balance:.2f}, "
                        f"Profit: ${episode_profit:.2f}, "
                        f"Trades: {trades}, "
                        f"Win Rate: {win_rate:.1f}%"
                    )
                else:
                    self.logger.info(
                        f"Episode {episode + 1} completed - "
                        f"No trades executed, Balance: ${self.virtual_balance:.2f}"
                    )
        
        # Training completion summary
        print("\n=== TRAINING COMPLETED ===")
        print(f"{'='*50}")
        print(f"🏆 Best Episode: {best_episode}")
        print(f"💰 Best Profit: ${best_profit:.2f}")
        print(f"📈 Final Epsilon: {epsilon:.4f}")
        
        # Strategy performance summary
        print("\n📊 STRATEGY PERFORMANCE SUMMARY:")
        for strategy, history in strategy_history.items():
            if history['win_rates']:
                avg_win_rate = np.mean(history['win_rates'])
                avg_profit = np.mean(history['profits'])
                print(f"\n{strategy}:")
                print(f"  Average Win Rate: {avg_win_rate:.1f}%")
                print(f"  Average Profit: ${avg_profit:.2f}")
        
        # Save final model
        final_model_path = 'models/final_model.keras'
        self.save_model(final_model_path)
        print(f"\n💾 Final model saved: {final_model_path}")

    def _run_training_episode(self, episode: int, epsilon: float, memory: deque,
                          batch_size: int, gamma: float) -> Dict:
        """Run single training episode with strategy optimization"""
        try:
            # Reset episode state
            self.virtual_balance = self.initial_balance
            self.current_index = 50
            self.open_positions = []
            self.trade_history = []
            trades_this_episode = 0
            last_market_update = 0
            
            # Strategy performance tracking
            strategy_stats = {
                strategy: {
                    'trades': 0,
                    'wins': 0,
                    'losses': 0,
                    'profit': 0.0
                } for strategy in self.strategies.keys()
            }
            
            episode_start = time.time()
            print(f"\n=== EPISODE {episode + 1} ===")
            
            while self.current_index < len(self.historical_data) - 50:
                # Early stopping conditions
                if self.virtual_balance <= 0:
                    print("❌ Account balance depleted")
                    break
                if trades_this_episode >= 100:
                    print("✋ Maximum trades reached")
                    break
                    
                # Update market conditions periodically
                if self.current_index - last_market_update >= 20:
                    self._update_market_conditions()
                    last_market_update = self.current_index
                
                # Try each strategy
                for strategy in self.strategies.keys():
                    # Get current state
                    current_state = self._get_market_state()
                    
                    # Use epsilon-greedy for trading decision
                    if np.random.random() < epsilon:
                        should_trade = True
                    else:
                        # Use model prediction
                        q_values = self.model.predict(
                            current_state.reshape(1, 50, 8), 
                            verbose=0
                        )[0]
                        should_trade = np.max(q_values) > 0.5
                    
                    # Execute trade if conditions are met
                    if should_trade and self.can_open_new_order(strategy):
                        window = self.historical_data[self.current_index-50:self.current_index]
                        current_price = self.historical_data[self.current_index]['close']
                        
                        # Get trade parameters
                        trade_type = self._get_trade_type(strategy, current_price, window)
                        if trade_type:
                            trades_this_episode += 1
                            strategy_stats[strategy]['trades'] += 1
                            
                            # Execute trade and track result
                            result = self.execute_trade(strategy, trade_type)
                            if result > 0:
                                strategy_stats[strategy]['wins'] += 1
                                strategy_stats[strategy]['profit'] += result
                            else:
                                strategy_stats[strategy]['losses'] += 1
                                strategy_stats[strategy]['profit'] += result
                            
                            # Store experience
                            next_state = self._get_market_state()
                            memory.append({
                                'state': current_state,
                                'action': list(self.strategies.keys()).index(strategy),
                                'reward': result,
                                'next_state': next_state
                            })
                
                # Check and update open positions
                if self.open_positions:
                    self._handle_closed_positions(
                        self.historical_data[self.current_index]['close'],
                        datetime.fromtimestamp(self.historical_data[self.current_index]['time'])
                    )
                
                # Train model periodically
                if len(memory) >= batch_size:
                    self._train_on_batch(memory, batch_size, gamma)
                
                self.current_index += 1
            
            # Episode summary
            total_trades = sum(s['trades'] for s in strategy_stats.values())
            if total_trades > 0:
                total_profit = sum(s['profit'] for s in strategy_stats.values())
                total_wins = sum(s['wins'] for s in strategy_stats.values())
                win_rate = (total_wins / total_trades) * 100
                
                print(f"\n{'='*50}")
                print(f"📊 EPISODE {episode + 1} RESULTS:")
                print(f"⏱️ Time: {time.time() - episode_start:.1f}s")
                print(f"💰 Final Balance: ${self.virtual_balance:.2f}")
                print(f"📈 Net Profit: ${total_profit:.2f}")
                print(f"🎯 Total Trades: {total_trades}")
                print(f"✅ Win Rate: {win_rate:.1f}%\n")
                
                # Strategy breakdown
                print("📊 STRATEGY PERFORMANCE:")
                for strategy, stats in strategy_stats.items():
                    if stats['trades'] > 0:
                        s_win_rate = (stats['wins'] / stats['trades']) * 100
                        print(f"\n{strategy}:")
                        print(f"  Trades: {stats['trades']}")
                        print(f"  Win Rate: {s_win_rate:.1f}%")
                        print(f"  Profit: ${stats['profit']:.2f}")
                
            return strategy_stats
            
        except Exception as e:
            self.logger.error(f"Error in episode {episode + 1}: {str(e)}")
            return None

    def _get_cached_state(self) -> np.ndarray:
        """Get market state with caching"""
        try:
            # Check if state exists in cache
            if self.current_index in self.market_state_cache:
                return self.market_state_cache[self.current_index]
            
            # Calculate new state
            state = self._get_market_state()
            
            # Cache the state
            self.market_state_cache[self.current_index] = state
            
            # Clear old cache entries periodically
            if len(self.market_state_cache) > 1000:
                self.market_state_cache = {k: v for k, v in self.market_state_cache.items() 
                                        if k > self.current_index - 100}
            
            return state
        
        except Exception as e:
            self.logger.error(f"Error getting cached state: {str(e)}")
            return np.zeros((50, 8))

    def _execute_episode_trade(self, state: np.ndarray, epsilon: float) -> Optional[Tuple]:
        """Execute trade during training episode with exploration"""
        try:
            if np.random.random() < epsilon:
                action = np.random.choice(len(self.strategies))
            else:
                q_values = self.model.predict(state.reshape(1, -1, 8), verbose=0)[0]
                action = np.argmax(q_values)
            
            reward = self.execute_trade(action)
            
            if reward != 0:
                current_state = self._get_market_state()
                next_state = self._get_cached_state()
                
                return (current_state, action, reward, next_state)
            
            return None
        
        except Exception as e:
            self.logger.error(f"Error executing episode trade: {str(e)}")
            return None

    def _update_model(self, state: np.ndarray, trade_result: Tuple,
                    gamma: float, memory: deque, batch_size: int) -> None:
        """Update model with new experience"""
        try:
            current_state, action, reward, next_state = trade_result
            
            # Store experience
            memory.append({
                'state': current_state,
                'action': action,
                'reward': reward,
                'next_state': next_state
            })
            
            # Train on batch if enough samples
            if len(memory) >= batch_size:
                self._train_on_batch(memory, batch_size, gamma)
            
        except Exception as e:
            self.logger.error(f"Error updating model: {str(e)}")

    def _print_episode_summary(self, episode: int, best_profit: float, 
                            best_episode: int) -> None:
        """Print detailed episode summary with all statistics"""
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
                    strategy_trades = [t for t in self.trade_history 
                                    if t['strategy'] == strategy]
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
            
        except Exception as e:
            self.logger.error(f"Error printing episode summary: {str(e)}")

    def _calculate_trade_statistics(self, trades_this_episode: int) -> Dict:
        """Calculate detailed trade statistics"""
        try:
            if trades_this_episode == 0:
                return {
                    'win_rate': 0,
                    'avg_profit': 0,
                    'total_profit': 0,
                    'roi': 0
                }
                
            win_trades = len([t for t in self.trade_history if t['profit'] > 0])
            total_profit = sum(t['profit'] for t in self.trade_history)
            avg_profit = total_profit / trades_this_episode
            roi = (self.virtual_balance - self.initial_balance) / self.initial_balance * 100
            
            return {
                'win_rate': (win_trades/trades_this_episode*100),
                'avg_profit': avg_profit,
                'total_profit': total_profit,
                'roi': roi
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating trade statistics: {str(e)}")
            return {
                'win_rate': 0,
                'avg_profit': 0,
                'total_profit': 0,
                'roi': 0
            }

    def _train_on_batch(self, memory: deque, batch_size: int, gamma: float) -> None:
        """Train model on a batch of experiences"""
        try:
            # Random sample from memory
            batch = random.sample(list(memory), batch_size)
            
            # Prepare arrays for training
            states = np.array([x['state'] for x in batch])
            next_states = np.array([x['next_state'] for x in batch])
            
            # Get current Q values for all states
            current_q = self.model.predict(states, verbose=0)
            
            # Get future Q values for next states
            future_q = self.model.predict(next_states, verbose=0)
            
            # Update Q values for actions taken
            for i, transition in enumerate(batch):
                if transition['reward'] != 0:  # If we got a reward
                    current_q[i][transition['action']] = transition['reward'] + \
                        gamma * np.max(future_q[i])
            
            # Train model on the batch
            self.model.fit(
                states, 
                current_q,
                batch_size=batch_size,
                epochs=1,
                verbose=0
            )
            
        except Exception as e:
            self.logger.error(f"Error in batch training: {str(e)}")

    def _execute_episode_trade(self, state: np.ndarray, epsilon: float) -> Optional[Tuple]:
        """Execute trade during training episode"""
        if np.random.random() < epsilon:
            action = np.random.choice(len(self.strategies))
        else:
            q_values = self.model.predict(
                state.reshape(1, *state.shape), verbose=0)[0]
            action = np.argmax(q_values)
            
        reward = self.execute_trade(action)
        if reward != 0:
            return (state, action, reward)
        return None
        
    def _update_model(self, state: np.ndarray, trade_result: Tuple,
                     gamma: float, memory: deque, batch_size: int) -> None:
        """Update model with new experience"""
        current_state, action, reward = trade_result
        next_state = self._get_market_state()
        
        # Store experience
        memory.append((current_state, action, reward, next_state))
        
        # Train on batch if enough samples
        if len(memory) >= batch_size:
            # Get random batch
            indices = np.random.choice(len(memory), batch_size, replace=False)
            states = []
            targets = []
            
            for idx in indices:
                s, a, r, n_s = memory[idx]
                # Get current Q values
                target = self.model.predict(
                    s.reshape(1, *s.shape), verbose=0)[0]
                # Get future Q values
                Q_future = np.max(self.model.predict(
                    n_s.reshape(1, *n_s.shape), verbose=0)[0])
                # Update target for action taken
                target[a] = r + gamma * Q_future
                
                states.append(s)
                targets.append(target)
            
            # Batch train
            states_array = np.array(states)
            self.model.fit(states_array, np.array(targets),
                          epochs=1, verbose=0, batch_size=batch_size)
                          
    def _handle_episode_completion(self, episode: int, episode_start: float,
                                trades_this_episode: int, best_profit: float,
                                best_episode: int) -> Tuple[float, int]:
        """Handle end of training episode and save results"""
        episode_time = time.time() - episode_start
        episode_profit = self.virtual_balance - self.initial_balance
        
        # Calculate statistics
        if trades_this_episode > 0:
            win_trades = len([t for t in self.trade_history if t['profit'] > 0])
            win_rate = (win_trades/trades_this_episode*100)
            
            print(f"\n{'='*50}")
            print(f"📊 EPISODE {episode + 1} RESULTS:")
            print(f"⏱️ Time: {episode_time:.1f}s")
            print(f"💰 Balance: ${self.virtual_balance:.2f}")
            print(f"📈 Profit: ${episode_profit:.2f}")
            print(f"🎯 Trades: {trades_this_episode} (Win Rate: {win_rate:.1f}%)")
            
            # Check for improvement
            if episode_profit > best_profit:
                print("✨ New best profit!")
                best_profit = episode_profit
                best_episode = episode + 1
                # Save model
                self.save_model(f'models/best_model_episode_{episode + 1}.keras')
                
            self.logger.info(
                f"Episode {episode + 1} completed - "
                f"Balance: ${self.virtual_balance:.2f}, "
                f"Profit: ${episode_profit:.2f}, "
                f"Trades: {trades_this_episode}, "
                f"Win Rate: {win_rate:.1f}%"
            )
        else:
            print("\n⚠️ No trades executed in this episode")
            self.logger.info(
                f"Episode {episode + 1} completed - "
                f"No trades executed, Balance: ${self.virtual_balance:.2f}"
            )
            
        return best_profit, best_episode
    
    def execute_trade(self, action: int) -> float:
        """Execute a trade with optimized decision making and logging"""
        if self.current_index >= len(self.historical_data):
            return 0
            
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
            
            # Get trade parameters
            trade_type = self._get_trade_type(strategy, current_price, window)
            if not trade_type:
                return 0
                
            atr = self.precomputed_indicators['atr'][self.current_index]
            tp_points, sl_points = self.get_dynamic_tp_sl(strategy, atr)
            volume = self.calculate_position_size(strategy, sl_points)
            
            if volume <= 0:
                return 0
                
            # Create and execute position
            position = self._create_position(
                strategy, trade_type, volume, current_price,
                tp_points, sl_points, trade_time
            )
            
            self._log_trade_execution(position, trade_time)
            
            # Log trade execution
            self.logger.info(
                f"Opening {position['type']} order: "
                f"Strategy={strategy}, "
                f"Volume={volume:.2f}, "
                f"Price={current_price:.5f}, "
                f"TP={position['tp']:.5f}, "
                f"SL={position['sl']:.5f}"
            )
            
            self.open_positions.append(position)
            self.last_order_times[strategy] = self.current_index
            
            expected_profit = self._calculate_expected_profit(position, current_price)
            
            # Clear memory cache periodically
            if len(self.market_state_cache) > 500:
                self.market_state_cache.clear()
            
            return expected_profit
            
        except Exception as e:
            self.logger.error(f"Error executing trade: {str(e)}")
            return 0
            
    def _get_trade_type(self, strategy: str, price: float, window: np.ndarray) -> Optional[str]:
        """Determine trade type based on strategy and market conditions"""
        try:
            current_idx = self.current_index
            rsi = self.precomputed_indicators['rsi'][current_idx]
            ema_fast = self.precomputed_indicators['ema20'][current_idx]
            ema_slow = self.precomputed_indicators['ema50'][current_idx]
            ema200 = self.precomputed_indicators['ema200'][current_idx]
            
            volume_avg = np.mean(window['tick_volume'][-20:])
            current_volume = window['tick_volume'][-1]
            momentum = window['close'][-1] - window['close'][-5]
            
            if strategy == 'Scalping':
                if (rsi < 30 and ema_fast > ema_slow and
                    price > ema200 and momentum > 0 and
                    current_volume > volume_avg):
                    return 'buy'
                elif (rsi > 70 and ema_fast < ema_slow and
                      price < ema200 and momentum < 0 and
                      current_volume > volume_avg):
                    return 'sell'
                    
            elif strategy == 'Breakout':
                highs = window['high'][-20:]
                lows = window['low'][-20:]
                resistance = np.max(highs[:-1])
                support = np.min(lows[:-1])
                atr = self.precomputed_indicators['atr'][current_idx]
                
                if (price > resistance + atr * 0.5 and
                    current_volume > volume_avg * 1.2 and
                    momentum > 0):
                    return 'buy'
                elif (price < support - atr * 0.5 and
                      current_volume > volume_avg * 1.2 and
                      momentum < 0):
                    return 'sell'
                    
            elif strategy == 'OrderBlock':
                if (price > ema200 and ema_fast > ema_slow and
                    40 < rsi < 75 and momentum > 0 and
                    current_volume > volume_avg and
                    self.market_conditions['trend_strength'] == 1):
                    return 'buy'
                elif (price < ema200 and ema_fast < ema_slow and
                      25 < rsi < 60 and momentum < 0 and
                      current_volume > volume_avg and
                      self.market_conditions['trend_strength'] == -1):
                    return 'sell'
                    
            return None
            
        except Exception as e:
            self.logger.error(f"Error in trade type calculation: {str(e)}")
            return None
        
    def _get_market_state(self) -> np.ndarray:
        """Get current market state using pre-computed indicators"""
        try:
            idx = self.current_index
            window_start = max(0, idx-50)
            
            state = np.column_stack((
                self.historical_data['open'][window_start:idx],
                self.historical_data['high'][window_start:idx],
                self.historical_data['low'][window_start:idx],
                self.historical_data['close'][window_start:idx],
                self.precomputed_indicators['ema50'][window_start:idx],
                self.precomputed_indicators['rsi'][window_start:idx],
                self.precomputed_indicators['atr'][window_start:idx],
                self.precomputed_indicators['ema200'][window_start:idx]
            ))
            
            return np.nan_to_num(state, nan=0.0)
            
        except Exception as e:
            self.logger.error(f"Error getting market state: {str(e)}")
            return np.zeros((50, 8))
        
    def _calculate_expected_profit(self, position: Dict, current_price: float) -> float:
        """Calculate expected profit for position"""
        if position['type'] == 'buy':
            return (position['tp'] - current_price) * position['volume'] * 100000
        else:
            return (current_price - position['tp']) * position['volume'] * 100000
            
    def _log_trade_execution(self, position: Dict, trade_time: datetime) -> None:
        """Log details of executed trade"""
        print(f"\n{'='*50}")
        print(f"🔄 NEW ORDER #{position['ticket']} | {trade_time}")
        print(f"🎯 Strategy: {position['strategy']}")
        print(f"📈 Type: {position['type'].upper()}")
        print(f"📦 Volume: {position['volume']:.2f} lot")
        print(f"💰 Entry: {position['price']:.5f}")
        print(f"🎯 Take Profit: {position['tp']:.5f}")
        print(f"🛑 Stop Loss: {position['sl']:.5f}")
        print(f"💵 Risk Amount: ${position['risk_amount']:.2f}")
        print(f"⚖️ Current Balance: ${self.virtual_balance:.2f}")

    def _batch_check_positions(self) -> float:
        """Check multiple positions efficiently"""
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
                positions_to_close.append((pos, profit, close_reason))
                
        # Process closed positions
        for pos, profit, reason in positions_to_close:
            self._process_closed_position(pos, profit, reason, current_price, current_time)

    def can_open_new_order(self, strategy: str) -> bool:
        """Check if new order can be opened with improved conditions"""
        try:
            # Check maximum open positions
            if len(self.open_positions) >= 5:  # Global limit
                return False
                
            # Check strategy-specific positions
            strategy_positions = [pos for pos in self.open_positions 
                                if pos['strategy'] == strategy]
            if len(strategy_positions) >= self.strategies[strategy]['max_orders']:
                return False
                
            # Check minimum interval between orders
            if self.current_index - self.last_order_times[strategy] < self.strategies[strategy]['interval']:
                return False
                
            # Check total risk exposure
            total_risk = sum(pos['risk_amount'] for pos in self.open_positions)
            max_risk_percent = 30  # Maximum 30% risk exposure
            if total_risk >= self.virtual_balance * (max_risk_percent / 100):
                return False
                
            # Market condition checks
            market = self.market_conditions
            volume_condition = market['volume_ratio'] > 1.0
            
            # Strategy specific conditions
            if strategy == 'Scalping':
                return (market['volatility'] < 2 and
                    market['session_activity'] > 0 and
                    volume_condition and
                    abs(market['trend_strength']) > 0)
                    
            elif strategy == 'Breakout':
                return (market['volatility'] > 0 and
                    market['session_activity'] > 1 and
                    volume_condition and
                    market['volume_ratio'] > 1.5)
                    
            elif strategy == 'OrderBlock':
                return (abs(market['trend_strength']) == 1 and
                    market['session_activity'] > 0 and
                    volume_condition and
                    market['volatility'] < 2)
                    
            return True
            
        except Exception as e:
            self.logger.error(f"Error checking order conditions: {str(e)}")
            return False
        
    def calculate_position_size(self, strategy: str, sl_points: float) -> float:
        """Calculate optimal position size with risk management"""
        try:
            strategy_params = self.strategies[strategy]
            
            # Base risk calculation
            base_risk = strategy_params['risk_percent'] / 100
            
            # Adjust risk based on market conditions
            market = self.market_conditions
            trend_strength = abs(market['trend_strength'])
            volatility = market['volatility']
            volume_ratio = market['volume_ratio']
            session = market['session_activity']
            
            # Risk multiplier based on conditions
            if trend_strength == 1 and session > 1:
                risk_mult = 1.5
            elif trend_strength == 1 or session > 1:
                risk_mult = 1.2
            else:
                risk_mult = 1.0
                
            # Volatility adjustment
            if volatility == 2:
                risk_mult *= 0.7  # Reduce risk in high volatility
            elif volatility == 1:
                risk_mult *= 0.9  # Slightly reduce risk in medium volatility
                
            # Volume impact
            if volume_ratio > 1.5:
                risk_mult *= 1.2  # Increase risk with higher volume
                
            # Final risk calculation
            adjusted_risk = base_risk * risk_mult
            
            # Check current exposure
            open_risk = sum(pos['risk_amount'] for pos in self.open_positions)
            max_risk = self.virtual_balance * 0.1  # Max 10% total risk
            
            if open_risk + adjusted_risk > max_risk:
                adjusted_risk = max(0, max_risk - open_risk)
                
            # Calculate lot size
            point_value = self.point_value if self.point_value > 0 else 0.0001
            volume = round((self.virtual_balance * adjusted_risk) / (sl_points * point_value * 10), 2)
            
            # Apply min/max limits
            volume = max(strategy_params['min_volume'],
                    min(volume, strategy_params['max_volume']))
            
            return volume
            
        except Exception as e:
            self.logger.error(f"Error calculating position size: {str(e)}")
            return 0.01  # Return minimum volume on error

    def calculate_dynamic_risk(self, strategy: str, market_conditions: Dict) -> float:
        """Calculate dynamic risk percentage based on market conditions"""
        try:
            base_risk = self.strategies[strategy]['risk_percent']
            trend = abs(market_conditions['trend_strength'])
            volatility = market_conditions['volatility']
            
            # Adjust risk based on conditions
            if trend == 1 and volatility < 2:
                risk = base_risk * 1.2  # Increase risk in strong trend
            elif trend == 0 or volatility == 2:
                risk = base_risk * 0.8  # Reduce risk in no trend or high volatility
            else:
                risk = base_risk
                
            # Apply maximum risk limit
            return min(risk, 5.0)  # Never risk more than 5% per trade
            
        except Exception as e:
            self.logger.error(f"Error calculating dynamic risk: {str(e)}")
            return 1.0  # Default to 1% risk

    def get_position_exposure(self) -> float:
        """Calculate total position exposure as percentage of balance"""
        try:
            total_exposure = sum(pos['risk_amount'] for pos in self.open_positions)
            return (total_exposure / self.virtual_balance) * 100
            
        except Exception as e:
            self.logger.error(f"Error calculating position exposure: {str(e)}")
            return 0.0
        
    def _create_position(self, strategy: str, trade_type: str, volume: float,
                     current_price: float, tp_points: float, sl_points: float,
                     trade_time: datetime) -> Dict:
        """Create a new position with calculated parameters"""
        try:
            point_value = self.point_value if self.point_value > 0 else 0.0001
            
            # Calculate TP and SL levels based on trade type
            if trade_type == 'buy':
                sl = current_price - (sl_points * point_value)
                tp = current_price + (tp_points * point_value)
            else:  # sell
                sl = current_price + (sl_points * point_value)
                tp = current_price - (tp_points * point_value)
                
            # Create position dictionary with all parameters
            position = {
                'ticket': len(self.open_positions) + 1,  # Unique identifier
                'type': trade_type,
                'volume': volume,
                'price': current_price,
                'sl': sl,
                'tp': tp,
                'strategy': strategy,
                'open_time': trade_time,
                'risk_amount': self.virtual_balance * (self.strategies[strategy]['risk_percent'] / 100)
            }
            
            # Log position details
            self.logger.info(
                f"Position created: {strategy}, {trade_type.upper()}, "
                f"Volume: {volume:.2f}, Price: {current_price:.5f}, "
                f"SL: {sl:.5f}, TP: {tp:.5f}"
            )
            
            return position
            
        except Exception as e:
            self.logger.error(f"Error creating position: {str(e)}")
            return None
        
    def _calculate_position_profit(self, position: Dict, current_price: float) -> float:
        """Calculate position profit efficiently"""
        try:
            if position['type'] == 'buy':
                points = current_price - position['price']
            else:  # sell
                points = position['price'] - current_price
                
            # Calculate profit using point value and volume
            profit = points * position['volume'] * 100000
            return profit
            
        except Exception as e:
            self.logger.error(f"Error calculating position profit: {str(e)}")
            return 0.0

    def _check_close_conditions(self, position: Dict, current_price: float) -> Optional[str]:
        """Check if position should be closed"""
        try:
            if position['type'] == 'buy':
                if current_price >= position['tp']:
                    return 'TP'
                elif current_price <= position['sl']:
                    return 'SL'
            else:  # sell
                if current_price <= position['tp']:
                    return 'TP'
                elif current_price >= position['sl']:
                    return 'SL'
            return None
            
        except Exception as e:
            self.logger.error(f"Error checking close conditions: {str(e)}")
            return None

    def _process_closed_position(self, position: Dict, profit: float, 
                            close_reason: str, current_price: float,
                            current_time: datetime) -> None:
        """Process and log closed position details"""
        try:
            roi = (profit / position['risk_amount']) * 100
            
            # Log position closure details
            print(f"\n{'='*50}")
            print(f"📍 ORDER #{position['ticket']} CLOSED | {current_time}")
            print(f"📊 Strategy: {position['strategy']}")
            print(f"📈 Type: {position['type'].upper()}")
            print(f"📦 Volume: {position['volume']:.2f} lot")
            print(f"💰 Entry → Exit: {position['price']:.5f} → {current_price:.5f}")
            print(f"💵 P/L: ${profit:.2f} ({roi:.1f}% ROI)")
            print(f"❗ Reason: {close_reason}")
            print(f"⚖️ New Balance: ${self.virtual_balance:.2f}")
            print(f"⏱️ Duration: {current_time - position['open_time']}")
            
            # Store trade history
            self.trade_history.append({
                **position,
                'close_price': current_price,
                'close_time': current_time,
                'profit': profit,
                'roi': roi,
                'close_reason': close_reason
            })
            
        except Exception as e:
            self.logger.error(f"Error processing closed position: {str(e)}")

    def _handle_closed_positions(self, current_price: float, current_time: datetime) -> None:
        """Handle all closed positions efficiently"""
        try:
            if not self.open_positions:
                return
                
            # Check all positions
            positions_to_close = []
            total_profit = 0
            
            for pos in self.open_positions:
                profit = self._calculate_position_profit(pos, current_price)
                close_reason = self._check_close_conditions(pos, current_price)
                
                if close_reason:
                    total_profit += profit
                    positions_to_close.append((pos, profit, close_reason))
                    
            # Process closed positions
            for pos, profit, reason in positions_to_close:
                self.virtual_balance += profit
                self._process_closed_position(pos, profit, reason, current_price, current_time)
                self.open_positions.remove(pos)
                
        except Exception as e:
            self.logger.error(f"Error handling closed positions: {str(e)}")