# analysis/technical.py
import pandas as pd
import numpy as np
import logging
from ta import trend, momentum, volatility, volume

class TechnicalAnalyzer:
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger('analysis.technical')

    async def analyze(self, data):
        """Run technical analysis"""
        try:
            # Calculate all indicators
            df = self.calculate_all_indicators(data)
            
            # Generate signals
            signals = self.generate_signals(df)
            
            # Detect patterns
            patterns = self.detect_patterns(df)
            
            # Calculate signal strength and confidence
            signal_strength = self.calculate_signal_strength(signals)
            confidence = self.calculate_confidence(signals, patterns)
            
            return {
                'signals': signals,
                'patterns': patterns,
                'strength': signal_strength,
                'confidence': confidence,
                'indicators': df
            }
            
        except Exception as e:
            self.logger.error(f"Error in technical analysis: {str(e)}")
            return None

    def calculate_all_indicators(self, df):
        """Calculate technical indicators using ta library"""
        try:
            # Trend indicators
            df['ema_9'] = trend.ema_indicator(df['close'], window=9)
            df['ema_21'] = trend.ema_indicator(df['close'], window=21)
            df['ema_50'] = trend.ema_indicator(df['close'], window=50)
            df['ema_200'] = trend.ema_indicator(df['close'], window=200)
            
            # MACD
            df['macd'] = trend.macd_diff(df['close'])
            df['macd_signal'] = trend.macd_signal(df['close'])
            
            # Momentum indicators
            df['rsi'] = momentum.rsi(df['close'])
            df['stoch_k'] = momentum.stoch(df['high'], df['low'], df['close'])
            df['stoch_d'] = momentum.stoch_signal(df['high'], df['low'], df['close'])
            
            # Volatility indicators
            df['bb_upper'] = volatility.bollinger_hband(df['close'])
            df['bb_middle'] = volatility.bollinger_mavg(df['close'])
            df['bb_lower'] = volatility.bollinger_lband(df['close'])
            df['atr'] = volatility.average_true_range(df['high'], df['low'], df['close'])
            
            # Volume indicators
            df['volume_ema'] = volume.volume_weighted_average_price(df['high'], df['low'], df['close'], df['volume'])
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error calculating indicators: {str(e)}")
            return df

    def generate_signals(self, df):
        """Generate trading signals from indicators"""
        try:
            signals = {}
            
            # Trend signals
            signals['trend'] = self._calculate_trend_signals(df)
            
            # Momentum signals
            signals['momentum'] = self._calculate_momentum_signals(df)
            
            # Volatility signals
            signals['volatility'] = self._calculate_volatility_signals(df)
            
            return signals
            
        except Exception as e:
            self.logger.error(f"Error generating signals: {str(e)}")
            return {}

    def _calculate_trend_signals(self, df):
        """Calculate trend based signals"""
        signals = {}
        
        # EMA crossovers
        signals['ema_cross'] = np.where(df['ema_9'] > df['ema_21'], 1, 
                                      np.where(df['ema_9'] < df['ema_21'], -1, 0))
        
        # MACD signals
        signals['macd'] = np.where(df['macd'] > df['macd_signal'], 1, -1)
        
        return signals

    def _calculate_momentum_signals(self, df):
        """Calculate momentum based signals"""
        signals = {}
        
        # RSI signals
        signals['rsi'] = np.where(df['rsi'] < 30, 1, 
                                np.where(df['rsi'] > 70, -1, 0))
        
        # Stochastic signals
        signals['stoch'] = np.where((df['stoch_k'] < 20) & (df['stoch_k'] > df['stoch_d']), 1,
                                  np.where((df['stoch_k'] > 80) & (df['stoch_k'] < df['stoch_d']), -1, 0))
        
        return signals

    def _calculate_volatility_signals(self, df):
        """Calculate volatility based signals"""
        signals = {}
        
        # Bollinger Bands signals
        signals['bb'] = np.where(df['close'] < df['bb_lower'], 1,
                               np.where(df['close'] > df['bb_upper'], -1, 0))
        
        return signals

    def detect_patterns(self, df):
        """Detect chart patterns"""
        patterns = {}
        
        # Engulfing patterns
        patterns['engulfing'] = self._detect_engulfing(df)
        
        # Doji patterns
        patterns['doji'] = self._detect_doji(df)
        
        return patterns

    def _detect_engulfing(self, df):
        """Detect engulfing candlestick patterns"""
        bullish_engulfing = (df['open'] < df['close']) & \
                           (df['open'].shift(1) > df['close'].shift(1)) & \
                           (df['open'] < df['close'].shift(1)) & \
                           (df['close'] > df['open'].shift(1))
                           
        bearish_engulfing = (df['open'] > df['close']) & \
                           (df['open'].shift(1) < df['close'].shift(1)) & \
                           (df['open'] > df['close'].shift(1)) & \
                           (df['close'] < df['open'].shift(1))
                           
        return np.where(bullish_engulfing, 1, np.where(bearish_engulfing, -1, 0))

    def _detect_doji(self, df):
        """Detect doji candlestick patterns"""
        body_size = abs(df['close'] - df['open'])
        total_size = df['high'] - df['low']
        
        return np.where(body_size / total_size < 0.1, 1, 0)

    def calculate_signal_strength(self, signals):
        """Calculate overall signal strength"""
        try:
            all_signals = []
            
            # Combine all signals
            for category in signals.values():
                for signal in category.values():
                    all_signals.append(signal[-1])  # Get latest signal
                    
            # Calculate average signal
            avg_signal = np.mean(all_signals)
            
            # Normalize to [-1, 1]
            return np.clip(avg_signal, -1, 1)
            
        except Exception as e:
            self.logger.error(f"Error calculating signal strength: {str(e)}")
            return 0

    def calculate_confidence(self, signals, patterns):
        """Calculate confidence level of signals"""
        try:
            # Count confirming signals
            confirming_signals = 0
            total_signals = 0
            
            # Check trend signals
            trend_direction = np.sign(self.calculate_signal_strength(signals))
            
            for category in signals.values():
                for signal in category.values():
                    if np.sign(signal[-1]) == trend_direction:
                        confirming_signals += 1
                    total_signals += 1
                    
            # Add pattern confirmation
            for pattern in patterns.values():
                if np.sign(pattern[-1]) == trend_direction:
                    confirming_signals += 1
                total_signals += 1
                
            return confirming_signals / total_signals if total_signals > 0 else 0
            
        except Exception as e:
            self.logger.error(f"Error calculating confidence: {str(e)}")
            return 0