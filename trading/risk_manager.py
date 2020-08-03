# trading/risk_manager.py
import numpy as np
from datetime import datetime, timedelta
import logging

class RiskManager:
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger('trading.risk')
        
        # Risk limits
        self.max_daily_loss = config.MAX_DAILY_LOSS
        self.max_position_size = config.MAX_POSITION_SIZE
        self.risk_per_trade = 0.02  # 2% risk per trade
        
    def check_risk_limits(self, account_info, new_position):
        """Check if new position meets risk limits"""
        try:
            # Check daily loss limit
            if self._check_daily_loss_limit(account_info):
                self.logger.warning("Daily loss limit reached")
                return False
            
            # Check position size limit
            if not self._check_position_size_limit(new_position):
                self.logger.warning("Position size limit exceeded")
                return False
            
            # Check margin requirements
            if not self._check_margin_requirements(account_info, new_position):
                self.logger.warning("Insufficient margin")
                return False
            
            # Check correlation risk
            if not self._check_correlation_risk(new_position):
                self.logger.warning("High correlation risk detected")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error in risk check: {str(e)}")
            return False
            
    def calculate_position_size(self, signal_strength, account_info, atr):
        """Calculate optimal position size"""
        try:
            # Base position size calculation
            equity = account_info['equity']
            risk_amount = equity * self.risk_per_trade
            
            # ATR-based stop loss
            stop_loss_pips = atr * 1.5
            
            # Calculate position size in lots
            position_size = (risk_amount / stop_loss_pips) / 10
            
            # Adjust based on signal strength
            adjusted_size = position_size * signal_strength
            
            # Apply limits
            final_size = min(adjusted_size, self.max_position_size)
            
            return max(0.01, round(final_size, 2))  # Minimum 0.01 lots
            
        except Exception as e:
            self.logger.error(f"Error calculating position size: {str(e)}")
            return 0.01  # Return minimum size on error
            
    def calculate_stop_loss(self, entry_price, position_type, atr):
        """Calculate stop loss level"""
        try:
            atr_multiplier = 1.5
            
            if position_type == 'buy':
                stop_loss = entry_price - (atr * atr_multiplier)
            else:
                stop_loss = entry_price + (atr * atr_multiplier)
                
            return round(stop_loss, 5)
            
        except Exception as e:
            self.logger.error(f"Error calculating stop loss: {str(e)}")
            return None
            
    def calculate_take_profit(self, entry_price, stop_loss, position_type):
        """Calculate take profit level"""
        try:
            risk = abs(entry_price - stop_loss)
            reward_ratio = 2  # Risk:Reward ratio
            
            if position_type == 'buy':
                take_profit = entry_price + (risk * reward_ratio)
            else:
                take_profit = entry_price - (risk * reward_ratio)
                
            return round(take_profit, 5)
            
        except Exception as e:
            self.logger.error(f"Error calculating take profit: {str(e)}")
            return None
            
    def _check_daily_loss_limit(self, account_info):
        """Check if daily loss limit has been reached"""
        try:
            daily_pnl = account_info.get('daily_pnl', 0)
            equity = account_info.get('equity', 0)
            
            max_loss_amount = equity * abs(self.max_daily_loss)
            
            return daily_pnl <= -max_loss_amount
            
        except Exception as e:
            self.logger.error(f"Error checking daily loss limit: {str(e)}")
            return True  # Conservative approach
            
    def _check_position_size_limit(self, new_position):
        """Check if position size is within limits"""
        try:
            total_exposure = self._calculate_total_exposure(new_position)
            return total_exposure <= self.max_position_size
            
        except Exception as e:
            self.logger.error(f"Error checking position size limit: {str(e)}")
            return False
            
    def _check_margin_requirements(self, account_info, new_position):
        """Check if account has sufficient margin"""
        try:
            required_margin = self._calculate_required_margin(new_position)
            free_margin = account_info.get('free_margin', 0)
            
            return free_margin >= required_margin * 1.5  # 50% margin buffer
            
        except Exception as e:
            self.logger.error(f"Error checking margin requirements: {str(e)}")
            return False