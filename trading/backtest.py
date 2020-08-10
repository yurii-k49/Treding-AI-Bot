# trading/backtest.py

import pandas as pd
import numpy as np
from datetime import datetime
import logging
from tqdm import tqdm

class Backtester:
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger('trading.backtest')
        self.initial_balance = 10000  # Starting balance
        self.commission = 0.0001      # Commission per trade (0.01%)
        self.positions = []
        self.trades = []
        self.equity_curve = []
        
    async def run_backtest(self, data, predictions):
        """Run backtest simulation"""
        try:
            self.logger.info("\n=== Starting Backtest ===")
            balance = self.initial_balance
            position = None
            
            # Initialize metrics tracking
            total_trades = 0
            winning_trades = 0
            losing_trades = 0
            max_drawdown = 0
            peak_balance = balance
            
            # Convert data to DataFrame if necessary
            df = pd.DataFrame(data) if not isinstance(data, pd.DataFrame) else data
            
            # Progress bar
            with tqdm(total=len(df), desc="Running backtest") as pbar:
                for i in range(1, len(df)):
                    current_price = df['close'].iloc[i]
                    signal = predictions['signal'][i-1]
                    confidence = predictions['confidence'][i-1]
                    
                    # Update equity curve
                    self.equity_curve.append({
                        'date': df.index[i],
                        'balance': balance,
                        'equity': balance + (position['pnl'] if position else 0)
                    })
                    
                    # Check for position exit
                    if position is not None:
                        # Calculate P/L
                        if position['type'] == 'buy':
                            pnl = (current_price - position['entry_price']) * position['size']
                        else:
                            pnl = (position['entry_price'] - current_price) * position['size']
                            
                        position['pnl'] = pnl
                        
                        # Check exit conditions
                        if (position['type'] == 'buy' and signal == 2) or \
                           (position['type'] == 'sell' and signal == 1):
                            # Close position
                            balance += pnl - (current_price * position['size'] * self.commission)
                            
                            # Record trade
                            trade = {
                                'entry_date': position['entry_date'],
                                'exit_date': df.index[i],
                                'type': position['type'],
                                'entry_price': position['entry_price'],
                                'exit_price': current_price,
                                'size': position['size'],
                                'pnl': pnl,
                                'balance': balance
                            }
                            self.trades.append(trade)
                            
                            # Update metrics
                            total_trades += 1
                            if pnl > 0:
                                winning_trades += 1
                            else:
                                losing_trades += 1
                                
                            position = None
                    
                    # Check for new position entry
                    elif signal != 0 and confidence > 0.6:  # Only trade on high confidence
                        # Calculate position size (1% risk per trade)
                        risk_amount = balance * 0.01
                        position_size = risk_amount / current_price
                        
                        # Open new position
                        position = {
                            'type': 'buy' if signal == 1 else 'sell',
                            'entry_price': current_price,
                            'entry_date': df.index[i],
                            'size': position_size,
                            'pnl': 0
                        }
                        
                        # Deduct commission
                        balance -= current_price * position_size * self.commission
                    
                    # Update max drawdown
                    if balance > peak_balance:
                        peak_balance = balance
                    drawdown = (peak_balance - balance) / peak_balance
                    max_drawdown = max(max_drawdown, drawdown)
                    
                    pbar.update(1)
            
            # Calculate final metrics
            metrics = {
                'initial_balance': self.initial_balance,
                'final_balance': balance,
                'total_return': ((balance / self.initial_balance) - 1) * 100,
                'total_trades': total_trades,
                'winning_trades': winning_trades,
                'losing_trades': losing_trades,
                'win_rate': winning_trades / total_trades if total_trades > 0 else 0,
                'max_drawdown': max_drawdown * 100,
                'profit_factor': self._calculate_profit_factor(),
                'sharpe_ratio': self._calculate_sharpe_ratio(),
                'avg_trade': self._calculate_average_trade(),
            }
            
            self.logger.info("\nBacktest Results:")
            for metric, value in metrics.items():
                self.logger.info(f"{metric}: {value:.2f}" if isinstance(value, float) else f"{metric}: {value}")
            
            return {
                'metrics': metrics,
                'trades': self.trades,
                'equity_curve': self.equity_curve
            }
            
        except Exception as e:
            self.logger.error(f"Error in backtest: {str(e)}")
            return None
            
    def _calculate_profit_factor(self):
        """Calculate profit factor"""
        winning_trades = [t['pnl'] for t in self.trades if t['pnl'] > 0]
        losing_trades = [abs(t['pnl']) for t in self.trades if t['pnl'] < 0]
        
        total_profit = sum(winning_trades)
        total_loss = sum(losing_trades)
        
        return total_profit / total_loss if total_loss > 0 else float('inf')
        
    def _calculate_sharpe_ratio(self):
        """Calculate Sharpe ratio"""
        if not self.equity_curve:
            return 0
            
        returns = pd.Series([e['equity'] for e in self.equity_curve]).pct_change().dropna()
        
        if len(returns) < 2:
            return 0
            
        return np.sqrt(252) * (returns.mean() / returns.std())
        
    def _calculate_average_trade(self):
        """Calculate average trade profit/loss"""
        if not self.trades:
            return 0
            
        return np.mean([t['pnl'] for t in self.trades])
        
    def get_monthly_returns(self):
        """Calculate monthly returns"""
        if not self.equity_curve:
            return pd.Series()
            
        equity = pd.DataFrame(self.equity_curve).set_index('date')['equity']
        return equity.resample('M').last().pct_change()
        
    def get_drawdown_periods(self):
        """Get significant drawdown periods"""
        if not self.equity_curve:
            return []
            
        equity = pd.DataFrame(self.equity_curve).set_index('date')['equity']
        
        drawdown = equity / equity.cummax() - 1
        drawdown_periods = []
        
        # Find periods with >5% drawdown
        threshold = -0.05
        is_drawdown = False
        start_date = None
        
        for date, value in drawdown.items():
            if value < threshold and not is_drawdown:
                is_drawdown = True
                start_date = date
            elif value >= threshold and is_drawdown:
                is_drawdown = False
                drawdown_periods.append({
                    'start': start_date,
                    'end': date,
                    'drawdown': abs(drawdown[start_date:date].min()) * 100
                })
                
        return drawdown_periods