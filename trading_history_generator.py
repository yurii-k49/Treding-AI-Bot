import os
import json
import asyncio
from datetime import datetime, timedelta
import pandas as pd
from config.settings import Config
from models.model_manager import ModelManager
from utils.data_loader import DataLoader
from utils.preprocessing import DataPreprocessor

class TradingHistoryGenerator:
    def __init__(self):
        self.config = Config()
        self.data_loader = DataLoader(self.config)
        self.model_manager = ModelManager(self.config)
        self.preprocessor = DataPreprocessor(self.config)

    async def generate_history(self):
        try:
            print("Getting historical data...")
            # Get historical data
            historical_data = await self.data_loader.get_historical_data(
                symbol=self.config.SYMBOL,
                timeframe=self.data_loader.timeframe,
                bars=10000
            )

            if historical_data is None or len(historical_data) == 0:
                raise Exception("No historical data available")

            print("Preprocessing data...")
            # Prepare features
            features = self.preprocessor.prepare_technical_features(historical_data)

            print("Getting model predictions...")
            # Get model predictions
            predictions = await self.model_manager.technical_model.predict(features)

            print("Generating trades based on predictions...")
            # Generate trades based on predictions
            trades = []
            balance = 10000  # Starting balance
            ticket = 1000

            for i, signal in enumerate(predictions['signal']):
                confidence = predictions['confidence'][i]
                
                if abs(signal) > 0.5 and confidence > 0.6:  # Only trade on strong signals
                    # Calculate position size (risk 2% of balance)
                    risk_amount = balance * 0.02
                    
                    # Get price data
                    open_price = historical_data.iloc[i]['open']
                    close_price = historical_data.iloc[i]['close']
                    high = historical_data.iloc[i]['high']
                    low = historical_data.iloc[i]['low']
                    
                    # Determine trade type
                    trade_type = 0 if signal > 0 else 1  # 0=BUY, 1=SELL
                    
                    # Calculate profit based on trade direction
                    direction = 1 if trade_type == 0 else -1
                    price_change = direction * (close_price - open_price)
                    profit = risk_amount * (price_change / open_price)

                    trade = {
                        'date': historical_data.index[i].strftime('%Y-%m-%d %H:%M:%S'),
                        'ticket': ticket,
                        'symbol': self.config.SYMBOL,
                        'type': trade_type,
                        'volume': round(risk_amount/10000, 2),  # Convert to lot size
                        'open_price': float(open_price),
                        'close_price': float(close_price),
                        'high': float(high),
                        'low': float(low),
                        'profit': round(profit, 2),
                        'balance': round(balance + profit, 2),
                        'signal_strength': float(signal),
                        'confidence': float(confidence)
                    }
                    
                    balance += profit
                    trades.append(trade)
                    ticket += 1

            # Save trades to file
            os.makedirs('data/history', exist_ok=True)
            with open('data/history/trading_history.json', 'w') as f:
                json.dump(trades, f, indent=2)

            # Print summary
            winning_trades = len([t for t in trades if t['profit'] > 0])
            total_trades = len(trades)
            win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0

            print("\nTrading Summary")
            print("===============")
            print(f"Total Trades: {total_trades}")
            print(f"Winning Trades: {winning_trades}")
            print(f"Win Rate: {win_rate:.1f}%")
            print(f"Initial Balance: ${10000:.2f}")
            print(f"Final Balance: ${balance:.2f}")
            print(f"Total Profit: ${(balance - 10000):.2f}")

            return trades

        except Exception as e:
            print(f"Error generating trading history: {str(e)}")
            raise

async def generate_trading_history():
    generator = TradingHistoryGenerator()
    return await generator.generate_history()

if __name__ == "__main__":
    asyncio.run(generate_trading_history())