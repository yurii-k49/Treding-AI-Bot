# main.py
import argparse
import sys
import logging
import traceback
from datetime import datetime
import MetaTrader5 as mt5
from historical_trader import HistoricalTrader
from trading_execution import TradingExecution
import multiprocessing as mp

logging.basicConfig(
    filename=f'trading_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def main():
    parser = argparse.ArgumentParser(description='Trading Bot')
    parser.add_argument('mode', choices=['train', 'trade'],
                      help='Bot mode: train or trade')
    parser.add_argument('--symbol', type=str, default='EURUSD',
                      help='Trading symbol')
    parser.add_argument('--timeframe', type=int, default=mt5.TIMEFRAME_M5,
                      help='Trading timeframe')
    parser.add_argument('--episodes', type=int, default=100,
                      help='Number of episodes')
    parser.add_argument('--model', type=str, default='models/latest_model.h5',
                      help='Model file for trading')
    parser.add_argument('--days', type=int, default=30,
                      help='Days of historical data to load')
    
    args = parser.parse_args()
    
    try:
        if not mt5.initialize():
            raise Exception("MT5 ishga tushmadi!")
            
        logging.info("MT5 ishga tushdi")
        
        if args.mode == 'train':
            trader = HistoricalTrader(symbol=args.symbol, timeframe=args.timeframe)
            if not trader.load_historical_data(days=args.days):
                raise Exception("Tarixiy ma'lumotlarni yuklashda xatolik!")
            trader.train_on_historical(episodes=args.episodes)
        else:
            trader = TradingExecution(symbol=args.symbol, timeframe=args.timeframe)
            if os.path.exists(args.model):
                trader.load_model(args.model)
                trader.trade()
            else:
                logging.error(f"Model fayli topilmadi: {args.model}")
            
    except Exception as e:
        logging.error(f"Kritik xatolik: {str(e)}")
        logging.error(traceback.format_exc())
        
    finally:
        mt5.shutdown()
        logging.info("Dastur yakunlandi\n" + "="*50 + "\n")

if __name__ == "__main__":
    mp.set_start_method('spawn')
    main()