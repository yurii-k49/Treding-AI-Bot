# test_mt5.py
import MetaTrader5 as mt5
from datetime import datetime
import time

def test_mt5_connection():
    # MT5 ga ulanish
    print("MetaTrader5 package author: ", mt5.__author__)
    print("MetaTrader5 package version: ", mt5.__version__)
    
    # MT5 ni ishga tushirish
    if not mt5.initialize():
        print("initialize() failed")
        mt5_error = mt5.last_error()
        print(f"Error: {mt5_error}")
        return False
        
    # Terminal ma'lumotlari
    terminal_info = mt5.terminal_info()
    if terminal_info is not None:
        print("\nTerminal Info:")
        print("Connected:", terminal_info.connected)
        print("Enable Trading:", terminal_info.trade_allowed)
        print("Path:", terminal_info.path)
    
    # Account ma'lumotlari
    account_info = mt5.account_info()
    if account_info is not None:
        print("\nAccount Info:")
        print("Login:", account_info.login)
        print("Server:", account_info.server)
        print("Balance:", account_info.balance)
        print("Equity:", account_info.equity)
        print("Margin:", account_info.margin)
        print("Free Margin:", account_info.margin_free)
        
    # Mavjud symbollar
    symbols = mt5.symbols_get()
    if symbols is not None:
        print("\nTotal symbols:", len(symbols))
        print("First 5 symbols:")
        for s in symbols[:5]:
            print(s.name)
            
    # EURUSD ma'lumotlarini olish
    eurusd_info = mt5.symbol_info("EURUSD")
    if eurusd_info is not None:
        print("\nEURUSD Info:")
        print("Bid:", eurusd_info.bid)
        print("Ask:", eurusd_info.ask)
        print("Spread:", eurusd_info.spread)
        
    mt5.shutdown()
    return True

if __name__ == "__main__":
    success = test_mt5_connection()
    if success:
        print("\nTest completed successfully")
    else:
        print("\nTest failed")