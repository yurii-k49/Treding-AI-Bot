import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime
import time
from config.settings import Config

def test_mt5_connection():
    config = Config()
    
    print("\nMT5 bog'lanish testini boshlash...")
    print(f"Login: {config.MT5_LOGIN}")
    print(f"Server: {config.MT5_SERVER}")
    
    try:
        # MT5 ni ishga tushirish
        if not mt5.initialize(
            login=config.MT5_LOGIN,
            password=config.MT5_PASSWORD,
            server=config.MT5_SERVER
        ):
            print(f"\nMT5 ishga tushmadi. Xato: {mt5.last_error()}")
            return False
            
        print("\nMT5 muvaffaqiyatli ishga tushdi!")
        
        # Terminal ma'lumotlari
        terminal_info = mt5.terminal_info()
        if terminal_info is None:
            print(f"Terminal ma'lumotlarini ololmadik. Xato: {mt5.last_error()}")
            return False
            
        print("\nTerminal ma'lumotlari:")
        print(f"Nomi: {terminal_info.name}")
        print(f"Path: {terminal_info.path}")
        print(f"Ulanish holati: {'Ulangan' if terminal_info.connected else 'Ulanmagan'}")
        
        # Account ma'lumotlari
        account_info = mt5.account_info()
        if account_info is None:
            print(f"Account ma'lumotlarini ololmadik. Xato: {mt5.last_error()}")
            return False
            
        print("\nAccount ma'lumotlari:")
        print(f"Login: {account_info.login}")
        print(f"Server: {account_info.server}")
        print(f"Balance: {account_info.balance} {account_info.currency}")
        print(f"Leverage: 1:{account_info.leverage}")
        
        # EURUSD test ma'lumotlari
        symbol = config.SYMBOL
        print(f"\n{symbol} ma'lumotlarini olish...")
        
        timeframe_map = {
            'M1': mt5.TIMEFRAME_M1,
            'M5': mt5.TIMEFRAME_M5,
            'M15': mt5.TIMEFRAME_M15,
            'M30': mt5.TIMEFRAME_M30,
            'H1': mt5.TIMEFRAME_H1,
            'H4': mt5.TIMEFRAME_H4,
            'D1': mt5.TIMEFRAME_D1,
        }
        
        timeframe = timeframe_map.get(config.TIMEFRAME, mt5.TIMEFRAME_H1)
        
        rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, 1)
        if rates is None:
            print(f"{symbol} ma'lumotlarini ololmadik. Xato: {mt5.last_error()}")
            return False
            
        # Numpy array ni pandas DataFrame ga o'tkazish
        df = pd.DataFrame(rates)
        
        # time ustunini datetime formatiga o'tkazish
        df['time'] = pd.to_datetime(df['time'], unit='s')
        
        print(f"\n{symbol} oxirgi bar ma'lumotlari:")
        print(df.to_string(index=False))
        
        print("\nBarcha testlar muvaffaqiyatli o'tdi!")
        
        # Mavjud symbollarni ko'rish
        symbols = mt5.symbols_get()
        if symbols is None:
            print("\nSymbollar ro'yxatini ololmadik")
        else:
            print(f"\nJami symbollar soni: {len(symbols)}")
            print("\nBirinchi 5 ta symbol:")
            for symbol in symbols[:5]:
                print(f"{symbol.name}: {symbol.path}")
        
        return True
        
    except Exception as e:
        print(f"\nXatolik yuz berdi: {str(e)}")
        return False
        
    finally:
        mt5.shutdown()
        print("\nMT5 yopildi")

if __name__ == "__main__":
    test_mt5_connection()
    time.sleep(5)