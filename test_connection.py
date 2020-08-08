import MetaTrader5 as mt5
import time

def test_mt5_connection():
    try:
        # MT5 ni ishga tushirish
        if not mt5.initialize():
            print(f"MT5 ishga tushirilmadi, xato kodi: {mt5.last_error()}")
            return False
            
        # Terminal ma'lumotlarini olish
        terminal_info = mt5.terminal_info()
        if terminal_info is None:
            print(f"Terminal ma'lumotlarini ololmadik, xato kodi: {mt5.last_error()}")
            return False
            
        print("\nMT5 Terminal ma'lumotlari:")
        print(f"Terminal nomi: {terminal_info.name}")
        print(f"Terminal versiyasi: {terminal_info.version}")
        print(f"Terminal papkasi: {terminal_info.path}")
        print(f"Trading serveri: {terminal_info.connected}")
        
        # Account ma'lumotlarini olish
        account_info = mt5.account_info()
        if account_info is None:
            print(f"Account ma'lumotlarini ololmadik, xato kodi: {mt5.last_error()}")
            return False
            
        print("\nAccount ma'lumotlari:")
        print(f"Login: {account_info.login}")
        print(f"Server: {account_info.server}")
        print(f"Balance: {account_info.balance}")
        print(f"Currency: {account_info.currency}")
        
        # Test uchun EURUSD ma'lumotlarini olish
        symbol = "EURUSD"
        timeframe = mt5.TIMEFRAME_D1
        
        rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, 1)
        if rates is None:
            print(f"Ma'lumotlarni ololmadik {symbol}, xato kodi: {mt5.last_error()}")
            return False
            
        print(f"\n{symbol} oxirgi ma'lumoti:")
        print(rates[0])
        
        return True
        
    except Exception as e:
        print(f"Xatolik yuz berdi: {e}")
        return False
        
    finally:
        mt5.shutdown()

if __name__ == "__main__":
    print("MT5 bilan bog'lanishni tekshirish...")
    if test_mt5_connection():
        print("\nMT5 bilan bog'lanish muvaffaqiyatli!")
    else:
        print("\nMT5 bilan bog'lanishda xatolik!")
    time.sleep(5)  # Natijalarni ko'rish uchun 5 sekund kutish