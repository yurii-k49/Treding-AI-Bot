# test_connection.py
import MetaTrader5 as mt5
import os
from dotenv import load_dotenv

def test_connection():
    # .env dan ma'lumotlarni olish
    load_dotenv()
    login = int(os.getenv('MT5_LOGIN', '313025394'))
    password = os.getenv('MT5_PASSWORD', '5579187Er@')
    server = os.getenv('MT5_SERVER', 'XMGlobal-MT5 7')
    
    # MT5 ni ishga tushirish
    if not mt5.initialize():
        print("MT5 initialization failed")
        return False
        
    # Hisobga kirish
    if not mt5.login(login=login, password=password, server=server):
        print("Login failed")
        return False
        
    # Hisob ma'lumotlarini olish
    account_info = mt5.account_info()
    if account_info is not None:
        print("\nConnection successful!")
        print(f"Login: {account_info.login}")
        print(f"Server: {account_info.server}")
        print(f"Balance: {account_info.balance}")
        print(f"Equity: {account_info.equity}")
    else:
        print("Failed to get account info")
        return False
        
    mt5.shutdown()
    return True

if __name__ == "__main__":
    test_connection()