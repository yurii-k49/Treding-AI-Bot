# test_data_loader.py
import asyncio
import pandas as pd
import logging
from config.settings import Config
from utils.data_loader import DataLoader

async def test_data_loading():
    config = Config()
    loader = DataLoader(config)
    
    print("\n1. Kichik hajmdagi ma'lumotlarni olish testi...")
    historical_small = await loader.get_historical_data(bars=10)
    if historical_small is not None:
        print(f"Muvaffaqiyatli! {len(historical_small)} qator olindi")
        print(historical_small.head())

    print("\n2. Joriy bozor ma'lumotlarini olish...")
    market_data = await loader.get_market_data()
    if market_data:
        print("\nBozor ma'lumotlari:")
        
        print("\na) Narx ma'lumotlari:")
        print(f"- Bid: {market_data['bid']}")
        print(f"- Ask: {market_data['ask']}")
        print(f"- Last: {market_data['last']}")
        print(f"- Spread: {market_data['spread']}")
        
        print("\nb) Hajm ma'lumotlari:")
        print(f"- Volume: {market_data['volume']}")
        print(f"- Volume Min: {market_data['volume_min']}")
        print(f"- Volume Max: {market_data['volume_max']}")
        print(f"- Volume Step: {market_data['volume_step']}")
        
        print("\nc) Margin ma'lumotlari:")
        print(f"- Initial Margin: {market_data['margin_initial']}")
        print(f"- Maintenance Margin: {market_data['margin_maintenance']}")

    print("\n3. Pozitsiyalar ma'lumotlarini olish...")
    positions = await loader.get_positions_data()
    if not positions.empty:
        print("\nOchiq pozitsiyalar:")
        print(positions[['symbol', 'type', 'volume', 'profit', 'price_open', 'price_current']])
    else:
        print("Ochiq pozitsiyalar yo'q")

    print("\n4. Orderlar tarixini olish...")
    orders = await loader.get_orders_history(days=7)
    if not orders.empty:
        print(f"\nOxirgi 7 kunlik orderlar ({len(orders)} ta):")
        print(orders[['symbol', 'type', 'volume', 'profit', 'price', 'time']])
    else:
        print("Order'lar tarixi topilmadi")

    print("\n5. Training ma'lumotlarini olish...")
    try:
        training_data = await loader.get_training_data(bars=1000)
        if training_data:
            print("\nTraining ma'lumotlari tarkibi:")
            for key, value in training_data.items():
                if isinstance(value, pd.DataFrame):
                    print(f"\n{key.upper()} ma'lumotlari ({len(value)} qator):")
                    if not value.empty:
                        print(value.head())
    except Exception as e:
        print(f"Training ma'lumotlarini olishda xatolik: {str(e)}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(test_data_loading())