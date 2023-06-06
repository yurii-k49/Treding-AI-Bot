# config/settings.py
import os
import logging
from dotenv import load_dotenv

load_dotenv()


class Config:
    # MT5 settings
    MT5_LOGIN = int(os.getenv('MT5_LOGIN', '313025394'))
    MT5_PASSWORD = os.getenv('MT5_PASSWORD', '5579187Er@')
    MT5_SERVER = os.getenv('MT5_SERVER', 'XMGlobal-MT5 7')
    MODE = os.getenv('MODE', 'demo')
    
    # Trading settings
    SYMBOL = os.getenv('SYMBOL', 'EURUSD')
    TIMEFRAME = os.getenv('TIMEFRAME', 'H1')  # M1, M5, M15, M30, H1, H4, D1, W1, MN1
    LOT_SIZE = float(os.getenv('LOT_SIZE', '0.01'))
    
    # API kalitlari
    ECONOMIC_API_KEY = os.getenv('ECONOMIC_API_KEY')
    NEWS_API_KEY = os.getenv('NEWS_API_KEY')
    CRYPTO_API_KEY = os.getenv('CRYPTO_API_KEY')
    
    # Fundamental analiz
    USE_FUNDAMENTAL_ANALYSIS = os.getenv('USE_FUNDAMENTAL_ANALYSIS', 'true').lower() == 'true'
    MACRO_ANALYSIS = os.getenv('MACRO_ANALYSIS', 'true').lower() == 'true'
    COMPANY_ANALYSIS = os.getenv('COMPANY_ANALYSIS', 'true').lower() == 'true'
    SECTOR_ANALYSIS = os.getenv('SECTOR_ANALYSIS', 'true').lower() == 'true'
    
    # Risk management
    USE_RISK_MANAGEMENT = os.getenv('USE_RISK_MANAGEMENT', 'true').lower() == 'true'
    MAX_DAILY_LOSS = float(os.getenv('MAX_DAILY_LOSS', '-2.0'))
    MAX_POSITION_SIZE = float(os.getenv('MAX_POSITION_SIZE', '0.1'))
    
    # AI model parametrlari
    MODEL_UPDATE_INTERVAL = int(os.getenv('MODEL_UPDATE_INTERVAL', '168'))
    FEATURE_ENGINEERING = os.getenv('FEATURE_ENGINEERING', 'true').lower() == 'true'
    DEEP_LEARNING = os.getenv('DEEP_LEARNING', 'false').lower() == 'true'
    
    # Signal weights
    SIGNAL_WEIGHTS = {
        'technical': float(os.getenv('TECHNICAL_WEIGHT', '0.4')),
        'fundamental': float(os.getenv('FUNDAMENTAL_WEIGHT', '0.4')),
        'sentiment': float(os.getenv('SENTIMENT_WEIGHT', '0.2'))
    }

# config/logging_config.py
import logging
import os
from datetime import datetime

def setup_logging():
    # Logging directory yaratish
    log_dir = 'logs'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # Log fayl nomi
    log_file = f"{log_dir}/trading_{datetime.now().strftime('%Y%m%d')}.log"
    
    # Logging konfiguratsiyasi
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    # Trading specific loggers
    trading_logger = logging.getLogger('trading')
    trading_logger.setLevel(logging.INFO)
    
    analysis_logger = logging.getLogger('analysis')
    analysis_logger.setLevel(logging.INFO)
    
    return trading_logger, analysis_logger