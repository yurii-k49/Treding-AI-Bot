# utils/data_loader.py
import pandas as pd
import numpy as np
import MetaTrader5 as mt5
from datetime import datetime, timedelta

class DataLoader:
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger('utils.data_loader')
        
    async def get_historical_data(self, symbol, timeframe, bars=1000):
        """Get historical data from MT5"""
        try:
            rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, bars)
            
            if rates is None:
                raise Exception("Failed to get historical data")
                
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error loading historical data: {str(e)}")
            return None
            
    async def get_fundamental_data(self, symbol):
        """Get fundamental data from various sources"""
        try:
            # Economic calendar data
            calendar_data = await self._get_economic_calendar()
            
            # Financial statements
            financial_data = await self._get_financial_statements(symbol)
            
            # Market news
            news_data = await self._get_market_news(symbol)
            
            return {
                'calendar': calendar_data,
                'financial': financial_data,
                'news': news_data
            }
            
        except Exception as e:
            self.logger.error(f"Error loading fundamental data: {str(e)}")
            return None
    
    async def save_model_data(self, model_data, model_name):
        """Save model data to disk"""
        try:
            path = f"data/models/{model_name}_{datetime.now().strftime('%Y%m%d')}.pkl"
            pd.to_pickle(model_data, path)
            self.logger.info(f"Model data saved to {path}")
        except Exception as e:
            self.logger.error(f"Error saving model data: {str(e)}")

# utils/preprocessing.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

class DataPreprocessor:
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger('utils.preprocessor')
        self.scaler = StandardScaler()
        
    def prepare_technical_features(self, df):
        """Prepare features for technical analysis"""
        try:
            # Calculate returns
            df['returns'] = df['close'].pct_change()
            df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
            
            # Volume features
            df['volume_ma'] = df['tick_volume'].rolling(window=20).mean()
            df['volume_std'] = df['tick_volume'].rolling(window=20).std()
            
            # Price features
            df['price_ma'] = df['close'].rolling(window=20).mean()
            df['price_std'] = df['close'].rolling(window=20).std()
            
            # Clean and normalize
            df = self._clean_data(df)
            df = self._normalize_features(df)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error in technical preprocessing: {str(e)}")
            return None
            
    def prepare_fundamental_features(self, data):
        """Prepare features for fundamental analysis"""
        try:
            features = pd.DataFrame()
            
            # Economic indicators
            if 'calendar' in data:
                features = self._process_economic_data(data['calendar'], features)
            
            # Financial metrics
            if 'financial' in data:
                features = self._process_financial_data(data['financial'], features)
            
            # News sentiment
            if 'news' in data:
                features = self._process_news_data(data['news'], features)
            
            return features
            
        except Exception as e:
            self.logger.error(f"Error in fundamental preprocessing: {str(e)}")
            return None
            
    def _clean_data(self, df):
        """Clean the dataset"""
        # Remove duplicates
        df = df.drop_duplicates()
        
        # Handle missing values
        df = df.fillna(method='ffill')
        
        # Remove outliers
        df = self._remove_outliers(df)
        
        return df
        
    def _normalize_features(self, df):
        """Normalize numerical features"""
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        df[numeric_columns] = self.scaler.fit_transform(df[numeric_columns])
        return df