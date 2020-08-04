# utils/preprocessing.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from datetime import datetime
import logging


class DataPreprocessor:
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger('utils.preprocessor')
        self.technical_scaler = StandardScaler()
        self.fundamental_scaler = MinMaxScaler()
        
    def prepare_technical_features(self, df):
        """Prepare features for technical analysis"""
        try:
            # Basic features
            df = self._calculate_returns(df)
            df = self._calculate_volatility_features(df)
            df = self._calculate_volume_features(df)
            
            if self.config.FEATURE_ENGINEERING:
                df = self._engineer_technical_features(df)
            
            # Clean and normalize
            df = self._clean_data(df)
            df = self._normalize_technical_features(df)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error in technical preprocessing: {str(e)}")
            return None
            
    def prepare_fundamental_features(self, data):
        """Prepare features for fundamental analysis"""
        try:
            features = pd.DataFrame()
            
            # Process different data types
            if 'economic' in data:
                features = self._process_economic_data(data['economic'], features)
                
            if 'financial' in data:
                features = self._process_financial_data(data['financial'], features)
                
            if 'news' in data:
                features = self._process_news_data(data['news'], features)
                
            # Normalize features
            features = self._normalize_fundamental_features(features)
            
            return features
            
        except Exception as e:
            self.logger.error(f"Error in fundamental preprocessing: {str(e)}")
            return None
            
    def _calculate_returns(self, df):
        """Calculate various return metrics"""
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        df['cumulative_returns'] = (1 + df['returns']).cumprod()
        
        return df
        
    def _calculate_volatility_features(self, df):
        """Calculate volatility related features"""
        # Price ranges
        df['daily_range'] = df['high'] - df['low']
        df['body_size'] = abs(df['close'] - df['open'])
        
        # Rolling volatility
        for window in [5, 10, 20]:
            df[f'volatility_{window}'] = df['returns'].rolling(window).std()
            
        return df
        
    def _calculate_volume_features(self, df):
        """Calculate volume related features"""
        # Volume momentum
        df['volume_momentum'] = df['tick_volume'].pct_change()
        
        # Rolling volume metrics
        for window in [5, 10, 20]:
            df[f'volume_ma_{window}'] = df['tick_volume'].rolling(window).mean()
            df[f'volume_std_{window}'] = df['tick_volume'].rolling(window).std()
            
        return df
        
    def _engineer_technical_features(self, df):
        """Create advanced technical features"""
        # Price patterns
        df['upper_shadow'] = df['high'] - df[['open', 'close']].max(axis=1)
        df['lower_shadow'] = df[['open', 'close']].min(axis=1) - df['low']
        
        # Momentum features
        df['price_acceleration'] = df['returns'].diff()
        df['volume_acceleration'] = df['volume_momentum'].diff()
        
        # Interaction features
        df['price_volume_correlation'] = (
            df['returns'].rolling(10).corr(df['volume_momentum'])
        )
        
        return df
        
    def _clean_data(self, df):
        """Clean the dataset"""
        # Remove duplicates
        df = df.drop_duplicates()
        
        # Handle missing values
        df = df.ffill().bfill()
        
        # Remove outliers
        df = self._remove_outliers(df)
        
        return df
        
    def _remove_outliers(self, df, n_sigmas=3):
        """Remove outliers using z-score method"""
        for column in df.select_dtypes(include=[np.number]).columns:
            mean = df[column].mean()
            std = df[column].std()
            
            df[column] = df[column].clip(
                lower=mean - n_sigmas * std,
                upper=mean + n_sigmas * std
            )
            
        return df
        
    def _normalize_technical_features(self, df):
        """Normalize technical features"""
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        df[numeric_columns] = self.technical_scaler.fit_transform(df[numeric_columns])
        return df
        
    def _normalize_fundamental_features(self, df):
        """Normalize fundamental features"""
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        df[numeric_columns] = self.fundamental_scaler.fit_transform(df[numeric_columns])
        return df
        
    def save_scalers(self):
        """Save scalers for later use"""
        try:
            # Save scalers using joblib
            import joblib
            
            joblib.dump(self.technical_scaler, 'data/scalers/technical_scaler.pkl')
            joblib.dump(self.fundamental_scaler, 'data/scalers/fundamental_scaler.pkl')
            
            self.logger.info("Scalers saved successfully")
            
        except Exception as e:
            self.logger.error(f"Error saving scalers: {str(e)}")

            