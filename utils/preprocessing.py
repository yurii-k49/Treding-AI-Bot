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
        self.sentiment_scaler = MinMaxScaler()

    def prepare_technical_features(self, df):
        """Enhanced technical features"""
        try:
            features = pd.DataFrame(index=df.index)
            
            # Narx o'zgarishlari
            features['returns'] = df['close'].pct_change()
            features['log_returns'] = np.log(df['close'] / df['close'].shift(1))
            
            # Moving averages va ularning signallari
            for window in [5, 10, 20, 50, 100]:
                features[f'ma_{window}'] = df['close'].rolling(window=window).mean()
                features[f'ma_signal_{window}'] = (df['close'] > features[f'ma_{window}']).astype(int)
            
            # Volatility indikatorlari
            for window in [5, 10, 20]:
                features[f'volatility_{window}'] = df['close'].rolling(window).std()
                
            # RSI
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            features['rsi'] = 100 - (100 / (1 + rs))
            
            # MACD
            exp1 = df['close'].ewm(span=12, adjust=False).mean()
            exp2 = df['close'].ewm(span=26, adjust=False).mean()
            features['macd'] = exp1 - exp2
            features['macd_signal'] = features['macd'].ewm(span=9, adjust=False).mean()
            
            # NaN qiymatlarni to'ldirish
            features = features.bfill().fillna(0)
            
            return features
        
        except Exception as e:
            self.logger.error(f"Error preparing technical features: {str(e)}")
            return pd.DataFrame()

    def prepare_fundamental_features(self, data):
        """Prepare fundamental analysis features"""
        try:
            if data is None:
                return None
                
            features = pd.DataFrame()
            
            # Process economic data
            if 'economic' in data:
                features = self._process_economic_data(data['economic'], features)
            
            # Process financial data
            if 'financial' in data:
                features = self._process_financial_data(data['financial'], features)
            
            # Clean and normalize
            features = self._clean_data(features)
            features = self._normalize_features(features, self.fundamental_scaler)
            
            return features
            
        except Exception as e:
            self.logger.error(f"Error in fundamental preprocessing: {str(e)}")
            return None

    def prepare_sentiment_features(self, data):
        """Prepare sentiment analysis features"""
        try:
            if data is None:
                return None
                
            features = pd.DataFrame()
            
            # Process news sentiment
            if 'news' in data:
                news_features = self._process_news_data(data['news'])
                features = pd.concat([features, news_features], axis=1)
            
            # Process social sentiment
            if 'social' in data:
                social_features = self._process_social_data(data['social'])
                features = pd.concat([features, social_features], axis=1)
            
            # Clean and normalize
            features = self._clean_data(features)
            features = self._normalize_features(features, self.sentiment_scaler)
            
            return features
            
        except Exception as e:
            self.logger.error(f"Error in sentiment preprocessing: {str(e)}")
            return None

    def _process_economic_data(self, data, features):
        """Process economic indicators"""
        for indicator, values in data.items():
            df = pd.DataFrame(values)
            df.set_index('date', inplace=True)
            features[f'economic_{indicator}'] = df['value']
        return features

    def _process_financial_data(self, data, features):
        """Process financial data"""
        for metric, values in data.items():
            df = pd.DataFrame(values)
            df.set_index('date', inplace=True)
            features[f'financial_{metric}'] = df['value']
        return features

    def _process_news_data(self, news_data):
        """Process news sentiment data"""
        df = pd.DataFrame(news_data)
        df.set_index('date', inplace=True)
        
        # Calculate daily sentiment
        daily_sentiment = df.groupby(df.index)['sentiment'].agg([
            'mean',
            'std',
            'count'
        ]).fillna(0)
        
        daily_sentiment.columns = [
            'news_sentiment_mean',
            'news_sentiment_std',
            'news_count'
        ]
        
        return daily_sentiment

    def _process_social_data(self, social_data):
        """Process social media sentiment data"""
        df = pd.DataFrame(social_data)
        df.set_index('date', inplace=True)
        
        # Calculate daily metrics
        daily_metrics = df.groupby(df.index).agg({
            'sentiment': ['mean', 'std'],
            'volume': 'sum'
        }).fillna(0)
        
        daily_metrics.columns = [
            'social_sentiment_mean',
            'social_sentiment_std',
            'social_volume'
        ]
        
        return daily_metrics

    def _clean_data(self, df):
        """Clean dataset"""
        if df is None:
            return None
            
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

    def _normalize_features(self, df, scaler):
        """Normalize numerical features"""
        if df is None:
            return None
            
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        if len(numeric_columns) > 0:
            df[numeric_columns] = scaler.fit_transform(df[numeric_columns])
        
        return df
    
    def prepare_market_features(self, market_df):
        try:
            if market_df is None or market_df.empty:
                return pd.DataFrame()
                
            features = pd.DataFrame()
            
            # Add market features
            if 'volatility' in market_df.columns:
                features['market_volatility'] = market_df['volatility']
            if 'volume_ma' in market_df.columns:
                features['market_volume'] = market_df['volume_ma']
            if 'spread_avg' in market_df.columns:
                features['market_spread'] = market_df['spread_avg']
                
            # Add calculated features
            if not features.empty:
                features['volatility_ma'] = features['market_volatility'].rolling(window=20).mean()
                features['volume_change'] = features['market_volume'].pct_change()
                
            # NaN qiymatlarni to'ldirish
            features = features.bfill().fillna(0)  # fillna(method='bfill') o'rniga
            
            return features
            
        except Exception as e:
            self.logger.error(f"Error preparing market features: {str(e)}")
            return pd.DataFrame()