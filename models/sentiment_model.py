# models/sentiment_model.py
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from textblob import TextBlob
import logging
import joblib
import os

class SentimentModel:
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger('models.sentiment')
        self.setup_model()
        
    def setup_model(self):
        """Initialize sentiment analysis model"""
        try:
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=10,
                random_state=42
            )
            self.logger.info("Sentiment model initialized: Traditional ML")
        except Exception as e:
            self.logger.error(f"Error initializing sentiment model: {str(e)}")
            raise
            
    async def train(self, data):
        """Train the sentiment model"""
        try:
            if data is None:
                raise Exception("No training data provided")
                
            # Prepare training data
            X, y = self._prepare_training_data(data)
            
            if X is None or y is None:
                raise Exception("Failed to prepare training data")
                
            # Train model
            self.model.fit(X, y)
            self.logger.info("Sentiment model trained successfully")
            
        except Exception as e:
            self.logger.error(f"Error training sentiment model: {str(e)}")
            raise
            
    def _prepare_training_data(self, data):
        """Prepare data for training"""
        try:
            if data is None:
                return None, None
                
            # Process news and social sentiment
            news_features = self._process_news_data(data.get('news'))
            social_features = self._process_social_data(data.get('social'))
            
            if news_features is None or social_features is None:
                return None, None
                
            # Combine features
            X = pd.concat([news_features, social_features], axis=1)
            
            # Create target (using simple strategy - if next day's sentiment improves)
            y = (X['news_sentiment_mean'].diff().shift(-1) > 0).astype(int)
            
            # Remove last row (no future value)
            X = X.iloc[:-1]
            y = y[:-1]
            
            # Remove any remaining NaN
            mask = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
            X = X[mask]
            y = y[mask]
            
            return X.values, y.values
            
        except Exception as e:
            self.logger.error(f"Error preparing training data: {str(e)}")
            return None, None
            
    def _process_news_data(self, news_data):
        """Process news sentiment data"""
        try:
            if news_data is None:
                return None
                
            df = pd.DataFrame(news_data)
            df.set_index('date', inplace=True)
            
            # Calculate daily sentiment metrics
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
            
        except Exception as e:
            self.logger.error(f"Error processing news data: {str(e)}")
            return None
            
    def _process_social_data(self, social_data):
        """Process social media sentiment data"""
        try:
            if social_data is None:
                return None
                
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
            
        except Exception as e:
            self.logger.error(f"Error processing social data: {str(e)}")
            return None
            
    async def predict(self, data):
        """Make sentiment predictions"""
        try:
            if data is None:
                return None
                
            # Prepare features
            X = self._prepare_prediction_data(data)
            
            if X is None:
                return None
                
            # Make prediction
            predictions = self.model.predict_proba(X)[:, 1]
            
            return {
                'signal': predictions[-1],  # Latest prediction
                'confidence': np.abs(predictions[-1] - 0.5) * 2  # Convert to 0-1 scale
            }
            
        except Exception as e:
            self.logger.error(f"Error making prediction: {str(e)}")
            return None
            
    def _prepare_prediction_data(self, data):
        """Prepare data for prediction"""
        try:
            return self._prepare_training_data(data)[0]
        except Exception as e:
            self.logger.error(f"Error preparing prediction data: {str(e)}")
            return None
            
    def save_model(self):
        """Save model to disk"""
        try:
            # Create directory if it doesn't exist
            os.makedirs('data/models', exist_ok=True)
            
            # Save model
            joblib.dump(self.model, 'data/models/sentiment_model.joblib')
            self.logger.info("Sentiment model saved successfully")
            
        except Exception as e:
            self.logger.error(f"Error saving model: {str(e)}")