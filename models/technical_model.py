# models/technical_model.py
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

class TechnicalModel:
    def __init__(self, config):
        self.config = config
        self.setup_model()
        
    def setup_model(self):
        """Initialize the model based on configuration"""
        if self.config.DEEP_LEARNING:
            self.model = self._create_deep_model()
        else:
            self.model = self._create_rf_model()
            
    def _create_deep_model(self):
        """Create deep learning model"""
        model = Sequential([
            LSTM(100, input_shape=(100, 20), return_sequences=True),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(25, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model
        
    def _create_rf_model(self):
        """Create random forest model"""
        return RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        
    async def update(self, data=None):
        """Update model with new data"""
        if data is not None:
            X, y = self._prepare_data(data)
            if self.config.DEEP_LEARNING:
                self.model.fit(X, y, epochs=10, batch_size=32, validation_split=0.2)
            else:
                self.model.fit(X, y)
                
    def predict(self, features):
        """Make predictions"""
        if self.config.DEEP_LEARNING:
            return self.model.predict(features)
        return self.model.predict_proba(features)[:, 1]

# models/fundamental_model.py
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

class FundamentalModel:
    def __init__(self, config):
        self.config = config
        self.setup_models()
        
    def setup_models(self):
        """Initialize models for different aspects"""
        self.macro_model = self._create_macro_model()
        self.company_model = self._create_company_model()
        self.sector_model = self._create_sector_model()
        
    def _create_macro_model(self):
        """Model for macroeconomic analysis"""
        if self.config.DEEP_LEARNING:
            model = Sequential([
                Dense(64, activation='relu', input_shape=(20,)),
                Dropout(0.2),
                Dense(32, activation='relu'),
                Dense(1, activation='sigmoid')
            ])
            model.compile(optimizer='adam', loss='mse')
            return model
        return GradientBoostingRegressor()
        
    async def update(self, data=None):
        """Update all fundamental models"""
        if data is not None:
            macro_data, company_data, sector_data = self._prepare_fundamental_data(data)
            await asyncio.gather(
                self._update_macro_model(macro_data),
                self._update_company_model(company_data),
                self._update_sector_model(sector_data)
            )

# models/sentiment_model.py
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding
from transformers import AutoModelForSequenceClassification, AutoTokenizer

class SentimentModel:
    def __init__(self, config):
        self.config = config
        self.setup_model()
        
    def setup_model(self):
        """Initialize sentiment analysis model"""
        if self.config.DEEP_LEARNING:
            self.model = self._create_deep_model()
            self.tokenizer = AutoTokenizer.from_pretrained('finbert-sentiment')
        else:
            self.model = self._create_basic_model()
            
    def _create_deep_model(self):
        """Create deep learning model for sentiment analysis"""
        return AutoModelForSequenceClassification.from_pretrained('finbert-sentiment')
        
    async def analyze_text(self, texts):
        """Analyze sentiment of multiple texts"""
        if self.config.DEEP_LEARNING:
            inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
            outputs = self.model(**inputs)
            return outputs.logits.softmax(dim=-1)
        return self._basic_sentiment_analysis(texts)