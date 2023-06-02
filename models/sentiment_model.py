# models/sentiment_model.py
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import numpy as np
from textblob import TextBlob
import logging
from sklearn.ensemble import RandomForestClassifier
import joblib

class SentimentModel:
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger('model.sentiment')
        self.setup_model()
        
    def setup_model(self):
        """Initialize sentiment analysis models"""
        try:
            if self.config.DEEP_LEARNING:
                # Load FinBERT model for financial sentiment analysis
                self.model = AutoModelForSequenceClassification.from_pretrained('ProsusAI/finbert')
                self.tokenizer = AutoTokenizer.from_pretrained('ProsusAI/finbert')
                
                # Default device configuration
                self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                self.model.to(self.device)
            else:
                # Fallback to traditional ML model
                self.model = RandomForestClassifier(
                    n_estimators=100,
                    max_depth=10,
                    random_state=42
                )
            
            self.logger.info(f"Sentiment model initialized: {'Deep Learning' if self.config.DEEP_LEARNING else 'Traditional ML'}")
        except Exception as e:
            self.logger.error(f"Error initializing sentiment model: {str(e)}")
            raise

    async def analyze_text(self, texts, batch_size=8):
        """Analyze sentiment of multiple texts"""
        try:
            if self.config.DEEP_LEARNING:
                results = []
                for i in range(0, len(texts), batch_size):
                    batch_texts = texts[i:i + batch_size]
                    batch_results = await self._analyze_batch_deep(batch_texts)
                    results.extend(batch_results)
                return results
            else:
                return self._analyze_traditional(texts)
                
        except Exception as e:
            self.logger.error(f"Error in sentiment analysis: {str(e)}")
            return None

    async def _analyze_batch_deep(self, texts):
        """Analyze a batch of texts using deep learning model"""
        try:
            inputs = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                predictions = torch.softmax(outputs.logits, dim=-1)
                
            results = []
            for pred in predictions:
                sentiment_score = {
                    'positive': pred[0].item(),
                    'neutral': pred[1].item(),
                    'negative': pred[2].item()
                }
                results.append(sentiment_score)
                
            return results
            
        except Exception as e:
            self.logger.error(f"Error in batch analysis: {str(e)}")
            return None

    def _analyze_traditional(self, texts):
        """Analyze texts using traditional ML approach"""
        try:
            results = []
            for text in texts:
                blob = TextBlob(text)
                
                # Get polarity and subjectivity
                polarity = blob.sentiment.polarity
                subjectivity = blob.sentiment.subjectivity
                
                # Convert to sentiment scores
                if polarity > 0.1:
                    sentiment = {'positive': 0.7 + polarity * 0.3, 'neutral': 0.3, 'negative': 0}
                elif polarity < -0.1:
                    sentiment = {'positive': 0, 'neutral': 0.3, 'negative': 0.7 + abs(polarity) * 0.3}
                else:
                    sentiment = {'positive': 0.2, 'neutral': 0.6, 'negative': 0.2}
                    
                results.append(sentiment)
                
            return results
            
        except Exception as e:
            self.logger.error(f"Error in traditional analysis: {str(e)}")
            return None

    async def train(self, labeled_data):
        """Train the sentiment model"""
        try:
            if not self.config.DEEP_LEARNING and labeled_data:
                X = [text for text, _ in labeled_data]
                y = [label for _, label in labeled_data]
                
                # Extract features using TextBlob
                features = self._extract_features(X)
                
                # Train the model
                self.model.fit(features, y)
                self.logger.info("Traditional sentiment model trained successfully")
                
        except Exception as e:
            self.logger.error(f"Error training sentiment model: {str(e)}")
            raise

    def _extract_features(self, texts):
        """Extract features from texts for traditional ML"""
        features = []
        for text in texts:
            blob = TextBlob(text)
            features.append([
                blob.sentiment.polarity,
                blob.sentiment.subjectivity,
                len(blob.words),
                len([word for word in blob.words if word.lower() in self.positive_words]),
                len([word for word in blob.words if word.lower() in self.negative_words])
            ])
        return np.array(features)

    def save_model(self):
        """Save the sentiment model"""
        try:
            if not self.config.DEEP_LEARNING:
                model_path = 'data/models/sentiment_model.joblib'
                joblib.dump(self.model, model_path)
                self.logger.info(f"Sentiment model saved to {model_path}")
        except Exception as e:
            self.logger.error(f"Error saving sentiment model: {str(e)}")
            raise