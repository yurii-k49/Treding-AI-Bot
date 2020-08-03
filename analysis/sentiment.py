# analysis/sentiment.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import requests
from bs4 import BeautifulSoup
from textblob import TextBlob

class SentimentAnalyzer:
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger('analysis.sentiment')
        self.news_api_key = config.NEWS_API_KEY
        
    async def analyze(self, symbol):
        """Run sentiment analysis"""
        try:
            # Get various types of sentiment data
            news_data = await self._get_news_sentiment(symbol)
            social_data = await self._get_social_sentiment(symbol)
            market_data = await self._get_market_sentiment(symbol)
            
            # Combine all sentiment data
            combined_sentiment = self._combine_sentiment_scores(
                news_data,
                social_data,
                market_data
            )
            
            return {
                'overall_sentiment': combined_sentiment,
                'components': {
                    'news': news_data,
                    'social': social_data,
                    'market': market_data
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error in sentiment analysis: {str(e)}")
            return None

    async def _get_news_sentiment(self, symbol):
        """Analyze news sentiment"""
        try:
            # Get news articles
            articles = await self._fetch_news(symbol)
            
            if not articles:
                return {'score': 0, 'confidence': 0}
                
            # Analyze each article
            sentiments = []
            for article in articles:
                title = article.get('title', '')
                description = article.get('description', '')
                content = f"{title} {description}"
                
                blob = TextBlob(content)
                sentiments.append({
                    'polarity': blob.sentiment.polarity,
                    'subjectivity': blob.sentiment.subjectivity
                })
                
            # Calculate average sentiment
            avg_polarity = np.mean([s['polarity'] for s in sentiments])
            avg_subjectivity = np.mean([s['subjectivity'] for s in sentiments])
            
            return {
                'score': avg_polarity,
                'confidence': 1 - avg_subjectivity,
                'articles_analyzed': len(articles)
            }
            
        except Exception as e:
            self.logger.error(f"Error in news sentiment analysis: {str(e)}")
            return {'score': 0, 'confidence': 0}

    async def _get_social_sentiment(self, symbol):
        """Analyze social media sentiment"""
        try:
            # Get social media posts
            posts = await self._fetch_social_posts(symbol)
            
            if not posts:
                return {'score': 0, 'confidence': 0}
                
            # Analyze sentiment for each post
            sentiments = []
            for post in posts:
                blob = TextBlob(post['text'])
                sentiments.append({
                    'polarity': blob.sentiment.polarity,
                    'subjectivity': blob.sentiment.subjectivity,
                    'reach': post.get('reach', 1)
                })
                
            # Calculate weighted sentiment
            total_reach = sum(s['reach'] for s in sentiments)
            weighted_polarity = sum(s['polarity'] * s['reach'] for s in sentiments) / total_reach
            
            return {
               'score': weighted_polarity,
                'confidence': 1 - np.mean([s['subjectivity'] for s in sentiments]),
                'posts_analyzed': len(posts)
            }
            
        except Exception as e:
            self.logger.error(f"Error in social sentiment analysis: {str(e)}")
            return {'score': 0, 'confidence': 0}

    async def _get_market_sentiment(self, symbol):
        """Analyze market sentiment indicators"""
        try:
            indicators = await self._fetch_market_indicators(symbol)
            
            if not indicators:
                return {'score': 0, 'confidence': 0}
                
            # Calculate market sentiment score
            vix = indicators.get('vix', 20)  # Default VIX value
            put_call_ratio = indicators.get('put_call_ratio', 1)
            
            # VIX scoring (inverse relationship)
            vix_score = max(0, min(1, (50 - vix) / 30))
            
            # Put-Call ratio scoring (neutral at 1.0)
            pc_score = max(0, min(1, 1 - abs(put_call_ratio - 1)))
            
            return {
                'score': (vix_score + pc_score) / 2,
                'confidence': 0.8,  # Market indicators are generally reliable
                'indicators': indicators
            }
            
        except Exception as e:
            self.logger.error(f"Error in market sentiment analysis: {str(e)}")
            return {'score': 0, 'confidence': 0}

    def _combine_sentiment_scores(self, news, social, market):
        """Combine different sentiment scores"""
        weights = {
            'news': 0.4,
            'social': 0.3,
            'market': 0.3
        }
        
        try:
            # Calculate weighted average
            total_score = (
                news['score'] * weights['news'] * news['confidence'] +
                social['score'] * weights['social'] * social['confidence'] +
                market['score'] * weights['market'] * market['confidence']
            )
            
            total_confidence = (
                weights['news'] * news['confidence'] +
                weights['social'] * social['confidence'] +
                weights['market'] * market['confidence']
            )
            
            if total_confidence > 0:
                final_score = total_score / total_confidence
            else:
                final_score = 0
                
            return {
                'score': final_score,
                'confidence': total_confidence,
                'interpretation': self._interpret_sentiment(final_score)
            }
            
        except Exception as e:
            self.logger.error(f"Error combining sentiment scores: {str(e)}")
            return {'score': 0, 'confidence': 0, 'interpretation': 'neutral'}

    def _interpret_sentiment(self, score):
        """Interpret sentiment score"""
        if score > 0.2:
            return 'bullish'
        elif score < -0.2:
            return 'bearish'
        else:
            return 'neutral'