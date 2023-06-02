# analysis/technical.py
import numpy as np
import pandas as pd
from ta.trend import EMAIndicator, MACD
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands, AverageTrueRange

class TechnicalAnalyzer:
    def __init__(self, config):
        self.config = config
        
    async def analyze(self, data):
        """Run technical analysis"""
        df = self._calculate_all_indicators(data)
        signals = self._generate_signals(df)
        patterns = self._detect_patterns(df)
        
        return {
            'indicators': df,
            'signals': signals,
            'patterns': patterns
        }
        
    def _calculate_all_indicators(self, df):
        """Calculate all technical indicators"""
        # Trend indicators
        df['ema9'] = EMAIndicator(close=df['close'], window=9).ema_indicator()
        df['ema21'] = EMAIndicator(close=df['close'], window=21).ema_indicator()
        df['ema50'] = EMAIndicator(close=df['close'], window=50).ema_indicator()
        df['ema200'] = EMAIndicator(close=df['close'], window=200).ema_indicator()
        
        # MACD
        macd = MACD(close=df['close'])
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        
        # RSI
        df['rsi'] = RSIIndicator(close=df['close']).rsi()
        
        # Stochastic
        stoch = StochasticOscillator(high=df['high'], low=df['low'], close=df['close'])
        df['stoch_k'] = stoch.stoch()
        df['stoch_d'] = stoch.stoch_signal()
        
        # Volatility
        bb = BollingerBands(close=df['close'])
        df['bb_upper'] = bb.bollinger_hband()
        df['bb_lower'] = bb.bollinger_lband()
        df['bb_middle'] = bb.bollinger_mavg()
        
        # ATR
        df['atr'] = AverageTrueRange(high=df['high'], low=df['low'], close=df['close']).average_true_range()
        
        return df
        
    def _generate_signals(self, df):
        """Generate trading signals from indicators"""
        signals = {}
        
        # Trend signals
        signals['trend'] = np.where(df['ema9'] > df['ema21'], 1, 
                                  np.where(df['ema9'] < df['ema21'], -1, 0))
        
        # MACD signals
        signals['macd'] = np.where(df['macd'] > df['macd_signal'], 1, -1)
        
        # RSI signals
        signals['rsi'] = np.where(df['rsi'] < 30, 1, 
                                np.where(df['rsi'] > 70, -1, 0))
        
        # Stochastic signals
        signals['stoch'] = np.where((df['stoch_k'] < 20) & (df['stoch_k'] > df['stoch_d']), 1,
                                  np.where((df['stoch_k'] > 80) & (df['stoch_k'] < df['stoch_d']), -1, 0))
        
        return signals
        
    def _detect_patterns(self, df):
        """Detect chart patterns"""
        patterns = {}
        
        patterns['engulfing'] = self._detect_engulfing(df)
        patterns['doji'] = self._detect_doji(df)
        patterns['hammer'] = self._detect_hammer(df)
        
        return patterns

# analysis/fundamental.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
import yfinance as yf

class FundamentalAnalyzer:
    def __init__(self, config):
        self.config = config
        self.apis = {
            'economic': config.ECONOMIC_API_KEY,
            'news': config.NEWS_API_KEY,
            'crypto': config.CRYPTO_API_KEY
        }
        
    async def analyze(self, symbol):
        """Run fundamental analysis"""
        results = {}
        
        if self.config.MACRO_ANALYSIS:
            results['macro'] = await self._analyze_macro_economics()
            
        if self.config.COMPANY_ANALYSIS:
            results['company'] = await self._analyze_company(symbol)
            
        if self.config.SECTOR_ANALYSIS:
            results['sector'] = await self._analyze_sector(symbol)
            
        return results
        
    async def _analyze_macro_economics(self):
        """Analyze macroeconomic indicators"""
        try:
            # Get economic data
            inflation = await self._get_inflation_data()
            interest_rates = await self._get_interest_rates()
            gdp = await self._get_gdp_data()
            employment = await self._get_employment_data()
            
            # Calculate economic score
            economic_score = self._calculate_economic_score(
                inflation, interest_rates, gdp, employment
            )
            
            return {
                'economic_score': economic_score,
                'details': {
                    'inflation': inflation,
                    'interest_rates': interest_rates,
                    'gdp': gdp,
                    'employment': employment
                }
            }
            
        except Exception as e:
            logger.error(f"Error in macro analysis: {str(e)}")
            return None
            
    async def _analyze_company(self, symbol):
        """Analyze company fundamentals"""
        try:
            stock = yf.Ticker(symbol)
            
            # Get financial metrics
            metrics = {
                'pe_ratio': stock.info.get('forwardPE'),
                'pb_ratio': stock.info.get('priceToBook'),
                'debt_to_equity': stock.info.get('debtToEquity'),
                'profit_margin': stock.info.get('profitMargins'),
                'revenue_growth': stock.info.get('revenueGrowth'),
                'roa': stock.info.get('returnOnAssets'),
                'roe': stock.info.get('returnOnEquity')
            }
            
            # Calculate company score
            company_score = self._calculate_company_score(metrics)
            
            return {
                'company_score': company_score,
                'metrics': metrics
            }
            
        except Exception as e:
            logger.error(f"Error in company analysis: {str(e)}")
            return None

# analysis/sentiment.py
from textblob import TextBlob
import numpy as np
import requests
from datetime import datetime, timedelta

class SentimentAnalyzer:
    def __init__(self, config):
        self.config = config
        self.news_api_key = config.NEWS_API_KEY
        
    async def analyze(self, symbol):
        """Run sentiment analysis"""
        try:
            # Get news data
            news = await self._get_news(symbol)
            social_media = await self._get_social_media_data(symbol)
            market_sentiment = await self._get_market_sentiment(symbol)
            
            # Analyze sentiment
            news_sentiment = await self._analyze_news_sentiment(news)
            social_sentiment = await self._analyze_social_sentiment(social_media)
            
            # Combine all sentiments
            combined_sentiment = self._combine_sentiments(
                news_sentiment,
                social_sentiment,
                market_sentiment
            )
            
            return {
                'overall_sentiment': combined_sentiment,
                'details': {
                    'news': news_sentiment,
                    'social': social_sentiment,
                    'market': market_sentiment
                }
            }
            
        except Exception as e:
            logger.error(f"Error in sentiment analysis: {str(e)}")
            return None
            
    async def _get_news(self, symbol, days=7):
        """Get news articles for the symbol"""
        endpoint = "https://newsapi.org/v2/everything"
        
        params = {
            'q': symbol,
            'from': (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d'),
            'sortBy': 'relevancy',
            'apiKey': self.news_api_key
        }
        
        response = requests.get(endpoint, params=params)
        return response.json().get('articles', [])
        
    async def _analyze_news_sentiment(self, articles):
        """Analyze sentiment of news articles"""
        sentiments = []
        
        for article in articles:
            text = f"{article['title']} {article['description']}"
            blob = TextBlob(text)
            sentiments.append(blob.sentiment.polarity)
            
        return np.mean(sentiments) if sentiments else 0