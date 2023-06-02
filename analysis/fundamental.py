# analysis/fundamental.py
import pandas as pd
import numpy as np
import yfinance as yf
import requests
from datetime import datetime, timedelta
import logging
from bs4 import BeautifulSoup

class FundamentalAnalyzer:
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger('analysis.fundamental')
        self.apis = {
            'economic': config.ECONOMIC_API_KEY,
            'news': config.NEWS_API_KEY,
            'crypto': config.CRYPTO_API_KEY
        }
        
    async def analyze(self, symbol, asset_type='forex'):
        """Run comprehensive fundamental analysis"""
        try:
            results = {}
            
            # Macroeconomic analysis
            if self.config.MACRO_ANALYSIS:
                results['macro'] = await self.analyze_macro_economics()
                
            # Asset-specific analysis
            if asset_type == 'forex':
                results['specific'] = await self.analyze_forex_fundamentals(symbol)
            elif asset_type == 'crypto':
                results['specific'] = await self.analyze_crypto_fundamentals(symbol)
            elif asset_type == 'stock':
                results['specific'] = await self.analyze_stock_fundamentals(symbol)
                
            # Market sentiment
            if self.config.USE_NEWS_ANALYSIS:
                results['sentiment'] = await self.analyze_market_sentiment(symbol)
                
            return results
            
        except Exception as e:
            self.logger.error(f"Error in fundamental analysis: {str(e)}")
            return None

    async def analyze_macro_economics(self):
        """Analyze macroeconomic indicators"""
        try:
            indicators = {}
            
            # Interest rates
            rates = await self._get_interest_rates()
            indicators['interest_rates'] = rates
            
            # Inflation
            inflation = await self._get_inflation_data()
            indicators['inflation'] = inflation
            
            # GDP growth
            gdp = await self._get_gdp_data()
            indicators['gdp'] = gdp
            
            # Employment
            employment = await self._get_employment_data()
            indicators['employment'] = employment
            
            # Calculate economic score
            score = self._calculate_economic_score(indicators)
            indicators['economic_score'] = score
            
            return indicators
            
        except Exception as e:
            self.logger.error(f"Error in macro analysis: {str(e)}")
            return None

    async def analyze_forex_fundamentals(self, pair):
        """Analyze forex pair fundamentals"""
        try:
            # Split pair into base and quote currencies
            base_currency = pair[:3]
            quote_currency = pair[3:]
            
            # Analyze each currency
            base_analysis = await self._analyze_currency(base_currency)
            quote_analysis = await self._analyze_currency(quote_currency)
            
            # Calculate relative strength
            strength_score = self._calculate_relative_strength(
                base_analysis,
                quote_analysis
            )
            
            return {
                'strength_score': strength_score,
                'base_currency': base_analysis,
                'quote_currency': quote_analysis
            }
            
        except Exception as e:
            self.logger.error(f"Error in forex analysis: {str(e)}")
            return None

    async def analyze_stock_fundamentals(self, symbol):
        """Analyze stock fundamentals"""
        try:
            stock = yf.Ticker(symbol)
            
            # Get financial ratios
            ratios = {
                'pe_ratio': stock.info.get('forwardPE'),
                'pb_ratio': stock.info.get('priceToBook'),
                'debt_equity': stock.info.get('debtToEquity'),
                'profit_margin': stock.info.get('profitMargins'),
                'revenue_growth': stock.info.get('revenueGrowth')
            }
            
            # Get financial statements
            income_stmt = stock.financials
            balance_sheet = stock.balance_sheet
            cash_flow = stock.cashflow
            
            # Calculate financial health score
            health_score = self._calculate_financial_health(
                ratios,
                income_stmt,
                balance_sheet,
                cash_flow
            )
            
            return {
                'health_score': health_score,
                'ratios': ratios,
                'financials': {
                    'income_stmt': income_stmt,
                    'balance_sheet': balance_sheet,
                    'cash_flow': cash_flow
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error in stock analysis: {str(e)}")
            return None

    def _calculate_economic_score(self, indicators):
        """Calculate overall economic score"""
        weights = {
            'interest_rates': 0.3,
            'inflation': 0.3,
            'gdp': 0.2,
            'employment': 0.2
        }
        
        score = 0
        for indicator, weight in weights.items():
            if indicator in indicators:
                score += indicators[indicator] * weight
                
        return score

    def _calculate_financial_health(self, ratios, income_stmt, balance_sheet, cash_flow):
        """Calculate financial health score for stocks"""
        try:
            score = 0
            
            # Profitability
            if ratios['profit_margin']:
                score += min(ratios['profit_margin'] * 100, 30) / 30 * 0.3
                
            # Growth
            if ratios['revenue_growth']:
                score += min(ratios['revenue_growth'] * 100, 50) / 50 * 0.3
                
            # Financial stability
            if ratios['debt_equity']:
                debt_score = max(0, 1 - ratios['debt_equity'] / 200) * 0.4
                score += debt_score
                
            return min(score, 1.0)
            
        except Exception as e:
            self.logger.error(f"Error calculating financial health: {str(e)}")
            return 0.5  # Return neutral score on error