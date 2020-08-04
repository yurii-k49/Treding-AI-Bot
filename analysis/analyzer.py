# analysis/analyzer.py
import logging
import asyncio
from analysis.technical import TechnicalAnalyzer
from analysis.fundamental import FundamentalAnalyzer
from analysis.sentiment import SentimentAnalyzer

class MainAnalyzer:
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger('analysis.main')
        self.setup_analyzers()

    def setup_analyzers(self):
        """Initialize all analyzers"""
        try:
            self.technical = TechnicalAnalyzer(self.config)
            self.fundamental = FundamentalAnalyzer(self.config)
            self.sentiment = SentimentAnalyzer(self.config)
            self.logger.info("All analyzers initialized successfully")
        except Exception as e:
            self.logger.error(f"Error initializing analyzers: {str(e)}")
            raise

    async def run_analysis(self, market_data):
        """Run comprehensive market analysis"""
        try:
            # Run all analyses in parallel
            technical_results, fundamental_results, sentiment_results = await asyncio.gather(
                self.technical.analyze(market_data),
                self.fundamental.analyze(market_data.get('symbol')),
                self.sentiment.analyze(market_data.get('symbol'))
            )
            
            # Combine results
            combined_analysis = self.combine_analysis({
                'technical': technical_results,
                'fundamental': fundamental_results,
                'sentiment': sentiment_results
            })
            
            # Log analysis summary
            self.logger.info(f"Analysis completed: {self._get_analysis_summary(combined_analysis)}")
            
            return combined_analysis
            
        except Exception as e:
            self.logger.error(f"Error in market analysis: {str(e)}")
            return None

    def combine_analysis(self, results):
        """Combine different analysis results"""
        try:
            # Get individual signals
            technical_signal = self._extract_signal(results['technical'])
            fundamental_signal = self._extract_signal(results['fundamental'])
            sentiment_signal = self._extract_signal(results['sentiment'])
            
            # Calculate weighted signal
            weights = self.config.SIGNAL_WEIGHTS
            combined_signal = (
                technical_signal * weights['technical'] +
                fundamental_signal * weights['fundamental'] +
                sentiment_signal * weights['sentiment']
            )
            
            # Calculate signal strength and confidence
            signal_strength = abs(combined_signal)
            confidence = self._calculate_confidence(results)
            
            return {
                'signal': combined_signal,
                'strength': signal_strength,
                'confidence': confidence,
                'components': {
                    'technical': results['technical'],
                    'fundamental': results['fundamental'],
                    'sentiment': results['sentiment']
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error combining analysis: {str(e)}")
            return None

    def _extract_signal(self, analysis):
        """Extract signal from analysis result"""
        if not analysis:
            return 0
            
        if 'signal' in analysis:
            return analysis['signal']
        elif 'score' in analysis:
            return analysis['score']
            
        return 0

    def _calculate_confidence(self, results):
        """Calculate overall confidence level"""
        confidences = []
        
        if results['technical'] and 'confidence' in results['technical']:
            confidences.append(results['technical']['confidence'])
            
        if results['fundamental'] and 'confidence' in results['fundamental']:
            confidences.append(results['fundamental']['confidence'])
            
        if results['sentiment'] and 'confidence' in results['sentiment']:
            confidences.append(results['sentiment']['confidence'])
            
        return sum(confidences) / len(confidences) if confidences else 0

    def _get_analysis_summary(self, analysis):
        """Generate analysis summary"""
        return {
            'signal_direction': 'buy' if analysis['signal'] > 0 else 'sell',
            'signal_strength': analysis['strength'],
            'confidence': analysis['confidence']
        }