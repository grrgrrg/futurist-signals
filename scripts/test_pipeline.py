#!/usr/bin/env python3
"""
Test script for the Futurist Signal Detection System
Run this to test the basic functionality of each component
"""

import asyncio
import sys
import os
import logging

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.collectors.crawler_factory import CrawlerFactory
from src.processors.signal_extractor import SignalExtractor, process_signals
from src.analyzers.trend_synthesizer import TrendSynthesizer, analyze_trends
from src.analyzers.anomaly_detector import AnomalyDetector, detect_anomalies

# Setup basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_crawler():
    """Test the web crawler functionality"""
    logger.info("Testing web crawler...")
    
    # Create a simple test crawler
    test_source = {
        'name': 'Test Source',
        'url': 'https://futurity.org',
        'type': 'test',
        'selectors': {
            'articles': 'article',
            'title': 'h2',
            'link': 'a',
            'summary': 'p'
        }
    }
    
    crawler = CrawlerFactory.create_crawler(test_source)
    signals = await crawler.crawl()
    
    logger.info(f"Crawler test: collected {len(signals)} raw signals")
    
    if signals:
        logger.info(f"Sample signal: {signals[0]}")
    
    return signals

def test_signal_extractor():
    """Test signal extraction"""
    logger.info("Testing signal extractor...")
    
    # Test with sample texts
    test_texts = [
        "Scientists achieve breakthrough in quantum computing, potentially revolutionizing cybersecurity by 2030.",
        "New AI system shows 300% improvement in energy efficiency, could transform data centers.",
        "Researchers develop room-temperature superconductor for the first time in history.",
        "Study shows 45% increase in renewable energy adoption, contradicting previous predictions.",
        "Company announces impossible perpetual motion machine, defying laws of physics."
    ]
    
    extractor = SignalExtractor()
    all_signals = []
    
    for text in test_texts:
        signals = extractor.extract_signals(text)
        all_signals.extend(signals)
        logger.info(f"Text: '{text[:50]}...' -> {len(signals)} signals")
    
    logger.info(f"Signal extraction test: {len(all_signals)} total signals extracted")
    return all_signals

def test_trend_synthesizer(signals):
    """Test trend synthesis"""
    logger.info("Testing trend synthesizer...")
    
    synthesizer = TrendSynthesizer()
    trends = synthesizer.synthesize_trends(signals)
    
    logger.info(f"Trend synthesis test: identified {len(trends)} trends")
    
    for trend in trends:
        logger.info(f"Trend: {trend.get('name')} - {trend.get('description')}")
    
    return trends

def test_anomaly_detector(signals):
    """Test anomaly detection"""
    logger.info("Testing anomaly detector...")
    
    detector = AnomalyDetector()
    anomalies = detector.detect_anomalies(signals)
    
    logger.info(f"Anomaly detection test: found {len(anomalies)} anomalies")
    
    for anomaly in anomalies:
        logger.info(f"Anomaly: {anomaly.get('content')[:100]}... (score: {anomaly.get('overall_anomaly_score', 0):.2f})")
    
    return anomalies

async def test_full_pipeline():
    """Test the complete pipeline"""
    logger.info("Starting full pipeline test...")
    
    try:
        # 1. Test crawler (optional - requires internet)
        try:
            raw_signals = await test_crawler()
        except Exception as e:
            logger.warning(f"Crawler test failed (expected if no internet): {e}")
            raw_signals = []
        
        # 2. Test signal extraction with sample data
        sample_signals = test_signal_extractor()
        
        # Convert to format expected by downstream components
        processed_signals = []
        for signal in sample_signals:
            processed_signals.append({
                'content': signal['content'],
                'category': signal['category'],
                'confidence': signal['confidence'],
                'entities': signal['entities'],
                'temporal_refs': signal['temporal_refs'],
                'extracted_at': signal['extracted_at'],
                'source_name': 'Test Source',
                'source_type': 'test'
            })
        
        # 3. Test trend synthesis
        trends = test_trend_synthesizer(processed_signals)
        
        # 4. Test anomaly detection
        anomalies = test_anomaly_detector(processed_signals)
        
        # Summary
        logger.info("="*50)
        logger.info("PIPELINE TEST SUMMARY")
        logger.info("="*50)
        logger.info(f"Raw signals collected: {len(raw_signals)}")
        logger.info(f"Processed signals: {len(processed_signals)}")
        logger.info(f"Trends identified: {len(trends)}")
        logger.info(f"Anomalies detected: {len(anomalies)}")
        logger.info("="*50)
        
        return {
            'raw_signals': len(raw_signals),
            'processed_signals': len(processed_signals),
            'trends': len(trends),
            'anomalies': len(anomalies)
        }
        
    except Exception as e:
        logger.error(f"Pipeline test failed: {str(e)}")
        raise

def main():
    """Main test function"""
    print("Futurist Signal Detection System - Component Test")
    print("=" * 60)
    
    try:
        result = asyncio.run(test_full_pipeline())
        print("\n✅ All tests passed!")
        print(f"Results: {result}")
        return 0
    except Exception as e:
        print(f"\n❌ Tests failed: {str(e)}")
        return 1

if __name__ == "__main__":
    exit(main())