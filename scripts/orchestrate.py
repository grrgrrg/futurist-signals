import asyncio
import schedule
import time
import logging
import yaml
from datetime import datetime
from typing import List, Dict
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.collectors.crawler_factory import crawl_all_sources
from src.processors.signal_extractor import process_signals
from src.analyzers.trend_synthesizer import analyze_trends
from src.analyzers.anomaly_detector import detect_anomalies
from src.storage.database import db
from src.storage.models import Signal, Trend, Source, AnomalyScore, TrendSignal, DailyReport
from config.settings import settings

# Setup logging
logging.config.dictConfig(yaml.safe_load(open('config/logging.yaml')))
logger = logging.getLogger(__name__)

class SignalPipeline:
    def __init__(self):
        self.stats = {
            'runs': 0,
            'total_signals': 0,
            'total_trends': 0,
            'total_anomalies': 0,
            'last_run': None,
            'errors': 0
        }
    
    async def run_pipeline(self) -> Dict:
        """Main pipeline for signal detection and analysis"""
        start_time = time.time()
        run_id = datetime.utcnow().isoformat()
        
        logger.info(f"Starting signal detection pipeline - Run ID: {run_id}")
        
        try:
            # Initialize database if needed
            self._ensure_database()
            
            # 1. Collect signals from all sources
            logger.info("Phase 1: Collecting signals from sources")
            raw_signals = await crawl_all_sources()
            logger.info(f"Collected {len(raw_signals)} raw signals")
            
            if not raw_signals:
                logger.warning("No raw signals collected, skipping processing")
                return self._create_run_summary(0, 0, 0, time.time() - start_time)
            
            # 2. Process and extract signals
            logger.info("Phase 2: Processing and extracting signals")
            processed_signals = await process_signals(raw_signals)
            logger.info(f"Processed {len(processed_signals)} valid signals")
            
            # 3. Save signals to database
            logger.info("Phase 3: Saving signals to database")
            saved_signals = await self._save_signals(processed_signals)
            logger.info(f"Saved {len(saved_signals)} signals to database")
            
            # 4. Analyze trends
            logger.info("Phase 4: Analyzing trends")
            trends = await analyze_trends(saved_signals)
            logger.info(f"Identified {len(trends)} trends")
            
            # 5. Save trends to database
            saved_trends = await self._save_trends(trends, saved_signals)
            logger.info(f"Saved {len(saved_trends)} trends to database")
            
            # 6. Detect anomalies
            logger.info("Phase 5: Detecting anomalies")
            anomalies = await detect_anomalies(saved_signals)
            logger.info(f"Detected {len(anomalies)} anomalies")
            
            # 7. Save anomalies to database
            saved_anomalies = await self._save_anomalies(anomalies)
            logger.info(f"Saved {len(saved_anomalies)} anomalies to database")
            
            # 8. Generate daily report if needed
            await self._maybe_generate_daily_report(saved_signals, saved_trends, saved_anomalies)
            
            # Update stats
            processing_time = time.time() - start_time
            self.stats.update({
                'runs': self.stats['runs'] + 1,
                'total_signals': self.stats['total_signals'] + len(saved_signals),
                'total_trends': self.stats['total_trends'] + len(saved_trends),
                'total_anomalies': self.stats['total_anomalies'] + len(saved_anomalies),
                'last_run': datetime.utcnow(),
            })
            
            summary = self._create_run_summary(
                len(saved_signals), len(saved_trends), len(saved_anomalies), processing_time
            )
            
            logger.info(f"Pipeline completed successfully in {processing_time:.2f}s")
            return summary
            
        except Exception as e:
            self.stats['errors'] += 1
            logger.error(f"Pipeline failed: {str(e)}", exc_info=True)
            return {
                'success': False,
                'error': str(e),
                'processing_time': time.time() - start_time,
                'timestamp': datetime.utcnow()
            }
    
    def _ensure_database(self):
        """Ensure database tables exist"""
        try:
            db.create_tables()
            self._ensure_sources()
        except Exception as e:
            logger.error(f"Database initialization failed: {str(e)}")
            raise
    
    def _ensure_sources(self):
        """Ensure sources from config are in database"""
        try:
            with open('config/sources.yaml', 'r') as file:
                sources_config = yaml.safe_load(file)
            
            with db.get_session() as session:
                for tier_name, sources in sources_config.get('sources', {}).items():
                    tier_number = {'tier_1': 1, 'tier_2': 2, 'tier_3': 3}.get(tier_name, 3)
                    
                    for source_config in sources:
                        existing = session.query(Source).filter(Source.name == source_config['name']).first()
                        
                        if not existing:
                            source = Source(
                                name=source_config['name'],
                                url=source_config['url'],
                                type=source_config['type'],
                                tier=tier_number,
                                crawl_frequency=source_config.get('crawl_frequency', 'daily')
                            )
                            session.add(source)
                            logger.info(f"Added new source: {source.name}")
                
                session.commit()
        except Exception as e:
            logger.error(f"Error ensuring sources: {str(e)}")
    
    async def _save_signals(self, processed_signals: List[Dict]) -> List[Signal]:
        """Save processed signals to database"""
        saved_signals = []
        
        try:
            with db.get_session() as session:
                for signal_data in processed_signals:
                    # Find or create source
                    source_name = signal_data.get('source_name')
                    source = session.query(Source).filter(Source.name == source_name).first()
                    
                    if not source:
                        # Create source if not exists
                        source = Source(
                            name=source_name,
                            url=signal_data.get('url', ''),
                            type=signal_data.get('source_type', 'unknown'),
                            tier=3
                        )
                        session.add(source)
                        session.flush()  # Get the ID
                    
                    # Create signal
                    signal = Signal(
                        title=signal_data.get('original_title', signal_data.get('content', '')[:100]),
                        content=signal_data.get('content', ''),
                        url=signal_data.get('url'),
                        source_id=source.id,
                        category=signal_data.get('category'),
                        relevance_score=signal_data.get('confidence', 0.5),
                        confidence_score=signal_data.get('confidence', 0.5),
                        entities=signal_data.get('entities', []),
                        temporal_refs=signal_data.get('temporal_refs', []),
                        timestamp=signal_data.get('extracted_at', datetime.utcnow())
                    )
                    
                    session.add(signal)
                    saved_signals.append(signal)
                
                session.commit()
                
                # Refresh objects to get IDs
                for signal in saved_signals:
                    session.refresh(signal)
                
        except Exception as e:
            logger.error(f"Error saving signals: {str(e)}")
            raise
        
        return saved_signals
    
    async def _save_trends(self, trends: List[Dict], signals: List[Signal]) -> List[Trend]:
        """Save trends to database"""
        saved_trends = []
        
        try:
            with db.get_session() as session:
                for trend_data in trends:
                    # Create trend
                    trend = Trend(
                        name=trend_data.get('name', 'Unnamed Trend'),
                        description=trend_data.get('description', ''),
                        category=trend_data.get('category'),
                        velocity=trend_data.get('velocity', 0.0),
                        trajectory=trend_data.get('trajectory', 'stable'),
                        confidence=trend_data.get('confidence', 0.5),
                        first_detected=trend_data.get('first_detected', datetime.utcnow()),
                        last_updated=trend_data.get('last_updated', datetime.utcnow())
                    )
                    
                    session.add(trend)
                    session.flush()  # Get the ID
                    
                    # Associate signals with trend (simplified - using all signals for now)
                    # In production, you'd match based on trend_data['constituent_signals']
                    for signal in signals[:min(len(signals), trend_data.get('signal_count', 0))]:
                        trend_signal = TrendSignal(
                            trend_id=trend.id,
                            signal_id=signal.id,
                            relevance_score=0.8  # Default relevance
                        )
                        session.add(trend_signal)
                    
                    saved_trends.append(trend)
                
                session.commit()
                
        except Exception as e:
            logger.error(f"Error saving trends: {str(e)}")
            raise
        
        return saved_trends
    
    async def _save_anomalies(self, anomalies: List[Dict]) -> List[AnomalyScore]:
        """Save anomalies to database"""
        saved_anomalies = []
        
        try:
            with db.get_session() as session:
                for anomaly_data in anomalies:
                    # Find the signal this anomaly refers to
                    # This is simplified - in production you'd have better signal matching
                    content = anomaly_data.get('content', '')
                    signal = session.query(Signal).filter(Signal.content.contains(content[:100])).first()
                    
                    if signal:
                        anomaly_scores = anomaly_data.get('anomaly_scores', {})
                        
                        anomaly = AnomalyScore(
                            signal_id=signal.id,
                            oddity_score=anomaly_scores.get('oddity', 0.0),
                            statistical_anomaly=anomaly_scores.get('statistical', False),
                            contradiction_score=anomaly_scores.get('contradiction', 0.0),
                            anomaly_type=anomaly_data.get('anomaly_type', 'general'),
                            overall_score=anomaly_data.get('overall_anomaly_score', 0.0)
                        )
                        
                        session.add(anomaly)
                        saved_anomalies.append(anomaly)
                
                session.commit()
                
        except Exception as e:
            logger.error(f"Error saving anomalies: {str(e)}")
            raise
        
        return saved_anomalies
    
    async def _maybe_generate_daily_report(self, signals: List[Signal], trends: List[Trend], anomalies: List[AnomalyScore]):
        """Generate daily report if it's a new day"""
        try:
            today = datetime.utcnow().date()
            
            with db.get_session() as session:
                existing_report = session.query(DailyReport).filter(
                    DailyReport.report_date >= today
                ).first()
                
                if not existing_report:
                    # Count signals by category
                    category_counts = {}
                    for signal in signals:
                        cat = signal.category or 'unknown'
                        category_counts[cat] = category_counts.get(cat, 0) + 1
                    
                    top_categories = sorted(category_counts.items(), key=lambda x: x[1], reverse=True)[:5]
                    
                    # Generate summary
                    summary = f"Daily Signal Report: {len(signals)} signals analyzed, {len(trends)} trends identified, {len(anomalies)} anomalies detected."
                    
                    report = DailyReport(
                        report_date=datetime.utcnow(),
                        total_signals=len(signals),
                        new_trends=len(trends),
                        anomalies_detected=len(anomalies),
                        top_categories=[{"category": cat, "count": count} for cat, count in top_categories],
                        summary=summary,
                        report_data={
                            "categories": dict(top_categories),
                            "trend_categories": [t.category for t in trends],
                            "anomaly_types": [a.anomaly_type for a in anomalies]
                        }
                    )
                    
                    session.add(report)
                    session.commit()
                    logger.info("Generated daily report")
                
        except Exception as e:
            logger.error(f"Error generating daily report: {str(e)}")
    
    def _create_run_summary(self, signals: int, trends: int, anomalies: int, processing_time: float) -> Dict:
        """Create run summary"""
        return {
            'success': True,
            'signals': signals,
            'trends': trends,
            'anomalies': anomalies,
            'processing_time': processing_time,
            'timestamp': datetime.utcnow(),
            'total_runs': self.stats['runs'] + 1
        }

# Global pipeline instance
pipeline = SignalPipeline()

def run_scheduled_job():
    """Run the async pipeline in sync context"""
    try:
        result = asyncio.run(pipeline.run_pipeline())
        logger.info(f"Scheduled job completed: {result}")
    except Exception as e:
        logger.error(f"Scheduled job failed: {str(e)}")

def setup_scheduler():
    """Setup scheduled jobs"""
    # Run every 6 hours
    schedule.every(6).hours.do(run_scheduled_job)
    
    # Also run daily at 9 AM for comprehensive analysis
    schedule.every().day.at("09:00").do(run_scheduled_job)
    
    logger.info("Scheduler configured: every 6 hours and daily at 9 AM")

async def run_once():
    """Run pipeline once for testing"""
    return await pipeline.run_pipeline()

def main():
    """Main entry point"""
    logger.info("Starting Futurist Signal Detection System Orchestrator")
    
    # Run once immediately for testing
    logger.info("Running initial pipeline execution...")
    try:
        result = asyncio.run(run_once())
        logger.info(f"Initial run completed: {result}")
    except Exception as e:
        logger.error(f"Initial run failed: {str(e)}")
    
    # Setup scheduler
    setup_scheduler()
    
    # Run scheduler
    logger.info("Starting scheduled execution loop...")
    try:
        while True:
            schedule.run_pending()
            time.sleep(60)  # Check every minute
    except KeyboardInterrupt:
        logger.info("Orchestrator stopped by user")
    except Exception as e:
        logger.error(f"Orchestrator error: {str(e)}")

if __name__ == "__main__":
    main()