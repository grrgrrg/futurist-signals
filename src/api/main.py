from fastapi import FastAPI, HTTPException, Query, Depends
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
import asyncio
from pydantic import BaseModel
import logging
from config.settings import settings
from src.storage.database import db
from src.storage.models import Signal, Trend, Source, AnomalyScore, DailyReport
from sqlalchemy.orm import Session
from sqlalchemy import desc, and_, func

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Futurist Signal Detection API",
    description="API for accessing and analyzing future change signals",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for API responses
class SignalResponse(BaseModel):
    id: int
    title: str
    content: str
    url: Optional[str]
    category: str
    confidence: float
    source_name: str
    timestamp: datetime

class TrendResponse(BaseModel):
    id: int
    name: str
    description: str
    category: str
    velocity: float
    trajectory: str
    confidence: float
    signal_count: int
    first_detected: datetime
    last_updated: datetime

class AnomalyResponse(BaseModel):
    id: int
    signal_id: int
    signal_title: str
    anomaly_type: str
    overall_score: float
    detected_at: datetime

class StatsResponse(BaseModel):
    total_signals: int
    total_trends: int
    total_anomalies: int
    signals_today: int
    top_categories: List[Dict[str, Any]]
    recent_activity: List[Dict[str, Any]]

# Dependency to get database session
def get_db():
    with db.get_session() as session:
        yield session

@app.get("/")
async def root():
    return {
        "message": "Futurist Signal Detection System API",
        "version": "1.0.0",
        "endpoints": {
            "signals": "/signals - Get recent signals",
            "trends": "/trends - Get synthesized trends", 
            "anomalies": "/anomalies - Get detected anomalies",
            "stats": "/stats - Get system statistics",
            "search": "/search - Search signals",
            "reports": "/reports/daily - Get daily reports"
        }
    }

@app.get("/signals", response_model=List[SignalResponse])
async def get_signals(
    category: Optional[str] = None,
    source: Optional[str] = None,
    days: int = Query(7, ge=1, le=90),
    limit: int = Query(100, ge=1, le=1000),
    min_confidence: float = Query(0.0, ge=0.0, le=1.0),
    db: Session = Depends(get_db)
):
    """Get recent signals with filtering options"""
    try:
        # Build query
        query = db.query(Signal).join(Source)
        
        # Apply filters
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        query = query.filter(Signal.timestamp >= cutoff_date)
        
        if category:
            query = query.filter(Signal.category == category)
        
        if source:
            query = query.filter(Source.name == source)
        
        if min_confidence > 0:
            query = query.filter(Signal.confidence_score >= min_confidence)
        
        # Order by timestamp and limit
        signals = query.order_by(desc(Signal.timestamp)).limit(limit).all()
        
        # Convert to response format
        return [
            SignalResponse(
                id=signal.id,
                title=signal.title,
                content=signal.content[:500] + "..." if len(signal.content) > 500 else signal.content,
                url=signal.url,
                category=signal.category or "unknown",
                confidence=signal.confidence_score or 0.0,
                source_name=signal.source.name,
                timestamp=signal.timestamp
            )
            for signal in signals
        ]
    
    except Exception as e:
        logger.error(f"Error fetching signals: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/trends", response_model=List[TrendResponse])
async def get_trends(
    trajectory: Optional[str] = None,
    category: Optional[str] = None,
    min_velocity: Optional[float] = None,
    days: int = Query(30, ge=1, le=90),
    limit: int = Query(50, ge=1, le=200),
    db: Session = Depends(get_db)
):
    """Get synthesized trends"""
    try:
        # Build query
        query = db.query(Trend).filter(Trend.is_active == True)
        
        # Apply filters
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        query = query.filter(Trend.last_updated >= cutoff_date)
        
        if trajectory:
            query = query.filter(Trend.trajectory == trajectory)
        
        if category:
            query = query.filter(Trend.category == category)
        
        if min_velocity is not None:
            query = query.filter(Trend.velocity >= min_velocity)
        
        # Order by velocity and confidence
        trends = query.order_by(desc(Trend.velocity), desc(Trend.confidence)).limit(limit).all()
        
        # Get signal counts for each trend
        return [
            TrendResponse(
                id=trend.id,
                name=trend.name,
                description=trend.description or "",
                category=trend.category or "unknown",
                velocity=trend.velocity or 0.0,
                trajectory=trend.trajectory or "stable",
                confidence=trend.confidence or 0.0,
                signal_count=len(trend.signal_associations),
                first_detected=trend.first_detected,
                last_updated=trend.last_updated
            )
            for trend in trends
        ]
    
    except Exception as e:
        logger.error(f"Error fetching trends: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/anomalies", response_model=List[AnomalyResponse])
async def get_anomalies(
    anomaly_type: Optional[str] = None,
    min_score: float = Query(0.7, ge=0.0, le=1.0),
    days: int = Query(30, ge=1, le=90),
    limit: int = Query(100, ge=1, le=500),
    db: Session = Depends(get_db)
):
    """Get detected anomalies"""
    try:
        # Build query
        query = db.query(AnomalyScore).join(Signal)
        
        # Apply filters
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        query = query.filter(AnomalyScore.detected_at >= cutoff_date)
        
        if anomaly_type:
            query = query.filter(AnomalyScore.anomaly_type == anomaly_type)
        
        query = query.filter(AnomalyScore.overall_score >= min_score)
        
        # Order by score and date
        anomalies = query.order_by(desc(AnomalyScore.overall_score), desc(AnomalyScore.detected_at)).limit(limit).all()
        
        return [
            AnomalyResponse(
                id=anomaly.id,
                signal_id=anomaly.signal_id,
                signal_title=anomaly.signal.title,
                anomaly_type=anomaly.anomaly_type or "unknown",
                overall_score=anomaly.overall_score or 0.0,
                detected_at=anomaly.detected_at
            )
            for anomaly in anomalies
        ]
    
    except Exception as e:
        logger.error(f"Error fetching anomalies: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/stats", response_model=StatsResponse)
async def get_stats(db: Session = Depends(get_db)):
    """Get system statistics"""
    try:
        # Basic counts
        total_signals = db.query(Signal).count()
        total_trends = db.query(Trend).filter(Trend.is_active == True).count()
        total_anomalies = db.query(AnomalyScore).count()
        
        # Signals today
        today = datetime.utcnow().date()
        signals_today = db.query(Signal).filter(
            func.date(Signal.timestamp) == today
        ).count()
        
        # Top categories in last 7 days
        week_ago = datetime.utcnow() - timedelta(days=7)
        category_stats = db.query(
            Signal.category,
            func.count(Signal.id).label('count')
        ).filter(
            Signal.timestamp >= week_ago,
            Signal.category.isnot(None)
        ).group_by(Signal.category).order_by(desc('count')).limit(5).all()
        
        top_categories = [
            {"category": stat.category, "count": stat.count}
            for stat in category_stats
        ]
        
        # Recent activity (last 24 hours)
        yesterday = datetime.utcnow() - timedelta(hours=24)
        recent_signals = db.query(Signal).filter(
            Signal.timestamp >= yesterday
        ).order_by(desc(Signal.timestamp)).limit(10).all()
        
        recent_activity = [
            {
                "type": "signal",
                "title": signal.title[:100],
                "category": signal.category,
                "timestamp": signal.timestamp,
                "source": signal.source.name
            }
            for signal in recent_signals
        ]
        
        return StatsResponse(
            total_signals=total_signals,
            total_trends=total_trends,
            total_anomalies=total_anomalies,
            signals_today=signals_today,
            top_categories=top_categories,
            recent_activity=recent_activity
        )
    
    except Exception as e:
        logger.error(f"Error fetching stats: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/search")
async def search_signals(
    q: str = Query(..., min_length=3),
    category: Optional[str] = None,
    fuzzy: bool = False,
    limit: int = Query(50, ge=1, le=200),
    db: Session = Depends(get_db)
):
    """Search signals by keyword"""
    try:
        # Build query
        query = db.query(Signal).join(Source)
        
        # Search in title and content
        search_filter = Signal.title.contains(q) | Signal.content.contains(q)
        query = query.filter(search_filter)
        
        if category:
            query = query.filter(Signal.category == category)
        
        # Order by relevance (confidence score) and recency
        signals = query.order_by(
            desc(Signal.confidence_score),
            desc(Signal.timestamp)
        ).limit(limit).all()
        
        results = []
        for signal in signals:
            # Calculate relevance score based on keyword matches
            title_matches = signal.title.lower().count(q.lower())
            content_matches = signal.content.lower().count(q.lower())
            relevance = title_matches * 2 + content_matches
            
            results.append({
                "id": signal.id,
                "title": signal.title,
                "content": signal.content[:300] + "..." if len(signal.content) > 300 else signal.content,
                "url": signal.url,
                "category": signal.category,
                "confidence": signal.confidence_score,
                "source": signal.source.name,
                "timestamp": signal.timestamp,
                "relevance": relevance
            })
        
        # Sort by relevance
        results.sort(key=lambda x: x['relevance'], reverse=True)
        
        return {
            "query": q,
            "total_results": len(results),
            "results": results
        }
    
    except Exception as e:
        logger.error(f"Error searching signals: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/reports/daily")
async def get_daily_report(
    date: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """Get daily signal digest"""
    try:
        if date:
            try:
                report_date = datetime.strptime(date, "%Y-%m-%d").date()
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD")
        else:
            report_date = datetime.utcnow().date()
        
        # Look for existing report
        report = db.query(DailyReport).filter(
            func.date(DailyReport.report_date) == report_date
        ).first()
        
        if report:
            return {
                "date": report.report_date,
                "total_signals": report.total_signals,
                "new_trends": report.new_trends,
                "anomalies_detected": report.anomalies_detected,
                "top_categories": report.top_categories,
                "summary": report.summary,
                "generated_at": report.generated_at,
                "data": report.report_data
            }
        else:
            # Generate report on the fly
            start_date = datetime.combine(report_date, datetime.min.time())
            end_date = start_date + timedelta(days=1)
            
            day_signals = db.query(Signal).filter(
                and_(Signal.timestamp >= start_date, Signal.timestamp < end_date)
            ).all()
            
            category_counts = {}
            for signal in day_signals:
                cat = signal.category or "unknown"
                category_counts[cat] = category_counts.get(cat, 0) + 1
            
            top_categories = sorted(category_counts.items(), key=lambda x: x[1], reverse=True)[:5]
            
            return {
                "date": report_date,
                "total_signals": len(day_signals),
                "new_trends": 0,  # Would need trend detection for this date
                "anomalies_detected": 0,  # Would need anomaly detection for this date
                "top_categories": [{"category": cat, "count": count} for cat, count in top_categories],
                "summary": f"Found {len(day_signals)} signals across {len(category_counts)} categories",
                "generated_at": datetime.utcnow(),
                "data": None
            }
    
    except Exception as e:
        logger.error(f"Error fetching daily report: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/analyze")
async def trigger_analysis():
    """Trigger immediate analysis cycle"""
    try:
        # This would trigger the analysis pipeline
        # For now, return a placeholder response
        return {
            "message": "Analysis triggered",
            "status": "processing",
            "timestamp": datetime.utcnow()
        }
    except Exception as e:
        logger.error(f"Error triggering analysis: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/sources")
async def get_sources(db: Session = Depends(get_db)):
    """Get list of configured sources"""
    try:
        sources = db.query(Source).filter(Source.is_active == True).all()
        
        return [
            {
                "id": source.id,
                "name": source.name,
                "url": source.url,
                "type": source.type,
                "tier": source.tier,
                "reliability_score": source.reliability_score,
                "last_checked": source.last_checked
            }
            for source in sources
        ]
    
    except Exception as e:
        logger.error(f"Error fetching sources: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.api.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.DEBUG
    )