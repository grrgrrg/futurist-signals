from sqlalchemy import Column, Integer, String, DateTime, Float, Text, ForeignKey, Boolean, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime

Base = declarative_base()

class Source(Base):
    __tablename__ = 'sources'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(255), unique=True, nullable=False)
    url = Column(String(500), nullable=False)
    type = Column(String(50), nullable=False)
    tier = Column(Integer, default=3)  # 1=highest quality, 3=lowest
    reliability_score = Column(Float, default=0.5)
    last_checked = Column(DateTime)
    is_active = Column(Boolean, default=True)
    crawl_frequency = Column(String(20), default="daily")
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    signals = relationship('Signal', back_populates='source')
    crawl_stats = relationship('CrawlStats', back_populates='source')

class Signal(Base):
    __tablename__ = 'signals'
    
    id = Column(Integer, primary_key=True)
    title = Column(String(500), nullable=False)
    content = Column(Text, nullable=False)
    url = Column(String(1000))
    source_id = Column(Integer, ForeignKey('sources.id'), nullable=False)
    category = Column(String(50))
    relevance_score = Column(Float)
    confidence_score = Column(Float)
    timestamp = Column(DateTime, default=datetime.utcnow)
    raw_html = Column(Text)
    entities = Column(JSON)  # Store extracted entities as JSON
    temporal_refs = Column(JSON)  # Store temporal references
    processed = Column(Boolean, default=False)
    
    # Relationships
    source = relationship('Source', back_populates='signals')
    trend_associations = relationship('TrendSignal', back_populates='signal')
    anomaly_scores = relationship('AnomalyScore', back_populates='signal', uselist=False)

class Trend(Base):
    __tablename__ = 'trends'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(255), nullable=False)
    description = Column(Text)
    category = Column(String(50))
    velocity = Column(Float)  # signals per day
    trajectory = Column(String(50))  # accelerating, stable, decelerating
    confidence = Column(Float)
    first_detected = Column(DateTime, default=datetime.utcnow)
    last_updated = Column(DateTime, default=datetime.utcnow)
    is_active = Column(Boolean, default=True)
    
    # Relationships
    signal_associations = relationship('TrendSignal', back_populates='trend')
    implications = relationship('TrendImplication', back_populates='trend')

class TrendSignal(Base):
    __tablename__ = 'trend_signals'
    
    id = Column(Integer, primary_key=True)
    trend_id = Column(Integer, ForeignKey('trends.id'), nullable=False)
    signal_id = Column(Integer, ForeignKey('signals.id'), nullable=False)
    relevance_score = Column(Float)
    added_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    trend = relationship('Trend', back_populates='signal_associations')
    signal = relationship('Signal', back_populates='trend_associations')

class AnomalyScore(Base):
    __tablename__ = 'anomaly_scores'
    
    id = Column(Integer, primary_key=True)
    signal_id = Column(Integer, ForeignKey('signals.id'), nullable=False)
    oddity_score = Column(Float)
    statistical_anomaly = Column(Boolean, default=False)
    contradiction_score = Column(Float)
    anomaly_type = Column(String(50))
    overall_score = Column(Float)
    detected_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    signal = relationship('Signal', back_populates='anomaly_scores')

class TrendImplication(Base):
    __tablename__ = 'trend_implications'
    
    id = Column(Integer, primary_key=True)
    trend_id = Column(Integer, ForeignKey('trends.id'), nullable=False)
    implication_text = Column(Text, nullable=False)
    category = Column(String(50))  # economic, social, technological, etc.
    probability = Column(Float)
    time_horizon = Column(String(20))  # short, medium, long
    impact_level = Column(String(20))  # low, medium, high
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    trend = relationship('Trend', back_populates='implications')

class CrawlStats(Base):
    __tablename__ = 'crawl_stats'
    
    id = Column(Integer, primary_key=True)
    source_id = Column(Integer, ForeignKey('sources.id'), nullable=False)
    crawl_timestamp = Column(DateTime, default=datetime.utcnow)
    articles_found = Column(Integer, default=0)
    signals_extracted = Column(Integer, default=0)
    processing_time = Column(Float)  # seconds
    success = Column(Boolean, default=True)
    error_message = Column(Text)
    
    # Relationships
    source = relationship('Source', back_populates='crawl_stats')

class DailyReport(Base):
    __tablename__ = 'daily_reports'
    
    id = Column(Integer, primary_key=True)
    report_date = Column(DateTime, nullable=False)
    total_signals = Column(Integer, default=0)
    new_trends = Column(Integer, default=0)
    anomalies_detected = Column(Integer, default=0)
    top_categories = Column(JSON)
    summary = Column(Text)
    report_data = Column(JSON)  # Full report as JSON
    generated_at = Column(DateTime, default=datetime.utcnow)