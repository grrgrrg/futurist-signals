import os
from typing import List, Dict
from dotenv import load_dotenv

load_dotenv()

class Settings:
    # Database
    DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://user:password@localhost:5432/futurist_signals")
    REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
    
    # Application
    DEBUG = os.getenv("DEBUG", "False").lower() == "true"
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-here")
    
    # Crawling
    MAX_CRAWL_WORKERS = int(os.getenv("MAX_CRAWL_WORKERS", "10"))
    CRAWL_TIMEOUT = 30
    MAX_RETRIES = 3
    
    # Signal Processing
    SIGNAL_RETENTION_DAYS = int(os.getenv("SIGNAL_RETENTION_DAYS", "365"))
    MIN_SIGNAL_CONFIDENCE = 0.3
    TREND_MIN_SIGNALS = 3
    TREND_TIME_WINDOW_DAYS = 7
    
    # ML Models
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")
    SENTENCE_TRANSFORMER_MODEL = "all-MiniLM-L6-v2"
    
    # API
    API_TOKEN = os.getenv("API_TOKEN", "your-api-token-here")
    API_HOST = "0.0.0.0"
    API_PORT = 8000
    
    # Monitoring
    PROMETHEUS_PORT = int(os.getenv("PROMETHEUS_PORT", "8001"))
    
    # Signal Detection Patterns
    SIGNAL_PATTERNS = [
        r"for the first time",
        r"breakthrough in",
        r"could revolutionize",
        r"by 20\d{2}",
        r"\d+% increase",
        r"unexpected discovery",
        r"paradigm shift",
        r"game.?changer",
        r"disrupt(?:ive|ing|ion)",
        r"transform(?:ative|ing|ation)",
        r"unprecedented",
        r"never before seen",
        r"world.?first",
        r"cutting.?edge",
        r"next.?generation"
    ]
    
    # Anomaly Detection
    ANOMALY_PATTERNS = [
        r"contrary to expectations",
        r"scientists? puzzled",
        r"unexplained",
        r"anomal(?:y|ous)",
        r"strange(?:ly)?",
        r"weird(?:ly)?",
        r"unusual(?:ly)?",
        r"first time ever",
        r"never before seen",
        r"defies? .* laws?"
    ]
    
    # Trend Thresholds
    TREND_VELOCITY_THRESHOLD = 0.5  # signals per day
    ACCELERATION_THRESHOLD = 1.2    # velocity multiplier
    DECELERATION_THRESHOLD = 0.8    # velocity multiplier

settings = Settings()