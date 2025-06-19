# Futurist Signal Detection System

An automated system for detecting and analyzing signals of future change from multiple online sources.

## Features

- **Multi-source crawling**: Monitors 30+ news sources, research publications, and analysis sites
- **Intelligent signal extraction**: Uses NLP and ML to identify potential future change indicators
- **Trend synthesis**: Clusters related signals to identify emerging trends
- **Anomaly detection**: Flags unusual or contradictory information that may indicate paradigm shifts
- **RESTful API**: Provides programmatic access to signals, trends, and anomalies
- **Real-time monitoring**: Continuous crawling and analysis with configurable frequencies

## Quick Start

### Prerequisites

- Python 3.11+
- PostgreSQL 15+
- Redis 7+
- Docker & Docker Compose (optional)

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd futurist-signal-system
```

2. Install dependencies:
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

3. Configure environment:
```bash
cp .env.example .env
# Edit .env with your database and API configurations
```

4. Initialize database:
```bash
python -c "from src.storage.database import db; db.create_tables()"
```

### Running with Docker

```bash
docker-compose up -d
```

This will start:
- API server on http://localhost:8000
- PostgreSQL database on port 5432
- Redis on port 6379
- Elasticsearch on port 9200
- Grafana dashboard on http://localhost:3000

### Running Locally

1. Start the API server:
```bash
uvicorn src.api.main:app --reload
```

2. Run the orchestration pipeline:
```bash
python scripts/orchestrate.py
```

## API Endpoints

### Signals
- `GET /signals` - Get recent signals with filtering
- `GET /signals?category=technological&days=7` - Filter by category and timeframe
- `GET /search?q=artificial intelligence` - Search signals by keyword

### Trends
- `GET /trends` - Get synthesized trends
- `GET /trends?trajectory=accelerating` - Filter by trend trajectory

### Anomalies
- `GET /anomalies` - Get detected anomalies
- `GET /anomalies?min_score=0.8` - Filter by anomaly score

### Reports
- `GET /reports/daily` - Get daily signal digest
- `GET /reports/daily?date=2024-01-15` - Get report for specific date

### System
- `GET /stats` - Get system statistics
- `POST /analyze` - Trigger immediate analysis
- `GET /sources` - Get configured sources

## Configuration

### Sources (`config/sources.yaml`)

Add new sources by configuring:
- URL and selectors for crawling
- Source type and reliability tier
- Crawl frequency

Example:
```yaml
sources:
  tier_1:
    - name: "MIT Technology Review"
      url: "https://www.technologyreview.com"
      type: "tech_news"
      crawl_frequency: "6h"
      selectors:
        articles: "article.content-card"
        title: "h3"
        link: "a[href]"
        summary: "p.content-card__deck"
```

### Settings (`config/settings.py`)

Key configuration options:
- Database connections
- Signal detection patterns
- Trend analysis thresholds
- API authentication

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Data Sources  │───▶│    Collectors   │───▶│   Processors    │
│                 │    │                 │    │                 │
│ • News Sites    │    │ • Web Crawlers  │    │ • Signal Extract│
│ • Research Pubs │    │ • API Clients   │    │ • Text Analysis │
│ • Analysis Sites│    │ • Feed Readers  │    │ • Classification│
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                        │
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Database      │◀───│    Analyzers    │◀───│   Raw Signals   │
│                 │    │                 │    │                 │
│ • Signals       │    │ • Trend Synth   │    │ • Categorized   │
│ • Trends        │    │ • Anomaly Det   │    │ • Scored        │
│ • Anomalies     │    │ • Pattern Rec   │    │ • Timestamped   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │
┌─────────────────┐    ┌─────────────────┐
│   API/Dashboard │◀───│   Orchestrator  │
│                 │    │                 │
│ • REST API      │    │ • Scheduler     │
│ • Web Interface │    │ • Pipeline Mgmt │
│ • Reports       │    │ • Monitoring    │
└─────────────────┘    └─────────────────┘
```

## Signal Detection

The system identifies signals using:

1. **Pattern Matching**: Predefined patterns for breakthrough language
2. **NLP Analysis**: Entity extraction and semantic analysis
3. **Temporal Analysis**: Time references and future projections
4. **Confidence Scoring**: Multi-factor scoring based on source reliability and content

Example signal patterns:
- "breakthrough in", "could revolutionize", "by 2030"
- "unprecedented", "never before seen", "paradigm shift"
- Quantitative claims with future implications

## Trend Analysis

Trends are synthesized from signals using:

1. **Semantic Clustering**: Group related signals by content similarity
2. **Entity Clustering**: Group signals mentioning same organizations/technologies
3. **Velocity Calculation**: Track signal frequency over time
4. **Trajectory Analysis**: Determine if trends are accelerating/decelerating

## Anomaly Detection

The system flags signals that are:

1. **Linguistically Odd**: Unusual language patterns or extreme claims
2. **Statistically Anomalous**: Impossible percentages or numbers
3. **Contradictory**: Conflicting with established knowledge
4. **Temporally Impossible**: Unrealistic timeframes

## Development

### Adding New Sources

1. Add source configuration to `config/sources.yaml`
2. Test with: `python -c "from src.collectors.crawler_factory import CrawlerFactory; c = CrawlerFactory.create_crawler({'name': 'Test', 'url': 'https://example.com', 'type': 'test'}); print(asyncio.run(c.crawl()))"`

### Extending Signal Patterns

1. Add patterns to `config/settings.py`
2. Update `SignalExtractor` class for new detection logic
3. Test with sample texts

### Custom Analysis

1. Create new analyzer in `src/analyzers/`
2. Add to pipeline in `scripts/orchestrate.py`
3. Update API endpoints as needed

## Monitoring

The system provides monitoring through:

- **Logs**: Structured logging in `logs/` directory
- **Metrics**: Prometheus metrics on port 8001
- **Dashboard**: Grafana dashboard on port 3000
- **API Statistics**: `/stats` endpoint

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## License

MIT License - see LICENSE file for details.

## Troubleshooting

### Common Issues

1. **spaCy model not found**: Run `python -m spacy download en_core_web_sm`
2. **Database connection failed**: Check PostgreSQL is running and credentials are correct
3. **Crawling errors**: Check source websites haven't changed structure
4. **Memory issues**: Reduce `MAX_CRAWL_WORKERS` in settings

### Performance Tuning

1. **Increase workers**: Adjust `MAX_CRAWL_WORKERS` for more parallel crawling
2. **Database optimization**: Add indexes for frequently queried fields
3. **Caching**: Enable Redis caching for API responses
4. **Resource limits**: Monitor memory usage during trend analysis

For more detailed documentation, see the `docs/` directory.