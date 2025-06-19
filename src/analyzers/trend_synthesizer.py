from typing import List, Dict, Optional
import numpy as np
from sklearn.cluster import DBSCAN
from collections import defaultdict, Counter
from datetime import datetime, timedelta
import logging
from config.settings import settings

logger = logging.getLogger(__name__)

class TrendSynthesizer:
    def __init__(self):
        self.sentence_transformer = None
        self._initialize_models()
        
    def _initialize_models(self):
        """Initialize ML models with fallback"""
        try:
            from sentence_transformers import SentenceTransformer
            self.sentence_transformer = SentenceTransformer(settings.SENTENCE_TRANSFORMER_MODEL)
            logger.info("Sentence transformer model loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to load sentence transformer: {str(e)}")
            self.sentence_transformer = None
    
    def synthesize_trends(self, signals: List[Dict], time_window_days: int = None) -> List[Dict]:
        """Identify trends from signals within time window"""
        if not signals:
            return []
        
        if time_window_days is None:
            time_window_days = settings.TREND_TIME_WINDOW_DAYS
        
        # Filter signals by time window
        cutoff_date = datetime.utcnow() - timedelta(days=time_window_days)
        recent_signals = [
            s for s in signals 
            if s.get('extracted_at', datetime.utcnow()) > cutoff_date
        ]
        
        if len(recent_signals) < settings.TREND_MIN_SIGNALS:
            logger.info(f"Not enough signals ({len(recent_signals)}) to identify trends")
            return []
        
        logger.info(f"Analyzing {len(recent_signals)} recent signals for trends")
        
        # Group signals by category first for more focused trend detection
        category_groups = defaultdict(list)
        for signal in recent_signals:
            category = signal.get('category', 'unknown')
            category_groups[category].append(signal)
        
        all_trends = []
        
        for category, category_signals in category_groups.items():
            if len(category_signals) >= settings.TREND_MIN_SIGNALS:
                trends = self._find_trends_in_category(category_signals, category)
                all_trends.extend(trends)
        
        # Also look for cross-category trends
        cross_category_trends = self._find_cross_category_trends(recent_signals)
        all_trends.extend(cross_category_trends)
        
        logger.info(f"Identified {len(all_trends)} potential trends")
        return all_trends
    
    def _find_trends_in_category(self, signals: List[Dict], category: str) -> List[Dict]:
        """Find trends within a specific category"""
        if len(signals) < settings.TREND_MIN_SIGNALS:
            return []
        
        trends = []
        
        # Use semantic clustering if available
        if self.sentence_transformer:
            trends.extend(self._semantic_clustering(signals, category))
        
        # Use keyword-based clustering as backup/supplement
        trends.extend(self._keyword_clustering(signals, category))
        
        return trends
    
    def _semantic_clustering(self, signals: List[Dict], category: str) -> List[Dict]:
        """Use semantic similarity to cluster signals"""
        try:
            texts = [s.get('content', '') for s in signals]
            embeddings = self.sentence_transformer.encode(texts)
            
            # Use DBSCAN for clustering
            clustering = DBSCAN(eps=0.3, min_samples=settings.TREND_MIN_SIGNALS, metric='cosine')
            clusters = clustering.fit_predict(embeddings)
            
            trends = []
            cluster_signals = defaultdict(list)
            
            for idx, cluster_id in enumerate(clusters):
                if cluster_id != -1:  # Ignore noise
                    cluster_signals[cluster_id].append(signals[idx])
            
            for cluster_id, cluster_sigs in cluster_signals.items():
                if len(cluster_sigs) >= settings.TREND_MIN_SIGNALS:
                    trend = self._build_trend(cluster_sigs, f"{category}_semantic_{cluster_id}")
                    trends.append(trend)
            
            return trends
            
        except Exception as e:
            logger.error(f"Error in semantic clustering: {str(e)}")
            return []
    
    def _keyword_clustering(self, signals: List[Dict], category: str) -> List[Dict]:
        """Use keyword overlap to cluster signals"""
        # Extract entities and keywords from signals
        signal_keywords = []
        for signal in signals:
            keywords = set()
            
            # Add entities
            entities = signal.get('entities', [])
            for entity, _ in entities:
                keywords.add(entity.lower())
            
            # Add significant words from content
            content = signal.get('content', '').lower()
            words = content.split()
            # Filter for meaningful words (longer than 3 chars, not common words)
            meaningful_words = [
                w for w in words 
                if len(w) > 3 and w not in ['that', 'this', 'with', 'from', 'they', 'will', 'have', 'been']
            ]
            keywords.update(meaningful_words[:10])  # Top 10 words
            
            signal_keywords.append(keywords)
        
        # Find clusters based on keyword overlap
        clusters = []
        used_signals = set()
        
        for i, signal_i in enumerate(signals):
            if i in used_signals:
                continue
            
            cluster = [i]
            keywords_i = signal_keywords[i]
            
            for j, signal_j in enumerate(signals[i+1:], i+1):
                if j in used_signals:
                    continue
                
                keywords_j = signal_keywords[j]
                
                # Calculate Jaccard similarity
                intersection = len(keywords_i & keywords_j)
                union = len(keywords_i | keywords_j)
                
                if union > 0 and intersection / union > 0.3:  # 30% similarity threshold
                    cluster.append(j)
                    used_signals.add(j)
            
            if len(cluster) >= settings.TREND_MIN_SIGNALS:
                cluster_signals = [signals[idx] for idx in cluster]
                trend = self._build_trend(cluster_signals, f"{category}_keyword_{len(clusters)}")
                clusters.append(trend)
                used_signals.update(cluster)
        
        return clusters
    
    def _find_cross_category_trends(self, signals: List[Dict]) -> List[Dict]:
        """Find trends that span multiple categories"""
        # Look for signals with shared entities across categories
        entity_signals = defaultdict(list)
        
        for signal in signals:
            entities = signal.get('entities', [])
            for entity, entity_type in entities:
                if entity_type in ['ORG', 'PERSON', 'GPE', 'PRODUCT']:
                    entity_signals[entity.lower()].append(signal)
        
        cross_trends = []
        for entity, entity_sigs in entity_signals.items():
            if len(entity_sigs) >= settings.TREND_MIN_SIGNALS:
                # Check if signals span multiple categories
                categories = [s.get('category') for s in entity_sigs]
                unique_categories = len(set(categories))
                
                if unique_categories >= 2:
                    trend = self._build_trend(entity_sigs, f"cross_category_{entity}")
                    trend['trend_type'] = 'cross_category'
                    trend['entity_focus'] = entity
                    cross_trends.append(trend)
        
        return cross_trends
    
    def _build_trend(self, signals: List[Dict], trend_id: str) -> Dict:
        """Build trend from clustered signals"""
        if not signals:
            return {}
        
        # Calculate velocity (signals per day)
        timestamps = [s.get('extracted_at', datetime.utcnow()) for s in signals]
        min_time = min(timestamps)
        max_time = max(timestamps)
        time_span_days = max((max_time - min_time).days, 1)
        velocity = len(signals) / time_span_days
        
        # Determine trajectory
        trajectory = self._calculate_trajectory(signals)
        
        # Generate trend summary
        categories = [s.get('category') for s in signals if s.get('category')]
        dominant_category = Counter(categories).most_common(1)[0][0] if categories else 'unknown'
        
        # Calculate average confidence
        confidences = [s.get('confidence', 0.5) for s in signals]
        avg_confidence = np.mean(confidences) if confidences else 0.5
        
        return {
            'name': self._generate_trend_name(signals),
            'description': self._generate_description(signals),
            'signal_count': len(signals),
            'velocity': velocity,
            'trajectory': trajectory,
            'category': dominant_category,
            'confidence': avg_confidence,
            'first_detected': min_time,
            'last_updated': max_time,
            'constituent_signals': [s.get('id') for s in signals if s.get('id')],
            'trend_id': trend_id,
            'created_at': datetime.utcnow()
        }
    
    def _calculate_trajectory(self, signals: List[Dict]) -> str:
        """Determine if trend is accelerating, stable, or decelerating"""
        if len(signals) < 4:
            return "stable"
        
        # Sort signals by timestamp
        sorted_signals = sorted(signals, key=lambda x: x.get('extracted_at', datetime.utcnow()))
        
        # Split into halves and compare velocity
        mid = len(sorted_signals) // 2
        first_half = sorted_signals[:mid]
        second_half = sorted_signals[mid:]
        
        if not first_half or not second_half:
            return "stable"
        
        # Calculate velocities for each half
        first_span = max((first_half[-1].get('extracted_at', datetime.utcnow()) - 
                         first_half[0].get('extracted_at', datetime.utcnow())).days, 1)
        second_span = max((second_half[-1].get('extracted_at', datetime.utcnow()) - 
                          second_half[0].get('extracted_at', datetime.utcnow())).days, 1)
        
        first_velocity = len(first_half) / first_span
        second_velocity = len(second_half) / second_span
        
        if second_velocity > first_velocity * settings.ACCELERATION_THRESHOLD:
            return "accelerating"
        elif second_velocity < first_velocity * settings.DECELERATION_THRESHOLD:
            return "decelerating"
        else:
            return "stable"
    
    def _generate_trend_name(self, signals: List[Dict]) -> str:
        """Generate concise trend name"""
        # Extract most common entities
        all_entities = []
        for signal in signals:
            entities = signal.get('entities', [])
            all_entities.extend([e[0] for e in entities if e[1] in ['ORG', 'PRODUCT', 'GPE']])
        
        if all_entities:
            most_common = Counter(all_entities).most_common(1)[0][0]
            return f"{most_common} Development Trend"
        
        # Fallback to category-based name
        categories = [s.get('category', 'unknown') for s in signals]
        dominant_category = Counter(categories).most_common(1)[0][0]
        return f"{dominant_category.title()} Trend"
    
    def _generate_description(self, signals: List[Dict]) -> str:
        """Generate trend description"""
        # Extract common themes
        all_entities = []
        temporal_refs = []
        
        for signal in signals:
            entities = signal.get('entities', [])
            all_entities.extend([e[0] for e in entities])
            temporal_refs.extend(signal.get('temporal_refs', []))
        
        # Get most common elements
        common_entities = [item for item, count in Counter(all_entities).most_common(3)]
        common_temporal = [item for item, count in Counter(temporal_refs).most_common(2)]
        
        description_parts = []
        
        if common_entities:
            description_parts.append(f"Involving: {', '.join(common_entities)}")
        
        if common_temporal:
            description_parts.append(f"Timeline: {', '.join(common_temporal)}")
        
        description_parts.append(f"Based on {len(signals)} signals")
        
        return " | ".join(description_parts)

async def analyze_trends(processed_signals: List[Dict]) -> List[Dict]:
    """Analyze processed signals to identify trends"""
    synthesizer = TrendSynthesizer()
    
    try:
        trends = synthesizer.synthesize_trends(processed_signals)
        logger.info(f"Synthesized {len(trends)} trends from {len(processed_signals)} signals")
        return trends
    except Exception as e:
        logger.error(f"Error analyzing trends: {str(e)}")
        return []