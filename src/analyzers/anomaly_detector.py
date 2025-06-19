import numpy as np
from typing import List, Dict, Tuple
from sklearn.ensemble import IsolationForest
from collections import Counter
import re
import logging
from datetime import datetime
from config.settings import settings

logger = logging.getLogger(__name__)

class AnomalyDetector:
    def __init__(self):
        self.statistical_model = IsolationForest(contamination=0.1, random_state=42)
        self.oddity_patterns = settings.ANOMALY_PATTERNS
        self.trained = False
        
    def detect_anomalies(self, signals: List[Dict]) -> List[Dict]:
        """Detect anomalous signals"""
        if not signals:
            return []
        
        anomalies = []
        
        # Train statistical model if we have enough data
        if len(signals) >= 10 and not self.trained:
            self._train_statistical_model(signals)
        
        logger.info(f"Analyzing {len(signals)} signals for anomalies")
        
        for signal in signals:
            try:
                anomaly_scores = self._analyze_signal(signal, signals)
                
                if self._is_anomaly(anomaly_scores):
                    signal_copy = signal.copy()
                    signal_copy['anomaly_scores'] = anomaly_scores
                    signal_copy['anomaly_type'] = self._classify_anomaly_type(anomaly_scores)
                    signal_copy['overall_anomaly_score'] = self._calculate_overall_score(anomaly_scores)
                    signal_copy['detected_at'] = datetime.utcnow()
                    anomalies.append(signal_copy)
                    
            except Exception as e:
                logger.error(f"Error analyzing signal for anomalies: {str(e)}")
        
        logger.info(f"Detected {len(anomalies)} anomalous signals")
        return anomalies
    
    def _analyze_signal(self, signal: Dict, all_signals: List[Dict]) -> Dict:
        """Analyze a signal for various types of anomalies"""
        return {
            'oddity': self._calculate_oddity_score(signal),
            'statistical': self._check_statistical_anomaly(signal),
            'contradiction': self._check_contradictions(signal, all_signals),
            'temporal': self._check_temporal_anomaly(signal),
            'quantitative': self._check_quantitative_anomaly(signal),
            'source_deviation': self._check_source_deviation(signal, all_signals)
        }
    
    def _calculate_oddity_score(self, signal: Dict) -> float:
        """Calculate how 'odd' a signal is based on language patterns"""
        content = signal.get('content', '').lower()
        score = 0.0
        
        # Check for oddity patterns
        pattern_matches = sum(1 for p in self.oddity_patterns if re.search(p, content))
        score += min(pattern_matches * 0.25, 0.7)
        
        # Check for unusual word combinations
        unusual_combos = [
            ("quantum", "biology"),
            ("ai", "consciousness"),
            ("reverse", "aging"),
            ("room temperature", "superconductor"),
            ("time", "travel"),
            ("teleportation", "achieved"),
            ("perpetual", "motion"),
            ("faster than light", "communication"),
            ("mind", "uploading"),
            ("warp", "drive")
        ]
        
        for combo in unusual_combos:
            if all(word in content for word in combo):
                score += 0.4
                break
        
        # Check for extreme language
        extreme_words = [
            "revolutionary", "breakthrough", "unprecedented", "impossible",
            "miraculous", "shocking", "stunning", "unbelievable"
        ]
        extreme_count = sum(1 for word in extreme_words if word in content)
        score += min(extreme_count * 0.1, 0.3)
        
        return min(score, 1.0)
    
    def _check_statistical_anomaly(self, signal: Dict) -> bool:
        """Check for statistical anomalies in quantitative claims"""
        content = signal.get('content', '')
        
        # Extract percentage claims
        percentages = re.findall(r'(\d+(?:\.\d+)?)\s*%', content)
        for pct_str in percentages:
            pct = float(pct_str)
            # Flag extreme percentages
            if pct > 1000 or (pct > 200 and "increase" in content.lower()):
                return True
        
        # Extract large numbers that might be suspicious
        large_numbers = re.findall(r'\b(\d{8,})\b', content)  # 8+ digit numbers
        if large_numbers:
            return True
        
        # Check for impossible timeframes
        impossible_patterns = [
            r'in \d+ (?:seconds?|minutes?)',  # Very short timeframes for major changes
            r'overnight',
            r'instantly',
            r'immediately'
        ]
        
        for pattern in impossible_patterns:
            if re.search(pattern, content.lower()):
                return True
        
        return False
    
    def _check_contradictions(self, signal: Dict, all_signals: List[Dict]) -> float:
        """Check if signal contradicts other signals"""
        signal_entities = set()
        for entity, _ in signal.get('entities', []):
            signal_entities.add(entity.lower())
        
        if not signal_entities:
            return 0.0
        
        contradiction_score = 0.0
        signal_content = signal.get('content', '').lower()
        
        # Keywords that suggest contradiction
        contradiction_keywords = [
            "however", "contrary", "despite", "opposite", "reverses",
            "debunks", "disproves", "contradicts", "challenges"
        ]
        
        has_contradiction_language = any(
            keyword in signal_content for keyword in contradiction_keywords
        )
        
        if not has_contradiction_language:
            return 0.0
        
        # Look for signals with shared entities that might be contradicted
        related_signals = 0
        for other_signal in all_signals[:100]:  # Limit for performance
            if other_signal.get('id') == signal.get('id'):
                continue
            
            other_entities = set()
            for entity, _ in other_signal.get('entities', []):
                other_entities.add(entity.lower())
            
            # Check for shared entities
            if signal_entities & other_entities:
                related_signals += 1
        
        if related_signals > 0:
            contradiction_score = min(related_signals / 20.0, 1.0)
        
        return contradiction_score
    
    def _check_temporal_anomaly(self, signal: Dict) -> float:
        """Check for temporal anomalies (things happening too fast/slow)"""
        content = signal.get('content', '').lower()
        temporal_refs = signal.get('temporal_refs', [])
        
        score = 0.0
        
        # Check for unrealistic timeframes
        unrealistic_patterns = [
            (r'solved? in \d+ days?', 0.5),  # Major problems solved in days
            (r'developed? in \d+ weeks?', 0.3),  # Complex tech in weeks
            (r'cure for .* in \d+ months?', 0.6),  # Medical cures very quickly
            (r'colonize .* by 20\d{2}', 0.4),  # Space colonization soon
        ]
        
        for pattern, weight in unrealistic_patterns:
            if re.search(pattern, content):
                score += weight
        
        # Check temporal references for consistency
        future_years = []
        for ref in temporal_refs:
            year_match = re.search(r'20(\d{2})', ref)
            if year_match:
                year = int(year_match.group(0))
                future_years.append(year)
        
        current_year = datetime.now().year
        for year in future_years:
            if year < current_year:  # Past year mentioned as future
                score += 0.3
            elif year > current_year + 50:  # Very far future
                score += 0.2
        
        return min(score, 1.0)
    
    def _check_quantitative_anomaly(self, signal: Dict) -> float:
        """Check for quantitative anomalies"""
        content = signal.get('content', '')
        score = 0.0
        
        # Check for impossible efficiency claims
        efficiency_matches = re.findall(r'(\d+(?:\.\d+)?)\s*%\s*efficient', content.lower())
        for eff in efficiency_matches:
            if float(eff) > 100:
                score += 0.5
        
        # Check for impossible speeds/sizes
        impossible_patterns = [
            (r'\d+x faster than light', 0.8),
            (r'\d{3,}% more efficient', 0.6),
            (r'zero energy', 0.4),
            (r'infinite .* capacity', 0.7),
        ]
        
        for pattern, weight in impossible_patterns:
            if re.search(pattern, content.lower()):
                score += weight
        
        return min(score, 1.0)
    
    def _check_source_deviation(self, signal: Dict, all_signals: List[Dict]) -> float:
        """Check if signal deviates significantly from source's typical content"""
        source_name = signal.get('source_name')
        if not source_name:
            return 0.0
        
        # Get other signals from same source
        source_signals = [
            s for s in all_signals[:200]  # Limit for performance
            if s.get('source_name') == source_name and s.get('id') != signal.get('id')
        ]
        
        if len(source_signals) < 5:
            return 0.0
        
        # Compare categories
        signal_category = signal.get('category')
        source_categories = [s.get('category') for s in source_signals]
        category_counter = Counter(source_categories)
        
        if signal_category not in category_counter:
            return 0.3  # New category for this source
        
        # Compare confidence scores
        signal_confidence = signal.get('confidence', 0.5)
        source_confidences = [s.get('confidence', 0.5) for s in source_signals]
        avg_confidence = np.mean(source_confidences)
        std_confidence = np.std(source_confidences)
        
        if std_confidence > 0:
            z_score = abs(signal_confidence - avg_confidence) / std_confidence
            if z_score > 2:  # More than 2 standard deviations
                return min(z_score / 5.0, 0.5)
        
        return 0.0
    
    def _train_statistical_model(self, signals: List[Dict]):
        """Train the statistical anomaly detection model"""
        try:
            # Create feature vectors from signals
            features = []
            for signal in signals:
                feature_vector = self._extract_features(signal)
                features.append(feature_vector)
            
            if len(features) >= 10:
                X = np.array(features)
                self.statistical_model.fit(X)
                self.trained = True
                logger.info("Statistical anomaly model trained successfully")
        except Exception as e:
            logger.error(f"Error training statistical model: {str(e)}")
    
    def _extract_features(self, signal: Dict) -> List[float]:
        """Extract numerical features from signal for statistical analysis"""
        content = signal.get('content', '')
        
        features = [
            len(content),  # Content length
            signal.get('confidence', 0.5),  # Confidence score
            len(signal.get('entities', [])),  # Number of entities
            len(signal.get('temporal_refs', [])),  # Number of temporal references
            len(re.findall(r'\d+', content)),  # Number of digits
            len(re.findall(r'%', content)),  # Number of percentages
            content.count('!'),  # Exclamation marks
            content.count('?'),  # Question marks
        ]
        
        return features
    
    def _is_anomaly(self, anomaly_scores: Dict) -> bool:
        """Determine if signal is anomalous based on scores"""
        overall_score = self._calculate_overall_score(anomaly_scores)
        return overall_score > 0.6
    
    def _calculate_overall_score(self, anomaly_scores: Dict) -> float:
        """Calculate overall anomaly score"""
        weights = {
            'oddity': 0.25,
            'statistical': 0.20,
            'contradiction': 0.20,
            'temporal': 0.15,
            'quantitative': 0.15,
            'source_deviation': 0.05
        }
        
        total_score = 0.0
        for score_type, weight in weights.items():
            score = anomaly_scores.get(score_type, 0.0)
            if isinstance(score, bool):
                score = 1.0 if score else 0.0
            total_score += score * weight
        
        return min(total_score, 1.0)
    
    def _classify_anomaly_type(self, anomaly_scores: Dict) -> str:
        """Classify the type of anomaly"""
        max_score = 0.0
        anomaly_type = "general"
        
        for score_name, score in anomaly_scores.items():
            if isinstance(score, bool):
                score = 1.0 if score else 0.0
            
            if score > max_score:
                max_score = score
                anomaly_type = score_name
        
        # Map to human-readable types
        type_mapping = {
            'oddity': 'linguistic_oddity',
            'statistical': 'statistical_outlier',
            'contradiction': 'contradictory_claim',
            'temporal': 'temporal_impossibility',
            'quantitative': 'quantitative_impossibility',
            'source_deviation': 'source_deviation'
        }
        
        return type_mapping.get(anomaly_type, 'general_anomaly')

async def detect_anomalies(processed_signals: List[Dict]) -> List[Dict]:
    """Detect anomalies in processed signals"""
    detector = AnomalyDetector()
    
    try:
        anomalies = detector.detect_anomalies(processed_signals)
        logger.info(f"Detected {len(anomalies)} anomalies from {len(processed_signals)} signals")
        return anomalies
    except Exception as e:
        logger.error(f"Error detecting anomalies: {str(e)}")
        return []