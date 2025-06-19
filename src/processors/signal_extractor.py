import re
from typing import List, Dict, Tuple, Optional
import spacy
import logging
from datetime import datetime
from config.settings import settings

logger = logging.getLogger(__name__)

class SignalExtractor:
    def __init__(self):
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            logger.warning("spaCy model not found. Run: python -m spacy download en_core_web_sm")
            self.nlp = None
        
        self.signal_patterns = settings.SIGNAL_PATTERNS
        self.categories = [
            "technological",
            "social", 
            "economic",
            "environmental",
            "political",
            "wildcard"
        ]
    
    def extract_signals(self, text: str, title: str = "") -> List[Dict]:
        """Extract potential signals from text"""
        if not text:
            return []
        
        signals = []
        combined_text = f"{title} {text}".strip()
        
        # Split into sentences for analysis
        sentences = self._split_sentences(combined_text)
        
        for sentence in sentences:
            if self._is_signal(sentence):
                signal = {
                    'content': sentence,
                    'category': self._categorize(sentence),
                    'entities': self._extract_entities(sentence),
                    'temporal_refs': self._extract_temporal_refs(sentence),
                    'confidence': self._calculate_confidence(sentence),
                    'extracted_at': datetime.utcnow()
                }
                
                # Only include signals above minimum confidence
                if signal['confidence'] >= settings.MIN_SIGNAL_CONFIDENCE:
                    signals.append(signal)
        
        return signals
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        if self.nlp:
            doc = self.nlp(text)
            return [sent.text.strip() for sent in doc.sents if len(sent.text.strip()) > 20]
        else:
            # Fallback: simple sentence splitting
            sentences = re.split(r'[.!?]+', text)
            return [s.strip() for s in sentences if len(s.strip()) > 20]
    
    def _is_signal(self, text: str) -> bool:
        """Check if text contains signal patterns"""
        text_lower = text.lower()
        
        # Check for explicit signal patterns
        for pattern in self.signal_patterns:
            if re.search(pattern, text_lower):
                return True
        
        # Check for quantitative changes that might indicate signals
        quantity_patterns = [
            r'\d+(?:\.\d+)?%\s+(?:increase|decrease|rise|fall|growth|decline)',
            r'doubled?|tripled?|halved?',
            r'\d+x\s+(?:faster|slower|more|less)',
            r'record\s+(?:high|low|level)',
        ]
        
        for pattern in quantity_patterns:
            if re.search(pattern, text_lower):
                return True
        
        return False
    
    def _categorize(self, text: str) -> str:
        """Categorize signal using keyword matching"""
        text_lower = text.lower()
        
        category_keywords = {
            "technological": [
                "ai", "artificial intelligence", "robot", "automation", "quantum",
                "blockchain", "virtual reality", "augmented reality", "5g", "iot",
                "machine learning", "neural network", "algorithm", "software",
                "computer", "digital", "cyber", "tech", "innovation"
            ],
            "environmental": [
                "climate", "environment", "carbon", "emission", "renewable",
                "solar", "wind", "battery", "electric", "sustainable",
                "pollution", "biodiversity", "ecosystem", "green", "clean"
            ],
            "social": [
                "society", "culture", "behavior", "demographic", "population",
                "education", "health", "lifestyle", "trend", "movement",
                "social media", "community", "relationship", "family"
            ],
            "economic": [
                "economy", "market", "financial", "investment", "business",
                "trade", "money", "bank", "cryptocurrency", "stock",
                "inflation", "gdp", "revenue", "profit", "cost", "price"
            ],
            "political": [
                "government", "policy", "regulation", "law", "election",
                "political", "democracy", "authority", "administration",
                "legislation", "congress", "parliament", "vote", "campaign"
            ]
        }
        
        category_scores = {}
        for category, keywords in category_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            if score > 0:
                category_scores[category] = score
        
        if category_scores:
            return max(category_scores, key=category_scores.get)
        else:
            return "wildcard"
    
    def _extract_entities(self, text: str) -> List[Tuple[str, str]]:
        """Extract named entities"""
        if not self.nlp:
            return []
        
        try:
            doc = self.nlp(text)
            entities = [(ent.text, ent.label_) for ent in doc.ents 
                       if ent.label_ in ['ORG', 'PERSON', 'GPE', 'PRODUCT', 'MONEY', 'PERCENT']]
            return entities
        except Exception as e:
            logger.error(f"Error extracting entities: {str(e)}")
            return []
    
    def _extract_temporal_refs(self, text: str) -> List[str]:
        """Extract time references"""
        temporal_patterns = [
            r"20\d{2}",  # Years like 2024, 2025
            r"next \w+ years?",  # "next 5 years"
            r"by the end of \d{4}",  # "by the end of 2025"
            r"within \d+ (?:years?|months?|decades?)",  # "within 10 years"
            r"in \d+ (?:years?|months?|decades?)",  # "in 5 years"
            r"by \d{4}",  # "by 2030"
            r"(?:early|mid|late) 20\d{2}s?",  # "early 2020s"
            r"(?:short|medium|long).?term",  # "long-term"
        ]
        
        refs = []
        for pattern in temporal_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            refs.extend(matches)
        
        return list(set(refs))  # Remove duplicates
    
    def _calculate_confidence(self, text: str) -> float:
        """Calculate signal confidence score"""
        score = 0.3  # Base score
        text_lower = text.lower()
        
        # Boost for multiple signal patterns
        pattern_matches = sum(1 for p in self.signal_patterns if re.search(p, text_lower))
        score += min(pattern_matches * 0.15, 0.4)
        
        # Boost for future time references
        temporal_refs = self._extract_temporal_refs(text)
        if temporal_refs:
            score += 0.1
        
        # Boost for credible source indicators
        credible_indicators = [
            "study", "research", "scientist", "university", "report",
            "analysis", "data", "findings", "survey", "institute"
        ]
        credible_count = sum(1 for indicator in credible_indicators if indicator in text_lower)
        score += min(credible_count * 0.05, 0.2)
        
        # Boost for quantitative information
        if re.search(r'\d+(?:\.\d+)?%', text):
            score += 0.1
        
        # Penalty for uncertainty language
        uncertainty_words = ["might", "could", "possibly", "perhaps", "maybe", "speculation"]
        uncertainty_count = sum(1 for word in uncertainty_words if word in text_lower)
        score -= min(uncertainty_count * 0.05, 0.2)
        
        # Ensure score is between 0 and 1
        return max(0.0, min(score, 1.0))

async def process_signals(raw_signals: List[Dict]) -> List[Dict]:
    """Process raw signals and extract structured signal data"""
    extractor = SignalExtractor()
    processed_signals = []
    
    for raw_signal in raw_signals:
        try:
            title = raw_signal.get('title', '')
            content = raw_signal.get('content', '')
            
            if not content and not title:
                continue
            
            signals = extractor.extract_signals(content, title)
            
            for signal in signals:
                # Add metadata from raw signal
                signal.update({
                    'source_name': raw_signal.get('source_name'),
                    'source_type': raw_signal.get('source_type'),
                    'url': raw_signal.get('url'),
                    'original_title': title,
                    'raw_content': content
                })
                processed_signals.append(signal)
        
        except Exception as e:
            logger.error(f"Error processing signal: {str(e)}")
    
    logger.info(f"Processed {len(processed_signals)} signals from {len(raw_signals)} raw items")
    return processed_signals