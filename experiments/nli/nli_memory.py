"""
NLI Memory: Memory-Based Learning System

Stores and retrieves patterns from SNLI training data for improved classification.
"""

import numpy as np
import json
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict

from .native_chain_encoder import NativeEncodedPair
from .inference_detectors import NLIClassifier


@dataclass
class MemoryPattern:
    """A stored pattern in memory."""
    premise: str
    hypothesis: str
    label: str
    encoded_pair: Optional[NativeEncodedPair] = None
    features: Dict[str, Any] = field(default_factory=dict)
    classification_result: Optional[Dict[str, Any]] = None
    importance: float = 1.0
    access_count: int = 0
    last_accessed: int = 0


class NLIMemory:
    """
    Memory system for NLI pattern storage and retrieval.
    
    Stores patterns by label (entailment, contradiction, neutral) and
    retrieves similar patterns for classification assistance.
    """
    
    def __init__(self, max_patterns_per_label: int = 1000):
        """
        Initialize NLI memory.
        
        Args:
            max_patterns_per_label: Maximum patterns to store per label
        """
        self.max_patterns_per_label = max_patterns_per_label
        
        # Separate memory stores for each label
        self.entailment_memory: List[MemoryPattern] = []
        self.contradiction_memory: List[MemoryPattern] = []
        self.neutral_memory: List[MemoryPattern] = []
        
        # Statistics
        self.stats = {
            'total_stored': 0,
            'entailment_count': 0,
            'contradiction_count': 0,
            'neutral_count': 0,
            'retrievals': 0
        }
    
    def store_pattern(self,
                     premise: str,
                     hypothesis: str,
                     label: str,
                     encoded_pair: Optional[NativeEncodedPair] = None,
                     classification_result: Optional[Dict[str, Any]] = None,
                     importance: float = 1.0):
        """
        Store pattern in memory.
        
        Args:
            premise: Premise sentence
            hypothesis: Hypothesis sentence
            label: Gold label (entailment/contradiction/neutral)
            encoded_pair: Optional encoded pair
            classification_result: Optional classification result
            importance: Importance weight (0-1)
        """
        # Extract features
        features = self._extract_features(premise, hypothesis, encoded_pair)
        
        # Create pattern
        pattern = MemoryPattern(
            premise=premise,
            hypothesis=hypothesis,
            label=label,
            encoded_pair=encoded_pair,
            features=features,
            classification_result=classification_result,
            importance=importance
        )
        
        # Store in appropriate memory
        memory_store = self._get_memory_store(label)
        
        if memory_store is not None:
            # Check if pattern already exists
            existing = self._find_existing(pattern, memory_store)
            if existing:
                # Update existing pattern
                existing.access_count += 1
                existing.importance = max(existing.importance, importance)
                if classification_result:
                    existing.classification_result = classification_result
            else:
                # Add new pattern
                memory_store.append(pattern)
                self.stats['total_stored'] += 1
                self.stats[f'{label}_count'] = self.stats.get(f'{label}_count', 0) + 1
                
                # Limit size
                if len(memory_store) > self.max_patterns_per_label:
                    # Remove least important/accessed
                    memory_store.sort(key=lambda p: (p.importance, p.access_count))
                    memory_store.pop(0)
    
    def retrieve_similar(self,
                        premise: str,
                        hypothesis: str,
                        encoded_pair: Optional[NativeEncodedPair] = None,
                        top_k: int = 5) -> List[MemoryPattern]:
        """
        Retrieve similar patterns from memory.
        
        Args:
            premise: Premise sentence
            hypothesis: Hypothesis sentence
            encoded_pair: Optional encoded pair
            top_k: Number of similar patterns to return
            
        Returns:
            List of similar patterns sorted by similarity
        """
        self.stats['retrievals'] += 1
        
        # Extract features
        query_features = self._extract_features(premise, hypothesis, encoded_pair)
        
        # Search all memory stores
        all_patterns = []
        all_patterns.extend(self.entailment_memory)
        all_patterns.extend(self.contradiction_memory)
        all_patterns.extend(self.neutral_memory)
        
        # Compute similarities
        similarities = []
        for pattern in all_patterns:
            similarity = self._compute_similarity(query_features, pattern.features)
            similarities.append((similarity, pattern))
        
        # Sort by similarity and return top_k
        similarities.sort(key=lambda x: x[0], reverse=True)
        
        # Update access counts
        for similarity, pattern in similarities[:top_k]:
            pattern.access_count += 1
        
        return [pattern for _, pattern in similarities[:top_k]]
    
    def get_memory_based_prediction(self,
                                   premise: str,
                                   hypothesis: str,
                                   encoded_pair: Optional[NativeEncodedPair] = None,
                                   top_k: int = 5) -> Dict[str, Any]:
        """
        Get prediction based on memory retrieval.
        
        Args:
            premise: Premise sentence
            hypothesis: Hypothesis sentence
            encoded_pair: Optional encoded pair
            top_k: Number of similar patterns to consider
            
        Returns:
            Dict with memory-based prediction
        """
        # Retrieve similar patterns
        similar = self.retrieve_similar(premise, hypothesis, encoded_pair, top_k)
        
        if len(similar) == 0:
            return {
                'label': 'neutral',
                'confidence': 0.0,
                'memory_votes': {'entailment': 0, 'contradiction': 0, 'neutral': 0}
            }
        
        # Count votes by label
        votes = defaultdict(float)
        total_similarity = 0.0
        
        for pattern in similar:
            similarity = self._compute_similarity(
                self._extract_features(premise, hypothesis, encoded_pair),
                pattern.features
            )
            votes[pattern.label] += similarity * pattern.importance
            total_similarity += similarity
        
        # Normalize votes
        if total_similarity > 0:
            for label in votes:
                votes[label] /= total_similarity
        
        # Determine label
        label = max(votes, key=votes.get) if votes else 'neutral'
        confidence = votes[label] if votes else 0.0
        
        return {
            'label': label,
            'confidence': float(confidence),
            'memory_votes': dict(votes),
            'similar_patterns': len(similar)
        }
    
    def _extract_features(self,
                         premise: str,
                         hypothesis: str,
                         encoded_pair: Optional[NativeEncodedPair] = None) -> Dict[str, Any]:
        """
        Extract features from premise-hypothesis pair.
        
        Args:
            premise: Premise sentence
            hypothesis: Hypothesis sentence
            encoded_pair: Optional encoded pair
            
        Returns:
            Feature dictionary
        """
        features = {
            'premise_length': len(premise.split()),
            'hypothesis_length': len(hypothesis.split()),
            'length_ratio': len(hypothesis.split()) / max(1, len(premise.split())),
            'premise_tokens': set(premise.lower().split()),
            'hypothesis_tokens': set(hypothesis.lower().split()),
            'token_overlap': 0.0,
            'premise_set': set(premise.lower().split()),
            'hypothesis_set': set(hypothesis.lower().split()),
        }
        
        # Compute token overlap
        if features['premise_set'] and features['hypothesis_set']:
            overlap = len(features['premise_set'] & features['hypothesis_set'])
            union = len(features['premise_set'] | features['hypothesis_set'])
            features['token_overlap'] = overlap / union if union > 0 else 0.0
        
        # Add Native Chain features if available
        if encoded_pair:
            features['premise_length'] = len(encoded_pair.premise_chain.tokens)
            features['hypothesis_length'] = len(encoded_pair.hypothesis_chain.tokens)
            features['resonance'] = encoded_pair.get_resonance()
        
        return features
    
    def _compute_similarity(self, features1: Dict[str, Any], features2: Dict[str, Any]) -> float:
        """
        Compute similarity between two feature sets.
        
        Args:
            features1: First feature set
            features2: Second feature set
            
        Returns:
            Similarity score [0, 1]
        """
        similarity = 0.0
        
        # Token overlap similarity
        if 'premise_set' in features1 and 'premise_set' in features2:
            premise_overlap = len(features1['premise_set'] & features2['premise_set'])
            premise_union = len(features1['premise_set'] | features2['premise_set'])
            premise_sim = premise_overlap / premise_union if premise_union > 0 else 0.0
            
            hypothesis_overlap = len(features1['hypothesis_set'] & features2['hypothesis_set'])
            hypothesis_union = len(features1['hypothesis_set'] | features2['hypothesis_set'])
            hypothesis_sim = hypothesis_overlap / hypothesis_union if hypothesis_union > 0 else 0.0
            
            similarity += 0.4 * premise_sim + 0.4 * hypothesis_sim
        
        # Length similarity
        if 'length_ratio' in features1 and 'length_ratio' in features2:
            length_diff = abs(features1['length_ratio'] - features2['length_ratio'])
            length_sim = max(0.0, 1.0 - length_diff)
            similarity += 0.2 * length_sim
        
        return float(np.clip(similarity, 0.0, 1.0))
    
    def _get_memory_store(self, label: str) -> Optional[List[MemoryPattern]]:
        """Get memory store for label."""
        label = label.lower()
        if label == 'entailment':
            return self.entailment_memory
        elif label == 'contradiction':
            return self.contradiction_memory
        elif label == 'neutral':
            return self.neutral_memory
        return None
    
    def _find_existing(self, pattern: MemoryPattern, memory_store: List[MemoryPattern]) -> Optional[MemoryPattern]:
        """Find existing pattern in memory store."""
        for existing in memory_store:
            if (existing.premise == pattern.premise and 
                existing.hypothesis == pattern.hypothesis):
                return existing
        return None
    
    def clear(self):
        """Clear all memory stores and reset statistics."""
        self.entailment_memory.clear()
        self.contradiction_memory.clear()
        self.neutral_memory.clear()
        self.stats = {
            'total_stored': 0,
            'entailment_count': 0,
            'contradiction_count': 0,
            'neutral_count': 0,
            'retrievals': 0
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get memory statistics."""
        return {
            **self.stats,
            'total_stored': self.stats['total_stored'],
            'entailment': len(self.entailment_memory),  # Use actual memory store size
            'contradiction': len(self.contradiction_memory),
            'neutral': len(self.neutral_memory),
            'entailment_count': self.stats.get('entailment_count', 0),
            'contradiction_count': self.stats.get('contradiction_count', 0),
            'neutral_count': self.stats.get('neutral_count', 0),
        }

