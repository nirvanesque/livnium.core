"""
Omcube System: Coupling and Classification

Combines cross-omcube coupling (basin dynamics) and omcube classifier (quantum collapse).
"""

import numpy as np
from typing import Dict, Tuple, Optional, Any
from dataclasses import dataclass

from .native_chain_encoder import NativeEncodedPair
from .inference_detectors import EntailmentDetector, ContradictionDetector, NLIClassifier
from core.classical.livnium_core_system import LivniumCoreSystem
from core.quantum.quantum_cell import QuantumCell
from core.quantum.measurement_engine import MeasurementEngine


# ============================================================================
# Coupling System: Basin Dynamics
# ============================================================================

@dataclass
class OmcubeBasin:
    """Represents a basin (attractor) for one omcube."""
    omcube_index: int  # 0=entailment, 1=contradiction, 2=neutral
    depth: float = 1.0  # Basin depth (deeper = stronger attractor)
    curvature: float = 1.0  # Basin curvature (sharper = more confident)
    energy: float = 0.0  # Current energy level
    
    def __init__(self, omcube_index: int, initial_depth: float = 1.0):
        self.omcube_index = omcube_index
        self.depth = initial_depth
        self.curvature = 1.0
        self.energy = 0.0


class CrossOmcubeCoupling:
    """
    Couples the 3 omcubes (3-class classification) so they influence each other.
    
    When one omcube is reinforced, it affects the others:
    - Correct omcube → deepens its basin
    - Wrong omcube → gets zero reinforcement (no punishment)
    """
    
    def __init__(self, initial_depth: float = 1.0):
        """Initialize coupling system."""
        self.basins = {
            0: OmcubeBasin(0, initial_depth),  # Entailment
            1: OmcubeBasin(1, initial_depth),  # Contradiction
            2: OmcubeBasin(2, initial_depth),  # Neutral
        }
        
        # Coupling strength between omcubes (3x3 matrix)
        self.coupling_matrix = np.array([
            [1.0, -0.3, -0.3],  # Entailment affects others
            [-0.3, 1.0, -0.3],  # Contradiction affects others
            [-0.3, -0.3, 1.0],  # Neutral affects others
        ])
        
        # Learning rates (reward-only)
        self.reinforcement_rate = 0.15  # Standard reinforcement rate
        self.coupling_strength = 0.5  # Cross-omcube influence
        
        # NATURAL DECAY RATES:
        # FIX: Set all decay rates EQUAL so Neutral doesn't bleed out
        # Equal decay prevents "rich get richer" - all basins decay at same rate
        self.decay_rate_entailment = 0.015
        self.decay_rate_contradiction = 0.015
        self.decay_rate_neutral = 0.015  # Was 0.025 - set to same as others
        
        # Lower the suppression threshold so Neutral stays in the game longer
        self.suppression_threshold_multiplier = 0.5  # Was 0.7 - more lenient
    
    def reinforce_basin(self, correct_omcube: int, strength: float = 1.0):
        """
        Reinforce correct omcube basin with Proportional Growth/Decay (Logistic Curve).
        
        FIX: Proportional decay creates contrast - winners stabilize, losers shrink.
        This prevents infinite growth and creates clear separation between basins.
        
        Args:
            correct_omcube: Index of correct omcube (0, 1, or 2)
            strength: Reinforcement strength (0-1)
        """
        # 1. Proportional Reinforcement (Logistic Growth)
        # Growth slows as basin approaches capacity: dH/dt = r * (1 - H/K)
        current_depth = self.basins[correct_omcube].depth
        capacity = 200.0  # Carrying capacity (stabilization point)
        growth = self.reinforcement_rate * strength * (1.0 - current_depth / capacity)
        self.basins[correct_omcube].depth += max(0.01, growth)  # Always grow a little
        
        # Update energy (lower energy = stronger attractor)
        self.basins[correct_omcube].energy = max(-5.0,
            self.basins[correct_omcube].energy - 0.1 * strength)
        
        # 2. Aggressive Proportional Decay for Losers
        # Losers lose 1% of their depth every time they lose
        # This creates exponential decay: depth * 0.99^n → shrinks to near zero
        decay_factor = 0.01  # 1% decay per loss
        for i in range(3):
            if i != correct_omcube:
                self.basins[i].depth *= (1.0 - decay_factor)
                self.basins[i].depth = max(0.1, self.basins[i].depth)
        
        # 3. DISABLE NORMALIZATION (Let the winner take all)
        # Commented out to allow basins to grow freely and diverge
        # self._normalize_basins()  # <-- DISABLED: Let winner grow!
    
    def _apply_natural_decay(self):
        """
        Apply natural decay to all basins (physics-based).
        
        FIX: Equal decay rates prevent "rich get richer" loop.
        All basins decay at the same rate, so no class bleeds out faster.
        """
        # All basins decay at the same rate (equal treatment)
        decay_rate = self.decay_rate_entailment  # Same for all now
        
        self.basins[0].depth = max(0.1, self.basins[0].depth - decay_rate)
        self.basins[1].depth = max(0.1, self.basins[1].depth - decay_rate)
        self.basins[2].depth = max(0.1, self.basins[2].depth - decay_rate)
    
    def get_basin_weights(self) -> np.ndarray:
        """
        Get weights for each omcube based on basin depths.
        
        FIX: Dampen exponential explosion to prevent "rich get richer" loop.
        Uses temperature scaling (softmax with temperature) to soften competition.
        
        Returns:
            Array of 3 weights (one per omcube)
        """
        depths = np.array([self.basins[i].depth for i in [0, 1, 2]])
        
        # Lower threshold calculation (more lenient)
        threshold = np.mean(depths) * self.suppression_threshold_multiplier
        
        # Apply suppression (less harsh penalty)
        suppressed_depths = depths.copy()
        for i in range(3):
            if depths[i] < threshold:
                suppressed_depths[i] = depths[i] * 0.2  # Less harsh: was 0.1
        
        # FIX: Dampen the exponential explosion
        # Divide by temperature so e^11 doesn't crush e^7
        # Temperature = 3.0 means: e^(11/3) ≈ 39 vs e^(8/3) ≈ 14 (ratio ~3:1, not 36:1)
        temperature = 3.0
        exp_depths = np.exp(suppressed_depths / temperature)
        
        weights = exp_depths / np.sum(exp_depths)
        
        return weights
    
    def apply_collapse_feedback(self, 
                                   correct_omcube: int,
                                   correct_omcube_unused: int,  # Keep for compatibility
                                   learning_strength: float):
        """
        Apply feedback (clean routing law).
        
        Always reinforce ground-truth basin (correct_omcube).
        All basins decay naturally (Neutral decays faster).
        
        Args:
            correct_omcube: Ground-truth omcube (0, 1, or 2) - THIS is what gets reinforced
            correct_omcube_unused: Unused (kept for compatibility)
            learning_strength: Learning strength (always 1.0)
        """
        if learning_strength > 0.0:
            # Reinforce ground-truth basin (decay applied inside)
            self.reinforce_basin(correct_omcube, learning_strength)
        else:
            # Just decay (shouldn't happen with routing law, but handle it)
            self._apply_natural_decay()
    
    def get_energy_landscape(self) -> Dict[int, float]:
        """Get energy landscape for visualization."""
        return {i: self.basins[i].energy for i in [0, 1, 2]}
    
    def get_basin_depths(self) -> Dict[int, float]:
        """Get basin depths."""
        return {i: self.basins[i].depth for i in [0, 1, 2]}
    
    def _normalize_basins(self):
        """
        Normalize basin depths to maintain stability without enforcing perfect equality.
        
        SOFT CONSTRAINT:
        Only normalize if total depth exceeds a safety ceiling (e.g. 60.0).
        This allows one basin to grow significantly larger than others (breaking symmetry)
        while preventing infinite explosion.
        """
        depths = np.array([self.basins[i].depth for i in [0, 1, 2]])
        
        # Ensure no zeros
        depths = np.maximum(depths, 0.1)
        
        current_total = np.sum(depths)
        target_ceiling = 60.0  # Increased from 30.0 to allow growth
        
        # Only scale down if we hit the ceiling
        if current_total > target_ceiling:
            scale_factor = target_ceiling / current_total
            
            for i in [0, 1, 2]:
                self.basins[i].depth = depths[i] * scale_factor
                # Update curvature
                self.basins[i].curvature = 1.0 + (self.basins[i].depth / 5.0)
        else:
            # No normalization needed - update curvature only
            for i in [0, 1, 2]:
                self.basins[i].curvature = 1.0 + (self.basins[i].depth / 5.0)


class GeometricFeedback:
    """
    Propagates collapse feedback into geometry.
    
    Updates:
    - Symbolic weights
    - Face exposure
    - Quantum amplitudes
    - Polarity fields
    """
    
    def __init__(self, encoded_pair: NativeEncodedPair):
        """Initialize geometric feedback."""
        self.encoded_pair = encoded_pair
    
    def reinforce_geometry(self, 
                          correct_omcube: int,
                          strength: float = 1.0):
        """
        Reinforce Native Chain using Relational Physics.
        
        Instead of updating symbolic weights, we update:
        - Chain entanglement strength
        - Quantum state amplitudes
        - WordOmcube masses
        
        - Entailment (0): Strengthen resonance (increase entanglement)
        - Contradiction (1): Weaken resonance (decrease entanglement)
        - Neutral (2): Weak reinforcement
        """
        premise_chain = self.encoded_pair.premise_chain
        hypothesis_chain = self.encoded_pair.hypothesis_chain
        
        learning_rate = 0.5 * strength
        
        if correct_omcube == 0:  # ENTAILMENT -> Strengthen resonance
            # Increase entanglement between matching words
            for p_cube in premise_chain.chain:
                for h_cube in hypothesis_chain.chain:
                    if p_cube.word == h_cube.word:
                        # Strengthen quantum coupling
                        p_cube.quantum_state.amplitudes *= (1.0 + learning_rate * 0.1)
                        h_cube.quantum_state.amplitudes *= (1.0 + learning_rate * 0.1)
                        p_cube.quantum_state.normalize()
                        h_cube.quantum_state.normalize()
                        
        elif correct_omcube == 1:  # CONTRADICTION -> Weaken resonance
            # Decrease entanglement (polarize)
            for p_cube in premise_chain.chain:
                for h_cube in hypothesis_chain.chain:
                    if p_cube.word == h_cube.word:
                        # Weaken quantum coupling
                        p_cube.quantum_state.amplitudes *= (1.0 - learning_rate * 0.1)
                        h_cube.quantum_state.amplitudes *= (1.0 - learning_rate * 0.1)
                        p_cube.quantum_state.normalize()
                        h_cube.quantum_state.normalize()
                        
        elif correct_omcube == 2:  # NEUTRAL -> Slight strengthening
            # Weak reinforcement - just normalize
            for cube in premise_chain.chain + hypothesis_chain.chain:
                cube.quantum_state.normalize()
        
        # CRITICAL: Commit changes to Global Memory
        # This prevents "amnesia" by saving learned states across examples
        self.encoded_pair.premise_chain.commit_learning()
        self.encoded_pair.hypothesis_chain.commit_learning()
    
    def apply_collapse_feedback(self,
                                collapsed_omcube: int,
                                correct_omcube: int,
                                learning_strength: float):
        """
        Apply reward-only collapse feedback to geometry (basin physics).
        
        Args:
            collapsed_omcube: Which omcube collapsed
            correct_omcube: Which omcube should have collapsed
            learning_strength: Learning strength (1.0 if correct, 0.0 if wrong)
        """
        if learning_strength > 0.0:
            # Correct collapse: reinforce geometry
            self.reinforce_geometry(correct_omcube, learning_strength)


# ============================================================================
# Classifier: Quantum Collapse
# ============================================================================

@dataclass
class OmcubeCollapseResult:
    """Result of 3-omcube collapse (3-class classification)."""
    label: str  # 'entailment', 'contradiction', or 'neutral'
    collapsed_omcube: int  # 0=entailment, 1=contradiction, 2=neutral
    probabilities: Dict[str, float]  # Probabilities before collapse
    confidence: float  # Probability of collapsed state
    amplitudes: np.ndarray  # Quantum amplitudes before collapse


class OmcubeNLIClassifier:
    """
    3-Omcube Collapse Classifier (3-Class Classification).
    
    Uses quantum measurement to collapse to one of three states:
    - Omcube 0: Entailment
    - Omcube 1: Contradiction
    - Omcube 2: Neutral
    """
    
    def __init__(self, encoded_pair: NativeEncodedPair):
        """
        Initialize 3-omcube classifier (3-class classification).
        
        Args:
            encoded_pair: Native Chain encoded premise-hypothesis pair
        """
        self.encoded_pair = encoded_pair
        self.entailment_detector = EntailmentDetector(encoded_pair)
        self.contradiction_detector = ContradictionDetector(encoded_pair)
        # FIX: Use NLIClassifier to get improved neutral detection
        self.nli_classifier = NLIClassifier(encoded_pair)
        self.measurement_engine = MeasurementEngine()
        
        # Cross-omcube coupling (enables communication between omcubes)
        self.coupling = CrossOmcubeCoupling(initial_depth=1.0)
        
        # Geometric feedback (propagates collapse into Native Chain)
        self.geometric_feedback = GeometricFeedback(encoded_pair)
        
        # Create 3 omcubes (quantum cells) for 3-class classification
        initial_amplitudes = np.array([1.0/np.sqrt(3), 1.0/np.sqrt(3), 1.0/np.sqrt(3)], dtype=complex)
        self.omcube_cell = QuantumCell(
            coordinates=(0, 0, 0),  # Dummy coordinates
            num_levels=3,  # 3 levels = 3 omcubes
            amplitudes=initial_amplitudes
        )
    
    def classify(self, 
                 collapse: bool = True,
                 deterministic_threshold: float = 0.1) -> OmcubeCollapseResult:
        """
        Classify using 3-omcube collapse (3-class classification).
        
        Args:
            collapse: Whether to collapse after measurement
            deterministic_threshold: Threshold for deterministic selection
            
        Returns:
            OmcubeCollapseResult with collapsed label
        """
        # Get base scores
        # FIX: Use NLIClassifier to get improved neutral detection with moderate resonance checks
        nli_result = self.nli_classifier.classify()
        entailment_score = nli_result['scores']['entailment']
        contradiction_score = nli_result['scores']['contradiction']
        neutral_score = nli_result['scores']['neutral']  # This has the improved logic!
        
        # CROSS-OMCUBE COUPLING: Get basin weights and depths (learned from previous collapses)
        basin_weights = self.coupling.get_basin_weights()
        basin_depths = self.coupling.get_basin_depths()
        
        # CLEAN 3-WAY COMPETITION: All classes compete equally
        depth_E = basin_depths.get(0, 1.0)
        depth_C = basin_depths.get(1, 1.0)
        depth_N = basin_depths.get(2, 1.0)
        
        # FIX: Dynamic Temperature (Thermodynamic Intelligence)
        # Prevents "Black Hole Problem" - system resists falling into deepest basin
        # when semantic signal is weak (ambiguous text).
        
        # 1. Calculate Semantic Confidence (How clear is the text?)
        # If scores are [0.9, 0.1], confidence is high.
        # If scores are [0.3, 0.3], confidence is low.
        semantic_confidence = max(entailment_score, contradiction_score)
        
        # 2. Calculate Dynamic Temperature (The "Anti-Gravity" Force)
        # High Confidence -> Low Temp (Freeze/Commit) - Trust the basin
        # Low Confidence -> High Temp (Explore/Resist Gravity) - Trust the signal
        # Range: 1.0 (Cold) to 10.0 (Hot)
        temperature = 1.0 + (1.0 - semantic_confidence) * 9.0
        
        # 3. Apply Basin Gravity (Modulated by Temperature)
        # If we are "Hot" (Confused), we divide the basin depth, making it shallow.
        # This prevents "Falling In" when we shouldn't.
        # When Hot (temp=8.0), depth 200 → effective_depth 25 (weak gravity)
        # When Cold (temp=1.0), depth 200 → effective_depth 200 (strong gravity)
        effective_depth_E = depth_E / temperature
        effective_depth_C = depth_C / temperature
        effective_depth_N = depth_N / temperature
        
        # 4. Boost semantic signal (keep the 5.0 multiplier)
        adj_entailment = entailment_score * 5.0
        adj_contradiction = contradiction_score * 5.0
        
        # 5. Compute Neutral Signal (FIXED: Use improved neutral score from NLIClassifier)
        # NLIClassifier already handles:
        # - Moderate resonance checks (0.4-0.65)
        # - Low lexical overlap (< 0.3)
        # - Both moderate scores (0.3-0.65)
        # - Boosts neutral to 0.85 when conditions are met
        # We just need to apply the same 5.0 multiplier for consistency
        neutral_semantic = neutral_score * 5.0
        
        # 6. Compute Final Scores
        # Score = Signal × Effective_Depth
        score_E = adj_entailment * effective_depth_E
        score_C = adj_contradiction * effective_depth_C
        score_N = neutral_semantic * effective_depth_N
        
        # Apply basin weights (cross-coupling influence)
        if len(basin_weights) >= 3:
            score_E *= (1.0 + basin_weights[0] * 0.5)
            score_C *= (1.0 + basin_weights[1] * 0.5)
            score_N *= (1.0 + basin_weights[2] * 0.5)
        
        # 3-way competition: winner is argmax(score_E, score_C, score_N)
        scores = [score_E, score_C, score_N]
        measured_omcube = scores.index(max(scores))
        
        # Probabilities proportional to scores (softmax-like)
        score_sum = sum(scores)
        prob_E = score_E / score_sum
        prob_C = score_C / score_sum
        prob_N = score_N / score_sum
        
        # Normalize probabilities
        probs = np.array([prob_E, prob_C, prob_N])
        probs = np.clip(probs, 0.0, 1.0)
        probs = probs / np.sum(probs)
        
        # FIX: Winner-Take-All for Training Stability
        # If one probability dominates, force it to 1.0 to stabilize the geometry
        # This breaks the probabilistic cycle: Clear Winner → Deterministic Collapse → 
        # Consistent Feedback → Stable Geometry → Moksha
        max_prob = np.max(probs)
        if max_prob > 0.4:  # If any class has > 40% probability (slightly better than random)
            # Hard collapse to the winner
            best_idx = np.argmax(probs)
            probs = np.zeros_like(probs)
            probs[best_idx] = 1.0
            # Update measured_omcube to match deterministic choice
            measured_omcube = best_idx
        
        # Set quantum amplitudes (square root of probabilities for Born rule)
        self.omcube_cell.amplitudes = np.sqrt(probs).astype(complex)
        self.omcube_cell.normalize()
        
        # Collapse quantum state to measured omcube
        if collapse:
            self.omcube_cell.amplitudes = np.zeros(3, dtype=complex)
            self.omcube_cell.amplitudes[measured_omcube] = 1.0 + 0j
        
        # Map to label
        label_map = {0: 'entailment', 1: 'contradiction', 2: 'neutral'}
        label = label_map[measured_omcube]
        
        result = OmcubeCollapseResult(
            label=label,
            collapsed_omcube=int(measured_omcube),
            probabilities={
                'entailment': float(probs[0]),
                'contradiction': float(probs[1]),
                'neutral': float(probs[2])
            },
            confidence=float(probs[measured_omcube]),
            amplitudes=self.omcube_cell.amplitudes.copy()
        )
        
        # Store coupling and feedback references for learning
        result._coupling = self.coupling
        result._geometric_feedback = self.geometric_feedback
        
        return result
    
    def apply_learning_feedback(self,
                                   result: OmcubeCollapseResult,
                                   correct_omcube: int,
                                   learning_strength: float):
        """
        Apply learning feedback (clean routing law).
        
        Always reinforce ground-truth basin (correct_omcube).
        This ensures all classes grow independently.
        
        Args:
            result: Collapse result (used for statistics)
            correct_omcube: Ground-truth omcube (0, 1, or 2) - THIS is what gets reinforced
            learning_strength: Learning strength (always 1.0)
        """
        if learning_strength > 0.0:
            # Reinforce ground-truth basin
            self.coupling.apply_collapse_feedback(
                correct_omcube, correct_omcube, learning_strength
            )
            
            # Update geometry
            self.geometric_feedback.apply_collapse_feedback(
                correct_omcube, correct_omcube, learning_strength
            )
            
            # Update quantum amplitudes
            self.omcube_cell.amplitudes[correct_omcube] *= (1.0 + learning_strength * 0.1)
            self.omcube_cell.normalize()
    
    def get_superposition_state(self) -> Dict[str, Any]:
        """
        Get current superposition state (before collapse).
        
        Returns:
            Dict with amplitudes and probabilities
        """
        probs = self.omcube_cell.get_probabilities()
        return {
            'amplitudes': self.omcube_cell.amplitudes.copy(),
            'probabilities': {
                'entailment': float(probs[0]),
                'contradiction': float(probs[1]),
                'neutral': float(probs[2])
            },
            'entropy': float(-np.sum(probs * np.log(probs + 1e-10)))  # Shannon entropy
        }
    
    def collapse_to(self, omcube_index: int) -> OmcubeCollapseResult:
        """
        Manually collapse to a specific omcube.
        
        Args:
            omcube_index: 0=entailment, 1=contradiction, 2=neutral
            
        Returns:
            OmcubeCollapseResult
        """
        if omcube_index not in [0, 1, 2]:
            raise ValueError("omcube_index must be 0, 1, or 2")
        
        probs = self.omcube_cell.get_probabilities()
        self.omcube_cell.amplitudes = np.zeros(3, dtype=complex)
        self.omcube_cell.amplitudes[omcube_index] = 1.0 + 0j
        
        label_map = {0: 'entailment', 1: 'contradiction', 2: 'neutral'}
        
        return OmcubeCollapseResult(
            label=label_map[omcube_index],
            collapsed_omcube=omcube_index,
            probabilities={
                'entailment': float(probs[0]),
                'contradiction': float(probs[1]),
                'neutral': float(probs[2])
            },
            confidence=float(probs[omcube_index]),
            amplitudes=self.omcube_cell.amplitudes.copy()
        )

