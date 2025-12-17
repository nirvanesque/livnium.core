"""
Contradiction Reconciler: Truth Reconciliation Loop

Implements Phase 1 of Domain Maturity: Contradiction Collapse.
Given a set of conflicting claims, this module runs a reconciliation loop that:
1. Minimizes global tension by clustering consistent claims.
2. Push contradictory claims into separate narrative basins.
3. Provides a reconciled "Truth Map" of the document.
"""

import torch
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass

from livnium.kernel.physics import alignment, divergence, tension
from livnium.engine.ops_torch import TorchOps
from livnium.domains.document.encoder import Claim, DocumentEncoder
from livnium.engine.config.defaults import RECONCILE_NOISE, RECONCILE_CONSENSUS

@dataclass
class ReconciliationResult:
    """Result of the contradiction reconciliation loop."""
    clusters: List[List[str]]  # List of groups of claim IDs that are consistent
    contradictions: List[Tuple[str, str]]  # Pairs of claim IDs that contradict
    narrative_centroids: torch.Tensor  # Centroids of the resulting basins [K, dim]
    global_tension_history: List[float]
    final_claims_map: Dict[str, torch.Tensor]  # Final state of each claim

class ContradictionReconciler:
    """
    Reconciles conflicting claims using Livnium mutual exclusion/attraction physics.
    """
    
    def __init__(
        self,
        encoder: DocumentEncoder,
        iterations: int = 10,
        attraction_strength: float = 0.2,
        repulsion_strength: float = 0.4,
        convergence_threshold: float = 1e-4
    ):
        self.encoder = encoder
        self.iterations = iterations
        self.attraction_strength = attraction_strength
        self.repulsion_strength = repulsion_strength
        self.convergence_threshold = convergence_threshold
        self.ops = TorchOps()

    def reconcile(
        self,
        claims: List[Claim],
        device: torch.device = torch.device("cpu")
    ) -> ReconciliationResult:
        """
        Runs the reconciliation loop on a set of claims.
        """
        if not claims:
            raise ValueError("No claims provided for reconciliation")

        # 1. Encode all claims [N, dim]
        claim_ids = [c.claim_id for c in claims]
        claim_vectors = torch.stack([self.encoder.encode_claim(c) for c in claims]).to(device)
        num_claims = len(claim_ids)
        dim = claim_vectors.shape[1]

        # Initialize states (h) as normalized encoders + noise for symmetry breaking
        h = F.normalize(claim_vectors, dim=-1)
        h = h + RECONCILE_NOISE * torch.randn_like(h)
        h = F.normalize(h, dim=-1)

        tension_history = []
        
        # 2. Iterative Reconciliation Loop
        for step in range(self.iterations):
            state_prev = h.clone()
            
            # Compute Mutual Tension Matrix
            # [N, N] alignments
            align_matrix = torch.mm(h, h.t()) # Simplified dot product since h is normalized
            
            # Livnium Law: divergence = pivot - alignment
            from livnium.kernel.constants import DIVERGENCE_PIVOT
            div_matrix = DIVERGENCE_PIVOT - align_matrix
            
            # Global Tension for monitoring
            current_tension = float(torch.abs(div_matrix).mean().item())
            tension_history.append(current_tension)
            
            # Compute Force Field
            # For each claim i, compute net force from all j != i
            # Force_ij = -strength * divergence * direction
            forces = torch.zeros_like(h)
            
            for i in range(num_claims):
                for j in range(num_claims):
                    if i == j: continue
                    
                    div = div_matrix[i, j]
                    # Direction: h[i] - h[j]
                    direction = F.normalize(h[i] - h[j], dim=0)
                    
                    # If div < 0 (Consistency/Entailment) -> Pull together
                    # If div > 0 (Contradiction) -> Push apart
                    # Use different strengths for attraction/repulsion to favor resolution
                    strength = self.attraction_strength if div < 0 else self.repulsion_strength
                    
                    force = -strength * div * direction
                    forces[i] = forces[i] + force
            
            # Apply update
            h = h + forces
            h = F.normalize(h, dim=-1)
            
            # Check convergence
            diff = torch.norm(h - state_prev)
            if diff < self.convergence_threshold:
                break

        # 3. Post-processing: Extract Basins (Clusters)
        # Group claims that collapsed into the same basin (high alignment)
        final_align = torch.mm(h, h.t())
        clusters = []
        visited = set()
        
        for i in range(num_claims):
            if i in visited: continue
            
            # Find all j that align with i > threshold
            # We use a threshold significantly higher than DIVERGENCE_PIVOT
            # because they should have collapsed together
            group_indices = torch.where(final_align[i] > RECONCILE_CONSENSUS)[0].tolist()
            cluster = [claim_ids[idx] for idx in group_indices]
            clusters.append(cluster)
            visited.update(group_indices)

        # 4. Extract Contradictions
        # Pairs of IDs where final alignment is very low / divergence is high
        contradictions = []
        for i in range(num_claims):
            for j in range(i + 1, num_claims):
                if final_align[i, j] < DIVERGENCE_PIVOT - 0.2: # Strong contradiction
                    contradictions.append((claim_ids[i], claim_ids[j]))

        # Calculate centroids for each cluster
        centroids = []
        for cluster_ids in clusters:
            indices = [claim_ids.index(cid) for cid in cluster_ids]
            centroid = h[indices].mean(dim=0)
            centroids.append(F.normalize(centroid, dim=0))
        
        final_centroids = torch.stack(centroids) if centroids else torch.empty((0, dim))

        return ReconciliationResult(
            clusters=clusters,
            contradictions=contradictions,
            narrative_centroids=final_centroids,
            global_tension_history=tension_history,
            final_claims_map={cid: h[i] for i, cid in enumerate(claim_ids)}
        )
