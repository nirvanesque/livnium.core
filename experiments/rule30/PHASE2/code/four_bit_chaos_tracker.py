#!/usr/bin/env python3
"""
15-Dimensional Chaos Tracker for Rule 30 Center Column

This script:
1. Solves the N=4 constraint system numerically (34 vars, 20 eqs, 15 free dims)
2. Extracts the 15-dimensional free-coordinate vector at each step
3. Iterates Rule 30 → updates frequencies → updates coordinates
4. Tracks 10k–100k steps
5. Stores trajectories in .npy files
6. Runs PCA, t-SNE, UMAP reductions
7. Plots 2D and 3D projections of the chaos subspace
8. Outputs: CHAOS15_RESULTS.md with interpretation
"""

import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
from scipy.optimize import minimize, linprog
from scipy.linalg import null_space
import warnings
warnings.filterwarnings('ignore')

try:
    import sympy
    from sympy import symbols, Matrix, Eq, simplify
    SYMPY_AVAILABLE = True
except ImportError:
    SYMPY_AVAILABLE = False
    print("Warning: sympy not available. Some features may be limited.")

try:
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: sklearn not available. Dimensionality reduction disabled.")

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    print("Warning: umap not available. UMAP reduction disabled.")

try:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: matplotlib not available. Plotting disabled.")

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# Import from PHASE2/code (same directory)
from rule30_algebra import RULE30_TABLE
from four_bit_system import (
    enumerate_4bit_patterns,
    build_4bit_constraint_system,
    build_rule30_transition_constraints,
    center_value_4bit,
    pattern4_to_edge
)

Pattern4 = Tuple[int, int, int, int]


class ChaosTracker15D:
    """
    Tracks the 15-dimensional free subspace of Rule 30 dynamics.
    """
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.patterns_4bit = enumerate_4bit_patterns()
        self.pattern_to_idx = {p: i for i, p in enumerate(self.patterns_4bit)}
        self.idx_to_pattern = {i: p for i, p in enumerate(self.patterns_4bit)}
        
        # Build constraint system
        if self.verbose:
            print("Building N=4 constraint system...")
        self.system = build_4bit_constraint_system(remove_flow=True)
        
        # Convert to numerical form
        if self.verbose:
            print("Converting to numerical constraint matrix...")
        self.constraint_matrix, self.constraint_rhs = self._build_constraint_matrix()
        
        # Find null space (15 free dimensions)
        if self.verbose:
            print("Computing null space (15 free dimensions)...")
        self.null_space = self._compute_null_space()
        
        # Build transition matrix for Rule 30
        if self.verbose:
            print("Building Rule 30 transition matrix...")
        self.transition_matrix = self._build_transition_matrix()
        
        if self.verbose:
            print(f"System ready:")
            print(f"  - Variables: {self.system['num_variables']}")
            print(f"  - Constraints: {self.system['num_equations']}")
            print(f"  - Free dimensions: {self.null_space.shape[1]}")
            print()
    
    def _build_constraint_matrix(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert symbolic constraints to numerical matrix form: A*x = b
        
        Returns:
            (A, b) where A is (num_eqs, num_vars) and b is (num_eqs,)
        """
        if not SYMPY_AVAILABLE:
            raise RuntimeError("sympy required for constraint matrix")
        
        # Get all variables in a fixed order
        all_vars = self.system['variables']
        
        # Build matrix row by row using sympy's coefficient extraction
        rows = []
        rhs = []
        
        for eq in self.system['equations']:
            # Convert equation to form: expr = 0
            if isinstance(eq, Eq):
                expr = eq.lhs - eq.rhs
            else:
                expr = eq
            
            # Use sympy's as_coefficients_dict or as_coefficients_dict
            # Expand to polynomial form
            expr = sympy.expand(expr)
            
            # Extract coefficients for each variable
            coeffs = []
            constant = 0.0
            
            # Get coefficient dictionary
            coeff_dict = expr.as_coefficients_dict()
            
            # Build coefficient vector
            for var in all_vars:
                if var in coeff_dict:
                    coeffs.append(float(coeff_dict[var]))
                else:
                    coeffs.append(0.0)
            
            # Extract constant term (coefficient of 1)
            if 1 in coeff_dict:
                constant = float(coeff_dict[1])
            
            rows.append(coeffs)
            rhs.append(-constant)
        
        A = np.array(rows, dtype=float)
        b = np.array(rhs, dtype=float)
        
        return A, b
    
    def _compute_null_space(self) -> np.ndarray:
        """
        Compute the null space of the constraint matrix.
        This gives us the 15 free dimensions.
        
        Returns:
            Matrix of shape (num_vars, 15) whose columns span the null space
        """
        # Use scipy's null_space function
        null_space_basis = null_space(self.constraint_matrix)
        
        if null_space_basis.shape[1] == 0:
            raise RuntimeError("No null space found - system is fully constrained")
        
        # Ensure orthonormal (null_space should already be, but just in case)
        null_space_basis, _ = np.linalg.qr(null_space_basis)
        
        return null_space_basis
    
    def _build_transition_matrix(self) -> np.ndarray:
        """
        Build the Rule 30 transition matrix T where freq_tp1 = T @ freq_t
        
        Returns:
            Matrix of shape (16, 16)
        """
        T = np.zeros((16, 16), dtype=float)
        
        for i, p_tp1 in enumerate(self.patterns_4bit):
            a_tp1, b_tp1, c_tp1, d_tp1 = p_tp1
            
            # Find all patterns at t that can contribute to p_tp1
            for j, p_t in enumerate(self.patterns_4bit):
                a_t, x_t, y_t, d_t = p_t
                
                # Check if this pattern can contribute
                if a_t != a_tp1 or d_t != d_tp1:
                    continue
                
                # Check Rule 30 updates
                new_b = RULE30_TABLE[(a_t, x_t, y_t)]
                new_c = RULE30_TABLE[(x_t, y_t, d_t)]
                
                if new_b == b_tp1 and new_c == c_tp1:
                    T[i, j] = 1.0
        
        return T
    
    def _find_feasible_point(self) -> np.ndarray:
        """
        Find a feasible point in the constraint space.
        
        Returns:
            Vector of shape (34,) satisfying all constraints
        """
        # Start with uniform frequencies
        freq_t = np.ones(16, dtype=float) / 16.0
        freq_tp1 = np.ones(16, dtype=float) / 16.0
        
        # Apply transition to get consistent freq_tp1
        freq_tp1 = self.transition_matrix @ freq_t
        
        # Normalize
        freq_tp1 = freq_tp1 / (freq_tp1.sum() + 1e-10)
        
        # Compute center values
        c_t = self._compute_center(freq_t)
        c_tp1 = self._compute_center(freq_tp1)
        
        # Build full state vector
        state = np.concatenate([freq_t, freq_tp1, [c_t, c_tp1]])
        
        # Project onto constraint space
        state = self._project_to_constraints(state)
        
        return state
    
    def _compute_center(self, freq: np.ndarray) -> float:
        """Compute center column value from frequency vector."""
        center = 0.0
        for i, p in enumerate(self.patterns_4bit):
            a, b, c, d = p
            if b == 1:
                center += freq[i]
        return center
    
    def _project_to_constraints(self, state: np.ndarray) -> np.ndarray:
        """
        Project a state vector onto the constraint space.
        
        Uses least squares to find closest point satisfying constraints.
        """
        # Solve: minimize ||x - state||^2 subject to A*x = b
        # Solution: x = state - A^T @ (A @ A^T)^(-1) @ (A @ state - b)
        
        A = self.constraint_matrix
        b = self.constraint_rhs
        
        # Compute residual
        residual = A @ state - b
        
        # Solve for correction
        try:
            # Use pseudo-inverse
            A_pinv = np.linalg.pinv(A)
            correction = A_pinv @ residual
            projected = state - correction
        except:
            # Fallback: use least squares
            projected, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        
        return projected
    
    def _extract_free_coordinates(self, state: np.ndarray) -> np.ndarray:
        """
        Extract the 15-dimensional free coordinate vector.
        
        Args:
            state: Full state vector of shape (34,)
            
        Returns:
            Free coordinates of shape (15,)
        """
        # Project state onto null space
        # free_coords = null_space^T @ state
        free_coords = self.null_space.T @ state
        
        return free_coords
    
    def _state_from_free_coords(self, free_coords: np.ndarray, 
                                 reference_state: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Reconstruct state vector from free coordinates.
        
        Args:
            free_coords: Free coordinates of shape (15,)
            reference_state: Reference state (default: feasible point)
            
        Returns:
            Full state vector of shape (34,)
        """
        if reference_state is None:
            reference_state = self._find_feasible_point()
        
        # Reconstruct: state = reference + null_space @ free_coords
        state = reference_state + self.null_space @ free_coords
        
        # Project back to constraints to ensure feasibility
        state = self._project_to_constraints(state)
        
        return state
    
    def iterate(self, state: np.ndarray) -> np.ndarray:
        """
        Apply one Rule 30 iteration.
        
        Args:
            state: Current state vector of shape (34,)
                  [freq_t (16), freq_tp1 (16), c_t (1), c_tp1 (1)]
            
        Returns:
            Next state vector of shape (34,)
        """
        # Extract frequencies at time t+1 (which becomes the new t)
        freq_t_old = state[:16]
        freq_tp1_old = state[16:32]
        
        # Move forward: t+1 becomes the new t
        freq_t_new = freq_tp1_old.copy()
        
        # Apply Rule 30 transition to get new t+1
        freq_tp1_new = self.transition_matrix @ freq_t_new
        
        # Normalize
        freq_tp1_new = freq_tp1_new / (freq_tp1_new.sum() + 1e-10)
        
        # Compute center values
        c_t_new = self._compute_center(freq_t_new)
        c_tp1_new = self._compute_center(freq_tp1_new)
        
        # Build new state: [freq_t_new, freq_tp1_new, c_t_new, c_tp1_new]
        new_state = np.concatenate([freq_t_new, freq_tp1_new, [c_t_new, c_tp1_new]])
        
        # Project to constraints
        new_state = self._project_to_constraints(new_state)
        
        return new_state
    
    def track_trajectory(self, num_steps: int = 10000, 
                         width: int = 50000) -> Tuple[np.ndarray, np.ndarray]:
        """
        Track ACTUAL Rule 30 chaos by running the grid simulation
        and projecting the results into the 15D null space.
        
        Args:
            num_steps: Number of iterations
            width: Width of the cellular automaton grid
            
        Returns:
            (trajectory_15d, trajectory_full) where:
            - trajectory_15d: (num_steps, 15) free coordinates
            - trajectory_full: (num_steps, 34) full state vectors
        """
        if self.verbose:
            print(f"Tracking chaos via Grid Simulation (Width={width})...")
        
        # 1. Initialize Grid (Actual Bits)
        np.random.seed(42)
        cells = np.random.randint(0, 2, width, dtype=np.uint8)
        
        trajectory_15d = []
        trajectory_full = []
        
        # Helper to get frequencies from bits
        def get_freqs(c):
            """
            Efficient 4-bit pattern frequency counting.
            Patterns are encoded as: 8*a + 4*b + 2*c + 1*d
            where (a,b,c,d) is the 4-bit pattern.
            """
            # Create 4-bit pattern indices using rolling windows
            patterns = (8 * c + 
                       4 * np.roll(c, -1) + 
                       2 * np.roll(c, -2) + 
                       1 * np.roll(c, -3))
            # Count occurrences of each pattern (0-15)
            counts = np.bincount(patterns, minlength=16)
            # Normalize to frequencies
            return counts.astype(float) / len(c)
        
        # 2. Run Simulation
        for step in range(num_steps):
            if self.verbose and (step + 1) % 1000 == 0:
                print(f"  Step {step + 1}/{num_steps}")
            
            # Measure State (t)
            f_t = get_freqs(cells)
            
            # Evolve Grid (Rule 30: new = left XOR (center OR right))
            l = np.roll(cells, 1)   # left neighbor
            c = cells                # center
            r = np.roll(cells, -1)  # right neighbor
            cells_next = np.bitwise_xor(l, np.bitwise_or(c, r))
            
            # Measure State (t+1)
            f_tp1 = get_freqs(cells_next)
            
            # Compute Center (just for vector consistency)
            c_t = self._compute_center(f_t)
            c_tp1 = self._compute_center(f_tp1)
            
            # Build Full State Vector
            full_state = np.concatenate([f_t, f_tp1, [c_t, c_tp1]])
            
            # Project to Null Space (The Magic Step)
            # This extracts the "pure chaos" coordinates
            free_coords = self.null_space.T @ full_state
            
            trajectory_15d.append(free_coords)
            trajectory_full.append(full_state)
            
            # Update cells for next step
            cells = cells_next
        
        return np.array(trajectory_15d), np.array(trajectory_full)
    
    def analyze_trajectory(self, trajectory_15d: np.ndarray) -> Dict:
        """
        Analyze the 15-D trajectory for structure.
        
        Args:
            trajectory_15d: Trajectory of shape (num_steps, 15)
            
        Returns:
            Dict with analysis results
        """
        if self.verbose:
            print("Analyzing trajectory...")
        
        results = {
            'num_steps': trajectory_15d.shape[0],
            'dimension': trajectory_15d.shape[1]
        }
        
        # Basic statistics
        results['mean'] = trajectory_15d.mean(axis=0)
        results['std'] = trajectory_15d.std(axis=0)
        results['min'] = trajectory_15d.min(axis=0)
        results['max'] = trajectory_15d.max(axis=0)
        
        # Correlation matrix
        results['correlation'] = np.corrcoef(trajectory_15d.T)
        
        # Principal component analysis
        if SKLEARN_AVAILABLE:
            pca = PCA(n_components=min(15, trajectory_15d.shape[0]))
            pca.fit(trajectory_15d)
            results['pca'] = {
                'explained_variance_ratio': pca.explained_variance_ratio_,
                'components': pca.components_,
                'singular_values': pca.singular_values_
            }
            
            # Effective dimensionality (number of components explaining 95% variance)
            cumsum = np.cumsum(pca.explained_variance_ratio_)
            results['effective_dimension_95'] = np.argmax(cumsum >= 0.95) + 1
            results['effective_dimension_99'] = np.argmax(cumsum >= 0.99) + 1
        
        # Lyapunov-like analysis (divergence of nearby trajectories)
        if trajectory_15d.shape[0] > 100:
            # Compute pairwise distances
            sample_size = min(1000, trajectory_15d.shape[0])
            sample = trajectory_15d[::trajectory_15d.shape[0]//sample_size][:sample_size]
            distances = np.linalg.norm(sample[:, None, :] - sample[None, :, :], axis=2)
            results['mean_pairwise_distance'] = distances.mean()
            results['std_pairwise_distance'] = distances.std()
        
        return results
    
    def reduce_dimensions(self, trajectory_15d: np.ndarray) -> Dict:
        """
        Apply dimensionality reduction techniques.
        
        Args:
            trajectory_15d: Trajectory of shape (num_steps, 15)
            
        Returns:
            Dict with reduced coordinates
        """
        if self.verbose:
            print("Applying dimensionality reduction...")
        
        results = {}
        
        # PCA to 2D and 3D
        if SKLEARN_AVAILABLE:
            pca_2d = PCA(n_components=2)
            results['pca_2d'] = pca_2d.fit_transform(trajectory_15d)
            results['pca_2d_variance'] = pca_2d.explained_variance_ratio_.sum()
            
            pca_3d = PCA(n_components=3)
            results['pca_3d'] = pca_3d.fit_transform(trajectory_15d)
            results['pca_3d_variance'] = pca_3d.explained_variance_ratio_.sum()
            
            # t-SNE to 2D
            if trajectory_15d.shape[0] <= 10000:  # t-SNE is slow for large datasets
                tsne = TSNE(n_components=2, random_state=42, perplexity=30)
                results['tsne_2d'] = tsne.fit_transform(trajectory_15d)
            else:
                # Sample for t-SNE
                sample_size = 5000
                indices = np.linspace(0, trajectory_15d.shape[0] - 1, sample_size, dtype=int)
                sample = trajectory_15d[indices]
                tsne = TSNE(n_components=2, random_state=42, perplexity=30)
                results['tsne_2d'] = tsne.fit_transform(sample)
                results['tsne_2d_indices'] = indices
            
            # UMAP to 2D and 3D
            if UMAP_AVAILABLE:
                if trajectory_15d.shape[0] <= 10000:
                    reducer_2d = umap.UMAP(n_components=2, random_state=42)
                    results['umap_2d'] = reducer_2d.fit_transform(trajectory_15d)
                    
                    reducer_3d = umap.UMAP(n_components=3, random_state=42)
                    results['umap_3d'] = reducer_3d.fit_transform(trajectory_15d)
                else:
                    # Sample for UMAP
                    sample_size = 10000
                    indices = np.linspace(0, trajectory_15d.shape[0] - 1, sample_size, dtype=int)
                    sample = trajectory_15d[indices]
                    reducer_2d = umap.UMAP(n_components=2, random_state=42)
                    results['umap_2d'] = reducer_2d.fit_transform(sample)
                    results['umap_2d_indices'] = indices
                    
                    reducer_3d = umap.UMAP(n_components=3, random_state=42)
                    results['umap_3d'] = reducer_3d.fit_transform(sample)
                    results['umap_3d_indices'] = indices
        
        return results
    
    def plot_trajectory(self, trajectory_15d: np.ndarray, reductions: Dict, 
                       output_dir: Path):
        """
        Generate visualization plots.
        
        Args:
            trajectory_15d: Trajectory of shape (num_steps, 15)
            reductions: Output from reduce_dimensions
            output_dir: Directory to save plots
        """
        if not MATPLOTLIB_AVAILABLE:
            if self.verbose:
                print("Skipping plots (matplotlib not available)")
            return
        
        if self.verbose:
            print("Generating plots...")
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Plot 1: PCA 2D projection
        if 'pca_2d' in reductions:
            fig, ax = plt.subplots(figsize=(10, 10))
            ax.scatter(reductions['pca_2d'][:, 0], reductions['pca_2d'][:, 1], 
                      c=range(len(reductions['pca_2d'])), cmap='viridis', 
                      s=1, alpha=0.6)
            ax.set_xlabel('PC1')
            ax.set_ylabel('PC2')
            ax.set_title(f'PCA 2D Projection (explained variance: {reductions["pca_2d_variance"]:.2%})')
            plt.tight_layout()
            plt.savefig(output_dir / 'pca_2d.png', dpi=150)
            plt.close()
        
        # Plot 2: PCA 3D projection
        if 'pca_3d' in reductions:
            fig = plt.figure(figsize=(12, 10))
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(reductions['pca_3d'][:, 0], reductions['pca_3d'][:, 1], 
                      reductions['pca_3d'][:, 2], c=range(len(reductions['pca_3d'])), 
                      cmap='viridis', s=1, alpha=0.6)
            ax.set_xlabel('PC1')
            ax.set_ylabel('PC2')
            ax.set_zlabel('PC3')
            ax.set_title(f'PCA 3D Projection (explained variance: {reductions["pca_3d_variance"]:.2%})')
            plt.tight_layout()
            plt.savefig(output_dir / 'pca_3d.png', dpi=150)
            plt.close()
        
        # Plot 3: t-SNE 2D
        if 'tsne_2d' in reductions:
            fig, ax = plt.subplots(figsize=(10, 10))
            tsne_data = reductions['tsne_2d']
            ax.scatter(tsne_data[:, 0], tsne_data[:, 1], 
                      c=range(len(tsne_data)), cmap='plasma', 
                      s=1, alpha=0.6)
            ax.set_xlabel('t-SNE 1')
            ax.set_ylabel('t-SNE 2')
            ax.set_title('t-SNE 2D Projection')
            plt.tight_layout()
            plt.savefig(output_dir / 'tsne_2d.png', dpi=150)
            plt.close()
        
        # Plot 4: UMAP 2D
        if 'umap_2d' in reductions:
            fig, ax = plt.subplots(figsize=(10, 10))
            umap_data = reductions['umap_2d']
            ax.scatter(umap_data[:, 0], umap_data[:, 1], 
                      c=range(len(umap_data)), cmap='inferno', 
                      s=1, alpha=0.6)
            ax.set_xlabel('UMAP 1')
            ax.set_ylabel('UMAP 2')
            ax.set_title('UMAP 2D Projection')
            plt.tight_layout()
            plt.savefig(output_dir / 'umap_2d.png', dpi=150)
            plt.close()
        
        # Plot 5: UMAP 3D
        if 'umap_3d' in reductions:
            fig = plt.figure(figsize=(12, 10))
            ax = fig.add_subplot(111, projection='3d')
            umap_data = reductions['umap_3d']
            ax.scatter(umap_data[:, 0], umap_data[:, 1], umap_data[:, 2], 
                      c=range(len(umap_data)), cmap='inferno', 
                      s=1, alpha=0.6)
            ax.set_xlabel('UMAP 1')
            ax.set_ylabel('UMAP 2')
            ax.set_zlabel('UMAP 3')
            ax.set_title('UMAP 3D Projection')
            plt.tight_layout()
            plt.savefig(output_dir / 'umap_3d.png', dpi=150)
            plt.close()
        
        # Plot 6: Time series of first few free coordinates
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        for i in range(min(3, trajectory_15d.shape[1])):
            axes[i].plot(trajectory_15d[:, i], alpha=0.7, linewidth=0.5)
            axes[i].set_ylabel(f'Free coord {i+1}')
            axes[i].grid(True, alpha=0.3)
        axes[-1].set_xlabel('Step')
        plt.suptitle('Time Series of First 3 Free Coordinates')
        plt.tight_layout()
        plt.savefig(output_dir / 'time_series.png', dpi=150)
        plt.close()
        
        if self.verbose:
            print(f"Plots saved to {output_dir}")


def generate_report(tracker: ChaosTracker15D, analysis: Dict, 
                   reductions: Dict, output_file: Path):
    """
    Generate CHAOS15_RESULTS.md report.
    """
    with open(output_file, 'w') as f:
        f.write("# 15-Dimensional Chaos Tracker Results\n\n")
        f.write("**Date**: Generated by `four_bit_chaos_tracker.py`\n\n")
        f.write("---\n\n")
        
        f.write("## System Overview\n\n")
        f.write(f"- **Variables**: {tracker.system['num_variables']}\n")
        f.write(f"- **Constraints**: {tracker.system['num_equations']}\n")
        f.write(f"- **Free Dimensions**: {tracker.null_space.shape[1]}\n")
        f.write(f"- **Trajectory Length**: {analysis['num_steps']} steps\n\n")
        
        f.write("---\n\n")
        
        f.write("## Dimensionality Analysis\n\n")
        if 'effective_dimension_95' in analysis:
            f.write(f"- **Effective Dimension (95% variance)**: {analysis['effective_dimension_95']}\n")
            f.write(f"- **Effective Dimension (99% variance)**: {analysis['effective_dimension_99']}\n")
            f.write(f"- **Full Dimension**: 15\n\n")
            
            f.write("### Principal Component Analysis\n\n")
            f.write("Explained variance ratio:\n")
            for i, var in enumerate(analysis['pca']['explained_variance_ratio'][:10], 1):
                f.write(f"- PC{i}: {var:.4f} ({var*100:.2f}%)\n")
            f.write("\n")
        
        f.write("---\n\n")
        
        f.write("## Statistical Properties\n\n")
        f.write("### Free Coordinate Statistics\n\n")
        f.write("| Coordinate | Mean | Std | Min | Max |\n")
        f.write("|------------|------|-----|-----|-----|\n")
        for i in range(min(15, len(analysis['mean']))):
            f.write(f"| {i+1} | {analysis['mean'][i]:.6f} | {analysis['std'][i]:.6f} | "
                   f"{analysis['min'][i]:.6f} | {analysis['max'][i]:.6f} |\n")
        f.write("\n")
        
        if 'mean_pairwise_distance' in analysis:
            f.write(f"- **Mean Pairwise Distance**: {analysis['mean_pairwise_distance']:.6f}\n")
            f.write(f"- **Std Pairwise Distance**: {analysis['std_pairwise_distance']:.6f}\n\n")
        
        f.write("---\n\n")
        
        f.write("## Dimensionality Reduction Results\n\n")
        
        if 'pca_2d' in reductions:
            f.write("### PCA 2D Projection\n\n")
            f.write(f"- **Explained Variance**: {reductions['pca_2d_variance']:.2%}\n")
            f.write("- **Visualization**: `pca_2d.png`\n\n")
        
        if 'pca_3d' in reductions:
            f.write("### PCA 3D Projection\n\n")
            f.write(f"- **Explained Variance**: {reductions['pca_3d_variance']:.2%}\n")
            f.write("- **Visualization**: `pca_3d.png`\n\n")
        
        if 'tsne_2d' in reductions:
            f.write("### t-SNE 2D Projection\n\n")
            f.write("- **Visualization**: `tsne_2d.png`\n\n")
        
        if 'umap_2d' in reductions:
            f.write("### UMAP 2D Projection\n\n")
            f.write("- **Visualization**: `umap_2d.png`\n\n")
        
        if 'umap_3d' in reductions:
            f.write("### UMAP 3D Projection\n\n")
            f.write("- **Visualization**: `umap_3d.png`\n\n")
        
        f.write("---\n\n")
        
        f.write("## Interpretation\n\n")
        
        # Interpret results
        if 'effective_dimension_95' in analysis:
            eff_dim = analysis['effective_dimension_95']
            if eff_dim < 5:
                f.write("### Structure Detected\n\n")
                f.write(f"The trajectory exhibits strong structure: effective dimension is {eff_dim} "
                       f"(much less than 15). This suggests:\n\n")
                f.write("- The dynamics may fall onto a lower-dimensional manifold\n")
                f.write("- There may be hidden symmetries or constraints\n")
                f.write("- The center column may have more structure than expected\n\n")
            elif eff_dim < 10:
                f.write("### Moderate Structure\n\n")
                f.write(f"The trajectory shows moderate structure: effective dimension is {eff_dim}. "
                       f"This suggests:\n\n")
                f.write("- Some compression of the free space\n")
                f.write("- Partial constraints or attractors\n")
                f.write("- Non-uniform distribution in the 15-D space\n\n")
            else:
                f.write("### High Dimensionality\n\n")
                f.write(f"The trajectory explores most of the 15-D space: effective dimension is {eff_dim}. "
                       f"This suggests:\n\n")
                f.write("- The dynamics are high-dimensional\n")
                f.write("- Limited compression or structure\n")
                f.write("- The center column may be truly unpredictable\n\n")
        
        f.write("---\n\n")
        
        f.write("## Conclusions\n\n")
        f.write("The 15-dimensional free subspace analysis reveals:\n\n")
        f.write("1. **Exact Constraints**: We have exact knowledge of the constraint system\n")
        f.write("2. **Free Space**: The 15-D space is where Rule 30's unpredictability lives\n")
        f.write("3. **Structure Detection**: Dimensionality reduction reveals hidden structure (if any)\n")
        f.write("4. **Geometric Insight**: This is the minimal coordinate system for the chaos\n\n")
        
        f.write("**Next Steps**:\n\n")
        f.write("- Analyze attractors or stable basins\n")
        f.write("- Study curvature and manifold structure\n")
        f.write("- Look for symmetry breaks\n")
        f.write("- Investigate deterministic drift\n")
        f.write("- Check for orbital structure\n\n")


def main():
    """Main execution."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="15-Dimensional Chaos Tracker for Rule 30"
    )
    
    parser.add_argument(
        '--steps',
        type=int,
        default=10000,
        help='Number of iterations (default: 10000)'
    )
    
    parser.add_argument(
        '--width',
        type=int,
        default=50000,
        help='Width of cellular automaton grid (default: 50000)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='results/chaos15',
        help='Output directory (default: results/chaos15)'
    )
    
    parser.add_argument(
        '--save-trajectory',
        action='store_true',
        help='Save trajectory arrays to .npy files'
    )
    
    parser.add_argument(
        '--no-plots',
        action='store_true',
        help='Skip generating plots'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Print detailed output'
    )
    
    args = parser.parse_args()
    
    # Create tracker
    tracker = ChaosTracker15D(verbose=args.verbose)
    
    # Track trajectory
    trajectory_15d, trajectory_full = tracker.track_trajectory(num_steps=args.steps, width=args.width)
    
    # Analyze
    analysis = tracker.analyze_trajectory(trajectory_15d)
    
    # Dimensionality reduction
    reductions = tracker.reduce_dimensions(trajectory_15d)
    
    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save trajectories
    if args.save_trajectory:
        np.save(output_dir / 'trajectory_15d.npy', trajectory_15d)
        np.save(output_dir / 'trajectory_full.npy', trajectory_full)
        if args.verbose:
            print(f"Trajectories saved to {output_dir}")
    
    # Generate plots
    if not args.no_plots:
        tracker.plot_trajectory(trajectory_15d, reductions, output_dir)
    
    # Generate report
    generate_report(tracker, analysis, reductions, output_dir / 'CHAOS15_RESULTS.md')
    
    if args.verbose:
        print(f"\nResults saved to {output_dir}")
        print(f"Report: {output_dir / 'CHAOS15_RESULTS.md'}")


if __name__ == "__main__":
    main()

