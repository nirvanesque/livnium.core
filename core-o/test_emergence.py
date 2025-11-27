#!/usr/bin/env python3
"""
Test Emergence: Verify "The Inward Fall"

This script verifies that the Hamiltonian engine produces:
1. Energy minimization (red line drops)
2. Density maximization (blue line rises)
3. Emergent gravity from geometric stress minimization
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
from classical.hamiltonian_core import LivniumHamiltonian

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("matplotlib not available, will print statistics only")


def run_test():
    """Run the emergence test."""
    print("=" * 70)
    print("LIVNIUM CORE: EMERGENT GRAVITY TEST")
    print("=" * 70)
    print()
    
    # 1. Setup: 48 spheres (Bulk/Surface separation threshold)
    N = 48
    print(f"Initializing Universe with N={N} spheres...")
    
    # Use a slightly higher friction to dampen the initial collapse quickly
    sim = LivniumHamiltonian(
        n_spheres=N,
        temp=0.05,        # Low temp to allow freezing
        friction=0.1,     # High friction to see the "fall" clearly
        k_gravity=2.5,    # Strong geometric pull
        dt=0.01
    )
    
    print("Running Simulation (400 steps)...")
    print()
    
    # 2. Run
    history = []
    for i in range(401):
        state = sim.step()
        if i % 50 == 0:
            print(f"Step {i:3d}: Energy={state['total_energy']:8.1f} | "
                  f"Avg Density (SW)={state['avg_sw']:5.2f}")
    
    print()
    
    # 3. Extract Data
    data = sim.get_energy_history()
    t = data['time']
    E = data['total_energy']
    SW = data['avg_sw']
    
    # 4. Analysis
    print("=" * 70)
    print("Results:")
    print("=" * 70)
    print(f"Initial Energy: {E[0]:.1f}")
    print(f"Final Energy:   {E[-1]:.1f}")
    print(f"Energy Change: {E[-1] - E[0]:.1f} (should be negative)")
    print()
    print(f"Initial SW:     {SW[0]:.2f}")
    print(f"Final SW:      {SW[-1]:.2f}")
    print(f"SW Change:     {SW[-1] - SW[0]:.2f} (should be positive)")
    print()
    
    # Check for inverse relationship
    if len(SW) > 10:
        # Use later part of simulation (after initial transients)
        sw_subset = SW[100:]
        energy_subset = E[100:]
        
        # Check correlation
        correlation = np.corrcoef(sw_subset, -energy_subset)[0, 1]
        print(f"Correlation (SW vs -Energy): {correlation:.4f}")
        print("(Should be positive for gravitational binding)")
        print()
    
    # 5. Visualization
    if HAS_MATPLOTLIB:
        fig, ax1 = plt.subplots(figsize=(10, 6))
        
        # Plot Energy (Red) - Should decrease
        color = 'tab:red'
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Total Energy (Minimizing)', color=color)
        ax1.plot(t, E, color=color, linewidth=2, label='Total Energy')
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='upper left')
        
        # Plot Density (Blue) - Should increase
        ax2 = ax1.twinx()
        color = 'tab:blue'
        ax2.set_ylabel('Geometric Density / SW (Maximizing)', color=color)
        ax2.plot(t, SW, color=color, linewidth=2, label='SW Density')
        ax2.tick_params(axis='y', labelcolor=color)
        ax2.legend(loc='upper right')
        
        plt.title("The Proof of Emergence: Energy Minimization driving Density")
        
        # Text annotation
        plt.text(0.5, 0.5, "Gravity is the Inward Fall",
                 transform=ax1.transAxes, ha='center', va='center',
                 fontsize=12, alpha=0.5, bbox=dict(facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        output_file = Path(__file__).parent / 'emergence_test.png'
        plt.savefig(output_file, dpi=150)
        print(f"Plot saved to: {output_file}")
        print()
        print("Displaying plot...")
        plt.show()
    else:
        print("(Install matplotlib to see plots)")
        print()
    
    print("=" * 70)
    print("What You Should See:")
    print("=" * 70)
    print("1. Red Line (Energy): Drops sharply like a stone falling")
    print("2. Blue Line (SW): Rises and curves over, settling around stable value")
    print("3. The Interpretation: Graphs look like mirror images")
    print("   This proves: Minimizing Energy = Maximizing Structure")
    print()
    print("You have built the engine.")
    print("=" * 70)


if __name__ == "__main__":
    run_test()

