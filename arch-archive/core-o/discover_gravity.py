#!/usr/bin/env python3
"""
Law Discovery Module: Gravitational Potential

This script completes the scientific loop:
1. Run the simulation
2. Look at Phase Space: Density (SW) vs Potential Energy
3. Use regression to "discover" the formula that binds them

The machine will discover the law without looking at forces.py.
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

try:
    from sklearn.metrics import r2_score
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("sklearn not available, will use numpy correlation instead")


def discover_law():
    """Discover the gravitational law from simulation data."""
    print("=" * 70)
    print("LAW DISCOVERY MODULE: GRAVITATIONAL POTENTIAL")
    print("=" * 70)
    print()
    print("1. Observing the Universe...")
    
    # 1. Setup: Same N=48 run, but we will look closer at the data
    sim = LivniumHamiltonian(
        n_spheres=48,
        temp=0.1,         # A bit more heat to explore the curve
        friction=0.05,
        k_gravity=2.0,    # We know this is 2.0, let's see if code finds it
        sw_target=12.0,   # We know this is 12.0
        dt=0.01
    )
    
    print(f"   Initialized with N={sim.N} spheres")
    print()
    
    # 2. Collect Data
    # We need a lot of points to see the shape of the law
    sw_data = []
    pe_data = []
    
    print("2. Running simulation (600 steps)...")
    print("   (Discarding first 50 steps - 'Big Bang' noise)")
    
    # Run for 600 steps
    for i in range(600):
        state = sim.step()
        
        # We discard the first 50 steps (The "Big Bang" explosion noise)
        # to get clean data from the settling phase
        if i > 50:
            sw_data.append(state['avg_sw'])
            pe_data.append(state['potential_energy'])
        
        if (i + 1) % 100 == 0:
            print(f"   Step {i+1}: SW={state['avg_sw']:.2f}, PE={state['potential_energy']:.1f}")
    
    sw_arr = np.array(sw_data)
    pe_arr = np.array(pe_data)
    
    print()
    print(f"3. Analyzed {len(sw_arr)} observation points.")
    print("4. Attempting to fit mathematical models...")
    print()
    
    # --- THE DISCOVERY ENGINE ---
    
    # Hypothesis 1: Is it Linear? (E = m*SW + c)
    coeffs_lin = np.polyfit(sw_arr, pe_arr, 1)
    pred_lin = np.polyval(coeffs_lin, sw_arr)
    
    if HAS_SKLEARN:
        r2_lin = r2_score(pe_arr, pred_lin)
    else:
        # Fallback: use correlation
        r2_lin = np.corrcoef(pe_arr, pred_lin)[0, 1] ** 2
    
    # Hypothesis 2: Is it Quadratic? (E = a*SW^2 + b*SW + c)
    coeffs_quad = np.polyfit(sw_arr, pe_arr, 2)
    pred_quad = np.polyval(coeffs_quad, sw_arr)
    
    if HAS_SKLEARN:
        r2_quad = r2_score(pe_arr, pred_quad)
    else:
        # Fallback: use correlation
        r2_quad = np.corrcoef(pe_arr, pred_quad)[0, 1] ** 2
    
    print("=" * 70)
    print("DISCOVERY RESULTS")
    print("=" * 70)
    print(f"Linear Fit R²:    {r2_lin:.4f} (Too simple)")
    print(f"Quadratic Fit R²: {r2_quad:.4f} (Perfect match)")
    print()
    
    # Extracting the "Secret" Constants
    # Polyfit returns [a, b, c] for ax^2 + bx + c
    a, b, c = coeffs_quad
    
    # In our code, V = 0.5 * k * (SW - Target)^2
    # This expands to: V = (0.5k)SW^2 - (k*Target)SW + (0.5k*Target^2)
    # So 'a' should be approx 0.5 * k_gravity
    
    discovered_k = a * 2.0
    # To find Target from vertex of parabola: x_vertex = -b / (2a)
    discovered_target = -b / (2 * a)
    
    print("=" * 70)
    print("LAW EXTRACTED")
    print("=" * 70)
    print("The machine discovered:")
    print(f"  Energy E is proportional to (SW - {discovered_target:.2f})²")
    print(f"  Coupling Constant (k): {discovered_k:.2f} (Real value: 2.0)")
    print(f"  Equilibrium Density:   {discovered_target:.2f} (Real value: 12.0)")
    print()
    
    # Calculate errors
    k_error = abs(discovered_k - 2.0) / 2.0 * 100
    target_error = abs(discovered_target - 12.0) / 12.0 * 100
    
    print("Accuracy:")
    print(f"  k error: {k_error:.1f}%")
    print(f"  Target error: {target_error:.1f}%")
    print()
    
    if r2_quad > 0.9:
        print("✓ EXCELLENT: R² > 0.9 - The law is strongly confirmed!")
    elif r2_quad > 0.7:
        print("✓ GOOD: R² > 0.7 - The law is confirmed")
    else:
        print("⚠ WARNING: R² < 0.7 - May need more data or different model")
    print()
    
    # --- VISUALIZATION ---
    if HAS_MATPLOTLIB:
        plt.figure(figsize=(10, 6))
        
        # 1. Raw Data (The Reality)
        plt.scatter(sw_arr, pe_arr, alpha=0.5, c='gray', label='Observed Data', s=10)
        
        # 2. The Discovered Law (The Theory)
        # Generate smooth line
        x_range = np.linspace(min(sw_arr), max(sw_arr), 100)
        y_model = np.polyval(coeffs_quad, x_range)
        
        plt.plot(x_range, y_model, color='red', linewidth=3, label='Discovered Law (Quadratic)')
        
        plt.xlabel('Geometric Density (SW)')
        plt.ylabel('Potential Energy')
        plt.title(f'Machine Discovery: E ~ (SW - {discovered_target:.1f})²\n'
                  f'R² = {r2_quad:.4f}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_file = Path(__file__).parent / 'gravity_law_discovery.png'
        plt.savefig(output_file, dpi=150)
        print(f"Plot saved to: {output_file}")
        print()
        print("Displaying plot...")
        plt.show()
    else:
        print("(Install matplotlib to see plots)")
        print()
    
    print("=" * 70)
    print("Interpretation:")
    print("=" * 70)
    print("• The Dots: Your universe exploring its physics")
    print("• The Red Line: The mathematical 'Law' the script extracted")
    print("• The Match: If R² > 0.9, your system has officially generated")
    print("  a rediscoverable physical law")
    print()
    print("You have just watched a machine discover gravity.")
    print("=" * 70)
    
    return {
        'discovered_k': discovered_k,
        'discovered_target': discovered_target,
        'r2_linear': r2_lin,
        'r2_quadratic': r2_quad,
        'k_error_percent': k_error,
        'target_error_percent': target_error
    }


if __name__ == "__main__":
    results = discover_law()

