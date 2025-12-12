"""
Ramsey Dynamic Search: Single-Universe Constraint Descent with Local Healing

Uses Dynamic Basin Search with local triangle/K₄ healing.
"""

from typing import Dict, Any, Tuple, List, Optional
import numpy as np

from core.classical.livnium_core_system import LivniumCoreSystem

# Handle relative imports
try:
    from .ramsey_encoder import RamseyEncoder
    from .ramsey_tension import (
        compute_ramsey_tension,
        count_monochromatic_k3,
        count_monochromatic_k4,
        get_all_k3_subsets,
        get_all_k4_subsets,
        get_k3_edges,
        get_k4_edges,
    )
    from .ramsey_local_feedback_patch import (
        heal_with_violation_priority,
        apply_local_feedback,
    )
except ImportError:
    from ramsey_encoder import RamseyEncoder
    from ramsey_tension import (
        compute_ramsey_tension,
        count_monochromatic_k3,
        count_monochromatic_k4,
        get_all_k3_subsets,
        get_all_k4_subsets,
        get_k3_edges,
        get_k4_edges,
    )
    from ramsey_local_feedback_patch import (
        heal_with_violation_priority,
        apply_local_feedback,
    )

# Optional imports (available if modules exist)
try:
    from ramsey_curvature_healing import (
        heal_with_curvature_guidance,
        heal_with_global_coherence,
    )
except ImportError:
    # Fallback if curvature healing not available
    heal_with_curvature_guidance = None
    heal_with_global_coherence = None

try:
    from ramsey_basin_escape import (
        BasinTracker,
        break_false_vacuum_aggressive,
        escape_basin_with_constraint_flips,
    )
except ImportError:
    BasinTracker = None
    break_false_vacuum_aggressive = None
    escape_basin_with_constraint_flips = None

try:
    from ramsey_geometric_inversion import (
        deep_basin_descent,
        geometric_sw_inversion,
        targeted_violation_fix,
    )
except ImportError:
    deep_basin_descent = None
    geometric_sw_inversion = None
    targeted_violation_fix = None

Edge = Tuple[int, int]
Coloring = Dict[Edge, int]


def _heal_k3_violations(
    system: LivniumCoreSystem,
    encoder: RamseyEncoder,
    coloring: Coloring,
    vertices: List[int],
    max_heal: int = 10,
):
    """
    Local healing for K3 violations.

    For each violated triangle, pick one edge and flip its SW bias.
    """
    triangles = get_all_k3_subsets(vertices)
    healed = 0

    for tri in triangles:
        if healed >= max_heal:
            break

        e1, e2, e3 = get_k3_edges(tri)
        if e1 not in coloring or e2 not in coloring or e3 not in coloring:
            continue
        c1, c2, c3 = coloring[e1], coloring[e2], coloring[e3]
        if not (c1 == c2 == c3):
            continue  # not a violation

        # Choose an edge to flip (simple heuristic: the one with largest |SW| to push)
        candidate_edges = [e1, e2, e3]
        best_edge = None
        best_mag = -1.0

        for e in candidate_edges:
            if e not in encoder.edge_to_coords:
                continue
            coord = encoder.edge_to_coords[e]
            cell = system.get_cell(coord)
            if cell is None:
                continue
            mean_sw = abs(cell.symbolic_weight)
            if mean_sw > best_mag:
                best_mag = mean_sw
                best_edge = e

        if best_edge is None:
            continue

        # Flip this edge's SW bias to opposite color (STRONG flip)
        cur_color = coloring[best_edge]
        new_color = 1 - cur_color
        target = +10.0 if new_color == 1 else -10.0

        coord = encoder.edge_to_coords[best_edge]
        cell = system.get_cell(coord)
        if cell is None:
            continue
        # STRONG push: 90% target, 10% current (almost direct flip)
        cell.symbolic_weight = 0.1 * cell.symbolic_weight + 0.9 * target

        healed += 1


def _heal_k4_violations(
    system: LivniumCoreSystem,
    encoder: RamseyEncoder,
    coloring: Coloring,
    vertices: List[int],
    max_heal: int = 10,
):
    """
    Local healing for K4 violations.

    For each violated K₄, pick one edge and flip its SW bias.
    """
    k4s = get_all_k4_subsets(vertices)
    healed = 0

    for quad in k4s:
        if healed >= max_heal:
            break

        edges = get_k4_edges(quad)
        if any(e not in coloring for e in edges):
            continue
        
        colors = {coloring[e] for e in edges}
        if len(colors) != 1:
            continue  # not a violation

        # Choose an edge to flip
        best_edge = None
        best_mag = -1.0

        for e in edges:
            if e not in encoder.edge_to_coords:
                continue
            coord = encoder.edge_to_coords[e]
            cell = system.get_cell(coord)
            if cell is None:
                continue
            mean_sw = abs(cell.symbolic_weight)
            if mean_sw > best_mag:
                best_mag = mean_sw
                best_edge = e

        if best_edge is None:
            continue

        # Flip this edge's SW bias to opposite color (STRONG flip)
        cur_color = coloring[best_edge]
        new_color = 1 - cur_color
        target = +10.0 if new_color == 1 else -10.0

        coord = encoder.edge_to_coords[best_edge]
        cell = system.get_cell(coord)
        if cell is None:
            continue
        # STRONG push: 90% target, 10% current (almost direct flip)
        cell.symbolic_weight = 0.1 * cell.symbolic_weight + 0.9 * target

        healed += 1


def solve_ramsey_dynamic(
    system: LivniumCoreSystem,
    encoder: RamseyEncoder,
    max_steps: int = 1000,
    verbose: bool = True,
    constraint_type: str = "k4",  # "k3" for R(3,3)
    initialize_random: bool = True,
    checkpoint_path: Optional[str] = None,
    save_checkpoint_interval: int = 500,
    visualize: bool = False,
    visualize_interval: int = 50,
) -> Dict[str, Any]:
    """
    Single-universe Dynamic Basin descent for Ramsey.

    constraint_type:
      - "k3" → triangles (R(3,3))
      - "k4" → K4 (R(4,4))
    
    initialize_random: If True, start with random balanced coloring
    checkpoint_path: Path to checkpoint file (optional)
    save_checkpoint_interval: Save checkpoint every N steps
    visualize: If True, show live visualization
    visualize_interval: Update visualization every N steps
    """
    # Handle checkpoint loading
    start_step = 1
    vertices = encoder.vertices
    
    if checkpoint_path:
        try:
            from .ramsey_checkpoint import load_checkpoint, restore_system_state
        except ImportError:
            from ramsey_checkpoint import load_checkpoint, restore_system_state
        
        from pathlib import Path
        ckpt_path = Path(checkpoint_path)
        checkpoint = load_checkpoint(ckpt_path)
        
        if checkpoint:
            if verbose:
                # Calculate percentage for checkpoint
                if constraint_type == "k3":
                    total_constraints = len(get_all_k3_subsets(vertices))
                else:
                    total_constraints = len(get_all_k4_subsets(vertices))
                
                if total_constraints > 0:
                    best_viols = checkpoint['best_violations']
                    best_satisfied = total_constraints - best_viols
                    best_percent = (best_satisfied / total_constraints) * 100.0
                    print(f"  📍 Resuming from checkpoint: step {checkpoint['step']}, "
                          f"{best_percent:.2f}% satisfied")
                else:
                    print(f"  📍 Resuming from checkpoint: step {checkpoint['step']}, "
                          f"best violations: {checkpoint['best_violations']}")
            restore_system_state(system, encoder, checkpoint)
            start_step = checkpoint['step'] + 1
            best_violations = checkpoint['best_violations']
            best_coloring = checkpoint['best_coloring']
            steps_to_best = checkpoint.get('steps_to_best', checkpoint['step'])
        else:
            # No checkpoint, start fresh
            if initialize_random:
                import random
                random_coloring = {edge: random.randint(0, 1) for edge in encoder.edges}
                encoder.encode_coloring(random_coloring, initial_only=True)
            best_violations = float("inf")
            best_coloring: Coloring = {}
            steps_to_best = 0
    else:
        # No checkpoint path, start fresh
        if initialize_random:
            import random
            random_coloring = {edge: random.randint(0, 1) for edge in encoder.edges}
            encoder.encode_coloring(random_coloring, initial_only=True)
        best_violations = float("inf")
        best_coloring: Coloring = {}
        steps_to_best = 0
    
    stuck_counter = 0  # Track if we're stuck
    last_reset_step = 0  # Track when we last reset (cooldown)
    reset_cooldown = 500  # Don't reset more than once every 500 steps
    last_escape_step = -10000  # Track when we last escaped (cooldown) - start far in past
    escape_cooldown = 1000  # Don't escape more than once every 1000 steps
    last_escape_violations = None  # Track violations before escape (None = no escape yet)
    
    # Setup basin tracker to detect false vacuum re-formation
    basin_tracker = None
    try:
        if BasinTracker is not None:
            basin_tracker = BasinTracker(window_size=2000)
    except NameError:
        # BasinTracker not imported (optional feature)
        basin_tracker = None
    
    # Setup visualizer if requested
    visualizer = None
    if visualize:
        try:
            from .ramsey_visualizer import RamseyVisualizer
        except ImportError:
            from ramsey_visualizer import RamseyVisualizer
        visualizer = RamseyVisualizer(len(vertices), constraint_type)
        visualizer.setup_plot()

    for step in range(start_step, max_steps + 1):
        # Decode current coloring from SW field
        coloring = encoder.decode_coloring()

        # Count violations
        if constraint_type == "k3":
            violations = count_monochromatic_k3(coloring, vertices)
        else:
            violations = count_monochromatic_k4(coloring, vertices)

        # Update best
        if violations < best_violations:
            best_violations = violations
            best_coloring = dict(coloring)
            steps_to_best = step
            stuck_counter = 0  # Reset stuck counter
        else:
            stuck_counter += 1

        # Compute normalized tension
        tension = compute_ramsey_tension(coloring, vertices, constraint_type)
        
        # Calculate percentage satisfied
        if constraint_type == "k3":
            total_constraints = len(get_all_k3_subsets(vertices))
        else:
            total_constraints = len(get_all_k4_subsets(vertices))
        
        if total_constraints > 0:
            satisfied = total_constraints - violations
            percent_satisfied = (satisfied / total_constraints) * 100.0
            best_satisfied = total_constraints - best_violations
            best_percent = (best_satisfied / total_constraints) * 100.0
        else:
            percent_satisfied = 0.0
            best_percent = 0.0
        
        # Track basin state
        if basin_tracker is not None:
            basin_tracker.add(step, violations, percent_satisfied)

        if verbose and step % 500 == 0:
            print(
                f"    Step {step}: {percent_satisfied:.2f}% satisfied "
                f"(Best: {best_percent:.2f}% at step {steps_to_best})"
            )
        
        # Update visualization
        if visualizer and step % visualize_interval == 0:
            visualizer.update(system, encoder, coloring, step, violations, best_violations)

        # Check solved
        if violations == 0:
            break

        # --- Curvature-Guided Multi-Edge Healing (The Compass) ---
        # This gives the solver direction and stops chaotic oscillations
        prev_violations = violations
        
        # Apply local feedback: push edges away from violated constraints
        apply_local_feedback(
            system, encoder, coloring, vertices,
            constraint_type=constraint_type,
            lambda_weight=0.5  # Stronger push - helps escape false vacua
        )
        
        # Re-decode after feedback
        coloring = encoder.decode_coloring()
        
        # Use curvature-guided healing if available (gives compass/direction)
        # Otherwise fall back to violation-count priority
        if heal_with_curvature_guidance is not None:
            # Curvature-guided multi-edge healing (restores gradient descent)
            for heal_pass in range(3):  # Fewer passes needed (more intelligent)
                healed_count = heal_with_curvature_guidance(
                    system, encoder, coloring, vertices,
                    constraint_type=constraint_type,
                    max_edges_to_flip=20,  # Multi-edge flips
                    curvature_threshold=0.2  # Only heal high-curvature regions
                )
                
                if healed_count == 0:
                    break  # No more violations to heal
                
                # Re-decode after healing
                coloring = encoder.decode_coloring()
                if constraint_type == "k3":
                    violations = count_monochromatic_k3(coloring, vertices)
                else:
                    violations = count_monochromatic_k4(coloring, vertices)
                
                # If we made progress, continue; otherwise try global coherence
                if violations < prev_violations:
                    prev_violations = violations
                else:
                    # Try global coherence healing (more aggressive)
                    if heal_with_global_coherence is not None:
                        healed_count = heal_with_global_coherence(
                            system, encoder, coloring, vertices,
                            constraint_type=constraint_type,
                            max_edges_to_flip=15
                        )
                        if healed_count > 0:
                            coloring = encoder.decode_coloring()
                            if constraint_type == "k3":
                                violations = count_monochromatic_k3(coloring, vertices)
                            else:
                                violations = count_monochromatic_k4(coloring, vertices)
                            if violations < prev_violations:
                                prev_violations = violations
                    break
        else:
            # Fallback: violation-count priority (old method)
            for heal_pass in range(5):  # More healing passes
                healed_count = heal_with_violation_priority(
                    system, encoder, coloring, vertices,
                    constraint_type=constraint_type,
                    max_heal=50  # Heal more edges per pass - target hot patches
                )
                
                if healed_count == 0:
                    break  # No more violations to heal
                
                # Re-decode after healing to check if we improved
                coloring = encoder.decode_coloring()
                if constraint_type == "k3":
                    violations = count_monochromatic_k3(coloring, vertices)
                else:
                    violations = count_monochromatic_k4(coloring, vertices)
                
                # If we made progress, continue; otherwise stop
                if violations >= prev_violations:
                    break
                
                prev_violations = violations
        
        # --- Basin Escape: Break False Vacuum Attractor ---
        # Detect if we're stuck in false vacuum (98.61% re-formation pattern) OR collapsed (0%)
        if basin_tracker is not None:
            # FIRST: Check for collapse (0% satisfied) - most urgent
            # BUT: Only if cooldown has passed (prevent reset loop)
            # AND: Only if current percent is actually low (not just historical)
            if (basin_tracker.detect_collapse() and 
                (step - last_reset_step) >= reset_cooldown and
                percent_satisfied < 1.0):  # Current state must also be collapsed
                if verbose:
                    print(f"    🚨 COLLAPSE DETECTED ({percent_satisfied:.2f}%) - applying emergency escape...")
                
                # Emergency: Reset to random balanced coloring
                import random
                random_coloring = {edge: random.randint(0, 1) for edge in encoder.edges}
                encoder.encode_coloring(random_coloring, initial_only=True)
                coloring = encoder.decode_coloring()
                if constraint_type == "k3":
                    violations = count_monochromatic_k3(coloring, vertices)
                else:
                    violations = count_monochromatic_k4(coloring, vertices)
                prev_violations = violations
                stuck_counter = 0  # Reset stuck counter
                last_reset_step = step  # Update cooldown timer
                if verbose:
                    print(f"    🔄 Emergency reset: violations now {violations}")
            
            # SECOND: Check for re-formation pattern (collapse → same basin)
            # BUT: Only if cooldown passed AND (no previous escape OR we improved since last escape)
            elif (basin_tracker.detect_reformation() and 
                  (step - last_escape_step) >= escape_cooldown and
                  (last_escape_violations is None or violations < last_escape_violations)):  # Only if we improved since last escape
                if verbose:
                    print(f"    ⚠️  Detected false vacuum re-formation - applying basin escape...")
                
                # Remember violations before escape
                violations_before_escape = violations
                
                # Aggressive basin breaking
                if break_false_vacuum_aggressive is not None:
                    escaped = break_false_vacuum_aggressive(
                        system, encoder, coloring, vertices,
                        constraint_type=constraint_type,
                        max_flips=50  # More aggressive escape
                    )
                    if escaped > 0:
                        coloring = encoder.decode_coloring()
                        if constraint_type == "k3":
                            violations = count_monochromatic_k3(coloring, vertices)
                        else:
                            violations = count_monochromatic_k4(coloring, vertices)
                        
                        # Only update if escape actually helped (or at least didn't make it much worse)
                        if violations <= violations_before_escape * 1.1:  # Allow 10% tolerance
                            prev_violations = violations
                            last_escape_step = step
                            last_escape_violations = violations
                            if verbose:
                                print(f"    🔥 Basin escape: flipped {escaped} edges, violations: {violations}")
                        else:
                            # Escape made things worse - revert by not updating state
                            if verbose:
                                print(f"    ⚠️  Escape made things worse ({violations} > {violations_before_escape}) - skipping")
                            # Revert coloring by re-decoding (this restores previous state)
                            coloring = encoder.decode_coloring()
            
            # THIRD: Check for deep basin (99%+) - use geometric inversion
            elif percent_satisfied >= 99.0:
                if verbose and step % 500 == 0:
                    print(f"    🌟 Deep basin detected ({percent_satisfied:.2f}%) - applying geometric SW inversion...")
                
                # Geometric SW inversion (preserves structure, pushes deeper)
                if deep_basin_descent is not None:
                    inverted = deep_basin_descent(
                        system, encoder, coloring, vertices,
                        constraint_type=constraint_type,
                        current_percent=percent_satisfied
                    )
                    if inverted > 0:
                        coloring = encoder.decode_coloring()
                        if constraint_type == "k3":
                            violations = count_monochromatic_k3(coloring, vertices)
                        else:
                            violations = count_monochromatic_k4(coloring, vertices)
                        prev_violations = violations
                        if verbose:
                            print(f"    ✨ Geometric inversion: {inverted} edges, violations: {violations}")
            
            # FOURTH: Check for false vacuum (stuck at same % repeatedly, but <99%)
            elif basin_tracker.detect_false_vacuum(target_percent=percent_satisfied, tolerance=1.0):
                if verbose and step % 1000 == 0:
                    print(f"    ⚠️  Stuck in false vacuum ({percent_satisfied:.2f}%) - applying constraint-based escape...")
                
                # Constraint-based escape (directly fix violated constraints)
                if escape_basin_with_constraint_flips is not None:
                    escaped = escape_basin_with_constraint_flips(
                        system, encoder, coloring, vertices,
                        constraint_type=constraint_type,
                        max_flips=40  # More aggressive
                    )
                    if escaped > 0:
                        coloring = encoder.decode_coloring()
                        if constraint_type == "k3":
                            violations = count_monochromatic_k3(coloring, vertices)
                        else:
                            violations = count_monochromatic_k4(coloring, vertices)
                        # Update prev_violations after escape
                        prev_violations = violations
        
        # Add small exploration noise if stuck (escape local minima)
        if stuck_counter > 100 and violations > 0:
            # Small random perturbation to escape local minima
            import random
            edges_to_perturb = random.sample(list(encoder.edges), min(3, len(encoder.edges)))
            for edge in edges_to_perturb:
                coord = encoder.edge_to_coords[edge]
                cell = system.get_cell(coord)
                if cell:
                    # Small random push
                    noise = random.uniform(-2.0, 2.0)
                    cell.symbolic_weight += noise
            stuck_counter = 0  # Reset after perturbation
        
        # Save checkpoint periodically
        if checkpoint_path and step % save_checkpoint_interval == 0:
            try:
                from .ramsey_checkpoint import save_checkpoint
            except ImportError:
                from ramsey_checkpoint import save_checkpoint
            
            from pathlib import Path
            ckpt_path = Path(checkpoint_path)
            
            # Save system state (SW values)
            system_state = {}
            for edge, coord in encoder.edge_to_coords.items():
                cell = system.get_cell(coord)
                if cell:
                    system_state[coord] = cell.symbolic_weight
            
            # Save encoder state (edge mapping)
            encoder_state = dict(encoder.edge_to_coords)
            
            # Metadata
            metadata = {
                'constraint_type': constraint_type,
                'n_vertices': len(vertices),
                'tension': tension,
                'steps_to_best': steps_to_best,
            }
            
            save_checkpoint(
                ckpt_path,
                best_coloring,
                best_violations,
                step,
                system_state,
                encoder_state,
                metadata
            )
            
            if verbose:
                if total_constraints > 0:
                    best_satisfied = total_constraints - best_violations
                    best_percent = (best_satisfied / total_constraints) * 100.0
                    print(f"  💾 Checkpoint saved: step {step}, {best_percent:.2f}% satisfied")
                else:
                    print(f"  💾 Checkpoint saved: step {step}, best violations: {best_violations}")

    # Final report uses best coloring seen
    final_coloring = best_coloring if best_coloring else coloring
    final_tension = compute_ramsey_tension(final_coloring, vertices, constraint_type)
    is_solved = best_violations == 0
    
    # Close visualizer
    if visualizer:
        visualizer.update(system, encoder, final_coloring, max_steps, 
                         best_violations, best_violations)  # Final update
        try:
            print("\n  Press Enter to close visualization (or Ctrl+C to exit)...")
            input()
        except (EOFError, KeyboardInterrupt):
            # Non-interactive mode or user cancelled
            pass
        visualizer.close()

    return {
        "coloring": final_coloring,
        "violations": best_violations,
        "tension": final_tension,
        "steps": steps_to_best if is_solved else max_steps,
        "solved": is_solved,
        "best_violations": best_violations,
        "steps_to_best": steps_to_best,
    }
