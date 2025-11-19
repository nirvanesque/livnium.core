# Curvature-Guided Multi-Edge Healing

## The Problem: Chaotic Oscillations

When the solver escapes the 98.6% false vacuum, it enters a **frustrated energy landscape**:
- No clear gradient/direction
- Bouncing between shallow basins (78%, 62%, 91%, 84%, 70%)
- Looks random but is actually **bounded chaos** - same edges stay broken, same clusters stay tense

## The Solution: Curvature-Guided Healing

This module implements the **"compass"** that gives the solver direction:

### 1. **Curvature Computation** (`compute_edge_curvature`)
- Measures geometric curvature for each edge
- High curvature = edge in conflicted region (many constraints pulling different ways)
- Low curvature = edge in stable region (constraints agree)
- **Replaces SW-based inference** with constraint-based geometric signals

### 2. **Violation Clusters** (`find_violation_clusters`)
- Finds sets of edges that participate in overlapping violated constraints
- These should be healed **together** (multi-edge flip) to maintain coherence
- Prevents: "fix one edge, break another" oscillation

### 3. **Flip Impact** (`compute_flip_impact`)
- Computes: if we flip this edge, how many constraints get fixed vs broken?
- Only flips edges with **positive net impact** (fix more than they break)
- This is the **true gradient** - not SW magnitude

### 4. **Curvature-Guided Healing** (`heal_with_curvature_guidance`)
- Processes clusters by curvature (highest first)
- For each cluster, computes flip impact for all edges
- Flips edges with positive net impact
- **Multi-edge flips** maintain global coherence

### 5. **Global Coherence Healing** (`heal_with_global_coherence`)
- More aggressive version
- Computes flip impact for ALL edges (not just in clusters)
- Selects edges with best net impact
- Ensures flips don't conflict with each other

## How It Works

```python
# 1. Compute curvature (geometric signal, not SW)
edge_curvature = compute_edge_curvature(encoder, coloring, vertices, constraint_type)

# 2. Find violation clusters (edges that should be healed together)
clusters = find_violation_clusters(coloring, vertices, constraint_type)

# 3. Process clusters by curvature (highest first)
for cluster in sorted_clusters:
    # 4. Compute flip impact for each edge
    for edge in cluster:
        fixed, created = compute_flip_impact(coloring, edge, vertices, constraint_type)
        net_impact = fixed - created  # Positive = good
    
    # 5. Flip edges with positive net impact
    flip_edges_with_positive_impact()
```

## Why This Stops Oscillations

### Before (Chaotic):
- Flip edge A → fixes 8 constraints, breaks 11 → net -3
- Flip edge B → fixes 5 constraints, breaks 7 → net -2
- System bounces around, never improving

### After (Gradient Descent):
- Compute flip impact for all edges
- Only flip edges with positive net impact
- System **descends** instead of oscillating

## Integration

The curvature healing is automatically used if available:

```python
if heal_with_curvature_guidance is not None:
    # Use curvature-guided healing (gives compass/direction)
    heal_with_curvature_guidance(...)
else:
    # Fallback to violation-count priority (old method)
    heal_with_violation_priority(...)
```

## Expected Behavior

### Before Curvature Healing:
- Oscillations: 78%, 62%, 91%, 84%, 70% (chaotic)
- Same edges stay broken
- No convergence

### After Curvature Healing:
- **Stable descent**: 70% → 75% → 80% → 85% → 90% → 95% → 100%
- Hot patches get resolved
- Convergence to solution

## Parameters

- `max_edges_to_flip`: How many edges to flip per pass (default: 20)
- `curvature_threshold`: Only heal edges above this curvature (default: 0.2)
- `min_cluster_size`: Minimum cluster size to consider (default: 3)

## Technical Details

- **Curvature formula**: `curvature = violations / total_constraints`
  - 0.0 = flat (no violations)
  - 1.0 = maximum conflict (all constraints violated)
  
- **Flip impact**: `net_impact = violations_fixed - violations_created`
  - Positive = beneficial flip
  - Negative = harmful flip
  
- **Cluster detection**: Uses graph overlap (two constraints overlap if they share an edge)

## Future Improvements

1. **Adaptive thresholds**: Adjust `curvature_threshold` based on system state
2. **Multi-step lookahead**: Consider 2-3 flips ahead, not just one
3. **Constraint weighting**: Weight constraints by how many times they've been violated
4. **Symmetry breaking**: Detect symmetric solutions and break symmetry

