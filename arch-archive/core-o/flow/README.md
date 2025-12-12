# Flow Engine: O-A7 Implementation

**The Flow Law transforms Livnium-O from static geometry into dynamic universe.**

---

## Why Flow Engine?

**Without Flow Engine:**
- Livnium-O is a **frozen universe**
- Perfect geometry, but **dead still**
- Cannot simulate movement
- Cannot run search
- Cannot optimize
- Cannot propagate meaning

**With Flow Engine:**
- Neighbors glide around Om
- Exposure changes continuously
- SW redistributes
- Geometry becomes computation
- Semantics become movement
- Optimization becomes real
- Dynamics are reversible
- States become trajectories

**This is the heartbeat.**

---

## O-A7: The Flow Law

Each neighbor moves **only along the tangent plane**, preserving tangency:

\[
v_i(t) \cdot (N_i - Om) = 0
\]

Update rule:

\[
N_i(t + \Delta t) = Om + (1+r_i) \cdot R_i(t) \cdot \hat{u}_i(t)
\]

Where:
- \(\hat{u}_i(t)\) is the unit direction of neighbor i
- \(R_i(t)\) is a small incremental rotation in SO(3)
- defined by the local velocity field

---

## Core Functions

### `move_neighbor()`

Move one neighbor along tangent plane:

```python
from flow.flow_engine import move_neighbor

# Create tangential velocity
tangential_velocity = np.array([0.1, 0.0, 0.0])

# Move neighbor
new_system = move_neighbor(system, neighbor_id=1, tangential_velocity, dt=0.01)
```

### `evolve_system()`

Evolve entire system forward in time:

```python
from flow.flow_engine import evolve_system

# Create velocity field
velocity_field = {
    1: np.array([0.1, 0.0, 0.0]),
    2: np.array([0.0, 0.1, 0.0]),
    3: np.array([0.0, 0.0, 0.1]),
}

# Evolve
new_system = evolve_system(system, velocity_field, dt=0.01)
```

### `FlowEngine` Class

Complete dynamics system with history and reversibility:

```python
from flow.flow_engine import FlowEngine

# Create engine
engine = FlowEngine(system, dt=0.01)

# Define force function
def repulsion_force(system, neighbor_id):
    # Compute repulsion from other neighbors
    return force_vector

# Evolve with forces
engine.step_with_forces(repulsion_force)

# Reverse step
engine.reverse_step()

# Reset to initial state
engine.reset()
```

---

## Usage Examples

### Simple Movement

```python
from classical.livnium_o_system import LivniumOSystem
from flow.flow_engine import move_neighbor
import numpy as np

# Create system
system = LivniumOSystem(neighbor_radii=[1.0]*6, core_radius=1.0)

# Move neighbor 1
neighbor = system.get_neighbor_nodes()[0]
pos_vec = np.array(neighbor.position)
radial_unit = pos_vec / np.linalg.norm(pos_vec)

# Create tangential velocity
tangential_velocity = np.cross(radial_unit, np.array([1, 0, 0]))
tangential_velocity = tangential_velocity / np.linalg.norm(tangential_velocity) * 0.1

# Move
new_system = move_neighbor(system, neighbor.node_id, tangential_velocity, dt=0.1)
```

### Force-Based Evolution

```python
from flow.flow_engine import FlowEngine

def repulsion_force(system, neighbor_id):
    """Repulsion from other neighbors."""
    neighbor = system.get_node(neighbor_id)
    pos_vec = np.array(neighbor.position)
    
    force = np.zeros(3)
    for other in system.get_neighbor_nodes():
        if other.node_id == neighbor_id:
            continue
        
        other_pos = np.array(other.position)
        diff = pos_vec - other_pos
        dist = np.linalg.norm(diff)
        if dist > 1e-10:
            force += diff / (dist ** 2) * 0.01
    
    return force

# Create engine
engine = FlowEngine(system, dt=0.01)

# Evolve
for _ in range(100):
    engine.step_with_forces(repulsion_force)
```

### Gradient-Based Optimization

```python
def exposure_gradient(system, neighbor_id):
    """Gradient to maximize exposure."""
    neighbor = system.get_node(neighbor_id)
    # Compute gradient of exposure with respect to position
    # (simplified example)
    return gradient_vector

engine = FlowEngine(system, dt=0.01)
engine.step_with_gradient(exposure_gradient)
```

---

## Properties

**Tangency Preservation:**
- Distance \(|N_i - Om| = 1 + r_i\) always preserved
- Neighbors slide around sphere surface
- No radial movement

**SO(3) Rotations:**
- All movements are SO(3) rotations
- Perfect reversibility
- Determinant = 1

**Ledger Conservation:**
- Total SW preserved
- Kissing constraint preserved
- Core radius fixed
- Om position fixed

**Reversibility:**
- Every step can be reversed
- History maintained
- Perfect inversion

---

## Status

✅ **Implementation Complete**
✅ **Tests Passing** (8 tests)
✅ **Demo Working**
✅ **Ready for Use**

---

**This is where Livnium-O becomes alive.**

