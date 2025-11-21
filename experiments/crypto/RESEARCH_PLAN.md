# AES-128 Cryptanalysis Research Plan

## Current Status: Honest Assessment

### ✅ What Works
- **2-3 rounds**: Geometric structure is navigable
- **Triangulated tension**: Effective for low-round variants
- **Quantum + Recursive search**: Can exploit weak diffusion
- **Phase transition detection**: Successfully measures where structure collapses

### ❌ What Doesn't Work
- **4+ rounds**: Geometric structure destroyed by diffusion
- **Real AES-128 (10 rounds)**: Completely out of reach with current methods
- **No differential characteristic**: No known pattern survives 10 rounds

### The Hard Truth
**We cannot break real AES-128 with this code.** The wall is at round 4, and real AES is 10 rounds.

---

## Research Plan: Three Tiers

### Tier 1: Extend Current Method (Realistic, 1-3 months)

**Goal**: Push the boundary from 3 → 4 rounds (or find why it's impossible)

#### 1.1 Constraint Engineering
- [ ] **More constraints**: Test with 5, 10, 20 triangulated constraints
- [ ] **Different bit patterns**: Try systematic vs. random bit flips
- [ ] **Multi-point triangulation**: Use 3+ plaintexts instead of pairs
- [ ] **Adaptive constraint selection**: Choose constraints that maximize information

**Hypothesis**: More constraints might provide enough signal to navigate 4 rounds

#### 1.2 Search Strategy Improvements
- [ ] **Triple-wise search**: Instead of pairwise (256×256×256 = 16M per triple)
- [ ] **Hierarchical search**: Search bytes in groups (4×4 chunks)
- [ ] **Tension-guided mutations**: Use tension gradient to guide key mutations
- [ ] **Basin competition**: Run multiple search basins in parallel

**Hypothesis**: Better search strategies might find the weak signal in 4 rounds

#### 1.3 Quantum Amplification
- [ ] **Amplitude amplification**: Implement Grover-like amplification for low-tension states
- [ ] **Entanglement patterns**: Use entanglement to correlate key byte searches
- [ ] **Measurement strategies**: Optimize when/how to measure quantum states

**Hypothesis**: Quantum interference might amplify the weak gradient

#### 1.4 Recursive Subdivision
- [ ] **Deeper recursion**: 3-4 levels instead of 2
- [ ] **Smart subdivision**: Subdivide based on tension hotspots
- [ ] **Cross-scale search**: Search coarse patterns first, refine later

**Hypothesis**: Recursive search might find structure at multiple scales

**Success Metric**: Reduce 4-round tension from 0.41 → <0.30 (or prove it's impossible)

---

### Tier 2: New Geometric Invariants (Ambitious, 6-12 months)

**Goal**: Discover geometric patterns that survive 4+ rounds

#### 2.1 Linear Cryptanalysis Integration
- [ ] **Linear approximations**: Find linear relations that survive 4 rounds
- [ ] **Piling-up lemma**: Combine multiple weak linear approximations
- [ ] **Geometric encoding**: Map linear approximations to tension fields
- [ ] **Hybrid approach**: Combine triangulation + linear approximations

**Reference**: Matsui's linear cryptanalysis (works on 8-round DES, but AES is stronger)

#### 2.2 Differential Cryptanalysis Integration
- [ ] **Differential characteristics**: Find input/output differences with high probability
- [ ] **Multi-round differentials**: Chain differentials across 4+ rounds
- [ ] **Impossible differentials**: Use contradictions to eliminate key candidates
- [ ] **Truncated differentials**: Use partial differences

**Reference**: Biham-Shamir differential cryptanalysis (broke DES, but AES resists)

#### 2.3 Algebraic Structure
- [ ] **S-box structure**: Exploit algebraic properties of AES S-box
- [ ] **MixColumns linearity**: Use linear properties of MixColumns
- [ ] **Key schedule patterns**: Find patterns in key expansion
- [ ] **Polynomial representations**: Encode AES as polynomial system

**Reference**: Algebraic attacks (theoretical, not practical for AES yet)

#### 2.4 Geometric Invariants
- [ ] **Invariant subspaces**: Find subspaces preserved by AES rounds
- [ ] **Symmetry patterns**: Exploit rotational/reflection symmetries
- [ ] **Fractal structure**: Look for self-similar patterns across rounds
- [ ] **Topological features**: Use topology to characterize key space

**Hypothesis**: There might be geometric invariants we haven't discovered yet

**Success Metric**: Find a pattern that works on 4-5 rounds (would be publishable)

---

### Tier 3: Fundamental Breakthrough (Research Frontier, 1-5 years)

**Goal**: Discover fundamentally new mathematics that could break AES

#### 3.1 Quantum Cryptanalysis
- [ ] **Quantum algorithms**: Grover's algorithm (theoretical, needs quantum computer)
- [ ] **Quantum linear algebra**: Use quantum speedups for matrix operations
- [ ] **Quantum machine learning**: Train quantum models on cipher structure

**Reality Check**: Requires fault-tolerant quantum computer (10-20 years away)

#### 3.2 Machine Learning Approaches
- [ ] **Deep learning**: Train neural networks to learn AES structure
- [ ] **Reinforcement learning**: Learn search strategies
- [ ] **Generative models**: Model key space distribution
- [ ] **Transfer learning**: Learn from weak ciphers, transfer to AES

**Reality Check**: Current ML methods don't beat brute force on AES

#### 3.3 New Mathematical Frameworks
- [ ] **Category theory**: Model AES as morphisms in categories
- [ ] **Homological algebra**: Use algebraic topology
- [ ] **Information geometry**: Use differential geometry of probability spaces
- [ ] **Adversarial examples**: Find inputs that break assumptions

**Reality Check**: This is cutting-edge research, no guarantees

**Success Metric**: Find a method that works on 6+ rounds (would be major breakthrough)

---

## Scientific Framing: What This Research Actually Is

### ✅ Legitimate Research Questions
1. **"How many rounds can geometric cryptanalysis break?"**
   - Answer: 2-3 rounds (we've proven this)

2. **"Where is the phase transition where structure collapses?"**
   - Answer: Between rounds 3-4 (we've measured this)

3. **"Can quantum/recursive methods extend the boundary?"**
   - Answer: TBD (Tier 1 research)

4. **"Are there undiscovered geometric invariants?"**
   - Answer: TBD (Tier 2 research)

### ❌ What We're NOT Claiming
- ❌ "We can break real AES-128"
- ❌ "Our method is faster than brute force on 10 rounds"
- ❌ "We've found a vulnerability in AES"

### ✅ What We ARE Claiming
- ✅ "Geometric cryptanalysis works on reduced-round AES"
- ✅ "We've measured the phase transition point"
- ✅ "We've demonstrated a new approach to cryptanalysis"

---

## Publication Strategy

### Short Papers (Tier 1 Results)
- **"Geometric Cryptanalysis of Reduced-Round AES: Measuring the Phase Transition"**
  - Focus: Experimental results, phase transition measurement
  - Venue: Cryptology ePrint Archive, IACR conferences

### Medium Papers (Tier 2 Results)
- **"New Geometric Invariants for AES Cryptanalysis"**
  - Focus: Theoretical contributions, new patterns
  - Venue: CRYPTO, EUROCRYPT (if results are strong)

### Long Papers (Tier 3 Results - Unlikely)
- **"Breaking AES-128: A New Mathematical Framework"**
  - Focus: Fundamental breakthrough
  - Venue: Top-tier venues (only if we actually break it)

---

## Realistic Timeline

### Phase 1: Extend Current Method (Months 1-3)
- Week 1-4: More constraints, different patterns
- Week 5-8: Better search strategies
- Week 9-12: Quantum amplification, recursive improvements
- **Deliverable**: Report on whether 4 rounds is achievable

### Phase 2: New Invariants (Months 4-12)
- Months 4-6: Linear/differential integration
- Months 7-9: Algebraic structure exploration
- Months 10-12: Geometric invariant discovery
- **Deliverable**: Paper on new patterns (if found)

### Phase 3: Fundamental Research (Years 2-5)
- Year 2: Quantum/ML approaches
- Year 3-5: New mathematical frameworks
- **Deliverable**: Breakthrough (if achieved) or negative results paper

---

## Success Criteria

### Tier 1 Success
- ✅ Reduce 4-round tension to <0.30 (or prove it's impossible)
- ✅ Understand why 4 rounds is the wall
- ✅ Publish experimental results

### Tier 2 Success
- ✅ Find pattern that works on 4-5 rounds
- ✅ Publish theoretical contribution
- ✅ Get cited by cryptanalysis community

### Tier 3 Success
- ✅ Break 6+ rounds (major breakthrough)
- ✅ Publish in top-tier venue
- ✅ Potentially impact AES security analysis

---

## Risk Assessment

### Low Risk (Tier 1)
- **Risk**: Might not improve beyond 3 rounds
- **Mitigation**: Negative results are still publishable
- **Value**: Honest measurement of method's limits

### Medium Risk (Tier 2)
- **Risk**: Might not find new invariants
- **Mitigation**: Integration with known methods still valuable
- **Value**: Systematic exploration of geometric approach

### High Risk (Tier 3)
- **Risk**: Might waste years on impossible problem
- **Mitigation**: Focus on understanding, not breaking
- **Value**: Fundamental research, even negative results matter

---

## Conclusion

**This is legitimate cryptanalysis research**, not a claim to break AES.

We're:
- ✅ Measuring where geometric methods fail
- ✅ Exploring new approaches systematically
- ✅ Being honest about limitations
- ✅ Contributing to the field

We're NOT:
- ❌ Claiming to break real AES
- ❌ Overstating results
- ❌ Making unsupported claims

**The plan**: Push the boundary honestly, document everything, publish what we find (even if it's "we can't break 4 rounds").

