# The Shadow Rule 30 Project

What if we could understand chaos by mapping it to geometry? This project explores Rule 30—one of the simplest cellular automata that produces seemingly random behavior—by translating its dynamics into continuous geometric space.

## What is Rule 30?

Rule 30 is a one-dimensional cellular automaton discovered by Stephen Wolfram. Start with a single black cell, then apply a simple rule: each new cell's color depends on its three neighbors above. Despite this simplicity, Rule 30's center column appears completely random—no pattern, no predictability.

The question: Can we find structure in this apparent randomness by moving to a different representation?

## The Approach

Instead of working with bits directly, we map Rule 30's behavior into continuous geometric space. Think of it like translating a digital signal into an analog waveform—the same information, but viewed through a different lens.

The pipeline has seven phases:

- **Phase 1**: Found exact algebraic invariants—patterns that always hold, no matter how chaotic things get
- **Phase 2**: Built a geometric representation of Rule 30's state space
- **Phase 3**: Learned how the geometry evolves over time (the "motion law")
- **Phase 4**: Created a decoder that maps geometry back to bits, matching Rule 30's density with 94% accuracy
- **Phase 5**: Reserved for future work
- **Phase 6**: Added Livnium steering to guide the dynamics
- **Phase 7**: Proof experiments showing the system works without Livnium

## Quick Start

See [QUICK_START.md](./QUICK_START.md) for step-by-step instructions to run the full pipeline.

Each phase has its own directory with detailed documentation:
- `PHASE1/` through `PHASE7/` contain code, docs, and results
- `results/` contains generated trajectories and analysis outputs
- `archive/` holds legacy scripts for reference

## What You'll Find

- **Chaos trajectories**: Raw geometric data from Phase 2 (`results/chaos14/`, `results/chaos15/`)
- **PCA models**: Learned dynamics in reduced dimensions (Phase 3)
- **Shadow trajectories**: Synthetic sequences that match Rule 30's statistics (Phase 4)
- **Proof experiments**: Validation that the system works independently (Phase 7)

## Notes

- Run scripts from within their phase directories
- Python 3.10+ required; dependencies vary by phase (see each phase's README)
- Some phases have known caveats documented in their `docs/` folders

## Why This Matters

Rule 30 is a testbed for understanding how simple rules can produce complex behavior. By mapping it to geometry, we're exploring whether there's hidden structure in apparent randomness—and whether that structure can be learned, predicted, and controlled.
