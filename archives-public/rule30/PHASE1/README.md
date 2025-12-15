# Phase 1: Finding the Invariants

**Status**: ✅ Complete

## What We Did

Before diving into geometry, we needed to understand what stays constant in Rule 30's chaos. This phase searched for exact algebraic invariants—patterns that always hold true, no matter how random things look.

## Key Findings

We discovered **4 exact linear invariants** that always hold for Rule 30:
- Patterns that balance each other out
- Weighted sums that stay constant
- Relationships between 3-bit pattern frequencies

These were verified exhaustively for small cases (up to N=12) and statistically for larger ones.

## The Negative Result

We also proved something important: there's no closed 3-bit Markov system that can exactly reproduce Rule 30. This means you can't compress Rule 30's complexity into a simple finite-state machine—the chaos is real, not just apparent.

## What's Here

- `code/` - Scripts for finding and verifying invariants
- `docs/` - Detailed results and analysis
- `results/` - Output files from the analysis

## Next Steps

Phase 2 builds on these invariants to create a geometric representation. See `../PHASE2/` to continue.
