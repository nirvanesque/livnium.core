# SNLI: Geometric Mapping (Phase 2)

In Livnium, SNLI is not a simple 3-way classification problem. It is a **geometric occupancy problem** where logical categories are mapped to specific zones in the semantic Basin Field.

## 1. The Physics of Entailment
-   **Law**: High Alignment ($\cos \theta \approx 1.0$).
-   **Observation**: Hypotheses that lie within the "Semantic Shadow" of the premise collapse towards the **Entailment Anchor**.
-   **Energy State**: Low Tension, high similarity.

## 2. The Physics of Contradiction
-   **Law**: High Divergence ($D = \text{PIVOT} - \text{Alignment} > 0.4$).
-   **Observation**: Claims that actively push against each other create a repulsion force that separates the states until they reach **Contradiction Basins**.
-   **Energy State**: Peak Tension at the start, followed by a violent separation.

## 3. The Physics of NeutralITY
-   **Law**: Orthogonality or High Residual Tension.
-   **Observation**: When $P$ and $H$ share no semantic energy, they drift toward the **Neutral Anchor**. Neutrality in Livnium is the "Ground State" when no attractive or repulsive work is performed.
-   **Energy State**: Constant, moderate tension.

## 4. Phase 2 Features
-   **Step-by-Step Auditing**: Using the `TensionLedger`, we can now visualize the exact moment a sentence pair "decides" to contradict.
-   **Entropy Capture**: The `NLIHead` now receives the radius ($|r|$) of the final state, which acts as a measure of "reasoning confidence."
