# Nova v3 Architecture (Livnium Core v1.0)

Vector-physics SNLI model. A single state vector evolves under frozen geometric laws, then a thin head reads out entailment / neutral / contradiction.

- **Physics (core/):** collapse engine, divergence/tension laws, optional dynamic basins.
- **Task (tasks/snli/, text/):** quantum text encoder + SNLI head.
- **Orchestration (training/, chat/):** data loading, training, evaluation, visualization.

## End-to-End Flow (SNLI)
1. Encode premise/hypothesis with the pretrained quantum encoder (`tasks/snli/encoding_snli.py`, `text/quantum_text_encoder.py`) to get `v_p`, `v_h` (dim inferred from quantum checkpoint).
2. Initialize state: `h0 = v_p + v_h + eps` (eps ~ N(0, 0.01) to break symmetry).
3. Collapse (`core/vector_collapse_engine.py`):
   - Static: three learned anchors (E/C/N) apply divergence forces.
   - Dynamic (train-only, optional): per-label micro-basins replace the static anchors; routing uses the gold label.
4. Head (`tasks/snli/head_snli.py`): combines collapsed state with geometric features to produce logits for E/N/C.
5. Training loop (`training/train_snli_vector.py`): cross-entropy with optional label smoothing and neutral up-weighting. Evaluation always uses static collapse even if dynamic basins were trained.

## Physics Primitives (`core/physics_laws.py`)
- Alignment: `align(a, b) = cos(a, b)`.
- Divergence (frozen law): `div = 0.38 - align`.
  - `div < 0` -> inward pull (entailment-like)
  - `div ~ 0` -> neutral equilibrium
  - `div > 0` -> outward push (contradiction-like)
- Tension: `tension = |div|` (stress magnitude).

## Collapse Dynamics (`core/vector_collapse_engine.py`)
Per-step update for static anchors:

```
h_hat_t = normalize(h_t)
align_k = h_hat_t dot a_hat_k       # k in {E, C, N}; a_hat_k is normalized anchor
div_k   = 0.38 - align_k
Delta_t = 0                         # physics-only update (MLP removed)
dir_k   = normalize(h_t - a_hat_k)

h_{t+1} = h_t + Delta_t - s_k * div_k * dir_k
h_{t+1} = clip_norm(h_{t+1}, max=10)
```

- `s_k` are learned strengths (defaults: E=0.1, C=0.1, N=0.05).
- Trace of per-step alignment/divergence/tension is returned for inspection.

### Dynamic Basins (Train-Time Only)
Implemented in `core/basin_field.py`, invoked via `collapse_dynamic`.

- Routing: for label `y in {E, N, C}`, pick the basin with highest cosine to `h_hat`; if none, seed with `h_hat`.
- Spawn rule: if `tension > 0.15` and `alignment < 0.6`, create a new basin at `h_hat` (capped per label, default 64).
- EMA update: after collapse, update basin center `c <- normalize((1 - lr) * c + lr * h_hat_final)` with lr=0.05.
- Prune/merge (optional cadence): prune anchors with `count < min_count` (default 10); merge if `cos(c_i, c_j) > 0.97`.
- Eval safety: dynamic routing is disabled during validation/test to avoid label leakage; static collapse is used instead.
- Entropy pressure knob: pass `entropy_pressure` to `collapse_dynamic` to decay basin utilities globally; basins with utility ≤ 0 are removed, with optional budget and audit log. Training can learn this scalar via `--learn-entropy-pressure`; default is 0 (no pruning).
- Anchor sync for eval: after dynamic training, static anchors are derived from basin centers (utility-weighted) so static collapse at eval has meaningful directions.

## Text Encoding & Initialization (`tasks/snli/encoding_snli.py`, `text/quantum_text_encoder.py`)
- Uses pretrained Livnium quantum embeddings (`../quantum_embed/quantum_embeddings_final.pt`).
- Tokenization + lookup from checkpoint vocab; mean pooling to sentence vectors.
- Initial state: `h0 = v_p + v_h (+ noise)`; `dim` comes from the quantum checkpoint so collapse and head stay aligned.

## SNLI Head Geometry (`tasks/snli/head_snli.py`)
Concatenated features before the head MLP:

```
align        = cos(v_p, v_h)
opp          = cos(-v_p, v_h)
energy       = 9 * (1 + align) / 2
expose_neg   = (1 - align) / 2
dist_p_h     = ||v_h - v_p||_2
r_p, r_h     = ||v_p||_2, ||v_h||_2
r_final      = ||h_final||_2
align_neu_p  = cos(v_p, n_hat) * neutral_scale
align_neu_h  = cos(v_h, n_hat) * neutral_scale
features = [h_final, align, opp, energy, expose_neg,
            dist_p_h, r_p, r_h, r_final,
            align_neu_p, align_neu_h]
logits = MLP(features) -> (E, N, C)
```

`n_hat` is a learned neutral direction; `neutral_scale` controls its weight.

## Training Loop Highlights (`training/train_snli_vector.py`)
- Data curation: `load_snli_data` drops invalid labels and ambiguous pairs; `--max-samples` for smoke tests.
- Balancing: neutral oversampling (`--neutral-oversample`) and neutral class weight (`--neutral-weight`); optional label smoothing.
- Dynamic basins: on by default; disable via `--disable-dynamic-basins`. Basin spawn/prune thresholds configurable with `--basin-*`.
- Collapse update: no learned MLP; forces are purely physics-based.
- Entropy pressure (optional): `--learn-entropy-pressure` adds a tiny head that outputs a non-negative scalar; `--entropy-budget` caps removal per step. Defaults keep pressure at 0.
- Checkpoint: `best_model.pt` stores collapse engine, encoder, head, optional basin field, optimizer, args, and `use_dynamic_basins`.

## Evaluation (`chat/test_snli_vector.py`)
- Restores checkpoint, rebuilds quantum encoder from saved path.
- Always uses static collapse (no label-conditioned routing), reports accuracy, confusion matrix, per-class accuracy; optional JSONL of misclassifications.

## Geometry Visualization (`chat/visualize_snli_geometry.py`)
- Builds E/N/C plane basis:
  - `u1 = normalize(e - n)`
  - `u2 = normalize((c - n) - proj_{u1}(c - n))`
- Projects anchors, dynamic basins, initial/final states, and sampled trajectories via `(x, y) = (h_hat dot u1, h_hat dot u2)`.
- Saves plot; can run in trajectories-only mode or with the dark "brain wires" style.

## Key Files
- `core/vector_collapse_engine.py`, `core/physics_laws.py`, `core/basin_field.py`
- `tasks/snli/encoding_snli.py`, `tasks/snli/head_snli.py`
- `training/train_snli_vector.py`, `chat/test_snli_vector.py`, `chat/visualize_snli_geometry.py`
