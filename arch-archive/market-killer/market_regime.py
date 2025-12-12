# market_regime.py

from enum import Enum


class Regime3(str, Enum):
    BULL = "bull"
    BEAR = "bear"
    NEUTRAL = "neutral"


class Regime5(str, Enum):
    BULL = "bull"
    BEAR = "bear"
    NEUTRAL = "neutral"
    PANIC = "panic"
    EUPHORIA = "euphoria"


def classify_regime3(
    alignment: float,
    tension: float,
    align_pos: float = 0.15,
    align_neg: float = -0.15,
    low_tension: float = 0.35,
    high_tension: float = 0.80,
) -> Regime3:
    """
    Simple 3-basin classifier used earlier.

    BULL   → reasonably positive alignment, tension not extreme.
    BEAR   → reasonably negative alignment, tension not tiny.
    NEUTRAL→ everything else.
    """
    if alignment >= align_pos and tension <= high_tension:
        return Regime3.BULL

    if alignment <= align_neg and tension >= low_tension:
        return Regime3.BEAR

    return Regime3.NEUTRAL


def classify_regime5(
    alignment: float,
    tension: float,
    align_pos: float = 0.15,
    align_strong_pos: float = 0.40,
    align_neg: float = -0.15,
    align_strong_neg: float = -0.40,
    low_tension: float = 0.35,
    med_tension: float = 0.80,
    panic_tension: float = 1.00,
) -> Regime5:
    """
    5-basin classifier: Bull / Bear / Neutral / Panic / Euphoria.

    Geometry (heuristic):

        - EUPHORIA → very positive alignment, medium–high tension
                      (strong, overheated bull run).
        - PANIC    → very negative alignment, very high tension
                      (crash / liquidation behaviour).
        - BULL     → positive alignment, tension not too high.
        - BEAR     → negative alignment, non-trivial tension.
        - NEUTRAL  → everything else (chop / equilibrium / unclear).

    All thresholds are tunable later based on empirical histograms.
    """

    # --- extremes first ---

    # PANIC: strong negative alignment + very high tension
    if alignment <= align_strong_neg and tension >= panic_tension:
        return Regime5.PANIC

    # EUPHORIA: strong positive alignment + at least medium tension
    if alignment >= align_strong_pos and tension >= med_tension:
        return Regime5.EUPHORIA

    # --- normal bull / bear ---

    # Bull: positive alignment, not too unstable
    if alignment >= align_pos and tension <= med_tension:
        return Regime5.BULL

    # Bear: negative alignment, some tension
    if alignment <= align_neg and tension >= low_tension:
        return Regime5.BEAR

    # --- fallback ---

    return Regime5.NEUTRAL
