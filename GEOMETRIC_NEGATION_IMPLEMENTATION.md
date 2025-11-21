# Geometric Negation Detection - Implementation

## ✅ What Was Implemented

Three geometric methods to detect negation, replacing the lexical token-matching hack:

### **METHOD 1: Geometric Opposition Detection**
**Location**: `NativeLogic.detect_negation_geometric()`

**How it works:**
- Checks if resonance is negative (< -0.2) → direct geometric opposition
- Checks if high overlap (> 0.4) but low resonance (< 0.3) → semantic gap = negation
- Returns `(has_negation, opposition_strength)`

**Example:**
- "happy" vs "not happy" → negative resonance → negation detected
- "runs" vs "doesn't run" → high overlap, low resonance → negation detected

---

### **METHOD 2: Word-Level Geometric Opposition**
**Location**: `NativeLogic.detect_negation_word_level()`

**How it works:**
- Compares each word pair's geometric similarity
- If `geo_sim < 0` → negative similarity = opposition
- Checks if negation words exist (fallback), but prefers geometric detection
- Returns `(has_negation, opposition_strength)`

**Example:**
- "happy" vs "sad" → negative geo_sim → opposition detected
- "not" word + negative geo_sim → stronger negation signal

---

### **METHOD 3: Resonance Pattern Detection**
**Location**: `NativeLogic.detect_negation_resonance_pattern()`

**How it works:**
- Pattern 1: Negative resonance (< -0.15) → direct opposition
- Pattern 2: High overlap (> 0.5) + very low resonance (< 0.2) → semantic gap
- Pattern 3: Moderate overlap (> 0.3) + negative resonance → opposition
- Returns `(has_negation, confidence)`

**Example:**
- "A dog runs" vs "A dog does not run" → negative resonance → negation
- "happy" vs "not sad" → high overlap, low resonance → semantic gap

---

## 🔄 How It's Used

**In `EntailmentDetector` and `ContradictionDetector`:**

```python
# GEOMETRIC NEGATION DETECTION (replaces lexical hack)
has_negation_geo, geo_opposition = NativeLogic.detect_negation_geometric(encoded_pair)
has_negation_word, word_opposition = NativeLogic.detect_negation_word_level(encoded_pair)
has_negation_res, res_confidence = NativeLogic.detect_negation_resonance_pattern(encoded_pair)

# Combine geometric signals (any method detecting negation = negation)
has_negation = has_negation_geo or has_negation_word or has_negation_res
opposition_strength = max(geo_opposition, word_opposition, res_confidence)

# Fallback to lexical for edge cases (but prefer geometric)
has_negation_lexical = NativeLogic.detect_negation(h_tokens)
if has_negation_lexical and not has_negation:
    has_negation = True
    opposition_strength = 0.5  # Medium confidence for lexical-only
```

**Strategy:**
1. Try all 3 geometric methods
2. If any detects negation → use it
3. If none detect but lexical does → use lexical as fallback
4. This ensures we prefer geometric but don't miss edge cases

---

## 🎯 Benefits

1. **Geometric-first**: Uses actual geometry/resonance, not token matching
2. **Multiple signals**: 3 independent methods increase robustness
3. **Fallback safety**: Lexical still available for edge cases
4. **No training required**: Works immediately with existing geometries
5. **Language-agnostic**: Geometric opposition works regardless of language

---

## 📊 Comparison

| Method | Input | Output | Pros | Cons |
|--------|-------|--------|------|------|
| **Lexical (old)** | Token list | Boolean | Simple, fast | Language-specific, not geometric |
| **Geometric Opposition** | Resonance + overlap | (bool, strength) | Pure geometric | May miss subtle cases |
| **Word-Level** | Word geometries | (bool, strength) | Fine-grained | Computationally heavier |
| **Resonance Pattern** | Resonance patterns | (bool, confidence) | Pattern-based | Requires tuning thresholds |

**Combined approach uses all 3 + lexical fallback = best of all worlds**

---

## 🧪 Testing

Test with:
```bash
python experiments/nli/test_golden_label_collapse.py \
    --premise "A dog runs" \
    --hypothesis "A dog does not run"
```

Expected: Should detect negation geometrically (negative resonance or word-level opposition)

---

## 🔧 Future Improvements

1. **Quantum Phase Inversion**: Make negation words flip quantum phase
2. **Polarity-Based**: Compute semantic polarity per word, check for negative polarity
3. **Learned Patterns**: Let training reinforce negation geometries
4. **Threshold Tuning**: Optimize thresholds based on test results

---

## 📝 Status

✅ **Implemented**: All 3 methods + integration into detectors
✅ **Tested**: No syntax errors, ready for testing
⚠️ **Needs**: Real-world testing on SNLI dataset to validate accuracy

