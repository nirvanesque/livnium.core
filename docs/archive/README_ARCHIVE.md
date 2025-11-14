# ✅ LIVNIUM Archive (Experimental / Broken / Legacy Components)

### *Final Polished Version*

This directory contains **experimental**, **incomplete**, or **legacy** components of the LIVNIUM quantum-inspired system.

These files are kept exclusively for **historical and research reference**.

They are **not part of the working system**, and may be:

* incomplete
* mathematically incorrect
* non-functional
* misleading if used directly
* based on shortcuts rather than simulation
* incompatible with the current architecture
* unable to handle realistic entanglement
* early prototypes replaced by better designs

---

# 📁 Archive Sections

---

## **1. `broken_tests/`**

Tests that fail due to memory limits, entanglement explosion, or incorrect assumptions.

**Files:**

* `REAL_TEST_MAX_ENTANGLEMENT.py` — fails with exit code 137 (memory kill)
* `example_500_qubit.py` — crashes during 500-qubit entanglement
* `run_500_qubits.py` — relies on impossible entanglement scaling

**Reason for Archive:**

These tests attempt scenarios that classical hardware fundamentally cannot support.

They demonstrate limitations, not real system behavior.

---

## **2. `broken_simulators/`**

Simulators that:

* require exponential memory (2^N)
* aggressively truncate entanglement
* produce incorrect or unstable results

**Files:**

* `livnium_entanglement_preserving.py` — real O(2^N) simulator, usable only for ~15–20 qubits
* `projection_based_simulator.py` — heavy truncation → incorrect output

**Reason for Archive:**

These simulators either misrepresent their capabilities or generate inaccurate results.

Not safe for production use.

---

## **3. `calculators/`**

Algorithms that use **closed-form mathematical formulas** instead of real simulation.

Examples include shortcuts like:

* `1 / √N`
* `sin²(kθ)`
* direct QFT eigenvalue formulas

**Files:**

* `qft_30_qubit.py` — formula-based QFT, no amplitude simulation
* `grovers_26_qubit_geometry.py` — uses sin² formula
* `grovers_26_qubit.py` — formula-based amplitude prediction

**Reason for Archive:**

These are **calculators**, not simulators.

They are valid mathematically but should not be confused with quantum-inspired simulation engines.

---

## **4. `legacy/`**

Older experimental systems that have been replaced by the unified architecture.

**Contents:**

* `livnium_core_demo/`
* `quantum_2/`
* `quantum_computer/`
* `quantum_computer_code.zip`
* `livnium-quantum-7b27e33.tar.gz`

**Reason for Archive:**

These are early prototypes.

They are preserved for historical continuity only.

---

## **5. `incomplete_tests/`**

Tests that were never finished or no longer match the current architecture.

**Files:**

* `test_geometry_capacity.py` — incomplete, import issues
* `test_error_rate.py` — inconsistent, unfinished

**Reason for Archive:**

The tests were abandoned before completion.

They are not compatible with the unified codebase.

---

# ⚠️ Important Notes

* These files are **not** used by the working system.
* Do **not** import or reference them in production code.
* Some contain mathematically incorrect or impossible assumptions.
* Use them only for research purposes (e.g., understanding failure cases).

---

# ✅ What Remains in the Working System

Your main `quantum/` directory now contains only:

* **working modules**
* **functional simulators** (with documented limitations)
* **real algorithms**
* **accurate classical amplitude simulations**
* **verified tests**
* **clean, maintainable code**
* **honest capability constraints**

This ensures scientific clarity and protects the credibility of the LIVNIUM system.

---

# 📚 Research Use Guidelines

If you wish to study archived components:

1. Treat them as failed prototypes
2. Do not reuse their designs directly
3. Use them to understand what **doesn't** work
4. Compare them against the stable system for insights

---

### **Archive Organization Complete — Clean, Professional, Stable**

Your codebase now reflects an academically responsible, logically structured research system.
