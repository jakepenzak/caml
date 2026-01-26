# CaML Development Guide for AI Agents

## TL;DR

**CaML** is being refactored into a production-ready package for automated CATE estimation wrapping 14 EconML estimators with custom scoring, dual AutoML (FLAML + Optuna), and protocol-based architecture.

**Status**: Phase 1/7 complete (~30%), Phase 2 (Estimator Wrappers) starting now.

**Immediate Priority**: Implement `caml/estimators/wrappers/dml.py` (Phase 2, HIGH priority).

## Current Status

| Phase | Status | Priority |
|-------|--------|----------|
| Phase 1: Data Structures & Protocols | ✅ Complete (100%) | - |
| Phase 2: Estimator Wrappers | ⚠️ Partial (~7%) | **HIGH** |
| Phase 3: Nuisance Models | 🔶 Not Started (0%) | **HIGH** |
| Phase 4: Cross-Fitting & Scoring | 🔶 Not Started (0%) | MEDIUM |
| Phase 5: AutoML Integration | 🔶 Not Started (0%) | MEDIUM |
| Phase 6-7: Registry & Polish | 🔶 Not Started (0%) | LOW |

## Core Principles

1. **EconML-First**: Wrap 14 EconML estimators (DML, DR, Meta-learners, ORF, Causal Forest)
2. **Protocol-Based**: Use `CATEEstimator` + `InferenceProvider` protocols for extensibility
3. **Custom Scoring**: R-loss, DR-loss, Qini, policy value, calibration metrics
4. **Dual AutoML**: FLAML for nuisance models, Optuna for CATE selection
5. **Modern Python**: Python 3.10+ type hints (`int | float`), dataclasses, NumPy docstrings

## Naming Conventions (CRITICAL)

**Use actual implementation names, NOT refactoring plan names:**

| Type | ✅ USE | ❌ NOT |
|------|-------|--------|
| Directories | `sampling/`, `registry/` | `validation/`, `modeling/` |
| Files | `uplift_.py`, `_validation.py` | `uplift.py`, `validation.py` |
| Methods | `effect()`, `from_dataframe()` | `predict_cate()` |
| Classes | `EstimatorCapabilities` (typo!) | `EstimatorCapabilities` |

## Documentation Standards

**NumPy-style docstrings required for all public APIs:**

- **CRITICAL**: Docstrings are MANDATORY before marking any phase as complete
- Rendered via quarto.qmd, so leverage markdown syntax and advanced features (callout-blocks, etc.) when warranted.
- **PRIORITIZE BEAUTIFUL DOCUMENTATION. BUT DO NOT BE OVERLY VERBOSE. KEEP EXAMPLES AND WORDING AT A "COMPREHENSIVE MINIMUM"**
- Include "See Also" section where applicable. Link to respective file like this:
```markdown
See Also
--------
[`TreatmentType`](data_schema.qmd#caml.data.data_schema.TreatmentType) : Enum defining treatment variable types.

[`OutcomeType`](data_schema.qmd#caml.data.data_schema.OutcomeType) : Enum defining outcome variable types.
```
- Always include module docstring
- Include: Parameters, Returns, Raises, Examples (type hints in parameters not needed, taken from fn signature)
- Any reference to a code object in descriptions, etc. should be outlined with "``"
- Use Quarto code blocks: ` ```{python} ` (NOT plain ` ```python `)
- Make examples runnable with `SyntheticDataGenerator`
- Type hints: Modern syntax (`int | float`, `str | None`)
- Code references: Use backticks (``CATEEstimator``)

**Example:**
```python
def effect(self, x: np.ndarray) -> np.ndarray:
    """Estimate CATE for given covariates.

    Parameters
    ----------
    x
        Covariate matrix.

    Returns
    -------
    np.ndarray
        Estimated CATE values.

    Examples
    --------
    ```{python}
    from caml.extensions.synthetic_data import SyntheticDataGenerator

    gen = SyntheticDataGenerator(n_samples=500)
    data = gen.generate()
    # ... use data
    ```
    """
```

## Quick Start Checklist

When starting work:

1. **Check status**: Review `.context/REFACTORING_PLAN.md` for latest phase details
2. **Code patterns**: Review `.context/CODE_EXAMPLES.md` for implementation examples
3. **Reference implementations**:
   - `caml/data/dataset.py` — Complete docstring examples
   - `caml/protocols/estimator.py` — Protocol definitions
   - `caml/data/_validation.py` — Validation utilities
4. **Use `SyntheticDataGenerator`** from `caml/extensions/synthetic_data.py` for all testing/examples
5. **Validate thoroughly** using `caml/data/_validation.py` utilities
6. **Follow naming** from table above (e.g., `effect()` not `predict_cate()`)

## Phase 2 Implementation Plan (IMMEDIATE)

**Next files to implement in order:**

1. `caml/estimators/wrappers/dml.py` (200 lines) — Wrap 4 DML variants from EconML
2. `caml/estimators/wrappers/dr.py` (160 lines) — Wrap 2 DR learners from EconML
3. `caml/estimators/wrappers/meta.py` (180 lines) — Wrap S/T/X-Learners from EconML
4. `caml/estimators/wrappers/orf.py` (120 lines) — Wrap Orthogonal Random Forest from EconML
5. `caml/registry/registry.py` (100 lines) — Estimator auto-discovery and compatibility filtering

**Dependencies**: Phase 1 complete (protocols defined). Can start immediately.

**Completion Criteria**: All public classes/methods must have complete NumPy-style docstrings with runnable examples.

## Key Files Reference

**Must-read documentation:**
- `.context/REFACTORING_PLAN.md` — Master plan with metrics and phase details
- `.context/CODE_EXAMPLES.md` — Code patterns for all 7 phases

**Reference implementations:**
- `caml/data/dataset.py` — Complete NumPy docstrings, validation patterns
- `caml/protocols/estimator.py` — `CATEEstimator` interface
- `caml/data/_validation.py` — Reusable validation utilities
- `caml/extensions/synthetic_data.py` — Test data generator

**Schemas & structures:**
- `caml/data/data_schema.py` — Enums (`TreatmentType`, `OutcomeType`, `Estimand`)
- `caml/inference/results.py` — `InferenceResult` dataclass
- `caml/inference/inference_schema.py` — `InferenceType` enum

## Common Pitfalls

1. **Wrong method names**: Use `effect()` not `predict_cate()`
2. **Wrong directories**: Use `sampling/` not `validation/`, `registry/` not `modeling/`
3. **Old type hints**: Use `int | float` not `Union[int, float]`
4. **Plain code blocks**: Use ` ```{python} ` not ` ```python `
5. **Manual test data**: Use `SyntheticDataGenerator` not `np.random`
6. **Missing validation**: Always use `caml/data/_validation.py` utilities
7. **Known typo**: Keep `EstimatorCapabilities` (missing 'i') for consistency

---

**Last updated**: 2026-01-24
**Next task**: `caml/estimators/wrappers/dml.py`
**Detailed docs**: See `.context/REFACTORING_PLAN.md` and `.context/CODE_EXAMPLES.md`
