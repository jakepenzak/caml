# CaML Development Guide for AI Agents

## TL;DR

**CaML** is being refactored into a production-ready package for automated CATE estimation wrapping 14 EconML estimators with custom scoring, dual AutoML (FLAML + Optuna), and protocol-based architecture.

**Status**: Phases 1-3/7 complete (~55%), Phase 4 (Scoring & Cross-Fitting) starting now.

**Immediate Priority**: Implement `caml/scorers/r_loss.py` and `caml/samplers/cross_fit.py` (Phase 4, HIGH priority).

## Current Status

| Phase | Status | Priority |
|-------|--------|----------|
| Phase 1: Data Structures & Protocols | ✅ Complete (100%) | - |
| Phase 2: Estimator Wrappers | ✅ Complete (100%) | - |
| Phase 3: Nuisance Models | ✅ Complete (100%) | - |
| Phase 4: Cross-Fitting & Scoring | 🔶 Not Started (0%) | **HIGH** |
| Phase 5: AutoML Integration | 🔶 Not Started (0%) | MEDIUM |
| Phase 6-7: Native Estimators & Polish | 🔶 Not Started (0%) | LOW |

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
| Directories | `samplers/`, `scorers/`, `registry/` | `sampling/`, `validation/`, `scoring/`, `modeling/` |
| Directory (native) | `estimators/native/` | `estimators/benchmark/` |
| Files | `uplift_.py`, `_validation.py` | `uplift.py`, `validation.py` |
| Methods | `effect()`, `from_dataframe()` | `predict_cate()` |
| Protocols Location | `estimators/base_estimator.py` | `protocols/estimator.py` (deprecated) |
| Classes | `EstimatorCapabilities` (typo!) | `EstimatorCapabilities` |
| Nuisance Attributes | `treatment_model_`, `outcome_model_`, `regression_model_` | `propensity_model_` |

## Documentation Standards

**NumPy-style docstrings required for all public APIs:**

- **CRITICAL**: Docstrings are MANDATORY before marking any phase as complete
- Rendered via quarto.qmd, so leverage markdown syntax and advanced features (callout-blocks, etc.) when warranted.
- **PRIORITIZE BEAUTIFUL DOCUMENTATION. BUT DO NOT BE OVERLY VERBOSE. KEEP EXAMPLES AND WORDING AT A "COMPREHENSIVE MINIMUM"**
- Include "See Also" section where applicable. Link to respective file like this:
```markdown
See Also
--------
[`TreatmentType`](data_enums.qmd#caml.data.data_enums.TreatmentType) : Enum defining treatment variable types.

[`OutcomeType`](data_enums.qmd#caml.data.data_enums.OutcomeType) : Enum defining outcome variable types.
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

## Phase 4 Implementation Plan (IMMEDIATE)

**Next files to implement in order:**

1. `caml/samplers/cross_fit.py` (200 lines) — Cross-fitting engine for orthogonal scores
2. `caml/scorers/base_scorer.py` (50 lines) — BaseCateScorerMixin abstract class
3. `caml/scorers/r_loss.py` (150 lines) — R-loss scorer for CATE model selection
4. `caml/scorers/dr_loss.py` (180 lines) — Doubly-robust loss scorer
5. `caml/scorers/uplift_.py` (120 lines) — Qini, AUUC uplift metrics
6. `caml/scorers/policy.py` (100 lines) — Policy value evaluation
7. `caml/scorers/calibration.py` (80 lines) — CATE calibration diagnostics
8. `caml/scorers/diagnostics.py` (60 lines) — Stability and sensitivity metrics
9. `caml/samplers/splitters.py` (40 lines) — Splitter utilities (KFold, GroupKFold, etc.)
10. `caml/samplers/bootstrap.py` (100 lines) — Bootstrap inference

**Dependencies**: Phase 3 complete (nuisance tuner available). Can start immediately.

**Critical**: Cross-fitting (`cross_fit.py`) must be implemented before R-loss/DR-loss scorers, as they depend on out-of-fold predictions.

**Completion Criteria**: All public classes/methods must have complete NumPy-style docstrings with runnable examples.

## Key Files Reference

**Long-form documentation (only reference if prompt necesitates needed):**
- `.context/REFACTORING_PLAN.md` — Master plan with metrics and phase details
- `.context/CODE_EXAMPLES.md` — Code patterns for all 7 phases

**Testing Docs:**
- `.context/TESTING_GUIDELINES.md` — Testing standards and examples. Always refer before writing tests.

**Reference implementations:**
- `caml/data/dataset.py` — Complete NumPy docstrings, validation patterns
- `caml/estimators/base_estimator.py` — Protocols (AutoCateEstimator, InferenceProvider), BaseWrapperMixin
- `caml/estimators/wrappers/dml.py` — Complete DML wrapper implementation (779 lines)
- `caml/nuisance/tuner.py` — FLAML-based nuisance tuner with Ray/Spark support (224 lines)
- `caml/nuisance/spec.py` — NuisanceTunerSpec dataclass (54 lines)
- `caml/registry/registry.py` — Estimator auto-discovery and compatibility filtering (162 lines)
- `caml/data/_validation.py` — Reusable validation utilities
- `caml/extensions/synthetic_data.py` — Test data generator

**Schemas & structures:**
- `caml/data/data_enums.py` — Enums (`TreatmentType`, `OutcomeType`, `Estimand`)
- `caml/inference/results.py` — `InferenceResult` dataclass
- `caml/inference/inference_enums.py` — `InferenceType` enum

## Common Pitfalls

1. **Wrong method names**: Use `effect()` not `predict_cate()`
2. **Wrong directories**: Use `samplers/` not `sampling/` or `validation/`, `scorers/` not `scoring/`, `registry/` not `modeling/`
3. **Wrong protocols location**: Protocols are in `estimators/base.py` NOT `protocols/estimator.py` (deprecated)
4. **Missing BaseWrapperMixin**: All wrappers inherit from `BaseWrapperMixin` ABC - provides `__getattr__` delegation
5. **Old type hints**: Use `int | float` not `Union[int, float]`
6. **Plain code blocks**: Use ` ```{python} ` not ` ```python `
7. **Manual test data**: Use `SyntheticDataGenerator` not `np.random`
8. **Missing validation**: Always use `caml/data/_validation.py` utilities
9. **Known typo**: Keep `EstimatorCapabilities` (missing 'i') for consistency
10. **Nuisance model naming**: Use `treatment_model_` not `propensity_model_` (attribute names)

---

**Last updated**: 2026-01-26
**Next task**: `caml/samplers/cross_fit.py` and `caml/scorers/r_loss.py`
**Detailed docs**: See `.context/REFACTORING_PLAN.md` and `.context/CODE_EXAMPLES.md`
