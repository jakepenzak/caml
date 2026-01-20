# CaML: Scope, Architecture, and Thin-Slice Plan

## Reframed vision (v0)

**CaML provides sklearn-style causal estimators that choose safe default specifications, run robust inference, and return standardized effect results across common causal designs.**

What CaML *is* (near-term):
- A **workflow + contracts** library for causal estimation in Python.
- A **unified API + unified result objects** (effects, uncertainty, diagnostics).
- A **backend orchestration layer** wrapping proven libraries (EconML/DoubleML/statsmodels/linearmodels) when possible.

What CaML is *not* (near-term):
- A reimplementation of the full econometrics ecosystem.
- A “push-button causal truth machine” that can validate identification.

The differentiation is “**standardize causal workflows, not models**”: CaML standardizes data specification, estimands, inference outputs, diagnostics, and reporting. Estimation is delegated to mature backends when possible.

---

## Product narrative (why users care)

Applied data scientists often fail at causal tooling because:
- they don’t know which regression specification is correct,
- they mix estimands (ATE vs ATT vs event-study curves) without noticing,
- and different causal libraries return outputs with incompatible conventions.

CaML addresses this by:
- requiring **minimal column declarations** (`y`, `treatment`, and when needed `unit`, `time`),
- providing **opinionated defaults** for specifications and inference,
- and returning **standardized `EffectResult` objects** regardless of backend.

---

## Core contracts (the unifying layer)

### 1) Data spec contract
All estimators should accept an explicit data spec; users provide column names.

A conceptual schema:
- `outcome: str`
- `treatment: str`
- `covariates: list[str] | None`
- `heterogeneity_features: list[str] | None`
- `unit: str | None` (required for panel/DiD)
- `time: str | None` (required for panel/DiD)
- `cluster: str | None` (defaults to `unit` for DiD)
- `weights: str | None`

Notes:
- “Sklearn-feel”: do not require formulas.
- Allow formulas only for debug/baseline estimators (optional).

### 2) Estimator contract
A sklearn-style causal estimator should provide:
- `fit(df) -> self`
- `effect(df=None) -> pd.Series | np.ndarray | pd.DataFrame`
  - Unit-level effects where meaningful (CATE). For estimators where it doesn’t make sense (DiD), raise a clear error.
- `estimate(estimand=..., df=None, query=None, **kwargs) -> EffectResult`
  - Scalar/group estimands, plus structured outputs like event-study paths.
- `summary() -> str | pd.DataFrame`
- `get_diagnostics() -> dict`

Avoid overloading `predict()`:
- Reserve `predict()` for outcome prediction where meaningful.
- Prefer `effect()` as the causal analogue of prediction.

### 3) Result contract
Standardize outputs through an `EffectResult` (and specializations).

Common fields:
- `estimand: str` (e.g., `"att"`, `"ate"`, `"cate"`, `"event_study"`, `"att_by_cohort"`)
- `value`: scalar or tabular
- `stderr`: same shape or `None`
- `ci(level=0.95)`: computed or stored
- `inference`: metadata (`cov_type`, `cluster` columns, bootstrap info, etc.)
- `n_obs`, `n_clusters`
- `diagnostics`: dict (overlap, cohort sizes, pretrend tests)
- `spec_`: the resolved spec CaML used (what it chose “for you”)
- `backend_`: the underlying backend fitted object(s)

Specializations that will matter early:
- `ATTResult` (scalar ATT + inference)
- `EventStudyResult` (relative_time-indexed table with estimates/SE/CI)
- `CATEPredictionResult` (unit-level effects + optional intervals)

---

## How existing code fits

### `AutoCATE`
Role: **automated heterogeneity engine** (unconfoundedness; EconML backend).
- Primary product output: `effect(df_new)` returns CATE predictions.
- Secondary: `estimate("ate")` returns average effect (with inference if supported).
- Must conform to the common `EffectResult` contract while retaining access to the best EconML estimator.

### `InteractiveLinearRegression`
Role: **explainable baseline** for heterogeneous effects (unconfoundedness).
- Outputs `estimate("ate")`, `estimate("gate")`, and `effect(df)`.
- This becomes the “sanity-check” estimator that complements AutoCATE.

---

## AutoDiD: v0 specification

User constraints agreed:
- Binary treatment, **single adoption date per unit** (staggered adoption allowed).
- Users explicitly provide `unit` and `time` column names.
- First-class outputs: overall ATT + event-study curve + cohort-specific effects.

### What AutoDiD should do (honest and safe)
AutoDiD must avoid pretending that TWFE/event-study/CS are “just different models for the same object.” Instead:
- Users declare **target estimand(s)** (`att`, `event_study`, `att_by_cohort`).
- AutoDiD chooses an appropriate backend implementation per estimand and treatment timing structure.
- TWFE should exist as a baseline and for diagnostics, but the pipeline should prefer heterogeneity-robust approaches when available.

### AutoDiD interface (target)
- `AutoDiD.fit(df) -> self`
- `AutoDiD.estimate("att") -> ATTResult`
- `AutoDiD.estimate("event_study") -> EventStudyResult`
- `AutoDiD.estimate("att_by_cohort") -> EffectResult` (tidy table)
- `AutoDiD.get_diagnostics() -> dict`

### AutoDiD diagnostics (always)
- adoption/cohort summary table
- counts of treated/control by time
- relative-time support by cohort
- pre-trends estimates/joint test outputs where feasible
- inference settings used (cluster column, cov_type)

### Implementation strategy
- Wrap backend libraries for TWFE/event study and robust inference (statsmodels/linearmodels).
- Add Callaway–Sant’Anna style estimator as either:
  - a wrapper if a stable Python package is selected, or
  - a minimal CaML-native implementation (ATTgt + aggregation + SE strategy) if no suitable backend exists.

---

## Proposed repo structure (target state)

Keep convenience namespaces (e.g., `caml.cross_section`) but move the “brain” into `core/`, `designs/`, and `adapters/`.

```text
caml/
  core/
    contracts.py      # base estimator + EffectResult types
    specs.py          # data/design specs + default resolution
    estimands.py      # canonical estimand names + validators
    splits.py         # cross-fitting + GroupKFold utilities
    inference.py      # CI helpers, cov_type normalization
    diagnostics.py    # overlap, DiD cohort/pretrend helpers

  designs/
    unconfoundedness/
      interactive_linear.py
      auto_cate.py
      dml_ate.py          # thin-slice addition

    did/
      twfe.py
      auto_did.py         # orchestrator
      cs.py               # Callaway–Sant’Anna adapter/impl

  adapters/
    econml.py
    statsmodels.py
    linearmodels.py

  reporting/
    summary.py

  extensions/
    synthetic_data.py
    plots.py
```

---

## Thin-slice roadmap

### Phase 0: Contract stabilization (high priority)
- Define canonical estimands and output shapes.
- Introduce `EffectResult` and require new work to return it.
- Decide standard column roles (`outcome`, `treatment`, `unit`, `time`, `covariates`).

### Phase 1: Unconfoundedness suite (build around what exists)
- Keep and stabilize `InteractiveLinearRegression` as the explainable baseline.
- Keep and stabilize `AutoCATE` as the automated heterogeneity engine.
- Add a thin DML ATE/ATT estimator (wrap DoubleML/EconML initially) that conforms to the contract.

### Phase 2: DiD suite (AutoDiD MVP)
- Implement TWFE + event-study baseline with cluster-robust SE.
- Add cohort/adoption diagnostics.
- Add a heterogeneity-robust DiD estimator (Callaway–Sant’Anna style) for correct ATT and event-study style outputs.
- Implement `AutoDiD` as an orchestrator that produces:
  - overall ATT
  - event-study curve
  - ATT by cohort

### Phase 3: Hardening
- Golden tests comparing CaML outputs to backend outputs on synthetic datasets.
- Version pinning strategy for backends.
- Documentation of estimand semantics and what is/ isn’t checked.

---

## Guardrails (to prevent scope creep)

- Support a **small number of designs** extremely well.
- Prefer wrapping mature inference implementations.
- Always expose the backend object.
- Treat “auto” estimators as **workflow engines**, not magic.
- Add estimators only when contracts are proven and outputs stay consistent.
