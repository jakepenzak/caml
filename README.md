<div align="center">
<center>

<img src="https://raw.githubusercontent.com/jakepenzak/caml/main/docs/assets/main_logo.svg" align="center" alt="CaML Logo" height="auto" width=500px/>

<br>
<br>

[![image](https://img.shields.io/pypi/v/caml.svg)](https://pypi.python.org/pypi/caml)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/caml)](https://pypi.python.org/pypi/caml)
[![lifecycle](https://img.shields.io/badge/Lifecycle-Experimental-blue?style=flat)](https://img.shields.io/badge/Lifecycle-Experimental-blue?style=flat)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
<br>
[![Caml CI/CD](https://github.com/jakepenzak/caml/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/jakepenzak/caml/actions/workflows/ci.yml)
[![Build & Publish Docs](https://github.com/jakepenzak/caml/actions/workflows/docs.yml/badge.svg)](https://github.com/jakepenzak/caml/actions/workflows/docs.yml)
[![Pre-Commit & Linting Checks](https://github.com/jakepenzak/caml/actions/workflows/lint.yml/badge.svg?branch=main)](https://github.com/jakepenzak/caml/actions/workflows/lint.yml)
<br>
<a href="https://app.codacy.com/gh/jakepenzak/caml/dashboard?utm_source=gh&utm_medium=referral&utm_content=&utm_campaign=Badge_grade"><img src="https://app.codacy.com/project/badge/Grade/cd6cc54c704e4a7aafe20f851bc39236"/></a>
[![codecov](https://codecov.io/gh/jakepenzak/caml/graph/badge.svg?token=UBABBZXO85)](https://codecov.io/gh/jakepenzak/caml)

</center>
</div>

<br>

## 🎯 What is CaML?

**CaML** is a framework for **automated heterogeneous treatment effect estimation and validation**. It combines state-of-the-art causal inference methods with AutoML optimization to deliver robust, validated CATE (Conditional Average Treatment Effect) models at scale.

### Why CaML?

**The Challenge**: Estimating heterogeneous treatment effects requires choosing among dozens of estimators, tuning nuisance models, and validating with specialized metrics. This process is complex, time-consuming, and error-prone.

**The Solution**: CaML automates the entire pipeline—from estimator selection to model validation—while maintaining rigorous statistical foundations and production-grade reliability, modularity, and extensibility.

> **⚠️ Important**: CaML provides tools, not magic. Standard causal inference assumptions (unconfoundedness, overlap, SUTVA) must hold for valid inference.


## ✨ Key Features

### 🔬 **Rigorous Causal Inference**
- **14 EconML Estimators**: DML, DR learners, Meta-learners, Causal Forests, Orthogonal Random Forests
- **Custom CATE Scoring**: R-loss, DR-loss, Qini curves, policy value, calibration diagnostics

### 🤖 **Dual AutoML Architecture**
- **FLAML for Nuisance Models**: Automated tuning of propensity and outcome models with Ray/Spark support
- **Optuna for CATE Selection**: Hyperparameter optimization across estimator families
- **Intelligent Model Registry**: Automatic compatibility filtering based on data characteristics

### 🏗️ **Production-Ready Design**
- **Protocol-Based**: Extensible interfaces for custom estimators and scorers
- **Type-Safe**: Modern Python 3.10+ with comprehensive type hints
- **Validated Data Structures**: Rich metadata tracking and automatic validation

## 🏛️ Architecture Highlights

CaML is built on three core pillars:

### 1️⃣ **EconML-First Philosophy**
We **wrap** proven estimators from [EconML](https://github.com/py-why/EconML) rather than reimplementing them, ensuring statistical rigor while adding AutoML capabilities and production tooling.

### 2️⃣ **Custom Validation Framework**
Traditional ML metrics fail for CATE models. CaML implements specialized scoring and loss functions, including:

- **Orcale** - Gold-standard metrics comparing predictions to true CATE. Only computable when ground truth is available (simulations).
- **Plug-In** - Surrogate metrics using a learned reference CATE model.
- **Pseudo-Outcome** - Metrics based on transformed outcomes that provide unbiased CATE proxies under model correctness.
- **Ranking (Proxy)** - Relative performance proxies that rank estimators by reformulating PEHE into identifiable quantities.
- **Ranking (Curve)** - Qini curves, uplift metrics, and calibration metrics that evaluate the ability to rank individuals by treatment effect magnitude.
- **Policy** - Decision-value metrics assessing expected outcomes under treatment assignment policies.

### 3️⃣ **Protocol-Based Extensibility**
Every component follows clear protocols and base classes, making it trivial to:

- Add custom estimators alongside EconML wrappers
- Implement domain-specific scoring metrics

## 📚 Documentation

- **API Reference**: [Coming Soon]
- **User Guide**: [Coming Soon]
- **Tutorials**: [Coming Soon]
- **Research Papers**: [Coming Soon]

## 🤝 Contributing

CaML is an experimental project welcoming contributions! Key areas:

- **CATE Estimators**: Additional estimators
- **Scoring Metrics**: Domain-specific CATE validation approaches
- **Benchmarking**: Comparative studies on synthetic/real datasets
- **Documentation**: Examples, tutorials, case studies
- **Parallelization**: Distributed training support

See [contribution guidelines](https://caml-docs.com/05_Contributors/getting_started.html)

---

<div align="center">
<center>

**Built with ❤️ for the causal inference community**

[⭐ Star us on GitHub](https://github.com/jakepenzak/caml) • [📖 Read the Docs](#) • [🐛 Report Issues](https://github.com/jakepenzak/caml/issues)

</center>
</div>
