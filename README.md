<div align="center">

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
[![Codacy Badge](https://app.codacy.com/project/badge/Grade/cd6cc54c704e4a7aafe20f851bc39236)](https://app.codacy.com/gh/jakepenzak/caml/dashboard?utm_source=gh&utm_medium=referral&utm_content=&utm_campaign=Badge_grade)
[![codecov](https://codecov.io/gh/jakepenzak/caml/graph/badge.svg?token=UBABBZXO85)](https://codecov.io/gh/jakepenzak/caml)

**Ca**usal **M**achine **L**earning

</div>

## Welcome!

CaML provides a high-level API for an _opinionated_ framework in performing Causal ML to estimate Average Treatment Effects (ATEs),
Group Average Treatment Effects (GATEs), and Conditional Average Treatment Effects (CATEs), and to provide mechanisms to utilize these
models for out of sample validation, prediction, & policy prescription.

The codebase is comprised primarily of extensions & abstractions over top of [EconML](https://github.com/py-why/EconML)
& [DoubleML](https://docs.doubleml.org/stable/api/generated/doubleml.datasets.make_confounded_irm_data.html#doubleml.datasets.make_confounded_irm_data)
with techniques motivated heavily by [Causal ML Book](https://causalml-book.org/) and additional research.

## Background

The origins of CaML are rooted in a desire to develop a set of helper tools to abstract and streamline techniques
& best pratices in Causal ML/Econometrics for estimating ATEs, GATEs, and CATEs, along with policy prescription. In
addition, we seek to provide a framework for validating & scoring these models on out of sample data to help
set the foundations for an AutoML framework for CATE models.

As we began working on these helper tools, we begun to see the value in reformulating this framework into a reusable
package for wider use amongst the community and to provide an opinionated framework that can be integrated into productionalized
systems, particularly experimentation platforms, for efficient estimation of causal parameters for reporting & decision-making
purposes.

**All of the standard assumptions for causal inference still apply in order for these tools & techniques to provide
unbiased inference.** A great resource for the CausalML landscape is the [CausalML book](https://www.causalml-book.org/) written and
publicly available generously by V. Chernozhukov, C. Hansen, N. Kallus, M. Spindler, & V. Syrgkanis.

Given a key motivation is to provide a tool for productionalized systems, we are building this package with interoperability
and extensibility as core values. As of now, the tools utilized still rely on in-memory datasets for estimation (via [EconML](https://github.com/py-why/EconML)
for causal models & [flaml](https://microsoft.github.io/FLAML/) for AutoML of nuissance functions), but we leverage Ray & Spark for distributing
certain processes where appropriate and if available for the user.
