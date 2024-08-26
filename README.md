![](https://img.shields.io/badge/version-0.0.0.dev3-purple)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
![](https://img.shields.io/badge/Lifecycle-Experimental-blue?style=flat)

<h1>
<img src="docs/assets/main_logo.svg" align="center" alt="CaML Logo" height="auto" width=500px/>
</h1>

## Welcome!

CaML = **Ca**usal **M**achine **L**earning

CaML provides a high-level API for an *opinionated* framework in performing Causal ML to estimate Average Treatment Effects (ATEs), Group Average Treatment Effects (GATEs), and Conditional Average Treatment Effects (CATEs), and to provide mechanisms to utilize these models for out of sample prediction & policy prescription. 

The codebase is comprised primarily of extensions & abstractions over top of [EconML](https://github.com/py-why/EconML) & [DoubleML](https://docs.doubleml.org/stable/api/generated/doubleml.datasets.make_confounded_irm_data.html#doubleml.datasets.make_confounded_irm_data) with techniques motivated heavily by [Causal ML Book](https://causalml-book.org/) and additional research. 

## Background
The origins of CaML are rooted in a desire to develop a set of helper tools to abstract and streamline techniques & best pratices in Causal ML/Econometrics for estimating ATEs, GATEs, and CATEs, along with policy prescription.   

As we began working on these helper tools, we begun to see the value in reformulating this framework into a reusable package for wider use amongst the community and to provide an opinionated framework that can be integrated into productionalized systems, particularly experimentation platforms, for efficient estimation of causal parameters for reporting & decision-making purposes.

Admittedly, we were tempted to include the term "Auto" in the name of this package (e.g., AutoCATE, AutoCausal, etc.), but we quickly realized the potential for misapplication & naive usage that could arise from that type of "branding." Indeed, the misapplication of many Causal AI/ML techniques is all too commonplace in the data science community. **All of the standard assumptions for causal inference still apply in order for these tools & techniques to provide unbiased inference.**

Given a key motivation is to provide a tool for productionalized systems, we are building this package with interoperability and extensibility as core values - a key motivation for using [Ibis](https://ibis-project.org/) to ensure we are backend agnostic for end users (e.g., instantiate with a pyspark dataframe and get a pyspark dataframe back). The degree of interoperability will be limited in scope at first, but we hope to expand this as the code base develops. As of now, the tools utilized still rely on in-memory datasets for estimation (via [EconML](https://github.com/py-why/EconML) for causal models & [flaml](https://microsoft.github.io/FLAML/) for AutoML of nuissance functions), but we leverage Ray & Spark for distributing certain processes where appropriate.  

