# Welcome to HyperSHAP's documentation! <img src="https://raw.githubusercontent.com/automl/hypershap/main/docs/source/_static/logo/hypershap-logo.png" alt="HyperSHAP Logo" align="right" height="200px"/>

[![Release](https://img.shields.io/github/v/release/automl/HyperSHAP)](https://img.shields.io/github/v/release/automl/hypershap)
[![Build status](https://img.shields.io/github/actions/workflow/status/automl/hypershap/main.yml?branch=main)](https://github.com/automl/hypershap/actions/workflows/main.yml?query=branch%3Amain)
[![Coverage Status](https://coveralls.io/repos/github/automl/HyperSHAP/badge.svg?branch=dev)](https://coveralls.io/github/automl/HyperSHAP?branch=dev)
[![Commit activity](https://img.shields.io/github/commit-activity/m/automl/hypershap)](https://img.shields.io/github/commit-activity/m/automl/hypershap)
[![License](https://img.shields.io/github/license/automl/hypershap)](https://img.shields.io/github/license/automl/hypershap)


`HyperSHAP` is a framework to explain the outcomes of hyperparameter optimization (HPO) by leveraging cooperative game theory concepts such as Shapley values and interaction indices. It is designed to provide actionable insights into the role and interplay of hyperparameters, thereby reducing the manual effort typically required to interpret HPO results. While its primary audience is researchers and practitioners in machine learning and artificial intelligence, its use is not restricted to these target groups.

The analysis of HPO results often involves comparing tuned configurations, assessing hyperparameter importance, and identifying optimizer biases. These tasks are typically performed in an ad-hoc and dataset-specific manner, requiring extensive manual inspection of results. Moreover, existing approaches often lack the ability to systematically capture interactions among hyperparameters or to generalize explanations across datasets and optimizers.

`HyperSHAP` addresses these challenges by formulating HPO explanations as cooperative games, where hyperparameters form coalitions whose contributions to performance are quantified. This unified framework enables fine-grained analyses, such as ablation studies, sensitivity attribution, tunability assessment, and optimizer behavior characterization. All computations are naturally parallelizable, making `HyperSHAP` scalable to modern HPO scenarios. By automating and standardizing the generation of interpretable explanations, `HyperSHAP` alleviates much of the overhead in analyzing HPO results and provides practitioners with clear guidance on which hyperparameters to focus on and how optimizers behave across tasks.

- **Github repository**: <https://github.com/automl/hypershap/>


## Features
- **Additive Shapley decomposition** of any performance metric across hyper‑parameters.
- **Interaction analysis** via the Faithful Shapley Interaction Index (FSII).
- Ready‑made explanation tasks for **Ablation**, **Tunability**, and **Optimizer Bias** studies.
- Integrated **visualisation** (SI‑graph) for interaction effects.
- Works with any surrogate model that follows the `ExplanationTask` interface.

---

## Acknowledgements
This work was partially supported by the European Union (ERC, “ixAutoML”, grant no.101041029).


---

**Enjoy exploring your HPO pipelines with HyperSHAP!** 🎉
