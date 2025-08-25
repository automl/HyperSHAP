# HyperSHAP <img src="https://raw.githubusercontent.com/automl/hypershap/main/docs/source/_static/logo/hypershap-logo.png" alt="HyperSHAP Logo" align="right" height="200px"/>

[![Release](https://img.shields.io/github/v/release/automl/HyperSHAP)](https://img.shields.io/github/v/release/automl/hypershap)
[![Build status](https://img.shields.io/github/actions/workflow/status/automl/hypershap/main.yml?branch=main)](https://github.com/automl/hypershap/actions/workflows/main.yml?query=branch%3Amain)
[![Coverage Status](https://coveralls.io/repos/github/automl/HyperSHAP/badge.svg?branch=dev)](https://coveralls.io/github/automl/HyperSHAP?branch=dev)
[![Commit activity](https://img.shields.io/github/commit-activity/m/automl/hypershap)](https://img.shields.io/github/commit-activity/m/automl/hypershap)
[![License](https://img.shields.io/github/license/automl/hypershap)](https://img.shields.io/github/license/automl/hypershap)

HyperSHAP – a game‑theoretic Python library for explaining Hyperparameter Optimization (HPO). It uses Shapley values and interaction indices to provide both local and global insights into how individual hyper‑parameters (and their interactions) affect a model’s performance.

- **Github repository**: <https://github.com/automl/hypershap/>
- **Documentation** <https://automl.github.io/hypershap/>


## Table of Contents
- [Features](#features)
- [Installation](#installation)
- [Getting Started](#getting-started)
- [API Overview](#api-overview)
- [Example Notebook](#example-notebook)
- [Citation](#citation)
- [Contributing](#contributing)
- [License](#license)

---

## Features
- **Additive Shapley decomposition** of any performance metric across hyper‑parameters.
- **Interaction analysis** via the Faithful Shapley Interaction Index (FSII).
- Ready‑made explanation tasks for **Ablation**, **Tunability**, and **Optimizer Bias** studies.
- Integrated **visualisation** (SI‑graph) for interaction effects.
- Works with any surrogate model that follows the `ExplanationTask` interface.

---

## Installation

```sh
$ make install
```

## Getting Started
Given an existing setup with a ConfigurationSpace from the [ConfigSpace package](https://github.com/automl/ConfigSpace) and black-box function as follows:
```Python
from ConfigSpace import ConfigurationSpace, Configuration

# ConfigurationSpace describing the hyperparameter space
cs = ConfigurationSpace()
  ...

# A black-box function, evaluating ConfigSpace.Configuration objects
def blackbox_function(cfg: Configuration) -> float:
  ...
```

You can use HyperSHAP as follows:
```Python
from hypershap import ExplanationTask, HyperSHAP

# Instantiate HyperSHAP
hypershap = HyperSHAP(ExplanationTask.from_function(config_space=cs,function=blackbox_function))
# Conduct tunability analysis
hypershap.tunability(baseline_config=cs.get_default_configuration())
# Plot results as a Shapley Interaction graph
hypershap.plot_si_graph()
```

The example demonstrates how to:
1. Wrap a black-box function in an explanation task.
2. Use `HyperSHAP` to obtain interaction values for the **tunability** game.
3. Plot the corresponding SI-graph.

---

## API Overview

| Method | Purpose | Key Arguments |
|--------|---------|---------------|
| `HyperSHAP(explanation_task)` | Initialise the explainer with a generic `ExplanationTask`. |
| `ablation(config_of_interest, baseline_config, index="FSII", order=2)` | Explain the contribution of each hyperparameter value (and interactions) when moving from a baseline to a specific configuration. |
| `tunability(baseline_config=None, index="FSII", order=2, n_samples=10_000)` | Quantify how much performance can be gained by tuning subsets of hyper‑parameters. |
| `optimizer_bias(optimizer_of_interest, optimizer_ensemble, index="FSII", order=2)` | Attribute performance differences to a particular optimizer vs. an ensemble of optimizers. |
| `plot_si_graph(interaction_values=None, save_path=None)` | Plot the Shapley Interaction (SI) graph; uses the most recent interaction values if none are supplied. |
| `ExplanationTask.get_hyperparameter_names()` | Helper to retrieve ordered hyper‑parameter names (used for visualisation). |

All methods return an `InteractionValues` object (from **shapiq**) that can be inspected, saved, or passed to the visualisation routine.

---

## Example Notebooks
Full Jupyter notebooks illustrating all three explanation tasks (ablation, tunability, optimizer bias) are included in the repository under `examples/`. The notebookts walk through:

- Building a mockup environment
- Creating the corresponding explanation task
- Loading explanation tasks from different setups: data, black-box function, and existing surrogate model.
- Computing interaction values with HyperSHAP
- Visualizing results with `plot_si_graph`

---

## Citation
If you use HyperSHAP in your research, please cite the original paper:

```bibtex
@article{wever-arxiv25,
  author       = {Marcel Wever and
                  Maximilian Muschalik and
                  Fabian Fumagalli and
                  Marius Lindauer},
  title        = {HyperSHAP: Shapley Values and Interactions for Hyperparameter Importance},
  journal      = {CoRR},
  volume       = {abs/2502.01276},
  year         = {2025},
  doi          = {10.48550/ARXIV.2502.01276},
}
```

The paper introduces the underlying game-theoretic framework and demonstrates its usefulness for HPO explainability.

## Contributing

Contributions are welcome! Please follow these steps:

     Fork the repo and create a feature branch (git checkout -b feat/your-feature).
     Write tests (the project uses pytest).
     Ensure all tests pass (pytest).
     Update documentation if you add new functionality.
     Submit a Pull Request with a clear description of the changes.


See CONTRIBUTING.md for detailed guidelines.

---

## License
HyperSHAP is released under the BSD 3-Clause License. See the `LICENSE` file for full terms.

---

**Enjoy exploring your HPO pipelines with HyperSHAP!** 🎉

---
Repository initiated with [fpgmaas/cookiecutter-uv](https://github.com/fpgmaas/cookiecutter-uv).
