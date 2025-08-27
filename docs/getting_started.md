# Getting Started

## Installation

The `HyperSHAP` package can be easily installed via PyPI:

```bash
pip install hypershap
```

Alternatively, one can clone the GitHub repository and install hypershap via the following command:

```bash
make install
```

## General Usage

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
