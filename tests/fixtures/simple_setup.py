"""The module contains simple setup fixtures for more convenient testing."""

from __future__ import annotations

import pytest
from ConfigSpace import Configuration, ConfigurationSpace, UniformFloatHyperparameter


@pytest.fixture(scope="session")
def simple_config_space() -> ConfigurationSpace:
    """Return a simple config space for testing."""
    config_space = ConfigurationSpace()
    config_space.add(UniformFloatHyperparameter("a", 0, 1, 0))
    config_space.add(UniformFloatHyperparameter("b", 0, 1, 0))
    return config_space


class SimpleBlackboxFunction:
    """A very simple black box function for testing."""

    def __init__(self, a_coeff: float, b_coeff: float) -> None:
        """Initialize the simple black box function.

        Args:
            a_coeff: Coefficient for hyperparameter a.
            b_coeff: Coefficient for hyperparameter b.

        """
        self.a_coeff = a_coeff
        self.b_coeff = b_coeff

    def evaluate(self, x: Configuration) -> float:
        """Evaluate the value of a configuration.

        Args:
            x: The configuration to be evaluated.

        Returns: The value of the configuration.

        """
        return self.a_coeff * x["a"] + self.b_coeff * x["b"]


@pytest.fixture(scope="session")
def simple_blackbox_function() -> SimpleBlackboxFunction:
    """Return a simple blackbox function for testing.

    Returns: The simple blackbox function.

    """
    return SimpleBlackboxFunction(0.7, 2.0)
