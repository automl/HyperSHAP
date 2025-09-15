"""The module contains simple setup fixtures for more convenient testing."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

import pytest
from ConfigSpace import Configuration, ConfigurationSpace, UniformFloatHyperparameter

from hypershap import ExplanationTask

NUM_PARAMS = 15


@pytest.fixture(scope="session")
def large_config_space() -> ConfigurationSpace:
    """Return a simple config space for testing."""
    config_space = ConfigurationSpace()
    for i in range(NUM_PARAMS):
        config_space.add(UniformFloatHyperparameter("p" + str(i), 0, 1, 0))
    return config_space


class LargeBlackboxFunction:
    """A very simple black box function for testing."""

    def __init__(self, coeff: float) -> None:
        """Initialize the simple black box function.

        Args:
            coeff (int): The coefficient of the black box function.

        """
        self.coeff = coeff

    def evaluate(self, x: Configuration) -> float:
        """Evaluate the value of a configuration.

        Args:
            x: The configuration to be evaluated.

        Returns: The value of the configuration.

        """
        return self.value(x.get_array())

    def value(self, x: np.ndarray) -> float:
        """Evaluate the value of a configuration.

        Args:
            x: The hyperparameter configuration array.

        """
        return (x * self.coeff).sum()


@pytest.fixture(scope="session")
def large_blackbox_function() -> LargeBlackboxFunction:
    """Return a simple blackbox function for testing.

    Returns: The simple blackbox function.

    """
    return LargeBlackboxFunction(coeff=0.5)


@pytest.fixture(scope="session")
def large_base_et(
    large_config_space: ConfigurationSpace,
    large_blackbox_function: LargeBlackboxFunction,
) -> ExplanationTask:
    """Return a base explanation task for the simple setup."""
    return ExplanationTask.from_function(large_config_space, large_blackbox_function.evaluate)
