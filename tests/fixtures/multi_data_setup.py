"""The module contains simple setup fixtures for more convenient testing."""

from __future__ import annotations

import pytest
from ConfigSpace import ConfigurationSpace, UniformFloatHyperparameter

from hypershap import ExplanationTask
from tests.fixtures.large_setup import LargeBlackboxFunction

NUM_DATA = 3
NUM_PARAMS = 6


@pytest.fixture(scope="session")
def multi_data_config_space() -> ConfigurationSpace:
    """Return a simple config space for testing."""
    config_space = ConfigurationSpace()
    for i in range(NUM_PARAMS):
        config_space.add(UniformFloatHyperparameter("p" + str(i), 0, 1, 0))
    return config_space


@pytest.fixture(scope="session")
def large_base_et(
    large_config_space: ConfigurationSpace,
) -> ExplanationTask:
    """Return a base explanation task for the simple setup."""
    blackbox_functions = [LargeBlackboxFunction(coeff=i * 0.1).evaluate for i in range(NUM_DATA)]
    return ExplanationTask.from_function_multidata(large_config_space, blackbox_functions)
