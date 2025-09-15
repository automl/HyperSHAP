"""The module contains simple setup fixtures for more convenient testing."""

from __future__ import annotations

import pytest
from ConfigSpace import Configuration, ConfigurationSpace, UniformFloatHyperparameter

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
def multi_data_blackbox_functions() -> list[LargeBlackboxFunction]:
    """Return a list of multi-data blackbox functions for testing."""
    return [LargeBlackboxFunction(coeff=i * 0.1) for i in range(NUM_DATA)]


@pytest.fixture(scope="session")
def multi_data_et(
    multi_data_config_space: ConfigurationSpace,
    multi_data_blackbox_functions: list[LargeBlackboxFunction],
) -> ExplanationTask:
    """Return a base explanation task for the simple setup."""
    return ExplanationTask.from_function_multidata(
        config_space=multi_data_config_space,
        functions=[fun.evaluate for fun in multi_data_blackbox_functions],
    )


@pytest.fixture(scope="session")
def multi_data_baseline_config(multi_data_config_space: ConfigurationSpace) -> Configuration:
    """Return a base configuration for the multi-data setup."""
    return multi_data_config_space.get_default_configuration()
