"""Test the surrogate model classes."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from hypershap import ExplanationTask
    from tests.fixtures.simple_setup import SimpleBlackboxFunction

EPSILON = 0.05


def test_batch_config_prediction(
    simple_base_et: ExplanationTask,
    simple_blackbox_function: SimpleBlackboxFunction,
) -> None:
    """Test the config batch prediction functionality of the surrogate model."""
    simple_base_et.config_space.seed(42)
    config_batch = simple_base_et.config_space.sample_configuration(10)
    gts = np.array([simple_blackbox_function.evaluate(config) for config in config_batch])
    preds = np.array(simple_base_et.surrogate_model.evaluate_config_batch(config_batch))
    mean_diff = (gts - preds).mean()
    assert mean_diff < EPSILON


def test_single_config_prediction(
    simple_base_et: ExplanationTask,
    simple_blackbox_function: SimpleBlackboxFunction,
) -> None:
    """Test the single config prediction functionality of the surrogate model."""
    simple_base_et.config_space.seed(42)
    config = simple_base_et.config_space.sample_configuration()
    gt = simple_blackbox_function.evaluate(config)
    pred = simple_base_et.surrogate_model.evaluate_config(config)
    diff = gt - pred
    assert diff < EPSILON


def test_vector_prediction(simple_base_et: ExplanationTask, simple_blackbox_function: SimpleBlackboxFunction) -> None:
    """Test the vector prediction functionality of the surrogate model."""
    simple_base_et.config_space.seed(42)
    config = simple_base_et.config_space.sample_configuration()
    gt = simple_blackbox_function.evaluate(config)
    pred = simple_base_et.surrogate_model.evaluate(config.get_array())
    diff = gt - pred
    assert diff < EPSILON
