"""The explanation task test tests all types of explanaation tasks and their functionalities as a major data class of HyperSHAP."""

from __future__ import annotations

from typing import TYPE_CHECKING

from sklearn.ensemble import RandomForestRegressor

if TYPE_CHECKING:
    from ConfigSpace import ConfigurationSpace

    from tests.fixtures.simple_setup import SimpleBlackboxFunction

from hypershap import ExplanationTask
from hypershap.task import (
    BaselineExplanationTask,
    MistunabilityExplanationTask,
    MultiBaselineExplanationTask,
    OptimizerBiasExplanationTask,
    SensitivityExplanationTask,
    TunabilityExplanationTask,
)
from hypershap.utils import RandomConfigSpaceSearcher


def _check_explanation_task(
    explanation_task: ExplanationTask,
    config_space: ConfigurationSpace,
    blackbox_function: SimpleBlackboxFunction,
) -> None:
    # trivial check whether explanation task exists
    assert explanation_task is not None

    # assert that config space has roughly the same size
    assert len(explanation_task.config_space) == len(config_space), "Config space has not been set correctly"

    # validate also the function of explanation task to work properly
    assert explanation_task.get_num_hyperparameters() == len(config_space)

    # validate whether hyperparameter names match
    hyperparameter_names_et = explanation_task.get_hyperparameter_names()
    hyperparameter_names_cs = config_space.keys()
    assert set(hyperparameter_names_et) == set(hyperparameter_names_cs), "Hyperparameter names don't match"

    # assert that surrogate model is reasonable
    epsilon = 0.05
    for cfg in config_space.sample_configuration(10):
        expected = blackbox_function.evaluate(cfg)
        actual = explanation_task.surrogate_model.evaluate_config(cfg)
        assert abs(expected - actual) < epsilon, f"Expected surrogate model to predict {expected}, but got {actual}."


def test_explanation_task_from_function(
    simple_config_space: ConfigurationSpace,
    simple_blackbox_function: SimpleBlackboxFunction,
) -> None:
    """Test creating the explanation task from the function."""
    simple_config_space.seed(42)
    explanation_task = ExplanationTask.from_function(simple_config_space, simple_blackbox_function.evaluate)
    _check_explanation_task(explanation_task, simple_config_space, simple_blackbox_function)


def test_explanation_task_from_data(
    simple_config_space: ConfigurationSpace,
    simple_blackbox_function: SimpleBlackboxFunction,
) -> None:
    """Test creating the explanation task from data."""
    simple_config_space.seed(42)
    configs = simple_config_space.sample_configuration(10_000)
    values = [simple_blackbox_function.evaluate(config) for config in configs]

    explanation_task = ExplanationTask.from_data(simple_config_space, list(zip(configs, values, strict=False)))
    _check_explanation_task(explanation_task, simple_config_space, simple_blackbox_function)


def test_explanation_task_from_model(
    simple_config_space: ConfigurationSpace,
    simple_blackbox_function: SimpleBlackboxFunction,
) -> None:
    """Test creating the explanation task from an already fitted model."""
    simple_config_space.seed(42)
    configs = simple_config_space.sample_configuration(10_000)
    values = [simple_blackbox_function.evaluate(config) for config in configs]
    config_arrays = [config.get_array() for config in configs]

    model = RandomForestRegressor()
    model.fit(config_arrays, values)

    explanation_task = ExplanationTask.from_base_model(simple_config_space, model)
    _check_explanation_task(explanation_task, simple_config_space, simple_blackbox_function)


def test_baseline_explanation_task(simple_base_et: ExplanationTask) -> None:
    """Test the baseline explanation task."""
    config = simple_base_et.config_space.sample_configuration()
    et = BaselineExplanationTask(simple_base_et.config_space, simple_base_et.surrogate_model, baseline_config=config)
    assert et.baseline_config == config, "Baseline explanation task should have the proper baseline config."


def test_multibaseline_explanation_task(simple_base_et: ExplanationTask) -> None:
    """Test the multibaseline explanation task."""
    baseline_configs = simple_base_et.config_space.sample_configuration(2)
    et = MultiBaselineExplanationTask(
        simple_base_et.config_space,
        simple_base_et.surrogate_model,
        baseline_configs=baseline_configs,
    )
    assert et.baseline_configs == baseline_configs, (
        "Multibaseline explanation task should have the proper baseline configs."
    )


def test_tunability_explanation_task(simple_base_et: ExplanationTask) -> None:
    """Test the tunability explanation task."""
    config = simple_base_et.config_space.sample_configuration()
    et = TunabilityExplanationTask(simple_base_et.config_space, simple_base_et.surrogate_model, baseline_config=config)
    assert et.baseline_config == config, "Tunability explanation task should have the proper baseline config."


def test_sensitivity_explanation_task(simple_base_et: ExplanationTask) -> None:
    """Test the sensitivity explanation task."""
    config = simple_base_et.config_space.sample_configuration()
    et = SensitivityExplanationTask(simple_base_et.config_space, simple_base_et.surrogate_model, baseline_config=config)
    assert et.baseline_config == config, "Sensitivity explanation task should have the proper baseline config."


def test_mistunability_explanation_task(simple_base_et: ExplanationTask) -> None:
    """Test the mistunability explanation task."""
    config = simple_base_et.config_space.sample_configuration()
    et = MistunabilityExplanationTask(
        simple_base_et.config_space,
        simple_base_et.surrogate_model,
        baseline_config=config,
    )
    assert et.baseline_config == config, "Mistunability explanation task should have the proper baseline config."


def test_optimizer_bias_explanation_task(simple_base_et: ExplanationTask) -> None:
    """Test the optimizer bias explanation task."""
    baseline_et = BaselineExplanationTask(
        simple_base_et.config_space,
        simple_base_et.surrogate_model,
        baseline_config=simple_base_et.config_space.get_default_configuration(),
    )

    opt_of_interest = RandomConfigSpaceSearcher(baseline_et, mode="min")
    ensemble = [RandomConfigSpaceSearcher(baseline_et)]

    obet = OptimizerBiasExplanationTask(
        simple_base_et.config_space,
        simple_base_et.surrogate_model,
        opt_of_interest,
        ensemble,
    )
    assert obet.optimizer_of_interest == opt_of_interest, "Optimizer of interest is not set correctly."
    assert obet.optimizer_ensemble == ensemble, "Optimizer ensemble is not set correctly."
