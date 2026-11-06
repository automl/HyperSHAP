"""Tests for the optuna integration (hypershap.optuna_task).

All tests are skipped automatically when optuna is not installed.
"""

from __future__ import annotations

import pytest

optuna = pytest.importorskip("optuna", reason="optuna is not installed")

import math  # noqa: E402
from typing import TYPE_CHECKING  # noqa: E402

from ConfigSpace import (  # noqa: E402
    CategoricalHyperparameter,
    ConfigurationSpace,
    Float,
    UniformFloatHyperparameter,
    UniformIntegerHyperparameter,
)

if TYPE_CHECKING:
    import optuna as opt

from hypershap import ExplanationTask, HyperSHAP, from_optuna_study  # noqa: E402
from hypershap.optuna_task import _require_optuna, study_to_config_space, study_to_data  # noqa: E402

# ---------------------------------------------------------------------------
# Shared objective (matches SimpleBlackboxFunction: 0.7*a + 2.0*b)
# ---------------------------------------------------------------------------

A_COEFF = 0.7
B_COEFF = 2.0
N_TRIALS = 300
N_HPS = 3  # a, b, c
B_UPPER = 10  # upper bound for integer HP "b"


def _linear_objective_max(trial: opt.Trial) -> float:
    """Maximisation objective: 0.7*a + 2.0*b (float + int + categorical HPs)."""
    a = trial.suggest_float("a", 0.0, 1.0)
    b = trial.suggest_int("b", 0, 10)
    c = trial.suggest_categorical("c", ["X", "Y"])
    bonus = math.sin(a) * 0.01 if c == "X" else math.cos(a) * 0.01  # tiny noise
    return A_COEFF * a + B_COEFF * b + bonus


def _linear_objective_min(trial: opt.Trial) -> float:
    """Minimisation objective: -(0.7*a + 2.0*b)."""
    return -_linear_objective_max(trial)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def maximize_study() -> opt.Study:
    """Optuna maximisation study with float, int and categorical HPs."""
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=0),
    )
    study.optimize(_linear_objective_max, n_trials=N_TRIALS)
    return study


@pytest.fixture(scope="module")
def minimize_study() -> opt.Study:
    """Optuna minimisation study (negated objective, same landscape)."""
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=0),
    )
    study.optimize(_linear_objective_min, n_trials=N_TRIALS)
    return study


@pytest.fixture(scope="module")
def float_only_study() -> opt.Study:
    """Minimal study with a single float HP for targeted conversion tests."""
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def obj(trial: opt.Trial) -> float:
        x = trial.suggest_float("x", -1.0, 1.0)
        return -(x**2)  # maximise at x=0

    study = optuna.create_study(direction="maximize")
    study.optimize(obj, n_trials=50)
    return study


# ---------------------------------------------------------------------------
# study_to_config_space
# ---------------------------------------------------------------------------


def test_config_space_hp_count(maximize_study: opt.Study) -> None:
    """The inferred ConfigSpace must contain exactly the HPs used in the study."""
    cs = study_to_config_space(maximize_study)
    assert len(cs) == N_HPS


def test_config_space_hp_names(maximize_study: opt.Study) -> None:
    """HP names must match the suggest_* names used in the study."""
    cs = study_to_config_space(maximize_study)
    assert set(cs.keys()) == {"a", "b", "c"}


def test_config_space_float_bounds(maximize_study: opt.Study) -> None:
    """Float HP bounds must be transferred correctly."""
    cs = study_to_config_space(maximize_study)
    hp: UniformFloatHyperparameter = cs["a"]  # type: ignore[assignment]
    assert isinstance(hp, UniformFloatHyperparameter)
    assert hp.lower == pytest.approx(0.0)
    assert hp.upper == pytest.approx(1.0)


def test_config_space_int_bounds(maximize_study: opt.Study) -> None:
    """Integer HP bounds must be transferred correctly."""
    cs = study_to_config_space(maximize_study)
    hp: UniformIntegerHyperparameter = cs["b"]  # type: ignore[assignment]
    assert isinstance(hp, UniformIntegerHyperparameter)
    assert hp.lower == 0
    assert hp.upper == B_UPPER


def test_config_space_categorical_choices(maximize_study: opt.Study) -> None:
    """Categorical choices must be transferred correctly."""
    cs = study_to_config_space(maximize_study)
    hp: CategoricalHyperparameter = cs["c"]  # type: ignore[assignment]
    assert isinstance(hp, CategoricalHyperparameter)
    assert set(hp.choices) == {"X", "Y"}


def test_config_space_float_only(float_only_study: opt.Study) -> None:
    """Single-float study produces a one-dimensional ConfigSpace."""
    cs = study_to_config_space(float_only_study)
    assert len(cs) == 1
    assert "x" in cs
    hp: UniformFloatHyperparameter = cs["x"]  # type: ignore[assignment]
    assert hp.lower == pytest.approx(-1.0)
    assert hp.upper == pytest.approx(1.0)


def test_config_space_no_completed_trials_raises() -> None:
    """study_to_config_space must raise ValueError when no completed trials exist."""
    empty_study = optuna.create_study()
    with pytest.raises(ValueError, match="No completed trials"):
        study_to_config_space(empty_study)


def test_config_space_unsupported_distribution_raises() -> None:
    """study_to_config_space must raise TypeError for unsupported distributions."""
    from unittest.mock import MagicMock

    unknown_dist = MagicMock()
    unknown_dist.__class__ = type("UnknownDist", (), {})  # type: ignore[assignment]

    mock_trial = MagicMock()
    mock_trial.state = optuna.trial.TrialState.COMPLETE
    mock_trial.distributions = {"z": unknown_dist}
    mock_trial.params = {"z": 0.5}

    mock_study = MagicMock()
    mock_study.trials = [mock_trial]

    with pytest.raises(TypeError, match="Unsupported optuna distribution"):
        study_to_config_space(mock_study)


# ---------------------------------------------------------------------------
# study_to_data
# ---------------------------------------------------------------------------


def test_data_length(maximize_study: opt.Study) -> None:
    """study_to_data must return one entry per completed trial."""
    data = study_to_data(maximize_study)
    completed = sum(1 for t in maximize_study.trials if t.value is not None)
    assert len(data) == completed


def test_data_tuples_structure(maximize_study: opt.Study) -> None:
    """Each element must be a (Configuration, float) tuple."""
    from ConfigSpace import Configuration

    data = study_to_data(maximize_study)
    for config, value in data:
        assert isinstance(config, Configuration)
        assert isinstance(value, float)


def test_data_config_belongs_to_space(maximize_study: opt.Study) -> None:
    """Every configuration in the converted data must be valid in the inferred CS."""
    cs = study_to_config_space(maximize_study)
    data = study_to_data(maximize_study, config_space=cs)
    for config, _ in data:
        # ConfigSpace raises if the config doesn't belong to the space
        cs.check_configuration_vector_representation(config.get_array())


def test_data_maximize_not_negated(maximize_study: opt.Study) -> None:
    """Values from a maximisation study must NOT be negated."""
    data = study_to_data(maximize_study)
    trial_values = [t.value for t in maximize_study.trials if t.value is not None]
    data_values = [v for _, v in data]
    assert sorted(trial_values) == pytest.approx(sorted(data_values), rel=1e-6)


def test_data_minimize_negated(minimize_study: opt.Study) -> None:
    """Values from a minimisation study must be negated (sign-flipped)."""
    data = study_to_data(minimize_study)
    trial_values = [t.value for t in minimize_study.trials if t.value is not None]
    data_values = [v for _, v in data]
    assert sorted([-v for v in trial_values]) == pytest.approx(sorted(data_values), rel=1e-6)


def test_data_negate_override(maximize_study: opt.Study) -> None:
    """Passing negate=True must negate even a maximisation study."""
    data_normal = study_to_data(maximize_study, negate=False)
    data_negated = study_to_data(maximize_study, negate=True)
    for (_, v_normal), (_, v_negated) in zip(data_normal, data_negated, strict=True):
        assert v_normal == pytest.approx(-v_negated)


def test_data_no_completed_trials_raises() -> None:
    """study_to_data must raise ValueError when no completed trials can be converted."""
    empty_study = optuna.create_study()
    cs = ConfigurationSpace()
    cs.add(Float("x", bounds=(0.0, 1.0)))
    with pytest.raises(ValueError, match="No trials could be converted"):
        study_to_data(empty_study, config_space=cs)


def test_data_multi_objective_raises() -> None:
    """study_to_data must raise ValueError for multi-objective studies."""
    multi_study = optuna.create_study(directions=["minimize", "maximize"])

    def multi_obj(trial: opt.Trial) -> tuple[float, float]:
        x = trial.suggest_float("x", 0.0, 1.0)
        return x, 1.0 - x

    multi_study.optimize(multi_obj, n_trials=10)
    with pytest.raises(ValueError, match="Multi-objective"):
        study_to_data(multi_study)


# ---------------------------------------------------------------------------
# from_optuna_study
# ---------------------------------------------------------------------------


def test_from_optuna_study_returns_explanation_task(maximize_study: opt.Study) -> None:
    """from_optuna_study must return a valid ExplanationTask."""
    task = from_optuna_study(maximize_study)
    assert isinstance(task, ExplanationTask)


def test_from_optuna_study_hp_names(maximize_study: opt.Study) -> None:
    """ExplanationTask must contain all HPs from the study."""
    task = from_optuna_study(maximize_study)
    assert set(task.get_hyperparameter_names()) == {"a", "b", "c"}


def test_from_optuna_study_hp_count(maximize_study: opt.Study) -> None:
    """ExplanationTask must have the correct number of HPs."""
    task = from_optuna_study(maximize_study)
    assert task.get_num_hyperparameters() == N_HPS


def test_from_optuna_study_single_surrogate(maximize_study: opt.Study) -> None:
    """ExplanationTask from a single study must have a single (not list) surrogate."""
    task = from_optuna_study(maximize_study)
    assert not task.is_multi_data()
    model = task.get_single_surrogate_model()
    assert model is not None


def test_from_optuna_study_surrogate_quality(maximize_study: opt.Study) -> None:
    """The surrogate must be a reasonable fit: b matters more than a."""
    task = from_optuna_study(maximize_study)
    cs = task.config_space

    # Evaluate two configs that differ only in b (0 vs 10).
    from ConfigSpace import Configuration

    low_b = Configuration(cs, values={"a": 0.5, "b": 0, "c": "X"})
    high_b = Configuration(cs, values={"a": 0.5, "b": 10, "c": "X"})

    val_low = task.get_single_surrogate_model().evaluate_config(low_b)
    val_high = task.get_single_surrogate_model().evaluate_config(high_b)

    # The linear objective grows strongly with b, so high_b must score higher.
    assert val_high > val_low, "Surrogate should predict higher value for b=10 vs b=0"


def test_from_optuna_study_minimize_auto_negate(minimize_study: opt.Study) -> None:
    """from_optuna_study on a minimisation study must auto-negate values.

    After negation, high-b configs must still score better than low-b configs.
    """
    task = from_optuna_study(minimize_study)
    cs = task.config_space

    from ConfigSpace import Configuration

    low_b = Configuration(cs, values={"a": 0.5, "b": 0, "c": "X"})
    high_b = Configuration(cs, values={"a": 0.5, "b": 10, "c": "X"})

    val_low = task.get_single_surrogate_model().evaluate_config(low_b)
    val_high = task.get_single_surrogate_model().evaluate_config(high_b)

    assert val_high > val_low, "After auto-negation of a minimisation study, higher b should still score better"


def test_from_optuna_study_custom_base_model(maximize_study: opt.Study) -> None:
    """from_optuna_study must accept a custom sklearn surrogate model."""
    from sklearn.ensemble import GradientBoostingRegressor

    task = from_optuna_study(maximize_study, base_model=GradientBoostingRegressor(n_estimators=50, random_state=0))
    assert isinstance(task, ExplanationTask)
    assert not task.is_multi_data()


# ---------------------------------------------------------------------------
# HyperSHAP integration
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def optuna_hypershap(maximize_study: opt.Study) -> HyperSHAP:
    """HyperSHAP instance built from an optuna study."""
    task = from_optuna_study(maximize_study)
    return HyperSHAP(explanation_task=task)


def test_hypershap_tunability(optuna_hypershap: HyperSHAP) -> None:
    """Tunability must run end-to-end and return non-None InteractionValues."""
    baseline = optuna_hypershap.explanation_task.config_space.get_default_configuration()
    iv = optuna_hypershap.tunability(baseline_config=baseline, n_samples=5_000)
    assert iv is not None
    # b (index 1) must be identified as more important than a (index 0)
    assert iv.dict_values[(1,)] > iv.dict_values[(0,)], (
        "b should have higher tunability than a (B_COEFF=2.0 > A_COEFF=0.7)"
    )


def test_hypershap_ablation(optuna_hypershap: HyperSHAP, maximize_study: opt.Study) -> None:
    """Ablation must run end-to-end and return non-None InteractionValues."""
    from ConfigSpace import Configuration

    cs = optuna_hypershap.explanation_task.config_space
    baseline = cs.get_default_configuration()
    config_of_interest = Configuration(cs, values=maximize_study.best_params)

    iv = optuna_hypershap.ablation(config_of_interest=config_of_interest, baseline_config=baseline)
    assert iv is not None


# ---------------------------------------------------------------------------
# _require_optuna
# ---------------------------------------------------------------------------


def test_require_optuna_raises_when_not_installed() -> None:
    """_require_optuna must raise ImportError with install instructions when optuna is absent."""
    import sys
    from unittest.mock import patch

    with patch.dict(sys.modules, {"optuna": None}), pytest.raises(ImportError, match="pip install optuna"):
        _require_optuna()


# ---------------------------------------------------------------------------
# _distribution_to_hp — step-size warnings
# ---------------------------------------------------------------------------


def test_float_distribution_step_warns() -> None:
    """FloatDistribution with a step size must trigger a UserWarning."""
    from optuna.distributions import FloatDistribution  # type: ignore[import-untyped]

    from hypershap.optuna_task import _distribution_to_hp

    dist = FloatDistribution(0.0, 1.0, step=0.1)
    with pytest.warns(UserWarning, match="step size"):
        hp = _distribution_to_hp("x", dist)
    # Step is ignored; bounds must still be transferred correctly.
    assert hp.lower == pytest.approx(0.0)
    assert hp.upper == pytest.approx(1.0)


def test_int_distribution_step_warns() -> None:
    """IntDistribution with step != 1 must trigger a UserWarning."""
    from optuna.distributions import IntDistribution  # type: ignore[import-untyped]

    from hypershap.optuna_task import _distribution_to_hp

    upper = 20
    dist = IntDistribution(0, upper, step=2)
    with pytest.warns(UserWarning, match="step size"):
        hp = _distribution_to_hp("n", dist)
    assert hp.lower == 0
    assert hp.upper == upper


# ---------------------------------------------------------------------------
# study_to_data — skipped-trials warning
# ---------------------------------------------------------------------------


def test_study_to_data_skipped_invalid_value_warns() -> None:
    """study_to_data must skip and warn when a trial's HP value is outside the CS bounds.

    We mock a trial whose params contain all required HP names but with a value
    that is out of the ConfigSpace bounds, causing Configuration(...) to raise —
    exercising the except-branch at lines 190-192.
    """
    from unittest.mock import MagicMock

    cs = ConfigurationSpace()
    cs.add(Float("x", bounds=(0.0, 1.0)))

    # A mock trial that passes all filters but whose value is out of CS bounds.
    mock_trial = MagicMock()
    mock_trial.state = optuna.trial.TrialState.COMPLETE
    mock_trial.value = 0.5
    mock_trial.params = {"x": 999.0}  # way out of [0, 1] → Configuration() raises
    mock_trial.distributions = {"x": optuna.distributions.FloatDistribution(0.0, 1.0)}  # type: ignore[import-untyped]

    mock_study = MagicMock()
    mock_study.trials = [mock_trial]
    mock_study.directions = [optuna.study.StudyDirection.MAXIMIZE]

    with (
        pytest.warns(UserWarning, match="skipped"),
        pytest.raises(ValueError, match="No trials could be converted"),
    ):
        study_to_data(mock_study, config_space=cs)


def test_study_to_data_skipped_missing_hp_warns() -> None:
    """study_to_data must warn when trials are skipped due to missing HPs.

    We pass a ConfigSpace that contains an extra HP 'z' not present in any
    trial, so every trial is missing 'z' and gets skipped.  The function must
    emit a UserWarning about skipped trials and then raise ValueError because
    no data remains.
    """

    # Create a tiny real study with one HP "x".
    def obj(trial: optuna.Trial) -> float:  # type: ignore[name-defined]
        return trial.suggest_float("x", 0.0, 1.0)

    study = optuna.create_study(direction="maximize")
    study.optimize(obj, n_trials=5)

    # Build a CS that expects BOTH "x" and "z"; no trial has "z".
    cs = ConfigurationSpace()
    cs.add([Float("x", bounds=(0.0, 1.0)), Float("z", bounds=(0.0, 1.0))])

    with (
        pytest.warns(UserWarning, match="skipped"),
        pytest.raises(ValueError, match="No trials could be converted"),
    ):
        study_to_data(study, config_space=cs)
