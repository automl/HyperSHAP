"""Test suite for extended settings of HyperSHAP, e.g., multi data extensions or large spaces that can only be approximated."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

if TYPE_CHECKING:
    from ConfigSpace import Configuration, ConfigurationSpace

    from hypershap.surrogate_model import SurrogateModel

from hypershap import ExplanationTask, HyperSHAP
from hypershap.games import MultiDataHPIGame, TunabilityGame
from hypershap.task import TunabilityExplanationTask
from hypershap.utils import Aggregation, RandomConfigSpaceSearcher


@pytest.fixture(scope="module")
def hypershap_inst(multi_data_et: ExplanationTask) -> HyperSHAP:
    """Return an instance of hypershap with a simple explanation task."""
    return HyperSHAP(multi_data_et)


def test_large_ablation(large_base_et: ExplanationTask) -> None:
    """Test HyperSHAP with large config space."""
    baseline = large_base_et.config_space.sample_configuration()
    comparison = large_base_et.config_space.sample_configuration()
    hypershap = HyperSHAP(explanation_task=large_base_et, approximation_budget=2**7)
    hypershap.ablation(comparison, baseline)


def test_large_ablation_kernelshap(large_base_et: ExplanationTask) -> None:
    """Test HyperSHAP with large config space."""
    baseline = large_base_et.config_space.sample_configuration()
    comparison = large_base_et.config_space.sample_configuration()
    hypershap = HyperSHAP(explanation_task=large_base_et, approximation_budget=2**7)
    hypershap.ablation(comparison, baseline, index="k-SII")


def test_multi_data_ablation(
    multi_data_baseline_config: Configuration,
    multi_data_config_space: ConfigurationSpace,
    hypershap_inst: HyperSHAP,
) -> None:
    """Test the multi-data ablation task."""
    config_of_interest = multi_data_config_space.sample_configuration()
    iv = hypershap_inst.ablation(config_of_interest=config_of_interest, baseline_config=multi_data_baseline_config)
    assert iv is not None, "Interaction values should not be none."


def test_multi_data__multi_baseline_ablation(
    multi_data_config_space: ConfigurationSpace,
    hypershap_inst: HyperSHAP,
) -> None:
    """Test the multi-data multi-baseline ablation task."""
    baseline_configs = multi_data_config_space.sample_configuration(3)
    config_of_interest = multi_data_config_space.sample_configuration()
    iv = hypershap_inst.ablation_multibaseline(config_of_interest=config_of_interest, baseline_configs=baseline_configs)
    assert iv is not None, "Interaction values should not be none."


def test_multi_data_tunability(multi_data_baseline_config: Configuration, hypershap_inst: HyperSHAP) -> None:
    """Test the multi-data tunability task."""
    iv = hypershap_inst.tunability(baseline_config=multi_data_baseline_config)
    assert iv is not None, "Interaction values should not be none."


def test_multi_data_mistunability(multi_data_baseline_config: Configuration, hypershap_inst: HyperSHAP) -> None:
    """Test the multi-data mistunability task."""
    iv = hypershap_inst.mistunability(baseline_config=multi_data_baseline_config)
    assert iv is not None, "Interaction values should not be none."


def test_multi_data_sensitivity(multi_data_baseline_config: Configuration, hypershap_inst: HyperSHAP) -> None:
    """Test the multi-data sesntivity task."""
    iv = hypershap_inst.sensitivity(baseline_config=multi_data_baseline_config)
    assert iv is not None, "Interaction values should not be none."


def test_tunability_with_conditions(simple_cond_base_et: ExplanationTask) -> None:
    """Test the tunability task with a configuration space that has conditions."""
    hypershap = HyperSHAP(simple_cond_base_et)
    iv = hypershap.tunability(simple_cond_base_et.config_space.get_default_configuration())
    assert iv is not None, "Interaction values should not be none."


def test_tunability_with_activation_structures(simple_act_base_et: ExplanationTask) -> None:
    """Test the tunability task with a configuration space that has conditions."""
    hypershap = HyperSHAP(simple_act_base_et)
    iv = hypershap.tunability(simple_act_base_et.config_space.get_default_configuration())
    assert iv is not None, "Interaction values should not be none."


def test_multi_data_tunability_fast_mode(multi_data_baseline_config: Configuration, hypershap_inst: HyperSHAP) -> None:
    """Test the multi-data tunability task in fast mode."""
    iv = hypershap_inst.tunability(baseline_config=multi_data_baseline_config, n_samples=1_000, fast_mode=True)
    assert iv is not None, "Interaction values should not be none."


def test_multi_data_fast_mode_sub_games_use_own_surrogate(
    multi_data_et: ExplanationTask,
    multi_data_baseline_config: Configuration,
) -> None:
    """Fast-mode values cached by the base game must not leak into the per-dataset sub-games."""
    n_players = multi_data_et.get_num_hyperparameters()
    coalitions = [np.zeros(n_players, dtype=bool), np.ones(n_players, dtype=bool)]

    def fast_game(surrogate_model: SurrogateModel) -> TunabilityGame:
        task = TunabilityExplanationTask(multi_data_et.config_space, surrogate_model, multi_data_baseline_config)
        searcher = RandomConfigSpaceSearcher(task, n_samples=500, seed=0)
        return TunabilityGame(task, searcher, fast_mode=True, fast_mode_order=2)

    surrogates = multi_data_et.get_surrogate_model_list()
    multi_game = MultiDataHPIGame(multi_data_et, fast_game(surrogates[0]), aggregation=Aggregation.AVG)

    for sub_game, surrogate in zip(multi_game.sub_games, surrogates, strict=True):
        reference = fast_game(surrogate)
        for coalition in coalitions:
            assert sub_game.evaluate_single_coalition(coalition) == pytest.approx(
                reference.evaluate_single_coalition(coalition),
            )
