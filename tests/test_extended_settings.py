"""Test suite for extended settings of HyperSHAP, e.g., multi data extensions or large spaces that can only be approximated."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from ConfigSpace import Configuration, ConfigurationSpace

from hypershap import ExplanationTask, HyperSHAP


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
