"""General test for the HyperSHAP explanation games."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from hypershap import ExplanationTask

from hypershap.games import MistunabilityGame, SensitivityGame, TunabilityGame
from hypershap.task import (
    BaselineExplanationTask,
    MistunabilityExplanationTask,
    SensitivityExplanationTask,
    TunabilityExplanationTask,
)
from hypershap.utils import RandomConfigSpaceSearcher


def test_reparametrization(simple_base_et: ExplanationTask) -> None:
    """Test the reparametrization of search-based games to ensure proper functionality."""
    # prepare baseline explanation task
    bet = BaselineExplanationTask(
        simple_base_et.config_space,
        simple_base_et.surrogate_model,
        simple_base_et.config_space.get_default_configuration(),
    )
    rccs = RandomConfigSpaceSearcher(bet)

    # Test tunability
    tet = TunabilityExplanationTask(
        bet.config_space,
        bet.surrogate_model,
        bet.baseline_config,
    )
    # expect max as search mode so set it to min before
    rccs.mode = "min"
    TunabilityGame(tet, rccs)
    assert rccs.mode == "max", "Game mode should have been set to 'max'."

    game = TunabilityGame(tet)
    assert game.cs_searcher is not None, "ConfigSpace searcher should be set."

    # Test sensitivity
    senset = SensitivityExplanationTask(
        bet.config_space,
        bet.surrogate_model,
        bet.baseline_config,
    )
    # expect var as search mode and should be max before according to assert
    SensitivityGame(senset, rccs)
    assert rccs.mode == "var", "Game mode should have been set to 'var'."

    game = SensitivityGame(senset)
    assert game.cs_searcher is not None, "ConfigSpace searcher should be set."

    met = MistunabilityExplanationTask(
        bet.config_space,
        bet.surrogate_model,
        bet.baseline_config,
    )
    MistunabilityGame(met, rccs)
    assert rccs.mode == "min", "Game mode should have been set to 'min'."

    game = MistunabilityGame(met)
    assert game.cs_searcher is not None, "ConfigSpace searcher should be set."
