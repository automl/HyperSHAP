"""General test for the HyperSHAP explanation games."""

from __future__ import annotations

from itertools import combinations, product
from math import comb

import numpy as np
import pytest
from ConfigSpace import ConfigurationSpace, Float

from hypershap import ExplanationTask, HyperSHAP
from hypershap.games import MistunabilityGame, SearchBasedGame, SensitivityGame, TunabilityGame
from hypershap.task import (
    BaselineExplanationTask,
    MistunabilityExplanationTask,
    SensitivityExplanationTask,
    TunabilityExplanationTask,
)
from hypershap.utils import Aggregation, ConfigSpaceSearcher, RandomConfigSpaceSearcher


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
    rccs.mode = Aggregation.MIN
    TunabilityGame(tet, rccs)
    assert rccs.mode == Aggregation.MAX, "Game mode should have been set to 'max'."

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
    assert rccs.mode == Aggregation.VAR, "Game mode should have been set to 'var'."

    game = SensitivityGame(senset)
    assert game.cs_searcher is not None, "ConfigSpace searcher should be set."

    met = MistunabilityExplanationTask(
        bet.config_space,
        bet.surrogate_model,
        bet.baseline_config,
    )
    MistunabilityGame(met, rccs)
    assert rccs.mode == Aggregation.MIN, "Game mode should have been set to 'min'."

    game = MistunabilityGame(met)
    assert game.cs_searcher is not None, "ConfigSpace searcher should be set."


# ---------------------------------------------------------------------------
# Fast mode for max-aggregating search-based games
# ---------------------------------------------------------------------------

FAST_N_PLAYERS = 6


class _DeterministicSearcher(ConfigSpaceSearcher):
    """Searcher returning a fixed, non-monotone pseudo-random value per coalition and counting calls."""

    def __init__(self, explanation_task: BaselineExplanationTask, mode: Aggregation = Aggregation.MAX) -> None:
        super().__init__(explanation_task, mode=mode)
        self.searched: list[tuple[int, ...]] = []

    @staticmethod
    def value_of(members: tuple[int, ...]) -> float:
        return float(np.random.default_rng(hash(members) % 2**32).random())

    def search(self, coalition: np.ndarray) -> float:
        members = tuple(int(i) for i in np.flatnonzero(coalition))
        self.searched.append(members)
        return self.value_of(members)


@pytest.fixture(scope="module")
def fast_tet() -> TunabilityExplanationTask:
    """Tunability task with a handful of hyperparameters to brute-force the fast mode."""
    cs = ConfigurationSpace()
    cs.add([Float(f"x{i}", bounds=(0.0, 1.0), default=0.5) for i in range(FAST_N_PLAYERS)])
    et = ExplanationTask.from_function(cs, lambda c: float(np.sum(c.get_array())))
    return TunabilityExplanationTask(et.config_space, et.surrogate_model, cs.get_default_configuration())


def _expected_fast_value(members: tuple[int, ...], order: int) -> float:
    """Brute force: maximum searcher value over all subsets with at most `order` members (incl. the coalition)."""
    return max(
        _DeterministicSearcher.value_of(sub)
        for size in range(min(order, len(members)) + 1)
        for sub in combinations(members, size)
    )


def _all_coalitions(n: int) -> np.ndarray:
    return np.array(list(product([False, True], repeat=n)))


@pytest.mark.parametrize("order", [1, 2, 3])
@pytest.mark.parametrize("ascending", [True, False])
def test_fast_mode_values(fast_tet: TunabilityExplanationTask, order: int, ascending: bool) -> None:
    """Every coalition takes the max over its own (if small enough) and its subsets' searcher values, in any order."""
    searcher = _DeterministicSearcher(fast_tet)
    game = TunabilityGame(fast_tet, searcher, fast_mode=True, fast_mode_order=order)

    coalitions = _all_coalitions(FAST_N_PLAYERS)
    coalitions = coalitions[np.argsort(coalitions.sum(axis=1), kind="stable")]
    if not ascending:  # largest coalitions first so that the subset cache cannot be used
        coalitions = coalitions[::-1]

    for coalition in coalitions:
        members = tuple(int(i) for i in np.flatnonzero(coalition))
        assert game.evaluate_single_coalition(coalition) == pytest.approx(_expected_fast_value(members, order))

    # only coalitions up to the given order are ever passed to the searcher, each exactly once
    assert all(len(m) <= order for m in searcher.searched)
    assert len(searcher.searched) == len(set(searcher.searched))
    assert len(searcher.searched) == sum(comb(FAST_N_PLAYERS, k) for k in range(order + 1))


def test_fast_mode_parallel_matches_sequential(fast_tet: TunabilityExplanationTask) -> None:
    """Fast-mode values must not depend on the number of workers."""
    coalitions = _all_coalitions(FAST_N_PLAYERS)
    seq = TunabilityGame(fast_tet, _DeterministicSearcher(fast_tet), fast_mode=True)
    par = TunabilityGame(fast_tet, _DeterministicSearcher(fast_tet), n_workers=4, fast_mode=True)
    assert np.allclose(seq.value_function(coalitions), par.value_function(coalitions))


def test_fast_mode_empty_coalition_is_baseline(fast_tet: TunabilityExplanationTask) -> None:
    """The empty coalition must keep the searcher value of the baseline configuration."""
    game = TunabilityGame(fast_tet, _DeterministicSearcher(fast_tet), fast_mode=True)
    empty = np.zeros(FAST_N_PLAYERS, dtype=bool)
    assert game.evaluate_single_coalition(empty) == _DeterministicSearcher.value_of(())


def test_fast_mode_small_coalition_not_below_subsets(fast_tet: TunabilityExplanationTask) -> None:
    """Coalitions within the fast-mode order must not have a lower value than any of their subsets."""
    game = TunabilityGame(fast_tet, _DeterministicSearcher(fast_tet), fast_mode=True, fast_mode_order=3)
    for coalition in _all_coalitions(FAST_N_PLAYERS):
        members = tuple(int(i) for i in np.flatnonzero(coalition))
        value = game.evaluate_single_coalition(coalition)
        for i in members:
            sub = coalition.copy()
            sub[i] = False
            assert value >= game.evaluate_single_coalition(sub)


def test_fast_mode_disabled_uses_searcher_for_all(fast_tet: TunabilityExplanationTask) -> None:
    """Without fast mode every coalition is evaluated by the searcher."""
    searcher = _DeterministicSearcher(fast_tet)
    game = TunabilityGame(fast_tet, searcher)
    full = np.ones(FAST_N_PLAYERS, dtype=bool)
    assert game.evaluate_single_coalition(full) == _DeterministicSearcher.value_of(tuple(range(FAST_N_PLAYERS)))


def test_fast_mode_requires_max(fast_tet: TunabilityExplanationTask) -> None:
    """Fast mode must be rejected for searchers that do not aggregate via max."""
    with pytest.raises(ValueError, match="only supported"):
        SearchBasedGame(fast_tet, _DeterministicSearcher(fast_tet, mode=Aggregation.VAR), fast_mode=True)


def test_fast_mode_invalid_order(fast_tet: TunabilityExplanationTask) -> None:
    """fast_mode_order must be positive."""
    with pytest.raises(ValueError, match="at least 1"):
        TunabilityGame(fast_tet, _DeterministicSearcher(fast_tet), fast_mode=True, fast_mode_order=0)


def test_hypershap_tunability_fast_mode(large_base_et: ExplanationTask) -> None:
    """HyperSHAP.tunability must support fast mode end-to-end, also in the approximation regime."""
    hypershap = HyperSHAP(large_base_et, approximation_budget=2**10)
    iv = hypershap.tunability(n_samples=200, fast_mode=True)
    assert iv is not None
    assert iv.n_players == large_base_et.get_num_hyperparameters()
