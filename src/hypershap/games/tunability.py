"""Hyperparameter tunability games for analyzing the impact of tuning hyperparameters on performance.

This module provides a suite of game-theoretic tools for analyzing the tunability
of hyperparameters within a surrogate model of a black-box optimization
process.  It defines classes that implement search-based games, allowing
exploration of scenarios involving coalitions of hyperparameters and assessment
of their impact on optimization performance.  The module leverages the
`ExplanationTask` from `hypershap.task` to represent the hyperparameter
search space and surrogate model, and provides flexible configuration options
through the `ConfigSpaceSearcher` interface.

The core functionality revolves around defining games like `TunabilityGame`,
`SensitivityGame`, and `MistunabilityGame`, each representing a different
aspect of hyperparameter behavior under coalition-based constraints. These
games are built upon search strategies and allow for in-depth understanding of
hyperparameter dependencies and potential for optimization gains.
"""

from __future__ import annotations

import logging
from itertools import combinations
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from hypershap.task import (
        BaselineExplanationTask,
        MistunabilityExplanationTask,
        SensitivityExplanationTask,
        TunabilityExplanationTask,
    )

from hypershap.games.abstract import AbstractHPIGame
from hypershap.utils import Aggregation, ConfigSpaceSearcher, RandomConfigSpaceSearcher

logger = logging.getLogger(__name__)

DEFAULT_FAST_MODE_ORDER = 3


class SearchBasedGame(AbstractHPIGame):
    """Base class for games that rely on searching the configuration space."""

    def __init__(  # noqa: PLR0913
        self,
        explanation_task: BaselineExplanationTask,
        cs_searcher: ConfigSpaceSearcher,
        n_workers: int | None = None,
        verbose: bool | None = None,
        *,
        fast_mode: bool = False,
        fast_mode_order: int = DEFAULT_FAST_MODE_ORDER,
    ) -> None:
        """Initialize the search-based game.

        Args:
            explanation_task: The explanation task containing the configuration
                space and surrogate model.
            cs_searcher: The configuration space searcher. If None, a
                RandomConfigSpaceSearcher is used by default.
            n_workers: The number of worker threads to use for parallel evaluation
                of coalitions. Defaults to None meaning no parallelization.  Using more workers can significantly
                speed up the computation of Shapley values.  The maximum number of workers is capped by the number of coalitions.
            verbose:  A boolean indicating whether to print verbose messages during
                computation. Defaults to None.  When set to True, the method prints
                debugging information and progress updates.
            fast_mode: Whether to use the fast mode for max-aggregating games. In fast mode, only coalitions with
                at most `fast_mode_order` hyperparameters are evaluated with the configuration space searcher. The
                value of every coalition is the maximum of its own searcher value (if evaluated) and the values of
                all its subsets, which saves surrogate evaluations for larger coalitions. Only supported if the
                searcher aggregates via `Aggregation.MAX`. Defaults to False.
            fast_mode_order: The largest coalition size that is evaluated with the configuration space searcher
                when `fast_mode` is enabled. Defaults to 3.

        """
        if fast_mode:
            if cs_searcher.mode != Aggregation.MAX:
                msg = f"Fast mode is only supported for searchers aggregating via max, got {cs_searcher.mode}."
                raise ValueError(msg)
            if fast_mode_order < 1:
                msg = f"fast_mode_order must be at least 1, got {fast_mode_order}."
                raise ValueError(msg)

        self.cs_searcher = cs_searcher
        self.fast_mode = fast_mode
        self.fast_mode_order = fast_mode_order
        # caches of raw searcher values and of fast-mode coalition values keyed by the sorted tuple of player indices
        self._search_cache: dict[tuple[int, ...], float] = {}
        self._fast_mode_cache: dict[tuple[int, ...], float] = {}
        super().__init__(explanation_task, n_workers=n_workers, verbose=verbose)

    def evaluate_single_coalition(self, coalition: np.ndarray) -> float:
        """Evaluate the value of a single coalition using the configuration space searcher.

        Args:
            coalition: A boolean array indicating which hyperparameters are
                constrained by the coalition.

        Returns:
            The value of the coalition based on the search results.

        """
        if not self.fast_mode:
            return self.cs_searcher.search(coalition)
        return self._fast_mode_value(tuple(int(i) for i in np.flatnonzero(coalition)))

    def reset_fast_mode_cache(self) -> None:
        """Clear the cached fast-mode coalition values, e.g., after exchanging the surrogate model."""
        self._search_cache = {}
        self._fast_mode_cache = {}

    def _search_value(self, members: tuple[int, ...]) -> float:
        """Return the (cached) raw searcher value of the coalition given by its member indices."""
        cached = self._search_cache.get(members)
        if cached is not None:
            return cached
        coalition = np.zeros(self.explanation_task.get_num_hyperparameters(), dtype=bool)
        coalition[list(members)] = True
        value = float(self.cs_searcher.search(coalition))
        self._search_cache[members] = value
        return value

    def _fast_mode_value(self, members: tuple[int, ...]) -> float:
        """Compute the value of a coalition in fast mode.

        The value of a coalition is the maximum of its own searcher value (only for coalitions with at most
        `fast_mode_order` members) and the values of all its subsets with one member less. The empty coalition
        thus takes the searcher value of the baseline configuration, and, recursively, every coalition takes the
        maximum searcher value over all its subsets with at most `fast_mode_order` members. If the values of the
        subsets with one member less are cached, they are reused. Otherwise, these subsets are enumerated directly.
        """
        cached = self._fast_mode_cache.get(members)
        if cached is not None:
            return cached

        # the empty coalition has no proper subsets and thus keeps the searcher value of the baseline configuration
        sub_values = (
            [self._fast_mode_cache.get(sub) for sub in combinations(members, len(members) - 1)] if members else []
        )
        if all(v is not None for v in sub_values):
            candidates = [v for v in sub_values if v is not None]
            if len(members) <= self.fast_mode_order:
                candidates.append(self._search_value(members))
            value = max(candidates)
        else:
            # evaluate subsets in ascending size so that the searcher can ensure monotonicity via its own cache
            value = max(
                self._search_value(sub)
                for size in range(min(self.fast_mode_order, len(members)) + 1)
                for sub in combinations(members, size)
            )

        self._fast_mode_cache[members] = value
        return value


class TunabilityGame(SearchBasedGame):
    """Game representing the tunability of hyperparameters."""

    def __init__(  # noqa: PLR0913
        self,
        explanation_task: TunabilityExplanationTask,
        cs_searcher: ConfigSpaceSearcher | None = None,
        n_workers: int | None = None,
        verbose: bool | None = None,
        *,
        fast_mode: bool = False,
        fast_mode_order: int = DEFAULT_FAST_MODE_ORDER,
    ) -> None:
        """Initialize the tunability game.

        Args:
            explanation_task: The explanation task containing the configuration
                space and surrogate model.
            cs_searcher: The configuration space searcher. If None, a
                RandomConfigSpaceSearcher is used by default.
            n_workers: The number of worker threads to use for parallel evaluation
                of coalitions. Defaults to None meaning no parallelization.  Using more workers can significantly
                speed up the computation of Shapley values.  The maximum number of workers is capped by the number of coalitions.
            verbose:  A boolean indicating whether to print verbose messages during
                computation. Defaults to None.  When set to True, the method prints
                debugging information and progress updates.
            fast_mode: Whether to only evaluate coalitions with at most `fast_mode_order` hyperparameters with the
                configuration space searcher and to assign larger coalitions the maximum value of all their subsets.
                Defaults to False.
            fast_mode_order: The largest coalition size evaluated with the configuration space searcher in fast mode.
                Defaults to 3.

        """
        # set cs searcher if not given by default to a random config space searcher.
        if cs_searcher is None:
            cs_searcher = RandomConfigSpaceSearcher(explanation_task, mode=Aggregation.MAX)
        elif cs_searcher.mode != Aggregation.MAX:  # ensure that cs_searcher is maximizing
            logger.warning("WARN: Tunability game set mode of given ConfigSpaceSearcher to maximize.")
            cs_searcher.mode = Aggregation.MAX
        super().__init__(
            explanation_task,
            cs_searcher,
            n_workers=n_workers,
            verbose=verbose,
            fast_mode=fast_mode,
            fast_mode_order=fast_mode_order,
        )


class SensitivityGame(SearchBasedGame):
    """Game representing the sensitivity of hyperparameters."""

    def __init__(
        self,
        explanation_task: SensitivityExplanationTask,
        cs_searcher: ConfigSpaceSearcher | None = None,
        n_workers: int | None = None,
        verbose: bool | None = None,
    ) -> None:
        """Initialize the sensitivity game.

        Args:
            explanation_task: The explanation task containing the configuration
                space and surrogate model.
            cs_searcher: The configuration space searcher. If None, a
                RandomConfigSpaceSearcher is used by default.
            n_workers: The number of worker threads to use for parallel evaluation
                of coalitions. Defaults to None meaning no parallelization.  Using more workers can significantly
                speed up the computation of Shapley values.  The maximum number of workers is capped by the number of coalitions.
            verbose:  A boolean indicating whether to print verbose messages during
                computation. Defaults to None.  When set to True, the method prints
                debugging information and progress updates.

        """
        # set cs searcher if not given by default to a random config space searcher.
        if cs_searcher is None:
            cs_searcher = RandomConfigSpaceSearcher(explanation_task, mode=Aggregation.VAR)
        elif cs_searcher.mode != Aggregation.VAR:  # ensure that cs_searcher is maximizing
            logger.warning("WARN: Sensitivity game set mode of given ConfigSpaceSearcher to variance.")
            cs_searcher.mode = Aggregation.VAR

        super().__init__(explanation_task, cs_searcher, n_workers=n_workers, verbose=verbose)


class MistunabilityGame(SearchBasedGame):
    """Game representing the mistunability of hyperparameters."""

    def __init__(
        self,
        explanation_task: MistunabilityExplanationTask,
        cs_searcher: ConfigSpaceSearcher | None = None,
        n_workers: int | None = None,
        verbose: bool | None = None,
    ) -> None:
        """Initialize the mistunability game.

        Args:
            explanation_task: The explanation task containing the configuration
                space and surrogate model.
            cs_searcher: The configuration space searcher. If None, a
                RandomConfigSpaceSearcher is used by default.
            n_workers: The number of worker threads to use for parallel evaluation
                of coalitions. Defaults to None meaning no parallelization.  Using more workers can significantly
                speed up the computation of Shapley values.  The maximum number of workers is capped by the number of coalitions.
            verbose:  A boolean indicating whether to print verbose messages during
                computation. Defaults to None.  When set to True, the method prints
                debugging information and progress updates.

        """
        # set cs searcher if not given by default to a random config space searcher.
        if cs_searcher is None:
            cs_searcher = RandomConfigSpaceSearcher(explanation_task, mode=Aggregation.MIN)
        elif cs_searcher.mode != Aggregation.MIN:  # ensure that cs_searcher is maximizing
            logger.warning("WARN: Mistunability game set mode of given ConfigSpaceSearcher to minimize.")
            cs_searcher.mode = Aggregation.MIN

        super().__init__(explanation_task, cs_searcher, n_workers=n_workers, verbose=verbose)
