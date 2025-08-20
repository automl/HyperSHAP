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
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

    from hypershap.task import (
        BaselineExplanationTask,
        MistunabilityExplanationTask,
        SensitivityExplanationTask,
        TunabilityExplanationTask,
    )

from hypershap.games.abstract import AbstractHPIGame
from hypershap.utils import ConfigSpaceSearcher, RandomConfigSpaceSearcher

logger = logging.getLogger(__name__)


class SearchBasedGame(AbstractHPIGame):
    """Base class for games that rely on searching the configuration space."""

    def __init__(
        self,
        explanation_task: BaselineExplanationTask,
        cs_searcher: ConfigSpaceSearcher,
        n_workers: int | None = None,
        verbose: bool | None = None,
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

        """
        self.cs_searcher = cs_searcher
        super().__init__(explanation_task, n_workers=n_workers, verbose=verbose)

    def evaluate_single_coalition(self, coalition: np.ndarray) -> float:
        """Evaluate the value of a single coalition using the configuration space searcher.

        Args:
            coalition: A boolean array indicating which hyperparameters are
                constrained by the coalition.

        Returns:
            The value of the coalition based on the search results.

        """
        return self.cs_searcher.search(coalition)


class TunabilityGame(SearchBasedGame):
    """Game representing the tunability of hyperparameters."""

    def __init__(
        self,
        explanation_task: TunabilityExplanationTask,
        cs_searcher: ConfigSpaceSearcher | None = None,
        n_workers: int | None = None,
        verbose: bool | None = None,
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

        """
        # set cs searcher if not given by default to a random config space searcher.
        if cs_searcher is None:
            cs_searcher = RandomConfigSpaceSearcher(explanation_task, mode="max")
        elif cs_searcher.mode != "max":  # ensure that cs_searcher is maximizing
            logger.warning("WARN: Tunability game set mode of given ConfigSpaceSearcher to maximize.")
            cs_searcher.mode = "max"
        super().__init__(explanation_task, cs_searcher, n_workers=n_workers, verbose=verbose)


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
            cs_searcher = RandomConfigSpaceSearcher(explanation_task, mode="var")
        elif cs_searcher.mode != "var":  # ensure that cs_searcher is maximizing
            logger.warning("WARN: Sensitivity game set mode of given ConfigSpaceSearcher to variance.")
            cs_searcher.mode = "var"

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
            cs_searcher = RandomConfigSpaceSearcher(explanation_task, mode="min")
        elif cs_searcher.mode != "min":  # ensure that cs_searcher is maximizing
            logger.warning("WARN: Mistunability game set mode of given ConfigSpaceSearcher to minimize.")
            cs_searcher.mode = "min"

        super().__init__(explanation_task, cs_searcher, n_workers=n_workers, verbose=verbose)
