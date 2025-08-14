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
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from hypershap.task import TunabilityExplanationTask

from hypershap.games.abstract import AbstractHPIGame

logger = logging.getLogger(__name__)


class UnknownModeError(ValueError):
    """Raised when an unknown mode is encountered."""

    def __init__(self) -> None:
        """Initialize the unknown mode error."""
        super().__init__("Unknown mode for the config space searcher.")


class ConfigSpaceSearcher(ABC):
    """Abstract base class for searching the configuration space.

    Provides an interface for retrieving performance values based on a coalition
    of hyperparameters.
    """

    def __init__(self, explanation_task: TunabilityExplanationTask) -> None:
        """Initialize the searcher with the explanation task.

        Args:
            explanation_task: The explanation task containing the configuration
                space and surrogate model.

        """
        self.explanation_task = explanation_task

    @abstractmethod
    def search(self, coalition: np.ndarray) -> float:
        """Search the configuration space based on the coalition.

        Args:
            coalition: A boolean array indicating which hyperparameters are
                constrained by the coalition.

        Returns:
            The aggregated performance value based on the search results.

        """


class RandomConfigSpaceSearcher(ConfigSpaceSearcher):
    """A searcher that randomly samples the configuration space and evaluates them using the surrogate model.

    Useful for establishing baseline performance or approximating game values.
    """

    def __init__(self, explanation_task: TunabilityExplanationTask, n_samples: int = 10_000, mode: str = "max") -> None:
        """Initialize the random configuration space searcher.

        Args:
            explanation_task: The explanation task containing the configuration
                space and surrogate model.
            n_samples: The number of configurations to sample.
            mode: The aggregation mode for performance values ('max', 'min', 'avg', 'var').

        """
        super().__init__(explanation_task)

        sampled_configurations = self.explanation_task.config_space.sample_configuration(size=n_samples)
        self.random_sample = np.array([config.get_array() for config in sampled_configurations])

        allowed_modes = ["max", "min", "avg", "var"]
        if mode in allowed_modes:
            self.mode = mode
        else:
            raise UnknownModeError

        # cache coalition values to ensure monotonicity for min/max
        self.coalition_cache = {}

    def search(self, coalition: np.ndarray) -> float:
        """Search the configuration space based on the coalition.

        Args:
            coalition: A boolean array indicating which hyperparameters are
                constrained by the coalition.

        Returns:
            The aggregated performance value based on the search results.

        """
        # copy the sampled configurations
        temp_random_sample = self.random_sample.copy()

        # blind configurations according to coalition
        blind_coalition = ~coalition
        column_index = np.where(blind_coalition)
        temp_random_sample[:, column_index] = self.explanation_task.baseline_config.get_array()[column_index]

        # predict performance values with the help of the surrogate model
        vals: np.ndarray = np.array(self.explanation_task.surrogate_model.evaluate(temp_random_sample))

        if self.mode == "max":
            return vals.max()
        if self.mode == "avg":
            return vals.mean()
        if self.mode == "min":
            return vals.min()
        if self.mode == "var":
            return vals.var()

        raise UnknownModeError


class SearchBasedGame(AbstractHPIGame):
    """Base class for games that rely on searching the configuration space."""

    def __init__(self, explanation_task: TunabilityExplanationTask, cs_searcher: ConfigSpaceSearcher = None) -> None:
        """Initialize the search-based game.

        Args:
            explanation_task: The explanation task containing the configuration
                space and surrogate model.
            cs_searcher: The configuration space searcher. If None, a
                RandomConfigSpaceSearcher is used by default.

        """
        self.cs_searcher = cs_searcher
        super().__init__(explanation_task)

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

    def __init__(self, explanation_task: TunabilityExplanationTask, cs_searcher: ConfigSpaceSearcher = None) -> None:
        """Initialize the tunability game.

        Args:
            explanation_task: The explanation task containing the configuration
                space and surrogate model.
            cs_searcher: The configuration space searcher. If None, a
                RandomConfigSpaceSearcher is used by default.

        """
        # set cs searcher if not given by default to a random config space searcher.
        if cs_searcher is None:
            cs_searcher = RandomConfigSpaceSearcher(explanation_task, mode="max")
        elif cs_searcher.mode != "max":  # ensure that cs_searcher is maximizing
            logger.warning("WARN: Tunability game set mode of given ConfigSpaceSearcher to maximize.")
            cs_searcher.mode = "max"
        super().__init__(explanation_task, cs_searcher)


class SensitivityGame(SearchBasedGame):
    """Game representing the sensitivity of hyperparameters."""

    def __init__(self, explanation_task: TunabilityExplanationTask, cs_searcher: ConfigSpaceSearcher = None) -> None:
        """Initialize the sensitivity game.

        Args:
            explanation_task: The explanation task containing the configuration
                space and surrogate model.
            cs_searcher: The configuration space searcher. If None, a
                RandomConfigSpaceSearcher is used by default.

        """
        # set cs searcher if not given by default to a random config space searcher.
        if cs_searcher is None:
            cs_searcher = RandomConfigSpaceSearcher(explanation_task, mode="var")
        elif cs_searcher.mode != "var":  # ensure that cs_searcher is maximizing
            logger.warning("WARN: Sensitivity game set mode of given ConfigSpaceSearcher to variance.")
            cs_searcher.mode = "var"

        super().__init__(explanation_task, cs_searcher)


class MistunabilityGame(TunabilityGame):
    """Game representing the mistunability of hyperparameters."""

    def __init__(self, explanation_task: TunabilityExplanationTask, cs_searcher: ConfigSpaceSearcher = None) -> None:
        """Initialize the mistunability game.

        Args:
            explanation_task: The explanation task containing the configuration
                space and surrogate model.
            cs_searcher: The configuration space searcher. If None, a
                RandomConfigSpaceSearcher is used by default.

        """
        # set cs searcher if not given by default to a random config space searcher.
        if cs_searcher is None:
            cs_searcher = RandomConfigSpaceSearcher(explanation_task, mode="min")
        elif cs_searcher.mode != "min":  # ensure that cs_searcher is maximizing
            logger.warning("WARN: Mistunability game set mode of given ConfigSpaceSearcher to minimize.")
            cs_searcher.mode = "min"

        super().__init__(explanation_task, cs_searcher)
