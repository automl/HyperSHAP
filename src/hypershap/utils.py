"""Utils module for specifying custom error classes and config space search interfaces.

This module defines specific error classes for simpler debugging and interfaces for searching config spaces.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from hypershap.task import BaselineExplanationTask

import numpy as np


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

    def __init__(
        self,
        explanation_task: BaselineExplanationTask,
        mode: str = "max",
        allowed_modes: list[str] | None = None,
    ) -> None:
        """Initialize the searcher with the explanation task.

        Args:
            explanation_task: The explanation task containing the configuration
                space and surrogate model.
            mode: The aggregation mode for performance values.
            allowed_modes: The list of allowed aggregation mode for performance values.

        """
        self.explanation_task = explanation_task
        self.mode = mode
        self.allowed_modes = allowed_modes

        if self.allowed_modes is None or mode in self.allowed_modes:
            self.mode = mode
        else:
            raise UnknownModeError

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

    def __init__(self, explanation_task: BaselineExplanationTask, mode: str = "max", n_samples: int = 10_000) -> None:
        """Initialize the random configuration space searcher.

        Args:
            explanation_task: The explanation task containing the configuration
                space and surrogate model.
            mode: The aggregation mode for performance values ('max', 'min', 'avg', 'var').
            n_samples: The number of configurations to sample.

        """
        allowed_modes = ["max", "min", "avg", "var"]
        super().__init__(explanation_task, mode=mode, allowed_modes=allowed_modes)

        sampled_configurations = self.explanation_task.config_space.sample_configuration(size=n_samples)
        self.random_sample = np.array([config.get_array() for config in sampled_configurations])

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
