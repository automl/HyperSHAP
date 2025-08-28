"""Utils module for specifying custom error classes and config space search interfaces.

This module defines specific error classes for simpler debugging and interfaces for searching config spaces.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from hypershap.task import BaselineExplanationTask

import numpy as np


class Aggregation(Enum):
    """Enum of aggregation functions for summarizing numpy arrays."""

    AVG = "avg"
    MAX = "max"
    MIN = "min"
    VAR = "var"


def evaluate_aggregation(aggregation: Aggregation, values: np.ndarray) -> float:
    """Evaluate an aggregation function for a numpy array summarizing it to a single float."""
    if aggregation == Aggregation.AVG:
        return values.mean()
    if aggregation == Aggregation.MAX:
        return values.max()
    if aggregation == Aggregation.MIN:
        return values.min()
    if aggregation == Aggregation.VAR:
        return values.var()
    raise ValueError


class ConfigSpaceSearcher(ABC):
    """Abstract base class for searching the configuration space.

    Provides an interface for retrieving performance values based on a coalition
    of hyperparameters.
    """

    def __init__(
        self,
        explanation_task: BaselineExplanationTask,
        mode: Aggregation,
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

    def __init__(
        self,
        explanation_task: BaselineExplanationTask,
        mode: Aggregation = Aggregation.MAX,
        n_samples: int = 10_000,
    ) -> None:
        """Initialize the random configuration space searcher.

        Args:
            explanation_task: The explanation task containing the configuration
                space and surrogate model.
            mode: The aggregation mode for performance values.
            n_samples: The number of configurations to sample.

        """
        super().__init__(explanation_task, mode=mode)

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
        return evaluate_aggregation(self.mode, vals)
