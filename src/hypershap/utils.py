"""Utils module for specifying custom error classes and config space search interfaces.

This module defines specific error classes for simpler debugging and interfaces for searching config spaces.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from copy import deepcopy
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from hypershap.task import BaselineExplanationTask

import numpy as np
from ConfigSpace.exceptions import (
    ActiveHyperparameterNotSetError,
    ForbiddenValueError,
    IllegalVectorizedValueError,
    InactiveHyperparameterSetError,
)

logger = logging.getLogger(__name__)


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
    return values.var()


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

        """
        self.explanation_task = explanation_task
        self.mode = mode
        self.coalition_cache: dict[str, float] = {}

    def _ensure_monotonicity(self, coalition: np.ndarray, value: float) -> float:
        """Ensure that the value function is monotonically increasing/decreasing depending on whether we want to maximize or minimize respectively.

        Args:
            coalition: The current coalition.
            value: The value of the coalition as determined by searching.

        Returns: The monotonicity-ensured value of the coalition.

        """
        monotone_value = value
        checked_one = False

        for i in range(len(coalition)):
            if coalition[i]:  # check whether the entry is True and set it to False to check for a cached result
                temp_coalition = coalition.copy()
                temp_coalition[i] = False
                if str(temp_coalition.tolist()) in self.coalition_cache:
                    checked_one = True
                    monotone_value = evaluate_aggregation(
                        self.mode,
                        np.array([
                            monotone_value,
                            self.coalition_cache[str(temp_coalition.tolist())],
                        ]),
                    )

        if not checked_one and coalition.any():
            logger.warning(
                "Could not ensure monotonicity as none of the coalitions with one player less has been cached so far.",
            )

        if value < monotone_value:  # pragma: no cover
            logger.debug(
                "Ensured monotonicity with a sub-coalition's value. Increased the value of the current coalition from %s to %s.",
                value,
                monotone_value,
            )

        return monotone_value

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
        seed: int | None = 0,
    ) -> None:
        """Initialize the random configuration space searcher.

        Args:
            explanation_task: The explanation task containing the configuration
                space and surrogate model.
            mode: The aggregation mode for performance values.
            n_samples: The number of configurations to sample.
            seed: The random seed for sampling configurations from the config space.

        """
        super().__init__(explanation_task, mode=mode)
        cs = deepcopy(explanation_task.config_space)
        if seed is not None:
            cs.seed(seed)
        sampled_configurations = cs.sample_configuration(size=n_samples)
        self.random_sample = np.array([config.get_array() for config in sampled_configurations])

    def _is_valid(self, config: np.ndarray) -> bool:
        """Check whether a configuration is valid with respect to conditions of the configuration space."""
        try:
            self.explanation_task.config_space.check_configuration_vector_representation(config)
        except (
            ActiveHyperparameterNotSetError,
            IllegalVectorizedValueError,
            InactiveHyperparameterSetError,
            ForbiddenValueError,
        ):
            return False
        else:
            return True

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

        # in case of conditions in the config space, it might happen that through blinding hyperparameter values
        # configurations might become invalid and those should not be considered for calculating vals
        if len(self.explanation_task.config_space.conditions) > 0:
            # filter invalid configurations
            validity = np.apply_along_axis(self._is_valid, axis=1, arr=temp_random_sample)
            filtered_samples = temp_random_sample[validity]

            if len(filtered_samples) < 0.05 * len(temp_random_sample):  # pragma: no cover
                logger.warning(
                    "WARNING: Due to blinding less than 5% of the samples in the random search remain valid. "
                    "Consider increasing the sampling budget of the random search.",
                )

            # predict performance values with the help of the surrogate model for the filtered configurations
            vals: np.ndarray = np.array(self.explanation_task.get_single_surrogate_model().evaluate(filtered_samples))
        else:
            vals: np.ndarray = np.array(self.explanation_task.get_single_surrogate_model().evaluate(temp_random_sample))

        # determine the final, aggregated value of the coalition
        res = evaluate_aggregation(self.mode, vals)

        # in case we are maximizing or minimizing, ensure that the value function is monotone
        if self.mode in (Aggregation.MAX, Aggregation.MIN):
            res = self._ensure_monotonicity(coalition, res)

        # cache the coalition's value
        self.coalition_cache[str(coalition.tolist())] = res

        return res
