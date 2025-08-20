"""The abstract module defines the abstract base class `AbstractHPIGame` for analyzing HPI.

The `AbstractHPIGame` class provides a foundational structure for constructing
and analyzing games that represent interactions between hyperparameters,
enabling a deeper understanding of their impact on optimization performance.
It handles initialization, normalization, and provides a framework for
parallel evaluation of coalitions. Subclasses must implement the
`evaluate_single_coalition` method to define the game-specific evaluation logic.
"""

from __future__ import annotations

from abc import abstractmethod
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING

import numpy as np
from shapiq import Game

if TYPE_CHECKING:
    from collections.abc import Iterable

    from hypershap.task import ExplanationTask


class AbstractHPIGame(Game):
    """Abstract base class for Hyperparameter Importance Games (HPIGames).

    Represents a game-theoretic framework for analyzing the importance of
    hyperparameters for HPO. It leverages the `shapiq` library to compute
    Shapley values and analyze coalition behavior.

    Args:
        explanation_task: The `ExplanationTask` containing information about
            the configuration space and surrogate model.
        n_workers: The number of worker threads to use for parallel evaluation
            of coalitions. Defaults to 1.  Using more workers can significantly
            speed up the computation of Shapley values.
        verbose:  A boolean indicating whether to print verbose messages during
            computation. Defaults to False.

    """

    def __init__(
        self,
        explanation_task: ExplanationTask,
        n_workers: int | None = None,
        verbose: bool | None = None,
    ) -> None:
        """Initialize the Hyperparameter Interaction Game (HPIGame).

        Args:
            explanation_task: The `ExplanationTask` containing information about
                the configuration space and surrogate model. This task defines the
                hyperparameter search space and the model used to estimate performance.
            n_workers: The number of worker threads to use for parallel evaluation
                of coalitions. Defaults to None meaning no parallelization.  Using more workers can significantly
                speed up the computation of Shapley values.  The maximum number of workers is capped by the number of coalitions.
            verbose:  A boolean indicating whether to print verbose messages during
                computation. Defaults to None.  When set to True, the method prints
                debugging information and progress updates.

        """
        self.explanation_task = explanation_task
        self.n_workers = n_workers if n_workers is not None else 1
        self.verbose = verbose if verbose is not None else False

        # determine the value of the empty coalition so that we can normalize wrt to that performance
        normalization_value = self.evaluate_single_coalition(
            np.array([False] * explanation_task.get_num_hyperparameters()),
        )

        super().__init__(
            n_players=explanation_task.get_num_hyperparameters(),
            normalize=True,
            normalization_value=normalization_value,
        )

    def _process_chunk(self, chunk: np.ndarray) -> list:
        """Process a chunk of coalitions, evaluating each coalition using the `evaluate_single_coalition` method.

        Args:
            chunk: A NumPy array representing a subset of coalitions.

        Returns:
            A list of floats, where each float is the value of a coalition
            in the input chunk.

        """
        return [self.evaluate_single_coalition(c) for c in chunk]

    def value_function(self, coalitions: np.ndarray) -> np.ndarray:
        """Calculate the value of a list of coalitions.

        This method handles both single-worker and multi-worker scenarios for efficient computation.

        Args:
            coalitions: A NumPy array representing a list of coalitions.  Each
                coalition is a Boolean array indicating which hyperparameters
                are included in that coalition.

        Returns:
            A NumPy array containing the values of the input coalitions.

        """
        if self.n_workers == 1:
            value_list = []
            for coalition in coalitions:
                value_list += [self.evaluate_single_coalition(coalition)]
        else:
            m = len(coalitions)
            num_workers = min(self.n_workers, m)
            base_size = m // num_workers
            remainder = m % num_workers

            chunk_indices: Iterable[tuple[int, int]] = []
            start = 0
            for i in range(num_workers):
                size = base_size + (1 if i < remainder else 0)
                chunk_indices.append((start, start + size))
                start += size

            chunks = [coalitions[start:end] for start, end in chunk_indices]

            with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
                partial_results = list(executor.map(self._process_chunk, chunks))

            value_list = [val for sublist in partial_results for val in sublist]
        return np.array(value_list)

    @abstractmethod
    def evaluate_single_coalition(self, coalition: np.ndarray) -> float:
        """Evaluate the value of a single coalition.

        This method *must* be implemented by subclasses.

        Args:
            coalition: A boolean array representing a coalition of hyperparameters.

        Returns:
            The value of the coalition (a float).

        """

    def get_num_hyperparameters(self) -> int:
        """Return the number of hyperparameters being considered.

        Returns:
            The number of hyperparameters (an integer).

        """
        return self.explanation_task.get_num_hyperparameters()

    def get_hyperparameter_names(self) -> list[str]:
        """Return a list of the names of the hyperparameters.

        Returns:
            A list of strings, where each string is the name of a hyperparameter.

        """
        return self.explanation_task.get_hyperparameter_names()
