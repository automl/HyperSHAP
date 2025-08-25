"""optimizerbias: Quantifying Bias in Hyperparameter Optimizers.

This module provides tools for measuring and explaining bias in hyperparameter optimizers using cooperative game
theory frameworks. It implements an explanation game specifically designed to assess how hyperparameter coalitions
influence the performance of specific HPO algorithms compared to diverse alternatives.

Key Features:
- Provides a mechanism to measure optimizer biases through mathematical games
- Enables identification of hyperparameters contributing most significantly to biased outcomes

Classes:
    OptimizerBiasGame: An explanation game class that quantifies bias in optimization algorithms by comparing
    performance against an ensemble of diverse optimizers.

Usage Notes:
    This module requires integration with the HyperSHAP library and appropriate input data structures. It is designed
    to help researchers understand and mitigate biases in machine learning optimization processes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

from hypershap.games import AbstractHPIGame
from hypershap.task import OptimizerBiasExplanationTask


class OptimizerBiasGame(AbstractHPIGame):
    """An explanation game to measure bias in hyperparameter optimizers.

    This class extends the `AbstractHPIGame` base class and is specifically designed
    to quantify how much an optimizer is biased toward tuning certain hyperparameters.
    To this end a set of diverse optimizers is used as a reference.

    Attributes:
        explanation_task (OptimizerBiasExplanationTask): The task that defines the game.
            This includes the configuration sapce, the surrogate model and a baseline configuration, as well as the
            optimizer of interest and the ensemble of diverse optimizers.


    Methods:
        evaluate_single_coalition: Computes the marginal contribution of a coalition (subset of features)
            by comparing the optimizer of interest against an optimizer ensemble. This method is called internally
            during the game's main evaluation process.

    Note:
        The game evaluates coalitions based on the difference between the outcome using the primary optimizer
        and the maximum outcome achieved by any optimizer in the provided ensemble.

    """

    def __init__(
        self,
        explanation_task: OptimizerBiasExplanationTask,
        n_workers: int | None = None,
        verbose: bool | None = None,
    ) -> None:
        """Initialize an instance of `OptimizerBiasGame`.

        Args:
            explanation_task (OptimizerBiasExplanationTask): The task that contains all necessary
                information for defining the game. This includes the configuration space, the surrogate model, the
                optimizer of interest, and the ensemble of diverse optimizers.
            n_workers: The number of worker threads to use for parallel evaluation
                of coalitions. Defaults to None meaning no parallelization.  Using more workers can significantly
                speed up the computation of Shapley values.  The maximum number of workers is capped by the number of coalitions.
            verbose:  A boolean indicating whether to print verbose messages during
                computation. Defaults to None.  When set to True, the method prints
                debugging information and progress updates.

        Example:
            >>> from hypershap.task import OptimizerBiasExplanationTask
            >>> # Create an explanation task instance with specific parameters
            >>> expl_task = OptimizerBiasExplanationTask(...)
            >>> game = OptimizerBiasGame(expl_task)

        """
        super().__init__(explanation_task, n_workers=n_workers, verbose=verbose)

    def _get_explanation_task(self) -> OptimizerBiasExplanationTask:
        if isinstance(self.explanation_task, OptimizerBiasExplanationTask):
            return self.explanation_task
        raise ValueError  # pragma: no cover

    def evaluate_single_coalition(self, coalition: np.ndarray) -> float:
        """Evaluate a single coalition by comparing against an optimizer ensemble.

        Args:
            coalition (np.ndarray): A binary array indicating which hyperparameters are included in the coalition.
                The array has shape (n_hyperparameters,) where True means hyperparameters is included and False
                 otherwise.

        Returns:
            float: The marginal contribution of the coalition. It is computed as the difference between
                the outcome when using the optimizer of interest with the given coalition, versus the best outcome
                achievable by any optimizer in the ensemble (including the one of interest).

        Note:
            This method overrides a base class abstract method and must be implemented to provide specific game logic.
            The value returned here is used for computing Shapley values or other cooperative game theory measures.

        Example:
            >>> coalition = np.array([1, 0, 1])  # Hyperparameters 0 and 2 are included
            >>> marginal_contribution = game.evaluate_single_coalition(coalition)
            >>> print(marginal_contribution)  # Outputs the difference in outcomes for this coalition.

        """
        optimizer_res = self._get_explanation_task().optimizer_of_interest.search(coalition)
        optimizer_ensemble_res = [
            optimizer.search(coalition) for optimizer in self._get_explanation_task().optimizer_ensemble
        ]
        return optimizer_res - max([*optimizer_ensemble_res, optimizer_res])
