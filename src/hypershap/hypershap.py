"""HyperSHAP main interface to work on explanation for a given task.

This module provides the main interface for working with HyperSHAP to access explanations regarding ablation, tunability,
sensitivity, and optimizer bias.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ConfigSpace import Configuration

    from hypershap.utils import ConfigSpaceSearcher

import logging

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from shapiq import ExactComputer, InteractionValues

from hypershap.games import (
    AblationGame,
    AbstractHPIGame,
    MistunabilityGame,
    OptimizerBiasGame,
    SensitivityGame,
    TunabilityGame,
)
from hypershap.task import (
    AblationExplanationTask,
    ExplanationTask,
    MistunabilityExplanationTask,
    OptimizerBiasExplanationTask,
    SensitivityExplanationTask,
    TunabilityExplanationTask,
)
from hypershap.utils import RandomConfigSpaceSearcher

logger = logging.getLogger(__name__)


class NoInteractionValuesError(ValueError):
    """Exception raised when no interaction values are present for plotting."""

    def __init__(self) -> None:
        """Initialize the no interaction values error."""
        super().__init__("No interaction values present for plotting.")


class HyperSHAP:
    """A class for computing and visualizing HyperSHAP Shapley values and interactions.

    Attributes:
        explanation_task (ExplanationTask): The task responsible for generating explanations.
        last_interaction_values (InteractionValues | None): The cached interaction values for plotting shortcuts.

    Methods:
        __init__(explanation_task: ExplanationTask):
            Initializes the HyperSHAP instance with an explanation task.

        ablation(config_of_interest: Configuration, baseline_config: Configuration, index: str = "FSII", order: int = 2) -> InteractionValues:
            Computes and returns the interaction values for ablation analysis.

        tunability(baseline_config: Configuration | None, index: str = "FSII", order: int = 2) -> InteractionValues:
            Computes and returns the interaction values for tunability analysis.

        optimizer_bias(optimizer_of_interest: ConfigSpaceSearcher, optimizer_ensemble: list[ConfigSpaceSearcher], index: str = "FSII", order: int = 2) -> InteractionValues:
            Computes and returns the interaction values for optimizer bias analysis.

        plot_si_graph(interaction_values: InteractionValues | None = None, save_path: str | None = None):
            Plots the SHAP interaction values as a graph.

    """

    def __init__(
        self,
        explanation_task: ExplanationTask,
        n_workers: int | None = None,
        verbose: bool | None = None,
    ) -> None:
        """Initialize the HyperSHAP instance with an explanation task.

        Args:
            explanation_task (ExplanationTask): The task responsible for generating explanations.
            n_workers: The number of worker threads to use for parallel evaluation
                of coalitions. Defaults to None meaning no parallelization.  Using more workers can significantly
                speed up the computation of Shapley values.  The maximum number of workers is capped by the number of coalitions.
            verbose:  A boolean indicating whether to print verbose messages during
                computation. Defaults to None.  When set to True, the method prints
                debugging information and progress updates.

        """
        self.explanation_task = explanation_task
        self.last_interaction_values = None
        self.n_workers = n_workers
        self.verbose = verbose

    def __get_interaction_values(self, game: AbstractHPIGame, index: str = "FSII", order: int = 2) -> InteractionValues:
        # instantiate exact computer if number of hyperparameters is small enough
        ec = ExactComputer(n_players=game.get_num_hyperparameters(), game=game)  # pyright: ignore

        # compute interaction values with the given index and order
        interaction_values = ec(index=index, order=order)

        # cache current interaction values for plotting shortcuts
        self.last_interaction_values = interaction_values

        return interaction_values

    def ablation(
        self,
        config_of_interest: Configuration,
        baseline_config: Configuration,
        index: str = "FSII",
        order: int = 2,
    ) -> InteractionValues:
        """Compute and return the interaction values for ablation analysis.

        Args:
            config_of_interest (Configuration): The configuration of interest.
            baseline_config (Configuration): The baseline configuration.
            index (str, optional): The index to use for computing interaction values. Defaults to "FSII".
            order (int, optional): The order of the interaction values. Defaults to 2.

        Returns:
            InteractionValues: The computed interaction values.

        """
        # setup explanation task
        ablation_task: AblationExplanationTask = AblationExplanationTask(
            config_space=self.explanation_task.config_space,
            surrogate_model=self.explanation_task.surrogate_model,
            baseline_config=baseline_config,
            config_of_interest=config_of_interest,
        )

        # setup ablation game and get interaction values
        ag = AblationGame(
            explanation_task=ablation_task,
            n_workers=self.n_workers,
            verbose=self.verbose,
        )
        return self.__get_interaction_values(game=ag, index=index, order=order)

    def tunability(
        self,
        baseline_config: Configuration | None = None,
        index: str = "FSII",
        order: int = 2,
        n_samples: int = 10_000,
    ) -> InteractionValues:
        """Compute and return the interaction values for tunability analysis.

        Args:
            baseline_config (Configuration | None, optional): The baseline configuration. Defaults to None.
            index (str, optional): The index to use for computing interaction values. Defaults to "FSII".
            order (int, optional): The order of the interaction values. Defaults to 2.
            n_samples (int, optional): The number of samples to use for simulating HPO. Defaults to 10_000.

        Returns:
            InteractionValues: The computed interaction values.

        """
        if baseline_config is None:
            baseline_config = self.explanation_task.config_space.get_default_configuration()

        # setup explanation task
        tunability_task: TunabilityExplanationTask = TunabilityExplanationTask(
            config_space=self.explanation_task.config_space,
            surrogate_model=self.explanation_task.surrogate_model,
            baseline_config=baseline_config,
        )

        # setup tunability game and get interaction values
        tg = TunabilityGame(
            explanation_task=tunability_task,
            cs_searcher=RandomConfigSpaceSearcher(
                explanation_task=tunability_task,
                n_samples=n_samples,
                mode="max",
            ),
            n_workers=self.n_workers,
            verbose=self.verbose,
        )
        return self.__get_interaction_values(game=tg, index=index, order=order)

    def sensitivity(
        self,
        baseline_config: Configuration | None = None,
        index: str = "FSII",
        order: int = 2,
        n_samples: int = 10_000,
    ) -> InteractionValues:
        """Compute and return the interaction values for sensitivity analysis.

        Args:
            baseline_config (Configuration | None, optional): The baseline configuration. Defaults to None.
            index (str, optional): The index to use for computing interaction values. Defaults to "FSII".
            order (int, optional): The order of the interaction values. Defaults to 2.
            n_samples (int, optional): The number of samples to use for simulating HPO. Defaults to 10_000.

        Returns:
            InteractionValues: The computed interaction values.

        """
        if baseline_config is None:
            baseline_config = self.explanation_task.config_space.get_default_configuration()

        # setup explanation task
        sensitivity_task: SensitivityExplanationTask = SensitivityExplanationTask(
            config_space=self.explanation_task.config_space,
            surrogate_model=self.explanation_task.surrogate_model,
            baseline_config=baseline_config,
        )

        # setup tunability game and get interaction values
        tg = SensitivityGame(
            explanation_task=sensitivity_task,
            cs_searcher=RandomConfigSpaceSearcher(
                explanation_task=sensitivity_task,
                n_samples=n_samples,
                mode="var",
            ),
            n_workers=self.n_workers,
            verbose=self.verbose,
        )
        return self.__get_interaction_values(game=tg, index=index, order=order)

    def mistunability(
        self,
        baseline_config: Configuration | None = None,
        index: str = "FSII",
        order: int = 2,
        n_samples: int = 10_000,
    ) -> InteractionValues:
        """Compute and return the interaction values for mistunability analysis.

        Args:
            baseline_config (Configuration | None, optional): The baseline configuration. Defaults to None.
            index (str, optional): The index to use for computing interaction values. Defaults to "FSII".
            order (int, optional): The order of the interaction values. Defaults to 2.
            n_samples (int, optional): The number of samples to use for simulating HPO. Defaults to 10_000.

        Returns:
            InteractionValues: The computed interaction values.

        """
        if baseline_config is None:
            baseline_config = self.explanation_task.config_space.get_default_configuration()

        # setup explanation task
        mistunability_task: MistunabilityExplanationTask = MistunabilityExplanationTask(
            config_space=self.explanation_task.config_space,
            surrogate_model=self.explanation_task.surrogate_model,
            baseline_config=baseline_config,
        )

        # setup tunability game and get interaction values
        tg = MistunabilityGame(
            explanation_task=mistunability_task,
            cs_searcher=RandomConfigSpaceSearcher(
                explanation_task=mistunability_task,
                n_samples=n_samples,
                mode="min",
            ),
            n_workers=self.n_workers,
            verbose=self.verbose,
        )
        return self.__get_interaction_values(game=tg, index=index, order=order)

    def optimizer_bias(
        self,
        optimizer_of_interest: ConfigSpaceSearcher,
        optimizer_ensemble: list[ConfigSpaceSearcher],
        index: str = "FSII",
        order: int = 2,
    ) -> InteractionValues:
        """Compute and return the interaction values for optimizer bias analysis.

        Args:
            optimizer_of_interest (ConfigSpaceSearcher): The optimizer of interest.
            optimizer_ensemble (list[ConfigSpaceSearcher]): The ensemble of optimizers.
            index (str, optional): The index to use for computing interaction values. Defaults to "FSII".
            order (int, optional): The order of the interaction values. Defaults to 2.

        Returns:
            InteractionValues: The computed interaction values.

        """
        # setup explanation task
        optimizer_bias_task: OptimizerBiasExplanationTask = OptimizerBiasExplanationTask(
            config_space=self.explanation_task.config_space,
            surrogate_model=self.explanation_task.surrogate_model,
            optimizer_of_interest=optimizer_of_interest,
            optimizer_ensemble=optimizer_ensemble,
        )

        # setup optimizer bias game and get interaction values
        og = OptimizerBiasGame(explanation_task=optimizer_bias_task, n_workers=self.n_workers, verbose=self.verbose)
        return self.__get_interaction_values(game=og, index=index, order=order)

    def plot_si_graph(
        self,
        interaction_values: InteractionValues | None = None,
        save_path: str | None = None,
        no_show: bool | None = None,
    ) -> None:
        """Plot the SHAP interaction values as a graph.

        Args:
            interaction_values (InteractionValues | None, optional): The interaction values to plot. Defaults to None.
            save_path (str | None, optional): The path to save the plot. Defaults to None.
            no_show (bool | None, optional): Do not show the plot if set to true. Defaults to None.

        """
        if interaction_values is None and self.last_interaction_values is None:
            raise NoInteractionValuesError

        # if given interaction values use those, else use cached interaction values
        iv = interaction_values if interaction_values is not None else self.last_interaction_values

        if not isinstance(iv, InteractionValues):  # pragma: no cover
            raise TypeError  # pragma: no cover

        hyperparameter_names = self.explanation_task.get_hyperparameter_names()

        def get_circular_layout(n_players: int) -> dict:
            original_graph, graph_nodes = nx.Graph(), []
            for i in range(n_players):
                original_graph.add_node(i, label=i)
                graph_nodes.append(i)
            return nx.circular_layout(original_graph)

        pos = get_circular_layout(n_players=self.explanation_task.get_num_hyperparameters())
        iv.plot_si_graph(
            show=False,
            size_factor=3.0,
            feature_names=hyperparameter_names,
            pos=pos,
            n_interactions=1_000,
            compactness=1e50,
        )
        plt.tight_layout()

        if save_path is not None:
            plt.savefig(save_path)
            logger.info("Saved SI graph to %s", save_path)

        if no_show is None or not no_show:  # pragma: no cover
            plt.show()  # pragma: no cover

    def plot_upset(
        self,
        interaction_values: InteractionValues | None = None,
        save_path: str | None = None,
        no_show: bool | None = None,
    ) -> None:
        """Plot the SHAP interaction values as an upset plot graph.

        Args:
            interaction_values (InteractionValues | None, optional): The interaction values to plot. Defaults to None.
            save_path (str | None, optional): The path to save the plot. Defaults to None.
            no_show (bool | None, optional): Do not show the plot if set to true. Defaults to None.

        """
        if interaction_values is None and self.last_interaction_values is None:
            raise NoInteractionValuesError

        # if given interaction values use those, else use cached interaction values
        iv = interaction_values if interaction_values is not None else self.last_interaction_values

        if not isinstance(iv, InteractionValues):  # pragma: no cover
            raise TypeError  # pragma: no cover

        hyperparameter_names = self.explanation_task.get_hyperparameter_names()

        fig = iv.plot_upset(feature_names=hyperparameter_names, show=False)

        if fig is None:  # pragma: no cover
            raise TypeError  # pragma: no cover

        ax = fig.get_axes()[0]
        ax.set_ylabel("Performance Gain")
        # also add "parameter" to the y-axis label
        ax = fig.get_axes()[1]
        ax.set_ylabel("Hyperparameter")

        plt.tight_layout()

        if save_path is not None:
            plt.savefig(save_path)

        if no_show is None or not no_show:  # pragma: no cover
            plt.show()  # pragma: no cover

    def plot_force(
        self,
        interaction_values: InteractionValues | None = None,
        save_path: str | None = None,
        no_show: bool | None = None,
    ) -> None:
        """Plot the SHAP interaction values as a forceplot graph.

        Args:
            interaction_values: Interaction values to plot. Defaults to None.
            save_path: The path to save the plot. Defaults to None.
            no_show (bool | None, optional): Do not show the plot if set to true. Defaults to None.

        """
        if interaction_values is None and self.last_interaction_values is None:
            raise NoInteractionValuesError

        # if given interaction values use those, else use cached interaction values
        iv = interaction_values if interaction_values is not None else self.last_interaction_values

        if not isinstance(iv, InteractionValues):  # pragma: no cover
            raise TypeError  # pragma: no cover

        hyperparameter_names = self.explanation_task.get_hyperparameter_names()

        iv.plot_force(feature_names=np.array(hyperparameter_names), show=False)
        plt.tight_layout()

        if save_path is not None:
            plt.savefig(save_path)

        if no_show is None or not no_show:  # pragma: no cover
            plt.show()  # pragma: no cover
