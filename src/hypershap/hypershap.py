from typing import Optional

import matplotlib.pyplot as plt
from ConfigSpace import Configuration
from shapiq import ExactComputer, InteractionValues

from hypershap.games.ablation import AblationGame
from hypershap.games.abstract import AbstractHPIGame
from hypershap.games.optimizerbias import OptimizerBiasGame
from hypershap.games.tunability import TunabilityGame
from hypershap.optimizer import Optimizer
from hypershap.task import (
    AblationExplanationTask,
    ExplanationTask,
    OptimizerBiasExplanationTask,
    TunabilityExplanationTask,
)


class NoInteractionValuesError(ValueError):
    """Exception raised when no interaction values are present for plotting."""

    def __init__(self):
        super().__init__("No interaction values present for plotting.")


class HyperSHAP:
    def __init__(self, explanation_task: ExplanationTask):
        self.explanation_task = explanation_task
        self.last_interaction_values = None

    def __get_interaction_values(self, game: AbstractHPIGame, index: str = "FSII", order: int = 2):
        # instantiate exact computer if number of hyperparameters is small enough
        ec = ExactComputer(n_players=game.get_num_hyperparameters(), game=game)

        # compute interaction values with the given index and order
        interaction_values = ec(index=index, order=order)

        return interaction_values

    def ablation(
        self, config_of_interest: Configuration, baseline_config: Configuration, index: str = "FSII", order: int = 2
    ) -> InteractionValues:
        # setup explanation task
        ablation_task: AblationExplanationTask = AblationExplanationTask(
            config_space=self.explanation_task.config_space,
            surrogate_model=self.explanation_task.surrogate_model,
            baseline_config=baseline_config,
            config_of_interest=config_of_interest,
        )

        # setup ablation game and get interaction values
        ag = AblationGame(explanation_task=ablation_task)
        interaction_values = self.__get_interaction_values(game=ag, index=index, order=order)

        # cache current interaction values for plotting shortcuts
        self.last_interaction_values = interaction_values

        return interaction_values

    def tunability(
        self, baseline_config: Configuration = None, index: str = "FSII", order: int = 2
    ) -> InteractionValues:
        # setup explanation task
        tunability_task: TunabilityExplanationTask = TunabilityExplanationTask(
            config_space=self.explanation_task.config_space,
            surrogate_model=self.explanation_task.surrogate_model,
            baseline_config=baseline_config,
        )

        # setup tunability game and get interaction values
        tg = TunabilityGame(explanation_task=tunability_task)
        interaction_values = self.__get_interaction_values(game=tg, index=index, order=order)

        # cache current interaction values for plotting shortcuts
        self.last_interaction_values = interaction_values

        return interaction_values

    def optimizer_bias(
        self, optimizer_of_interest: Optimizer, optimizer_ensemble: list[Optimizer], index: str = "FSII", order: int = 2
    ) -> InteractionValues:
        # setup explanation task
        optimizer_bias_task: OptimizerBiasExplanationTask = OptimizerBiasExplanationTask(
            config_space=self.explanation_task.config_space,
            surrogate_model=self.explanation_task.surrogate_model,
            optimizer_of_interest=optimizer_of_interest,
            optimizer_ensemble=optimizer_ensemble,
        )

        # setup optimizer bias game and get interaction values
        og = OptimizerBiasGame(explanation_task=optimizer_bias_task)
        interaction_values = self.__get_interaction_values(hpi_game=og, index=index, order=2)

        # cache current interaction values for plotting shortcuts
        self.last_interaction_values = interaction_values

        return interaction_values

    def plot_si_graph(self, interaction_values: Optional[InteractionValues] = None, save_path: Optional[str] = None):
        if interaction_values is None and self.last_interaction_values is None:
            raise NoInteractionValuesError()

        # if given interaction values use those, else use cached interaction values
        iv = interaction_values if interaction_values is not None else self.last_interaction_values
        hyperparameter_names = self.explanation_task.get_hyperparameter_names()

        import networkx as nx

        def get_circular_layout(n_players: int):
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
            print(f"Saved SI graph to {save_path}")

        plt.show()
