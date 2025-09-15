"""The multi-data module provides a wrapper for extending explanation games to a multi-data setting."""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING

import numpy as np

from hypershap.games.ablation import MultiBaselineAblationGame
from hypershap.games.abstract import AbstractHPIGame
from hypershap.games.tunability import SearchBasedGame

if TYPE_CHECKING:
    from hypershap.task import ExplanationTask

from hypershap.utils import Aggregation, evaluate_aggregation


class MultiDataHPIGame(AbstractHPIGame):
    """The multi-data game generalizes an explanation game to multiple datasets."""

    def __init__(
        self,
        explanation_task: ExplanationTask,
        base_game: AbstractHPIGame,
        aggregation: Aggregation,
    ) -> None:
        """Initialize the multi-data game wrapper.

        Args:
            explanation_task: The explanation task containing the configuration space and surrogate model.
            base_game: The base game instance.
            aggregation: The aggregation method to use.

        """
        self.aggregation = aggregation
        self.base_game = base_game

        self.sub_games = []
        for surrogate_model in explanation_task.get_surrogate_model_list():
            game_copy = copy.deepcopy(base_game)
            game_copy.explanation_task.surrogate_model = surrogate_model
            if isinstance(game_copy, SearchBasedGame):
                game_copy.cs_searcher.explanation_task.surrogate_model = surrogate_model
            if isinstance(game_copy, MultiBaselineAblationGame):
                for ag in game_copy.ablation_games:
                    ag.explanation_task.surrogate_model = surrogate_model
            self.sub_games.append(game_copy)

        super().__init__(explanation_task)

    def evaluate_single_coalition(self, coalition: np.ndarray) -> float:
        """Evaluate the multi-data game on the coalition.

        Args:
            coalition: The coalition to evaluate.

        Returns: The value of the multi-data game on the coalition.

        """
        vals = np.array([game.evaluate_single_coalition(coalition) for game in self.sub_games])
        return evaluate_aggregation(self.aggregation, vals)
