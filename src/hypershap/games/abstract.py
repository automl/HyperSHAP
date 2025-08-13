from abc import abstractmethod

import numpy as np
from shapiq import Game

from hypershap.task import ExplanationTask


class AbstractHPIGame(Game):
    def __init__(self, explanation_task: ExplanationTask, n_workers: int = 1, verbose: bool = False):
        self.explanation_task = explanation_task
        self.n_workers = n_workers
        self.verbose = verbose

        # determine the value of the empty coalition so that we can normalize wrt to that performance
        normalization_value = self.evaluate_single_coalition(
            np.array([False] * explanation_task.get_num_hyperparameters())
        )

        super().__init__(
            n_players=explanation_task.get_num_hyperparameters(),
            normalize=True,
            normalization_value=normalization_value,
        )

    def value_function(self, coalitions: np.ndarray) -> np.ndarray:
        value_list = []
        for coalition in coalitions:
            value_list += [self.evaluate_single_coalition(coalition)]
        return np.array(value_list)

    @abstractmethod
    def evaluate_single_coalition(self, coalition: np.ndarray) -> float:
        pass

    def get_num_hyperparameters(self) -> int:
        return self.explanation_task.get_num_hyperparameters()

    def get_hyperparameter_names(self) -> list[str]:
        return self.explanation_task.get_hyperparameter_names()
