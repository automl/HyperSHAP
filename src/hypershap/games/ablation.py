import numpy as np

from hypershap.games.abstract import AbstractHPIGame
from hypershap.task import AblationExplanationTask


class AblationGame(AbstractHPIGame):
    def __init__(
        self,
        explanation_task: AblationExplanationTask,
        n_workers: int = 1,
        verbose: bool = False,
    ):
        """
        The ablation game generates local explanations for hyperparameter configurations by evaluating all potential
        ablation paths switching from a baseline configuration to an optimized configuration value by value.
        Args:
            explanation_task (AblationExplanationTask):
        """
        super().__init__(explanation_task, n_workers=n_workers, verbose=verbose)

    def evaluate_single_coalition(self, coalition: np.ndarray) -> float:
        baseline_cfg = self._get_explanation_task().baseline_config.get_array()
        cfg_of_interest = self._get_explanation_task().config_of_interest.get_array()
        blend = np.where(coalition == 0, baseline_cfg, cfg_of_interest)
        coalition_value = self._get_explanation_task().surrogate_model.evaluate(blend)
        return coalition_value

    def _get_explanation_task(self) -> AblationExplanationTask:
        return self.explanation_task
