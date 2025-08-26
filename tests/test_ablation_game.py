"""Test suite for the ablation games.

In this test suite, we test the functionality of the ablation game and its subcomponents.
"""

from __future__ import annotations

import numpy as np
import numpy.testing as npt
from ConfigSpace import ConfigurationSpace

from hypershap.games.ablation import AblationGame
from hypershap.surrogate_model import SurrogateModel
from hypershap.task import AblationExplanationTask


def test_ablation_game() -> None:
    """Tests the ablation game with a mockup config space, surrogate model, baseline config and config of interest."""
    cs = ConfigurationSpace(
        name="myspace",
        space={
            "a": (0.1, 1.5),  # UniformFloat
            "b": (2, 10),  # UniformInt
            "c": ["mouse", "cat", "dog"],  # Categorical
        },
    )

    class MockupSurrogateModel(SurrogateModel):
        def __init__(self) -> None:
            """Initialize the mockup surrogate model."""
            super().__init__(None)
            self.last_queried_config = None

        def evaluate(self, config: np.ndarray) -> float:
            """Evaluate the surrogate model for a given configuration.

            Args:
                config: The configuration to evaluate.

            Returns: The performance value obtained for the given configuration.

            """
            self.last_queried_config = config
            return 0.0

    surrogate_model = MockupSurrogateModel()

    baseline_config = cs.sample_configuration()
    config_of_interest = cs.sample_configuration()

    ablation_task = AblationExplanationTask(
        config_space=cs,
        surrogate_model=surrogate_model,
        baseline_config=baseline_config,
        config_of_interest=config_of_interest,
    )

    ablation_game = AblationGame(ablation_task)
    ablation_game.evaluate_single_coalition(np.zeros(len(cs)))

    npt.assert_array_equal(
        baseline_config.get_array(),
        surrogate_model.last_queried_config,
        "Mismatch between expected config and actual config",
    )

    ablation_game.evaluate_single_coalition(np.ones(len(cs)))
    npt.assert_array_equal(
        config_of_interest.get_array(),
        surrogate_model.last_queried_config,
        "Mismatch between expected config and actual config",
    )

    game_hp_names = ablation_game.get_hyperparameter_names()
    assert game_hp_names == ["a", "b", "c"], "Hyperparameter names mismatch"
