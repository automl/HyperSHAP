"""Test suite for the tunability games.

In this test suite, we test the functionality of the tunability game, its variants and its subcomponents.
"""

from __future__ import annotations

import numpy as np
from ConfigSpace import ConfigurationSpace

from hypershap.surrogate_model import SurrogateModel
from hypershap.task import TunabilityExplanationTask
from hypershap.utils import RandomConfigSpaceSearcher


class MockupSurrogateModel(SurrogateModel):
    """A mock surrogate model class that inherits from SurrogateModel.

    This class is used for testing purposes and provides a simple implementation of the evaluate method.
    """

    def __init__(self) -> None:
        """Initialize the MockupSurrogateModel instance.

        Args:
            None

        Returns:
            None

        """
        super().__init__(None)

    def evaluate(self, config_array: np.ndarray) -> float:
        """Evaluate the given configuration array using a simple linear function.

        The function returns the predicted value for the given configuration.

        Args:
            config_array (np.ndarray): A 2D numpy array containing the configuration values.

        Returns:
            float: The predicted value for the given configuration.

        """
        if config_array.ndim == 1:
            config_array = config_array.reshape(1, -1)

        vals = 0.5 * config_array[:, 0] + 0.2 * config_array[:, 1]

        if vals.shape == (1,):  # Check for a 1-element array (scalar)
            return float(vals[0])  # Convert to a Python float
        return vals.tolist()  # Convert to a Python list


def test_config_space_searcher() -> None:
    """Test the RandomConfigSpaceSearcher class.

    This function creates an instance of ConfigurationSpace, MockupSurrogateModel,
    and TunabilityExplanationTask, then uses these instances to create an
    instance of RandomConfigSpaceSearcher.

    Args:
        None

    Returns:
        None

    """
    cs = ConfigurationSpace(
        name="myspace",
        space={
            "a": (0.1, 1.5),  # UniformFloat
            "b": (2, 10),  # UniformInt
        },
    )

    surrogate_model = MockupSurrogateModel()

    cs.seed(42)
    baseline_config = cs.sample_configuration()

    tet = TunabilityExplanationTask(config_space=cs, surrogate_model=surrogate_model, baseline_config=baseline_config)
    rccs = RandomConfigSpaceSearcher(tet, n_samples=1_000)
    res = rccs.search(np.array([True, True]))

    assert res is not None, "No result has been returned by RandomConfigSpaceSearcher."
    assert res >= surrogate_model.evaluate_config(baseline_config), "Returned value is worse than baseline config."
