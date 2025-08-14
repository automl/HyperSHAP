"""The surrogate module defines the basic classes for surrogate models.

It provides methods for training and evaluating a model that approximates
the relationship between input hyperparameters and performance.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ConfigSpace import Configuration, ConfigurationSpace
    from sklearn.base import BaseEstimator

from abc import ABC, abstractmethod

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OrdinalEncoder


class SurrogateModel(ABC):
    """An abstract class for defining the interface of surrogate models.

    This class defines the basic methods that all surrogate models should implement,
    allowing for a consistent interface for evaluating different models.
    """

    def __init__(self, config_space: ConfigurationSpace) -> None:
        """Initialize the SurrogateModel with a configuration space.

        Args:
            config_space: The configuration space for the surrogate model.

        """
        self.config_space = config_space

    def evaluate_config(self, config: Configuration) -> float:
        """Evaluate a single configuration using the surrogate model.

        Args:
            config: The configuration to evaluate.

        Returns:
            The predicted performance for the given configuration.

        """
        return self.evaluate(np.array(config.get_array()))

    def evaluate_config_batch(self, config_batch: list[Configuration]) -> list[float]:
        """Evaluate a batch of configurations using the surrogate model.

        Args:
            config_batch: A list of configurations to evaluate.

        Returns:
            A list of predicted performances for the given configurations.

        """
        return self.evaluate(np.array([config.get_array() for config in config_batch]))

    @abstractmethod
    def evaluate(self, config_array: np.ndarray) -> float | list[float]:
        """Evaluate a configuration (or batch of configurations) represented as a numpy array.

        Args:
            config_array: A numpy array representing the configuration(s).

        Returns:
            The predicted performance(s).

        """


class ModelBasedSurrogateModel(SurrogateModel):
    """A surrogate model based on a pre-trained machine learning model."""

    def __init__(self, config_space: ConfigurationSpace, base_model: BaseEstimator) -> None:
        """Initialize the ModelBasedSurrogateModel with a configuration space and a base model.

        Args:
            config_space: The configuration space.
            base_model: The base machine learning model.

        """
        super().__init__(config_space)
        self.base_model = base_model

    def evaluate_config(self, config: Configuration) -> float:
        """Evaluate a single configuration.

        Args:
            config: The configuration to evaluate.

        Returns:
            The predicted performance for the given configuration.

        """
        return self.evaluate(config.get_array())

    def evaluate_config_batch(self, config_batch: list[Configuration]) -> list[float]:
        """Evaluate a batch of configurations.

        Args:
            config_batch: A list of configurations to evaluate.

        Returns:
            A list of predicted performances for the given configurations.

        """
        return self.evaluate(np.array([config.get_array() for config in config_batch]))

    def evaluate(self, config_array: np.ndarray) -> float | list[float]:
        """Evaluate a configuration (or batch of configurations).

        Args:
            config_array: A numpy array representing the configuration(s).

        Returns:
            The predicted performance(s).

        """
        if config_array.ndim == 1:
            config_array = config_array.reshape(1, -1)

        predictions = self.base_model.predict(config_array)
        if predictions.shape == (1,):  # Check for a 1-element array (scalar)
            return float(predictions[0])  # Convert to a Python float
        return predictions.tolist()  # Convert to a Python list


class DataBasedSurrogateModel(ModelBasedSurrogateModel):
    """A surrogate model trained on a dataset of configurations and their performance."""

    def __init__(
        self,
        config_space: ConfigurationSpace,
        data: list[tuple[Configuration, float]],
        base_model: BaseEstimator | None = None,
    ) -> None:
        """Initialize the DataBasedSurrogateModel with data and an optional base model.

        Args:
            config_space: The configuration space.
            data: The data to be used for fitting the surrogate model.  Each element
                  is a tuple of (Configuration, float).
            base_model: The base model to be used for fitting the surrogate model.
                        If None, a RandomForestRegressor is used.

        """
        train_x = np.array([obs[0].get_array() for obs in data])
        train_y = np.array([obs[1] for obs in data])

        if base_model is None:
            base_model = RandomForestRegressor()

        pipeline = make_pipeline(
            OrdinalEncoder(categories="auto", handle_unknown="use_encoded_value", unknown_value=np.nan),
            SimpleImputer(missing_values=np.nan, strategy="median"),
            base_model,
        )
        pipeline.fit(train_x, train_y)
        super().__init__(config_space, pipeline)
