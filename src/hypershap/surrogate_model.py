"""The surrogate module defines the basic classes for surrogate models.

It provides methods for training and evaluating a model that approximates
the relationship between input hyperparameters and performance.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, cast

if TYPE_CHECKING:
    from ConfigSpace import Configuration, ConfigurationSpace
    from sklearn.base import BaseEstimator

from abc import ABC, abstractmethod

import numpy as np
from sklearn.ensemble import RandomForestRegressor


class SklearnRegressorProtocol(Protocol):
    """Defines the interface for scikit-learn-like regression models.

    This protocol specifies the required methods for a class to be considered
    a compatible scikit-learn regression model.  It mandates the presence of `fit`,
    `predict`, and `score` methods, mirroring the structure of many
    scikit-learn estimators.

    Attributes:
        fit (callable): A method that fits the model to the provided data.
                         It should accept training data (X) and target variables (y)
                         as arguments, and any optional fit parameters. It should return
                         the fitted model instance itself (allowing for chaining).
        predict (callable): A method that generates predictions for a given input dataset.
                           It accepts input data (X) and returns predictions as a NumPy array.
        score (callable): A method that evaluates the model's performance on a given dataset.
                          It accepts input data (X) and corresponding target variables (y)
                          and returns a scalar performance score (e.g., R-squared, MSE).

    """

    def fit(self, X: Any, y: Any, **fit_params: Any) -> None:
        """Fit the regression model to the provided training data.

        Args:
            X (np.ndarray): The training data features.
            y (np.ndarray): The training data target variables.
            **fit_params (Any): Optional keyword arguments passed to the fit method.

        Returns:
            'SklearnRegressorProtocol': The fitted regression model instance itself, allowing for method chaining.

        """
        ...

    def predict(self, X: Any) -> np.ndarray:
        """Generate predictions for a given input dataset.

        Args:
            X (np.ndarray): The input data features for prediction.

        Returns:
            np.ndarray: The predicted target values as a NumPy array.

        """
        ...

    def score(self, X: Any, y: Any) -> float:
        """Evaluate the model's performance on a given dataset.

        Args:
            X (np.ndarray): The input data features.
            y (np.ndarray): The corresponding target variables.

        Returns:
            float: A scalar performance score, representing the model's accuracy on the dataset.

        """
        ...


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
        res = self.evaluate(np.array(config.get_array()))
        if not isinstance(res, float):  # pragma: no cover
            raise TypeError  # pragma: no cover
        return res

    def evaluate_config_batch(self, config_batch: list[Configuration]) -> list[float]:
        """Evaluate a batch of configurations using the surrogate model.

        Args:
            config_batch: A list of configurations to evaluate.

        Returns:
            A list of predicted performances for the given configurations.

        """
        res = self.evaluate(np.array([config.get_array() for config in config_batch]))
        if not isinstance(res, list):  # pragma: no cover
            raise TypeError  # pragma: no cover
        return res

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
        res = self.evaluate(config.get_array())
        if not isinstance(res, float):  # pragma: no cover
            raise TypeError  # pragma: no cover
        return res

    def evaluate_config_batch(self, config_batch: list[Configuration]) -> list[float]:
        """Evaluate a batch of configurations.

        Args:
            config_batch: A list of configurations to evaluate.

        Returns:
            A list of predicted performances for the given configurations.

        """
        res = self.evaluate(np.array([config.get_array() for config in config_batch]))
        if not isinstance(res, list):  # pragma: no cover
            raise TypeError  # pragma: no cover
        return res

    def evaluate(self, config_array: np.ndarray) -> float | list[float]:
        """Evaluate a configuration (or batch of configurations).

        Args:
            config_array: A numpy array representing the configuration(s).

        Returns:
            The predicted performance(s).

        """
        if config_array.ndim == 1:
            config_array = config_array.reshape(1, -1)

        base_model = cast("SklearnRegressorProtocol", self.base_model)
        predictions = base_model.predict(config_array)

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

        pipeline = cast("SklearnRegressorProtocol", base_model)
        pipeline.fit(train_x, train_y)

        super().__init__(config_space, base_model)
