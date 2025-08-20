"""The task module implements a hierarchy of *explanation tasks* that can be used to explain HPO.

The tasks provide a convenient API to construct surrogate models from different data sources (pretrained estimators,
empirical data, or a black box function) and to add domain specific information such as a baseline configuration or
an optimizer of interest.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

    from ConfigSpace import Configuration, ConfigurationSpace
    from sklearn.base import BaseEstimator

    from hypershap import ConfigSpaceSearcher

from sklearn.ensemble import RandomForestRegressor

from hypershap.surrogate_model import DataBasedSurrogateModel, ModelBasedSurrogateModel, SurrogateModel


class ExplanationTask:
    """Defines the base class for explanation tasks, providing access to the configuration space and surrogate model."""

    def __init__(self, config_space: ConfigurationSpace, surrogate_model: SurrogateModel) -> None:
        """Initialize an ExplanationTask with a configuration space and surrogate model.

        Args:
            config_space: The configuration space for the explanation task.
            surrogate_model: The surrogate model used for the explanation task.

        """
        self.config_space: ConfigurationSpace = config_space
        self.surrogate_model: SurrogateModel = surrogate_model

    def get_num_hyperparameters(self) -> int:
        """Return the number of hyperparameters in the configuration space.

        Returns:
            The number of hyperparameters.

        """
        return len(self.config_space)

    def get_hyperparameter_names(self) -> list[str]:
        """Return the names of the hyperparameters in the configuration space.

        Returns:
            A list of hyperparameter names.

        """
        return list(self.config_space.keys())

    @staticmethod
    def from_base_model(config_space: ConfigurationSpace, base_model: BaseEstimator) -> ExplanationTask:
        """Create an ExplanationTask from a pre-trained base model.

        Args:
            config_space: The configuration space.
            base_model: The pre-trained base model.

        Returns:
            An ExplanationTask instance.

        """
        surrogate_model = ModelBasedSurrogateModel(config_space=config_space, base_model=base_model)
        return ExplanationTask(config_space=config_space, surrogate_model=surrogate_model)

    @staticmethod
    def from_data(
        config_space: ConfigurationSpace,
        data: list[tuple[Configuration, float]],
        base_model: BaseEstimator | None = None,
    ) -> ExplanationTask:
        """Create an ExplanationTask from a dataset of configurations and their performance.

        Args:
            config_space: The configuration space.
            data: A list of tuples, where each tuple contains a configuration and its corresponding performance.
            base_model: The base model to use for training the surrogate model. Defaults to RandomForestRegressor.

        Returns:
            An ExplanationTask instance.

        """
        surrogate_model = DataBasedSurrogateModel(config_space=config_space, data=data, base_model=base_model)
        return ExplanationTask(config_space=config_space, surrogate_model=surrogate_model)

    @staticmethod
    def from_function(
        config_space: ConfigurationSpace,
        function: Callable[[Configuration], float],
        n_samples: int = 1_000,
        base_model: BaseEstimator | None = None,
    ) -> ExplanationTask:
        """Create an ExplanationTask from a function that evaluates configurations.

        Args:
            config_space: The configuration space.
            function: A callable that takes a configuration and returns its performance.
            n_samples: The number of configurations to sample for training the surrogate model. Defaults to 1000.
            base_model: The base model to use for training the surrogate model. Defaults to RandomForestRegressor.

        Returns:
            An ExplanationTask instance.

        """
        samples: list[Configuration] = config_space.sample_configuration(n_samples)
        values: list[float] = [function(config) for config in samples]
        data: list[tuple[Configuration, float]] = list(zip(samples, values, strict=False))
        base_model = base_model if base_model is not None else RandomForestRegressor()

        return ExplanationTask.from_data(config_space=config_space, data=data, base_model=base_model)


class BaselineExplanationTask(ExplanationTask):
    """Defines an explanation task with a baseline configuration."""

    def __init__(
        self,
        config_space: ConfigurationSpace,
        surrogate_model: SurrogateModel,
        baseline_config: Configuration,
    ) -> None:
        """Initialize a BaselineExplanationTask with a baseline configuration.

        Args:
            config_space: The configuration space.
            surrogate_model: The surrogate model.
            baseline_config: The baseline configuration.

        """
        super().__init__(config_space, surrogate_model)
        self.baseline_config = baseline_config


class MultiBaselineExplanationTask(ExplanationTask):
    """Defines an explanation task with multiple baseline configurations."""

    def __init__(
        self,
        config_space: ConfigurationSpace,
        surrogate_model: SurrogateModel,
        baseline_configs: list[Configuration],
    ) -> None:
        """Initialize a MultiBaselineExplanationTask with a list of baseline configurations.

        Args:
            config_space: The configuration space.
            surrogate_model: The surrogate model.
            baseline_configs: A list of baseline configurations.

        """
        super().__init__(config_space, surrogate_model)
        self.baseline_configs = baseline_configs


class AblationExplanationTask(BaselineExplanationTask):
    """Defines an ablation explanation task, comparing a configuration of interest to a baseline."""

    def __init__(
        self,
        config_space: ConfigurationSpace,
        surrogate_model: SurrogateModel,
        baseline_config: Configuration,
        config_of_interest: Configuration,
    ) -> None:
        """Initialize an AblationExplanationTask with a baseline and a configuration of interest.

        Args:
            config_space: The configuration space.
            surrogate_model: The surrogate model.
            baseline_config: The baseline configuration.
            config_of_interest: The configuration of interest.

        """
        super().__init__(config_space, surrogate_model, baseline_config)
        self.config_of_interest = config_of_interest


class SensitivityExplanationTask(BaselineExplanationTask):
    """Defines a sensitivity explanation task."""

    def __init__(
        self,
        config_space: ConfigurationSpace,
        surrogate_model: SurrogateModel,
        baseline_config: Configuration,
    ) -> None:
        """Initialize a SensitivityExplanationTask.

        Args:
            config_space: The configuration space.
            surrogate_model: The surrogate model.
            baseline_config: The baseline configuration.

        """
        super().__init__(config_space, surrogate_model, baseline_config)


class TunabilityExplanationTask(BaselineExplanationTask):
    """Defines a tunability explanation task."""

    def __init__(
        self,
        config_space: ConfigurationSpace,
        surrogate_model: SurrogateModel,
        baseline_config: Configuration,
    ) -> None:
        """Initialize a TunabilityExplanationTask.

        Args:
            config_space: The configuration space.
            surrogate_model: The surrogate model.
            baseline_config: The baseline configuration.

        """
        super().__init__(config_space, surrogate_model, baseline_config)


class MistunabilityExplanationTask(BaselineExplanationTask):
    """Defines a mistunability explanation task."""

    def __init__(
        self,
        config_space: ConfigurationSpace,
        surrogate_model: SurrogateModel,
        baseline_config: Configuration,
    ) -> None:
        """Initialize a MistunabilityExplanationTask.

        Args:
            config_space: The configuration space.
            surrogate_model: The surrogate model.
            baseline_config: The baseline configuration.

        """
        super().__init__(config_space, surrogate_model, baseline_config)


class OptimizerBiasExplanationTask(ExplanationTask):
    """Defines an optimizer bias explanation task."""

    def __init__(
        self,
        config_space: ConfigurationSpace,
        surrogate_model: SurrogateModel,
        optimizer_of_interest: ConfigSpaceSearcher,
        optimizer_ensemble: list[ConfigSpaceSearcher],
    ) -> None:
        """Initialize an OptimizerBiasExplanationTask.

        Args:
            config_space: The configuration space.
            surrogate_model: The surrogate model.
            optimizer_of_interest: The optimizer of interest.
            optimizer_ensemble: The ensemble of optimizers.

        """
        super().__init__(config_space, surrogate_model)
        self.optimizer_of_interest = optimizer_of_interest
        self.optimizer_ensemble = optimizer_ensemble
