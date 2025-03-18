import abc
import numpy as np


class BaseAdvantageEstimator(abc.ABC):
    """Abstract base class for advantage estimators."""

    @abc.abstractmethod
    def __call__(
        self,
        rewards: np.typing.ArrayLike,
        values: np.typing.ArrayLike,
        truncated: bool,
        termination_value: float = 100,
    ) -> np.typing.ArrayLike:
        """Estimate advantages based on rewards and values."""
        pass
