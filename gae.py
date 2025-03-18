from base_advantage_estimator import BaseAdvantageEstimator

import numpy as np


class GeneralizedAdvantageEstimator(BaseAdvantageEstimator):
    """
    Generalized Advantage Estimator (GAE) class.

    This class implements the Generalized Advantage Estimation algorithm,
    which is used to estimate the advantages of actions in reinforcement
    learning.

    Attributes:
        gamma (float): The discount factor (gamma) for future rewards.
        lambd (float): The advantage estimator exponential moving average decay rate.
    """

    def __init__(self, gamma: float = 0.99, lambd: float = 0.95):
        """
        Initializes the Generalized Advantage Estimator.

        Args:
            gamma (float, optional): The discount factor (gamma). Defaults to 0.99.
            lambd (float, optional): The GAE lambda parameter. Defaults to 0.95.

        Raises:
            ValueError: If gamma or lambd are not within the valid range [0, 1].
        """
        if not 0 <= gamma <= 1:
            raise ValueError("Gamma must be between 0 and 1 inclusive.")
        if not 0 <= lambd <= 1:
            raise ValueError("Lambda must be between 0 and 1 inclusive.")

        self.gamma = gamma
        self.lambd = lambd

    def __call__(
        self,
        rewards: np.typing.ArrayLike,
        values: np.typing.ArrayLike,
        truncated: bool,
        termination_value: float = 100,
    ) -> np.typing.ArrayLike:
        """
        Calculates the advantages using the GAE algorithm.

        Args:
            rewards (np.ndarray): A 1D array of rewards received at each timestep.
            values (np.ndarray): A 1D array of value estimates at each timestep.

        Returns:
            np.ndarray: A 1D array of estimated advantages at each timestep.

        Raises:
          ValueError: if rewards and values are not numpy arrays.
          ValueError: if rewards and values are not 1 dimensional.
          ValueError: if rewards and values do not have the same length.
        """
        if not isinstance(rewards, np.ndarray) or not isinstance(values, np.ndarray):
            raise ValueError("Rewards and values must be numpy arrays.")
        if rewards.ndim != 1 or values.ndim != 1:
            raise ValueError(
                f"Rewards and values must be 1 dimensional, but rewards: {rewards}, values: {values}"
            )
        if len(rewards) != len(values):
            raise ValueError(
                "Rewards and values must have the same length, but rewards: {rewards}, values: {values}"
            )

        advantages = np.zeros_like(rewards)

        # Calculate TD errors (deltas): difference between actual reward and the difference
        # between discounted future value and current value.
        deltas = rewards[:-1] + self.gamma * values[1:] - values[:-1]

        # Special handling for final delta - if it is a truncation, simply do not have
        # it contribute to the advantage calculations; else, use the provided termination
        # value as the expected value from us achieving a real termination condition for
        # the episode.
        last_delta = 0 if truncated else termination_value - values[-1]
        deltas = np.concatenate([deltas, np.array([last_delta])])

        # Calculate GAE recursively
        gae_advantage = 0
        for i in reversed(range(len(advantages))):
            gae_advantage = deltas[i] + self.gamma * self.lambd * gae_advantage
            advantages[i] = gae_advantage
        return advantages
