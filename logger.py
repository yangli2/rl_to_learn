"""A Logger class for RL episodes."""

from dataclasses import dataclass, field
import numpy as np

from base_advantage_estimator import BaseAdvantageEstimator


@dataclass
class TrainingMinibatch:
    states: np.typing.ArrayLike
    rewards_to_go: np.typing.ArrayLike
    actions_and_weights: np.typing.ArrayLike


@dataclass
class Logger:
    """A data class for logging episode information.

    Attributes:
        rewards (list[float]): List of rewards received at each step.
        actions (list[float]): List of actions taken at each step.
        action_logits (list[array]): List of action logits from which the action was sampled
          at each step.
        values (list[float]): List of value estimates at each step.
        advantage_estimator: A BaseAdvantageEstimator object for estimating advantages.
        rewards_to_go (list[float]): List of rewards-to-go (calculated at the end of an episode).
        action_weights (list[float]): List of values corresponding to actions at each step that
          encodes whether the action should be encouraged or discouraged (calculated at the end
          of an episode). If advantage_estimator is present, this will be the estimated advantages.
          Otherwise, this will just be the rewards-to-go.
    """

    rewards: list[float] = field(default_factory=list)
    states: list[np.typing.ArrayLike] = field(default_factory=list)
    actions: list[float] = field(default_factory=list)
    action_logits: list[np.typing.ArrayLike] = field(default_factory=list)
    values: list[float] = field(default_factory=list)
    # If pre
    advantage_estimator: BaseAdvantageEstimator | None = None
    # rewards_to_go and action_weights require a complete episode (which ends w/ a call to `log()` w/
    # either `done` or `truncated` being `True`). Once the episode is complete, these vectors will
    # be computed and `log()` cannot be called again on the object (an error will be thrown if it is).
    rewards_to_go: None | np.typing.ArrayLike = None
    action_weights: None | np.typing.ArrayLike = None

    @property
    def is_done(self) -> bool:
        """Checks if the episode is done.

        Returns:
            bool: True if the episode is done, False otherwise.
        """
        return self.rewards_to_go is not None

    def log(
        self,
        reward: float,
        state: np.typing.ArrayLike,
        action: float,
        action_logits: np.typing.ArrayLike,
        value: float | None = None,
        done: bool = False,
        truncated: bool = False,
    ) -> None:
        """Logs information for a single step.

        Args:
            reward (float): The reward received.
            state (array): The state of the environment.
            action (float): The action taken.
            action_logits (array): the logits that determined the probabilities that the action was sampled from.
            value (float): The optional value estimate.
            done (bool, optional): Whether the episode has terminated. Defaults to False.
            truncated (bool, optional): Whether the episode has been truncated. Defaults to False.

        Raises:
            RuntimeError: If `log()` is called after the episode is finished.
        """
        if self.is_done:
            raise RuntimeError(
                "`log()` cannot be called on a `Logger` object if it had "
                "already been called previously w/ `done` or `truncated` being"
                " True. Each new episode should be logged by a new Logger "
                "object."
            )
        self.rewards.append(reward)
        self.states.append(state)
        self.actions.append(action)
        self.action_logits.append(action_logits)
        if value is not None:
            self.values.append(value)
        else:
            assert (
                self.advantage_estimator is None
            ), "If we are not logging value, then we cannot estimate advantage."
        if done or truncated:
            self.rewards_to_go = np.cumsum(list(reversed(self.rewards)))[::-1]
            if self.advantage_estimator is not None:
                self.action_weights = self.advantage_estimator(
                    np.array(self.rewards), np.array(self.values), truncated
                )
            else:
                self.action_weights = self.rewards_to_go

    def training_minibatch(self) -> TrainingMinibatch:
        assert (
            self.is_done
        ), "The episode must be complete before examples can be extracted."
        return TrainingMinibatch(
            states=np.vstack(self.states),
            rewards_to_go=np.reshape(self.rewards_to_go, [-1, 1]),
            actions_and_weights=np.hstack(
                [np.vstack(self.actions), np.reshape(self.action_weights, [-1, 1])]
            ),
        )
