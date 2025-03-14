"""Custom MountainCar Env w/ a reward based on minimum distance from flag."""

import abc
import jax

import gymnasium.envs.classic_control
import gymnasium as gym

from dataclasses import dataclass, field

import keras
import numpy as np
import functools

import os
os.environ["JAX_TRACEBACK_FILTERING"] = "off"


class MountainCar(gym.envs.classic_control.MountainCarEnv):
    """Custom MountainCar Env w/ a reward based on minimum distance from flag.

    This environment modifies the standard MountainCar environment by:
        - Increasing the reward when the car reaches a new best position.
        - Setting a maximum episode length of 500 steps.
        - Providing a large reward (100) upon reaching the goal.
    """

    def __init__(self, *args, **kwargs):
        """Initializes the MountainCar environment.

        Args:
            *args: Positional arguments passed to the parent class constructor.
            **kwargs: Keyword arguments passed to the parent class constructor.
        """
        super().__init__(*args, **kwargs)
        self.best_p = -2
        self.current_step = 0

    def step(self, *args, **kwargs):
        """Performs a step in the environment.

        This method overrides the parent class's `step` method to provide
        custom reward logic and episode termination conditions.

        Args:
            *args: Positional arguments passed to the parent class's `step` method.
            **kwargs: Keyword arguments passed to the parent class's `step` method.

        Returns:
            tuple: A tuple containing:
                - observation: The next observation.
                - reward: The reward received.
                - terminated: Whether the episode has terminated.
                - truncated: Whether the episode has been truncated.
                - info: Additional information.
        """
        observation, reward, terminated, truncated, info = super().step(*args, **kwargs)
        self.current_step += 1
        if self.current_step > 500:
            truncated = True
        if terminated:
            reward = 100
        if terminated or truncated:
            return observation, reward, terminated, truncated, info
        p, v = observation
        if p > self.best_p:
            reward = 2
            self.best_p = p
        return observation, reward, terminated, truncated, info

    def reset(self, *args, **kwargs):
        """Resets the environment.

        This method overrides the parent class's `reset` method to reset the
        `best_p` attribute.

        Args:
            *args: Positional arguments passed to the parent class's `reset` method.
            **kwargs: Keyword arguments passed to the parent class's `reset` method.

        Returns:
            tuple: A tuple containing:
                - observation: The initial observation.
                - info: Additional information.
        """
        observation, info = super().reset(*args, **kwargs)
        self.best_p = observation[0]
        self.current_step = 0
        return observation, info


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


@dataclass
class TrainingMinibatch:
    states: np.typing.ArrayLike
    value_labels: np.typing.ArrayLike
    actions_and_advantages: np.typing.ArrayLike


@dataclass
class Logger:
    """A data class for logging episode information.

    Attributes:
        advantage_estimator: A BaseAdvantageEstimator object for estimating advantages.
        rewards (list[float]): List of rewards received at each step.
        action_probs (list[float]): List of action probabilities at each step.
        actions (list[float]): List of actions taken at each step.
        values (list[float]): List of value estimates at each step.
        rewards_to_go (list[float]): List of rewards-to-go (calculated at the end of an episode).
        advantages (list[float]): List of advantages (calculated at the end of an episode).
    """

    advantage_estimator: BaseAdvantageEstimator
    rewards: list[float] = field(default_factory=list)
    action_probs: list[float] = field(default_factory=list)
    states: list[np.typing.ArrayLike] = field(default_factory=list)
    actions: list[float] = field(default_factory=list)
    values: list[float] = field(default_factory=list)
    # rewards_to_go and advantages require a complete episode (which ends w/ a call to `log()` w/
    # either `done` or `truncated` being `True`). Once the episode is complete, these vectors will
    # be computed and `log()` cannot be called again on the object (an error will be thrown if it is).
    rewards_to_go: None | np.typing.ArrayLike = None
    advantages: None | np.typing.ArrayLike = None

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
        action_prob: float,
        value: float,
        done: bool = False,
        truncated: bool = False,
    ) -> None:
        """Logs information for a single step.

        Args:
            reward (float): The reward received.
            action (float): The action taken.
            action_prob (float): The probability of the action taken.
            value (float): The value estimate.
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
        self.action_probs.append(action_prob)
        self.values.append(value)
        if done or truncated:
            self.rewards_to_go = np.cumsum(list(reversed(self.rewards)))[::-1]
            self.advantages = self.advantage_estimator(
                np.array(self.rewards), np.array(self.values), truncated
            )

    def training_minibatch(self) -> TrainingMinibatch:
        assert (
            self.is_done
        ), "The episode must be complete before examples can be extracted."
        return TrainingMinibatch(
            states=np.vstack(self.states),
            value_labels=np.reshape(self.rewards_to_go, [-1, 1]),
            actions_and_advantages=np.hstack(
                [np.vstack(self.actions), np.reshape(self.advantages, [-1, 1])]
            ),
        )


@dataclass
class TrainingLoop:
    ac: keras.Model
    optimizer: keras.Optimizer
    value_only_epochs: int = 5
    value_and_policy_epochs: int = 2
    pg_loss_weight: float = 0.001
    jit_compile: bool = True

    def __post_init__(self):
        self.optimizer.build(self.ac.trainable_variables)
        self.train_state = (
            [v.value for v in self.ac.trainable_variables],
            [v.value for v in self.ac.non_trainable_variables],
            [v.value for v in self.optimizer.variables],
        )

        # jax.value_and_grad only differentiates against the first input.
        self.grad_fn = jax.value_and_grad(
            functools.partial(
                self.compute_loss_and_updates, pg_loss_weight=self.pg_loss_weight
            ),
            has_aux=True,
        )
        self.value_only_grad_fn = jax.value_and_grad(
            functools.partial(self.compute_loss_and_updates, pg_loss_weight=0.0),
            has_aux=True,
        )
        if self.jit_compile:
            self.train_step = jax.jit(
                functools.partial(self._train_step_helper, grad_fn=self.grad_fn)
            )
            self.value_only_train_step = jax.jit(
                functools.partial(
                    self._train_step_helper, grad_fn=self.value_only_grad_fn
                )
            )
        else:
            self.train_step = functools.partial(
                self._train_step_helper, grad_fn=self.grad_fn
            )
            self.value_only_train_step = functools.partial(
                self._train_step_helper, grad_fn=self.value_only_grad_fn
            )

    def compute_loss_and_updates(
        self,
        trainable_variables,
        non_trainable_variables,
        states,
        actions_and_advantages,
        value_labels,
        pg_loss_weight,
    ):
        outputs, non_trainable_variables = self.ac.stateless_call(
            trainable_variables, non_trainable_variables, states, training=True
        )
        values = outputs["value"]
        action_probs = outputs["action_prob"]
        loss_fn = make_total_loss(pg_loss_weight=pg_loss_weight)
        loss = loss_fn(actions_and_advantages, action_probs, value_labels, values)

        return loss, non_trainable_variables

    def _train_step_helper(self, state, data, grad_fn):
        trainable_variables, non_trainable_variables, optimizer_variables = state
        states, actions_and_advantages, value_labels = data

        (loss, non_trainable_variables), grads = grad_fn(
            trainable_variables,
            non_trainable_variables,
            states,
            actions_and_advantages,
            value_labels,
        )
        print(f"Grad min: {[jax.numpy.min(g) for g in grads]}")
        print(f"Grad max: {[jax.numpy.max(g) for g in grads]}")
        trainable_variables, optimizer_variables = optimizer.stateless_apply(
            optimizer_variables, grads, trainable_variables
        )
        # Return updated state
        return loss, (
            trainable_variables,
            non_trainable_variables,
            optimizer_variables,
        )

    def train(self, data: list[list[np.typing.ArrayLike]]):
        for _ in range(self.value_only_epochs):
            for batch in data:
                loss, self.train_state = self.value_only_train_step(
                    self.train_state, batch
                )
            print(f"Value-only loss: {loss}")
        for _ in range(self.value_and_policy_epochs):
            for batch in data:
                loss, self.train_state = self.train_step(self.train_state, batch)
            print(f"Value-and-policy loss: {loss}")

    def eval(self, states):
        trainable_variables, non_trainable_variables, _ = self.train_state
        outputs, _ = self.ac.stateless_call(
            trainable_variables, non_trainable_variables, states, training=False
        )
        return outputs


def make_actor_critic(
    state_dims: int = 2, num_actions: int = 3, layer_sizes: None | list[int] = None
) -> keras.Model:
    """Creates an actor-critic model.

    Args:
        state_dims (int, optional): The number of dimensions in the state space. Defaults to 2.
        num_actions (int, optional): The number of possible actions. Defaults to 3.
        layer_sizes (None | list[int], optional): A list of hidden layer sizes. Defaults to None.

    Returns:
        keras.Model: The actor-critic model.
    """
    # There are two observations from MountainCar
    inputs = keras.Input(shape=(state_dims,))
    x = inputs
    for ls in layer_sizes:
        x = keras.layers.Dense(ls, activation="swish")(x)
    action_scores = keras.layers.Dense(3)(x)
    softmax = keras.layers.Softmax(name="action_prob")(action_scores)
    value = keras.layers.Dense(1, name="value")(x)
    return keras.Model(inputs, {"action_prob": softmax, "value": value})


def simulate_episode(ac: TrainingLoop, env: gym.Env, render: bool = False) -> Logger:
    """Simulates a single episode.

    Args:
        ac (keras.Model): The actor-critic model wrapped in a TrainingLoop.
        env (gym.Env): The environment.
        render (bool, optional): Whether to render the environment. Defaults to False.

    Returns:
        Logger: The logger containing the episode data.
    """
    env.render_mode = "human" if render else None
    obs, _ = env.reset()
    logger = Logger(advantage_estimator=GeneralizedAdvantageEstimator())
    done = False
    while not done:
        output = ac.eval(np.reshape(obs, [1, -1]))
        action_probs = np.asarray(output["action_prob"][0]).copy()
        value = output["value"][0][0]
        # Jax has some numerical jitter making the probs negative or not sum to 1.
        action_probs = np.clip(action_probs, 0.0, 1.0)
        action_probs[-1] = 1 - sum(action_probs[:-1])
        action = np.random.choice(range(3), p=action_probs)
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        logger.log(
            reward=reward,
            state=obs,
            action=action,
            action_prob=action_probs[action],
            value=value,
            done=terminated,
            truncated=truncated,
        )
    return logger


def make_pg_loss(loss_weight: float = 0.1):
    def _pg_loss(actions_and_advantages, action_probs):
        actions = actions_and_advantages[:, 0]
        advantages = keras.ops.reshape(actions_and_advantages[:, 1], [-1, 1])
        logps = keras.ops.log(action_probs[keras.ops.cast(actions, dtype="int32")])
        return -loss_weight * keras.ops.sum(logps * advantages)

    return _pg_loss


def make_total_loss(pg_loss_weight: float = 0.1):
    pg_loss = make_pg_loss(loss_weight=pg_loss_weight)

    def _total_loss(actions_and_advantages, action_probs, value_labels, values):
        return pg_loss(
            actions_and_advantages, action_probs
        ) + keras.losses.MeanSquaredError()(value_labels, values)

    return _total_loss


env = MountainCar(render_mode="human")
env.metadata["render_fps"] = 60 * 10
env = MountainCar()

ac = make_actor_critic(layer_sizes=[256, 256, 256])
optimizer = keras.optimizers.RMSprop(learning_rate=0.001)

training_loop = TrainingLoop(
    ac=ac,
    optimizer=optimizer,
    value_only_epochs=100,
    value_and_policy_epochs=2,
    pg_loss_weight=1e-4,
    jit_compile=True,
)

inner_iters = 5
for iter in range(50):
    batches = []
    for sim_iter in range(inner_iters):
        logger = simulate_episode(
            training_loop, env, render=iter % 10 == 0 and sim_iter == 0
        )
        print(f"Total reward: {sum(logger.rewards)}")
        minibatch = logger.training_minibatch()
        batches.append(
            [minibatch.states, minibatch.actions_and_advantages, minibatch.value_labels]
        )
    training_loop.train(batches)
