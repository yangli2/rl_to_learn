"""Custom MountainCar Env w/ a reward based on minimum distance from flag."""

import argparse
from dataclasses import dataclass, field
import pickle
import time

import gymnasium.envs.classic_control
import gymnasium as gym
import jax
import scipy

import keras
import numpy as np
import scipy.special

from gae import GeneralizedAdvantageEstimator
from logger import Logger
from training_loop import ValueModel, PolicyModel

from matplotlib import pyplot as plt


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
            reward = 1
            self.best_p = p
        else:
            reward = -0.1
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


def make_policy_network(
    state_dims: int = 2, num_actions: int = 3, layer_sizes: None | list[int] = None
) -> keras.Model:
    """Creates an actor-critic model.

    Args:
        state_dims (int, optional): The number of dimensions in the state space. Defaults to 2.
        num_actions (int, optional): The number of possible actions. Defaults to 3.
        layer_sizes (None | list[int], optional): A list of hidden layer sizes. Defaults to None.

    Returns:
        keras.Model: The model mapping state to action probabilities.
    """
    # There are two observations from MountainCar
    inputs = keras.Input(shape=(state_dims,))
    x = inputs
    for ls in layer_sizes:
        x = keras.layers.Dense(ls, activation="swish")(x)
    logits = keras.layers.Dense(num_actions)(x)
    return keras.Model(inputs, logits)


def make_value_network(
    state_dims: int = 2, layer_sizes: None | list[int] = None
) -> keras.Model:
    """Creates an actor-critic model.

    Args:
        state_dims (int, optional): The number of dimensions in the state space. Defaults to 2.
        layer_sizes (None | list[int], optional): A list of hidden layer sizes. Defaults to None.

    Returns:
        keras.Model: The model mapping state to its predicted value.
    """
    # There are two observations from MountainCar
    inputs = keras.Input(shape=(state_dims,))
    x = inputs
    for ls in layer_sizes:
        x = keras.layers.Dense(ls, activation="swish")(x)
    value = keras.layers.Dense(1, name="value")(x)
    return keras.Model(inputs, value)


def simulate_episode(
    m_policy: PolicyModel,
    env: gym.Env,
    m_value: ValueModel | None = None,
    render: bool = False,
) -> Logger:
    """Simulates a single episode.

    Args:
        m_policy: The policy model.
        env (gym.Env): The environment.
        m_value: The (optional) value model. If present, will use advantage as the weight; otherwise, will use reward or reward-to-go.
        render (bool, optional): Whether to render the environment. Defaults to False.

    Returns:
        Logger: The logger containing the episode data.
    """
    env.render_mode = "human" if render else None
    obs, _ = env.reset()
    ae = None if m_value is None else GeneralizedAdvantageEstimator()
    logger = Logger(advantage_estimator=ae)
    done = False
    while not done:
        action_logits = m_policy.eval(np.reshape(obs, [1, -1]))
        num_actions = action_logits.shape[1]
        try:
            action_probs = scipy.special.softmax(action_logits[0])
            action = np.random.choice(range(num_actions), p=action_probs)
        except:
            print(action_probs)
            raise
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        value = (
            m_value.eval(np.reshape(obs, [1, -1]))[0][0]
            if m_value is not None
            else None
        )
        logger.log(
            reward=reward,
            state=obs,
            action=action,
            action_logits=action_logits[0],
            value=value,
            done=terminated,
            truncated=truncated,
        )
    return logger


@dataclass
class ReplayBuffer:
    num_batches: int = 128
    batches_buffer: list[list[np.typing.ArrayLike]] = field(default_factory=list)

    def maybe_add(
        self, batch: list[np.typing.ArrayLike], replacement_prob: float = 0.1
    ) -> None:
        if len(self.batches_buffer) < self.num_batches:
            self.batches_buffer.append(batch)
            return
        # We have a full buffer. Replace an existing batch w/ some probability
        if np.random.random() <= replacement_prob:
            self.batches_buffer[np.random.randint(self.num_batches)] = batch


@dataclass
class Schedule:
    key_points: list[int]
    key_values: list[float]
    interpolation_methods: list[str] = field(default_factory=list)

    def __post_init__(self):
        assert self.key_points == sorted(self.key_points)
        assert self.key_points[0] == 0
        if not self.interpolation_methods:
            self.interpolation_methods = ["step"] * (len(self.key_points) - 1)

    def value(self, x: int) -> float:
        stage = sum(x > kp for kp in self.key_points)
        if stage >= len(self.key_points):
            return self.key_values[-1]
        if stage == 0:
            return self.key_values[0]
        # stage is somewhere in between
        interpolation_method = self.interpolation_methods[stage - 1]
        if interpolation_method == "step":
            return self.key_values[stage - 1]
        elif interpolation_method == "linear":
            p = float(x - self.key_points[stage - 1]) / float(
                self.key_points[stage] - self.key_points[stage - 1]
            )
            return p * self.key_values[stage] + (1 - p) * self.key_values[stage - 1]
        else:
            raise ValueError(f"Unknown interpolation method {interpolation_method}")


@dataclass
class EpisodeMetrics:
    total_reward: float = 0.0
    action_prob_entropy_mean: float = 0.0
    action_prob_entropy_std: float = 0.0


def simulate_and_log(
    m_policy, env, policy_batches, m_value=None, value_batches=None, render=False
) -> EpisodeMetrics:
    metrics = EpisodeMetrics()
    if value_batches is not None:
        assert m_value is not None
    logger = simulate_episode(m_policy, env, m_value=m_value, render=render)
    action_probs = [scipy.special.softmax(logit) for logit in logger.action_logits]
    action_prob_entropy = [np.sum(p * np.log(p)) for p in action_probs]
    metrics = EpisodeMetrics(
        total_reward=logger.rewards_to_go[0],
        action_prob_entropy_mean=np.mean(action_prob_entropy),
        action_prob_entropy_std=np.std(action_prob_entropy),
    )

    minibatch = logger.training_minibatch()
    policy_batches.append([minibatch.states, minibatch.actions_and_weights])
    if value_batches is not None:
        value_batches.append([minibatch.states, minibatch.rewards_to_go])

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_path_prefix", type=str, default="result")
    result_path_prefix = parser.parse_args().result_path_prefix
    timestamp = str(int(time.time() * 1000))
    result_path = f"{result_path_prefix}_{timestamp}.pkl"

    env = MountainCar(render_mode="human")
    env.metadata["render_fps"] = 60 * 100
    env = MountainCar()

    value_net = make_value_network(layer_sizes=[256, 256, 256])
    value_optimizer = keras.optimizers.Adam(learning_rate=0.001)
    policy_net = make_policy_network(layer_sizes=[32, 32, 32])
    policy_optimizer = keras.optimizers.Adadelta(learning_rate=0.01, clipnorm=10)
    # optimizer = keras.optimizers.RMSprop(learning_rate=0.001)

    # m_value = ValueModel(value_net, optimizer=value_optimizer)
    m_value = None
    m_policy = PolicyModel(policy_net, optimizer=policy_optimizer)

    inner_iters = 10
    policy_epochs_schedule = Schedule(key_points=[0, 0], key_values=[0, 5])
    mean_total_rewards = []
    mean_action_prob_entropy_means = []
    mean_action_prob_entropy_stds = []
    plt.ion()
    plt.show(block=False)
    plt.pause(0.1)

    for iter in range(500):
        # value_batches = []
        policy_batches = []
        episode_metrics = []
        for sim_iter in range(inner_iters):
            episode_metrics.append(
                simulate_and_log(
                    m_policy,
                    env,
                    policy_batches,
                    m_value=m_value,
                    value_batches=None,
                    render=iter % 10 == 0 and sim_iter == 0,
                )
            )

        # m_value.train(value_batches, num_epochs=1)
        m_policy.train(policy_batches, num_epochs=1)
        mean_total_rewards.append(np.mean([em.total_reward for em in episode_metrics]))
        mean_action_prob_entropy_means.append(
            np.mean([em.action_prob_entropy_mean for em in episode_metrics])
        )
        mean_action_prob_entropy_stds.append(
            np.mean([em.action_prob_entropy_std for em in episode_metrics])
        )
        plt.subplot(3, 1, 1)
        plt.plot(mean_total_rewards)
        plt.subplot(3, 1, 2)
        plt.plot(mean_action_prob_entropy_means)
        plt.subplot(3, 1, 3)
        plt.plot(mean_action_prob_entropy_stds)
        plt.pause(0.1)

    with open(result_path, "wb") as f:
        pickle.dump(
            [
                mean_total_rewards,
                mean_action_prob_entropy_means,
                mean_action_prob_entropy_stds,
            ],
            f,
        )


if __name__ == "__main__":
    main()
