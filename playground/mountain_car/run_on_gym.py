from typing import Any, Union
import datetime
import timeit
import functools
import pickle

import chex
import gymnax
import gymnax.visualize
import gymnasium as gym
import gymnasium.envs.classic_control.mountain_car
import jax
import jax.numpy as jnp
import numpy as np
from flax import serialization, struct
import yaml
from matplotlib import pyplot as plt

from gymnax.environments.classic_control.mountain_car import (
    MountainCar,
    EnvParams,
    EnvState,
)

from rejax import get_algo


@struct.dataclass
class MountainCarShapedRewardState(EnvState):
    position: jnp.ndarray
    velocity: jnp.ndarray
    time: int
    best_position: jnp.ndarray


class MountainCarShapedReward(MountainCar):
    def step_env(
        self,
        key: chex.PRNGKey,
        state: MountainCarShapedRewardState,
        action: Union[int, float, chex.Array],
        params: EnvParams,
    ) -> tuple[chex.Array, EnvState, jnp.ndarray, jnp.ndarray, dict[Any, Any]]:
        """Perform single timestep state transition."""
        obs, parent_state, reward, done, info = super().step_env(
            key, state, action, params
        )
        success = (parent_state.position >= params.goal_position) * (
            parent_state.velocity >= params.goal_velocity
        )
        reward = jnp.where(
            success,
            jnp.array(100.0),
            jnp.where(
                parent_state.position > state.best_position,
                jnp.array(1.0),
                0.1 * reward,
            ),
        )
        new_state = MountainCarShapedRewardState(
            time=parent_state.time,
            position=parent_state.position,
            velocity=parent_state.velocity,
            best_position=jnp.maximum(state.best_position, parent_state.position),
        )
        return obs, new_state, reward, done, info

    def reset_env(
        self, key: chex.PRNGKey, params: EnvParams
    ) -> tuple[chex.Array, MountainCarShapedRewardState]:
        """Reset environment state by sampling initial position."""
        init_state = jax.random.uniform(key, shape=(), minval=-0.6, maxval=-0.4)
        state = MountainCarShapedRewardState(
            position=init_state,
            velocity=jnp.array(0.0),
            time=0,
            best_position=init_state,
        )
        return self.get_obs(state), state


def main(algo_str, config, save_path, seed_id, num_trials):
    key = jax.random.PRNGKey(seed_id)
    algo_cls = get_algo(algo_str)
    if config["env"] == "MountainCar-ShapedReward-v0":
        # My own environment
        config["env"] = MountainCarShapedReward()
    algo = algo_cls.create(**config)
    print(algo.config)
    ts = algo.init_state(key)

    with open(save_path, "rb") as f:
        ts = serialization.from_state_dict(ts, pickle.load(f))

    gym_env = gym.make("MountainCar-v0", 200, render_mode="human")
    gym_env.metadata["render_fps"] = 1000

    @jax.jit
    @functools.partial(jax.vmap, in_axes=[0, None, None])
    def _act(_ts, _rng, _obs):
        return algo.make_act(_ts)(_obs, _rng)

    def act(_obs):
        return int(_act(ts, key, jnp.array(_obs))[0])

    for _ in range(num_trials):
        done = False
        obs, _ = gym_env.reset()
        while not done:
            obs, _, terminated, truncated, _ = gym_env.step(act(obs))
            done = terminated or truncated


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="configs/cartpole.yaml",
        help="Path to configuration file.",
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--seed_id",
        type=int,
        default=0,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--num_trials",
        type=int,
        default=10,
        help="How many episodes to simulate.",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="",
        help="Path to save the trained model weights.",
    )

    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f.read())[args.algorithm]

    assert args.save_path

    main(
        args.algorithm,
        config,
        args.save_path,
        args.seed_id,
        args.num_trials,
    )
