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


def main(algo_str, config, seed_id, num_seeds, time_fit, save_path_prefix):
    algo_cls = get_algo(algo_str)
    if config["env"] == "MountainCar-ShapedReward-v0":
        # My own environment, hijack and create my own python obj
        config["env"] = MountainCarShapedReward()
    algo = algo_cls.create(**config)
    print(algo.config)

    old_eval_callback = algo.eval_callback

    def eval_callback(algo, ts, rng):
        lengths, returns = old_eval_callback(algo, ts, rng)
        jax.debug.print(
            "Step {}, Mean episode length: {}, Mean return: {}",
            ts.global_step,
            lengths.mean(),
            returns.mean(),
        )
        return lengths, returns

    algo = algo.replace(eval_callback=eval_callback)

    # Train it
    key = jax.random.PRNGKey(seed_id)
    keys = jax.random.split(key, num_seeds)

    vmap_train = jax.jit(jax.vmap(algo_cls.train, in_axes=(None, 0)))
    ts, (_, returns) = vmap_train(algo, keys)
    returns.block_until_ready()

    print(f"Achieved mean return of {returns.mean(axis=-1)[:, -1]}")

    t = jnp.arange(returns.shape[1]) * algo.eval_freq
    colors = plt.cm.cool(jnp.linspace(0, 1, num_seeds))
    for i in range(num_seeds):
        plt.plot(t, returns.mean(axis=-1)[i], c=colors[i])
    plt.show()

    if time_fit:
        print("Fitting 3 times, getting a mean time of... ", end="", flush=True)

        def time_fn():
            return vmap_train(algo, keys)

        time = timeit.timeit(time_fn, number=3) / 3
        print(
            f"{time:.1f} seconds total, equalling to "
            f"{time / num_seeds:.1f} seconds per seed"
        )

    if save_path_prefix:
        suffix = str(int(datetime.datetime.now().timestamp()))
        with open(f"{save_path_prefix}_{suffix}", "wb") as f:
            pickle.dump(serialization.to_state_dict(ts), f)

    # Move local variables to global scope for debugging (run with -i)
    globals().update(locals())

    gym_env = gym.make("MountainCar-v0", 200, render_mode="human")
    gym_env.metadata["render_fps"] = 1000

    @jax.jit
    @functools.partial(jax.vmap, in_axes=[0, None, None])
    def _act(_ts, _rng, _obs):
        return algo.make_act(_ts)(_obs, _rng)

    def act(_obs):
        return int(_act(ts, key, jnp.array(_obs))[0])

    for _ in range(10):
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
        "--time-fit",
        action="store_true",
        help="Time how long it takes to fit the agent by fitting 3 times.",
    )
    parser.add_argument(
        "--seed_id",
        type=int,
        default=0,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--num-seeds",
        type=int,
        default=1,
        help="Number of seeds to roll out.",
    )
    parser.add_argument(
        "--save_path_prefix",
        type=str,
        default="",
        help="If present, path to save the trained model weights.",
    )

    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f.read())[args.algorithm]

    main(
        args.algorithm,
        config,
        args.seed_id,
        args.num_seeds,
        args.time_fit,
        args.save_path_prefix,
    )
