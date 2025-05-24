import functools
import pickle
import random

import jax
import jax.numpy as jnp
from flax import serialization
import yaml

from mountaincar_shaped_reward_env import MountainCarShapedReward
from eval import run_episode

from rejax import get_algo


def main(algo_str, config, save_path, seed_id, num_trials):
    if seed_id == -1:
        seed_id = int.from_bytes(random.randbytes(8))
    keys = jax.random.split(jax.random.PRNGKey(seed_id), num_trials)
    algo_cls = get_algo(algo_str)
    env = MountainCarShapedReward()
    config["env"] = env
    algo = algo_cls.create(**config)
    print(algo.config)
    # It does not matter what random we use here, since it'll get overwritten right below.
    ts = algo.init_state(keys[0])

    with open(save_path, "rb") as f:
        ts = serialization.from_state_dict(ts, pickle.load(f))

    @jax.jit
    @functools.partial(jax.vmap, in_axes=[0, None, None])
    def _act(_ts, _rng, _obs):
        return algo.make_act(_ts)(_obs, _rng)

    def act(_obs):
        rng = jax.random.PRNGKey(
            int.from_bytes(random.randbytes(8), signed=True, byteorder="little")
        )
        return int(_act(ts, rng, _obs)[0])

    env_params = env.default_params
    for key in keys:
        run_episode(env, env_params, act, key, render=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="mountaincar_shaped_reward.yaml",
        help="Path to configuration file.",
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        default="ppo",
    )
    parser.add_argument(
        "--seed_id",
        type=int,
        default=-1,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--num_trials",
        type=int,
        default=1,
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
