import datetime
import timeit
import functools
import pickle
import random

import jax
import jax.numpy as jnp
import numpy as np
from flax import serialization
import yaml
from matplotlib import pyplot as plt

from rejax import get_algo
from mountaincar_shaped_reward_env import MountainCarShapedReward
from eval import run_episode


def main(algo_str, config, seed_id, num_seeds, time_fit, save_path_prefix):
    algo_cls = get_algo(algo_str)
    env = MountainCarShapedReward()
    config["env"] = env
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
        suffix = f"{str(int(datetime.datetime.now().timestamp()))}.pkl"
        with open(f"{save_path_prefix}_{suffix}", "wb") as f:
            pickle.dump(serialization.to_state_dict(ts), f)

    # Move local variables to global scope for debugging (run with -i)
    globals().update(locals())

    @jax.jit
    @functools.partial(jax.vmap, in_axes=[0, None, None])
    def _act(_ts, _rng, _obs):
        return algo.make_act(_ts)(_obs, _rng)

    def act(_obs):
        rng = jax.random.PRNGKey(
            int.from_bytes(random.randbytes(8), signed=True, byteorder="little")
        )
        return int(_act(ts, rng, _obs)[0])

    for _ in range(10):
        rng = jax.random.PRNGKey(
            int.from_bytes(random.randbytes(8), signed=True, byteorder="little")
        )
        run_episode(env, env.default_params, act, rng, render=True)


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
