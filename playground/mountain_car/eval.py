import jax
import jax.numpy as jnp
from gymnax.visualize import Visualizer

def render_episode(env, env_params, state_seq, reward_seq):
    # Create a memory filesystem
    cum_rewards = jnp.cumsum(jnp.array(reward_seq))
    vis = Visualizer(env, env_params, state_seq, cum_rewards)
    vis.animate()

def run_episode(env, env_params, step_fn, rng, render=False):
    state_seq, reward_seq = [], []
    rng, rng_reset = jax.random.split(rng)
    obs, env_state = env.reset(rng_reset, env_params)
    while True:
        state_seq.append(env_state)
        rng, rng_step = jax.random.split(rng)
        action = step_fn(obs)
        next_obs, next_env_state, reward, done, _ = env.step(
            rng_step, env_state, action, env_params
        )
        reward_seq.append(reward)
        if done:
            break
        obs = next_obs
        env_state = next_env_state

    if render:
        render_episode(env, env_params, state_seq, reward_seq)

    return state_seq, reward_seq

