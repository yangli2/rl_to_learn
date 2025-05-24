from typing import Union, Any

import jax
import jax.numpy as jnp

import chex
from flax import struct
from gymnax.environments.classic_control.mountain_car import (
    MountainCar,
    EnvParams,
    EnvState,
)


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
