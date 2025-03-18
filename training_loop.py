import abc
from dataclasses import dataclass
import functools

import numpy as np
import keras
import jax


@dataclass
class JaxModel(abc.ABC):
    model: keras.Model
    optimizer: keras.Optimizer
    jit_compile: bool = True

    def __post_init__(self):
        self.optimizer.build(self.model.trainable_variables)
        self.train_state = (
            [v.value for v in self.model.trainable_variables],
            [v.value for v in self.model.non_trainable_variables],
            [v.value for v in self.optimizer.variables],
        )

        # jax.value_and_grad only differentiates against the first input.
        self.grad_fn = jax.value_and_grad(
            self.compute_loss_and_updates,
            has_aux=True,
        )
        self.train_step = functools.partial(
            self._train_step_helper, grad_fn=self.grad_fn
        )
        if self.jit_compile:
            self.train_step = jax.jit(self.train_step)

    @abc.abstractmethod
    def loss_fn(self, outputs: jax.Array, labels: jax.Array) -> jax.Array:
        pass

    def compute_loss_and_updates(
        self,
        trainable_variables,
        non_trainable_variables,
        inputs,
        labels,
    ):
        num_model_non_trainables = len(self.model.non_trainable_variables)
        output, non_trainable_variables = self.model.stateless_call(
            trainable_variables,
            non_trainable_variables[:num_model_non_trainables],
            inputs,
            training=True
        )
        loss, loss_non_trainable_variables = self.loss_fn(output, labels)
        return loss, non_trainable_variables + loss_non_trainable_variables

    def _train_step_helper(self, state, data, grad_fn):
        trainable_variables, non_trainable_variables, optimizer_variables = state
        inputs, labels = data

        (loss, non_trainable_variables), grads = grad_fn(
            trainable_variables,
            non_trainable_variables,
            inputs,
            labels
        )
        trainable_variables, optimizer_variables = self.optimizer.stateless_apply(
            optimizer_variables, grads, trainable_variables
        )
        # Return updated state
        return loss, (
            trainable_variables,
            non_trainable_variables,
            optimizer_variables,
        )

    def train(self, data: list[list[np.typing.ArrayLike]], num_epochs: int):
        for _ in range(num_epochs):
            for batch in data:
                loss, self.train_state = self.train_step(
                    self.train_state, batch
                )
            print(f"Loss: {loss}")
            all_non_trainables = self.train_state[1]
            num_model_non_trainables = len(self.model.non_trainable_variables)
            if num_model_non_trainables == 0:
                loss_non_trainables = all_non_trainables
            else:
                loss_non_trainables = all_non_trainables[num_model_non_trainables-1:]
            if loss_non_trainables:
                print(f"Loss stats: {loss_non_trainables}")

    def eval(self, states):
        trainable_variables, non_trainable_variables, _ = self.train_state
        num_model_non_trainables = len(self.model.non_trainable_variables)
        outputs, _ = self.model.stateless_call(
            trainable_variables,
            non_trainable_variables[:num_model_non_trainables],
            states,
            training=False
        )
        return outputs


class PolicyModel(JaxModel):
    def loss_fn(self, outputs: jax.Array, labels: jax.Array) -> jax.Array:
        actions_and_advantages = labels
        logits = outputs
        actions = actions_and_advantages[:, 0]
        advantages = keras.ops.reshape(actions_and_advantages[:, 1], [-1, 1])
        advantage_stats = keras.ops.stack([keras.ops.mean(advantages), keras.ops.std(
            advantages), keras.ops.max(advantages), keras.ops.min(advantages)])
        logps = keras.ops.log_softmax(logits)[:, keras.ops.cast(actions, dtype="int32")]
        actions_stats = keras.ops.stack([keras.ops.sum(
            actions == 0), keras.ops.sum(actions == 1), keras.ops.sum(actions == 2)])
        return -keras.ops.mean(logps * advantages), [advantage_stats, actions_stats]


class ValueModel(JaxModel):
    def loss_fn(self, outputs: jax.Array, labels: jax.Array) -> jax.Array:
        return keras.losses.MeanSquaredError()(labels, outputs), []
