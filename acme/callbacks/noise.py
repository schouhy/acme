"""Noise layers (for exploration)."""

import sonnet as snt
import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions

from acme.callbacks.base import AgentCallback


class ClippedGaussianCallback(AgentCallback):
    """adding clipped Gaussian noise to each output."""
    def __init__(self, stddev: float):
        super().__init__()
        self._noise = tfd.Normal(loc=0., scale=stddev)

    def after_select_action(self):
        self.owner._action +=  self._noise.sample(self.owner._action.shape)
        #action[0] = tf.clip_by_value(action[0], -1.0, 1.0)