"""Rescaling layers (e.g. to match action specs)."""

from typing import Union
from acme import specs
import sonnet as snt
import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions
tfb = tfp.bijectors

from acme.callbacks.base import AgentCallback


class ClipToSpec(AgentCallback):
  """Clipping inputs to within a BoundedArraySpec."""
  def __init__(self, spec: specs.BoundedArray):
    super().__init__()
    self._min = spec.minimum
    self._max = spec.maximum

  def after_select_action(self):
      self.owner._action = tf.clip_by_value(self.owner._action, self._min, self._max)


# class RescaleToSpec(snt.Module):
#   """Sonnet module rescaling inputs in [-1, 1] to match a BoundedArraySpec."""
#
#   def __init__(self, spec: specs.BoundedArray, name: str = 'rescale_to_spec'):
#     super().__init__(name=name)
#     self._scale = spec.maximum - spec.minimum
#     self._offset = spec.minimum
#
#   def __call__(self, inputs: tf.Tensor) -> tf.Tensor:
#     inputs = 0.5 * (inputs + 1.0)  # [0, 1]
#     output = inputs * self._scale + self._offset  # [minimum, maximum]
#
#     return output
#
#
# class TanhToSpec(snt.Module):
#   """Sonnet module squashing real-valued inputs to match a BoundedArraySpec."""
#
#   def __init__(self, spec: specs.BoundedArray, name: str = 'tanh_to_spec'):
#     super().__init__(name=name)
#     self._scale = spec.maximum - spec.minimum
#     self._offset = spec.minimum
#
#   def __call__(
#       self, inputs: Union[tf.Tensor, tfd.Distribution]
#       ) -> Union[tf.Tensor, tfd.Distribution]:
#     if isinstance(inputs, tfd.Distribution):
#       inputs = tfb.Tanh()(inputs)
#       inputs = tfb.ScaleMatvecDiag(0.5 * self._scale)(inputs)
#       output = tfb.Shift(self._offset + 0.5 * self._scale)(inputs)
#     else:
#       inputs = tf.tanh(inputs)  # [-1, 1]
#       inputs = 0.5 * (inputs + 1.0)  # [0, 1]
#       output = inputs * self._scale + self._offset  # [minimum, maximum]
#     return output
