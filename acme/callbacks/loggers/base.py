import abc
from typing import Any, Mapping
from acme.callbacks import base
from acme.utils import counting
import tree
import time
from acme import types

# Internal imports.
import dm_env


LoggingData = Mapping[str, Any]


class BaseLoggerCallback(base.BaseCallback, abc.ABC):
    def __init__(self):
        self._counter = counting.Counter()
        super(BaseLoggerCallback, self).__init__()

    """A logger has a `write` method."""
    def on_episode_begin(self, timestep: dm_env.TimeStep):
        # Reset any counts and start the environment.
        self._start_time = time.time()
        self._episode_steps = 0
        self._episode_return = 0

    def on_feedback(self, action: types.NestedArray, next_timestep: dm_env.TimeStep):
        # Book-keeping.
        self._episode_steps += 1
        self._episode_return += next_timestep.reward

    def on_episode_end(self):
        # Record counts.
        counts = self._counter.increment(episodes=1, steps=self._episode_steps)

        # Collect the results and combine with counts.
        steps_per_second = self._episode_steps / (time.time() - self._start_time)
        result = {
            'episode_length': self._episode_steps,
            'episode_return': self._episode_return,
            'steps_per_second': steps_per_second,
        }
        result.update(counts)

        # Log the given results.
        self.write(result)

    @abc.abstractmethod
    def write(self, data: LoggingData):
        """Writes `data` to destination (file, terminal, database, etc)."""


def tensor_to_numpy(value: Any):
    if hasattr(value, 'numpy'):
        # Assuming TensorFlow.
        return value.numpy()
    else:
        return value


def to_numpy(values: Any):
    """Converts tensors in a nested structure to numpy.

    Converts tensors from Tensorflow to Numpy if needed without importing TF
    dependency.

    Args:
      values: nested structure with numpy and / or TF tensors.

    Returns:
      Same nested structure as values, but with numpy tensors.
    """
    return tree.map_structure(tensor_to_numpy, values)
