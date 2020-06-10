import abc
from acme import types

from acme.callbacks.base import ActorCallback
from acme import core

import dm_env


class LearnerCallback(ActorCallback, core.Learner, abc.ABC):
    def __init__(self, min_observations: int, observations_per_step: float):
        # We'll ignore the first min_observations when determining whether to take
        # a step and we'll do so by making sure num_observations >= 0.
        self._num_observations = -min_observations

        # Rather than work directly with the observations_per_step ratio we can
        # figure out how many observations or steps to run per update, one of which
        # should be one.
        if observations_per_step >= 1.0:
            self._observations_per_update = int(observations_per_step)
            self._steps_per_update = 1
        else:
            self._observations_per_update = 1
            self._steps_per_update = int(1.0 / observations_per_step)
        super(LearnerCallback, self).__init__()

    def on_feedback(self, action: types.NestedArray, next_timestep: dm_env.TimeStep):
        self._num_observations += 1

        # Only allow updates after some minimum number of observations have been and
        # then at some period given by observations_per_update.
        if (self._num_observations >= 0 and
                self._num_observations % self._observations_per_update == 0):
            self._num_observations = 0

            # Run a number of learner steps (usually gradient steps).
            for _ in range(self._steps_per_update):
                self.step()
