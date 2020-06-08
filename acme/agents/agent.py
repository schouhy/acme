# python3
# Copyright 2018 DeepMind Technologies Limited. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""The base agent interface."""

from typing import List

from acme import core
from acme import types
from acme.callbacks import base
# Internal imports.

import dm_env
import numpy as np


class Agent(core.VariableSource):
    """Agent class which combines acting and learning.

    This provides an implementation of the `Actor` interface which acts and
    learns. It takes as input instances of both `acme.Actor` and `acme.Learner`
    classes, and implements the policy, observation, and update methods which
    defer to the underlying actor and learner.

    The only real logic implemented by this class is that it controls the number
    of observations to make before running a learner step. This is done by
    passing the number of `min_observations` to use and a ratio of
    `observations_per_step`

    Note that the number of `observations_per_step` which can also be in the range
    [0, 1] in order to allow more steps per update.
    """

    def __init__(self, actor: core.Actor, callbacks=None):
        self._actor = actor

        callbacks = callbacks or {}
        callbacks.update({'actor': actor})
        ## callbacks
        self._callback_list = base.AgentCallbackList(list(callbacks.values()))
        for name, cb in callbacks.items():
            cb.set_agent(self)
            assert not hasattr(self, name)
            setattr(self, name, cb)

    @property
    def callback_list(self):
        return self._callback_list.callbacks

    def select_action(self, observation: types.NestedArray) -> types.NestedArray:
        return self._actor.select_action(observation)

    def get_variables(self, names: List[str]) -> List[List[np.ndarray]]:
        return self._learner.get_variables(names)

# Internal class.
