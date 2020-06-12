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
import abc

from typing import List

from acme import core
from acme import types
from acme.callbacks import base

# Internal imports.

import numpy as np


class Agent:
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

    def __init__(self, actor, callbacks):
        self._actor = actor
        self._callback_list = self._init_callbacks(callbacks)

    def _init_callbacks(self, callbacks):
        callback_list = base.CallbackList(list(callbacks.values()))
        for name, cb in callbacks.items():
            attr_name = f'_{name}'
            assert not hasattr(self, attr_name)
            setattr(self, attr_name, cb)
            cb.set_owner(self)
        return callback_list

    @property
    def callbacks(self):
        return self._callback_list.callbacks

    def select_action(self, observation: types.NestedArray) -> types.NestedArray:
        self._callback_list.call('before_select_action', observation=observation)
        action = self._actor.select_action(observation)
        # TODO: improve this ugly workaround to make action mutable and let callbacks modify it
        action = [action]
        self._callback_list.call('after_select_action', action=action)
        action = action[0]
        return action

    @abc.abstractmethod
    def get_variables(self, names: List[str]) -> List[List[np.ndarray]]:
        pass
        #return self._learner.owner.get_variables(names)

# Internal class.
