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

"""A simple agent-environment training loop."""

import itertools
import time

from typing import Optional

from acme.agents import agent
from acme import core
# Internal imports.
from acme.utils import counting
from acme.callbacks import loggers
from acme.callbacks import base

import dm_env


class EnvironmentLoop(core.Worker):
    """A simple RL environment loop.

    This takes `Environment` and `Actor` instances and coordinates their
    interaction. This can be used as:

      loop = EnvironmentLoop(environment, actor)
      loop.run(num_episodes)

    A `Counter` instance can optionally be given in order to maintain counts
    between different Acme components. If not given a local Counter will be
    created to maintain counts between calls to the `run` method.

    A `Logger` instance can also be passed in order to control the output of the
    loop. If not given a platform-specific default logger will be used as defined
    by utils.loggers.make_default_logger. A string `label` can be passed to easily
    change the label associated with the default logger; this is ignored if a
    `Logger` instance is given.
    """

    def __init__(
            self,
            environment: dm_env.Environment,
            agent: agent.Agent,
            logger: loggers.BaseLoggerCallback = None,
            label: str = 'environment_loop'
    ):
        # Internalize agent and environment.
        self._environment = environment
        self._agent = agent
        logger = logger or loggers.make_default_logger(label)
        callbacks = [logger] + self._agent.callback_list
        self._callbacks = base.CallbackList(callback_list=callbacks)

    def run(self, num_episodes: Optional[int] = None):
        """Perform the run loop.

        Run the environment loop for `num_episodes` episodes. Each episode is itself
        a loop which interacts first with the environment to get an observation and
        then give that observation to the agent in order to retrieve an action. Upon
        termination of an episode a new episode will be started. If the number of
        episodes is not given then this will interact with the environment
        infinitely.

        Args:
          num_episodes: number of episodes to run the loop for. If `None` (default),
            runs without limit.
        """

        iterator = range(num_episodes) if num_episodes else itertools.count()

        for _ in iterator:
            timestep = self._environment.reset()
            self._callbacks.call('on_episode_begin', timestep=timestep)

            # Run an episode.
            while not timestep.last():
                # Generate an action from the agent's policy and step the environment.
                action = self._agent.select_action(timestep.observation)
                timestep = self._environment.step(action)

                # Have the agent observe the timestep and let the actor update itself.
                self._callbacks.call('on_feedback', action=action, next_timestep=timestep)

            self._callbacks.call('on_episode_end')

# Internal class.
