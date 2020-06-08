import dm_env

from acme.callbacks.base import AgentCallback
from acme import core, types

import abc


class ActorCallback(AgentCallback, core.Actor, abc.ABC):
    def on_feedback(self, action: types.NestedArray, next_timestep: dm_env.TimeStep):
        self.observe(action=action, next_timestep=next_timestep)
        ## TODO: check that it is ok to update on every tick and not only after learners update
        self.update()

    def on_episode_begin(self, timestep: dm_env.TimeStep):
        self.observe_first(timestep)
