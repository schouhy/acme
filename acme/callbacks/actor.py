# import dm_env
#
# from acme.callbacks.base import ActorCallback
# from acme import core, types
#
# import abc
#
#
# class ActorCallback(ActorCallback, core.Actor, abc.ABC):
#     def on_feedback(self, action: types.NestedArray, next_timestep: dm_env.TimeStep):
#         self.observe(action=action, next_timestep=next_timestep)
#         ## TODO: check that it is ok to update on every tick and not only after learners update
#         self.update()
#
#     def on_episode_begin(self, timestep: dm_env.TimeStep):
#         self.observe_first(timestep)
from acme.callbacks import base
from acme import types
from acme import core
import weakref


class ActorCallback(base.BaseCallback, core.Actor):
    def __init__(self, callbacks=None):
        callbacks = callbacks or []
        self._callbacks = base.CallbackList(callbacks)
        super(ActorCallback, self).__init__()

    def set_agent(self, actor, enable=True):
        if hasattr(self, '_actor_ref'):
            raise Exception(f'Callback ({self}) already initialized')
        self._actor_ref = weakref.ref(actor)
        self.on_set_actor(actor)
        self._is_enabled = enable

    @property
    def actor(self):
        return self._actor_ref()

    def on_set_actor(self, actor):
        pass

    def before_act(self, observation):
        pass

    def after_act(self, action):
        pass

    def act(self, observation: types.NestedArray) -> types.NestedArray:


class ActorCallbackList(base.CallbackList):
    def add_callback(self, callback, actor, enable=True):
        callback.set_agent(actor=actor, enable=enable)
        self._callbacks.append(callback)
