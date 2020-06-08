import weakref

from acme import types

# Internal imports.
import dm_env


class BaseCallback:
    def __init__(self):
        self._is_enabled = True

    @property
    def is_enabled(self):
        return self._is_enabled

    def enable(self):
        self._is_enabled = True

    def disable(self):
        self._is_enabled = False

    ## events

    def on_feedback(self, action: types.NestedArray, next_timestep: dm_env.TimeStep):
        pass

    def on_episode_begin(self, timestep: dm_env.TimeStep):
        pass

    def on_episode_end(self):
        pass


class CallbackList:
    def __init__(self, callback_list=None):
        self._callbacks = callback_list or []

    @property
    def callbacks(self):
        return self._callbacks

    def call(self, event, **params):
        for cb in self._callbacks:
            if cb.is_enabled:
                getattr(cb, event)(**params)


class ActorCallback(BaseCallback):
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

class ActorCallbackList(CallbackList):
    def add_callback(self, callback, agent, enable=True):
        callback.set_agent(actor=agent, enable=enable)
        self._callbacks.append(callback)

