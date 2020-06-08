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


class AgentCallback(BaseCallback):
    def set_agent(self, agent, enable=True):
        if hasattr(self, 'agent'):
            raise Exception(f'Callback ({self}) already initialized')
        self._agent_ref = weakref.ref(agent)
        self.on_set_agent(agent)
        self._is_enabled = enable

    @property
    def agent(self):
        return self._agent_ref()

    def on_set_agent(self, agent):
        pass

    def before_act(self, observation):
        pass

    def after_act(self, action):
        pass

class AgentCallbackList(CallbackList):
    def add_callback(self, callback, agent, enable=True):
        callback.set_agent(agent=agent, enable=enable)
        self._callbacks.append(callback)

