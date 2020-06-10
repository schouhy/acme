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


class AgentCallback(BaseCallback):
    def __init__(self):
        self._owner_ref = None
        super(AgentCallback, self).__init__()

    def set_owner(self, owner):
        if self._owner_ref is not None:
            raise Exception(f'Callback ({self}) already initialized')
        self.on_set_owner(owner)
        self._owner_ref = weakref.ref(owner)
        return self

    @property
    def owner(self):
        return self._owner_ref()

    ## events
    def on_set_owner(self, owner):
        """This may be used to assert that the owner is of certain type"""
        pass

    def before_select_action(self, observation):
        pass

    def after_select_action(self, action):
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

    def add_callback(self, callback, enable=True):
        self._callbacks.append(callback)
