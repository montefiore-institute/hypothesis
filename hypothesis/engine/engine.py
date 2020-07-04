from hypothesis.exception import NoSuchEventException
from hypothesis.exception import NoEventRegistrationException



class Events: pass



class Procedure:
    r""""""

    def __init__(self):
        self.hooks = {}
        self.events = Events()
        self.num_events = 0
        self._register_events()

    def _register_events(self):
        raise NoEventRegistrationException

    def _event_exists(self, event):
        return event in self.hooks.keys()

    def register_event(self, event):
        # Check if the event already exists.
        event = event.lower()
        if not hasattr(self.events, event):
            setattr(self.events, event, self.num_events)
            event_index = self.num_events
            self.hooks[event_index] = []
            self.num_events += 1

    def registered_events(self):
        return self.events

    def add_event_handler(self, event, f):
        # Check if the specified event exists.
        if not self._event_exists(event):
            raise NoSuchEventException
        self.hooks[event].append(f)

    def clear_event_handler(self, event):
        # Check if the specified event exists.
        if not self._event_exists(event):
            raise NoSuchEventException()
        self.hooks[event] = []

    def clear_event_handlers(self):
        for key in self.hooks.keys():
            self.clear_event_handler(key)

    def call_event(self, event, **kwargs):
        # Check if the specified event exists.
        if not self._event_exists(event):
            raise NoSuchEventException
        handlers = self.hooks[event]
        for handler in handlers:
            handler(self, **kwargs)

    def on(self, event):
        def wrap(f):
            self.add_event_handler(event, f)

        return wrap
