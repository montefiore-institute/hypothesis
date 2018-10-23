"""
Event definitions.
"""



class Event:

    def __init__(self):
        self._next_identifier = 0
        self.add_event("start")
        self.add_event("terminate")

    def add_event(self, identifier):
        setattr(self, identifier, self._next_identifier)
        self._next_identifier += 1


event = Event()
