"""
Event definitions.
"""



class Event:

    def __init__(self):
        self._next_identifier = 0
        self.add_event("start")
        self.add_event("start_iteration")
        self.add_event("end_batch")
        self.add_event("start_iteration")
        self.add_event("end_epoch")
        self.add_event("terminate")
        self.add_event("log")
        self.add_event("loss")

    def add_event(self, identifier):
        setattr(self, identifier, self._next_identifier)
        self._next_identifier += 1


event = Event()
