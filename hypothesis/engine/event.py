"""
Base event definitions.
"""



class Event:

    def __init__(self):
        self.add_event("start")
        self.add_event("terminate")
        self.add_event("batch_start")
        self.add_event("batch_end")
        self.add_event("iteration_start")
        self.add_event("iteration_end")
        self.add_event("epoch_start")
        self.add_event("epoch_end")
        self.add_event("log")

    def add_event(self, identifier):
        if not self.has_event(identifier):
            setattr(self, identifier, 0);

    def has_event(self, identifier):
        return hasattr(self, identifier)


event = Event()
