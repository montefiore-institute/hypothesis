"""
Event definitions.
"""



class Event:

    def __init__(self):
        self.add_event("start")
        self.add_event("start_batch")
        self.add_event("end_batch")
        self.add_event("start_iteration")
        self.add_event("end_iteration")
        self.add_event("start_epoch")
        self.add_event("end_epoch")
        self.add_event("terminate")
        self.add_event("log")

    def add_event(self, identifier):
        if not self.has_event(identifier):
            setattr(self, identifier, identifier)

    def has_event(self, identifier):
        return hasattr(self, identifier)


event = Event()
