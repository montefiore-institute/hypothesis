"""
Base class for an event handler.
"""



class Handler:

    def process(self, event_type, message):
        raise NotImplementedError
