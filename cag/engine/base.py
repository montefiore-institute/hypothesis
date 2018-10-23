"""
Module base class.
"""

import multiprocessing

from .event import event


class Module:

    def __init__(self, workers=1):

        self._event_handlers = {}
        self._handlers = []
        self._queue = None
        self._running = True
        self.start()

    def _process_queue(self):
        while self._running or self._queue.peek():
            event_type, message = self._queue.pop()
            # Run it through all event handlers.
            if event_type in self._event_handlers.keys():
                for event_handler in self._event_handlers[event_type]:
                    event_handler(message)
            # Run it through all handlers.
            for handler in self._handlers:
                handler(event_type, message)

    def start(self):
        self.fire_event(event.start)
        self._running = True
        # TODO Start queue.

    def terminate(self):
        self.fire_event(event.terminate)
        self._running = False
        # TODO Cleanup queue.

    def fire_event(self, event_type, message):
        self._queue.put((event_type, message))

    def add_handler(self, handler):
        self._handlers.append(handler)

    def handlers(self):
        return self._handlers

    def add_event_handler(self, event_type, handler):
        if not event_type in self._event_handlers.keys():
            self._event_handlers[event_type] = []
        self._event_handlers[event_type].append(handler)

    def event_handlers(self, event_type):
        if not event_type in self._event_handlers.keys():
            return []

        return self._event_handlers[event_type]
