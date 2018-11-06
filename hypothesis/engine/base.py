"""
Module base class.
"""

import torch.multiprocessing

from .event import event



class Module:

    def __init__(self, workers=1):
        self._event_handlers = {}
        self._handlers = []
        self._num_workers = workers
        self._queue = torch.multiprocessing.Queue()
        self._running = True
        self._workers = []
        self.start()

    def _process_queue(self):
        while self._running or self._queue.peek():
            event_type, message = self._queue.get()
            # Run the event through all handlers.
            if event_type in self._event_handlers.keys():
                for event_handler in self._event_handlers[event_type]:
                    event_handler(message);
            # Run it through all handlers.
            for handler in self._handlers:
                handler(event_type, message)

    def start(self):
        self._running = True
        # Start the workers.
        for worker_index in range(self._num_workers):
            worker = torch.multiprocessing.Process(target=self._process_queue)
            worker.start()

    def terminate(self):
        self._running = False
        # Terminate and wait for the processes to clean up.
        for worker in self._workers:
            worker.terminate()
            worker.join()
        self._workers = []

    def fire_event(self, event_type, message=None):
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
