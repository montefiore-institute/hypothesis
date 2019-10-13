class BaseMetric:
    r""""""

    def update(self, value=None):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def __getitem__(self, pattern):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError



class BaseValueMetric(BaseMetric):
    r""""""

    def __init__(self, initial_value=None):
        self.initial_value = initial_value
        self.current_value = initial_value
        self.history = []
        self.reset()

    def _set_current_value(self, value):
        self.history.append(value)
        self.current_value = value

    def reset(self):
        self.current_value = self.initial_value
        if initial_value is not None:
            self.history = [self.initial_value]

    def __getitem__(self, pattern):
        return self.history[pattern]

    def __len__(self):
        return len(self.history)



class BaseStateMetric(BaseValueMetric):
    r""""""

    def __init__(initial_value=None):
        super(BaseStateMetric, self).__init__(initial_value)

    def update(self, value):
        self._set_current_value(value)

    def update(self):
        raise NotImplementedError
