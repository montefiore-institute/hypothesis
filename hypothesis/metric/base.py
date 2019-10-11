class BaseMetric:

    def update(self, value):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def __getitem__(self, pattern):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError
