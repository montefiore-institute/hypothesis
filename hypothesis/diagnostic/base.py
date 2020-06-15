import hypothesis



class BaseDiagnostic:

    def reset(self):
        raise NotImplementedError

    def test(self, **kwargs):
        raise NotImplementedError
