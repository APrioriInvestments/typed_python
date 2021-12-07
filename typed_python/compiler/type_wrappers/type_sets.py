class SubclassOf:
    def __init__(self, T):
        self.T = T

    def __getattr__(self, attr):
        return getattr(self.T, attr)


class Either:
    def __init__(self, types):
        self.Types = types

    def __getattr__(self, attr):
        vals = [getattr(T, attr) for T in self.Types]

        if len(set(vals)) == 1:
            return vals[0]
