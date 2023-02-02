from typed_python import Generator, Final, Member, PointerTo, pointerTo


class IteratorAdaptor(Generator(object), Final):
    """Adapts a regular iterator to a __fastnext__ iterator"""
    _iterator = Member(object)
    _instance = Member(object)

    def __init__(self, iterator):
        self._iterator = iterator

    def __next__(self) -> object:
        return self._iterator.__next__()

    def __fastnext__(self) -> PointerTo(object):
        try:
            self._instance = self._iterator.__next__()
        except StopIteration:
            return PointerTo(object)()

        return pointerTo(self)._instance
