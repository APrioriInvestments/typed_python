import types
from typed_python import Class, TypeFunction, PointerTo, Member, Final, pointerTo


@TypeFunction
def Generator(T):
    """TypeFunction producing base classes for typed Generator objects."""
    from typed_python import Entrypoint

    class Generator_(Class, __name__=f"Generator({T})"):
        IteratorType = T

        @Entrypoint
        def __iter__(self) -> Generator(T):
            return self

        def __next__(self) -> T:
            res = self.__fastnext__()
            if res:
                return res.get()
            else:
                raise StopIteration()

        def __fastnext__(self) -> PointerTo(T):
            raise NotImplementedError()

        @staticmethod
        def __convert_from__(iterator: types.GeneratorType) -> Generator(T):
            return GeneratorAdaptor(T)(iterator)

    return Generator_


@TypeFunction
def GeneratorAdaptor(T):
    """Adapt an untyped generator to a typed generator."""
    class GeneratorAdaptor_(Generator(T), Final, __name__=f"GeneratorAdaptor({T})"):
        IteratorType = T

        _iterator = Member(object)
        _result = Member(T)

        def __init__(self, iterator):
            self._iterator = iterator

        def __iter__(self) -> Generator(T):
            return self

        def __next__(self) -> T:
            res = self.__fastnext__()
            if res:
                return res.get()
            else:
                raise StopIteration()

        def __fastnext__(self) -> PointerTo(T):
            try:
                self._result = self._iterator.__next__()
            except StopIteration:
                return PointerTo(T)()

            return pointerTo(self)._result

    return GeneratorAdaptor_
