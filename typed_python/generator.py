from typed_python import Class, TypeFunction, PointerTo


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

    return Generator_
