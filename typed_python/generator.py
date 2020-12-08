from typed_python import Class, TypeFunction, PointerTo


@TypeFunction
def Generator(T):
    """TypeFunction producing base classes for typed Generator objects."""
    class Generator_(Class, __name__=f"Generator({T})"):
        def __iter__(self) -> Generator(T):
            return self

        def __next__(self) -> T:
            raise NotImplementedError()

        def __fastnext__(self, tPtr: PointerTo(T)) -> bool:
            raise NotImplementedError()

    return Generator_
