from typed_python import Class, Final, Member, TypeFunction, ListOf, OneOf


@TypeFunction
def TypedQueue(T):
    """Create a synchronizing Queue with typed elements."""

    class TypedQueue(Class, Final):
        _pushable = Member(ListOf(T))
        _poppable = Member(ListOf(T))

        def get(self) -> OneOf(None, T):
            """Return a value from the Queue, or None if no value exists."""
            if not self._poppable and self._pushable:
                for i in range(len(self._pushable)):
                    self._poppable.append(self._pushable[-1 - i])
                self._pushable.clear()

            if self._poppable:
                return self._poppable.pop()

            return None

        def put(self, element: T) -> None:
            self._pushable.append(element)

        def peek(self) -> OneOf(None, T):
            if self._poppable:
                return self._poppable[-1]
            if self._pushable:
                return self._pushable[0]

        def __len__(self) -> int:
            return len(self._pushable) + len(self._poppable)

    return TypedQueue
