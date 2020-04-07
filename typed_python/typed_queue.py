from typed_python import Class, Final, Member, TypeFunction, ListOf, OneOf, Entrypoint

from threading import Lock


@TypeFunction
def TypedQueue(T):
    """Create a Queue with typed elements."""
    class TypedQueue(Class, Final):
        _pushable = Member(ListOf(T))
        _poppable = Member(ListOf(T))
        _lock = Member(Lock)
        _isEmptyLock = Member(Lock)

        def __init__(self):
            self._pushable = ListOf(T)()
            self._poppable = ListOf(T)()

            # this lock is held while making changes to the datastructure
            self._lock = Lock()

            # this lock is active whenever the queue is empty. you may
            # acquire this lock if you also have _lock, but not the other
            # way around (to prevent deadlock)
            self._isEmptyLock = Lock()
            self._isEmptyLock.acquire()

        @Entrypoint
        def getNonblocking(self) -> OneOf(None, T):
            """Return a value from the Queue, or None if no value exists."""
            self._lock.acquire()

            if not self._poppable and self._pushable:
                for i in range(len(self._pushable)):
                    self._poppable.append(self._pushable[-1 - i])
                self._pushable.clear()

            if self._poppable:
                result = self._poppable.pop()

                if self._len == 0:
                    # leave the lock locked if we're out of elements.
                    self._isEmptyLock.acquire()

                self._lock.release()
                return result

            self._lock.release()
            return None

        @Entrypoint
        def get(self) -> T:
            """Return a value from the Queue"""
            self._lock.acquire()

            while not self._len:
                # we need to wait until the queue is nonempty
                self._lock.release()
                self._isEmptyLock.acquire()
                self._isEmptyLock.release()
                self._lock.acquire()

            if not self._poppable and self._pushable:
                for i in range(len(self._pushable)):
                    self._poppable.append(self._pushable[-1 - i])
                self._pushable.clear()

            if self._poppable:
                result = self._poppable.pop()

                if self._len == 0:
                    # leave the lock locked if we're out of elements.
                    self._isEmptyLock.acquire()

                self._lock.release()
                return result

            raise Exception("impossible")

        @Entrypoint
        def getMany(self, minCount: int, maxCount: int) -> ListOf(T):
            """Return a list of values from the Queue.

            Block until we get 'minCount'.
            """
            try:
                self._lock.acquire()

                res = ListOf(T)()

                while len(res) < maxCount:
                    if self._len == 0 and len(res) >= minCount:
                        return res

                    while not self._len:
                        # we need to wait until the queue is nonempty
                        try:
                            self._lock.release()
                            try:
                                self._isEmptyLock.acquire()
                            finally:
                                self._isEmptyLock.release()
                        finally:
                            self._lock.acquire()

                    if not self._poppable and self._pushable:
                        for i in range(len(self._pushable)):
                            self._poppable.append(self._pushable[-1 - i])
                        self._pushable.clear()

                    if self._poppable:
                        res.append(self._poppable.pop())

                        if self._len == 0:
                            # leave the lock locked if we're out of elements.
                            self._isEmptyLock.acquire()

                return res
            finally:
                self._lock.release()

        @Entrypoint
        def put(self, element: T) -> None:
            self._lock.acquire()

            self._pushable.append(element)

            if self._len == 1:
                # wake listeners up
                self._isEmptyLock.release()

            self._lock.release()

        @Entrypoint
        def putMany(self, elementSeq: ListOf(T)) -> None:
            with self._lock:
                curLen = self._len

                for elt in elementSeq:
                    self._pushable.append(elt)

                    curLen += 1

                    if curLen == 1:
                        # wake listeners up
                        self._isEmptyLock.release()

        @Entrypoint
        def peek(self) -> OneOf(None, T):
            with self._lock:
                if self._poppable:
                    return self._poppable[-1]
                if self._pushable:
                    return self._pushable[0]

        @Entrypoint
        def __len__(self) -> int:
            with self._lock:
                return self._len

        @property
        def _len(self) -> int:
            return len(self._pushable) + len(self._poppable)

    return TypedQueue
