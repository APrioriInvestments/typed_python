from typed_python import (
    TypeFunction, Class, Member, Final, Entrypoint, OneOf, Generator, Tuple,
    Forward, ListOf
)


def less(x, y):
    return x < y


@TypeFunction
def SortedDict(K, V, comparator=less):
    Node = Forward("Node")

    @Node.define
    class Node(Class, Final):
        key = Member(K)
        value = Member(V)

        left = Member(OneOf(None, Node), nonempty=True)
        right = Member(OneOf(None, Node), nonempty=True)
        count = Member(int, nonempty=True)

        def __contains__(self, k: K) -> bool:
            if comparator(k, self.key):
                if self.left is not None:
                    return k in self.left
                else:
                    return False
            elif comparator(self.key, k):
                if self.right is not None:
                    return k in self.right
                else:
                    return False
            else:
                return True

        def get(self, k: K) -> V:
            if comparator(k, self.key):
                if self.left is None:
                    raise KeyError(k)
                return self.left.get(k)
            elif comparator(self.key, k):
                if self.right is None:
                    raise KeyError(k)
                return self.right.get(k)
            else:
                return self.value

        def set(self, k: K, v: V) -> bool:
            if comparator(k, self.key):
                if self.left is None:
                    self.left = Node(key=k, value=v, count=1)
                    self.count += 1
                    return True
                else:
                    if self.left.set(k, v):
                        self.count += 1
                        self.rebalance()
                        return True
                    return False
            elif comparator(self.key, k):
                if self.right is None:
                    self.right = Node(key=k, value=v, count=1)
                    self.count += 1
                    return True
                else:
                    if self.right.set(k, v):
                        self.count += 1
                        self.rebalance()
                        return True
                    return False
            else:
                self.value = v
                return False

        def first(self) -> K:
            if self.left is not None:
                return self.left.first()
            return self.key

        def last(self) -> K:
            if self.right is not None:
                return self.right.last()
            return self.key

        def become(self, otherNode: Node):
            self.key = otherNode.key
            self.value = otherNode.value
            self.left = otherNode.left
            self.right = otherNode.right
            self.count = otherNode.count

        def _checkInvariants(self):
            assert self.count == (
                1 + (0 if not self.left else self.left.count)
                + (0 if not self.right else self.right.count)
            )
            if self.left:
                assert comparator(self.left.key, self.key)
                self.left._checkInvariants()

            if self.right:
                assert comparator(self.key, self.right.key)
                self.right._checkInvariants()

        def remove(self, k: K) -> bool:
            """Remove 'k' and return True if we are now empty."""
            if comparator(k, self.key):
                if self.left is None:
                    raise KeyError(k)

                if self.left.remove(k):
                    self.left = None

                self.count -= 1
                self.rebalance()
                return False
            elif comparator(self.key, k):
                if self.right is None:
                    raise KeyError(k)

                if self.right.remove(k):
                    self.right = None

                self.count -= 1
                self.rebalance()
                return False
            else:
                if self.count == 1:
                    return True  # just remove us

                if self.left is not None and self.right is None:
                    # just become 'left'
                    self.become(self.left)
                    return False

                if self.right is not None and self.left is None:
                    self.become(self.right)
                    return False

                if self.right.count < self.left.count:
                    self.key = self.left.last()
                    self.value = self.left.get(self.key)
                    if self.left.remove(self.key):
                        self.left = None
                    self.count -= 1

                    self.rebalance()
                    return False
                else:
                    self.key = self.right.first()
                    self.value = self.right.get(self.key)

                    if self.right.remove(self.key):
                        self.right = None
                    self.count -= 1
                    self.rebalance()
                    return False

        def rebalance(self):
            if self.left is None and self.right is None:
                assert self.count == 1
                return

            if self.left is None and self.right is not None:
                if self.right.count <= 2:
                    return

                k = self.key
                v = self.value
                self.right.set(k, v)
                self.become(self.right)
                return

            if self.right is None and self.left is not None:
                if self.left.count <= 2:
                    return

                k = self.key
                v = self.value
                self.left.set(k, v)
                self.become(self.left)
                return

            if self.right is not None and self.left is not None:
                # both are populated. we should have that the
                # left count and right count imbalance is no greater than
                # a factor of two
                ll = 0 if self.left.left is None else self.left.left.count
                lr = 0 if self.left.right is None else self.left.right.count
                rl = 0 if self.right.left is None else self.right.left.count
                rr = 0 if self.right.right is None else self.right.right.count

                # if ll is much bigger than it should be, make 'll' the
                # new left side
                if ll > (3 + lr + rl + rr) * 2:
                    leftKey = self.left.key
                    leftVal = self.left.value
                    rootKey = self.key
                    rootValue = self.value

                    lNode = self.left
                    rNode = self.right
                    llNode = self.left.left
                    lrNode = self.left.right

                    self.left = llNode
                    self.key = leftKey
                    self.value = leftVal

                    self.right = Node(
                        key=rootKey,
                        value=rootValue,
                        left=lrNode,
                        right=rNode,
                        count=1 + lr + rNode.count
                    )
                    self.count = 1 + self.left.count + self.right.count

                elif rr > (3 + rl + lr + ll) * 2:
                    rightKey = self.right.key
                    rightVal = self.right.value
                    rootKey = self.key
                    rootValue = self.value

                    lNode = self.left
                    rNode = self.right
                    rlNode = self.right.left
                    rrNode = self.right.right

                    self.right = rrNode
                    self.key = rightKey
                    self.value = rightVal

                    self.left = Node(
                        key=rootKey,
                        value=rootValue,
                        left=lNode,
                        right=rlNode,
                        count=1 + rl + lNode.count
                    )
                    self.count = 1 + self.left.count + self.right.count

        def height(self):
            return max(
                0,
                1 + (0 if self.left is None else self.left.height()),
                1 + (0 if self.right is None else self.right.height())
            )

    class SortedDict_(Class, Final):
        _root = Member(OneOf(None, Node), nonempty=True)

        def __init__(self):
            pass

        def __init__(self, other):  # noqa
            for key in other:
                self[key] = other[key]

        def height(self):
            if self._root is None:
                return 0
            return self._root.height()

        @Entrypoint
        def __getitem__(self, key) -> V:
            if self._root is None:
                raise KeyError(key)

            return self._root.get(key)

        @Entrypoint
        def __contains__(self, key) -> bool:
            if self._root is None:
                return False

            return key in self._root

        @Entrypoint
        def __setitem__(self, k: K, v: V) -> None:
            if self._root is None:
                self._root = Node(key=k, value=v, count=1)
            else:
                self._root.set(k, v)

        @Entrypoint
        def __delitem__(self, k: K) -> None:
            if self._root is None:
                raise KeyError(k)

            if self._root.remove(k):
                self._root = None

        @Entrypoint
        def pop(self, k: K) -> V:
            if self._root is None:
                raise KeyError(k)

            res = self._root.get(k)
            if self._root.remove(k):
                self._root = None
            return res

        @Entrypoint
        def pop(self, k: K, v: V) -> V:  # noqa
            if self._root is None or k not in self._root:
                return v

            res = self._root.get(k)
            if self._root.remove(k):
                self._root = None
            return res

        @Entrypoint
        def first(self) -> OneOf(None, K):
            if self._root is None:
                return None

            return self._root.first()

        @Entrypoint
        def last(self) -> OneOf(None, K):
            if self._root is None:
                return None

            return self._root.last()

        @Entrypoint
        def get(self, k: K) -> V:
            return self[k]

        @Entrypoint
        def get(self, k: K, v: V) -> V:  # noqa
            if k in self:
                return self[k]
            return v

        @Entrypoint
        def setdefault(self, k: K) -> V:
            if k not in self:
                self[k] = V()
            return self[k]

        @Entrypoint
        def setdefault(self, k: K, v: V) -> V:  # noqa
            if k not in self:
                self[k] = v
            return self[k]

        @Entrypoint
        def __str__(self):
            return '{' + ",".join(f'{k}: {v}' for k, v in self.items()) + '}'

        @Entrypoint
        def __repr__(self):
            return '{' + ",".join(f'{k}: {v}' for k, v in self.items()) + '}'

        def __len__(self):
            return self._root.count if self._root is not None else 0

        @Entrypoint
        def _checkInvariants(self):
            if not self._root:
                return
            self._root._checkInvariants()

        @Entrypoint
        def items(self) -> Generator(Tuple(K, V)):
            stack = ListOf(Tuple(Node, bool))()
            if self._root is None:
                return

            stack.append((self._root, True))

            while stack:
                node, wayDown = stack.pop()

                if wayDown:
                    if node.left:
                        stack.append((node, False))
                        stack.append((node.left, True))
                    else:
                        yield (node.key, node.value)

                        if node.right:
                            stack.append((node.right, True))
                else:
                    yield (node.key, node.value)

                    if node.right:
                        stack.append((node.right, True))

        @Entrypoint
        def __iter__(self) -> Generator(K):
            stack = ListOf(Tuple(Node, bool))()
            if self._root is None:
                return

            stack.append((self._root, True))

            while stack:
                node, wayDown = stack.pop()

                if wayDown:
                    if node.left:
                        stack.append((node, False))
                        stack.append((node.left, True))
                    else:
                        yield node.key

                        if node.right:
                            stack.append((node.right, True))
                else:
                    yield node.key

                    if node.right:
                        stack.append((node.right, True))

    return SortedDict_
