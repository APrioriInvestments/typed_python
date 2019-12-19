from typed_python import OneOf
from typed_python.compiler.type_wrappers.one_of_wrapper import OneOfWrapper


def mergeTypes(t1, t2):
    if t1 is None:
        return t2
    if t2 is None:
        return t1
    if t1 == t2:
        return t1
    return OneOf(t1, t2)


def removeTypeFrom(type, toRemove):
    if getattr(type, '__typed_python_category__', None) == "OneOf":
        types = [x for x in type.Types if x is not toRemove]
        if len(types) == 0:
            return None

        if len(types) == 1:
            return types[0]

        return OneOfWrapper.mergeTypes(types).interpreterTypeRepresentation
    return type


class FunctionStackState:
    """Model what's known about a set of variables at a particular point in a program.

    Each variable may be one of a set of types, or 'Uninitialized'
    """
    def __init__(self, types=None, maybeUnintialized=None):
        self._types = types or dict()
        self._maybeUnintialized = maybeUnintialized or set()

    def __str__(self):
        res = []
        for name, t in self._types.items():
            res.append(f"{name} -> {t} {'[or uninit]' if name in self._maybeUnintialized else ''}")
        return "\n".join(res) + "\n"

    def isDefinitelyInitialized(self, name):
        return name in self._types and name not in self._maybeUnintialized

    def couldBeUninitialized(self, name):
        return not self.isDefinitelyInitialized(name)

    def isDefinitelyUninitialized(self, name):
        return name not in self._types

    def variablesThatMightBeActive(self):
        """Return a list of variables that are either initialized or maybe initialized."""
        return sorted(self._types)

    def variableAssigned(self, varname, varType):
        """Indicate that local variable 'varname' now has type 'type' which must be a typed python type."""
        self._types[varname] = varType
        self._maybeUnintialized.discard(varname)

    def currentType(self, varname):
        return self._types.get(varname)

    def clone(self):
        return FunctionStackState(dict(self._types), set(self._maybeUnintialized))

    def becomeMerge(self, left, right):
        """Become the result of two control flow paths merging.

        If left or right is None, then don't take their information, because
        control flow does not return from that path.
        """
        assert left is not self
        assert right is not self

        if left is None and right is None:
            self._types = {}
            self._maybeUnintialized = set()
            return

        if left is None:
            self._types = dict(right._types)
            self._maybeUnintialized = set(right._maybeUnintialized)
            return

        if right is None:
            self._types = dict(left._types)
            self._maybeUnintialized = set(left._maybeUnintialized)
            return

        allNames = (
            set(left._types)
            .union(right._types)
            .union(left._maybeUnintialized)
            .union(right._maybeUnintialized)
        )

        self._types = {}
        self._maybeUnintialized = set()

        for name in allNames:
            self._types[name] = mergeTypes(left._types.get(name), right._types.get(name))

            if (name in left._maybeUnintialized or name in right._maybeUnintialized
                    or not (name in left._types and name in right._types)):
                self._maybeUnintialized.add(name)

    def mergeWithSelf(self, other):
        self.becomeMerge(self.clone(), other)

    def restrictTypeFor(self, varname, toType, shouldHaveType):
        """Indicate that 'varname' has type toType (or does not have, if shouldHaveType is False)"""
        if varname not in self._types:
            return

        if shouldHaveType:
            self._types[varname] = toType
        else:
            newType = removeTypeFrom(self._types[varname], toType)

            if newType is not None:
                self._types[varname] = newType

    def __eq__(self, other):
        if not isinstance(other, FunctionStackState):
            return False

        return self._types == other._types and self._maybeUnintialized == other._maybeUnintialized
