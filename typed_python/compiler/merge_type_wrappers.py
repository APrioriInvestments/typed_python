import typed_python

from typed_python.compiler.type_wrappers.wrapper import Wrapper
from typed_python import OneOf, Value, Class


def typeWrapper(t):
    return typed_python.compiler.python_object_representation.typedPythonTypeToTypeWrapper(t)


_mergedWrapperCache = {}


def mergeTypes(types):
    """Return a Type that covers all of 'types'."""
    types = [t for t in types if t is not None]

    if not types:
        return None

    if len(types) == 1:
        return types[0]

    if len(types) == 2 and types[0] == types[1]:
        return types[0]

    return mergeTypeWrappers(
        [typeWrapper(x) for x in types]
    ).interpreterTypeRepresentation


def mergeTypeWrappers(wrappers):
    """Return a Wrapper for a type that covers all of 'wrappers'.

    In particular, we'll produce a OneOf if the types are unalike (e.g. None and
    float), but we are careful to ensure that we don't produce an overly large
    OneOf by merging overlapping types. For instance, we drop child classes
    if their base class is present, and we drop constants if the concrete type
    is present (e.g. drop 'Value(5)' if 'int' is present).

    If the set is empty we will return None.

    Args:
        wrappers - a tuple of Type or Wrapper instances. None will be filtered.

    Returns:
        None, or a Wrapper instance.
    """
    wrappers = [typeWrapper(w) if not isinstance(w, Wrapper) else w for w in wrappers if w is not None]

    if not wrappers:
        return None

    if len(wrappers) == 1:
        return wrappers[0]

    if len(wrappers) == 2 and wrappers[0] == wrappers[1]:
        return wrappers[0]

    wrappers = tuple(wrappers)

    if wrappers in _mergedWrapperCache:
        return _mergedWrapperCache[wrappers]

    _mergedWrapperCache[wrappers] = _mergeTypeWrappers(wrappers)

    return _mergedWrapperCache[wrappers]


def typeSubsumes(bigType, smallType):
    """Return True if we can drop 'smallType' because of 'bigType'"""
    if bigType == smallType:
        return True

    if bigType is object:
        return True

    if issubclass(smallType, bigType) and issubclass(bigType, Class):
        return True

    if issubclass(smallType, Value) and isinstance(smallType.Value, bigType):
        return True

    return False


def typeSortKey(t):
    return (0 if issubclass(t, Value) else 1, str(t), id(t))


def _mergeTypeWrappers(wrappers):
    assert wrappers

    types = []

    for w in wrappers:
        newType = w.interpreterTypeRepresentation

        newTypes = list(newType.Types) if issubclass(newType, OneOf) else [newType]

        for t in newTypes:
            shouldAdd = True
            ix = 0
            while ix < len(types):
                if typeSubsumes(types[ix], t):
                    shouldAdd = False
                    break
                elif typeSubsumes(t, types[ix]):
                    types.pop(ix)
                    ix -= 1
                ix += 1

            if shouldAdd:
                types.append(t)

    if len(types) == 1:
        return typeWrapper(types[0])

    assert len(types) > 1, wrappers

    return typeWrapper(OneOf(*sorted(types, key=typeSortKey)))
