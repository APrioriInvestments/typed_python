#   Copyright 2019 Nativepython Authors
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

from types import FunctionType

_type_to_typefunction = {}


def isTypeFunctionType(type):
    """Is 'type' the result of a type function?

    If so, return a tuple of (func, args, kwargs). otherwise none
    """
    if type in _type_to_typefunction:
        (func, (args, kwargs)) = _type_to_typefunction[type]
        return (func, args, kwargs)
    return None


def reconstructTypeFunctionType(typeFunction, args, kwargs):
    """Reconstruct a type from the values returned by 'isTypeFunctionType'"""

    # note that our 'key' objects are dict-in-tuple-form, because dicts are
    # not hashable. So to keyword-call with them, we have to convert back to a dict...
    return typeFunction(*args, **dict(kwargs))


class ConcreteTypeFunction(object):
    def __init__(self, concreteTypeFunction):
        self._concreteTypeFunction = concreteTypeFunction

        self._memoForKey = {}

    def __str__(self):
        return self._concreteTypeFunction.__name__

    def __repr__(self):
        return self._concreteTypeFunction.__qualname__

    def nameFor(self, args, kwargs):
        def toStr(x):
            if isinstance(x, type):
                return x.__qualname__
            return str(x)

        return self._concreteTypeFunction.__qualname__ + "(" + ",".join(
            [toStr(x) for x in args] + ["%s=%s" % (k, v) for k, v in kwargs]
        ) + ")"

    def applyNameChangesToType(self, type, name):
        # if hasattr(type, '__typed_python_category__'):
        #     if type.__typed_python_category__ == "Alternative" or type.__typed_python_category__ == "Class":
        #         return _types.RenameType(type, name)
        return type

    def mapArg(self, arg):
        """Map a type argument to a valid value. Type arguments must be hashable,
        and if they're forward types, they must be resolvable.
        """
        if arg is None:
            return None

        if isinstance(arg, (type, float, int, bool, str, bytes)):
            return arg

        if isinstance(arg, (tuple, list)):
            return tuple(self.mapArg(x) for x in arg)

        if isinstance(arg, dict):
            return tuple(sorted([(k, self.mapArg(v)) for k, v in arg.items()]))

        if isinstance(arg, FunctionType):
            return self.mapArg(arg())

        raise TypeError("Instance of type '%s' is not a valid argument to a type function" % type(arg))

    def __call__(self, *args, **kwargs):
        args = tuple(self.mapArg(a) for a in args)
        kwargs = tuple(sorted([(k, self.mapArg(v)) for k, v in kwargs.items()]))

        key = (args, kwargs)

        if key in self._memoForKey:
            return self._memoForKey[key]

        def forward():
            if isinstance(self._memoForKey[key], Exception):
                raise self._memoForKey[key]

            if self._memoForKey[key] is forward:
                # if this gets triggered, it means we're asking for concrete information about the type
                # when dependent type-functions have yet to resolve.
                raise TypeError("Forward declaration for %s has not resolved yet" % self.nameFor(args, kwargs))

            return self._memoForKey[key]

        forward.__name__ = self.nameFor(args, kwargs)

        self._memoForKey[key] = forward
        _type_to_typefunction[forward] = (self, key)

        try:
            resultType = self._concreteTypeFunction(*args, **dict(kwargs))

            resultType = self.applyNameChangesToType(resultType, self.nameFor(args, kwargs))

            self._memoForKey[key] = resultType

            _type_to_typefunction[resultType] = (self, key)

            return resultType
        except Exception as e:
            self._memoForKey[key] = e
            raise


def TypeFunction(f):
    """Decorate 'f' to be a 'TypeFunction'.

    The resulting function is expected to take a set of hashable arguments and
    produce a type object.  The function is memoized, so code in the
    decorated function is executed once only for each distinct set of
    arguments. The order of keyword arguments is not considered in the memo.
    The function should not have sideeffects. The resulting type, if it is a
    Class or  Alternative, will have its name amended to include the type
    arguments.

    TypeFunctions may call each other recursively and in self-referential
    cycles. If the function calls back into itself, a forward type lambda will
    be returned instead of a concrete type, which lets you express recursive
    types in a natural way.

    Don't stash the type you return, since the actual type returned by the
    function may not be the one you returned (name changes require creating a
    new object). """

    return ConcreteTypeFunction(f)
