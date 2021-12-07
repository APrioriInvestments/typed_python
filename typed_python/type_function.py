#   Copyright 2019-2020 typed_python Authors
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

import logging

from types import FunctionType
from typed_python.compiler.runtime_lock import runtimeLock
from typed_python._types import Forward, ListOf, TupleOf, Dict, ConstDict, Class
import typed_python


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


def makeTypeFunction(f):
    """Decorate 'f' to be a 'TypeFunction'.

    The resulting function is expected to take a set of hashable arguments and
    produce a type object.  The function is memoized, so code in the
    decorated function is executed once only for each distinct set of
    arguments. The order of keyword arguments is not considered in the memo.
    The function should not have sideeffects.

    TypeFunctions may call each other recursively and in self-referential
    cycles. If the function calls back into itself, a Forward will
    be returned instead of a concrete type, which lets you express recursive
    types in a natural way.

    Don't stash the type you return, since the actual type returned by the
    function may not be the one you returned.
    """
    def nameFor(args, kwargs):
        def toStr(x):
            if isinstance(x, type):
                return x.__qualname__
            return str(x)

        return f.__qualname__ + "(" + ",".join(
            [toStr(x) for x in args] + ["%s=%s" % (k, v) for k, v in kwargs]
        ) + ")"

    def mapArg(arg):
        """Map a type argument to a valid value. Type arguments must be hashable,
        and if they're forward types, they must be resolvable.
        """
        if arg is None:
            return None

        if isinstance(arg, (type, float, int, bool, str, bytes)):
            return arg

        if isinstance(arg, (tuple, list, TupleOf, ListOf)):
            return tuple(mapArg(x) for x in arg)

        if isinstance(arg, (dict, Dict, ConstDict)):
            return tuple(sorted([(k, mapArg(v)) for k, v in arg.items()]))

        if isinstance(arg, FunctionType):
            return mapArg(typed_python.Function(arg))

        if isinstance(arg, typed_python._types.Function):
            return arg

        raise TypeError("Instance of type '%s' is not a valid argument to a type function" % type(arg))

    _memoForKey = {}

    def buildType(*args, **kwargs):
        args = tuple(mapArg(a) for a in args)
        kwargs = tuple(sorted([(k, mapArg(v)) for k, v in kwargs.items()]))

        key = (args, kwargs)

        if key in _memoForKey:
            res = _memoForKey[key]
            if isinstance(res, Exception):
                raise res

            if getattr(res, '__typed_python_category__', None) != 'Forward':
                # only return fully resolved TypeFunction values without
                # locking.
                return res

        with runtimeLock:
            if key in _memoForKey:
                res = _memoForKey[key]
                if isinstance(res, Exception):
                    raise res
                return res

            forward = Forward(nameFor(args, kwargs))

            _memoForKey[key] = forward
            _type_to_typefunction[forward] = (TypeFunction_, key)

            try:
                resultType = f(*args, **dict(kwargs))

                forward.define(resultType)

                _type_to_typefunction.pop(forward)

                if resultType not in _type_to_typefunction:
                    _type_to_typefunction[resultType] = (TypeFunction_, key)

                _memoForKey[key] = resultType

                return resultType
            except Exception as e:
                _memoForKey[key] = e
                logging.exception("TypeFunction errored")
                raise

    class TypeFunction_(TypeFunction):
        __module__ = f.__module__
        __qualname__ = f.__qualname__
        __name__ = f.__name__
        __typed_python_template__ = buildType

    return TypeFunction_


class TypeFunction(Class):
    __typed_python_template__ = makeTypeFunction
