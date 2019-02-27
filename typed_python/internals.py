#   Copyright 2018 Braxton Mckee
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

import typed_python._types as _types
import typed_python.inspect_override as inspect

from typed_python._types import NamedTuple


class UndefinedBehaviorException(BaseException):
    """An unsafe operation with known undefined behavior was performed.

    This Exception is deliberately not a subclass of Exception because by
    default it should not be caught by normal exception handlers. In compiled
    code, the operation that raised this exception is likely to segfault.
    """


# needed by the C api
object = object


def forwardToName(fwdLambda):
    """Unwrap a 'forward definition' lambda to a name.

    Maps functions like 'lambda: X' to the string 'X'.
    """
    if hasattr(fwdLambda, "__code__"):
        if fwdLambda.__code__.co_code == b't\x00S\x00':
            return fwdLambda.__code__.co_names[0]
        if fwdLambda.__code__.co_code == b'\x88\x00S\x00':
            return fwdLambda.__code__.co_freevars[0]

    if fwdLambda.__name__ == "<lambda>":
        return "UnknownForward"
    else:
        return fwdLambda.__name__


class Member:
    """A member of a Class object."""

    def __init__(self, t, default_value=None):
        self._type = t
        self._default_value = default_value
        if self._default_value is not None:
            assert isinstance(self._default_value, self._type)

    @property
    def type(self):
        if isinstance(self._type, FunctionType):
            # resolve the function type.
            self._type = self._type()
        return self._type

    def __eq__(self, other):
        if not isinstance(other, Member):
            return False
        return self._type == other._type


class ClassMetaNamespace:
    def __init__(self):
        self.ns = {}
        self.order = []

    def __getitem__(self, k):
        return self.ns[k]

    def __setitem__(self, k, v):
        self.ns[k] = v
        self.order.append((k, v))


def makeFunction(name, f, firstArgType=None):
    spec = inspect.getfullargspec(f)

    def getAnn(argname):
        if argname not in spec.annotations:
            return None
        else:
            ann = spec.annotations.get(argname)
            if ann is None:
                return type(None)
            else:
                return ann

    arg_types = []
    for i, argname in enumerate(spec.args):
        if spec.defaults is not None:
            if i >= len(spec.args) - len(spec.defaults):
                default = (spec.defaults[i-(len(spec.args) - len(spec.defaults))],)
            else:
                default = None
        else:
            default = None

        ann = getAnn(argname)
        if ann is None and i == 0 and firstArgType is not None:
            ann = firstArgType

        arg_types.append((argname, ann, default, False, False))

    return_type = None

    if 'return' in spec.annotations:
        ann = spec.annotations.get('return')
        if ann is None:
            ann = type(None)
        return_type = ann

    if spec.varargs is not None:
        arg_types.append((spec.varargs, getAnn(spec.varargs), None, True, False))

    for arg in spec.kwonlyargs:
        arg_types.append((arg, getAnn(arg), (spec.kwonlydefaults.get(arg),), False, False))

    if spec.varkw is not None:
        arg_types.append((spec.varkw, getAnn(spec.varkw), None, False, True))

    return _types.Function(name, return_type, f, tuple(arg_types))


class ClassMetaclass(type):
    @classmethod
    def __prepare__(cls, *args, **kwargs):
        return ClassMetaNamespace()

    def __new__(cls, name, bases, namespace, **kwds):
        if not bases:
            return type.__new__(cls, name, bases, namespace.ns, **kwds)

        members = []
        memberFunctions = {}
        staticFunctions = {}
        classMembers = []
        properties = {}

        for eltName, elt in namespace.order:
            if isinstance(elt, Member):
                members.append((eltName, elt._type, elt._default_value))
                classMembers.append((eltName, elt))
            elif isinstance(elt, property):
                properties[eltName] = makeFunction(eltName, elt.fget)
            elif isinstance(elt, staticmethod):
                if eltName not in staticFunctions:
                    staticFunctions[eltName] = makeFunction(eltName, elt.__func__)
                else:
                    staticFunctions[eltName] = _types.Function(staticFunctions[eltName], makeFunction(eltName, elt.__func__))
            elif isinstance(elt, FunctionType):
                if eltName not in memberFunctions:
                    memberFunctions[eltName] = makeFunction(eltName, elt, lambda: actualClass)
                else:
                    memberFunctions[eltName] = _types.Function(memberFunctions[eltName], makeFunction(eltName, elt, lambda: actualClass))
            else:
                classMembers.append((eltName, elt))

        actualClass = _types.Class(
            name,
            tuple(members),
            tuple(memberFunctions.items()),
            tuple(staticFunctions.items()),
            tuple(properties.items()),
            tuple(classMembers)
        )
        return actualClass


class Class(metaclass=ClassMetaclass):
    """Base class for all typed python Class objects."""
    pass


def Function(f):
    """Turn a normal python function into a 'typed_python.Function' which obeys type restrictions."""
    return makeFunction(f.__name__, f)()


class FunctionOverloadArg:
    def __init__(self, name, defaultVal, typeFilter, isStarArg, isKwarg):
        self.name = name
        self.defaultValue = defaultVal
        self.typeFilter = typeFilter
        self.isStarArg = isStarArg
        self.isKwarg = isKwarg


class FunctionOverload:
    def __init__(self, functionTypeObject, index, f, returnType, *huh):
        self.functionTypeObject = functionTypeObject
        self.index = index

        self.functionObj = f
        self.returnType = returnType
        self.args = ()

    def addArg(self, name, defaultVal, typeFilter, isStarArg, isKwarg):
        self.args = self.args + (FunctionOverloadArg(name, defaultVal, typeFilter, isStarArg, isKwarg),)

    def matchesTypes(self, argTypes):
        """Do the types in 'argTypes' match our argument typeFilters at a binary level"""
        if len(argTypes) == len(self.args) and not any(x.isStarArg or x.isKwarg for x in self.args):
            for i in range(len(argTypes)):
                if self.args[i].typeFilter is not None and not _types.isBinaryCompatible(self.args[i].typeFilter, argTypes[i]):
                    return False

            return True

        return False

    def __str__(self):
        return "FunctionOverload(%s->%s, %s)" % (self.functionTypeObject, self.returnType, self.args)

    def _installNativePointer(self, fp, returnType, argumentTypes):
        _types.installNativeFunctionPointer(self.functionTypeObject, self.index, fp, returnType, tuple(argumentTypes))


class DisableCompiledCode:
    def __init__(self):
        pass

    def __enter__(self):
        _types.disableNativeDispatch()

    def __exit__(self, *args):
        _types.enableNativeDispatch()


def makeNamedTuple(**kwargs):
    return NamedTuple(**{k: type(v) for k, v in kwargs.items()})(kwargs)
