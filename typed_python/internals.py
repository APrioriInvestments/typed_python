#   Coyright 2017-2019 Nativepython Authors
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

import typed_python
import typed_python.inspect_override as inspect


class UndefinedBehaviorException(BaseException):
    """An unsafe operation with known undefined behavior was performed.

    This Exception is deliberately not a subclass of Exception because by
    default it should not be caught by normal exception handlers. In compiled
    code, the operation that raised this exception is likely to segfault.
    """


# needed by the C api
object = object
type = type


class Member:
    """A member of a Class object."""

    def __init__(self, t, default_value=None):
        self._type = t
        self._default_value = default_value
        if self._default_value is not None:
            assert isinstance(self._default_value, self._type)

    @property
    def type(self):
        if getattr(self._type, '__typed_python_category__', None) == "Forward":
            return self._type.get()
        return self._type

    def __eq__(self, other):
        if not isinstance(other, Member):
            return False
        return self.type == other.type


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
        """ Return the annotated type for the given argument or None. """
        if argname not in spec.annotations:
            return None
        else:
            ann = spec.annotations.get(argname)
            if ann is None:
                return type(None)
            else:
                return ann

    def getDefault(idx: int):
        """ Return the default value for a positional argument given its index. """
        if spec.defaults is not None:
            if idx >= len(spec.args) - len(spec.defaults):
                default = (spec.defaults[idx - (len(spec.args) - len(spec.defaults))],)
            else:
                default = None
        else:
            default = None

        return default

    arg_types = []
    for i, argname in enumerate(spec.args):
        default = getDefault(i)

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

    return typed_python._types.Function(name, return_type, f, tuple(arg_types))


class ClassMetaclass(type):
    @classmethod
    def __prepare__(cls, *args, **kwargs):
        return ClassMetaNamespace()

    def __new__(cls, name, bases, namespace, **kwds):
        if not bases:
            return type.__new__(cls, name, bases, namespace.ns, **kwds)

        members = []
        bases = [x for x in bases if x is not typed_python._types.Class]
        memberFunctions = {}
        staticFunctions = {}
        classMembers = []
        properties = {}

        actualClass = typed_python._types.Forward(name)

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
                    staticFunctions[eltName] = typed_python._types.Function(
                        staticFunctions[eltName],
                        makeFunction(eltName, elt.__func__)
                    )
            elif isinstance(elt, FunctionType):
                if eltName not in memberFunctions:
                    memberFunctions[eltName] = makeFunction(eltName, elt, actualClass)
                else:
                    memberFunctions[eltName] = typed_python._types.Function(
                        memberFunctions[eltName],
                        makeFunction(eltName, elt, actualClass)
                    )
            else:
                classMembers.append((eltName, elt))

        actualClass = actualClass.define(typed_python._types.Class(
            name,
            tuple(bases),
            tuple(members),
            tuple(memberFunctions.items()),
            tuple(staticFunctions.items()),
            tuple(properties.items()),
            tuple(classMembers)
        ))

        return actualClass


def Function(f):
    """Turn a normal python function into a 'typed_python.Function' which obeys type restrictions."""
    return makeFunction(f.__name__, f)()


class FunctionOverloadArg:
    def __init__(self, name, defaultVal, typeFilter, isStarArg, isKwarg):
        self.name = name
        self.defaultValue = defaultVal
        self._typeFilter = typeFilter
        self.isStarArg = isStarArg
        self.isKwarg = isKwarg

    @property
    def typeFilter(self):
        if getattr(self._typeFilter, '__typed_python_category__', None) == "Forward":
            return self._typeFilter.get()
        return self._typeFilter

    def __repr__(self):
        res = f"{self.name}: {self.typeFilter}"
        if self.defaultValue is not None:
            res += " = " + str(self.defaultValue)
        if self.isKwarg:
            res = "**" + res
        if self.isStarArg:
            res = "*" + res

        return res


class FunctionOverload:
    def __init__(self, functionTypeObject, index, f, returnType):
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
                if (self.args[i].typeFilter is not None and
                        not typed_python._types.isBinaryCompatible(self.args[i].typeFilter, argTypes[i])):
                    return False

            return True

        return False

    def __str__(self):
        return "FunctionOverload(returns %s, %s, %s)" % (
            self.returnType,
            self.args,
            "<signature>" if self.functionObj is None else "<impl>"
        )

    def _installNativePointer(self, fp, returnType, argumentTypes):
        typed_python._types.installNativeFunctionPointer(self.functionTypeObject, self.index, fp, returnType, tuple(argumentTypes))


class DisableCompiledCode:
    def __init__(self):
        pass

    def __enter__(self):
        typed_python._types.disableNativeDispatch()

    def __exit__(self, *args):
        typed_python._types.enableNativeDispatch()


def makeNamedTuple(**kwargs):
    return typed_python._types.NamedTuple(**{k: type(v) for k, v in kwargs.items()})(kwargs)
