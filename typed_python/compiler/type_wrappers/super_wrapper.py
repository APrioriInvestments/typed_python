#   Copyright 2017-2019 typed_python Authors
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

import typed_python
import typed_python.compiler.native_ast as native_ast

from typed_python import Class
from typed_python.compiler.type_wrappers.wrapper import Wrapper


typeWrapper = lambda t: typed_python.compiler.python_object_representation.typedPythonTypeToTypeWrapper(t)


_superWrappers = {}
_superBoundMethodWrappers = {}


def superWrapper(heldType, knownType):
    if (heldType, knownType) not in _superWrappers:
        _superWrappers[heldType, knownType] = SuperInstanceWrapper(
            SuperInstance(heldType, knownType)
        )

    return _superWrappers[heldType, knownType]


def superBoundMethodWrapper(heldType, knownType, attribute):
    if (heldType, knownType) not in _superBoundMethodWrappers:
        _superBoundMethodWrappers[heldType, knownType, attribute] = SuperBoundMethodWrapper(
            SuperBoundMethod(heldType, knownType, attribute)
        )

    return _superBoundMethodWrappers[heldType, knownType, attribute]


class SuperInstance:
    def __init__(self, heldType, knownType):
        self.heldType = heldType
        self.knownType = knownType

    def __hash__(self):
        return hash(self.asTup())

    def __eq__(self, other):
        if not isinstance(other, SuperInstance):
            return False

        return self.asTup() == other.asTup()

    def asTup(self):
        return (self.heldType, self.knownType)


class SuperBoundMethod:
    def __init__(self, heldType, knownType, attribute):
        self.heldType = heldType
        self.knownType = knownType
        self.attribute = attribute

    def __hash__(self):
        return hash(self.asTup())

    def __eq__(self, other):
        if not isinstance(other, SuperBoundMethod):
            return False

        return self.asTup() == other.asTup()

    def asTup(self):
        return (self.heldType, self.knownType, self.attribute)


class SuperWrapper(Wrapper):
    is_pod = True
    is_empty = False
    is_pass_by_ref = False

    def __init__(self):
        super().__init__(super)

    def getNativeLayoutType(self):
        return native_ast.Type.Void()

    def convert_call(self, context, expr, args, kwargs):
        firstArg = context.functionContext.funcArgNames[0]
        cls = context.functionContext._globals.get('__class__')

        assert issubclass(cls, typed_python.Class)

        var = context.namedVariableLookup(firstArg)

        # we need to find 'cls' in the MRO of the instance itself
        # and then dispatch to the _next_ type in the list
        mro = var.expr_type.typeRepresentation.MRO

        if cls not in mro:
            context.pushException(
                TypeError,
                f"Instance of type {var.expr_type.typeRepresentation.__name__} is not "
                f"an instance of {cls.__name__}"
            )
        else:
            ix = mro.index(cls) + 1

            if ix >= len(mro):
                nextBase = Class
            else:
                nextBase = mro[ix]

        return var.changeType(superWrapper(var.expr_type.typeRepresentation, nextBase))


class SuperInstanceWrapper(Wrapper):
    is_pod = False
    is_empty = False
    is_pass_by_ref = False

    def __init__(self, T):
        assert isinstance(T, SuperInstance)
        super().__init__(T)

        self.heldWrapper = typeWrapper(T.heldType)
        self.knownAsType = T.knownType

    def __str__(self):
        return "super"

    def getNativeLayoutType(self):
        return self.heldWrapper.getNativeLayoutType()

    def convert_attribute(self, context, instance, attribute):
        if self.knownAsType is not Class and attribute in self.knownAsType.MemberFunctions:
            return instance.changeType(
                superBoundMethodWrapper(self.heldWrapper, self.knownAsType, attribute)
            )

        return super().convert_attribute(context, instance, attribute)


class SuperBoundMethodWrapper(Wrapper):
    is_pod = False
    is_empty = False
    is_pass_by_ref = False

    def __init__(self, T):
        assert isinstance(T, SuperBoundMethod)
        super().__init__(T)

        self.heldWrapper = typeWrapper(T.heldType)
        self.knownAsType = T.knownType
        self.attribute = T.attribute

    def getNativeLayoutType(self):
        return self.heldWrapper.getNativeLayoutType()

    def convert_call(self, context, instance, args, kwargs):
        func = type(getattr(self.knownAsType, self.attribute))

        return typeWrapper(func).convert_call(
            context,
            None,
            (instance.changeType(self.heldWrapper),) + tuple(args),
            kwargs
        )
