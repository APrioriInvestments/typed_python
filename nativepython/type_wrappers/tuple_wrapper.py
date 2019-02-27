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

from nativepython.type_wrappers.wrapper import Wrapper

from typed_python import _types

import nativepython.native_ast as native_ast
import nativepython

typeWrapper = lambda t: nativepython.python_object_representation.typedPythonTypeToTypeWrapper(t)


class TupleWrapper(Wrapper):
    is_empty = False
    is_pass_by_ref = True

    def __init__(self, t):
        assert hasattr(t, '__typed_python_category__')
        super().__init__(t)
        bytecount = _types.bytecount(t)

        self.subTypeWrappers = tuple(typeWrapper(sub_t) for sub_t in t.ElementTypes)
        self.byteOffsets = [0]

        for i in range(len(self.subTypeWrappers)-1):
            self.byteOffsets.append(self.byteOffsets[-1] + _types.bytecount(t.ElementTypes[i]))

        self.layoutType = native_ast.Type.Array(element_type=native_ast.UInt8, count=bytecount)

        self._is_pod = all(typeWrapper(possibility).is_pod for possibility in self.subTypeWrappers)
        self.is_default_constructible = _types.is_default_constructible(t)

    @property
    def is_pod(self):
        return self._is_pod

    def getNativeLayoutType(self):
        return self.layoutType

    def convert_initialize_from_args(self, context, target, *args):
        assert len(args) == len(self.byteOffsets)
        for i in range(len(args)):
            self.refAs(context, target, i).convert_copy_initialize(args[i])

    def convert_default_initialize(self, context, target):
        if not self.is_default_constructible:
            context.pushException(TypeError, "Can't default-initialize any subtypes of %s" % self.typeRepresentation.__qualname__)
            return

        for i, t in enumerate(self.typeRepresentation.ElementTypes):
            if _types.is_default_constructible(t):
                self.refAs(context, target, i).convert_default_initialize()

    def refAs(self, context, expr, which):
        return context.pushReference(
            self.subTypeWrappers[which],
            expr.expr.cast(native_ast.UInt8Ptr)
                .ElementPtrIntegers(self.byteOffsets[which])
                .cast(self.subTypeWrappers[which].getNativeLayoutType().pointer())
        )

    def convert_assign(self, context, expr, other):
        assert expr.isReference

        for i in range(len(self.subTypeWrappers)):
            self.refAs(context, expr, i).convert_assign(self.refAs(context, other, i))

    def convert_copy_initialize(self, context, expr, other):
        for i in range(len(self.subTypeWrappers)):
            self.refAs(context, expr, i).convert_copy_initialize(self.refAs(context, other, i))

    def convert_destroy(self, context, expr):
        if not self.is_pod:
            for i in range(len(self.subTypeWrappers)):
                self.refAs(context, expr, i).convert_destroy()


class NamedTupleWrapper(TupleWrapper):
    def __init__(self, t):
        super().__init__(t)

        self.namesToIndices = {n: i for i, n in enumerate(t.ElementNames)}
        self.namesToTypes = {n: t.ElementTypes[i] for i, n in enumerate(t.ElementNames)}

    def convert_attribute(self, context, instance, attribute):
        ix = self.namesToIndices.get(attribute)
        if ix is None:
            context.pushException(AttributeError, "'%s' object has no attribute '%s'" % (str(self.typeRepresentation), attribute))
            return

        return self.refAs(context, instance, ix)
