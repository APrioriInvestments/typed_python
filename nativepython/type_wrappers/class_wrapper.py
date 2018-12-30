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

from nativepython.type_wrappers.refcounted_wrapper import RefcountedWrapper
from nativepython.typed_expression import TypedExpression
from nativepython.type_wrappers.exceptions import generateThrowException
import nativepython.type_wrappers.runtime_functions as runtime_functions

from typed_python import NoneType, Int64, _types

import nativepython.native_ast as native_ast
import nativepython


typeWrapper = lambda x: nativepython.python_object_representation.typedPythonTypeToTypeWrapper(x)

        
class ClassWrapper(RefcountedWrapper):
    is_pod = False
    is_empty = False
    is_pass_by_ref = True

    def __init__(self, t):
        super().__init__(t)

        self.nameToIndex = {}
        self.indexToByteOffset = {}

        element_types = [('refcount', native_ast.Int64), ('data',native_ast.UInt8)]
        
        #this follows the general layout of 'held class' which is 1 bit per field for initialization and then
        #each field packed directly according to byte size
        byteOffset = 8 + (len(t.MemberNames) // 8 + 1)

        for i,name in enumerate(t.MemberNames):
            self.nameToIndex[name] = i
            self.indexToByteOffset[i] = byteOffset

            byteOffset += _types.bytecount(t.MemberTypes[i])

        self.layoutType = native_ast.Type.Struct(element_types=element_types,name=t.__qualname__+"Layout").pointer()

    def getNativeLayoutType(self):
        return self.layoutType

    def on_refcount_zero(self, context, instance):
        expr = native_ast.callFree(instance.expr)

        for i in range(len(self.typeRepresentation.MemberTypes)):
            if not typeWrapper(self.typeRepresentation.MemberTypes[i]).is_pod:
                expr = context.decref(self.convert_attribute(context, instance, i)).expr >> expr

        return expr

    def convert_bin_op(self, context, left, op, right):
        raise ConvsersionException("Not convertible")

    def convert_attribute(self, context, instance, attribute):
        if not isinstance(attribute, int):
            ix = self.nameToIndex.get(attribute)
        else:
            ix = attribute

        if ix is None:
            return TypedExpression(
                generateThrowException(context, AttributeError("Attribute %s doesn't exist in %s" % (attribute, self.typeRepresentation))),
                None,
                False
                )

        instance = instance.ensureNonReference()

        return context.incref(
            TypedExpression(
                instance.expr.cast(native_ast.UInt8.pointer()).ElementPtrIntegers(self.indexToByteOffset[ix])
                    .cast(
                        typeWrapper(self.typeRepresentation.MemberTypes[ix]).getNativeLayoutType().pointer()
                        ).load(),
                typeWrapper(self.typeRepresentation.MemberTypes[ix]),
                False
                )
            )


