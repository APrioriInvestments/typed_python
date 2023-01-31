#   Copyright 2017-2023 typed_python Authors
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

from typed_python.compiler.type_wrappers.wrapper import Wrapper
import typed_python.compiler.native_ast as native_ast
import typed_python.compiler.type_wrappers.runtime_functions as runtime_functions
from typed_python import Type, SerializationContext
from typed_python.compiler.conversion_level import ConversionLevel
from typed_python._types import serialize


class SerializeWrapper(Wrapper):
    is_pod = True
    is_empty = False
    is_pass_by_ref = False

    def __init__(self):
        super().__init__(serialize)

    def getNativeLayoutType(self):
        return native_ast.Type.Void()

    def convert_call(self, context, expr, args, kwargs):
        if len(args) != 3:
            return super().convert_call(context, expr, args, kwargs)

        # check if the first argument is an actual known type
        if not args[0].expr_type.is_py_type_object_wrapper:
            return context.constant(serialize, allowArbitrary=True).convert_call(args, kwargs)

        T = args[0].expr_type.typeRepresentation.Value

        if not issubclass(T, (Type, str, int, float, bytes, bool)):
            return context.constant(serialize, allowArbitrary=True).convert_call(args, kwargs)

        asT = args[1].convert_to_type(T, ConversionLevel.New)

        if asT is None:
            return

        outBytesRef = context.allocateUninitializedSlot(bytes)

        isNone = args[2].convert_to_type_with_target(
            context.push(type(None), lambda n: None),
            ConversionLevel.Signature
        )

        serContext = context.allocateUninitializedSlot(SerializationContext)

        isSerializationContext = args[2].convert_to_type_with_target(
            serContext,
            ConversionLevel.Signature
        )

        with context.ifelse(isNone.nonref_expr) as (ifTrue, ifFalse):
            with ifTrue:
                context.pushEffect(
                    outBytesRef.expr.store(
                        runtime_functions.serialize_no_context.call(
                            asT.expr.cast(native_ast.VoidPtr),
                            context.getTypePointer(T).cast(native_ast.VoidPtr)
                        ).cast(outBytesRef.expr_type.layoutType)
                    )
                )

            with ifFalse:
                with context.ifelse(isSerializationContext.nonref_expr) as (ifTrueSc, ifFalseSc):
                    with ifTrueSc:
                        context.markUninitializedSlotInitialized(serContext)

                        context.pushEffect(
                            outBytesRef.expr.store(
                                runtime_functions.serialize.call(
                                    asT.expr.cast(native_ast.VoidPtr),
                                    context.getTypePointer(T).cast(native_ast.VoidPtr),
                                    context.getTypePointer(SerializationContext).cast(native_ast.VoidPtr),
                                    serContext.expr.cast(native_ast.VoidPtr)
                                ).cast(outBytesRef.expr_type.layoutType)
                            )
                        )

                    with ifFalseSc:
                        context.pushException(TypeError, "Expected a SerializationContext")

        return outBytesRef
