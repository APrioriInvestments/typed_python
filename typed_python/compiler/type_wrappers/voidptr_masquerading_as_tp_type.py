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

from typed_python.compiler.type_wrappers.masquerade_wrapper import MasqueradeWrapper
from typed_python import PointerTo
import typed_python.compiler.type_wrappers.runtime_functions as runtime_functions
import typed_python.compiler.native_ast as native_ast
from typed_python.compiler.conversion_level import ConversionLevel


class VoidPtrMasqueradingAsTPType(MasqueradeWrapper):
    is_pod = True
    is_pass_by_ref = False

    DEFAULT_IS_TYPED = False

    def __init__(self, baseType):
        super().__init__(PointerTo(None))

        # our pointers are all subclasses of this
        self.baseType = baseType

    @property
    def interpreterTypeRepresentation(self):
        return type

    def convert_masquerade_to_untyped(self, context, instance):
        return context.push(
            object,
            lambda targetSlot: targetSlot.expr.store(
                runtime_functions.convertTypePtrToTypeObj.call(
                    instance.nonref_expr.cast(native_ast.VoidPtr)
                ).cast(targetSlot.expr_type.getNativeLayoutType())
            )
        )

    def convert_issubclass(self, context, typeInstance, instance, isSubclassCall):
        if instance.expr_type.can_convert_to_type(type, ConversionLevel.Signature) is False:
            # its definitely not a type. throw an exception
            return super().convert_issubclass(context, typeInstance, instance, isSubclassCall)

        if not isinstance(instance.expr_type, VoidPtrMasqueradingAsTPType):
            return typeInstance.convert_to_type(object, ConversionLevel.Signature).convert_issubclass(
                instance, isSubclassCall
            )

        return context.pushPod(
            bool,
            runtime_functions.typePtrIsSubclass.call(instance.nonref_expr, typeInstance.nonref_expr)
        )
