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

from typed_python import Class
import typed_python.compiler
import typed_python.compiler.native_ast as native_ast
from typed_python.compiler.type_wrappers.wrapper import Wrapper
import typed_python.compiler.type_wrappers.runtime_functions as runtime_functions


typeWrapper = lambda t: typed_python.compiler.python_object_representation.typedPythonTypeToTypeWrapper(t)


class SubclassOfWrapper(Wrapper):
    """Models an object held as a Type* implementing SubclassOf"""
    is_pod = True
    is_pass_by_ref = False

    def getNativeLayoutType(self):
        return native_ast.VoidPtr

    def convert_default_initialize(self, context, target):
        context.pushEffect(
            target.expr.store(context.getTypePointer(self.typeRepresentation.Type))
        )

    def convert_assign(self, context, target, toStore):
        assert target.isReference
        context.pushEffect(
            target.expr.store(toStore.nonref_expr)
        )

    def convert_copy_initialize(self, context, target, toStore):
        assert target.isReference
        context.pushEffect(
            target.expr.store(toStore.nonref_expr)
        )

    def convert_destroy(self, context, instance):
        pass

    def convert_issubclass(self, context, instance, ofType, isSubclassCall):
        classCouldBeInstanceOf = (
            typed_python.compiler.type_wrappers.class_wrapper.classCouldBeInstanceOf
        )

        PythonTypeObjectWrapper = (
            typed_python.compiler.type_wrappers.python_type_object_wrapper.PythonTypeObjectWrapper
        )

        # if both of us are Type* represented
        if isinstance(ofType.expr_type, SubclassOfWrapper):
            couldBe = classCouldBeInstanceOf(
                self.typeRepresentation.Type,
                ofType.expr_type.typeRepresentation.Type
            )

            # because the other type could be _more_ specific than the second
            # arg, the only information we can infer is whether its impossible
            # to match.
            if couldBe is False:
                return context.constant(couldBe)

            return context.pushPod(
                bool,
                runtime_functions.typePtrIsSubclass.call(instance.nonref_expr, ofType.nonref_expr)
            )

        # if we're a Type* and the other is an actual known type
        if isinstance(ofType.expr_type, PythonTypeObjectWrapper):
            # this is a crappy hack that relies on the fact that SubclassOf only accepts
            # subclasses of Class.
            if issubclass(self.typeRepresentation.Type, Class) and not issubclass(ofType.expr_type.typeRepresentation.Value, Class):
                return context.constant(False)

            couldBe = classCouldBeInstanceOf(
                self.typeRepresentation.Type,
                ofType.expr_type.typeRepresentation.Value
            )

            if couldBe != "Maybe":
                return context.constant(couldBe)

            if issubclass(ofType.expr_type.typeRepresentation.Value, Class) and ofType.expr_type.typeRepresentation.Value.IsFinal:
                return context.pushPod(
                    bool,
                    instance.nonref_expr.cast(native_ast.UInt64).eq(
                        context.getTypePointer(ofType.expr_type.typeRepresentation.Value).cast(native_ast.UInt64)
                    )
                )

            return context.pushPod(
                bool,
                runtime_functions.typePtrIsSubclass.call(
                    instance.nonref_expr,
                    context.getTypePointer(ofType.expr_type.typeRepresentation.Value)
                )
            )

        return super().convert_issubclass(context, instance, ofType, isSubclassCall)

    def convert_typeof(self, context, instance):
        pythonObjectRepresentation = (
            typed_python.compiler.python_object_representation.pythonObjectRepresentation
        )

        return pythonObjectRepresentation(context, type)

    def convert_attribute(self, context, instance, attribute):
        if hasattr(self.typeRepresentation.Type, attribute):
            assert False

        return instance.toPyObj().convert_attribute(attribute)

    def convert_method_call(self, context, instance, methodname, args, kwargs):
        return instance.toPyObj().convert_method_call(methodname, args, kwargs)
