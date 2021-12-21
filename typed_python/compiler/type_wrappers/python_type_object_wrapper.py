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

import typed_python.compiler
from typed_python import Type, TupleOf
from typed_python.compiler.conversion_level import ConversionLevel
from typed_python.compiler.type_wrappers.wrapper import Wrapper
from typed_python.compiler.type_wrappers.python_free_object_wrapper import PythonFreeObjectWrapper
from typed_python.compiler.type_wrappers.compilable_builtin import CompilableBuiltin
from typed_python.compiler.type_wrappers.typed_tuple_masquerading_as_tuple_wrapper import TypedTupleMasqueradingAsTuple
from typed_python.compiler.type_wrappers.subclass_of_wrapper import SubclassOfWrapper

typeWrapper = lambda t: typed_python.compiler.python_object_representation.typedPythonTypeToTypeWrapper(t)


class PythonTypeObjectWrapper(PythonFreeObjectWrapper):
    is_py_type_object_wrapper = True

    def __init__(self, f):
        super().__init__(f, hasSideEffects=False)

    def __repr__(self):
        return "Wrapper(%s)" % self.typeRepresentation.Value.__qualname__

    def __str__(self):
        return self.typeRepresentation.Value.__qualname__

    def _can_convert_to_type(self, otherType, conversionLevel):
        if isinstance(otherType, SubclassOfWrapper):
            canConvert = otherType.typeRepresentation.Type in self.typeRepresentation.Value.MRO
            return canConvert

        return super()._can_convert_to_type(otherType, conversionLevel)

    def convert_to_type_with_target(self, context, instance, targetVal, conversionLevel, mayThrowOnFailure=False):
        canConvert = self._can_convert_to_type(targetVal.expr_type, conversionLevel)

        if canConvert is True and isinstance(targetVal.expr_type, SubclassOfWrapper):
            context.pushEffect(
                targetVal.expr.store(
                    context.getTypePointer(self.typeRepresentation.Value)
                )
            )
            return context.constant(True)

        return super().convert_to_type_with_target(context, instance, targetVal, conversionLevel, mayThrowOnFailure)

    @Wrapper.unwrapOneOfAndValue
    def convert_call_on_container_expression(self, context, inst, argExpr):
        if issubclass(self.typeRepresentation.Value, CompilableBuiltin):
            return self.typeRepresentation.Value.convert_type_call_on_container_expression(context, inst, argExpr)

        if self.typeRepresentation.Value in (tuple, list) and (argExpr.matches.ListComp or argExpr.matches.GeneratorExp):
            compResult = context.convert_generator_as_list_comprehension(argExpr)

            if compResult is None:
                return compResult

            if self.typeRepresentation.Value is list:
                return compResult
            else:
                return compResult.changeType(
                    TypedTupleMasqueradingAsTuple(
                        TupleOf(compResult.expr_type.typeRepresentation.ElementType)
                    )
                )

        return typeWrapper(self.typeRepresentation.Value).convert_type_call_on_container_expression(context, inst, argExpr)

    def convert_attribute(self, context, instance, attribute, allowDefer=True):
        if allowDefer:
            return typeWrapper(self.typeRepresentation.Value).convert_type_attribute(context, instance, attribute)
        else:
            return super().convert_attribute(context, instance, attribute)

    def convert_method_call(self, context, instance, methodname, args, kwargs, allowDefer=True):
        if allowDefer:
            return typeWrapper(self.typeRepresentation.Value).convert_type_method_call(context, instance, methodname, args, kwargs)
        else:
            return super().convert_method_call(context, instance, methodname, args, kwargs)

    @Wrapper.unwrapOneOfAndValue
    def convert_call(self, context, left, args, kwargs):
        if self.typeRepresentation.Value is type:
            if len(args) != 1 or kwargs:
                return super().convert_call(context, left, args, kwargs)

            return args[0].convert_typeof()

        if len(args) == 1 and not kwargs:
            if self.typeRepresentation.Value in (bool, int, float, str, bytes):
                return args[0].convert_to_type(self.typeRepresentation.Value, ConversionLevel.New)

        if Type in self.typeRepresentation.Value.__bases__:
            # this is one of the type factories (ListOf, Dict, etc.)
            return super().convert_call(context, left, args, kwargs)

        if issubclass(self.typeRepresentation.Value, CompilableBuiltin):
            return self.typeRepresentation.Value.convert_type_call(context, left, args, kwargs)

        return typeWrapper(self.typeRepresentation.Value).convert_type_call(context, left, args, kwargs)

    def convert_typeof(self, context, instance):
        pythonObjectRepresentation = (
            typed_python.compiler.python_object_representation.pythonObjectRepresentation
        )

        return pythonObjectRepresentation(context, type)

    def convert_issubclass(self, context, instance, ofType, isSubclassCall):
        if isinstance(ofType.expr_type, PythonTypeObjectWrapper):
            return context.constant(
                issubclass(
                    instance.expr_type.typeRepresentation.Value,
                    ofType.expr_type.typeRepresentation.Value
                )
            )

        if isinstance(ofType.expr_type, SubclassOfWrapper):
            return context.constant(
                issubclass(
                    instance.expr_type.typeRepresentation.Value,
                    ofType.expr_type.typeRepresentation.Type
                )
            )

        return super().convert_issubclass(context, instance, ofType, isSubclassCall)
