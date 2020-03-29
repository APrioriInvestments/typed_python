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
from typed_python import String, Int64, Bool, NoneType, Float64, Type, PythonObjectOfType
from typed_python.compiler.type_wrappers.wrapper import Wrapper
from typed_python.compiler.type_wrappers.python_free_object_wrapper import PythonFreeObjectWrapper
from typed_python.compiler.type_wrappers.compilable_builtin import CompilableBuiltin

typeWrapper = lambda t: typed_python.compiler.python_object_representation.typedPythonTypeToTypeWrapper(t)


class PythonTypeObjectWrapper(PythonFreeObjectWrapper):
    def __init__(self, f):
        super().__init__(f, hasSideEffects=False)

    def __repr__(self):
        return "Wrapper(TypeObject(%s))" % self.typeRepresentation.Value.__qualname__

    def __str__(self):
        return "TypeObject(%s)" % self.typeRepresentation.Value.__qualname__

    def convert_str_cast(self, context, instance):
        return context.constant(str(self.typeRepresentation.Value))

    @staticmethod
    def typedPythonTypeToRegularType(typeRep):
        """Unwrap a typed_python type back to the normal python representation.

        For things where typed_python has its own internal representation,
        like Int64 <-> int, we convert back to normal python values.
        """
        # internally, we track int, bool, float, and str as Int64, Bool, Float64, etc.
        # but that's now how python programs would see them. So, we have to convert
        # to the python object representation of those objects.
        if typeRep == Int64:
            return int
        if typeRep == Float64:
            return float
        if typeRep == Bool:
            return bool
        if typeRep == String:
            return str
        if typeRep == NoneType:
            return type(None)
        if isinstance(typeRep, type) and issubclass(typeRep, PythonObjectOfType):
            return typeRep.PyType

        return typeRep

    @Wrapper.unwrapOneOfAndValue
    def convert_call_on_container_expression(self, context, inst, argExpr):
        if issubclass(self.typeRepresentation.Value, CompilableBuiltin):
            return self.typeRepresentation.Value.convert_type_call_on_container_expression(context, inst, argExpr)

        return typeWrapper(self.typeRepresentation.Value).convert_type_call_on_container_expression(context, inst, argExpr)

    @Wrapper.unwrapOneOfAndValue
    def convert_call(self, context, left, args, kwargs):
        if self.typeRepresentation.Value is type:
            # make sure we don't have any masquerades in here
            if len(args) != 1 or kwargs:
                return super().convert_call(context, left, args, kwargs)

            argtype = args[0].expr_type

            if isinstance(argtype, PythonTypeObjectWrapper):
                res = typed_python.compiler.python_object_representation.pythonObjectRepresentation(
                    context,
                    type
                )
            else:
                typeRep = self.typedPythonTypeToRegularType(argtype.interpreterTypeRepresentation)

                res = typed_python.compiler.python_object_representation.pythonObjectRepresentation(
                    context,
                    typeRep
                )

            return res

        if len(args) == 1 and not kwargs:
            if self.typeRepresentation.Value is bool:
                return args[0].convert_bool_cast()
            if self.typeRepresentation.Value is int:
                return args[0].convert_int_cast()
            if self.typeRepresentation.Value is float:
                return args[0].convert_float_cast()
            if self.typeRepresentation.Value is str:
                return args[0].convert_str_cast()
            if self.typeRepresentation.Value is bytes:
                return args[0].convert_bytes_cast()

        if Type in self.typeRepresentation.Value.__bases__:
            # this is one of the type factories (ListOf, Dict, etc.)
            return super().convert_call(context, left, args, kwargs)

        if issubclass(self.typeRepresentation.Value, CompilableBuiltin):
            return self.typeRepresentation.Value.convert_type_call(context, left, args, kwargs)

        return typeWrapper(self.typeRepresentation.Value).convert_type_call(context, left, args, kwargs)
