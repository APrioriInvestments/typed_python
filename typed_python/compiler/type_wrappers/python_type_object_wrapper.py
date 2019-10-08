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
from typed_python import String, Int64, Bool, Float64, Type, PythonObjectOfType
from typed_python.compiler.type_wrappers.wrapper import Wrapper
from typed_python.compiler.type_wrappers.python_free_object_wrapper import PythonFreeObjectWrapper

typeWrapper = lambda t: typed_python.compiler.python_object_representation.typedPythonTypeToTypeWrapper(t)


class PythonTypeObjectWrapper(PythonFreeObjectWrapper):
    def __repr__(self):
        return "Wrapper(TypeObject(%s))" % self.typeRepresentation.Value.__qualname__

    def __str__(self):
        return "TypeObject(%s)" % self.typeRepresentation.Value.__qualname__

    def convert_cast(self, context, instance, targetType):
        if targetType == typeWrapper(str):
            typ = self.typedPythonTypeToRegularType(self.typeRepresentation.Value)
            return context.constant(str(typ))

        return super().convert_call(context, instance, targetType)

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
        elif typeRep == Bool:
            return bool
        elif typeRep == String:
            return str
        elif issubclass(typeRep, PythonObjectOfType):
            return typeRep.PyType
        return typeRep

    @Wrapper.unwrapOneOfAndValue
    def convert_call(self, context, left, args, kwargs):
        if self.typeRepresentation.Value is type:
            if len(args) != 1 or kwargs:
                return super().convert_call(context, left, args, kwargs)

            argtype = args[0].expr_type

            if isinstance(argtype, PythonTypeObjectWrapper):
                res = typed_python.compiler.python_object_representation.pythonObjectRepresentation(
                    context,
                    type
                )
            else:
                typeRep = self.typedPythonTypeToRegularType(argtype.typeRepresentation)

                res = typed_python.compiler.python_object_representation.pythonObjectRepresentation(
                    context,
                    typeRep
                )
            return res

        if Type in self.typeRepresentation.Value.__bases__:
            # this is one of the type factories (ListOf, Dict, etc.)
            return super().convert_call(context, left, args, kwargs)

        return typeWrapper(self.typeRepresentation.Value).convert_type_call(context, left, args, kwargs)
