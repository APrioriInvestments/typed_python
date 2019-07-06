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

import nativepython
from typed_python import String, Int64, Bool, Float64
from nativepython.type_wrappers.python_free_object_wrapper import PythonFreeObjectWrapper

typeWrapper = lambda t: nativepython.python_object_representation.typedPythonTypeToTypeWrapper(t)


class PythonTypeObjectWrapper(PythonFreeObjectWrapper):
    def __repr__(self):
        return "Wrapper(TypeObject(%s))" % self.typeRepresentation.__qualname__

    def __str__(self):
        return "TypeObject(%s)" % self.typeRepresentation.__qualname__

    def convert_call(self, context, left, args, kwargs):
        if self.typeRepresentation is type:
            if len(args) != 1 or kwargs:
                return super().convert_call(context, left, args, kwargs)

            argtype = args[0].expr_type
            if isinstance(argtype, PythonTypeObjectWrapper):
                res = nativepython.python_object_representation.pythonObjectRepresentation(
                    context,
                    type
                )
            else:
                typeRep = argtype.typeRepresentation

                # internally, we track int, bool, float, and str as Int64, Bool, Float64, etc.
                # but that's now how python programs would see them. So, we have to convert
                # to the python object representation of those objects.
                if typeRep == Int64:
                    typeRep = int
                if typeRep == Float64:
                    typeRep = float
                elif typeRep == Bool:
                    typeRep = bool
                elif typeRep == String:
                    typeRep = str

                res = nativepython.python_object_representation.pythonObjectRepresentation(
                    context,
                    typeRep
                )
            return res

        return typeWrapper(self.typeRepresentation).convert_type_call(context, left, args, kwargs)
