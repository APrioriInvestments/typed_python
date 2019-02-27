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

import nativepython
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
                res = nativepython.python_object_representation.pythonObjectRepresentation(
                    context,
                    argtype.typeRepresentation
                )
            return res

        return typeWrapper(self.typeRepresentation).convert_type_call(context, left, args, kwargs)
