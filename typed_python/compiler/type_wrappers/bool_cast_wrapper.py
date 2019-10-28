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
from typed_python.compiler.type_wrappers.wrapper import Wrapper
from typed_python.compiler.type_wrappers.python_free_object_wrapper import PythonFreeObjectWrapper

typeWrapper = lambda t: typed_python.compiler.python_object_representation.typedPythonTypeToTypeWrapper(t)


class BoolCastWrapper(PythonFreeObjectWrapper):
    def __init__(self):
        super().__init__(bool)

    def __repr__(self):
        return "Wrapper(TypeObject(%s))" % self.typeRepresentation.Value.__qualname__

    def __str__(self):
        return "TypeObject(%s)" % self.typeRepresentation.Value.__qualname__

    def convert_str_cast(self, context, instance):
        return context.constant(str(bool))

    @Wrapper.unwrapOneOfAndValue
    def convert_call(self, context, left, args, kwargs):
        return args[0].convert_bool_cast()
