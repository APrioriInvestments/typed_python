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

import typed_python

from typed_python import NamedTuple

from typed_python.compiler.type_wrappers.masquerade_wrapper import MasqueradeWrapper

typeWrapper = lambda t: typed_python.compiler.python_object_representation.typedPythonTypeToTypeWrapper(t)


def slice_repr(sliceObj):
    return f"slice({repr(sliceObj.start)}, {repr(sliceObj.stop)}, {repr(sliceObj.step)})"


class NamedTupleMasqueradingAsSlice(MasqueradeWrapper):
    """Models a 'NamedTuple' that's masquerading as a 'slice'

    It must have 'start', 'stop', and 'step' arguments.
    """

    def __init__(self, typeRepresentation):
        assert issubclass(typeRepresentation, NamedTuple)
        assert typeRepresentation.ElementNames == ('start', 'stop', 'step'), typeRepresentation.ElementNames
        super().__init__(typeRepresentation)

    @property
    def interpreterTypeRepresentation(self):
        return slice

    def convert_masquerade_to_untyped(self, context, instance):
        return context.constant(slice).convert_to_type(object).convert_call(
            [instance.convert_attribute("start"), instance.convert_attribute("stop"), instance.convert_attribute("step")],
            {}
        )

    def can_cast_to_primitive(self, context, instance, primitiveType):
        return instance.convert_masquerade_to_typed().can_cast_to_primitive(primitiveType)

    def convert_bool_cast(self, context, instance):
        return context.constant(True)

    def convert_int_cast(self, context, instance):
        return instance.convert_masquerade_to_typed().convert_int_cast()

    def convert_float_cast(self, context, instance):
        return instance.convert_masquerade_to_typed().convert_float_cast()

    def convert_str_cast(self, context, instance):
        return context.call_py_function(slice_repr, (instance,), {})

    def convert_repr(self, context, instance):
        return context.call_py_function(slice_repr, (instance,), {})
