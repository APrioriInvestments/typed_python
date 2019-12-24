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

from typed_python.compiler.type_wrappers.masquerade_wrapper import MasqueradeWrapper

typeWrapper = lambda t: typed_python.compiler.python_object_representation.typedPythonTypeToTypeWrapper(t)


class TypedTupleMasqueradingAsTuple(MasqueradeWrapper):
    """Models a 'NamedTuple' that's masquerading as a 'dict' for use in **kwargs."""

    def __init__(self, typeRepresentation):
        super().__init__(typeRepresentation)

    @property
    def interpreterTypeRepresentation(self):
        return tuple

    def convert_mutable_masquerade_to_untyped_type(self):
        return typeWrapper(tuple)

    def convert_mutable_masquerade_to_untyped(self, context, instance):
        return self.convert_masquerade_to_untyped(context, instance)

    def convert_masquerade_to_untyped(self, context, instance):
        return context.constant(tuple).convert_call([instance], {}).changeType(tuple)

    def convert_bool_cast(self, context, expr):
        return expr.convert_masquerade_to_typed().convert_bool_cast()

    def convert_int_cast(self, context, expr):
        return expr.convert_masquerade_to_typed().convert_int_cast()

    def convert_float_cast(self, context, expr):
        return expr.convert_masquerade_to_typed().convert_float_cast()

    def convert_getitem(self, context, instance, item):
        return instance.convert_masquerade_to_typed().convert_getitem(item)
