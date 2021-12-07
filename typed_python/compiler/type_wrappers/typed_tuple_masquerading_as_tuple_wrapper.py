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

    def __init__(self, typeRepresentation, interiorTypeWrappers=None):
        super().__init__(typeRepresentation)
        self.interiorTypeWrappers = tuple(interiorTypeWrappers) if interiorTypeWrappers is not None else None

    def __hash__(self):
        return hash(
            (type(self), self.typeRepresentation, self.interiorTypeWrappers)
        )

    def __eq__(self, other):
        if type(self) != type(other):
            return False

        return (
            self.typeRepresentation == other.typeRepresentation
            and self.interiorTypeWrappers == other.interiorTypeWrappers
        )

    @property
    def interpreterTypeRepresentation(self):
        return tuple

    def convert_getitem(self, context, instance, index):
        actualRes = super().convert_getitem(context, instance, index)

        if actualRes is None:
            return None

        if (
            index.isConstant
            and isinstance(index.constantValue, int)
            and self.interiorTypeWrappers is not None
        ):
            return actualRes.changeType(
                self.interiorTypeWrappers[index.constantValue]
            )

        return actualRes

    def convert_masquerade_to_untyped(self, context, instance):
        return context.constant(tuple).convert_call(
            [instance.convert_masquerade_to_typed()],
            {}
        ).changeType(tuple)

    def refAs(self, context, instance, which):
        return self.convert_masquerade_to_typed(context, instance).refAs(which)
