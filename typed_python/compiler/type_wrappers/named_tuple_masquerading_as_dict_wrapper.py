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
from typed_python import Tuple
from typed_python.compiler.type_wrappers.typed_tuple_masquerading_as_tuple_wrapper import (
    TypedTupleMasqueradingAsTuple
)
from typed_python.compiler.type_wrappers.bound_method_wrapper import BoundMethodWrapper
from typed_python.compiler.type_wrappers.masquerade_wrapper import MasqueradeWrapper

typeWrapper = lambda t: typed_python.compiler.python_object_representation.typedPythonTypeToTypeWrapper(t)


class NamedTupleMasqueradingAsDict(MasqueradeWrapper):
    """Models a 'NamedTuple' that's masquerading as a 'dict' for use in **kwargs."""

    def __init__(self, typeRepresentation):
        super().__init__(typeRepresentation)

    @property
    def interpreterTypeRepresentation(self):
        return dict

    def convert_getitem(self, context, instance, key):
        if key.constantValue is not None and isinstance(key.constantValue, str):
            return self.convert_attribute(context, instance, key.constantValue)

        return self.convert_masquerade_to_untyped(context, instance).convert_getitem(key)

    def convert_masquerade_to_untyped(self, context, instance):
        emptyDict = context.constant(dict).convert_call([], {})

        typedForm = instance.convert_masquerade_to_typed()

        for ix, name in enumerate(self.typeRepresentation.ElementNames):
            emptyDict.convert_setitem(
                context.constant(name),
                typedForm.convert_attribute(name)
            )

        return emptyDict

    def get_iteration_expressions(self, context, instance):
        return [context.constant(name) for name in self.typeRepresentation.ElementNames]

    def convert_attribute(self, context, instance, attribute):
        if attribute == "items":
            return instance.changeType(BoundMethodWrapper.Make(self, attribute))

        return super().convert_attribute(context, instance, attribute)

    def convert_method_call(self, context, instance, methodname, args, kwargs):
        if methodname == "items" and len(args) == 0 and len(kwargs) == 0:
            return instance.changeType(
                ItemsOfNamedTupleMasqueradingAsDict(self.typeRepresentation)
            )

        return super().convert_method_call(context, instance, methodname, args, kwargs)


class ItemsOfNamedTupleMasqueradingAsDict(MasqueradeWrapper):
    def get_iteration_expressions(self, context, instance):
        res = []

        for ix in range(len(self.typeRepresentation.ElementNames)):
            name = self.typeRepresentation.ElementNames[ix]
            T = Tuple(str, self.typeRepresentation.ElementTypes[ix])

            res.append(
                typeWrapper(T).createFromArgs(
                    context,
                    (
                        context.constant(name),
                        instance.changeType(
                            self.typeRepresentation
                        ).convert_attribute(name)
                    ),
                ).changeType(
                    TypedTupleMasqueradingAsTuple(T)
                )
            )

        return res
