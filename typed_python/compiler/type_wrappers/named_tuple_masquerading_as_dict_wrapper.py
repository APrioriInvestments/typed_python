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
