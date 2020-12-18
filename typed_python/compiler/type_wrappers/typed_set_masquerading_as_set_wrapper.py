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

from typed_python.compiler.type_wrappers.masquerade_wrapper import MasqueradeWrapper


class TypedSetMasqueradingAsSet(MasqueradeWrapper):
    def __init__(self, typeRepresentation):
        super().__init__(typeRepresentation)

    @property
    def interpreterTypeRepresentation(self):
        return set

    def convert_masquerade_to_untyped(self, context, instance):
        return context.constant(set).convert_call(
            [instance.convert_masquerade_to_typed()],
            {}
        ).changeType(set)
