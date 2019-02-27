#   Copyright 2019 Nativepython Authors
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

from nativepython.type_wrappers.wrapper import Wrapper

import nativepython

typeWrapper = lambda x: nativepython.python_object_representation.typedPythonTypeToTypeWrapper(x)


class BoundCompiledMethodWrapper(Wrapper):
    def __init__(self, wrapped_type, method_name):
        super().__init__((wrapped_type, method_name))
        self.wrapped_type = typeWrapper(wrapped_type)
        self.method_name = method_name

    def convert_assign(self, context, target, toStore):
        return self.wrapped_type.convert_assign(
            context,
            target.changeType(self.wrapped_type),
            toStore.changeType(self.wrapped_type)
        )

    def convert_copy_initialize(self, context, target, toStore):
        return self.wrapped_type.convert_copy_initialize(
            context,
            target.changeType(self.wrapped_type),
            toStore.changeType(self.wrapped_type)
        )

    def convert_destroy(self, context, instance):
        return self.wrapped_type.convert_destroy(
            context,
            target.changeType(self.wrapped_type),
            toStore.changeType(self.wrapped_type)
        )

    def convert_call(self, context, left, args, kwargs):
        return self.wrapped_type.convert_method_call(
            context,
            left.changeType(self.wrapped_type),
            self.method_name,
            args,
            kwargs
        )
