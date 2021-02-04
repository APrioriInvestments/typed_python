#   Copyright 2017-2020 typed_python Authors
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

import typed_python.compiler.native_ast as native_ast
from typed_python.compiler.function_metadata import FunctionMetadata
from typed_python.compiler.type_wrappers.wrapper import Wrapper


class TypedCallTarget:
    def __init__(self, named_call_target, input_types, output_type, alwaysRaises=False, functionMetadata=None):
        super().__init__()

        assert isinstance(output_type, Wrapper) or output_type is None
        assert isinstance(named_call_target, native_ast.NamedCallTarget)

        assert named_call_target.name != "throws"

        # if we know _ahead of time_ that this will always throw an exception
        self.alwaysRaises = alwaysRaises

        self.named_call_target = named_call_target
        self.input_types = input_types
        self.output_type = output_type
        self.functionMetadata = functionMetadata or FunctionMetadata()

    def call(self, *args):
        return native_ast.CallTarget.Named(target=self.named_call_target).call(*args)

    @property
    def name(self):
        return self.named_call_target.name

    def __str__(self):
        return "TypedCallTarget(name=%s,inputs=%s,outputs=%s)" % (
            self.name,
            [str(x) for x in self.input_types],
            str(self.output_type)
        )
