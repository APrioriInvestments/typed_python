#   Copyright 2020 typed_python Authors
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

from typed_python.compiler.type_wrappers.compilable_builtin import CompilableBuiltin


class CompilerType(CompilableBuiltin):
    """A builtin for determing the type of a variable as known to the compiler."""
    def __eq__(self, other):
        return isinstance(other, CompilerType)

    def __hash__(self):
        return hash("CompilerType")

    def __call__(self, x):
        return "<interpreted>"

    def convert_call(self, context, expr, args, kwargs):
        if len(args) != 1 or len(kwargs):
            context.pushException(TypeError, "CompilerType takes exactly one unnamed argument")
            return

        return context.constant(str(args[0].expr_type))


typeAsKnownToCompiler = CompilerType()
