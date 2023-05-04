#   Copyright 2017-2023 typed_python Authors
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


import typed_python.compiler.native_compiler.native_ast as native_ast


class TypedLLVMValue:
    def __init__(self, llvm_value, native_type):
        object.__init__(self)

        assert isinstance(native_type, native_ast.Type)

        if native_type.matches.Void:
            assert llvm_value is None
        else:
            assert llvm_value is not None

        self.llvm_value = llvm_value
        self.native_type = native_type

    def __str__(self):
        return "TypedLLVMValue(%s)" % self.native_type

    def __repr__(self):
        return str(self)
