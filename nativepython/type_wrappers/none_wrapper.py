#   Copyright 2018 Braxton Mckee
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
from typed_python import NoneType
import nativepython.native_ast as native_ast

class NoneWrapper(Wrapper):
    is_pod = True
    is_empty = True
    is_pass_by_ref = False

    def __init__(self):
        super().__init__(NoneType())

    def getNativeLayoutType(self):
        return native_ast.Type.Void()

    def convert_assign(self, context, target, toStore):
        return context.NoneExpr()

    def convert_copy_initialize(self, context, target, toStore):
        return context.NoneExpr()

    def convert_destroy(self, context, instance):
        return context.NoneExpr()