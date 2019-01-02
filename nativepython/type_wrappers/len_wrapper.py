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

class LenWrapper(Wrapper):
    is_pod = True
    is_empty = False
    is_pass_by_ref = True

    def __init__(self):
        super().__init__(len)

    def getNativeLayoutType(self):
        return native_ast.Type.Void()

    def convert_call(self, context, expr, args):
        if len(args) == 1:
            return args[0].convert_len()

        assert False, "we should be raising a python exception here but we dont know how yet"

