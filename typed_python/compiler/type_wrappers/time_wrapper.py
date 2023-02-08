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
import time


from typed_python.compiler.type_wrappers.wrapper import Wrapper
import typed_python.compiler.native_ast as native_ast
from typed_python.compiler.type_wrappers.runtime_functions import secondsSinceEpoch


class TimeWrapper(Wrapper):
    is_pod = True
    is_empty = False
    is_pass_by_ref = False

    def __init__(self):
        super().__init__(time.time)

    def getNativeLayoutType(self):
        return native_ast.Type.Struct()

    def convert_call(self, context, expr, args, kwargs):
        return context.pushPod(
            float,
            secondsSinceEpoch.call()
        )
