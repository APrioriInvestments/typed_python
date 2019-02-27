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
import nativepython.native_ast as native_ast


class PythonTypedFunctionWrapper(Wrapper):
    is_pod = True
    is_empty = True
    is_pass_by_ref = False

    def __init__(self, f):
        super().__init__(f)

    def getNativeLayoutType(self):
        return native_ast.Type.Void()

    def convert_call(self, context, left, args, kwargs):
        for a in list(args) + list(kwargs.items()):
            if not hasattr(a.expr_type.typeRepresentation, '__typed_python_category__'):
                # we don't know how to push around non-typed-python argument types yet. Eventually we should
                # defer to the interpreter in these cases.
                context.pushException(TypeError, "Can't pass arguments of type %s yet" % a.typeRepresentation)
                return

        if kwargs:
            raise NotImplementedError("can't dispatch to native code with kwargs yet as our matcher doesn't understand it")

        f = self.typeRepresentation

        for overload in f.overloads:
            if overload.matchesTypes([a.expr_type.typeRepresentation for a in args]):
                return context.call_py_function(
                    overload.functionObj,
                    args,
                    kwargs,
                    overload.returnType
                )

        context.pushException(TypeError, "No overload for %s with args of type (%s)" % (
            self.typeRepresentation.__qualname__,
            ",".join([str(x.expr_type) for x in args])
        ))
