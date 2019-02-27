#   Copyright 2017 Braxton Mckee
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

import nativepython.python_to_native_converter as python_to_native_converter
import nativepython.llvm_compiler as llvm_compiler
from typed_python import Function, NoneType
from typed_python.internals import FunctionOverload

_singleton = [None]

typeWrapper = lambda t: python_to_native_converter.typedPythonTypeToTypeWrapper(t)


class Runtime:
    @staticmethod
    def singleton():
        if _singleton[0] is None:
            _singleton[0] = Runtime()
        return _singleton[0]

    def __init__(self):
        self.llvm_compiler = llvm_compiler.Compiler()
        self.converter = python_to_native_converter.PythonToNativeConverter()

    def verboselyDisplayNativeCode(self):
        self.llvm_compiler.mark_converter_verbose()
        self.llvm_compiler.mark_llvm_codegen_verbose()

    def compile(self, f, argument_types=None):
        """Compile a single FunctionOverload and install the pointer

        If provided, 'argument_types' can be a dictionary from variable name to a type.
        this will take precedence over any types specified on the function. Keep in mind
        that function overloads already filter by type, so if you specify a type that's
        not compatible with the type argument of the existing overload, the resulting
        specialization will never be called.
        """
        argument_types = argument_types or {}

        if isinstance(f, FunctionOverload):
            for a in f.args:
                assert not a.isStarArg, 'dont support star args yet'
                assert not a.isKwarg, 'dont support keyword yet'

            def chooseTypeFilter(a):
                return argument_types.pop(a.name, a.typeFilter or object)

            input_wrappers = [typeWrapper(chooseTypeFilter(a)) for a in f.args]

            if len(argument_types):
                raise Exception("No argument exists for type overrides %s" % argument_types)

            callTarget = self.converter.convert(f.functionObj, input_wrappers, f.returnType, assertIsRoot=True)

            assert callTarget is not None

            wrappingCallTargetName = self.converter.generateCallConverter(callTarget)

            targets = self.converter.extract_new_function_definitions()

            function_pointers = self.llvm_compiler.add_functions(targets)

            fp = function_pointers[wrappingCallTargetName]

            f._installNativePointer(
                fp.fp,
                callTarget.output_type.typeRepresentation if callTarget.output_type is not None else NoneType,
                [i.typeRepresentation for i in input_wrappers]
            )

            return targets

        if hasattr(f, '__typed_python_category__') and f.__typed_python_category__ == 'Function':
            for o in f.overloads:
                self.compile(o, argument_types)
            return f

        if hasattr(f, '__typed_python_category__') and f.__typed_python_category__ == 'BoundMethod':
            for o in f.Function.overloads:
                arg_types = dict(argument_types)
                arg_types[o.args[0].name] = typeWrapper(f.Class)
                self.compile(o, arg_types)
            return f

        if callable(f):
            result = Function(f)
            self.compile(result, argument_types)
            return result

        assert False, f


def Entrypoint(f):
    """Indicate that a function is a natural entrypoint into compiled code.

    By default, this means that if we hit an entrypoint, we'll attempt to compile
    the code when we first execute the function.

    However, the runtime can be placed into other modes, where we precompile
    all entrypoints.
    """
    if not hasattr(f, '__typed_python_category__'):
        if not callable(f):
            raise Exception("Can only compile functions.")
        f = Function(f)

    compiled = [None]

    def inner(*args, **kwargs):
        if compiled[0] is None:
            compiled[0] = Runtime.singleton().compile(f)
        return compiled[0](*args, **kwargs)

    inner.__qualname__ = str(f)

    return inner
