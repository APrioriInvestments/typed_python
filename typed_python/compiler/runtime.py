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

import threading
import os
import types
import typed_python.compiler.python_to_native_converter as python_to_native_converter
import typed_python.compiler.llvm_compiler as llvm_compiler
from typed_python import Function, NoneType, OneOf, _types, Value
from typed_python.internals import FunctionOverload, DisableCompiledCode

_singleton = [None]

typeWrapper = lambda t: python_to_native_converter.typedPythonTypeToTypeWrapper(t)


def toSingleType(setOfTypes):
    if not setOfTypes:
        return None

    if len(setOfTypes) == 1:
        return list(setOfTypes)[0]

    return OneOf(*setOfTypes)


class Runtime:
    @staticmethod
    def singleton():
        if _singleton[0] is None:
            _singleton[0] = Runtime()

        if os.getenv("TP_COMPILER_VERBOSE"):
            _singleton[0].verboselyDisplayNativeCode()

        return _singleton[0]

    def __init__(self):
        self.llvm_compiler = llvm_compiler.Compiler()
        self.converter = python_to_native_converter.PythonToNativeConverter()
        self.lock = threading.RLock()
        self.timesCompiled = 0

    def verboselyDisplayNativeCode(self):
        self.llvm_compiler.mark_converter_verbose()
        self.llvm_compiler.mark_llvm_codegen_verbose()

    def resultTypes(self, f, argument_types):
        with self.lock:
            argument_types = dict(argument_types or {})

            if isinstance(f, FunctionOverload):
                for a in f.args:
                    assert not a.isStarArg, 'dont support star args yet'
                    assert not a.isKwarg, 'dont support keyword yet'

                def chooseTypeFilter(a):
                    return argument_types.pop(a.name, a.typeFilter or object)

                input_wrappers = [typeWrapper(chooseTypeFilter(a)) for a in f.args]

                if len(argument_types):
                    raise Exception("No argument exists for type overrides %s" % argument_types)

                self.timesCompiled += 1

                callTarget = self.converter.convert(f.functionObj, input_wrappers, f.returnType, assertIsRoot=True)

                assert callTarget is not None

                wrappingCallTargetName = self.converter.generateCallConverter(callTarget)

                targets = self.converter.extract_new_function_definitions()

                self.llvm_compiler.add_functions(targets)

                # if the callTargetName isn't in the list, then we already compiled it and installed it.
                fp = self.llvm_compiler.function_pointer_by_name(wrappingCallTargetName)

                f._installNativePointer(
                    fp.fp,
                    callTarget.output_type.typeRepresentation if callTarget.output_type is not None else NoneType,
                    [i.typeRepresentation for i in input_wrappers]
                )

                self._collectLinktimeHooks()

                return toSingleType(set([callTarget.output_type.typeRepresentation] if callTarget.output_type is not None else []))

            if hasattr(f, '__typed_python_category__') and f.__typed_python_category__ == 'Function':
                results = set()

                for o in f.overloads:
                    results.add(self.resultTypes(o, argument_types))

                return toSingleType(results)

            if hasattr(f, '__typed_python_category__') and f.__typed_python_category__ == 'BoundMethod':
                results = set()

                for o in f.Function.overloads:
                    arg_types = dict(argument_types)
                    arg_types[o.args[0].name] = typeWrapper(f.Class)
                    results.add(self.resultTypes(o, arg_types))

                return results

            if callable(f):
                result = Function(f)
                return self.resultTypes(result, argument_types)

            assert False, f

    def _collectLinktimeHooks(self):
        while True:
            identityAndCallback = self.converter.popLinktimeHook()
            if identityAndCallback is None:
                return

            identity, callback = identityAndCallback

            name = self.converter.identityToName(identity)

            fp = self.llvm_compiler.function_pointer_by_name(name)

            callback(fp)

    def compile(self, f, argument_types=None):
        """Compile a single FunctionOverload and install the pointer

        If provided, 'argument_types' can be a dictionary from variable name to a type.
        this will take precedence over any types specified on the function. Keep in mind
        that function overloads already filter by type, so if you specify a type that's
        not compatible with the type argument of the existing overload, the resulting
        specialization will never be called.
        """
        with self.lock:
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

                self.timesCompiled += 1

                callTarget = self.converter.convert(f.functionObj, input_wrappers, f.returnType, assertIsRoot=True)

                assert callTarget is not None

                wrappingCallTargetName = self.converter.generateCallConverter(callTarget)

                targets = self.converter.extract_new_function_definitions()

                self.llvm_compiler.add_functions(targets)

                # if the callTargetName isn't in the list, then we already compiled it and installed it.
                fp = self.llvm_compiler.function_pointer_by_name(wrappingCallTargetName)

                f._installNativePointer(
                    fp.fp,
                    callTarget.output_type.typeRepresentation if callTarget.output_type is not None else NoneType,
                    [i.typeRepresentation for i in input_wrappers]
                )

                self._collectLinktimeHooks()

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


def pickSpecializationTypeFor(x):
    if isinstance(x, types.FunctionType):
        return Function(x)

    if isinstance(x, type):
        return Value(x)

    return type(x)


def pickSpecializationValueFor(x):
    if isinstance(x, types.FunctionType):
        return Function(x)

    if isinstance(x, type):
        return x

    return x


def NotCompiled(pyFunc):
    """Decorate 'pyFunc' to prevent it from being compiled.

    Any type annotations on this function will apply, but the function's body itself
    will stay in the interpreter. This lets us avoid accidentally compiling code that
    we can't understand but still use the functions when we need them (and provide type
    hints).

    The actual python object for the function gets used, so it can have references
    to global state in the program.
    """
    assert isinstance(pyFunc, types.FunctionType), "Can't apply NotCompiled to anything but functions."

    pyFunc.__typed_python_no_compile__ = True
    return Function(pyFunc)


def Entrypoint(pyFunc):
    """Decorate 'pyFunc' to JIT-compile it based on the signature of the arguments.

    Each time you call 'pyFunc', we look at the argument signature and see whether
    we have already compiled a form of that function. If so, we dispatch to that.
    Otherwise, we compile a new form (which blocks) and then use that when
    compilation has completed.
    """
    wrapInStatic = False
    if not isinstance(pyFunc, _types.Function):
        if isinstance(pyFunc, staticmethod):
            pyFunc = pyFunc.__func__
            wrapInStatic = True

        if not callable(pyFunc):
            raise Exception(f"Can only compile functions, not {pyFunc}")

        pyFunc = Function(pyFunc)

    lock = threading.RLock()
    signatures = set()

    def inner(*args):
        signature = tuple(pickSpecializationTypeFor(x) for x in args)
        args = tuple(pickSpecializationValueFor(x) for x in args)

        if True or not DisableCompiledCode.isDisabled():
            with lock:
                if signature not in signatures:
                    i = pyFunc.indexOfOverloadMatching(*args)

                    if i is not None:
                        o = pyFunc.overloads[i]
                        argTypes = {o.args[i].name: o.args[i].typeToUse(signature[i]) for i in range(len(args))}
                        Runtime.singleton().compile(o, argTypes)

                    signatures.add(signature)

        return pyFunc(*args)

    inner.__qualname__ = str(pyFunc)
    inner.__typed_python_function__ = pyFunc

    def resultTypeFor(*args):
        signature = tuple(pickSpecializationTypeFor(x) for x in args)
        args = tuple(pickSpecializationValueFor(x) for x in args)

        i = pyFunc.indexOfOverloadMatching(*args)

        if i is not None:
            o = pyFunc.overloads[i]

            argTypes = {o.args[i].name: o.args[i].typeToUse(signature[i]) for i in range(len(args))}
            return Runtime.singleton().resultTypes(o, argTypes)

        raise Exception("No compilable dispatch found for these arguments.")

    inner.resultTypeFor = resultTypeFor

    if wrapInStatic:
        inner = staticmethod(inner)

    return inner
