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
import typed_python
from typed_python.type_function import ConcreteTypeFunction
from typed_python.compiler.type_wrappers.one_of_wrapper import OneOfWrapper
from typed_python.compiler.type_wrappers.typed_tuple_masquerading_as_tuple_wrapper import TypedTupleMasqueradingAsTuple
from typed_python.compiler.type_wrappers.named_tuple_masquerading_as_dict_wrapper import NamedTupleMasqueradingAsDict
from typed_python.compiler.type_wrappers.python_typed_function_wrapper import PythonTypedFunctionWrapper
from typed_python import Function, NoneType, _types, Value

_singleton = [None]
_singletonLock = threading.Lock()

typeWrapper = lambda t: python_to_native_converter.typedPythonTypeToTypeWrapper(t)


def toInterpreterType(setOfTypes):
    res = OneOfWrapper.mergeTypes(setOfTypes)
    if res is None:
        return None

    res = res.interpreterTypeRepresentation

    if issubclass(res, typed_python.PythonObjectOfType):
        return res.PyType

    return res


class RuntimeEventVisitor:
    """Base class for a Visitor that gets to see what's going on in the runtime.

    Clients should subclass this and pass it to 'addEventVisitor' in the runtime
    to find out about events like function typing assignments.
    """
    def onNewFunction(self, function, inputTypes, outputType, variableTypes):
        pass

    def __enter__(self):
        Runtime.singleton().addEventVisitor(self)

    def __exit__(self, *args):
        Runtime.singleton().removeEventVisitor(self)


class PrintNewFunctionVisitor(RuntimeEventVisitor):
    """A simple visitor that prints out all the functions that get compiled.

    Usage:
        @Entrypoint
        def f():
            return 0

        with PrintNewFunctionVisitor():
            # this should print out the fact that we compiled 'f'
            f()
    """
    def onNewFunction(self, f, inputTypes, outputType, variables):
        print("compiling ", f)
        print("   inputs: ", inputTypes)
        print("   output: ", outputType)
        print("   vars: ")
        for varname, varVal in variables.items():
            if isinstance(varVal, type) and issubclass(varVal, Value):
                print("        ", varname, " which is a Value(", varVal.Value, " of type ", type(varVal.Value), ")")
            else:
                print("        ", varname, varVal)


class Runtime:
    @staticmethod
    def singleton():
        with _singletonLock:
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

    def addEventVisitor(self, visitor: RuntimeEventVisitor):
        self.converter.addVisitor(visitor)

    def removeEventVisitor(self, visitor: RuntimeEventVisitor):
        self.converter.removeVisitor(visitor)

    def _collectLinktimeHooks(self):
        while True:
            identityAndCallback = self.converter.popLinktimeHook()
            if identityAndCallback is None:
                return

            identity, callback = identityAndCallback

            name = self.converter.identityToName(identity)

            fp = self.llvm_compiler.function_pointer_by_name(name)

            callback(fp)

    @staticmethod
    def passingTypeForValue(arg):
        if isinstance(arg, types.FunctionType):
            return type(Function(arg))

        elif isinstance(arg, type):
            return Value(arg)

        elif isinstance(arg, ConcreteTypeFunction):
            return Value(arg)

        return type(arg)

    @staticmethod
    def pickSpecializationTypeFor(overloadArg, argValue, argumentsAreTypes=False):
        """Compute the typeWrapper we'll use for this particular argument based on 'argValue'.

        Args:
            overloadArg - the internals.FunctionOverloadArg instance representing this argument.
                This tells us whether we're dealing with a normal positional/keyword argument or
                a *arg / **kwarg, where the typeFilter applies to the items of the tuple but
                not the tuple itself.
            argValue - the value being passed for this argument. If 'argumentsAreTypes' is true,
                then this is the actual type, not the value.

        Returns:
            the Wrapper or type instance to use for this argument.
        """
        if not argumentsAreTypes:
            if overloadArg.isStarArg:
                argType = TypedTupleMasqueradingAsTuple(
                    typed_python.Tuple(*[Runtime.passingTypeForValue(v) for v in argValue])
                )
            elif overloadArg.isKwarg:
                argType = NamedTupleMasqueradingAsDict(
                    typed_python.NamedTuple(
                        **{k: Runtime.passingTypeForValue(v) for k, v in argValue.items()}
                    )
                )
            else:
                argType = typeWrapper(Runtime.passingTypeForValue(argValue))
        else:
            argType = typeWrapper(argValue)

        resType = PythonTypedFunctionWrapper.pickSpecializationTypeFor(overloadArg, argType)

        if argType.can_convert_to_type(resType, True) is False:
            return None

        if (overloadArg.isStarArg or overloadArg.isKwarg) and resType != argType:
            return None

        return resType

    def compileFunctionOverload(self, typedFunc, overloadIx, arguments, argumentsAreTypes=False):
        """Attempt to compile typedFunc.overloads[overloadIx]' with the given arguments.

        Args:
            typedFunc - a typed_python.Function instance
            overloadIx - an integer giving the index of the overload we're interested in
            arguments - a list of values (or types if 'argumentsAreTypes') for each of the
                named function arguments contained in the overload
            argumentsAreTypes - if true, then we've provided types or Wrapper instances
                instead of actual values.

        Returns:
            None if it is not possible to match this overload with these arguments or
            a TypedCallTarget.
        """

        overload = typedFunc.overloads[overloadIx]

        assert len(arguments) == len(overload.args)

        with self.lock:
            inputWrappers = []

            for i in range(len(arguments)):
                inputWrappers.append(
                    self.pickSpecializationTypeFor(overload.args[i], arguments[i], argumentsAreTypes)
                )

            if any(x is None for x in inputWrappers):
                # this signature is unmatchable with these arguments.
                return None

            self.timesCompiled += 1

            callTarget = self.converter.convert(overload.functionObj, inputWrappers, overload.returnType, assertIsRoot=True)

            callTarget = self.converter.demasqueradeCallTargetOutput(callTarget)

            assert callTarget is not None

            wrappingCallTargetName = self.converter.generateCallConverter(callTarget)

            targets = self.converter.extract_new_function_definitions()

            self.llvm_compiler.add_functions(targets)

            # if the callTargetName isn't in the list, then we already compiled it and installed it.
            fp = self.llvm_compiler.function_pointer_by_name(wrappingCallTargetName)

            overload._installNativePointer(
                fp.fp,
                callTarget.output_type.typeRepresentation if callTarget.output_type is not None else NoneType,
                [i.typeRepresentation for i in inputWrappers]
            )

            self._collectLinktimeHooks()

            return callTarget

    def resultTypeForCall(self, funcObj, argTypes, kwargTypes):
        """Determine the result of calling funcObj with things of type 'argTypes' and 'kwargTypes'

        Args:
            funcObj - a typed_python.Function object.
            argTypes - a list of Type or Wrapper objects
            kwargs - a dict of keyword Type or Wrapper objects

        Returns:
            None if control flow doesn't return a result (say, because it always
            throws an exception) or a Wrapper object describing the return types.
        """
        assert isinstance(funcObj, typed_python._types.Function)

        argTypes = [typeWrapper(a) for a in argTypes]
        kwargTypes = {k: typeWrapper(v) for k, v in kwargTypes.items()}

        possibleTypes = []

        for overloadIx in range(len(funcObj.overloads)):
            overload = funcObj.overloads[overloadIx]

            ExpressionConversionContext = typed_python.compiler.expression_conversion_context.ExpressionConversionContext

            argumentSignature = ExpressionConversionContext.computeFunctionArgumentTypeSignature(overload, argTypes, kwargTypes)

            if argumentSignature is not None:
                callTarget = self.compileFunctionOverload(funcObj, overloadIx, argumentSignature, argumentsAreTypes=True)

                if callTarget is not None and callTarget.output_type is not None:
                    possibleTypes.append(callTarget.output_type)

        return OneOfWrapper.mergeTypes(possibleTypes)


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

    typedFunc = pyFunc
    if not isinstance(typedFunc, _types.Function):
        if isinstance(typedFunc, staticmethod):
            typedFunc = typedFunc.__func__
            wrapInStatic = True

        if not callable(typedFunc):
            raise Exception(f"Can only compile functions, not {typedFunc}")

        typedFunc = Function(typedFunc)

    typedFunc = typedFunc.withEntrypoint(True)

    if wrapInStatic:
        return staticmethod(typedFunc)

    return typedFunc


def Compiled(pyFunc):
    """Compile a pyFunc, which must have a type annotation for all arguments"""
    f = Entrypoint(pyFunc)
    types = []
    for a in f.overloads[0].args:
        if a.typeFilter is None:
            raise Exception(f"@Compiled requires that {pyFunc} has an explicit type annotation for every argument.")
        if a.isStarArg or a.isKwarg:
            raise Exception(f"@Compiled is not compatible with *args or **kwargs because the signature is not fully known.")
        types.append(a.typeFilter)

    f.resultTypeFor(*types)

    return f
