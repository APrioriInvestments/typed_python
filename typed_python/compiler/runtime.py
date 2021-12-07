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
import time
import types
import typed_python.compiler.python_to_native_converter as python_to_native_converter
import typed_python.compiler.llvm_compiler as llvm_compiler
import typed_python
from typed_python.compiler.runtime_lock import runtimeLock
from typed_python.compiler.conversion_level import ConversionLevel
from typed_python.compiler.compiler_cache import CompilerCache
from typed_python.type_function import TypeFunction
from typed_python.compiler.type_wrappers.typed_tuple_masquerading_as_tuple_wrapper import TypedTupleMasqueradingAsTuple
from typed_python.compiler.type_wrappers.named_tuple_masquerading_as_dict_wrapper import NamedTupleMasqueradingAsDict
from typed_python.compiler.type_wrappers.python_typed_function_wrapper import PythonTypedFunctionWrapper, NoReturnTypeSpecified
from typed_python import Function, _types, Value
from typed_python.compiler.merge_type_wrappers import mergeTypeWrappers

_singleton = [None]
_singletonLock = threading.RLock()

typeWrapper = lambda t: python_to_native_converter.typedPythonTypeToTypeWrapper(t)

_resultTypeCache = {}


class RuntimeEventVisitor:
    """Base class for a Visitor that gets to see what's going on in the runtime.

    Clients should subclass this and pass it to 'addEventVisitor' in the runtime
    to find out about events like function typing assignments.
    """
    def onNewFunction(
        self,
        identifier,
        functionConverter,
        nativeFunction,
        funcName,
        funcCode,
        funcGlobals,
        closureVars,
        inputTypes,
        outputType,
        yieldType,
        variableTypes,
        conversionType
    ):
        pass

    def __enter__(self):
        Runtime.singleton().addEventVisitor(self)
        return self

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
    def __init__(self, short=False):
        self.short = short

    def onNewFunction(
        self,
        identifier,
        functionConverter,
        nativeFunction,
        funcName,
        funcCode,
        funcGlobals,
        closureVars,
        inputTypes,
        outputType,
        yieldType,
        variableTypes,
        conversionType
    ):
        if self.short:
            print(
                f"[complexity={len(str(nativeFunction)):8d}] ",
                funcName,
                "(" + ",".join([str(x.typeRepresentation) for x in inputTypes]) + ")",
                "->",
                outputType,
            )
        else:
            print("compiling ", funcName)
            print("   inputs: ", inputTypes)
            print("   output: ", outputType)
            print("   vars: ")

            for varname, varVal in variableTypes.items():
                if isinstance(varVal, type) and issubclass(varVal, Value):
                    print(
                        "        ",
                        varname,
                        "which is a Value(",
                        varVal.Value,
                        "of type",
                        type(varVal.Value),
                        ")"
                    )
                else:
                    print("        ", varname, varVal)


class CountCompilationsVisitor(RuntimeEventVisitor):
    def __init__(self):
        self.count = 0

    def onNewFunction(
        self,
        identifier,
        functionConverter,
        nativeFunction,
        funcName,
        funcCode,
        funcGlobals,
        closureVars,
        inputTypes,
        outputType,
        yieldType,
        variableTypes,
        conversionType
    ):
        self.count += 1


class Runtime:
    @staticmethod
    def singleton():
        with _singletonLock:
            if _singleton[0] is None:
                _singleton[0] = Runtime()
        return _singleton[0]

    def __init__(self):
        if os.getenv("TP_COMPILER_CACHE"):
            self.compilerCache = CompilerCache(
                os.path.abspath(os.getenv("TP_COMPILER_CACHE"))
            )
        else:
            self.compilerCache = None
        self.llvm_compiler = llvm_compiler.Compiler(inlineThreshold=100)
        self.converter = python_to_native_converter.PythonToNativeConverter(
            self.llvm_compiler,
            self.compilerCache
        )
        self.lock = runtimeLock
        self.timesCompiled = 0

        if os.getenv("TP_COMPILER_VERBOSE"):
            self.verbosityLevel = int(os.getenv("TP_COMPILER_VERBOSE"))
            if self.verbosityLevel >= 2:
                if self.verbosityLevel >= 3:
                    self.addEventVisitor(PrintNewFunctionVisitor(False))
                else:
                    self.addEventVisitor(PrintNewFunctionVisitor(True))
            if self.verbosityLevel >= 4:
                self.llvm_compiler.mark_converter_verbose()
            if self.verbosityLevel >= 5:
                self.llvm_compiler.mark_llvm_codegen_verbose()
        else:
            self.verbosityLevel = 0

    def addEventVisitor(self, visitor: RuntimeEventVisitor):
        self.converter.addVisitor(visitor)

    def removeEventVisitor(self, visitor: RuntimeEventVisitor):
        self.converter.removeVisitor(visitor)

    @staticmethod
    def passingTypeForValue(arg):
        if isinstance(arg, types.FunctionType):
            return type(Function(arg))

        elif isinstance(arg, type) and issubclass(arg, TypeFunction) and len(arg.MRO) == 2:
            return Value(arg)

        elif isinstance(arg, type):
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

        if argType.can_convert_to_type(resType, ConversionLevel.Implicit) is False:
            return None

        if (overloadArg.isStarArg or overloadArg.isKwarg) and resType != argType:
            return None

        return resType

    def compileFunctionOverload(self, functionType, overloadIx, arguments, argumentsAreTypes=False):
        """Attempt to compile typedFunc.overloads[overloadIx]' with the given arguments.

        Args:
            functionType - a typed_python.Function _type_
            overloadIx - an integer giving the index of the overload we're interested in
            arguments - a list of values (or types if 'argumentsAreTypes') for each of the
                named function arguments contained in the overload
            argumentsAreTypes - if true, then we've provided types or Wrapper instances
                instead of actual values.

        Returns:
            None if it is not possible to match this overload with these arguments or
            a TypedCallTarget.
        """
        overload = functionType.overloads[overloadIx]

        assert len(arguments) == len(overload.args)

        try:
            t0 = time.time()
            t1 = None
            t2 = None
            defCount = self.converter.getDefinitionCount()

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

                callTarget = self.converter.convertTypedFunctionCall(
                    functionType,
                    overloadIx,
                    inputWrappers,
                    assertIsRoot=True
                )

                callTarget = self.converter.demasqueradeCallTargetOutput(callTarget)

                assert callTarget is not None

                wrappingCallTargetName = self.converter.generateCallConverter(callTarget)

                t1 = time.time()
                self.converter.buildAndLinkNewModule()
                t2 = time.time()

                fp = self.converter.functionPointerByName(wrappingCallTargetName)

                overload._installNativePointer(
                    fp.fp,
                    callTarget.output_type.typeRepresentation if callTarget.output_type is not None else type(None),
                    [i.typeRepresentation for i in callTarget.input_types]
                )

                return callTarget
        finally:
            if self.verbosityLevel > 0:
                print(
                    f"typed_python runtime spent {time.time()-t0:.3f} seconds "
                    + (f"({t2 - t1:.3f})" if t2 is not None else "")
                    + " adding " +
                    f"{self.converter.getDefinitionCount() - defCount} functions."
                )

    def compileClassDispatch(self, interfaceClass, implementingClass, slotIndex):
        with self.lock:
            self.converter.compileSingleClassDispatch(interfaceClass, implementingClass, slotIndex)

        return True

    def compileClassDestructor(self, cls):
        with self.lock:
            self.converter.compileClassDestructor(cls)

        return True

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

        # walk over the closure of funcObj and figure out the appropriate types
        # of each cell.
        funcObj = _types.prepareArgumentToBePassedToCompiler(funcObj)

        argTypes = [typeWrapper(a) for a in argTypes]
        kwargTypes = {k: typeWrapper(v) for k, v in kwargTypes.items()}

        key = (type(funcObj), tuple(argTypes), tuple(kwargTypes.items()))

        if key in _resultTypeCache:
            return _resultTypeCache[key]

        possibleTypes = []

        for overloadIx in range(len(funcObj.overloads)):
            overload = funcObj.overloads[overloadIx]

            ExpressionConversionContext = typed_python.compiler.expression_conversion_context.ExpressionConversionContext

            argumentSignature = ExpressionConversionContext.computeFunctionArgumentTypeSignature(overload, argTypes, kwargTypes)

            if argumentSignature is not None:
                if funcObj.isNocompile:
                    actualArgTypes, actualKwargTypes = ExpressionConversionContext.computeOverloadSignature(
                        overload,
                        argTypes,
                        kwargTypes
                    )

                    returnType = PythonTypedFunctionWrapper.computeFunctionOverloadReturnType(
                        overload, actualArgTypes, actualKwargTypes
                    )

                    if returnType is not NoReturnTypeSpecified:
                        possibleTypes.append(typeWrapper(returnType))
                    else:
                        possibleTypes.append(typeWrapper(object))
                else:
                    callTarget = self.compileFunctionOverload(funcObj, overloadIx, argumentSignature, argumentsAreTypes=True)

                    if callTarget is not None and callTarget.output_type is not None:
                        possibleTypes.append(callTarget.output_type)

        _resultTypeCache[key] = mergeTypeWrappers(possibleTypes)
        return _resultTypeCache[key]


def NotCompiled(pyFunc, returnTypeOverride=None):
    """Decorate 'pyFunc' to prevent it from being compiled.

    Any type annotations on this function will apply, but the function's body itself
    will stay in the interpreter. This lets us avoid accidentally compiling code that
    we can't understand but still use the functions when we need them (and provide type
    hints).

    The actual python object for the function gets used, so it can have references
    to global state in the program.
    """
    pyFunc = Function(
        pyFunc,
        returnTypeOverride=returnTypeOverride,
    ).withNocompile(True)

    return pyFunc


def Entrypoint(pyFunc):
    """Decorate 'pyFunc' to JIT-compile it based on the signature of the arguments.

    Each time you call 'pyFunc', we look at the argument signature and see whether
    we have already compiled a form of that function. If so, we dispatch to that.
    Otherwise, we compile a new form (which blocks) and then use that when
    compilation has completed.
    """
    Runtime.singleton()

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

    # note that we have to call 'prepareArgumentToBePassedToCompiler' which
    # captures the current closure of 'pyFunc' as it currently stands, since 'Compiled'
    # is supposed to compile the function _as it currently stands_.
    Runtime.singleton()

    entrypoint = Entrypoint(pyFunc)
    f = _types.prepareArgumentToBePassedToCompiler(entrypoint)

    types = []
    for a in f.overloads[0].args:
        if a.typeFilter is None:
            raise Exception(f"@Compiled requires that {pyFunc} has an explicit type annotation for every argument.")
        if a.isStarArg or a.isKwarg:
            raise Exception("@Compiled is not compatible with *args or **kwargs because the signature is not fully known.")
        types.append(a.typeFilter)

    f.resultTypeFor(*types)

    return f
