import types

import typed_python
from typed_python.compiler.python_object_representation import typedPythonTypeToTypeWrapper
from typed_python.compiler.type_wrappers.python_typed_function_wrapper import (
    PythonTypedFunctionWrapper,
    CannotBeDetermined,
    NoReturnTypeSpecified,
)
from typed_python.compiler.type_wrappers.typed_tuple_masquerading_as_tuple_wrapper import (
    TypedTupleMasqueradingAsTuple,
)
from typed_python.compiler.type_wrappers.named_tuple_masquerading_as_dict_wrapper import (
    NamedTupleMasqueradingAsDict,
)
from typed_python.compiler.conversion_level import ConversionLevel
from typed_python import Function, Value
from typed_python.type_function import TypeFunction

import typed_python.compiler.python_to_native_converter as python_to_native_converter

typeWrapper = lambda t: python_to_native_converter.typedPythonTypeToTypeWrapper(t)


class CompilerInput:
    """
    Represents a parcel of input code and its input types + closure, containing everything the
    compiler needs in order to do the compilation. Typed_python supports function overloading,
    so everything here is specific to a given 'overload' - a function for a given set of input
    types, and inferred output type.
    """
    def __init__(self, overload, closure_type, input_wrappers):
        self._overload = overload
        self.closure_type = closure_type
        self._input_wrappers = input_wrappers

        # cached properties
        self._realized_input_wrappers = None
        self._return_type = None
        self._return_type_calculated = False

    @property
    def realized_input_wrappers(self):
        if self._realized_input_wrappers is None:
            self._realized_input_wrappers = self._compute_realized_input_wrappers()
            assert self._realized_input_wrappers is not None

        return self._realized_input_wrappers

    @property
    def return_type(self):
        if not self._return_type_calculated:
            self._return_type = self._compute_return_type()
            self._return_type_calculated = True
        return self._return_type

    @property
    def args(self):
        return self._overload.args

    @property
    def name(self):
        return self._overload.name

    @property
    def functionCode(self):
        return self._overload.functionCode

    @property
    def realizedGlobals(self):
        return self._overload.realizedGlobals

    @property
    def functionGlobals(self):
        return self._overload.functionGlobals

    @property
    def funcGlobalsInCells(self):
        return self._overload.funcGlobalsInCells

    @property
    def closureVarLookups(self):
        return self._overload.closureVarLookups

    def _compute_realized_input_wrappers(self) -> None:
        """
        Extend the list of wrappers (representing the input args) using the free variables in
        the function closure.
        """
        res = []
        for closure_var_path in self.closureVarLookups.values():
            res.append(
                typedPythonTypeToTypeWrapper(
                    PythonTypedFunctionWrapper.closurePathToCellType(closure_var_path, self.closure_type)
                )
            )
        res.extend(self._input_wrappers)
        return res

    def _compute_return_type(self) -> None:
        """Determine the return type, if possible."""
        res = PythonTypedFunctionWrapper.computeFunctionOverloadReturnType(
            self._overload, self._input_wrappers, {}
        )

        if res is CannotBeDetermined:
            res = object

        elif res is NoReturnTypeSpecified:
            res = None

        return res

    def install_native_pointer(self, fp, returnType, argumentTypes) -> None:
        return self._overload._installNativePointer(fp, returnType, argumentTypes)

    @staticmethod
    def make(functionType, overloadIx, arguments, argumentsAreTypes):
        overload = functionType.overloads[overloadIx]

        if len(arguments) != len(overload.args):
            raise Exception(
                "CompilerInput mismatch: overload has %s args, but we were given "
                "%s arguments" % (len(overload.args), len(arguments))
            )

        inputWrappers = []

        for i in range(len(arguments)):
            specialization = pickSpecializationTypeFor(
                overload.args[i], arguments[i], argumentsAreTypes
            )

            if specialization is None:
                return None

            inputWrappers.append(specialization)

        return CompilerInput(overload, functionType.ClosureType, inputWrappers)


def pickSpecializationTypeFor(overloadArg, argValue, argumentsAreTypes):
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
                typed_python.Tuple(*[passingTypeForValue(v) for v in argValue])
            )
        elif overloadArg.isKwarg:
            argType = NamedTupleMasqueradingAsDict(
                typed_python.NamedTuple(
                    **{k: passingTypeForValue(v) for k, v in argValue.items()}
                )
            )
        else:
            argType = typeWrapper(passingTypeForValue(argValue))
    else:
        argType = typeWrapper(argValue)

    resType = PythonTypedFunctionWrapper.pickSpecializationTypeFor(overloadArg, argType)

    if argType.can_convert_to_type(resType, ConversionLevel.Implicit) is False:
        return None

    if (overloadArg.isStarArg or overloadArg.isKwarg) and resType != argType:
        return None

    return resType


def passingTypeForValue(arg):
    if isinstance(arg, types.FunctionType):
        return type(Function(arg))

    elif isinstance(arg, type) and issubclass(arg, TypeFunction) and len(arg.MRO) == 2:
        return Value(arg)

    elif isinstance(arg, type):
        return Value(arg)

    return type(arg)
