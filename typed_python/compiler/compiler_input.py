from typing import List
from typed_python.compiler.python_object_representation import typedPythonTypeToTypeWrapper
from typed_python.compiler.type_wrappers.python_typed_function_wrapper import (
    PythonTypedFunctionWrapper, CannotBeDetermined, NoReturnTypeSpecified
)
from typed_python.internals import FunctionOverload


class CompilerInput:
    """
    Represents a parcel of input code and its input types + closure, containing everything the
    compiler needs in order to do the compilation. Typed_python supports function overloading,
    so everything here is specific to a given 'overload' - a function for a given set of input
    types, and inferred output type.

    Args:
        function_type: a typed_python.Function _type_.
        overload_index: the integer index of the overload we are interested in.
        input_wrappers: A list of types (or TypeWrappers) for each input argument.
    """
    def __init__(self, function_type, overload_index: int, input_wrappers=None) -> None:
        self._overload: FunctionOverload = function_type.overloads[overload_index]
        self._closure_type = function_type.ClosureType
        self._input_wrappers: List = input_wrappers
        self._realized_input_wrappers: List = None
        self._return_type = None
        self._return_type_calculated = False

    def expand_input_wrappers(self) -> None:
        """
        Extend the list of wrappers (representing the input args) using the free variables in
        the function closure.
        """
        realized_input_wrappers = []
        for closure_var_path in self.closureVarLookups.values():
            realized_input_wrappers.append(
                typedPythonTypeToTypeWrapper(
                    PythonTypedFunctionWrapper.closurePathToCellType(closure_var_path, self.closure_type)
                )
            )
        realized_input_wrappers.extend(self.input_wrappers)
        self._realized_input_wrappers = realized_input_wrappers

    def compute_return_type(self) -> None:
        """Determine the return type, if possible."""
        return_type = PythonTypedFunctionWrapper.computeFunctionOverloadReturnType(self._overload,
                                                                                   self.input_wrappers,
                                                                                   {}
                                                                                   )
        if return_type is CannotBeDetermined:
            return_type = object

        elif return_type is NoReturnTypeSpecified:
            return_type = None

        self._return_type = return_type
        self._return_type_calculated = True

    def install_native_pointer(self, fp, returnType, argumentTypes) -> None:
        return self._overload._installNativePointer(fp, returnType, argumentTypes)

    @property
    def return_type(self):
        if not self._return_type_calculated:
            self.compute_return_type()
        return self._return_type

    @property
    def realized_input_wrappers(self):
        return self._realized_input_wrappers

    @property
    def input_wrappers(self):
        return self._input_wrappers

    @input_wrappers.setter
    def input_wrappers(self, wrappers: List):
        self._input_wrappers = wrappers

    @property
    def closure_type(self):
        return self._closure_type

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
