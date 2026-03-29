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

"""introspect.py

Contains helper functions for exploring the objects and syntax trees generatied in the process
of going from python -> native IR -> LLVM IR.
"""

import typed_python
import typed_python.compiler.python_to_native_converter as python_to_native_converter

from typed_python import Runtime, Function


def getNativeIRString(
    typedFunc: Function, args=None, kwargs=None
) -> str:
    """
    Given a function compiled with Entrypoint, return a text representation
    of the generated native (one layer prior to LLVM) code.

    Args:
        typedFunc (Function): a decorated python function.
        args (Optional(list)): these optional args should be the Types of the functions' positional arguments
        kwargs (Optional(dict)): these keyword args should be the Types of the functions' keyword arguments

    Returns:
        A string for the function bodies generated (including constructors and destructors)
    """
    converter = Runtime.singleton().llvm_compiler.converter

    function_name = getFullFunctionNameWithArgs(typedFunc, args, kwargs)
    # relies on us maintaining our naming conventions (tests would break otherwise)
    output_str = ""
    for key, value in converter._function_definitions.items():
        if function_name in key:
            output_str += f"Function {key}" + "_" * 20 + "\n"
            output_str += str(value.body.body) + "\n"
            output_str += "_" * 80 + "\n"

    if not output_str:
        raise ValueError(
            "no matching function definitions found - has the code been compiled (and run)?"
        )

    return output_str


def getLLVMString(
    typedFunc: Function, args=None, kwargs=None
) -> str:
    """
    Given a function compiled with Entrypoint, return a text representation
    of the generated LLVM code.

    Args:
        typedFunc (Function): a decorated python function.
        args (Optional(list)): these optional args should be the Types of the functions' positional arguments
        kwargs (Optional(dict)): these keyword args should be the Types of the functions' keyword arguments

    Returns:
        A string for the function bodies generated (including constructors and destructors)
    """
    converter = Runtime.singleton().llvm_compiler.converter

    function_name = getFullFunctionNameWithArgs(typedFunc, args, kwargs)

    output_str = ""
    for key, value in converter._functions_by_name.items():
        if function_name in key:
            output_str += f"Function {key}" + "_" * 20 + "\n"
            output_str += str(value) + "\n"
            output_str += "_" * 80 + "\n"

    return output_str


def getFullFunctionNameWithArgs(funcObj, argTypes, kwargTypes):
    """
    Given a Function and a set of types, compile the function to generate the unique name
    for that function+argument combination.

    Args:
        funcObj (Function): a typed_python Function.
        argTypes (List): a list of the position arguments for the function.
        kwargTypes (Dict): a key:value mapping for the functions' keywords arguments.
    """
    assert isinstance(funcObj, typed_python._types.Function)
    typeWrapper = lambda t: python_to_native_converter.typedPythonTypeToTypeWrapper(t)
    funcObj = typed_python._types.prepareArgumentToBePassedToCompiler(funcObj)
    argTypes = [typeWrapper(a) for a in argTypes] if argTypes is not None else []
    kwargTypes = (
        {k: typeWrapper(v) for k, v in kwargTypes.items()}
        if kwargTypes is not None
        else {}
    )

    overload_index = 0
    overload = funcObj.overloads[overload_index]

    ExpressionConversionContext = (
        typed_python.compiler.expression_conversion_context.ExpressionConversionContext
    )
    argumentSignature = (
        ExpressionConversionContext.computeFunctionArgumentTypeSignature(
            overload, argTypes, kwargTypes
        )
    )

    if argumentSignature is not None:
        callTarget = (
            Runtime()
            .singleton()
            .compileFunctionOverload(
                funcObj, overload_index, argumentSignature, argumentsAreTypes=True
            )
        )
        return callTarget.name
    else:
        raise ValueError("no signature found.")
