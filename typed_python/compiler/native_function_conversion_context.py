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

import typed_python.compiler.native_ast as native_ast
from typed_python.compiler.expression_conversion_context import ExpressionConversionContext
from typed_python.compiler.function_metadata import FunctionMetadata


class NativeFunctionConversionContext:
    """A FunctionConversionContext look-alike for converting native helper code (e.g. destructors).

    In this case, we know all the types up-front, so we don't need as much infrastructure for tracking
    return types etc.
    """
    def __init__(self, converter, input_types, output_type, generatingFunction, identity):
        self.varnames = 0
        self.converter = converter
        self._input_types = input_types
        self._output_type = output_type
        self._generatingFunction = generatingFunction
        self._identity = identity
        self.functionMetadata = FunctionMetadata()

    def getInputTypes(self):
        return self._input_types

    def knownOutputType(self):
        """If the output type is known ahead, then that type (as a wrapper). Else, None"""
        return self._output_type

    def __str__(self):
        return f"NativeFunctionConversionContext({self._identity}, {self._generatingFunction})"

    @property
    def identity(self):
        return self._identity

    def allocateLetVarname(self):
        self.varnames += 1
        return ".var_%s" % self.varnames

    def allocateStackVarname(self):
        return self.allocateLetVarname()

    def alwaysRaises(self):
        return False

    def typesAreUnstable(self):
        return False

    def resetTypeInstabilityFlag(self):
        pass

    def convertToNativeFunction(self):
        return self.getFunction(), self._output_type

    def currentReturnType(self):
        return self._output_type

    def getFunction(self):
        self.varnames = 0
        # reset the constant return value.
        self.functionMetadata = FunctionMetadata()

        try:
            subcontext = ExpressionConversionContext(self, None)
            output_type = self._output_type
            input_types = self._input_types
            generatingFunction = self._generatingFunction

            if output_type.is_pass_by_ref:
                outputArg = subcontext.inputArg(output_type, '.return')
            else:
                outputArg = None

            inputArgs = [subcontext.inputArg(input_types[i], 'a_%s' % i) if not input_types[i].is_empty
                         else subcontext.pushPod(input_types[i], native_ast.nullExpr)
                         for i in range(len(input_types))]

            generatingFunction(subcontext, outputArg, *inputArgs)

            native_args = [
                ('a_%s' % i, input_types[i].getNativePassingType())
                for i in range(len(input_types)) if not input_types[i].is_empty
            ]

            if output_type.is_pass_by_ref:
                # the first argument is actually the output
                native_output_type = native_ast.Void
                native_args = [('.return', output_type.getNativePassingType())] + native_args
            else:
                native_output_type = output_type.getNativeLayoutType()

            return native_ast.Function(
                args=native_args,
                output_type=native_output_type,
                body=native_ast.FunctionBody.Internal(subcontext.finalize(None))
            )
        except Exception:
            print("Failing function is ", self.identity)
            raise
