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

import typed_python.python_ast as python_ast
import typed_python.ast_util as ast_util

import nativepython
import nativepython.native_ast as native_ast
from nativepython.type_wrappers.none_wrapper import NoneWrapper
from nativepython.python_object_representation import typedPythonTypeToTypeWrapper
from nativepython.expression_conversion_context import ExpressionConversionContext
from nativepython.function_conversion_context import FunctionConversionContext
from typed_python import *

NoneExprType = NoneWrapper()

typeWrapper = lambda t: nativepython.python_object_representation.typedPythonTypeToTypeWrapper(t)


class NativeFunctionConversionContext:
    def __init__(self, converter, input_types, output_type, generatingFunction):
        self.varnames = 0
        self.converter = converter
        self._input_types = input_types
        self._output_type = output_type
        self._generatingFunction = generatingFunction

    def let_varname(self):
        self.varnames += 1
        return ".var_%s" % self.varnames

    def stack_varname(self):
        return self.let_varname()

    def typesAreUnstable(self):
        return False

    def resetTypeInstabilityFlag(self):
        pass

    def convertToNativeFunction(self):
        return self.getFunction(), self._output_type

    def getFunction(self):
        subcontext = ExpressionConversionContext(self)
        output_type = self._output_type
        input_types = self._input_types
        generatingFunction = self._generatingFunction

        if output_type.is_pass_by_ref:
            outputArg = subcontext.inputArg(output_type, '.return')
        else:
            outputArg = None

        inputArgs = [subcontext.inputArg(input_types[i], 'a_%s' % i) if not input_types[i].is_empty
                     else subcontext.pushPod(native_ast.nullExpr, input_types[i])
                     for i in range(len(input_types))]

        native_input_types = [t.getNativePassingType() for t in input_types if not t.is_empty]

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


class TypedCallTarget(object):
    def __init__(self, named_call_target, input_types, output_type):
        super().__init__()

        self.named_call_target = named_call_target
        self.input_types = input_types
        self.output_type = output_type

    def call(self, *args):
        return native_ast.CallTarget.Named(target=self.named_call_target).call(*args)

    @property
    def name(self):
        return self.named_call_target.name

    def __str__(self):
        return "TypedCallTarget(name=%s,inputs=%s,outputs=%s)" % (
            self.name,
            [str(x) for x in self.input_types],
            str(self.output_type)
        )


class PythonToNativeConverter(object):
    def __init__(self):
        object.__init__(self)
        self._names_for_identifier = {}
        self._definitions = {}
        self._targets = {}
        self._inflight_function_conversions = {}
        self._new_native_functions = set()
        self._used_names = set()

        self.verbose = False

    def extract_new_function_definitions(self):
        """Return a list of all new function definitions from the last conversion."""
        res = {}

        for u in self._new_native_functions:
            res[u] = self._definitions[u]

            if self.verbose:
                print(self._targets[u])

        self._new_native_functions = set()

        return res

    def new_name(self, name, prefix="py."):
        suffix = None
        getname = lambda: prefix + name + ("" if suffix is None else ".%s" % suffix)
        while getname() in self._used_names:
            suffix = 1 if not suffix else suffix+1
        res = getname()
        self._used_names.add(res)
        return res

    def createConversionContext(self, identity, f, input_types, output_type):
        pyast, freevars = self._callable_to_ast_and_vars(f)

        if isinstance(pyast, python_ast.Statement.FunctionDef):
            body = pyast.body
        else:
            body = [python_ast.Statement.Return(
                value=ast.body,
                line_number=ast.body.line_number,
                col_offset=ast.body.col_offset,
                filename=ast.body.filename
            )]

        return FunctionConversionContext(self, identity, pyast.args, body, input_types, output_type, freevars)

    def defineNativeFunction(self, name, identity, input_types, output_type, generatingFunction):
        """Define a native function if we haven't defined it before already.

            name - the name to actually give the function.
            identity - a unique identifier for this function to allow us to cache it.
            input_types - list of Wrapper objects for the incoming types
            output_type - Wrapper object for the output type.
            generatingFunction - a function producing a native_function_definition

        returns a TypedCallTarget. 'generatingFunction' may call this recursively if it wants.
        """
        output_type = typeWrapper(output_type)
        input_types = [typeWrapper(x) for x in input_types]

        identity = ("native", identity, tuple(input_types))

        if identity in self._names_for_identifier:
            return self._targets[self._names_for_identifier[identity]]

        new_name = self.new_name(name, "runtime.")

        self._names_for_identifier[identity] = new_name
        self._inflight_function_conversions[identity] = NativeFunctionConversionContext(self, input_types, output_type, generatingFunction)

        self._targets[new_name] = self.getTypedCallTarget(new_name, input_types, output_type)

        return self._targets[new_name]

    def getTypedCallTarget(self, name, input_types, output_type):
        native_input_types = [a.getNativePassingType() for a in input_types if not a.is_empty]
        if output_type is None:
            native_output_type = native_ast.Type.Void()
        elif output_type.is_pass_by_ref:
            native_input_types = [output_type.getNativePassingType()] + native_input_types
            native_output_type = native_ast.Type.Void()
        else:
            native_output_type = output_type.getNativeLayoutType()

        return TypedCallTarget(
            native_ast.NamedCallTarget(
                name=name,
                arg_types=native_input_types,
                output_type=native_output_type,
                external=False,
                varargs=False,
                intrinsic=False,
                can_throw=True
            ),
            input_types,
            output_type
        )

    def _callable_to_ast_and_vars(self, f):
        pyast = ast_util.pyAstFor(f)

        _, lineno = ast_util.getSourceLines(f)
        _, fname = ast_util.getSourceFilenameAndText(f)

        pyast = ast_util.functionDefOrLambdaAtLineNumber(pyast, lineno)

        pyast = python_ast.convertPyAstToAlgebraic(pyast, fname)

        freevars = dict(f.__globals__)

        if f.__closure__:
            for i in range(len(f.__closure__)):
                try:
                    freevars[f.__code__.co_freevars[i]] = f.__closure__[i].cell_contents
                except Exception:
                    print("BAD IS ", f)
                    raise

        return pyast, freevars

    def generateCallConverter(self, callTarget):
        """Given a call target that's optimized for C-style dispatch, produce a (native) call-target that
        we can dispatch to from our C extension.

        we are given
            T f(A1, A2, A3 ...)
        and want to produce
            f(T*, X**)
        where X is the union of A1, A2, etc.

        returns the name of the defined native function
        """
        identifier = ("call_converter", callTarget.name)

        if identifier in self._names_for_identifier:
            return self._names_for_identifier[identifier]

        args = []
        for i in range(len(callTarget.input_types)):
            if not callTarget.input_types[i].is_empty:
                argtype = callTarget.input_types[i].getNativeLayoutType()

                untypedPtr = native_ast.var('input').ElementPtrIntegers(i).load()

                if callTarget.input_types[i].is_pass_by_ref:
                    # we've been handed a pointer, and it's already a pointer
                    args.append(untypedPtr.cast(argtype.pointer()))
                else:
                    args.append(untypedPtr.cast(argtype.pointer()).load())

        if callTarget.output_type is not None and callTarget.output_type.is_pass_by_ref:
            body = callTarget.call(
                native_ast.var('return').cast(callTarget.output_type.getNativeLayoutType().pointer()),
                *args
            )
        else:
            body = callTarget.call(*args)

            if not (callTarget.output_type is None or callTarget.output_type.is_empty):
                body = native_ast.var('return').cast(callTarget.output_type.getNativeLayoutType().pointer()).store(body)

        body = native_ast.FunctionBody.Internal(body=body)

        definition = native_ast.Function(
            args=(
                ('return', native_ast.Type.Void().pointer()),
                ('input', native_ast.Type.Void().pointer().pointer())
            ),
            body=body,
            output_type=native_ast.Type.Void()
        )

        new_name = self.new_name(callTarget.name + ".dispatch")
        self._names_for_identifier[identifier] = new_name

        self._definitions[new_name] = definition
        self._new_native_functions.add(new_name)

        return new_name

    def _resolveInflightOnePass(self):
        oldCount = len(self._inflight_function_conversions)
        repeat = False

        for identity, functionConverter in list(self._inflight_function_conversions.items()):
            nativeFunction, actual_output_type = functionConverter.convertToNativeFunction()

            if nativeFunction is None:
                repeat = True
            else:
                if functionConverter.typesAreUnstable():
                    functionConverter.resetTypeInstabilityFlag()
                    repeat = True

                name = self._names_for_identifier[identity]

                self._targets[name] = self.getTypedCallTarget(name, functionConverter._input_types, actual_output_type)

        return repeat or len(self._inflight_function_conversions) != oldCount

    def _resolveAllInflightFunctions(self):
        passCt = 0
        while self._resolveInflightOnePass():
            passCt += 1
            if passCt > 100:
                print("We've done ", passCt, " with ", len(self._inflight_function_conversions))
                for c in self._inflight_function_conversions.values():
                    print("    ", c.identity[1].__name__, c.identity[2], "->", c._output_type)
                raise Exception("Exceed max pass count")

    def convert(self, f, input_types, output_type, assertIsRoot=False):
        """Convert a single pure python function using args of 'input_types'.

        It will return no more than 'output_type'. if output_type is None we produce
        the tightest output type possible.
        """
        input_types = tuple([typedPythonTypeToTypeWrapper(i) for i in input_types])

        identifier = ("pyfunction", f, input_types, output_type)

        if identifier in self._names_for_identifier:
            name = self._names_for_identifier[identifier]
        else:
            name = self.new_name(f.__name__)
            self._names_for_identifier[identifier] = name

        if name in self._targets:
            return self._targets[name]

        isRoot = len(self._inflight_function_conversions) == 0

        if assertIsRoot:
            assert isRoot

        if identifier not in self._inflight_function_conversions:
            functionConverter = self.createConversionContext(identifier, f, input_types, output_type)

            self._inflight_function_conversions[identifier] = functionConverter

        if isRoot:
            try:
                self._resolveAllInflightFunctions()
                self._installInflightFunctions()
                return self._targets[name]
            finally:
                self._inflight_function_conversions.clear()
        else:
            # above us on the stack, we are walking a set of function conversions.
            # if we have ever calculated this function before, we'll have a call
            # target with an output type and we can return that. Otherwise we have to
            # return None, which will cause callers to replace this with a throw
            # until we have had a chance to do a full pass of conversion.
            if name in self._targets:
                return self._targets[name]
            else:
                return None

    def _installInflightFunctions(self):
        for identifier, functionConverter in self._inflight_function_conversions.items():
            nativeFunction, actual_output_type = functionConverter.convertToNativeFunction()

            assert nativeFunction is not None
            name = self._names_for_identifier[identifier]

            self._definitions[name] = nativeFunction
            self._new_native_functions.add(name)

        self._inflight_function_conversions.clear()
