#   Copyright 2017-2022 typed_python Authors
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
import importlib
import typed_python.compiler.codegen_helpers as codegen_helpers
from typed_python.compiler.for_loop_codegen import rewriteForLoops, rewriteIntiterForLoop
from typed_python.compiler.generator_codegen import GeneratorCodegen
from typed_python.compiler.withblock_codegen import expandWithBlockIntoTryCatch
from typed_python.compiler.python_ast_analysis import (
    computeAssignedVariables,
    computeReadVariables,
    computeFunctionArgVariables,
    computeVariablesAssignedOnlyOnce,
    computeVariablesReadByClosures,
    countYieldStatements,
    extractFunctionDefs,
)
from typed_python.internals import makeFunctionType, checkOneOfType, checkType
from typed_python.compiler.conversion_level import ConversionLevel
import typed_python.compiler
import typed_python.compiler.native_ast as native_ast
from typed_python import _types, Type, ListOf, PointerTo, pointerTo, Set, Dict, Member
from typed_python.generator import Generator
import typed_python.compiler.type_wrappers.runtime_functions as runtime_functions
from typed_python.compiler.expression_conversion_context import ExpressionConversionContext
from typed_python.compiler.function_stack_state import FunctionStackState
from typed_python.compiler.type_wrappers.none_wrapper import NoneWrapper
from typed_python.compiler.typed_expression import TypedExpression
from typed_python.compiler.conversion_exception import ConversionException
from typed_python import OneOf, Function, Tuple, Forward, Class
from typed_python.compiler.function_metadata import FunctionMetadata
from typed_python.compiler.merge_type_wrappers import mergeTypeWrappers
from typed_python.python_ast import evaluateFunctionDefWithLocalsInCells

# Constants for control flow instructions
CONTROL_FLOW_DEFAULT = 0
CONTROL_FLOW_EXCEPTION = 1
CONTROL_FLOW_BREAK = 2
CONTROL_FLOW_CONTINUE = 3
CONTROL_FLOW_RETURN = 4

typeWrapper = lambda t: typed_python.compiler.python_object_representation.typedPythonTypeToTypeWrapper(t)


# storage for mutually recursive function types
_closureCycleMemo = {}


class FunctionOutput:
    pass


class FunctionYield:
    pass


class ControlFlowBlocks:
    """Models info about where control flow should go.

    When we're generating code within a try/catch or a while loop we need
    to know where break/continue/return statements should go. This class
    models how codegen handles those cases.

    return_to - label to return to. Only set within a 'try' statement.
        If None, return normally.
    try_flow - variable containing deferred control flow instruction.  Specifically, what
        control flow should be followed after the 'finally' block.
        Only set within a 'try' statement.
        0=default, 1=unhandled exception, 2=break, 3=continue, 4=return
    in_loop - would a break/continue directly hit a surrounding while or for loop?
    """

    def __init__(self, return_to=None, try_flow=None, in_loop=False, in_loop_ever=False):
        self.return_to = return_to
        self.try_flow = try_flow
        self.in_loop = in_loop
        self.in_loop_ever = in_loop_ever

    def pushTryCatch(self, return_to, try_flow):
        return ControlFlowBlocks(
            return_to=return_to, try_flow=try_flow, in_loop=False, in_loop_ever=self.in_loop_ever
        )

    def pushLoop(self):
        return ControlFlowBlocks(
            return_to=self.return_to, try_flow=self.try_flow, in_loop=True, in_loop_ever=True
        )


class ConversionContextBase:
    """Helper function for converting a single python function given some input and output types"""

    def __init__(
        self,
        converter,
        name,
        identity,
        input_types,
        output_type,
        funcArgNames,
        closureVarnames,
        globalVars,
        globalVarsRaw,
    ):
        """Initialize a FunctionConverter

        Args:
            converter - a PythonToNativeConverter
            name - the function name
            identity - an object to uniquely identify this instance of the function
            input_types - a list of the input types actually passed to us. There must be a type
                for each closure varname, and then again for each function argument.
            output_type - the output type (if proscribed), or None
            funcArgNames - the stated list of argument names to this function.
            closureVarnames - names of the variables in this function's closure. These will be passed
                before the actual func args.
            globalVars - a dict from name to the actual python object in the globals for this function
            globalVarsRaw - the original dict where these globals live.
        """
        self.name = name
        self.funcArgNames = funcArgNames

        self.variablesAssigned = set()
        self.variablesBound = set()

        # the set of variables that are captured in closures in this function.
        # this includes recursive functions, which will not be in the closure itself
        # since they get bound in the closure varnames.
        self.variablesReadByClosures = set()

        # the set of variables that have exactly one definition. If these are 'deffed'
        # functions, we don't have to worry about them changing type and so they can be
        # bound to the closure directly (in which case we don't even assign them to slots)
        self.variablesAssignedOnlyOnce = set()

        # the list of 'def' statements and 'Lambda' expressions. each one engenders a function type.
        self.functionDefs = []
        self.generators = []
        self.comprehensions = []

        # all 'def' operations that are assigned exactly once. These defs are special
        # because we just assume that the binding is active without even evaluating the
        # def. Other bindings (lambdas, etc), require us to track slots for the closure itself
        self.functionDefsAssignedOnce = {}

        # the current _type_ that we're using for this def,
        self.functionDefToType = {}

        # variables in closure slots that are not single-assignment function defs need slots
        self.variablesNeedingClosureSlots = set()

        # for all typed functions we have ever defined, the original untyped function.
        # This grows with each pass and is there to help us when we're walking types
        # looking for our own closures to replace.
        self.typedFunctionTypeToClosurelessFunctionType = {}

        # bidirectional map between the 'def' and the resulting function object
        self.functionDefToClosurelessFunctionTypeCache = {}
        self.closurelessFunctionTypeToDef = {}

        # if we assign a closure, the type of it. Initially None because we have to fill it
        # out with something...
        self.closureType = None

        self.converter = converter
        self.identity = identity
        self._argnames = None
        self._argtypes = {}
        self._input_types = input_types
        self._output_type = output_type
        self._argumentsWithoutStackslots = set()  # arguments that we don't bother to copy into the stack
        self._varname_to_type = {}
        self._globals = globalVars
        self._globalsRaw = globalVarsRaw
        self._closureVarnames = closureVarnames

        self.tempLetVarIx = 0
        self._tempStackVarIx = 0
        self._tempIterVarIx = 0

        self._typesAreUnstable = False
        self._functionOutputTypeKnown = False
        self._functionYieldTypeKnown = False
        self._native_args = None
        self.functionMetadata = FunctionMetadata()

    @property
    def isGenerator(self):
        """Override to true if the function should produce a Generator object."""
        return False

    def alwaysRaises(self):
        return False

    def getInputTypes(self):
        return self._input_types

    def knownOutputType(self):
        """If the output type is known ahead, then that type (as a wrapper). Else, None"""
        return self._output_type

    def currentReturnType(self):
        return self._varname_to_type.get(FunctionOutput)

    def isLocalVariable(self, name):
        return name in self.variablesBound or name in self.variablesAssigned

    def isClosureVariable(self, name):
        return name in self._closureVarnames

    def shouldReadAndWriteVariableFromClosure(self, varname):
        return varname in self.variablesReadByClosures

    def localVariableExpression(self, context: ExpressionConversionContext, name):
        """Return an TypedExpression reference for the local variable given by  'name'"""
        if name in self.functionDefsAssignedOnce:
            return self.localVariableExpression(context, ".closure").changeType(
                self.functionDefToType[self.functionDefsAssignedOnce[name]]
            )

        if self.shouldReadAndWriteVariableFromClosure(name):
            return self.localVariableExpression(context, ".closure").convert_attribute(name, nocheck=True)

        if name != ".closure" and self._varname_to_type.get(name) is None:
            context.pushException(NameError, f"name '{name}' is not defined")
            return None

        if name == ".closure":
            slot_type = typeWrapper(self.closureType)
        else:
            slot_type = self._varname_to_type[name]

        return TypedExpression(
            context,
            native_ast.Expression.StackSlot(name=name, type=slot_type.getNativeLayoutType()),
            slot_type,
            isReference=True,
        )

    def variableIsAlwaysEmpty(self, name):
        assert self.isLocalVariable(name), f"{name} is not a local variable here."

        # we have never assigned to this thing, so we need to upcast it
        if name not in self._varname_to_type:
            return True

        if self._varname_to_type[name] is None:
            return True

        return self._varname_to_type[name].is_empty

    def variableNeedsDestructor(self, name):
        if name in self._argumentsWithoutStackslots:
            return False

        if name in self.variablesReadByClosures:
            # destroying the closure itself will handle this
            return False

        varType = self._varname_to_type.get(name)

        if varType is None or varType.is_empty or varType.is_pod:
            return False

        return True

    def closurePathToName(self, name):
        if name in self.functionDefsAssignedOnce:
            # this is another function in the closure. We want to just bind it directly
            # to our closure
            return [self.functionDefToType[self.functionDefsAssignedOnce[name]]]

        return [name]

    def computeTypeForFunctionDef(self, ast):
        untypedFuncType = self.functionDefToClosurelessFunction(ast)

        if hasattr(ast, "decorator_list"):
            for decorator in ast.decorator_list:
                ctx = ExpressionConversionContext(self, FunctionStackState())
                decoratorEval = ctx.convert_expression_ast(decorator)
                if decoratorEval is None:
                    raise Exception("Decorators other than NotCompiled or Entrypoint not supported yet")

                if decoratorEval.expr_type.typeRepresentation is typed_python.NotCompiled:
                    untypedFuncType = untypedFuncType.typeWithNocompile(True)
                elif decoratorEval.expr_type.typeRepresentation is typed_python.Entrypoint:
                    untypedFuncType = untypedFuncType.typeWithEntrypoint(True)
                else:
                    raise Exception("Decorators other than NotCompiled or Entrypoint not supported yet")

        typedFuncType = untypedFuncType.withClosureType(self.closureType)
        typedFuncType = typedFuncType.withOverloadVariableBindings(
            0, {name: self.closurePathToName(name) for name in untypedFuncType.overloads[0].closureVarLookups}
        )

        return untypedFuncType, typedFuncType

    def currentClosureLookupKey(self):
        usedTypes = {}
        for var in self.variablesNeedingClosureSlots:
            if var in self._varname_to_type:
                usedTypes[var] = self.stripClosureTypes(self._varname_to_type[var].typeRepresentation)

        return (self.identity, tuple(sorted(usedTypes.items())))

    def stripClosureTypes(self, nativeType):
        res = self.replaceClosureTypeWith(nativeType, wantsCurrentClosure=False)
        if res is None:
            return nativeType
        return res

    def replaceClosureTypesIn(self, nativeType):
        res = self.replaceClosureTypeWith(nativeType, wantsCurrentClosure=True)
        if res is None:
            return nativeType
        return res

    def replaceClosureTypeWith(self, nativeType, wantsCurrentClosure):
        """Return 'nativeType' stripped of any references to our closure, or None if unchanged."""
        assert isinstance(nativeType, type) and (
            issubclass(nativeType, Type),
            (nativeType, type(nativeType)) or nativeType in (int, float, bool, str, bytes, type(None)),
        )

        if nativeType in self.typedFunctionTypeToClosurelessFunctionType:
            untypedFuncType = self.typedFunctionTypeToClosurelessFunctionType[nativeType]

            if not wantsCurrentClosure:
                return untypedFuncType

            origDef = self.closurelessFunctionTypeToDef[untypedFuncType]

            resType = self.functionDefToType[origDef]

            if resType == nativeType:
                return None

            return resType

        # this is a deliberately simple implementation - not good enough at all.
        # we're ignoring tuples, named tuples, etc, all of which we'll need to be
        # able to handle, including recursively defined forwards, eventually.
        # for now, oneof is good enough. you have to write some weird stuff where
        # you put mutually recursive functions into objects to break this.
        if issubclass(nativeType, OneOf):
            elts = []
            anyDifferent = False

            for e in nativeType.Types:
                elts.append(self.replaceClosureTypeWith(e, wantsCurrentClosure))
                if elts[-1] is None:
                    elts[-1] = e
                else:
                    anyDifferent = True

            if anyDifferent:
                return OneOf(*elts)

        return None

    def buildClosureTypes(self):
        """Determine the type of the closure we'll build, plus any recursive function definitions.

        We rely on passing over the function multiple times to build up the information about each
        closure we need to generate. On the final pass, this set of types will be stable and we can
        generate an appropriate closure.
        """
        closureKey = self.currentClosureLookupKey()

        if closureKey in _closureCycleMemo:
            self.closureType, self.functionDefToType = _closureCycleMemo[closureKey]
            return

        # we need to build a closure type.
        if not self.variablesNeedingClosureSlots:
            # we don't need any slots at all
            self.closureType = Tuple()
        else:
            self.closureType = Forward(self.name + ".closure")

        self.functionDefToType = {
            fd: Forward(fd.name if fd.matches.FunctionDef else "<lambda>")
            for fd in list(self.functionDefs) + self.generators + self.comprehensions
        }

        # walk over the function defs and instantiate their forward types
        for ast in list(self.functionDefs) + self.generators + self.comprehensions:
            untypedFuncType, typedFuncType = self.computeTypeForFunctionDef(ast)

            self.functionDefToType[ast] = self.functionDefToType[ast].define(typedFuncType)
            self.typedFunctionTypeToClosurelessFunctionType[typedFuncType] = untypedFuncType

        if self.variablesNeedingClosureSlots:
            # now build the closure type itself and replace the forward with the defined class
            closureMembers = []
            classMembers = []

            replacedVarTypes = {}

            for var in sorted(self.variablesNeedingClosureSlots):
                if var in self._varname_to_type:
                    replacedVarTypes[var] = self.replaceClosureTypesIn(
                        self._varname_to_type[var].typeRepresentation
                    )

                    closureMembers.append((var, replacedVarTypes[var], None, False))
                    classMembers.append((var, Member(replacedVarTypes[var], None, False)))

            memberFunctions = {
                "__init__": makeFunctionType(
                    "__init__", lambda self: None, classname=self.name + ".closure", assumeClosuresGlobal=True
                )
            }

            self.closureType = self.closureType.define(
                _types.Class(
                    self.name + ".closure",
                    (),
                    True,
                    tuple(closureMembers),
                    tuple(memberFunctions.items()),
                    (),
                    (),
                    tuple(classMembers),
                    ()
                )
            )

            assert not _types.is_default_constructible(self.closureType)

            # now rebuild each of our var types. we have to do this after
            # setting 'closureType' because otherwise we'll end up with undefined
            # forwards floating around.
            for var in self._varname_to_type:
                if var in replacedVarTypes:
                    self._varname_to_type[var] = typeWrapper(replacedVarTypes[var])
                elif isinstance(self._varname_to_type[var], type):
                    self._varname_to_type[var] = typeWrapper(
                        self.replaceClosureTypesIn(self._varname_to_type[var].typeRepresentation)
                    )

        _closureCycleMemo[closureKey] = (self.closureType, dict(self.functionDefToType))

    def convertToNativeFunction(self):
        self.tempLetVarIx = 0
        self._tempStackVarIx = 0
        self._tempIterVarIx = 0
        self.functionMetadata = FunctionMetadata()

        variableStates = FunctionStackState()

        self.buildClosureTypes()

        initializer_expr = self.initializeVariableStates(self._argnames, variableStates)
        body_native_expr, controlFlowReturns = self.convert_function_body(variableStates)
        assert not controlFlowReturns

        if self.isGenerator:
            # discard the native expression and generate code to produce the closure
            # object instead
            body_native_expr = self.convert_build_generator()
        else:
            # destroy our variables if they are in scope
            destructors = self.generateDestructors(variableStates)

            body_native_expr = initializer_expr >> body_native_expr

            if destructors:
                body_native_expr = native_ast.Expression.Finally(teardowns=destructors, expr=body_native_expr)

        return_type = self._varname_to_type.get(FunctionOutput, None)

        if return_type is None:
            return (
                native_ast.Function(
                    args=self._native_args,
                    body=native_ast.FunctionBody.Internal(body=body_native_expr),
                    output_type=native_ast.Void,
                ),
                return_type,
            )

        if return_type.is_pass_by_ref:
            return (
                native_ast.Function(
                    args=(
                        ((".return", return_type.getNativeLayoutType().pointer()),) + tuple(self._native_args)
                    ),
                    body=native_ast.FunctionBody.Internal(body=body_native_expr),
                    output_type=native_ast.Void,
                ),
                return_type,
            )
        else:
            return (
                native_ast.Function(
                    args=self._native_args,
                    body=native_ast.FunctionBody.Internal(body=body_native_expr),
                    output_type=return_type.getNativeLayoutType(),
                ),
                return_type,
            )

    def createGeneratorFun(self):
        """Modify our function code to be the '__next__' of a class defining the generator.

        Basically, 'yield' gets turned to 'return', variable accesses become 'self.', and
        we introduce some extra code to route ourselves to the correct place in the code
        based on the last 'yield' statement.
        """
        return GeneratorCodegen(
            set(self._varname_to_type), self._statements[0].line_number
        ).convertStatementsToFunctionDef(
            rewriteForLoops(self._statements) if self.isGenerator else self.statements
        )

    def convert_build_generator(self):
        """Generate code that returns a 'generator' object."""
        if FunctionYield not in self._varname_to_type:
            context = ExpressionConversionContext(self, FunctionStackState())
            context.pushException("TypeInference not complete yet")
            return context.finalize(None)

        # if we have a FunctionYield, we have a type for our generator
        T = self._varname_to_type[FunctionYield].typeRepresentation

        generatorMembers = [("..slot", int, None, True), ("..value", T, None, True)]

        classMembers = [("..slot", Member(int, None, True)), ("..value", Member(T, None, True))]

        for k, v in self._varname_to_type.items():
            if isinstance(k, str):
                generatorMembers.append(("." + k, v.typeRepresentation, None, False))
                classMembers.append(("." + k, Member(v.typeRepresentation, None, False)))

        smts = self.createGeneratorFun()

        generatorFun = evaluateFunctionDefWithLocalsInCells(
            smts, self._globals, {".PointerType": PointerTo(T), ".pointerTo": pointerTo}
        )

        generatorBaseclass = Generator(T)

        generatorSubclass = Forward(self.name + ".generator")

        def __iter__(self) -> generatorBaseclass:
            return self

        memberFunctions = {
            "__fastnext__": makeFunctionType(
                "__fastnext__",
                generatorFun,
                classname=self.name + ".generator",
                assumeClosuresGlobal=True,
                returnTypeOverride=PointerTo(T),
            ).typeWithEntrypoint(True),
            "__iter__": makeFunctionType(
                "__iter__", __iter__, classname=self.name + ".generator", assumeClosuresGlobal=True
            ).typeWithEntrypoint(True),
        }

        generatorSubclass = generatorSubclass.define(
            _types.Class(
                self.name + ".generator",
                (generatorBaseclass,),
                True,
                tuple(generatorMembers),
                tuple(memberFunctions.items()),
                (),
                (),
                tuple(classMembers),
                ()
            )
        )

        generatorType = typeWrapper(generatorSubclass)

        self._varname_to_type[FunctionOutput] = generatorType

        assert generatorType.is_pass_by_ref

        variableStates = FunctionStackState()
        context = ExpressionConversionContext(self, variableStates)

        args = {}

        for argByName in self._argnames:
            varType = self._varname_to_type[argByName]

            initializer = TypedExpression(
                context, native_ast.Expression.Variable(name=argByName), varType, varType.is_pass_by_ref
            )

            if initializer is not None:
                args["." + argByName] = initializer

        args["..slot"] = context.constant(-1)

        output = generatorType.convert_type_call(context, None, [], args)

        assert output is not None

        returnSlot = TypedExpression(
            context, native_ast.Expression.Variable(name=".return"), generatorType, isReference=True
        )

        returnSlot.convert_copy_initialize(output)
        return context.finalize(native_ast.Expression.Return())

    def _constructInitialVarnameToType(self):
        input_types = self._input_types

        self._argnames = list(self._closureVarnames) + list(self.funcArgNames)

        if len(input_types) != len(self._argnames):
            raise ConversionException(
                "%s at %s:%s, with closure %s, expected at least %s arguments but got %s. Expected argnames are %s. Input types are %s"
                % (
                    self.name,
                    self._statements[0].filename,
                    self._statements[0].line_number,
                    self._closureVarnames,
                    len(self._argnames),
                    len(input_types),
                    self._argnames,
                    input_types,
                )
            )

        self._native_args = []
        for i, argName in enumerate(self._argnames):
            self._varname_to_type[self._argnames[i]] = input_types[i]
            self._argtypes[self._argnames[i]] = input_types[i]

            if not input_types[i].is_empty:
                self._native_args.append((self._argnames[i], input_types[i].getNativePassingType()))

        if self._output_type is not None:
            self._varname_to_type[FunctionOutput] = typeWrapper(self._output_type)

        self._functionOutputTypeKnown = FunctionOutput in self._varname_to_type

        if self.isGenerator and self._functionOutputTypeKnown:
            if hasattr(self._output_type, "IteratorType"):
                self._varname_to_type[FunctionYield] = typeWrapper(self._output_type.IteratorType)
                self._functionYieldTypeKnown = True

    def typesAreUnstable(self):
        return self._typesAreUnstable

    def resetTypeInstabilityFlag(self):
        self._typesAreUnstable = False

    def markTypesAreUnstable(self):
        self._typesAreUnstable = True

    def allocateLetVarname(self):
        self.tempLetVarIx += 1
        return "letvar.%s" % (self.tempLetVarIx - 1)

    def allocateStackVarname(self):
        self._tempStackVarIx += 1
        return "stackvar.%s" % (self._tempStackVarIx - 1)

    def externalScopeVarExpr(self, subcontext, varname):
        """If 'varname' refers to a known variable that doesn't use a stack slot, return an expression for it.

        This can happen when a variable is passed to us as a function argument
        but not assigned to in our scope, in which case we don't have a stackslot
        for it.

        Args:
            subcontext - the expression conversion context we're using
            varname - the python identifier we're looking up

        Returns:
            a TypedExpression for the given name.
        """
        if varname not in self._argumentsWithoutStackslots:
            return None

        varType = self._varname_to_type[varname]

        return TypedExpression(
            subcontext, native_ast.Expression.Variable(name=varname), varType, varType.is_pass_by_ref
        )

    def setVariableType(self, varname, new_type):
        if self._varname_to_type.get(varname) == new_type:
            return

        self._varname_to_type[varname] = new_type
        self.markTypesAreUnstable()

    def upsizeVariableType(self, varname, new_type):
        if self._varname_to_type.get(varname) is None:
            if new_type is None:
                return

            self._varname_to_type[varname] = new_type
            self.markTypesAreUnstable()
            return

        if self._varname_to_type[varname] == new_type:
            return

        existingTypeWrapper = self._varname_to_type[varname]

        finalType = mergeTypeWrappers([existingTypeWrapper, new_type])

        if finalType != existingTypeWrapper:
            self.markTypesAreUnstable()
            self._varname_to_type[varname] = finalType

    def closureDestructor(self, variableStates):
        if not issubclass(self.closureType, Class):
            return []

        context = ExpressionConversionContext(self, variableStates)

        closure = self.localVariableExpression(context, ".closure")
        closure.convert_destroy()

        return [context.finalize(None)]

    def closureInitializer(self, variableStates):
        if not issubclass(self.closureType, Class):
            return []

        context = ExpressionConversionContext(self, variableStates)

        self.localVariableExpression(context, ".closure").convert_default_initialize(force=True)

        return [context.finalize(None)]

    def functionVariableInitializations(self, variableStates):
        """Produce any initializer expressions that get run when the function is first called.

        Args:
            variableStates - the current VariableStates
        Returns:
            a list of native_ast expressions
        """
        return []

    def initializeVariableStates(self, argnames, variableStates):
        to_add = self.closureInitializer(variableStates)

        to_add.extend(self.functionVariableInitializations(variableStates))

        # reset this
        self._argumentsWithoutStackslots = set()

        # first, mark every variable that we plan on assigning to as not initialized.
        for name in self.variablesAssigned:
            # this is a variable in the function that we assigned to. we need to ensure that
            # the initializer flag is zero
            if not self.variableIsAlwaysEmpty(name) and name not in argnames:
                context = ExpressionConversionContext(self, variableStates)
                context.markVariableNotInitialized(name)
                to_add.append(context.finalize(None))

        for name in self.variablesBound:
            if name not in self.variablesAssigned:
                variableStates.variableAssigned(name, self._varname_to_type[name].typeRepresentation)

        for name in argnames:
            if name is not FunctionOutput:
                if name not in self._varname_to_type:
                    raise ConversionException("Couldn't find a type for argument %s" % name)
                slot_type = self._varname_to_type[name]

                if slot_type.is_empty:
                    # we don't need to generate a stackslot for this value. Whenever we look it up
                    # we'll simply make a void expression
                    pass
                elif slot_type is not None:
                    context = ExpressionConversionContext(self, variableStates)

                    if slot_type.is_empty:
                        pass
                    elif (
                        name in self.variablesBound
                        and name not in self.variablesAssigned
                        and name not in self.variablesReadByClosures
                    ):
                        # this variable is bound but never assigned, so we don't need to
                        # generate a stackslot. We can just read it directly from our arguments
                        self._argumentsWithoutStackslots.add(name)
                    elif slot_type.is_pod:
                        # we can just copy this into the stackslot directly. no destructor needed
                        context.pushEffect(
                            native_ast.Expression.Store(
                                ptr=self.localVariableExpression(context, name).expr,
                                val=(
                                    native_ast.Expression.Variable(name=name)
                                    if not slot_type.is_pass_by_ref
                                    else native_ast.Expression.Variable(name=name).load()
                                ),
                            )
                        )
                        context.markVariableInitialized(name)
                        variableStates.variableAssigned(name, slot_type.typeRepresentation)
                    else:
                        # need to make a stackslot for this variable
                        var_expr = context.inputArg(self._argtypes[name], name)
                        converted = var_expr.convert_to_type(slot_type, ConversionLevel.Signature)
                        assert converted is not None, (
                            "It makes no sense we can't convert an argument to its"
                            " type representation in the function stack."
                        )

                        self.assignToLocalVariable(name, var_expr, variableStates)

                        context.markVariableInitialized(name)

                    to_add.append(context.finalize(None))

        return native_ast.makeSequence(to_add)

    def generateDestructors(self, variableStates):
        destructors = []

        for name in self._varname_to_type:
            if isinstance(name, str) and self.variableNeedsDestructor(name):
                context = ExpressionConversionContext(self, variableStates)

                slot_expr = self.localVariableExpression(context, name)

                with context.ifelse(context.isInitializedVarExpr(name)) as (true, false):
                    with true:
                        slot_expr.convert_destroy()

                destructors.append(
                    native_ast.Teardown.Always(
                        expr=context.finalize(None).with_comment(f"Cleanup for variable {name}")
                    )
                )

        for expr in self.closureDestructor(variableStates):
            destructors.append(native_ast.Teardown.Always(expr=expr))

        return destructors

    def isInitializedVarExpr(self, context, name):
        if self.variableIsAlwaysEmpty(name):
            return context.constant(True)

        return TypedExpression(
            context,
            native_ast.Expression.StackSlot(name=name + ".isInitialized", type=native_ast.Bool),
            bool,
            isReference=True,
        )

    def assignToLocalVariable(self, varname, val_to_store, variableStates):
        """Ensure we have appropriate storage allocated for 'varname', and assign 'val_to_store' to it."""

        if varname not in self.variablesAssigned:
            # make sure we know this variable is new. We'll have to
            # re-execute this converter now that we know about this
            # variable, because right now we generate initializers
            # for our variables only when the converter executes
            # with a stable list of assigned variables (and types)
            self.variablesAssigned.add(varname)
            self.markTypesAreUnstable()

        subcontext = val_to_store.context

        self.upsizeVariableType(varname, val_to_store.expr_type)

        # we should already be filtering this out at the expression level
        assert varname not in self.functionDefsAssignedOnce

        if self.shouldReadAndWriteVariableFromClosure(varname):
            self.localVariableExpression(subcontext, ".closure").convert_set_attribute(varname, val_to_store)
            subcontext.markVariableInitialized(varname)
            return

        assignedType = val_to_store.expr_type.typeRepresentation

        slot_ref = self.localVariableExpression(subcontext, varname)

        if slot_ref is None:
            # this happens if the variable has never been assigned
            return

        # convert the value to the target type now that we've upsized it
        val_to_store = val_to_store.convert_to_type(slot_ref.expr_type, ConversionLevel.Signature)

        assert val_to_store is not None, "We should always be able to upsize"

        if slot_ref.expr_type.is_empty:
            pass
        elif slot_ref.expr_type.is_pod:
            slot_ref.convert_copy_initialize(val_to_store)
            if not variableStates.isDefinitelyInitialized(varname):
                subcontext.markVariableInitialized(varname)
        else:
            if variableStates.isDefinitelyInitialized(varname):
                slot_ref.convert_assign(val_to_store)
            elif variableStates.isDefinitelyUninitialized(varname):
                slot_ref.convert_copy_initialize(val_to_store)
                subcontext.markVariableInitialized(varname)
            else:
                with subcontext.ifelse(subcontext.isInitializedVarExpr(varname)) as (true_block, false_block):
                    with true_block:
                        slot_ref.convert_assign(val_to_store)
                    with false_block:
                        slot_ref.convert_copy_initialize(val_to_store)
                        subcontext.markVariableInitialized(varname)

        variableStates.variableAssigned(varname, assignedType)

    def functionDefToClosurelessFunction(self, ast):
        # parse the code into a function object with no closure.
        if ast not in self.functionDefToClosurelessFunctionTypeCache:
            untypedFunction = python_ast.evaluateFunctionDefWithLocalsInCells(
                ast,
                globals=self._globals,
                locals={name: None for name in (self.variablesBound | self.variablesAssigned)},
            )

            tpFunction = Function(untypedFunction)

            self.functionDefToClosurelessFunctionTypeCache[ast] = type(tpFunction)
            self.closurelessFunctionTypeToDef[type(tpFunction)] = ast

            for varname in tpFunction.overloads[0].closureVarLookups:
                assert varname in self.variablesReadByClosures, varname

        # this function object has a totally bogus closure - it will just have 'None'
        # for each variable it references. We'll need to replace the closure variable binding
        # rules and have it extract its closure
        return self.functionDefToClosurelessFunctionTypeCache[ast]

    def freeVariableLookup(self, name):
        if self.isLocalVariable(name):
            return None

        if name in self._globals:
            return self._globals[name]

        if name in __builtins__:
            return __builtins__[name]

        return None

    def restrictByCondition(self, variableStates, condition, result):
        if (
            condition.matches.Call
            and (
                condition.func.matches.Name
                and self.freeVariableLookup(condition.func.id) is isinstance
                or condition.func.matches.Constant and condition.func.value is isinstance
            ) and len(condition.args) == 2
            and condition.args[0].matches.Name
        ):
            context = ExpressionConversionContext(self, variableStates)
            typeExpr = context.convert_expression_ast(condition.args[1])

            if typeExpr is not None and typeExpr.expr_type.is_py_type_object_wrapper:
                variableStates.restrictTypeFor(
                    condition.args[0].id, typeExpr.expr_type.typeRepresentation.Value, result
                )

        # check if we are a 'var.matches.Y' expression
        if (
            condition.matches.Attribute
            and condition.value.matches.Attribute
            and condition.value.attr == "matches"
            and condition.value.value.matches.Name
        ):
            curType = variableStates.currentType(condition.value.value.id)
            if curType is not None and getattr(curType, "__typed_python_category__", None) == "Alternative":
                if result:
                    subType = [x for x in curType.__typed_python_alternatives__ if x.Name == condition.attr]
                    if subType:
                        variableStates.restrictTypeFor(condition.value.value.id, subType[0], result)


class ExpressionFunctionConversionContext(ConversionContextBase):
    """Helper function for converting a single python function given some input and output types"""

    def __init__(
        self, converter, name, identity, input_types, generator, outputType=None, alwaysRaises=False
    ):
        super().__init__(
            converter,
            name,
            identity,
            input_types,
            outputType,
            [f"a{i}" for i in range(len(input_types))],
            [],
            {},
            {},
        )

        self._generator = generator
        self.variablesBound = set(self.funcArgNames)
        self._alwaysRaises = alwaysRaises
        self._constructInitialVarnameToType()

    def alwaysRaises(self):
        return self._alwaysRaises

    def convert_function_body(self, variableStates: FunctionStackState):
        subcontext = ExpressionConversionContext(self, variableStates)

        try:
            expr = self._generator(*[subcontext.namedVariableLookup(a) for a in self.funcArgNames])
        except Exception as e:
            newMessage = f"\n{self.identity}\n"

            if e.args:
                e.args = (str(e.args[0]) + newMessage,)
            else:
                e.args = (newMessage,)
            raise

        if expr is not None:
            if not self._functionOutputTypeKnown:
                self.upsizeVariableType(FunctionOutput, expr.expr_type)

            subcontext.pushReturnValue(expr)

        return subcontext.finalize(None), False


class FunctionConversionContext(ConversionContextBase):
    """Helper function for converting a single python function given some input and output types"""

    def __init__(
        self,
        converter,
        name,
        identity,
        input_types,
        output_type,
        closureVarnames,
        globalVars,
        globalVarsRaw,
        ast_arg,
        ast,
    ):
        super().__init__(
            converter,
            name,
            identity,
            input_types,
            output_type,
            ast_arg.argumentNames(),
            closureVarnames,
            globalVars,
            globalVarsRaw,
        )

        self._statements = statements = self.extractStatements(ast)

        self.variablesAssigned = computeAssignedVariables(statements)
        self.variablesBound = computeFunctionArgVariables(ast_arg) | set(closureVarnames)

        # the set of variables that are captured in closures in this function.
        # this includes recursive functions, which will not be in the closure itself
        # since they get bound in the closure varnames.
        self.variablesReadByClosures = computeVariablesReadByClosures(statements)

        # if this is not zero, then we are a generator
        self._bodyHasYieldStatements = countYieldStatements(statements) > 0

        # the set of variables that have exactly one definition. If these are 'deffed'
        # functions, we don't have to worry about them changing type and so they can be
        # bound to the closure directly (in which case we don't even assign them to slots)
        self.variablesAssignedOnlyOnce = computeVariablesAssignedOnlyOnce(statements)

        (functionDefs, assignedLambdas, freeLambdas, comprehensions, generators) = extractFunctionDefs(
            statements
        )

        # the list of 'def' statements and 'Lambda' expressions. each one engenders a function type.
        self.functionDefs = functionDefs + freeLambdas
        self.generators = generators
        self.comprehensions = comprehensions

        # all 'def' operations that are assigned exactly once. These defs are special
        # because we just assume that the binding is active without even evaluating the
        # def. Other bindings (lambdas, etc), require us to track slots for the closure itself
        self.functionDefsAssignedOnce = {
            fd.name: fd for fd in functionDefs if fd.name in self.variablesAssignedOnlyOnce
        }

        # add any lambdas that get assigned exactly once.
        for name, lambdaFunc in assignedLambdas:
            if name in self.variablesAssignedOnlyOnce:
                self.functionDefsAssignedOnce[name] = lambdaFunc
            self.functionDefs.append(lambdaFunc)

        # the current _type_ that we're using for this def,
        self.functionDefToType = {}

        # variables in closure slots that are not single-assignment function defs need slots
        self.variablesNeedingClosureSlots = set(
            [c for c in self.variablesReadByClosures if c not in self.functionDefsAssignedOnce]
        )

        self._constructInitialVarnameToType()

    @property
    def isGenerator(self):
        return self._bodyHasYieldStatements > 0

    def extractStatements(self, pyast):
        """Given the AST we're converting, extract the statemets we're actually going to convert.

        Args:
            pyast - the FunctionDef, Lambda, etc. that we're converting
        Returns:
            a list of pythnn_ast.Statement objects
        """
        if isinstance(pyast, python_ast.Statement.FunctionDef):
            return pyast.body
        else:
            return [
                python_ast.Statement.Return(
                    value=pyast.body,
                    line_number=pyast.body.line_number,
                    col_offset=pyast.body.col_offset,
                    filename=pyast.body.filename,
                )
            ]

    def processYieldExpression(self, expr):
        """Called with the body of a yield statement so that subclasses can handle.

        expr is an expression of type self._varname_to_type[FunctionYield].

        We return a native expression handling the result. Flow must return.
        """

        # in the base class, we just drop this on the floor - we still generate
        # code for the function, but just to figure out what 'FunctionYield' will
        # be. Then at the end of function generation we discard the actual code and
        # produce the actual generator object
        pass

    def convert_function_body(self, variableStates: FunctionStackState):
        return self.convert_statement_list_ast(
            rewriteForLoops(self._statements) if self.isGenerator else self._statements,
            variableStates,
            ControlFlowBlocks(),
            toplevel=True,
        )

    def convert_delete(self, expression, variableStates):
        """Convert the target of a 'del' statement.

        Args:
            expression - a python_ast Expression

        Returns:
            a pair of native_ast.Expression and a bool indicating whether control flow
            returns to the caller.
        """
        expr_context = ExpressionConversionContext(self, variableStates)

        if expression.matches.Subscript:
            slicing = expr_context.convert_expression_ast(expression.value)
            if slicing is None:
                return expr_context.finalize(None), False

            index = expr_context.convert_expression_ast(expression.slice)

            if slicing is None:
                return expr_context.finalize(None), False

            res = slicing.convert_delitem(index)

            return expr_context.finalize(None), res is not None
        elif expression.matches.Attribute:
            slicing = expr_context.convert_expression_ast(expression.value)
            attr = expression.attr
            if attr is None:
                return expr_context.finalize(None), False

            res = slicing.convert_set_attribute(attr, None)
            return expr_context.finalize(None), res is not None
        else:
            expr_context.pushException(Exception, "Can't delete this")
            return expr_context.finalize(None), False

    def convert_assignment(self, target, op, val_to_store):
        subcontext = val_to_store.context

        if target.matches.Name and target.ctx.matches.Store:
            varname = target.id

            if varname not in self._varname_to_type:
                self._varname_to_type[varname] = None

            if op is not None:
                slot_ref = subcontext.namedVariableLookup(varname)
                if slot_ref is None:
                    return False

                val_to_store = slot_ref.convert_bin_op(op, val_to_store, True)

                if val_to_store is None:
                    return False

            self.assignToLocalVariable(varname, val_to_store, subcontext.variableStates)

            return True

        if target.matches.Subscript and target.ctx.matches.Store:
            slicing = subcontext.convert_expression_ast(target.value)
            if slicing is None:
                return False

            # we are assuming this is an index. We ought to be checking this
            # and doing something else if it's a Slice or an Ellipsis or whatnot
            index = subcontext.convert_expression_ast(target.slice)

            if index is None:
                return False

            if op is not None:
                getItem = slicing.convert_getitem(index)
                if getItem is None:
                    return False

                val_to_store = getItem.convert_bin_op(op, val_to_store, True)
                if val_to_store is None:
                    return False

            slicing.convert_setitem(index, val_to_store)
            return True

        if target.matches.Attribute and target.ctx.matches.Store:
            slicing = subcontext.convert_expression_ast(target.value)
            attr = target.attr

            if op is not None:
                input_val = slicing.convert_attribute(attr)
                if input_val is None:
                    return False

                val_to_store = input_val.convert_bin_op(op, val_to_store, True)
                if val_to_store is None:
                    return False

            slicing.convert_set_attribute(attr, val_to_store)
            return True

        if target.matches.Tuple and target.ctx.matches.Store and op is None:
            return self.convert_tuple_assign(target.elts, val_to_store)

        assert False, target

    def convert_tuple_assign(self, targets, val_to_store):
        subcontext = val_to_store.context
        variableStates = subcontext.variableStates

        iterated_expressions = val_to_store.get_iteration_expressions()

        if iterated_expressions is not None:
            if len(iterated_expressions) < len(targets):
                subcontext.pushException(
                    ValueError,
                    f"not enough values to unpack (expected {len(targets)}, got {len(iterated_expressions)})",
                )
                return False
            elif len(iterated_expressions) > len(targets):
                subcontext.pushException(ValueError, f"too many values to unpack (expected {len(targets)})")
                return False

            for i in range(len(iterated_expressions)):
                if not self.convert_assignment(targets[i], None, iterated_expressions[i]):
                    return False

            return True
        else:
            # create a variable to hold the iterator, and instantiate it there
            iter_varname = f".anonymous_iter.{targets[0].line_number}"

            # we are going to assign this
            iterator_object = val_to_store.convert_method_call("__iter__", (), {})
            if iterator_object is None:
                return False

            self.assignToLocalVariable(iter_varname, iterator_object, variableStates)
            iter_obj = subcontext.namedVariableLookup(iter_varname)

            tempVarnames = []

            for targetIndex in range(len(targets) + 1):
                next_ptr = iter_obj.convert_fastnext()

                if next_ptr is not None:
                    with subcontext.ifelse(next_ptr.nonref_expr) as (if_true, if_false):
                        if targetIndex < len(targets):
                            with if_true:
                                tempVarnames.append(f".anonyous_iter{targets[0].line_number}.{targetIndex}")
                                self.assignToLocalVariable(
                                    tempVarnames[-1], next_ptr.asReference(), variableStates
                                )

                            with if_false:
                                subcontext.pushException(
                                    ValueError,
                                    f"not enough values to unpack (expected {len(targets)}, got {targetIndex})",
                                )
                        else:
                            with if_true:
                                subcontext.pushException(
                                    ValueError, f"too many values to unpack (expected {len(targets)})"
                                )
                else:
                    return False

            for targetIndex in range(len(targets)):
                self.convert_assignment(
                    targets[targetIndex], None, subcontext.namedVariableLookup(tempVarnames[targetIndex])
                )

            return True

    @property
    def exception_occurred_slot(self):
        exception_occurred_name = ".exc_occurred"
        return native_ast.Expression.StackSlot(name=exception_occurred_name, type=native_ast.Bool)

    def convert_statement_ast(self, ast, variableStates: FunctionStackState, controlFlowBlocks):
        """Convert a single statement to native_ast.

        Args:
            ast - the python_ast.Statement to convert
            variableStates - a description of what's known about the types of our variables.
                This data structure will be _modified_ by the calling code to include what's
                known about the types of values when control flow leaves this statement.
            controlFlowBlocks - info about where control flow should return next
        Returns:
            a pair (native_ast.Expression, flowReturns) giving an expression representing the
            statement in native code, and a boolean indicating whether control flow might
            return to the caller. If false, then we can assume that the code throws an
            exception or 'returns' from the function.
        """

        try:
            return self._convert_statement_ast(ast, variableStates, controlFlowBlocks)
        except Exception as e:
            types = {
                varname: self._varname_to_type.get(varname)
                for varname in set(computeReadVariables(ast)).union(computeAssignedVariables(ast))
            }

            newMessage = f"\n{ast.filename}:{ast.line_number}\n" + "\n".join(
                f"    {k}={v}" for k, v in types.items()
            )
            if e.args:
                e.args = (str(e.args[0]) + newMessage,)
            else:
                e.args = (newMessage,)
            raise

    def _convert_statement_ast(self, ast, variableStates: FunctionStackState, controlFlowBlocks):
        """same as 'convert_statement_ast'."""

        if ast.matches.Expr and ast.value.matches.Str:
            return native_ast.Expression(), True

        if ast.matches.Expr and ast.value.matches.Yield:
            # for the moment we don't support co-routines, so this is where
            # yield handling happens
            subcontext = ExpressionConversionContext(self, variableStates)

            if ast.value.value is None:
                e = subcontext.constant(None)
            else:
                e = subcontext.convert_expression_ast(ast.value.value)

            if e is None:
                return subcontext.finalize(None, exceptionsTakeFrom=ast), False

            if not self._functionYieldTypeKnown:
                self.upsizeVariableType(FunctionYield, e.expr_type)

            if e.expr_type != self._varname_to_type[FunctionYield]:
                e = e.convert_to_type(
                    self._varname_to_type[FunctionYield], ConversionLevel.ImplicitContainers
                )

                if e is None:
                    return subcontext.finalize(None, exceptionsTakeFrom=ast), False

            return (subcontext.finalize(self.processYieldExpression(e), exceptionsTakeFrom=ast), True)

        if ast.matches.AugAssign:
            subcontext = ExpressionConversionContext(self, variableStates)
            val_to_store = subcontext.convert_expression_ast(ast.value)

            if val_to_store is None:
                return subcontext.finalize(None, exceptionsTakeFrom=ast), False

            succeeds = self.convert_assignment(ast.target, ast.op, val_to_store)

            return subcontext.finalize(None, exceptionsTakeFrom=ast), succeeds

        if ast.matches.AnnAssign:
            subcontext = ExpressionConversionContext(self, variableStates)

            if ast.value is None:
                # this is a no-op
                return subcontext.finalize(None), True

            val_to_store = subcontext.convert_expression_ast(ast.value)

            if val_to_store is None:
                return subcontext.finalize(None, exceptionsTakeFrom=ast), False

            succeeds = self.convert_assignment(ast.target, None, val_to_store)

            return subcontext.finalize(None, exceptionsTakeFrom=ast), succeeds

        if ast.matches.Assign:
            if (
                len(ast.targets) == 1
                and ast.targets[0].matches.Name
                and ast.value.matches.Lambda
                and ast.targets[0].id in self.variablesAssignedOnlyOnce
            ):
                # this is like a 'def'
                return native_ast.Expression(), True

            subcontext = ExpressionConversionContext(self, variableStates)

            val_to_store = subcontext.convert_expression_ast(ast.value)

            if val_to_store is None:
                return subcontext.finalize(None, exceptionsTakeFrom=ast), False

            if len(ast.targets) == 1:
                succeeds = self.convert_assignment(ast.targets[0], None, val_to_store)
                return subcontext.finalize(None, exceptionsTakeFrom=ast), succeeds
            else:
                for target in ast.targets:
                    succeeds = self.convert_assignment(target, None, val_to_store)

                    if not succeeds:
                        return subcontext.finalize(None, exceptionsTakeFrom=ast), False

                return subcontext.finalize(None, exceptionsTakeFrom=ast), True

        if ast.matches.Return:
            subcontext = ExpressionConversionContext(self, variableStates)

            if ast.value is None:
                e = subcontext.constant(None)
            else:
                e = subcontext.convert_expression_ast(ast.value)

            if e is None:
                return subcontext.finalize(None, exceptionsTakeFrom=ast), False

            if self.isGenerator:
                subcontext.pushException(StopIteration, e)
                return subcontext.finalize(None), False

            if not self._functionOutputTypeKnown:
                self.upsizeVariableType(FunctionOutput, e.expr_type)

            if e.expr_type != self._varname_to_type[FunctionOutput]:
                e = e.convert_to_type(
                    self._varname_to_type[FunctionOutput], ConversionLevel.ImplicitContainers
                )

            if e is None:
                return subcontext.finalize(None, exceptionsTakeFrom=ast), False

            if controlFlowBlocks.return_to is not None:
                if controlFlowBlocks.try_flow:
                    subcontext.pushEffect(
                        controlFlowBlocks.try_flow.store(native_ast.const_int_expr(CONTROL_FLOW_RETURN))
                    )
                self.assignToLocalVariable(".return_value", e, variableStates)

                subcontext.pushEffect(
                    native_ast.Expression.Branch(
                        cond=self.exception_occurred_slot.load(), true=runtime_functions.clear_exc_info.call()
                    )
                )

                subcontext.pushTerminal(
                    native_ast.Expression.Return(arg=None, blockName=controlFlowBlocks.return_to)
                )

                return subcontext.finalize(None, exceptionsTakeFrom=ast), False
            else:
                subcontext.pushReturnValue(e)

                return subcontext.finalize(None, exceptionsTakeFrom=ast), False

        if ast.matches.Expr:
            subcontext = ExpressionConversionContext(self, variableStates)

            result_expr = subcontext.convert_expression_ast(ast.value)

            return subcontext.finalize(None, exceptionsTakeFrom=ast), result_expr is not None

        if ast.matches.If:
            cond_context = ExpressionConversionContext(self, variableStates)
            cond = cond_context.convert_expression_ast(ast.test)
            if cond is None:
                return cond_context.finalize(None, exceptionsTakeFrom=ast), False

            cond = cond.toBool()
            if cond is None:
                return cond_context.finalize(None, exceptionsTakeFrom=ast), False

            if cond.expr.matches.Constant:
                truth_val = cond.expr.val.truth_value()

                branch, flow_returns = self.convert_statement_list_ast(
                    ast.body if truth_val else ast.orelse, variableStates, controlFlowBlocks
                )

                return cond_context.finalize(None, exceptionsTakeFrom=ast) >> branch, flow_returns

            variableStatesTrue = variableStates.clone()
            variableStatesFalse = variableStates.clone()

            self.restrictByCondition(variableStatesTrue, ast.test, result=True)
            self.restrictByCondition(variableStatesFalse, ast.test, result=False)

            true, true_returns = self.convert_statement_list_ast(
                ast.body, variableStatesTrue, controlFlowBlocks
            )
            false, false_returns = self.convert_statement_list_ast(
                ast.orelse, variableStatesFalse, controlFlowBlocks
            )

            variableStates.becomeMerge(
                variableStatesTrue if true_returns else None, variableStatesFalse if false_returns else None
            )

            return (
                native_ast.Expression.Branch(
                    cond=cond_context.finalize(cond.nonref_expr, exceptionsTakeFrom=ast),
                    true=true,
                    false=false,
                ),
                true_returns or false_returns,
            )

        if ast.matches.Pass:
            return native_ast.nullExpr, True

        if ast.matches.While:
            while True:
                # track the initial variable states
                initVariableStates = variableStates.clone()

                cond_context = ExpressionConversionContext(self, variableStates)

                cond = cond_context.convert_expression_ast(ast.test)
                if cond is None:
                    return cond_context.finalize(None, exceptionsTakeFrom=ast), False

                cond = cond.toBool()
                if cond is None:
                    return cond_context.finalize(None, exceptionsTakeFrom=ast), False

                if cond.expr.matches.Constant:
                    truth_value = cond.expr.val.truth_value()

                    if not truth_value:
                        branch, flow_returns = self.convert_statement_list_ast(
                            ast.orelse, variableStates, controlFlowBlocks
                        )
                        return cond_context.finalize(None, exceptionsTakeFrom=ast) >> branch, flow_returns
                    else:
                        isWhileTrue = True
                else:
                    isWhileTrue = False

                variableStatesTrue = variableStates.clone()
                variableStatesFalse = variableStates.clone()

                self.restrictByCondition(variableStatesTrue, ast.test, result=True)
                self.restrictByCondition(variableStatesFalse, ast.test, result=False)

                true, true_returns = self.convert_statement_list_ast(
                    ast.body, variableStatesTrue, controlFlowBlocks.pushLoop()
                )

                if isWhileTrue:
                    if "loop_break" in true.returnTargets():
                        isWhileTrue = False

                false, false_returns = self.convert_statement_list_ast(
                    ast.orelse, variableStatesFalse, controlFlowBlocks
                )

                variableStates.becomeMerge(
                    variableStatesTrue if true_returns else None,
                    variableStatesFalse if false_returns else None,
                )

                variableStates.mergeWithSelf(initVariableStates)

                if variableStates == initVariableStates:
                    return (
                        native_ast.Expression.While(
                            cond=cond_context.finalize(cond.nonref_expr, exceptionsTakeFrom=ast),
                            while_true=true.withReturnTargetName("loop_continue"),
                            orelse=false,
                        ).withReturnTargetName("loop_break"),
                        (true_returns or false_returns) and not isWhileTrue,
                    )

        if ast.matches.Try:
            # .exception_occurred turns on once any exception occurs
            # .control_flow indicates the control flow instruction that is deferred until after the 'finally' block
            #   0=default, 1=unhandled exception, 2=break, 3=continue, 4=return
            control_flow_name = f".control_flow{ast.line_number}.{ast.col_offset}"
            end_of_try_marker = f"end_of_try{ast.line_number}.{ast.col_offset}"
            end_of_finally_marker = f"end_of_finally{ast.line_number}.{ast.col_offset}"
            exception_occurred = self.exception_occurred_slot
            control_flow = native_ast.Expression.StackSlot(name=control_flow_name, type=native_ast.Int64)

            # the state of our variables going into the Try
            variableStatesInitial = variableStates.clone()

            # the state of the variables going into any handlers could be any of the states
            # we were in at any point before an exception was thrown. Rather than try to
            # reason about this precisely, we simply remove any information on any variables
            # that are assigned within the try body
            variableStatesGoingIntoHandler = variableStatesInitial.clone()
            for varname in computeAssignedVariables(ast.body):
                if varname in self._varname_to_type:
                    # if the varname is assigned but not in _varname_to_type, then the
                    # assignment is not reachable.
                    variableStatesGoingIntoHandler.markVariableStateUnknown(
                        varname,
                        self._varname_to_type[varname].typeRepresentation
                    )

            body_context = ExpressionConversionContext(self, variableStates)

            body, body_returns = self.convert_statement_list_ast(
                ast.body, variableStates, controlFlowBlocks.pushTryCatch(end_of_try_marker, control_flow)
            )

            body_context.pushEffect(body)

            handlers_context = ExpressionConversionContext(self, variableStates)

            working_context = ExpressionConversionContext(self, variableStates)

            working = runtime_functions.catch_exception.call() >> control_flow.store(
                native_ast.const_int_expr(CONTROL_FLOW_EXCEPTION)
            )
            working_returns = True

            for h in reversed(ast.handlers):
                variableStatesHandler = variableStatesGoingIntoHandler.clone()

                cond_context = ExpressionConversionContext(self, variableStatesHandler)

                if h.type is None:
                    exc_match = BaseException
                    exc_type = BaseException
                else:
                    if h.type.matches.Tuple:
                        # TODO: figure out compile-time Value for tuple of constant objects
                        types = [
                            cond_context.convert_expression_ast(elt).expr_type.typeRepresentation.Value
                            for elt in h.type.elts
                        ]
                        exc_type = tuple(types)
                        # exc_type = OneOf(*[Value(t) for t in types])
                        exc_type = BaseException
                    else:
                        exc_match = cond_context.convert_expression_ast(
                            h.type
                        ).expr_type.typeRepresentation.Value
                        exc_type = exc_match
                cond = cond_context.matchExceptionObject(exc_match)

                handler_context = ExpressionConversionContext(self, variableStatesHandler)
                if h.name is None:
                    handler_context.pushEffect(runtime_functions.catch_exception.call())
                else:
                    h_name = h.name
                    self.assignToLocalVariable(
                        h_name, handler_context.fetchExceptionObject(exc_type), variableStatesHandler
                    )

                handler, handler_returns = self.convert_statement_list_ast(
                    h.body,
                    variableStatesHandler,
                    controlFlowBlocks.pushTryCatch(end_of_try_marker, control_flow),
                )

                if h.name is not None:
                    cleanup_context = ExpressionConversionContext(self, variableStatesHandler)
                    self.localVariableExpression(cleanup_context, h_name).convert_destroy()
                    cleanup_context.markVariableNotInitialized(h_name)
                    handler = native_ast.Expression.Finally(
                        expr=handler,
                        teardowns=[
                            native_ast.Teardown.Always(
                                expr=cleanup_context.finalize(None).with_comment(f"Cleanup for {h_name}")
                            )
                        ],
                    )
                    variableStatesHandler.variableUninitialized(h_name)

                if handler_returns:
                    variableStates.becomeMerge(variableStates.clone(), variableStatesHandler)

                handler = handler >> native_ast.Expression.Branch(
                    cond=exception_occurred.load(), true=runtime_functions.clear_exc_info.call()
                )

                working = native_ast.Expression.Branch(
                    cond=cond_context.finalize(cond.nonref_expr),
                    true=handler_context.finalize(handler),
                    false=working_context.finalize(working),
                )

                working_returns = handler_returns or working_returns

            handlers_context.pushEffect(working_context.finalize(working))

            orelse_context = ExpressionConversionContext(self, variableStates)
            if len(ast.orelse) > 0:
                orelse, orelse_returns = self.convert_statement_list_ast(
                    ast.orelse,
                    variableStates,
                    controlFlowBlocks.pushTryCatch(end_of_try_marker, control_flow),
                )
                orelse_context.pushEffect(orelse)
            else:
                orelse, orelse_returns = None, True

            final_context = ExpressionConversionContext(self, variableStates)
            if ast.finalbody is not None:
                final, final_returns = self.convert_statement_list_ast(
                    ast.finalbody,
                    variableStates,
                    controlFlowBlocks.pushTryCatch(end_of_finally_marker, control_flow),
                )
                final = final.withReturnTargetName(end_of_finally_marker)
                final_returns = True
                final_context.pushEffect(final)
            else:
                final, final_returns = None, True

            complete = (
                exception_occurred.store(native_ast.falseExpr)
                >> control_flow.store(native_ast.const_int_expr(CONTROL_FLOW_DEFAULT))
                >> native_ast.Expression.TryCatch(
                    expr=body_context.finalize(None),
                    handler=exception_occurred.store(native_ast.trueExpr)
                    >> native_ast.Expression.TryCatch(
                        expr=handlers_context.finalize(None),
                        handler=runtime_functions.catch_exception.call()
                        >> control_flow.store(native_ast.const_int_expr(CONTROL_FLOW_EXCEPTION)),
                    ),
                )
            )
            if orelse:
                complete = complete >> native_ast.Expression.Branch(
                    cond=exception_occurred.load(),
                    false=native_ast.Expression.TryCatch(
                        expr=orelse_context.finalize(None) >> native_ast.nullExpr,
                        handler=exception_occurred.store(native_ast.trueExpr)
                        >> runtime_functions.catch_exception.call()
                        >> control_flow.store(native_ast.const_int_expr(CONTROL_FLOW_EXCEPTION)),
                    ),
                )

            complete = complete.withReturnTargetName(end_of_try_marker)

            if final:
                complete = complete >> native_ast.Expression.TryCatch(
                    expr=final_context.finalize(None),
                    handler=exception_occurred.store(native_ast.trueExpr)
                    >> runtime_functions.catch_exception.call()
                    >> control_flow.store(native_ast.const_int_expr(CONTROL_FLOW_EXCEPTION)),
                )

            if self.isLocalVariable(".return_value"):
                variableStatesReturnBlock = variableStates.clone()

                return_context = ExpressionConversionContext(self, variableStatesReturnBlock)
                rtn, _ = self.convert_statement_ast(
                    python_ast.Statement.Return(value=python_ast.Expr.Name(id=".return_value")),
                    variableStatesReturnBlock,
                    controlFlowBlocks,
                )

                complete = complete >> native_ast.Expression.Branch(
                    cond=control_flow.load().eq(CONTROL_FLOW_RETURN),
                    true=native_ast.Expression.Branch(
                        cond=exception_occurred.load(), true=runtime_functions.clear_exc_info.call()
                    )
                    >> return_context.finalize(rtn),
                )

            if controlFlowBlocks.in_loop_ever:
                break_context = ExpressionConversionContext(self, variableStates)
                brk, _ = self.convert_statement_ast(
                    python_ast.Statement.Break(), variableStates, controlFlowBlocks
                )
                complete = complete >> native_ast.Expression.Branch(
                    cond=control_flow.load().eq(CONTROL_FLOW_BREAK),
                    true=native_ast.Expression.Branch(
                        cond=exception_occurred.load(), true=runtime_functions.clear_exc_info.call()
                    )
                    >> break_context.finalize(brk),
                )

                cont_context = ExpressionConversionContext(self, variableStates)
                cont, _ = self.convert_statement_ast(
                    python_ast.Statement.Continue(), variableStates, controlFlowBlocks
                )
                complete = complete >> native_ast.Expression.Branch(
                    cond=control_flow.load().eq(CONTROL_FLOW_CONTINUE),
                    true=native_ast.Expression.Branch(
                        cond=exception_occurred.load(), true=runtime_functions.clear_exc_info.call()
                    )
                    >> cont_context.finalize(cont),
                )

            raise_context = ExpressionConversionContext(self, variableStates)
            raise_context.pushExceptionObject(None, clear_exc=True)
            complete = complete >> native_ast.Expression.Branch(
                cond=control_flow.load().eq(CONTROL_FLOW_EXCEPTION), true=raise_context.finalize(None)
            )

            return (complete, ((body_returns and orelse_returns) or working_returns) and final_returns)

        if ast.matches.For:
            context = ExpressionConversionContext(self, variableStates)

            to_iterate = context.convert_expression_ast(ast.iter)
            if to_iterate is None:
                return context.finalize(None, exceptionsTakeFrom=ast), False

            if isinstance(to_iterate.expr_type.typeRepresentation, type) and issubclass(
                to_iterate.expr_type.typeRepresentation, OneOf
            ):
                # split the code on the different possible 'oneof' values
                if not to_iterate.isReference:
                    to_iterate = context.pushMove(to_iterate)

                subExprs = []
                anyFlowsReturn = False
                subVariableStates = []

                for ix in range(len(to_iterate.expr_type.typeRepresentation.Types)):
                    subVS = variableStates.clone()
                    subcontext = ExpressionConversionContext(self, subVS)

                    expr, flowReturns = self.convert_iteration_expression(
                        to_iterate.refAs(ix).changeContext(subcontext), ast, "." + str(ix), controlFlowBlocks
                    )

                    subExprs.append(expr)
                    if flowReturns:
                        anyFlowsReturn = True

                    subVariableStates.append(subVS)

                switchExpr = subExprs[-1]
                for ix in reversed(range(len(subExprs) - 1)):
                    switchExpr = native_ast.Expression.Branch(
                        cond=to_iterate.expr_type.convert_which_native(to_iterate.expr)
                        .cast(native_ast.Int64)
                        .eq(native_ast.const_int_expr(ix)),
                        true=subExprs[ix],
                        false=switchExpr,
                    )

                variableStates.becomeMergeOf(subVariableStates)

                return context.finalize(switchExpr, exceptionsTakeFrom=ast), anyFlowsReturn
            else:
                return self.convert_iteration_expression(to_iterate, ast, "", controlFlowBlocks)

        if ast.matches.Raise:
            expr_context = ExpressionConversionContext(self, variableStates)

            if ast.exc is None:
                toThrow = None  # means reraise
            else:
                toThrow = expr_context.convert_expression_ast(ast.exc)

                if toThrow is None:
                    return (
                        expr_context.finalize(None, exceptionsTakeFrom=None if ast.exc is None else ast),
                        False,
                    )

                toThrow = toThrow.convert_to_type(object, ConversionLevel.Signature)
                if toThrow is None:
                    return (
                        expr_context.finalize(None, exceptionsTakeFrom=None if ast.exc is None else ast),
                        False,
                    )

            if ast.cause is None:
                expr_context.pushExceptionObject(toThrow)
            else:
                excCause = expr_context.convert_expression_ast(ast.cause)
                expr_context.pushExceptionObjectWithCause(toThrow, excCause)

            return expr_context.finalize(None, exceptionsTakeFrom=None if ast.exc is None else ast), False

        if ast.matches.Delete:
            exprs = []
            for target in ast.targets:
                subExprs, flowReturns = self.convert_delete(target, variableStates)
                exprs.append(subExprs)

                if not flowReturns:
                    return native_ast.makeSequence(exprs), flowReturns
            return native_ast.makeSequence(exprs), True

        if ast.matches.With:
            statements = expandWithBlockIntoTryCatch(ast)

            return self.convert_statement_list_ast(statements, variableStates, controlFlowBlocks)

        if ast.matches.Import:
            # TODO: correctly model the side-effectfulness of imports
            context = ExpressionConversionContext(self, variableStates)

            for alias in ast.names:
                if alias.asname is None and len(alias.name.split(".")) > 1:
                    module = importlib.import_module(alias.name.split("."))[0]
                    target = alias.name.split(".")
                elif alias.asname is None:
                    module = importlib.import_module(alias.name)
                    target = alias.name
                else:
                    module = importlib.import_module(alias.name)
                    target = alias.asname

                self.assignToLocalVariable(target, context.constant(module), variableStates)

            return context.finalize(None, exceptionsTakeFrom=ast), True

        if ast.matches.Break:
            if controlFlowBlocks.in_loop:
                return native_ast.Expression.Return(blockName="loop_break"), True

            assert controlFlowBlocks.return_to is not None

            if controlFlowBlocks.try_flow:
                return (
                    controlFlowBlocks.try_flow.store(native_ast.const_int_expr(CONTROL_FLOW_BREAK))
                    >> native_ast.Expression.Branch(
                        cond=self.exception_occurred_slot.load(), true=runtime_functions.clear_exc_info.call()
                    )
                    >> native_ast.Expression.Return(blockName=controlFlowBlocks.return_to),
                    True,
                )
            else:
                return native_ast.Expression.Return(blockName=controlFlowBlocks.return_to), True

        if ast.matches.Continue:
            if controlFlowBlocks.in_loop:
                return native_ast.Expression.Return(blockName="loop_continue"), True

            assert controlFlowBlocks.return_to is not None

            if controlFlowBlocks.try_flow:
                return (
                    controlFlowBlocks.try_flow.store(native_ast.const_int_expr(CONTROL_FLOW_CONTINUE))
                    >> native_ast.Expression.Branch(
                        cond=self.exception_occurred_slot.load(), true=runtime_functions.clear_exc_info.call()
                    )
                    >> native_ast.Expression.Return(blockName=controlFlowBlocks.return_to),
                    True,
                )
            else:
                return native_ast.Expression.Return(blockName=controlFlowBlocks.return_to), True

        if ast.matches.Assert:
            expr_context = ExpressionConversionContext(self, variableStates)

            testExpr = expr_context.convert_expression_ast(ast.test)

            if testExpr is not None:
                testExpr = testExpr.toBool()

            if testExpr is None:
                return expr_context.finalize(None, exceptionsTakeFrom=ast), False

            definitelyFails = testExpr.expr.matches.Constant and testExpr.expr.val.truth_value() is False

            with expr_context.ifelse(testExpr) as (ifTrue, ifFalse):
                with ifFalse:
                    if ast.msg is None:
                        expr_context.pushException(AssertionError)
                    else:
                        msgExpr = expr_context.convert_expression_ast(ast.msg)

                        if msgExpr is not None:
                            expr_context.pushException(AssertionError, msgExpr)

            return expr_context.finalize(None, exceptionsTakeFrom=ast), not definitelyFails

        if ast.matches.FunctionDef:
            if ast.name in self.functionDefsAssignedOnce:
                # this is a no-op. for performance reasons, we assume that the function
                # is valid (and are OK with not throwing an exception even though
                # that's a deviation from normal python), because otherwise we would
                # have to be checking in a slot every time we want to access this function
                return native_ast.nullExpr, True

            context = ExpressionConversionContext(self, variableStates)

            res = self.localVariableExpression(context, ".closure").changeType(self.functionDefToType[ast])

            self.assignToLocalVariable(ast.name, res, variableStates)

            return context.finalize(None, exceptionsTakeFrom=ast), True

        # if ast.matches.ImportFrom:
        # if ast.matches.ClassDef:
        # if ast.matches.Global:
        # if ast.matches.AsyncFunctionDef:
        # if ast.matches.AsyncWith:
        # if ast.matches.AsyncFor:
        # if ast.matches.NonLocal:

        raise ConversionException("Can't handle python ast Statement.%s" % ast.Name)

    def convert_iteration_expression(self, to_iterate, ast, variableSuffix, controlFlowBlocks):
        """Convert the 'For' statement in 'ast', where to_iterate is the iterable."""
        context = to_iterate.context
        variableStates = context.variableStates

        iteration_expressions = to_iterate.get_iteration_expressions()

        # we allow types to explicitly break themselves down into a fixed set of
        # expressions to unroll, so that we can retain typing information.
        if iteration_expressions is not None:
            for subexpr in iteration_expressions:
                self.convert_assignment(ast.target, None, subexpr)

                thisOne, thisOneReturns = self.convert_statement_list_ast(
                    ast.body, variableStates, controlFlowBlocks.pushLoop()
                )

                # if we hit 'continue', just come to the end of this expression
                thisOne = thisOne.withReturnTargetName("loop_continue")

                context.pushEffect(thisOne)

                if not thisOneReturns:
                    return (
                        context.finalize(None, exceptionsTakeFrom=ast).withReturnTargetName("loop_break"),
                        False,
                    )

            thisOne, thisOneReturns = self.convert_statement_list_ast(
                ast.orelse, variableStates, controlFlowBlocks
            )

            context.pushEffect(thisOne)

            wholeLoopExpr = context.finalize(None, exceptionsTakeFrom=ast)

            wholeLoopExpr = wholeLoopExpr.withReturnTargetName("loop_break")

            return wholeLoopExpr, thisOneReturns

        if to_iterate.expr_type.has_intiter():
            # execute the 'intiter' fastpath for iterators. Classes where iteration
            # is simply enumerating a list of values indexed by an integer can
            # be replaced with a simple loop, which downstream compilation steps
            # can see through better. Eventually, we'd like to be able to pull apart
            # everything we're doing with classes for optimization purposes,
            # but at the moment, this is more expedient.
            iter_varname = ".iterate_over." + str(ast.line_number) + variableSuffix

            self.assignToLocalVariable(iter_varname, to_iterate, variableStates)

            inner, innerReturns = self.convert_statement_list_ast(
                list(rewriteIntiterForLoop(iter_varname, ast.target, ast.body, ast.orelse)),
                variableStates,
                controlFlowBlocks,
            )

            return context.finalize(inner, exceptionsTakeFrom=ast), innerReturns

        # create a variable to hold the iterator, and instantiate it there
        iter_varname = ".iter." + str(ast.line_number) + variableSuffix

        iterator_object = to_iterate.convert_method_call("__iter__", (), {})
        if iterator_object is None:
            return context.finalize(None, exceptionsTakeFrom=ast), False

        self.assignToLocalVariable(iter_varname, iterator_object, variableStates)

        while True:
            # track the initial variable states
            initVariableStates = variableStates.clone()

            cond_context = ExpressionConversionContext(self, variableStates)

            iter_obj = cond_context.namedVariableLookup(iter_varname)
            if iter_obj is None:
                return (
                    context.finalize(None, exceptionsTakeFrom=ast)
                    >> cond_context.finalize(None, exceptionsTakeFrom=ast),
                    False,
                )

            next_ptr = iter_obj.convert_fastnext()
            if next_ptr is None:
                return (
                    context.finalize(None, exceptionsTakeFrom=ast)
                    >> cond_context.finalize(None, exceptionsTakeFrom=ast),
                    False,
                )

            with cond_context.ifelse(next_ptr.nonref_expr) as (if_true, if_false):
                with if_true:
                    self.convert_assignment(ast.target, None, next_ptr.asReference())

            variableStatesTrue = variableStates.clone()
            variableStatesFalse = variableStates.clone()

            true, true_returns = self.convert_statement_list_ast(
                ast.body, variableStatesTrue, controlFlowBlocks.pushLoop()
            )
            false, false_returns = self.convert_statement_list_ast(
                ast.orelse, variableStatesFalse, controlFlowBlocks.pushLoop()
            )

            variableStates.becomeMerge(
                variableStatesTrue if true_returns else None, variableStatesFalse if false_returns else None
            )

            variableStates.mergeWithSelf(initVariableStates)

            if variableStates == initVariableStates:
                # if nothing changed, the loop is stable.
                return (
                    context.finalize(None, exceptionsTakeFrom=ast)
                    >> native_ast.Expression.While(
                        cond=cond_context.finalize(next_ptr.nonref_expr, exceptionsTakeFrom=ast),
                        while_true=true.withReturnTargetName("loop_continue"),
                        orelse=false,
                    ).withReturnTargetName("loop_break"),
                    true_returns or false_returns,
                )

    def checkIfStatementIsSplitOnType(self, statement, variableStates):
        """Check if 'statement' is 'checkType(x, t1, t2, ...)' for some variable 'x'

        If so, return the name of the variable 'x' and the expressions t1, t2.

        Otherwise return None.
        """
        if not statement.matches.Expr:
            return None, None

        expr = statement.value

        if not expr.matches.Call or len(expr.args) < 1 or expr.keywords:
            return None, None

        if expr.args[0].matches.Starred:
            return None, None

        if not expr.args[0].matches.Name or not expr.func.matches.Name:
            return None, None

        c = ExpressionConversionContext(self, variableStates)
        callRes = c.convert_expression_ast(expr.func)

        if callRes is None:
            return None, None

        if callRes.expr_type.typeRepresentation is not checkType:
            return None, None

        return expr.args[0].id, expr.args[1:]

    def checkIfStatementIsSplitOnOneOf(self, statement, variableStates):
        """Check if 'statement' is 'checkOneOfType(x)' for some variable 'x'

        If so, return the name of the variable 'x'. Otherwise return None.
        """
        if not statement.matches.Expr:
            return None

        expr = statement.value

        if not expr.matches.Call or len(expr.args) != 1 or expr.keywords:
            return None

        if expr.args[0].matches.Starred:
            return None

        if not expr.args[0].matches.Name or not expr.func.matches.Name:
            return None

        c = ExpressionConversionContext(self, variableStates)
        callRes = c.convert_expression_ast(expr.func)

        if callRes is None:
            return None

        if callRes.expr_type.typeRepresentation is not checkOneOfType:
            return None

        varname = expr.args[0].id

        curType = variableStates.currentType(varname)

        if curType is None or not issubclass(curType, OneOf):
            return None

        return varname

    def splitOnOneOfAndConvertStatementList(
        self, varToSplitOn, statements, variableStates, toplevel, controlFlowBlocks
    ):
        subExprs = []
        anyFlowsReturn = False
        subVariableStates = []

        varType = variableStates.currentType(varToSplitOn)
        assert issubclass(varType, OneOf)

        for ix in range(len(varType.Types)):
            subVS = variableStates.clone()

            subVS.restrictTypeFor(varToSplitOn, varType.Types[ix], True)

            expr, flowReturns = self.convert_statement_list_ast(statements, subVS, controlFlowBlocks)

            subExprs.append(expr)
            if flowReturns:
                anyFlowsReturn = True

            subVariableStates.append(subVS)

        context = ExpressionConversionContext(self, variableStates)
        toSplit = context.namedVariableLookup(varToSplitOn)

        switchExpr = subExprs[-1]
        for ix in reversed(range(len(subExprs) - 1)):
            switchExpr = native_ast.Expression.Branch(
                cond=toSplit.expr_type.convert_which_native(toSplit.expr)
                .cast(native_ast.Int64)
                .eq(native_ast.const_int_expr(ix)),
                true=subExprs[ix],
                false=switchExpr,
            )

        variableStates.becomeMergeOf(subVariableStates)

        return context.finalize(switchExpr), anyFlowsReturn

    def splitOnSubtypesAndConvertStatementList(
        self, varToSplitOn, subtypeExprs, statements, variableStates, toplevel, controlFlowBlocks
    ):
        """Evaluate 'statements' as if 'varToSplitOn' had been checked against each of subtypes.

        In practice, we generate something like

            if isinstance(varToSplitOn, subtypes[0]):
                statements
            elif isinstance(varToSplitOn, subtypes[1]):
                statements
            else:
                ...
        """
        outStatements = statements

        for typeExpr in subtypeExprs:
            checkExpr = codegen_helpers.makeCallExpr(
                python_ast.Expr.Constant(value=isinstance),
                python_ast.Expr.Name(id=varToSplitOn),
                typeExpr
            )

            outStatements = [
                python_ast.Statement.If(
                    test=checkExpr,
                    body=statements,
                    orelse=outStatements
                )
            ]

        return self.convert_statement_list_ast(
            outStatements,
            variableStates,
            controlFlowBlocks,
            toplevel=toplevel
        )

    def convert_statement_list_ast(
        self, statements, variableStates: FunctionStackState, controlFlowBlocks, *, toplevel=False
    ):
        """Convert a sequence of statements to a native expression.

        After executing this statement, variableStates will contain the known states of the
        current variables.

        Args:
            statements - a list of python_ast.Statement objects
            variableStates - a FunctionStackState object,
            toplevel - is this at the root of a function, so that flowing off the end should
                produce a Return expression?
            controlFlowBLocks - where control flow should return to

        Returns:
            a tuple (expr: native_ast.Expression, controlFlowReturns: bool) giving the result, and
            whether control flow returns to the invoking native code.
        """
        exprAndReturns = []

        for statementIx in range(len(statements)):
            s = statements[statementIx]

            varToSplitOn = self.checkIfStatementIsSplitOnOneOf(s, variableStates)

            if varToSplitOn is not None:
                expr, controlFlowReturns = self.splitOnOneOfAndConvertStatementList(
                    varToSplitOn,
                    statements[statementIx + 1:],
                    variableStates,
                    toplevel,
                    controlFlowBlocks
                )
                exprAndReturns.append((expr, controlFlowReturns))
                break

            varToSplitOn, typeExprs = self.checkIfStatementIsSplitOnType(s, variableStates)

            if varToSplitOn is not None:
                expr, controlFlowReturns = self.splitOnSubtypesAndConvertStatementList(
                    varToSplitOn,
                    typeExprs,
                    statements[statementIx + 1:],
                    variableStates,
                    toplevel,
                    controlFlowBlocks
                )
                exprAndReturns.append((expr, controlFlowReturns))
                break

            res = self.convert_statement_ast(s, variableStates, controlFlowBlocks)

            if not isinstance(res, tuple) or len(res) != 2:
                raise Exception(
                    f"convert_statement_ast is supposed to return a pair. It returned {res}. "
                    f"Statement type is {type(s)}"
                )
            expr, controlFlowReturns = res

            exprAndReturns.append((expr, controlFlowReturns))

            if not controlFlowReturns:
                break

        if not exprAndReturns or exprAndReturns[-1][1]:
            flows_off_end = True
        else:
            flows_off_end = False

        if toplevel and flows_off_end:
            exprAndReturns.append(self.handleFlowsOffEnd(variableStates))
            assert exprAndReturns[-1][1] is False

        seq_expr = native_ast.makeSequence([expr for expr, _ in exprAndReturns])

        return seq_expr, exprAndReturns[-1][1] if exprAndReturns else True

    def handleFlowsOffEnd(self, variableStates: FunctionStackState):
        """Generate code to handle the case where we exit the function normally.

        Returns:
            a tuple (native_ast, controlFlowReturns)

        controlFlowReturns must be False.
        """
        if self.isGenerator:
            return native_ast.Expression.Return(arg=None, blockName=None), False

        if not self._functionOutputTypeKnown:
            self.upsizeVariableType(FunctionOutput, NoneWrapper())

        return self.convert_statement_ast(
            python_ast.Statement.Return(value=None, filename="", line_number=0, col_offset=0),
            variableStates,
            ControlFlowBlocks(),
        )


class ComprehensionConversionContextBase(FunctionConversionContext):
    """Convert a generator function, but instead of generating, build a list.

    We generate an accumulator at the start of the function, replace all yields
    with an '.append' operation, and then at the end append the list.
    """

    @property
    def isGenerator(self):
        return False

    def comprehensionAccumulatorType(self):
        raise Exception("SubclassesOverride")

    def localVariableExpression(self, context: ExpressionConversionContext, name):
        if name == ".comprehension_accumulator":
            listCompType = self.comprehensionAccumulatorType()

            return TypedExpression(
                context,
                native_ast.Expression.StackSlot(name=name, type=listCompType.getNativeLayoutType()),
                listCompType,
                isReference=True,
            )

        return super().localVariableExpression(context, name)

    def functionVariableInitializations(self, variableStates):
        context = ExpressionConversionContext(self, variableStates)

        self.localVariableExpression(context, ".comprehension_accumulator").convert_default_initialize()

        return [context.finalize(None)]

    def generateDestructors(self, variableStates):
        destructors = super().generateDestructors(variableStates)

        context = ExpressionConversionContext(self, variableStates)
        accumulator = self.localVariableExpression(context, ".comprehension_accumulator")
        accumulator.convert_destroy()

        nativeDestructor = context.finalize(None)

        return destructors + [native_ast.Teardown.Always(expr=nativeDestructor)]

    def processYieldExpression(self, expr):
        """Called with the body of a yield statement so that subclasses can handle.

        expr is an expression of type self._varname_to_type[FunctionYield].

        We return a native expression handling the result. Flow must return.
        """
        raise Exception("Subclasses Implement")

    def handleFlowsOffEnd(self, variableStates: FunctionStackState):
        """Generate code to handle the case where we exit the function normally.

        Returns:
            a tuple (native_ast, controlFlowReturns)

        controlFlowReturns must be False.
        """
        assert not self._functionOutputTypeKnown

        subcontext = ExpressionConversionContext(self, variableStates)

        resExpr = self.localVariableExpression(subcontext, ".comprehension_accumulator").changeType(
            self.getMasqueradeType()
        )

        self.setVariableType(FunctionOutput, resExpr.expr_type)

        subcontext.pushReturnValue(resExpr)

        return subcontext.finalize(None, exceptionsTakeFrom=None), False


class ListComprehensionConversionContext(ComprehensionConversionContextBase):
    """Convert a generator function, but instead of generating, build a list.

    We generate an accumulator at the start of the function, replace all yields
    with an '.append' operation, and then at the end append the list.
    """

    def comprehensionAccumulatorType(self):
        if FunctionYield in self._varname_to_type:
            return typeWrapper(ListOf(self._varname_to_type[FunctionYield].typeRepresentation))
        else:
            return typeWrapper(ListOf(None))

    def processYieldExpression(self, expr):
        """Called with the body of a yield statement so that subclasses can handle.

        expr is an expression of type self._varname_to_type[FunctionYield].

        We return a native expression handling the result. Flow must return.
        """
        self.localVariableExpression(expr.context, ".comprehension_accumulator").convert_method_call(
            "append", [expr], {}
        )

    def getMasqueradeType(self):
        from typed_python.compiler.type_wrappers.typed_list_masquerading_as_list_wrapper import (
            TypedListMasqueradingAsList,
        )

        return TypedListMasqueradingAsList(self.comprehensionAccumulatorType().typeRepresentation)


class SetComprehensionConversionContext(ComprehensionConversionContextBase):
    """Convert a generator function, but instead of generating, build a set.

    We generate an accumulator at the start of the function, replace all yields
    with an '.append' operation, and then at the end append the list.
    """

    def comprehensionAccumulatorType(self):
        if FunctionYield in self._varname_to_type:
            return typeWrapper(Set(self._varname_to_type[FunctionYield].typeRepresentation))
        else:
            return typeWrapper(Set(None))

    def processYieldExpression(self, expr):
        """Called with the body of a yield statement so that subclasses can handle.

        expr is an expression of type self._varname_to_type[FunctionYield].

        We return a native expression handling the result. Flow must return.
        """
        self.localVariableExpression(expr.context, ".comprehension_accumulator").convert_method_call(
            "add", [expr], {}
        )

    def getMasqueradeType(self):
        from typed_python.compiler.type_wrappers.typed_set_masquerading_as_set_wrapper import (
            TypedSetMasqueradingAsSet,
        )

        return TypedSetMasqueradingAsSet(self.comprehensionAccumulatorType().typeRepresentation)


class DictComprehensionConversionContext(ComprehensionConversionContextBase):
    """Convert a generator function, but instead of generating, build a dict.

    We generate an accumulator at the start of the function, replace all yields
    (which must be pairs) with an assignment operation, and then at the end append the list.
    """

    def comprehensionAccumulatorType(self):
        if FunctionYield in self._varname_to_type:
            # the tuple wrapper will already jam together the various types
            # into a single output tuple because there is only one 'yield' statement
            from typed_python.compiler.type_wrappers.typed_tuple_masquerading_as_tuple_wrapper import (
                TypedTupleMasqueradingAsTuple,
            )

            T = self._varname_to_type[FunctionYield]

            assert isinstance(T, TypedTupleMasqueradingAsTuple)
            assert len(T.typeRepresentation.ElementTypes) == 2

            return typeWrapper(
                Dict(T.typeRepresentation.ElementTypes[0], T.typeRepresentation.ElementTypes[1])
            )
        else:
            return typeWrapper(Dict(None, None))

    def processYieldExpression(self, expr):
        """Called with the body of a yield statement so that subclasses can handle.

        expr is an expression of type self._varname_to_type[FunctionYield].

        We return a native expression handling the result. Flow must return.
        """
        self.localVariableExpression(expr.context, ".comprehension_accumulator").convert_setitem(
            expr.convert_getitem(expr.context.constant(0)), expr.convert_getitem(expr.context.constant(1))
        )

    def getMasqueradeType(self):
        from typed_python.compiler.type_wrappers.typed_dict_masquerading_as_dict_wrapper import (
            TypedDictMasqueradingAsDict,
        )

        return TypedDictMasqueradingAsDict(self.comprehensionAccumulatorType().typeRepresentation)
