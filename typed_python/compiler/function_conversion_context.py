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

import typed_python.python_ast as python_ast

from typed_python.compiler.python_ast_analysis import (
    computeAssignedVariables,
    computeReadVariables,
    computeFunctionArgVariables,
    computeVariablesAssignedOnlyOnce,
    computeVariablesReadByClosures,
    extractFunctionDefs
)
from typed_python.internals import makeFunctionType

import typed_python.compiler
import typed_python.compiler.native_ast as native_ast
from typed_python import _types, Type
import typed_python.compiler.type_wrappers.runtime_functions as runtime_functions
from typed_python.compiler.expression_conversion_context import ExpressionConversionContext
from typed_python.compiler.function_stack_state import FunctionStackState
from typed_python.compiler.type_wrappers.none_wrapper import NoneWrapper
from typed_python.compiler.type_wrappers.python_type_object_wrapper import PythonTypeObjectWrapper
from typed_python.compiler.typed_expression import TypedExpression
from typed_python.compiler.conversion_exception import ConversionException
from typed_python import OneOf, Function, Tuple, Forward, Class

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


class ConversionContextBase(object):
    """Helper function for converting a single python function given some input and output types"""

    def __init__(self, converter, name, identity, input_types, output_type,
                 funcArgNames, closureVarnames, globalVars, globalVarsRaw):
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
        self._native_args = None

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
            return (
                self.localVariableExpression(context, ".closure")
                .convert_attribute(name, nocheck=True)
            )

        if name != ".closure" and self._varname_to_type.get(name) is None:
            context.pushException(NameError, f"name '{name}' is not defined")
            return None

        if name == ".closure":
            slot_type = typeWrapper(self.closureType)
        else:
            slot_type = self._varname_to_type[name]

        return TypedExpression(
            context,
            native_ast.Expression.StackSlot(
                name=name,
                type=slot_type.getNativeLayoutType()
            ),
            slot_type,
            isReference=True
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

        typedFuncType = untypedFuncType.withClosureType(self.closureType)
        typedFuncType = typedFuncType.withOverloadVariableBindings(
            0,
            {name: self.closurePathToName(name)
             for name in untypedFuncType.overloads[0].closureVarLookups}
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
            issubclass(nativeType, Type), (nativeType, type(nativeType))
            or nativeType in (int, float, bool, str, bytes, type(None))
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
            for fd in self.functionDefs
        }

        # walk over the function defs and actually build them
        for ast in self.functionDefs:
            untypedFuncType, typedFuncType = self.computeTypeForFunctionDef(ast)

            self.functionDefToType[ast] = self.functionDefToType[ast].define(typedFuncType)
            self.typedFunctionTypeToClosurelessFunctionType[typedFuncType] = untypedFuncType

        if self.variablesNeedingClosureSlots:
            # now build the closure type itself and replace the forward with the defined class
            closureMembers = []

            replacedVarTypes = {}

            for var in sorted(self.variablesNeedingClosureSlots):
                if var in self._varname_to_type:
                    replacedVarTypes[var] = self.replaceClosureTypesIn(
                        self._varname_to_type[var].typeRepresentation
                    )

                    closureMembers.append((var, replacedVarTypes[var], None))

            memberFunctions = {
                '__init__':
                makeFunctionType('__init__', lambda self: None, classname=self.name + ".closure", assumeClosuresGlobal=True)
            }

            self.closureType = self.closureType.define(
                _types.Class(self.name + ".closure", (), True, tuple(closureMembers), tuple(memberFunctions.items()), (), (), ())
            )

            assert not _types.is_default_constructible(self.closureType)

            # now rebuild each of our var types. we have to do this after
            # setting 'closureType' because otherwise we'll end up with undefined
            # forwards floating around.
            for var in self._varname_to_type:
                if var in replacedVarTypes:
                    self._varname_to_type[var] = typeWrapper(replacedVarTypes[var])
                elif isinstance(self._varname_to_type[var], type):
                    self._varname_to_type[var] = typeWrapper(self.replaceClosureTypesIn(self._varname_to_type[var].typeRepresentation))

        _closureCycleMemo[closureKey] = (self.closureType, dict(self.functionDefToType))

    def convertToNativeFunction(self):
        self.tempLetVarIx = 0
        self._tempStackVarIx = 0
        self._tempIterVarIx = 0

        variableStates = FunctionStackState()

        self.buildClosureTypes()

        initializer_expr = self.initializeVariableStates(self._argnames, variableStates)

        body_native_expr, controlFlowReturns = self.convert_function_body(variableStates)

        # destroy our variables if they are in scope
        destructors = self.generateDestructors(variableStates)

        assert not controlFlowReturns

        body_native_expr = initializer_expr >> body_native_expr

        if destructors:
            body_native_expr = native_ast.Expression.Finally(
                teardowns=destructors,
                expr=body_native_expr
            )

        return_type = self._varname_to_type.get(FunctionOutput, None)

        if return_type is None:
            return (
                native_ast.Function(
                    args=self._native_args,
                    body=native_ast.FunctionBody.Internal(body=body_native_expr),
                    output_type=native_ast.Void
                ),
                return_type
            )

        if return_type.is_pass_by_ref:
            return (
                native_ast.Function(
                    args=(
                        (('.return', return_type.getNativeLayoutType().pointer()),)
                        + tuple(self._native_args)
                    ),
                    body=native_ast.FunctionBody.Internal(body=body_native_expr),
                    output_type=native_ast.Void
                ),
                return_type
            )
        else:
            return (
                native_ast.Function(
                    args=self._native_args,
                    body=native_ast.FunctionBody.Internal(body=body_native_expr),
                    output_type=return_type.getNativeLayoutType()
                ),
                return_type
            )

    def _constructInitialVarnameToType(self):
        input_types = self._input_types

        self._argnames = list(self._closureVarnames) + list(self.funcArgNames)

        if len(input_types) != len(self._argnames):
            raise ConversionException(
                "%s at %s:%s, with closure %s, expected at least %s arguments but got %s. Expected argnames are %s. Input types are %s" %
                (
                    self.name,
                    self._statements[0].filename,
                    self._statements[0].line_number,
                    self._closureVarnames,
                    len(self._argnames),
                    len(input_types),
                    self._argnames, input_types
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

    def typesAreUnstable(self):
        return self._typesAreUnstable

    def resetTypeInstabilityFlag(self):
        self._typesAreUnstable = False

    def markTypesAreUnstable(self):
        self._typesAreUnstable = True

    def allocateLetVarname(self):
        self.tempLetVarIx += 1
        return "letvar.%s" % (self.tempLetVarIx-1)

    def allocateStackVarname(self):
        self._tempStackVarIx += 1
        return "stackvar.%s" % (self._tempStackVarIx-1)

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
            subcontext,
            native_ast.Expression.Variable(name=varname),
            varType,
            varType.is_pass_by_ref
        )

    def upsizeVariableType(self, varname, new_type):
        if self._varname_to_type.get(varname) is None:
            if new_type is None:
                return

            self._varname_to_type[varname] = new_type
            self.markTypesAreUnstable()
            return

        existingType = self._varname_to_type[varname].typeRepresentation

        if existingType == new_type.typeRepresentation:
            return

        if issubclass(existingType, OneOf):
            if new_type.typeRepresentation in existingType.Types:
                return

            if issubclass(new_type.typeRepresentation, OneOf):
                if all(x in existingType.Types for x in new_type.typeRepresentation.Types):
                    return

        final_type = OneOf(new_type.typeRepresentation, existingType)

        self.markTypesAreUnstable()

        self._varname_to_type[varname] = typeWrapper(final_type)

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

    def initializeVariableStates(self, argnames, variableStates):
        to_add = self.closureInitializer(variableStates)

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
                    elif name in self.variablesBound and name not in self.variablesAssigned and name not in self.variablesReadByClosures:
                        # this variable is bound but never assigned, so we don't need to
                        # generate a stackslot. We can just read it directly from our arguments
                        self._argumentsWithoutStackslots.add(name)
                    elif slot_type.is_pod:
                        # we can just copy this into the stackslot directly. no destructor needed
                        context.pushEffect(
                            native_ast.Expression.Store(
                                ptr=self.localVariableExpression(context, name).expr,
                                val=(
                                    native_ast.Expression.Variable(name=name) if not slot_type.is_pass_by_ref else
                                    native_ast.Expression.Variable(name=name).load()
                                )
                            )
                        )
                        context.markVariableInitialized(name)
                        variableStates.variableAssigned(name, slot_type.typeRepresentation)
                    else:
                        # need to make a stackslot for this variable
                        var_expr = context.inputArg(self._argtypes[name], name)
                        converted = var_expr.convert_to_type(slot_type)
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
            destructors.append(
                native_ast.Teardown.Always(
                    expr=expr
                )
            )

        return destructors

    def isInitializedVarExpr(self, context, name):
        if self.variableIsAlwaysEmpty(name):
            return context.constant(True)

        return TypedExpression(
            context,
            native_ast.Expression.StackSlot(
                name=name + ".isInitialized",
                type=native_ast.Bool
            ),
            bool,
            isReference=True
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
        val_to_store = val_to_store.convert_to_type(slot_ref.expr_type)

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
                locals={name: None for name in (self.variablesBound | self.variablesAssigned)}
            )

            tpFunction = Function(untypedFunction)

            self.functionDefToClosurelessFunctionTypeCache[ast] = type(tpFunction)
            self.closurelessFunctionTypeToDef[type(tpFunction)] = ast

            for varname in tpFunction.overloads[0].closureVarLookups:
                assert varname in self.variablesReadByClosures

            python_ast._originalAstCache[untypedFunction.__code__] = ast

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
        if condition.matches.Call and condition.func.matches.Name and len(condition.args) == 2 and condition.args[0].matches.Name:
            if self.freeVariableLookup(condition.func.id) is isinstance:
                context = ExpressionConversionContext(self, variableStates)
                typeExpr = context.convert_expression_ast(condition.args[1])

                if typeExpr is not None and isinstance(typeExpr.expr_type, PythonTypeObjectWrapper):
                    variableStates.restrictTypeFor(condition.args[0].id, typeExpr.expr_type.typeRepresentation.Value, result)

        # check if we are a 'var.matches.Y' expression
        if (condition.matches.Attribute and
                condition.value.matches.Attribute and
                condition.value.attr == "matches" and
                condition.value.value.matches.Name):
            curType = variableStates.currentType(condition.value.value.id)
            if curType is not None and getattr(curType, '__typed_python_category__', None) == "Alternative":
                if result:
                    subType = [x for x in curType.__typed_python_alternatives__ if x.Name == condition.attr]
                    if subType:
                        variableStates.restrictTypeFor(
                            condition.value.value.id,
                            subType[0],
                            result
                        )


class ExpressionFunctionConversionContext(ConversionContextBase):
    """Helper function for converting a single python function given some input and output types"""

    def __init__(self, converter, name, identity, input_types, generator, outputType=None, alwaysRaises=False):
        super().__init__(converter, name, identity, input_types, outputType, [f'a{i}' for i in range(len(input_types))], [], {}, {})

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
                if self._varname_to_type.get(FunctionOutput) is None:
                    self.markTypesAreUnstable()
                    self._varname_to_type[FunctionOutput] = expr.expr_type
                else:
                    self.upsizeVariableType(FunctionOutput, expr.expr_type)

            subcontext.pushReturnValue(expr)

        return subcontext.finalize(None), False


class FunctionConversionContext(ConversionContextBase):
    """Helper function for converting a single python function given some input and output types"""

    def __init__(self, converter, name, identity, input_types, output_type, closureVarnames,
                 globalVars, globalVarsRaw, ast_arg, statements):
        super().__init__(converter, name, identity, input_types, output_type, ast_arg.argumentNames(),
                         closureVarnames, globalVars, globalVarsRaw)

        self._statements = statements

        self.variablesAssigned = computeAssignedVariables(statements)
        self.variablesBound = computeFunctionArgVariables(ast_arg) | set(closureVarnames)

        # the set of variables that are captured in closures in this function.
        # this includes recursive functions, which will not be in the closure itself
        # since they get bound in the closure varnames.
        self.variablesReadByClosures = computeVariablesReadByClosures(statements)

        # the set of variables that have exactly one definition. If these are 'deffed'
        # functions, we don't have to worry about them changing type and so they can be
        # bound to the closure directly (in which case we don't even assign them to slots)
        self.variablesAssignedOnlyOnce = computeVariablesAssignedOnlyOnce(statements)

        functionDefs, assignedLambdas, freeLambdas = extractFunctionDefs(statements)

        # the list of 'def' statements and 'Lambda' expressions. each one engenders a function type.
        self.functionDefs = functionDefs + freeLambdas

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

        # the current _type_ that we're using for this def,
        self.functionDefToType = {}

        # variables in closure slots that are not single-assignment function defs need slots
        self.variablesNeedingClosureSlots = set([
            c for c in self.variablesReadByClosures
            if c not in self.functionDefsAssignedOnce
        ])

        self._constructInitialVarnameToType()

    def convert_function_body(self, variableStates: FunctionStackState):
        return self.convert_statement_list_ast(self._statements, variableStates, toplevel=True)

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

            # we are assuming this is an index. We ought to be checking this
            # and doing something else if it's a Slice or an Ellipsis or whatnot
            index = expr_context.convert_expression_ast(expression.slice.value)

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
            assert target.slice.matches.Index

            slicing = subcontext.convert_expression_ast(target.value)
            if slicing is None:
                return False

            # we are assuming this is an index. We ought to be checking this
            # and doing something else if it's a Slice or an Ellipsis or whatnot
            index = subcontext.convert_expression_ast(target.slice.value)

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
            return self.convert_multi_assign(target.elts, val_to_store)

        assert False, target

    def convert_multi_assign(self, targets, val_to_store):
        subcontext = val_to_store.context
        variableStates = subcontext.variableStates

        iterated_expressions = val_to_store.get_iteration_expressions()

        if iterated_expressions is not None:
            if len(iterated_expressions) < len(targets):
                subcontext.pushException(
                    ValueError,
                    f"not enough values to unpack (expected {len(targets)}, got {len(iterated_expressions)})"
                )
                return False
            elif len(iterated_expressions) > len(targets):
                subcontext.pushException(
                    ValueError,
                    f"too many values to unpack (expected {len(targets)})"
                )
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
                next_ptr, is_populated = iter_obj.convert_next()  # this conversion is special - it returns two values
                if next_ptr is not None:
                    with subcontext.ifelse(is_populated.nonref_expr) as (if_true, if_false):
                        if targetIndex < len(targets):
                            with if_true:
                                tempVarnames.append(f".anonyous_iter{targets[0].line_number}.{targetIndex}")
                                self.assignToLocalVariable(tempVarnames[-1], next_ptr, variableStates)

                            with if_false:
                                subcontext.pushException(
                                    ValueError,
                                    f"not enough values to unpack (expected {len(targets)}, got {targetIndex})"
                                )
                        else:
                            with if_true:
                                subcontext.pushException(ValueError, f"too many values to unpack (expected {len(targets)})")

            for targetIndex in range(len(targets)):
                self.convert_assignment(
                    targets[targetIndex],
                    None,
                    subcontext.namedVariableLookup(tempVarnames[targetIndex])
                )

            return True

    def convert_statement_ast(self, ast, variableStates: FunctionStackState, return_to=None, in_loop=False, try_flow=None):
        """Convert a single statement to native_ast.

        Args:
            ast - the python_ast.Statement to convert
            variableStates - a description of what's known about the types of our variables.
                This data structure will be _modified_ by the calling code to include what's
                known about the types of values when control flow leaves this statement.
            return_to - label to return to. Only set within a 'try' statement.
                If None, return normally.
            try_flow - variable containing deferred control flow instruction.  Specifically, what
                control flow should be followed after the 'finally' block.
                Only set within a 'try' statement.
                0=default, 1=unhandled exception, 2=break, 3=continue, 4=return
            in_loop - are we within a 'for' or 'while' statement?
        Returns:
            a pair (native_ast.Expression, flowReturns) giving an expression representing the
            statement in native code, and a boolean indicating whether control flow might
            return to the caller. If false, then we can assume that the code throws an
            exception or 'returns' from the function.
        """

        try:
            return self._convert_statement_ast(ast, variableStates, return_to=return_to, in_loop=in_loop, try_flow=try_flow)
        except Exception as e:
            types = {
                varname: self._varname_to_type.get(varname)
                for varname in set(computeReadVariables(ast)).union(computeAssignedVariables(ast))
            }

            newMessage = f"\n{ast.filename}:{ast.line_number}\n" + "\n".join(f"    {k}={v}" for k, v in types.items())
            if e.args:
                e.args = (str(e.args[0]) + newMessage,)
            else:
                e.args = (newMessage,)
            raise

    def _convert_statement_ast(self, ast, variableStates: FunctionStackState, return_to=None, in_loop=False, try_flow=None):
        """same as 'convert_statement_ast'."""

        if ast.matches.Expr and ast.value.matches.Str:
            return native_ast.Expression(), True

        if ast.matches.AugAssign:
            subcontext = ExpressionConversionContext(self, variableStates)
            val_to_store = subcontext.convert_expression_ast(ast.value)

            if val_to_store is None:
                return subcontext.finalize(None, exceptionsTakeFrom=ast), False

            succeeds = self.convert_assignment(ast.target, ast.op, val_to_store)

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
                succeeds = self.convert_multi_assign(ast.targets, val_to_store)
                return subcontext.finalize(None, exceptionsTakeFrom=ast), succeeds

        if ast.matches.Return:
            subcontext = ExpressionConversionContext(self, variableStates)

            if ast.value is None:
                e = subcontext.constant(None)
            else:
                e = subcontext.convert_expression_ast(ast.value)

            if e is None:
                return subcontext.finalize(None, exceptionsTakeFrom=ast), False

            if not self._functionOutputTypeKnown:
                if self._varname_to_type.get(FunctionOutput) is None:
                    self.markTypesAreUnstable()
                    self._varname_to_type[FunctionOutput] = e.expr_type
                else:
                    self.upsizeVariableType(FunctionOutput, e.expr_type)

            if e.expr_type != self._varname_to_type[FunctionOutput]:
                e = e.convert_to_type(self._varname_to_type[FunctionOutput])

            if e is None:
                return subcontext.finalize(None, exceptionsTakeFrom=ast), False

            if return_to is not None:
                if try_flow:
                    subcontext.pushEffect(try_flow.store(native_ast.const_int_expr(CONTROL_FLOW_RETURN)))
                self.assignToLocalVariable(".return_value", e, variableStates)
                subcontext.pushEffect(runtime_functions.clear_exc_info.call())

            subcontext.pushReturnValue(e, blockName=return_to)

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

                branch, flow_returns = self.convert_statement_list_ast(ast.body if truth_val else ast.orelse, variableStates,
                                                                       return_to=return_to, try_flow=try_flow)

                return cond_context.finalize(None, exceptionsTakeFrom=ast) >> branch, flow_returns

            variableStatesTrue = variableStates.clone()
            variableStatesFalse = variableStates.clone()

            self.restrictByCondition(variableStatesTrue, ast.test, result=True)
            self.restrictByCondition(variableStatesFalse, ast.test, result=False)

            true, true_returns = self.convert_statement_list_ast(ast.body, variableStatesTrue, return_to=return_to, try_flow=try_flow)
            false, false_returns = self.convert_statement_list_ast(ast.orelse, variableStatesFalse, return_to=return_to, try_flow=try_flow)

            variableStates.becomeMerge(
                variableStatesTrue if true_returns else None,
                variableStatesFalse if false_returns else None
            )

            return (
                native_ast.Expression.Branch(
                    cond=cond_context.finalize(cond.nonref_expr, exceptionsTakeFrom=ast), true=true, false=false
                ),
                true_returns or false_returns
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
                            ast.orelse, variableStates, return_to=return_to, try_flow=try_flow
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
                    ast.body, variableStatesTrue, return_to=return_to, in_loop=True, try_flow=try_flow
                )

                if isWhileTrue:
                    if "loop_break" in true.returnTargets():
                        isWhileTrue = False

                false, false_returns = self.convert_statement_list_ast(
                    ast.orelse, variableStatesFalse, return_to=return_to, try_flow=try_flow
                )

                variableStates.becomeMerge(
                    variableStatesTrue if true_returns else None,
                    variableStatesFalse if false_returns else None
                )

                variableStates.mergeWithSelf(initVariableStates)

                if variableStates == initVariableStates:
                    return (
                        native_ast.Expression.While(
                            cond=cond_context.finalize(cond.nonref_expr, exceptionsTakeFrom=ast),
                            while_true=true.withReturnTargetName("loop_continue"),
                            orelse=false
                        ).withReturnTargetName("loop_break"),
                        (true_returns or false_returns) and not isWhileTrue
                    )

        if ast.matches.Try:
            # .exception_occurred turns on once any exception occurs
            # .control_flow indicates the control flow instruction that is deferred until after the 'finally' block
            #   0=default, 1=unhandled exception, 2=break, 3=continue, 4=return
            exception_occurred = native_ast.Expression.StackSlot(name=f".exception_occurred{ast.line_number}", type=native_ast.Bool)
            control_flow = native_ast.Expression.StackSlot(name=f".control_flow{ast.line_number}", type=native_ast.Int64)

            body_context = ExpressionConversionContext(self, variableStates)
            body, body_returns = self.convert_statement_list_ast(
                ast.body, variableStates, in_loop=in_loop, return_to=f"end_of_try{ast.line_number}", try_flow=control_flow
            )
            body_context.pushEffect(body)

            handlers_context = ExpressionConversionContext(self, variableStates)

            working_context = ExpressionConversionContext(self, variableStates)
            working = runtime_functions.catch_exception.call() \
                >> control_flow.store(native_ast.const_int_expr(CONTROL_FLOW_EXCEPTION))
            working_returns = True

            for h in reversed(ast.handlers):
                cond_context = ExpressionConversionContext(self, variableStates)
                if h.type is None:
                    exc_match = BaseException
                    exc_type = BaseException
                else:
                    if h.type.matches.Tuple:
                        # TODO: figure out compile-time Value for tuple of constant objects
                        types = [cond_context.convert_expression_ast(elt).expr_type.typeRepresentation.Value for elt in h.type.elts]
                        exc_type = tuple(types)
                        # exc_type = OneOf(*[Value(t) for t in types])
                        exc_type = BaseException
                    else:
                        exc_match = cond_context.convert_expression_ast(h.type).expr_type.typeRepresentation.Value
                        exc_type = exc_match
                cond = cond_context.matchExceptionObject(exc_match)

                handler_context = ExpressionConversionContext(self, variableStates)
                if h.name is None:
                    handler_context.pushEffect(runtime_functions.catch_exception.call())
                else:
                    h_name = h.name
                    self.assignToLocalVariable(h_name, handler_context.fetchExceptionObject(exc_type), variableStates)

                handler, handler_returns = self.convert_statement_list_ast(
                    h.body, variableStates, in_loop=in_loop, return_to=f"end_of_try{ast.line_number}", try_flow=control_flow
                )

                if h.name is not None:
                    cleanup_context = ExpressionConversionContext(self, variableStates)
                    self.localVariableExpression(cleanup_context, h_name).convert_destroy()
                    cleanup_context.markVariableNotInitialized(h_name)
                    handler = native_ast.Expression.Finally(
                        expr=handler,
                        teardowns=[
                            native_ast.Teardown.Always(
                                expr=cleanup_context.finalize(None).with_comment(f"Cleanup for {h_name}")
                            )
                        ]
                    )
                    variableStates.variableUninitialized(h_name)

                variableStatesMatch = variableStates.clone()
                variableStatesOtherwise = variableStates.clone()

                variableStates.becomeMerge(
                    variableStatesMatch if handler_returns else None,
                    variableStatesOtherwise if working_returns else None
                )
                working = native_ast.Expression.Branch(
                    cond=cond_context.finalize(cond.nonref_expr),
                    true=handler_context.finalize(handler >> runtime_functions.clear_exc_info.call()),
                    false=working_context.finalize(working)
                )
                working_returns = handler_returns or working_returns

            handlers_context.pushEffect(working_context.finalize(working))

            orelse_context = ExpressionConversionContext(self, variableStates)
            if len(ast.orelse) > 0:
                orelse, orelse_returns = self.convert_statement_list_ast(
                    ast.orelse, variableStates, in_loop=in_loop, return_to=f"end_of_try{ast.line_number}", try_flow=control_flow
                )
                orelse_context.pushEffect(orelse)
            else:
                orelse, orelse_returns = None, True

            final_context = ExpressionConversionContext(self, variableStates)
            if ast.finalbody is not None:
                final, final_returns = self.convert_statement_list_ast(
                    ast.finalbody, variableStates, in_loop=in_loop, return_to=f"end_of_finally{ast.line_number}", try_flow=control_flow
                )
                final = final.withReturnTargetName(f"end_of_finally{ast.line_number}")
                final_returns = True
                final_context.pushEffect(final)
            else:
                final, final_returns = None, True

            complete = exception_occurred.store(native_ast.falseExpr) \
                >> control_flow.store(native_ast.const_int_expr(CONTROL_FLOW_DEFAULT)) \
                >> native_ast.Expression.TryCatch(
                    expr=body_context.finalize(None),
                    handler=exception_occurred.store(native_ast.trueExpr)
                    >> native_ast.Expression.TryCatch(
                        expr=handlers_context.finalize(None),
                        handler=runtime_functions.catch_exception.call()
                        >> control_flow.store(native_ast.const_int_expr(CONTROL_FLOW_EXCEPTION))
                    )
            )
            if orelse:
                complete = complete >> native_ast.Expression.Branch(
                    cond=exception_occurred.load(),
                    false=native_ast.Expression.TryCatch(
                        expr=orelse_context.finalize(None) >> native_ast.nullExpr,
                        handler=exception_occurred.store(native_ast.trueExpr)
                        >> runtime_functions.catch_exception.call()
                        >> control_flow.store(native_ast.const_int_expr(CONTROL_FLOW_EXCEPTION))
                    )
                )

            complete = complete.withReturnTargetName(f"end_of_try{ast.line_number}")

            if final:
                complete = complete >> native_ast.Expression.TryCatch(
                    expr=final_context.finalize(None),
                    handler=exception_occurred.store(native_ast.trueExpr)
                    >> runtime_functions.catch_exception.call()
                    >> control_flow.store(native_ast.const_int_expr(CONTROL_FLOW_EXCEPTION))
                )

            if self.isLocalVariable(".return_value"):
                return_context = ExpressionConversionContext(self, variableStates)
                rtn, _ = self.convert_statement_ast(
                    python_ast.Statement.Return(
                        value=python_ast.Expr.Name(id=".return_value"), filename="", line_number=0, col_offset=0
                    ),
                    variableStates,
                    in_loop=in_loop,
                    return_to=return_to,
                    try_flow=try_flow
                )

                complete = complete >> native_ast.Expression.Branch(
                    cond=control_flow.load().eq(CONTROL_FLOW_RETURN),
                    true=native_ast.Expression.Branch(
                        cond=exception_occurred.load(),
                        true=runtime_functions.clear_exc_info.call()
                    )
                    >> return_context.finalize(rtn)
                )
            if in_loop:
                break_context = ExpressionConversionContext(self, variableStates)
                brk, _ = self.convert_statement_ast(
                    python_ast.Statement.Break(filename="", line_number=0, col_offset=0),
                    variableStates,
                    in_loop=in_loop,
                    return_to=return_to,
                    try_flow=try_flow
                )
                complete = complete >> native_ast.Expression.Branch(
                    cond=control_flow.load().eq(CONTROL_FLOW_BREAK),
                    true=runtime_functions.clear_exc_info.call() >> break_context.finalize(brk)
                )

                cont_context = ExpressionConversionContext(self, variableStates)
                cont, _ = self.convert_statement_ast(
                    python_ast.Statement.Continue(filename="", line_number=0, col_offset=0),
                    variableStates,
                    in_loop=in_loop,
                    return_to=return_to,
                    try_flow=try_flow
                )
                complete = complete >> native_ast.Expression.Branch(
                    cond=control_flow.load().eq(CONTROL_FLOW_CONTINUE),
                    true=runtime_functions.clear_exc_info.call() >> cont_context.finalize(cont)
                )

            raise_context = ExpressionConversionContext(self, variableStates)
            raise_context.pushExceptionObject(None, clear_exc=True)
            complete = complete >> native_ast.Expression.Branch(
                cond=control_flow.load().eq(CONTROL_FLOW_EXCEPTION),
                true=raise_context.finalize(None)
            )

            return (complete, ((body_returns and orelse_returns) or working_returns) and final_returns)

        if ast.matches.For:
            if not ast.target.matches.Name:
                raise NotImplementedError("Can't handle multi-variable loop expressions")

            target_var_name = ast.target.id

            iterator_setup_context = ExpressionConversionContext(self, variableStates)

            to_iterate = iterator_setup_context.convert_expression_ast(ast.iter)
            if to_iterate is None:
                return iterator_setup_context.finalize(None, exceptionsTakeFrom=ast), False

            iteration_expressions = to_iterate.get_iteration_expressions()

            # we allow types to explicitly break themselves down into a fixed set of
            # expressions to unroll, so that we can retain typing information.
            if iteration_expressions is not None:
                for subexpr in iteration_expressions:
                    self.assignToLocalVariable(target_var_name, subexpr, variableStates)

                    thisOne, thisOneReturns = self.convert_statement_list_ast(
                        ast.body, variableStates, return_to=return_to, in_loop=True, try_flow=try_flow
                    )

                    # if we hit 'continue', just come to the end of this expression
                    thisOne = thisOne.withReturnTargetName("loop_continue")

                    iterator_setup_context.pushEffect(thisOne)

                    if not thisOneReturns:
                        return iterator_setup_context.finalize(None, exceptionsTakeFrom=ast).withReturnTargetName("loop_break"), False

                thisOne, thisOneReturns = self.convert_statement_list_ast(
                    ast.orelse, variableStates, return_to=return_to, in_loop=in_loop, try_flow=try_flow
                )

                iterator_setup_context.pushEffect(thisOne)

                wholeLoopExpr = iterator_setup_context.finalize(None, exceptionsTakeFrom=ast)

                wholeLoopExpr = wholeLoopExpr.withReturnTargetName("loop_break")

                return wholeLoopExpr, thisOneReturns
            else:
                # create a variable to hold the iterator, and instantiate it there
                iter_varname = target_var_name + ".iter." + str(ast.line_number)

                iterator_object = to_iterate.convert_method_call("__iter__", (), {})
                if iterator_object is None:
                    return iterator_setup_context.finalize(None, exceptionsTakeFrom=ast), False

                self.assignToLocalVariable(iter_varname, iterator_object, variableStates)

                while True:
                    # track the initial variable states
                    initVariableStates = variableStates.clone()

                    cond_context = ExpressionConversionContext(self, variableStates)

                    iter_obj = cond_context.namedVariableLookup(iter_varname)
                    if iter_obj is None:
                        return (
                            iterator_setup_context.finalize(None, exceptionsTakeFrom=ast)
                            >> cond_context.finalize(None, exceptionsTakeFrom=ast),
                            False
                        )

                    next_ptr, is_populated = iter_obj.convert_next()  # this conversion is special - it returns two values
                    if next_ptr is None:
                        return (
                            iterator_setup_context.finalize(None, exceptionsTakeFrom=ast)
                            >> cond_context.finalize(None, exceptionsTakeFrom=ast),
                            False
                        )

                    with cond_context.ifelse(is_populated.nonref_expr) as (if_true, if_false):
                        with if_true:
                            self.assignToLocalVariable(target_var_name, next_ptr, variableStates)

                    variableStatesTrue = variableStates.clone()
                    variableStatesFalse = variableStates.clone()

                    true, true_returns = self.convert_statement_list_ast(
                        ast.body, variableStatesTrue, in_loop=True, return_to=return_to, try_flow=try_flow
                    )
                    false, false_returns = self.convert_statement_list_ast(
                        ast.orelse, variableStatesFalse, in_loop=True, return_to=return_to, try_flow=try_flow
                    )

                    variableStates.becomeMerge(
                        variableStatesTrue if true_returns else None,
                        variableStatesFalse if false_returns else None
                    )

                    variableStates.mergeWithSelf(initVariableStates)

                    if variableStates == initVariableStates:
                        # if nothing changed, the loop is stable.
                        return (
                            iterator_setup_context.finalize(None, exceptionsTakeFrom=ast) >>
                            native_ast.Expression.While(
                                cond=cond_context.finalize(is_populated, exceptionsTakeFrom=ast),
                                while_true=true.withReturnTargetName("loop_continue"),
                                orelse=false
                            ).withReturnTargetName("loop_break"),
                            true_returns or false_returns
                        )

        if ast.matches.Raise:
            expr_context = ExpressionConversionContext(self, variableStates)

            if ast.exc is None:
                toThrow = None  # means reraise
            else:
                toThrow = expr_context.convert_expression_ast(ast.exc)

                if toThrow is None:
                    return None

                toThrow = toThrow.convert_to_type(object)
                if toThrow is None:
                    return None

            if ast.cause is None:
                expr_context.pushExceptionObject(toThrow)
            else:
                excCause = expr_context.convert_expression_ast(ast.cause)
                expr_context.pushExceptionObjectWithCause(toThrow, excCause)

            return expr_context.finalize(None, exceptionsTakeFrom=None if ast.exc is None else ast), False

        if ast.matches.Delete:
            exprs = None
            for target in ast.targets:
                subExprs, flowReturns = self.convert_delete(target, variableStates)
                if exprs is None:
                    exprs = subExprs
                else:
                    exprs = exprs >> subExprs

                if not flowReturns:
                    return exprs, flowReturns
            return exprs, True

        if ast.matches.With:
            # with EXPRESSION as TARGET:
            #     SUITE
            #
            # is semantically equivalent to:
            #
            # manager = (EXPRESSION)
            # enter = type(manager).__enter__
            # exit = type(manager).__exit__
            # value = enter(manager)
            # hit_except = False
            #
            # try:
            #     TARGET = value
            #     SUITE
            # except:
            #     hit_except = True
            #     if not exit(manager, *sys.exc_info()):
            #         raise
            # finally:
            #     if not hit_except:
            #         exit(manager, None, None, None)
            #             assert len(ast.items) == 1

            assert len(ast.items) == 1
            # TODO: 'with' with multiple context managers
            # with A() as a, B() as b:
            #     SUITE
            #
            # is semantically equivalent to:
            #
            # with A() as a:
            #     with B() as b:
            #         SUITE

            # .exception_occurred indicates if any exception has occurred
            # .control_flow indicates the control flow instruction that is deferred until after the '__exit__'
            #   0=default, 1=unhandled exception, 2=break, 3=continue, 4=return
            # .exception_tuple stores exc_info, if needed
            exception_occurred = native_ast.Expression.StackSlot(name=f".exception_occurred{ast.line_number}", type=native_ast.Bool)
            control_flow = native_ast.Expression.StackSlot(name=f".control_flow{ast.line_number}", type=native_ast.Int64)
            exception_tuple = native_ast.Expression.StackSlot(
                name=f".exception_tuple{ast.line_number}",
                type=typeWrapper(Tuple(object, object, object)).layoutType
            )

            enter_context = ExpressionConversionContext(self, variableStates)

            item = enter_context.convert_expression_ast(ast.items[0].context_expr)
            if item is None:
                return enter_context.finalize(None, exceptionsTakeFrom=ast), False
            item_var = f".item{ast.line_number}"
            self.assignToLocalVariable(item_var, item, variableStates)

            item_enter = self.localVariableExpression(enter_context, item_var).convert_context_manager_enter()
            if item_enter is None:
                return enter_context.finalize(None, exceptionsTakeFrom=ast), False

            body_context = ExpressionConversionContext(self, variableStates)
            v = ast.items[0].optional_vars
            if v is not None:
                self.assignToLocalVariable(v.id, item_enter.changeContext(body_context), variableStates)
            body, body_returns = self.convert_statement_list_ast(
                ast.body, variableStates, in_loop=in_loop, return_to=f"exit{ast.line_number}", try_flow=control_flow
            )

            cond_context = ExpressionConversionContext(self, variableStates)
            cond_context.pushEffect(runtime_functions.fetch_exception_tuple.call(exception_tuple.elemPtr(0).cast(native_ast.VoidPtr)))
            t = TypedExpression(cond_context, exception_tuple, typeWrapper(Tuple(object, object, object)), True)
            item_exit = self.localVariableExpression(cond_context, item_var).convert_context_manager_exit(
                [t.convert_getitem(cond_context.constant(0)),
                 t.convert_getitem(cond_context.constant(1)),
                 t.convert_getitem(cond_context.constant(2))]
            )
            if item_exit is None:
                return cond_context.finalize(None, exceptionsTakeFrom=ast), False
            item_exit = item_exit.toBool()

            handler_context = ExpressionConversionContext(self, variableStates)
            handler = native_ast.Expression.TryCatch(
                expr=native_ast.Expression.Branch(
                    cond=cond_context.finalize(item_exit.nonref_expr, exceptionsTakeFrom=ast),
                    true=control_flow.store(native_ast.const_int_expr(0)),
                    false=control_flow.store(native_ast.const_int_expr(1))
                ),
                handler=control_flow.store(native_ast.const_int_expr(1))
                >> runtime_functions.catch_exception.call()
            )

            complete = exception_occurred.store(native_ast.falseExpr) \
                >> control_flow.store(native_ast.const_int_expr(0)) \
                >> enter_context.finalize(None) \
                >> native_ast.Expression.TryCatch(
                    expr=body_context.finalize(body),
                    handler=exception_occurred.store(native_ast.trueExpr)
                    >> handler_context.finalize(handler)
            ).withReturnTargetName(f"exit{ast.line_number}")

            exit_context = ExpressionConversionContext(self, variableStates)
            item_exit = self.localVariableExpression(exit_context, item_var).convert_context_manager_exit(
                [exit_context.constant(None) for _ in range(3)]
            )
            if item_exit is None:
                return exit_context.finalize(None, exceptionsTakeFrom=ast), False

            complete = complete >> native_ast.Expression.Branch(
                cond=exception_occurred.load(),
                false=exit_context.finalize(None, exceptionsTakeFrom=ast)
            )

            if self.isLocalVariable(".return_value"):
                return_context = ExpressionConversionContext(self, variableStates)
                rtn, _ = self.convert_statement_ast(
                    python_ast.Statement.Return(
                        value=python_ast.Expr.Name(id=".return_value"), filename="", line_number=0, col_offset=0
                    ),
                    variableStates,
                    in_loop=in_loop,
                    return_to=return_to,
                    try_flow=try_flow
                )

                complete = complete >> native_ast.Expression.Branch(
                    cond=control_flow.load().eq(4),
                    true=return_context.finalize(rtn)
                )

            if in_loop:
                break_context = ExpressionConversionContext(self, variableStates)
                brk, _ = self.convert_statement_ast(
                    python_ast.Statement.Break(filename="", line_number=0, col_offset=0),
                    variableStates,
                    in_loop=in_loop,
                    return_to=return_to,
                    try_flow=try_flow
                )
                complete = complete >> native_ast.Expression.Branch(
                    cond=control_flow.load().eq(2),
                    true=break_context.finalize(brk)
                )

                cont_context = ExpressionConversionContext(self, variableStates)
                cont, _ = self.convert_statement_ast(
                    python_ast.Statement.Continue(filename="", line_number=0, col_offset=0),
                    variableStates,
                    in_loop=in_loop,
                    return_to=return_to,
                    try_flow=try_flow
                )
                complete = complete >> native_ast.Expression.Branch(
                    cond=control_flow.load().eq(3),
                    true=cont_context.finalize(cont)
                )

            raise_context = ExpressionConversionContext(self, variableStates)
            raise_context.pushExceptionObject(None)
            complete = complete >> native_ast.Expression.Branch(
                cond=control_flow.load().eq(1),
                true=raise_context.finalize(None)
            )

            return complete, True

        if ast.matches.Break:
            # for the moment, we have to pretend as if the 'break' did return control flow,
            # or else a while loop that always ends in break/continue will look like it doesn't
            # return, when in fact it does.
            if return_to is not None:
                if try_flow:
                    return try_flow.store(native_ast.const_int_expr(CONTROL_FLOW_BREAK)) \
                        >> runtime_functions.clear_exc_info.call() \
                        >> native_ast.Expression.Return(blockName=return_to), True
                else:
                    return native_ast.Expression.Return(blockName=return_to), True
            else:
                return native_ast.Expression.Return(blockName="loop_break"), True

        if ast.matches.Continue:
            # for the moment, we have to pretend as if the 'continue' did return control flow,
            # or else a while loop that always ends in break/continue will look like it doesn't
            # return, when in fact it does.
            if return_to is not None:
                if try_flow:
                    return try_flow.store(native_ast.const_int_expr(CONTROL_FLOW_CONTINUE)) \
                        >> runtime_functions.clear_exc_info.call() \
                        >> native_ast.Expression.Return(blockName=return_to), True
                else:
                    return native_ast.Expression.Return(blockName=return_to), True
            else:
                return native_ast.Expression.Return(blockName="loop_continue"), True

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

            res = self.localVariableExpression(context, ".closure").changeType(
                self.functionDefToType[ast]
            )

            self.assignToLocalVariable(ast.name, res, variableStates)

            return context.finalize(None, exceptionsTakeFrom=ast), True

        raise ConversionException("Can't handle python ast Statement.%s" % ast.Name)

    def convert_statement_list_ast(
            self, statements, variableStates: FunctionStackState, toplevel=False, return_to=None, in_loop=False, try_flow=None
    ):
        """Convert a sequence of statements to a native expression.

        After executing this statement, variableStates will contain the known states of the
        current variables.

        Args:
            statements - a list of python_ast.Statement objects
            variableStates - a FunctionStackState object,
            toplevel - is this at the root of a function, so that flowing off the end should
                produce a Return expression?
            return_to - if any of these statements alter control flow (return, break, continue), this is the name of the
                finally block that we should return to first

        Returns:
            a tuple (expr: native_ast.Expression, controlFlowReturns: bool) giving the result, and
            whether control flow returns to the invoking native code.
        """
        exprAndReturns = []
        for s in statements:
            expr, controlFlowReturns = self.convert_statement_ast(
                s, variableStates, return_to=return_to, in_loop=in_loop, try_flow=try_flow
            )

            exprAndReturns.append((expr, controlFlowReturns))

            if not controlFlowReturns:
                break

        if not exprAndReturns or exprAndReturns[-1][1]:
            flows_off_end = True
        else:
            flows_off_end = False

        if toplevel and flows_off_end:
            flows_off_end = False
            if not self._functionOutputTypeKnown:
                if self._varname_to_type.get(FunctionOutput) is None:
                    self._varname_to_type[FunctionOutput] = NoneWrapper()
                    self.markTypesAreUnstable()
                else:
                    self.upsizeVariableType(FunctionOutput, NoneWrapper())

            exprAndReturns.append(
                self.convert_statement_ast(
                    python_ast.Statement.Return(
                        value=None, filename="", line_number=0, col_offset=0
                    ),
                    variableStates,
                    return_to=return_to,
                    try_flow=try_flow
                )
            )

        seq_expr = native_ast.makeSequence(
            [expr for expr, _ in exprAndReturns]
        )

        return seq_expr, flows_off_end
