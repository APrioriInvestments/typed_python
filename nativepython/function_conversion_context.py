#   Coyright 2017-2019 Nativepython Authors
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

from nativepython.python_ast_analysis import (
    computeAssignedVariables,
    computeReadVariables,
    computeFunctionArgVariables
)

import nativepython
import nativepython.native_ast as native_ast
from nativepython.expression_conversion_context import ExpressionConversionContext
from nativepython.function_stack_state import FunctionStackState
from nativepython.type_wrappers.none_wrapper import NoneWrapper
from nativepython.typed_expression import TypedExpression
from nativepython.conversion_exception import ConversionException
from typed_python import OneOf

NoneExprType = NoneWrapper()

typeWrapper = lambda t: nativepython.python_object_representation.typedPythonTypeToTypeWrapper(t)


class FunctionOutput:
    pass


class FunctionConversionContext(object):
    """Helper function for converting a single python function given some input and output types"""

    def __init__(self, converter, identity, ast_arg, statements, input_types, output_type, free_variable_lookup):
        """Initialize a FunctionConverter

        Args:
            converter - a PythonToNativeConverter
            identity - an object to uniquely identify this instance of the function
            ast_arg - a python_ast.Arguments object
            statements - a list of python_ast.Statement objects making up the body of the function
            input_types - a list of the input types actually passed to us
            output_type - the output type (if proscribed), or None
            free_variable_lookup - a dict from name to the actual python object in this
                function's closure. We don't distinguish between local and global scope yet.
        """
        self.variablesAssigned = computeAssignedVariables(statements)
        self.variablesRead = computeReadVariables(statements)
        self.variablesBound = computeFunctionArgVariables(ast_arg)

        self.converter = converter
        self.identity = identity
        self._ast_arg = ast_arg
        self._argnames = None
        self._statements = statements
        self._input_types = input_types
        self._output_type = output_type
        self._argumentsWithoutStackslots = set()  # arguments that we don't bother to copy into the stack
        self._varname_to_type = {}
        self._free_variable_lookup = free_variable_lookup
        self._temp_let_var = 0
        self._temp_stack_var = 0
        self._temp_iter_var = 0
        self._typesAreUnstable = False
        self._functionOutputTypeKnown = False
        self._star_args_name = None
        self._native_args = None

        self._constructInitialVarnameToType()

    def isLocalVariable(self, name):
        return name in self.variablesBound or name in self.variablesAssigned

    def localVariableExpression(self, context: ExpressionConversionContext, name):
        """Return an TypedExpression reference for the local variable given by  'name'"""
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

        varType = self._varname_to_type.get(name)

        if varType is None or varType.is_empty or varType.is_pod:
            return False

        return True

    def convertToNativeFunction(self):
        variableStates = FunctionStackState()

        initializer_expr = self.initializeVariableStates(self._argnames, self._star_args_name, variableStates)

        body_native_expr, controlFlowReturns = self.convert_function_body(self._statements, variableStates)

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

        if self._ast_arg.vararg is not None:
            self._star_args_name = self._ast_arg.vararg.val.arg

        if self._star_args_name is None:
            if len(input_types) != len(self._ast_arg.args):
                raise ConversionException(
                    "Expected %s arguments but got %s" % (len(self._ast_arg.args), len(input_types))
                )
        else:
            if len(input_types) < len(self._ast_arg.args):
                raise ConversionException(
                    "Expected at least %s arguments but got %s" %
                    (len(self._ast_arg.args), len(input_types))
                )

        self._native_args = []
        for i in range(len(self._ast_arg.args)):
            self._varname_to_type[self._ast_arg.args[i].arg] = input_types[i]
            if not input_types[i].is_empty:
                self._native_args.append((self._ast_arg.args[i].arg, input_types[i].getNativePassingType()))

        self._argnames = [a.arg for a in self._ast_arg.args]

        if self._star_args_name is not None:
            star_args_count = len(input_types) - len(self._ast_arg.args)

            for i in range(len(self._ast_arg.args), len(input_types)):
                self._native_args.append(
                    ('.star_args.%s' % (i - len(self._ast_arg.args)),
                        input_types[i].getNativePassingType())
                )

            starargs_type = native_ast.Struct(
                [('f_%s' % i, input_types[i+len(self._ast_arg.args)])
                    for i in range(star_args_count)]
            )

            self._varname_to_type[self._star_args_name] = starargs_type

        if self._output_type is not None:
            self._varname_to_type[FunctionOutput] = typeWrapper(self._output_type)

        self._functionOutputTypeKnown = FunctionOutput in self._varname_to_type

    def typesAreUnstable(self):
        return self._typesAreUnstable

    def resetTypeInstabilityFlag(self):
        self._typesAreUnstable = False

    def markTypesAreUnstable(self):
        self._typesAreUnstable = True

    def let_varname(self):
        self._temp_let_var += 1
        return "letvar.%s" % (self._temp_let_var-1)

    def stack_varname(self):
        self._temp_stack_var += 1
        return "stackvar.%s" % (self._temp_stack_var-1)

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

        if hasattr(existingType, '__typed_python_category__') and \
                existingType.__typed_python_category__ == 'OneOf':
            if new_type.typeRepresentation in existingType.Types:
                return

        final_type = OneOf(new_type.typeRepresentation, existingType)

        self.markTypesAreUnstable()

        self._varname_to_type[varname] = typeWrapper(final_type)

    def initializeVariableStates(self, argnames, stararg_name, variableStates):
        to_add = []

        # first, mark every variable that we plan on assigning to as not initialized.
        for name in self.variablesAssigned:
            # this is a variable in the function that we assigned to. we need to ensure that
            # the initializer flag is zero
            if not self.variableIsAlwaysEmpty(name):
                context = ExpressionConversionContext(self, variableStates)
                context.markVariableNotInitialized(name)
                to_add.append(context.finalize(None))

        for name in self.variablesBound:
            if name not in self.variablesAssigned:
                variableStates.variableAssigned(name, self._varname_to_type[name].typeRepresentation)

        for name in argnames:
            if name is not FunctionOutput and name != stararg_name:
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
                    elif name in self.variablesBound and name not in self.variablesAssigned:
                        # this variable is bound but never assigned, so we don't need to
                        # generate a stackslot. We can just read it directly from our arguments
                        self._argumentsWithoutStackslots.add(name)
                    elif slot_type.is_pod:
                        # we can just copy this into the stackslot directly. no destructor needed
                        context.pushEffect(
                            native_ast.Expression.Store(
                                ptr=native_ast.Expression.StackSlot(
                                    name=name,
                                    type=slot_type.getNativeLayoutType()
                                ),
                                val=(
                                    native_ast.Expression.Variable(name=name) if not slot_type.is_pass_by_ref else
                                    native_ast.Expression.Variable(name=name).load()
                                )
                            )
                        )
                        context.markVariableInitialized(name)
                    else:
                        # need to make a stackslot for this variable
                        var_expr = context.inputArg(slot_type, name)

                        self.assignToLocalVariable(name, var_expr, variableStates)

                        context.markVariableInitialized(name)

                    to_add.append(context.finalize(None))

        return native_ast.makeSequence(to_add)

    def generateDestructors(self, variableStates):
        destructors = []

        for name in variableStates.variablesThatMightBeActive():
            if self.variableNeedsDestructor(name):
                context = ExpressionConversionContext(self, variableStates)

                slot_expr = self.localVariableExpression(context, name)

                with context.ifelse(context.isInitializedVarExpr(name)) as (true, false):
                    with true:
                        slot_expr.convert_destroy()

                destructors.append(
                    native_ast.Teardown.Always(
                        expr=context.finalize(None).with_comment("Cleanup for variable %s" % name)
                    )
                )

        return destructors

    def construct_starargs_around(self, res, star_args_name):
        args_type = self._varname_to_type[star_args_name]

        stararg_slot = self.named_var_expr(star_args_name)

        return (
            args_type.convert_initialize(
                self,
                stararg_slot,
                [
                    TypedExpression(
                        native_ast.Expression.Variable(".star_args.%s" % i),
                        args_type.element_types[i][1]
                    )
                    for i in range(len(args_type.element_types))]
            ).with_comment("initialize *args slot") + res
        )

    def assignToLocalVariable(self, varname, val_to_store, variableStates):
        """Ensure we have appropriate storage allocated for 'varname', and assign 'val_to_store' to it."""
        subcontext = val_to_store.context

        self.upsizeVariableType(varname, val_to_store.expr_type)

        assignedType = val_to_store.expr_type.typeRepresentation

        slot_ref = self.localVariableExpression(subcontext, varname)

        # convert the value to the target type now that we've upsized it
        val_to_store = val_to_store.convert_to_type(slot_ref.expr_type)

        assert val_to_store is not None, "We should always be able to upsize"

        if slot_ref.expr_type.is_empty:
            pass
        elif slot_ref.expr_type.is_pod:
            slot_ref.convert_copy_initialize(val_to_store)
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

    def convert_statement_ast(self, ast, variableStates: FunctionStackState):
        if ast.matches.Expr and ast.value.matches.Str:
            return native_ast.Expression(), True

        if ast.matches.Assign or ast.matches.AugAssign:
            if ast.matches.Assign:
                assert len(ast.targets) == 1
                op = None
                target = ast.targets[0]
            else:
                target = ast.target
                op = ast.op

            if target.matches.Name and target.ctx.matches.Store:
                varname = target.id

                if varname not in self._varname_to_type:
                    self._varname_to_type[varname] = None

                subcontext = ExpressionConversionContext(self, variableStates)

                val_to_store = subcontext.convert_expression_ast(ast.value)

                if val_to_store is None:
                    return subcontext.finalize(None), False

                if op is not None:
                    slot_ref = subcontext.namedVariableLookup(varname)
                    if slot_ref is None:
                        return subcontext.finalize(None), False

                    val_to_store = slot_ref.convert_bin_op(op, val_to_store)

                    if val_to_store is None:
                        return subcontext.finalize(None), False

                self.assignToLocalVariable(varname, val_to_store, variableStates)

                return subcontext.finalize(None).with_comment("Assign %s" % (varname)), True

            if target.matches.Subscript and target.ctx.matches.Store:
                assert target.slice.matches.Index

                subcontext = ExpressionConversionContext(self, variableStates)

                slicing = subcontext.convert_expression_ast(target.value)
                if slicing is None:
                    return subcontext.finalize(None), False

                # we are assuming this is an index. We ought to be checking this
                # and doing something else if it's a Slice or an Ellipsis or whatnot
                index = subcontext.convert_expression_ast(target.slice.value)

                if index is None:
                    return subcontext.finalize(None), False

                val_to_store = subcontext.convert_expression_ast(ast.value)

                if val_to_store is None:
                    return subcontext.finalize(None), False

                if op is not None:
                    getItem = slicing.convert_getitem(index)
                    if getItem is None:
                        return subcontext.finalize(None), False

                    val_to_store = getItem.convert_bin_op(op, val_to_store)
                    if val_to_store is None:
                        return subcontext.finalize(None), False

                slicing.convert_setitem(index, val_to_store)

                return subcontext.finalize(None), True

            if target.matches.Attribute and target.ctx.matches.Store:
                subcontext = ExpressionConversionContext(self, variableStates)

                slicing = subcontext.convert_expression_ast(target.value)
                attr = target.attr

                val_to_store = subcontext.convert_expression_ast(ast.value)
                if val_to_store is None:
                    return subcontext.finalize(None), False

                if op is not None:
                    input_val = slicing.convert_attribute(attr)
                    if input_val is None:
                        return subcontext.finalize(None), False

                    val_to_store = input_val.convert_bin_op(op, val_to_store)
                    if val_to_store is None:
                        return subcontext.finalize(None), False

                slicing.convert_set_attribute(attr, val_to_store)

                return subcontext.finalize(None), True

        if ast.matches.Return:
            subcontext = ExpressionConversionContext(self, variableStates)

            if ast.value is None:
                e = subcontext.convert_expression_ast(
                    python_ast.Expr.Num(n=python_ast.NumericConstant.None_())
                )
            else:
                e = subcontext.convert_expression_ast(ast.value)

            if e is None:
                return subcontext.finalize(None), False

            if not self._functionOutputTypeKnown:
                if self._varname_to_type.get(FunctionOutput) is None:
                    self.markTypesAreUnstable()
                    self._varname_to_type[FunctionOutput] = e.expr_type
                else:
                    self.upsizeVariableType(FunctionOutput, e.expr_type)

            if e.expr_type != self._varname_to_type[FunctionOutput]:
                e = e.convert_to_type(self._varname_to_type[FunctionOutput])

            if e is None:
                return subcontext.finalize(None), False

            if e.expr_type.is_pass_by_ref:
                returnTarget = TypedExpression(
                    subcontext,
                    native_ast.Expression.Variable(name=".return"),
                    self._varname_to_type[FunctionOutput],
                    True
                )

                returnTarget.convert_copy_initialize(e)

                subcontext.pushTerminal(
                    native_ast.Expression.Return(arg=None)
                )
            else:
                subcontext.pushTerminal(
                    native_ast.Expression.Return(arg=e.nonref_expr)
                )

            return subcontext.finalize(None), False

        if ast.matches.Expr:
            subcontext = ExpressionConversionContext(self, variableStates)

            result_expr = subcontext.convert_expression_ast(ast.value)

            return subcontext.finalize(result_expr), result_expr is not None

        if ast.matches.If:
            cond_context = ExpressionConversionContext(self, variableStates)
            cond = cond_context.convert_expression_ast(ast.test)
            if cond is None:
                return cond_context.finalize(None), False
            cond = cond.toBool()
            if cond is None:
                return cond_context.finalize(None), False

            if cond.expr.matches.Constant:
                truth_val = cond.expr.val.truth_value()

                branch, flow_returns = self.convert_statement_list_ast(ast.body if truth_val else ast.orelse, variableStates)

                return cond.expr >> branch, flow_returns

            variableStatesTrue = variableStates.clone()
            variableStatesFalse = variableStates.clone()

            self.restrictByCondition(variableStatesTrue, ast.test, result=True)
            self.restrictByCondition(variableStatesFalse, ast.test, result=False)

            true, true_returns = self.convert_statement_list_ast(ast.body, variableStatesTrue)
            false, false_returns = self.convert_statement_list_ast(ast.orelse, variableStatesFalse)

            variableStates.becomeMerge(
                variableStatesTrue if true_returns else None,
                variableStatesFalse if false_returns else None
            )

            return (
                native_ast.Expression.Branch(
                    cond=cond_context.finalize(cond.nonref_expr), true=true, false=false
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
                    return cond_context.finalize(None), False

                cond = cond.toBool()
                if cond is None:
                    return cond_context.finalize(None), False

                variableStatesTrue = variableStates.clone()
                variableStatesFalse = variableStates.clone()

                self.restrictByCondition(variableStatesTrue, ast.test, result=True)
                self.restrictByCondition(variableStatesFalse, ast.test, result=False)

                true, true_returns = self.convert_statement_list_ast(ast.body, variableStatesTrue)

                false, false_returns = self.convert_statement_list_ast(ast.orelse, variableStatesFalse)

                variableStates.becomeMerge(
                    variableStatesTrue if true_returns else None,
                    variableStatesFalse if false_returns else None
                )

                variableStates.mergeWithSelf(initVariableStates)

                if variableStates == initVariableStates:
                    return (
                        native_ast.Expression.While(
                            cond=cond_context.finalize(cond.nonref_expr), while_true=true, orelse=false
                        ),
                        true_returns or false_returns
                    )

        if ast.matches.Try:
            raise NotImplementedError()

        if ast.matches.For:
            if not ast.target.matches.Name:
                raise NotImplementedError("Can't handle multi-variable loop expressions")

            target_var_name = ast.target.id

            iterator_setup_context = ExpressionConversionContext(self, variableStates)

            to_iterate = iterator_setup_context.convert_expression_ast(ast.iter)
            if to_iterate is None:
                return iterator_setup_context.finalize(to_iterate), False

            iteration_expressions = to_iterate.get_iteration_expressions()

            # we allow types to explicitly break themselves down into a fixed set of
            # expressions to unroll, so that we can retain typing information.
            if iteration_expressions is not None:
                for subexpr in iteration_expressions:
                    self.assignToLocalVariable(target_var_name, subexpr, variableStates)

                    thisOne, thisOneReturns = self.convert_statement_list_ast(ast.body, variableStates)

                    iterator_setup_context.pushEffect(thisOne)

                    if not thisOneReturns:
                        return iterator_setup_context.finalize(None), False

                thisOne, thisOneReturns = self.convert_statement_list_ast(ast.orelse, variableStates)

                iterator_setup_context.pushEffect(thisOne)

                return iterator_setup_context.finalize(None), thisOneReturns
            else:
                # create a variable to hold the iterator, and instantiate it there
                iter_varname = target_var_name + ".iter." + str(ast.line_number)

                # we are going to assign this
                self.variablesAssigned.add(iter_varname)

                iterator_object = to_iterate.convert_method_call("__iter__", (), {})
                if iterator_object is None:
                    return iterator_setup_context.finalize(iterator_object), False

                self.assignToLocalVariable(iter_varname, iterator_object, variableStates)

                while True:
                    # track the initial variable states
                    initVariableStates = variableStates.clone()

                    cond_context = ExpressionConversionContext(self, variableStates)

                    iter_obj = cond_context.namedVariableLookup(iter_varname)
                    if iter_obj is None:
                        return iterator_setup_context.finalize(None) >> cond_context.finalize(None), False

                    next_ptr, is_populated = iter_obj.convert_next()  # this conversion is special - it returns two values
                    if next_ptr is None:
                        return iterator_setup_context.finalize(None) >> cond_context.finalize(None), False

                    with cond_context.ifelse(is_populated.nonref_expr) as (if_true, if_false):
                        with if_true:
                            self.assignToLocalVariable(target_var_name, next_ptr, variableStates)

                    variableStatesTrue = variableStates.clone()
                    variableStatesFalse = variableStates.clone()

                    true, true_returns = self.convert_statement_list_ast(ast.body, variableStatesTrue)
                    false, false_returns = self.convert_statement_list_ast(ast.orelse, variableStatesFalse)

                    variableStates.becomeMerge(
                        variableStatesTrue if true_returns else None,
                        variableStatesFalse if false_returns else None
                    )

                    variableStates.mergeWithSelf(initVariableStates)

                    if variableStates == initVariableStates:
                        # if nothing changed, the loop is stable.
                        return (
                            iterator_setup_context.finalize(None) >>
                            native_ast.Expression.While(
                                cond=cond_context.finalize(is_populated), while_true=true, orelse=false
                            ),
                            true_returns or false_returns
                        )

        if ast.matches.Raise:
            expr_contex = ExpressionConversionContext(self, variableStates)
            strVal = "Unknown Exception"
            if ast.exc.matches.Call:
                if ast.exc.func.matches.Name and ast.exc.func.id == "Exception":
                    if len(ast.exc.args) == 1 and ast.exc.args[0].matches.Str:
                        strVal = ast.exc.args[0].s

            expr_contex.pushException(KeyError, strVal)
            return expr_contex.finalize(None), False

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

        raise ConversionException("Can't handle python ast Statement.%s" % ast.Name)

    def freeVariableLookup(self, name):
        if self.isLocalVariable(name):
            return None

        if name in self._free_variable_lookup:
            return self._free_variable_lookup[name]

        if name in __builtins__:
            return __builtins__[name]

        return None

    def restrictByCondition(self, variableStates, condition, result):
        if condition.matches.Call and condition.func.matches.Name and len(condition.args) == 2 and condition.args[0].matches.Name:
            if self.freeVariableLookup(condition.func.id) is isinstance:
                context = ExpressionConversionContext(self, variableStates)
                typeExpr = context.convert_expression_ast(condition.args[1])

                PythonTypeObjectWrapper = nativepython.type_wrappers.python_type_object_wrapper.PythonTypeObjectWrapper
                if typeExpr is not None and isinstance(typeExpr.expr_type, PythonTypeObjectWrapper):
                    variableStates.restrictTypeFor(condition.args[0].id, typeExpr.expr_type.typeRepresentation, result)

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

    def convert_delete(self, expression, variableStates):
        """Convert the target of a 'del' statement.

        Args:
            expression - a python_ast Expression

        Returns:
            a pair of native_ast.Expression and a bool indicating whether control flow
            returns to the caller.
        """
        expr_contex = ExpressionConversionContext(self, variableStates)

        if expression.matches.Subscript:
            slicing = expr_contex.convert_expression_ast(expression.value)
            if slicing is None:
                return expr_contex.finalize(None), False

            # we are assuming this is an index. We ought to be checking this
            # and doing something else if it's a Slice or an Ellipsis or whatnot
            index = expr_contex.convert_expression_ast(expression.slice.value)

            if slicing is None:
                return expr_contex.finalize(None), False

            res = slicing.convert_delitem(index)

            return expr_contex.finalize(None), res is not None
        else:
            expr_contex.pushException(Exception, "Can't delete this")
            return expr_contex.finalize(None), False

    def convert_function_body(self, statements, variableStates: FunctionStackState):
        return self.convert_statement_list_ast(statements, variableStates, toplevel=True)

    def convert_statement_list_ast(self, statements, variableStates: FunctionStackState, toplevel=False):
        """Convert a sequence of statements to a native expression.

        After executing this statement, variableStates will contain the known states of the
        current variables.

        Args:
            statements - a list of python_ast.Statement objects
            variableStates - a FunctionStackState object,
            toplevel - is this at the root of a function, so that flowing off the end should
                produce a Return expression?

        Returns:
            a tuple (expr: native_ast.Expression, controlFlowReturns: bool) giving the result, and
            whether control flow returns to the invoking native code.
        """
        exprAndReturns = []
        for s in statements:
            expr, controlFlowReturns = self.convert_statement_ast(s, variableStates)

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
                    variableStates
                )
            )

        seq_expr = native_ast.makeSequence(
            [expr for expr, _ in exprAndReturns]
        )

        return seq_expr, flows_off_end
