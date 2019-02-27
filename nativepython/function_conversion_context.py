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

import nativepython
import nativepython.native_ast as native_ast
from nativepython.expression_conversion_context import ExpressionConversionContext
from nativepython.type_wrappers.none_wrapper import NoneWrapper
from nativepython.typed_expression import TypedExpression
from nativepython.conversion_exception import ConversionException
from typed_python import *

NoneExprType = NoneWrapper()

typeWrapper = lambda t: nativepython.python_object_representation.typedPythonTypeToTypeWrapper(t)


class FunctionOutput:
    pass


class FunctionConversionContext(object):
    """Helper function for converting a single python function."""

    def __init__(self, converter, identity, ast_arg, statements, input_types, output_type, free_variable_lookup):
        self.converter = converter
        self.identity = identity
        self._ast_arg = ast_arg
        self._argnames = None
        self._statements = statements
        self._input_types = input_types
        self._output_type = output_type
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

    def convertToNativeFunction(self):
        body_native_expr, controlFlowReturns = self.convert_function_body(self._statements)
        assert not controlFlowReturns

        if self._star_args_name is not None:
            body_native_expr = self.construct_starargs_around(body_native_expr, self._star_args_name)

        body_native_expr = self.construct_stackslots_around(body_native_expr, self._argnames, self._star_args_name)

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
                    args=(('.return', return_type.getNativeLayoutType().pointer()),) + tuple(self._native_args),
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

            starargs_type = Struct(
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

    def generateAssignmentExpr(self, varname, val_to_store):
        """Ensure we have appropriate storage allocated for 'varname', and assign 'val_to_store' to it."""
        subcontext = val_to_store.context

        self.upsizeVariableType(varname, val_to_store.expr_type)
        slot_ref = subcontext.named_var_expr(varname)

        # convert the value to the target type now that we've upsized it
        val_to_store = val_to_store.convert_to_type(slot_ref.expr_type)

        assert val_to_store is not None, "We should always be able to upsize"

        if slot_ref.expr_type.is_pod:
            slot_ref.convert_copy_initialize(val_to_store)
            subcontext.pushEffect(subcontext.isInitializedVarExpr(varname).expr.store(native_ast.trueExpr))
        else:
            with subcontext.ifelse(subcontext.isInitializedVarExpr(varname)) as (true_block, false_block):
                with true_block:
                    slot_ref.convert_assign(val_to_store)
                with false_block:
                    slot_ref.convert_copy_initialize(val_to_store)
                    subcontext.pushEffect(subcontext.isInitializedVarExpr(varname).expr.store(native_ast.trueExpr))

    def convert_statement_ast(self, ast):
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

                subcontext = ExpressionConversionContext(self)

                val_to_store = subcontext.convert_expression_ast(ast.value)

                if val_to_store is None:
                    return subcontext.finalize(None), False

                if op is not None:
                    if varname not in self._varname_to_type:
                        raise NotImplementedError()
                    else:
                        slot_ref = subcontext.named_var_expr(varname)
                        val_to_store = slot_ref.convert_bin_op(op, val_to_store)

                        if val_to_store is None:
                            return subcontext.finalize(None), False

                self.generateAssignmentExpr(varname, val_to_store)

                return subcontext.finalize(None).with_comment("Assign %s" % (varname)), True

            if target.matches.Subscript and target.ctx.matches.Store:
                assert target.slice.matches.Index

                subcontext = ExpressionConversionContext(self)

                slicing = subcontext.convert_expression_ast(target.value)
                if slicing is None:
                    return subcontext.finalize(None), False

                index = subcontext.convert_expression_ast(target.slice.value)

                if slicing is None:
                    return subcontext.finalize(None), False

                val_to_store = subcontext.convert_expression_ast(ast.value)

                if val_to_store is None:
                    return subcontext.finalize(None), False

                if op is not None:
                    val_to_store = slicing.convert_getitem(index).convert_bin_op(op, val_to_store)
                    if val_to_store is None:
                        return subcontext.finalize(None), False

                slicing.convert_setitem(index, val_to_store)

                return subcontext.finalize(None), True

            if target.matches.Attribute and target.ctx.matches.Store:
                subcontext = ExpressionConversionContext(self)

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
            subcontext = ExpressionConversionContext(self)

            if ast.value is None:
                e = subcontext.convert_expression_ast(python_ast.Expr.Num(n=python_ast.NumericConstant.None_()))
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
                returnTarget = TypedExpression(subcontext, native_ast.Expression.Variable(name=".return"), self._varname_to_type[FunctionOutput], True)

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
            subcontext = ExpressionConversionContext(self)

            result_expr = subcontext.convert_expression_ast(ast.value)

            return subcontext.finalize(result_expr), result_expr is not None

        if ast.matches.If:
            cond_context = ExpressionConversionContext(self)
            cond = cond_context.convert_expression_ast(ast.test)
            if cond is None:
                return cond_context.finalize(None), False
            cond = cond.toBool()
            if cond is None:
                return cond_context.finalize(None), False

            if cond.expr.matches.Constant:
                truth_val = cond.expr.val.truth_value()

                branch, flow_returns = self.convert_statement_list_ast(ast.body if truth_val else ast.orelse)

                return cond.expr + branch, flow_returns

            true, true_returns = self.convert_statement_list_ast(ast.body)
            false, false_returns = self.convert_statement_list_ast(ast.orelse)

            return (
                native_ast.Expression.Branch(cond=cond_context.finalize(cond.nonref_expr), true=true, false=false),
                true_returns or false_returns
            )

        if ast.matches.Pass:
            return native_ast.nullExpr, True

        if ast.matches.While:
            cond_context = ExpressionConversionContext(self)

            cond = cond_context.convert_expression_ast(ast.test)

            if cond is None:
                return cond_context.finalize(None), False
            cond = cond.toBool()
            if cond is None:
                return cond_context.finalize(None), False

            true, true_returns = self.convert_statement_list_ast(ast.body)

            false, false_returns = self.convert_statement_list_ast(ast.orelse)

            return (
                native_ast.Expression.While(cond=cond_context.finalize(cond.nonref_expr), while_true=true, orelse=false),
                true_returns or false_returns
            )

        if ast.matches.Try:
            raise NotImplementedError()

        if ast.matches.For:
            if not ast.target.matches.Name:
                raise NotImplementedError("Can't handle multi-variable loop expressions")

            target_var_name = ast.target.id

            # create a variable to hold the iterator, and instantiate it there
            iter_varname = target_var_name + ".iter." + str(ast.line_number)

            iterator_setup_context = ExpressionConversionContext(self)
            to_iterate = iterator_setup_context.convert_expression_ast(ast.iter)
            if to_iterate is None:
                return iterator_setup_context.finalize(to_iterate), False
            iterator_object = to_iterate.convert_method_call("__iter__", (), {})
            if iterator_object is None:
                return iterator_setup_context.finalize(iterator_object), False
            self.generateAssignmentExpr(iter_varname, iterator_object)

            cond_context = ExpressionConversionContext(self)
            iter_obj = cond_context.named_var_expr(iter_varname)
            next_ptr, is_populated = iter_obj.convert_next()  # this conversion is special - it returns two values

            with cond_context.ifelse(is_populated.nonref_expr) as (if_true, if_false):
                with if_true:
                    self.generateAssignmentExpr(target_var_name, next_ptr)

            true, true_returns = self.convert_statement_list_ast(ast.body)

            false, false_returns = self.convert_statement_list_ast(ast.orelse)

            return (
                iterator_setup_context.finalize(None) >>
                native_ast.Expression.While(cond=cond_context.finalize(is_populated), while_true=true, orelse=false),
                true_returns or false_returns
            )

        if ast.matches.Raise:
            raise NotImplementedError()

        raise ConversionException("Can't handle python ast Statement.%s" % ast._which)

    def convert_function_body(self, statements):
        return self.convert_statement_list_ast(statements, toplevel=True)

    def convert_statement_list_ast(self, statements, toplevel=False):
        exprAndReturns = []
        for s in statements:
            expr, controlFlowReturns = self.convert_statement_ast(s)
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

            exprAndReturns.append(self.convert_statement_ast(python_ast.Statement.Return(value=None, filename="", line_number=0, col_offset=0)))

        seq_expr = native_ast.Expression.Sequence(
            vals=[expr for expr, _ in exprAndReturns]
        )

        return seq_expr, flows_off_end

    def construct_stackslots_around(self, expr, argnames, stararg_name):
        to_add = []
        destructors = []

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
                    context = ExpressionConversionContext(self)

                    if slot_type.is_pod:
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
                        context.pushEffect(
                            context.isInitializedVarExpr(name).expr.store(native_ast.trueExpr)
                        )
                    else:
                        # need to make a stackslot for this variable
                        # the argument will be a pointer because it's POD
                        var_expr = context.inputArg(slot_type, name)

                        slot_expr = context.named_var_expr(name)

                        slot_type.convert_copy_initialize(context, slot_expr, var_expr)

                        context.pushEffect(
                            context.isInitializedVarExpr(name).expr.store(native_ast.trueExpr)
                        )

                    to_add.append(context.finalize(None))

        for name in self._varname_to_type:
            if name is not FunctionOutput and name != stararg_name:
                context = ExpressionConversionContext(self)

                if self._varname_to_type[name] is not None:
                    slot_expr = context.named_var_expr(name)

                    with context.ifelse(context.isInitializedVarExpr(name)) as (true, false):
                        with true:
                            slot_expr.convert_destroy()

                    destructors.append(
                        native_ast.Teardown.Always(
                            expr=context.finalize(None).with_comment("Cleanup for variable %s" % name)
                        )
                    )

                    if name not in argnames:
                        # this is a variable in the function that we assigned to. we need to ensure that
                        # the initializer flag is zero
                        context = ExpressionConversionContext(self)
                        context.pushEffect(context.isInitializedVarExpr(name).expr.store(native_ast.falseExpr))
                        to_add.append(context.finalize(None))

        if to_add:
            expr = native_ast.Expression.Sequence(
                vals=to_add + [expr]
            )

        if destructors:
            expr = native_ast.Expression.Finally(
                teardowns=destructors,
                expr=expr
            )

        return expr

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
