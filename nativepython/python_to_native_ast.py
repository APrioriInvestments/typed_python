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
from nativepython.expression_conversion_context import ExpressionConversionContext
from nativepython.type_wrappers.none_wrapper import NoneWrapper
from nativepython.python_object_representation import pythonObjectRepresentation, typedPythonTypeToTypeWrapper
from nativepython.typed_expression import TypedExpression
from nativepython.conversion_exception import ConversionException

NoneExprType = NoneWrapper()

from typed_python import *

typeWrapper = lambda t: nativepython.python_object_representation.typedPythonTypeToTypeWrapper(t)

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

class ExceptionHandlingHelper:
    def __init__(self, conversion_context):
        self.context = conversion_context

    def exception_pointer(self):
        return TypedExpression(
            native_ast.Expression.Cast(
                left=native_ast.Expression.Variable(".unnamed.exception.var"),
                to_type=self.InFlightException.getNativeLayoutType()
                ),
            InFlightException
            )

    def convert_tryexcept(self, ast):
        if len(ast.orelse):
            raise ConversionException("We dont handle try-except-else yet")
        if len(ast.finalbody):
            raise ConversionException("We dont handle try-except-finally yet")

        body = self.context.convert_statement_list_ast(ast.body)
        handlers = []

        handlers_and_conds = []

        for h in ast.handlers:
            if h.type is None:
                if h.name is not None:
                    raise ConversionException("Can't handle a typeless exception handler with a named variable")

                handlers_and_conds.append(
                    self.generate_exception_handler_expr(None, None, h.body)
                    )
            else:
                typexpr = self.context.consume_temporaries(self.context.convert_expression_ast(h.type.val))

                if not (typexpr.expr_type.is_compile_time
                        and isinstance(typexpr.expr_type.python_object_representation, Type)
                        ):
                    raise ConversionException("expected a type-constant, not %s" % typexpr.expr_type)

                handler_type = typexpr.expr_type.python_object_representation

                name = None
                if h.name is not None:
                    name = h.name.val
                    assert isinstance(name, str)

                handlers_and_conds.append(
                    self.generate_exception_handler_expr(handler_type, name, h.body)
                    )

        expr = self.context.consume_temporaries(self.generate_exception_rethrow())

        for cond, handler in reversed(handlers_and_conds):
            expr = TypedExpression(
                native_ast.Expression.Branch(
                    cond=cond.expr,
                    true=handler.expr,
                    false=expr.expr
                    ),
                NoneExprType if (handler.expr_type is not None or expr.expr_type is not None) else
                    None
                )

        return TypedExpression(
            native_ast.Expression.TryCatch(
                expr=body.expr,
                varname=".unnamed.exception.var",
                handler=expr.expr
                ),
            NoneExprType if body.expr_type is not None else None
            )

    def generate_exception_rethrow(self):
        import nativepython.lib.exception

        return TypedExpression(
            native_ast.Expression.Throw(
                native_ast.Expression.Variable(".unnamed.exception.var")
                ),
            None
            )

    def generate_exception_teardown(self):
        return self.context.consume_temporaries(
            self.context.call_py_function(
                nativepython.lib.exception.exception_teardown,
                [self.exception_pointer()]
                )
            )

    def generate_exception_handler_expr(self, val_type, binding_name, body):
        """Generate an expression that tests the current exception against
            'val_type', and if so, binds it into 'binding_name' and evaluates 'body'
        """
        if binding_name is not None and self.context._varname_to_type.get(binding_name, None) is not None:
            raise ConversionException("Variable %s is already defined" % binding_name)

        if binding_name is not None:
            if val_type is None:
                raise ConversionException("Can't bind a name without a type.")
            self.context._varname_to_type[binding_name] = val_type

        if val_type is None:
            cond_expr = TypedExpression(native_ast.trueExpr, Bool)
        else:
            cond_expr = self.context.consume_temporaries(
                self.context.call_py_function(
                    nativepython.lib.exception.exception_matches,
                    [pythonObjectRepresentation(self, val_type),
                     self.exception_pointer()
                     ]
                    )
                )

        if binding_name is not None:
            bind_expr = self.context.consume_temporaries(
                self.context.call_py_function(
                    nativepython.lib.exception.bind_exception_into,
                    [self.exception_pointer(),
                     self.context.named_var_expr(binding_name)
                     ]
                    )
                )
        else:
            bind_expr = self.context.consume_temporaries(self.generate_exception_teardown())

        handler_expr = TypedExpression.NoneExpr(
            native_ast.Expression.Finally(
                expr=self.context.convert_statement_list_ast(body).expr,
                teardowns=[
                    native_ast.Teardown.Always(
                        self.context.named_var_expr(binding_name).convert_destroy(self.context).expr
                        )
                    ] if binding_name is not None else []
                )
            )

        if binding_name is not None:
            del self.context._varname_to_type[binding_name]

        return (cond_expr, self.context.consume_temporaries(bind_expr + handler_expr))

class FunctionOutput:
    pass

class FunctionConversionContext(object):
    """Helper function for converting a single python function."""
    def __init__(self, converter, varname_to_type, free_variable_lookup):
        self.exception_helper = ExceptionHandlingHelper(self)
        self.converter = converter
        self._varname_to_type = varname_to_type
        self._free_variable_lookup = free_variable_lookup
        self._temp_let_var = 0
        self._temp_stack_var = 0
        self._typesAreUnstable = False
        self._functionOutputTypeKnown = FunctionOutput in varname_to_type

    def typesAreUnstable(self):
        return self._typesAreUnstable

    def resetTypeInstabilityFlag(self):
        self._typesAreUnstable = False

    def let_varname(self):
        self._temp_let_var += 1
        return ".letvar.%s" % (self._temp_let_var-1)

    def stack_varname(self):
        self._temp_stack_var += 1
        return ".stackvar.%s" % (self._temp_stack_var-1)

    def upsizeVariableType(self, varname, new_type):
        if self._varname_to_type.get(varname) is None:
            self._varname_to_type[varname] = new_type
            return

        existingType = self._varname_to_type[varname].typeRepresentation

        if existingType == new_type.typeRepresentation:
            return

        if hasattr(existingType, '__typed_python_category__') and \
                existingType.__typed_python_category__ == 'OneOf':
            if new_type.typeRepresentation in existingType.Types:
                return

        final_type = OneOf(new_type.typeRepresentation, existingType)

        self._typesAreUnstable = True
        self._varname_to_type[varname] = typeWrapper(final_type)

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

                self.upsizeVariableType(varname, val_to_store.expr_type)
                slot_ref = subcontext.named_var_expr(varname)

                #convert the value to the target type now that we've upsized it
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

                return subcontext.finalize(None).with_comment("Assign %s" % (varname)), True

            if target.matches.Subscript and target.ctx.matches.Store:
                raise NotImplementedError("Not implemented correctly yet")
                assert target.slice.matches.Index

                slicing = self.convert_expression_ast(target.value)
                index = self.convert_expression_ast(target.slice.value)
                val_to_store = self.convert_expression_ast(ast.value)

                if op is not None:
                    val_to_store = slicing.convert_getitem(index).convert_bin_op(op, val_to_store)

                return slicing.convert_setitem(index, val_to_store)

            if target.matches.Attribute and target.ctx.matches.Store:
                subcontext = ExpressionConversionContext(self)

                slicing = subcontext.convert_expression_ast(target.value)
                attr = target.attr
                val_to_store = subcontext.convert_expression_ast(ast.value)

                if op is not None:
                    raise NotImplementedError("Not implemented correctly.")
                    val_to_store = slicing.convert_attribute(attr).convert_bin_op(op, val_to_store)

                slicing.convert_set_attribute(attr, val_to_store)

                return subcontext.finalize(None), False

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
                return cond.finalize(None)
            cond = cond.toBool()
            if cond is None:
                return cond.finalize(None)

            if cond.expr.matches.Constant:
                truth_val = cond.expr.val.truth_value()

                branch, flow_returns = self.convert_statement_list_ast(ast.body if truth_val else ast.orelse)

                return cond.expr + branch, flow_returns

            true, true_returns = self.convert_statement_list_ast(ast.body)
            false, false_returns = self.convert_statement_list_ast(ast.orelse)

            return (
                native_ast.Expression.Branch(cond=cond_context.finalize(cond.nonref_expr),true=true,false=false),
                true_returns or false_returns
                )

        if ast.matches.Pass:
            return native_ast.nullExpr, True

        if ast.matches.While:
            cond_context = ExpressionConversionContext(self)
            cond = cond_context.convert_expression_ast(ast.test)
            if cond is None:
                return cond_context.finalize(None)
            cond = cond.toBool()
            if cond is None:
                return cond_context.finalize(None)

            true, true_returns = self.convert_statement_list_ast(ast.body)

            false, false_returns = self.convert_statement_list_ast(ast.orelse)

            return (
                native_ast.Expression.While(cond=cond_context.finalize(cond.nonref_expr),while_true=true,orelse=false),
                true_returns or false_returns
                )

        if ast.matches.Try:
            return self.exception_helper.convert_tryexcept(ast)

        if ast.matches.For:
            if not ast.target.matches.Name:
                raise ConversionException("For loops can only have simple targets for now")

            #this object needs to stay alive for the duration
            #of the expression
            iter_create_expr = (
                self.convert_expression_ast(ast.iter)
                    .convert_attribute("__iter__")
                    .convert_call([])
                )
            iter_type = iter_create_expr.expr_type.ensureNonReference_reference()

            iter_expr = self.allocate_temporary(iter_type)

            iter_setup_expr = (
                iter_type.convert_copy_initialize(iter_expr, iter_create_expr).expr +
                self.activates_temporary(iter_expr)
                )

            teardowns_for_iter = self.consume_temporaries(None)

            #now we need to generate a while loop
            while_cond_expr = (
                iter_expr.convert_attribute("has_next")
                    .convert_call([])
                    .convert_to_type(Bool, False)
                )

            while_cond_expr = self.consume_temporaries(while_cond_expr)

            next_val_expr = iter_expr.convert_attribute("next").convert_call([])

            if self._varname_to_type.get(ast.target.id, None) is not None:
                raise ConversionException(
                    "iterator target %s already bound to a variable of type %s" % (
                        ast.target.id,
                        self._varname_to_type.get(ast.target.id)
                        )
                    )

            self._varname_to_type[ast.target.id] = next_val_expr.expr_type

            next_val_ref = self.named_var_expr(ast.target.id)
            next_val_setup_native_expr = (
                next_val_expr.expr_type.convert_copy_initialize(next_val_ref, next_val_expr).expr +
                self.activates_temporary(next_val_ref)
                )
            next_val_teardowns = self.consume_temporaries(None)
            next_val_teardowns.append(
                self.named_variable_teardown(ast.target.id)
                )

            body_native_expr = native_ast.Expression.Finally(
                    expr =  next_val_setup_native_expr
                          + self.convert_statement_list_ast(ast.body).expr,
                    teardowns=next_val_teardowns
                    )

            del self._varname_to_type[ast.target.id]

            orelse_native_expr = self.convert_statement_list_ast(ast.orelse).expr

            res = self.NoneExpr(
                native_ast.Expression.Finally(
                    expr=iter_setup_expr +
                        native_ast.Expression.While(
                            cond=while_cond_expr.expr,
                            while_true=body_native_expr,
                            orelse=orelse_native_expr
                            ),
                    teardowns=teardowns_for_iter
                    )
                )

            return res

        if ast.matches.Raise:
            if ast.exc is None:
                #this is a naked raise
                raise ConversionException(
                    "We don't handle re-raise yet"
                    )

            if ast.exc is not None and ast.cause is None:
                expr = self.convert_expression_ast(ast.exc.val)

                import nativepython.lib.exception

                return self.call_py_function(
                    nativepython.lib.exception.throw,
                    [expr]
                    )
            else:
                raise ConversionException("We can only handle simple 'raise' statements")

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
                else:
                    self.upsizeVariableType(FunctionOutput, NoneWrapper())

            exprAndReturns.append(self.convert_statement_ast(python_ast.Statement.Return(value=None, filename="", line_number=0, col_offset=0)))

        seq_expr = native_ast.Expression.Sequence(
            vals=[expr for expr,_ in exprAndReturns]
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

                if slot_type is not None:
                    context = ExpressionConversionContext(self)

                    if slot_type.is_pod:
                        #we can just copy this into the stackslot directly. no destructor needed
                        context.pushEffect(
                            native_ast.Expression.Store(
                                ptr=native_ast.Expression.StackSlot(name=name,type=slot_type.getNativeLayoutType()),
                                val=
                                    native_ast.Expression.Variable(name=name) if not slot_type.is_pass_by_ref else
                                    native_ast.Expression.Variable(name=name).load()
                                )
                            )
                        context.pushEffect(
                            context.isInitializedVarExpr(name).expr.store(native_ast.trueExpr)
                            )
                    else:
                        #need to make a stackslot for this variable
                        #the argument will be a pointer because it's POD
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
                    #this is a variable in the function that we assigned to. we need to ensure that
                    #the initializer flag is zero
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

        return args_type.convert_initialize(
                self,
                stararg_slot,
                [TypedExpression(
                        native_ast.Expression.Variable(".star_args.%s" % i),
                        args_type.element_types[i][1]
                        )
                    for i in range(len(args_type.element_types))]
            ).with_comment("initialize *args slot") + res


class Converter(object):
    def __init__(self):
        object.__init__(self)
        self._names_for_identifier = {}
        self._definitions = {}
        self._targets = {}
        self._typeids = {}

        self._unconverted = set()

        self.verbose = False

    def get_typeid(self, t):
        if t in self._typeids:
            return self._typeids[t]

        self._typeids[t] = len(self._typeids) + 1024

        return self._typeids[t]

    def extract_new_function_definitions(self):
        res = {}

        for u in self._unconverted:
            res[u] = self._definitions[u]

            if self.verbose:
                print(self._targets[u])

        self._unconverted = set()

        return res

    def new_name(self, name, prefix="py."):
        suffix = None
        getname = lambda: prefix + name + ("" if suffix is None else ".%s" % suffix)
        while getname() in self._targets:
            suffix = 1 if not suffix else suffix+1
        return getname()

    def convert_function_ast(
                self,
                ast_arg,
                statements,
                input_types,
                local_variables,
                free_variable_lookup,
                output_type
                ):
        if ast_arg.vararg is not None:
            star_args_name = ast_arg.vararg.val.arg
        else:
            star_args_name = None

        if star_args_name is None:
            if len(input_types) != len(ast_arg.args):
                raise ConversionException(
                    "Expected %s arguments but got %s" % (len(ast_arg.args), len(input_types))
                    )
        else:
            if len(input_types) < len(ast_arg.args):
                raise ConversionException(
                    "Expected at least %s arguments but got %s" %
                        (len(ast_arg.args), len(input_types))
                    )

        varname_to_type = {}

        args = []
        for i in range(len(ast_arg.args)):
            varname_to_type[ast_arg.args[i].arg] = input_types[i]
            args.append((ast_arg.args[i].arg, input_types[i].getNativePassingType()))

        argnames = [a[0] for a in args]

        if star_args_name is not None:
            star_args_count = len(input_types) - len(ast_arg.args)

            starargs_elts = []
            for i in range(len(ast_arg.args), len(input_types)):
                args.append(
                    ('.star_args.%s' % (i - len(ast_arg.args)),
                        input_types[i].getNativePassingType())
                    )

            starargs_type = Struct(
                [('f_%s' % i, input_types[i+len(ast_arg.args)])
                    for i in range(star_args_count)]
                )

            varname_to_type[star_args_name] = starargs_type

        if output_type is not None:
            varname_to_type[FunctionOutput] = typeWrapper(output_type)

        functionConverter = FunctionConversionContext(self, varname_to_type, free_variable_lookup)

        while True:
            #repeatedly try converting as long as the types keep getting bigger.
            body_native_expr, controlFlowReturns = functionConverter.convert_function_body(statements)
            assert not controlFlowReturns

            if functionConverter.typesAreUnstable():
                functionConverter.resetTypeInstabilityFlag()
            else:
                break

        if star_args_name is not None:
            body_native_expr = functionConverter.construct_starargs_around(body_native_expr, star_args_name)

        body_native_expr = functionConverter.construct_stackslots_around(body_native_expr, argnames, star_args_name)

        return_type = functionConverter._varname_to_type.get(FunctionOutput, NoneExprType)

        if return_type.is_pass_by_ref:
            return (
                native_ast.Function(
                    args=(('.return',return_type.getNativeLayoutType().pointer()),) + tuple(args),
                    body=native_ast.FunctionBody.Internal(body=body_native_expr),
                    output_type=native_ast.Void
                    ),
                return_type
                )
        else:
            return (
                native_ast.Function(
                    args=args,
                    body=native_ast.FunctionBody.Internal(body=body_native_expr),
                    output_type=return_type.getNativeLayoutType()
                    ),
                return_type
                )


    def convert_lambda_ast(self, ast, input_types, local_variables, free_variable_lookup, output_type):
        return self.convert_function_ast(
            ast.args,
            [python_ast.Statement.Return(
                value=ast.body,
                line_number=ast.body.line_number,
                col_offset=ast.body.col_offset,
                filename=ast.body.filename
                )],
            input_types,
            local_variables,
            free_variable_lookup,
            output_type
            )

    def defineNativeFunction(self, name, identity, input_types, output_type, generatingFunction):
        """Define a native function if we haven't defined it before already.

            name - the name to actually give the function.
            identity - a unique identifier for this function to allow us to cache it.
            input_types - list of Wrapper objects for the incoming types
            output_ype - Wrapper object for the output type.
            generatingFunction - a function producing a native_function_definition

        returns a TypedCallTarget. 'generatingFunction' may call this recursively if it wants.
        """
        identity = ("native", identity)

        if identity in self._names_for_identifier:
            return self._targets[self._names_for_identifier[identity]]

        new_name = self.new_name(name, "runtime.")

        self._names_for_identifier[identity] = new_name

        native_input_types = [t.getNativePassingType() for t in input_types]

        if output_type.is_pass_by_ref:
            #the first argument is actually the output
            native_output_type = native_ast.Void
            native_input_types = [output_type.getNativePassingType()] + native_input_types
        else:
            native_output_type = output_type.getNativeLayoutType()

        self._targets[new_name] = TypedCallTarget(
            native_ast.NamedCallTarget(
                name=new_name,
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

        self._definitions[new_name] = generatingFunction()
        self._unconverted.add(new_name)

        return self._targets[new_name]

    def define(self, identifier, name, input_types, output_type, native_function_definition):
        identifier = ("defined", identifier)

        if identifier in self._names_for_identifier:
            name = self._names_for_identifier[identifier]

            return self._targets[name]

        new_name = self.new_name(name)
        self._names_for_identifier[identifier] = new_name

        self._targets[new_name] = TypedCallTarget(
            native_ast.NamedCallTarget(
                name=new_name,
                arg_types=[x[1] for x in native_function_definition.args],
                output_type=native_function_definition.output_type,
                external=False,
                varargs=False,
                intrinsic=False,
                can_throw=True
                ),
            input_types,
            output_type
            )

        self._definitions[new_name] = native_function_definition
        self._unconverted.add(new_name)

        return self._targets[new_name]

    def callable_to_ast_and_vars(self, f):
        pyast = ast_util.pyAstFor(f)

        _, lineno = ast_util.getSourceLines(f)
        _, fname = ast_util.getSourceFilenameAndText(f)

        pyast = ast_util.functionDefOrLambdaAtLineNumber(pyast, lineno)

        pyast = python_ast.convertPyAstToAlgebraic(pyast, fname)

        freevars = dict(f.__globals__)

        if f.__closure__:
            for i in range(len(f.__closure__)):
                freevars[f.__code__.co_freevars[i]] = f.__closure__[i].cell_contents

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

        underlyingDefinition = self._definitions[callTarget.name]

        args = []
        for i in range(len(callTarget.input_types)):
            argtype = callTarget.input_types[i].getNativeLayoutType()

            untypedPtr = native_ast.var('input').ElementPtrIntegers(i).load()

            if callTarget.input_types[i].is_pass_by_ref:
                #we've been handed a pointer, and it's already a pointer
                args.append(untypedPtr.cast(argtype.pointer()))
            else:
                args.append(untypedPtr.cast(argtype.pointer()).load())

        if callTarget.output_type.is_pass_by_ref:
            body = callTarget.call(
                native_ast.var('return').cast(callTarget.output_type.getNativeLayoutType().pointer()),
                *args
                )
        else:
            body = callTarget.call(*args)

            if not callTarget.output_type.is_empty:
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
        self._unconverted.add(new_name)

        return new_name

    def convert(self, f, input_types, output_type):
        input_types = tuple([typedPythonTypeToTypeWrapper(i) for i in input_types])

        identifier = ("pyfunction", f, input_types)

        if identifier in self._names_for_identifier:
            name = self._names_for_identifier[identifier]
            return self._targets[name]

        pyast, freevars = self.callable_to_ast_and_vars(f)

        if isinstance(pyast, python_ast.Statement.FunctionDef):
            definition, output_type = \
                self.convert_function_ast(
                    pyast.args,
                    pyast.body,
                    input_types,
                    f.__code__.co_varnames,
                    freevars,
                    output_type
                    )
        else:
            assert pyast.matches.Lambda

            definition,output_type = self.convert_lambda_ast(pyast, input_types, f.__code__.co_varnames, freevars, output_type)

        assert definition is not None

        new_name = self.new_name(f.__name__)

        self._names_for_identifier[identifier] = new_name

        self._targets[new_name] = TypedCallTarget(
            native_ast.NamedCallTarget(
                name=new_name,
                arg_types=[x[1] for x in definition.args],
                output_type=definition.output_type,
                external=False,
                varargs=False,
                intrinsic=False,
                can_throw=True
                ),
            input_types,
            output_type
            )

        self._definitions[new_name] = definition
        self._unconverted.add(new_name)

        return self._targets[new_name]

