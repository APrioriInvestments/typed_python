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
from nativepython.python_object_representation import pythonObjectRepresentation, typedPythonTypeToTypeWrapper
from nativepython.typed_expression import TypedExpression
from nativepython.conversion_exception import ConversionException

NoneExprType = NoneWrapper()

from typed_python import *

class TypedCallTarget(object):
    def __init__(self, named_call_target, input_types, output_type):
        super().__init__()

        self.named_call_target = named_call_target
        self.input_types = input_types
        self.output_type = output_type

    @property
    def name(self):
        return self.named_call_target.name

    def __str__(self):
        return "TypedCallTarget(name=%s,inputs=%s,outputs=%s)" % (
            self.name,
            [str(x) for x in self.input_types],
            str(self.output_type)
            )

class FunctionOutput:
    pass

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


class ConversionContext(object):
    def __init__(self, converter, varname_to_type, free_variable_lookup, init_fields):
        self.exception_helper = ExceptionHandlingHelper(self)
        self.converter = converter
        self._varname_to_type = varname_to_type
        self._varname_and_type_to_slot_name = {}
        self._varname_uses = {}
        self._new_variables = set()
        self._free_variable_lookup = free_variable_lookup
        self._temporaries = {}
        self._new_temporaries = set()
        self._temp_let_var = 0

    def let_varname(self):
        self._temp_let_var += 1
        return ".letvar.%s" % (self._temp_let_var-1)

    def let(self, e1, e2):
        v = self.let_varname()
        return native_ast.Expression.Let(
            var=v,
            val=e1,
            within=e2(v)
            )

    def NoneExpr(self, expr=None):
        return TypedExpression(self, expr if expr is not None else native_ast.nullExpr, NoneWrapper(), False)

    def TerminalExpr(self, expr):
        return TypedExpression(self, expr, None, False)

    def ValueExpr(self, expr, type):
        return TypedExpression(self, expr, type, False)

    def RefExpr(self, expr, type):
        return TypedExpression(self, expr, type, True)

    def activates_temporary(self, slot):
        if slot.expr_type.is_pod:
            return native_ast.nullExpr

        return native_ast.Expression.ActivatesTeardown(name=slot.expr.name)

    def allocate_temporary(self, slot_type):
        tname = '.temp.%s' % len(self._temporaries)
        self._new_temporaries.add(tname)
        self._temporaries[tname] = slot_type

        return self.RefExpr(
            native_ast.Expression.StackSlot(name=tname,type=slot_type.getNativeLayoutType()),
            slot_type
            )

    def named_var_expr(self, name):
        if self._varname_to_type[name] is None:
            raise ConversionException(
                "variable %s is not in scope here" % name
                )

        slot_type = self._varname_to_type[name]

        if name not in self._varname_uses:
            self._varname_uses[name] = 0

        if (slot_type, name) not in self._varname_and_type_to_slot_name:
            self._varname_uses[name] += 1
            if self._varname_uses[name] > 1:
                name_to_use = name + ".%s" % self._varname_uses[name]
            else:
                name_to_use = name

            self._varname_and_type_to_slot_name[slot_type,name] = name_to_use

        name_to_use = self._varname_and_type_to_slot_name[slot_type,name]

        return TypedExpression(
            self,
            native_ast.Expression.StackSlot(
                name=name_to_use,
                type=slot_type.getNativeLayoutType()
                ),
            slot_type,
            isReference=True
            )

    def call_py_function(self, f, args, name_override=None):
        #force arguments to a type appropriate for argpassing
        args = [a.as_call_arg() for a in args]
        native_args = [a.expr for a in args]

        call_target = \
            self.converter.convert(
                f,
                [a.expr_type for a in args],
                name_override=name_override
                )

        assert len(call_target.named_call_target.arg_types) == len(args)

        return self.ValueExpr(
            self.generate_call_expr(
                target=call_target.named_call_target,
                args=native_args
                ),
            call_target.output_type
            )

    def generate_call_expr(self, target, args):
        lets = []
        actual_args = []

        for a in args:
            while a.matches.Sequence:
                for sub_expr in a.vals[:-1]:
                    lets.append((None, sub_expr))
                a = a.vals[-1]

            if not a.is_simple():
                name = self.let_varname()
                lets.append((name, a))
                actual_args.append(native_ast.Expression.Variable(name=name))
            else:
                actual_args.append(a)

        e = native_ast.Expression.Call(
            target=native_ast.CallTarget.Named(target=target),
            args=actual_args
            )

        for k,v in reversed(lets):
            if k is not None:
                e = native_ast.Expression.Let(
                    var=k,
                    val=v,
                    within=e
                    )
            else:
                e = v + e

        return e



    def convert_expression_ast(self, ast):
        return self.convert_expression_ast_(ast)

    def wrapInTemporaries(self, fun, args):
        """Call 'fun' with args that are all simple reference to values or nonref pod."""
        for i,a in enumerate(args):
            if not (a.isReference or a.expr_type.is_pod):
                #'a' contains an increffed value that will be valid for the duration of the statement
                #we're in. we'll move it to a stack slot. We assume all of our types are moveable (no self pointers to the stack)
                slot = self.allocate_temporary(a.expr_type)

                result = self.wrapInTemporaries(fun, args[:i] + (slot,) + args[i+1:])

                return TypedExpression(
                    self,
                    slot.expr.store(a.expr)
                        >> result.expr,
                    result.expr_type,
                    result.isReference
                    )

            elif not a.expr.is_simple():
                let_varname = self.let_varname()

                result = self.wrapInTemporaries(
                    fun, 
                    args[:i]
                        + (TypedExpression(self, native_ast.Expression.Variable(name=let_varname), a.expr_type, a.isReference),) 
                        + args[i+1:]
                    )

                return TypedExpression(
                    self,
                    native_ast.Expression.Let(
                        var=let_varname,
                        val=a.expr,
                        within=result.expr
                        ),
                    result.expr_type,
                    result.isReference
                    )

        return fun(*args)

    def convert_expression_ast_(self, ast):
        if ast.matches.Attribute:
            attr = ast.attr
            val = self.convert_expression_ast(ast.value)

            return val.convert_attribute(attr)

        if ast.matches.Name:
            assert ast.ctx.matches.Load
            if ast.id in self._varname_to_type:
                return self.named_var_expr(ast.id)

            if ast.id in self._free_variable_lookup:
                return pythonObjectRepresentation(self, self._free_variable_lookup[ast.id])

            elif ast.id in __builtins__:
                return pythonObjectRepresentation(self, __builtins__[ast.id])

            if ast.id not in self._varname_to_type:
                raise ConversionException(
                    "can't find variable %s"  % ast.id
                    )

        if ast.matches.Num:
            if ast.n.matches.None_:
                return pythonObjectRepresentation(self, None)
            if ast.n.matches.Boolean:
                return pythonObjectRepresentation(self, bool(ast.n.value))
            if ast.n.matches.Int:
                return pythonObjectRepresentation(self, int(ast.n.value))
            if ast.n.matches.Float:
                return pythonObjectRepresentation(self, float(ast.n.value))

        if ast.matches.Str:
            return pythonObjectRepresentation(self, ast.s)

        if ast.matches.BoolOp:
            def exprForVals(values):
                op = ast.op
                
                expr_so_far = []

                for i in range(len(values)):
                    expr_so_far.append(values[i].toBool().expr)
                    if expr_so_far[-1].matches.Constant:
                        if (expr_so_far[-1].val.val and op.matches.Or or
                                    (not expr_so_far[-1].val.val) and op.matches.And):
                            #this is a short-circuit
                            if len(expr_so_far) == 1:
                                return self.ValueExpr(expr_so_far[0], Bool)

                            return self.ValueExpr(
                                native_ast.Expression.Sequence(expr_so_far),
                                Bool
                                )
                        else:
                            expr_so_far.pop()

                if not expr_so_far:
                    if op.matches.Or:
                        #must have had all False constants
                        return self.ValueExpr(native_ast.falseExpr, Bool)
                    else:
                        #must have had all True constants
                        return self.ValueExpr(native_ast.trueExpr, Bool)

                while len(expr_so_far) > 1:
                    l,r = expr_so_far[-2], expr_so_far[-1]
                    expr_so_far.pop()
                    expr_so_far.pop()
                    if op.matches.And:
                        new_expr = native_ast.Expression.Branch(cond=l, true=r, false=native_ast.falseExpr)
                    else:
                        new_expr = native_ast.Expression.Branch(cond=l, true=native_ast.trueExpr, false=r)
                    expr_so_far.append(new_expr)

                return self.ValueExpr(expr_so_far[0], Bool)

            return self.wrapInTemporaries(exprForVals, tuple(self.convert_expression_ast(v).toBool() for v in ast.values))

        if ast.matches.BinOp:
            l = self.convert_expression_ast(ast.left)
            r = self.convert_expression_ast(ast.right)

            return l.convert_bin_op(ast.op, r)

        if ast.matches.UnaryOp:
            operand = self.convert_expression_ast(ast.operand)

            return operand.convert_unary_op(ast.op)

        if ast.matches.Subscript:
            assert ast.slice.matches.Index

            val = self.convert_expression_ast(ast.value)
            index = self.convert_expression_ast(ast.slice.value)

            return val.convert_getitem(index)

        if ast.matches.Call:
            l = self.convert_expression_ast(ast.func)

            ast_args = ast.args
            stararg = None

            for a in ast_args:
                assert not a.matches.Starred, "not implemented yet"

            args = [self.convert_expression_ast(a) for a in ast_args]

            return l.convert_call(args)

        if ast.matches.Compare:
            assert len(ast.comparators) == 1, "multi-comparison not implemented yet"
            assert len(ast.ops) == 1

            l = self.convert_expression_ast(ast.left)
            r = self.convert_expression_ast(ast.comparators[0])

            return l.convert_bin_op(ast.ops[0], r)

        if ast.matches.Tuple:
            raise NotImplementedError("not implemented yet")

        if ast.matches.IfExp:
            test = self.convert_expression_ast(ast.test).toBool()
            body = self.convert_expression_ast(ast.body)
            orelse = self.convert_expression_ast(ast.orelse)

            if body.expr_type != orelse.expr_type:
                if body.expr_type.nonref_type == orelse.expr_type.nonref_type and \
                        body.expr_type.nonref_type.is_pod:
                    body = body.dereference()
                    orelse = orelse.dereference()
                else:
                    raise ConversionException("Expected IfExpr to have the same type, but got " +
                        "%s and %s" % (body.expr_type, orelse.expr_type))

            assert body.isReference == orelse.isReference

            return TypedExpression(
                    self,
                    native_ast.Expression.Branch(
                        cond=test.expr,
                        true=body.expr,
                        false=orelse.expr
                        ),
                    body.expr_type,
                    body.isReference
                    )

        raise ConversionException("can't handle python expression type %s" % ast._which)

    def convert_statement_ast_and_teardown_tmps(self, ast):
        if self._new_temporaries:
            raise ConversionException("Expected no temporaries on %s. instead have of types %s" %
                (ast._which, [str(self._temporaries[x]) for x in self._new_temporaries])
                )

        expr = self.convert_statement_ast(ast)

        return self.consume_temporaries(expr)

    def consume_temporaries(self, expr):
        teardowns = []

        for tname in sorted(self._new_temporaries):
            teardown = native_ast.Teardown.ByTag(
                tag=tname,
                expr=self._temporaries[tname].convert_destroy(
                    self,
                    TypedExpression(
                        native_ast.Expression.StackSlot(
                            name=tname,
                            type=self._temporaries[tname].getNativeLayoutType()
                            ),
                        self._temporaries[tname],
                        True
                        )
                    ).expr
                )

            if not teardown.expr.matches.Constant:
                teardowns.append(teardown)

        self._new_temporaries = set()

        if expr is None:
            return teardowns

        if not teardowns:
            return expr

        return TypedExpression(
            native_ast.Expression.Finally(expr=expr.expr, teardowns=teardowns),
            expr.expr_type,
            expr.isReference
            )

    def convert_statement_ast(self, ast):
        return self.convert_statement_ast_(ast)

    def convert_statement_ast_(self, ast):
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

                val_to_store = self.convert_expression_ast(ast.value)

                if self._varname_to_type[varname] is None:
                    self._new_variables.add(varname)

                    new_variable_type = val_to_store.expr_type

                    self._varname_to_type[varname] = new_variable_type

                    slot_ref = self.named_var_expr(varname)

                    #this is a new variable which we are constructing
                    return self._varname_to_type[varname].convert_copy_initialize(
                        self,
                        slot_ref,
                        val_to_store
                        ) >> self.NoneExpr(native_ast.Expression.ActivatesTeardown(name=varname))
                else:
                    #this is an existing variable.
                    slot_ref = self.named_var_expr(varname)

                    if op is not None:
                        val_to_store = slot_ref.convert_bin_op(op, val_to_store)

                    return slot_ref.convert_assign(val_to_store)

            if target.matches.Subscript and target.ctx.matches.Store:
                assert target.slice.matches.Index

                slicing = self.convert_expression_ast(target.value)
                index = self.convert_expression_ast(target.slice.value)
                val_to_store = self.convert_expression_ast(ast.value)

                if op is not None:
                    val_to_store = slicing.convert_getitem(index).convert_bin_op(op, val_to_store)

                return slicing.convert_setitem(index, val_to_store)

            if target.matches.Attribute and target.ctx.matches.Store:
                slicing = self.convert_expression_ast(target.value)
                attr = target.attr
                val_to_store = self.convert_expression_ast(ast.value)

                if op is not None:
                    val_to_store = slicing.convert_attribute(attr).convert_bin_op(op, val_to_store)

                return slicing.convert_set_attribute(attr, val_to_store)

        if ast.matches.Return:
            if ast.value is None:
                e = self.NoneExpr()
            else:
                e = self.convert_expression_ast(ast.value)

            if e.expr_type is None:
                return e
            
            if self._varname_to_type[FunctionOutput] is not None:
                if self._varname_to_type[FunctionOutput] != e.expr_type:
                    raise ConversionException(
                        "Function returning multiple types:\n\t%s\n\t%s" % (
                                e.expr_type,
                                self._varname_to_type[FunctionOutput]
                                )
                        )
            else:
                self._varname_to_type[FunctionOutput] = e.expr_type

            if e.isReference:
                return TypedExpression(
                    self, 
                    e.convert_incref().expr >> 
                        native_ast.Expression.Return(arg=e.nonref_expr),
                    None,
                    False
                    )
            else:
                return TypedExpression(
                    self, 
                    native_ast.Expression.Return(arg=e.nonref_expr),
                    None,
                    False
                    )

        if ast.matches.Expr:
            return self.NoneExpr(
                self.convert_expression_ast(ast.value).expr + native_ast.nullExpr
                )

        if ast.matches.If:
            cond = self.convert_expression_ast(ast.test).toBool()
            cond = self.consume_temporaries(cond)

            if cond.expr.matches.Constant:
                truth_val = cond.expr.val.truth_value()
                branch = self.convert_statement_list_ast(ast.body if truth_val else ast.orelse)

                return cond + branch

            true = self.convert_statement_list_ast(ast.body)
            false = self.convert_statement_list_ast(ast.orelse)

            if true.expr_type is None and false.expr_type is None:
                ret_type = None
            else:
                ret_type = NoneExprType

            return TypedExpression(
                native_ast.Expression.Branch(cond=cond.expr,true=true.expr,false=false.expr),
                ret_type
                )

        if ast.matches.Pass:
            return TypedExpression(native_ast.nullExpr, NoneExprType)

        if ast.matches.While:
            cond = self.convert_expression_ast(ast.test).toBool()
            cond = self.consume_temporaries(cond)

            true = self.convert_statement_list_ast(ast.body)
            false = self.convert_statement_list_ast(ast.orelse)

            if true.expr_type or false.expr_type:
                ret_type = typedPythonTypeToTypeWrapper(NoneType())
            else:
                ret_type = None

            return self.ValueExpr(
                native_ast.Expression.While(cond=cond.expr,while_true=true.expr,orelse=false.expr),
                ret_type
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

    def convert_statement_list_ast(self, statements, toplevel=False):
        orig_vars_in_scope = set(self._varname_to_type)

        if not statements:
            if toplevel:
                return self.TerminalExpr(native_ast.Expression.Return(None))

            return self.NoneExpr()

        exprs = []
        for s in statements:
            conversion = self.convert_statement_ast_and_teardown_tmps(s)
            exprs.append(conversion)

            if conversion.expr_type is None:
                break


        if exprs[-1].expr_type is not None:
            flows_off_end = True
        else:
            flows_off_end = False

        if toplevel and flows_off_end:
            if self._varname_to_type[FunctionOutput] is None:
                self._varname_to_type[FunctionOutput] = NoneWrapper()
            if isinstance(self._varname_to_type[FunctionOutput], NoneWrapper):
                exprs = exprs + [self.NoneExpr(native_ast.Expression.Return(arg=None))]
                flows_off_end = False
            else:
                raise ConversionException(
                    "Not all control flow paths return a value and the function returns %s" %
                        self._varname_to_type[FunctionOutput]
                    )

        teardowns = []
        for v in list(self._varname_to_type.keys()):
            if v not in orig_vars_in_scope:
                teardowns.append(self.named_variable_teardown(v))

                del self._varname_to_type[v]

        seq_expr = native_ast.Expression.Sequence(
                vals=[e.expr for e in exprs]
                )
        if teardowns:
            seq_expr = native_ast.Expression.Finally(
                expr=seq_expr,
                teardowns=teardowns
                )

        return TypedExpression(self, seq_expr, typedPythonTypeToTypeWrapper(NoneType()) if flows_off_end else None, False)

    def named_variable_teardown(self, v):
        return native_ast.Teardown.ByTag(
                tag=v,
                expr=self._varname_to_type[v].convert_destroy(
                    self,
                    self.RefExpr(
                        native_ast.Expression.StackSlot(
                            name=v,
                            type=self._varname_to_type[v].getNativeLayoutType()
                            ),
                        self._varname_to_type[v]
                        )
                    ).expr
                )

    def construct_stackslots_around(self, expr, argnames, stararg_name):
        to_add = []
        destructors = []
        for name in argnames:
            if name is not FunctionOutput and name != stararg_name:
                if name not in self._varname_to_type:
                    raise ConversionException("Couldn't find a type for argument %s" % name)
                slot_type = self._varname_to_type[name]

                if slot_type is not None:
                    if slot_type.is_pod:
                        #we can just copy this into the stackslot directly. no destructor needed
                        to_add.append(
                            native_ast.Expression.Store(
                                ptr=native_ast.Expression.StackSlot(name=name,type=slot_type.getNativeLayoutType()),
                                val=native_ast.Expression.Variable(name=name)
                                )
                            )
                    else:
                        #need to make a stackslot for this variable
                        #the argument will be a pointer because it's POD
                        var_expr = self.RefExpr(
                            native_ast.Expression.Variable(name=name),
                            slot_type
                            )

                        slot_expr = self.RefExpr(
                            native_ast.Expression.StackSlot(name=name,type=slot_type.getNativeLayoutType()),
                            slot_type
                            )

                        to_add.append(
                            slot_type.convert_copy_initialize(self, slot_expr, var_expr)
                                .expr.with_comment("initialize %s from arg" % name)
                            )

                        destructors.append(
                            native_ast.Teardown.Always(
                                expr=slot_type.convert_destroy(self, slot_expr).expr
                                )
                            )

        if to_add:
            expr = TypedExpression(
                self,
                native_ast.Expression.Sequence(
                    vals=to_add + [expr.expr]
                    ),
                expr.expr_type,
                expr.isReference
                )

        if destructors:
            expr = TypedExpression(
                self,
                native_ast.Expression.Finally(
                    teardowns=destructors,
                    expr=expr.expr
                    ),
                expr.expr_type,
                expr.isReference
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

    def new_name(self, name):
        suffix = None
        getname = lambda: "py." + name + ("" if suffix is None else ".%s" % suffix)
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
                members_of_arg0_to_initialize
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

        varname_to_type[FunctionOutput] = None

        if members_of_arg0_to_initialize:
            init_fields = InitFields(
                local_variables[0],
                members_of_arg0_to_initialize
                )
        else:
            init_fields = None

        subconverter = ConversionContext(self, varname_to_type, free_variable_lookup, init_fields)

        res = subconverter.convert_statement_list_ast(statements, toplevel=True)

        if init_fields:
            res = init_fields.init_expr(subconverter) + res

        if star_args_name is not None:
            res = subconverter.construct_starargs_around(res, star_args_name)

        res = subconverter.construct_stackslots_around(res, argnames, star_args_name)

        return_type = subconverter._varname_to_type[FunctionOutput] or NoneExprType

        return (
            native_ast.Function(
                args=args,
                body=native_ast.FunctionBody.Internal(body=res.expr),
                output_type=return_type.getNativeLayoutType()
                ),
            return_type
            )

    def convert_lambda_ast(self, ast, input_types, local_variables, free_variable_lookup):
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
            ()
            )

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

    def convert_lambda_as_expression(self, lambda_func):
        ast, free_variable_lookup = self.callable_to_ast_and_vars(lambda_func)

        varname_to_type = {}

        subconverter = ConversionContext(self, varname_to_type, free_variable_lookup, None)

        return subconverter.convert_expression_ast(ast.body)

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

    def convert_initializer_function(self, f, input_types, name_override, fields_and_types):
        return self.convert(f, input_types, name_override, fields_and_types)

    def generateCallConverter(self, callTarget, returnType):
        """Given a call target that's optimized for C-style dispatch, produce a (native) call-target that
        we can dispatch to from our C extension.

        we are given
            T f(A1, A2, A3 ...)
        and want to produce
            f(T*, X**)
        where X is the union of A1, A2, etc.

        returns the name of the defined native function
        """
        if callTarget.output_type != returnType: 
            raise Exception("Can't call a call target whose output type is %s when our return type is %s" % 
                (callTarget.output_type, returnType))

        identifier = ("call_converter", callTarget.name)

        if identifier in self._names_for_identifier:
            return self._names_for_identifier[identifier]


        underlyingDefinition = self._definitions[callTarget.name]

        args = []
        for i in range(len(underlyingDefinition.args)):
            argname, argtype = underlyingDefinition.args[i]

            untypedPtr = native_ast.var('input').ElementPtrIntegers(i).load()

            if callTarget.input_types[i].is_pass_by_ref:
                #we've been handed a pointer, and it's already a pointer
                args.append(untypedPtr.cast(argtype))
            else:
                args.append(untypedPtr.cast(argtype.pointer()).load())

        body = native_ast.Expression.Call(
            target=native_ast.CallTarget.Named(target=callTarget.named_call_target),
            args=args
            )

        if not underlyingDefinition.output_type.matches.NoneExprType:
            body = native_ast.var('return').cast(underlyingDefinition.output_type.pointer()).store(body)

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

    def convert(self, f, input_types, name_override=None, fields_and_types_for_initializing=None):
        input_types = tuple([typedPythonTypeToTypeWrapper(i) for i in input_types])

        identifier = ("pyfunction", f, input_types)

        if identifier in self._names_for_identifier:
            name = self._names_for_identifier[identifier]
            return self._targets[name]

        pyast, freevars = self.callable_to_ast_and_vars(f)

        if isinstance(pyast, python_ast.Statement.FunctionDef):
            definition,output_type = \
                self.convert_function_ast(
                    pyast.args,
                    pyast.body,
                    input_types,
                    f.__code__.co_varnames,
                    freevars,
                    fields_and_types_for_initializing
                    )
        else:
            assert pyast.matches.Lambda
            if fields_and_types_for_initializing:
                raise ConversionException("initializers can't be lambdas")
            definition,output_type = self.convert_lambda_ast(pyast, input_types, f.__code__.co_varnames, freevars)

        assert definition is not None

        new_name = self.new_name(name_override or f.__name__)

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

