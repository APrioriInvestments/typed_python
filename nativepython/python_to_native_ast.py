#   Copyright 2017 Braxton Mckee
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

import nativepython.python_ast as python_ast
import nativepython.python.ast_util as ast_util
import nativepython.native_ast as native_ast
import nativepython.exceptions as exceptions
from nativepython.type_model import *

class FunctionOutput:
    pass

class ConversionContext(object):
    def __init__(self, converter, varname_to_type, free_variable_lookup):
        self._converter = converter
        self._varname_to_type = varname_to_type
        self._new_variables = set()
        self._free_variable_lookup = free_variable_lookup
        self._temporaries = {}
        self._new_temporaries = set()
        self._temp_let_var = 0

    def let_varname(self):
        self._temp_let_var += 1
        return ".letvar.%s" % (self._temp_let_var-1)

    def activates_temporary(self, slot):
        if slot.expr_type.value_type.is_pod:
            return native_ast.nullExpr

        return native_ast.Expression.ActivatesTeardown(slot.expr.name)

    def allocate_temporary(self, slot_type):
        tname = '.temp.%s' % len(self._temporaries)
        self._new_temporaries.add(tname)
        self._temporaries[tname] = slot_type

        return TypedExpression(
            native_ast.Expression.StackSlot(name=tname,type=slot_type.lower()),
            slot_type.reference
            )

    def named_var_expr(self, name):
        if self._varname_to_type[name] is None:
            raise ConversionException(
                "variable %s is not in scope here" % name
                )

        slot_type = self._varname_to_type[name]

        return TypedExpression(
            native_ast.Expression.StackSlot(
                name=name,
                type=slot_type.lower()
                ),
            slot_type.reference
            )

    def call_py_function(self, f, args, name_override=None):
        #force arguments to a type appropriate for argpassing
        #e.g. drop out "CreateReference" and other syntactic sugar
        args = [a.as_call_arg(self) for a in args]
        native_args = [a.expr for a in args]

        call_target = \
            self._converter.convert(
                f, 
                [a.expr_type for a in args], 
                name_override=name_override
                )

        if not call_target.output_type.is_pod:
            assert len(call_target.native_call_target.arg_types) == len(args) + 1

            slot = self.allocate_temporary(call_target.output_type)

            return TypedExpression(
                self.generate_call_expr(
                    target=call_target.native_call_target,
                    args=[slot.expr] + native_args
                    ) 
                    + self.activates_temporary(slot)
                    + slot.expr
                    ,
                call_target.output_type.reference
                )
        else:
            assert len(call_target.native_call_target.arg_types) == len(args)

            return TypedExpression(
                self.generate_call_expr(
                    target=call_target.native_call_target,
                    args=native_args
                    ),
                call_target.output_type
                )

    def generate_call_expr(self, target, args):
        lets = []
        actual_args = []

        def is_simple(expr):
            if expr.matches.StackSlot:
                return True
            if expr.matches.Constant:
                return True
            if expr.matches.Variable:
                return True
            if expr.matches.Load:
                return is_simple(expr.ptr)
            return False

        for a in args:
            if not is_simple(a):
                name = self.let_varname()
                lets.append((name, a))
                actual_args.append(native_ast.Expression.Variable(name))
            else:
                actual_args.append(a)

        e = native_ast.Expression.Call(target=target, args=actual_args)

        for k,v in reversed(lets):
            e = native_ast.Expression.Let(
                var=k,
                val=v,
                within=e
                )
        
        return e


    def convert_expression_ast(self, ast):
        try:
            return self.convert_expression_ast_(ast)
        except ConversionException as e:
            e.add_scope(
                exceptions.ConversionScopeInfo.CreateFromAst(ast, self._varname_to_type)
                )
            raise

    def convert_expression_ast_(self, ast):
        if ast.matches.Attribute:
            attr = ast.attr
            val = self.convert_expression_ast(ast.value)
            return val.convert_attribute(self, attr)

        if ast.matches.Name:
            assert ast.ctx.matches.Load
            if ast.id in self._varname_to_type:
                return self.named_var_expr(ast.id).drop_double_references()

            if ast.id in self._free_variable_lookup:
                return pythonObjectRepresentation(self._free_variable_lookup[ast.id])
            elif ast.id in __builtins__:
                return pythonObjectRepresentation(__builtins__[ast.id])

            if ast.id not in self._varname_to_type:
                raise ConversionException(
                    "can't find variable %s"  % ast.id
                    )

        if ast.matches.Num:
            if ast.n.matches.Int:
                return TypedExpression(
                    native_ast.Expression.Constant(
                        native_ast.Constant.Int(val=ast.n.value,bits=64,signed=True)
                        ), 
                    Int64
                    )
            if ast.n.matches.Float:
                return TypedExpression(
                    native_ast.Expression.Constant(
                        native_ast.Constant.Float(val=ast.n.value,bits=64)
                        ), 
                    Float64
                    )

        if ast.matches.Str:
            return pythonObjectRepresentation(ast.s)

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

            return val.convert_getitem(self, index)

        if ast.matches.Call:
            l = self.convert_expression_ast(ast.func)
            args = [self.convert_expression_ast(a) for a in ast.args]

            init = native_ast.nullExpr

            if not ast.starargs.matches.Null:
                starargs = self.convert_expression_ast(ast.starargs.val)

                #starargs is now a reference to a tuple
                #now we want to take each element and turn into a reference.

                for attr, t in starargs.expr_type.nonref_type.element_types:
                    args.append(
                        starargs.convert_attribute(self, attr)
                        )

            return init + l.convert_call(self, args)

        if ast.matches.Compare:
            assert len(ast.comparators) == 1, "multi-comparison not implemented yet"
            assert len(ast.ops) == 1

            l = self.convert_expression_ast(ast.left)
            r = self.convert_expression_ast(ast.comparators[0])

            return l.convert_bin_op(ast.ops[0], r)

        if ast.matches.Tuple:
            elts = [self.convert_expression_ast(e) for e in ast.elts]

            struct_type = Struct([("f%s"%i,e.expr_type.as_call_arg) for i,e in enumerate(elts)])

            tmp_ref = self.allocate_temporary(struct_type)

            return TypedExpression(
                struct_type.convert_initialize(self, tmp_ref, elts).expr + 
                    self.activates_temporary(tmp_ref) + 
                    tmp_ref.expr,
                tmp_ref.expr_type
                )

        if ast.matches.IfExp:
            test = self.convert_expression_ast(ast.test)
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

            return TypedExpression(
                native_ast.Expression.Branch(
                    cond=test.expr, 
                    true=body.expr, 
                    false=orelse.expr
                    ),
                body.expr_type
                )

        raise ConversionException("can't handle python expression type %s" % ast._which)

    def convert_statement_ast_and_teardown_tmps(self, ast):
        if self._new_temporaries:
            raise ConversionException("Expected no temporaries on %s" % ast._which)
        
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
                            type=self._temporaries[tname].lower()
                            ),
                        self._temporaries[tname].reference
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
            expr.expr_type
            )

    def convert_statement_ast(self, ast):
        try:
            return self.convert_statement_ast_(ast)
        except ConversionException as e:
            e.add_scope(
                exceptions.ConversionScopeInfo.CreateFromAst(ast, self._varname_to_type)
                )

            raise

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

                    new_variable_type = val_to_store.expr_type.as_call_arg
                    assert new_variable_type.is_valid_as_variable()
                    self._varname_to_type[varname] = new_variable_type

                    slot_ref = self.named_var_expr(varname)
                    
                    #this is a new variable which we are constructing
                    return self._varname_to_type[varname].convert_initialize_copy(
                        self,
                        slot_ref,
                        val_to_store
                        ) + TypedExpression.Void(native_ast.Expression.ActivatesTeardown(varname))
                else:
                    #this is an existing variable.
                    slot_ref = self.named_var_expr(varname).drop_double_references()

                    if op is not None:
                        val_to_store = slot_ref.convert_bin_op(op, val_to_store)

                    return slot_ref.convert_assign(self, val_to_store)

            if target.matches.Subscript and target.ctx.matches.Store:
                assert target.slice.matches.Index

                slicing = self.convert_expression_ast(target.value)
                index = self.convert_expression_ast(target.slice.value)
                val_to_store = self.convert_expression_ast(ast.value)

                if op is not None:
                    val_to_store = slicing.convert_getitem(self, index).convert_bin_op(op, val_to_store)

                return slicing.convert_setitem(self, index, val_to_store)
        
            if target.matches.Attribute and target.ctx.matches.Store:
                slicing = self.convert_expression_ast(target.value)
                attr = target.attr
                val_to_store = self.convert_expression_ast(ast.value)

                if op is not None:
                    val_to_store = slicing.convert_attribute(self, attr).convert_bin_op(op, val_to_store)

                return slicing.convert_set_attribute(self, attr, val_to_store)
        
        if ast.matches.Return:
            if ast.value.matches.Null:
                e = TypedExpression(native_ast.nullExpr, Void)
            else:
                e = self.convert_expression_ast(ast.value.val)

            if self._varname_to_type[FunctionOutput] is not None:
                if self._varname_to_type[FunctionOutput] != e.expr_type.as_call_arg:
                    raise ConversionException(
                        "Function returning multiple types (%s and %s)" % (
                                e.expr_type, 
                                self._varname_to_type[FunctionOutput]
                                )
                        )
            else:
                self._varname_to_type[FunctionOutput] = e.expr_type.as_call_arg

            output_type = self._varname_to_type[FunctionOutput]

            if output_type.is_pod:
                if e.expr_type == Void:
                    return e + TypedExpression(native_ast.Expression.Return(arg=None), None)

                if e.expr_type.is_ref:
                    if output_type.is_ref:
                        return TypedExpression(native_ast.Expression.Return(e.expr), None)
                    else:
                        return TypedExpression(native_ast.Expression.Return(e.expr.load()), None)
                else:
                    assert not output_type.is_ref
                    return TypedExpression(native_ast.Expression.Return(e.expr), None)
            else:
                assert not output_type.is_ref

                return TypedExpression(
                    output_type
                        .convert_initialize_copy(
                            self,
                            TypedExpression(
                                native_ast.Expression.Variable(".return"),
                                output_type.reference
                                ),
                            e
                            ).expr + 
                        native_ast.Expression.Return(None),
                    None
                    )

        if ast.matches.Expr:
            return TypedExpression(
                self.convert_expression_ast(ast.value).expr + native_ast.nullExpr, 
                Void
                )

        if ast.matches.If:
            cond = self.convert_expression_ast(ast.test)
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
                ret_type = Void

            return TypedExpression(
                native_ast.Expression.Branch(cond=cond.expr,true=true.expr,false=false.expr),
                ret_type
                )

        if ast.matches.Pass:
            return TypedExpression(native_ast.nullExpr, Void)

        if ast.matches.While:
            cond = self.convert_expression_ast(ast.test)
            cond = self.consume_temporaries(cond)

            true = self.convert_statement_list_ast(ast.body)
            false = self.convert_statement_list_ast(ast.orelse)

            if true.expr_type or false.expr_type:
                ret_type = Void
            else:
                ret_type = None

            return TypedExpression(
                native_ast.Expression.While(cond=cond.expr,while_true=true.expr,orelse=false.expr),
                ret_type
                )

        if ast.matches.For:
            if not ast.target.matches.Name:
                raise ConversionException("For loops can only have simple targets for now")

            #this object needs to stay alive for the duration
            #of the expression
            iter_create_expr = (
                self.convert_expression_ast(ast.iter)
                    .convert_attribute(self, "__iter__")
                    .convert_call(self, [])
                )
            iter_type = iter_create_expr.expr_type.unwrap_reference()

            iter_expr = self.allocate_temporary(iter_type)

            iter_setup_expr = (
                iter_type.convert_initialize_copy(self, iter_expr, iter_create_expr).expr + 
                self.activates_temporary(iter_expr)
                )
                
            teardowns_for_iter = self.consume_temporaries(None)

            #now we need to generate a while loop
            while_cond_expr = (
                iter_expr.convert_attribute(self, "has_next")
                    .convert_call(self, [])
                    .convert_to_type(Bool)
                )

            while_cond_expr = self.consume_temporaries(while_cond_expr)

            next_val_expr = iter_expr.convert_attribute(self, "next").convert_call(self, [])
            
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
                next_val_expr.expr_type.convert_initialize_copy(self, next_val_ref, next_val_expr).expr +
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

            res = TypedExpression(
                native_ast.Expression.Finally(
                    expr=iter_setup_expr + 
                        native_ast.Expression.While(
                            cond=while_cond_expr.expr,
                            while_true=body_native_expr,
                            orelse=orelse_native_expr
                            ),
                    teardowns=teardowns_for_iter
                    ),
                Void
                )

            return res


        raise ConversionException("Can't handle python ast Statement.%s" % ast._which)

    def convert_statement_list_ast(self, statements, toplevel=False):
        orig_vars_in_scope = set(self._varname_to_type)

        if not statements:
            return TypedExpression(native_ast.nullExpr, Void)

        exprs = []
        for s in statements:
            conversion = self.convert_statement_ast_and_teardown_tmps(s)
            exprs.append(conversion)
            if conversion.expr_type is None:
                break

        if exprs[-1].expr_type is not None:
            exprs = exprs + [TypedExpression(native_ast.nullExpr, Void)]
            ret_type = Void
        else:
            ret_type = None

        if toplevel and ret_type is not None:
            exprs = exprs + [TypedExpression(native_ast.Expression.Return(None), None)]
            ret_type = None

        if toplevel:
            assert ret_type is None, "Not all control flow paths return a value"

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
        return TypedExpression(seq_expr, ret_type)

    def named_variable_teardown(self, v):
        return native_ast.Teardown.ByTag(
                tag=v,
                expr=self._varname_to_type[v].convert_destroy(
                    self,
                    TypedExpression(
                        native_ast.Expression.StackSlot(
                            name=v,
                            type=self._varname_to_type[v].lower()
                            ),
                        self._varname_to_type[v].reference
                        )
                    ).expr
                )
                    

    def call_expression_in_function(self, identity, name, args, expr_producer):
        varlist = []
        typelist = []
        exprlist = []

        for i in xrange(len(args)):
            varlist.append(native_ast.Expression.Variable(".var.%s" % i))

            if args[i].expr_type.is_pod:
                varexpr = varlist[i]
            else:
                varexpr = native_ast.Expression.Load(varlist[i])

            typelist.append(args[i].expr_type)
            exprlist.append(TypedExpression(varexpr, typelist[i]))

        expr = expr_producer(*exprlist)

        if expr.expr_type != Void:
            expr = native_ast.Expression.Return(expr)

        call_target = self._converter.define(
            (identity, tuple(typelist)),
            name,
            typelist,
            expr.expr_type,
            native_ast.Function(
                args=[(varlist[i].name, typelist[i].lower_as_function_arg()) 
                            for i in xrange(len(varlist))],
                body=native_ast.FunctionBody.Internal(expr.expr),
                output_type=expr.expr_type.lower()
                )
            )

        return TypedExpression(
            self.generate_call_expr(
                target=call_target.native_call_target,
                args=[a.expr for a in args]
                ),
            expr.expr_type
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
                                ptr=native_ast.Expression.StackSlot(name=name,type=slot_type.lower()),
                                val=native_ast.Expression.Variable(name)
                                )
                            )
                    else:
                        #need to make a stackslot for this variable
                        #the argument will be a pointer because it's POD
                        var_expr = TypedExpression(
                            native_ast.Expression.Variable(name),
                            slot_type.reference
                            )
                        
                        slot_expr = TypedExpression(
                            native_ast.Expression.StackSlot(name=name,type=slot_type.lower()),
                            slot_type.reference
                            )

                        to_add.append(
                            slot_type.convert_initialize_copy(self, slot_expr, var_expr)
                                .expr.with_comment("initialize %s from arg" % name)
                            )

                        destructors.append(
                            native_ast.Teardown.Always(
                                slot_type.convert_destroy(self, slot_expr).expr
                                )
                            )

        if to_add:
            expr = TypedExpression(
                native_ast.Expression.Sequence(
                    to_add + [expr.expr]
                    ),
                expr.expr_type
                )

        if destructors:
            expr = TypedExpression(
                native_ast.Expression.Finally(
                    teardowns=destructors,
                    expr=expr.expr
                    ),
                expr.expr_type
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
                    for i in xrange(len(args_type.element_types))]
            ).with_comment("initialize *args slot") + res


class TypedCallTarget(object):
    def __init__(self, native_call_target, input_types, output_type):
        object.__init__(self)
        self.native_call_target = native_call_target
        self.input_types = input_types
        self.output_type = output_type

    @property
    def name(self):
        return self.native_call_target.name

    def __str__(self):
        return "TypedCallTarget(name=%s,inputs=%s,outputs=%s)" % (
            self.name, 
            [str(x) for x in self.input_types],
            str(self.output_type)
            )

class Converter(object):
    def __init__(self):
        object.__init__(self)
        self._names_for_identifier = {}
        self._definitions = {}
        self._targets = {}

        self._unconverted = set()

        self.verbose = False

    def extract_new_function_definitions(self):
        res = {}

        for u in self._unconverted:
            res[u] = self._definitions[u]

            if self.verbose:
                print self._targets[u]
        
        self._unconverted = set()

        return res

    def new_name(self, name):
        suffix = None
        getname = lambda: "py." + name + ("" if suffix is None else ".%s" % suffix)
        while getname() in self._targets:
            suffix = 1 if not suffix else suffix+1
        return getname()

    def convert_function_ast(self, ast, input_types, local_variables, free_variable_lookup):
        if ast.args.vararg.matches.Value:
            star_args_name = ast.args.vararg.val
        else:
            star_args_name = None

        if star_args_name is None:
            if len(input_types) != len(ast.args.args):
                raise ConversionException(
                    "Exected %s arguments but got %s" % (len(ast.args.args), len(input_types))
                    )
        else:
            if len(input_types) < len(ast.args.args):
                raise ConversionException(
                    "Exected at least %s arguments but got %s" % 
                        (len(ast.args.args), len(input_types))
                    )

        varname_to_type = {}

        args = []
        for i in xrange(len(ast.args.args)):
            varname_to_type[ast.args.args[i].id] = input_types[i]
            args.append((ast.args.args[i].id, input_types[i].lower_as_function_arg()))

        argnames = [a[0] for a in args]

        if star_args_name is not None:
            star_args_count = len(input_types) - len(ast.args.args)

            starargs_elts = []
            for i in xrange(len(ast.args.args), len(input_types)):
                args.append(
                    ('.star_args.%s' % (i - len(ast.args.args)), 
                        input_types[i].lower_as_function_arg())
                    )

            def stararg_field_type_for(t):
                if t.is_ref:
                    return t
                if t.is_pod:
                    return t

            starargs_type = Struct(
                [('f_%s' % i, stararg_field_type_for(input_types[i+len(ast.args.args)]))
                    for i in xrange(star_args_count)]
                )

            varname_to_type[star_args_name] = starargs_type

        varname_to_type[FunctionOutput] = None

        subconverter = ConversionContext(self, varname_to_type, free_variable_lookup)

        res = subconverter.convert_statement_list_ast(ast.body, toplevel=True)

        if star_args_name is not None:
            res = subconverter.construct_starargs_around(res, star_args_name)

        res = subconverter.construct_stackslots_around(res, argnames, star_args_name)

        return_type = subconverter._varname_to_type[FunctionOutput] or Void

        return (
            native_ast.Function(
                args=args, 
                body=native_ast.FunctionBody.Internal(res.expr),
                output_type=return_type.lower()
                ),
            return_type
            )
                  

    def convert_lambda_ast(self, ast, input_types, free_variable_lookup):
        assert len(input_types) == len(ast.args.args)
        varname_to_type = {}

        args = []
        argnames = []
        for i in xrange(len(input_types)):
            varname_to_type[ast.args.args[i].id] = input_types[i]
            args.append((ast.args.args[i].id, input_types[i].lower_as_function_arg()))
            argnames.append(ast.args.args[i].id)

        subconverter = ConversionContext(self, varname_to_type, free_variable_lookup)

        expr = subconverter.convert_expression_ast(ast.body)

        expr = subconverter.construct_stackslots_around(expr, argnames, None)

        return (
            native_ast.Function(
                args=args, 
                body=native_ast.FunctionBody.Internal(
                    native_ast.Expression.Return(expr.expr)
                    ),
                output_type=expr.expr_type.lower()
                ),
            expr.expr_type
            )

    def define(self, identifier, name, input_types, output_type, native_function_definition):
        identifier = ("defined", identifier)

        if identifier in self._names_for_identifier:
            name = self._names_for_identifier[identifier]

            return self._targets[name]

        new_name = self.new_name(name)
        self._names_for_identifier[identifier] = new_name

        self._targets[new_name] = TypedCallTarget(
            native_ast.CallTarget(
                name=new_name, 
                arg_types=[x[1] for x in native_function_definition.args],
                output_type=native_function_definition.output_type,
                external=False,
                varargs=False
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

        subconverter = ConversionContext(self, varname_to_type, free_variable_lookup)

        return subconverter.convert_expression_ast(ast.body)

    def callable_to_ast_and_vars(self, f):
        pyast = ast_util.pyAstFor(f)

        _, lineno = ast_util.getSourceLines(f)
        _, fname = ast_util.getSourceFilenameAndText(f)

        pyast = ast_util.functionDefOrLambdaAtLineNumber(pyast, lineno)

        pyast = python_ast.convertPyAstToAlgebraic(pyast, fname)

        freevars = dict(f.func_globals)

        if f.func_closure:
            for i in xrange(len(f.func_closure)):
                freevars[f.func_code.co_freevars[i]] = f.func_closure[i].cell_contents

        return pyast, freevars

    def convert(self, f, input_types, name_override=None):
        for i in input_types:
            if not i.is_valid_as_variable():
                raise ConversionException("Invalid argument types for %s: %s" % (f, input_types))

        input_types = tuple(input_types)

        identifier = ("pyfunction", f, input_types)

        if identifier in self._names_for_identifier:
            name = self._names_for_identifier[identifier]
            return self._targets[name]

        pyast, freevars = self.callable_to_ast_and_vars(f)

        try:
            if isinstance(pyast, python_ast.Statement.FunctionDef):
                definition,output_type = \
                    self.convert_function_ast(pyast, input_types, f.func_code.co_varnames, freevars)
            else:
                assert pyast.matches.Lambda
                definition,output_type = self.convert_lambda_ast(pyast, input_types, freevars)

            assert definition is not None

            new_name = self.new_name(name_override or f.func_name)

            self._names_for_identifier[identifier] = new_name

            if not output_type.is_pod:
                definition = native_ast.Function(
                    args=(('.return', output_type.pointer.lower()),) + definition.args,
                    body=definition.body,
                    output_type=native_ast.Type.Void()
                    )

            self._targets[new_name] = TypedCallTarget(
                native_ast.CallTarget(
                    name=new_name, 
                    arg_types=[x[1] for x in definition.args],
                    output_type=definition.output_type,
                    external=False,
                    varargs=False
                    ),
                input_types,
                output_type
                )

            self._definitions[new_name] = definition
            self._unconverted.add(new_name)

            return self._targets[new_name]
        except ConversionException as e:
            e.add_scope(
                exceptions.ConversionScopeInfo.CreateFromAst(pyast, {})
                )
            raise

