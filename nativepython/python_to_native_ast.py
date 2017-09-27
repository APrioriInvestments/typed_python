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
import nativepython.util as util
import nativepython.native_ast as native_ast
import nativepython.exceptions as exceptions
from nativepython.type_model import *

class FunctionOutput:
    pass

class InitFields:
    def __init__(self, first_var_name, fields_and_types):
        self.first_var_name = first_var_name
        self.uninitialized_fields = set(x[0] for x in fields_and_types)
        self.field_types = dict(fields_and_types)
        self.field_order = [x[0] for x in fields_and_types]
        self.initialization_expressions = {}

    def mark_field_initialized(self, field, expr):
        self.initialization_expressions[field] = expr
        self.uninitialized_fields.discard(field)

    def finalize(self, context):
        for f in self.uninitialized_fields:
            self.initialization_expressions[f] = (
                context.named_var_expr(self.first_var_name)
                    .drop_double_references()
                    .convert_attribute(context, f, allow_double_refs=True)
                    .convert_initialize(context, ())
                )

        self.uninitialized_fields = set()

    def init_expr(self, context):
        self.finalize(context)

        if not self.field_order:
            return TypedExpression(native_ast.nullExpr, Void)

        expr = self.initialization_expressions[self.field_order[0]]

        for field in self.field_order[1:]:
            expr = expr + self.initialization_expressions[field]

        return expr

    def any_remaining(self):
        return bool(self.uninitialized_fields)

class ConversionContext(object):
    def __init__(self, converter, varname_to_type, free_variable_lookup, init_fields):
        self.converter = converter
        self._varname_to_type = varname_to_type
        self._varname_and_type_to_slot_name = {}
        self._varname_uses = {}
        self._new_variables = set()
        self._free_variable_lookup = free_variable_lookup
        self._temporaries = {}
        self._new_temporaries = set()
        self._temp_let_var = 0
        self._init_fields = init_fields

    def let_varname(self):
        self._temp_let_var += 1
        return ".letvar.%s" % (self._temp_let_var-1)

    def activates_temporary(self, slot):
        if slot.expr_type.value_type.is_pod:
            return native_ast.nullExpr

        return native_ast.Expression.ActivatesTeardown(slot.expr.name)

    def allocate_temporary(self, slot_type, type_is_temp_ref = True):
        tname = '.temp.%s' % len(self._temporaries)
        self._new_temporaries.add(tname)
        self._temporaries[tname] = slot_type

        return TypedExpression(
            native_ast.Expression.StackSlot(name=tname,type=slot_type.lower()),
            slot_type.reference_to_temporary if type_is_temp_ref else
                slot_type.reference
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
            native_ast.Expression.StackSlot(
                name=name_to_use,
                type=slot_type.lower()
                ),
            slot_type.reference
            )

    def call_typed_function(self, native_call_target, output_type, input_types, varargs, args):
        def make_arg_compatible(i, arg):
            if i >= len(input_types):
                if not varargs:
                    raise ConversionException("Calling with too many arguments and func is not varargs")
                #never pass references - always unpack
                return args[i].unwrap_reference(completely=True)

            to_type = input_types[i]

            if arg.expr_type.nonref_type != to_type.nonref_type:
                arg = arg.convert_to_type(to_type.nonref_type, implicitly=True)

            if not to_type.is_ref and arg.expr_type.is_ref:
                arg=arg.dereference()
        
            if arg.expr_type != to_type:
                raise ConversionException(
                    "Can't convert arg #%s from %s to %s" % (
                        i+1, 
                        args[i].expr_type, 
                        to_type
                        )
                    )

            return arg

        new_args = []
        for i in xrange(len(args)):
            new_args.append(make_arg_compatible(i, args[i]))

        if output_type.is_pod:
            return TypedExpression(
                native_ast.Expression.Call(
                    target=native_call_target,
                    args=[a.expr for a in new_args]
                    ),
                output_type
                )
        else:
            slot_ref = self.allocate_temporary(output_type)

            return TypedExpression(
                native_ast.Expression.Call(
                    target=native_ast.CallTarget.Pointer(instance.expr),
                    args=[slot_ref.expr] + [a.expr for a in new_args]
                    )
                    + self.activates_temporary(slot_ref)
                    + slot_ref.expr,
                slot_ref.expr_type
                )

    def call_py_function(self, f, args, name_override=None):
        #force arguments to a type appropriate for argpassing
        #e.g. drop out "CreateReference" and other syntactic sugar
        args = [a.as_call_arg(self) for a in args]
        native_args = [a.expr for a in args]

        call_target = \
            self.converter.convert(
                f, 
                [a.expr_type for a in args], 
                name_override=name_override
                )

        if not call_target.output_type.is_pod:
            assert len(call_target.named_call_target.arg_types) == len(args) + 1

            slot = self.allocate_temporary(call_target.output_type)

            return TypedExpression(
                self.generate_call_expr(
                    target=call_target.named_call_target,
                    args=[slot.expr] + native_args
                    ) 
                    + self.activates_temporary(slot)
                    + slot.expr
                    ,
                slot.expr_type
                )
        else:
            assert len(call_target.named_call_target.arg_types) == len(args)

            return TypedExpression(
                self.generate_call_expr(
                    target=call_target.named_call_target,
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

        e = native_ast.Expression.Call(
            target=native_ast.CallTarget.Named(target),
            args=actual_args
            )

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

        if ast.matches.BoolOp:
            op = ast.op
            values = ast.values

            expr_so_far = []

            for i in xrange(len(values)):
                expr_so_far.append(self.ensure_bool(self.convert_expression_ast(values[i])).expr)
                if expr_so_far[-1].matches.Constant:
                    if (expr_so_far[-1].val.val and op.matches.Or or 
                            (not expr_so_far[-1].val.val) and op.matches.And):
                        #this is a short-circuit
                        if len(expr_so_far) == 1:
                            return TypedExpression(expr_so_far[0], Bool)

                        return TypedExpression(
                            native_ast.Expression.Sequence(expr_so_far),
                            Bool
                            )
                    else:
                        expr_so_far.pop()

            if not expr_so_far:
                if op.matches.Or:
                    #must have had all False constants
                    return TypedExpression(native_ast.falseExpr, Bool)
                else:
                    #must have had all True constants
                    return TypedExpression(native_ast.trueExpr, Bool)

            while len(expr_so_far) > 1:
                l,r = expr_so_far[-2], expr_so_far[-1]
                expr_so_far.pop()
                expr_so_far.pop()
                if op.matches.And:
                    new_expr = native_ast.Expression.Branch(cond=l, true=r, false=native_ast.falseExpr)
                else:
                    new_expr = native_ast.Expression.Branch(cond=l, true=native_ast.trueExpr, false=r)
                expr_so_far.append(new_expr)

            return TypedExpression(expr_so_far[0], Bool)

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

            struct_type = Struct([("f%s"%i,e.expr_type.variable_storage_type) for i,e in enumerate(elts)])

            tmp_ref = self.allocate_temporary(struct_type)

            return TypedExpression(
                struct_type.convert_initialize(self, tmp_ref, elts).expr + 
                    self.activates_temporary(tmp_ref) + 
                    tmp_ref.expr,
                tmp_ref.expr_type
                )

        if ast.matches.IfExp:
            test = self.ensure_bool(self.convert_expression_ast(ast.test))
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

    def ensure_bool(self, expr):
        t = expr.expr_type.nonref_type

        if t == Bool:
            return expr.dereference()

        if not (t.is_primitive_numeric or t.is_pointer):
            if expr.expr.matches.Constant:
                return TypedExpression(native_ast.falseExpr, Bool)
            else:
                return expr + TypedExpression(native_ast.falseExpr, Bool)

        return expr.dereference()

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

    def check_statement_is_field_initializer(self, ast):
        if not ast.matches.Expr:
            return None

        ast = ast.value

        if not (ast.matches.Call and
                ast.func.matches.Attribute and
                ast.func.attr == "__init__" and
                ast.func.value.matches.Attribute and
                ast.func.value.value.matches.Name and
                ast.func.value.value.id == self._init_fields.first_var_name
                ):
            return None

        return (ast.func.value.attr, ast.args)

    def convert_statement_ast_(self, ast):
        field_and_args = self.check_statement_is_field_initializer(ast)
        if field_and_args is not None:
            field, args = field_and_args

            if self._init_fields is None:
                raise ConversionException("Invalid use of __init__ outside of constructor")

            if not self._init_fields.any_remaining():
                raise ConversionException("All members are initialized.")

            if field not in self._init_fields.field_types:
                raise ConversionException(
                    "Member %s is not a valid member to initialize. Did you mean %s?" % (
                        field,
                        string_util.closest(field, self._init_fields.field_order)
                        )
                    )

            if field not in self._init_fields.uninitialized_fields:
                raise ConversionException("Member %s is already initialized" % field)

            field_type = self._init_fields.field_types[field]

            args = [self.convert_expression_ast(a) for a in args]

            expr = (
                self.named_var_expr(self._init_fields.first_var_name)
                    .drop_double_references()
                    .convert_attribute(self, field, allow_double_refs=True)
                    .convert_initialize(self, args)
                )

            expr = self.consume_temporaries(expr)

            self._init_fields.mark_field_initialized(field, expr)

            return TypedExpression.Void(native_ast.nullExpr)

        if ast.matches.Assign or ast.matches.AugAssign:
            if self._init_fields is not None:
                self._init_fields.finalize(self)

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

                    new_variable_type = val_to_store.expr_type.variable_storage_type
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
            if self._init_fields is not None:
                self._init_fields.finalize(self)
            
            if ast.value.matches.Null:
                e = TypedExpression(native_ast.nullExpr, Void)
            else:
                e = self.convert_expression_ast(ast.value.val)

            if self._varname_to_type[FunctionOutput] is not None:
                if self._varname_to_type[FunctionOutput] != e.expr_type.variable_storage_type:
                    raise ConversionException(
                        "Function returning multiple types (%s and %s)" % (
                                e.expr_type, 
                                self._varname_to_type[FunctionOutput]
                                )
                        )
            else:
                self._varname_to_type[FunctionOutput] = e.expr_type.variable_storage_type

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
            cond = self.ensure_bool(self.convert_expression_ast(ast.test))
            cond = self.consume_temporaries(cond)

            if cond.expr.matches.Constant:
                truth_val = cond.expr.val.truth_value()
                branch = self.convert_statement_list_ast(ast.body if truth_val else ast.orelse)
                
                return cond + branch

            if self._init_fields is not None:
                self._init_fields.finalize(self)

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
            if self._init_fields is not None:
                self._init_fields.finalize(self)

            cond = self.ensure_bool(self.convert_expression_ast(ast.test))
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

        if ast.matches.TryExcept:
            if len(ast.orelse):
                raise ConversionException("We dont handle try-except-else")

            body = self.convert_statement_list_ast(ast.body)
            handlers = []

            #if 'orelse' is not None, then 
            any_handlers_exit = False

            for h in ast.handlers:
                if h.type.matches.Value:
                    typexpr = self.convert_expression_ast(h.type.val)

                    if not (    typexpr.expr_type.is_compile_time
                            and typexpr.expr_type.python_object_representation is Exception
                            ):
                        raise ConversionException("can't handle types in exceptions yet")

                name = None
                if not h.name.matches.Null:
                    name = h.name.val.id

                if name is not None:
                    if self._varname_to_type.get(name, None) is not None:
                        raise ConversionException("Variable %s is already defined" % name)
                    
                    self._varname_to_type[name] = Int8.pointer

                handler_expr = self.convert_statement_list_ast(h.body)

                if name is not None:
                    handler_expr = TypedExpression.Void(
                        native_ast.Expression.Store(
                            val=native_ast.Expression.Variable(".unnamed.exception.var"),
                            ptr=self.named_var_expr(name).expr
                            )
                        ) + handler_expr

                    del self._varname_to_type[name]

                if handler_expr.expr_type is not None:
                    any_handlers_exit = True

            return TypedExpression(
                native_ast.Expression.TryCatch(
                    expr=body.expr,
                    varname=".unnamed.exception.var",
                    handler=handler_expr.expr
                    ),
                Void if any_handlers_exit or body.expr_type is not None else None
                )

        if ast.matches.For:
            if self._init_fields is not None:
                self._init_fields.finalize(self)

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
                    .convert_to_type(Bool, False)
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

        if ast.matches.Raise:
            if ast.type.matches.Value and ast.inst.matches.Null and ast.tback.matches.Null:
                expr = self.convert_expression_ast(ast.type.val)

                return self.call_py_function(
                    util.throw, 
                    [expr]
                    )
            else:
                raise ConversionException("We can only handle simple 'raise' statements")

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
            assert exprs[-1].expr_type == Void
            flows_off_end = True
        else:
            flows_off_end = False

        if toplevel and flows_off_end:
            if self._varname_to_type[FunctionOutput] in (Void, None):
                exprs = exprs + [TypedExpression(native_ast.Expression.Return(None), None)]
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

        return TypedExpression(seq_expr, Void if flows_off_end else None)

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
        args = [a.as_call_arg(self) for a in args]

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

        call_target = self.converter.define(
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
                target=call_target.named_call_target,
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
    def __init__(self, named_call_target, input_types, output_type):
        object.__init__(self)
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

    def convert_function_ast(
                self, 
                ast_arg, 
                statements, 
                input_types, 
                local_variables, 
                free_variable_lookup,
                members_of_arg0_to_initialize
                ):
        if ast_arg.vararg.matches.Value:
            star_args_name = ast_arg.vararg.val
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
        for i in xrange(len(ast_arg.args)):
            varname_to_type[ast_arg.args[i].id] = input_types[i]
            args.append((ast_arg.args[i].id, input_types[i].lower_as_function_arg()))

        argnames = [a[0] for a in args]

        if star_args_name is not None:
            star_args_count = len(input_types) - len(ast_arg.args)

            starargs_elts = []
            for i in xrange(len(ast_arg.args), len(input_types)):
                args.append(
                    ('.star_args.%s' % (i - len(ast_arg.args)), 
                        input_types[i].lower_as_function_arg())
                    )

            starargs_type = Struct(
                [('f_%s' % i, input_types[i+len(ast_arg.args)])
                    for i in xrange(star_args_count)]
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

        return_type = subconverter._varname_to_type[FunctionOutput] or Void

        return (
            native_ast.Function(
                args=args, 
                body=native_ast.FunctionBody.Internal(res.expr),
                output_type=return_type.lower()
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

        freevars = dict(f.func_globals)

        if f.func_closure:
            for i in xrange(len(f.func_closure)):
                freevars[f.func_code.co_freevars[i]] = f.func_closure[i].cell_contents

        return pyast, freevars
    
    def convert_initializer_function(self, f, input_types, name_override, fields_and_types):
        return self.convert(f, input_types, name_override, fields_and_types)

    def convert(self, f, input_types, name_override=None, fields_and_types_for_initializing=None):
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
                    self.convert_function_ast(
                        pyast.args, 
                        pyast.body, 
                        input_types, 
                        f.func_code.co_varnames, 
                        freevars,
                        fields_and_types_for_initializing
                        )
            else:
                assert pyast.matches.Lambda
                if fields_and_types_for_initializing:
                    raise ConversionException("initializers can't be lambdas")
                definition,output_type = self.convert_lambda_ast(pyast, input_types, f.func_code.co_varnames, freevars)

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
        except ConversionException as e:
            e.add_scope(
                exceptions.ConversionScopeInfo.CreateFromAst(pyast, {})
                )
            raise

