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
import llvm_compiler as llvm_compiler

class FunctionOutput:
    pass

function_type = type(lambda d:d)
classobj = type(FunctionOutput)

class ConversionScopeInfo(object):
    def __init__(self, filename, line, col, types):
        object.__init__(self)

        self.filename = filename
        self.line = line
        self.col = col
        self.types = {k:v for k,v in types.iteritems() if isinstance(k,str) and v is not None}

    def __cmp__(self, other):
        return cmp(
            (self.filename,self.line,self.col,tuple(sorted(self.types.iteritems()))),
            (other.filename,other.line,other.col,tuple(sorted(other.types.iteritems())))
            )

    @staticmethod
    def CreateFromAst(ast, types):
        return ConversionScopeInfo(
            ast.filename,
            ast.line_number,
            ast.col_offset,
            dict(types)
            )

    def __str__(self):
        return "   File \"%s\", line %d\n%s" % (self.filename, self.line, 
            "".join(["\t\t%s=%s\n" % (k,v) for k,v in self.types.iteritems()])
            )

class ConversionException(Exception):
    def __init__(self, msg):
        Exception.__init__(self, msg)
        self.conversion_scope_infos=[]

    def add_scope(self, new_scope):
        if not self.conversion_scope_infos or self.conversion_scope_infos[0] != new_scope:
            self.conversion_scope_infos = [new_scope] + self.conversion_scope_infos

    def __str__(self):
        return self.message + "\n\n" + "".join(str(x) for x in self.conversion_scope_infos)

class UnassignableFieldException(ConversionException):
    def __init__(self, obj_type, attr, target_type):
        ConversionException.__init__(self, "missing attribute %s in type %s" % (attr,type))
        self.obj_type = obj_type
        self.attr = attr
        self.target_type = target_type

class Type(object):
    def __init__(self):
        object.__init__(self)

    def unwrap_reference(self):
        return self

    @property
    def is_pod(self):
        assert False,"can't directly references instances of %s" % self

    @property
    def null_value(self):
        assert False,"can't construct a null value of type %s" % self

    def lower_as_function_arg(self):
        if self.is_pod:
            return self.lower()
        return native_ast.Type.Pointer(self.lower())

    def lower(self):
        assert False, "Can't directly reference instances of %s" % self

    def convert_initialize_copy(self, context, instance_ptr, other_instance):
        assert self.is_pod, "can't initialize %s - need a real implementation" % self

        if other_instance.expr_type != self:
            other_instance = other_instance.expr_type.convert_to_type(other_instance, self)

        return TypedExpression(
            native_ast.Expression.Store(
                ptr=instance_ptr.expr,
                val=other_instance.expr
                ),
            Void
            )

    def convert_destroy(self, context, instance_ptr):
        assert self.is_pod, "can't destroy %s - need a real implementation" % self

        return TypedExpression(
            native_ast.nullExpr,
            Void
            )

    def convert_initialize(self, context, instance_ptr, args):
        assert instance_ptr.expr_type.value_type == self

        assert len(args) <= 1
        if len(args) == 1:
            return self.convert_initialize_copy(context, instance_ptr, args[0])
        else:
            assert self.is_pod, "can't initialize %s - need a real implementation" % self
            return TypedExpression(
                native_ast.Expression.Store(
                    ptr=instance_ptr.expr,
                    val=native_ast.Expression.Constant(self.null_value)
                    ),
                Void
                )

    def convert_assign(self, context, instance_ptr, arg):
        assert self.is_pod
        assert arg.expr_type == self, "can't assign %s to %s" % (arg.expr_type, self)

        return TypedExpression(
            native_ast.Expression.Store(
                ptr=instance_ptr.expr,
                val=arg.expr
                ),
            Void
            )

    def convert_unary_op(self, instance, op):
        assert False, "can't handle unary op %s on %s" % (op, self)

    def convert_bin_op(self, op, l, r):
        assert False, "can't handle binary op %s between %s and %s" % (op, l.expr_type, r.expr_type)

    def convert_to_type(self, instance, to_type):
        assert False, "can't convert %s to type %s" % (self, to_type)

    def convert_attribute(self, instance, attr):
        assert False, "%s has no attribute %s" % (self, attr)

    def convert_set_attribute(self, instance, attr, value):
        assert False, "%s has no attribute %s" % (self, attr)

    def convert_getitem(self, instance, index):
        assert False, "%s doesn't support getting items" % self

    def convert_setitem(self, instance, index, value):
        assert False, "%s doesn't support setting items" % self

    @property
    def sizeof(self):
        assert False, "can't compute the size of %s because we can't instantiate it" % self

    @property
    def pointer(self):
        return Pointer(self)

    def __cmp__(self, other):
        if not isinstance(other, type(self)):
            return cmp(type(self), type(other))
        for k in sorted(self.__dict__):
            c = cmp(getattr(self,k), getattr(other,k))
            if c:
                return c
        return 0

    def __hash__(self):
        return hash(tuple(sorted(self.__dict__.iteritems())))


class Struct(Type):
    def __init__(self, element_types=()):
        self.element_types = tuple(element_types)

    @property
    def is_pod(self):
        for _,e in self.element_types:
            if not e.is_pod:
                return False
        return True

    @property
    def null_value(self):
        return native_ast.Constant.Struct(
                [(name,t.null_value) for name,t in self.element_types]
                )

    def convert_initialize_copy(self, context, instance_ptr, other_instance):
        if self.is_pod:
            assert other_instance.expr_type == self

            return TypedExpression(
                native_ast.Expression.Store(
                    ptr=instance_ptr.expr,
                    val=other_instance.expr
                    ),
                Void
                )

        assert other_instance.expr_type == self
        assert instance_ptr.expr_type.value_type == self

        def make_body(instance_ptr, other_instance):
            body = native_ast.nullExpr

            for ix,(name,e) in enumerate(self.element_types):
                dest_field = self.pointer_to_field(instance_ptr.expr, name)
                source_field = other_instance.convert_attribute(name)

                dest_t = dest_field.expr_type.value_type

                body = body + dest_t.convert_initialize_copy(context, dest_field, source_field).expr

            return TypedExpression(body, Void)

        return context.call_expression_in_function(
            (self, "initialize_copy"), 
            "%s.initialize_copy" % (str(self)), 
            [instance_ptr, other_instance.reference], 
            make_body
            )

    def convert_destroy(self, context, instance_ptr):
        if self.is_pod:
            return TypedExpression(
                native_ast.nullExpr,
                Void
                )

        expr = native_ast.nullExpr

        for ix,(name,e) in enumerate(self.element_types):
            dest_field = self.pointer_to_field(instance_ptr.expr, name)

            expr = expr + dest_field.expr_type.value_type.convert_destroy(context, dest_field).expr

        return TypedExpression(expr, Void)

    def convert_initialize(self, context, instance_ptr, args):
        if len(args) != 0:
            assert len(args) == len(self.element_types), (self.element_types, args)

        def make_body(instance_ptr, *args):
            expr = native_ast.nullExpr

            for ix,(name,e) in enumerate(self.element_types):
                dest_field = self.pointer_to_field(instance_ptr.expr, name)

                dest_t = dest_field.expr_type.value_type
                if args:
                    expr = expr + dest_t.convert_initialize_copy(context, dest_field, args[ix]).expr
                else:
                    expr = expr + dest_t.convert_initialize(context, dest_field, ()).expr

            return TypedExpression(expr, Void)

        return context.call_expression_in_function(
            (self, "initialize"), 
            "%s.initialize" % (str(self)), 
            [instance_ptr] + list(args), 
            make_body
            )

    def convert_assign(self, context, instance_ptr, arg):
        assert arg.expr_type == self, "can't assign %s to %s" % (arg.expr_type, self)

        if arg.expr_type.is_pod:
            return TypedExpression(
                native_ast.Expression.Store(
                    ptr=instance_ptr.expr,
                    val=arg.expr
                    ),
                Void
                )

        def make_body(instance_ptr, arg):
            expr = native_ast.nullExpr

            for ix,(name,e) in enumerate(self.element_types):
                dest_field = self.pointer_to_field(instance_ptr.expr, name)
                source_field = arg.convert_attribute(name)

                dest_t = dest_field.expr_type.value_type

                expr = expr + dest_t.convert_assign(context, dest_field, source_field).expr

            return TypedExpression(expr, Void)

        return context.call_expression_in_function(
            (self, "assign"), 
            "%s.assign" % (str(self)), 
            [instance_ptr, arg.reference], 
            make_body
            )

    def lower(self):
        return native_ast.Type.Struct(tuple([(a[0], a[1].lower()) for a in self.element_types]))

    def with_field(self, name, type):
        if isinstance(name, TypedExpression):
            assert name.expr.matches.Constant and name.expr.val.matches.ByteArray, name.expr
            name = name.expr.val.val

        return Struct(element_types=self.element_types + ((name,type),))

    def convert_attribute(self, instance, attr):
        for i in xrange(len(self.element_types)):
            if self.element_types[i][0] == attr:
                if self.is_pod:
                    return TypedExpression(
                        native_ast.Expression.Attribute(left=instance.expr,attr=attr), 
                        self.element_types[i][1]
                        )
                else:
                    address = instance.address
                    return TypedExpression(
                        address.expr.ElementPtrIntegers(0,i).load(),
                        self.element_types[i][1]
                        )

        return super(Struct,self).convert_attribute(instance, attr)

    def convert_getitem(self, instance, index):
        assert index.expr.matches.Constant and index.expr.val.matches.Int, \
            "can't index %s with %s" % (self,index)
        i = index.expr.val.val
        assert i >= 0 and i < len(self.element_types), "can't index %s with %s" % (self, index)

        return self.convert_attribute(instance, self.element_types[i][0])

    def pointer_to_field(self, native_instance_ptr, attribute_name):
        for i in xrange(len(self.element_types)):
            if self.element_types[i][0] == attribute_name:
                return TypedExpression(
                    native_instance_ptr.ElementPtrIntegers(0, i),
                    self.element_types[i][1].pointer
                    )

    def field(self, native_instance, attribute_name):
        for i in xrange(len(self.element_types)):
            if self.element_types[i][0] == attribute_name:
                return TypedExpression(
                    native_ast.Expression.Attribute(
                        left=native_instance, 
                        attr=attribute_name
                        ),
                    self.element_types[i][1]
                    )


    def convert_set_attribute(self, instance, attr, val):
        field_ptr = self.pointer_to_field(instance.address.expr, attr)

        assert field_ptr is not None

        assert val.expr_type == field_ptr.expr_type.value_type, \
            "Can't assign value of type %s to struct field %s" % (
                val.expr_type,
                field_ptr.expr_type.value_type
                )

        return TypedExpression(
            native_ast.Expression.Store(
                ptr=field_ptr.expr,
                val=val.expr
                ),
            Void
            )

    @property
    def sizeof(self):
        return sum(t.sizeof for n,t in self.element_types)

    def __str__(self):
        return "Struct(%s)" % (",".join(["%s=%s" % t for t in self.element_types]))

class Reference(Type):
    def __init__(self, value_type):
        assert isinstance(value_type, Type)
        self.value_type = value_type

    def unwrap_reference(self):
        return self.value_type

    def lower(self):
        return native_ast.Type.Pointer(self.value_type.lower())

    @property
    def is_pod(self):
        return True

    @property
    def null_value(self):
        return native_ast.Constant.NullPointer(self.value_type.lower())

    @property
    def sizeof(self):
        return llvm_compiler.pointer_size

    @property
    def pointer(self):
        return Pointer(self.value_type)

    def convert_attribute(self, instance, attr):
        result = TypedExpression(instance.expr.load(), self.value_type).convert_attribute(attr)

        return result

    def convert_set_attribute(self, instance, attr, val):
        return TypedExpression(instance.expr.load(), self.value_type) \
            .convert_set_attribute(attr, val)

    def convert_bin_op(self, op, l, r):
        return TypedExpression(l.expr.load(), self.value_type).convert_bin_op(op, r)

    def convert_getitem(self, instance, index):
        return TypedExpression(instance.expr.load(), self.value_type).convert_getitem(index)

    def convert_setitem(self, instance, index, value):
        return TypedExpression(instance.expr.load(), self.value_type).convert_setitem(index, value)

    def convert_to_type(self, instance, to_type):
        return TypedExpression(instance.expr.load(), self.value_type).convert_to_type(to_type)

    def __repr__(self):
        return "Reference(%s)" % self.value_type

class Pointer(Type):
    def __init__(self, value_type):
        assert isinstance(value_type, Type)
        self.value_type = value_type

    def lower(self):
        return native_ast.Type.Pointer(self.value_type.lower())

    @property
    def is_pod(self):
        return True

    @property
    def null_value(self):
        return native_ast.Constant.NullPointer(self.value_type.lower())

    @property
    def sizeof(self):
        return llvm_compiler.pointer_size

    def convert_attribute(self, instance, attr):
        if not isinstance(self.value_type, Pointer):
            return instance.load.convert_attribute(attr)

        assert False, "no attribute %s in Pointer" % attr

    def convert_set_attribute(self, instance, attr, val):
        if not isinstance(self.value_type, Pointer):
            return instance.load.convert_set_attribute(attr,val)
        
        assert False, "no attribute %s in Pointer" % attr

    def convert_bin_op(self, op, l, r):
        if op._alternative is python_ast.BinaryOp:
            if op.matches.Add or op.matches.Sub:
                if isinstance(r.expr_type, PrimitiveNumericType) and r.expr_type.t.matches.Int:
                    if op.matches.Sub:
                        r = r.convert_unary_op(python_ast.PythonASTUnaryOp.USub())

                    return TypedExpression(
                        native_ast.Expression.ElementPtr(
                            left=l.expr,
                            offsets=(r.expr,)
                            ),
                        self
                        )

        return super(Pointer, self).convert_bin_op(op,l,r)


    def convert_getitem(self, instance, index):
        assert (isinstance(index.expr_type, PrimitiveNumericType) 
                and index.expr_type.t.matches.Int), \
            "can only index with integers, not %s" % index.expr_type

        return TypedExpression(
            native_ast.Expression.Load(
                ptr=native_ast.Expression.ElementPtr(
                    left=instance.expr,
                    offsets=(index.expr,)
                    )
                ),
            self.value_type
            )

    def convert_setitem(self, instance, index, value):
        assert value.expr_type == self.value_type, (
            "%s can only hold instances of %s, not %s" % 
                (self, self.value_type, value.expr_type)
            )

        assert (isinstance(index.expr_type, PrimitiveNumericType) 
                and index.expr_type.t.matches.Int), \
            "can only index with integers, not %s" % index.expr_type

        return TypedExpression(
            native_ast.Expression.Store(
                ptr=native_ast.Expression.ElementPtr(
                    left=instance.expr,
                    offsets=(index.expr,)
                    ),
                val=value.expr
                ),
            Void
            )

    def convert_to_type(self, instance, to_type):
        if isinstance(to_type, Pointer):
            return TypedExpression(
                native_ast.Expression.Cast(left=instance.expr, to_type=to_type.lower()), 
                to_type
                )

        if isinstance(to_type, PrimitiveType) and to_type.t.matches.Int:
            return TypedExpression(
                native_ast.Expression.Cast(left=instance.expr, to_type=to_type.lower()), 
                to_type
                )

        assert False, "can't convert %s to type %s" % (self, to_type)

    def __repr__(self):
        return "Pointer(%s)" % self.value_type


class PrimitiveType(Type):
    def __init__(self, t):
        Type.__init__(self)
        self.t = t

    @property
    def null_value(self):
        if self.t.matches.Float:
            return native_ast.Constant.Float(bits=self.t.bits,val=0.0)

        if self.t.matches.Int:
            return native_ast.Constant.Int(bits=self.t.bits,signed=self.t.signed,val=0)

        if self.t.matches.Void:
            return native_ast.Constant.Void()

        assert False, self.t

    @property
    def is_pod(self):
        return True

    def lower(self):
        return self.t

    def __repr__(self):
        return str(self.t)

class PrimitiveNumericType(PrimitiveType):
    def __init__(self, t):
        PrimitiveType.__init__(self, t)

        assert t.matches.Float or t.matches.Int

    @property
    def sizeof(self):
        if self.t.bits == 1:
            return 1
        assert self.t.bits % 8 == 0
        return self.t.bits / 8

    def bigger_type(self, other):
        if self.t.matches.Float and other.t.matches.Int:
            return self
        if self.t.matches.Int and other.t.matches.Float:
            return other
        if self.t.matches.Int:
            return PrimitiveNumericType(
                native_ast.Type.Int(
                    bits = max(self.t.bits, other.t.bits),
                    signed = self.t.signed or other.t.signed
                    )
                )
        else:
            return PrimitiveNumericType(
                native_ast.Type.Float(
                    bits = max(self.t.bits, other.t.bits)
                    )
                )

    def convert_unary_op(self, instance, op):
        if op.matches.UAdd or op.matches.USub:
            return TypedExpression(
                native_ast.Expression.Unaryop(
                    op=native_ast.UnaryOp.Add() if op.matches.UAdd else native_ast.UnaryOp.Negate(),
                    operand=instance.expr
                    ),
                self
                )

        return super(PrimitiveNumericType, self).convert_unary_op(instance, op)

    def convert_bin_op(self, op, l, r):
        target_type = self.bigger_type(r.expr_type)

        if r.expr_type != target_type:
            r = r.expr_type.convert_to_type(r, target_type)

        if l.expr_type != target_type:
            l = l.expr_type.convert_to_type(l, target_type)

        if op._alternative is python_ast.BinaryOp:
            for py_op, native_op in [('Add','Add'),('Sub','Sub'),('Mult','Mul'),('Div','Div')]:
                if getattr(op.matches, py_op):
                    return TypedExpression(
                        native_ast.Expression.Binop(
                            op=getattr(native_ast.BinaryOp,native_op)(),
                            l=l.expr,
                            r=r.expr
                            ),
                        target_type
                        )
        if op._alternative is python_ast.ComparisonOp:
            for opname in ['Gt','GtE','Lt','LtE','Eq','NotEq']:
                if getattr(op.matches, opname):
                    return TypedExpression(
                        native_ast.Expression.Binop(
                            op=getattr(native_ast.BinaryOp,opname)(),
                            l=l.expr,
                            r=r.expr
                            ),
                        Bool
                        )


        assert False, "can't handle binary op %s between %s and %s" % (op, l.expr_type, r.expr_type)

    def convert_to_type(self, e, other_type):
        if other_type == self:
            return e

        if isinstance(other_type, PrimitiveNumericType):
            return TypedExpression(
                native_ast.Expression.Cast(left=e.expr, to_type=other_type.lower()),
                other_type
                )

        if isinstance(other_type, Pointer) and self.t.matches.Int:
            return TypedExpression(
                native_ast.Expression.Cast(left=e.expr, to_type=other_type.lower()),
                other_type
                )

        raise ConversionException("can't convert %s to %s" % (self, other_type))

class RepresentationlessType(Type):
    @property
    def is_pod(self):
        return True
    
    @property
    def sizeof(self):
        return 0

    @property
    def null_value(self):
        return native_ast.Constant.Void()

    def lower(self):
        return native_ast.Type.Void()

    @property
    def python_object_representation(self):
        assert False, "Subclasses must implement"

def representation_for(obj):
    def decorator(override):
        FreePythonObjectReference.free_python_object_overrides[obj] = override
        return override
    return decorator

class FreePythonObjectReference(RepresentationlessType):
    free_python_object_overrides = {}

    def __init__(self, obj):
        object.__init__(self)
        self._original_obj = obj

        if obj in self.free_python_object_overrides:
            obj = self.free_python_object_overrides[obj]
        self._obj = obj

    @property
    def python_object_representation(self):
        return self._original_obj
        
    def convert_attribute(self, instance, attr):
        return pythonObjectRepresentation(getattr(self._obj, attr))

    def convert_call(self, context, instance, args):
        if isinstance(self._obj, function_type):
            return context.call_py_function(self._obj, args)

        if self._obj is float:
            assert len(args) == 1
            return args[0].convert_to_type(Float64)

        if self._obj is int:
            assert len(args) == 1
            return args[0].convert_to_type(Int64)

        if (isinstance(self._obj, type) 
                and self._obj.__module__ != '__builtin__' 
                and not issubclass(self._obj, Type) or isinstance(self._obj, classobj)):
            init_func = getattr(self._obj, "__init__").im_func
            
            cur_types = ()
            while True:
                try:
                    cls_type = PythonClass(self._obj, cur_types)
                    call_target = context._converter.convert(init_func, [Reference(cls_type)] + \
                        [a.expr_type for a in args], name_override=self._obj.__name__+".__init__")
                    break
                except UnassignableFieldException as e:
                    if e.obj_type == cls_type:
                        cur_types = cur_types + ((e.attr, e.target_type),)
                    else:
                        raise

            tmp_ptr = context.allocate_temporary(cls_type)

            return TypedExpression(
                cls_type.convert_initialize(context, tmp_ptr, ()).expr + 
                    context.activates_temporary(tmp_ptr) + 
                    context.generate_call_expr(
                        target=call_target.native_call_target,
                        args=[tmp_ptr.load.reference.expr] 
                              + [a.as_function_call_arg() for a in args]
                        ) + 
                    native_ast.Expression.Load(tmp_ptr.expr),
                cls_type
                )

        if isinstance(self._obj, Type):
            #we are initializing an element of the type
            for a in args:
                assert a.expr is not None

            tmp_ptr = context.allocate_temporary(self._obj)

            return TypedExpression(
                self._obj.convert_initialize(context, tmp_ptr, args).expr + 
                    context.activates_temporary(tmp_ptr) + 
                    tmp_ptr.expr.load(),
                self._obj
                )

        def to_py(x):
            if isinstance(x.expr_type, FreePythonObjectReference):
                return x.expr_type._obj
            if x.expr is not None and x.expr.matches.Constant:
                if x.expr.val.matches.Int or x.expr.val.matches.Float:
                    return x.expr.val.val
            return x

        call_args = [to_py(x) for x in args]
        try:
            py_call_result = self._obj(*call_args)
        except Exception as e:
            raise ConversionException("Failed to call %s with %s" % (self._obj, call_args))

        return pythonObjectRepresentation(py_call_result)

    def __repr__(self):
        return "FreePythonObject(%s)" % self._obj

class PythonClass(Type):
    def __init__(self, cls, element_types):
        self.cls = cls
        self.element_types = element_types

    @property
    def is_pod(self):
        return False

    def lower(self):
        native_ast.Type.Struct([(n,t.lower()) for t in self.element_types])

    @property
    def null_value(self):
        return native_ast.Constant.Struct(
                [(name,t.null_value) for name,t in self.element_types]
                )

    def convert_initialize_copy(self, context, instance_ptr, other_instance):
        if hasattr(self.cls, "__copy_constructor__"):
            def make_body(instance_ptr, other_instance):
                init_func = self.cls.__copy_constructor__.im_func

                call_target = context._converter.convert(
                    init_func, 
                    [instance_ptr.expr_type, other_instance.expr_type],
                    name_override=self.cls.__name__+".__copy_constructor__"
                    )
                    
                return TypedExpression(
                    self.convert_initialize(context, instance_ptr, ()).expr + 
                        context.generate_call_expr(
                            target=call_target.native_call_target,
                            args=[instance_ptr.as_function_call_arg(), 
                                  other_instance.as_function_call_arg()]
                            ),
                    Void
                    )

            return context.call_expression_in_function(
                (self, "initialize_copy"), "%s.initialize_copy" % (self.cls.__name__), 
                [instance_ptr.load.reference, other_instance.reference], 
                make_body
                )
        else:
            assert other_instance.expr_type == self
            assert instance_ptr.expr_type.value_type == self

            def make_body(instance_ptr, other_instance):
                body = native_ast.nullExpr

                for ix,(name,e) in enumerate(self.element_types):
                    dest_field = self.pointer_to_field(instance_ptr.expr, name)
                    source_field = other_instance.convert_attribute(name)

                    dest_t = dest_field.expr_type.value_type

                    body = body + \
                        dest_t.convert_initialize_copy(context, dest_field, source_field).expr

                return TypedExpression(body, Void)

            return context.call_expression_in_function(
                (self, "initialize_copy"), 
                "%s.initialize_copy" % (str(self)), 
                [instance_ptr, other_instance.reference], 
                make_body
                )

    def convert_destroy(self, context, instance_ptr):
        assert instance_ptr.expr_type.value_type == self

        def make_body(instance_ptr):
            expr = native_ast.nullExpr

            if hasattr(self.cls, "__destructor__"):
                destructor_func = self.cls.__destructor__.im_func
                expr = context.call_py_function(
                    destructor_func, 
                    [instance_ptr.load.reference],
                    name_override=self.cls.__name__+".__destructor__").expr

            for ix,(name,e) in enumerate(self.element_types):
                dest_field = self.pointer_to_field(instance_ptr.expr, name)

                dest_t = dest_field.expr_type.value_type
                expr = expr + dest_t.convert_destroy(context, dest_field).expr

            return TypedExpression(expr, Void)

        return context.call_expression_in_function(
            (self,'destroy'), 
            "%s.destroy" % (self.cls.__name__),
            [instance_ptr],
            make_body
            )

    def convert_initialize(self, context, instance_ptr, args):
        if len(args) != 0:
            assert len(args) == len(self.element_types), (len(self.element_types), len(args))

        def make_body(instance_ptr, *args):
            body = native_ast.nullExpr

            for ix,(name,e) in enumerate(self.element_types):
                dest_field = self.pointer_to_field(instance_ptr.expr, name)

                dest_t = dest_field.expr_type.value_type
                if args:
                    body = body + dest_t.convert_initialize(context, dest_field, (args[ix],)).expr
                else:
                    body = body + dest_t.convert_initialize(context, dest_field, ()).expr

            return TypedExpression(body, Void)

        return context.call_expression_in_function(
            (self, "initialize"), 
            "%s.initialize" % (self.cls.__name__), 
            [instance_ptr] + list(args), 
            make_body
            )

    def convert_assign(self, context, instance_ptr, arg):
        assert arg.expr_type == self, "can't assign %s to %s" % (arg.expr_type, self)

        def make_body(instance_ptr, arg):
            expr = native_ast.nullExpr

            if hasattr(self.cls, "__assign__"):
                assign_func = self.cls.__assign__.im_func
                return context.call_py_function(
                    assign_func, 
                    [instance_ptr.load.reference, arg],
                    name_override=self.cls.__name__+".__assign__"
                    )

            for ix,(name,e) in enumerate(self.element_types):
                dest_field = self.pointer_to_field(instance_ptr.expr, name)
                source_field = arg.convert_attribute(name)

                dest_t = dest_field.expr_type.value_type
                expr = expr + dest_t.convert_assign(context, dest_field, source_field).expr

            return TypedExpression(expr, Void)

        return context.call_expression_in_function(
            (self,"assign"), 
            "%s.assign" % self.cls.__name__, 
            [instance_ptr, arg.reference], 
            make_body
            )

    def lower(self):
        return native_ast.Type.Struct(tuple([(a[0], a[1].lower()) for a in self.element_types]))

    def convert_attribute(self, instance, attr):
        for i in xrange(len(self.element_types)):
            if self.element_types[i][0] == attr:
                return TypedExpression(
                    native_ast.Expression.Attribute(left=instance.expr,attr=attr), 
                    self.element_types[i][1]
                    )

        func = None
        try:
            func = getattr(self.cls, attr).im_func
        except AttributeError:
            pass

        if func is not None:
            return TypedExpression(instance.address.expr, PythonClassMemberFunc(self, attr))

        return super(PythonClass,self).convert_attribute(instance, attr)

    def pointer_to_field(self, native_instance_ptr, attribute_name):
        for i in xrange(len(self.element_types)):
            if self.element_types[i][0] == attribute_name:
                return TypedExpression(
                    native_instance_ptr.ElementPtrIntegers(0, i),
                    self.element_types[i][1].pointer
                    )

    def field(self, native_instance, attribute_name):
        for i in xrange(len(self.element_types)):
            if self.element_types[i][0] == attribute_name:
                return TypedExpression(
                    native_ast.Expression.Attribute(
                        left=native_instance, 
                        attr=attribute_name
                        ),
                    self.element_types[i][1]
                    )


    def convert_set_attribute(self, instance, attr, val):
        field_ptr = self.pointer_to_field(instance.address.expr, attr)

        if field_ptr is None:
            raise UnassignableFieldException(self, attr, val.expr_type)

        if val.expr_type != field_ptr.expr_type.value_type:
            raise ConversionException(
                "Can't assign value of type %s to struct field %s" % (
                    val.expr_type,
                    field_ptr.expr_type.value_type
                    )
                )

        return TypedExpression(
            native_ast.Expression.Store(
                ptr=field_ptr.expr,
                val=val.expr
                ),
            Void
            )

    @property
    def sizeof(self):
        return sum(t.sizeof for n,t in self.element_types)

    def __str__(self):
        return "Class(%s,%s)" % (self.cls, ",".join(["%s=%s" % t for t in self.element_types]))

class PythonClassMemberFunc(Type):
    def __init__(self, python_class_type, attr):
        self.python_class_type = python_class_type
        self.attr = attr

    def lower(self):
        return self.python_class_type.pointer.lower()

    @property
    def is_pod(self):
        return True

    @property
    def sizeof(self):
        return self.python_class_type.pointer.sizeof

    def convert_call(self, context, instance, args):
        func = getattr(self.python_class_type.cls, self.attr).im_func
        
        obj_ptr = TypedExpression(instance.expr, self.python_class_type.pointer)

        return context.call_py_function(
            func, 
            [obj_ptr.load.reference] + args, 
            name_override=self.python_class_type.cls.__name__ + "." + self.attr
            )

    def __str__(self):
        return "ClassMemberFunction(%s,%s,%s)" % (
            self.python_class_type.cls, 
            self.attr, 
            ",".join(["%s=%s" % t for t in self.python_class_type.element_types])
            )

def pythonObjectRepresentation(o):
    if isinstance(o,TypedExpression):
        return o

    if isinstance(o, int):
        return TypedExpression(
            native_ast.Expression.Constant(
                native_ast.Constant.Int(val=o,bits=64,signed=True)
                ), 
            Int64
            )

    if isinstance(o, str):
        return TypedExpression(
            native_ast.Expression.Constant(
                native_ast.Constant.ByteArray(bytes(o))
                ), 
            UInt8.pointer
            )

    if isinstance(o, float):
        return TypedExpression(
            native_ast.Expression.Constant(
                native_ast.Constant.Float(val=o,bits=64)
                ), 
            Float64
            )

    if isinstance(o,RepresentationlessType):
        return TypedExpression(native_ast.nullExpr, o)

    return TypedExpression(native_ast.nullExpr, FreePythonObjectReference(o))

class TypedExpression(object):
    def __init__(self, expr, expr_type):
        object.__init__(self)

        assert expr._alternative is native_ast.Expression
        assert isinstance(expr_type, Type) or expr_type is None

        if expr_type and not expr_type.is_pod:
            assert self.is_addressable_expr(expr), expr
            

        self.expr = expr
        self.expr_type = expr_type

    @classmethod
    def is_addressable_expr(cls, e):
        if e.matches.Load:
            return True
        if e.matches.Sequence:
            return cls.is_addressable_expr(e.vals[-1])
        return False

    def __add__(self, other):
        if self.expr_type is None:
            return self

        return TypedExpression(self.expr + other.expr, other.expr_type)

    def as_function_call_arg(self):
        """This expression is being passed to a function. Our calling convention
        requires non-pod to be passed as a pointer. This means we need to be
        able to find an actual stackslot for this expression."""

        if self.expr_type.is_pod:
            return self.expr

        return self.address.expr

    def convert_unary_op(self, op):
        return self.expr_type.convert_unary_op(self, op)

    def convert_bin_op(self, op, r):
        return self.expr_type.convert_bin_op(op, self, r)

    def convert_call(self, context, args):
        return self.expr_type.convert_call(context, self, args)

    def convert_attribute(self, attr):
        return self.expr_type.convert_attribute(self, attr)

    def convert_set_attribute(self, attr, val):
        return self.expr_type.convert_set_attribute(self, attr, val)

    def convert_to_type(self, type):
        return self.expr_type.convert_to_type(self, type)

    def convert_setitem(self, index, value):
        return self.expr_type.convert_setitem(self, index, value)

    def convert_getitem(self, index):
        return self.expr_type.convert_getitem(self, index)

    @property
    def load(self):
        assert isinstance(self.expr_type, Pointer)
        return TypedExpression(self.expr.load(), self.expr_type.value_type)

    @property
    def address(self):
        if self.expr.matches.Sequence:
            sub = TypedExpression(self.expr.vals[-1], self.expr_type).address
            return TypedExpression(
                native_ast.Expression.Sequence(self.expr.vals[:-1] + (sub.expr,)), 
                sub.expr_type
                )

        assert self.expr.matches.Load, \
            "Can't take the address of expression %s: %s" % (self, self.expr)

        return TypedExpression(self.expr.ptr, Pointer(self.expr_type))

    @property
    def reference(self):
        if isinstance(self.expr_type, Reference):
            return self

        return TypedExpression(self.address.expr, Reference(self.expr_type))

    @property
    def dereference(self):
        if isinstance(self.expr_type, Reference):
            return TypedExpression(self.expr.load(), self.expr_type.value_type)
        else:
            return self

    def __repr__(self):
        return "TypedExpression(t=%s)" % (self.expr_type)

class TypedCallTarget(object):
    def __init__(self, native_call_target, input_types, output_type):
        object.__init__(self)
        self.native_call_target = native_call_target
        self.input_types = input_types
        self.output_type = output_type

    @property
    def name(self):
        return self.native_call_target.name

Float64 = PrimitiveNumericType(native_ast.Type.Float(bits=64))
Int64 = PrimitiveNumericType(native_ast.Type.Int(bits=64, signed=True))
Int32 = PrimitiveNumericType(native_ast.Type.Int(bits=32, signed=True))
Int8 = PrimitiveNumericType(native_ast.Type.Int(bits=8, signed=True))
UInt8 = PrimitiveNumericType(native_ast.Type.Int(bits=8, signed=False))
Bool = PrimitiveNumericType(native_ast.Type.Int(bits=1, signed=False))
Void = PrimitiveType(native_ast.Type.Void())

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
            slot_type.pointer
            )

    def call_py_function(self, f, args, name_override=None):
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
                    args=[slot.expr] + [a.as_function_call_arg() for a in args]
                    ) 
                    + self.activates_temporary(slot)
                    + slot.load.expr
                    ,
                call_target.output_type
                )        
        else:
            assert len(call_target.native_call_target.arg_types) == len(args)

            return TypedExpression(
                self.generate_call_expr(
                    target=call_target.native_call_target,
                    args=[a.as_function_call_arg() for a in args]
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
                ConversionScopeInfo.CreateFromAst(ast, self._varname_to_type)
                )
            raise

    def convert_expression_ast_(self, ast):
        if ast.matches.Attribute:
            attr = ast.attr
            val = self.convert_expression_ast(ast.value)
            return val.convert_attribute(attr)

        if ast.matches.Name:
            assert ast.ctx.matches.Load
            if ast.id in self._varname_to_type:
                assert self._varname_to_type[ast.id] is not None, \
                    "variable %s has no type inferred yet" % ast.id

                return TypedExpression(
                    native_ast.Expression.Load(native_ast.Expression.Variable(ast.id+".slot")),
                    self._varname_to_type[ast.id]
                    ).dereference

            if ast.id in self._free_variable_lookup:
                return pythonObjectRepresentation(self._free_variable_lookup[ast.id])
            elif ast.id in __builtins__:
                return pythonObjectRepresentation(__builtins__[ast.id])

            assert ast.id in self._varname_to_type, (
                "can't find %s in %s. free are %s" 
                    % (ast.id, self._varname_to_type.keys(), self._free_variable_lookup.keys())
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

            return val.convert_getitem(index)

        if ast.matches.Call:
            l = self.convert_expression_ast(ast.func)
            args = [self.convert_expression_ast(a) for a in ast.args]

            init = native_ast.nullExpr

            if not ast.starargs.matches.Null:
                starargs = self.convert_expression_ast(ast.starargs.val)

                assert isinstance(starargs.expr_type, Struct), starargs.expr_type

                tmp_ptr = self.allocate_temporary(starargs.expr_type)
                init = starargs.expr_type.convert_initialize_copy(self, tmp_ptr, starargs)
                init = init + TypedExpression(self.activates_temporary(tmp_ptr), Void)

                for i in xrange(len(starargs.expr_type.element_types)):
                    args.append(
                        TypedExpression(
                            native_ast.Expression.Load(
                                tmp_ptr.expr.ElementPtrIntegers(0,i)
                                ),
                            starargs.expr_type.element_types[i][1]
                            )
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

            struct_type = Struct([("f%s"%i,e.expr_type) for i,e in enumerate(elts)])

            tmp_ptr = self.allocate_temporary(struct_type)

            return TypedExpression(
                struct_type.convert_initialize(self, tmp_ptr, elts).expr + 
                    self.activates_temporary(tmp_ptr) + 
                    tmp_ptr.expr.load(),
                struct_type
                )

        assert False, "can't handle %s" % ast

    def convert_statement_ast_and_teardown_tmps(self, ast):
        assert not self._new_temporaries
        
        expr = self.convert_statement_ast(ast)

        assert expr is not None
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
                        self._temporaries[tname].pointer
                        )
                    ).expr
                )

            if not teardown.expr.matches.Constant:
                teardowns.append(teardown)
        
        self._new_temporaries = set()

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
            e.add_scope(ConversionScopeInfo.CreateFromAst(ast, self._varname_to_type))

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

                val_to_store = self.convert_expression_ast(ast.value)

                if varname not in self._varname_to_type:
                    raise ConversionException("Can't store in variable %s" % varname)

                if self._varname_to_type[varname] is None:
                    self._new_variables.add(varname)

                    #we never construct new stackslots with references
                    self._varname_to_type[varname] = val_to_store.expr_type.unwrap_reference()
                else:
                    assert (self._varname_to_type[varname].unwrap_reference() == 
                                    val_to_store.expr_type),\
                        "can't assign both %s and %s to %s" % (
                                self._varname_to_type[varname], 
                                val_to_store.expr_type, varname
                                )

                if isinstance(self._varname_to_type[varname], Reference):
                    ptr_expr = TypedExpression(
                        native_ast.Expression.Load(
                            native_ast.Expression.Variable(varname+".slot")
                            ), 
                        self._varname_to_type[varname]
                        )
                else:
                    ptr_expr = TypedExpression(
                                native_ast.Expression.Variable(varname+".slot"), 
                                self._varname_to_type[varname].pointer
                                )

                if op is not None:
                    val_to_store = ptr_expr.load.convert_bin_op(op, val_to_store)

                return ptr_expr.expr_type.value_type.convert_assign(self, ptr_expr, val_to_store)

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
            if ast.value.matches.Null:
                e = TypedExpression(native_ast.nullExpr, Void)
            else:
                e = self.convert_expression_ast(ast.value.val)

            if self._varname_to_type[FunctionOutput] is not None:
                assert self._varname_to_type[FunctionOutput] == e.expr_type, \
                    "Function returning multiple types (%s and %s)" % (
                            e.expr_type, 
                            self._varname_to_type[FunctionOutput]
                            )
            else:
                self._varname_to_type[FunctionOutput] = e.expr_type

            output_type = self._varname_to_type[FunctionOutput]

            if output_type.is_pod:
                if e.expr_type == Void:
                    return e + TypedExpression(native_ast.Expression.Return(arg=None), None)

                return TypedExpression(native_ast.Expression.Return(e.expr), None)
            else:
                return TypedExpression(
                    output_type
                        .convert_initialize_copy(
                            self,
                            TypedExpression(
                                native_ast.Expression.Variable(".return"),
                                output_type.pointer
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

        if ast.matches.While:
            cond = self.convert_expression_ast(ast.test)
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

        if ast.matches.If:
            cond = self.convert_expression_ast(ast.test)

            if cond.expr.matches.Constant:
                truth_val = cond.expr.val.truth_value()
                branch = self.convert_statement_list_ast(ast.body if truth_val else ast.orelse)
                
                return cond + branch

            true = self.convert_statement_list_ast(ast.body)
            false = self.convert_statement_list_ast(ast.orelse)

            if true.expr_type or false.expr_type:
                ret_type = Void
            else:
                ret_type = None

            return TypedExpression(
                native_ast.Expression.Branch(cond=cond.expr,true=true.expr,false=false.expr),
                ret_type
                )

        if ast.matches.Pass:
            return TypedExpression(native_ast.nullExpr, Void)

        if ast.matches.For:
            statements = []
            ctx_load = python_ast.ExprContext.Load()
            ctx_store = python_ast.ExprContext.Store()
            iterator_name = self.let_varname()
            self._varname_to_type[iterator_name] = None

            def scope(a):
                return {
                    'filename': a.filename, 
                    'line_number': a.line_number,
                    'col_offset': a.col_offset
                    }

            with_scope = scope(ast.iter)

            def name(n,c=ctx_load):
                return python_ast.Expr.Name(id=n,ctx=c,**with_scope)

            def memb(f, member):
                return python_ast.Expr.Attribute(value=f, attr=member, ctx=ctx_load,**with_scope)

            def call(f, member, args):
                if member is not None:
                    f = memb(f, member)

                return python_ast.Expr.Call(func=f, args=args,keywords=(),
                                            starargs=None,kwargs=None, **with_scope)

            def assign(n, val):
                return python_ast.Statement.Assign(
                    targets=(name(n,ctx_store),),
                    value=val,
                    **with_scope
                    )

            statements.append(
                assign(iterator_name, call(ast.iter, "__iter__",()))
                )

            cond = call(name(iterator_name), 'has_next',())

            assign_statement = python_ast.Statement.Assign(
                targets=(ast.target,),
                value=call(name(iterator_name),"next",()),
                **with_scope
                )

            statements.append(
                python_ast.Statement.While(
                    test=cond,
                    body=(assign_statement,)+ast.body,
                    orelse=ast.orelse,
                    **with_scope
                    )
                )

            return self.convert_statement_list_ast(statements)


        raise ConversionException("Can't handle python ast Statement.%s" % ast._which)

    def convert_statement_list_ast(self, statements, toplevel=False):
        if not statements:
            return TypedExpression(native_ast.nullExpr, Void)

        exprs = [self.convert_statement_ast_and_teardown_tmps(s) for s in statements]

        #all paths must return
        i = 0
        while i < len(exprs) and exprs[i].expr_type is not None:
            i += 1

        #i contains index of first statement that definitely returns
        if i < len(exprs):
            exprs = exprs[:i+1]
            ret_type = None
        else:
            #we run off the end
            exprs = exprs + [TypedExpression(native_ast.nullExpr, Void)]
            ret_type = Void

        if toplevel and ret_type is not None:
            exprs = exprs + [TypedExpression(native_ast.Expression.Return(None), None)]
            ret_type = None

        if toplevel:
            assert ret_type is None, "Not all control flow paths return a value"

        return TypedExpression(
            native_ast.Expression.Sequence(
                vals=[e.expr for e in exprs]
                ),
            ret_type
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
                args=[a.as_function_call_arg() for a in args]
                ),
            expr.expr_type
            )

    def construct_stackslots_around(self, expr, stararg_name):
        ss_vars = sorted(list(self._varname_to_type))

        to_add = []
        for name in ss_vars:
            if name is not FunctionOutput and name != stararg_name:
                slot_type = self._varname_to_type[name]

                if slot_type is not None:
                    slot_expr = TypedExpression(
                        native_ast.Expression.Variable(name + ".slot"),
                        slot_type.pointer
                        )

                    if name in self._new_variables:
                        to_add.append(
                            slot_type.convert_initialize(self, slot_expr,())
                                .expr.with_comment("initialize empty local %s" % name)
                            )
                    else:
                        if slot_type.is_pod:
                            var_expr = TypedExpression(
                                native_ast.Expression.Variable(name), 
                                slot_type
                                )
                        else:
                            var_expr = TypedExpression(
                                native_ast.Expression.Load(native_ast.Expression.Variable(name)), 
                                slot_type
                                )

                        to_add.append(
                            slot_type.convert_initialize_copy(self, slot_expr, var_expr)
                                .expr.with_comment("initialize %s from arg" % name)
                            )

        if to_add:
            expr = TypedExpression(
                native_ast.Expression.Sequence(
                    to_add + [expr.expr]
                    ),
                expr.expr_type
                )

        destructors = []
        for name in ss_vars:
            if name is not FunctionOutput:
                slot_type = self._varname_to_type[name]

                if slot_type is not None:
                    slot_expr = TypedExpression(
                        native_ast.Expression.Variable(name + ".slot"),
                        slot_type.pointer
                        )

                    destructors.append(
                        native_ast.Teardown.Always(
                            slot_type.convert_destroy(self, slot_expr).expr
                            )
                        )
        
        expr = TypedExpression(
            native_ast.Expression.Finally(
                teardowns=destructors,
                expr=expr.expr
                ),
            expr.expr_type
            )
        
        for name in ss_vars:
            if name is not FunctionOutput:
                slot_type = self._varname_to_type[name]

                if slot_type is not None and name != stararg_name:
                    slot_expr = TypedExpression(
                        native_ast.Expression.Variable(name + ".slot"),
                        slot_type.pointer
                        )

                    expr = TypedExpression(
                        native_ast.Expression.Let(
                            var=name + ".slot",
                            val=native_ast.Expression.Alloca(slot_type.lower()),
                            within=expr.expr
                            ),
                        expr.expr_type
                        )

        return expr

    def construct_starargs_around(self, res, star_args_name):
        args_type = self._varname_to_type[star_args_name]

        return TypedExpression(
            native_ast.Expression.Let(
                var=star_args_name + ".slot",
                val=native_ast.Expression.Alloca(args_type.lower()),
                within=
                    args_type.convert_initialize(
                        self,
                        TypedExpression(
                            native_ast.Expression.Variable(star_args_name + ".slot"),
                            args_type.pointer
                            ),
                        [TypedExpression(
                                native_ast.Expression.Variable(".star_args.%s" % i),
                                args_type.element_types[i][1]
                                ) 
                            for i in xrange(len(args_type.element_types))]
                        ).expr.with_comment("initialize *args slot") +
                    res.expr
                ),
            res.expr_type
            )


class Converter(object):
    def __init__(self):
        object.__init__(self)
        self._names_for_identifier = {}
        self._definitions = {}
        self._targets = {}

        self._unconverted = set()

    def extract_new_function_definitions(self):
        res = {}

        for u in self._unconverted:
            res[u] = self._definitions[u]
        
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
            assert len(input_types) == len(ast.args.args), (input_types, ast.args.args)
        else:
            assert len(input_types) >= len(ast.args.args), (input_types, ast.args.args)

        varname_to_type = {}

        args = []
        for i in xrange(len(ast.args.args)):
            varname_to_type[ast.args.args[i].id] = input_types[i]
            args.append((ast.args.args[i].id, input_types[i].lower_as_function_arg()))

        if star_args_name is not None:
            star_args_count = len(input_types) - len(ast.args.args)

            for i in xrange(len(ast.args.args), len(input_types)):
                args.append(
                    ('.star_args.%s' % (i - len(ast.args.args)), 
                        input_types[i].lower_as_function_arg())
                    )

            starargs_type = Struct(
                [('f_%s' % i, input_types[i+len(ast.args.args)]) 
                    for i in xrange(star_args_count)]
                )

            varname_to_type[star_args_name] = starargs_type

        for v in local_variables:
            if v not in varname_to_type:
                #indicate that this is a local variable but we don't have a type for it yet
                varname_to_type[v] = None

        varname_to_type[FunctionOutput] = None

        subconverter = ConversionContext(self, varname_to_type, free_variable_lookup)

        res = subconverter.convert_statement_list_ast(ast.body, toplevel=True)

        if star_args_name is not None:
            res = subconverter.construct_starargs_around(res, star_args_name)

        res = subconverter.construct_stackslots_around(res, star_args_name)

        return_type = varname_to_type[FunctionOutput] or Void

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
        for i in xrange(len(input_types)):
            varname_to_type[ast.args.args[i].id] = input_types[i]
            args.append((ast.args.args[i].id, input_types[i].lower_as_function_arg()))

        subconverter = ConversionContext(self, varname_to_type, free_variable_lookup)

        expr = subconverter.convert_expression_ast(ast.body)

        expr = subconverter.construct_stackslots_around(expr, None)

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


    def convert(self, f, input_types, name_override=None):
        input_types = tuple(input_types)

        identifier = ("pyfunction", f, input_types)

        if identifier in self._names_for_identifier:
            name = self._names_for_identifier[identifier]
            return self._targets[name]

        pyast = ast_util.pyAstFor(f)

        _, lineno = ast_util.getSourceLines(f)
        _, fname = ast_util.getSourceFilenameAndText(f)

        pyast = ast_util.functionDefOrLambdaAtLineNumber(pyast, lineno)

        pyast = python_ast.convertPyAstToAlgebraic(pyast, fname)

        freevars = dict(f.func_globals)
        if f.func_closure:
            for i in xrange(len(f.func_closure)):
                freevars[f.func_code.co_freevars[i]] = f.func_closure[i].cell_contents

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
