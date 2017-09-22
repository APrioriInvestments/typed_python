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

import types

from nativepython.type_model.type_base import Type
from nativepython.type_model.typed_expression import TypedExpression
from nativepython.exceptions import ConversionException, UnassignableFieldException

import nativepython.native_ast as native_ast

class ClassType(Type):
    def __init__(self, cls, element_types):
        self.cls = cls
        self.element_types = tuple(element_types)

    @staticmethod
    def object_is_class(o):
        return (isinstance(o, type) 
                and o.__module__ != '__builtin__' 
                and not issubclass(o, Type) or isinstance(o, types.ClassType))

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

    @staticmethod
    def convert_class_call(context, cls, args):
        if not hasattr(cls, "__init__"):
            cls_type = ClassType(cls, ())

            tmp_ptr = context.allocate_temporary(cls_type)

            return TypedExpression(
                cls_type.convert_initialize(context, tmp_ptr, ()).expr + 
                    context.activates_temporary(tmp_ptr) + 
                    native_ast.Expression.Load(tmp_ptr.expr),
                cls_type
                )
        else:
            init_func = getattr(cls, "__init__").im_func
            
            cur_types = ()
            while True:
                try:
                    cls_type = ClassType(cls, cur_types)
                    call_target = context._converter.convert(init_func, [cls_type.reference] + \
                        [a.expr_type for a in args], name_override=cls.__name__+".__init__")
                    break
                except UnassignableFieldException as e:
                    if e.obj_type == cls_type:
                        cur_types = cur_types + ((e.attr, e.target_type.nonref_type),)
                    else:
                        raise

            tmp_ref = context.allocate_temporary(cls_type)

            return TypedExpression(
                cls_type.convert_initialize(context, tmp_ref, ()).expr + 
                    context.activates_temporary(tmp_ref) + 
                    context.generate_call_expr(
                        target=call_target.native_call_target,
                        args=[tmp_ref.expr] 
                              + [a.native_expr_for_function_call() for a in args]
                        ) + 
                    tmp_ref.expr,
                tmp_ref.expr_type
                )


    def convert_initialize_copy(self, context, instance_ref, other_instance):
        self.assert_is_instance_ref(instance_ref)

        if hasattr(self.cls, "__copy_constructor__"):
            def make_body(instance_ref, other_instance):
                init_func = self.cls.__copy_constructor__.im_func

                call_target = context._converter.convert(
                    init_func, 
                    [instance_ref.expr_type, other_instance.expr_type],
                    name_override=self.cls.__name__+".__copy_constructor__"
                    )
                    
                return TypedExpression.Void(
                    self.convert_initialize(context, instance_ref, ()).expr + 
                        context.generate_call_expr(
                            target=call_target.native_call_target,
                            args=[instance_ref.native_expr_for_function_call(), 
                                  other_instance.native_expr_for_function_call()]
                            )
                    )

            return context.call_expression_in_function(
                (self, "initialize_copy"), "%s.initialize_copy" % (self.cls.__name__), 
                [instance_ref, other_instance], 
                make_body
                )
        else:
            if not self.is_pod:
                self.assert_is_instance_ref(other_instance)

            def make_body(instance_ref, other_instance):
                body = native_ast.nullExpr

                for ix,(name,e) in enumerate(self.element_types):
                    dest_field = self.reference_to_field(instance_ref.expr, name)
                    source_field = other_instance.convert_attribute(context, name)

                    body = body + dest_field.convert_initialize_copy(context, source_field)

                return TypedExpression.Void(body.expr)

            return context.call_expression_in_function(
                (self, "initialize_copy"), 
                "%s.initialize_copy" % (str(self)), 
                [instance_ref, other_instance], 
                make_body
                )

    def convert_destroy(self, context, instance_ref):
        self.assert_is_instance_ref(instance_ref)
        
        def make_body(instance_ref):
            expr = native_ast.nullExpr

            if hasattr(self.cls, "__destructor__"):
                destructor_func = self.cls.__destructor__.im_func
                expr = context.call_py_function(
                    destructor_func, 
                    [instance_ref],
                    name_override=self.cls.__name__+".__destructor__").expr

            for ix,(name,e) in enumerate(self.element_types):
                dest_field = self.reference_to_field(instance_ref.expr, name)

                dest_t = dest_field.expr_type.value_type
                expr = expr + dest_t.convert_destroy(context, dest_field).expr

            return TypedExpression.Void(expr)

        return context.call_expression_in_function(
            (self,'destroy'), 
            "%s.destroy" % (self.cls.__name__),
            [instance_ref],
            make_body
            )

    def convert_initialize(self, context, instance_ref, args):
        self.assert_is_instance_ref(instance_ref)
        
        if len(args) != 0:
            assert len(args) == len(self.element_types), (len(self.element_types), len(args))

        def make_body(instance_ref, *args):
            body = native_ast.nullExpr

            for ix,(name,e) in enumerate(self.element_types):
                dest_field = self.reference_to_field(instance_ref.expr, name)

                dest_t = dest_field.expr_type.value_type
                if args:
                    body = body + dest_t.convert_initialize_copy(context, dest_field, args[ix]).expr
                else:
                    body = body + dest_t.convert_initialize(context, dest_field, ()).expr

            return TypedExpression.Void(body)

        return context.call_expression_in_function(
            (self, "initialize"), 
            "%s.initialize" % (self.cls.__name__), 
            [instance_ref] + list(args), 
            make_body
            )

    def convert_assign(self, context, instance_ref, arg):
        self.assert_is_instance_ref(instance_ref)
        self.assert_is_instance_ref(arg)
        
        def make_body(instance_ref, arg):
            expr = native_ast.nullExpr

            if hasattr(self.cls, "__assign__"):
                assign_func = self.cls.__assign__.im_func
                return context.call_py_function(
                    assign_func, 
                    [instance_ref, arg],
                    name_override=self.cls.__name__+".__assign__"
                    )

            for ix,(name,e) in enumerate(self.element_types):
                dest_field = self.reference_to_field(instance_ref.expr, name)
                source_field = arg.convert_attribute(context, name)

                dest_t = dest_field.expr_type.value_type
                expr = expr + dest_t.convert_assign(context, dest_field, source_field).expr

            return TypedExpression.Void(expr)

        return context.call_expression_in_function(
            (self,"assign"), 
            "%s.assign" % self.cls.__name__, 
            [instance_ref, arg], 
            make_body
            )

    def lower(self):
        return native_ast.Type.Struct(tuple([(a[0], a[1].lower()) for a in self.element_types]))

    def convert_attribute(self, context, instance_or_ref, attr):
        if self.is_pod and instance_or_ref.expr_type == self:
            instance = instance_or_ref

            for i in xrange(len(self.element_types)):
                if self.element_types[i][0] == attr:
                    return TypedExpression(
                        native_ast.Expression.Attribute(
                            left=instance.expr, 
                            attr=attr
                            ),
                        self.element_types[i][1]
                        )
        else:
            instance_ref = instance_or_ref

            self.assert_is_instance_ref(instance_ref)

            for i in xrange(len(self.element_types)):
                if self.element_types[i][0] == attr:
                    return TypedExpression(
                        instance_ref.expr.ElementPtrIntegers(0,i),
                        self.element_types[i][1].reference
                        ).drop_double_references()

            func = None
            try:
                func = getattr(self.cls, attr).im_func
            except AttributeError:
                pass

            if func is not None:
                return TypedExpression(instance_ref.expr, PythonClassMemberFunc(self, attr))

        return super(ClassType,self).convert_attribute(context, instance_ref, attr)

    def reference_to_field(self, native_instance_ptr, attribute_name):
        for i in xrange(len(self.element_types)):
            if self.element_types[i][0] == attribute_name:
                return TypedExpression(
                    native_instance_ptr.ElementPtrIntegers(0, i),
                    self.element_types[i][1].reference
                    )

    def convert_set_attribute(self, context, instance_ref, attr, val):
        self.assert_is_instance_ref(instance_ref)

        field_ref = self.reference_to_field(instance_ref.expr, attr)

        if field_ref is None:
            raise UnassignableFieldException(self, attr, val.expr_type)

        return field_ref.convert_assign(context, val)

    def convert_getitem(self, context, instance, item):
        if hasattr(self.cls, "__getitem__"):
            getitem = self.cls.__getitem__.im_func
            return context.call_py_function(
                getitem, 
                [instance, item],
                name_override=self.cls.__name__+".__getitem__"
                )

        return super(ClassType, self).convert_getitem(context, instance, item)

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
        instance = instance.dereference()

        func = getattr(self.python_class_type.cls, self.attr).im_func
        
        obj_ref = TypedExpression(instance.expr, self.python_class_type.reference)

        return context.call_py_function(
            func, 
            [obj_ref] + args, 
            name_override=self.python_class_type.cls.__name__ + "." + self.attr
            )

    def __str__(self):
        return "ClassMemberFunction(%s,%s,%s)" % (
            self.python_class_type.cls, 
            self.attr, 
            ",".join(["%s=%s" % t for t in self.python_class_type.element_types])
            )
