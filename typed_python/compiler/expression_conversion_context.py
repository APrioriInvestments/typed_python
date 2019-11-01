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

import typed_python.compiler
import typed_python.compiler.native_ast as native_ast
import typed_python.compiler.type_wrappers.runtime_functions as runtime_functions
from typed_python.compiler.function_stack_state import FunctionStackState
from typed_python.compiler.type_wrappers.none_wrapper import NoneWrapper
from typed_python.compiler.type_wrappers.one_of_wrapper import OneOfWrapper
from typed_python.compiler.python_object_representation import pythonObjectRepresentation
from typed_python.compiler.typed_expression import TypedExpression
from typed_python.compiler.conversion_exception import ConversionException
from typed_python import NoneType, Alternative, OneOf, Int32, ListOf, String
from typed_python._types import getTypePointer

builtinValueIdToNameAndValue = {id(v): (k, v) for k, v in __builtins__.items()}

NoneExprType = NoneWrapper()

typeWrapper = lambda t: typed_python.compiler.python_object_representation.typedPythonTypeToTypeWrapper(t)

ExpressionIntermediate = Alternative(
    "ExpressionIntermediate",
    Effect={"expr": native_ast.Expression},
    Terminal={"expr": native_ast.Expression},
    Simple={"name": str, "expr": native_ast.Expression},
    StackSlot={"name": str, "expr": native_ast.Expression}
)

_memoizedThingsById = {}


class ExpressionConversionContext(object):
    """Context class when we're converting a single compound expression.

    This class tracks creation of temporaries so we can destroy them at the end of expression
    evaluation, and provides convenience methods to allow expression generators to stash
    compound expressions and get back simple variable references.
    """

    def __init__(self, functionContext, variableStates: FunctionStackState):
        self.functionContext = functionContext
        self.intermediates = []
        self.teardowns = []
        self.variableStates = variableStates

    @property
    def converter(self):
        return self.functionContext.converter

    def isEmpty(self):
        return not self.intermediates

    def inputArg(self, type, name):
        return TypedExpression(self, native_ast.Expression.Variable(name), type, type.is_pass_by_ref)

    def zero(self, T):
        """Return a TypedExpression matching the Zero form of the native layout of type T.

        Args:
            T - a wrapper, or a type that will get turned into a wrapper.
        """

        T = typeWrapper(T)

        return TypedExpression(self, T.getNativeLayoutType().zero(), T, False)

    def constantPyObject(self, x):
        """Get a TypedExpression that represents a specific python object as 'object'.

        We do this (at the moment) by encoding the pointer value directly in the generated
        code. Later, when we want our compiled code to be reusable, we'll have to have
        a secondary link stage for this.
        """

        # this guarantees the object stays alive as long as this module
        _memoizedThingsById[id(x)] = x

        return self.push(
            object,
            lambda oExpr:
            oExpr.expr.store(
                runtime_functions.create_pyobj.call(
                    native_ast.const_uint64_expr(id(x)).cast(native_ast.Void.pointer())
                ).cast(oExpr.expr_type.getNativeLayoutType())
            )
        )

    def constant(self, x):
        if isinstance(x, str):
            return typed_python.compiler.type_wrappers.string_wrapper.StringWrapper().constant(self, x)
        if isinstance(x, bool):
            return TypedExpression(self, native_ast.const_bool_expr(x), bool, False)
        if isinstance(x, int):
            return TypedExpression(self, native_ast.const_int_expr(x), int, False)
        if isinstance(x, Int32):
            return TypedExpression(self, native_ast.const_int32_expr(int(x)), Int32, False)
        if isinstance(x, float):
            return TypedExpression(self, native_ast.const_float_expr(x), float, False)
        if x is None:
            return TypedExpression(self, native_ast.nullExpr, type(None), False)

        if id(x) in builtinValueIdToNameAndValue and builtinValueIdToNameAndValue[id(x)][1] is x:
            return self.push(
                object,
                lambda oPtr:
                oPtr.expr.store(
                    runtime_functions.builtin_pyobj_by_name.call(
                        native_ast.const_utf8_cstr(builtinValueIdToNameAndValue[id(x)][0])
                    ).cast(oPtr.expr_type.getNativeLayoutType())
                )
            )

        if isinstance(x, type):
            return pythonObjectRepresentation(self, x)

        raise Exception(f"Couldn't convert {x} to a constant expression.")

    def pushVoid(self, t=None):
        if t is None:
            t = typeWrapper(type(None))

        assert t.is_empty, t
        return TypedExpression(self, native_ast.nullExpr, t, False)

    def pushPod(self, type, expression):
        """stash an expression that generates POD passed as a value"""
        type = typeWrapper(type)

        assert type.is_pod
        assert not type.is_pass_by_ref

        varname = self.functionContext.let_varname()

        self.intermediates.append(
            ExpressionIntermediate.Simple(name=varname, expr=expression)
        )

        return TypedExpression(self, native_ast.Expression.Variable(varname), type, False)

    def pushLet(self, type, expression, isReference):
        """Push an arbitrary expression onto the stack."""
        varname = self.functionContext.let_varname()

        self.intermediates.append(
            ExpressionIntermediate.Simple(name=varname, expr=expression)
        )

        return TypedExpression(self, native_ast.Expression.Variable(varname), type, isReference)

    def pushEffect(self, expression):
        """Push a native expression that has a side effect but no value, and that returns control flow."""
        if expression is None:
            return

        if expression == native_ast.nullExpr:
            return

        self.intermediates.append(
            ExpressionIntermediate.Effect(expression)
        )

    def pushReturnValue(self, expression, isMove=False):
        """Push an expression returning 'expression' in the current function context.

        If 'isMove', then don't incref the result.
        """
        returnType = self.functionContext.currentReturnType()

        assert returnType == expression.expr_type

        if returnType.is_pass_by_ref:
            returnTarget = TypedExpression(
                self,
                native_ast.Expression.Variable(name=".return"),
                returnType,
                True
            )

            if isMove:
                returnTarget.expr.store(expression.nonref_expr)
            else:
                # TODO: Is this really what I want to do?
                if not expression.isReference:
                    expression = self.pushMove(expression)
                returnTarget.convert_copy_initialize(expression)

            self.pushTerminal(
                native_ast.Expression.Return(arg=None)
            )
        else:
            self.pushTerminal(
                native_ast.Expression.Return(arg=expression.nonref_expr)
            )

    def pushTerminal(self, expression):
        """Push a native expression that does not return control flow."""
        self.intermediates.append(
            ExpressionIntermediate.Terminal(expression)
        )

    def pushMove(self, typed_expression):
        """Given a typed expression, allocate space for it on the stack and 'move' it
        (copy its bits, but don't inc or decref it)
        """
        return self.push(
            typed_expression.expr_type,
            lambda other: other.expr.store(typed_expression.nonref_expr),
            wantsTeardown=False
        )

    def let(self, e1, e2):
        v = self.functionContext.let_varname()

        return native_ast.Expression.Let(
            var=v,
            val=e1,
            within=e2(native_ast.Expression.Variable(name=v))
        )

    def pushReferenceCopy(self, type, expression):
        """Given a native expression that returns a reference, duplicate the object
        and return a handle."""
        type = typeWrapper(type)

        toCopy = self.pushReference(type, expression)

        return self.push(type, lambda target: type.convert_copy_initialize(target, toCopy))

    def pushReference(self, type, expression):
        """Push a reference to an object that's guaranteed to be alive for the duration of the expression."""
        type = typeWrapper(type)

        varname = self.functionContext.let_varname()

        self.intermediates.append(
            ExpressionIntermediate.Simple(name=varname, expr=expression)
        )

        return TypedExpression(self, native_ast.Expression.Variable(varname), type, True)

    def allocateUninitializedSlot(self, type):
        type = typeWrapper(type)

        varname = self.functionContext.stack_varname()

        resExpr = TypedExpression(
            self,
            native_ast.Expression.StackSlot(name=varname, type=type.getNativeLayoutType()),
            type,
            True
        )

        if not type.is_pod:
            with self.subcontext() as sc:
                type.convert_destroy(self, resExpr)

            self.teardowns.append(
                native_ast.Teardown.ByTag(
                    tag=varname,
                    expr=sc.result
                )
            )

        return resExpr

    def markUninitializedSlotInitialized(self, slot):
        if slot.expr_type.is_pod:
            return

        assert slot.expr.matches.StackSlot

        self.pushEffect(native_ast.Expression.ActivatesTeardown(slot.expr.name))

    def push(self, type, callback, wantsTeardown=True):
        """Allocate a stackvariable of type 'type' and pass it to 'callback' which should return
        a native_ast.Expression or TypedExpression(None) initializing it.
        """
        type = typeWrapper(type)

        if type.is_pod:
            wantsTeardown = False

        varname = self.functionContext.stack_varname()

        resExpr = TypedExpression(
            self,
            native_ast.Expression.StackSlot(name=varname, type=type.getNativeLayoutType()),
            type,
            True
        )

        expr = callback(resExpr)

        if expr is None:
            expr = native_ast.nullExpr

        if isinstance(expr, TypedExpression):
            assert expr.expr_type.typeRepresentation is NoneType, expr.expr_type
            expr = expr.expr
        else:
            assert isinstance(expr, native_ast.Expression)

        self.intermediates.append(
            ExpressionIntermediate.StackSlot(
                name=varname,
                expr=expr if not wantsTeardown else expr >> native_ast.Expression.ActivatesTeardown(varname)
            )
        )

        if wantsTeardown:
            with self.subcontext() as sc:
                type.convert_destroy(self, resExpr)

            self.teardowns.append(
                native_ast.Teardown.ByTag(
                    tag=varname,
                    expr=sc.result
                )
            )

        return resExpr

    def subcontext(self):
        class Scope:
            def __init__(scope):
                scope.intermediates = None
                scope.teardowns = None

            def __enter__(scope):
                scope.intermediates = self.intermediates
                scope.teardowns = self.teardowns
                self.intermediates = []
                self.teardowns = []

                scope.expr = None

                return scope

            def __exit__(scope, *args):
                scope.result = self.finalize(scope.expr)

                self.intermediates = scope.intermediates
                self.teardowns = scope.teardowns

        return Scope()

    def whileLoop(self, conditionExpr):
        if isinstance(conditionExpr, TypedExpression):
            conditionExpr = conditionExpr.nonref_expr

        class Scope:
            def __init__(scope):
                scope.intermediates = None
                scope.teardowns = None

            def __enter__(scope):
                scope.intermediates = self.intermediates
                scope.teardowns = self.teardowns
                self.intermediates = []
                self.teardowns = []

            def __exit__(scope, *args):
                result = self.finalize(None)
                self.intermediates = scope.intermediates
                self.teardowns = scope.teardowns

                self.pushEffect(
                    native_ast.Expression.While(
                        cond=conditionExpr,
                        while_true=result,
                        orelse=native_ast.nullExpr
                    )
                )
        return Scope()

    def loop(self, countExpr):
        class Scope:
            def __init__(scope):
                scope.intermediates = None
                scope.teardowns = None

            def __enter__(scope):
                scope.counter = self.push(int, lambda counter: counter.expr.store(native_ast.const_int_expr(0)))

                scope.intermediates = self.intermediates
                scope.teardowns = self.teardowns
                self.intermediates = []
                self.teardowns = []

                return scope.counter

            def __exit__(scope, *args):
                result = self.finalize(None)
                self.intermediates = scope.intermediates
                self.teardowns = scope.teardowns

                self.pushEffect(
                    native_ast.Expression.While(
                        cond=scope.counter.nonref_expr.lt(countExpr.nonref_expr),
                        while_true=result >> scope.counter.expr.store(
                            scope.counter.nonref_expr.add(native_ast.const_int_expr(1))
                        ),
                        orelse=native_ast.nullExpr
                    )
                )
        return Scope()

    def switch(self, expression, targets, wantsBailout):
        results = {}

        if wantsBailout:
            targets = tuple(targets) + (None,)
        else:
            targets = tuple(targets)

        class Scope:
            def __init__(scope, target):
                scope.intermediates = []
                scope.teardowns = []
                scope.target = target

            def __enter__(scope):
                scope.intermediates, self.intermediates = self.intermediates, scope.intermediates
                scope.teardowns, self.teardowns = self.teardowns, scope.teardowns

            def __exit__(scope, *args):
                results[scope.target] = self.finalize(None)

                scope.intermediates, self.intermediates = self.intermediates, scope.intermediates
                scope.teardowns, self.teardowns = self.teardowns, scope.teardowns

        class MainScope:
            def __init__(scope):
                pass

            def __enter__(scope):
                return [(target, Scope(target)) for target in targets]

            def __exit__(scope, t, v, traceback):
                if t is None:
                    expr = results.get(targets[-1], native_ast.nullExpr)

                    for t in reversed(targets[:-1]):
                        expr = native_ast.Expression.Branch(
                            cond=expression.cast(native_ast.Int64).eq(native_ast.const_int_expr(t)),
                            true=results.get(t, native_ast.nullExpr),
                            false=expr
                        )

                    self.pushEffect(expr)

        return MainScope()

    def ifelse(self, condition):
        if isinstance(condition, TypedExpression):
            condition = condition.toBool().nonref_expr

        results = {}

        class Scope:
            def __init__(scope, isTrue):
                scope.intermediates = []
                scope.teardowns = []
                scope.isTrue = isTrue

            def __enter__(scope):
                scope.intermediates, self.intermediates = self.intermediates, scope.intermediates
                scope.teardowns, self.teardowns = self.teardowns, scope.teardowns

            def __exit__(scope, *args):
                results[scope.isTrue] = self.finalize(None)

                scope.intermediates, self.intermediates = self.intermediates, scope.intermediates
                scope.teardowns, self.teardowns = self.teardowns, scope.teardowns

        class MainScope:
            def __init__(scope):
                pass

            def __enter__(scope):
                return Scope(True), Scope(False)

            def __exit__(scope, t, v, traceback):
                if t is None:
                    true = results.get(True, native_ast.nullExpr)
                    false = results.get(False, native_ast.nullExpr)

                    if condition.matches.Constant:
                        if condition.val.truth_value():
                            self.pushEffect(true)
                        else:
                            self.pushEffect(false)
                    else:
                        self.pushEffect(
                            native_ast.Expression.Branch(
                                cond=condition,
                                true=true,
                                false=false
                            )
                        )

        return MainScope()

    def finalize(self, expr, exceptionsTakeFrom=None):
        if expr is None:
            expr = native_ast.nullExpr
        elif isinstance(expr, native_ast.Expression):
            pass
        else:
            assert isinstance(expr, TypedExpression), type(expr)
            expr = expr.expr

        for i in reversed(self.intermediates):
            if i.matches.Terminal or i.matches.Effect:
                expr = i.expr >> expr
            elif i.matches.StackSlot:
                expr = i.expr >> expr
            elif i.matches.Simple:
                expr = native_ast.Expression.Let(var=i.name, val=i.expr, within=expr)

        if self.teardowns:
            expr = native_ast.Expression.Finally(expr=expr, teardowns=self.teardowns)

        if exceptionsTakeFrom:
            expr = native_ast.Expression.TryCatch(
                expr=expr,
                varname=self.functionContext.let_varname(),
                handler=runtime_functions.add_traceback.call(
                    native_ast.const_utf8_cstr(self.functionContext.name),
                    native_ast.const_utf8_cstr(exceptionsTakeFrom.filename),
                    native_ast.const_int_expr(exceptionsTakeFrom.line_number)
                ) >> native_ast.Expression.Throw(
                    expr=native_ast.Expression.Constant(
                        val=native_ast.Constant.NullPointer(value_type=native_ast.UInt8.pointer())
                    )
                )
            )

        return expr

    def call_function_pointer(self, funcPtr, args, kwargs, returnType):
        if kwargs:
            raise NotImplementedError("Kwargs not implemented for py-function dispatch yet")

        # force arguments to a type appropriate for argpassing
        native_args = [a.as_native_call_arg() for a in args if not a.expr_type.is_empty]

        if returnType.is_pass_by_ref:
            nativeFunType = native_ast.Type.Function(
                output=native_ast.Void,
                args=[returnType.getNativePassingType()] + [a.expr_type.getNativePassingType() for a in args],
                varargs=False,
                can_throw=True
            )

            return self.push(
                returnType,
                lambda output_slot:
                    native_ast.CallTarget.Pointer(expr=funcPtr.cast(nativeFunType.pointer()))
                    .call(output_slot.expr, *native_args)
            )
        else:
            nativeFunType = native_ast.Type.Function(
                output=returnType.getNativePassingType(),
                args=[a.expr_type.getNativePassingType() for a in args],
                varargs=False,
                can_throw=True
            )

            return self.pushPod(
                returnType,
                native_ast.CallTarget.Pointer(expr=funcPtr.cast(nativeFunType.pointer())).call(*native_args)
            )

    def call_py_function(self, f, args, kwargs, returnTypeOverload=None):
        if kwargs:
            raise NotImplementedError("Kwargs not implemented for py-function dispatch yet")

        call_target = self.functionContext.converter.convert(f, [a.expr_type for a in args], returnTypeOverload)

        if call_target is None:
            self.pushException(TypeError, "Function %s was not convertible." % f.__qualname__)
            return

        return self.call_typed_call_target(call_target, args, kwargs)

    def call_typed_call_target(self, call_target, args, kwargs):
        # force arguments to a type appropriate for argpassing
        native_args = [a.as_native_call_arg() for a in args if not a.expr_type.is_empty]

        if call_target.output_type is None:
            # this always throws
            assert len(call_target.named_call_target.arg_types) == len(native_args)

            self.pushTerminal(call_target.call(*native_args))

            # if you get this spuriously, perhaps one of your code-conversion functions
            # returned None when it meant to return context.pushVoid(), which actually returns Void.
            self.pushException(TypeError, "Unreachable code.")
            return

        if call_target.output_type.is_pass_by_ref:
            return self.push(
                call_target.output_type,
                lambda output_slot: call_target.call(output_slot.expr, *native_args)
            )
        else:
            assert call_target.output_type.is_pod
            if len(call_target.named_call_target.arg_types) != len(native_args):
                raise Exception(
                    f"Can't call {call_target} with {args} because we need "
                    f"{len(call_target.named_call_target.arg_types)} and we got {len(native_args)}"
                )

            return self.pushPod(
                call_target.output_type,
                call_target.call(*native_args)
            )

    def pushComment(self, c):
        self.pushEffect(native_ast.nullExpr.with_comment(c))

    def pushException(self, excType, *args, **kwargs):
        """Push a side-effect that throws an exception of type 'excType'.

        Args:
            excType - either a TypedExpression containing the type of the
                exception, or an actual builtin type.
            args - a list of TypedExpressions or constants that make
                up the arguments to the exception
            kwargs - like args, but keyword arguments for the Exception
                to throw.

        Returns:
            None
        """

        def toTyped(x):
            if isinstance(x, TypedExpression):
                return x
            return self.constant(x)

        excType = toTyped(excType)
        args = [toTyped(x) for x in args]
        kwargs = {k: toTyped(v) for k, v in kwargs.items()}

        if excType is None:
            return None

        exceptionVal = excType.convert_call(args, kwargs)

        if exceptionVal is None:
            return None

        return self.pushExceptionObject(exceptionVal)

    def pushExceptionObject(self, exceptionObject):
        nativeExpr = (
            runtime_functions.initialize_exception.call(
                exceptionObject.nonref_expr.cast(native_ast.VoidPtr)
            )
        )

        nativeExpr = nativeExpr >> native_ast.Expression.Throw(
            expr=native_ast.Expression.Constant(
                val=native_ast.Constant.NullPointer(value_type=native_ast.UInt8.pointer())
            )
        )

        self.pushEffect(nativeExpr)

    def isInitializedVarExpr(self, name):
        if self.functionContext.variableIsAlwaysEmpty(name):
            return self.constant(True)

        return TypedExpression(
            self,
            native_ast.Expression.StackSlot(
                name=name + ".isInitialized",
                type=native_ast.Bool
            ),
            bool,
            isReference=True
        )

    def markVariableInitialized(self, varname):
        if self.functionContext.variableIsAlwaysEmpty(varname):
            raise ConversionException(
                f"Can't mark {varname} initialized because its instances have no content."
            )

        self.pushEffect(self.isInitializedVarExpr(varname).expr.store(native_ast.trueExpr))

    def markVariableNotInitialized(self, varname):
        if self.functionContext.variableIsAlwaysEmpty(varname):
            raise ConversionException(
                f"Can't mark {varname} not initialized because its instances have no content."
            )

        self.pushEffect(self.isInitializedVarExpr(varname).expr.store(native_ast.falseExpr))

    def recastVariableAsRestrictedType(self, expr, varType):
        if varType is not None and varType != expr.expr_type.typeRepresentation:
            if getattr(expr.expr_type.typeRepresentation, "__typed_python_category__", None) == "OneOf":
                return expr.convert_to_type(varType)

            if getattr(expr.expr_type.typeRepresentation, "__typed_python_category__", None) == "Alternative" and \
                    getattr(varType, "__typed_python_category__", None) == "ConcreteAlternative":
                return expr.changeType(varType)
        return expr

    def namedVariableLookup(self, name):
        if self.functionContext.isLocalVariable(name):
            if self.functionContext.externalScopeVarExpr(self, name) is not None:
                res = self.functionContext.externalScopeVarExpr(self, name)

                varType = self.variableStates.currentType(name)
                return self.recastVariableAsRestrictedType(res, varType)

            if self.variableStates.couldBeUninitialized(name):
                isInitExpr = self.isInitializedVarExpr(name)

                with self.ifelse(isInitExpr) as (true, false):
                    with false:
                        self.pushException(UnboundLocalError, "local variable '%s' referenced before assignment" % name)

            res = self.functionContext.localVariableExpression(self, name)
            if res is None:
                return None

            varType = self.variableStates.currentType(name)
            return self.recastVariableAsRestrictedType(res, varType)

        if name in self.functionContext._free_variable_lookup:
            return pythonObjectRepresentation(self, self.functionContext._free_variable_lookup[name])

        if name in __builtins__:
            return pythonObjectRepresentation(self, __builtins__[name])

        self.pushException(NameError, "name '%s' is not defined" % name)
        return None

    def convert_expression_ast(self, ast):
        if ast.matches.Attribute:
            attr = ast.attr
            val = self.convert_expression_ast(ast.value)

            if val is None:
                return None

            return val.convert_attribute(attr)

        if ast.matches.Name:
            assert ast.ctx.matches.Load

            return self.namedVariableLookup(ast.id)

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

        if ast.matches.Bytes:
            return pythonObjectRepresentation(self, ast.s)

        if ast.matches.BoolOp:
            with self.subcontext():
                return_type = OneOfWrapper.mergeTypes((self.convert_expression_ast(v).expr_type.typeRepresentation for v in ast.values))

            def convertBoolOp(return_type, depth=0):
                with self.subcontext() as sc:
                    value = self.convert_expression_ast(ast.values[depth])
                    bool_value = TypedExpression.asBool(value)
                    value_expr = value.convert_to_type(return_type).nonref_expr
                    if bool_value is not None:
                        if depth == len(ast.values) - 1:
                            sc.expr = value_expr
                        else:
                            tail_expr = convertBoolOp(return_type, depth + 1)

                            if ast.op.matches.And:
                                sc.expr = native_ast.Expression.Branch(
                                    cond=bool_value.nonref_expr, true=tail_expr, false=value_expr)
                            elif ast.op.matches.Or:
                                sc.expr = native_ast.Expression.Branch(
                                    cond=bool_value.nonref_expr, true=value_expr, false=tail_expr)
                            else:
                                raise Exception(f"Unknown kind of Boolean operator: {ast.op.Name}")

                return sc.result

            return TypedExpression(self, convertBoolOp(return_type), typeWrapper(return_type), False)

        if ast.matches.BinOp:
            lhs = self.convert_expression_ast(ast.left)

            if lhs is None:
                return None

            rhs = self.convert_expression_ast(ast.right)

            if rhs is None:
                return None

            return lhs.convert_bin_op(ast.op, rhs)

        if ast.matches.UnaryOp:
            operand = self.convert_expression_ast(ast.operand)

            if operand is None:
                return None

            return operand.convert_unary_op(ast.op)

        if ast.matches.Subscript:
            assert ast.slice.matches.Index

            val = self.convert_expression_ast(ast.value)
            if val is None:
                return None
            index = self.convert_expression_ast(ast.slice.value)
            if index is None:
                return None

            return val.convert_getitem(index)

        if ast.matches.Call:
            lhs = self.convert_expression_ast(ast.func)

            if lhs is None:
                return None

            for a in ast.args:
                assert not a.matches.Starred, "not implemented yet"

            args = []
            kwargs = {}

            for a in ast.args:
                args.append(self.convert_expression_ast(a))
                if args[-1] is None:
                    return None

            for keywordArg in ast.keywords:
                argname = keywordArg.arg

                kwargs[argname] = self.convert_expression_ast(keywordArg.value)
                if kwargs[argname] is None:
                    return None

            return lhs.convert_call(args, kwargs)

        if ast.matches.Compare:
            assert len(ast.comparators) == 1, "multi-comparison not implemented yet"
            assert len(ast.ops) == 1

            lhs = self.convert_expression_ast(ast.left)

            if lhs is None:
                return None

            r = self.convert_expression_ast(ast.comparators[0])

            if r is None:
                return None

            return lhs.convert_bin_op(ast.ops[0], r)

        if ast.matches.Tuple:
            raise NotImplementedError("not implemented yet")

        if ast.matches.IfExp:
            test = self.convert_expression_ast(ast.test)
            if test is None:
                return None
            test = test.toBool()
            if test is None:
                return None

            with self.ifelse(test) as (true_block, false_block):
                with true_block:
                    true_res = self.convert_expression_ast(ast.body)
                with false_block:
                    false_res = self.convert_expression_ast(ast.orelse)

                if true_res.expr_type != false_res.expr_type:
                    out_type = typeWrapper(OneOf(true_res.expr_type.typeRepresentation, false_res.expr_type.typeRepresentation))
                else:
                    out_type = true_res.expr_type

                out_slot = self.allocateUninitializedSlot(out_type)

                with true_block:
                    true_res = true_res.convert_to_type(out_type)
                    out_slot.convert_copy_initialize(true_res)
                    self.markUninitializedSlotInitialized(out_slot)

                with false_block:
                    false_res = false_res.convert_to_type(out_type)
                    out_slot.convert_copy_initialize(false_res)
                    self.markUninitializedSlotInitialized(out_slot)

            return out_slot

        if ast.matches.FormattedValue:
            value = self.convert_expression_ast(ast.value)

            if value is None:
                return None

            result = value.convert_format(ast.format_spec)

            if result is None:
                return

            if result.expr_type.typeRepresentation is not String:
                self.pushException(TypeError, "Expected string, but got %s" % result.expr_type.typeRepresentation)
                return None

            return result

        if ast.matches.JoinedStr:
            items_to_join = self.push(ListOf(str), lambda s: s.convert_default_initialize())

            for v in ast.values:
                value = self.convert_expression_ast(v)
                if value is None:
                    return None

                if items_to_join.convert_method_call("append", (value,), {}) is None:
                    return None

            return pythonObjectRepresentation(self, "").convert_method_call("join", (items_to_join,), {})

        raise ConversionException("can't handle python expression type %s" % ast.Name)

    def getTypePointer(self, t):
        """Return a raw type pointer for type t

        Args:
            t - python representation of Type, e.g. int, UInt64, ListOf(String), ...
        """
        return getTypePointer(t)
