#   Copyright 2017-2020 typed_python Authors
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

from typed_python.compiler.global_variable_definition import GlobalVariableMetadata
import typed_python.compiler
import typed_python.compiler.native_ast as native_ast
from typed_python.compiler.type_wrappers.wrapper import Wrapper
from typed_python.compiler.merge_type_wrappers import mergeTypeWrappers
import typed_python.compiler.type_wrappers.runtime_functions as runtime_functions
import types

from typed_python.compiler.type_wrappers.slice_type_object_wrapper import SliceWrapper
from typed_python.compiler.conversion_level import ConversionLevel
from typed_python.SerializationContext import SerializationContext
from typed_python.internals import makeFunctionType, FunctionOverload
from typed_python.compiler.function_stack_state import FunctionStackState
from typed_python.compiler.python_object_representation import pythonObjectRepresentation
from typed_python.compiler.python_object_representation import pythonObjectRepresentationType
from typed_python.compiler.typed_expression import TypedExpression
from typed_python.compiler.conversion_exception import ConversionException
from typed_python import (
    Alternative, OneOf, Int8, Int16, Int32, UInt8, UInt16, UInt32, UInt64, ListOf, Tuple, NamedTuple, TupleOf
)
from typed_python._types import TypeFor, pyInstanceHeldObjectAddress
from typed_python.compiler.type_wrappers.named_tuple_masquerading_as_dict_wrapper import NamedTupleMasqueradingAsDict
from typed_python.compiler.type_wrappers.typed_tuple_masquerading_as_tuple_wrapper import TypedTupleMasqueradingAsTuple
from typed_python.compiler.type_wrappers.python_typed_function_wrapper import PythonTypedFunctionWrapper
from typed_python.compiler.type_wrappers.typed_cell_wrapper import TypedCellWrapper
from typed_python import bytecount

builtinValueIdToNameAndValue = {id(v): (k, v) for k, v in __builtins__.items()}

typeWrapper = lambda t: typed_python.compiler.python_object_representation.typedPythonTypeToTypeWrapper(t)


ExpressionIntermediate = native_ast.ExpressionIntermediate


_pyFuncToFuncCache = {}


FunctionArgMapping = Alternative(
    "FunctionArgMapping",
    Arg=dict(value=object),
    Constant=dict(value=object),
    StarArgs=dict(value=ListOf(object)),
    Kwargs=dict(value=TupleOf(Tuple(str, object)))
)


class ExpressionConversionContext:
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
        return TypedExpression(
            self,
            native_ast.Expression.Variable(name) if not type.is_empty else native_ast.nullExpr,
            type,
            type.is_pass_by_ref
        )

    def zero(self, T):
        """Return a TypedExpression matching the Zero form of the native layout of type T.

        Args:
            T - a wrapper, or a type that will get turned into a wrapper.
        """

        T = typeWrapper(T)

        return TypedExpression(self, T.getNativeLayoutType().zero(), T, False)

    def getTypePointer(self, t):
        """Return a native expression with the type pointer for 't' as a void pointer

        Args:
            t - python representation of Type, e.g. int, UInt64, ListOf(str), ...
        """
        if not isinstance(t, type):
            raise Exception(f"Can't give a type pointer to {t} because its not a type")

        return native_ast.Expression.GlobalVariable(
            name="type_pointer_" + str(id(t)) + "_" + str(t)[:20],
            type=native_ast.VoidPtr,
            metadata=GlobalVariableMetadata.RawTypePointer(
                value=t
            )
        ).load()

    def allocateClassMethodDispatchSlot(self, clsType, methodName, retType, argTupleType, kwargTupleType):
        # the first argument indicates whether this is an instance or type-level dispatch
        assert argTupleType.ElementTypes[0].Value in ('type', 'instance')

        identHash = self.converter.hashObjectToIdentity(
            (clsType, methodName, retType, argTupleType, kwargTupleType)
        )

        return native_ast.Expression.GlobalVariable(
            name="class_dispatch_slot_" + identHash.hexdigest,
            type=native_ast.Int64,
            metadata=GlobalVariableMetadata.ClassMethodDispatchSlot(
                clsType=clsType,
                methodName=methodName,
                retType=retType,
                argTupleType=argTupleType,
                kwargTupleType=kwargTupleType
            )
        ).load()

    def constantTypedPythonObject(self, x, owningGlobalScopeAndName=None):
        wrapper = typeWrapper(type(x))

        meta = None

        if owningGlobalScopeAndName:
            # this object is visible as the member of a module. Instead of
            # serializing it (and its state), we want to make sure we encode
            # the object's location, so that the compiler cache can get the
            # correct version of it.
            globallyVisibleDict, name = owningGlobalScopeAndName

            if SerializationContext().nameForObject(globallyVisibleDict) is not None:
                meta = GlobalVariableMetadata.PointerToTypedPythonObjectAsMemberOfDict(
                    sourceDict=globallyVisibleDict, name=name, type=type(x)
                )

        if meta is None:
            meta = GlobalVariableMetadata.PointerToTypedPythonObject(
                value=x,
                type=type(x)
            )

        return TypedExpression(
            self,
            native_ast.Expression.GlobalVariable(
                # this is bad - we should be using a formal name for
                # this object plus its type, which would be enough to
                # uniquely identify it
                name="typed_python_object_" + str(pyInstanceHeldObjectAddress(x)),
                type=wrapper.getNativeLayoutType(),
                metadata=meta
            ).cast(wrapper.getNativeLayoutType().pointer()),
            wrapper,
            True
        )

    def constantPyObject(self, x, owningGlobalScopeAndName=None):
        """Get a TypedExpression that represents a specific python object as 'object'."""
        wrapper = typeWrapper(object)
        meta = None

        if owningGlobalScopeAndName:
            # this object is visible as the member of a module. Instead of
            # serializing it (and its state), we want to make sure we encode
            # the object's location, so that the compiler cache can get the
            # correct version of it.
            globallyVisibleDict, name = owningGlobalScopeAndName

            if SerializationContext().nameForObject(globallyVisibleDict) is not None:
                meta = GlobalVariableMetadata.PointerToTypedPythonObjectAsMemberOfDict(
                    sourceDict=globallyVisibleDict, name=name, type=object
                )

        if meta is None:
            meta = GlobalVariableMetadata.PointerToPyObject(
                value=x
            )

        return TypedExpression(
            self,
            native_ast.Expression.GlobalVariable(
                # this is bad - we should be using a formal name for
                # this object plus its type, which would be enough to
                # uniquely identify it
                name="py_object_" + str(id(x)),
                type=native_ast.VoidPtr,
                metadata=meta
            ).cast(wrapper.getNativeLayoutType().pointer()),
            wrapper,
            True
        )

    @staticmethod
    def constantType(x, allowArbitrary=False):
        """Return the Wrapper for the type we'd get if we called self.constant(x)
        """
        if isinstance(x, str):
            return typed_python.compiler.type_wrappers.string_wrapper.StringWrapper()

        if isinstance(x, (bool, int, Int32, float)):
            return typeWrapper(type(x))
        if x is None:
            return typeWrapper(type(None))

        if id(x) in builtinValueIdToNameAndValue and builtinValueIdToNameAndValue[id(x)][1] is x:
            return object

        if isinstance(x, type):
            return pythonObjectRepresentationType(x)

        if allowArbitrary:
            return typeWrapper(object)

        raise Exception(f"Couldn't get a type for {x} to a constant expression.")

    def matchExceptionObject(self, exc):
        """Return expression that tests whether current exception is an instance of exception class exc
        """
        return self.push(
            bool,
            lambda oExpr:
            oExpr.expr.store(
                runtime_functions.match_exception.call(
                    native_ast.Expression.GlobalVariable(
                        name="py_exception_type_" + str(exc),
                        type=native_ast.VoidPtr,
                        metadata=GlobalVariableMetadata.IdOfPyObject(
                            value=exc
                        )
                    ).load().cast(native_ast.Void.pointer())
                )
            )
        )

    def matchGivenExceptionObject(self, given, exc):
        """Return expression that tests whether current exception is an instance of exception class exc
        """
        return self.push(
            bool,
            lambda oExpr:
            oExpr.expr.store(
                runtime_functions.match_given_exception.call(
                    given.nonref_expr.cast(native_ast.Void.pointer()),
                    native_ast.Expression.GlobalVariable(
                        name="py_exception_type_" + str(exc),
                        type=native_ast.VoidPtr,
                        metadata=GlobalVariableMetadata.IdOfPyObject(
                            value=exc
                        )
                    ).cast(native_ast.Void.pointer())
                )
            )
        )

    def fetchExceptionObject(self, exc):
        """Get a TypedExpression that represents the currently raised exception, as an object typed as ObjectOfType(exc)
        Don't generate unless you know there is an exception.
        The exception state is cleared.
        """
        return self.push(
            TypeFor(exc),
            lambda oExpr:
            oExpr.expr.store(
                runtime_functions.fetch_exception.call().cast(oExpr.expr_type.getNativeLayoutType())
            )
        )

    def getExceptionObject(self, exc):
        """Get a TypedExpression that represents the last caught exception, as an object typed as ObjectOfType(exc)
        Don't generate unless you know there is an exception.
        The exception state is not cleared.
        """
        return self.push(
            TypeFor(exc),
            lambda oExpr:
            oExpr.expr.store(
                runtime_functions.get_exception.call().cast(oExpr.expr_type.getNativeLayoutType())
            )
        )

    def constant(self, x, allowArbitrary=False):
        """Return a TypedExpression representing 'x'.

        By default, we allow only simple constants that map directly into machine code,
        builtins for which we have a definition, and types.

        If 'allowArbitrary' then all other values will be memoized and held as PyObject*
        in the generated code. Otherwise we'll throw an exception.
        """

        if isinstance(x, str):
            return typed_python.compiler.type_wrappers.string_wrapper.StringWrapper().constant(self, x)
        if isinstance(x, bytes):
            return typed_python.compiler.type_wrappers.bytes_wrapper.BytesWrapper().constant(self, x)
        if isinstance(x, bool):
            return TypedExpression(self, native_ast.const_bool_expr(x), bool, False, constantValue=x)
        if isinstance(x, int):
            return TypedExpression(self, native_ast.const_int_expr(x), int, False, constantValue=x)
        if isinstance(x, Int32):
            return TypedExpression(self, native_ast.const_int32_expr(int(x)), Int32, False, constantValue=x)
        if isinstance(x, Int16):
            return TypedExpression(self, native_ast.const_int16_expr(int(x)), Int16, False, constantValue=x)
        if isinstance(x, Int8):
            return TypedExpression(self, native_ast.const_int8_expr(int(x)), Int16, False, constantValue=x)
        if isinstance(x, UInt64):
            return TypedExpression(self, native_ast.const_uint64_expr(int(x)), UInt64, False, constantValue=x)
        if isinstance(x, UInt32):
            return TypedExpression(self, native_ast.const_uint32_expr(int(x)), UInt32, False, constantValue=x)
        if isinstance(x, UInt16):
            return TypedExpression(self, native_ast.const_uint16_expr(int(x)), UInt16, False, constantValue=x)
        if isinstance(x, UInt8):
            return TypedExpression(self, native_ast.const_uint8_expr(int(x)), UInt8, False, constantValue=x)
        if isinstance(x, float):
            return TypedExpression(self, native_ast.const_float_expr(x), float, False, constantValue=x)
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

        if isinstance(x, (type, types.ModuleType)):
            return pythonObjectRepresentation(self, x)

        if allowArbitrary:
            return self.constantPyObject(x)

        raise Exception(f"Couldn't convert {x} to a constant expression.")

    def pushVoid(self, t=None):
        if t is None:
            t = typeWrapper(type(None))

        assert t.is_empty, t
        return TypedExpression(self, native_ast.nullExpr, t, False)

    def pushPod(self, type, expression):
        """stash an expression that generates POD passed as a value"""
        if not isinstance(type, Wrapper):
            type = typeWrapper(type)

        if not type.is_pod:
            raise Exception(f"Type {type} is not pod. {type.is_pod}")

        if type.is_pass_by_ref:
            raise Exception(f"Type {type} is pass_by_ref")

        varname = self.functionContext.allocateLetVarname()

        self.intermediates.append(
            ExpressionIntermediate.Simple(name=varname, expr=expression)
        )

        return TypedExpression(self, native_ast.Expression.Variable(varname), type, False)

    def pushLet(self, type, expression, isReference):
        """Push an arbitrary expression onto the stack."""
        varname = self.functionContext.allocateLetVarname()

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

        assert returnType == expression.expr_type, (returnType, expression.expr_type)

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
        v = self.functionContext.allocateLetVarname()

        return native_ast.Expression.Let(
            var=v,
            val=e1,
            within=e2(native_ast.Expression.Variable(name=v))
        )

    def pushReference(self, type, expression):
        """Push a reference to an object that's guaranteed to be alive for the duration of the expression."""
        type = typeWrapper(type)

        varname = self.functionContext.allocateLetVarname()

        self.intermediates.append(
            ExpressionIntermediate.Simple(name=varname, expr=expression)
        )

        return TypedExpression(self, native_ast.Expression.Variable(varname), type, True)

    def allocateUninitializedSlot(self, type):
        type = typeWrapper(type)

        varname = self.functionContext.allocateStackVarname()

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

    def pushStackSlot(self, nativeType):
        varname = self.functionContext.allocateStackVarname()

        return native_ast.Expression.StackSlot(name=varname, type=nativeType)

    def push(self, type, callback, wantsTeardown=True):
        """Allocate a stackvariable of type 'type' and pass it to 'callback' which should return
        a native_ast.Expression or TypedExpression(None) initializing it.
        """
        type = typeWrapper(type)

        if type.is_pod:
            wantsTeardown = False

        varname = self.functionContext.allocateStackVarname()

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
            assert expr.expr_type.typeRepresentation is type(None), expr.expr_type  # noqa
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
                    if not targets:
                        return

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
            expr = expr.nonref_expr

        if len(self.intermediates):
            expr = native_ast.Expression.ApplyIntermediates(base=expr, intermediates=self.intermediates)

        if self.teardowns:
            expr = native_ast.Expression.Finally(expr=expr, teardowns=self.teardowns)

        if exceptionsTakeFrom and expr.couldThrow() and exceptionsTakeFrom.filename:
            expr = native_ast.Expression.ExceptionPropagator(
                expr=expr,
                varname=self.functionContext.allocateLetVarname(),
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

    def call_function_pointer(self, funcPtr, args, returnType):
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

    @staticmethod
    def mapFunctionArguments(functionOverload: FunctionOverload, args, kwargs) -> OneOf(str, ListOf(FunctionArgMapping)):
        """Figure out how to call 'functionOverload' with 'args' and 'kwargs'.

        This takes care of doing things like mapping keyword arguments, default values, etc.

        It does _not_ deal at all with types, so it's fine to use the typed-form of a non-typed
        function. The args in 'args/kwargs' can be any object.

        Args:
            functionOverload - a FunctionOverload we're trying to map to
            args - a list of positional arguments. They can be of any type.
            kwargs - a dict of keyword arguments. They can be of any type.

        Returns:
            A ListOf(FunctionArgMapping) mapping to the arguments of the function,
            in the order in which the names appear.

            Otherwise, a string error message.
        """
        name = functionOverload.name

        outArgs = ListOf(FunctionArgMapping)()
        curTargetIx = 0

        minPositional = functionOverload.minPositionalCount()
        maxPositional = functionOverload.maxPositionalCount()

        consumedPositionalNames = set()

        if minPositional == maxPositional:
            positionalMsg = f"{minPositional}"
        else:
            positionalMsg = f"from {minPositional} to {maxPositional}"

        if args and not functionOverload.args:
            return f"{name}() takes {positionalMsg} positional arguments but {len(args)} were given"

        if kwargs and not functionOverload.args:
            return f"{name}() got an unexpected keyword argument '{list(kwargs)[0]}'"

        for a in args:
            if curTargetIx >= len(functionOverload.args):
                return f"{name}() takes {positionalMsg} positional arguments but {len(args)} were given"

            if functionOverload.args[curTargetIx].isKwarg:
                return f"{name}() takes {positionalMsg} positional arguments but {len(args)} were given"

            if functionOverload.args[curTargetIx].isStarArg:
                if len(outArgs) <= curTargetIx:
                    outArgs.append(FunctionArgMapping.StarArgs(ListOf(object)([a])))
                else:
                    outArgs[-1].value.append(a)
            else:
                consumedPositionalNames.add(functionOverload.args[curTargetIx].name)

                outArgs.append(FunctionArgMapping.Arg(value=a))
                curTargetIx += 1

        unconsumedKwargs = dict(kwargs)

        while len(outArgs) < len(functionOverload.args):
            arg = functionOverload.args[len(outArgs)]

            if arg.isStarArg:
                outArgs.append(FunctionArgMapping.StarArgs())
            elif arg.isKwarg:
                for name in unconsumedKwargs:
                    if name in consumedPositionalNames:
                        return f"{name}() got multiple values for argument '{name}'"

                outArgs.append(FunctionArgMapping.Kwargs(value=unconsumedKwargs.items()))
                assert len(outArgs) == len(functionOverload.args)
                unconsumedKwargs = {}
            elif arg.name in kwargs:
                outArgs.append(FunctionArgMapping.Arg(value=unconsumedKwargs[arg.name]))
                del unconsumedKwargs[arg.name]
            elif arg.defaultValue is not None:
                outArgs.append(FunctionArgMapping.Constant(value=arg.defaultValue[0]))
            else:
                return f"{name}() missing required positional argument: {arg.name}"

        for argName in unconsumedKwargs:
            return f"{name}() got an unexpected keyword argument '{argName}'"

        return outArgs

    def buildFunctionArguments(self, functionOverload: FunctionOverload, args, kwargs):
        """Figure out how to call 'functionOverload' with 'args' and 'kwargs'.

        This takes care of doing things like mapping keyword arguments, default values, etc.

        It does _not_ deal at all with types, so it's fine to use the typed-form of a non-typed
        function.

        This function may generate code to construct the relevant arguments.

        Args:
            functionOverload - a FunctionOverload we're trying to map to
            args - a list of positional argument TypedExpression objects.
            kwargs - a dict of keyword argument TypedExpression objects.

        Returns:
            If we can map, a list of TypedExpression objects mapping to
            the argument names of the function, in the order that they appear.

            Otherwise, None, and an exception will have been generated.
        """
        argsOrErr = self.mapFunctionArguments(functionOverload, args, kwargs)

        if isinstance(argsOrErr, str):
            self.pushException(TypeError, argsOrErr)
            return

        outArgs = []

        for mappingArg in argsOrErr:
            if mappingArg.matches.Arg:
                outArgs.append(mappingArg.value)
            elif mappingArg.matches.Constant:
                outArgs.append(self.constant(mappingArg.value, allowArbitrary=True))
            elif mappingArg.matches.StarArgs:
                outArgs.append(self.makeStarArgTuple(mappingArg.value))
            elif mappingArg.matches.Kwargs:
                outArgs.append(self.makeKwargDict(dict(mappingArg.value)))

        return outArgs

    @staticmethod
    def computeOverloadSignature(functionOverload: FunctionOverload, args, kwargs):
        """Figure out the concrete type assignments we'd need to give to each _argument_ to call 'functionOverload'

        Args:
            functionOverload - a FunctionOverload we're trying to map to
            args - a list of positional argument Wrapper objects.
            kwargs - a dict of keyword argument Wrapper objects.

        Returns:
            If we can map, a pair (argsOut, kwargsOut) giving the updated type wrapper assignments.
            There will be one entry in each of argsOut/kwargsOut for each entry in the inputs,
            updated to reflect the required typing judgments that are applied by the overload's
            signature.

            Note that we _dont_ actually check if the argument Wrappers are convertible.
            So it's possible to return a signature that can't be reached by 'args' and
            'kwargs'.

            Otherwise, if the mapping cannot be done, return None.
        """
        argsOrErr = ExpressionConversionContext.mapFunctionArguments(
            functionOverload,
            [i for i in range(len(args))],
            {argName: argName for argName in kwargs}
        )

        if isinstance(argsOrErr, str):
            return None

        outArgTypes = list(args)
        outKwargTypes = dict(kwargs)

        def setType(which, T):
            if isinstance(which, int):
                outArgTypes[which] = typeWrapper(T)
            elif isinstance(which, str):
                outKwargTypes[which] = typeWrapper(T)
            else:
                assert False, type(which)

        for overloadArg, mappingArg in zip(functionOverload.args, argsOrErr):
            if overloadArg.typeFilter is not None:
                if mappingArg.matches.Arg:
                    setType(mappingArg.value, overloadArg.typeFilter)
                elif mappingArg.matches.Constant:
                    pass
                elif mappingArg.matches.StarArgs:
                    for v in mappingArg.value:
                        setType(v, overloadArg.typeFilter)
                elif mappingArg.matches.Kwargs:
                    for name, _ in mappingArg.value:
                        setType(name, overloadArg.typeFilter)
                else:
                    assert False, "Unreachable"

        return outArgTypes, outKwargTypes

    @staticmethod
    def computeFunctionArgumentTypeSignature(functionOverload: FunctionOverload, args, kwargs):
        """Figure out the concrete type assignments we'd give to each variable if we call with args/kwargs.

        This takes care of doing things like mapping keyword arguments, default values, etc.

        It's also careful to apply the typing annotations on typed functions to the types.
        It does also checks whether it's possible to map to those types, and if its definitely
        _not_ possible, this will return None. However, it's always possible
        you will receive a typing form that you can't successfully map to at runtime.

        Args:
            functionOverload - a FunctionOverload we're trying to map to
            args - a list of positional argument Wrapper objects.
            kwargs - a dict of keyword argument Wrapper objects.

        Returns:
            If we can map, a list of Wrapper objects mapping to the argument
            types of the function, in the order that they appear.

            Otherwise, None, and an exception will have been generated.
        """
        argsOrErr = ExpressionConversionContext.mapFunctionArguments(functionOverload, args, kwargs)

        if isinstance(argsOrErr, str):
            return None

        outArgs = []

        for overloadArg, mappingArg in zip(functionOverload.args, argsOrErr):
            if mappingArg.matches.Arg:
                if overloadArg.typeFilter is None:
                    outArgs.append(mappingArg.value)
                else:
                    if mappingArg.value.can_convert_to_type(typeWrapper(overloadArg.typeFilter), ConversionLevel.Implicit) is False:
                        return None
                    outArgs.append(typeWrapper(overloadArg.typeFilter))

            elif mappingArg.matches.Constant:
                constType = ExpressionConversionContext.constantType(mappingArg.value, allowArbitrary=True)

                if overloadArg.typeFilter is None:
                    outArgs.append(constType)
                else:
                    if constType.can_convert_to_type(typeWrapper(overloadArg.typeFilter), ConversionLevel.Implicit) is False:
                        return None

                    outArgs.append(typeWrapper(overloadArg.typeFilter))
            elif mappingArg.matches.StarArgs:
                if overloadArg.typeFilter is None:
                    outArgs.append(ExpressionConversionContext.makeStarArgTupleType(mappingArg.value))
                else:
                    for t in mappingArg.value:
                        if t.can_convert_to_type(typeWrapper(overloadArg.typeFilter), ConversionLevel.Implicit) is False:
                            return None

                    outArgs.append(ExpressionConversionContext.makeStarArgTupleType(
                        [typeWrapper(overloadArg.typeFilter) for _ in mappingArg.value]
                    ))
            elif mappingArg.matches.Kwargs:
                if overloadArg.typeFilter is None:
                    outArgs.append(ExpressionConversionContext.makeKwargDictType(dict(mappingArg.value)))
                else:
                    for _, t in mappingArg.value:
                        if t.can_convert_to_type(typeWrapper(overloadArg.typeFilter), ConversionLevel.Implicit) is False:
                            return None

                    outArgs.append(ExpressionConversionContext.makeKwargDictType(
                        dict({k: typeWrapper(overloadArg.typeFilter) for k, v in mappingArg.value})
                    ))

        return outArgs

    @staticmethod
    def makeStarArgTupleType(tupleArgs):
        tupleType = Tuple(*[a.interpreterTypeRepresentation for a in tupleArgs])

        return TypedTupleMasqueradingAsTuple(tupleType)

    def makeStarArgTuple(self, tupleArgs):
        """Produce the argument that will be passed to a *args argument.

        Note that this version returns a typed tuple, which is not identical
        to what the interpreter does which is return a normal tuple. Eventually
        we should try to do better than this.

        Args:
            tupleArgs - a list of TypedExpression objects giving the arguments
                that were packed into the tuple.

        Returns:
            a TypedExpression for the tuple
        """
        tupleType = Tuple(*[a.expr_type.interpreterTypeRepresentation for a in tupleArgs])

        return typeWrapper(tupleType).createFromArgs(self, tupleArgs).changeType(
            TypedTupleMasqueradingAsTuple(tupleType)
        )

    @staticmethod
    def makeKwargDictType(kwargs):
        tupleType = NamedTuple(**{k: v.interpreterTypeRepresentation for k, v in kwargs.items()})

        return NamedTupleMasqueradingAsDict(tupleType)

    def makeKwargDict(self, kwargs):
        tupleType = NamedTuple(**{k: v.expr_type.interpreterTypeRepresentation for k, v in kwargs.items()})

        return typeWrapper(tupleType).convert_type_call(self, None, (), kwargs).changeType(
            NamedTupleMasqueradingAsDict(tupleType)
        )

    def logDiagnostic(self, *args):
        realArgs = []

        for a in args:
            if isinstance(a, TypedExpression):
                realArgs.append(a)
            else:
                realArgs.append(self.constant(a))

        self.constant(print).convert_call(realArgs, {})

    def call_py_function(self, f, args, kwargs, returnTypeOverload=None):
        """Call a 'free' python function 'f'.

        The function will be memoized by its actual value. This means that
        different functions with the same code and closures but different
        identities will be compiled separately.
        """
        if not isinstance(f, types.FunctionType):
            raise Exception(f"Can't convert a py function of type {type(f)}")

        if f not in _pyFuncToFuncCache:
            _pyFuncToFuncCache[f] = makeFunctionType(f.__name__, f)
        typedFunc = _pyFuncToFuncCache[f]

        concreteArgs = self.buildFunctionArguments(typedFunc.overloads[0], args, kwargs)

        if concreteArgs is None:
            return None

        funcGlobals = dict(f.__globals__)

        globalsInCells = []

        if f.__closure__:
            for i in range(len(f.__closure__)):
                funcGlobals[f.__code__.co_freevars[i]] = f.__closure__[i].cell_contents
                globalsInCells.append(f.__code__.co_freevars[i])

        call_target = self.functionContext.converter.convert(
            f.__name__,
            f.__code__,
            funcGlobals,
            f.__globals__,
            globalsInCells,
            [],
            [a.expr_type for a in concreteArgs],
            returnTypeOverload
        )

        if call_target is None:
            self.pushException(TypeError, "Function %s was not convertible." % f.__qualname__)
            return

        return self.call_typed_call_target(call_target, concreteArgs)

    def call_overload(self, overload, funcObj, args, kwargs, returnTypeOverload=None):
        concreteArgs = self.buildFunctionArguments(overload, args, kwargs)

        if concreteArgs is None:
            return None

        if funcObj is not None:
            closureTuple = funcObj.changeType(funcObj.expr_type.closureWrapper)
        else:
            # if the overload has an empty closure, then callers can just pass 'None'.
            # check to make sure it's really an empty closure. This happens with
            # things like class methods, which are expected to be entirely static.
            assert bytecount(overload.functionTypeObject.ClosureType) == 0
            closureTuple = self.push(typeWrapper(overload.functionTypeObject.ClosureType), lambda x: None)

        closureArgs = [
            PythonTypedFunctionWrapper.closurePathToCellValue(path, closureTuple)
            for path in overload.closureVarLookups.values()
        ]

        if returnTypeOverload is not None:
            returnType = returnTypeOverload
        elif overload.signatureFunction:
            raise Exception("At this point, the signature should be fully known")
        else:
            returnType = typeWrapper(overload.returnType) if overload.returnType is not None else None

        call_target = self.functionContext.converter.convert(
            overload.name,
            overload.functionCode,
            overload.realizedGlobals,
            overload.functionGlobals,
            list(overload.funcGlobalsInCells),
            list(overload.closureVarLookups),
            [a.expr_type for a in closureArgs]
            + [a.expr_type for a in concreteArgs],
            returnType
        )

        if call_target is None:
            self.pushException(TypeError, "Function %s was not convertible." % overload.name)
            return

        return self.call_typed_call_target(call_target, closureArgs + concreteArgs)

    def call_typed_call_target(self, call_target, args):
        # make sure that any non-reference objects that need to be passed by reference
        # get stackslots
        args = list(args)
        for i in range(len(args)):
            if args[i].expr_type.is_pass_by_ref and not args[i].isReference:
                args[i] = self.pushMove(args[i])

        native_args = [a.as_native_call_arg() for a in args if not a.expr_type.is_empty]

        if call_target.output_type is None:
            # this always throws
            assert len(call_target.named_call_target.arg_types) == len(native_args)

            self.pushTerminal(call_target.call(*native_args))

            if not call_target.alwaysRaises:
                # if you get this spuriously, perhaps one of your code-conversion functions
                # returned None when it meant to return context.pushVoid(), which actually returns Void.
                self.pushException(TypeError, "Unreachable code")

            return

        constantRetval = call_target.functionMetadata.getConstantReturnValue()

        if call_target.output_type.is_pass_by_ref:
            assert len(call_target.named_call_target.arg_types) == len(native_args) + 1, "\n\n%s\n%s" % (
                call_target.named_call_target.arg_types,
                [a.expr_type for a in args]
            )

            if constantRetval is not None:
                self.push(
                    call_target.output_type,
                    lambda output_slot: call_target.call(output_slot.expr, *native_args)
                )
                return self.constant(constantRetval)

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

            if constantRetval is not None:
                self.pushEffect(call_target.call(*native_args))
                return self.constant(constantRetval)

            return self.pushPod(
                call_target.output_type,
                call_target.call(*native_args)
            )

    def pushComment(self, c):
        self.pushEffect(native_ast.nullExpr.with_comment(c))

    def pushAttributeError(self, attribute):
        self.pushEffect(
            runtime_functions.raiseAttributeError.call(
                native_ast.const_utf8_cstr(attribute)
            )
        )

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
        if len(args) == 1 and isinstance(args[0], str) and isinstance(excType, type):
            # this is the most common pathway
            if id(excType) in builtinValueIdToNameAndValue:
                self.pushEffect(
                    runtime_functions.raise_exception_fastpath.call(
                        native_ast.const_utf8_cstr(args[0]),
                        native_ast.const_utf8_cstr(builtinValueIdToNameAndValue[id(excType)][0])
                    )
                )
                self.pushEffect(
                    native_ast.Expression.Throw(
                        expr=native_ast.Expression.Constant(
                            val=native_ast.Constant.NullPointer(value_type=native_ast.UInt8.pointer())
                        )
                    )
                )
                return None

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

    def pushExceptionClear(self):
        nativeExpr = (
            runtime_functions.clear_exception.call()
        )
        self.pushEffect(nativeExpr)

    def pushExceptionObject(self, exceptionObject, clear_exc=False):
        if exceptionObject is None:
            exceptionObject = self.zero(object)

        nativeExpr = (
            runtime_functions.initialize_exception.call(
                exceptionObject.nonref_expr.cast(native_ast.VoidPtr)
            )
        )

        if clear_exc:
            nativeExpr = nativeExpr >> native_ast.Expression.Branch(
                cond=self.functionContext.exception_occurred_slot.load(),
                true=runtime_functions.clear_exc_info.call()
            )

        nativeExpr = nativeExpr >> native_ast.Expression.Throw(
            expr=native_ast.Expression.Constant(
                val=native_ast.Constant.NullPointer(value_type=native_ast.UInt8.pointer())
            )
        )

        self.pushEffect(nativeExpr)

    def pushExceptionObjectWithCause(self, exceptionObject, causeObject, deferred=False):
        if exceptionObject is None:
            exceptionObject = self.zero(object)

        if causeObject.expr_type.typeRepresentation is type(None):  # noqa
            causeObject = self.zero(object)

        nativeExpr = (
            runtime_functions.initialize_exception_w_cause.call(
                exceptionObject.nonref_expr.cast(native_ast.VoidPtr),
                causeObject.nonref_expr.cast(native_ast.VoidPtr)
            )
        )
        if deferred:
            nativeExpr = nativeExpr >> native_ast.Expression.Branch(
                cond=self.functionContext.exception_occurred_slot.load(),
                true=runtime_functions.clear_exc_info.call()
            )

        nativeExpr = nativeExpr >> native_ast.Expression.Throw(
            expr=native_ast.Expression.Constant(
                val=native_ast.Constant.NullPointer(value_type=native_ast.UInt8.pointer())
            )
        )

        self.pushEffect(nativeExpr)

    def isInitializedVarExpr(self, name):
        return self.functionContext.isInitializedVarExpr(self, name)

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
            if expr.expr_type.is_oneof_wrapper:
                # if it's a concrete type, then it must be one of our subtypes
                if varType in expr.expr_type.typeRepresentation.Types:
                    # this is only true if we know for certain that exactly one of these
                    # types can be cast to varType and that's it
                    index = expr.expr_type.unambiguousTypeIndexFor(varType)
                    if index is not None:
                        return expr.refAs(index)

                # it's possible that it's a subset of the oneof types
                return expr.convert_to_type(varType, ConversionLevel.Signature, assumeSuccessful=True)

            if getattr(expr.expr_type.typeRepresentation, "__typed_python_category__", None) == "Alternative" and \
                    getattr(varType, "__typed_python_category__", None) == "ConcreteAlternative":
                return expr.changeType(varType)

            # if its a class, and the variable is a subclass of our type
            if expr.expr_type.is_class_wrapper and issubclass(varType, expr.expr_type.classType):
                return expr.convert_to_type(typeWrapper(varType), ConversionLevel.Signature, assumeSuccessful=True)

        return expr

    def namedVariableLookup(self, name):
        if self.functionContext.isLocalVariable(name):
            if self.functionContext.externalScopeVarExpr(self, name) is not None:
                res = self.functionContext.externalScopeVarExpr(self, name)

                if self.functionContext.isClosureVariable(name) and isinstance(res.expr_type, TypedCellWrapper):
                    # explicitly unwrap cells
                    return res.convert_method_call("get", (), {})

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

        if name in self.functionContext._globals:
            if (
                name in self.functionContext._globalsRaw
                and self.functionContext._globalsRaw[name] is self.functionContext._globals[name]
            ):
                scopeAndName = (self.functionContext._globalsRaw, name)
            else:
                scopeAndName = None

            return pythonObjectRepresentation(
                self,
                self.functionContext._globals[name],
                owningGlobalScopeAndName=scopeAndName
            )

        if name in __builtins__:
            return pythonObjectRepresentation(self, __builtins__[name])

        self.pushException(NameError, "name '%s' is not defined" % name)
        return None

    def expressionAsFunctionCall(self, name, args, generatingFunction, identity, outputType=None, alwaysRaises=False):
        callTarget = self.converter.defineNonPythonFunction(
            name,
            ("expression", identity),
            typed_python.compiler.function_conversion_context.ExpressionFunctionConversionContext(
                self.functionContext.converter,
                name,
                ("expression", identity),
                [x.expr_type for x in args],
                generatingFunction,
                outputType=outputType,
                alwaysRaises=alwaysRaises
            )
        )

        if callTarget is None:
            self.pushException(TypeError, "Expression %s was not convertible." % name)
            return

        return self.call_typed_call_target(callTarget, args)

    def convert_expression_ast(self, ast):
        """Convert a python_ast.Expression node to a TypedExpression.

        Returns:
            None if the expression doesn't return control flow to the caller
            or a TypedExpression.
        """
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
            # compute the resulting bool-op type by evaluating each subexpression
            # this is a little wasteful because we hit the expression
            # once here to find out the subtype and then again to actually render it
            # below.
            resultTypes = []
            for argIx, e in enumerate(ast.values):
                with self.subcontext():
                    resValue = self.convert_expression_ast(e)

                    if resValue is not None:
                        # check whether converting to bool definitely causes an exception
                        # if so, we can't produce this output type.
                        if resValue.toBool() is None:
                            resValue = None

                if resValue is None:
                    if argIx == 0:
                        # we always throw. we swallowed the value with the subcontext
                        # so we need to actually produce the code here.
                        return self.convert_expression_ast(e)
                    else:
                        # we throw on this subexpression. this means the whole
                        # expression may throw but may also return because the
                        # first expression could possibly return. We exit the loop
                        # because no new types will be produced, but will still
                        # need to evaluate each subexpression and try writing
                        # it into the output.
                        break
                else:
                    # otherwise, there's a pathway that returns this type and
                    # we need to be able to accept it.
                    resultTypes.append(resValue.expr_type.typeRepresentation)

            # this is the merge of all the possible types that could
            # come out of the subexpression.
            return_type = mergeTypeWrappers(resultTypes)

            # allocate a new slot for the final value. every pathway
            # will either write into it and intialize it, or throw an
            # exception. we are guaranteed that if control flow returns
            # then the value is initialized.
            result = self.allocateUninitializedSlot(return_type)

            # define a function to recursively walk through the expressions in our chain.
            # calling it pushes code that evaluates ast.values[depth:] in the current
            # context.
            assert ast.values

            def convertBoolOp(depth=0):
                value = self.convert_expression_ast(ast.values[depth])

                if value is None:
                    # evaluating this particular subexpression produced an exception.
                    # we can stop evaluating.
                    return

                if depth == len(ast.values) - 1:
                    # this is the last node in the expression, and so therefore
                    # the value of the slot in this particular pathway.
                    value.convert_to_type_with_target(result, ConversionLevel.Signature)
                    self.markUninitializedSlotInitialized(result)
                else:
                    bool_value = TypedExpression.asBool(value)

                    if bool_value is None:
                        return

                    if ast.op.matches.And:
                        with self.ifelse(bool_value) as (ifTrue, ifFalse):
                            with ifTrue:
                                convertBoolOp(depth + 1)
                            with ifFalse:
                                value.convert_to_type_with_target(result, ConversionLevel.Signature)
                                self.markUninitializedSlotInitialized(result)
                    elif ast.op.matches.Or:
                        with self.ifelse(bool_value) as (ifTrue, ifFalse):
                            with ifTrue:
                                value.convert_to_type_with_target(result, ConversionLevel.Signature)
                                self.markUninitializedSlotInitialized(result)
                            with ifFalse:
                                convertBoolOp(depth + 1)

            convertBoolOp()

            return result

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
            val = self.convert_expression_ast(ast.value)
            if val is None:
                return None

            if ast.slice.matches.Slice:
                if ast.slice.lower is None:
                    lower = None
                else:
                    lower = self.convert_expression_ast(ast.slice.lower)
                    if lower is None:
                        return None

                if ast.slice.upper is None:
                    upper = None
                else:
                    upper = self.convert_expression_ast(ast.slice.upper)
                    if upper is None:
                        return None

                if ast.slice.step is None:
                    step = None
                else:
                    step = self.convert_expression_ast(ast.slice.step)
                    if step is None:
                        return None

                return val.convert_getslice(lower, upper, step)
            elif ast.slice.matches.Tuple:
                args = []
                for dim in ast.slice.elts:
                    if dim.matches.Slice:
                        if dim.lower is None:
                            lower = self.constant(None)
                        else:
                            lower = self.convert_expression_ast(dim.lower)
                            if lower is None:
                                return None

                        if dim.upper is None:
                            upper = self.constant(None)
                        else:
                            upper = self.convert_expression_ast(dim.upper)
                            if upper is None:
                                return None

                        if dim.step is None:
                            step = self.constant(None)
                        else:
                            step = self.convert_expression_ast(dim.step)
                            if step is None:
                                return None

                        args.append(
                            SliceWrapper().convert_call(self, None, [lower, upper, step], {})
                        )
                    else:
                        index = self.convert_expression_ast(ast.slice)
                        if index is None:
                            return None

                        args.append(index)

                tupType = Tuple(*[x.expr_type.typeRepresentation for x in args])

                instance = typeWrapper(tupType).createFromArgs(self, args)

                if instance is None:
                    return None

                index = instance.changeType(
                    TypedTupleMasqueradingAsTuple(
                        tupType,
                        interiorTypeWrappers=[x.expr_type for x in args]
                    )
                )

                return val.convert_getitem(index)
            else:
                index = self.convert_expression_ast(ast.slice)
                if index is None:
                    return None

                return val.convert_getitem(index)

        if ast.matches.Call:
            lhs = self.convert_expression_ast(ast.func)

            if lhs is None:
                return None

            if len(ast.args) == 1 and len(ast.keywords) == 0:
                arg = ast.args[0]

                if (
                    arg.matches.List
                    or arg.matches.Tuple
                    or arg.matches.Dict
                    or arg.matches.Set
                    or arg.matches.ListComp
                    or arg.matches.GeneratorExp
                    or arg.matches.DictComp
                    or arg.matches.SetComp
                ):
                    return lhs.expr_type.convert_call_on_container_expression(self, lhs, arg)

            args = []
            kwargs = {}

            for a in ast.args:
                if a.matches.Starred:
                    toStar = self.convert_expression_ast(a.value)

                    if toStar is None:
                        return None

                    fixedExpressions = toStar.get_iteration_expressions()

                    if fixedExpressions is None:
                        # we don't know how to iterate over these at the compiler level.
                        # really we should be passing this entire call to the interpreter
                        # now and seeing if it can handle it.
                        raise Exception(
                            f"Don't know how to *args call with argument of type "
                            f"{toStar.expr_type.typeRepresentation} yet"
                        )

                    args.extend(fixedExpressions)
                else:
                    args.append(self.convert_expression_ast(a))

                    if args[-1] is None:
                        return None

            for keywordArg in ast.keywords:
                argname = keywordArg.arg

                if argname is None:
                    # this is a **kwarg
                    toStarKwarg = self.convert_expression_ast(keywordArg.value)
                    if toStarKwarg is None:
                        return None

                    if not isinstance(toStarKwarg.expr_type, NamedTupleMasqueradingAsDict):
                        raise Exception(
                            f"Don't know how to **args call with argument of type "
                            f"{toStarKwarg.expr_type} yet"
                        )

                    # unwrap this to a NamedTuple
                    toStarKwarg = toStarKwarg.convert_masquerade_to_typed()

                    for ix, name in enumerate(toStarKwarg.expr_type.typeRepresentation.ElementNames):
                        kwargs[name] = toStarKwarg.expr_type.refAs(self, toStarKwarg, ix)
                else:
                    kwargs[argname] = self.convert_expression_ast(keywordArg.value)
                    if kwargs[argname] is None:
                        return None

            return lhs.convert_call(args, kwargs)

        if ast.matches.Compare:
            if len(ast.ops) == 1:
                lhs = self.convert_expression_ast(ast.left)

                if lhs is None:
                    return None

                r = self.convert_expression_ast(ast.comparators[0])

                if r is None:
                    return None

                return lhs.convert_bin_op(ast.ops[0], r)
            else:
                result = self.allocateUninitializedSlot(bool)
                result.convert_copy_initialize(self.constant(True))
                self.markUninitializedSlotInitialized(result)
                left = self.convert_expression_ast(ast.left)
                if left is None:
                    return None
                for i in range(len(ast.comparators)):
                    with self.ifelse(result) as (ifTrue, ifFalse):
                        with ifTrue:
                            t = self.convert_expression_ast(ast.comparators[i])
                            if t is None:
                                return None
                            right = self.allocateUninitializedSlot(t.expr_type.typeRepresentation)
                            right.convert_copy_initialize(t)
                            self.markUninitializedSlotInitialized(right)
                            cond = left.convert_bin_op(ast.ops[i], right)
                            if cond is None:
                                return None
                            result.convert_copy_initialize(cond)
                            left = right

                return result

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

                if true_res is None and false_res is None:
                    return None

                if true_res is None:
                    out_type = false_res.expr_type
                elif false_res is None:
                    out_type = true_res.expr_type
                elif true_res.expr_type != false_res.expr_type:
                    out_type = mergeTypeWrappers([true_res.expr_type, false_res.expr_type])
                else:
                    out_type = true_res.expr_type

                out_slot = self.allocateUninitializedSlot(out_type)

                if true_res is not None:
                    with true_block:
                        true_res = true_res.convert_to_type(out_type, ConversionLevel.Signature)
                        out_slot.convert_copy_initialize(true_res)
                        self.markUninitializedSlotInitialized(out_slot)

                if false_res is not None:
                    with false_block:
                        false_res = false_res.convert_to_type(out_type, ConversionLevel.Signature)
                        out_slot.convert_copy_initialize(false_res)
                        self.markUninitializedSlotInitialized(out_slot)

            return out_slot

        if ast.matches.FormattedValue:
            value = self.convert_expression_ast(ast.value)

            if value is None:
                return None

            if ast.format_spec is not None:
                format_spec = self.convert_expression_ast(ast.format_spec)
                if format_spec is None:
                    return None
            else:
                format_spec = None

            result = value.convert_format(format_spec)

            if result is None:
                return

            if result.expr_type.typeRepresentation is not str:
                self.pushException(TypeError, "Expected string, but got %s" % result.expr_type.typeRepresentation)
                return None

            return result

        if ast.matches.JoinedStr:
            converted = [self.convert_expression_ast(v) for v in ast.values]

            for v in converted:
                if v is None:
                    return None

            if all(v.isConstant and isinstance(v.constantValue, str) for v in converted):
                return self.constant("".join(v.constantValue for v in converted))

            items_to_join = self.push(ListOf(str), lambda s: s.convert_default_initialize())

            for v in converted:
                if items_to_join.convert_method_call("append", (v,), {}) is None:
                    return None

            return pythonObjectRepresentation(self, "").convert_method_call("join", (items_to_join,), {})

        if ast.matches.List:
            aList = self.constant(list).convert_call([], {})

            for e in ast.elts:
                eVal = self.convert_expression_ast(e)
                if eVal is None:
                    return None

                aList.convert_method_call("append", (eVal,), {})

            return aList

        if ast.matches.Tuple:
            # return a masquerading Tuple object
            tupArgs = []

            for e in ast.elts:
                tupArgs.append(self.convert_expression_ast(e))
                if tupArgs[-1] is None:
                    return None

            tupType = Tuple(*[x.expr_type.typeRepresentation for x in tupArgs])

            instance = typeWrapper(tupType).createFromArgs(self, tupArgs)

            if instance is None:
                return None

            return instance.changeType(
                TypedTupleMasqueradingAsTuple(tupType)
            )

        if ast.matches.Set:
            aSet = self.constant(set).convert_call([], {})

            for e in ast.elts:
                eVal = self.convert_expression_ast(e)
                if eVal is None:
                    return None

                aSet.convert_method_call("add", (eVal,), {})

            return aSet

        if ast.matches.Dict:
            aList = self.constant(dict).convert_call([], {})

            for keyExpr, valExpr in zip(ast.keys, ast.values):
                keyVal = self.convert_expression_ast(keyExpr)
                if keyVal is None:
                    return None

                valVal = self.convert_expression_ast(valExpr)
                if valVal is None:
                    return None

                aList.convert_method_call("__setitem__", (keyVal, valVal), {})

            return aList

        if ast.matches.Yield:
            raise Exception("Yield should be handled at the statement level")

        if ast.matches.Lambda:
            return self.functionContext.localVariableExpression(self, ".closure").changeType(
                self.functionContext.functionDefToType[ast]
            )

        if ast.matches.ListComp:
            return self.convert_generator_as_list_comprehension(ast)

        if ast.matches.SetComp:
            return self.convert_generator_as_set_comprehension(ast)

        if ast.matches.DictComp:
            return self.convert_generator_as_dict_comprehension(ast)

        if ast.matches.GeneratorExp:
            return self.functionContext.localVariableExpression(self, ".closure").changeType(
                self.functionContext.functionDefToType[ast]
            ).convert_call([], {})

        if ast.matches.Constant:
            if ast.value is isinstance:
                return pythonObjectRepresentation(self, ast.value)

            return self.constant(ast.value, allowArbitrary=True)

        raise ConversionException("can't handle python expression type %s" % ast.Name)

    def convert_generator_as_list_comprehension(self, ast):
        generatorFunc = self.functionContext.localVariableExpression(self, ".closure").changeType(
            self.functionContext.functionDefToType[ast]
        )
        return generatorFunc.expr_type.convert_list_comprehension(
            generatorFunc.context,
            generatorFunc
        )

    def convert_generator_as_set_comprehension(self, ast):
        generatorFunc = self.functionContext.localVariableExpression(self, ".closure").changeType(
            self.functionContext.functionDefToType[ast]
        )
        return generatorFunc.expr_type.convert_set_comprehension(
            generatorFunc.context,
            generatorFunc
        )

    def convert_generator_as_dict_comprehension(self, ast):
        generatorFunc = self.functionContext.localVariableExpression(self, ".closure").changeType(
            self.functionContext.functionDefToType[ast]
        )
        return generatorFunc.expr_type.convert_dict_comprehension(
            generatorFunc.context,
            generatorFunc
        )
