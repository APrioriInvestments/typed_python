#   Copyright 2020-2020 typed_python Authors
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

from typed_python.compiler.type_wrappers.wrapper import Wrapper
from typed_python.compiler.merge_type_wrappers import mergeTypes
from typed_python.compiler.conversion_level import ConversionLevel
import typed_python.compiler.native_ast as native_ast
import typed_python.python_ast as python_ast
import typed_python.compiler

typeWrapper = lambda t: typed_python.compiler.python_object_representation.typedPythonTypeToTypeWrapper(t)


# at or below this threshold, use code space O(n^2) and just one copy
# above this threshold, use code space O(n) but also O(n) copies
_MINMAX_SPACE_VS_COPY_THRESHOLD = 10


def i_min(v):
    first = 1
    for e in v:
        if first or e < ret:  # noqa: F821
            first = 0
            ret = e
    if first:
        raise ValueError("min() arg is an empty sequence")
    return ret


def i_min_default(v, default):
    first = 1
    for e in v:
        if first or e < ret:  # noqa: F821
            first = 0
            ret = e
    if first:
        return default
    return ret


def i_min_key(v, key_f):
    first = 1
    for e in v:
        elt_key = key_f(e)
        if first or elt_key < ret_key:  # noqa: F821
            first = 0
            ret_key = elt_key  # noqa: F841
            ret = e
    if first:
        raise ValueError("min() arg is an empty sequence")
    return ret


def i_min_key_default(v, key_f, default):
    first = 1
    for e in v:
        elt_key = key_f(e)
        if first or elt_key < ret_key:  # noqa: F821
            first = 0
            ret_key = elt_key  # noqa: F841
            ret = e
    if first:
        return default
    return ret


def i_max(v):
    first = 1
    for e in v:
        if first or e > ret:  # noqa: F821
            first = 0
            ret = e
    if first:
        raise ValueError("max() arg is an empty sequence")
    return ret


def i_max_default(v, default):
    first = 1
    for e in v:
        if first or e > ret:  # noqa: F821
            first = 0
            ret = e
    if first:
        return default
    return ret


def i_max_key(v, key_f):
    first = 1
    for e in v:
        elt_key = key_f(e)
        if first or elt_key > ret_key:  # noqa: F821
            first = 0
            ret_key = elt_key  # noqa: F841
            ret = e
    if first:
        raise ValueError("max() arg is an empty sequence")
    return ret


def i_max_key_default(v, key_f, default):
    first = 1
    for e in v:
        elt_key = key_f(e)
        if first or elt_key > ret_key:  # noqa: F821
            first = 0
            ret_key = elt_key  # noqa: F841
            ret = e
    if first:
        return default
    return ret


def no_more_kwargs(context, **kwargs):
    for e in kwargs:
        context.pushException(TypeError, f"'{e}' is an invalid keyword argument for this function")
        # just need to generate the first exception
        break


class MinMaxWrapper(Wrapper):
    is_pod = True
    is_empty = False
    is_pass_by_ref = False

    def __init__(self, comparison1, comparison2, i_call, i_default_call, i_key_call, i_key_default_call, builtin):
        self.comparison_op1 = comparison1
        self.comparison_op2 = comparison2
        self.i_call = i_call
        self.i_default_call = i_default_call
        self.i_key_call = i_key_call
        self.i_key_default_call = i_key_default_call
        super().__init__(builtin)

    def getNativeLayoutType(self):
        return native_ast.Type.Void()

    def convert_call(self, context, expr, args, kwargs0):
        kwargs = kwargs0.copy()
        # iterable case
        if len(args) == 1:
            if 'key' in kwargs:
                key_f = kwargs['key']
                del kwargs['key']
                if 'default' in kwargs:
                    default = kwargs['default']
                    del kwargs['default']
                    no_more_kwargs(context, **kwargs)
                    return context.call_py_function(self.i_key_default_call, (args[0], key_f, default), {})
                else:
                    no_more_kwargs(context, **kwargs)
                    return context.call_py_function(self.i_key_call, (args[0], key_f), {})
            if 'default' in kwargs:
                default = kwargs['default']
                del kwargs['default']
                no_more_kwargs(context, **kwargs)
                return context.call_py_function(self.i_default_call, (args[0], default), {})
            no_more_kwargs(context, **kwargs)
            return context.call_py_function(self.i_call, (args[0],), {})

        if len(args) >= 2 and 'key' in kwargs:
            outT = mergeTypes([a.expr_type.typeRepresentation for a in args])

            selected = context.allocateUninitializedSlot(outT)
            selected.convert_copy_initialize(args[0].convert_to_type(outT, ConversionLevel.Signature))
            context.markUninitializedSlotInitialized(selected)

            key_f = kwargs['key']
            del kwargs['key']
            no_more_kwargs(context, **kwargs)

            # determine key type
            keyT = None
            for i in range(len(args)):
                k = key_f.expr_type.convert_call(context, None, (args[i],), {})
                if k is None:
                    return None
                if keyT is None:
                    keyT = k.expr_type.typeRepresentation
                else:
                    keyT = mergeTypes([keyT, k.expr_type.typeRepresentation])

            # evaluate key for each arg
            keys = []
            for i in range(len(args)):
                keys.append(context.allocateUninitializedSlot(keyT))
                keys[i].convert_copy_initialize(key_f.expr_type.convert_call(context, None, (args[i],), {}))
                context.markUninitializedSlotInitialized(keys[i])

            selected_key = context.allocateUninitializedSlot(keyT)
            selected_key.convert_copy_initialize(keys[0].convert_to_type(keyT, ConversionLevel.Signature))
            context.markUninitializedSlotInitialized(selected_key)
            for i in range(1, len(args)):
                cond = typeWrapper(keyT).convert_bin_op(context, selected_key, self.comparison_op1, keys[i], False)
                if cond is None:
                    return None
                cond = cond.toBool()
                if cond is None:
                    return None
                with context.ifelse(cond.nonref_expr) as (ifTrue, ifFalse):
                    with ifTrue:
                        selected.convert_copy_initialize(args[i].convert_to_type(outT, ConversionLevel.Signature))
                        selected_key.convert_copy_initialize(keys[i].convert_to_type(keyT, ConversionLevel.Signature))

            return selected

        if len(args) >= 2 and not kwargs:
            outT = mergeTypes([a.expr_type.typeRepresentation for a in args])
            selected = context.allocateUninitializedSlot(outT)

            for i in range(len(args)):
                convertedArg = args[i].convert_to_type(outT, ConversionLevel.Signature)
                if convertedArg is None:
                    return None

                if i == 0:
                    selected.convert_copy_initialize(convertedArg)
                    context.markUninitializedSlotInitialized(selected)
                else:
                    cond = typeWrapper(outT).convert_bin_op(context, selected, self.comparison_op1, convertedArg, False)
                    if cond is None:
                        return None

                    cond = cond.toBool()
                    if cond is None:
                        return None

                    with context.ifelse(cond.nonref_expr) as (ifTrue, ifFalse):
                        with ifTrue:
                            selected.convert_assign(convertedArg)

            return selected

        return super().convert_call(context, expr, args, kwargs)


class MinWrapper(MinMaxWrapper):
    def __init__(self):
        super().__init__(
            python_ast.ComparisonOp.Gt(),
            python_ast.ComparisonOp.GtE(),
            i_min,
            i_min_default,
            i_min_key,
            i_min_key_default,
            min
        )


class MaxWrapper(MinMaxWrapper):
    def __init__(self):
        super().__init__(
            python_ast.ComparisonOp.Lt(),
            python_ast.ComparisonOp.LtE(),
            i_max,
            i_max_default,
            i_max_key,
            i_max_key_default,
            max
        )
