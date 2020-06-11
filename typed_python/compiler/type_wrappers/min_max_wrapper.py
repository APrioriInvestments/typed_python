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
from typed_python.compiler.type_wrappers.one_of_wrapper import OneOfWrapper
import typed_python.compiler.native_ast as native_ast
import typed_python.python_ast as python_ast
import typed_python.compiler


typeWrapper = lambda t: typed_python.compiler.python_object_representation.typedPythonTypeToTypeWrapper(t)


# at or below this threshold, use code space O(n^2) and just one copy
# above this threshold, use code space O(n) but also O(n) copies
_MINMAX_SPACE_VS_COPY_THRESHOLD = 10


class MinWrapper(Wrapper):
    is_pod = True
    is_empty = False
    is_pass_by_ref = False

    def __init__(self):
        super().__init__(min)

    def getNativeLayoutType(self):
        return native_ast.Type.Void()

    def convert_call(self, context, expr, args, kwargs):
        # this algorithm generates O(n) code and does O(n) copies
        if len(args) > _MINMAX_SPACE_VS_COPY_THRESHOLD and not kwargs:
            outT = OneOfWrapper.mergeTypes([a.expr_type.typeRepresentation for a in args]).typeRepresentation
            selected = context.allocateUninitializedSlot(outT)
            selected.convert_copy_initialize(args[0].convert_to_type(outT))
            context.markUninitializedSlotInitialized(selected)

            for i in range(1, len(args)):
                cond = typeWrapper(outT).convert_bin_op(context, selected, python_ast.ComparisonOp.Gt(), args[i], False)
                if cond is None:
                    return None
                with context.ifelse(cond.nonref_expr) as (ifTrue, ifFalse):
                    with ifTrue:
                        selected.convert_copy_initialize(args[i].convert_to_type(outT))

            return selected

        # this algorithm generates O(n^2) code and does only one copy
        if len(args) >= 2 and not kwargs:
            outT = OneOfWrapper.mergeTypes([a.expr_type.typeRepresentation for a in args]).typeRepresentation
            selected = context.allocateUninitializedSlot(outT)

            # returns False to abort this attempt
            def comparison_tree(j, k):
                cond = args[j].expr_type.convert_bin_op(context, args[j], python_ast.ComparisonOp.LtE(), args[k], False)
                if cond is None:
                    return False
                with context.ifelse(cond.nonref_expr) as (ifTrue, ifFalse):
                    if j + 1 == k:
                        with ifTrue:
                            selected.convert_copy_initialize(args[j].convert_to_type(outT))
                        with ifFalse:
                            selected.convert_copy_initialize(args[k].convert_to_type(outT))
                    else:
                        with ifTrue:
                            if not comparison_tree(j, k-1):
                                return False
                        with ifFalse:
                            if not comparison_tree(j+1, k):
                                return False
                return True

            if not comparison_tree(0, len(args) - 1):
                return None
            context.markUninitializedSlotInitialized(selected)
            return selected

        return super().convert_call(context, expr, args, kwargs)


class MaxWrapper(Wrapper):
    is_pod = True
    is_empty = False
    is_pass_by_ref = False

    def __init__(self):
        super().__init__(max)

    def getNativeLayoutType(self):
        return native_ast.Type.Void()

    def convert_call(self, context, expr, args, kwargs):
        # this algorithm generates O(n) code and does O(n) copies
        if len(args) > _MINMAX_SPACE_VS_COPY_THRESHOLD and not kwargs:
            outT = OneOfWrapper.mergeTypes([a.expr_type.typeRepresentation for a in args]).typeRepresentation
            selected = context.allocateUninitializedSlot(outT)
            selected.convert_copy_initialize(args[0].convert_to_type(outT))
            context.markUninitializedSlotInitialized(selected)

            for i in range(1, len(args)):
                cond = typeWrapper(outT).convert_bin_op(context, selected, python_ast.ComparisonOp.Lt(), args[i], False)
                if cond is None:
                    return None
                with context.ifelse(cond.nonref_expr) as (ifTrue, ifFalse):
                    with ifTrue:
                        selected.convert_copy_initialize(args[i].convert_to_type(outT))

            return selected

        # this algorithm generates O(n^2) code and does only one copy
        if len(args) >= 2 and not kwargs:
            outT = OneOfWrapper.mergeTypes([a.expr_type.typeRepresentation for a in args]).typeRepresentation
            selected = context.allocateUninitializedSlot(outT)

            # returns False to abort this attempt
            def comparison_tree(j, k):
                cond = args[j].expr_type.convert_bin_op(context, args[j], python_ast.ComparisonOp.GtE(), args[k], False)
                if cond is None:
                    return False
                with context.ifelse(cond.nonref_expr) as (ifTrue, ifFalse):
                    if j + 1 == k:
                        with ifTrue:
                            selected.convert_copy_initialize(args[j].convert_to_type(outT))
                        with ifFalse:
                            selected.convert_copy_initialize(args[k].convert_to_type(outT))
                    else:
                        with ifTrue:
                            if not comparison_tree(j, k-1):
                                return False
                        with ifFalse:
                            if not comparison_tree(j+1, k):
                                return False
                return True

            if not comparison_tree(0, len(args) - 1):
                return None
            context.markUninitializedSlotInitialized(selected)
            return selected

        return super().convert_call(context, expr, args, kwargs)
