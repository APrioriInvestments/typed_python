#   Copyright 2020 typed_python Authors
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

from typed_python._types import Tuple, NamedTuple
from typed_python.compiler.type_wrappers.compilable_builtin import CompilableBuiltin
import typed_python.compiler

typeWrapper = lambda t: typed_python.compiler.python_object_representation.typedPythonTypeToTypeWrapper(t)


class Reduce(CompilableBuiltin):
    def __eq__(self, other):
        return isinstance(other, Reduce)

    def __hash__(self):
        return hash("Reduce")

    def __call__(self, reducer, tup):
        if isinstance(tup, (Tuple, NamedTuple)):
            return self.reduceTuple(reducer, tup)

        raise TypeError("Currently, 'reduce' only works on typed python Tuple or NamedTuple objects")

    def reduceTuple(self, reducer, tup):
        isFirst = True
        result = None

        for elt in tup:
            if isFirst:
                result = reducer(elt)
                isFirst = False
            else:
                result = reducer(result, elt)

        return result

    def convert_call(self, context, expr, args, kwargs):
        if len(args) != 2 or len(kwargs):
            context.pushException(TypeError, "reduce takes two positional arguments: 'f' and 'iterable'")
            return

        argT = args[1].expr_type.typeRepresentation

        if isinstance(argT, type) and issubclass(argT, (NamedTuple, Tuple)):
            return self.convert_call_on_tuple(context, args[0], args[1])

        context.pushException(TypeError, "Currently, 'reduce' only works on typed python Tuple or NamedTuple objects")

    def convert_call_on_tuple(self, context, fArg, tupArg):
        outputExpr = None

        argT = tupArg.expr_type.typeRepresentation

        for i in range(len(argT.ElementTypes)):
            if i == 0:
                outputExpr = fArg.convert_call((tupArg.refAs(i),), {})
            else:
                outputExpr = fArg.convert_call((outputExpr, tupArg.refAs(i)), {})

            if outputExpr is None:
                return None

        return outputExpr


reduce = Reduce()
