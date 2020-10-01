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
from typed_python import NamedTuple
from typed_python.compiler.type_wrappers.wrapper import Wrapper
from typed_python.compiler.type_wrappers.python_type_object_wrapper import PythonTypeObjectWrapper
from typed_python.compiler.type_wrappers.named_tuple_masquerading_as_slice import NamedTupleMasqueradingAsSlice

typeWrapper = lambda t: typed_python.compiler.python_object_representation.typedPythonTypeToTypeWrapper(t)


class SliceWrapper(PythonTypeObjectWrapper):
    def __init__(self):
        super().__init__(slice)

    def __repr__(self):
        return "SliceWrapper()"

    def __str__(self):
        return "SliceWrapper()"

    @Wrapper.unwrapOneOfAndValue
    def convert_call(self, context, left, args, kwargs):
        if kwargs:
            return context.pushException(TypeError, "slice() does not take keyword arguments")

        if len(args) > 3:
            return context.pushException(TypeError, f"slice() expected at most 3 arguments, got {len(args)}")

        if len(args) == 0:
            return context.pushException(TypeError, "slice() expected at least 1 argument, got 0")

        if len(args) == 1:
            args = [context.constant(None), args[0], context.constant(None)]

        while len(args) < 3:
            args = tuple(args) + (context.constant(None),)

        ntType = NamedTuple(
            start=args[0].expr_type.typeRepresentation,
            stop=args[1].expr_type.typeRepresentation,
            step=args[2].expr_type.typeRepresentation
        )

        return typeWrapper(ntType).convert_type_call(
            context,
            None,
            [],
            dict(start=args[0], stop=args[1], step=args[2])
        ).changeType(
            NamedTupleMasqueradingAsSlice(ntType)
        )
