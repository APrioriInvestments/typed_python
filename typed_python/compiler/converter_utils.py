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

from typed_python.compiler.type_wrappers.compilable_builtin import CompilableBuiltin
from typed_python.compiler.conversion_level import ConversionLevel
from typed_python._types import convertObjectToTypeAtLevel


class ConvertImplicit(CompilableBuiltin):
    def __init__(self, T):
        self.T = T

    def __eq__(self, other):
        return isinstance(other, ConvertImplicit) and self.T == other.T

    def __hash__(self):
        return hash(("ConvertImplicit", self.T))

    def convert_call(self, context, instance, args, kwargs):
        """ConvertImplicit(T)(sourceVal).

        Construct a new instance of type 'T' using 'sourceVal' converted
        using ConversionLevel.Implicit
        """
        if len(args) == 1:
            return args[0].convert_to_type(self.T, ConversionLevel.Implicit)

        return super().convert_call(context, instance, args, kwargs)

    def __call__(self, x):
        return convertObjectToTypeAtLevel(x, self.T, ConversionLevel.Implicit.LEVEL)


class ConvertImplicitContainers(CompilableBuiltin):
    def __init__(self, T):
        self.T = T

    def __eq__(self, other):
        return isinstance(other, ConvertImplicitContainers) and self.T == other.T

    def __hash__(self):
        return hash(("ConvertImplicitContainers", self.T))

    def convert_call(self, context, instance, args, kwargs):
        """ConvertImplicitContainers(T)(sourceVal).

        Construct a new instance of type 'T' using 'sourceVal' converted
        using ConversionLevel.ImplicitContainers
        """
        if len(args) == 1:
            return args[0].convert_to_type(self.T, ConversionLevel.ImplicitContainers)

        return super().convert_call(context, instance, args, kwargs)

    def __call__(self, x):
        return convertObjectToTypeAtLevel(x, self.T, ConversionLevel.ImplicitContainers.LEVEL)


class ConvertUpcastContainers(CompilableBuiltin):
    def __init__(self, T):
        self.T = T

    def __eq__(self, other):
        return isinstance(other, ConvertUpcastContainers) and self.T == other.T

    def __hash__(self):
        return hash(("ConvertImplicit", self.T))

    def convert_call(self, context, instance, args, kwargs):
        """ConvertImplicit(T)(sourceVal).

        Construct a new instance of type 'T' using 'sourceVal' converted
        using ConversionLevel.UpcastContainers
        """
        if len(args) == 1:
            return args[0].convert_to_type(self.T, ConversionLevel.UpcastContainers)

        return super().convert_call(context, instance, args, kwargs)

    def __call__(self, x):
        return convertObjectToTypeAtLevel(x, self.T, ConversionLevel.UpcastContainers.LEVEL)


class InitializeRefAsImplicitContainers(CompilableBuiltin):
    def __eq__(self, other):
        return isinstance(other, InitializeRefAsImplicit)

    def __hash__(self):
        return hash("InitializeRefAsImplicitContainers")

    def convert_call(self, context, instance, args, kwargs):
        """InitializeRefAsImplicit()(target, sourceVal) -> bool.

        Initializes 'target' with the contents of 'sourceVal', returning True on success
        and False on failure.

        'target' must be a reference expression to an uninitialized value.

        We attempt to convert types using 'ImplicitContainers'.
        """
        if len(args) == 2:
            return args[0].expr_type.convert_to_type_with_target(
                context,
                args[0],
                args[1],
                ConversionLevel.ImplicitContainers
            )

        return super().convert_call(context, instance, args, kwargs)


class InitializeRefAsImplicit(CompilableBuiltin):
    def __eq__(self, other):
        return isinstance(other, InitializeRefAsImplicit)

    def __hash__(self):
        return hash("InitializeRefAsImplicit")

    def convert_call(self, context, instance, args, kwargs):
        """InitializeRefAsImplicit()(target, sourceVal) -> bool.

        Initializes 'target' with the contents of 'sourceVal', returning True on success
        and False on failure.

        'target' must be a reference expression to an uninitialized value.

        We attempt to convert types using 'Implicit'.
        """
        if len(args) == 2:
            return args[0].expr_type.convert_to_type_with_target(
                context,
                args[0],
                args[1],
                ConversionLevel.Implicit
            )

        return super().convert_call(context, instance, args, kwargs)


class InitializeRefAsUpcastContainers(CompilableBuiltin):
    def __eq__(self, other):
        return isinstance(other, InitializeRefAsUpcastContainers)

    def __hash__(self):
        return hash("InitializeRefAsUpcastContainers")

    def convert_call(self, context, instance, args, kwargs):
        """InitializeRefAsUpcastContainers()(target, sourceVal) -> bool.

        Initializes 'target' with the contents of 'sourceVal', returning True on success
        and False on failure.

        'target' must be a reference expression to an uninitialized value.

        We attempt to convert types using 'UpcastContainers'
        """
        if len(args) == 2:
            return args[0].expr_type.convert_to_type_with_target(
                context,
                args[0],
                args[1],
                ConversionLevel.UpcastContainers
            )

        return super().convert_call(context, instance, args, kwargs)
