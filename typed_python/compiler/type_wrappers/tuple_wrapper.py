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

from typed_python.compiler.type_wrappers.wrapper import Wrapper

from typed_python import _types, Int32, OneOf, Bool

from typed_python.compiler.type_wrappers.bound_compiled_method_wrapper import BoundCompiledMethodWrapper
import typed_python.compiler.native_ast as native_ast
import typed_python.compiler

typeWrapper = lambda t: typed_python.compiler.python_object_representation.typedPythonTypeToTypeWrapper(t)


class TupleWrapper(Wrapper):
    is_empty = False
    is_pass_by_ref = True

    def __init__(self, t):
        assert hasattr(t, '__typed_python_category__')
        super().__init__(t)
        bytecount = _types.bytecount(t)

        self.subTypeWrappers = tuple(typeWrapper(sub_t) for sub_t in t.ElementTypes)
        self.unionType = OneOf(*tuple(t.ElementTypes))
        self.byteOffsets = [0]

        for i in range(len(self.subTypeWrappers)-1):
            self.byteOffsets.append(self.byteOffsets[-1] + _types.bytecount(t.ElementTypes[i]))

        self.layoutType = native_ast.Type.Array(element_type=native_ast.UInt8, count=bytecount)

        self._is_pod = all(typeWrapper(possibility).is_pod for possibility in self.subTypeWrappers)
        self.is_default_constructible = _types.is_default_constructible(t)

    def convert_hash(self, context, expr):
        val = context.constant(Int32(0))
        for i in range(len(self.subTypeWrappers)):
            subHash = self.refAs(context, expr, i).convert_hash()
            if subHash is None:
                return None
            val = (val * context.constant(Int32(1000003))) ^ subHash
        return val

    @property
    def is_pod(self):
        return self._is_pod

    def getNativeLayoutType(self):
        return self.layoutType

    def convert_initialize_from_args(self, context, target, *args):
        assert len(args) == len(self.byteOffsets)
        for i in range(len(args)):
            self.refAs(context, target, i).convert_copy_initialize(args[i])

    def convert_default_initialize(self, context, target):
        if not self.is_default_constructible:
            context.pushException(TypeError, "Can't default-initialize any subtypes of %s" % self.typeRepresentation.__qualname__)
            return

        for i, t in enumerate(self.typeRepresentation.ElementTypes):
            if _types.is_default_constructible(t):
                self.refAs(context, target, i).convert_default_initialize()

    def refAs(self, context, expr, which):
        return context.pushReference(
            self.subTypeWrappers[which],
            expr.expr.cast(native_ast.UInt8Ptr)
                .ElementPtrIntegers(self.byteOffsets[which])
                .cast(self.subTypeWrappers[which].getNativeLayoutType().pointer())
        )

    def convert_len(self, context, instance):
        return context.constant(len(self.subTypeWrappers))

    def convert_bool_cast(self, context, e):
        return context.constant(len(self.subTypeWrappers) != 0)

    def convert_getitem(self, context, expr, index):
        index = index.convert_to_type(int)
        if index is None:
            return None

        # if the argument is a constant, we can be very precise about what type
        # we're going to get out of the indexing operation
        if index.expr.matches.Constant:
            if index.expr.val.matches.Int:
                indexVal = index.expr.val.val

                if indexVal >= - len(self.subTypeWrappers) and indexVal < len(self.subTypeWrappers):
                    if indexVal < 0:
                        indexVal += len(self.subTypeWrappers)

                    return self.refAs(context, expr, indexVal)

        index = index.convert_to_type(int)
        if index is None:
            return None

        result = context.allocateUninitializedSlot(self.unionType)
        with context.switch(
            index.nonref_expr,
            range(len(self.subTypeWrappers)),
            True
        ) as indicesAndContexts:
            for i, subcontext in indicesAndContexts:
                with subcontext:
                    if i is not None:
                        converted = self.refAs(context, expr, i).convert_to_type(self.unionType)
                        if converted is not None:
                            result.convert_copy_initialize(converted)
                            context.markUninitializedSlotInitialized(result)
                    else:
                        context.pushException(IndexError, f"{i} not in [0, {len(self.subTypeWrappers)})")

        return result

    def convert_assign(self, context, expr, other):
        assert expr.isReference

        for i in range(len(self.subTypeWrappers)):
            self.refAs(context, expr, i).convert_assign(self.refAs(context, other, i))

    def convert_copy_initialize(self, context, expr, other):
        for i in range(len(self.subTypeWrappers)):
            self.refAs(context, expr, i).convert_copy_initialize(self.refAs(context, other, i))

    def convert_destroy(self, context, expr):
        if not self.is_pod:
            for i in range(len(self.subTypeWrappers)):
                self.refAs(context, expr, i).convert_destroy()

    def get_iteration_expressions(self, context, expr):
        return [self.refAs(context, expr, i) for i in range(len(self.subTypeWrappers))]

    def convert_type_call(self, context, typeInst, args, kwargs):
        context.pushException(TypeError, f"Can't initialize {self.typeRepresentation} with this signature")
        return

    def convert_type_call_on_container_expression(self, context, typeInst, argExpr):
        if not (argExpr.matches.Tuple or argExpr.matches.List):
            return super().convert_type_call_on_container_expression(context, typeInst, argExpr)

        if len(self.typeRepresentation.ElementTypes) != len(argExpr.elts):
            context.pushException(TypeError, f"Wrong number of arguments to construct '{self.typeRepresentation}'")
            return

        args = []
        for tupArg in argExpr.elts:
            convertedArg = context.convert_expression_ast(tupArg)
            if convertedArg is None:
                return None
            args.append(convertedArg)

        return self.createFromArgs(context, args)

    def createFromArgs(self, context, args):
        """Initialize a new tuple of this type from a set of arguments.

        This will attempt to convert the tuple.
        """
        typeConvertedArgs = []
        for i in range(len(args)):
            typeConvertedArg = args[i].convert_to_type(self.typeRepresentation.ElementTypes[i])
            if typeConvertedArg is None:
                return None
            typeConvertedArgs.append(typeConvertedArg)

        uninitializedTuple = context.allocateUninitializedSlot(self)

        for i in range(len(args)):
            uninitializedChildElement = self.refAs(context, uninitializedTuple, i)
            uninitializedChildElement.convert_copy_initialize(typeConvertedArgs[i])

        context.markUninitializedSlotInitialized(uninitializedTuple)

        # the tuple is now initialized
        return uninitializedTuple

    def convert_to_type_with_target(self, context, e, targetVal, explicit):
        if not explicit:
            return super().convert_to_type_with_target(context, e, targetVal, explicit)

        target_type = targetVal.expr_type

        if target_type.typeRepresentation == Bool:
            context.pushEffect(
                targetVal.expr.store(
                    context.constant(len(self.subTypeWrappers) != 0)
                )
            )
            return context.constant(True)

        return super().convert_to_type_with_target(context, e, targetVal, explicit)


class NamedTupleWrapper(TupleWrapper):
    def __init__(self, t):
        super().__init__(t)

        self.namesToIndices = {n: i for i, n in enumerate(t.ElementNames)}
        self.namesToTypes = {n: t.ElementTypes[i] for i, n in enumerate(t.ElementNames)}

    def convert_attribute(self, context, instance, attribute):
        if attribute in ["replacing"]:
            return instance.changeType(BoundCompiledMethodWrapper(self, attribute))

        ix = self.namesToIndices.get(attribute)
        if ix is None:
            context.pushException(AttributeError, "'%s' object has no attribute '%s'" % (str(self.typeRepresentation), attribute))
            return

        return self.refAs(context, instance, ix)

    def convert_type_call(self, context, typeInst, args, kwargs):
        if len(args) == 0:
            for name in kwargs:
                if name not in self.namesToTypes:
                    context.pushException(TypeError, f"Couldn't initialize type of {self} with an argument named {name}")
                    return

            needsDefaultInitializer = set()

            for name, argType in self.namesToTypes.items():
                if name not in kwargs:
                    if _types.is_default_constructible(name):
                        needsDefaultInitializer.add(name)
                    else:
                        context.pushException(TypeError, f"Can't default initialize member {name} of {self}")
                        return

            uninitializedNamedTuple = context.allocateUninitializedSlot(self)

            for name, expr in kwargs.items():
                actualExpr = expr.convert_to_type(self.namesToTypes[name])
                if actualExpr is None:
                    return None

                uninitializedChildElement = self.refAs(context, uninitializedNamedTuple, self.namesToIndices[name])
                uninitializedChildElement.convert_copy_initialize(actualExpr)

            for name in needsDefaultInitializer:
                self.refAs(context, uninitializedNamedTuple, self.namesToIndices[name]).convert_default_initialize()

            context.markUninitializedSlotInitialized(uninitializedNamedTuple)

            # the tuple is now initialized
            return uninitializedNamedTuple

        if len(args) == 1 and not kwargs:
            return args[0].convert_to_type(self, True)

        return super().convert_type_call(context, typeInst, args, kwargs)

    def convert_method_call(self, context, instance, methodname, args, kwargs):
        if methodname == 'replacing' and not args:
            return context.push(self, lambda newInstance: self.initializeReplacing(context, newInstance, instance, kwargs))

        return super().convert_method_call(context, instance, methodname, args, kwargs)

    def initializeReplacing(self, context, toInitialize, existingInstance, kwargs):
        # check if all the passed arguments are in the list of the names
        additional_arguments = sorted(list(set(kwargs.keys()) - set(self.typeRepresentation.ElementNames)))
        if additional_arguments:
            context.pushException(
                ValueError,
                "The arguments list contain names '{}' which are not in the tuple definition."
                .format(", ".join(additional_arguments))
            )
            return None

        for i in range(len(self.subTypeWrappers)):
            field_name = self.typeRepresentation.ElementNames[i]
            field_type = self.typeRepresentation.ElementTypes[i]

            if field_name not in kwargs:
                self.refAs(context, toInitialize, i).convert_copy_initialize(self.refAs(context, existingInstance, i))
            else:
                converted = kwargs[field_name].convert_to_type(field_type)
                if converted is None:
                    return None
                self.refAs(context, toInitialize, i).convert_copy_initialize(converted)
