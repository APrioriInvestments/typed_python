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

import types
import logging

from typed_python.hash import Hash

import typed_python.python_ast as python_ast
import typed_python._types as _types
import typed_python.compiler
import typed_python.compiler.native_ast as native_ast
from sortedcontainers import SortedSet
from typed_python.compiler.directed_graph import DirectedGraph
from typed_python.compiler.type_wrappers.wrapper import Wrapper
from typed_python.compiler.type_wrappers.class_wrapper import ClassWrapper
from typed_python.compiler.python_object_representation import typedPythonTypeToTypeWrapper
from typed_python.compiler.function_conversion_context import FunctionConversionContext, FunctionOutput
from typed_python.compiler.native_function_conversion_context import NativeFunctionConversionContext
from typed_python.compiler.type_wrappers.python_typed_function_wrapper import PythonTypedFunctionWrapper
from typed_python.compiler.typed_call_target import TypedCallTarget

typeWrapper = lambda t: typed_python.compiler.python_object_representation.typedPythonTypeToTypeWrapper(t)


VALIDATE_FUNCTION_DEFINITIONS_STABLE = False


class FunctionDependencyGraph:
    def __init__(self):
        self._dependencies = DirectedGraph()

        # the search depth in the dependency to find 'identity'
        # the _first_ time we ever saw it. We prefer to update
        # nodes with higher search depth, so we don't recompute
        # earlier nodes until their children are complete.
        self._identity_levels = {}

        # nodes that need to recompute
        self._dirty_inflight_functions = set()

        # (priority, node) pairs that need to recompute
        self._dirty_inflight_functions_with_order = SortedSet(key=lambda pair: pair[0])

    def dropNode(self, node):
        self._dependencies.dropNode(node, False)
        if node in self._identity_levels:
            del self._identity_levels[node]
        self._dirty_inflight_functions.discard(node)

    def getNextDirtyNode(self):
        while self._dirty_inflight_functions_with_order:
            priority, identity = self._dirty_inflight_functions_with_order.pop()

            if identity in self._dirty_inflight_functions:
                self._dirty_inflight_functions.discard(identity)

                return identity

    def addRoot(self, identity):
        if identity not in self._identity_levels:
            self._identity_levels[identity] = 0
            self.markDirty(identity)

    def addEdge(self, caller, callee):
        if caller not in self._identity_levels:
            raise Exception(f"unknown identity {caller} found in the graph")

        if callee not in self._identity_levels:
            self._identity_levels[callee] = self._identity_levels[caller] + 1

            self.markDirty(callee, isNew=True)

        self._dependencies.addEdge(caller, callee)

    def getNamesDependedOn(self, caller):
        return self._dependencies.outgoing(caller)

    def markDirtyWithLowPriority(self, callee):
        # mark this dirty, but call it back after new functions.
        self._dirty_inflight_functions.add(callee)

        level = self._identity_levels[callee]
        self._dirty_inflight_functions_with_order.add((-1000000 + level, callee))

    def markDirty(self, callee, isNew=False):
        self._dirty_inflight_functions.add(callee)

        if isNew:
            # if its a new node, compute it with higher priority the _higher_ it is in the stack
            # so that we do a depth-first search on the way down
            level = 1000000 - self._identity_levels[callee]
        else:
            level = self._identity_levels[callee]

        self._dirty_inflight_functions_with_order.add((level, callee))

    def functionReturnSignatureChanged(self, identity):
        for caller in self._dependencies.incoming(identity):
            self.markDirty(caller)


class PythonToNativeConverter(object):
    def __init__(self, llvmCompiler, compilerCache):
        object.__init__(self)

        self.llvmCompiler = llvmCompiler
        self.compilerCache = compilerCache

        # if True, then insert additional code to check for undefined behavior.
        self.generateDebugChecks = False

        # all link names for which we have a definition.
        self._allDefinedNames = set()

        self._link_name_for_identity = {}
        self._definitions = {}
        self._targets = {}
        self._inflight_definitions = {}
        self._inflight_function_conversions = {}
        self._identifier_to_pyfunc = {}
        self._times_calculated = {}

        # function names that have been defined but not yet compiled
        self._new_native_functions = set()

        self._linktimeHooks = []
        self._visitors = []

        # the identity of the function we're currently evaluating.
        # we use this to track which functions need to get rebuilt when
        # other functions change types.
        self._currentlyConverting = None

        self._dependencies = FunctionDependencyGraph()

    def addVisitor(self, visitor):
        self._visitors.append(visitor)

    def removeVisitor(self, visitor):
        self._visitors.remove(visitor)

    def identityToName(self, identity):
        """Convert a function identity to the link-time name for the function.

        Args:
            identity - an identity tuple that uniquely identifies the function

        Returns:
            name - the linker name of the native function this represents, or None
                if the identity is unknown
        """
        return self._link_name_for_identity.get(identity)

    def buildAndLinkNewModule(self):
        targets = self.extract_new_function_definitions()

        if not targets:
            return

        if self.compilerCache is None:
            loadedModule = self.llvmCompiler.buildModule(targets)
            loadedModule.linkGlobalVariables()
            return

        # get a set of function names that we depend on
        externallyUsed = set()

        for funcName in targets:
            for dep in self._dependencies.getNamesDependedOn(funcName):
                if dep not in targets:
                    externallyUsed.add(dep)

        binary = self.llvmCompiler.buildSharedObject(targets)

        self.compilerCache.addModule(
            binary,
            {name: self._targets[name] for name in targets if name in self._targets},
            externallyUsed
        )

    def extract_new_function_definitions(self):
        """Return a list of all new function definitions from the last conversion."""
        res = {}

        for u in self._new_native_functions:
            res[u] = self._definitions[u]

        self._new_native_functions = set()

        return res

    def identityHashToLinkerName(self, name, identityHash, prefix="tp."):
        assert isinstance(name, str)
        assert isinstance(identityHash, str)
        assert isinstance(prefix, str)

        return prefix + name + "." + identityHash

    def createConversionContext(self, identity, funcName, funcCode, funcGlobals, funcGlobalsRaw, closureVars, input_types, output_type):
        pyast = self._code_to_ast(funcCode)

        if isinstance(pyast, python_ast.Statement.FunctionDef):
            body = pyast.body
        else:
            body = [python_ast.Statement.Return(
                value=pyast.body,
                line_number=pyast.body.line_number,
                col_offset=pyast.body.col_offset,
                filename=pyast.body.filename
            )]

        return FunctionConversionContext(
            self,
            funcName,
            identity,
            input_types,
            output_type,
            closureVars,
            funcGlobals,
            funcGlobalsRaw,
            pyast.args,
            body,
        )

    def installLinktimeHook(self, identity, callback):
        """Call 'callback' with the native function pointer for 'identity' after compilation has finished."""
        self._linktimeHooks.append((identity, callback))

    def popLinktimeHook(self):
        if self._linktimeHooks:
            return self._linktimeHooks.pop()
        else:
            return None

    def defineNonPythonFunction(self, name, identityTuple, context, callback=None):
        """Define a non-python generating function (if we haven't defined it before already)

            name - the name to actually give the function.
            identityTuple - a unique (sha)hashable tuple
            context - a FunctionConvertsionContext lookalike
            callback - a function taking a function pointer that gets called after codegen
                to allow us to install this function pointer.

        returns a TypedCallTarget, or None if it's not known yet
        """
        identity = self.hashObjectToIdentity(identityTuple).hexdigest
        linkName = self.identityHashToLinkerName(name, identity, "runtime.")

        if self._currentlyConverting is not None:
            self._dependencies.addEdge(self._currentlyConverting, identity)
        else:
            self._dependencies.addRoot(identity)

        if callback is not None:
            self.installLinktimeHook(identity, callback)

        if linkName in self._allDefinedNames:
            return self._targets.get(linkName)

        self._allDefinedNames.add(linkName)
        self._link_name_for_identity[identity] = linkName
        self._inflight_function_conversions[identity] = context

        if context.knownOutputType() is not None or context.alwaysRaises():
            self._targets[linkName] = self.getTypedCallTarget(
                name,
                context.getInputTypes(),
                context.knownOutputType(),
                alwaysRaises=context.alwaysRaises()
            )

        if self._currentlyConverting is None:
            # force the function to resolve immediately
            self._resolveAllInflightFunctions()
            self._installInflightFunctions(name)
            self._inflight_function_conversions.clear()

        return self._targets.get(linkName)

    def defineNativeFunction(self, name, identity, input_types, output_type, generatingFunction, callback=None):
        """Define a native function if we haven't defined it before already.

            name - the name to actually give the function.
            identity - a tuple consisting of strings, ints, type wrappers, and tuples of same
            input_types - list of Wrapper objects for the incoming types
            output_type - Wrapper object for the output type.
            generatingFunction - a function producing a native_function_definition.
                It should accept an expression_conversion_context, an expression for the output
                if it's not pass-by-value (or None if it is), and a bunch of TypedExpressions
                and produce code that always ends in a terminal expression, (or if it's pass by value,
                flows off the end of the function)
            callback - a function taking a function pointer that gets called after codegen
                to allow us to install this function pointer.

        returns a TypedCallTarget. 'generatingFunction' may call this recursively if it wants.
        """
        output_type = typeWrapper(output_type)
        input_types = [typeWrapper(x) for x in input_types]

        identity = (
            Hash.from_integer(2) + self.hashObjectToIdentity(identity)
        ).hexdigest

        return self.defineNonPythonFunction(
            name,
            identity,
            NativeFunctionConversionContext(
                self, input_types, output_type, generatingFunction, identity
            ),
            callback
        )

    def getTypedCallTarget(self, name, input_types, output_type, alwaysRaises=False):
        native_input_types = [a.getNativePassingType() for a in input_types if not a.is_empty]
        if output_type is None:
            native_output_type = native_ast.Type.Void()
        elif output_type.is_pass_by_ref:
            native_input_types = [output_type.getNativePassingType()] + native_input_types
            native_output_type = native_ast.Type.Void()
        else:
            native_output_type = output_type.getNativeLayoutType()

        res = TypedCallTarget(
            native_ast.NamedCallTarget(
                name=name,
                arg_types=native_input_types,
                output_type=native_output_type,
                external=False,
                varargs=False,
                intrinsic=False,
                can_throw=True
            ),
            input_types,
            output_type,
            alwaysRaises=alwaysRaises
        )

        return res

    def _code_to_ast(self, f):
        return python_ast.convertFunctionToAlgebraicPyAst(f)

    def demasqueradeCallTargetOutput(self, callTarget: TypedCallTarget):
        """Ensure we are returning the correct 'interpreterType' from callTarget.

        In some cases, we may return a 'masquerade' type in compiled code. This is fine
        for other compiled code, but the interpreter needs us to transform the result back
        to the right interpreter type. For instance, we may be returning a *args tuple.

        Returns:
            a new TypedCallTarget where the output type has the right return type.
        """
        if callTarget.output_type is None:
            return callTarget

        if callTarget.output_type.interpreterTypeRepresentation == callTarget.output_type.typeRepresentation:
            return callTarget

        def generator(context, out, *args):
            assert out is not None, "we should have an output because no masquerade types are pass-by-value"

            res = context.call_typed_call_target(callTarget, args)

            out.convert_copy_initialize(res.convert_mutable_masquerade_to_untyped())

        res = self.defineNativeFunction(
            "demasquerade_" + callTarget.name,
            ("demasquerade", callTarget.name),
            callTarget.input_types,
            typeWrapper(callTarget.output_type.interpreterTypeRepresentation),
            generator
        )

        return res

    def generateCallConverter(self, callTarget: TypedCallTarget):
        """Given a call target that's optimized for llvm-level dispatch (with individual
        arguments packed into registers), produce a (native) call-target that
        we can dispatch to from our C extension, where arguments are packed into
        an array.

        For instance, we are given
            T f(A1, A2, A3 ...)
        and want to produce
            f(T*, X**)
        where X is the union of A1, A2, etc.

        Args:
            callTarget - a TypedCallTarget giving the function we need
                to generate an alternative entrypoint for
        Returns:
            the linker name of the defined native function
        """
        identifier = "call_converter_" + callTarget.name
        linkName = callTarget.name + ".dispatch"

        if linkName in self._allDefinedNames:
            return linkName

        args = []
        for i in range(len(callTarget.input_types)):
            if not callTarget.input_types[i].is_empty:
                argtype = callTarget.input_types[i].getNativeLayoutType()

                untypedPtr = native_ast.var('input').ElementPtrIntegers(i).load()

                if callTarget.input_types[i].is_pass_by_ref:
                    # we've been handed a pointer, and it's already a pointer
                    args.append(untypedPtr.cast(argtype.pointer()))
                else:
                    args.append(untypedPtr.cast(argtype.pointer()).load())

        if callTarget.output_type is not None and callTarget.output_type.is_pass_by_ref:
            body = callTarget.call(
                native_ast.var('return').cast(callTarget.output_type.getNativeLayoutType().pointer()),
                *args
            )
        else:
            body = callTarget.call(*args)

            if not (callTarget.output_type is None or callTarget.output_type.is_empty):
                body = native_ast.var('return').cast(callTarget.output_type.getNativeLayoutType().pointer()).store(body)

        body = native_ast.FunctionBody.Internal(body=body)

        definition = native_ast.Function(
            args=(
                ('return', native_ast.Type.Void().pointer()),
                ('input', native_ast.Type.Void().pointer().pointer())
            ),
            body=body,
            output_type=native_ast.Type.Void()
        )

        self._link_name_for_identity[identifier] = linkName
        self._allDefinedNames.add(linkName)

        self._definitions[linkName] = definition
        self._new_native_functions.add(linkName)

        return linkName

    def _resolveAllInflightFunctions(self):
        while True:
            identity = self._dependencies.getNextDirtyNode()
            if not identity:
                return

            functionConverter = self._inflight_function_conversions[identity]

            hasDefinitionBeforeConversion = identity in self._inflight_definitions

            try:
                self._currentlyConverting = identity

                self._times_calculated[identity] = self._times_calculated.get(identity, 0) + 1

                nativeFunction, actual_output_type = functionConverter.convertToNativeFunction()

                if nativeFunction is not None:
                    self._inflight_definitions[identity] = (nativeFunction, actual_output_type)
            except Exception:
                for i in self._inflight_function_conversions:
                    if i in self._link_name_for_identity:
                        name = self._link_name_for_identity[i]
                        if name in self._targets:
                            self._targets.pop(name)
                        self._allDefinedNames.discard(name)
                        self._link_name_for_identity.pop(i)

                    self._dependencies.dropNode(i)

                self._inflight_function_conversions.clear()
                self._inflight_definitions.clear()
                self._linktimeHooks.clear()
                raise
            finally:
                self._currentlyConverting = None

            dirtyUpstream = False

            # figure out whether we ought to recalculate all the upstream nodes of this
            # node. we do that if we get a definition and we didn't have one before, or if
            # our type stability changed
            if nativeFunction is not None:
                if not hasDefinitionBeforeConversion:
                    dirtyUpstream = True

                if functionConverter.typesAreUnstable():
                    functionConverter.resetTypeInstabilityFlag()
                    self._dependencies.markDirtyWithLowPriority(identity)
                    dirtyUpstream = True

                name = self._link_name_for_identity[identity]

                self._targets[name] = self.getTypedCallTarget(
                    name,
                    functionConverter._input_types,
                    actual_output_type,
                    functionConverter.alwaysRaises()
                )

            if dirtyUpstream:
                self._dependencies.functionReturnSignatureChanged(identity)

            # when we define an entrypoint to a class, we actually need to compile
            # a version of that function for every override of that function as well.
            # typed_python keeps track of all the entries in all class vtables that
            # need pointers (we generate one dispatch entry for each class that implements
            # a function that gets triggered in a base class). As we resolve inflight
            # functions, we trigger compilation on each of the individual instantiations
            # we receive.
            while self.compileClassDispatch():
                pass

    def compileClassDispatch(self):
        dispatch = _types.getNextUnlinkedClassMethodDispatch()

        if dispatch is None:
            return False

        interfaceClass, implementingClass, slotIndex = dispatch

        name, retType, argTypeTuple, kwargTypeTuple = _types.getClassMethodDispatchSignature(interfaceClass, implementingClass, slotIndex)

        # generate a callback that takes the linked function pointer and jams
        # it into the relevant slot in the vtable once it's produced
        def installOverload(fp):
            _types.installClassMethodDispatch(interfaceClass, implementingClass, slotIndex, fp.fp)

        # we are compiling the function 'name' in 'implementingClass' to be installed when
        # viewing an instance of 'implementingClass' as 'interfaceClass' that's function
        # 'name' called with signature '(*argTypeTuple, **kwargTypeTuple) -> retType'
        assert ClassWrapper.compileMethodInstantiation(
            self,
            interfaceClass,
            implementingClass,
            name,
            retType,
            argTypeTuple,
            kwargTypeTuple,
            callback=installOverload
        )

        return True

    def convertTypedFunctionCall(self, functionType, overloadIx, inputWrappers, assertIsRoot=False):
        overload = functionType.overloads[overloadIx]

        realizedInputWrappers = []

        closureType = functionType.ClosureType

        for closureVarName, closureVarPath in overload.closureVarLookups.items():
            realizedInputWrappers.append(
                typeWrapper(
                    PythonTypedFunctionWrapper.closurePathToCellType(closureVarPath, closureType)
                )
            )

        realizedInputWrappers.extend(inputWrappers)

        return self.convert(
            overload.name,
            overload.functionCode,
            overload.realizedGlobals,
            overload.functionGlobals,
            list(overload.closureVarLookups),
            realizedInputWrappers,
            overload.returnType,
            assertIsRoot=assertIsRoot
        )

    def hashObjectToIdentity(self, hashable):
        if isinstance(hashable, Hash):
            return hashable

        if isinstance(hashable, int):
            return Hash.from_integer(hashable)

        if isinstance(hashable, str):
            return Hash.from_string(hashable)

        if hashable is None:
            return Hash.from_integer(1) + Hash.from_integer(0)

        if isinstance(hashable, (tuple, list)):
            res = Hash.from_integer(len(hashable))
            for t in hashable:
                res += self.hashObjectToIdentity(t)
            return res

        if isinstance(hashable, dict):
            res = Hash.from_integer(len(hashable))
            for t in sorted(hashable):
                res += self.hashObjectToIdentity(t)
                res += self.hashObjectToIdentity(hashable[t])
            return res

        if isinstance(hashable, Wrapper):
            return hashable.identityHash()

        return Hash(_types.identityHash(hashable))

    def convert(
        self,
        funcName,
        funcCode,
        funcGlobals,
        funcGlobalsRaw,
        closureVars,
        input_types,
        output_type,
        assertIsRoot=False,
        callback=None,
    ):
        """Convert a single pure python function using args of 'input_types'.

        It will return no more than 'output_type'. if output_type is None we produce
        the tightest output type possible.

        Args:
            funcName - the name of the function
            funcCode - a Code object representing the code to compile
            funcGlobals - the globals object from the relevant function
            funcGlobalsRaw - the original globals object (with no merging or filtering done)
                which we use to figure out the location of global variables that are not in cells.
            input_types - a type for each free variable in the function closure, and
                then again for each input argument
            output_type - the output type of the function, if known. if this is None,
                then we use type inference to produce the tightest type we can.
                If not None, then we will produce this type or throw an exception.
            assertIsRoot - if True, then assert that no other functions are using
                the converter right now.
            callback - if not None, then a function that gets called back with the
                function pointer to the compiled function when it's known.
        """
        assert isinstance(funcName, str)
        assert isinstance(funcCode, types.CodeType)
        assert isinstance(funcGlobals, dict)

        input_types = tuple([typedPythonTypeToTypeWrapper(i) for i in input_types])

        identityHash = (
            Hash.from_integer(1)
            + self.hashObjectToIdentity((
                funcCode,
                funcName,
                funcGlobals,
                input_types,
                output_type,
                closureVars
            ))
        )
        assert not identityHash.isPoison()

        identity = identityHash.hexdigest

        if callback is not None:
            self.installLinktimeHook(identity, callback)

        name = self.identityHashToLinkerName(funcName, identity)

        if name not in self._allDefinedNames:
            self._link_name_for_identity[identity] = name
            self._allDefinedNames.add(name)

            if self.compilerCache:
                if self.compilerCache.hasSymbol(name):
                    newTypedCallTargets, newNativeFunctionTypes = self.compilerCache.loadForSymbol(name)

                    self._targets.update(newTypedCallTargets)
                    self.llvmCompiler.markExternal(newNativeFunctionTypes)

                    self._allDefinedNames.update(newTypedCallTargets)
                    self._allDefinedNames.update(newNativeFunctionTypes)

        if identity not in self._identifier_to_pyfunc:
            self._identifier_to_pyfunc[identity] = (
                funcName, funcCode, funcGlobals, closureVars, input_types, output_type
            )

        isRoot = len(self._inflight_function_conversions) == 0

        if assertIsRoot:
            assert isRoot

        if self._currentlyConverting is not None:
            self._dependencies.addEdge(self._currentlyConverting, identity)
        else:
            self._dependencies.addRoot(identity)

        if name in self._targets:
            return self._targets[name]

        if identity not in self._inflight_function_conversions:
            functionConverter = self.createConversionContext(
                identity,
                funcName,
                funcCode,
                funcGlobals,
                funcGlobalsRaw,
                closureVars,
                input_types,
                output_type
            )

            self._inflight_function_conversions[identity] = functionConverter

        if isRoot:
            try:
                self._resolveAllInflightFunctions()
                self._installInflightFunctions(name)
                return self._targets[name]
            finally:
                self._inflight_function_conversions.clear()

        else:
            # above us on the stack, we are walking a set of function conversions.
            # if we have ever calculated this function before, we'll have a call
            # target with an output type and we can return that. Otherwise we have to
            # return None, which will cause callers to replace this with a throw
            # until we have had a chance to do a full pass of conversion.
            if name in self._targets:
                return self._targets[name]
            else:
                return None

    def _installInflightFunctions(self, name):
        if VALIDATE_FUNCTION_DEFINITIONS_STABLE:
            # this should always be true, but its expensive so we have it off by default
            for identifier, functionConverter in self._inflight_function_conversions.items():
                try:
                    self._currentlyConverting = identifier

                    nativeFunction, actual_output_type = functionConverter.convertToNativeFunction()

                    assert nativeFunction == self._inflight_definitions[identifier]
                finally:
                    self._currentlyConverting = None

        for identifier, functionConverter in self._inflight_function_conversions.items():
            if identifier in self._identifier_to_pyfunc:
                for v in self._visitors:

                    funcName, funcCode, funcGlobals, closureVars, input_types, output_type = (
                        self._identifier_to_pyfunc[identifier]
                    )

                    try:
                        v.onNewFunction(
                            funcName,
                            funcCode,
                            funcGlobals,
                            closureVars,
                            input_types,
                            functionConverter._varname_to_type.get(FunctionOutput),
                            {k: v.typeRepresentation for k, v in functionConverter._varname_to_type.items() if isinstance(k, str)}
                        )
                    except Exception:
                        logging.exception("event handler %s threw an unexpected exception", v.onNewFunction)

            if identifier not in self._inflight_definitions:
                raise Exception(
                    f"Expected a definition for {identifier} depended on by:\n"
                    + "\n".join("    " + str(i) for i in self._dependencies._dependencies.incoming(identifier))
                )

            nativeFunction, actual_output_type = self._inflight_definitions.get(identifier)

            name = self._link_name_for_identity[identifier]

            self._definitions[name] = nativeFunction
            self._new_native_functions.add(name)
