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

import logging
import types
from types import ModuleType
from typing import Dict, List, Optional, Tuple, Type

from sortedcontainers import SortedSet

import typed_python._types as _types
import typed_python.compiler
import typed_python.compiler.native_ast as native_ast
import typed_python.python_ast as python_ast
from typed_python import Class
from typed_python.compiler.compiler_cache import CompilerCache
from typed_python.compiler.directed_graph import DirectedGraph
from typed_python.compiler.function_conversion_context import (FunctionConversionContext, FunctionOutput, FunctionYield)
from typed_python.compiler.llvm_compiler import Compiler
from typed_python.compiler.native_function_conversion_context import NativeFunctionConversionContext
from typed_python.compiler.native_function_pointer import NativeFunctionPointer
from typed_python.compiler.python_object_representation import typedPythonTypeToTypeWrapper
from typed_python.compiler.type_wrappers.class_wrapper import ClassWrapper
from typed_python.compiler.type_wrappers.python_typed_function_wrapper import (
    CannotBeDetermined, NoReturnTypeSpecified, PythonTypedFunctionWrapper)
from typed_python.compiler.type_wrappers.wrapper import Wrapper
from typed_python.compiler.typed_call_target import TypedCallTarget
from typed_python.hash import Hash
from typed_python.internals import ClassMetaclass


__all__ = ['PythonToNativeConverter']

type_wrapper = lambda t: typed_python.compiler.python_object_representation.typedPythonTypeToTypeWrapper(t)


VALIDATE_FUNCTION_DEFINITIONS_STABLE = False


class FunctionDependencyGraph:
    """
    A wrapper for DirectedGraph with identity levels and the ability to tag nodes for
    recomputation. The nodes of the graph are function hashes, and the edges represent
    dependency.
    """
    def __init__(self):
        self.dependency_graph = DirectedGraph()

        # the search depth in the dependency to find 'identity'
        # the _first_ time we ever saw it. We prefer to update
        # nodes with higher search depth, so we don't recompute
        # earlier nodes until their children are complete.
        self._identity_levels = {}

        # nodes that need to recompute
        self._dirty_inflight_functions = set()

        # (priority, node) pairs that need to recompute
        self._dirty_inflight_functions_with_order = SortedSet(key=lambda pair: pair[0])

    def drop_node(self, node):
        self.dependency_graph.dropNode(node, False)
        if node in self._identity_levels:
            del self._identity_levels[node]
        self._dirty_inflight_functions.discard(node)

    def get_next_dirty_node(self):
        while self._dirty_inflight_functions_with_order:
            priority, identity = self._dirty_inflight_functions_with_order.pop()

            if identity in self._dirty_inflight_functions:
                self._dirty_inflight_functions.discard(identity)

                return identity

    def add_root(self, identity, dirty=True):
        if identity not in self._identity_levels:
            self._identity_levels[identity] = 0
            if dirty:
                self.mark_dirty(identity)

    def add_edge(self, caller, callee, dirty=True):
        if caller not in self._identity_levels:
            raise Exception(f"unknown identity {caller} found in the graph")

        if callee not in self._identity_levels:
            self._identity_levels[callee] = self._identity_levels[caller] + 1

            if dirty:
                self.mark_dirty(callee, isNew=True)

        self.dependency_graph.addEdge(caller, callee)

    def get_names_depended_on(self, caller):
        return self.dependency_graph.outgoing(caller)

    def mark_dirty_with_low_priority(self, callee):
        # mark this dirty, but call it back after new functions.
        self._dirty_inflight_functions.add(callee)

        level = self._identity_levels[callee]
        self._dirty_inflight_functions_with_order.add((-1000000 + level, callee))

    def mark_dirty(self, callee, isNew=False):
        self._dirty_inflight_functions.add(callee)

        if isNew:
            # if its a new node, compute it with higher priority the _higher_ it is in the stack
            # so that we do a depth-first search on the way down
            level = 1000000 - self._identity_levels[callee]
        else:
            level = self._identity_levels[callee]

        self._dirty_inflight_functions_with_order.add((level, callee))

    def function_return_signature_changed(self, identity):
        for caller in self.dependency_graph.incoming(identity):
            self.mark_dirty(caller)


class PythonToNativeConverter:
    """
    What do you do?
    - init

    Parses Python functions into an intermediate representation of the AST. These representations
    are held in self._definitions, and added how?!

    'installed' but new (and hence not fully converted) definitions are stored in
    _new_native_functions, and are sent downstream when build_and_link_new_module is called.


    Holds optional


    Usage is effectively, in compileFunctionOverload, we do convert_typed_function_call,
    then demasquerade_call_target_output, then generate_call_converter, then
    build_and_link_new_module,
    then function_pointer_by_name.


    x = understanding

    'identity' is the hexdigest of the function hash
    'link_name' is the symbol name, prefix + func_name + hash

    Attributes:
        llvm_compiler: The WHAT?
        compiler_cache: The WHAT?
        generate_debug_checks: ?
        _all_defined_names: ?
        _all_cached_names:
        _link_name_for_identity: ?
        _identity_for_link_name: ?
        _definitions: definition blablabla
        _targets: ?
        _inflight_definitions: ?
        _inflight_function_conversions: ?
        _identifier_to_pyfunc: ?
        _times_calculated: ?
        _new_native_functions:
       _visitors: A list of context managers which run on each new function installed with
            _install_inflight_functions, giving info on e.g. compilation counts.
        _currently_converting:
        _dependencies: A directed graph in which nodes are function hashes and edges are a dependence.
    """

    def __init__(self, llvm_compiler: Compiler, compiler_cache: CompilerCache):
        self.llvm_compiler = llvm_compiler
        self.compiler_cache = compiler_cache

        # all LoadedModule objects that we have created. We need to keep them alive so
        # that any python metadata objects the've created stay alive as well. Ultimately, this
        # may not be the place we put these objects (for instance, you could imagine a
        # 'dummy' compiler cache or something). But for now, we need to keep them alive.
        self.loaded_uncached_modules = []

        # if True, then insert additional code to check for undefined behavior.
        self.generate_debug_checks = False

        self._link_name_for_identity: Dict[str, str] = {}
        self._identity_for_link_name: Dict[str, str] = {}
        self._definitions: Dict[str, native_ast.Function] = {}
        self._targets: Dict[str, TypedCallTarget] = {}
        # NB i don't think the Wrapper is ever actually used.
        self._inflight_definitions: Dict[str, Tuple[native_ast.Function, Type[Wrapper]]] = {}
        self._inflight_function_conversions: Dict[str, FunctionConversionContext] = {}
        self._identifier_to_pyfunc = {}
        self._times_calculated = {}

        # function names that have been defined but not yet compiled
        self._new_native_functions = set()

        self._visitors = []

        # the identity of the function we're currently evaluating.
        # we use this to track which functions need to get rebuilt when
        # other functions change types.
        self._currently_converting = None

        # tuple of (baseClass, childClass, slotIndex) containing
        # virtual methods that need to get instantiated during the
        # current compilation unit.
        self._delayedVMIs = []
        self._delayed_destructors = []

        self._installedVMIs = set()
        self._installed_destructors = set()

        self._dependencies = FunctionDependencyGraph()

    def convert_typed_function_call(self,
                                    function_type,
                                    overload_index: int,
                                    input_wrappers: List[Type[Wrapper]],
                                    assert_is_root=False) -> Optional[TypedCallTarget]:
        """Does what?
        Args:
            function_type: Normally a subclass of typed_python._types.Function? sometimes not a class at all?
            overload_index: The index of function_type.overloads to access, corresponding to a specific set of input argument types
            input_wrappers: The input wrappers? TODO not always a Wrapper, sometimes just a type?!
            assert_is_root: If True, then assert that no other functions are using
                the converter right now.
        Trash! TODO fix
        Returns:
            bla
        """
        overload = function_type.overloads[overload_index]

        realized_input_wrappers = []

        closure_type = function_type.ClosureType

        for closure_var_name, closure_var_path in overload.closureVarLookups.items():
            realized_input_wrappers.append(
                type_wrapper(
                    PythonTypedFunctionWrapper.closurePathToCellType(closure_var_path, closure_type)
                )
            )

        realized_input_wrappers.extend(input_wrappers)

        return_type = PythonTypedFunctionWrapper.computeFunctionOverloadReturnType(
            overload,
            input_wrappers,
            {}
        )

        if return_type is CannotBeDetermined:
            return_type = object

        if return_type is NoReturnTypeSpecified:
            return_type = None

        return self.convert_python_to_native(
            overload.name,
            overload.functionCode,
            overload.realizedGlobals,
            overload.functionGlobals,
            overload.funcGlobalsInCells,
            list(overload.closureVarLookups),
            realized_input_wrappers,
            return_type,
            assert_is_root=assert_is_root
        )

    def convert_python_to_native(
        self,
        func_name: str,
        func_code: types.CodeType,
        func_globals: Dict,
        func_glocals_raw,
        func_globals_from_cells,
        closure_vars,
        input_types,
        output_type,
        assert_is_root=False,
        conversion_type=None
    ):
        """Convert a single pure python function using args of 'input_types'.

        It will return no more than 'output_type'. if output_type is None we produce
        the tightest output type possible.

        Args:
            func_name: the name of the function
            func_code: a Code object representing the code to compile
            func_globals: the globals object from the relevant function
            func_globals_raw: the original globals object (with no merging or filtering done)
                which we use to figure out the location of global variables that are not in cells.
            func_globals_from_cells:  a list of the names that are globals that are actually accessed
                as cells.
            closure_vars: a list of the names of the variables that are accessed as free variables. TODO check
            input_types - a type for each free variable in the function closure, and
                then again for each input argument
            output_type - the output type of the function, if known. if this is None,
                then we use type inference to produce the tightest type we can.
                If not None, then we will produce this type or throw an exception.
            assert_is_root - if True, then assert that no other functions are using
                the converter right now.
            conversion_type - if None, this is a normal function conversion. Otherwise,
                this must be a subclass of FunctionConversionContext
        """
        assert isinstance(func_name, str)
        assert isinstance(func_code, types.CodeType)
        assert isinstance(func_globals, dict)

        input_types = tuple([typedPythonTypeToTypeWrapper(i) for i in input_types])

        identity_hash = (
            Hash.from_integer(1)
            + self.hash_object_to_identity((
                func_code,
                func_name,
                input_types,
                output_type,
                closure_vars,
                conversion_type
            )) +
            self._hash_globals(func_globals, func_code, func_globals_from_cells)
        )

        assert not identity_hash.isPoison()

        identity = identity_hash.hexdigest

        name = self._identity_hash_to_function_name(func_name, identity)

        self._define_link_name(identity, name)

        if identity not in self._identifier_to_pyfunc:
            self._identifier_to_pyfunc[identity] = (
                func_name, func_code, func_globals, closure_vars, input_types, output_type, conversion_type
            )

        is_root = len(self._inflight_function_conversions) == 0

        if assert_is_root:
            assert is_root

        target = self._get_target(name)

        if self._currently_converting is not None:
            self._dependencies.add_edge(self._currently_converting, identity, dirty=(target is None))
        else:
            self._dependencies.add_root(identity, dirty=(target is None))

        if target is not None:
            return target

        if identity not in self._inflight_function_conversions:
            function_converter = self._create_conversion_context(
                identity,
                func_name,
                func_code,
                func_globals,
                func_glocals_raw,
                closure_vars,
                input_types,
                output_type,
                conversion_type
            )
            self._inflight_function_conversions[identity] = function_converter

        if is_root:
            try:
                self._resolve_all_inflight_functions()
                self._install_inflight_functions()
                return self._get_target(name)
            finally:
                self._inflight_function_conversions.clear()

        else:
            # above us on the stack, we are walking a set of function conversions.
            # if we have ever calculated this function before, we'll have a call
            # target with an output type and we can return that. Otherwise we have to
            # return None, which will cause callers to replace this with a throw
            # until we have had a chance to do a full pass of conversion.
            if self._get_target(name) is not None:
                raise RuntimeError(f"Unexpected conversion error for {name}")
            return None

    def demasquerade_call_target_output(self, call_target: TypedCallTarget) -> Optional[TypedCallTarget]:
        """Ensure we are returning the correct 'interpreterType' from callTarget.

        In some cases, we may return a 'masquerade' type in compiled code. This is fine
        for other compiled code, but the interpreter needs us to transform the result back
        to the right interpreter type. For instance, we may be returning a *args tuple.


        Args:
            call_target: the input TypedCallTarget to demasquerade (or return unchanged)
        Returns:
            a new TypedCallTarget where the output type has the right return type.
        """
        if call_target.output_type is None:
            return call_target

        if call_target.output_type.interpreterTypeRepresentation == call_target.output_type.typeRepresentation:
            return call_target

        def generator(context, out, *args):
            assert out is not None, "we should have an output because no masquerade types are pass-by-value"

            res = context.call_typed_call_target(call_target, args)

            out.convert_copy_initialize(res.convert_masquerade_to_untyped())

        res = self.define_native_function(
            "demasquerade_" + call_target.name,
            ("demasquerade", call_target.name),
            call_target.input_types,
            type_wrapper(call_target.output_type.interpreterTypeRepresentation),
            generator
        )

        return res

    def generate_call_converter(self, call_target: TypedCallTarget) -> str:
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
            call_target: a TypedCallTarget giving the function we need
                to generate an alternative entrypoint for
        Returns:
            the linker name of the defined native function
        """
        identifier = "call_converter_" + call_target.name
        link_name = call_target.name + ".dispatch"

        # we already made a definition for this in this process so don't do it again
        if link_name in self._definitions:
            return link_name

        # we already defined it in another process so don't do it again
        if self.compiler_cache is not None and self.compiler_cache.has_symbol(link_name):
            return link_name

        # N.B. there aren't targets for call converters. We make the definition directly.
        args = []
        for i in range(len(call_target.input_types)):
            if not call_target.input_types[i].is_empty:
                argtype = call_target.input_types[i].getNativeLayoutType()

                untyped_ptr = native_ast.var('input').ElementPtrIntegers(i).load()

                if call_target.input_types[i].is_pass_by_ref:
                    # we've been handed a pointer, and it's already a pointer
                    args.append(untyped_ptr.cast(argtype.pointer()))
                else:
                    args.append(untyped_ptr.cast(argtype.pointer()).load())

        if call_target.output_type is not None and call_target.output_type.is_pass_by_ref:
            body = call_target.call(
                native_ast.var('return').cast(call_target.output_type.getNativeLayoutType().pointer()),
                *args
            )
        else:
            body = call_target.call(*args)

            if not (call_target.output_type is None or call_target.output_type.is_empty):
                body = native_ast.var('return').cast(call_target.output_type.getNativeLayoutType().pointer()).store(body)

        body = native_ast.FunctionBody.Internal(body=body)

        definition = native_ast.Function(
            args=(
                ('return', native_ast.Type.Void().pointer()),
                ('input', native_ast.Type.Void().pointer().pointer())
            ),
            body=body,
            output_type=native_ast.Type.Void()
        )

        self._link_name_for_identity[identifier] = link_name
        self._identity_for_link_name[link_name] = identifier

        self._definitions[link_name] = definition
        self._new_native_functions.add(link_name)

        return link_name

    def build_and_link_new_module(self) -> None:
        """Pull all in-flight conversions and compile as a module.
        This grabs all new functions, where the python->native conversion has been completed,
        and collates these + their dependencies and global variables into a module, performing the
        native -> llvm conversion. The list of in-flight functions is cleared once parsed to avoid
        double-compilation.
        """

        definitions = self._extract_new_function_definitions()

        if not definitions:
            return

        if self.compiler_cache is None:
            loaded_module = self.llvm_compiler.buildModule(definitions)  # TODO handle None
            loaded_module.linkGlobalVariables()
            self.loaded_uncached_modules.append(loaded_module)
            return

        # get a set of function names that we depend on
        externally_used = set()
        dependency_edgelist = []

        for func_name in definitions:
            ident = self._identity_for_link_name.get(func_name)
            if ident is not None:
                for dep in self._dependencies.get_names_depended_on(ident):
                    depLN = self._link_name_for_identity.get(dep)
                    dependency_edgelist.append([func_name, depLN])
                    if depLN not in definitions:
                        externally_used.add(depLN)

        binary = self.llvm_compiler.buildSharedObject(definitions)

        self.compiler_cache.add_module(
            binary,
            {name: self._get_target(name) for name in definitions if self._has_target(name)},
            externally_used,
            dependency_edgelist
        )

    def define_non_python_function(self, name: str, identity_tuple: Tuple, context) -> Optional[TypedCallTarget]:
        """Define a non-python generating function (if we haven't defined it before already)

            name: the name to actually give the function.
            identity_tuple: a unique (sha)hashable tuple
            context: a FunctionConversionContext lookalike

        Returns:
            A TypedCallTarget, or None if it's not known yet
        """
        identity = self.hash_object_to_identity(identity_tuple).hexdigest
        linkName = self._identity_hash_to_function_name(name, identity, "runtime.")

        self._define_link_name(identity, linkName)

        target = self._get_target(linkName)

        if self._currently_converting is not None:
            self._dependencies.add_edge(self._currently_converting, identity, dirty=(target is None))
        else:
            self._dependencies.add_root(identity, dirty=(target is None))

        if target is not None:
            return target

        self._inflight_function_conversions[identity] = context

        if context.knownOutputType() is not None or context.alwaysRaises():
            self._set_target(
                linkName,
                self._get_typed_call_target(
                    name,
                    context.getInputTypes(),
                    context.knownOutputType(),
                    always_raises=context.alwaysRaises(),
                    function_metadata=context.functionMetadata,
                )
            )

        if self._currently_converting is None:
            # force the function to resolve immediately
            self._resolve_all_inflight_functions()
            self._install_inflight_functions()
            self._inflight_function_conversions.clear()

        return self._get_target(linkName)

    def define_native_function(self,
                               name: str,
                               identity: Tuple,
                               input_types,
                               output_type,
                               generatingFunction) -> Optional[TypedCallTarget]:
        """Define a native function if we haven't defined it before already.


        Args:
            name: the name to actually give the function.
            identity: a tuple consisting of strings, ints, type wrappers, and tuples of same
            input_types: list of Wrapper objects for the incoming types
            output_type: Wrapper object for the output type, or None if the function doesn't
                ever return
            generatingFunction: a function producing a native_function_definition.
                It should accept an expression_conversion_context, an expression for the output
                if it's not pass-by-value (or None if it is), and a bunch of TypedExpressions
                and produce code that always ends in a terminal expression, (or if it's pass by value,
                flows off the end of the function)

        Returns:
            A TypedCallTarget. 'generatingFunction' may call this recursively if it wants.
        """
        if output_type is not None:
            output_type = type_wrapper(output_type)

        input_types = [type_wrapper(x) for x in input_types]

        identity = (
            Hash.from_integer(2) +
            self.hash_object_to_identity(identity) +
            self.hash_object_to_identity(output_type) +
            self.hash_object_to_identity(input_types)
        ).hexdigest

        return self.define_non_python_function(
            name,
            identity,
            NativeFunctionConversionContext(
                self, input_types, output_type, generatingFunction, identity
            )
        )

    def compile_single_class_dispatch(self, interface_class: ClassMetaclass, implementing_class: ClassMetaclass, slot_index: int):
        if (interface_class, implementing_class, slot_index) in self._installedVMIs:
            return

        name, ret_type, arg_type_tuple, kwarg_type_tuple = _types.getClassMethodDispatchSignature(interface_class,
                                                                                                  implementing_class,
                                                                                                  slot_index)

        # we are compiling the function 'name' in 'implementingClass' to be installed when
        # viewing an instance of 'implementingClass' as 'interfaceClass' that's function
        # 'name' called with signature '(*argTypeTuple, **kwargTypeTuple) -> retType'
        typed_call_target = ClassWrapper.compileVirtualMethodInstantiation(
            self,
            interface_class,
            implementing_class,
            name,
            ret_type,
            arg_type_tuple,
            kwarg_type_tuple
        )

        assert typed_call_target is not None

        self.build_and_link_new_module()

        fp = self.function_pointer_by_name(typed_call_target.name)

        if fp is None:
            raise Exception(f"Couldn't find a function pointer for {typed_call_target.name}")

        _types.installClassMethodDispatch(interface_class, implementing_class, slot_index, fp.fp)

        self._installedVMIs.add(
            (interface_class, implementing_class, slot_index)
        )

    def compile_class_destructor(self, cls):
        if cls in self._installed_destructors:
            return

        typed_call_target = type_wrapper(cls).compileDestructor(self)

        assert typed_call_target is not None

        self.build_and_link_new_module()

        fp = self.function_pointer_by_name(typed_call_target.name)

        _types.installClassDestructor(cls, fp.fp)
        self._installed_destructors.add(cls)

    def function_pointer_by_name(self, func_name) -> NativeFunctionPointer:
        """Find a NativeFunctionPointer for a given link-time name.

        Args:
            func_name (str) - the name of the compiled symbol we want.

        Returns:
            a NativeFunctionPointer or None
        """
        if self.compiler_cache is None:
            # the llvm compiler holds it all
            return self.llvm_compiler.function_pointer_by_name(func_name)
        else:
            # the llvm compiler is just building shared objects, but the
            # compiler cache has all the pointers.
            return self.compiler_cache.function_pointer_by_name(func_name)

    def hash_object_to_identity(self, hashable, is_module_val=False):
        if isinstance(hashable, Hash):
            return hashable

        if isinstance(hashable, int):
            return Hash.from_integer(hashable)

        if isinstance(hashable, str):
            return Hash.from_string(hashable)

        if hashable is None:
            return Hash.from_integer(1) + Hash.from_integer(0)

        if isinstance(hashable, (dict, list)) and is_module_val:
            # don't look into dicts and lists at module level
            return Hash.from_integer(2)

        if isinstance(hashable, (tuple, list)):
            res = Hash.from_integer(len(hashable))
            for t in hashable:
                res += self.hash_object_to_identity(t, is_module_val)
            return res

        if isinstance(hashable, Wrapper):
            return hashable.identityHash()

        return Hash(_types.identityHash(hashable))

    def is_currently_converting(self):
        return len(self._inflight_function_conversions) > 0

    def get_definition_count(self):
        return len(self._definitions)

    def add_visitor(self, visitor):
        self._visitors.append(visitor)

    def remove_visitor(self, visitor):
        self._visitors.remove(visitor)

    def identity_to_name(self, identity):
        """Convert a function identity to the link-time name for the function.

        Args:
            identity - an identity tuple that uniquely identifies the function

        Returns:
            name - the linker name of the native function this represents, or None
                if the identity is unknown
        """
        return self._link_name_for_identity.get(identity)

    def trigger_virtual_destructor(self, instance_type):
        self._delayed_destructors.append(instance_type)

    def trigger_virtual_method_instantiation(self, instance_type, method_name, return_type, arg_tuple_type, kwarg_tuple_type):
        """Instantiate a virtual method as part of this batch of compilation.

        Normally, compiling 'virtual' methods (method instantiations on subclasses
        that are known as a base class to the compiler) happens lazily.  In some cases
        (for instance, generators) we want to force compilation of specific methods
        when we define the class, since otherwise we end up with irregular performance
        because we're lazily triggering an expensive operation.

        This method forces the current compilation operation to compile and link
        instantiating 'methodName' on instances of 'instanceType'.
        """
        for baseClass in instance_type.__mro__:
            if issubclass(baseClass, Class) and baseClass is not Class:
                slot = _types.allocateClassMethodDispatch(
                    baseClass,
                    method_name,
                    return_type,
                    arg_tuple_type,
                    kwarg_tuple_type
                )

                self._delayedVMIs.append(
                    (baseClass, instance_type, slot)
                )

    def flush_delayed_VMIs(self):
        while self._delayedVMIs or self._delayed_destructors:
            vmis = self._delayedVMIs
            self._delayedVMIs = []

            for baseClass, instanceClass, dispatchSlot in vmis:
                self.compile_single_class_dispatch(baseClass, instanceClass, dispatchSlot)

            delayedDestructors = self._delayed_destructors
            self._delayed_destructors = []

            for T in delayedDestructors:
                self.compile_class_destructor(T)

    def _extract_new_function_definitions(self):
        """Return a list of all new function definitions from the last conversion."""
        res = {}

        for u in self._new_native_functions:
            res[u] = self._definitions[u]

        self._new_native_functions = set()

        return res

    def _identity_hash_to_function_name(self, name, identity_hash, prefix="tp."):
        assert isinstance(name, str)
        assert isinstance(identity_hash, str)
        assert isinstance(prefix, str)

        return prefix + name + "." + identity_hash

    def _create_conversion_context(
        self,
        identity,
        funcName,
        funcCode,
        funcGlobals,
        funcGlobalsRaw,
        closureVars,
        input_types,
        output_type,
        conversionType
    ):
        ConverterType = conversionType or FunctionConversionContext

        pyast = self._code_to_ast(funcCode)

        return ConverterType(
            self,
            funcName,
            identity,
            input_types,
            output_type,
            closureVars,
            funcGlobals,
            funcGlobalsRaw,
            pyast.args,
            pyast,
        )

    def _define_link_name(self, identity, link_name):
        if identity in self._link_name_for_identity:
            if self._link_name_for_identity[identity] != link_name:
                raise Exception(
                    f"For identity {identity}:\n\n"
                    f"{self._link_name_for_identity[identity]}\n\n!=\n\n{link_name}"
                )
            assert self._identity_for_link_name[link_name] == identity
        else:
            self._link_name_for_identity[identity] = link_name
            self._identity_for_link_name[link_name] = identity

    def _has_target(self, link_name):
        return self._get_target(link_name) is not None

    def _delete_target(self, link_name):
        self._targets.pop(link_name)

    def _set_target(self, link_name, target):
        assert (isinstance(target, TypedCallTarget))
        self._targets[link_name] = target

    def _get_target(self, link_name) -> Optional[TypedCallTarget]:
        if link_name in self._targets:
            return self._targets[link_name]

        if self.compiler_cache is not None and self.compiler_cache.has_symbol(link_name):
            return self.compiler_cache.get_target(link_name)

        return None

    def _get_typed_call_target(self, name, input_types, output_type, always_raises=False, function_metadata=None) -> TypedCallTarget:
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
            alwaysRaises=always_raises,
            functionMetadata=function_metadata
        )

        return res

    def _code_to_ast(self, f):
        return python_ast.convertFunctionToAlgebraicPyAst(f)

    def _resolve_all_inflight_functions(self):
        while True:
            identity = self._dependencies.get_next_dirty_node()
            if not identity:
                return

            functionConverter = self._inflight_function_conversions[identity]

            hasDefinitionBeforeConversion = identity in self._inflight_definitions

            try:
                self._currently_converting = identity

                self._times_calculated[identity] = self._times_calculated.get(identity, 0) + 1

                # this calls back into convert with dependencies
                # they get registered as dirty
                nativeFunction, actual_output_type = functionConverter.convertToNativeFunction()

                if nativeFunction is not None:
                    self._inflight_definitions[identity] = (nativeFunction, actual_output_type)
            except Exception:
                for i in self._inflight_function_conversions:
                    if i in self._link_name_for_identity:
                        name = self._link_name_for_identity[i]
                        if self._has_target(name):
                            self._delete_target(name)
                        ln = self._link_name_for_identity.pop(i)
                        self._identity_for_link_name.pop(ln)

                    self._dependencies.drop_node(i)

                self._inflight_function_conversions.clear()
                self._inflight_definitions.clear()
                raise
            finally:
                self._currently_converting = None

            dirtyUpstream = False

            # figure out whether we ought to recalculate all the upstream nodes of this
            # node. we do that if we get a definition and we didn't have one before, or if
            # our type stability changed
            if nativeFunction is not None:
                if not hasDefinitionBeforeConversion:
                    dirtyUpstream = True

                if functionConverter.typesAreUnstable():
                    functionConverter.resetTypeInstabilityFlag()
                    self._dependencies.mark_dirty_with_low_priority(identity)
                    dirtyUpstream = True

                name = self._link_name_for_identity[identity]

                self._set_target(
                    name,
                    self._get_typed_call_target(
                        name,
                        functionConverter._input_types,
                        actual_output_type,
                        always_raises=functionConverter.alwaysRaises(),
                        function_metadata=functionConverter.functionMetadata,
                    ),
                )

            if dirtyUpstream:
                self._dependencies.function_return_signature_changed(identity)

    def _hash_globals(self, funcGlobals, code, funcGlobalsFromCells):
        """Hash a given piece of code's accesses to funcGlobals.

        We're trying to make sure that if we have a reference to module 'x'
        in our globals, but we only ever use 'x' by writing 'x.f' or 'x.g', then
        we shouldn't depend on the entirety of the definition of 'x'.
        """

        res = Hash.from_integer(0)

        for dotSeq in _types.getCodeGlobalDotAccesses(code):
            res += self._hash_dot_seq(dotSeq, funcGlobals)

        for globalName in funcGlobalsFromCells:
            res += self._hash_dot_seq([globalName], funcGlobals)

        return res

    def _hash_dot_seq(self, dotSeq, funcGlobals):
        if not dotSeq or dotSeq[0] not in funcGlobals:
            return Hash.from_integer(0)

        item = funcGlobals[dotSeq[0]]

        if not isinstance(item, ModuleType) or len(dotSeq) == 1:
            return Hash.from_string(dotSeq[0]) + self.hash_object_to_identity(item, True)

        if not hasattr(item, dotSeq[1]):
            return Hash.from_integer(0)

        return Hash.from_string(dotSeq[0] + "." + dotSeq[1]) + self.hash_object_to_identity(getattr(item, dotSeq[1]), True)

    def _install_inflight_functions(self):
        """Add all function definitions corresponding to keys in inflight_function_conversions to the relevant dictionaries."""
        if VALIDATE_FUNCTION_DEFINITIONS_STABLE:
            # this should always be true, but its expensive so we have it off by default
            for identifier, functionConverter in self._inflight_function_conversions.items():
                try:
                    self._currently_converting = identifier

                    nativeFunction, actual_output_type = functionConverter.convertToNativeFunction()

                    assert nativeFunction == self._inflight_definitions[identifier]
                finally:
                    self._currently_converting = None

        for identifier, functionConverter in self._inflight_function_conversions.items():
            outboundTargets = []
            for outboundFuncId in self._dependencies.get_names_depended_on(identifier):
                name = self._link_name_for_identity[outboundFuncId]
                target = self._get_target(name)
                if target is not None:
                    outboundTargets.append(target)
                else:
                    raise RuntimeError(f'dependency not found for {name}.')

            nativeFunction, actual_output_type = self._inflight_definitions.get(identifier)

            if identifier in self._identifier_to_pyfunc:
                for v in self._visitors:
                    funcName, funcCode, funcGlobals, closureVars, input_types, output_type, conversionType = (
                        self._identifier_to_pyfunc[identifier]
                    )

                    try:
                        v.onNewFunction(
                            identifier,
                            functionConverter,
                            nativeFunction,
                            funcName,
                            funcCode,
                            funcGlobals,
                            closureVars,
                            input_types,
                            functionConverter._varname_to_type.get(FunctionOutput),
                            functionConverter._varname_to_type.get(FunctionYield),
                            {k: v.typeRepresentation for k, v in functionConverter._varname_to_type.items() if isinstance(k, str)},
                            conversionType,
                            outboundTargets
                        )
                    except Exception:
                        logging.exception("event handler %s threw an unexpected exception", v.onNewFunction)
            else:
                for v in self._visitors:
                    v.onNewNonpythonFunction(
                        identifier,
                        self._link_name_for_identity[identifier],
                        functionConverter,
                        nativeFunction,
                        outboundTargets
                    )

            if identifier not in self._inflight_definitions:
                raise Exception(
                    f"Expected a definition for {identifier} depended on by:\n"
                    + "\n".join("    " + str(i) for i in self._dependencies.dependency_graph.incoming(identifier))
                )

            nativeFunction, actual_output_type = self._inflight_definitions.get(identifier)

            name = self._link_name_for_identity[identifier]

            self._definitions[name] = nativeFunction
            self._new_native_functions.add(name)
