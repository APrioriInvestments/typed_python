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

from sortedcontainers import SortedSet
from types import ModuleType
from typing import Optional, Dict, Type, List, Set, Tuple, Union

import typed_python.python_ast as python_ast
import typed_python._types as _types
import typed_python.compiler
import typed_python.compiler.native_ast as native_ast

from typed_python.compiler.compiler_cache import CompilerCache
from typed_python.compiler.llvm_compiler import Compiler
from typed_python.compiler.native_function_pointer import NativeFunctionPointer
from typed_python.compiler.directed_graph import DirectedGraph
from typed_python.compiler.type_wrappers.wrapper import Wrapper
from typed_python.compiler.type_wrappers.class_wrapper import ClassWrapper
from typed_python.compiler.python_object_representation import typedPythonTypeToTypeWrapper
# from typed_python.compiler.runtime import RuntimeEventVisitor (not possible until 3.11 due to circular imports)
from typed_python.compiler.function_conversion_context import FunctionConversionContext, FunctionOutput, FunctionYield
from typed_python.compiler.native_function_conversion_context import NativeFunctionConversionContext
from typed_python.compiler.type_wrappers.python_typed_function_wrapper import (
    PythonTypedFunctionWrapper, CannotBeDetermined, NoReturnTypeSpecified
)
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

    def add_root(self, identity):
        if identity not in self._identity_levels:
            self._identity_levels[identity] = 0
            self.mark_dirty(identity)

    def add_edge(self, caller, callee):
        if caller not in self._identity_levels:
            raise Exception(f"unknown identity {caller} found in the graph")

        if callee not in self._identity_levels:
            self._identity_levels[callee] = self._identity_levels[caller] + 1

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
        # if True, then insert additional code to check for undefined behavior.
        self.generate_debug_checks = False

        # all link names for which we have a definition.
        self._all_defined_names: Set[str] = set()

        # all names we loaded from the cache (not necessarily every function loaded in the cache,
        # only those dependencies of a function we have called _loadFromCompilerCache on).
        self._all_cached_names: Set[str] = set()

        self._link_name_for_identity: Dict[str, str] = {}
        self._identity_for_link_name: Dict[str, str] = {}
        # a mapping from link_name to native-level function definition
        self._definitions: Dict[str, native_ast.Function] = {}
        # how is this different from _definitions?
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

        self._dependencies = FunctionDependencyGraph()

    def convert_typed_function_call(self,
                                    function_type,
                                    overload_index: int,
                                    input_wrappers: List[Type[Wrapper]],
                                    assert_is_root=False) -> TypedCallTarget:
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
        func_globals_raw,
        func_globals_from_cells,
        closure_vars,
        input_types,
        output_type,
        assert_is_root=False,
        conversion_type=None
    ) -> TypedCallTarget:
        """Convert a single pure python function using args of 'input_types'.

        It will return no more than 'output_type'. if output_type is None we produce
        the tightest output type possible.

        Args:
            func_name: the name of the function
            func_code: a Code object representing the code to compile
            func_globals: the globals object from the relevant function
            func_globals_raw: the original globals object (with no merging or filtering done)
                which we use to figure out the location of global variables that are not in cells.
            func_globals_from_cells: a list of the names that are globals that are actually accessed
                as cells.
            closure_vars: TODO
            input_types: a type for each free variable in the function closure, and
                then again for each input argument
            output_type: the output type of the function, if known. if this is None,
                then we use type inference to produce the tightest type we can.
                If not None, then we will produce this type or throw an exception.
            assert_is_root: if True, then assert that no other functions are using
                the converter right now.
            conversion_type: if None, this is a normal function conversion. Otherwise,
                this must be a subclass of FunctionConversionContext

        Returns:
            A TypedCallTarget TODO doing what?
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

        name = self._identity_hash_to_linker_name(func_name, identity)

        self._define_link_name(identity, name)

        if identity not in self._identifier_to_pyfunc:
            self._identifier_to_pyfunc[identity] = (
                func_name, func_code, func_globals, closure_vars, input_types, output_type, conversion_type
            )

        is_root = len(self._inflight_function_conversions) == 0

        if assert_is_root:
            assert is_root

        if self._currently_converting is not None:
            self._dependencies.add_edge(self._currently_converting, identity)
        else:
            self._dependencies.add_root(identity)

        if name in self._targets:
            return self._targets[name]

        if identity not in self._inflight_function_conversions:
            function_converter = self._create_conversion_context(
                identity,
                func_name,
                func_code,
                func_globals,
                func_globals_raw,
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

    def demasquerade_call_target_output(self, call_target: TypedCallTarget) -> TypedCallTarget:
        """Ensure we are returning the correct 'interpreterType' from callTarget.

        In some cases, we may return a 'masquerade' type in compiled code. This is fine
        for other compiled code, but the interpreter needs us to transform the result back
        to the right interpreter type. For instance, we may be returning a *args tuple.

        Args:
            call_target: the input TypedCallTarget to be demasqueraded (or returned unchanged).
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
            callTarget: a TypedCallTarget giving the function we need
                to generate an alternative entrypoint for
        Returns:
            the linker name of the defined native function
        """
        identifier = "call_converter_" + call_target.name
        link_name = call_target.name + ".dispatch"

        # consider something better
        if link_name in self._all_defined_names:
            return link_name

        self._load_from_compiler_cache(link_name)
        if link_name in self._all_defined_names:
            return link_name

        args = []
        for i in range(len(call_target.input_types)):
            if not call_target.input_types[i].is_empty:
                argtype = call_target.input_types[i].getNativeLayoutType()

                untypedPtr = native_ast.var('input').ElementPtrIntegers(i).load()

                if call_target.input_types[i].is_pass_by_ref:
                    # we've been handed a pointer, and it's already a pointer
                    args.append(untypedPtr.cast(argtype.pointer()))
                else:
                    args.append(untypedPtr.cast(argtype.pointer()).load())

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

        assert type(identifier) == type(link_name) == str
        self._identity_for_link_name[link_name] = identifier
        self._all_defined_names.add(link_name)

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
        targets = self._extract_new_function_definitions()

        if not targets:
            return

        if self.compiler_cache is None:
            # todo - what is loadedModule doing? what is stored?
            loaded_module = self.llvm_compiler.buildModule(targets)
            loaded_module.linkGlobalVariables()
            return

        # get a set of function names that we depend on
        externally_used = set()

        for funcName in targets:
            ident = self._identity_for_link_name.get(funcName)
            if ident is not None:
                for dep in self._dependencies.get_names_depended_on(ident):
                    depLN = self._link_name_for_identity.get(dep)
                    if depLN not in targets:
                        externally_used.add(depLN)

        binary = self.llvm_compiler.buildSharedObject(targets)

        self.compiler_cache.addModule(
            binary,
            {name: self._targets[name] for name in targets if name in self._targets},
            externally_used
        )

    def define_non_python_function(self, name: str, identity_tuple: Tuple, context) -> Optional[TypedCallTarget]:
        """Define a non-python generating function (if we haven't defined it before already)

        Args:
            name: the name to actually give the function.
            identity_tuple: a unique (sha)hashable tuple
            context: a FunctionConversionContext lookalike

        Returns:
            A TypedCallTarget, or None if it's not known yet
        """
        identity = self.hash_object_to_identity(identity_tuple).hexdigest
        link_name = self._identity_hash_to_linker_name(name, identity, "runtime.")

        self._define_link_name(identity, link_name)

        if self._currently_converting is not None:
            self._dependencies.add_edge(self._currently_converting, identity)
        else:
            self._dependencies.add_root(identity)

        if link_name in self._targets:
            return self._targets.get(link_name)

        self._inflight_function_conversions[identity] = context

        if context.knownOutputType() is not None or context.alwaysRaises():
            self._targets[link_name] = self._get_typed_call_target(
                name,
                context.getInputTypes(),
                context.knownOutputType(),
                alwaysRaises=context.alwaysRaises(),
                functionMetadata=context.functionMetadata
            )

        if self._currently_converting is None:
            # force the function to resolve immediately
            self._resolve_all_inflight_functions()
            self._install_inflight_functions()
            self._inflight_function_conversions.clear()

        return self._targets.get(link_name)

    def define_native_function(self,
                               name: str,
                               identity: Tuple,
                               input_types,
                               output_type,
                               generating_function) -> Optional[TypedCallTarget]:
        """Define a native function if we haven't defined it before already.

        Args:
            name: the name to actually give the function.
            identity: a tuple consisting of strings, ints, type wrappers, and tuples of same
            input_types: list of Wrapper objects for the incoming types (TODO not always true! sometimes it is a type!)
            output_type: Wrapper object for the output type. (TODO as above, plus sometimes None)
            generating_function: a function producing a native_function_definition.
                It should accept an expression_conversion_context, an expression for the output
                if it's not pass-by-value (or None if it is), and a bunch of TypedExpressions
                and produce code that always ends in a terminal expression, (or if it's pass by value,
                flows off the end of the function)

        Returns:
            A TypedCallTarget. `generating_function` may call this recursively if it wants. TODO can this be None?
                define_non_python_function can return None. But the tests in pytest never have it doing so.
        """

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
                self, input_types, output_type, generating_function, identity
            )
        )

    def compile_single_class_dispatch(self, interface_class: ClassMetaclass, implementing_class: ClassMetaclass, slot_index: int):
        """TODO what do you do?

        Runs build_and_link_new_module to generate LLVM code.
        """
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

    def compile_class_destructor(self, cls):
        """Generate a destructor for class `cls`, then hand off to the native->llvm compiler."""
        typedCallTarget = type_wrapper(cls).compileDestructor(self)

        assert typedCallTarget is not None

        self.build_and_link_new_module()

        fp = self.function_pointer_by_name(typedCallTarget.name)

        _types.installClassDestructor(cls, fp.fp)

    def function_pointer_by_name(self, link_name: str) -> Optional[NativeFunctionPointer]:
        """Find a NativeFunctionPointer for a given link-time name.

        Args:
            link_name: the name of the compiled symbol we want

        Returns:
            a NativeFunctionPointer or None
        """
        if self.compiler_cache is None:
            # the llvm compiler holds it all
            return self.llvm_compiler.function_pointer_by_name(link_name)
        else:
            # the llvm compiler is just building shared objects, but the
            # compiler cache has all the pointers.
            return self.compiler_cache.function_pointer_by_name(link_name)

    def hash_object_to_identity(self, hashable, is_module_val=False) -> Hash:
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

    def is_currently_converting(self) -> bool:
        """Return True if there are inflight function conversions."""
        return len(self._inflight_function_conversions) > 0

    def get_definition_count(self) -> int:
        """Returns the total number of functions in _definitions TODO naff"""
        return len(self._definitions)

    def add_visitor(self, visitor) -> None:
        """Add a RuntimeEventVisitor to instrument the conversion process"""
        self._visitors.append(visitor)

    def remove_visitor(self, visitor) -> None:
        """Remove a RuntimeEventVisitor from the set of conversion process instruments."""
        self._visitors.remove(visitor)

    def _define_link_name(self, identity: str, link_name: str) -> bool:
        """
        Registers the identity <-> link_name mapping, and tries to load the link_name
        from the cache.

        Args:
            identity: The hash for the function
            link_name: The unique function name (prefix+func_name+hash)

        Returns:
            False if the link_name is already defined, True otherwise.
        """
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

        if link_name in self._all_defined_names:
            return False

        self._all_defined_names.add(link_name)

        self._load_from_compiler_cache(link_name)

        return True

    def _load_from_compiler_cache(self, link_name: str) -> None:
        """Attempt to load `link_name` from the cache, and add the contents of the function's
        module to relevant registers if successful.
        """
        if self.compiler_cache:
            if self.compiler_cache.hasSymbol(link_name):
                callTargetsAndTypes = self.compiler_cache.loadForSymbol(link_name)

                if callTargetsAndTypes is not None:
                    newTypedCallTargets, newNativeFunctionTypes = callTargetsAndTypes

                    self._targets.update(newTypedCallTargets)
                    self.llvm_compiler.markExternal(newNativeFunctionTypes)
                    self._all_defined_names.update(newNativeFunctionTypes)
                    self._all_cached_names.update(newNativeFunctionTypes)

    def _create_conversion_context(
        self,
        identity: str,
        func_name: str,
        func_code: types.CodeType,
        func_globals: Dict,
        func_globals_raw: Dict,
        closure_vars: List[str],
        input_types,  # as before, its not always a Wrapper, sometimes its just a type.
        output_type,  # TODO check
        conversion_type: Optional[Type[FunctionConversionContext]]
    ):
        """TODO placeholder

        Usage: in convert_python_to_native.

        Args:
            identity: the function + arguments hash.
            func_name: the name of the function
            func_code: a Code object representing the code to compile
            func_globals: the globals object from the relevant function
            func_globals_raw: the original globals object (with no merging or filtering done)
                which we use to figure out the location of global variables that are not in cells.
            closure_vars: a list of strings giving the names of closure variables
            input_types: a Wrapper for each free variable in the function closure, and
                then again for each input argument.
            output_type: the output type of the function, if known. if this is None,
                then we use type inference to produce the tightest type we can.
                If not None, then we will produce this type or throw an exception.
            conversion_type: If provided, must be a subclass of FunctionConversionContext.

        """

        ConverterType = conversion_type or FunctionConversionContext

        pyast = self._code_to_ast(func_code)

        return ConverterType(
            self,
            func_name,
            identity,
            input_types,
            output_type,
            closure_vars,
            func_globals,
            func_globals_raw,
            pyast.args,
            pyast,
        )

    def _resolve_all_inflight_functions(self):
        """TODO placeholder"""
        while True:
            identity = self._dependencies.get_next_dirty_node()
            if not identity:
                return

            link_name = self._link_name_for_identity[identity]
            if link_name in self._all_cached_names:
                continue

            function_converter = self._inflight_function_conversions[identity]

            has_definition_before_conversion = identity in self._inflight_definitions

            try:
                self._currently_converting = identity

                self._times_calculated[identity] = self._times_calculated.get(identity, 0) + 1

                native_function, actual_output_type = function_converter.convertToNativeFunction()

                if native_function is not None:
                    self._inflight_definitions[identity] = (native_function, actual_output_type)
            except Exception:
                for i in self._inflight_function_conversions:
                    if i in self._link_name_for_identity:
                        name = self._link_name_for_identity[i]
                        if name in self._targets:
                            self._targets.pop(name)
                        self._all_defined_names.discard(name)
                        ln = self._link_name_for_identity.pop(i)
                        self._identity_for_link_name.pop(ln)

                    self._dependencies.drop_node(i)
                self._inflight_function_conversions.clear()
                self._inflight_definitions.clear()
                raise
            finally:
                self._currently_converting = None

            dirty_upstream = False

            # figure out whether we ought to recalculate all the upstream nodes of this
            # node. we do that if we get a definition and we didn't have one before, or if
            # our type stability changed
            if native_function is not None:
                if not has_definition_before_conversion:
                    dirty_upstream = True

                if function_converter.typesAreUnstable():
                    function_converter.resetTypeInstabilityFlag()
                    self._dependencies.mark_dirty_with_low_priority(identity)
                    dirty_upstream = True

                name = self._link_name_for_identity[identity]

                self._targets[name] = self._get_typed_call_target(
                    name,
                    function_converter._input_types,
                    actual_output_type,
                    alwaysRaises=function_converter.alwaysRaises(),
                    functionMetadata=function_converter.functionMetadata
                )

            if dirty_upstream:
                self._dependencies.function_return_signature_changed(identity)

    def _install_inflight_functions(self):
        """placeholder"""
        if VALIDATE_FUNCTION_DEFINITIONS_STABLE:
            # this should always be true, but its expensive so we have it off by default
            for identifier, function_converter in self._inflight_function_conversions.items():
                try:
                    self._currently_converting = identifier

                    native_function, actual_output_type = function_converter.convertToNativeFunction()

                    assert native_function == self._inflight_definitions[identifier]
                finally:
                    self._currently_converting = None

        for identifier, function_converter in self._inflight_function_conversions.items():
            if identifier in self._identifier_to_pyfunc:
                for v in self._visitors:

                    func_name, func_code, func_globals, closure_vars, input_types, output_type, conversion_type = (
                        self._identifier_to_pyfunc[identifier]
                    )

                    native_function, actual_output_type = self._inflight_definitions.get(identifier)

                    try:
                        v.onNewFunction(
                            identifier,
                            function_converter,
                            native_function,
                            func_name,
                            func_code,
                            func_globals,
                            closure_vars,
                            input_types,
                            function_converter._varname_to_type.get(FunctionOutput),
                            function_converter._varname_to_type.get(FunctionYield),
                            {k: v.typeRepresentation for k, v in function_converter._varname_to_type.items() if isinstance(k, str)},
                            conversion_type
                        )
                    except Exception:
                        logging.exception("event handler %s threw an unexpected exception", v.onNewFunction)

            if identifier not in self._inflight_definitions:
                raise Exception(
                    f"Expected a definition for {identifier} depended on by:\n"
                    + "\n".join("    " + str(i) for i in self._dependencies.dependency_graph.incoming(identifier))
                )

            native_function, actual_output_type = self._inflight_definitions.get(identifier)

            name = self._link_name_for_identity[identifier]
            self._definitions[name] = native_function
            self._all_defined_names.add(name)
            self._new_native_functions.add(name)

    def _get_typed_call_target(self, name, input_types, output_type, alwaysRaises=False, functionMetadata=None) -> TypedCallTarget:
        """placeholder"""
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
            alwaysRaises=alwaysRaises,
            functionMetadata=functionMetadata
        )

        return res

    def _extract_new_function_definitions(self) -> Dict[str, native_ast.Function]:
        """Return a list of all new function definitions since the last time this function was run."""
        res = {}

        for u in self._new_native_functions:
            res[u] = self._definitions[u]

        self._new_native_functions = set()
        for key, val in res.items():
            assert type(val) == native_ast.Function
            assert type(key) == str
        return res

    def _identity_hash_to_linker_name(self, name: str, identity_hash: str, prefix: str = "tp.") -> str:
        assert isinstance(name, str)
        assert isinstance(identity_hash, str)
        assert isinstance(prefix, str)

        return prefix + name + "." + identity_hash

    def _code_to_ast(self, f: Union[types.CodeType, types.FunctionType]):
        ast_f = python_ast.convertFunctionToAlgebraicPyAst(f)
        return ast_f

    def _hash_globals(self, func_globals, code, func_globals_from_cells):
        """Hash a given piece of code's accesses to funcGlobals.

        We're trying to make sure that if we have a reference to module 'x'
        in our globals, but we only ever use 'x' by writing 'x.f' or 'x.g', then
        we shouldn't depend on the entirety of the definition of 'x'.
        """

        res = Hash.from_integer(0)

        for dot_seq in _types.getCodeGlobalDotAccesses(code):
            res += self._hash_dot_seq(dot_seq, func_globals)

        for global_name in func_globals_from_cells:
            res += self._hash_dot_seq([global_name], func_globals)

        return res

    def _hash_dot_seq(self, dot_seq, func_globals):
        """placeholder"""
        if not dot_seq or dot_seq[0] not in func_globals:
            return Hash.from_integer(0)

        item = func_globals[dot_seq[0]]

        if not isinstance(item, ModuleType) or len(dot_seq) == 1:
            return Hash.from_string(dot_seq[0]) + self.hash_object_to_identity(item, True)

        if not hasattr(item, dot_seq[1]):
            return Hash.from_integer(0)

        return Hash.from_string(dot_seq[0] + "." + dot_seq[1]) + self.hash_object_to_identity(getattr(item, dot_seq[1]), True)
