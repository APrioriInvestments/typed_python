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

import os
import uuid
import shutil
import llvmlite.ir

from typing import List

from typed_python.compiler.binary_shared_object import LoadedBinarySharedObject, BinarySharedObject
from typed_python.compiler.directed_graph import DirectedGraph
from typed_python.compiler.native_function_pointer import NativeFunctionPointer
from typed_python.compiler.typed_call_target import TypedCallTarget
import typed_python.compiler.native_ast as native_ast
from typed_python.SerializationContext import SerializationContext
from typed_python import Dict, ListOf


def _ensure_dir_exists(cache_dir):
    if not os.path.exists(cache_dir):
        try:
            os.makedirs(cache_dir)
        except IOError:
            # this can happen because of race conditions with
            # other writers
            pass

    if not os.path.exists(cache_dir):
        raise Exception("Failed to create the cache directory.")


class CompilerCache:
    """Implements an on-disk cache of compiled code.

    This is a pretty simple implementation - it needs to be threadsafe,
    which we achieve by only ever writing to it, and using directory renames
    to guarantee atomicity.

    The biggest drawback here is that we have to load a bunch of 'manifest' objects
    when we first boot up, which could be slow. We could improve this substantially
    by making it possible to determine if a given function is in the cache by organizing
    the manifests by, say, function name.

    Due to the potential for race conditions, we must distinguish between the following:
        func_name -  The identifier for the function, based on its identity hash.
        link_name -  The identifier for the specific realization of that function, which lives in a specific
            cache module.

    Attributes:
        cache_dir: the relative path to the cache directory.
        loaded_binary_shared_objects: a mapping from a module hash to a LoadedBinarySharedObject.
        name_to_module_hash: a mapping from link_name (defining a function) to the module hash
            it's contained in. (assumes each function only defined in one module).
        modules_marked_valid: a set of module hashes which have sucessfully been loaded
            using _load_name_manifest_from_stored_module_by_hash.
        modules_marked_invalid: a set of module hashes which have marked_invalid in their
            associated directory. NB: this attribute is never referenced.
        initialised: True if the initialise() method has been run, setting the cache dir and
            loading all manifest files.
        added_modules: a set of module hashes which have been added during this compiler_caches
            lifetime, rather than loaded on initialisation. TODO fix this
    """
    def __init__(self, cache_dir):
        self.cache_dir = cache_dir

        _ensure_dir_exists(cache_dir)

        self.loaded_binary_shared_objects = Dict(str, LoadedBinarySharedObject)()
        self.link_name_to_module_hash = Dict(str, str)()
        self.module_manifests_loaded = set()
        # link_names with an associated module in loadedBinarySharedObjects
        self.targets_loaded: Dict[str, TypedCallTarget] = {}
        # the set of link_names for functions with linked and validated globals (i.e. ready to be run).
        self.targets_validated = set()
        # the total number of instructions for each link_name
        self.target_complexity = Dict(str, int)()
        # link_name -> link_name
        self.function_dependency_graph = DirectedGraph()
        # dict from link_name to list of global names (should be llvm keys in serialisedGlobalDefinitions)
        self.global_dependencies = Dict(str, ListOf(str))()
        self.func_name_to_link_names = Dict(str, ListOf(str))()
        for module_hash in os.listdir(self.cache_dir):
            if len(module_hash) == 40:
                self._load_name_manifest_from_stored_module_by_hash(module_hash)

    def has_symbol(self, func_name: str) -> bool:
        """Returns True if there are any versions of `func_name` in the cache.

        There may be multiple copies in different modules with different link_names.
        """
        return any(link_name in self.link_name_to_module_hash for link_name in self.func_name_to_link_names.get(func_name, []))

    def get_target(self, func_name: str) -> TypedCallTarget:
        if not self.has_symbol(func_name):
            raise ValueError(f'symbol not found for func_name {func_name}')
        link_name = self._select_link_name(func_name)
        self._load_for_symbol(link_name)
        return self.targets_loaded[link_name]

    def get_IR(self, func_name: str) -> llvmlite.ir.Function:
        if not self.has_symbol(func_name):
            raise ValueError(f'symbol not found for func_name {func_name}')
        link_name = self._select_link_name(func_name)
        module_hash = self.link_name_to_module_hash[link_name]
        return self.loaded_binary_shared_objects[module_hash].binarySharedObject.functionIRs[func_name]

    def get_definition(self, func_name: str) -> native_ast.Function:
        if not self.has_symbol(func_name):
            raise ValueError(f'symbol not found for func_name {func_name}')
        link_name = self._select_link_name(func_name)
        module_hash = self.link_name_to_module_hash[link_name]
        serialized_definition = self.loaded_binary_shared_objects[module_hash].binarySharedObject.serializedFunctionDefinitions[func_name]
        return SerializationContext().deserialize(serialized_definition)

    def _generate_link_name(self, func_name: str, module_hash: str) -> str:
        return func_name + "." + module_hash

    def _select_link_name(self, func_name) -> str:
        """choose a link name for a given func name.

        Currently we just choose the first available option.
        Throws a KeyError if func_name isn't in the cache.
        """
        link_name_candidates = self.func_name_to_link_names[func_name]
        return link_name_candidates[0]

    def dependencies(self, link_name: str) -> List[str]:
        """Returns all the function names that `link_name` depends on, or an empty list."""
        return list(self.function_dependency_graph.outgoing(link_name))

    def _load_for_symbol(self, link_name: str) -> None:
        """Loads the whole module, and any dependant modules, into LoadedBinarySharedObjects.

        Link only the necessary globals for the link_name.
        """
        module_hash = self.link_name_to_module_hash[link_name]

        self._load_module_by_hash(module_hash)

        if link_name not in self.targets_validated:
            self.targets_validated.add(link_name)
            for dependant_func in self.dependencies(link_name):
                self._load_for_symbol(dependant_func)

            globals_to_link = self.global_dependencies.get(link_name, [])
            if globals_to_link:
                definitions_to_link = {x: self.loaded_binary_shared_objects[module_hash].serializedGlobalVariableDefinitions[x]
                                       for x in globals_to_link
                                       }
                self.loaded_binary_shared_objects[module_hash].linkGlobalVariables(definitions_to_link)
                if not self.loaded_binary_shared_objects[module_hash].validateGlobalVariables(definitions_to_link):
                    raise RuntimeError('failed to validate globals when loading:', link_name)

    def complexity_for_symbol(self, func_name: str) -> int:
        """Get the total number of LLVM instructions for a given symbol."""
        try:
            link_name = self._select_link_name(func_name)
            return self.target_complexity[link_name]
        except KeyError as e:
            raise ValueError(f'No complexity value cached for {func_name}') from e

    def _load_module_by_hash(self, module_hash: str) -> None:
        """Load a module by name.

        Add the module contents to targetsLoaded, generate a LoadedBinarySharedObject,
        and update the function and global dependency graphs.
        """
        if module_hash in self.loaded_binary_shared_objects:
            return

        target_dir = os.path.join(self.cache_dir, module_hash)

        # TODO (Will) - store these names as module consts, use one .dat only
        with open(os.path.join(target_dir, "type_manifest.dat"), "rb") as f:
            call_targets = SerializationContext().deserialize(f.read())
        with open(os.path.join(target_dir, "globals_manifest.dat"), "rb") as f:
            serialized_global_var_defs = SerializationContext().deserialize(f.read())
        with open(os.path.join(target_dir, "native_type_manifest.dat"), "rb") as f:
            function_name_to_native_type = SerializationContext().deserialize(f.read())
        with open(os.path.join(target_dir, "submodules.dat"), "rb") as f:
            submodules = SerializationContext().deserialize(f.read(), ListOf(str))
        with open(os.path.join(target_dir, "function_dependencies.dat"), "rb") as f:
            dependency_edgelist = SerializationContext().deserialize(f.read())
        with open(os.path.join(target_dir, "global_dependencies.dat"), "rb") as f:
            global_dependencies = SerializationContext().deserialize(f.read())
        with open(os.path.join(target_dir, "function_complexities.dat"), "rb") as f:
            function_complexities = SerializationContext().deserialize(f.read())
        with open(os.path.join(target_dir, "function_irs.dat"), "rb") as f:
            function_IRs = SerializationContext().deserialize(f.read())
        with open(os.path.join(target_dir, "function_definitions.dat"), "rb") as f:
            function_definitions = SerializationContext().deserialize(f.read())

        # load the submodules first
        for submodule in submodules:
            self._load_module_by_hash(submodule)

        module_path = os.path.join(target_dir, "module.so")

        loaded = BinarySharedObject.fromDisk(
            module_path,
            serialized_global_var_defs,
            function_name_to_native_type,
            global_dependencies,
            function_complexities,
            function_IRs,
            function_definitions
        ).loadFromPath(module_path)

        self.loaded_binary_shared_objects[module_hash] = loaded

        for func_name, call_target in call_targets.items():
            link_name = self._generate_link_name(func_name, module_hash)
            assert link_name not in self.targets_loaded
            self.targets_loaded[link_name] = call_target

        for func_name, complexity in function_complexities.items():
            link_name = self._generate_link_name(func_name, module_hash)
            self.target_complexity[link_name] = complexity

        link_name_global_dependencies = {self._generate_link_name(x, module_hash): y for x, y in global_dependencies.items()}
        assert not any(key in self.global_dependencies for key in link_name_global_dependencies)

        self.global_dependencies.update(link_name_global_dependencies)
        # update the cache's dependency graph with our new edges.
        for function_name, dependant_function_name in dependency_edgelist:
            self.function_dependency_graph.addEdge(source=function_name, dest=dependant_function_name)

    def add_module(self, binary_shared_object, name_to_typed_call_target, link_dependencies, dependency_edge_list):
        """Add new code to the compiler cache.

        Generate the link_name to link_name dependency graph, and write the binary_shared_object to disk
        along with its types, and dependency mappings. Then load that object back from disk into the set of
        loaded binary_shared_objects, link & validate everything, and update the set of loaded symbols accordingly.

        Args:
            binary_shared_object: a BinarySharedObject containing the actual assembler
                we've compiled.
            name_to_typed_call_target: a dict from func_name to TypedCallTarget telling us
                the formal python types for all the objects.
            link_dependencies: a set of func_names we depend on directly. (this becomes submodules)
            dependency_edge_list (list): a list of source, dest pairs giving the set of dependency graph for the
                module.

        TODO (Will): the notion of submodules/linkDependencies can be refactored out.
        """

        hash_to_use = SerializationContext().sha_hash(str(uuid.uuid4())).hexdigest

        # the linkDependencies and dependencyEdgelist are in terms of func_name.
        dependent_hashes = set()
        for name in link_dependencies:
            link_name = self._select_link_name(name)
            dependent_hashes.add(self.link_name_to_module_hash[link_name])

        link_name_dependency_edgelist = []
        for source, dest in dependency_edge_list:
            assert source in binary_shared_object.definedSymbols
            source_link_name = self._generate_link_name(source, hash_to_use)
            if dest in binary_shared_object.definedSymbols:
                dest_link_name = self._generate_link_name(dest, hash_to_use)
            else:
                dest_link_name = self._select_link_name(dest)
            link_name_dependency_edgelist.append([source_link_name, dest_link_name])

        path = self._write_module_to_disk(binary_shared_object,
                                          hash_to_use,
                                          name_to_typed_call_target,
                                          dependent_hashes,
                                          link_name_dependency_edgelist)

        for func_name, complexity in binary_shared_object.functionComplexities.items():
            link_name = self._generate_link_name(func_name, hash_to_use)
            self.target_complexity[link_name] = complexity

        self.loaded_binary_shared_objects[hash_to_use] = (
            binary_shared_object.loadFromPath(os.path.join(path, "module.so"))
        )

        for func_name in binary_shared_object.definedSymbols:
            link_name = self._generate_link_name(func_name, hash_to_use)
            self.link_name_to_module_hash[link_name] = hash_to_use
            self.func_name_to_link_names.setdefault(func_name, []).append(link_name)

        # link & validate all globals for the new module
        self.loaded_binary_shared_objects[hash_to_use].linkGlobalVariables()
        if not self.loaded_binary_shared_objects[hash_to_use].validateGlobalVariables(
                self.loaded_binary_shared_objects[hash_to_use].serializedGlobalVariableDefinitions):
            raise RuntimeError('failed to validate globals in new module:', hash_to_use)

    def _load_name_manifest_from_stored_module_by_hash(self, module_hash) -> None:
        """
        Initialise the cache by reading the module from disk, populating the list of cached functions.

        Args:
            module_hash: 40-character string representing a directory, with a name_manifest.dat
                specifying the functions within.
        """
        if module_hash in self.module_manifests_loaded:
            return

        targetDir = os.path.join(self.cache_dir, module_hash)

        # TODO (Will) the name_manifest module_hash is the same throughout so this doesn't need to be a dict.
        with open(os.path.join(targetDir, "name_manifest.dat"), "rb") as f:
            func_name_to_module_hash = SerializationContext().deserialize(f.read(), Dict(str, str))

            for func_name, module_hash in func_name_to_module_hash.items():
                link_name = self._generate_link_name(func_name, module_hash)
                self.func_name_to_link_names.setdefault(func_name, []).append(link_name)
                self.link_name_to_module_hash[link_name] = module_hash

        self.module_manifests_loaded.add(module_hash)

    def _write_module_to_disk(self, binary_shared_object, hash_to_use, name_to_typed_call_target, submodules, dependency_edgelist):
        """Write out a disk representation of this module.

        This includes writing both the shared object, a manifest of the function names
        to the module path (so when we start up we can read from the compiler cache),
        and the typed call targets and global definitions, so we know how to link the
        objects when we do load them.

        In order to make this atomic, we store data for each module in its own
        directory, which we write out under a tempname and then rename to the
        proper name in case we see conflicts. This allows multiple processes
        to interact with the compiler cache simultaneously without relying on
        individual file-level locking.

        Args:
            binary_shared_object: The compiled module code.
            hash_to_use: The unique hash for the module contents.
            name_to_typed_call_target: A dict from linkname to TypedCallTarget telling us
                the formal python types for all the objects.
            submodules: The list of hashes of dependent modules.
            dependency_edgelist: The function dependency graph for all functions in this module.
        Returns:
            The absolute path to the new module in the cache.
        """

        target_dir = os.path.join(
            self.cache_dir,
            hash_to_use
        )

        assert not os.path.exists(target_dir)

        temp_target_dir = target_dir + "_" + str(uuid.uuid4())
        _ensure_dir_exists(temp_target_dir)

        # write the binary module
        with open(os.path.join(temp_target_dir, "module.so"), "wb") as f:
            f.write(binary_shared_object.binaryForm)

        # write the manifest. Every TP process using the cache will have to
        # load the manifest every time, so we try to use compiled code to load it
        manifest = Dict(str, str)()
        for n in binary_shared_object.functionTypes:
            manifest[n] = hash_to_use

        with open(os.path.join(temp_target_dir, "name_manifest.dat"), "wb") as f:
            f.write(SerializationContext().serialize(manifest, Dict(str, str)))

        with open(os.path.join(temp_target_dir, "name_manifest.txt"), "w") as f:
            for source_name in manifest:
                f.write(source_name + "\n")

        with open(os.path.join(temp_target_dir, "type_manifest.dat"), "wb") as f:
            f.write(SerializationContext().serialize(name_to_typed_call_target))

        with open(os.path.join(temp_target_dir, "native_type_manifest.dat"), "wb") as f:
            f.write(SerializationContext().serialize(binary_shared_object.functionTypes))

        with open(os.path.join(temp_target_dir, "globals_manifest.dat"), "wb") as f:
            f.write(SerializationContext().serialize(binary_shared_object.serializedGlobalVariableDefinitions))

        with open(os.path.join(temp_target_dir, "submodules.dat"), "wb") as f:
            f.write(SerializationContext().serialize(ListOf(str)(submodules), ListOf(str)))

        with open(os.path.join(temp_target_dir, "function_dependencies.dat"), "wb") as f:
            f.write(SerializationContext().serialize(dependency_edgelist))

        with open(os.path.join(temp_target_dir, "global_dependencies.dat"), "wb") as f:
            f.write(SerializationContext().serialize(binary_shared_object.globalDependencies))

        with open(os.path.join(temp_target_dir, "function_complexities.dat"), "wb") as f:
            f.write(SerializationContext().serialize(binary_shared_object.functionComplexities))

        with open(os.path.join(temp_target_dir, "function_irs.dat"), "wb") as f:
            f.write(SerializationContext().serialize(binary_shared_object.functionIRs))

        with open(os.path.join(temp_target_dir, "function_definitions.dat"), "wb") as f:
            f.write(SerializationContext().serialize(binary_shared_object.serializedFunctionDefinitions))

        try:
            os.rename(temp_target_dir, target_dir)
        except IOError:
            if not os.path.exists(target_dir):
                raise
            else:
                shutil.rmtree(temp_target_dir)

        return target_dir

    def function_pointer_by_name(self, func_name: str) -> NativeFunctionPointer:
        """
        Find the module hash associated with an instance of <func_name>. If not loaded, load the module,
        and return the associated function pointer.
        Args:
            func_name: the symbol name, normally of the form <module>.<function_name>.<unique_hash>
        Returns:
           The NativeFunctionPointer for that symbol.
        """
        linkName = self._select_link_name(func_name)
        module_hash = self.link_name_to_module_hash.get(linkName)
        if module_hash is None:
            raise Exception("Can't find a module for " + linkName)

        if module_hash not in self.loaded_binary_shared_objects:
            self._load_for_symbol(linkName)

        return self.loaded_binary_shared_objects[module_hash].functionPointers[func_name]
