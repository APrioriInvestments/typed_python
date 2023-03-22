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

from typing import Optional, List

from typed_python.compiler.binary_shared_object import LoadedBinarySharedObject, BinarySharedObject
from typed_python.compiler.directed_graph import DirectedGraph
from typed_python.compiler.typed_call_target import TypedCallTarget
from typed_python.SerializationContext import SerializationContext
from typed_python import Dict, ListOf


def ensureDirExists(cacheDir):
    if not os.path.exists(cacheDir):
        try:
            os.makedirs(cacheDir)
        except IOError:
            # this can happen because of race conditions with
            # other writers
            pass

    if not os.path.exists(cacheDir):
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
    """
    def __init__(self, cacheDir):
        self.cacheDir = cacheDir

        ensureDirExists(cacheDir)

        self.loadedBinarySharedObjects = Dict(str, LoadedBinarySharedObject)()
        self.link_name_to_module_hash = Dict(str, str)()
        self.moduleManifestsLoaded = set()
        # link_names with an associated module in loadedBinarySharedObjects
        self.targetsLoaded = Dict(str, TypedCallTarget)()
        # the set of link_names for functions with linked and validated globals (i.e. ready to be run).
        self.targetsValidated = set()
        # link_name -> link_name
        self.function_dependency_graph = DirectedGraph()
        # dict from link_name to list of global names (should be llvm keys in serialisedGlobalDefinitions)
        self.global_dependencies = Dict(str, ListOf(str))()
        self.func_name_to_link_names = Dict(str, ListOf(str))()
        for moduleHash in os.listdir(self.cacheDir):
            if len(moduleHash) == 40:
                self.loadNameManifestFromStoredModuleByHash(moduleHash)

    def hasSymbol(self, func_name: str) -> bool:
        """Returns true if there are any versions of `func_name` in the cache.

        There may be multiple copies in different modules with different link_names.
        """
        return any(link_name in self.link_name_to_module_hash for link_name in self.func_name_to_link_names.get(func_name, []))

    def getTarget(self, func_name: str) -> TypedCallTarget:
        if not self.hasSymbol(func_name):
            raise ValueError(f'symbol not found for func_name {func_name}')
        link_name = self._select_link_name(func_name)
        self.loadForSymbol(link_name)
        return self.targetsLoaded[link_name]

    def _generate_link_name(self, func_name: str, module_hash: str) -> str:
        return func_name + "." + module_hash

    def _select_link_name(self, func_name) -> str:
        """choose a link name for a given func name.

        Currently we just choose the first available option.
        Throws a KeyError if func_name isn't in the cache.
        """
        link_name_candidates = self.func_name_to_link_names[func_name]
        return link_name_candidates[0]

    def dependencies(self, link_name: str) -> Optional[List[str]]:
        """Returns all the function names that `link_name` depends on"""
        return list(self.function_dependency_graph.outgoing(link_name))

    def loadForSymbol(self, linkName: str, parents=()) -> None:
        """Loads the whole module, and any dependant modules, into LoadedBinarySharedObjects"""
        moduleHash = self.link_name_to_module_hash[linkName]

        self.loadModuleByHash(moduleHash)

        # print()
        # print()
        # print(f"Loading for {linkName}")
        # if 'runtime.oneof_attribute.7a785' in linkName:
        #     import pdb;pdb.set_trace()

        if linkName not in self.targetsValidated:
            self.targetsValidated.add(linkName)
            for dependant_func in self.dependencies(linkName):
                self.loadForSymbol(dependant_func, parents=parents + (linkName,))

            globalsToLink = self.global_dependencies.get(linkName, [])
            if globalsToLink:
                definitionsToLink = {x: self.loadedBinarySharedObjects[moduleHash].serializedGlobalVariableDefinitions[x]
                                     for x in globalsToLink
                                     }

                # print(f"While loading for {linkName}, loaded:")
                # for k, v in definitionsToLink.items():
                #     print(k)
                #     print(SerializationContext().deserialize(v).metadata)


                for k, v in definitionsToLink.items():
                    if 'type_pointer_' in k and 'ProxyResp' in k:
                        print("Symbol ", linkName, " defined in module ", moduleHash)

                        for i in range(len(parents)):
                            print("    " * i + parents[i])
                        print("    " * len(parents) + "links to ", k)

                self.loadedBinarySharedObjects[moduleHash].linkGlobalVariables(definitionsToLink)
                if not self.loadedBinarySharedObjects[moduleHash].validateGlobalVariables(definitionsToLink):
                    raise RuntimeError('failed to validate globals when loading:', linkName)

    def loadModuleByHash(self, moduleHash: str) -> None:
        """Load a module by name.

        Add the module contents to targetsLoaded, generate a LoadedBinarySharedObject,
        and update the function and global dependency graphs.
        """
        if moduleHash in self.loadedBinarySharedObjects:
            return

        targetDir = os.path.join(self.cacheDir, moduleHash)

        # TODO (Will) - store these names as module consts, use one .dat only
        with open(os.path.join(targetDir, "type_manifest.dat"), "rb") as f:
            # func_name -> typedcalltarget
            callTargets = SerializationContext().deserialize(f.read())

        with open(os.path.join(targetDir, "globals_manifest.dat"), "rb") as f:
            serializedGlobalVarDefs = SerializationContext().deserialize(f.read())

        with open(os.path.join(targetDir, "native_type_manifest.dat"), "rb") as f:
            functionNameToNativeType = SerializationContext().deserialize(f.read())

        with open(os.path.join(targetDir, "submodules.dat"), "rb") as f:
            submodules = SerializationContext().deserialize(f.read(), ListOf(str))

        with open(os.path.join(targetDir, "function_dependencies.dat"), "rb") as f:
            dependency_edgelist = SerializationContext().deserialize(f.read())

        with open(os.path.join(targetDir, "global_dependencies.dat"), "rb") as f:
            globalDependencies = SerializationContext().deserialize(f.read())

        # load the submodules first
        for submodule in submodules:
            self.loadModuleByHash(submodule)

        print("LOADING ", moduleHash)

        modulePath = os.path.join(targetDir, "module.so")

        loaded = BinarySharedObject.fromDisk(
            modulePath,
            serializedGlobalVarDefs,
            functionNameToNativeType,
            globalDependencies
        ).loadFromPath(modulePath)

        self.loadedBinarySharedObjects[moduleHash] = loaded

        for func_name, callTarget in callTargets.items():
            link_name = self._generate_link_name(func_name, moduleHash)
            assert link_name not in self.targetsLoaded
            self.targetsLoaded[link_name] = callTarget

        link_name_global_dependencies = {self._generate_link_name(x, moduleHash): y for x, y in globalDependencies.items()}

        assert not any(key in self.global_dependencies for key in link_name_global_dependencies)

        self.global_dependencies.update(link_name_global_dependencies)
        # update the cache's dependency graph with our new edges.
        for function_name, dependant_function_name in dependency_edgelist:
            self.function_dependency_graph.addEdge(source=function_name, dest=dependant_function_name)

    def addModule(self, binarySharedObject, nameToTypedCallTarget, linkDependencies, dependencyEdgelist):
        """Add new code to the compiler cache.


        Args:
            binarySharedObject: a BinarySharedObject containing the actual assembler
                we've compiled.
            nameToTypedCallTarget: a dict from func_name to TypedCallTarget telling us
                the formal python types for all the objects.
            linkDependencies: a set of func_names we depend on directly. (this becomes submodules)
            dependencyEdgelist (list): a list of source, dest pairs giving the set of dependency graph for the
                module.

        TODO (Will): the notion of submodules/linkDependencies can be refactored out.
        """

        hashToUse = SerializationContext().sha_hash(str(uuid.uuid4())).hexdigest

        # the linkDependencies and dependencyEdgelist are in terms of func_name.
        dependentHashes = set()
        for name in linkDependencies:
            link_name = self._select_link_name(name)
            dependentHashes.add(self.link_name_to_module_hash[link_name])

        link_name_dependency_edgelist = []
        for source, dest in dependencyEdgelist:
            assert source in binarySharedObject.definedSymbols
            source_link_name = self._generate_link_name(source, hashToUse)
            if dest in binarySharedObject.definedSymbols:
                dest_link_name = self._generate_link_name(dest, hashToUse)
            else:
                dest_link_name = self._select_link_name(dest)
            link_name_dependency_edgelist.append([source_link_name, dest_link_name])

        path = self.writeModuleToDisk(binarySharedObject, hashToUse, nameToTypedCallTarget, dependentHashes, link_name_dependency_edgelist)

        self.loadedBinarySharedObjects[hashToUse] = (
            binarySharedObject.loadFromPath(os.path.join(path, "module.so"))
        )

        for func_name in binarySharedObject.definedSymbols:
            link_name = self._generate_link_name(func_name, hashToUse)
            self.link_name_to_module_hash[link_name] = hashToUse
            self.func_name_to_link_names.setdefault(func_name, []).append(link_name)

        # link & validate all globals for the new module
        self.loadedBinarySharedObjects[hashToUse].linkGlobalVariables()
        if not self.loadedBinarySharedObjects[hashToUse].validateGlobalVariables(
                self.loadedBinarySharedObjects[hashToUse].serializedGlobalVariableDefinitions):
            raise RuntimeError('failed to validate globals in new module:', hashToUse)

    def loadNameManifestFromStoredModuleByHash(self, moduleHash) -> None:
        if moduleHash in self.moduleManifestsLoaded:
            return

        targetDir = os.path.join(self.cacheDir, moduleHash)

        # TODO (Will) the name_manifest module_hash is the same throughout so this doesn't need to be a dict.
        with open(os.path.join(targetDir, "name_manifest.dat"), "rb") as f:
            func_name_to_module_hash = SerializationContext().deserialize(f.read(), Dict(str, str))

            for func_name, module_hash in func_name_to_module_hash.items():
                link_name = self._generate_link_name(func_name, module_hash)
                self.func_name_to_link_names.setdefault(func_name, []).append(link_name)
                self.link_name_to_module_hash[link_name] = module_hash

        self.moduleManifestsLoaded.add(moduleHash)

    def writeModuleToDisk(self, binarySharedObject, hashToUse, nameToTypedCallTarget, submodules, dependencyEdgelist):
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
        """

        targetDir = os.path.join(
            self.cacheDir,
            hashToUse
        )

        assert not os.path.exists(targetDir)

        tempTargetDir = targetDir + "_" + str(uuid.uuid4())
        ensureDirExists(tempTargetDir)

        # write the binary module
        with open(os.path.join(tempTargetDir, "module.so"), "wb") as f:
            f.write(binarySharedObject.binaryForm)

        # write the manifest. Every TP process using the cache will have to
        # load the manifest every time, so we try to use compiled code to load it
        manifest = Dict(str, str)()
        for n in binarySharedObject.functionTypes:
            manifest[n] = hashToUse

        with open(os.path.join(tempTargetDir, "name_manifest.dat"), "wb") as f:
            f.write(SerializationContext().serialize(manifest, Dict(str, str)))

        with open(os.path.join(tempTargetDir, "name_manifest.txt"), "w") as f:
            for sourceName in manifest:
                f.write(sourceName + "\n")

        with open(os.path.join(tempTargetDir, "type_manifest.dat"), "wb") as f:
            f.write(SerializationContext().serialize(nameToTypedCallTarget))

        with open(os.path.join(tempTargetDir, "native_type_manifest.dat"), "wb") as f:
            f.write(SerializationContext().serialize(binarySharedObject.functionTypes))

        with open(os.path.join(tempTargetDir, "globals_manifest.dat"), "wb") as f:
            f.write(SerializationContext().serialize(binarySharedObject.serializedGlobalVariableDefinitions))

        with open(os.path.join(tempTargetDir, "submodules.dat"), "wb") as f:
            f.write(SerializationContext().serialize(ListOf(str)(submodules), ListOf(str)))

        with open(os.path.join(tempTargetDir, "function_dependencies.dat"), "wb") as f:
            f.write(SerializationContext().serialize(dependencyEdgelist))

        with open(os.path.join(tempTargetDir, "global_dependencies.dat"), "wb") as f:
            f.write(SerializationContext().serialize(binarySharedObject.globalDependencies))

        try:
            os.rename(tempTargetDir, targetDir)
        except IOError:
            if not os.path.exists(targetDir):
                raise
            else:
                shutil.rmtree(tempTargetDir)

        return targetDir

    def function_pointer_by_name(self, func_name):
        linkName = self._select_link_name(func_name)
        moduleHash = self.link_name_to_module_hash.get(linkName)
        if moduleHash is None:
            raise Exception("Can't find a module for " + linkName)

        # if moduleHash not in self.loadedBinarySharedObjects:
        if linkName not in self.targetsValidated:
            self.loadForSymbol(linkName)

        return self.loadedBinarySharedObjects[moduleHash].functionPointers[func_name]
