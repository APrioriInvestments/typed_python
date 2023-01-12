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
# from typed_python.compiler.loaded_module import LoadedModule
from typed_python.compiler.binary_shared_object import LoadedBinarySharedObject, BinarySharedObject

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
    """
    def __init__(self, cacheDir):
        self.cacheDir = cacheDir

        ensureDirExists(cacheDir)

        self.loadedBinarySharedObjects = Dict(str, LoadedBinarySharedObject)()
        self.nameToModuleHash = Dict(str, str)()

        # self.modulesMarkedValid = set()
        # self.modulesMarkedInvalid = set()

        for moduleHash in os.listdir(self.cacheDir):
            if len(moduleHash) == 40:
                self.loadNameManifestFromStoredModuleByHash(moduleHash)

        # the module for this function has been read into a LoadedBinaryObject
        self.targetsLoaded = {}  # : dict[str, CallTarget]

        # the function's globals have been linked and validated and it is good to go.
        self.targetsValidated = set()

    def hasSymbol(self, linkName: str) -> bool:
        """NB this will return True even if the linkName is ultimately unretrievable."""
        return linkName in self.nameToModuleHash

    def getTarget(self, linkName):

        assert self.hasSymbol(linkName)

        self.loadForSymbol(linkName)

        return self.targetsLoaded[linkName]

    # def markModuleHashInvalid(self, hashstr):
    #     with open(os.path.join(self.cacheDir, hashstr, "marked_invalid"), "w"):
    #         pass

    def loadForSymbol(self, linkName: str) -> None:
        """Loads the whole module, and any submodules, into LoadedBinarySharedObjects"""
        moduleHash = self.nameToModuleHash[linkName]

        self.loadModuleByHash(moduleHash)

        # TODO - the linking and validation for the linkName and its dependencies.
        if linkName not in self.targetsValidated:
            dependentFuncs = self.dependants(linkName)  # optimise - just the unlinked dependents?
            newGlobalVars = self.getNecessaryGlobals([linkName] + dependentFuncs)
            self.loadedBinarySharedObjects[moduleHash].linkGlobalVariables(newGlobalVars)  # this doesn't account for cross-module walk.
            self.loadedBinarySharedObjects[moduleHash].validateGlobalVariables(newGlobalVars)
            self.targetsValidated.update(dependentFuncs)

    def loadModuleByHash(self, moduleHash: str) -> None:
        """Load a module by name.

        As we load, place all the newly imported typed call targets into
        'nameToTypedCallTarget' so that the rest of the system knows what functions
        have been uncovered.
        """
        if moduleHash in self.loadedBinarySharedObjects:
            return   # True

        targetDir = os.path.join(self.cacheDir, moduleHash)

        with open(os.path.join(targetDir, "type_manifest.dat"), "rb") as f:
            callTargets = SerializationContext().deserialize(f.read())

        with open(os.path.join(targetDir, "globals_manifest.dat"), "rb") as f:
            serializedGlobalVarDefs = SerializationContext().deserialize(f.read())

        with open(os.path.join(targetDir, "native_type_manifest.dat"), "rb") as f:
            functionNameToNativeType = SerializationContext().deserialize(f.read())

        with open(os.path.join(targetDir, "submodules.dat"), "rb") as f:
            submodules = SerializationContext().deserialize(f.read(), ListOf(str))

        # dependency graph and global dependency graph here.

        # if not LoadedModule.validateGlobalVariables(globalVarDefs):
        #     self.markModuleHashInvalid(moduleHash)
        #     return False

        # load the submodules first
        for submodule in submodules:
            # if not self.loadModuleByHash(submodule):
            #     return False
            self.loadModuleByHash(submodule)

        modulePath = os.path.join(targetDir, "module.so")

        loaded = BinarySharedObject.fromDisk(
            modulePath,
            serializedGlobalVarDefs,  # dependencies and global dependencies? no.
            functionNameToNativeType
        ).loadFromPath(modulePath)

        self.loadedBinarySharedObjects[moduleHash] = loaded

        self.targetsLoaded.update(callTargets)

        # return True

    def addModule(self, binarySharedObject, nameToTypedCallTarget, linkDependencies):
        """Add new code to the compiler cache.

        Args:
            binarySharedObject - a BinarySharedObject containing the actual assembler
                we've compiled
            nameToTypedCallTarget - a dict from linkname to TypedCallTarget telling us
                the formal python types for all the objects
            linkDependencies - a set of linknames we depend on directly.
        """
        dependentHashes = set()

        for name in linkDependencies:
            dependentHashes.add(self.nameToModuleHash[name])

        path, hashToUse = self.writeModuleToDisk(binarySharedObject, nameToTypedCallTarget, dependentHashes)

        self.loadedBinarySharedObjects[hashToUse] = (
            binarySharedObject.loadFromPath(os.path.join(path, "module.so"))
        )

        for n in binarySharedObject.definedSymbols:
            self.nameToModuleHash[n] = hashToUse

    def loadNameManifestFromStoredModuleByHash(self, moduleHash):
        if moduleHash in self.modulesMarkedValid:
            return True

        targetDir = os.path.join(self.cacheDir, moduleHash)

        # ignore 'marked invalid'
        if os.path.exists(os.path.join(targetDir, "marked_invalid")):
            # just bail - don't try to read it now

            # for the moment, we don't try to clean up the cache, because
            # we can't be sure that some process is not still reading the
            # old files.
            self.modulesMarkedInvalid.add(moduleHash)
            return False

        with open(os.path.join(targetDir, "submodules.dat"), "rb") as f:
            submodules = SerializationContext().deserialize(f.read(), ListOf(str))

        for subHash in submodules:
            if not self.loadNameManifestFromStoredModuleByHash(subHash):
                self.markModuleHashInvalid(subHash)
                return False

        with open(os.path.join(targetDir, "name_manifest.dat"), "rb") as f:
            self.nameToModuleHash.update(
                SerializationContext().deserialize(f.read(), Dict(str, str))
            )

        self.modulesMarkedValid.add(moduleHash)

        return True

    def writeModuleToDisk(self, binarySharedObject, nameToTypedCallTarget, submodules):
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
        hashToUse = SerializationContext().sha_hash(str(uuid.uuid4())).hexdigest

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

        # write the type manifest
        with open(os.path.join(tempTargetDir, "type_manifest.dat"), "wb") as f:
            f.write(SerializationContext().serialize(nameToTypedCallTarget))

        # write the nativetype manifest
        with open(os.path.join(tempTargetDir, "native_type_manifest.dat"), "wb") as f:
            f.write(SerializationContext().serialize(binarySharedObject.functionTypes))

        # write the type manifest
        with open(os.path.join(tempTargetDir, "globals_manifest.dat"), "wb") as f:
            f.write(SerializationContext().serialize(binarySharedObject.globalVariableDefinitions))

        with open(os.path.join(tempTargetDir, "submodules.dat"), "wb") as f:
            f.write(SerializationContext().serialize(ListOf(str)(submodules), ListOf(str)))

        try:
            os.rename(tempTargetDir, targetDir)
        except IOError:
            if not os.path.exists(targetDir):
                raise
            else:
                shutil.rmtree(tempTargetDir)

        return targetDir, hashToUse

    def function_pointer_by_name(self, linkName):
        moduleHash = self.nameToModuleHash.get(linkName)
        if moduleHash is None:
            raise Exception("Can't find a module for " + linkName)

        if moduleHash not in self.loadedBinarySharedObjects:
            self.loadForSymbol(linkName)

        return self.loadedBinarySharedObjects[moduleHash].functionPointers[linkName]


# class CacheDependencyGraph:
#     """
#     Holds the directed dependency graph for the functions in the compiler cache.
#     TODO - annotate the graph with loadable/not loadable, deprecate mark_invalid, account
#         for required global variables
#     TODO allow for partial graph creation, by only loading the dependencies as required.
#     """

#     def __init__(self, compiler_cache):
#         self._compiler_cache = compiler_cache
#         self._edgelist = []
#         self._directed_graph = None
#         self._module_dependencies_loaded = set()

#     @property
#     def directed_graph(self):
#         if self._directed_graph is None:
#             self._directed_graph = self._compute_full_graph()
#         return self._directed_graph

#     def _read_module_dependency_graph(self, module_hash: str) -> ListOf(ListOf(str)):
#         """deserialise the edgelist corresponding to the module_hash in the cache dir."""
#         target_dir = os.path.join(self._compiler_cache.cacheDir, module_hash)

#         with open(os.path.join(target_dir, "dependency_graph.dat"), "rb") as f:
#             edge_list = SerializationContext().deserialize(
#                 f.read(), ListOf(ListOf(str))
#             )
#         return edge_list

#     def _compute_full_graph(self) -> nx.DiGraph:
#         """Read every unread module's dependency graph and collate."""
#         for module_hash in os.listdir(self._compiler_cache.cacheDir):
#             if (
#                 len(module_hash) == 40
#                 and module_hash not in self._module_dependencies_loaded
#             ):
#                 self._edgelist += self._read_module_dependency_graph(module_hash)
#                 self._module_dependencies_loaded.add(module_hash)

#         full_graph = nx.DiGraph()

#         # add the function/global distinction
#         for source, dest, edge_type in self._edgelist:
#             full_graph.add_edge(source, dest)
#             full_graph.nodes[source]["is_global"] = False
#             if edge_type == "global":
#                 full_graph.nodes[dest]["is_global"] = True  # horrendous coding.

#         full_graph.remove_node("None")

#         return full_graph
