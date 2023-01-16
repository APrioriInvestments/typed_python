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

# from typed_python.compiler.loaded_module import LoadedModule
from typed_python.compiler.binary_shared_object import LoadedBinarySharedObject, BinarySharedObject
from typed_python.compiler.directed_graph import DirectedGraph
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

        self.moduleManifestsLoaded = set()
        # self.modulesMarkedInvalid = set()

        for moduleHash in os.listdir(self.cacheDir):
            if len(moduleHash) == 40:
                self.loadNameManifestFromStoredModuleByHash(moduleHash)

        # the module for this function has been read into a LoadedBinaryObject
        self.targetsLoaded = {}  # : dict[str, CallTarget]

        # the function's globals have been linked and validated and it is good to go.
        self.targetsValidated = set()

        # the total number of instructions for each linkName.
        self.targetComplexity = {}

        # function dependency graph. DirectedGraph?
        self.function_dependency_graph = DirectedGraph()
        # dict from function linkname to list of global names (should be llvm keys in serialisedGlobalDefinitions)
        self.global_dependencies = Dict(str, ListOf(str))()

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

    def dependencies(self, linkName: str) -> Optional[List[str]]:
        """Returns all the function names that `linkName` depends on"""
        return list(self.function_dependency_graph.outgoing(linkName))

    def loadForSymbol(self, linkName: str) -> None:
        """Loads the whole module, and any submodules, into LoadedBinarySharedObjects"""
        moduleHash = self.nameToModuleHash[linkName]

        self.loadModuleByHash(moduleHash)

        # TODO - the linking and validation for the linkName and its dependencies.
        if linkName not in self.targetsValidated:
            dependantFuncs = self.dependencies(linkName) + [linkName]
            globalsToLink = {}  # this is a dict from modulehash to list of globals.
            for funcName in dependantFuncs:
                if funcName not in self.targetsValidated:
                    funcModuleHash = self.nameToModuleHash[funcName]
                    # append to the list of globals to link for a given module.  TODO: optimise this, don't double-link.
                    globalsToLink[funcModuleHash] = globalsToLink.get(funcModuleHash, []) + self.global_dependencies.get(funcName, [])

            for moduleHash, globs in globalsToLink.items():  # this works because loadModuleByHash loads submodules too.
                if globs:
                    definitionsToLink = {x: self.loadedBinarySharedObjects[moduleHash].serializedGlobalVariableDefinitions[x] for x in globs}
                    self.loadedBinarySharedObjects[moduleHash].linkGlobalVariables(definitionsToLink)
                    if not self.loadedBinarySharedObjects[moduleHash].validateGlobalVariables(definitionsToLink):
                        raise RuntimeError('failed to validate globals when loading:', linkName)

            self.targetsValidated.update(dependantFuncs)

    def complexityForSymbol(self, linkName: str) -> int:
        """Get the total number of instructions for a given symbol (cached when first compiled)."""
        try:
            return self.targetComplexity[linkName]
        except KeyError as e:
            raise ValueError(f'No complexity value cached for {linkName}') from e

    def loadModuleByHash(self, moduleHash: str) -> None:
        """Load a module by name.

        As we load, place all the newly imported typed call targets into
        'nameToTypedCallTarget' so that the rest of the system knows what functions
        have been uncovered.
        """
        if moduleHash in self.loadedBinarySharedObjects:
            return

        targetDir = os.path.join(self.cacheDir, moduleHash)

        # TODO (Will) - store these names as module consts.
        with open(os.path.join(targetDir, "type_manifest.dat"), "rb") as f:
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

        with open(os.path.join(targetDir, "function_complexities.dat"), "rb") as f:
            functionComplexities = SerializationContext().deserialize(f.read())

        # load the submodules first
        for submodule in submodules:
            self.loadModuleByHash(submodule)

        modulePath = os.path.join(targetDir, "module.so")

        loaded = BinarySharedObject.fromDisk(
            modulePath,
            serializedGlobalVarDefs,
            functionNameToNativeType,
            globalDependencies,
            functionComplexities

        ).loadFromPath(modulePath)

        self.loadedBinarySharedObjects[moduleHash] = loaded

        self.targetsLoaded.update(callTargets)
        self.targetComplexity.update(functionComplexities)
        assert not any(key in self.global_dependencies for key in globalDependencies)  # should only happen if there's a hash collision.
        self.global_dependencies.update(globalDependencies)

        # update the cache's dependency graph with our new edges.
        for function_name, dependant_function_name in dependency_edgelist:
            self.function_dependency_graph.addEdge(source=function_name, dest=dependant_function_name)


    def addModule(self, binarySharedObject, nameToTypedCallTarget, linkDependencies, dependencyEdgelist):
        """Add new code to the compiler cache.

        Args:
            binarySharedObject: a BinarySharedObject containing the actual assembler
                we've compiled.
            nameToTypedCallTarget: a dict from linkname to TypedCallTarget telling us
                the formal python types for all the objects.
            linkDependencies: a set of linknames we depend on directly.
            dependencyEdgelist (list): a list of source, dest pairs giving the set of dependency graph for the
                module.
        """
        dependentHashes = set()

        for name in linkDependencies:
            dependentHashes.add(self.nameToModuleHash[name])

        path, hashToUse = self.writeModuleToDisk(binarySharedObject, nameToTypedCallTarget, dependentHashes, dependencyEdgelist)

        self.loadedBinarySharedObjects[hashToUse] = (
            binarySharedObject.loadFromPath(os.path.join(path, "module.so"))
        )

        for n in binarySharedObject.definedSymbols:
            self.nameToModuleHash[n] = hashToUse

        # link & validate all globals for the new module
        self.loadedBinarySharedObjects[hashToUse].linkGlobalVariables()
        if not self.loadedBinarySharedObjects[hashToUse].validateGlobalVariables(
                self.loadedBinarySharedObjects[hashToUse].serializedGlobalVariableDefinitions):
            raise RuntimeError('failed to validate globals in new module:', hashToUse)

    def loadNameManifestFromStoredModuleByHash(self, moduleHash):
        if moduleHash in self.moduleManifestsLoaded:
            return True

        targetDir = os.path.join(self.cacheDir, moduleHash)

        # ignore 'marked invalid'
        # if os.path.exists(os.path.join(targetDir, "marked_invalid")):
        #     # just bail - don't try to read it now

        #     # for the moment, we don't try to clean up the cache, because
        #     # we can't be sure that some process is not still reading the
        #     # old files.
        #     self.modulesMarkedInvalid.add(moduleHash)
        #     return False

        with open(os.path.join(targetDir, "submodules.dat"), "rb") as f:
            submodules = SerializationContext().deserialize(f.read(), ListOf(str))

        for subHash in submodules:
            if not self.loadNameManifestFromStoredModuleByHash(subHash):
                # self.markModuleHashInvalid(subHash)
                return False

        with open(os.path.join(targetDir, "name_manifest.dat"), "rb") as f:
            self.nameToModuleHash.update(
                SerializationContext().deserialize(f.read(), Dict(str, str))
            )

        self.moduleManifestsLoaded.add(moduleHash)

        return True

    def writeModuleToDisk(self, binarySharedObject, nameToTypedCallTarget, submodules, dependencyEdgelist):
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
            f.write(SerializationContext().serialize(binarySharedObject.serializedGlobalVariableDefinitions))

        with open(os.path.join(tempTargetDir, "submodules.dat"), "wb") as f:
            f.write(SerializationContext().serialize(ListOf(str)(submodules), ListOf(str)))

        with open(os.path.join(tempTargetDir, "function_dependencies.dat"), "wb") as f:
            f.write(SerializationContext().serialize(dependencyEdgelist))  # might need a listof

        with open(os.path.join(tempTargetDir, "global_dependencies.dat"), "wb") as f:
            f.write(SerializationContext().serialize(binarySharedObject.globalDependencies))

        with open(os.path.join(tempTargetDir, "function_complexities.dat"), "wb") as f:
            f.write(SerializationContext().serialize(binarySharedObject.functionComplexities))

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
