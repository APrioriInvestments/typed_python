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
from typed_python.compiler.native_compiler.native_ast import Function
from typed_python.compiler.native_compiler.native_ast_analysis import extractNamedCallTargets
from typed_python.compiler.native_compiler.loaded_module import LoadedModule
from typed_python.compiler.native_compiler.binary_shared_object import BinarySharedObject

from typed_python.SerializationContext import SerializationContext
from typed_python import Dict, ListOf, Set


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
    def __init__(self, cacheDir, checkModuleValidity=True):
        self.cacheDir = cacheDir
        self.checkModuleValidity = checkModuleValidity

        ensureDirExists(cacheDir)

        self.loadedModules = Dict(str, LoadedModule)()

        # for each symbol, the first module we loaded that has that symbol
        self.symbolToLoadedModuleHash = Dict(str, str)()

        # for each module that we loaded or might load, the contents
        self.symbolToModuleHashes = Dict(str, Set(str))()
        self.moduleHashToSymbols = Dict(str, Set(str))()

        # modules we might be able to load
        self.modulesMarkedValid = set()

        # modules we definitely can't load
        self.modulesMarkedInvalid = set()

        for moduleHash in os.listdir(self.cacheDir):
            if len(moduleHash) == 40:
                self.loadNameManifestFromStoredModuleByHash(moduleHash)

    def hasSymbol(self, symbol):
        """Do we have this symbol defined somewhere?

        Note that this can change: if we attempt to laod a symbol and fail,
        then it may no longer be defined anywhere. To really know if you have
        a symbol, you have to load it.
        """
        return symbol in self.symbolToModuleHashes

    def markModuleHashInvalid(self, moduleHash):
        """Mark this module unloadable on disk and remove its symbols."""
        with open(os.path.join(self.cacheDir, moduleHash, "marked_invalid"), "w"):
            pass

        # remove any symbols that we can't see anymore
        for symbol in self.moduleHashToSymbols.pop(moduleHash, Set(str)()):
            hashes = self.symbolToModuleHashes[symbol]
            hashes.discard(moduleHash)
            if not hashes:
                del self.symbolToModuleHashes[symbol]

    def loadForSymbol(self, symbol):
        # check if this symbol is already loaded
        if symbol in self.symbolToLoadedModuleHash:
            return None

        while symbol in self.symbolToModuleHashes:
            moduleHash = list(self.symbolToModuleHashes[symbol])[0]

            nameToTypedCallTarget = {}
            nameToNativeFunctionType = {}
            nameToDefinition = {}

            if self.loadModuleByHash(moduleHash, nameToTypedCallTarget, nameToNativeFunctionType, nameToDefinition):
                return nameToTypedCallTarget, nameToNativeFunctionType, nameToDefinition
            else:
                assert (
                    # either we can't load this symbol at all anymore
                    symbol not in self.symbolToModuleHashes
                    # or confirm we can't try to load this again
                    or moduleHash not in self.symbolToModuleHashes[symbol]
                )

    def loadModuleByHash(
        self,
        moduleHash,
        nameToTypedCallTarget,
        nameToNativeFunctionType,
        nameToDefinition
    ):
        """Load a module by name.

        As we load, place all the newly imported typed call targets into
        'nameToTypedCallTarget' so that the rest of the system knows what functions
        have been uncovered.
        """
        if moduleHash in self.loadedModules:
            return True

        targetDir = os.path.join(self.cacheDir, moduleHash)

        try:
            with open(os.path.join(targetDir, "type_manifest.dat"), "rb") as f:
                callTargets = SerializationContext().deserialize(f.read())

            with open(os.path.join(targetDir, "globals_manifest.dat"), "rb") as f:
                globalVarDefs = SerializationContext().deserialize(f.read())

            with open(os.path.join(targetDir, "native_type_manifest.dat"), "rb") as f:
                functionNameToNativeType = SerializationContext().deserialize(f.read())

            with open(os.path.join(targetDir, "submodules.dat"), "rb") as f:
                submodules = SerializationContext().deserialize(f.read(), ListOf(str))

            with open(os.path.join(targetDir, "linkDependencies.dat"), "rb") as f:
                linkDependencies = SerializationContext().deserialize(f.read(), ListOf(str))

            with open(os.path.join(targetDir, "functionDefinitions.dat"), "rb") as f:
                functionDefinitions = SerializationContext().deserialize(
                    f.read(), Dict(str, Function)
                )

        except Exception:
            self.markModuleHashInvalid(moduleHash)
            return False

        if not LoadedModule.validateGlobalVariables(globalVarDefs):
            self.markModuleHashInvalid(moduleHash)
            return False

        # load the submodules first
        for submodule in submodules:
            if not self.loadModuleByHash(
                submodule,
                nameToTypedCallTarget,
                nameToNativeFunctionType,
                nameToDefinition
            ):
                return False

        modulePath = os.path.join(targetDir, "module.so")

        loaded = BinarySharedObject.fromDisk(
            modulePath,
            globalVarDefs,
            functionNameToNativeType,
            linkDependencies,
            functionDefinitions
        ).loadFromPath(modulePath)

        self.loadedModules[moduleHash] = loaded

        nameToTypedCallTarget.update(callTargets)
        nameToNativeFunctionType.update(functionNameToNativeType)
        nameToDefinition.update(functionDefinitions)

        for symbol in functionNameToNativeType:
            if symbol not in self.symbolToLoadedModuleHash:
                self.symbolToLoadedModuleHash[symbol] = moduleHash

        return True

    def addModule(self, binarySharedObject, nameToTypedCallTarget):
        """Add new code to the compiler cache.

        Args:
            binarySharedObject - a BinarySharedObject containing the actual assembler
                we've compiled
            nameToTypedCallTarget - a dict from linkname to TypedCallTarget telling us
                the formal python types for all the objects
            linkDependencies - a set of linknames we depend on directly.
        """
        if self.checkModuleValidity:
            externals = extractNamedCallTargets(
                binarySharedObject.functionDefinitions
            )

            statedNames = set(binarySharedObject.usedExternalFunctions)

            expectedNames = set(e.name for e in externals if not e.external) - set(
                binarySharedObject.functionDefinitions
            )

            if statedNames != expectedNames:
                if expectedNames - statedNames:
                    raise Exception(
                        "Invalid shared object - link dependencies don't match "
                        + "stated shared object dependencies:\n\n"
                        + "".join(
                            ['    ' + x + "\n" for x in sorted(expectedNames - statedNames)]
                        )
                        + "\nwere referenced but not claimed in the manifest."
                    )
                else:
                    raise Exception(
                        "Invalid shared object - link dependencies don't match "
                        + "stated shared object dependencies:\n\n"
                        + "".join(
                            ['    ' + x + "\n" for x in sorted(statedNames - expectedNames)]
                        )
                        + "\nwere claimed in the manifest but don't seem to be referenced"
                    )

        dependentHashes = set()

        for name in binarySharedObject.usedExternalFunctions:
            dependentHashes.add(self.symbolToLoadedModuleHash[name])

        path, hashToUse = self.writeModuleToDisk(
            binarySharedObject,
            nameToTypedCallTarget,
            dependentHashes
        )

        self.loadedModules[hashToUse] = (
            binarySharedObject.loadFromPath(os.path.join(path, "module.so"))
        )

        for symbol in binarySharedObject.definedSymbols:
            if symbol not in self.symbolToLoadedModuleHash:
                self.symbolToLoadedModuleHash[symbol] = hashToUse
            self.symbolToModuleHashes.setdefault(symbol).add(hashToUse)
        self.moduleHashToSymbols[hashToUse] = Set(str)(binarySharedObject.definedSymbols)

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
            manifest = SerializationContext().deserialize(f.read(), Set(str))

            for symbolName in manifest:
                self.symbolToModuleHashes.setdefault(symbolName).add(moduleHash)
            self.moduleHashToSymbols[moduleHash] = manifest

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
        manifest = Set(str)(binarySharedObject.functionTypes)

        with open(os.path.join(tempTargetDir, "name_manifest.dat"), "wb") as f:
            f.write(SerializationContext().serialize(manifest, Set(str)))

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

        with open(os.path.join(tempTargetDir, "linkDependencies.dat"), "wb") as f:
            f.write(
                SerializationContext().serialize(
                    ListOf(str)(binarySharedObject.usedExternalFunctions),
                    ListOf(str)
                )
            )

        with open(os.path.join(tempTargetDir, "functionDefinitions.dat"), "wb") as f:
            f.write(
                SerializationContext().serialize(
                    Dict(str, Function)(binarySharedObject.functionDefinitions),
                    Dict(str, Function)
                )
            )

        try:
            os.rename(tempTargetDir, targetDir)
        except IOError:
            if not os.path.exists(targetDir):
                raise
            else:
                shutil.rmtree(tempTargetDir)

        return targetDir, hashToUse

    def function_pointer_by_name(self, linkName):
        moduleHash = self.symbolToLoadedModuleHash.get(linkName)
        if moduleHash is None:
            raise Exception("Can't find a module for " + linkName)

        if moduleHash not in self.loadedModules:
            raise Exception("You need to call 'loadForSymbol' on this linkName first")

        return self.loadedModules[moduleHash].functionPointers[linkName]
