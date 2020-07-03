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
import time
import shutil

from typed_python.compiler.loaded_module import LoadedModule
from typed_python.compiler.binary_shared_object import BinarySharedObject

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

        self.loadedModules = Dict(str, LoadedModule)()
        self.nameToModuleHash = Dict(str, str)()

        t0 = time.time()
        count = 0
        for moduleHash in os.listdir(self.cacheDir):
            if len(moduleHash) == 40:
                self.loadNameManifestFromStoredModuleByHash(moduleHash)
                count += 1
        print("took ", time.time() - t0, " to load ", len(self.nameToModuleHash), " functions from ", count, " modules.")

    def hasSymbol(self, linkName):
        return linkName in self.nameToModuleHash

    def loadForSymbol(self, linkName):
        moduleHash = self.nameToModuleHash[linkName]

        nameToTypedCallTarget = {}
        nameToNativeFunctionType = {}
        self.loadModuleByHash(moduleHash, nameToTypedCallTarget, nameToNativeFunctionType)

        return nameToTypedCallTarget, nameToNativeFunctionType

    def loadModuleByHash(self, moduleHash, nameToTypedCallTarget, nameToNativeFunctionType):
        """Load a module by name.

        As we load, place all the newly imported typed call targets into
        'nameToTypedCallTarget' so that the rest of the system knows what functions
        have been uncovered.
        """
        if moduleHash in self.loadedModules:
            return

        targetDir = os.path.join(self.cacheDir, moduleHash)

        # write the type manifest
        with open(os.path.join(targetDir, "type_manifest.dat"), "rb") as f:
            callTargets = SerializationContext().deserialize(f.read())

        with open(os.path.join(targetDir, "globals_manifest.dat"), "rb") as f:
            globalVarDefs = SerializationContext().deserialize(f.read())

        with open(os.path.join(targetDir, "native_type_manifest.dat"), "rb") as f:
            functionNameToNativeType = SerializationContext().deserialize(f.read())

        with open(os.path.join(targetDir, "submodules.dat"), "rb") as f:
            submodules = SerializationContext().deserialize(f.read(), ListOf(str))

        # load the submodules first
        for submodule in submodules:
            self.loadModuleByHash(
                submodule,
                nameToTypedCallTarget,
                nameToNativeFunctionType
            )

        modulePath = os.path.join(targetDir, "module.so")

        loaded = BinarySharedObject.fromDisk(
            modulePath,
            globalVarDefs,
            functionNameToNativeType
        ).loadFromPath(modulePath)

        self.loadedModules[moduleHash] = loaded

        nameToTypedCallTarget.update(callTargets)
        nameToNativeFunctionType.update(functionNameToNativeType)

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

        path = self.writeModuleToDisk(binarySharedObject, nameToTypedCallTarget, dependentHashes)

        self.loadedModules[binarySharedObject.hash.hexdigest] = (
            binarySharedObject.loadFromPath(os.path.join(path, "module.so"))
        )

        for n in binarySharedObject.definedSymbols:
            self.nameToModuleHash[n] = binarySharedObject.hash.hexdigest

    def loadNameManifestFromStoredModuleByHash(self, moduleHash):
        targetDir = os.path.join(self.cacheDir, moduleHash)

        with open(os.path.join(targetDir, "name_manifest.dat"), "rb") as f:
            self.nameToModuleHash.update(
                SerializationContext().deserialize(f.read(), Dict(str, str))
            )

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
        targetDir = os.path.join(self.cacheDir, binarySharedObject.hash.hexdigest)

        if os.path.exists(targetDir):
            # this already exists. we don't need to write anything
            return targetDir

        tempTargetDir = targetDir + "_" + str(uuid.uuid4())
        ensureDirExists(tempTargetDir)

        # write the binary module
        with open(os.path.join(tempTargetDir, "module.so"), "wb") as f:
            f.write(binarySharedObject.binaryForm)

        # write the manifest. Every TP process using the cache will have to
        # load the manifest every time, so we try to use compiled code to load it
        manifest = Dict(str, str)()
        for n in binarySharedObject.functionTypes:
            manifest[n] = binarySharedObject.hash.hexdigest

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

        return targetDir

    def function_pointer_by_name(self, linkName):
        moduleHash = self.nameToModuleHash.get(linkName)
        if moduleHash is None:
            raise Exception("Can't find a module for " + linkName)

        return self.loadedModules[moduleHash].functionPointers[linkName]
