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


import llvmlite.binding as llvm
import tempfile
import os
import subprocess
import ctypes
import struct

from typed_python.compiler.native_function_pointer import NativeFunctionPointer
from typed_python.compiler.loaded_module import LoadedModule
from typed_python.hash import sha_hash


class LoadedBinarySharedObject(LoadedModule):
    def __init__(self, binarySharedObject, storageDir, functionPointers, globalVariableDefinitions):
        super().__init__(functionPointers, globalVariableDefinitions)

        self.binarySharedObject = binarySharedObject
        self.storageDir = storageDir


class BinarySharedObject:
    """Models a shared object library (.so) loadable on linux systems."""

    def __init__(self, binaryForm, functionTypes, globalVariableDefinitions):
        """
        Args:
            binaryForm - a bytes object containing the actual compiled code for the module
            globalVariableDefinitions - a map from name to GlobalVariableDefinition
        """
        self.binaryForm = binaryForm
        self.functionTypes = functionTypes
        self.globalVariableDefinitions = globalVariableDefinitions

    @staticmethod
    def fromModule(module, globalVariableDefinitions, functionNameToType):
        target_triple = llvm.get_process_triple()
        target = llvm.Target.from_triple(target_triple)
        target_machine_shared_object = target.create_target_machine(reloc='pic', codemodel='default')

        # returns the contents of a '.o' file coming out of a c++ compiler like clang
        o_file_contents = target_machine_shared_object.emit_object(module)

        # we have to run it through 'ld' to link it. if we want to support windows,
        # we should use 'llvm' directly instead of 'llmvlite', in which case this
        # kind of linking operation would be easier to express directly without
        # resorting to subprocesses.
        with tempfile.TemporaryDirectory() as tf:
            with open(os.path.join(tf, "module.o"), "wb") as o_file:
                o_file.write(o_file_contents)

            subprocess.check_call(
                ["ld", "-shared", "-fPIC", os.path.join(tf, "module.o"), "-o", os.path.join(tf, "module.so")]
            )

            with open(os.path.join(tf, "module.so"), "rb") as so_file:
                return BinarySharedObject(so_file.read(), functionNameToType, globalVariableDefinitions)

    def load(self, storageDir):
        """Instantiate this .so in temporary storage and return a dict from symbol -> integer function pointer"""
        if not os.path.exists(storageDir):
            os.makedirs(storageDir)

        modulename = sha_hash(self.binaryForm).hexdigest + "_module.so"
        modulePath = os.path.join(storageDir, modulename)

        with open(modulePath, "wb") as f:
            f.write(self.binaryForm)

        dll = ctypes.CDLL(modulePath)

        functionPointers = {}

        for symbol in self.functionTypes:
            # if you ask for 'bytes' on a ctypes function you get the function pointer
            # encoded as a bytearray.
            functionPointers[symbol] = NativeFunctionPointer(
                symbol,
                struct.unpack("q", bytes(dll[symbol]))[0],
                self.functionTypes[symbol].args,
                self.functionTypes[symbol].output,
            )

        return LoadedBinarySharedObject(
            self,
            storageDir,
            functionPointers,
            self.globalVariableDefinitions
        )
