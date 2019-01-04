#   Copyright 2018 Braxton Mckee
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

import importlib
import types
import tempfile
import os
import sys
import threading
import logging

from typed_python.SerializationContext import SerializationContext

_lock = threading.Lock()
_root_level_module_codebase_cache = {}

class Codebase:
    """Represents a bundle of code and objects on disk somewhere.

    Also provides services for building a serialization context.
    """
    def __init__(self, rootDirectory, filesToContents, modules):
        self.rootDirectory = rootDirectory
        filesToContents = filesToContents
        self.modules = modules

        nameToObject = {}
        objectsSeen = set()
        for modulename, module in modules.items():
            for membername, member in module.__dict__.items():
                if isinstance(member, type) or isinstance(member, types.FunctionType):
                    nameToObject[modulename + "." + membername] = member
                    objectsSeen.add(id(member))

            #also add the module so we can serialize it.
            nameToObject[".modules." + modulename] = module
            objectsSeen.add(id(module))

        for modulename, module in modules.items():
            for membername, member in module.__dict__.items():
                if isinstance(member, type) and hasattr(member, '__dict__'):
                    for sub_name, sub_obj in member.__dict__.items():
                        if not (sub_name[:2] == "__" and sub_name[-2:] == "__"):
                            if isinstance(member, type) or isinstance(member, types.FunctionType):
                                if id(sub_obj) not in objectsSeen:
                                    nameToObject[membername + "." + sub_name] = sub_obj
                                    objectsSeen.add(id(sub_obj))

        self.serializationContext = SerializationContext(nameToObject)

    def getModuleByName(self, module_name):
        if module_name not in self.modules:
            raise ImportError(module_name)
        return self.modules[module_name]

    def getClassByName(self, qualifiedName):
        modulename, classname = qualifiedName.rsplit(".")
        return getattr(self.getModuleByName(modulename), classname)

    @staticmethod
    def FromRootlevelModule(module):
        with _lock:
            if module in _root_level_module_codebase_cache:
                return _root_level_module_codebase_cache[module]

            assert "." not in module.__name__
            assert module.__file__.endswith("__init__.py") or module.__file__.endswith("__init__.pyc")

            dirpart = os.path.dirname(module.__file__)
            root, moduleDir = os.path.split(dirpart)

            files = {}

            def walkDisk(path, so_far):
                for name in os.listdir(path):
                    fullpath = os.path.join(path, name)
                    so_far_with_name = os.path.join(so_far, name) if so_far else name
                    if os.path.isdir(fullpath):
                        walkDisk(fullpath, so_far_with_name)
                    else:
                        if os.path.splitext(name)[1] == ".py":
                            with open(fullpath, "r") as f:
                                contents = f.read()

                            files[so_far_with_name] = contents

            walkDisk(os.path.abspath(dirpart), moduleDir)

            modules_by_name = Codebase.filesToModuleNames(files)
            modules = Codebase.importModulesByName(modules_by_name)

            _root_level_module_codebase_cache[module] = Codebase(root, files, modules)

            return _root_level_module_codebase_cache[module]

    @staticmethod
    def filesToModuleNames(files):
        modules_by_name = set()

        for fpath in files:
            if fpath.endswith(".py"):
                module_parts = fpath.split("/")
                if module_parts[-1] == "__init__.py":
                    module_parts = module_parts[:-1]
                else:
                    module_parts[-1] = module_parts[-1][:-3]

                modules_by_name.add(".".join(module_parts))

        return modules_by_name

    @staticmethod
    def Instantiate(filesToContents, rootDirectory=None):
        """Instantiate a codebase on disk and import the modules."""
        with _lock:
            if rootDirectory is None:
                rootDirectory = tempfile.TemporaryDirectory().name

            for fpath, fcontents in filesToContents.items():
                path, name = os.path.split(fpath)

                fullpath = os.path.join(rootDirectory, path)

                if not os.path.exists(fullpath):
                    os.makedirs(fullpath)

                with open(os.path.join(fullpath, name), "wb") as f:
                    f.write(fcontents.encode("utf-8"))

            importlib.invalidate_caches()

            sys.path = [rootDirectory] + sys.path

            #get a list of all modules and import each one
            modules_by_name = Codebase.filesToModuleNames(filesToContents)

            try:
                modules = Codebase.importModulesByName(modules_by_name)
            finally:
                sys.path.pop(0)
                Codebase.removeUserModules([rootDirectory])

            return Codebase(rootDirectory, filesToContents, modules)

    @staticmethod
    def importModulesByName(modules_by_name):
        modules = {}
        for mname in modules_by_name:
            try:
                modules[mname] = importlib.import_module(mname)
            except Exception as e:
                logging.getLogger(__name__).warn(
                    "Error importing module %s from codebase: %s", mname,  e)
        return modules

    @staticmethod
    def removeUserModules(paths):
        paths = [os.path.abspath(path) for path in paths]

        for f in list(sys.path_importer_cache):
            if any(os.path.abspath(f).startswith(disk_path) for disk_path in paths):
                del sys.path_importer_cache[f]

        for m, sysmodule in list(sys.modules.items()):
            if hasattr(sysmodule, '__file__') and any(sysmodule.__file__.startswith(p) for p in paths):
                del sys.modules[m]
            elif hasattr(sysmodule, '__path__') and hasattr(sysmodule.__path__, '_path'):
                if any(any(pathElt.startswith(p) for p in paths) for pathElt in sysmodule.__path__._path):
                    del sys.modules[m]
