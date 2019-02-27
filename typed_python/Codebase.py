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
import tempfile
import os
import sys
import threading
import logging

from typed_python.SerializationContext import SerializationContext

_lock = threading.RLock()
_root_level_module_codebase_cache = {}
_coreSerializationContext = [None]


class Codebase:
    """Represents a bundle of code and objects on disk somewhere.

    Also provides services for building a serialization context.
    """

    def __init__(self, rootDirectory, filesToContents, modules):
        self.rootDirectory = rootDirectory
        self.filesToContents = filesToContents
        self.modules = modules

        self.serializationContext = Codebase.coreSerializationContext().union(
            SerializationContext.FromModules(modules.values())
        )

    def getIsolatedSerializationContext(self):
        return SerializationContext.FromModules(self.modules.values())

    def allModuleLevelValues(self):
        """Iterate over all module-level values. Yields (name, object) pairs."""
        for mname, module in self.modules.items():
            for item in dir(module):
                yield (mname + "." + item, getattr(module, item))

    def getModuleByName(self, module_name):
        if module_name not in self.modules:
            raise ImportError(module_name)
        return self.modules[module_name]

    def getClassByName(self, qualifiedName):
        modulename, classname = qualifiedName.rsplit(".", 1)
        return getattr(self.getModuleByName(modulename), classname)

    @staticmethod
    def coreSerializationContext():
        with _lock:
            if _coreSerializationContext[0] is None:
                import object_database
                import typed_python

                context1 = SerializationContext.FromModules(
                    Codebase._walkModuleDiskRepresentation(typed_python)[2].values()
                )
                context2 = SerializationContext.FromModules(
                    Codebase._walkModuleDiskRepresentation(object_database)[2].values()
                )

                _coreSerializationContext[0] = context1.union(context2)

            return _coreSerializationContext[0]

    @staticmethod
    def FromRootlevelModule(module, **kwargs):
        assert '.' not in module.__name__
        return Codebase._FromModule(module, **kwargs)

    @staticmethod
    def _FromModule(module, **kwargs):
        if '.' in module.__name__:
            prefix = module.__name__.rsplit(".", 1)[0]
        else:
            prefix = None

        with _lock:
            if module in _root_level_module_codebase_cache:
                return _root_level_module_codebase_cache[module]

            assert module.__file__.endswith("__init__.py") or module.__file__.endswith("__init__.pyc")

            root, files, modules = Codebase._walkModuleDiskRepresentation(module, prefix=prefix, **kwargs)

            codebase = Codebase(root, files, modules)

            _root_level_module_codebase_cache[module] = codebase

            return _root_level_module_codebase_cache[module]

    @staticmethod
    def FromRootlevelPath(rootPath, **kwargs):
        root, files, modules = Codebase._walkDiskRepresentation(rootPath, **kwargs)
        codebase = Codebase(root, files, modules)
        return codebase

    @staticmethod
    def _walkDiskRepresentation(rootPath, prefix=None, extensions=('.py',), maxTotalBytes=100 * 1024 * 1024):
        """ Utility method that collects the code for a given root module.

            Parameters:
            -----------
            rootPath : str
                the root path for which to gather code

            Returns:
            --------
            tuple(parentDir:str, files:dict(str->str), modules:dict(str->module))
                parentDir:str is the path of the parent directory of the module
                files:dict(str->str) maps file paths (relative to the parentDir) to their contents
                modules:dict(str->module) maps module names to modules
        """
        parentDir, moduleDir = os.path.split(rootPath)

        # map: path:str -> contents:str
        files = {}
        total_bytes = [0]

        def walkDisk(path, so_far):
            if so_far.startswith("."):
                return  # skip hidden directories

            for name in os.listdir(path):
                fullpath = os.path.join(path, name)
                so_far_with_name = os.path.join(so_far, name) if so_far else name
                if os.path.isdir(fullpath):
                    walkDisk(fullpath, so_far_with_name)
                else:
                    if os.path.splitext(name)[1] in extensions:
                        with open(fullpath, "r") as f:
                            contents = f.read()

                        total_bytes[0] += len(contents)

                        if total_bytes[0] > maxTotalBytes:
                            raise Exception(
                                "exceeded bytecount with %s of size %s" % (fullpath, len(contents))
                            )

                        files[so_far_with_name] = contents

        walkDisk(os.path.abspath(rootPath), moduleDir)

        modules_by_name = Codebase.filesToModuleNames(files, prefix)
        modules = Codebase.importModulesByName(modules_by_name)

        return parentDir, files, modules

    @staticmethod
    def _walkModuleDiskRepresentation(module, **kwargs):
        dirpart = os.path.dirname(module.__file__)
        return Codebase._walkDiskRepresentation(dirpart, **kwargs)

    @staticmethod
    def filesToModuleNames(files, prefix=None):
        modules_by_name = set()

        for fpath in files:
            if fpath.endswith(".py"):
                module_parts = fpath.split("/")
                if module_parts[-1] == "__init__.py":
                    module_parts = module_parts[:-1]
                else:
                    module_parts[-1] = module_parts[-1][:-3]

                if prefix is not None:
                    module_parts = [prefix] + module_parts

                modules_by_name.add(".".join(module_parts))

        return modules_by_name

    @staticmethod
    def Instantiate(filesToContents, rootDirectory=None):
        """Instantiate a codebase on disk and import the modules."""
        with _lock:
            if rootDirectory is None:
                # this works, despite the fact that we immediately destroy
                # the directory, because we use 'makedirs' below to repopulate.
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

            # get a list of all modules and import each one
            modules_by_name = Codebase.filesToModuleNames(filesToContents)

            try:
                modules = Codebase.importModulesByName(modules_by_name)
            finally:
                sys.path.pop(0)
                Codebase.removeUserModules([rootDirectory])

            codebase = Codebase(rootDirectory, filesToContents, modules)

            # now make sure we install these modules in our cache so that later
            # we can walk them if we need to.
            for m in modules:
                if "." not in m:
                    _root_level_module_codebase_cache[modules[m]] = codebase

            return codebase

    @staticmethod
    def importModulesByName(modules_by_name):
        """ Returns a dict mapping module names (str) to modules. """
        modules = {}
        for mname in modules_by_name:
            try:
                modules[mname] = importlib.import_module(mname)
            except Exception as e:
                logging.getLogger(__name__).warn(
                    "Error importing module '%s' from codebase: %s", mname, e)
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

    @staticmethod
    def rootlevelPathFromModule(module):
        module_path = os.path.abspath(module.__file__)

        if os.path.basename(module_path) == '__init__.py':
            module_path = os.path.dirname(module_path)

        # drop as many parts of the module_path as there dots to the
        # module name
        for _ in range(len(module.__name__.split("."))-1):
            module_path = os.path.dirname(module_path)

        return module_path
