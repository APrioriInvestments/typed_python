#   Copyright 2017-2019 typed_python Authors
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

from typed_python import sha_hash


# lock to guard our process-wide state
_lock = threading.RLock()

# map from sha-hash to installed codebase
_installed_codebases = {}

# map from root modulename to the installed codebase
_installed_rootlevel_modules = {}


class Codebase:
    """Represents a bundle of code and objects on disk somewhere.

    Codebases can be builtin (e.g. they come from an existing module
    on the system already) or they can be 'foreign' (meaning they
    don't already exist on the system) and can then be instantiated,
    which means we can import their modules.

    You may instantiate multiple codebases per process, but they
    must have disjoint root module names. You may not instantiate a codebase
    with a module name that's already defined locally.
    """

    def __init__(self, rootDirectory, filesToContents, rootModuleNames):
        """Initialize a codebase.

        Args:
            rootDirectory - the path to the root where the filesystem lives.
                For instance, if the code is in /home/ubuntu/code/typed_python,
                this would be '/home/ubuntu/code'
            filesToContents - a dict containing the filename (relative to
                rootDirectory) of each file, mapping to the byte contents.
            rootModuleNames - a list of root-level module names
            modules - None, or a dict from dotted module name to the actual
                module object, if its known.
        """
        self.rootDirectory = rootDirectory
        self.filesToContents = filesToContents
        self.rootModuleNames = rootModuleNames
        self._sha_hash = None

    @staticmethod
    def FromFileMap(filesToContents):
        modules_by_name = Codebase.filesToModuleNames(list(filesToContents), None)

        rootLevelModuleNames = set([x.split(".")[0] for x in modules_by_name])

        return Codebase(
            None,
            filesToContents,
            rootLevelModuleNames
        )

    @property
    def sha_hash(self):
        if self._sha_hash is None:
            self._sha_hash = sha_hash(self.filesToContents).hexdigest
        return self._sha_hash

    def isInstantiated(self):
        with _lock:
            return self.rootDirectory or self.sha_hash in _installed_codebases

    @property
    def moduleNames(self):
        return set(self.filesToModuleNames(self.filesToContents))

    def allModuleLevelValues(self):
        """Iterate over all module-level values. Yields (name, object) pairs."""
        for moduleName, module in self.importModulesByName(self.moduleNames).items():
            for item in dir(module):
                yield (moduleName + "." + item, getattr(module, item))

    def getModuleByName(self, module_name):
        return importlib.import_module(module_name)

    def getClassByName(self, qualifiedName):
        modulename, classname = qualifiedName.rsplit(".", 1)
        return getattr(self.getModuleByName(modulename), classname)

    def markNative(self):
        """Indicate that this codebase is already instantiated."""
        with _lock:
            for mname in self.rootModuleNames:
                _installed_rootlevel_modules[mname] = self

            _installed_codebases[self.sha_hash] = self

    @staticmethod
    def FromRootlevelModule(module, **kwargs):
        assert '.' not in module.__name__
        assert module.__file__.endswith("__init__.py") or module.__file__.endswith("__init__.pyc")

        prefix = module.__name__.rsplit(".", 1)[0]
        dirpart = os.path.dirname(module.__file__)

        codebase = Codebase.FromRootlevelPath(
            dirpart,
            prefix=prefix,

        )

        codebase.markNative()

        return codebase

    @staticmethod
    def FromRootlevelPath(
        rootPath,
        prefix=None,
        extensions=('.py',),
        maxTotalBytes=100 * 1024 * 1024,
        suppressFun=None
    ):
        """Build a codebase from the path to the root directory containing a module.

        Args:
            rootPath (str) - the root path we're going to pull in. This should point
                to a directory with the name of the python module this codebase
                will represent.
            extensions (tuple of strings) - a list of file extensions with the files
                we want to grab
            maxTotalBytes - a maximum bytecount before we'll throw an exception
            suppressFun - a function from module path (a dotted name) that returns
                True if we should stop walking into the path.
        """
        root, files, rootModuleNames = Codebase._walkDiskRepresentation(
            rootPath,
            prefix=prefix,
            extensions=extensions,
            maxTotalBytes=maxTotalBytes,
            suppressFun=suppressFun
        )

        return Codebase(root, files, rootModuleNames)

    @staticmethod
    def _walkDiskRepresentation(
        rootPath,
        prefix=None,
        extensions=('.py',),
        maxTotalBytes=100 * 1024 * 1024,
        suppressFun=None
    ):
        """ Utility method that collects the code for a given root module.

            Parameters:
            -----------
            rootPath : str
                the root path for which to gather code

            suppressFun : a function(path) that returns True if the module path shouldn't
                be included in the codebase.

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

            if suppressFun is not None:
                if suppressFun(so_far):
                    return

            for name in os.listdir(path):
                fullpath = os.path.join(path, name)
                so_far_with_name = os.path.join(so_far, name) if so_far else name
                if os.path.isdir(fullpath):
                    walkDisk(fullpath, so_far_with_name)
                else:
                    if os.path.splitext(name)[1] in extensions:
                        with open(fullpath, "r", encoding='utf-8') as f:
                            try:
                                contents = f.read()
                            except UnicodeDecodeError:
                                raise Exception(f"Failed to parse code in {fullpath} because of a unicode error.")

                        total_bytes[0] += len(contents)

                        if total_bytes[0] > maxTotalBytes:
                            raise Exception(
                                "exceeded bytecount with %s of size %s" % (fullpath, len(contents))
                            )

                        files[so_far_with_name] = contents

        walkDisk(os.path.abspath(rootPath), moduleDir)

        modules_by_name = Codebase.filesToModuleNames(files, prefix)

        rootLevelModuleNames = set([x.split(".")[0] for x in modules_by_name])

        return parentDir, files, rootLevelModuleNames

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

    def instantiate(self, rootDirectory=None):
        """Instantiate a codebase on disk

        Args:
            rootDirectory - if None, then pick a directory. otherwise,
                this is where to put the code. This directory must be
                persistent for the life of the process.
        """
        if self.isInstantiated():
            return

        if self.rootDirectory is not None:
            raise Exception("Codebase is already instantiated, but not marked as such?")

        with _lock:
            if self.sha_hash in _installed_codebases:
                # we're already installed
                return

            for rootMod in self.rootModuleNames:
                if rootMod in _installed_rootlevel_modules:
                    # we can't have the same root-level module instantiated in a different codebase.
                    raise Exception(f"Module {rootMod} is instantiated in another codebase already")

            if rootDirectory is None:
                # this works, despite the fact that we immediately destroy
                # the directory, because we use 'makedirs' below to repopulate.
                rootDirectory = tempfile.TemporaryDirectory().name

            for fpath, fcontents in self.filesToContents.items():
                path, name = os.path.split(fpath)
                fullpath = os.path.join(rootDirectory, path)

                if not os.path.exists(fullpath):
                    os.makedirs(fullpath)

                with open(os.path.join(fullpath, name), "wb") as f:
                    f.write(fcontents.encode("utf-8"))

            sys.path = [rootDirectory] + sys.path

            for rootMod in self.rootModuleNames:
                _installed_rootlevel_modules[rootMod] = self

            _installed_codebases[self.sha_hash] = self
            self.rootDirectory = rootDirectory

    @staticmethod
    def importModulesByName(modules_by_name):
        """ Returns a dict mapping module names (str) to modules. """
        modules = {}
        for mname in sorted(modules_by_name):
            try:
                modules[mname] = importlib.import_module(mname)
            except Exception as e:
                logging.getLogger(__name__).warn(
                    "Error importing module '%s' from codebase: %s", mname, e)
        return modules

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
