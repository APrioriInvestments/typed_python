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

import logging
import importlib
import os
import sys
import six
import importlib
import time
import os
import tempfile
import urllib.parse
import object_database

from object_database import Schema, Indexed, Index, core_schema, SubscribeLazilyByDefault
from typed_python import *
import threading

from object_database.service_manager.ServiceSchema import service_schema

#singleton state objects for the codebase cache
_codebase_lock = threading.Lock()
_codebase_cache = {}
_codebase_instantiation_dir = None

class InstantiatedCodebase:
    def __init__(self, targetDirectory, filesToContents):
        self.targetDirectory = targetDirectory
        self.files = filesToContents
        self.modules = {}

        self._instantiate()

        self.typeSet = TypeSet.fromModules(self.modules)

    def getModuleByName(self, module_name):
        if module_name not in self.modules:
            raise ImportError(module_name)
        return self.modules[module_name]

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

    def _instantiate(self):
        for fpath, fcontents in self.files.items():
            path, name = os.path.split(fpath)

            fullpath = os.path.join(self.targetDirectory, path)

            if not os.path.exists(fullpath):
                try:
                    os.makedirs(fullpath)
                except:
                    logging.warn("Exception trying to make directory %s", _codebase_instantiation_dir)

            with open(os.path.join(fullpath, name), "wb") as f:
                f.write(fcontents.encode("utf-8"))

        importlib.invalidate_caches()

        sys.path = [self.targetDirectory] + sys.path

        #get a list of all modules and import each one
        modules_by_name = set()
        for fpath in self.files:
            if fpath.endswith(".py"):
                module_parts = fpath.split("/")
                if module_parts[-1] == "__init__.py":
                    module_parts = module_parts[:-1]
                else:
                    module_parts[-1] = module_parts[-1][:-3]

                modules_by_name.add(".".join(module_parts))

        try:
            for mname in modules_by_name:
                try:
                    self.modules[mname] = importlib.import_module(mname)
                except Exception as e:
                    logging.warn("Error importing module %s from codebase %s: %s", mname, self, e)
        finally:
            sys.path.pop(0)
            self.removeUserModules([self.targetDirectory])


def setCodebaseInstantiationDirectory(dir, forceReset=False):
    """Called at program invocation to specify where we can instantiate codebases."""
    with _codebase_lock:
        global _codebase_instantiation_dir
        global _codebase_cache

        if forceReset:
            _codebase_instantiation_dir = None
            _codebase_cache = {}

        if _codebase_instantiation_dir == dir:
            return

        assert _codebase_instantiation_dir is None, "Can't modify the codebase instantiation location."


        _codebase_instantiation_dir = os.path.abspath(dir)

@service_schema.define
@SubscribeLazilyByDefault
class File:
    hash = Indexed(str)
    contents = str

    @staticmethod
    def create(contents):
        hash = sha_hash(contents).hexdigest
        f = File.lookupAny(hash=hash)
        if f:
            return f
        else:
            return File(hash=hash, contents=contents)

@service_schema.define
@SubscribeLazilyByDefault
class Codebase:
    hash = Indexed(str)

    #filename (at root of project import) to contents
    files = ConstDict(str, service_schema.File)

    @staticmethod
    def create(root_paths, extensions=('.py',), maxTotalBytes = 100 * 1024 * 1024):
        files = {}
        total_bytes = [0]

        def walk(path, so_far):
            for name in os.listdir(path):
                fullpath = os.path.join(path, name)
                so_far_with_name = os.path.join(so_far, name) if so_far else name
                if os.path.isdir(fullpath):
                    walk(fullpath, so_far_with_name)
                else:
                    if os.path.splitext(name)[1] in extensions:
                        with open(fullpath, "r") as f:
                            contents = f.read()

                        total_bytes[0] += len(contents)

                        assert total_bytes[0] < maxTotalBytes, "exceeded bytecount with %s of size %s" % (fullpath, len(contents))

                        files[so_far_with_name] = File.create(contents)

        for path in root_paths:
            walk(path, '')

        return Codebase.createFromFiles(files)

    @staticmethod
    def createFromFiles(files):
        assert files

        files = {k: File.create(v) if not isinstance(v, File) else v for k,v in files.items()}

        hashval = sha_hash(files).hexdigest

        c = Codebase.lookupAny(hash=hashval)
        if c:
            return c

        return Codebase(hash=hashval, files=files)

    def instantiate(self, module_name=None):
        """Instantiate a codebase on disk and load it."""
        with _codebase_lock:
            if self.hash not in _codebase_cache:
                try:
                    if not os.path.exists(_codebase_instantiation_dir):
                        os.makedirs(_codebase_instantiation_dir)
                except:
                    logging.warn("Exception trying to make directory %s", _codebase_instantiation_dir)

                disk_path = os.path.join(_codebase_instantiation_dir, self.hash)

                #preload the files, since they're lazy.
                object_database.current_transaction().db().requestLazyObjects(set(self.files.values()))

                fileContents = {fpath: file.contents for fpath, file in self.files.items()}

                _codebase_cache[self.hash] = InstantiatedCodebase(disk_path, fileContents)

            if module_name is None:
                return _codebase_cache[self.hash]

            return _codebase_cache[self.hash].getModuleByName(module_name)
