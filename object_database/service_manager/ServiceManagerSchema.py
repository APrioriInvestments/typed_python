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

service_schema = Schema("core.service")

codebase_lock = threading.Lock()
codebase_module_cache = {}
codebase_is_imported = set()

MAX_BAD_BOOTS = 5

@service_schema.define
@SubscribeLazilyByDefault
class Codebase:
    hash = Indexed(str)

    #filename (at root of project import) to contents
    files = ConstDict(str, str)

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

                        files[so_far_with_name] = contents

        for path in root_paths:
            walk(path, '')

        return Codebase.createFromFiles(files)

    @staticmethod
    def createFromFiles(files):
        assert files

        hashval = sha_hash(files).hexdigest

        c = Codebase.lookupAny(hash=hashval)
        if c:
            return c

        return Codebase(hash=hashval, files=files)

    @staticmethod
    def instantiateFromLocalSource(paths, service_module_name):
        """Given a set of paths (that we would use to create a codebase), instantiate it directly."""
        sys.path = paths + sys.path

        try:
            module = importlib.import_module(service_module_name)
        finally:
            for _ in range(len(paths)):
                sys.path.pop(0)

            Codebase.removeUserModules(paths)

        return module


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

    def instantiate(self, service_module_name, root_path_override=None):
        """Instantiate a codebase on disk and load it."""
        if root_path_override is not None:
            root_path = root_path_override
        else:
            root_path = object_database.service_manager.ServiceBase.ServiceBase.currentServiceSourceRootImpliedByDbTransaction()

        with codebase_lock:
            if (self.hash, service_module_name) in codebase_module_cache:
                return codebase_module_cache[self.hash, service_module_name]

            if self.hash not in codebase_is_imported:
                root_path = os.path.abspath(root_path)

                importlib.invalidate_caches()

                try:
                    if not os.path.exists(root_path):
                        os.makedirs(root_path)
                except:
                    logging.warn("Exception trying to make directory %s", root_path)

                disk_path = os.path.join(root_path, self.hash)

                for fpath, fcontents in self.files.items():
                    path, name = os.path.split(fpath)

                    fullpath = os.path.join(disk_path, path)

                    if not os.path.exists(fullpath):
                        try:
                            os.makedirs(fullpath)
                        except:
                            logging.warn("Exception trying to make directory %s", root_path)

                    with open(os.path.join(fullpath, name), "wb") as f:
                        f.write(fcontents.encode("utf-8"))

                sys.path = [disk_path] + sys.path

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
                            codebase_module_cache[self.hash, mname] = importlib.import_module(mname)
                        except Exception as e:
                            logging.warn("Error importing module %s from codebase %s: %s", mname, self, e)
                finally:
                    sys.path.pop(0)
                    self.removeUserModules([disk_path])

                codebase_is_imported.add(self.hash)

            if (self.hash, service_module_name) not in codebase_module_cache:
                raise Exception("Imported codebase doesn't have module %s" % service_module_name)

            return codebase_module_cache[self.hash, service_module_name]

@service_schema.define
@SubscribeLazilyByDefault
class LogResponse:
    data = str

@service_schema.define
class LogRequest:
    host = Indexed(service_schema.ServiceHost)
    serviceInstance = service_schema.ServiceInstance
    maxBytes = int

    response = OneOf(None, service_schema.LogResponse)
    timestamp = float #so we can clean up old requests that timed out

@service_schema.define
class ServiceHost:
    connection = Indexed(core_schema.Connection)
    isMaster = bool
    hostname = str
    maxGbRam = float
    maxCores = int

    gbRamUsed = float
    coresUsed = float

    cpuUse = float
    actualMemoryUseGB = float
    statsLastUpdateTime = float

@service_schema.define
class Service:
    name = Indexed(str)
    codebase = OneOf(None, service_schema.Codebase)

    service_module_name = str
    service_class_name = str

    #per service, how many do we use?
    gbRamUsed = int
    coresUsed = int
    placement = OneOf("Master", "Worker", "Any")
    isSingleton = bool

    #how many do we want?
    target_count = int

    #how many would we like but we can't boot?
    unbootable_count = int

    timesBootedUnsuccessfully = int
    timesCrashed = int
    lastFailureReason = OneOf(None, str)

    def isThrottled(self):
        return self.timesBootedUnsuccessfully >= MAX_BAD_BOOTS

    def resetCounters(self):
        self.timesBootedUnsuccessfully = 0
        self.timesCrashed = 0
        self.lastFailureReason = None

    def setCodebase(self, codebase, moduleName, className):
        if codebase != self.codebase or moduleName != self.service_module_name or className != self.service_class_name:
            self.codebase = codebase
            self.service_module_name = moduleName
            self.service_class_name = className
            self.resetCounters()

    def effectiveTargetCount(self):
        if self.timesBootedUnsuccessfully >= MAX_BAD_BOOTS:
            return 0

        if self.isSingleton:
            return min(1, max(self.target_count, 0))
        else:
            return max(self.target_count, 0)

    def instantiateType(self, typename, root_path_override=None):
        modulename = ".".join(typename.split(".")[:-1])
        typename = typename.split(".")[-1]

        if self.codebase:
            module = self.codebase.instantiate(modulename, root_path_override=root_path_override)

            if typename not in module.__dict__:
                return None

            return module.__dict__[typename]
        else:
            def _getobject(modname, attribute):
                mod = __import__(modname, fromlist=[attribute])
                return mod.__dict__[attribute]

            return _getobject(modulename, typename)

    def urlForObject(self, obj, **queryParams):
        return "/services/%s/%s/%s" % (
            self.name,
            type(obj).__schema__.name + "." + type(obj).__qualname__,
            obj._identity
            ) + ("" if not queryParams else "?" + urllib.parse.urlencode({k:str(v) for k,v in queryParams.items()}))

    def findModuleSchemas(self):
        """Find all Schema objects in the same module as our type object."""
        if self.codebase:
            module = self.codebase.instantiate(self.service_module_name)
        else:
            module = importlib.import_module(self.service_module_name)

        res = []

        for o in dir(module):
            if isinstance(getattr(module, o), Schema):
                res.append(getattr(module, o))

        return res

    def instantiateServiceType(self, root_path_override=None):
        """Instantiate the codebase and return the instance of the Service type.

        root_path_override - set the path where we instantiate codebases. If not given,
            we assume we're already running in a service context and can look up the current
            service (and its instantiation path).
        """
        if self.codebase:
            module = self.codebase.instantiate(self.service_module_name, root_path_override=root_path_override)

            if self.service_class_name not in module.__dict__:
                raise Exception("Provided module %s at %s has no class %s. Options are:\n%s" % (
                    self.service_module_name,
                    module.__file__,
                    self.service_class_name,
                    "\n".join(["  " + x for x in sorted(module.__dict__)])
                    ))

            service_type = module.__dict__[self.service_class_name]
        else:
            def _getobject(modname, attribute):
                mod = __import__(modname, fromlist=[attribute])
                return mod.__dict__[attribute]

            service_type = _getobject(
                self.service_module_name,
                self.service_class_name
                )

        return service_type


@service_schema.define
class ServiceInstance:
    host = Indexed(ServiceHost)
    service = Indexed(service_schema.Service)
    codebase = OneOf(None, service_schema.Codebase)
    connection = Indexed(OneOf(None, core_schema.Connection))

    shouldShutdown = bool
    shutdownTimestamp = OneOf(None, float)

    def triggerShutdown(self):
        if not self.shouldShutdown:
            self.shouldShutdown = True
            self.shutdownTimestamp = time.time()

    def markFailedToStart(self, reason):
        self.boot_timestamp = time.time()
        self.end_timestamp = self.boot_timestamp
        self.failureReason = reason
        self.state = "FailedToStart"

    def isNotRunning(self):
        return self.state in ("Stopped", "FailedToStart", "Crashed") or (self.connection and not self.connection.exists())

    def isActive(self):
        """Is this service instance up and intended to be up?"""
        return (
            self.state in ("Running", "Initializing", "Booting")
                and not self.shouldShutdown
                and (self.connection is None or self.connection.exists())
            )

    state = Indexed(OneOf("Booting", "Initializing", "Running", "Stopped", "FailedToStart", "Crashed"))

    boot_timestamp = OneOf(None, float)
    start_timestamp = OneOf(None, float)
    end_timestamp = OneOf(None, float)
    failureReason = str

