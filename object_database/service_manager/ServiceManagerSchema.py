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
import os
import sys
import importlib
import time
import os
import tempfile
from object_database import Schema, Indexed, Index, core_schema
from typed_python import *
import threading

service_schema = Schema("core.service")

codebase_lock = threading.Lock()
codebase_cache = {}

MAX_BAD_BOOTS = 5

@service_schema.define
class Codebase:
    hash = Indexed(str)

    #filename (at root of project import) to contents
    files = ConstDict(str, str) 

    @staticmethod
    def create(root_paths, extensions=('.py',), maxTotalBytes = 1024 * 1024):
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
        modules = dict(sys.modules)
        sys.path = paths + sys.path

        try:
            module = importlib.import_module(service_module_name)
        finally:
            sys.path.pop(len(paths))
            sys.modules = modules

        return module

    def instantiate(self, disk_path, service_module_name):
        """Instantiate a codebase on disk and load it."""
        with codebase_lock:
            root_path = os.path.abspath(disk_path)

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

            if (self.hash, service_module_name) in codebase_cache:
                return codebase_cache[self.hash, service_module_name]

            modules = dict(sys.modules)
            sys.path = [disk_path] + sys.path

            try:
                module = importlib.import_module(service_module_name)
            finally:
                sys.path.pop(0)
                sys.modules = modules

            codebase_cache[self.hash, service_module_name] = module

            return codebase_cache[self.hash, service_module_name]

@service_schema.define
class ServiceHost:
    connection = Indexed(core_schema.Connection)
    isMaster = bool
    hostname = str
    maxGbRam = float
    maxCores = int

    gbRamUsed = float
    coresUsed = float

@service_schema.define
class Service:
    name = Indexed(str)
    codebase = OneOf(service_schema.Codebase, None)

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

    def isThrottled(self):
        return self.timesBootedUnsuccessfully >= MAX_BAD_BOOTS
        
    def resetCounters(self):
        self.timesBootedUnsuccessfully = 0
        self.timesCrashed = 0

    def setCodebase(self, codebase, moduleName, className):
        if codebase != self.codebase or moduleName != self.service_module_name or className != self.service_class_name:
            self.codebase = codebase
            self.service_module_name = moduleName
            self.service_class_name = className
            self.timesBootedUnsuccessfully = 0
            self.timesCrashed = 0

    def effectiveTargetCount(self):
        if self.timesBootedUnsuccessfully >= MAX_BAD_BOOTS:
            return 0

        if self.isSingleton:
            return min(1, max(self.target_count, 0))
        else:
            return max(self.target_count, 0)

    def instantiateServiceObject(self, diskPath):
        if self.codebase:
            module = self.codebase.instantiate(diskPath, self.service_module_name)

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
    codebase = OneOf(service_schema.Codebase, None)
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

