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

from object_database import Schema, Indexed, Index, core_schema
from typed_python import *

service_schema = Schema("core.service")

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

    def instantiate(self, disk_path, service_module):
        """Instantiate a codebase on disk and load it."""
        logging.info(disk_path)

        for fpath, fcontents in self.files.items():
            path, name = os.path.split(fpath)

            fullpath = os.path.join(disk_path, path)

            if not os.path.exists(fullpath):
                os.makedirs(fullpath)
            
            with open(os.path.join(fullpath, name), "w") as f:
                f.write(fcontents)

        sys.path = [disk_path] + sys.path

        try:
            return importlib.import_module(service_module)
        finally:
            sys.path.pop(0)

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

    gbRamUsed = int
    coresUsed = int
    placement = OneOf("Master", "Worker", "Any")

    #how many do we want?
    target_count = int

    #how many would we like but we can't boot?
    unbootable_count = int

@service_schema.define
class ServiceInstance:
    host = Indexed(ServiceHost)
    service = Indexed(service_schema.Service)
    codebase = OneOf(service_schema.Codebase, None)
    connection = Indexed(OneOf(None, core_schema.Connection))

    shouldShutdown = bool

    def markFailedToStart(self, reason):
        self.boot_timestamp = time.time()
        self.end_timestamp = self.boot_timestamp
        self.failureReason = reason
        self.state = "Failed"

    def isNotRunning(self):
        return self.state in ("Stopped", "Failed") or (self.connection and not self.connection.exists())

    def isActive(self):
        """Is this service instance up and intended to be up?"""
        return (
            self.state in ("Running", "Initializing", "Booting") 
                and not self.shouldShutdown 
                and (self.connection is None or self.connection.exists())
            )

    state = Indexed(OneOf("Booting", "Initializing", "Running", "Stopped", "Failed"))

    boot_timestamp = OneOf(None, float)
    start_timestamp = OneOf(None, float)
    end_timestamp = OneOf(None, float)
    failureReason = str

