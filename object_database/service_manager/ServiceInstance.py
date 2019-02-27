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
import logging
import time
import urllib.parse

import object_database

from object_database.service_manager.ServiceSchema import service_schema
from object_database import Schema, Indexed, core_schema
from typed_python.Codebase import Codebase as TypedPythonCodebase
from typed_python import OneOf

MAX_BAD_BOOTS = 5


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


class CodebaseLockedException(Exception):
    pass


@service_schema.define
class Service:
    name = Indexed(str)
    _codebase = OneOf(None, service_schema.Codebase)
    _codebaseStatus = OneOf("PREPARED", "LOCKED", "UNLOCKED")  # protects _codebase from modification

    service_module_name = str
    service_class_name = str

    # per service, how many do we use?
    gbRamUsed = int
    coresUsed = int
    placement = OneOf("Master", "Worker", "Any")
    isSingleton = bool

    # how many do we want?
    target_count = int

    # how many would we like but we can't boot?
    unbootable_count = int

    timesBootedUnsuccessfully = int
    timesCrashed = int
    lastFailureReason = OneOf(None, str)

    @property
    def codebaseStatus(self):
        return self._codebaseStatus

    @property
    def isUnlocked(self):
        return self._codebaseStatus == "UNLOCKED"

    @property
    def isLocked(self):
        return self._codebaseStatus == "LOCKED"

    @property
    def isPrepared(self):
        return self._codebaseStatus == "PREPARED"

    def unlock(self):
        self._codebaseStatus = "UNLOCKED"

    def lock(self):
        self._codebaseStatus = "LOCKED"

    def prepare(self):
        if self._codebaseStatus == "LOCKED":
            self._codebaseStatus = "PREPARED"

    def deploy(self):
        if self._codebaseStatus == "PREPARED":
            self._codebaseStatus = "LOCKED"

    @property
    def codebase(self):
        return self._codebase

    def _setCodebase(self, other):
        if self.isLocked:
            logging.getLogger(__name__).warning("Cannot set codebase of locked service")
        else:
            self._codebase = other
            self.deploy()

    def setCodebase(self, codebase, moduleName=None, className=None):
        if (codebase != self.codebase or
                (moduleName is not None and moduleName != self.service_module_name) or
                (className is not None and className != self.service_class_name)):
            if self.isLocked:
                raise CodebaseLockedException(
                    "Cannot set codebase of locked service '{}'".format(self.name))

            self._setCodebase(codebase)

            if moduleName is not None:
                self.service_module_name = moduleName

            if className is not None:
                self.service_class_name = className

            self.resetCounters()

    def trySetCodebase(self, codebase, moduleName=None, className=None):
        try:
            self.setCodebase(codebase, moduleName, className)
            return True
        except CodebaseLockedException:
            logging.getLogger(__name__).warning(
                "Cannot set codebase of locked service '{}'".format(self.name)
            )
            return False

    def getSerializationContext(self):
        if self.codebase is None:
            return TypedPythonCodebase.FromRootlevelModule(object_database).serializationContext
        else:
            return self.codebase.instantiate().serializationContext

    def isThrottled(self):
        return self.timesBootedUnsuccessfully >= MAX_BAD_BOOTS

    def resetCounters(self):
        self.timesBootedUnsuccessfully = 0
        self.timesCrashed = 0
        self.lastFailureReason = None

    def effectiveTargetCount(self):
        if self.timesBootedUnsuccessfully >= MAX_BAD_BOOTS:
            return 0

        if self.isSingleton:
            return min(1, max(self.target_count, 0))
        else:
            return max(self.target_count, 0)

    def instantiateType(self, typename):
        modulename = ".".join(typename.split(".")[:-1])
        typename = typename.split(".")[-1]

        if self.codebase:
            module = self.codebase.instantiate(modulename)

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
        ) + ("" if not queryParams else "?" + urllib.parse.urlencode({k: str(v) for k, v in queryParams.items()}))

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

    def instantiateServiceType(self):
        """Instantiate the codebase and return the instance of the Service type.
        """
        if self.codebase:
            module = self.codebase.instantiate(self.service_module_name)

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
