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


from object_database.view import revisionConflictRetry
from object_database.core_schema import core_schema
from object_database.service_manager.ServiceManagerSchema import service_schema

import logging
import traceback
import threading
import time

class ServiceManager(object):
    def __init__(self, dbConnectionFactory, isMaster, ownHostname, maxGbRam=4, maxCores=4):
        object.__init__(self)
        self.ownHostname = ownHostname
        self.isMaster = isMaster
        self.maxGbRam = maxGbRam
        self.maxCores = maxCores
        self.serviceHostObject = None
        self.dbConnectionFactory = dbConnectionFactory
        self.db = dbConnectionFactory()

        self.shouldStop = threading.Event()
        self.thread = threading.Thread(target=self.doWork)
        self.thread.daemon = True

        self.SLEEP_INTERVAL = 0.5

    def start(self):
        self.thread.start()

    def stop(self):
        self.shouldStop.set()
        self.thread.join()

    @staticmethod
    def createService(serviceClass, serviceName, target_count=None, placement="Any"):
        service = service_schema.Service.lookupAny(name=serviceName)

        if not service:
            service = service_schema.Service(name=serviceName, placement=placement)
            service.service_module_name = serviceClass.__module__
            service.service_class_name = serviceClass.__qualname__

        if target_count is not None:
            service.target_count = target_count

        if placement is not None:
            service.placement = placement

        return service

    @staticmethod
    def createServiceWithCodebase(codebase, className, serviceName, targetCount=0, placement="Any", coresUsed=None, gbRamUsed=None):
        assert len(className.split(".")) > 1, "className should be a fully-qualified module.classname"

        service = service_schema.Service.lookupAny(name=serviceName)

        if not service:
            service = service_schema.Service(name=serviceName, placement="Any")

        service.codebase = codebase
        service.service_module_name = ".".join(className.split(".")[:-1])
        service.service_class_name = className.split(".")[-1]
        
        if coresUsed is not None:
            service.coresUsed = coresUsed

        if gbRamUsed is not None:
            service.gbRamUsed = gbRamUsed

        if targetCount is not None:
            service.target_count = targetCount

        if placement is not None:
            service.placement = placement

        return service

    @staticmethod
    def startService(serviceName, targetCount = 1):
        service = service_schema.Service.lookupOne(name=serviceName)
        service.target_count = targetCount

    @staticmethod
    def waitRunning(db, serviceName, timeout=5.0):
        def isRunning():
            service = service_schema.Service.lookupAny(name=serviceName)
            if not service:
                return False
            for i in service_schema.ServiceInstance.lookupAll(service=service):
                if i.state == "Running":
                    return True
            return False

        return db.waitForCondition(isRunning,timeout)

    def stopAllServices(self, timeout):
        with self.db.transaction():
            for s in service_schema.Service.lookupAll():
                s.target_count = 0

        def allStopped():
            for s in service_schema.ServiceInstance.lookupAll():
                if not s.isNotRunning():
                    return False
            return True

        self.db.waitForCondition(allStopped, timeout)

    def doWork(self):
        with self.db.transaction():
            self.serviceHostObject = service_schema.ServiceHost(
                connection=self.db.connectionObject,
                isMaster=self.isMaster,
                maxGbRam=self.maxGbRam,
                maxCores=self.maxCores
                )
            self.serviceHostObject.hostname = self.ownHostname

        logging.info("ServiceManager starting work loop.")

        while not self.shouldStop.is_set():
            #redeploy our own services
            self.redeployServicesIfNecessary()

            #if we're the master, do some allocation
            if self.isMaster:
                self.collectDeadHosts()
                self.createInstanceRecords()

            instances = self.instanceRecordsToBoot()
                
            bad_instances = {}

            for i in instances:
                try:
                    with self.db.view():
                        self._startServiceWorker(i.service, i._identity)
                except:
                    logging.error("Failed to start a worker for instance %s:\n%s", i, traceback.format_exc())
                    bad_instances[i] = traceback.format_exc()

            if bad_instances:
                with self.db.transaction():
                    for i in bad_instances:
                        i.markFailedToStart(bad_instances[i])

            time.sleep(self.SLEEP_INTERVAL)

    
    @revisionConflictRetry
    def collectDeadHosts(self):
        #reset the state
        with self.db.transaction():
            for serviceHost in service_schema.ServiceHost.lookupAll():
                instances = service_schema.ServiceInstance.lookupAll(host=serviceHost)

                if not serviceHost.connection.exists():
                    for sInst in instances:
                        sInst.delete()
                    serviceHost.delete()
                else:
                    serviceHost.gbRamUsed = sum([i.service.gbRamUsed for i in instances if i.isActive()])
                    serviceHost.coresUsed = sum([i.service.coresUsed for i in instances if i.isActive()])

        with self.db.transaction():
            for serviceInstance in service_schema.ServiceInstance.lookupAll():
                if not serviceInstance.host.exists() or serviceInstance.connection and not serviceInstance.connection.exists():
                    serviceInstance.delete()

    def redeployServicesIfNecessary(self):
        needRedeploy = []
        with self.db.view():
            for i in service_schema.ServiceInstance.lookupAll(host=self.serviceHostObject):
                if i.service.codebase != i.codebase and i.connection is not None and not i.shouldShutdown:
                    needRedeploy.append(i)

            if needRedeploy:
                logging.info(
                    "The following services need to be stopped because their codebases are out of date:\n%s",
                    "\n".join(["  " + i.service.name + "." + i._identity + ". " 
                            + str(i.service.codebase) + " != " + str(i.codebase) for i in needRedeploy])
                    )

        if needRedeploy:
            self.stopServices(needRedeploy)

    @revisionConflictRetry
    def stopServices(self, needRedeploy):
        with self.db.transaction():
            for i in needRedeploy:
                if i.exists():
                    i.shouldShutdown = True

    @revisionConflictRetry
    def createInstanceRecords(self):
        actual_by_service = {}

        with self.db.view():
            for service in service_schema.Service.lookupAll():
                actual_by_service[service] = [
                    x for x in service_schema.ServiceInstance.lookupAll(service=service)
                        if x.isActive()
                    ]

        for service, actual_records in actual_by_service.items():
            with self.db.transaction():
                if service.target_count != len(actual_records):
                    self._updateService(service, actual_records)

    def _pickHost(self, service):
        for h in service_schema.ServiceHost.lookupAll():
            if h.connection.exists():
                if h.isMaster:
                    canPlace = service.placement in ("Master", "Any")
                else:
                    canPlace = service.placement in ("Worker", "Any")

                if canPlace and h.gbRamUsed + service.gbRamUsed <= h.maxGbRam and h.coresUsed + service.coresUsed <= h.maxCores:
                    return h

    def _updateService(self, service, actual_records):
        service.unbootable_count = 0

        while service.target_count > len(actual_records):
            host = self._pickHost(service)
            
            if not host:
                service.unbootable_count = service.target_count - len(actual_records)
                return

            instance = service_schema.ServiceInstance(
                service=service, 
                host=host,
                state="Booting",
                start_timestamp=time.time()
                )

            actual_records.append(instance)

        while service.target_count < len(actual_records):
            sInst = actual_records.pop()
            sInst.shouldShutdown = True

    def instanceRecordsToBoot(self):
        res = []
        with self.db.view():
            for i in service_schema.ServiceInstance.lookupAll(host=self.serviceHostObject):
                if i.state == "Booting":
                    res.append(i)
        return res

    def _startServiceWorker(self, service, instanceIdentity):
        raise NotImplementedError()
