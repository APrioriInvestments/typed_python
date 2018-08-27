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
    def __init__(self, dbConnectionFactory):
        object.__init__(self)
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
    def createService(serviceClass, serviceName, targetCount=0):
        service = service_schema.Service.lookupAny(name=serviceName)

        if not service:
            service = service_schema.Service(name=serviceName)
            service.service_module_name = serviceClass.__module__
            service.service_class_name = serviceClass.__qualname__
            service.target_count = targetCount

        service.target_count = targetCount

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

        return self.db.waitForCondition(isRunning,timeout)

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
        #reset the state
        with self.db.transaction():
            for s in service_schema.Service.lookupAll():
                s.actual_count = 0
            for sInst in service_schema.ServiceInstance.lookupAll():
                sInst.delete()

        while not self.shouldStop.is_set():
            didOne = False

            instances = self.createInstanceRecords()
            
            bad_instances = []

            for i in instances:
                try:
                    with self.db.view():
                        self._startServiceWorker(i.service, i._identity)
                except:
                    logging.error("Failed to start a worker for instance %s:\n%s", i, traceback.format_exc())
                    bad_instances.append(i)

            if bad_instances:
                with self.db.transaction():
                    for i in bad_instances:
                        i.delete()

            if not instances:
                time.sleep(self.SLEEP_INTERVAL)

    @revisionConflictRetry
    def createInstanceRecords(self):
        res = []
        with self.db.transaction():
            for service in service_schema.Service.lookupAll():
                service.actual_count = len([
                    x for x in service_schema.ServiceInstance.lookupAll(service=service)
                        if x.isActive()
                    ])

                if service.target_count != service.actual_count:
                    res += self._updateService(service)

        return res

    def _updateService(self, service):
        updates = []

        while service.target_count > service.actual_count:
            instance = service_schema.ServiceInstance(
                service=service, 
                state="Booting",
                start_timestamp=time.time()
                )

            updates.append(instance)

            service.actual_count += 1

        while service.target_count < service.actual_count:
            for sInst in service_schema.ServiceInstance.lookupAll(service=service):
                if service.target_count < service.actual_count and sInst.isActive():
                    sInst.shouldShutdown = True
                    service.actual_count = service.actual_count - 1

        return updates

    def _startServiceWorker(self, service, instanceIdentity):
        raise NotImplementedError()
