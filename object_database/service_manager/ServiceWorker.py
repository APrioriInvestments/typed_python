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


from object_database.core_schema import core_schema
from object_database.service_manager.ServiceManagerSchema import service_schema
from object_database.service_manager.ServiceBase import ServiceBase

import traceback
import threading
import time
import logging
import tempfile


class ServiceWorker:
    def __init__(self, dbConnectionFactory, instance_id):
        self.dbConnectionFactory = dbConnectionFactory
        self.db = dbConnectionFactory()
        self.instance = service_schema.ServiceInstance.fromIdentity(instance_id)
        self.serviceObject = None
        self.serviceName = None

        self.serviceWorkerThread = threading.Thread(target=self.synchronouslyRunService)
        self.serviceWorkerThread.daemon = True
        self.shouldStop = threading.Event()

        self.shutdownPollThread = threading.Thread(target=self.checkForShutdown)
        self.shutdownPollThread.daemon = True

        self.tempdir = None

    def initialize(self):
        assert self.db.waitForCondition(lambda: self.instance.exists(), 5.0)

        with self.db.transaction():
            assert self.instance.exists(), "Service Instance object %s doesn't exist" % self.instance._identity
            self.serviceName = self.instance.service.name
            self.instance.connection = self.db.connectionObject
            self.instance.codebase = self.instance.service.codebase
            self.instance.start_timestamp = time.time()
            self.instance.state = "Initializing"

            try:
                self.serviceObject = self._instantiateServiceObject()
            except:
                logging.error('Service thread for %s failed:\n%s', self.instance._identity, traceback.format_exc())
                self.instance.state = "Failed"
                self.instance.failureReason = traceback.format_exc()
                self.instance.end_timestamp = time.time()
                return
        try:
            self.serviceObject.initialize()
        except:
            logging.error('Service thread for %s failed:\n%s', self.instance._identity, traceback.format_exc())
            
            self.serviceObject = None

            with self.db.transaction():
                self.instance.state = "Failed"
                self.instance.failureReason = traceback.format_exc()
                self.instance.end_timestamp = time.time()
                return

    def checkForShutdown(self):
        while not self.shouldStop.is_set():
            with self.db.view():
                if self.instance.shouldShutdown:
                    self.shouldStop.set()
                    return

            time.sleep(1.0)

    def synchronouslyRunService(self):
        self.initialize()

        if self.serviceObject is None:
            self.shouldStop.set()
            return

        with self.db.transaction():
            self.instance.state = "Running"

        try:
            self.serviceObject.doWork(self.shouldStop)
        except:
            logging.error("Service %s/%s failed: %s", 
                self.serviceName, 
                self.instance._identity, 
                traceback.format_exc()
                )
            
            with self.db.transaction():
                self.instance.state = "Failed"
                self.instance.end_timestamp = time.time()
                self.instance.failureReason = traceback.format_exc()
                return
        else:
            with self.db.transaction():
                logging.info(
                    "Service %s/%s exited gracefully. Setting stopped flag.", 
                    self.serviceName, 
                    self.instance._identity
                    )
                
                self.instance.state = "Stopped"
                self.instance.end_timestamp = time.time()

    def start(self):
        self.serviceWorkerThread.start()
        self.shutdownPollThread.start()

    def runAndWaitForShutdown(self):
        self.start()
        self.serviceWorkerThread.join()

    def stop(self):
        self.shouldStop.set()
        if self.serviceWorkerThread.isAlive():
            self.serviceWorkerThread.join()
        if self.shutdownPollThread.isAlive():
            self.shutdownPollThread.join()

        if self.tempdir:
            self.tempdir.__exit__(None, None, None)

    def _instantiateServiceObject(self):
        if self.instance.service.codebase:
            self.tempdir = tempfile.TemporaryDirectory()
            tempdirName = self.tempdir.__enter__()

            module = self.instance.service.codebase.instantiate(tempdirName, self.instance.service.service_module_name)

            if self.instance.service.service_class_name not in module.__dict__:
                raise Exception("Provided module %s at %s has no class %s. Options are:\n%s" % (
                    self.instance.service.service_module_name,
                    module.__file__,
                    self.instance.service.service_class_name,
                    "\n".join(["  " + x for x in sorted(module.__dict__)])
                    ))

            service_type = module.__dict__[self.instance.service.service_class_name]

            logging.info("ServiceWorker loaded module code for service %s.%s",
                self.instance.service.service_module_name,
                self.instance.service.service_class_name
                )
        else:
            def _getobject(modname, attribute):
                mod = __import__(modname, fromlist=[attribute])
                return mod.__dict__[attribute]

            service_type = _getobject(
                self.instance.service.service_module_name, 
                self.instance.service.service_class_name
                )
            
        assert isinstance(service_type, type), service_type
        assert issubclass(service_type, ServiceBase), service_type

        return service_type(self.db, self.instance)

    def isRunning(self):
        return self.serviceWorkerThread.isAlive()
