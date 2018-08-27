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


from object_database.service_manager.ServiceManager import ServiceManager
from object_database.service_manager.ServiceWorker import ServiceWorker
from object_database import connect

import logging
import sys
import subprocess
import os

ownDir = os.path.dirname(os.path.abspath(__file__))

class SubprocessServiceManager(ServiceManager):
    def __init__(self, host, port, logfileDirectory=None):
        self.host = host
        self.port = port
        self.logfileDirectory = logfileDirectory

        if logfileDirectory is not None:
            if not os.path.exists(logfileDirectory):
                os.makedirs(logfileDirectory)

        def dbConnectionFactory():
            return connect(host, port)

        ServiceManager.__init__(self, dbConnectionFactory)

        self.serviceProcesses = {}

    def _startServiceWorker(self, service, instanceIdentity):
        logfileName = service.name + "_" + instanceIdentity + ".log.txt"

        if self.logfileDirectory is not None:
            output_file = open(os.path.join(self.logfileDirectory, logfileName), "w")
        else:
            output_file = None
        
        process = subprocess.Popen(
            [sys.executable, os.path.join(ownDir, 'service_entrypoint.py'),
             self.host, str(self.port), instanceIdentity],
            stdin=subprocess.DEVNULL,
            stdout=output_file,
            stderr=subprocess.STDOUT
            )

        self.serviceProcesses[service] = self.serviceProcesses.get(service, ()) + (process,)

        if output_file:
            output_file.close()

        if self.logfileDirectory:
            logging.info(
                "Started a service logging to %s with pid %s",
                os.path.join(self.logfileDirectory, logfileName),
                process.pid
                )
        else:
            logging.info(
                "Started service %s/%s with pid %s",
                service.name,
                instanceIdentity,
                process.pid
                )

    def stop(self, timeout=10.0):
        self.stopAllServices(timeout)

        for service, workers in self.serviceProcesses.items():
            for worker in workers:
                worker.terminate()

        self.serviceProcesses = {}

        ServiceManager.stop(self)
