#   Coyright 2017-2019 Nativepython Authors
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

import os
import tempfile

from object_database.service_manager.ServiceManager import ServiceManager
from object_database.service_manager.ServiceWorker import ServiceWorker
from object_database.service_manager.Codebase import setCodebaseInstantiationDirectory


class InProcessServiceManager(ServiceManager):
    def __init__(self, dbConnectionFactory, auth_token):
        self.storageRoot = tempfile.TemporaryDirectory()
        self.sourceRoot = tempfile.TemporaryDirectory()
        self.auth_token = auth_token

        setCodebaseInstantiationDirectory(self.sourceRoot.name)

        ServiceManager.__init__(
            self, dbConnectionFactory, self.sourceRoot.name, isMaster=True, ownHostname="localhost"
        )

        self.serviceWorkers = {}

    def startServiceWorker(self, service, instanceIdentity):
        if instanceIdentity in self.serviceWorkers:
            return

        worker = ServiceWorker(
            self.dbConnectionFactory(),
            self.dbConnectionFactory,
            instanceIdentity,
            os.path.join(self.storageRoot.name, str(instanceIdentity)),
            self.auth_token,
            '127.0.0.1'
        )

        self.serviceWorkers[instanceIdentity] = self.serviceWorkers.get(service, ()) + (worker,)

        worker.start()

    def stop(self):
        self.stopAllServices(10.0)

        for instanceId, workers in self.serviceWorkers.items():
            for worker in workers:
                worker.stop()

        self.serviceWorkers = {}

        ServiceManager.stop(self)

    def cleanup(self):
        self.storageRoot.cleanup()
        self.sourceRoot.cleanup()
