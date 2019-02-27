#   Copyright 2019 Braxton Mckee
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
import subprocess
import sys
import tempfile

import object_database
from object_database.util import genToken
from object_database.service_manager.ServiceManager import ServiceManager

from object_database import (
    core_schema, connect, service_schema,
)

ownDir = os.path.dirname(os.path.abspath(__file__))

VERBOSE = True


class ServiceManagerTestCommon(object):
    WAIT_FOR_COUNT_TIMEOUT = 20.0 if os.environ.get('TRAVIS_CI', None) is not None else 5.0

    def schemasToSubscribeTo(self):
        """Subclasses can override to extend the schema set."""
        return []

    def waitRunning(self, serviceName):
        self.assertTrue(
            ServiceManager.waitRunning(self.database, serviceName, self.WAIT_FOR_COUNT_TIMEOUT),
            "Service " + serviceName + " never came up."
        )

    def setUp(self):
        self.token = genToken()
        self.tempDirObj = tempfile.TemporaryDirectory()
        self.tempDirectoryName = self.tempDirObj.name
        object_database.service_manager.Codebase.setCodebaseInstantiationDirectory(self.tempDirectoryName, forceReset=True)

        os.makedirs(os.path.join(self.tempDirectoryName, 'source'))
        os.makedirs(os.path.join(self.tempDirectoryName, 'storage'))

        if not VERBOSE:
            kwargs = {'stdout': subprocess.DEVNULL, 'stderr': subprocess.DEVNULL}
        else:
            kwargs = {}

        try:
            self.server = subprocess.Popen(
                [
                    sys.executable, os.path.join(ownDir, '..', 'frontends', 'service_manager.py'),
                    'localhost', 'localhost', "8023",
                    "--run_db",
                    '--source', os.path.join(self.tempDirectoryName, 'source'),
                    '--storage', os.path.join(self.tempDirectoryName, 'storage'),
                    '--service-token', self.token,
                    '--shutdownTimeout', '1.0',
                    '--ssl-path', os.path.join(ownDir, '..', '..', 'testcert.cert')
                ],
                **kwargs
            )
            # this should throw a subprocess.TimeoutExpired exception if the service did not crash
            self.server.wait(1.3)
        except subprocess.TimeoutExpired:
            pass
        else:
            raise Exception(
                f"Failed to start service_manager (retcode:{self.server.returncode})"
            )
        try:
            self.database = connect("localhost", 8023, self.token, retry=True)
            self.database.subscribeToSchema(core_schema, service_schema, *self.schemasToSubscribeTo())
        except Exception:
            self.server.terminate()
            self.server.wait()
            self.tempDirObj.cleanup()
            raise

    def newDbConnection(self):
        return connect("localhost", 8023, self.token, retry=True)

    def tearDown(self):
        self.server.terminate()
        self.server.wait()
        self.tempDirObj.__exit__(None, None, None)
