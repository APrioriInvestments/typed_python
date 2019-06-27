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

import logging
import os
import subprocess
import sys
import tempfile
import time

import object_database
from object_database.util import genToken
from object_database.service_manager.ServiceManager import ServiceManager

from object_database import (
    core_schema, connect, service_schema,
)
from object_database.frontends import service_manager

ownDir = os.path.dirname(os.path.abspath(__file__))

VERBOSE = True
# Turn VERBOSE off on TravisCI because subprocess.PIPE seems to lock things up
VERBOSE = False if os.environ.get('TRAVIS_CI', None) else VERBOSE


class ServiceManagerTestCommon(object):
    WAIT_FOR_COUNT_TIMEOUT = 40.0 if os.environ.get('TRAVIS_CI', None) is not None else 5.0

    def schemasToSubscribeTo(self):
        """Subclasses can override to extend the schema set."""
        return []

    def waitRunning(self, serviceName):
        self.assertTrue(
            ServiceManager.waitRunning(self.database, serviceName, self.WAIT_FOR_COUNT_TIMEOUT),
            "Service " + serviceName + " never came up."
        )

    def timeElapsed(self):
        return time.time() - self.test_start_time

    def setUp(self):
        self.logger = logging.getLogger(__name__)
        self.test_start_time = time.time()
        self.token = genToken()
        self.tempDirObj = tempfile.TemporaryDirectory()
        self.tempDirectoryName = self.tempDirObj.name
        object_database.service_manager.Codebase.setCodebaseInstantiationDirectory(
            self.tempDirectoryName, forceReset=True
        )

        os.makedirs(os.path.join(self.tempDirectoryName, 'source'))
        os.makedirs(os.path.join(self.tempDirectoryName, 'storage'))
        os.makedirs(os.path.join(self.tempDirectoryName, 'logs'))

        self.logDir = os.path.join(self.tempDirectoryName, 'logs')

        logLevelName = logging.getLevelName(
            logging.getLogger(__name__).getEffectiveLevel()
        )

        if VERBOSE:
            kwargs = {}
        else:
            kwargs = {'stdout': subprocess.DEVNULL, 'stderr': subprocess.DEVNULL}

        self.server = service_manager.start_service_manager(
            self.tempDirectoryName, 8023, self.token,
            loglevelName=logLevelName,
            sslPath=os.path.join(ownDir, '..', '..', 'testcert.cert')
        )

        try:
            self.database = connect("localhost", 8023, self.token, retry=True)
            self.database.subscribeToSchema(core_schema, service_schema, *self.schemasToSubscribeTo())
        except Exception:
            self.logger.error(f"Failed to initialize for test")
            self.server.terminate()
            self.server.wait()
            self.tempDirObj.cleanup()
            raise

    def newDbConnection(self):
        return connect("localhost", 8023, self.token, retry=True)

    def tearDown(self):
        self.server.terminate()
        self.server.wait()
        self.tempDirObj.cleanup()
