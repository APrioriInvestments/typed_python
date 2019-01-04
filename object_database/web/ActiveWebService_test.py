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
import requests
import subprocess
import sys
import tempfile
import time
import unittest
import websockets

from object_database.service_manager.ServiceManager import ServiceManager
from object_database.web.ActiveWebService import (
    active_webservice_schema,
    ActiveWebService
)
from object_database import core_schema, connect, service_schema
from object_database.util import genToken

ownDir = os.path.dirname(os.path.abspath(__file__))
ownName = os.path.basename(os.path.abspath(__file__))

DATABASE_SERVER_PORT=8023

WEB_SERVER_PORT=8025


class ActiveWebServiceTest(unittest.TestCase):
    def setUp(self):
        self.token = genToken()
        self.tempDirObj = tempfile.TemporaryDirectory()
        self.tempDirectoryName = self.tempDirObj.__enter__()
        log_level = logging.getLogger(__name__).getEffectiveLevel()
        log_level_name = logging.getLevelName(log_level)

        self.server = subprocess.Popen(
            [sys.executable, os.path.join(ownDir, '..', 'frontends', 'service_manager.py'),
                'localhost', 'localhost', str(DATABASE_SERVER_PORT), '--run_db',
                '--source', os.path.join(self.tempDirectoryName,'source'),
                '--storage', os.path.join(self.tempDirectoryName,'storage'),
                '--service-token', self.token,
                '--shutdownTimeout', '.5',
                '--log-level', log_level_name
            ]
        )
        try:
            # this should throw a subprocess.TimeoutExpired exception if the service did not crash
            self.server.wait(0.7)
        except subprocess.TimeoutExpired:
            pass
        else:
            raise Exception("Failed to start service_manager (retcode:{})"
                .format(self.server.returncode)
            )

        try:
            self.database = connect("localhost", DATABASE_SERVER_PORT, self.token, retry=True)

            self.database.subscribeToSchema(core_schema, service_schema, active_webservice_schema)

            with self.database.transaction():
                service = ServiceManager.createService(ActiveWebService, "ActiveWebService", target_count=0)
            ActiveWebService.configureFromCommandline(
                self.database,
                service,
                ['--port', str(WEB_SERVER_PORT), '--host', 'localhost', '--log-level', log_level_name]
            )

            with self.database.transaction():
                ServiceManager.startService("ActiveWebService", 1)

            self.waitUntilUp()
        except:
            self.server.terminate()
            self.server.wait()
            raise

    def waitUntilUp(self, timeout = 2.0):
        t0 = time.time()

        while time.time() - t0 < timeout:
            try:
                res = requests.get(
                    "http://localhost:{port}/content/object_database.css"
                    .format(port=WEB_SERVER_PORT)
                )
                return
            except Exception:
                time.sleep(.5)

        raise Exception("Webservice never came up.")

    def tearDown(self):
        self.server.terminate()
        self.server.wait()
        self.tempDirObj.__exit__(None, None, None)

    def test_start_web_service(self):
        res = requests.get(
            "http://localhost:{port}/content/object_database.css"
            .format(port=WEB_SERVER_PORT)
        )
        self.assertEqual(res.status_code, 200)

