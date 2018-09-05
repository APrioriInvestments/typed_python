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


import unittest
import requests
import os
import subprocess
import sys
import time
import asyncio
import websockets
import tempfile
from object_database.service_manager.ServiceManager import ServiceManager
from object_database.web.ActiveWebService import active_webservice_schema, ActiveWebService

from object_database import Schema, Indexed, Index, core_schema, TcpServer, connect, service_schema

ownDir = os.path.dirname(os.path.abspath(__file__))


class ActiveWebServiceTest(unittest.TestCase):
    def setUp(self):
        self.tempDirObj = tempfile.TemporaryDirectory()
        self.tempDirectoryName = self.tempDirObj.__enter__()

        self.server = subprocess.Popen(
            [sys.executable, os.path.join(ownDir, '..', 'frontends', 'service_manager.py'),
                'localhost', 'localhost', "8020", "--run_db",'--source',self.tempDirectoryName,
                '--shutdownTimeout', '.5'
                ]
            )
        try:
            self.database = connect("localhost", 8020, retry=True)
            self.database.subscribeToSchema(core_schema, service_schema, active_webservice_schema)

            with self.database.transaction():
                service = ServiceManager.createService(ActiveWebService, "ActiveWebService", target_count=0)
            ActiveWebService.configureFromCommandline(self.database, service, ['--port', '6000', '--host', 'localhost'])

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
                res = requests.get("http://localhost:6000/content/object_database.css")
                return
            except:
                time.sleep(.5)

        raise Exception("Webservice never came up.")

    def tearDown(self):
        self.server.terminate()
        self.server.wait()
        self.tempDirObj.__exit__(None, None, None)

    def test_start_web_service(self):
        res = requests.get("http://localhost:6000/content/object_database.css")
        self.assertEqual(res.status_code, 200)

    def test_connect_websocket(self):
        result = []
        async def hello(uri):
            async with websockets.connect(uri) as websocket:
                await websocket.send("Hello world!")
                result.append(await websocket.recv())

        asyncio.get_event_loop().run_until_complete(hello('ws://localhost:6000/echo'))

        self.assertEqual(result[0], "Hello world!")

