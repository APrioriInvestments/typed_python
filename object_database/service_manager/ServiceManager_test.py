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

from object_database.service_manager.ServiceManager import ServiceManager
from object_database.service_manager.ServiceBase import ServiceBase

from object_database import Schema, Indexed, Index, core_schema, TcpServer, connect, service_schema
from typed_python import *
import textwrap
import time
import numpy
import logging
import subprocess
import os
import sys

ownDir = os.path.dirname(os.path.abspath(__file__))

schema = Schema("core.ServiceManagerTest")

@schema.define
class TestServiceLastTimestamp:
    connection = Indexed(core_schema.Connection)
    lastPing = float
    triggerHardKill = bool
    triggerSoftKill = bool
    version = int

    @staticmethod
    def aliveServices(window = None):
        res = []

        for i in TestServiceLastTimestamp.lookupAll():
            if i.connection.exists() and (window is None or time.time() - i.lastPing < window):
                res.append(i)

        return res

    @staticmethod
    def aliveCount(window = None):
        return len(TestServiceLastTimestamp.aliveServices(window))

class TestService(ServiceBase):
    def initialize(self):
        with self.db.transaction():
            self.conn = TestServiceLastTimestamp(connection=self.db.connectionObject)
            self.version = 0

    def doWork(self, shouldStop):
        while not shouldStop.is_set():
            time.sleep(0.01)

            with self.db.transaction():
                if self.conn.triggerSoftKill:
                    return

                if self.conn.triggerHardKill:
                    os._exit(1)

                self.conn.lastPing = time.time()

def getTestServiceModule(version):
    return {
        'test_service/__init__.py': '',
        'test_service/service.py': textwrap.dedent("""
            from object_database import Schema, ServiceBase, Indexed, core_schema
            import os
            import time
            import logging

            schema = Schema("core.ServiceManagerTest")

            @schema.define
            class TestServiceLastTimestamp:
                connection = Indexed(core_schema.Connection)
                lastPing = float
                triggerHardKill = bool
                triggerSoftKill = bool
                version = int

            class Service(ServiceBase):
                def initialize(self):
                    with self.db.transaction():
                        self.conn = TestServiceLastTimestamp(connection=self.db.connectionObject)
                        self.conn.version = {version}

                def doWork(self, shouldStop):
                    while not shouldStop.is_set():
                        time.sleep(0.01)

                        with self.db.transaction():
                            if self.conn.triggerSoftKill:
                                return

                            if self.conn.triggerHardKill:
                                os._exit(1)

                            self.conn.lastPing = time.time()
            """.format(version=version))
    }

class ServiceManagerTest(unittest.TestCase):
    def setUp(self):
        self.server = subprocess.Popen(
            [sys.executable, os.path.join(ownDir, '..', 'frontends', 'service_manager.py'),
                'localhost', 'localhost', "8020", "--run_db"]
            )
        self.database = connect("localhost", 8020, retry=True)

    def tearDown(self):
        self.server.terminate()
        self.server.wait()

    def waitForCount(self, count):
        self.assertTrue(
            self.database.waitForCondition(
                lambda: TestServiceLastTimestamp.aliveCount() == count,
                timeout=5.0
                )
            )

    def test_starting_services(self):        
        with self.database.transaction():
            ServiceManager.createService(TestService, "TestService", target_count=1)

        self.waitForCount(1)

    def test_racheting_service_count_up_and_down(self):
        with self.database.transaction():
            ServiceManager.createService(TestService, "TestService", target_count=1)

        numpy.random.seed(42)

        for count in numpy.random.choice(15,size=20):
            logging.info("Setting count for TestService to %s and waiting for it to be alive.", count)

            with self.database.transaction():
                ServiceManager.startService("TestService", int(count))

            self.waitForCount(count)

    def test_service_restarts_after_soft_kill(self):
        with self.database.transaction():
            ServiceManager.createService(TestService, "TestService", target_count=1)

        self.waitForCount(1)

        with self.database.transaction():
            s = TestServiceLastTimestamp.aliveServices()[0]
            s.triggerSoftKill = True

        self.database.waitForCondition(lambda: not s.connection.exists(), timeout=5.0)

        self.waitForCount(1)
    
    def test_service_restarts_after_killing(self):
        with self.database.transaction():
            ServiceManager.createService(TestService, "TestService", target_count=1)

        self.waitForCount(1)

        with self.database.transaction():
            s = TestServiceLastTimestamp.aliveServices()[0]
            s.triggerHardKill = True

        self.database.waitForCondition(lambda: not s.connection.exists(), timeout=5.0)

        self.waitForCount(1)

    def test_update_module_code(self):
        with self.database.transaction():
            ServiceManager.createServiceWithCodebase(
                service_schema.Codebase.createFromFiles(getTestServiceModule(1)),
                "test_service.service.Service",
                "TestService",
                1
                )

        self.waitForCount(1)

        with self.database.transaction():
            s = TestServiceLastTimestamp.aliveServices()[0]
            self.assertEqual(s.version, 1)

        with self.database.transaction():
            ServiceManager.createServiceWithCodebase(
                service_schema.Codebase.createFromFiles(getTestServiceModule(2)),
                "test_service.service.Service",
                "TestService",
                1
                )

        self.database.waitForCondition(lambda: not s.connection.exists(), timeout=5.0)

        self.waitForCount(1)

        with self.database.transaction():
            s = TestServiceLastTimestamp.aliveServices()[0]
            self.assertEqual(s.version, 2)





