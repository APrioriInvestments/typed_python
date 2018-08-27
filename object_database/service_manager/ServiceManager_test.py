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

from object_database.service_manager.SubprocessServiceManager import SubprocessServiceManager
from object_database.service_manager.ServiceBase import ServiceBase

from object_database import Schema, Indexed, Index, core_schema, TcpServer
from typed_python import *

import time
import numpy
import logging
import os

schema = Schema("core.ServiceManagerTest")

@schema.define
class TestServiceLastTimestamp:
    connection = Indexed(core_schema.Connection)
    lastPing = float
    triggerHardKill = bool
    triggerSoftKill = bool

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

    def doWork(self, shouldStop):
        while not shouldStop.is_set():
            time.sleep(0.01)

            with self.db.transaction():
                if self.conn.triggerSoftKill:
                    return

                if self.conn.triggerHardKill:
                    os._exit(1)

                self.conn.lastPing = time.time()

class ServiceManagerTest(unittest.TestCase):
    def setUp(self):
        self.dbServer = TcpServer("localhost", 8020)
        self.dbServer.start()
        self.database = self.dbServer.connect()

        self.serviceManager = SubprocessServiceManager("localhost", 8020)
        self.serviceManager.start()

    def tearDown(self):
        self.serviceManager.stop()
        self.dbServer.stop()

    def waitForCount(self, count):
        self.assertTrue(
            self.database.waitForCondition(
                lambda: TestServiceLastTimestamp.aliveCount() == count,
                timeout=5.0
                )
            )

    def test_starting_services(self):        
        with self.database.transaction():
            self.serviceManager.createService(TestService, "TestService", 1)

        self.waitForCount(1)

    def test_racheting_service_count_up_and_down(self):
        self.serviceManager.SLEEP_INTERVAL = 0.01

        with self.database.transaction():
            self.serviceManager.createService(TestService, "TestService", 1)

        numpy.random.seed(42)

        for count in numpy.random.choice(15,size=20):
            logging.info("Setting count for TestService to %s and waiting for it to be alive.", count)

            self.serviceManager.startService("TestService", int(count))

            self.waitForCount(count)

    def test_service_restarts_after_soft_kill(self):
        with self.database.transaction():
            self.serviceManager.createService(TestService, "TestService", 1)

        self.waitForCount(1)

        with self.database.transaction():
            s = TestServiceLastTimestamp.aliveServices()[0]
            s.triggerSoftKill = True

        self.database.waitForCondition(lambda: not s.connection.exists(), timeout=5.0)

        self.waitForCount(1)
    
    def test_service_restarts_after_killing(self):
        with self.database.transaction():
            self.serviceManager.createService(TestService, "TestService", 1)

        self.waitForCount(1)

        with self.database.transaction():
            s = TestServiceLastTimestamp.aliveServices()[0]
            s.triggerHardKill = True

        self.database.waitForCondition(lambda: not s.connection.exists(), timeout=5.0)

        self.waitForCount(1)
