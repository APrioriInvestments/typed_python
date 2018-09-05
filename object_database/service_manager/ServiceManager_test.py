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

import object_database.service_manager.ServiceManagerSchema as ServiceManagerSchema

from object_database import Schema, Indexed, Index, core_schema, TcpServer, connect, service_schema
from typed_python import *
import psutil
import threading
import textwrap
import time
import numpy
import tempfile
import logging
import subprocess
import os
import sys

ownDir = os.path.dirname(os.path.abspath(__file__))

schema = Schema("core.ServiceManagerTest")

@schema.define
class TestServiceCounter:
    k = int

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
        self.db.subscribeToSchema(core_schema, service_schema, schema)

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

class HangingService(ServiceBase):
    def initialize(self):
        self.db.subscribeToSchema(core_schema, service_schema, schema)
        
        with self.db.transaction():
            self.conn = TestServiceLastTimestamp(connection=self.db.connectionObject)
            self.version = 0

    def doWork(self, shouldStop):
        time.sleep(120)

class UninitializableService(ServiceBase):
    def initialize(self):
        assert False

    def doWork(self, shouldStop):
        time.sleep(120)

class CrashingService(ServiceBase):
    def initialize(self):
        assert False
        
    def doWork(self, shouldStop):
        time.sleep(.5)
        assert False

def getTestServiceModule(version):
    return {
        'test_service/__init__.py': '',
        'test_service/service.py': textwrap.dedent("""
            from object_database import Schema, ServiceBase, Indexed, core_schema, service_schema
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
                    self.db.subscribeToSchema(core_schema, service_schema, schema)

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
        self.tempDirObj = tempfile.TemporaryDirectory()
        self.tempDirectoryName = self.tempDirObj.__enter__()

        self.server = subprocess.Popen(
            [sys.executable, os.path.join(ownDir, '..', 'frontends', 'service_manager.py'),
                'localhost', 'localhost', "8020", "--run_db",'--source',self.tempDirectoryName,
                '--shutdownTimeout', '1.0'
                ]
            )
        self.database = connect("localhost", 8020, retry=True)
        self.database.subscribeToSchema(core_schema, service_schema, schema)

    def tearDown(self):
        self.server.terminate()
        self.server.wait()
        self.tempDirObj.__exit__(None,None,None)

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

    def test_starting_uninitializable_services(self):        
        with self.database.transaction():
            svc = ServiceManager.createService(UninitializableService, "UninitializableService", target_count=1)

        self.assertTrue(
            self.database.waitForCondition(
                lambda: svc.timesBootedUnsuccessfully == ServiceManagerSchema.MAX_BAD_BOOTS,
                10
                )
            )

        with self.database.view():
            self.assertEqual(svc.effectiveTargetCount(), 0)

        with self.database.transaction():
            svc.resetCounters()

        with self.database.view():
            self.assertEqual(svc.effectiveTargetCount(), 1)

        self.assertTrue(
            self.database.waitForCondition(
                lambda: svc.timesBootedUnsuccessfully == ServiceManagerSchema.MAX_BAD_BOOTS,
                10
                )
            )


    def test_racheting_service_count_up_and_down(self):
        with self.database.transaction():
            ServiceManager.createService(TestService, "TestService", target_count=1)

        numpy.random.seed(42)

        for count in numpy.random.choice(6,size=20):
            logging.info("Setting count for TestService to %s and waiting for it to be alive.", count)

            with self.database.transaction():
                ServiceManager.startService("TestService", int(count))

            self.waitForCount(count)

        with self.database.transaction():
            ServiceManager.startService("TestService", 0)

        self.waitForCount(0)        

        #make sure we don't have a bunch of zombie processes hanging underneath the service manager
        self.assertEqual(len(psutil.Process().children()[0].children()), 0)

    def test_shutdown_hanging_services(self):
        with self.database.transaction():
            ServiceManager.createService(HangingService, "HangingService", target_count=10)

        self.waitForCount(10)

        t0 = time.time()
        
        with self.database.transaction():
            ServiceManager.startService("HangingService", 0)

        self.waitForCount(0)

        self.assertTrue(time.time() - t0 < 2.0)

        #make sure we don't have a bunch of zombie processes hanging underneath the service manager
        self.assertEqual(len(psutil.Process().children()[0].children()), 0)

    def test_conflicting_codebases(self):
        with self.database.transaction():
            v1 = service_schema.Codebase.createFromFiles({
                'test_service/__init__.py': '',
                'test_service/helper/__init__.py': 'g = 1',
                'test_service/service.py': textwrap.dedent("""
                    import test_service.helper as helper
                    def f():
                        assert helper.g == 1
                        return 1
                """)
                })

            v2 = service_schema.Codebase.createFromFiles({
                'test_service/__init__.py': '',
                'test_service/helper/__init__.py': 'g = 2',
                'test_service/service.py': textwrap.dedent("""
                    import test_service.helper as helper
                    def f():
                        assert helper.g == 2
                        return 2
                """)
                })

            i1 = v1.instantiate(self.tempDirectoryName, "test_service.service")
            i2 = v2.instantiate(self.tempDirectoryName, "test_service.service")
            i12 = v1.instantiate(self.tempDirectoryName, "test_service.service")
            i22 = v2.instantiate(self.tempDirectoryName, "test_service.service")

            self.assertTrue(i1.f() == 1)
            self.assertTrue(i2.f() == 2)
            self.assertTrue(i12.f() == 1)
            self.assertTrue(i22.f() == 2)

            self.assertTrue(i1 is i12)
            self.assertTrue(i2 is i22)



    def test_redeploy_hanging_services(self):
        with self.database.transaction():
            ServiceManager.createService(HangingService, "HangingService", target_count=10)

        self.waitForCount(10)
        
        with self.database.view():
            instances = service_schema.ServiceInstance.lookupAll()
            orig_codebase = instances[0].codebase

        with self.database.transaction():
            ServiceManager.createServiceWithCodebase(
                service_schema.Codebase.createFromFiles(getTestServiceModule(2)),
                "test_service.service.Service",
                "HangingService",
                10
                )

        #this should force a redeploy. 
        maxProcessesEver = 0
        for i in range(20):
            maxProcessesEver = max(maxProcessesEver, len(psutil.Process().children()[0].children()))
            time.sleep(.1)

        #after 2 seconds, we should be redeployed
        with self.database.view():
            instances_redeployed = service_schema.ServiceInstance.lookupAll()

            self.assertEqual(len(instances), 10)
            self.assertEqual(len(instances_redeployed), 10)
            self.assertEqual(len(set(instances).intersection(set(instances_redeployed))), 0)

            self.assertTrue(orig_codebase != instances_redeployed[0].codebase)

        #and we never became too big!
        self.assertLess(maxProcessesEver, 11)
    
    def measureThroughput(self, seconds):
        t0 = time.time()

        with self.database.transaction():
            c = TestServiceCounter()

        while time.time() - t0 < seconds:
            with self.database.transaction():
                c.k = c.k + 1

        with self.database.view():
            return c.k / seconds

    def test_throughput_while_adjusting_servicecount(self):
        with self.database.transaction():
            ServiceManager.createService(TestService, "TestService", target_count=0)


        emptyThroughputs = [self.measureThroughput(1.0)]
        fullThroughputs = []

        for i in range(2):
            with self.database.transaction():
                ServiceManager.startService("TestService", 20)
            
            self.waitForCount(20)

            fullThroughputs.append(self.measureThroughput(1.0))

            with self.database.transaction():
                ServiceManager.startService("TestService", 0)

            self.waitForCount(0)

            emptyThroughputs.append(self.measureThroughput(1.0))

        print("Throughput with no workers: ", emptyThroughputs)
        print("Throughput with 20 workers: ", fullThroughputs)

        #we want to ensure that we don't have some problem where our transaction throughput
        #goes down because we have left-over connections or something similar in the server,
        #which would be a real problem!
        self.assertTrue(emptyThroughputs[-1] * 2 > emptyThroughputs[0])

    def DISABLEDtest_throughput_with_many_workers(self):
        with self.database.transaction():
            ServiceManager.createService(TestService, "TestService", target_count=0)

        throughputs = []

        for ct in [16,18,20,22,24,26,28,30,32,34,0]:
            with self.database.transaction():
                ServiceManager.startService("TestService", ct)
            
            self.waitForCount(ct)

            throughputs.append(self.measureThroughput(5.0))

        print("Total throughput was", throughputs, " transactions per second")

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





