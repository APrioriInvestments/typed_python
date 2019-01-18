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
import numpy
import os
import psutil
import subprocess
import sys
import tempfile
import textwrap
import threading
import time
import unittest

import object_database
from object_database.service_manager.ServiceManager import ServiceManager
from object_database.service_manager.ServiceBase import ServiceBase
import object_database.service_manager.ServiceInstance as ServiceInstance
from object_database.util import genToken
from object_database.test_util import start_service_manager

from object_database import (
    Schema, Indexed, core_schema,
    connect, service_schema, current_transaction
)


ownDir = os.path.dirname(os.path.abspath(__file__))
ownName = os.path.basename(os.path.abspath(__file__))

schema = Schema("core.ServiceManagerTest")


@schema.define
class TestServiceCounter:
    k = int


@schema.define
class PointsToShow:
    timestamp = float
    y = float


@schema.define
class Feigenbaum:
    y = float


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



class GraphDisplayService(ServiceBase):
    def initialize(self):
        self.db.subscribeToSchema(core_schema, service_schema, schema)
        with self.db.transaction():
            if not Feigenbaum.lookupAny():
                Feigenbaum(y=2.0)

    @staticmethod
    def addAPoint():
        PointsToShow(timestamp = time.time(), y = len(PointsToShow.lookupAll()) ** 2.2)

    @staticmethod
    def serviceDisplay(serviceObject, instance=None, objType=None, queryArgs=None):
        ensureSubscribedType(PointsToShow)
        ensureSubscribedType(Feigenbaum)
        depth = Slot(50)

        return Tabs(
            Overlay=
                Card(Plot(lambda: {'single_array': [1,2,3,1,2,3],
                                'xy': {'x': [1,2,3,1,2,3], 'y': [4,5,6,7,8,9]},
                                }
                    ).width(600).height(400)
                ),
            Timestamps=
                Button("Add a point!", GraphDisplayService.addAPoint) +
                Card(Plot(GraphDisplayService.chartData)).width(600).height(400),
            feigenbaum=
                Dropdown("Depth", [(val, depth.setter(val)) for val in [10,50,100,250,500,750,1000]]) +
                Dropdown("Polynomial", [1.0, 1.5, 2.0], lambda polyVal: setattr(Feigenbaum.lookupAny(), 'y', float(polyVal))) +
                Card(Plot(lambda graph: GraphDisplayService.feigenbaum(graph, depth.get()))).width(600).height(400)
            )


    @staticmethod
    def chartData():
        points = sorted(PointsToShow.lookupAll(), key=lambda p: p.timestamp)

        return {'PointsToShow': {'timestamp': [p.timestamp for p in points], 'y': [p.y for p in points]}}

    @staticmethod
    def feigenbaum(linePlot, depth):
        if linePlot.curXYRanges.get() is None:
            left,right = 0.0, 4.0
        else:
            left, right = linePlot.curXYRanges.get()[0]
            left = max(0.0,left) if left is not None else 3
            right = min(4.0, right) if right is not None else 4
            left = min(left,right - 1e-6)
            right = max(left + 1e-6,right)

        values = numpy.linspace(left,right,1500,endpoint=True)

        def feigenbaum(values):
            x = numpy.ones(len(values)) * .5
            for _ in range(10000):
                x = (values * x * (1-x)) ** ((Feigenbaum.lookupAny().y) ** .5)

            its = []
            for _ in range(depth):
                x = (values * x * (1-x)) ** ((Feigenbaum.lookupAny().y) ** .5)
                its.append(x)

            return numpy.concatenate(its)

        fvals = feigenbaum(values)

        return {"feigenbaum": {'x': numpy.concatenate([values]*(len(fvals)//len(values))), 'y': fvals, 'type':'scattergl',
                'mode':'markers', 'opacity': .5, 'marker': {'size':2}}}


happy = Schema("core.test.happy")
@happy.define
class Happy:
    i = int

    def display(self, queryParams=None):
        ensureSubscribedType(Happy)
        return "Happy %s. " % self.i + str(queryParams)


class HappyService(ServiceBase):
    def initialize(self):
        pass

    @staticmethod
    def serviceDisplay(serviceObject, instance=None, objType=None, queryArgs=None):
        if not current_transaction().db().isSubscribedToType(Happy):
            raise SubscribeAndRetry(lambda db: db.subscribeToType(Happy))

        if instance:
            return instance.display(queryArgs)

        return Card(
            Subscribed(lambda: Text("There are %s happy objects" % len(Happy.lookupAll()))) +
            Expands(Text("Closed"),Subscribed(lambda: HappyService.serviceDisplay(serviceObject)))
            ) + Button("go to google", "http://google.com/") + SubscribedSequence(
                lambda: Happy.lookupAll(),
                lambda h: Button("go to the happy", serviceObject.urlForObject(h, x=10))
                )

    def doWork(self, shouldStop):
        self.db.subscribeToSchema(happy)

        with self.db.transaction():
            h = Happy(i = 1)
            h = Happy(i = 2)

        while not shouldStop.is_set():
            time.sleep(.5)
            with self.db.transaction():
                h = Happy()
            time.sleep(.5)
            with self.db.transaction():

                h.delete()


class StorageTest(ServiceBase):
    def initialize(self):
        with open(os.path.join(self.runtimeConfig.serviceTemporaryStorageRoot, "a.txt"), "w") as f:
            f.write("This exists")

        self.db.subscribeToSchema(core_schema, service_schema, schema)

        with self.db.transaction():
            self.conn = TestServiceLastTimestamp(connection=self.db.connectionObject)
            self.version = 0

    def doWork(self, shouldStop):
        shouldStop.wait()


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

VERBOSE = True


class ServiceManagerTest(unittest.TestCase):

    WAIT_FOR_COUNT_TIMEOUT = 20.0 if os.environ.get('TRAVIS_CI', None) is not None else 5.0

    def setUp(self):
        self.token = genToken()
        self.tempDirObj = tempfile.TemporaryDirectory()
        self.tempDirectoryName = self.tempDirObj.__enter__()
        object_database.service_manager.Codebase.setCodebaseInstantiationDirectory(self.tempDirectoryName, forceReset=True)

        os.makedirs(os.path.join(self.tempDirectoryName,'source'))
        os.makedirs(os.path.join(self.tempDirectoryName,'storage'))

        self.server = start_service_manager(
            self.tempDirectoryName,
            8023,
            self.token,
            timeout=2.0,
            verbose=VERBOSE
        )

        try:
            self.database = connect("localhost", 8023, self.token, retry=True)
            self.database.subscribeToSchema(core_schema, service_schema, schema)
        except Exception:
            self.server.terminate()
            self.server.wait()
            self.tempDirObj.__exit__(None,None,None)
            raise

    def tearDown(self):
        self.server.terminate()
        self.server.wait()
        self.tempDirObj.__exit__(None,None,None)

    def waitForCount(self, count):
        self.assertTrue(
            self.database.waitForCondition(
                lambda: TestServiceLastTimestamp.aliveCount() == count,
                timeout=self.WAIT_FOR_COUNT_TIMEOUT
                )
            )

    def test_starting_services(self):
        with self.database.transaction():
            ServiceManager.createOrUpdateService(TestService, "TestService", target_count=1)

        self.waitForCount(1)

    def test_service_storage(self):
        with self.database.transaction():
            ServiceManager.createOrUpdateService(StorageTest, "StorageTest", target_count=1)

        self.waitForCount(1)

    def test_starting_uninitializable_services(self):
        with self.database.transaction():
            svc = ServiceManager.createOrUpdateService(UninitializableService, "UninitializableService", target_count=1)

        self.assertTrue(
            self.database.waitForCondition(
                lambda: svc.timesBootedUnsuccessfully == ServiceInstance.MAX_BAD_BOOTS,
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
                lambda: svc.timesBootedUnsuccessfully == ServiceInstance.MAX_BAD_BOOTS,
                10
                )
            )


    def test_racheting_service_count_up_and_down(self):
        with self.database.transaction():
            ServiceManager.createOrUpdateService(TestService, "TestService", target_count=1)

        numpy.random.seed(42)

        for count in numpy.random.choice(6, size=20):
            logging.getLogger(__name__).info(
                "Setting count for TestService to %s and waiting for it to be alive.", count)

            with self.database.transaction():
                ServiceManager.startService("TestService", int(count))

            self.waitForCount(count)

        with self.database.transaction():
            ServiceManager.startService("TestService", 0)

        self.waitForCount(0)

        # make sure we don't have a bunch of zombie processes hanging underneath the service manager
        time.sleep(1.0)
        self.assertEqual(len(psutil.Process().children()[0].children()), 0)

    def test_shutdown_hanging_services(self):
        with self.database.transaction():
            ServiceManager.createOrUpdateService(HangingService, "HangingService", target_count=10)

        self.waitForCount(10)

        t0 = time.time()

        with self.database.transaction():
            ServiceManager.startService("HangingService", 0)

        self.waitForCount(0)

        self.assertTrue(time.time() - t0 < 2.0)

        # make sure we don't have a bunch of zombie processes hanging underneath the service manager
        time.sleep(1.0)
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

            i1 = v1.instantiate("test_service.service")
            i2 = v2.instantiate("test_service.service")
            i12 = v1.instantiate("test_service.service")
            i22 = v2.instantiate("test_service.service")

            self.assertTrue(i1.f() == 1)
            self.assertTrue(i2.f() == 2)
            self.assertTrue(i12.f() == 1)
            self.assertTrue(i22.f() == 2)

            self.assertTrue(i1 is i12)
            self.assertTrue(i2 is i22)



    def test_redeploy_hanging_services(self):
        with self.database.transaction():
            ServiceManager.createOrUpdateService(HangingService, "HangingService", target_count=10)

        self.waitForCount(10)

        with self.database.view():
            instances = service_schema.ServiceInstance.lookupAll()
            orig_codebase = instances[0].codebase

        with self.database.transaction():
            ServiceManager.createOrUpdateServiceWithCodebase(
                service_schema.Codebase.createFromFiles(getTestServiceModule(2)),
                "test_service.service.Service",
                "HangingService",
                10
                )

        # this should force a redeploy.
        maxProcessesEver = 0
        for i in range(40):
            maxProcessesEver = max(maxProcessesEver, len(psutil.Process().children()[0].children()))
            time.sleep(.1)

        self.database.flush()

        # after 2 seconds, we should be redeployed, but give Travis a bit more time
        if os.environ.get('TRAVIS_CI', None) is not None:
            time.sleep(5.0)

        with self.database.view():
            instances_redeployed = service_schema.ServiceInstance.lookupAll()

            self.assertEqual(len(instances), 10)
            self.assertEqual(len(instances_redeployed), 10)
            self.assertEqual(len(set(instances).intersection(set(instances_redeployed))), 0)

            self.assertTrue(orig_codebase != instances_redeployed[0].codebase)

        # and we never became too big!
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
            ServiceManager.createOrUpdateService(TestService, "TestService", target_count=0)


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

        # we want to ensure that we don't have some problem where our transaction throughput
        # goes down because we have left-over connections or something similar in the server,
        # which would be a real problem!
        self.assertTrue(emptyThroughputs[-1] * 2 > emptyThroughputs[0], (emptyThroughputs))

    def DISABLEDtest_throughput_with_many_workers(self):
        with self.database.transaction():
            ServiceManager.createOrUpdateService(TestService, "TestService", target_count=0)

        throughputs = []

        for ct in [16,18,20,22,24,26,28,30,32,34,0]:
            with self.database.transaction():
                ServiceManager.startService("TestService", ct)

            self.waitForCount(ct)

            throughputs.append(self.measureThroughput(5.0))

        print("Total throughput was", throughputs, " transactions per second")

    def test_service_restarts_after_soft_kill(self):
        with self.database.transaction():
            ServiceManager.createOrUpdateService(TestService, "TestService", target_count=1)

        self.waitForCount(1)

        with self.database.transaction():
            s = TestServiceLastTimestamp.aliveServices()[0]
            s.triggerSoftKill = True

        self.database.waitForCondition(lambda: not s.connection.exists(), timeout=5.0)

        self.waitForCount(1)

    def test_service_restarts_after_killing(self):
        with self.database.transaction():
            ServiceManager.createOrUpdateService(TestService, "TestService", target_count=1)

        self.waitForCount(1)

        with self.database.transaction():
            s = TestServiceLastTimestamp.aliveServices()[0]
            s.triggerHardKill = True

        self.database.waitForCondition(lambda: not s.connection.exists(), timeout=5.0)

        self.waitForCount(1)

    def test_update_module_code(self):
        serviceName = "TestService"

        def deploy_helper(codebase_version, expected_version, existing_service=None):
            with self.database.transaction():
                try:
                    ServiceManager.createOrUpdateServiceWithCodebase(
                        service_schema.Codebase.createFromFiles(getTestServiceModule(codebase_version)),
                        "test_service.service.Service",
                        serviceName,
                        targetCount=1
                        )
                except Exception:
                    pass

            if existing_service:
                self.database.waitForCondition(
                    lambda: not existing_service.connection.exists(), timeout=5.0)

            self.waitForCount(1)

            with self.database.transaction():
                s = TestServiceLastTimestamp.aliveServices()[0]
                self.assertEqual(s.version, expected_version)

            return s

        def lock_helper():
            with self.database.transaction():
                service = service_schema.Service.lookupAny(name=serviceName)
                self.assertIsNotNone(service)
                service.lock()
                self.assertTrue(service.isLocked)

        def unlock_helper():
            with self.database.transaction():
                service = service_schema.Service.lookupAny(name=serviceName)
                self.assertIsNotNone(service)
                service.unlock()
                self.assertFalse(service.isLocked)

        def prepare_helper():
            with self.database.transaction():
                service = service_schema.Service.lookupAny(name=serviceName)
                self.assertIsNotNone(service)
                service.prepare()
                self.assertFalse(service.isLocked)

        # Initial deploy should succeed
        s = deploy_helper(1, 1)

        # Trying to update the codebase without unlocking should fail
        s = deploy_helper(2, 1, s)

        # Trying to update the codebase after preparing for deployment should succeed
        prepare_helper()
        s = deploy_helper(3, 3, s)

        # Trying to update the codebase a second time after preparing for deployment should fail
        s = deploy_helper(4, 3, s)

        # Trying to update the codebase after unlocking should succeed
        unlock_helper()
        s = deploy_helper(5, 5, s)
        s = deploy_helper(6, 6, s)

        # Trying to update the codebase after locking should fail
        lock_helper()
        s = deploy_helper(7, 6, s)

