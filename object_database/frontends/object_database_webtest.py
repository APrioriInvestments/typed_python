#!/usr/bin/env python3

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

import os
import sys
import tempfile
import textwrap
import time

from object_database.service_manager.ServiceManager import ServiceManager

from object_database.service_manager.ServiceManager_test import (
    GraphDisplayService,
    TextEditorService,
    HappyService,
    # UninitializableService,
    DropdownTestService,
    BigGridTestService
)

from object_database.web.CellsTestService import CellsTestService

from object_database.web.ActiveWebServiceSchema import (
    active_webservice_schema,
)
from object_database.web.ActiveWebService import (
    ActiveWebService
)
from object_database import (
    connect,
    core_schema,
    service_schema,

)
from object_database.test_util import start_service_manager
from object_database.util import genToken
from object_database.web.LoginPlugin import LoginIpPlugin


ownDir = os.path.dirname(os.path.abspath(__file__))


def main(argv=None):
    if argv is not None:
        argv = sys.argv

    token = genToken()
    port = 8020
    loglevel_name = 'INFO'

    with tempfile.TemporaryDirectory() as tmpDirName:
        try:
            server = start_service_manager(tmpDirName, port, token,
                                           loglevel_name=loglevel_name)

            database = connect("localhost", port, token, retry=True)
            database.subscribeToSchema(core_schema, service_schema,
                                       active_webservice_schema)

            with database.transaction():
                service = ServiceManager.createOrUpdateService(
                    ActiveWebService, "ActiveWebService", target_count=0)

            ActiveWebService.configureFromCommandline(
                database, service,
                ['--port', '8000',
                 '--host', '0.0.0.0',
                 '--log-level', loglevel_name]
            )

            ActiveWebService.setLoginPlugin(
                database,
                service,
                LoginIpPlugin,
                [None],
                config={'company_name': 'A Testing Company'}
            )

            with database.transaction():
                ServiceManager.startService("ActiveWebService", 1)

            with database.transaction():
                service = ServiceManager.createOrUpdateService(
                    CellsTestService, "CellsTestService",
                    target_count=1
                )

            with database.transaction():
                service = ServiceManager.createOrUpdateService(
                    CellsTestService, "CellsTestService",
                    target_count=1
                )

            with database.transaction():
                service = ServiceManager.createOrUpdateService(
                    HappyService, "HappyService", target_count=1
                )
            with database.transaction():
                service = ServiceManager.createOrUpdateService(
                    GraphDisplayService, "GraphDisplayService", target_count=1
                )
            with database.transaction():
                service = ServiceManager.createOrUpdateService(
                    TextEditorService, "TextEditorService", target_count=1
                )

            with database.transaction():
                service = ServiceManager.createOrUpdateService(
                    DropdownTestService, "DropdownTestService", target_count=1
                )

            with database.transaction():
                service = ServiceManager.createOrUpdateService(
                    BigGridTestService, "BigGridTestService", target_count=1
                )

            with database.transaction():
                ServiceManager.createOrUpdateServiceWithCodebase(
                    service_schema.Codebase.createFromFiles({
                        'test_service/__init__.py': '',
                        'test_service/service.py': textwrap.dedent("""
                            from object_database.service_manager.ServiceBase import ServiceBase

                            class TestService(ServiceBase):
                                gbRamUsed = 0
                                coresUsed = 0

                                def initialize(self):
                                    with self.db.transaction():
                                        self.runtimeConfig.serviceInstance.statusMessage = "Loaded"

                                def doWork(self, shouldStop):
                                    shouldStop.wait()

                                def display(self, queryParams=None):
                                    return "test service display message"
                        """)
                    }),
                    "test_service.service.TestService",
                    "TestService",
                    10
                )

            print("SERVER IS BOOTED")

            while True:
                time.sleep(.1)
        finally:
            server.terminate()
            server.wait()

    return 0


if __name__ == '__main__':
    sys.exit(main())
