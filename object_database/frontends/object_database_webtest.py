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


import argparse
import os
import subprocess
import sys
import tempfile
import time

from object_database.service_manager.ServiceManager import ServiceManager
from object_database.service_manager.ServiceManager_test import (
    GraphDisplayService,
    HappyService,
    UninitializableService
)
from object_database.web.ActiveWebService import (
    active_webservice_schema,
    ActiveWebService
)
from object_database import (
    connect,
    core_schema,
    current_transaction,
    Index,
    Indexed,
    Schema,
    service_schema,
    TcpServer
)
from object_database.util import tokenFromString


ownDir = os.path.dirname(os.path.abspath(__file__))
ownName = os.path.basename(os.path.abspath(__file__))


def main(argv=None):
    if argv is not None:
        argv = sys.argv

    with tempfile.TemporaryDirectory() as tf:
        try:
            server = subprocess.Popen(
                [sys.executable, os.path.join(ownDir, '..', 'frontends', 'service_manager.py'),
                    'localhost', 'localhost', '8020', '--run_db',
                    '--source', os.path.join(tf,'source'),
                    '--storage', os.path.join(tf,'storage'),
                    '--service-token', tokenFromString(ownName),
                    #'--logdir', os.path.join(tf,'logs'),
                    '--shutdownTimeout', '.5'
                    ]
                )

            database = connect("localhost", 8020, retry=True)
            database.subscribeToSchema(core_schema, service_schema, active_webservice_schema)

            with database.transaction():
                service = ServiceManager.createService(ActiveWebService, "ActiveWebService", target_count=0)

            ActiveWebService.configureFromCommandline(database, service, ['--port', '8000', '--host', '0.0.0.0'])

            with database.transaction():
                ServiceManager.startService("ActiveWebService", 1)

            with database.transaction():
                service = ServiceManager.createService(UninitializableService, "UninitializableService", target_count=1)

            with database.transaction():
                service = ServiceManager.createService(HappyService, "HappyService", target_count=1)

            with database.transaction():
                service = ServiceManager.createService(GraphDisplayService, "GraphDisplayService", target_count=1)



            while True:
                time.sleep(.1)
        finally:
            server.terminate()
            server.wait()

    return 0

if __name__ == '__main__':
    sys.exit(main())