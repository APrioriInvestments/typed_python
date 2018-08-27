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
import threading
import argparse
import sys
import time
import signal
import logging
import traceback
import logging.config

from object_database import connect, service_schema
from object_database.util import configureLogging, formatTable, secondsToHumanReadable

def main(argv):
    configureLogging()

    parser = argparse.ArgumentParser("Install and configure services.")

    parser.add_argument("--hostname", default=os.getenv("ODB_HOST", "localhost"), required=False)
    parser.add_argument("--port", type=int, default=int(os.getenv("ODB_PORT", 8000)), required=False)

    subparsers = parser.add_subparsers()

    list_parser = subparsers.add_parser('list', help='list installed services')
    list_parser.set_defaults(command='list')

    hosts_parser = subparsers.add_parser('hosts', help='list running hosts')
    hosts_parser.set_defaults(command='hosts')

    start_parser = subparsers.add_parser('start', help='Start (or change target replicas for) a service')
    start_parser.set_defaults(command='start')
    start_parser.add_argument("name")
    start_parser.add_argument("--count", type=int, default=1, required=False)

    stop_parser = subparsers.add_parser('stop', help='Stop a service')
    stop_parser.set_defaults(command='stop')
    stop_parser.add_argument("name")
    
    parsedArgs = parser.parse_args(argv[1:])

    db = connect(parsedArgs.hostname, parsedArgs.port)

    if parsedArgs.command == 'list':
        table = [['Service', 'Codebase', 'Module', 'Class', 'Placement', 'TargetCount']]

        with db.view():
            for s in service_schema.Service.lookupAll():
                table.append([
                    s.name,
                    str(s.codebase),
                    s.service_module_name,
                    s.service_class_name,
                    s.placement,
                    str(s.target_count)
                    ])

        print(formatTable(table))

    if parsedArgs.command == 'hosts':
        table = [['Connection', 'IsMaster', 'Hostname', 'RAM USAGE', 'CORE USAGE', 'SERVICE COUNT']]

        with db.view():
            for s in service_schema.ServiceHost.lookupAll():
                table.append([
                    s.connection._identity,
                    str(s.isMaster),
                    s.hostname,
                    "%.1f / %.1f" % (s.gbRamUsed, s.maxGbRam),
                    "%s / %s" % (s.coresUsed, s.maxCores),
                    str(len(service_schema.ServiceInstance.lookupAll(host=s)))
                    ])

        print(formatTable(table))

    if parsedArgs.command == 'start':
        with db.transaction():
            s = service_schema.Service.lookupAny(name=parsedArgs.name)
            if not s:
                print("Can't find a service named", s)
                return 1

            s.target_count = max(parsedArgs.count, 0)

    if parsedArgs.command == 'stop':
        with db.transaction():
            s = service_schema.Service.lookupAny(name=parsedArgs.name)
            if not s:
                print("Can't find a service named", s)
                return 1

            s.target_count = 0

    return 0

if __name__ == '__main__':
    sys.exit(main(sys.argv))
