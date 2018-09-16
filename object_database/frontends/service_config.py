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
import tempfile
from object_database.service_manager.ServiceManager import ServiceManager
from object_database import connect, service_schema, core_schema, ServiceBase
from object_database.util import configureLogging, formatTable, secondsToHumanReadable

def findGitParent(p_root):
    p = os.path.abspath(p_root)
    while True:
        if os.path.exists(os.path.join(p, ".git")):
            return p
        p = os.path.dirname(p)
        if not p:
            raise Exception("Can't find a git worktree at " + p_root)

def main(argv):
    configureLogging()

    parser = argparse.ArgumentParser("Install and configure services.")

    parser.add_argument("--hostname", default=os.getenv("ODB_HOST", "localhost"), required=False)
    parser.add_argument("--port", type=int, default=int(os.getenv("ODB_PORT", 8000)), required=False)

    subparsers = parser.add_subparsers()

    connections_parser = subparsers.add_parser('connections', help='list live connections')
    connections_parser.set_defaults(command='connections')

    install_parser = subparsers.add_parser('install', help='install a service')
    install_parser.set_defaults(command='install')
    install_parser.add_argument("--path", action='append', dest='paths')
    install_parser.add_argument("--class")
    install_parser.add_argument("--name", required=False)
    install_parser.add_argument("--placement", required=False, default='Any', choices=['Any','Master','Worker'])
    install_parser.add_argument("--singleton", required=False, action='store_true')

    configure_parser = subparsers.add_parser('configure', help='configure a service')
    configure_parser.set_defaults(command='configure')
    configure_parser.add_argument("name")
    configure_parser.add_argument("-l", "--local", action='store_true', help='use the local codebase, not the remote')
    configure_parser.add_argument("args", nargs=argparse.REMAINDER)
    
    list_parser = subparsers.add_parser('list', help='list installed services')
    list_parser.set_defaults(command='list')

    instances_parser = subparsers.add_parser('instances', help='list running service instances')
    instances_parser.set_defaults(command='instances')

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
    db.subscribeToSchema(core_schema, service_schema)

    if parsedArgs.command == 'connections':
        table = [['Connection ID']]

        with db.view():
            for c in sorted(core_schema.Connection.lookupAll(), key=lambda c: c._identity):
                table.append([c._identity])

        print(formatTable(table))

    if parsedArgs.command == 'configure':
        try:
            with tempfile.TemporaryDirectory() as tf:
                with db.transaction():
                    s = service_schema.Service.lookupAny(name=parsedArgs.name)

                    if parsedArgs.local:
                        svcClass = getattr(
                            service_schema.Codebase.instantiateFromLocalSource([findGitParent(os.getcwd())], s.service_module_name), 
                            s.service_class_name
                            )
                    else:
                        svcClass = s.instantiateServiceObject(tf)

                svcClass.configureFromCommandline(db, s, parsedArgs.args)
        except Exception as e:
            print("Failed to configure %s: %s" % (parsedArgs.name, e))
            return 1

    if parsedArgs.command == 'install':
        if parsedArgs.paths:
            paths = parsedArgs.paths
        else:
            paths = [findGitParent(os.getcwd())]

        gbRamUsed = 1
        coresUsed = 1

        with db.transaction():
            fullClassname = getattr(parsedArgs, 'class')
            modulename, classname = fullClassname.rsplit(".",1)
            
            if modulename.startswith("object_database"):
                def _getobject(modname, attribute):
                    mod = __import__(modname, fromlist=[attribute])
                    return mod.__dict__[attribute]

                actualClass = _getobject(modulename, classname)


                if not isinstance(actualClass, type):
                    print("Named class %s is not a type." % fullClassname)
                    return 1

                if not issubclass(actualClass, ServiceBase):
                    print("Named class %s is not a ServiceBase subclass." % fullClassname)
                    return 1

                ServiceManager.createService(actualClass, classname, placement=parsedArgs.placement, isSingleton=parsedArgs.singleton)
            else:
                codebase = service_schema.Codebase.create(paths)

                #make sure the codebase is importable, etc
                module = service_schema.Codebase.instantiateFromLocalSource(paths, modulename)

                if module is None:
                    print("Can't find", module, "in the codebase")

                actualClass = module.__dict__.get(classname, None)
                if actualClass is None:
                    print("Can't find", module, "in module", modulename)

                if actualClass is None:
                    print("Can't find", classname, "in the codebase")
                    return 1

                if not isinstance(actualClass, type):
                    print("Named class %s is not a type." % fullClassname)
                    return 1

                if not issubclass(actualClass, ServiceBase):
                    print("Named class %s is not a ServiceBase subclass." % fullClassname)
                    return 1

                coresUsed = actualClass.coresUsed
                gbRamUsed = actualClass.gbRamUsed

                if not parsedArgs.name:
                    name = fullClassname.split(".")[-1]
                else:
                    name = parsedArgs.name

                ServiceManager.createServiceWithCodebase(codebase, fullClassname, name, targetCount=None, 
                        placement=parsedArgs.placement, coresUsed=coresUsed, gbRamUsed=gbRamUsed)

    if parsedArgs.command == 'list':
        table = [['Service', 'Codebase', 'Module', 'Class', 'Placement', 'TargetCount', 'Cores', 'RAM']]

        with db.view():
            for s in sorted(service_schema.Service.lookupAll(), key=lambda s:s.name):
                table.append([
                    s.name,
                    str(s.codebase),
                    s.service_module_name,
                    s.service_class_name,
                    s.placement,
                    str(s.target_count),
                    s.coresUsed,
                    s.gbRamUsed
                    ])

        print(formatTable(table))

    if parsedArgs.command == 'instances':
        table = [['Service', 'Host', 'Connection', 'State']]

        with db.view():
            for s in sorted(service_schema.ServiceInstance.lookupAll(), key=lambda s:(s.service.name, s.host.hostname, s.state)):
                table.append([
                    s.service.name,
                    s.host.hostname,
                    s.connection if s.connection.exists() else "<DEAD>",
                    s.state
                    ])

        print(formatTable(table))

    if parsedArgs.command == 'hosts':
        table = [['Connection', 'IsMaster', 'Hostname', 'RAM USAGE', 'CORE USAGE', 'SERVICE COUNT']]

        with db.view():
            for s in sorted(service_schema.ServiceHost.lookupAll(), key=lambda s:s.hostname):
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
