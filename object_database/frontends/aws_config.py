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
import argparse
import sys
import time

from object_database import connect
from object_database.util import configureLogging, formatTable, secondsToHumanReadable
from object_database.service_manager.ServiceManager import ServiceManager
from object_database.service_manager.ServiceSchema import service_schema
from object_database.service_manager.aws.AwsWorkerBootService import AwsWorkerBootService, AwsApi
from object_database.service_manager.aws.AwsWorkerBootService import schema as aws_worker_boot_schema


def main(argv):
    parser = argparse.ArgumentParser("Control the AWS service")

    parser.add_argument("--hostname", default=os.getenv("ODB_HOST", "localhost"), required=False)
    parser.add_argument("--port", type=int, default=int(os.getenv("ODB_PORT", 8000)), required=False)
    parser.add_argument("--auth", type=str, default=os.getenv("ODB_AUTH_TOKEN"), required=False, help="Auth token to use to connect.")

    subparsers = parser.add_subparsers()

    config_parser = subparsers.add_parser('config', help='configure the service')
    config_parser.set_defaults(command='config')

    config_parser.add_argument('--region', required=False)
    config_parser.add_argument('--vpc_id', required=False)
    config_parser.add_argument('--subnet', required=False)
    config_parser.add_argument('--security_group', required=False)
    config_parser.add_argument('--keypair', required=False)
    config_parser.add_argument('--worker_name', required=False)
    config_parser.add_argument('--worker_iam_role_name', required=False)
    config_parser.add_argument('--docker_image', required=False)
    config_parser.add_argument('--defaultStorageSize', required=False, type=int)
    config_parser.add_argument('--max_to_boot', required=False, type=int)

    install_parser = subparsers.add_parser('install', help='install the service')
    install_parser.set_defaults(command='install')

    list_parser = subparsers.add_parser('list', help='list machines')
    list_parser.set_defaults(command='list')

    boot_parser = subparsers.add_parser('boot', help='set the number of desired boxes')
    boot_parser.set_defaults(command='boot')
    boot_parser.add_argument("instance_type")
    boot_parser.add_argument("count", type=int)

    killall_parser = subparsers.add_parser('killall', help='kill everything')
    killall_parser.set_defaults(command='killall')

    reset_parser = subparsers.add_parser('reset', help='kill everything')
    reset_parser.set_defaults(command='reset')

    configureLogging()

    parsedArgs = parser.parse_args(argv[1:])

    db = connect(parsedArgs.hostname, parsedArgs.port, parsedArgs.auth)
    db.subscribeToSchema(service_schema, lazySubscription=True)
    db.subscribeToSchema(aws_worker_boot_schema)

    if parsedArgs.command == 'reset':
        with db.transaction():
            for s in aws_worker_boot_schema.State.lookupAll():
                s.delete()

    if parsedArgs.command == 'config':
        with db.transaction():
            AwsWorkerBootService.configure(
                db_hostname=parsedArgs.hostname,
                db_port=parsedArgs.port,
                region=parsedArgs.region,
                vpc_id=parsedArgs.vpc_id,
                subnet=parsedArgs.subnet,
                security_group=parsedArgs.security_group,
                keypair=parsedArgs.keypair,
                worker_name=parsedArgs.worker_name,
                worker_iam_role_name=parsedArgs.worker_iam_role_name,
                docker_image=parsedArgs.docker_image,
                defaultStorageSize=parsedArgs.defaultStorageSize,
                max_to_boot=parsedArgs.max_to_boot
            )

    if parsedArgs.command == 'install':
        with db.transaction():
            ServiceManager.createOrUpdateService(AwsWorkerBootService, "AwsWorkerBootService", placement="Master")

    if parsedArgs.command == 'list':
        print()
        print()

        with db.view():
            api = AwsApi()
            booted = AwsWorkerBootService.currentBooted()
            targets = AwsWorkerBootService.currentTargets()

            table = [["Instance Type", "Booted", "Desired"]]

            for instance_type in sorted(set(list(booted) + list(targets))):
                table.append([
                    instance_type,
                    booted.get(instance_type, 0),
                    targets.get(instance_type, 0)
                ])

            print(formatTable(table))

        print()
        print()

        with db.view():
            api = AwsApi()
            table = [["InstanceID", "InstanceType", "IP", "Uptime"]]
            for instanceId, instance in api.allRunningInstances(spot=None).items():
                table.append([
                    instance['InstanceId'],
                    instance['InstanceType'],
                    instance['PrivateIpAddress'],
                    secondsToHumanReadable(time.time() - instance['LaunchTime'].timestamp())
                ])
            print(formatTable(table))

        print()
        print()

    if parsedArgs.command == 'boot':
        with db.transaction():
            AwsWorkerBootService.setBootState(parsedArgs.instance_type, parsedArgs.count)

    if parsedArgs.command == 'killall':
        with db.transaction():
            AwsWorkerBootService.shutdownAll()

        with db.view():
            api = AwsApi()

            for i in api.allRunningInstances():
                try:
                    api.terminateInstanceById(i)
                except Exception:
                    pass

    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
