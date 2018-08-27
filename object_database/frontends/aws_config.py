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

def formatTable(rows):
    cols = [[r[i] for r in rows] for i in range(len(rows[0]))]
    colWidth = [max([len(c) for c in col]) for col in cols]

    formattedRows = [
        "  ".join(row[col] + " " * (colWidth[col] - len(row[col])) for col in range(len(cols)))
            for row in rows
        ]
    formattedRows = formattedRows[:1] + [
        "  ".join("-" * colWidth[col] for col in range(len(cols)))
        ] + formattedRows[1:]

    return "\n".join(formattedRows)

def configureLogging(preamble=""):
    logging.basicConfig(format='[%(asctime)s] %(levelname)6s %(filename)30s:%(lineno)4s' 
        + ("|" + preamble if preamble else '') 
        + '| %(message)s', level=logging.ERROR
        )

def secondsToHumanReadable(seconds):
    if seconds < 120:
        return "%.2f seconds" % (seconds)
    if seconds < 120 * 60:
        return "%.2f minutes" % (seconds / 60)
    if seconds < 120 * 60 * 24:
        return "%.2f hours" % (seconds / 60 / 60)
    return "%.2f days" % (seconds / 60 / 60 / 24)

from object_database import connect, InMemServer
from object_database.service_manager.aws.AwsWorkerBootService import AwsWorkerBootService, AwsApi

def main(argv):
    parser = argparse.ArgumentParser("Control the AWS service")

    parser.add_argument("--inmem", default=False, action='store_true')
    parser.add_argument("--hostname", default=os.getenv("ODB_HOST", "localhost"), required=False)
    parser.add_argument("--port", type=int, default=int(os.getenv("ODB_PORT", 8000)), required=False)

    parser.add_argument('--region', required=False)
    parser.add_argument('--vpc_id', required=False)
    parser.add_argument('--subnet', required=False)
    parser.add_argument('--security_group', required=False)
    parser.add_argument('--keypair', required=False)
    parser.add_argument('--worker_name', required=False)
    parser.add_argument('--worker_iam_role_name', required=False)
    parser.add_argument('--linux_ami', required=False)
    parser.add_argument('--defaultStorageSize', required=False, type=int)
    parser.add_argument('--max_to_boot', required=False, type=int)

    parser.add_argument('--configure', required=False, action='store_true')
    parser.add_argument('--list', required=False, action='store_true')
    parser.add_argument('--boot', required=False)
    parser.add_argument('--kill', required=False)

    configureLogging()

    parsedArgs = parser.parse_args(argv[1:])

    if parsedArgs.inmem:
        db = InMemServer().connect()
        assert (parsedArgs.region
            and parsedArgs.vpc_id
            and parsedArgs.subnet
            and parsedArgs.security_group
            and parsedArgs.keypair
            and parsedArgs.worker_name
            and parsedArgs.worker_iam_role_name
            and parsedArgs.linux_ami
            and parsedArgs.defaultStorageSize
            and parsedArgs.max_to_boot)
    else:
        db = connect(parsedArgs.hostname, parsedArgs.port)

    if parsedArgs.configure or parsedArgs.inmem:
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
                linux_ami=parsedArgs.linux_ami,
                defaultStorageSize=parsedArgs.defaultStorageSize,
                max_to_boot=parsedArgs.max_to_boot
                )

    if parsedArgs.list:
        with db.view():
            api = AwsApi()
            table = [["InstanceID", "InstanceType", "IP", "Uptime"]]
            for instanceId, instance in api.allRunningInstances().items():
                table.append([
                    instance['InstanceId'],
                    instance['InstanceType'], 
                    instance['PrivateIpAddress'], 
                    secondsToHumanReadable(time.time() - instance['LaunchTime'].timestamp())
                    ])
            print(formatTable(table))

    if parsedArgs.boot:
        with db.view():
            AwsWorkerBootService.bootOneDirectly(parsedArgs.boot)

    if parsedArgs.kill:
        with db.view():
            api = AwsApi()
            api.terminateInstanceById(parsedArgs.kill)

    return 0

if __name__ == '__main__':
    sys.exit(main(sys.argv))
