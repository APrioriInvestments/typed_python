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
import sys
import time

from object_database.persistence import InMemoryPersistence, RedisPersistence
from object_database.tcp_server import TcpServer
from object_database.util import sslContextFromCertPathOrNone


def main(argv):
    parser = argparse.ArgumentParser("Run an object_database server")

    parser.add_argument("host")
    parser.add_argument("port", type=int)
    parser.add_argument(
        "--service-token", type=str, required=True,
        help="the auth token to be used with this service"
    )
    parser.add_argument("--ssl-path", default=None, required=False, help="path to (self-signed) SSL certificate")
    parser.add_argument("--redis_port", type=int, default=None)
    parser.add_argument("--inmem", default=False, action='store_true')

    parsedArgs = parser.parse_args(argv[1:])

    if parsedArgs.inmem:
        mem_store = InMemoryPersistence()
    else:
        mem_store = RedisPersistence(port=parsedArgs.redis_port)

    ssl_ctx = sslContextFromCertPathOrNone(parsedArgs.ssl_path)
    databaseServer = TcpServer(
        parsedArgs.host,
        parsedArgs.port,
        mem_store,
        ssl_context=ssl_ctx,
        auth_token=parsedArgs.service_token
    )

    databaseServer.start()

    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        return


if __name__ == '__main__':
    main(sys.argv)
