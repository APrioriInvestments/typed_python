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


import threading
import argparse
import sys
import time
import signal
import logging
import traceback
import logging.config

def configureLogging(preamble=""):
    logging.basicConfig(format='[%(asctime)s] %(levelname)6s %(filename)30s:%(lineno)4s' 
        + ("|" + preamble if preamble else '') 
        + '| %(message)s', level=logging.INFO
        )

from object_database import connect, TcpServer, RedisStringStore, InMemoryStringStore
from object_database.service_manager.SubprocessServiceManager import SubprocessServiceManager

def main(argv):
    parser = argparse.ArgumentParser("Run the main service manager and the object_database_service.")

    parser.add_argument("own_hostname")
    parser.add_argument("db_hostname")
    parser.add_argument("port", type=int)

    parser.add_argument("--run_db", default=False, action='store_true')
    
    #if populated, run a db_server as well
    parser.add_argument("--redis_port", type=int, default=None, required=False)
    
    configureLogging()

    parsedArgs = parser.parse_args(argv[1:])

    if parsedArgs.redis_port is not None and not parsedArgs.run_db:
        sys.stderr.write('error: please add --run_db if you want to run a database\n')
        parser.print_help()
        return 2

    shouldStop = threading.Event()

    def shutdownCleanly(signalNumber, frame):
        logging.info("Received signal %s. Stopping.", signalNumber)
        shouldStop.set()

    signal.signal(signal.SIGINT, shutdownCleanly)
    signal.signal(signal.SIGTERM, shutdownCleanly)

    object_database_port = parsedArgs.port

    databaseServer = None
    serviceManager = None

    try:
        if parsedArgs.run_db:
            databaseServer = TcpServer(
                parsedArgs.own_hostname, 
                object_database_port, 
                RedisStringStore(port=parsedArgs.redis_port) if parsedArgs.redis_port is not None else
                    InMemoryStringStore()
                )

            databaseServer.start()

            logging.info("Started a database server on %s:%s", parsedArgs.own_hostname, object_database_port)
    
        #verify we can connect first.
        db = connect(parsedArgs.db_hostname, parsedArgs.port)

        logging.info("Successfully connected to object database at %s:%s", parsedArgs.db_hostname, object_database_port)

        #start the process manager
        serviceManager = SubprocessServiceManager(parsedArgs.own_hostname, parsedArgs.db_hostname, parsedArgs.port)
        
        serviceManager.start()

        logging.info("Started serviceManager.")

        try:
            while not shouldStop.is_set():
                shouldStop.wait(timeout=1.0)
        except KeyboardInterrupt:
            return 0

        return 0
    finally:
        if databaseServer is not None:
            try:
                databaseServer.stop()
            except:
                logging.info("Failed to stop the database server:\n%s", traceback.format_exc())

        if serviceManager is not None:
            try:
                serviceManager.stop()
            except:
                logging.info("Failed to stop the service manager:\n%s", traceback.format_exc())

if __name__ == '__main__':
    sys.exit(main(sys.argv))
