#!/usr/bin/env python3

#   Coyright 2017-2019 Nativepython Authors
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
import concurrent.futures
import logging
import multiprocessing
import os
import psutil
import resource
import signal
import subprocess
import sys
import tempfile
import threading
import time
import traceback

from object_database.util import (
    configureLogging,
    genToken,
    sslContextFromCertPathOrNone,
    validateLogLevel
)
from object_database import TcpServer, RedisPersistence, InMemoryPersistence, DisconnectedException
from object_database.service_manager.SubprocessServiceManager import SubprocessServiceManager


ownDir = os.path.dirname(os.path.abspath(__file__))


def startServiceManagerProcess(tempDirectoryName, port, authToken, *,
                               loglevelName="INFO",
                               timeout=1.0,
                               verbose=True,
                               ownHostname='localhost',
                               dbHostname='localhost',
                               runDb=True,
                               logDir=True,
                               sslPath=None):
    if not verbose:
        kwargs = dict(stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    else:
        kwargs = dict()

    cmd = [
        sys.executable,
        os.path.join(ownDir, 'service_manager.py'),
        ownHostname, dbHostname, str(port),
        '--service-token', authToken,
        '--shutdownTimeout', str(timeout),
        '--log-level', loglevelName,
        '--source', os.path.join(tempDirectoryName, 'source'),
        '--storage', os.path.join(tempDirectoryName, 'storage'),
    ]
    if runDb:
        cmd.append('--run_db')

    if logDir:
        cmd.extend(['--logdir', os.path.join(tempDirectoryName, 'logs')])
    if sslPath:
        cmd.extend(['--ssl-path', sslPath])

    server = subprocess.Popen(cmd, **kwargs)
    try:
        # this should throw a subprocess.TimeoutExpired exception if the service did not crash
        server.wait(timeout)
    except subprocess.TimeoutExpired:
        pass
    else:
        if server.returncode:
            msg = f"Failed to start service_manager (retcode:{server.returncode})"

            if verbose and server.stderr:
                error = b''.join(server.stderr.readlines())
                msg += "\n" + error.decode("utf-8")
            server.terminate()
            server.wait()
            raise Exception(msg)

    return server


def autoconfigureAndStartServiceManagerProcess(
        port=None, authToken=None, loglevelName=None, **kwargs):

    port = port or 8020
    authToken = authToken or genToken()

    if loglevelName is None:
        loglevelName = logging.getLevelName(
            logging.getLogger(__name__).getEffectiveLevel()
        )

    tempDirObj = tempfile.TemporaryDirectory()
    tempDirectoryName = tempDirObj.name

    server = startServiceManagerProcess(
        tempDirectoryName,
        port,
        authToken,
        loglevelName=loglevelName,
        **kwargs
    )

    def cleanupFn(error=False):
        server.terminate()
        try:
            server.wait(timeout=15.0)
        except subprocess.TimeoutExpired:
            logging.getLogger(__name__).warning(
                f"Failed to gracefully terminate service manager after 15 seconds."
                + " Sending KILL signal"
            )
            server.kill()
            try:
                server.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                logging.getLogger(__name__).warning(
                    f"Failed to kill service manager process."
                )

        if error or server.returncode:
            logging.getLogger(__name__).warning(
                "Exited with an error. Leaving temporary directory around for inspection: {}"
                .format(tempDirectoryName)
            )
        else:
            tempDirObj.cleanup()

    return server, cleanupFn


def main(argv=None):
    if argv is None:
        # this is a needed pathway for the 'console_scripts' in setup.py
        argv = sys.argv

    parser = argparse.ArgumentParser("Run the main service manager and the object_database_service.")

    parser.add_argument("own_hostname")
    parser.add_argument("db_hostname")
    parser.add_argument("port", type=int)
    parser.add_argument("--source", help="path for the source trees used by services", required=True)
    parser.add_argument("--storage", help="path for local storage used by services", required=True)
    parser.add_argument(
        "--service-token", type=str, required=True,
        help="the auth token to be used with this service"
    )
    parser.add_argument("--run_db", default=False, action='store_true')

    parser.add_argument("--ssl-path", default=None, required=False, help="path to (self-signed) SSL certificate")
    parser.add_argument("--redis_port", type=int, default=None, required=False)

    parser.add_argument("--max_gb_ram", type=float, default=None, required=False)
    parser.add_argument("--max_cores", type=int, default=None, required=False)
    parser.add_argument("--shutdownTimeout", type=float, default=None, required=False)

    parser.add_argument('--logdir', default=None, required=False)
    parser.add_argument("--log-level", required=False, default="INFO")

    parsedArgs = parser.parse_args(argv[1:])

    level_name = parsedArgs.log_level.upper()
    level_name = validateLogLevel(level_name, fallback='INFO')

    # getLevelName returns a name when given an int and an int when given a name,
    # and there doesn't seem to be an other way to get the int from the string.
    configureLogging("service_manager", level=logging.getLevelName(level_name))
    logger = logging.getLogger(__name__)

    parsedArgs = parser.parse_args(argv[1:])

    if parsedArgs.redis_port is not None and not parsedArgs.run_db:
        sys.stderr.write('error: please add --run_db if you want to run a database\n')
        parser.print_help()
        return 2

    logger.info(
        "ServiceManager on %s connecting to %s:%s",
        parsedArgs.own_hostname, parsedArgs.db_hostname, parsedArgs.port
    )
    shouldStop = threading.Event()

    def shutdownCleanly(signalNumber, frame):
        logger.info("Received signal %s. Stopping.", signalNumber)
        shouldStop.set()

    signal.signal(signal.SIGINT, shutdownCleanly)
    signal.signal(signal.SIGTERM, shutdownCleanly)

    resource.setrlimit(resource.RLIMIT_NOFILE, (2048, 4096))

    object_database_port = parsedArgs.port

    databaseServer = None
    serviceManager = None

    try:
        if parsedArgs.run_db:
            ssl_ctx = sslContextFromCertPathOrNone(parsedArgs.ssl_path)
            databaseServer = TcpServer(
                parsedArgs.own_hostname,
                object_database_port,
                RedisPersistence(port=parsedArgs.redis_port) if parsedArgs.redis_port is not None
                else InMemoryPersistence(),
                ssl_context=ssl_ctx,
                auth_token=parsedArgs.service_token
            )

            databaseServer.start()

            logger.info(
                "Started a database server on %s:%s",
                parsedArgs.own_hostname, object_database_port)

        serviceManager = None

        logger.info("Started object_database")

        try:
            while not shouldStop.is_set():
                if serviceManager is None:
                    try:
                        serviceManager = SubprocessServiceManager(
                            parsedArgs.own_hostname,
                            parsedArgs.db_hostname,
                            parsedArgs.port,
                            parsedArgs.source,
                            parsedArgs.storage,
                            parsedArgs.service_token,
                            isMaster=parsedArgs.run_db,
                            maxGbRam=parsedArgs.max_gb_ram or int(psutil.virtual_memory().total / 1024.0 / 1024.0 / 1024.0 + .1),
                            maxCores=parsedArgs.max_cores or multiprocessing.cpu_count(),
                            logfileDirectory=parsedArgs.logdir,
                            shutdownTimeout=parsedArgs.shutdownTimeout,
                            logLevelName=level_name
                        )
                        logger.info("Connected the service-manager")
                    except (ConnectionRefusedError, DisconnectedException, concurrent.futures._base.TimeoutError):
                        serviceManager = None

                    if serviceManager is None:
                        logger.error("Failed to connect to service manager. Sleeping and retrying")
                        time.sleep(10)
                    else:
                        serviceManager.start()
                else:
                    timeout = max(.1, serviceManager.shutdownTimeout / 10)
                    shouldStop.wait(timeout=timeout)
                    try:
                        serviceManager.cleanup()

                    except (ConnectionRefusedError, DisconnectedException, concurrent.futures._base.TimeoutError):
                        # try to reconnect
                        logger.error("Disconnected from object_database host. Attempting to reconnect.")
                        serviceManager.stop(gracefully=False)
                        serviceManager = None
                    except Exception:
                        logger.error("Service manager cleanup failed:\n%s", traceback.format_exc())
        except KeyboardInterrupt:
            logger.warning("Exiting due to KeyboardInterrupt")
            return 0

        return 0
    finally:
        if serviceManager is not None:
            try:
                serviceManager.stop(gracefully=True)
            except Exception:
                logger.error("Failed to stop the service manager:\n%s", traceback.format_exc())

        if databaseServer is not None:
            try:
                databaseServer.stop()
            except Exception:
                logger.error("Failed to stop the database server:\n%s", traceback.format_exc())


if __name__ == '__main__':
    sys.exit(main(sys.argv))
