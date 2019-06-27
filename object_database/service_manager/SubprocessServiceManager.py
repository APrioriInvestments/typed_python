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

import os
import shutil
import subprocess
import sys
import logging
import threading
import time
import traceback

from object_database.service_manager.ServiceManager import ServiceManager
from object_database.service_manager.ServiceSchema import service_schema
from object_database.util import validateLogLevel
from object_database import connect


ownDir = os.path.dirname(os.path.abspath(__file__))
_repoDir = None

def repoDir():
    if _repoDir is None:
        assert os.path.join(ownDir, '..').endswith('object_database')
        _repoDir = os.path.join(ownDir, '..', '..')
    return _repoDir


def parseLogfileToInstanceid(fname):
    """Parse a file name and return the integer instance id for the service."""
    if not fname.endswith(".log.txt") or "-" not in fname:
        return
    try:
        return int(fname[:-8].split("-")[-1])
    except ValueError:
        return


class SubprocessServiceManager(ServiceManager):
    def __init__(self, ownHostname, host, port,
                 sourceDir, storageDir, authToken,
                 isMaster, maxGbRam=4, maxCores=4, logfileDirectory=None,
                 shutdownTimeout=None, logLevelName="INFO",
                 metricUpdateInterval=2.0
                 ):
        self.cleanupLock = threading.Lock()
        self.host = host
        self.port = port
        self.storageDir = storageDir
        self.authToken = authToken
        self.logfileDirectory = logfileDirectory
        self.logLevelName = validateLogLevel(logLevelName, fallback='INFO')

        self.lock = threading.Lock()

        if logfileDirectory is not None:
            if not os.path.exists(logfileDirectory):
                os.makedirs(logfileDirectory)

        if not os.path.exists(storageDir):
            os.makedirs(storageDir)

        if not os.path.exists(sourceDir):
            os.makedirs(sourceDir)

        def dbConnectionFactory():
            return connect(host, port, self.authToken)

        ServiceManager.__init__(
            self, dbConnectionFactory, sourceDir, isMaster, ownHostname,
            maxGbRam=maxGbRam, maxCores=maxCores, shutdownTimeout=shutdownTimeout,
            metricUpdateInterval=metricUpdateInterval
        )
        self.serviceProcesses = {}
        self._logger = logging.getLogger(__name__)

    def startServiceWorker(self, service, instanceIdentity):
        assert isinstance(instanceIdentity, int)

        with self.db.view():
            if instanceIdentity in self.serviceProcesses:
                return

            with self.lock:
                logfileName = (
                    service.name
                    + "-" + str(instanceIdentity) + ".log.txt"
                )

                if self.logfileDirectory is not None:
                    output_file = open(os.path.join(self.logfileDirectory, logfileName), "w")
                else:
                    output_file = None
                process = subprocess.Popen(
                    [
                        sys.executable,
                        os.path.join(ownDir, '..', 'frontends', 'service_entrypoint.py'),
                        service.name,
                        self.host,
                        str(self.port),
                        str(instanceIdentity),
                        os.path.join(self.sourceDir, str(instanceIdentity)),
                        os.path.join(self.storageDir, str(instanceIdentity)),
                        self.authToken,
                        '--log-level', self.logLevelName
                    ],
                    cwd=self.storageDir,
                    stdin=subprocess.DEVNULL,
                    stdout=output_file,
                    stderr=subprocess.STDOUT
                )

                self.serviceProcesses[instanceIdentity] = process

                if output_file:
                    output_file.close()

            if self.logfileDirectory:
                self._logger.info(
                    "Started a service logging to %s with pid %s",
                    os.path.join(self.logfileDirectory, logfileName),
                    process.pid
                )
            else:
                self._logger.info(
                    "Started service %s/%s with pid %s",
                    service.name,
                    instanceIdentity,
                    process.pid
                )

    def stop(self, gracefully=True):
        if gracefully:
            self.stopAllServices(self.shutdownTimeout)

        ServiceManager.stop(self)

        with self.lock:
            for instanceIdentity, workerProcess in self.serviceProcesses.items():
                workerProcess.terminate()

            for instanceIdentity, workerProcess in self.serviceProcesses.items():
                workerProcess.wait()

        self.serviceProcesses = {}

    def cleanup(self):
        with self.cleanupLock:
            with self.lock:
                toCheck = list(self.serviceProcesses.items())

            for identity, workerProcess in toCheck:
                if workerProcess.poll() is not None:
                    workerProcess.wait()
                    with self.lock:
                        if identity in self.serviceProcesses:
                            del self.serviceProcesses[identity]

            with self.lock:
                toCheck = list(self.serviceProcesses.items())

            with self.db.view():
                for identity, workerProcess in toCheck:
                    serviceInstance = service_schema.ServiceInstance.fromIdentity(identity)

                    if (not serviceInstance.exists()
                            or (serviceInstance.shouldShutdown and
                                time.time() - serviceInstance.shutdownTimestamp > self.shutdownTimeout)):
                        if workerProcess:
                            workerProcess.terminate()
                            workerProcess.wait()

                            with self.lock:
                                if identity in self.serviceProcesses:
                                    del self.serviceProcesses[identity]

            self.cleanupOldLogfiles()

    def extractLogData(self, targetInstanceId, maxBytes):
        assert isinstance(targetInstanceId, int)

        if self.logfileDirectory:
            with self.lock:
                for file in os.listdir(self.logfileDirectory):
                    instanceId = parseLogfileToInstanceid(file)
                    if instanceId and instanceId == targetInstanceId:
                        fpath = os.path.join(self.logfileDirectory, file)
                        with open(fpath, "r") as f:
                            f.seek(0, 2)
                            curPos = f.tell()
                            f.seek(max(curPos-maxBytes, 0))

                            return f.read()

        return "<logfile not found>"

    def cleanupOldLogfiles(self):
        if self.logfileDirectory:
            with self.lock:
                for file in os.listdir(self.logfileDirectory):
                    instanceId = parseLogfileToInstanceid(file)

                    if instanceId is not None and not self.isLiveService(instanceId):
                        if not os.path.exists(os.path.join(self.logfileDirectory, "old")):
                            os.makedirs(os.path.join(self.logfileDirectory, "old"))
                        shutil.move(
                            os.path.join(self.logfileDirectory, file),
                            os.path.join(self.logfileDirectory, "old", file)
                        )

        if self.storageDir:
            with self.lock:
                if os.path.exists(self.storageDir):
                    for stringifiedInstanceId in os.listdir(self.storageDir):
                        path = os.path.join(self.storageDir, stringifiedInstanceId)

                        if stringifiedInstanceId.startswith(".coverage.") and os.path.isfile(path):
                            shutil.move(
                                path,
                                os.path.join(repoDir(), stringifiedInstanceId)
                            )

                        if os.path.isdir(path) and not self.isLiveService(stringifiedInstanceId):
                            try:
                                self._logger.info("Removing storage at path %s for dead service.", path)
                                shutil.rmtree(path)
                            except Exception:
                                self._logger.error(
                                    "Failed to remove storage at path %s for dead service:\n%s",
                                    path,
                                    traceback.format_exc()
                                )

        if self.sourceDir:
            with self.lock:
                if os.path.exists(self.sourceDir):
                    for stringifiedInstanceId in os.listdir(self.sourceDir):
                        if not self.isLiveService(stringifiedInstanceId):
                            try:
                                path = os.path.join(self.sourceDir, stringifiedInstanceId)
                                self._logger.info("Removing source caches at path %s for dead service.", path)
                                shutil.rmtree(path)
                            except Exception:
                                self._logger.error(
                                    "Failed to remove source cache at path %s for dead service:\n%s",
                                    path,
                                    traceback.format_exc()
                                )

    def isLiveService(self, instanceId):
        if isinstance(instanceId, str):
            try:
                instanceId = int(instanceId)
            except ValueError:
                return False

        return instanceId in self.serviceProcesses
