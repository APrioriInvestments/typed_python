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
import traceback
import logging
import time
import os
import weakref
import object_database

from object_database.web.cells import Card
from object_database.core_schema import core_schema
from object_database.service_manager.ServiceManagerSchema import service_schema

class ServiceRuntimeConfig:
    def __init__(self, serviceSourceRoot, serviceTemporaryStorageRoot):
        self.serviceSourceRoot = serviceSourceRoot
        self.serviceTemporaryStorageRoot = serviceTemporaryStorageRoot

_connectionToServiceSourceRoot = weakref.WeakKeyDictionary()

class ServiceBase:
    coresUsed = 1
    gbRamUsed = 1

    def __init__(self, db, serviceInstance, runtimeConfig):
        self.db = db
        self.serviceInstance = serviceInstance
        self.runtimeConfig = runtimeConfig

        _connectionToServiceSourceRoot[self.db] = runtimeConfig.serviceSourceRoot

        assert self.runtimeConfig.serviceSourceRoot is not None

    @staticmethod
    def currentServiceSourceRootImpliedByDbTransaction():
        return _connectionToServiceSourceRoot[object_database.current_transaction().db()]

    @staticmethod
    def associateServiceSourceRootWithDb(db, ssr):
        _connectionToServiceSourceRoot[db] = ssr

    @staticmethod
    def configureFromCommandline(db, serviceObject, args):
        """Subclasses should take the remaining args from the commandline and configure using them"""
        pass

    def initialize(self):
        pass

    def doWork(self, shouldStop):
        #subclasses actually do work in here.
        shouldStop.wait()

    @staticmethod
    def serviceDisplay(serviceObject, instance=None, objType=None, queryArgs=None):
        return Card("No details provided for service '%s'" % serviceObject.name)

