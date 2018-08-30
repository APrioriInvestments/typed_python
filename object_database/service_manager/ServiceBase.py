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

from object_database.core_schema import core_schema
from object_database.service_manager.ServiceManagerSchema import service_schema

class ServiceBase:
    coresUsed = 1
    gbRamUsed = 1
    
    def __init__(self, db, serviceInstance):
        self.db = db
        self.serviceInstance = serviceInstance

    @staticmethod
    def configureFromCommandline(db, serviceObject, args):
        """Subclasses should take the remaining args from the commandline and configure using them"""
        pass

    def initialize(self):
        pass

    def doWork(self, shouldStop):
        #subclasses actually do work in here.
        shouldStop.wait()