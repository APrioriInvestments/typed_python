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

class ServiceBase:
    def __init__(self, db, serviceInstance):
        self.db = db
        self.serviceInstance = serviceInstance

    def initialize(self):
        pass

    def doWork(self, shouldStop):
        #subclasses actually do work in here.
        shouldStop.wait()