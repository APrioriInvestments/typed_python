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

import traceback
import threading
import logging
import time
import base64
import json
import sys
import time
import argparse
import traceback
import datetime
import os
import json
import gevent.socket

from gevent import pywsgi
from geventwebsocket.handler import WebSocketHandler

from object_database.web.cells import *

from gevent.greenlet import Greenlet
from gevent import sleep

from flask import Flask, send_from_directory, redirect, url_for, request
from flask_sockets import Sockets
from flask_cors import CORS

from object_database import ServiceBase, service_schema, Schema, Indexed, Index, DatabaseObject

active_webservice_schema = Schema("core.active_webservice")

@active_webservice_schema.define
class Configuration:
    service = Indexed(service_schema.Service)

    port = int
    hostname = str

class ActiveWebService(ServiceBase):
    def __init__(self, db, serviceObject, serviceRuntimeConfig):
        ServiceBase.__init__(self, db, serviceObject, serviceRuntimeConfig)
        self._logger = logging.getLogger(__name__)

    @staticmethod
    def configureFromCommandline(db, serviceObject, args):
        """Subclasses should take the remaining args from the commandline and configure using them"""
        db.subscribeToType(Configuration)

        with db.transaction():
            c = Configuration.lookupAny(service=serviceObject)
            if not c:
                c = Configuration(service=serviceObject)

            parser = argparse.ArgumentParser("Configure a webservice")
            parser.add_argument("--hostname", type=str)
            parser.add_argument("--port", type=int)

            parsedArgs = parser.parse_args(args)

            c.port = parsedArgs.port
            c.hostname = parsedArgs.hostname

    def initialize(self):
        self.db.subscribeToType(Configuration)
        self.db.subscribeToSchema(service_schema)

        with self.db.transaction():
            self.app = Flask(__name__)
            CORS(self.app)
            self.sockets = Sockets(self.app)
            self.configureApp()

    def doWork(self, shouldStop):
        self._logger.info("Configuring ActiveWebService")
        with self.db.view():
            config = Configuration.lookupAny(service=self.serviceObject)
            assert config, "No configuration available."
            host,port = config.hostname, config.port

        self._logger.info("ActiveWebService listening on %s:%s", host, port)

        server = pywsgi.WSGIServer((host, port), self.app, handler_class=WebSocketHandler)

        server.serve_forever()

    def configureApp(self):
        instanceName = self.serviceObject.name
        self.app.route("/")(lambda: redirect("/services"))
        self.app.route('/content/<path:path>')(self.sendContent)
        self.app.route('/services')(self.sendPage)
        self.app.route('/services/<path:path>')(self.sendPage)
        self.sockets.route("/socket/<path:path>")(self.mainSocket)

    def sendPage(self, path=None):
        return self.sendContent("page.html")

    def mainDisplay(self):
        def serviceCountSetter(service, ct):
            def f():
                service.target_count = ct
            return f

        serviceCounts = list(range(5)) + list(range(10,100,10)) + list(range(100,400,25)) + list(range(400,1001,100))

        return Tabs(
            Services=Table(
                colFun=lambda: ['Service', 'Codebase', 'Module', 'Class', 'Placement', 'Active', 'TargetCount', 'Cores', 'RAM', 'Boot Status'],
                rowFun=lambda: sorted(service_schema.Service.lookupAll(), key=lambda s:s.name),
                headerFun=lambda x: x,
                rendererFun=lambda s,field: Subscribed(lambda:
                    Clickable(s.name, "/services/" + s.name) if field == 'Service' else
                    (str(s.codebase) if s.codebase else "") if field == 'Codebase' else
                    s.service_module_name if field == 'Module' else
                    s.service_class_name if field == 'Class' else
                    s.placement if field == 'Placement' else
                    Subscribed(lambda: len(service_schema.ServiceInstance.lookupAll(service=s))) if field == 'Active' else
                    Dropdown(s.target_count, [(str(ct), serviceCountSetter(s, ct)) for ct in serviceCounts])
                            if field == 'TargetCount' else
                    str(s.coresUsed) if field == 'Cores' else
                    str(s.gbRamUsed) if field == 'RAM' else
                    (Popover(Octicon("alert"), "Failed", Traceback(s.lastFailureReason or "<Unknown>")) if s.isThrottled() else "") if field == 'Boot Status' else
                    ""
                    ),
                maxRowsPerPage=10
                ),
            Hosts=Table(
                colFun=lambda: ['Connection', 'IsMaster', 'Hostname', 'RAM ALLOCATION', 'CORE ALLOCATION', 'SERVICE COUNT', 'CPU USE', 'RAM USE'],
                rowFun=lambda: sorted(service_schema.ServiceHost.lookupAll(), key=lambda s:s.hostname),
                headerFun=lambda x: x,
                rendererFun=lambda s,field: Subscribed(lambda:
                    s.connection._identity if field == "Connection" else
                    str(s.isMaster) if field == "IsMaster" else
                    s.hostname if field == "Hostname" else
                    "%.1f / %.1f" % (s.gbRamUsed, s.maxGbRam) if field == "RAM ALLOCATION" else
                    "%s / %s" % (s.coresUsed, s.maxCores) if field == "CORE ALLOCATION" else
                    str(len(service_schema.ServiceInstance.lookupAll(host=s))) if field == "SERVICE COUNT" else
                    "%2.1f" % (s.cpuUse * 100) + "%" if field == "CPU USE" else
                    ("%2.1f" % s.actualMemoryUseGB) + " GB" if field == "RAM USE" else
                    ""
                    ),
                maxRowsPerPage=10
                )
            )

    def pathToDisplay(self, path, queryArgs):
        if len(path) and path[0] == 'services':
            if len(path) == 1:
                return self.mainDisplay()
            serviceObj = service_schema.Service.lookupAny(name=path[1])

            if serviceObj is None:
                return Traceback("Unknown service %s" % path[1])

            serviceType = serviceObj.instantiateServiceType()

            if len(path) == 2:
                return (
                    Subscribed(lambda: serviceType.serviceDisplay(serviceObj, queryArgs=queryArgs))
                        .withSerializationContext(serviceObj.getSerializationContext())
                    )

            typename = path[2]

            schemas = serviceObj.findModuleSchemas()
            typeObj = None
            for s in schemas:
                typeObj = s.lookupFullyQualifiedTypeByName(typename)
                if typeObj:
                    break

            if typeObj is None:
                return Traceback("Can't find fully-qualified type %s" % typename)

            if len(path) == 3:
                return (
                    serviceType.serviceDisplay(serviceObj, objType=typename, queryArgs=queryArgs)
                        .withSerializationContext(serviceObj.getSerializationContext())
                    )

            instance = typeObj.fromIdentity(path[3])

            return (
                serviceType.serviceDisplay(serviceObj, instance=instance, queryArgs=queryArgs)
                    .withSerializationContext(serviceObj.getSerializationContext())
                )

        return Traceback("Invalid url path: %s" % path)


    def addMainBar(self, display):
        return (
            HeaderBar(
                [Subscribed(lambda:
                    Dropdown(
                        "Service",
                            [("All", "/services")] +
                            [(s.name, "/services/" + s.name) for
                                s in sorted(service_schema.Service.lookupAll(), key=lambda s:s.name)]
                        ),
                    )
                ]) +
            Main(display)
            )

    def mainSocket(self, ws, path):
        path = str(path).split("/")
        queryArgs = dict(request.args)
        self._logger.info("path = %s", path)
        reader = None

        try:
            self._logger.info("Starting main websocket handler with %s", ws)

            cells = Cells(self.db)
            cells.root.setRootSerializationContext(self.db.serializationContext)
            cells.root.setChild(self.addMainBar(Subscribed(lambda: self.pathToDisplay(path, queryArgs))))

            timestamps = []

            lastDumpTimestamp = time.time()
            lastDumpMessages = 0
            lastDumpFrames = 0
            lastDumpTimeSpentCalculating = 0.0

            def readThread():
                while not ws.closed:
                    msg = ws.receive()
                    if msg is None:
                        return
                    else:
                        try:
                            jsonMsg = json.loads(msg)

                            cell_id = jsonMsg.get('target_cell')
                            cell = cells.cells.get(cell_id)
                            if cell is not None:
                                cell.onMessage(jsonMsg)
                        except:
                            self._logger.error("Exception in inbound message: %s", traceback.format_exc())
                        cells.triggerIfHasDirty()

            reader = Greenlet.spawn(readThread)

            while not ws.closed:
                t0 = time.time()
                cells.recalculate()
                messages = cells.renderMessages()

                lastDumpTimeSpentCalculating += time.time() - t0

                for message in messages:
                    gevent.socket.wait_write(ws.stream.handler.socket.fileno())

                    ws.send(json.dumps(message))
                    lastDumpMessages += 1

                lastDumpFrames += 1
                if time.time() - lastDumpTimestamp > 5.0:
                    self._logger.info("In the last %.2f seconds, spent %.2f seconds calculating %s messages over %s frames",
                        time.time() - lastDumpTimestamp,
                        lastDumpTimeSpentCalculating,
                        lastDumpMessages,
                        lastDumpFrames
                        )

                    lastDumpFrames = 0
                    lastDumpMessages = 0
                    lastDumpTimeSpentCalculating = 0
                    lastDumpTimestamp = time.time()

                ws.send(json.dumps("postscripts"))

                cells.gEventHasTransactions.wait()

                timestamps.append(time.time())

                if len(timestamps) > MAX_FPS:
                    timestamps = timestamps[-MAX_FPS+1:]
                    if (time.time() - timestamps[0]) < 1.0:
                        sleep(1.0 / MAX_FPS + .001)

        except:
            self._logger.error("Websocket handler error: %s", traceback.format_exc())
        finally:
            if reader:
                reader.join()

    def echoSocket(self, ws):
        while not ws.closed:
            message = ws.receive()
            if message is not None:
                ws.send(message)

    def sendContent(self, path):
        own_dir = os.path.dirname(__file__)
        return send_from_directory(os.path.join(own_dir, "content"), path)

    @staticmethod
    def serviceDisplay(serviceObject, instance=None, objType=None, queryArgs=None):
        c = Configuration.lookupAny(service=serviceObject)

        return Card(Text("Host: " + c.hostname) + Text("Port: " + str(c.port)))
