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

import logging
import time
import argparse
import functools
import traceback
import os
import gevent.socket
import gevent.queue

from object_database.util import genToken, validateLogLevel
from object_database import ServiceBase, service_schema
from object_database.web.AuthPlugin import AuthPluginBase, LdapAuthPlugin
from object_database.web.LoginPlugin import LoginIpPlugin
from object_database.web.ActiveWebService_util import (
    Configuration, LoginPlugin, mainBar,
    displayAndHeadersForPathAndQueryArgs,
    writeJsonMessage,
    readThread
)

from object_database.web.cells import (
    Subscribed, Cells, Card, Text, MAX_FPS, SessionState
)

from typed_python import OneOf, TupleOf
from typed_python.Codebase import Codebase as TypedPythonCodebase

from gevent import pywsgi, sleep
from gevent.greenlet import Greenlet
from geventwebsocket.handler import WebSocketHandler

from flask import (
    Flask,
    jsonify,
    make_response,
    redirect,
    request,
    send_from_directory,
)
from flask_sockets import Sockets
from flask_cors import CORS
from flask_login import LoginManager, current_user, login_required


class ActiveWebService(ServiceBase):
    """
    See object_database.frontends.object_database_webtest.py for example
    useage.
    """
    def __init__(self, db, serviceObject, serviceRuntimeConfig):
        ServiceBase.__init__(self, db, serviceObject, serviceRuntimeConfig)
        self._logger = logging.getLogger(__name__)

    @staticmethod
    def configure(db, serviceObject, hostname, port, level_name="INFO"):
        db.subscribeToType(Configuration)

        with db.transaction():
            c = Configuration.lookupAny(service=serviceObject)
            if not c:
                c = Configuration(service=serviceObject)

            c.hostname = hostname
            c.port = port
            c.log_level = logging.getLevelName(level_name)

    @staticmethod
    def setLoginPlugin(db, serviceObject, loginPluginFactory, authPlugins,
                       codebase=None, config=None):
        db.subscribeToType(Configuration)
        db.subscribeToType(LoginPlugin)

        config = config or {}

        with db.transaction():
            c = Configuration.lookupAny(service=serviceObject)
            if not c:
                c = Configuration(service=serviceObject)
            login_plugin = LoginPlugin(
                name="an auth plugin",
                login_plugin_factory=loginPluginFactory,
                auth_plugins=TupleOf(OneOf(None, AuthPluginBase))(authPlugins),
                codebase=codebase,
                config=config
            )
            c.login_plugin = login_plugin

    @staticmethod
    def configureFromCommandline(db, serviceObject, args):
        """
            Subclasses should take the remaining args from the commandline and
            configure using them.
        """
        db.subscribeToType(Configuration)

        parser = argparse.ArgumentParser("Configure a webservice")
        parser.add_argument("--hostname", type=str)
        parser.add_argument("--port", type=int)
        # optional arguments
        parser.add_argument("--log-level", type=str, required=False,
                            default="INFO")

        parser.add_argument("--ldap-hostname", type=str, required=False)
        parser.add_argument("--ldap-base-dn", type=str, required=False)
        parser.add_argument("--ldap-ntlm-domain", type=str, required=False)
        parser.add_argument("--authorized-groups", type=str, required=False,
                            nargs="+")
        parser.add_argument("--company-name", type=str, required=False)

        parsedArgs = parser.parse_args(args)

        with db.transaction():
            c = Configuration.lookupAny(service=serviceObject)
            if not c:
                c = Configuration(service=serviceObject)

            level_name = parsedArgs.log_level.upper()
            level_name = validateLogLevel(level_name, fallback='INFO')

            c.port = parsedArgs.port
            c.hostname = parsedArgs.hostname

            c.log_level = logging.getLevelName(level_name)

        if parsedArgs.ldap_base_dn is not None:
            ActiveWebService.setLoginPlugin(
                db,
                serviceObject,
                LoginIpPlugin,
                [LdapAuthPlugin(
                    parsedArgs.ldap_hostname,
                    parsedArgs.ldap_base_dn,
                    parsedArgs.ldap_ntlm_domain,
                    parsedArgs.authorized_groups
                )],
                config={'company_name': parsedArgs.company_name}
            )

    def initialize(self):
        # dict from session id (cookie really) to a a list of
        # [cells.SessionState]
        self.sessionStates = {}

        self.db.subscribeToType(Configuration)
        self.db.subscribeToType(LoginPlugin)
        self.db.subscribeToSchema(service_schema)

        with self.db.transaction():
            self.app = Flask(__name__)
            CORS(self.app)
            self.sockets = Sockets(self.app)
            self.configureApp()
        self.login_manager = LoginManager(self.app)
        self.login_manager.login_view = 'login'

    def doWork(self, shouldStop):
        self._logger.info("Configuring ActiveWebService")
        with self.db.view() as view:
            config = Configuration.lookupAny(service=self.serviceObject)
            assert config, "No configuration available."
            self._logger.setLevel(config.log_level)
            host, port = config.hostname, config.port

            login_config = config.login_plugin

            codebase = login_config.codebase
            if codebase is None:
                ser_ctx = TypedPythonCodebase.coreSerializationContext()
            else:
                ser_ctx = codebase.instantiate().serializationContext
            view.setSerializationContext(ser_ctx)

            self.login_plugin = login_config.login_plugin_factory(
                self.db, login_config.auth_plugins, login_config.config
            )

            # register `load_user` method with login_manager
            self.login_plugin.load_user = self.login_manager.user_loader(
                self.login_plugin.load_user)

            self.authorized_groups_text = self.login_plugin.authorized_groups_text

            self.login_plugin.init_app(self.app)

        self._logger.info("ActiveWebService listening on %s:%s", host, port)

        server = pywsgi.WSGIServer(
            (host, port),
            self.app,
            handler_class=WebSocketHandler
        )

        server.serve_forever()

    def configureApp(self):
        self.app.config['SECRET_KEY'] = os.environ.get(
            'SECRET_KEY') or genToken()

        self.app.add_url_rule('/', endpoint='index', view_func=lambda:
                              redirect("/services"))
        self.app.add_url_rule('/content/<path:path>', endpoint=None,
                              view_func=self.sendContent)
        self.app.add_url_rule('/services', endpoint=None,
                              view_func=self.sendPage)
        self.app.add_url_rule('/services/<path:path>', endpoint=None,
                              view_func=self.sendPage)
        self.app.add_url_rule('/status', view_func=self.statusPage)
        self.sockets.add_url_rule('/socket/<path:path>', None, self.mainSocket)

    def statusPage(self):
        return make_response(jsonify("STATUS: service is up"))

    @login_required
    def sendPage(self, path=None):
        self._logger.info("Sending 'page.html'")
        return self.sendContent("page.html")

    def displayForPathAndQueryArgs(self, path, queryArgs):
        display, toggles = displayAndHeadersForPathAndQueryArgs(path, queryArgs)
        return mainBar(display, toggles, current_user.username,
                       self.authorized_groups_text)

    @login_required
    def mainSocket(self, ws, path):
        path = str(path).split("/")
        queryArgs = dict(request.args.items())

        sessionId = request.cookies.get("session")

        # wait for the other socket to close if we were bounced
        sleep(.25)

        sessionState = self._getSessionState(sessionId)

        self._logger.info("entering websocket with path %s", path)
        reader = None
        isFirstMessage = True

        # set up message tracking
        timestamps = []

        lastDumpTimestamp = time.time()
        lastDumpMessages = 0
        lastDumpFrames = 0
        lastDumpTimeSpentCalculating = 0.0

        # set up cells
        cells = Cells(self.db)

        # reset the session state. There's only one per cells (which is why
        # we keep a list of sessions.)
        sessionState._reset(cells)

        cells = cells.withRoot(
            Subscribed(
                lambda: self.displayForPathAndQueryArgs(path, queryArgs)
            ),
            serialization_context=self.db.serializationContext,
            session_state=sessionState
        )

        # large messages (more than frames_per_ack frames) send an ack
        # after every frames_per_ackth message
        largeMessageAck = gevent.queue.Queue()
        reader = Greenlet.spawn(functools.partial(readThread, ws,
                                                  cells, largeMessageAck,
                                                  self._logger))

        self._logger.info("Starting main websocket handler with %s", ws)

        while not ws.closed:
            t0 = time.time()
            try:
                # make sure user is authenticated
                user = self.login_plugin.load_user(current_user.username)
                if not user.is_authenticated:
                    ws.close()
                    return

                messages = cells.renderMessages()

                lastDumpTimeSpentCalculating += time.time() - t0

                if isFirstMessage:
                    self._logger.info("Completed first rendering loop")
                    isFirstMessage = False

                for message in messages:
                    gevent.socket.wait_write(ws.stream.handler.socket.fileno())

                    writeJsonMessage(message, ws, largeMessageAck,
                                     self._logger)

                    lastDumpMessages += 1

                lastDumpFrames += 1
                # log slow messages
                if time.time() - lastDumpTimestamp > 60.0:
                    self._logger.info(
                        "In the last %.2f seconds, spent %.2f seconds"
                        " calculating %s messages over %s frames",
                        time.time() - lastDumpTimestamp,
                        lastDumpTimeSpentCalculating,
                        lastDumpMessages,
                        lastDumpFrames
                    )

                    lastDumpFrames = 0
                    lastDumpMessages = 0
                    lastDumpTimeSpentCalculating = 0
                    lastDumpTimestamp = time.time()

                # tell the browser to execute the postscripts that its built up
                writeJsonMessage("postscripts", ws, largeMessageAck,
                                 self._logger)

                cells.wait()

                timestamps.append(time.time())

                if len(timestamps) > MAX_FPS:
                    timestamps = timestamps[-MAX_FPS+1:]
                    if (time.time() - timestamps[0]) < 1.0:
                        sleep(1.0 / MAX_FPS + .001)

            except Exception:
                self._logger.error("Websocket handler error: %s",
                                   traceback.format_exc())
                self.sessionStates[sessionId].append(sessionState)

                self._logger.info(
                    "Returning session state to pool for %s. Have %s",
                    sessionId,
                    len(self.sessionStates[sessionId])
                )

                cells.markStopProcessingTasks()

                if reader:
                    reader.join()

    def _getSessionState(self, sessionId):
        if sessionId is None:
            sessionState = SessionState()
        else:
            # we keep sessions in a list. This is not great, but if you
            # bounce your browser, you'll get the session state you just dropped.
            # if you have several windows open, close a few, and then reopen
            # you'll get a random one
            sessionStateList = self.sessionStates.setdefault(sessionId, [])
            if not sessionStateList:
                self._logger.info("Creating a new SessionState for %s",
                                  sessionId)
                sessionState = SessionState()
            else:
                sessionState = sessionStateList.pop()
        return sessionState

    @login_required
    def echoSocket(self, ws):
        while not ws.closed:
            message = ws.receive()
            if message is not None:
                ws.send(message)

    @login_required
    def sendContent(self, path):
        own_dir = os.path.dirname(__file__)
        return send_from_directory(os.path.join(own_dir, "content"), path)

    @staticmethod
    def serviceDisplay(serviceObject, instance=None, objType=None,
                       queryArgs=None):
        c = Configuration.lookupAny(service=serviceObject)

        return Card(Text("Host: " + c.hostname) + Text("Port: " + str(c.port)))
