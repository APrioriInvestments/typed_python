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
import gevent.queue

from object_database.view import revisionConflictRetry
from object_database.util import genToken, checkLogLevelValidity
from object_database import ServiceBase, service_schema, Schema, Indexed, Index, DatabaseObject
from object_database.web.AuthPlugin import AuthPluginBase, PermissiveAuthPlugin, LdapAuthPlugin
from object_database.web.cells import *
from typed_python import TupleOf, Float64

from gevent import pywsgi, sleep
from gevent.greenlet import Greenlet
from geventwebsocket.handler import WebSocketHandler

from flask import Flask, send_from_directory, redirect, url_for, request, render_template, flash
from flask_wtf import FlaskForm
from flask_sockets import Sockets
from flask_cors import CORS
from flask_login import LoginManager, current_user, login_user, logout_user, login_required

from wtforms import StringField, PasswordField, BooleanField, SubmitField
from wtforms.validators import DataRequired


active_webservice_schema = Schema("core.active_webservice")


@active_webservice_schema.define
class Configuration:
    service = Indexed(service_schema.Service)

    port = int
    hostname = str

    log_level = int

    # auth plugin args
    auth_type = str
    authorized_groups = TupleOf(str)
    auth_hostname = str
    ldap_base_dn = str
    ldap_ntlm_domain = str

    # HTML template rendering args
    company_name = str


USER_LOGIN_DURATION = 24 * 60 * 60  # 24 hours

@active_webservice_schema.define
class User:
    username = Indexed(str)
    login_expiration = Float64()

    def login(self):
        self.login_expiration = time.time() + USER_LOGIN_DURATION

    def logout(self):
        self.login_expiration = 0.0


class UserWrapper:
    #  must be called with an open view of the user argument
    @staticmethod
    def makeFromUser(user):
        if user is None:
            return None
        else:
            return UserWrapper(user.username, user.login_expiration)

    def __init__(self, username, login_expiration):
        self.username = username
        self.login_expiration = login_expiration

    @property
    def is_authenticated(self):
        return True if time.time() < self.login_expiration else False

    @property
    def is_active(self):
        return True

    @property
    def is_anonymous(self):
        return False

    def get_id(self):
        return self.username  # must return unicode by Python 3 strings are unicode


class LoginForm(FlaskForm):
    username = StringField('Username', validators=[DataRequired()])
    password = PasswordField('Password', validators=[DataRequired()])
    remember_me = BooleanField('Remember Me')
    submit = SubmitField('Sign In')


class ActiveWebService(ServiceBase):
    def __init__(self, db, serviceObject, serviceRuntimeConfig):
        ServiceBase.__init__(self, db, serviceObject, serviceRuntimeConfig)
        self._logger = logging.getLogger(__name__)

    @staticmethod
    def initialize_auth_plugin(auth_type, authorized_groups=None, hostname=None, ldap_base_dn=None, ldap_ntlm_domain=None):
        if auth_type == "NONE":
            return None
        elif auth_type == "PERMISSIVE":
            return PermissiveAuthPlugin()
        elif auth_type == "FORBIDDEN":
            return AuthPluginBase()
        elif auth_type == "LDAP":
            if not hostname:
                raise Exception("Missing required argument for LDAP: --hostname")

            if not ldap_base_dn:
                raise Exception("Missing required argument for LDAP: --base-dn")

            return LdapAuthPlugin(
                hostname=hostname,
                base_dn=ldap_base_dn,
                ntlm_domain=ldap_ntlm_domain,
                authorized_groups=authorized_groups
            )
        else:
            raise Exception("Unkown auth-type: {}".format(auth_type))

    @staticmethod
    def configure(db, serviceObject, hostname, port):
        db.subscribeToType(Configuration)

        with db.transaction():
            c = Configuration.lookupAny(service=serviceObject)
            if not c:
                c = Configuration(service=serviceObject)

            c.hostname = hostname
            c.port = port
            c.log_level = logging.getLevelName("INFO")
            c.auth_type = "NONE"


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

            parser.add_argument("--log-level", type=str, required=False, default="INFO")

            parser.add_argument("--auth", type=str, required=False, default="LDAP")
            parser.add_argument("--authorized-groups", nargs='+', type=str, required=False, default=())
            parser.add_argument("--auth-hostname", type=str, required=False, default="")
            parser.add_argument("--ldap-base-dn", type=str, required=False, default="")
            parser.add_argument("--ldap-ntlm-domain", type=str, required=False, default="")

            parser.add_argument("--company-name", type=str, required=False, default="")

            parsedArgs = parser.parse_args(args)

            VALID_AUTH_TYPES = ["NONE", "LDAP", "PERMISSIVE", "FORBIDDEN"]
            auth_type = parsedArgs.auth.upper()
            if  auth_type not in VALID_AUTH_TYPES:
                raise Exception("invalid --auth value: {auth}. Must be one of {options}"
                    .format(auth=parsedArgs.auth, options=VALID_AUTH_TYPES))

            level_name = parsedArgs.log_level.upper()
            checkLogLevelValidity(level_name)

            c.port = parsedArgs.port
            c.hostname = parsedArgs.hostname

            c.log_level = logging.getLevelName(level_name)

            c.auth_type = auth_type
            c.authorized_groups = TupleOf(str)(parsedArgs.authorized_groups)
            c.auth_hostname = parsedArgs.auth_hostname
            c.ldap_base_dn = parsedArgs.ldap_base_dn
            c.ldap_ntlm_domain = parsedArgs.ldap_ntlm_domain

            c.company_name = parsedArgs.company_name

    def initialize(self):
        self.db.subscribeToType(Configuration)
        self.db.subscribeToType(User)
        self.db.subscribeToSchema(service_schema)

        with self.db.transaction():
            self.app = Flask(__name__)
            CORS(self.app)
            self.sockets = Sockets(self.app)
            self.configureApp()
        self.login_manager = LoginManager(self.app)
        self.load_user = self.login_manager.user_loader(self.load_user)
        self.login_manager.login_view = 'login'

    def load_user(self, username):
        with self.db.view():
            return UserWrapper.makeFromUser(User.lookupAny(username=username))


    def doWork(self, shouldStop):
        self._logger.info("Configuring ActiveWebService")
        with self.db.view():
            config = Configuration.lookupAny(service=self.serviceObject)
            assert config, "No configuration available."
            self._logger.setLevel(config.log_level)
            host, port = config.hostname, config.port

            self.authorized_groups = config.authorized_groups

            self.authorized_groups_text = "All"
            if self.authorized_groups:
                self.authorized_groups_text = ", ".join(self.authorized_groups)

            self.auth_plugin = self.initialize_auth_plugin(
                config.auth_type,
                authorized_groups=config.authorized_groups,
                hostname=config.auth_hostname,
                ldap_base_dn=config.ldap_base_dn,
                ldap_ntlm_domain=config.ldap_ntlm_domain
            )

            self.company_name = config.company_name

        self._logger.info("ActiveWebService listening on %s:%s", host, port)

        server = pywsgi.WSGIServer((host, port), self.app, handler_class=WebSocketHandler)

        server.serve_forever()

    def configureApp(self):
        instanceName = self.serviceObject.name
        self.app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY') or genToken()

        self.app.add_url_rule('/', endpoint=None, view_func=lambda: redirect("/services"))
        self.app.add_url_rule('/content/<path:path>', endpoint=None, view_func=self.sendContent)
        self.app.add_url_rule('/services', endpoint=None, view_func=self.sendPage)
        self.app.add_url_rule('/services/<path:path>', endpoint=None, view_func=self.sendPage)
        self.app.add_url_rule('/login', endpoint=None, view_func=self.login, methods=['GET', 'POST'])
        self.app.add_url_rule('/logout', endpoint=None, view_func=self.logout)
        self.sockets.add_url_rule('/socket/<path:path>', None, self.mainSocket)

    @revisionConflictRetry
    def login_user(self, username):
        with self.db.transaction():
            users = User.lookupAll(username=username)

            if len(users) == 0:
                user = User(username=username)
            elif len(users) == 1:
                user = users[0]
            elif len(users) > 1:
                raise Exception("multiple users found with username={}".format(username))
            else:
                raise Exception("This should never happen: len(users)={}".format(len(users)))
            user.login()
            login_user(UserWrapper.makeFromUser(user))

    def login(self):
        if current_user.is_authenticated:
            return redirect('/')

        if self.auth_plugin is None:
            self.login_user('anonymous')
            return redirect('/')

        form = LoginForm()

        if form.validate_on_submit():
            username = form.username.data
            password = form.password.data
            if not self.auth_plugin.authenticate(username, password):
                flash(
                    'Invalid username or password or not a member of an '
                    'authorized group. Please try again.',
                    'danger')
                return render_template(
                    'login.html',
                    form=form,
                    title=self.company_name,
                    authorized_groups_text=self.authorized_groups_text
                )

            self.login_user(username)

            return redirect('/')

        if form.errors:
            flash(form.errors, 'danger')

        return render_template(
            'login.html',
            form=form,
            title=self.company_name,
            authorized_groups_text=self.authorized_groups_text
        )

    def logout(self):
        logout_user()
        return redirect('/')

    @login_required
    def sendPage(self, path=None):
        return self.sendContent("page.html")

    def mainDisplay(self):
        def serviceCountSetter(service, ct):
            def f():
                service.target_count = ct
            return f

        serviceCounts = list(range(5)) + list(range(10,100,10)) + list(range(100,400,25)) + list(range(400,1001,100))

        buttons = Sequence([
            Padding(),
            Button(
                Sequence([Octicon('shield').color('green'), Span('Lock ALL')]),
                lambda: [s.lock() for s in service_schema.Service.lookupAll()]),
            Button(
                Sequence([Octicon('shield').color('orange'), Span('Prepare ALL')]),
                lambda: [s.prepare() for s in service_schema.Service.lookupAll()]),
            Button(
                Sequence([Octicon('stop').color('red'), Span('Unlock ALL')]),
                lambda: [s.unlock() for s in service_schema.Service.lookupAll()]),
        ])
        tabs = Tabs(
            Services=Table(
                colFun=lambda: [
                    'Service', 'Codebase Status', 'Codebase', 'Module', 'Class',
                    'Placement', 'Active', 'TargetCount', 'Cores', 'RAM', 'Boot Status'],
                rowFun=lambda:
                    sorted(service_schema.Service.lookupAll(), key=lambda s:s.name),
                headerFun=lambda x: x,
                rendererFun=lambda s, field: Subscribed(lambda:
                    Clickable(s.name, "/services/" + s.name) if field == 'Service' else
                    (   Clickable(Sequence([Octicon('stop').color('red'), Span('Unlocked')]),
                                  lambda: s.lock()) if s.isUnlocked else
                        Clickable(Sequence([Octicon('shield').color('green'), Span('Locked')]),
                                  lambda: s.prepare()) if s.isLocked else
                        Clickable(Sequence([Octicon('shield').color('orange'), Span('Prepared')]),
                                  lambda: s.unlock())) if field == 'Codebase Status' else
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
                maxRowsPerPage=50
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
                maxRowsPerPage=50
                )
            )
        return Sequence([buttons, tabs])

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
        with self.db.view():
            current_username = current_user.username

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
                ],
                (),
                [
                LargePendingDownloadDisplay(),
                Octicon('person') + Span(current_username),
                Span('Authorized Groups: {}'.format(self.authorized_groups_text)),
                Button(Octicon('sign-out'),'/logout')
                ]) +
            Main(display)
            )

    @login_required
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

            FRAME_SIZE = 32 * 1024
            FRAMES_PER_ACK = 10 #this HAS to line up with the constant in page.html for our ad-hoc protocol to function.

            #large messages (more than FRAMES_PER_ACK frames) send an ack after every FRAMES_PER_ACKth message
            largeMessageAck = gevent.queue.Queue()

            def readThread():
                while not ws.closed:
                    msg = ws.receive()
                    if msg is None:
                        return
                    else:
                        try:
                            jsonMsg = json.loads(msg)
                            if 'ACK' in jsonMsg:
                                largeMessageAck.put(jsonMsg['ACK'])
                            else:
                                cell_id = jsonMsg.get('target_cell')
                                cell = cells[cell_id]
                                if cell is not None:
                                    cell.onMessageWithTransaction(jsonMsg)
                        except Exception:
                            self._logger.error("Exception in inbound message: %s", traceback.format_exc())

                        cells.triggerIfHasDirty()

                largeMessageAck.put(StopIteration)

            reader = Greenlet.spawn(readThread)

            def writeJsonMessage(message):
                """Send a message over the websocket. We have to chunk in 64kb frames
                to keep the websocket from disconnecting on chrome for very large messages.
                This appears to be a bug in the implementation?
                """
                msg = json.dumps(message)

                #split msg int 64kb frames
                frames = []
                i = 0
                while i < len(msg):
                    frames.append(msg[i:i+FRAME_SIZE])
                    i += FRAME_SIZE

                if len(frames) >= FRAMES_PER_ACK:
                    self._logger.info("Sending large message of %s bytes over %s frames", len(msg), len(frames))

                ws.send(json.dumps(len(frames)))

                for index, frame in enumerate(frames):
                    ws.send(frame)

                    #block until we get the ack for FRAMES_PER_ACK frames ago. That way we always
                    #have FRAMES_PER_ACK frames in the buffer.
                    framesSent = index+1
                    if framesSent % FRAMES_PER_ACK == 0 and framesSent > FRAMES_PER_ACK:
                        ack = largeMessageAck.get()
                        if ack is StopIteration:
                            return
                        else:
                            assert ack == framesSent - FRAMES_PER_ACK, (ack, framesSent - FRAMES_PER_ACK)


                framesSent = len(frames)

                if framesSent >= FRAMES_PER_ACK:
                    finalAckIx = framesSent - (framesSent % FRAMES_PER_ACK)

                    ack = largeMessageAck.get()
                    if ack is StopIteration:
                        return
                    else:
                        assert ack == finalAckIx, (ack, finalAckIx)


            while not ws.closed:
                t0 = time.time()
                messages = cells.renderMessages()

                user = self.load_user(current_user.username)
                if not user.is_authenticated:
                    ws.close()
                    return

                lastDumpTimeSpentCalculating += time.time() - t0

                for message in messages:
                    gevent.socket.wait_write(ws.stream.handler.socket.fileno())

                    writeJsonMessage(message)

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

                writeJsonMessage("postscripts")

                cells.wait()

                timestamps.append(time.time())

                if len(timestamps) > MAX_FPS:
                    timestamps = timestamps[-MAX_FPS+1:]
                    if (time.time() - timestamps[0]) < 1.0:
                        sleep(1.0 / MAX_FPS + .001)

        except Exception:
            self._logger.error("Websocket handler error: %s", traceback.format_exc())
        finally:
            if reader:
                reader.join()

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
    def serviceDisplay(serviceObject, instance=None, objType=None, queryArgs=None):
        c = Configuration.lookupAny(service=serviceObject)

        return Card(Text("Host: " + c.hostname) + Text("Port: " + str(c.port)))
