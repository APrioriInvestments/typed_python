from object_database.database_connection import DatabaseConnection
from object_database.server import Server
from object_database.messages import ClientToServer, ServerToClient, getHeartbeatInterval
from object_database.algebraic_protocol import AlgebraicProtocol
from object_database.persistence import InMemoryPersistence

import asyncio
import logging
import ssl
import time
import threading
import socket
import traceback


class ServerToClientProtocol(AlgebraicProtocol):
    def __init__(self, dbserver, loop):
        AlgebraicProtocol.__init__(self, ClientToServer, ServerToClient)
        self.dbserver = dbserver
        self.loop = loop
        self.connectionIsDead = False
        self._logger = logging.getLogger(__name__)

    def setClientToServerHandler(self, handler):
        def callHandler(*args):
            try:
                return handler(*args)
            except Exception:
                self._logger.error("Unexpected exception in %s:\n%s", handler.__name__, traceback.format_exc())

        self.handler = callHandler

    def messageReceived(self, msg):
        self.handler(msg)

    def onConnected(self):
        self.dbserver.addConnection(self)

    def write(self, msg):
        if not self.connectionIsDead:
            self.loop.call_soon_threadsafe(self.sendMessage, msg)

    def connection_lost(self, e):
        self.connectionIsDead = True
        _eventLoop.loop.call_later(0.01, self.completeDropConnection)

    def completeDropConnection(self):
        self.dbserver.dropConnection(self)

    def close(self):
        self.connectionIsDead = True
        self.transport.close()


class ClientToServerProtocol(AlgebraicProtocol):
    def __init__(self, host, port, eventLoop):
        AlgebraicProtocol.__init__(self, ServerToClient, ClientToServer)
        self.loop = eventLoop
        self.lock = threading.Lock()
        self.host = host
        self.port = port
        self.handler = None
        self.msgs = []
        self.disconnected = False
        self._stopHeartbeatingSet = False
        self._logger = logging.getLogger(__name__)

    def _stopHeartbeating(self):
        self._stopHeartbeatingSet = True

    def setServerToClientHandler(self, handler):
        with self.lock:
            def callHandler(*args):
                try:
                    return handler(*args)
                except Exception:
                    self._logger.error("Unexpected exception in %s:\n%s", handler.__name__, traceback.format_exc())

            self.handler = callHandler
            for m in self.msgs:
                self.loop.call_soon_threadsafe(self.handler, m)
            self.msgs = None

    def messageReceived(self, msg):
        with self.lock:
            if not self.handler:
                self.msgs.append(msg)
            else:
                self.loop.call_soon_threadsafe(self.handler, msg)

    def onConnected(self):
        self.loop.call_later(getHeartbeatInterval(), self.heartbeat)

    def heartbeat(self):
        if not self.disconnected and not self._stopHeartbeatingSet:
            self.sendMessage(ClientToServer.Heartbeat())
            self.loop.call_later(getHeartbeatInterval(), self.heartbeat)

    def close(self):
        self.loop.call_soon_threadsafe(self._close)

    def _close(self):
        self.disconnected = True
        self.transport.close()

    def connection_lost(self, e):
        self.disconnected = True
        self.messageReceived(ServerToClient.Disconnected())

    def write(self, msg):
        self.loop.call_soon_threadsafe(self.sendMessage, msg)


class EventLoopInThread:
    def __init__(self):
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self.runEventLoop)
        self.thread.daemon = True
        self.started = False

    def runEventLoop(self):
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    def start(self):
        if not self.started:
            self.started = True
            self.thread.start()

    def create_connection(self, protocol_factory, host, port, ssl):
        self.start()

        async def doit():
            return await self.loop.create_connection(protocol_factory, host=host, port=port, family=socket.AF_INET, ssl=ssl)

        res = asyncio.run_coroutine_threadsafe(doit(), self.loop)

        return res.result(10)

    def create_server(self, protocol_factory, host, port, ssl):
        self.start()

        async def doit():
            return await self.loop.create_server(protocol_factory, host=host, port=port, family=socket.AF_INET, ssl=ssl)

        res = asyncio.run_coroutine_threadsafe(doit(), self.loop)

        return res.result(10)


_eventLoop = EventLoopInThread()


def connect(host, port, auth_token, timeout=10.0, retry=False, eventLoop=_eventLoop):
    t0 = time.time()
    # With CLIENT_AUTH we are setting up the SSL to use encryption only, which is what we want.
    # If we also wanted authentication, we would use SERVER_AUTH.
    ssl_ctx = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)

    proto = None
    while proto is None:
        try:
            _, proto = eventLoop.create_connection(
                lambda: ClientToServerProtocol(host, port, eventLoop.loop),
                host=host,
                port=port,
                ssl=ssl_ctx
            )
        except Exception:
            if not retry or time.time() - t0 > timeout * .8:
                raise
            time.sleep(min(timeout, max(timeout / 100.0, 0.01)))

    if proto is None:
        raise ConnectionRefusedError()

    conn = DatabaseConnection(proto)
    conn.authenticate(auth_token)

    conn.initialized.wait(timeout=max(timeout - (time.time() - t0), 0.0))

    assert conn.initialized.is_set()

    return conn


_eventLoop2 = []


class TcpServer(Server):
    def __init__(self, host, port, mem_store, ssl_context, auth_token):
        Server.__init__(self, mem_store or InMemoryPersistence(), auth_token)

        self.mem_store = mem_store
        self.host = host
        self.port = port
        self.ssl_ctx = ssl_context
        self.socket_server = None
        self.stopped = False

    def start(self):
        Server.start(self)

        self.socket_server = _eventLoop.create_server(
            lambda: ServerToClientProtocol(self, _eventLoop.loop),
            host=self.host,
            port=self.port,
            ssl=self.ssl_ctx
        )
        _eventLoop.loop.call_soon_threadsafe(self.checkHeartbeatsCallback)

    def checkHeartbeatsCallback(self):
        if not self.stopped:
            _eventLoop.loop.call_later(getHeartbeatInterval(), self.checkHeartbeatsCallback)
            try:
                self.checkForDeadConnections()
            except Exception:
                logging.error("Caught exception in checkForDeadConnections:\n%s", traceback.format_exc())

    def stop(self):
        Server.stop(self)

        self.stopped = True
        if self.socket_server:
            self.socket_server.close()

    def connect(self, auth_token, useSecondaryLoop=False):
        if useSecondaryLoop:
            if not _eventLoop2:
                _eventLoop2.append(EventLoopInThread())
            loop = _eventLoop2[0]
        else:
            loop = _eventLoop

        return connect(self.host, self.port, auth_token, eventLoop=loop)

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, t, v, traceback):
        self.stop()
