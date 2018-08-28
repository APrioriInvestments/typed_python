from object_database.database_connection import DatabaseConnection
from object_database.server import Server
from object_database.messages import ClientToServer, ServerToClient, getHeartbeatInterval
from object_database.algebraic_protocol import AlgebraicProtocol
from object_database.persistence import InMemoryStringStore

import asyncio
import json
import queue
import logging
import time
import threading
import traceback


class ServerToClientProtocol(AlgebraicProtocol):
    def __init__(self, dbserver):
        AlgebraicProtocol.__init__(self, ClientToServer, ServerToClient)
        self.dbserver = dbserver

    def setClientToServerHandler(self, handler):
        self.handler = handler

    def messageReceived(self, msg):
        self.handler(msg)

    def onConnected(self):
        self.dbserver.addConnection(self)

    def write(self, msg):
        self.sendMessage(msg)

    def connection_lost(self, e):
        self.dbserver.dropConnection(self)

    def close(self):
        self.transport.close()

class ClientToServerProtocol(AlgebraicProtocol):
    def __init__(self, host, port):
        AlgebraicProtocol.__init__(self, ServerToClient, ClientToServer)
        self.lock = threading.Lock()
        self.host = host
        self.port = port
        self.handler = None
        self.msgs = []
        self.disconnected = False
        self._stopHeartbeatingSet = False

    def _stopHeartbeating(self):
        self._stopHeartbeatingSet = True
    
    def setServerToClientHandler(self, handler):
        with self.lock:
            self.handler = handler
            for m in self.msgs:
                _eventLoop.loop.call_soon_threadsafe(self.handler, m)
            self.msgs = None

    def messageReceived(self, msg):
        with self.lock:
            if not self.handler:
                self.msgs.append(msg)
            else:
                _eventLoop.loop.call_soon_threadsafe(self.handler, msg)
        
    def onConnected(self):
        _eventLoop.loop.call_later(getHeartbeatInterval(), self.heartbeat)

    def heartbeat(self):
        if not self.disconnected and not self._stopHeartbeatingSet:
            self.sendMessage(ClientToServer.Heartbeat())
            _eventLoop.loop.call_later(getHeartbeatInterval(), self.heartbeat)

    def connection_lost(self, e):
        self.disconnected = True
        self.messageReceived(ServerToClient.Disconnected())

    def write(self, msg):
        _eventLoop.loop.call_soon_threadsafe(self.sendMessage, msg)

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

    def create_connection(self, callback, host, port):
        self.start()

        async def doit():
            return await self.loop.create_connection(callback, host, port)

        return asyncio.run_coroutine_threadsafe(doit(), self.loop).result(10)

    def create_server(self, callback, host, port):
        self.start()

        async def doit():
            return await self.loop.create_server(callback, host, port)

        res = asyncio.run_coroutine_threadsafe(doit(), self.loop)

        return res.result(10)

_eventLoop = EventLoopInThread()

def connect(host, port, timeout=10.0, retry = False):
    t0 = time.time()

    proto = None
    while proto is None:
        try:
            _, proto = _eventLoop.create_connection(
                lambda: ClientToServerProtocol(host, port),
                host,
                port
                )
        except:
            if not retry or time.time() - t0 > timeout * .8:
                raise
            time.sleep(min(timeout, max(timeout / 100.0, 0.01)))

    conn = DatabaseConnection(proto)
    conn.initialized.wait(timeout=max(timeout - (time.time() - t0), 0.0))
    
    assert conn.initialized.is_set()

    return conn


class TcpServer(Server):
    def __init__(self, host, port, mem_store = None):
        Server.__init__(self, mem_store or InMemoryStringStore())

        self.mem_store = mem_store
        self.host = host
        self.port = port
        self.socket_server = None
        self.stopped = False

    def start(self):
        self.socket_server = _eventLoop.create_server(
            lambda: ServerToClientProtocol(self), 
            self.host, 
            self.port
            )
        _eventLoop.loop.call_soon_threadsafe(self.checkHeartbeatsCallback)

    def checkHeartbeatsCallback(self):
        if not self.stopped:
            _eventLoop.loop.call_later(getHeartbeatInterval(), self.checkHeartbeatsCallback)
            self.checkForDeadConnections()
        
    def stop(self):
        self.stopped = True
        if self.socket_server:
            self.socket_server.close()

    def connect(self):
        return connect(self.host, self.port)

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, t,v,traceback):
        self.stop()
