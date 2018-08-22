from object_database.database_connection import DatabaseConnection
from object_database.server import Server
from object_database.messages import ClientToServer, ServerToClient
from object_database.algebraic_protocol import AlgebraicProtocol

import asyncio
import json
import queue
import logging
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

class ClientToServerProtocol(AlgebraicProtocol):
    def __init__(self, host, port):
        AlgebraicProtocol.__init__(self, ServerToClient, ClientToServer)
        self.lock = threading.Lock()
        self.host = host
        self.port = port
        self.handler = None
        self.msgs = []
    
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
        pass

    def connection_lost(self, e):
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

def connect(host, port):
    _, proto = _eventLoop.create_connection(
        lambda: ClientToServerProtocol(host, port),
        host,
        port
        )

    return DatabaseConnection(proto)

class TcpServer(Server):
    def __init__(self, mem_store, host, port):
        Server.__init__(self, mem_store)

        self.mem_store = mem_store
        self.host = host
        self.port = port
        self.socket_server = None

    def start(self):
        self.socket_server = _eventLoop.create_server(
            lambda: ServerToClientProtocol(self), 
            self.host, 
            self.port
            )
        
    def stop(self):
        if self.socket_server:
            self.socket_server.close()

