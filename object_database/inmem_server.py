from object_database.server import Server
from object_database.database_connection import DatabaseConnection
from object_database.messages import ClientToServer, ServerToClient, getHeartbeatInterval
from object_database.persistence import InMemoryStringStore
import json
import time
import queue
import logging
import threading
import traceback

class InMemoryChannel:
    def __init__(self, server):
        self._server = server
        self._clientCallback = None
        self._serverCallback = None
        self._clientToServerMsgQueue = queue.Queue()
        self._serverToClientMsgQueue = queue.Queue()
        self._shouldStop = True

        self._pumpThreadServer = threading.Thread(target=self.pumpMessagesFromServer)
        self._pumpThreadServer.daemon = True
        
        self._pumpThreadClient = threading.Thread(target=self.pumpMessagesFromClient)
        self._pumpThreadClient.daemon = True

        self._stopHeartbeatingSet = False

    def _stopHeartbeating(self):
        self._stopHeartbeatingSet = True

    def close(self):
        self.stop()
        self._clientCallback(ServerToClient.Disconnected())
        self._server.dropConnection(self)

    def pumpMessagesFromServer(self):
        while not self._shouldStop:
            try:
                e = self._serverToClientMsgQueue.get(timeout=0.01)
            except queue.Empty:
                e = None

            if e:
                try:
                    self._clientCallback(e)
                except:
                    traceback.print_exc()
                    logging.error("Pump thread failed: %s", traceback.format_exc())
                    return
        
    def pumpMessagesFromClient(self):
        lastHeartbeat = time.time()
        while not self._shouldStop:
            if time.time() - lastHeartbeat > getHeartbeatInterval() and not self._stopHeartbeatingSet:
                lastHeartbeat = time.time()
                e = ClientToServer.Heartbeat()
            else:
                try:
                    e = self._clientToServerMsgQueue.get(timeout=0.01)
                except queue.Empty:
                    e = None

            if e:
                try:
                    self._serverCallback(e)
                except:
                    traceback.print_exc()
                    logging.error("Pump thread failed: %s", traceback.format_exc())
                    return

    def start(self):
        assert self._shouldStop
        self._shouldStop = False

    def stop(self):
        self._shouldStop = True
        self._pumpThreadServer.join()
        self._pumpThreadClient.join()

    def write(self, msg):
        if isinstance(msg, ClientToServer):
            self._clientToServerMsgQueue.put(msg)
        elif isinstance(msg, ServerToClient):
            self._serverToClientMsgQueue.put(msg)
        else:
            assert False

    def setServerToClientHandler(self, callback):
        assert not self._shouldStop
        
        self._clientCallback = callback
        self._pumpThreadServer.start()

    def setClientToServerHandler(self, callback):
        assert not self._shouldStop

        self._serverCallback = callback
        self._pumpThreadClient.start()

class InMemServer(Server):
    def __init__(self, kvstore=None):
        Server.__init__(self, kvstore or InMemoryStringStore())
        self.channels = []
        self.stopped = False
        self.checkForDeadConnectionsLoopThread = threading.Thread(target=self.checkForDeadConnectionsLoop)
        self.checkForDeadConnectionsLoopThread.daemon = True
        self.checkForDeadConnectionsLoopThread.start()

    def getChannel(self):
        channel = InMemoryChannel(self)
        channel.start()

        self.addConnection(channel)
        self.channels.append(channel)

        return channel

    def connect(self):
        return DatabaseConnection(self.getChannel())

    def checkForDeadConnectionsLoop(self):
        lastCheck = time.time()
        while not self.stopped:
            if time.time() - lastCheck > getHeartbeatInterval():
                self.checkForDeadConnections()
                lastCheck = time.time()
            else:
                time.sleep(0.1)

    def teardown(self):
        self.stopped = True
        for c in self.channels:
            c.stop()
        self.checkForDeadConnectionsLoopThread.join()

    def __enter__(self):
        return self

    def __exit__(self, type, val, tb):
        self.teardown()