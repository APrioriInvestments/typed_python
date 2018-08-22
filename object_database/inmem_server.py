from object_database.server import Server
from object_database.database_connection import DatabaseConnection
from object_database.messages import ClientToServer, ServerToClient
from object_database.persistence import InMemoryStringStore
import json
import queue
import logging
import threading
import traceback

class InMemoryChannel:
    def __init__(self):
        self._clientCallback = None
        self._serverCallback = None
        self._clientToServerMsgQueue = queue.Queue()
        self._serverToClientMsgQueue = queue.Queue()
        self._shouldStop = True

        self._pumpThreadServer = threading.Thread(target=self.pumpMessagesFromServer)
        self._pumpThreadServer.daemon = True
        
        self._pumpThreadClient = threading.Thread(target=self.pumpMessagesFromClient)
        self._pumpThreadClient.daemon = True
        
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
        while not self._shouldStop:
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

    def getChannel(self):
        channel = InMemoryChannel()
        channel.start()

        self.addConnection(channel)
        self.channels.append(channel)

        return channel

    def connect(self):
        return DatabaseConnection(self.getChannel())

    def teardown(self):
        for c in self.channels:
            c.stop()

    def __enter__(self):
        return self

    def __exit__(self, type, val, tb):
        self.teardown()