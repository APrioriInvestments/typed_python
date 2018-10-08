import asyncio
import struct
import threading
import json
import logging
import traceback
import socket
import time

from typed_python import serialize, deserialize

sizeType = '<L'
longLength = struct.calcsize(sizeType)

def longToString(l):
    return struct.pack(sizeType, l)

def stringToLong(l):
    return struct.unpack(sizeType, l)[0]

class AlgebraicProtocol(asyncio.Protocol):
    def __init__(self, receiveType, sendType):
        self.receiveType = receiveType
        self.sendType = sendType
        self.transport = None
        self.buffer = bytes()
        self.writelock = threading.Lock()

    def sendMessage(self, msg):
        try:
            assert isinstance(msg, self.sendType), "message %s is of type %s != %s" % (msg, type(msg), self.sendType)

            dataToSend = serialize(self.sendType, msg)
            dataToSend = longToString(len(dataToSend)) + dataToSend
            
            with self.writelock:
                self.transport.write(dataToSend)
        except:
            logging.error("Error in AlgebraicProtocol: %s", traceback.format_exc())
            self.transport.close()

    def messageReceived(self, msg):
        #subclasses override
        pass

    def onConnected(self):
        pass

    def connection_made(self, transport):
        self.transport = transport
        self.transport._sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, True)
        self.onConnected()

    def data_received(self, data):
        assert isinstance(data, bytes)
        self.buffer += data

        while len(self.buffer) >= longLength:
            bytesToRead = stringToLong(self.buffer[:longLength])
            if bytesToRead + longLength <= len(self.buffer):
                toConsume = self.buffer[longLength:bytesToRead + longLength]
                self.buffer = self.buffer[bytesToRead + longLength:]

                try:
                    self.messageReceived(deserialize(self.receiveType, toConsume))
                except:
                    logging.info("Error in AlgebraicProtocol: %s", traceback.format_exc())
                    self.transport.close()
            else:
                return

