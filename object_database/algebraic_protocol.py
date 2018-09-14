import asyncio
import struct
import threading
import json
import logging
import traceback
import socket
import time

from object_database.algebraic_to_json import Encoder

sizeType = '<L'
longLength = struct.calcsize(sizeType)

def longToString(l):
    return struct.pack(sizeType, l)

def stringToLong(l):
    return struct.unpack(sizeType, l)[0]

class AlgebraicProtocol(asyncio.Protocol):
    def __init__(self, receiveType, sendType):
        self.encoder = Encoder()
        self.receiveType = receiveType
        self.sendType = sendType
        self.transport = None
        self.buffer = bytes()
        self.writelock = threading.Lock()

    def sendMessage(self, msg):
        try:
            assert isinstance(msg, self.sendType), "message %s is of type %s != %s" % (msg, type(msg), self.sendType)

            #cache the encoded message on the object in case we're sending this to multiple
            #clients.
            if '__encoded_message__' in msg.__dict__:
                dataToSend = msg.__dict__['__encoded_message__']
            else:
                dataToSend = bytes(json.dumps(self.encoder.to_json(self.sendType, msg)), 'utf8')
                dataToSend = longToString(len(dataToSend)) + dataToSend
                msg.__dict__['__encoded_message__'] = dataToSend

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
                    self.messageReceived(self.encoder.from_json(json.loads(str(toConsume,'utf8')), self.receiveType))
                except:
                    logging.info("Error in AlgebraicProtocol: %s", traceback.format_exc())
                    self.transport.close()
            else:
                return

