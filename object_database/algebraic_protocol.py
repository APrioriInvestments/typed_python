import asyncio
import struct
import ujson as json
import logging
import traceback

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

    def sendMessage(self, msg):
        assert isinstance(msg, self.sendType), "message %s is of type %s != %s" % (msg, type(msg), self.sendType)
        dataToSend = bytes(json.dumps(self.encoder.to_json(self.sendType, msg)), 'utf8')
        self.transport.write(longToString(len(dataToSend)) + dataToSend)

    def messageReceived(self, msg):
        #subclasses override
        pass

    def onConnected(self):
        pass

    def connection_made(self, transport):
        self.transport = transport
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

