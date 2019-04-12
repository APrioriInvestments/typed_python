#   Copyright 2017-2019 Nativepython Authors
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

import asyncio
import struct
import threading
import logging
import traceback

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
        self.buffer = bytearray()
        self.writelock = threading.Lock()
        self._logger = logging.getLogger(__name__)

    def sendMessage(self, msg):
        try:
            assert isinstance(msg, self.sendType), "message %s is of type %s != %s" % (msg, type(msg), self.sendType)

            dataToSend = serialize(self.sendType, msg)
            dataToSend = longToString(len(dataToSend)) + dataToSend

            with self.writelock:
                self.transport.write(dataToSend)
        except Exception:
            self._logger.error("Error in AlgebraicProtocol: %s", traceback.format_exc())
            self.transport.close()

    def messageReceived(self, msg):
        # subclasses override
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

                if toConsume:
                    try:
                        self.messageReceived(deserialize(self.receiveType, bytes(toConsume)))
                    except Exception:
                        self._logger.error("Error in AlgebraicProtocol: %s", traceback.format_exc())
                        self.transport.close()
            else:
                return
