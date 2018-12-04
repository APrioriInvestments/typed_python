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

from typed_python import *
from object_database.algebraic_protocol import AlgebraicProtocol

import unittest
import asyncio
import threading
import queue

Message = Alternative(
    "Message",
    Ping = {},
    Pong = {}
    )

class PingPongProtocol(AlgebraicProtocol):
    def __init__(self):
        AlgebraicProtocol.__init__(self, Message, Message)

    def messageReceived(self, m):
        if m.matches.Ping:
            self.sendMessage(Message.Pong())

class SendAndReturn(AlgebraicProtocol):
    def __init__(self, messageToSend, responseQueue):
        AlgebraicProtocol.__init__(self, Message, Message)
        self.messageToSend = messageToSend
        self.responseQueue = responseQueue

    def onConnected(self):
        self.sendMessage(self.messageToSend)

    def messageReceived(self, msg):
        self.responseQueue.put(msg)
        self.transport.close()

class AlgebraicProtocolTests(unittest.TestCase):
    def test_basic_send_and_receive(self):
        loop = asyncio.get_event_loop()

        # Each client connection will create a new protocol instance
        serverCoro = loop.create_server(PingPongProtocol, '127.0.0.1', 8888)

        server = loop.run_until_complete(serverCoro)

        q = queue.Queue()

        clientCoro = loop.create_connection(
            lambda: SendAndReturn(Message.Ping(), q),
            '127.0.0.1',
            8888
            )
        loop.run_until_complete(clientCoro)

        self.assertEqual(q.get(timeout=1.0), Message.Pong())

        server.close()
        loop.run_until_complete(server.wait_closed())
        loop.close()