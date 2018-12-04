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

from object_database.algebraic_protocol import AlgebraicProtocol
from typed_python import Alternative

import asyncio
import queue
import ssl
import threading
import unittest


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
        print("SEND")
        self.sendMessage(self.messageToSend)

    def messageReceived(self, msg):
        self.responseQueue.put(msg)
        self.transport.close()


class AlgebraicProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.loop = asyncio.get_event_loop()

    @classmethod
    def tearDownClass(cls):
        cls.loop.close()

    def test_basic_send_and_receive(self):
        loop = self.loop

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

    def test_basic_send_and_receive_with_ssl(self):
        # https://gist.github.com/messa/22398173f039d1230e32
        #
        # Command to generate the self-signed key:
        #
        #     openssl req -x509 -newkey rsa:2048 -keyout testcert.key -nodes \
        #                 -out testcert.cert -sha256 -days 1000
        #
        # use 'localhost' as Common Name (CN)
        loop = self.loop
        host = 'localhost'
        port = 8888

        srv_ssl_ctx = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
        srv_ssl_ctx.load_cert_chain('testcert.cert', 'testcert.key')
        # Each client connection will create a new protocol instance
        serverCoro = loop.create_server(
            PingPongProtocol, host, port,
            ssl=srv_ssl_ctx,
            #ssl_handshake_timeout=2
        )

        server = loop.run_until_complete(serverCoro)

        cli_ssl_ctx = ssl.create_default_context(
            ssl.Purpose.SERVER_AUTH,
            cafile='testcert.cert'
        )

        q = queue.Queue()

        clientCoro = loop.create_connection(
            lambda: SendAndReturn(Message.Ping(), q),
            host,
            port,
            ssl=cli_ssl_ctx
        )
        try:
            loop.run_until_complete(clientCoro)
        except Exception as e:
            import pdb; pdb.set_trace()
            print(e)
        finally:
            self.assertEqual(q.get(timeout=1.0), Message.Pong())

        server.close()
        loop.run_until_complete(server.wait_closed())
