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

"""message_bus

Classes for maintaining a stronlyg-typed message bus over sockets,
along with classes to simulate this in tests.
"""

import ssl
import threading
import queue
import struct
import logging
import select
import os
import socket

from typed_python import Alternative, NamedTuple, TypeFunction, serialize, deserialize

from object_database.util import sslContextFromCertPathOrNone
from object_database.bytecount_limited_queue import BytecountLimitedQueue

MESSAGE_LEN_BYTES = 4  # sizeof an int32 used to pack messages
SELECT_TIMEOUT = .5
MSG_BUF_SIZE = 128 * 1024


class MessageBuffer:
    def __init__(self):
        # the buffer we're reading
        self.buffer = bytearray()
        self.messagesEver = 0

        # the current message length, if any.
        self.curMessageLen = None

    def pendingBytecount(self):
        return len(self.buffer)

    @staticmethod
    def encode(bytes):
        """Prepend a message-length prefix"""
        res = bytearray(struct.pack("i", len(bytes)))
        res.extend(bytes)
        res.extend(struct.pack("i", len(bytes)))

        return res

    def write(self, bytesToWrite):
        """Push bytes into the buffer and read any completed messages.

        Args:
            bytesToWrite (bytes) - a portion of the message stream

        Returns:
            A list of messages completed by the bytes.
        """
        messages = []

        self.buffer.extend(bytesToWrite)

        while True:
            if self.curMessageLen is None:
                if len(self.buffer) >= MESSAGE_LEN_BYTES:
                    self.curMessageLen = struct.unpack("i", self.buffer[:MESSAGE_LEN_BYTES])[0]
                    self.buffer[:MESSAGE_LEN_BYTES] = b""

            if self.curMessageLen is None:
                return messages

            if len(self.buffer) >= self.curMessageLen + MESSAGE_LEN_BYTES:
                messages.append(bytes(self.buffer[:self.curMessageLen]))
                self.messagesEver += 1
                checkSize = struct.unpack("i", self.buffer[self.curMessageLen:self.curMessageLen + MESSAGE_LEN_BYTES])[0]
                assert checkSize == self.curMessageLen, f"Corrupt message stream: {checkSize} != {self.curMessageLen}"

                self.buffer[:self.curMessageLen + MESSAGE_LEN_BYTES] = b""
                self.curMessageLen = None
            else:
                return messages


class Disconnected:
    """A singleton representing our disconnect state."""


class FailedToStart(Exception):
    """We failed to acquire the listening socket."""


class TriggerDisconnect:
    """A singleton for triggering a channel to disconnect."""


class TriggerConnect:
    """A singleton for signaling we should connect to a channel."""


Endpoint = NamedTuple(host=str, port=int)


ConnectionId = NamedTuple(id=int)


@TypeFunction
def MessageBusEvent(MessageType):
    return Alternative(
        "MessageBusEvent",
        # the entire bus was stopped (by us). This is always the last message
        Stopped=dict(),
        # someone connected to us. All messages sent on this particular socket connectionId
        # will be associated with the given connectionId.
        NewIncomingConnection=dict(source=Endpoint, connectionId=ConnectionId),
        # an incoming connection closed
        IncomingConnectionClosed=dict(connectionId=ConnectionId),
        # someone sent us a message one one of our channels
        IncomingMessage=dict(
            connectionId=ConnectionId,
            message=MessageType
        ),
        # we made a new outgoing connection. this connection is also
        # valid as an input connection (we may receive messages on it)
        OutgoingConnectionEstablished=dict(connectionId=ConnectionId),
        # an outgoing connection failed
        OutgoingConnectionFailed=dict(connectionId=ConnectionId),
        # an outgoing connection closed
        OutgoingConnectionClosed=dict(connectionId=ConnectionId)
    )


class MessageBus(object):
    def __init__(self, busIdentity, endpoint, messageType, onEvent,
                 authToken=None, serializationContext=None, certPath=None, wantsSSL=True):
        """Initialize a MessageBus

        Args:
            busIdentity - any object that identifies this message bus
            endpoint - a (host, port) tuple that we're supposed to listen on, or None if we accept no incoming.
            messageType - the wire-type of all messages. Can be 'object' in which case we'll require a serializationContext
                to know how to serialize the names of types.
            serializationContext - the serialization context to use for serializing things, or None to use naked serialization
                from typed_python without any 'object'.
            authToken - the authentication token that must be sent to us for the connection to succeed. If None, then don't
                require authentication. MessageBus objects must have the same authToken to work together.
            onEvent - a callback function recieving a stream of 'eventType' objects (MessageBusEvents).
            certPath - (str or None) if we use SSL, an optional path to a cert file.
            wantsSSL (bool) - should we encrypt our channel with SSL

        The MessageBus listen for connection on the endpoint and calls onEvent from the read thread whenever
        a new event occurs.

        Clients may establish connection to other MessageBus objects, and will receive a ConnectionId object
        representing that channel. Other clients connecting in will produce their own 'ConnectionId's associated
        with the incoming connection. ConnectionIds are unique for a given MessageBus instance.

        Clients may send messages to outgoing connections that have been established or to other incoming connections.
        The send function indicates whether the send _might_ succeed (meaning it returns False only if it's KNOWN that
        the message channel on the other side is closed.)

        All event callbacks are fired from the same internal thread. This function should never throw,
        and if it blocks, it will block execution across all threads.

        Clients are expected to call 'start' to start the bus, and 'stop' to stop it and tear down threads.

        Clients can call 'connect' to get a connection id back, which they can pass to 'closeConnection' or
        'sendMessage'.
        """
        if authToken is not None:
            assert isinstance(authToken, str)

        self._logger = logging.getLogger(__file__)

        self.busIdentity = busIdentity

        self._certPath = certPath
        self.onEvent = onEvent
        self.serializationContext = serializationContext
        self.messageType = messageType
        self.eventType = MessageBusEvent(messageType)
        self._authToken = authToken
        self._listeningEndpoint = Endpoint(endpoint)
        self._lock = threading.RLock()
        self.started = False
        self._acceptSocket = None

        self._connIdToIncomingSocket = {}  # connectionId -> socket
        self._connIdToOutgoingSocket = {}  # connectionId -> socket

        self._socketToIncomingConnId = {}  # socket -> connectionId
        self._socketToOutgoingConnId = {}  # socket -> connectionId

        self._unauthenticatedConnections = set()
        self._connIdToIncomingEndpoint = {}  # connectionId -> Endpoint
        self._connIdToOutgoingEndpoint = {}  # connectionId -> Endpoint

        self._outThreadWakePipe = None
        self._inThreadWakePipe = None

        self._currentlyClosingConnections = set()  # set of ConnectionId, while we are closing them

        # how many bytes do we actually have in our deserialized pump loop
        # waiting to be sent down the wire.
        self.totalBytesPendingInOutputLoop = 0

        # how many bytes have we actually written (to anybody)
        self.totalBytesWritten = 0

        # how many bytes are in the deserialization queue that have not
        # created full messages.
        self.totalBytesPendingInInputLoop = 0
        self.totalBytesPendingInInputLoopHighWatermark = 0

        # how many bytes have we actually read (from anybody)
        self.totalBytesRead = 0

        self._connectionIdCounter = 0

        # queue of messages to write to other endpoints
        self._messagesToSendQueue = BytecountLimitedQueue(self._bytesPerMsg)
        self._eventsToFireQueue = queue.Queue()

        self._inputThread = threading.Thread(target=self._inThreadLoop)
        self._outputThread = threading.Thread(target=self._outThreadLoop)
        self._inputThread.daemon = True
        self._outputThread.daemon = True
        self._wantsSSL = wantsSSL

    def setMaxWriteQueueSize(self, queueSize):
        """Insist that we block any _sending_ threads if our outgoing queue gets too large."""
        self._messagesToSendQueue.setMaxBytes(queueSize)

    def isWriteQueueBlocked(self):
        return self._messagesToSendQueue.isBlocked()

    def start(self):
        """
        Start the message bus. May create threads and connect sockets.
        """
        assert not self.started

        if not self._setupAcceptSocket():
            raise FailedToStart()

        self.started = True
        self._outThreadWakePipe = os.pipe()
        self._inThreadWakePipe = os.pipe()
        self._inputThread.start()
        self._outputThread.start()

    def stop(self, timeout=None):
        """
        Stop the message bus.

        This bus may not be started again. Client threads blocked reading on the bus
        will return immediately with no message.
        """
        with self._lock:
            if not self.started:
                return
            self.started = False

        self._logger.debug(
            "Stopping MessageBus (%s) on endpoint %s%s",
            self.busIdentity,
            self._listeningEndpoint[0],
            self._listeningEndpoint[1]
        )

        self._messagesToSendQueue.put(Disconnected)
        self._eventsToFireQueue.put(self.eventType.Stopped())

        self._wakeOutputThreadIfAsleep()
        self._wakeInputThreadIfAsleep()

        self._inputThread.join(timeout=timeout)
        self._outputThread.join(timeout=timeout)

        if self._inputThread.isAlive() or self._outputThread.isAlive():
            raise Exception("Failed to shutdown our threads!")

        if self._acceptSocket is not None:
            self._ensureSocketClosed(self._acceptSocket)

        os.close(self._outThreadWakePipe[0])
        os.close(self._outThreadWakePipe[1])
        os.close(self._inThreadWakePipe[0])
        os.close(self._inThreadWakePipe[1])

        for sock in self._connIdToIncomingSocket.values():
            self._ensureSocketClosed(sock)

        for sock in self._connIdToOutgoingSocket.values():
            self._ensureSocketClosed(sock)

    def connect(self, endpoint: Endpoint) -> ConnectionId:
        """Make a connection to another endpoint and return a ConnectionId for it.

        You can send messages on this ConnectionId immediately.

        Args:
            endpoint (Endpoint) - the host/port to connect to

        Returns:
            a ConnectionId representing the connection.
        """
        if not self.started:
            raise Exception(f"Bus {self.busIdentity} is not active")

        endpoint = Endpoint(endpoint)

        connId = self._newConnectionId()

        with self._lock:
            self._connIdToOutgoingEndpoint[connId] = endpoint
            self._messagesToSendQueue.put((connId, TriggerConnect))  # just trying to trigger the endpoint

        self._wakeOutputThreadIfAsleep()

        return connId

    def sendMessage(self, connectionId, message):
        """Send a message to another endpoint endpoint.

        Send a message and return immediately (before guaranteeding we've sent
        the message). This function may block if we have too much outgoing data on the wire,
        but doesn't have to.

        Args:
            targetEndpoint - a host and port tuple.
            message - a message of type (self.MessageType) to send to the other endpoint.

        Returns:
            True if the message was queued, False if we preemptively dropped it because the
            other endpoint is disconnected.
        """
        if not self.started:
            raise Exception(f"Bus {self.busIdentity} is not active")

        if self.serializationContext is None:
            serializedMessage = serialize(self.messageType, message)
        else:
            serializedMessage = self.serializationContext.serialize(message, serializeType=self.messageType)

        with self._lock:
            isDefinitelyDead = connectionId not in self._connIdToOutgoingEndpoint and connectionId not in self._connIdToIncomingEndpoint

        if isDefinitelyDead:
            return False

        self._messagesToSendQueue.put((connectionId, serializedMessage))

        # wake up the send loop if its asleep
        self._wakeOutputThreadIfAsleep()

        return True

    def closeConnection(self, connectionId):
        """Trigger a connection close."""
        with self._lock:
            isDefinitelyDead = connectionId not in self._connIdToOutgoingEndpoint and connectionId not in self._connIdToIncomingEndpoint

        if isDefinitelyDead:
            return

        self._messagesToSendQueue.put((connectionId, TriggerDisconnect))

        self._wakeOutputThreadIfAsleep()

    def _newConnectionId(self):
        with self._lock:
            self._connectionIdCounter += 1
            return ConnectionId(id=self._connectionIdCounter)

    def _bytesPerMsg(self, msg):
        if not isinstance(msg, tuple):
            return 0

        if msg[1] is TriggerConnect or msg[1] is TriggerDisconnect:
            return 0

        return len(msg[1])

    def _wakeOutputThreadIfAsleep(self):
        assert os.write(self._outThreadWakePipe[1], b" ") == 1

    def _wakeInputThreadIfAsleep(self):
        assert os.write(self._inThreadWakePipe[1], b" ") == 1

    def _scheduleEvent(self, event):
        """Schedule an event to get sent to the onEvent callback on the input loop"""
        self._eventsToFireQueue.put(event)
        self._wakeInputThreadIfAsleep()

    def _setupAcceptSocket(self):
        assert not self.started

        if self._listeningEndpoint is None:
            return True

        context = sslContextFromCertPathOrNone(self._certPath)

        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM, 0)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        try:
            sock.bind((self._listeningEndpoint.host, self._listeningEndpoint.port))
            sock.listen(2048)
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, True)

            with self._lock:
                self._acceptSocket = sock

                if self._wantsSSL:
                    self._acceptSocket = context.wrap_socket(self._acceptSocket, server_side=True)

                self._logger.debug(
                    "%s listening on %s:%s",
                    self.busIdentity,
                    self._listeningEndpoint[0],
                    self._listeningEndpoint[1]
                )

                return True

        except OSError:
            sock.close()

        return False

    def _inThreadLoop(self):
        # all the sockets we care about
        allSockets = set()
        incomingSocketBuffers = {}

        try:
            if self._acceptSocket is not None:
                allSockets.add(self._acceptSocket)
            allSockets.add(self._inThreadWakePipe[0])

            while True:
                readReady = select.select(allSockets, [], [], SELECT_TIMEOUT)[0]

                for socketWithData in readReady:
                    if socketWithData == self._inThreadWakePipe[0]:
                        for receivedMsgTrigger in os.read(socketWithData, MSG_BUF_SIZE):
                            # one message should be on the queue for each msg trigger on the
                            # wake thread pipe
                            readMessage = self._eventsToFireQueue.get_nowait()

                            if isinstance(readMessage, tuple) and readMessage[1] is TriggerDisconnect:
                                connIdToClose = readMessage[0]

                                if connIdToClose in self._connIdToIncomingSocket:
                                    socket = self._connIdToIncomingSocket[connIdToClose]
                                elif connIdToClose in self._connIdToOutgoingSocket:
                                    socket = self._connIdToOutgoingSocket[connIdToClose]
                                else:
                                    socket = None

                                if socket is not None:
                                    self._markSocketClosed(socket)
                                    allSockets.discard(socket)
                                    del incomingSocketBuffers[socket]
                                    self._currentlyClosingConnections.discard(socket)
                            else:
                                assert isinstance(readMessage, self.eventType)

                                self._fireEvent(readMessage)

                                if readMessage.matches.Stopped:
                                    # this is the only valid way to exit the loop
                                    return

                                elif readMessage.matches.OutgoingConnectionEstablished:
                                    sock = self._connIdToOutgoingSocket.get(readMessage.connectionId)
                                    if sock is not None:
                                        allSockets.add(sock)
                                        incomingSocketBuffers[sock] = MessageBuffer()

                    elif socketWithData is self._acceptSocket:
                        newSocket, newSocketSource = socketWithData.accept()
                        newSocket.setblocking(False)
                        allSockets.add(newSocket)

                        with self._lock:
                            connId = self._newConnectionId()

                            with self._lock:
                                if self._authToken is not None:
                                    self._unauthenticatedConnections.add(connId)
                                self._connIdToIncomingSocket[connId] = newSocket
                                self._socketToIncomingConnId[newSocket] = connId
                                self._connIdToIncomingEndpoint[connId] = newSocketSource

                                incomingSocketBuffers[newSocket] = MessageBuffer()

                            self._fireEvent(
                                self.eventType.NewIncomingConnection(
                                    source=Endpoint(newSocketSource),
                                    connectionId=connId
                                )
                            )
                    else:
                        assert socketWithData in allSockets

                        try:
                            bytesReceived = socketWithData.recv(MSG_BUF_SIZE)
                        except ssl.SSLWantReadError:
                            bytesReceived = None
                        except ConnectionResetError:
                            bytesReceived = b""
                        except Exception as e:
                            logging.info("MessageBus read socket shutting down because of exception: %s", e)
                            bytesReceived = b""

                        if bytesReceived is None:
                            # do nothing
                            pass
                        elif bytesReceived == b"":
                            self._markSocketClosed(socketWithData)
                            allSockets.discard(socketWithData)
                            del incomingSocketBuffers[socketWithData]
                        else:
                            self.totalBytesRead += len(bytesReceived)

                            oldBytecount = incomingSocketBuffers[socketWithData].pendingBytecount()
                            newMessages = incomingSocketBuffers[socketWithData].write(bytesReceived)

                            self.totalBytesPendingInInputLoop += (
                                incomingSocketBuffers[socketWithData].pendingBytecount() - oldBytecount
                            )

                            self.totalBytesPendingInInputLoopHighWatermark = max(
                                self.totalBytesPendingInInputLoop,
                                self.totalBytesPendingInInputLoopHighWatermark
                            )

                            for m in newMessages:
                                if not self._handleIncomingMessage(m, socketWithData):
                                    self._markSocketClosed(socketWithData)
                                    allSockets.discard(socketWithData)
                                    del incomingSocketBuffers[socketWithData]
                                    break

        except Exception:
            self._logger.exception("MessageBus input loop failed")

    def _ensureSocketClosed(self, sock):
        try:
            sock.close()
        except OSError:
            pass

    def _markSocketClosed(self, socket):
        with self._lock:
            if socket in self._socketToIncomingConnId:
                connId = self._socketToIncomingConnId[socket]
                del self._socketToIncomingConnId[socket]
                del self._connIdToIncomingSocket[connId]
                del self._connIdToIncomingEndpoint[connId]
                self._unauthenticatedConnections.discard(connId)
                self._fireEvent(self.eventType.IncomingConnectionClosed(connectionId=connId))
            elif socket in self._socketToOutgoingConnId:
                connId = self._socketToOutgoingConnId[socket]
                del self._socketToOutgoingConnId[socket]
                del self._connIdToOutgoingSocket[connId]
                del self._connIdToOutgoingEndpoint[connId]
                self._fireEvent(self.eventType.OutgoingConnectionClosed(connectionId=connId))

        self._ensureSocketClosed(socket)

    def isUnauthenticated(self, connId):
        with self._lock:
            return connId in self._unauthenticatedConnections

    def _handleIncomingMessage(self, serializedMessage, socket):
        if socket in self._socketToIncomingConnId:
            connId = self._socketToIncomingConnId[socket]
        elif socket in self._socketToOutgoingConnId:
            connId = self._socketToOutgoingConnId[socket]
        else:
            return False

        if connId in self._unauthenticatedConnections:
            try:
                if serializedMessage.decode('utf8') != self._authToken:
                    self._logger.error("Unauthorized socket connected to us.")
                    return False

                self._unauthenticatedConnections.discard(connId)
                return True

            except Exception:
                self._logger.exception("Failed to read incoming auth message for %s", connId)
                return False
        else:
            try:
                if self.serializationContext is None:
                    message = deserialize(self.messageType, serializedMessage)
                else:
                    message = self.serializationContext.deserialize(serializedMessage, self.messageType)
            except Exception:
                self._logger.exception("Failed to deserialize a message")
                return False

            self._fireEvent(
                self.eventType.IncomingMessage(
                    connectionId=connId,
                    message=message
                )
            )

            return True

    def _fireEvent(self, event):
        try:
            self.onEvent(event)
        except Exception:
            self._logger.exception("Message callback threw unexpected exception")
            return

    def _connectTo(self, ssl_context, connId: ConnectionId):
        """Actually form an outgoing connection.

        This should only get called from the internals.
        """
        try:
            endpoint = self._connIdToOutgoingEndpoint[connId]

            naked_socket = socket.create_connection((endpoint.host, endpoint.port))

            if self._wantsSSL:
                ssl_socket = ssl_context.wrap_socket(naked_socket)
            else:
                ssl_socket = naked_socket

            ssl_socket.setblocking(False)

            with self._lock:
                self._socketToOutgoingConnId[ssl_socket] = connId
                self._connIdToOutgoingSocket[connId] = ssl_socket

            # this message notifies the input loop that it needs to pay attention to this
            # connection.
            self._scheduleEvent(self.eventType.OutgoingConnectionEstablished(connId))

            return True
        except Exception:
            with self._lock:
                if connId in self._connIdToOutgoingEndpoint:
                    del self._connIdToOutgoingEndpoint[connId]

            self._scheduleEvent(self.eventType.OutgoingConnectionFailed(connectionId=connId))

            return False

    def _outThreadLoop(self):
        context = sslContextFromCertPathOrNone(self._certPath)
        socketToBytes = {}  # socket -> bytes that need to be written

        def writeBytes(connId, bytes):
            if not bytes:
                return

            if connId in self._currentlyClosingConnections:
                return
            if connId in self._connIdToOutgoingSocket:
                sslSock = self._connIdToOutgoingSocket.get(connId)
            elif connId in self._connIdToIncomingSocket:
                sslSock = self._connIdToIncomingSocket.get(connId)
            else:
                return

            bytes = MessageBuffer.encode(bytes)

            self.totalBytesPendingInOutputLoop += len(bytes)

            if sslSock not in socketToBytes:
                socketToBytes[sslSock] = bytearray(bytes)
            else:
                socketToBytes[sslSock].extend(bytes)

        try:
            while True:
                # don't read from the serialization queue unless we can handle the
                # bytes in our 'self.totalBytesPendingInOutputLoop' flow
                canRead = (
                    self._messagesToSendQueue.maxBytes is None or
                    self.totalBytesPendingInOutputLoop < self._messagesToSendQueue.maxBytes
                )

                readReady, writeReady = select.select([self._outThreadWakePipe[0]] if canRead else [], socketToBytes, [])[:2]

                if readReady:
                    connectionAndMsg = self._messagesToSendQueue.get()

                    if connectionAndMsg is Disconnected:
                        return

                    assert os.read(self._outThreadWakePipe[0], 1) == b" "

                    connId, msg = connectionAndMsg

                    if msg is TriggerDisconnect:
                        # take this message, and make sure we never put this socket in the selectloop again.
                        if connId in socketToBytes:
                            del socketToBytes[connId]

                        with self._lock:
                            self._currentlyClosingConnections.add(connId)

                        # Then trigger the input loop to remove it and gracefully close it.
                        self._eventsToFireQueue.put((connId, TriggerDisconnect))
                        self._wakeInputThreadIfAsleep()

                    elif msg is TriggerConnect:
                        # we're supposed to connect to this worker
                        connected = self._connectTo(context, connId)

                        # and immediately write the auth token
                        if connected and self._authToken is not None:
                            writeBytes(connId, self._authToken.encode('utf8'))
                    else:
                        writeBytes(connId, msg)

                for writeable in writeReady:
                    try:
                        bytesWritten = writeable.send(socketToBytes[writeable])
                    except ssl.SSLWantWriteError:
                        bytesWritten = -1
                    except OSError:
                        bytesWritten = 0
                    except BrokenPipeError:
                        bytesWritten = 0
                    except Exception as e:
                        logging.info("MessageBus write socket shutting down because of exception: %s", e)
                        bytesWritten = 0

                    if bytesWritten > 0:
                        self.totalBytesPendingInOutputLoop -= bytesWritten
                        self.totalBytesWritten += bytesWritten

                    if bytesWritten == 0:
                        # the primary socket close pathway is in the input handler.
                        del socketToBytes[writeable]
                    elif bytesWritten == -1:
                        # do nothing
                        pass
                    else:
                        socketToBytes[writeable][:bytesWritten] = b""

                        if not socketToBytes[writeable]:
                            # we have no bytes to flush
                            del socketToBytes[writeable]
        except Exception:
            self._logger.exception("Socket loop for MessageBus had unexpected exception")
