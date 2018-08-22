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

from object_database.messages import ClientToServer, ServerToClient
from object_database.schema import Schema
from object_database.core_schema import core_schema
from object_database.view import View, Transaction, _cur_view, data_key, index_key
from object_database.algebraic_protocol import AlgebraicProtocol
from typed_python.hash import sha_hash
from typed_python import *

import uuid
import logging
import threading
import traceback
import json

tupleOfString = TupleOf(str)

class ConnectedChannel:
    def __init__(self, initial_tid, channel, connectionObject):
        self.channel = channel
        self.initial_tid = initial_tid
        self.connectionObject = connectionObject

    def sendKeyVersion(self, key, value, tid):
        toSend = value if not isinstance(value, set) else tupleOfString(value)

        self.channel.write(
            ServerToClient.KeyInfo(
                key=key,
                data=toSend,
                transaction_id=tid
                )
            )

    def sendTransaction(self, key_value, setAdds, setRemoves, tid):
        self.channel.write(
            ServerToClient.Transaction(
                writes={k:v for k,v in key_value.items()},
                set_adds=setAdds,
                set_removes=setRemoves,
                transaction_id=tid
                )
            )

    def sendInitializationMessage(self):
        self.channel.write(
            ServerToClient.Initialize(transaction_num=self.initial_tid, connIdentity=self.connectionObject._identity)
            )

    def sendTransactionSuccess(self, guid, success):
        self.channel.write(
            ServerToClient.TransactionResult(transaction_guid=guid,success=success)
            )


class Server:
    def __init__(self, kvstore):
        self._lock = threading.Lock()
        self._kvstore = kvstore
        self._clientChannels = {}
        
        #id of the next transaction
        self._cur_transaction_num = 0

        #for each key, the last version number we committed if it wasn't a write.
        self._version_numbers = {}
        
    def dropConnection(self, channel):
        with self._lock:
            if channel not in self._clientChannels:
                logging.error('Tried to drop a nonexistant channel')
                return

            co = self._clientChannels[channel].connectionObject
            del self._clientChannels[channel]
            self._dropConnectionEntry(co)

    def _createConnectionEntry(self):
        identity = sha_hash(str(uuid.uuid4())).hexdigest
        exists_key = data_key(core_schema.Connection, identity, " exists")
        exists_index = index_key(core_schema.Connection, " exists", True)

        self._handleNewTransaction(
            {exists_key: "true"},
            {exists_index: set([identity])},
            {},
            [],
            [],
            self._cur_transaction_num
            )

        return core_schema.Connection.fromIdentity(identity)

    def _dropConnectionEntry(self, entry):
        identity = entry._identity

        exists_key = data_key(core_schema.Connection, identity, " exists")
        exists_index = index_key(core_schema.Connection, " exists", True)

        self._handleNewTransaction(
            {exists_key: None},
            {},
            {exists_index: set([identity])},
            [],
            [],
            self._cur_transaction_num
            )

    def addConnection(self, channel):
        try:
            with self._lock:
                connectionObject = self._createConnectionEntry()

                connectedChannel = ConnectedChannel(self._cur_transaction_num, channel, connectionObject)

                self._clientChannels[channel] = connectedChannel

                channel.setClientToServerHandler(
                    lambda msg: self._onClientToServerMessage(connectedChannel, msg)
                    )

                connectedChannel.sendInitializationMessage()
        except:
            logging.error(
                "Failed during addConnection which should never happen:\n%s", 
                traceback.format_exc()
                )

    def _onClientToServerMessage(self, connectedChannel, msg):
        assert isinstance(msg, ClientToServer)
        if msg.matches.SendSets:
            with self._lock:
                for key in msg.keys:
                    connectedChannel.sendKeyVersion(
                        key,
                        self._kvstore.setMembers(key),
                        self._version_numbers.get(key, connectedChannel.initial_tid)
                        )

        elif msg.matches.SendValues:
            with self._lock:
                for key in msg.keys:
                    connectedChannel.sendKeyVersion(
                        key, 
                        self._kvstore.get(key),
                        self._version_numbers.get(key, connectedChannel.initial_tid)
                        )
        elif msg.matches.NewTransaction:
            try:
                with self._lock:
                    isOK = self._handleNewTransaction(
                        {k: v for k,v in msg.writes.items()},
                        {k: set(a) for k,a in msg.set_adds.items() if a},
                        {k: set(a) for k,a in msg.set_removes.items() if a},
                        msg.key_versions,
                        msg.index_versions,
                        msg.as_of_version
                        )
            except:
                traceback.print_exc()
                logging.error("Unknown error: %s", traceback.format_exc())
                isOK = False

            connectedChannel.sendTransactionSuccess(msg.transaction_guid, isOK)

    def _handleNewTransaction(self, 
                key_value, 
                set_adds, 
                set_removes, 
                keys_to_check_versions, 
                indices_to_check_versions, 
                as_of_version
                ):
        """Commit a transaction. 

        key_value: a map
            db_key -> (json_representation, database_representation)
        that we want to commit. We cache the normal_representation for later.

        set_adds: a map:
            db_key -> set of identities added to an index
        set_removes: a map:
            db_key -> set of identities removed from an index
        """

        set_adds = {k:v for k,v in set_adds.items() if v}
        set_removes = {k:v for k,v in set_removes.items() if v}

        self._cur_transaction_num += 1
        transaction_id = self._cur_transaction_num
        assert transaction_id > as_of_version

        keysWritingTo = []
        setsWritingTo = []

        for key in key_value:
            keysWritingTo.append(key)

        for subset in [set_adds, set_removes]:
            for k in subset:
                if subset[k]:
                    setsWritingTo.append(k)

        for subset in [keys_to_check_versions, indices_to_check_versions]:
            for key in subset:
                last_tid = self._version_numbers.get(key, -1)
                if as_of_version < last_tid:
                    return False

        #broadcast any values that have not changed since the client subscribed.
        #we'll replace this with a proper subscription mechanism at some point.
        for key in keysWritingTo:
            for client in self._clientChannels.values():
                last_tid = self._version_numbers.get(key, client.initial_tid)

                if last_tid <= client.initial_tid:
                    client.sendKeyVersion(key, self._kvstore.get(key), last_tid)

            self._version_numbers[key] = transaction_id

        for key in setsWritingTo:
            for client in self._clientChannels.values():
                last_tid = self._version_numbers.get(key, client.initial_tid)

                if last_tid <= client.initial_tid:
                    client.sendKeyVersion(key, self._kvstore.setMembers(key), last_tid)

            self._version_numbers[key] = transaction_id

        #set the json representation in the database
        self._kvstore.setSeveral({k: v for k,v in key_value.items()}, set_adds, set_removes)

        for client in self._clientChannels.values():
            client.sendTransaction(
                key_value,
                set_adds,
                set_removes,
                transaction_id
                )

        return True
