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
from object_database.view import View, JsonWithPyRep, Transaction, _cur_view, data_key, index_key
from object_database.algebraic_protocol import AlgebraicProtocol
from typed_python.hash import sha_hash

import json
import uuid
import logging
import threading
import traceback

class VersionedBase:
    def _best_version_offset_for(self, version):
        i = len(self.version_numbers) - 1

        while i >= 0:
            if self.version_numbers[i] <= version:
                return i
            i -= 1

        return None

    def isEmpty(self):
        return not self.version_numbers

    def validVersionIncoming(self, version_read, transaction_id):
        if not self.version_numbers:
            return True
        top = self.version_numbers[-1]
        assert transaction_id > version_read
        return version_read >= top

    def hasVersionInfoNewerThan(self, tid):
        if not self.version_numbers:
            return False
        return tid < self.version_numbers[-1]

    def newestValue(self):
        if self.version_numbers:
            return self.valueForVersion(self.version_numbers[-1])
        else:
            return self.valueForVersion(None)

class VersionedValue(VersionedBase):
    def __init__(self, tailValue):
        self.version_numbers = []
        self.values = []

        #the value for the lowest possible revision
        self.tailValue = tailValue

    def setVersionedValue(self, version_number, val):
        assert isinstance(val, JsonWithPyRep), val

        self.version_numbers.append(version_number)
        self.values.append(val)

    def valueForVersion(self, version):
        i = self._best_version_offset_for(version)

        if i is None:
            return self.tailValue
        return self.values[i]

    def cleanup(self, version_number):
        assert self.version_numbers[0] == version_number

        self.tailValue = self.values[0]
        self.version_numbers.pop(0)
        self.values.pop(0)

class VersionedSet(VersionedBase):
    #values in sets are always strings
    def __init__(self, tailValue):
        self.version_numbers = []
        self.adds = []
        self.removes = []

        self.tailValue = tailValue

    def setVersionedAddsAndRemoves(self, version, adds, removes):
        assert not adds or not removes
        assert adds or removes
        assert isinstance(adds, set)
        assert isinstance(removes, set)

        self.adds.append(adds)
        self.removes.append(removes)
        self.version_numbers.append(version)

    def cleanup(self, version_number):
        assert self.version_numbers[0] == version_numbers

        self.tailValue.update(self.adds[0])
        self.tailValue.difference_update(self.removes[0])

        self.version_numbers.pop(0)
        self.values.pop(0)
        self.adds.pop(0)
        self.removes.pop(0)

    def valueForVersion(self, version):
        ix = self._best_version_offset_for(version)
        if ix is None:
            ix = 0
        else:
            ix += 1

        return SetWithEdits(self.tailValue, self.adds[:ix], self.removes[:ix])

class SetWithEdits:
    def __init__(self, s, adds, removes):
        self.s = s
        self.adds = adds
        self.removes = removes

    def toSet(self):
        res = set(self.s)
        for i in range(len(self.adds)):
            res.update(self.adds[i])
            res.difference_update(self.removes[i])
        return res

    def pickAny(self, toAvoid):
        removed = set()

        for i in reversed(range(len(self.adds))):
            for a in self.adds[i]:
                if a not in removed and a not in toAvoid:
                    return a
            removed.update(self.removes[i])

        for a in self.s:
            if a not in removed and a not in toAvoid:
                return a

class ConnectedChannel:
    def __init__(self, initial_tid, channel, connectionObject):
        self.channel = channel
        self.initial_tid = initial_tid
        self.connectionObject = connectionObject

    def sendInitialKeyVersion(self, key, value):
        if isinstance(value, SetWithEdits):
            value = list(value.toSet())
            is_set = True
        else:
            value = value.jsonRep
            is_set = False

        self.channel.write(
            ServerToClient.KeyInfo(
                key=key,
                data=json.dumps(value),
                is_set=is_set,
                transaction_id=self.initial_tid
                )
            )

    def sendTransaction(self, key_value, setAdds, setRemoves, tid):
        self.channel.write(
            ServerToClient.Transaction(
                writes={k:json.dumps(v.jsonRep) for k,v in key_value.items()},
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
        self._transactionWriteLock = threading.Lock()

        #transaction of what's in the KV store
        self._cur_transaction_num = 0

        #minimum transaction we can support. This is the implicit transaction
        #for all the 'tail values'
        self._min_transaction_num = 0

        #for each version number in _version_numbers, how many views referring to it
        self._version_number_counts = {}
        self._min_reffed_version_number = None

        #list of outstanding version numbers in increasing order where we have writes
        #_min_transaction_num is the minimum of these and the current transaction
        self._version_numbers = []

        #for each version number, a set of keys that were set
        self._version_number_objects = {}

        #for each key, a VersionedValue or VersionedSet
        self._versioned_objects = {}
        
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
        exists_key = data_key(core_schema.Connection, identity, ".exists")
        exists_index = index_key(core_schema.Connection, ".exists", True)

        self._handleNewTransaction(
            {exists_key: JsonWithPyRep(True, True)},
            {exists_index: set([identity])},
            {},
            [],
            [],
            self._cur_transaction_num
            )

        return core_schema.Connection.fromIdentity(identity)

    def _dropConnectionEntry(self, entry):
        identity = entry._identity

        exists_key = data_key(core_schema.Connection, identity, ".exists")
        exists_index = index_key(core_schema.Connection, ".exists", True)

        self._handleNewTransaction(
            {exists_key: JsonWithPyRep(None, None)},
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
                    if key not in self._versioned_objects:
                        members = self._kvstore.setMembers(key)
                        self._versioned_objects[key] = VersionedSet(members)

                    connectedChannel.sendInitialKeyVersion(
                        key, 
                        self._versioned_objects[key].valueForVersion(
                            connectedChannel.initial_tid
                            )
                        )
        elif msg.matches.SendValues:
            with self._lock:
                for key in msg.keys:
                    if key not in self._version_numbers:
                        self._versioned_objects[key] = VersionedValue(
                            JsonWithPyRep(
                                self._kvstore.get(key),
                                None
                                )
                            )

                    connectedChannel.sendInitialKeyVersion(
                        key, 
                        self._versioned_objects[key].valueForVersion(
                            connectedChannel.initial_tid
                            )
                        )
        elif msg.matches.NewTransaction:
            try:
                with self._lock:
                    isOK = self._handleNewTransaction(
                        {k: JsonWithPyRep(json.loads(v),None) for k,v in msg.writes.items()},
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

        assert as_of_version >= self._min_transaction_num

        self._cur_transaction_num += 1
        transaction_id = self._cur_transaction_num
        assert transaction_id > as_of_version

        keysWritingTo = []

        for key in key_value:
            keysWritingTo.append(key)

            if key not in self._versioned_objects:
                self._versioned_objects[key] = (
                        VersionedValue(
                            JsonWithPyRep(
                            self._kvstore.get(key),
                            None
                            )
                        )
                    )
        
        for subset in [set_adds, set_removes]:
            for k in subset:
                if subset[k]:
                    keysWritingTo.append(k)
                
                    if k not in self._versioned_objects:
                        self._versioned_objects[k] = VersionedSet(
                            self._kvstore.setMembers(k)
                            )

        for subset in [keys_to_check_versions, indices_to_check_versions]:
            for key in subset:
                if not self._versioned_objects[key].validVersionIncoming(as_of_version, transaction_id):
                    return False

        for key in keysWritingTo:
            obj = self._versioned_objects[key]

            for client in self._clientChannels.values():
                #see if this value has not changed since this client connected. If so, we need
                #to send the value we currently have, because otherwise we cannot recreate it
                #if it asks for it later.
                if not obj.hasVersionInfoNewerThan(client.initial_tid):
                    client.sendInitialKeyVersion(key, obj.newestValue())

        #set the json representation in the database
        self._kvstore.setSeveral({k: v.jsonRep for k,v in key_value.items()}, set_adds, set_removes)

        priors = {}

        for k,v in key_value.items():
            versioned = self._versioned_objects[k]
            priors[k] = versioned.newestValue()
            versioned.setVersionedValue(transaction_id, v)

        for k,a in set_adds.items():
            if a:
                self._versioned_objects[k].setVersionedAddsAndRemoves(transaction_id, a, set())
        for k,r in set_removes.items():
            if r:
                self._versioned_objects[k].setVersionedAddsAndRemoves(transaction_id, set(), r)

        #record what objects we touched
        self._version_number_objects[transaction_id] = list(key_value.keys())
        self._version_numbers.append(transaction_id)

        for client in self._clientChannels.values():
            client.sendTransaction(
                key_value,
                set_adds,
                set_removes,
                transaction_id
                )

        return True
