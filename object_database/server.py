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

from object_database.messages import ClientToServer, ServerToClient, getHeartbeatInterval
from object_database.schema import Schema
from object_database.messages import SchemaDefinition
from object_database.core_schema import core_schema
import object_database.keymapping as keymapping
from object_database.algebraic_protocol import AlgebraicProtocol
from typed_python.hash import sha_hash
from typed_python import *

import redis
import queue
import time
import uuid
import logging
import threading
import traceback
import json

class ConnectedChannel:
    def __init__(self, initial_tid, channel, connectionObject):
        self.channel = channel
        self.initial_tid = initial_tid
        self.connectionObject = connectionObject
        self.missedHeartbeats = 0
        self.definedSchemas = {}
        self.subscribedTypes = set() #schema, type
        self.subscribedIds = set() #identities
        self.subscribedIndexKeys = set() #full index keys

    def heartbeat(self):
        self.missedHeartbeats = 0

    def sendTransaction(self, msg):
        #we need to cut the transaction down
        self.channel.write(msg)

    def sendInitializationMessage(self):
        self.channel.write(
            ServerToClient.Initialize(transaction_num=self.initial_tid, connIdentity=self.connectionObject._identity)
            )

    def sendTransactionSuccess(self, guid, success, badKey):
        self.channel.write(
            ServerToClient.TransactionResult(transaction_guid=guid,success=success,badKey=badKey)
            )

class Server:
    def __init__(self, kvstore):
        self._lock = threading.RLock()
        self._kvstore = kvstore

        self.verbose = False

        self._removeOldDeadConnections()

        self._clientChannels = {}
        
        #id of the next transaction
        self._cur_transaction_num = 0

        #for each key, the last version number we committed
        self._version_numbers = {}

        #(schema,type) to set(subscribed channel)
        self._type_to_channel = {}

        #index-stringname to set(subscribed channel)
        self._index_to_channel = {}

        self._id_to_channel = {}

        self.longTransactionThreshold = 1.0
        self.logFrequency = 10.0

        self._transactions = 0
        self._keys_set = 0
        self._index_values_updated = 0
        self._subscriptions_written = 0

        self._subscriptionResponseThread = None

        self._shouldStop = threading.Event()

        #a queue of queue-subscription messages. we have to handle
        #these on another thread because they can be quite large, and we don't want
        #to prevent message processing on the main thread.
        self._subscriptionQueue = queue.Queue()

        #if we're building a subscription up, all the objects that have changed while our
        #lock was released.
        self._pendingSubscriptionRecheck = None

        #fault injector to test this thing
        self._subscriptionBackgroundThreadCallback = None

    def start(self):
        self._subscriptionResponseThread = threading.Thread(target=self.serviceSubscriptions)
        self._subscriptionResponseThread.daemon = True
        self._subscriptionResponseThread.start()

    def stop(self):
        self._shouldStop.set()
        self._subscriptionQueue.put((None,None))
        self._subscriptionResponseThread.join()

    def serviceSubscriptions(self):
        while not self._shouldStop.is_set():
            try:
                try:
                    (connectedChannel, msg) = self._subscriptionQueue.get(timeout=1.0)
                    if connectedChannel is not None:
                        self.handleSubscriptionOnBackgroundThread(connectedChannel, msg)
                except queue.Empty:
                    pass
            except:
                logging.error("Unexpected error in serviceSubscription thread:\n%s", traceback.format_exc())


    def _removeOldDeadConnections(self):        
        connection_index = keymapping.index_key(core_schema.Connection, " exists", True)
        oldIds = self._kvstore.getSetMembers(keymapping.index_key(core_schema.Connection, " exists", True))

        if oldIds:
            self._kvstore.setSeveral(
                {keymapping.data_key(core_schema.Connection, identity, " exists"):None for identity in oldIds},
                {},
                {connection_index: set(oldIds)}
                )

    def checkForDeadConnections(self):
        with self._lock:
            heartbeatCount = {}

            for c in list(self._clientChannels):
                missed = self._clientChannels[c].missedHeartbeats
                self._clientChannels[c].missedHeartbeats += 1

                heartbeatCount[missed] = heartbeatCount.get(missed,0) + 1

                if missed >= 4:
                    logging.info(
                        "Connection %s has not heartbeat in a long time. Killing it.", 
                        self._clientChannels[c].connectionObject._identity
                        )

                    c.close()

            logging.info("Connection heartbeat distribution is %s", heartbeatCount)

    def dropConnection(self, channel):
        with self._lock:
            if channel not in self._clientChannels:
                logging.error('Tried to drop a nonexistant channel')
                return

            connectedChannel = self._clientChannels[channel]

            for schema_name, typename in connectedChannel.subscribedTypes:
                self._type_to_channel[schema_name,typename].discard(connectedChannel)

            for index_key in connectedChannel.subscribedIndexKeys:
                self._index_to_channel[index_key].discard(connectedChannel)
                if not self._index_to_channel[index_key]:
                    del self._index_to_channel[index_key]

            for identity in connectedChannel.subscribedIds:
                self._id_to_channel[identity].discard(connectedChannel)
                if not self._id_to_channel[identity]:
                    del self._id_to_channel[identity]

            co = connectedChannel.connectionObject

            logging.info("Server dropping connection for connectionObject._identity = %s", co._identity)

            del self._clientChannels[channel]

            self._dropConnectionEntry(co)

    def _createConnectionEntry(self):
        identity = sha_hash(str(uuid.uuid4())).hexdigest
        exists_key = keymapping.data_key(core_schema.Connection, identity, " exists")
        exists_index = keymapping.index_key(core_schema.Connection, " exists", True)

        self._handleNewTransaction(
            None,
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

        exists_key = keymapping.data_key(core_schema.Connection, identity, " exists")
        exists_index = keymapping.index_key(core_schema.Connection, " exists", True)

        self._handleNewTransaction(
            None,
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
                    lambda msg: self.onClientToServerMessage(connectedChannel, msg)
                    )

                connectedChannel.sendInitializationMessage()
        except:
            logging.error(
                "Failed during addConnection which should never happen:\n%s", 
                traceback.format_exc()
                )

    def handleSubscriptionOnBackgroundThread(self, connectedChannel, msg):
        try:
            with self._lock:
                t0 = time.time()

                if connectedChannel.channel not in self._clientChannels:
                    logging.warn("Ignoring subscription from dead channel.")
                    return

                schema_name = msg.schema
                
                definition = connectedChannel.definedSchemas.get(schema_name)

                assert definition is not None, "can't subscribe to a schema we don't know about!"

                assert msg.typename is not None
                typename = msg.typename

                assert typename in definition, "Can't subscribe to a type we didn't define in the schema: %s not in %s" % (typename, list(definition))

                typedef = definition[typename]

                if msg.fieldname_and_value is None:
                    field, val = " exists", keymapping.index_value_to_hash(True)
                else:
                    field, val = msg.fieldname_and_value

                if field == '_identity':
                    identities = set([val])
                else:
                    identities = set(self._kvstore.getSetMembers(keymapping.index_key_from_names_encoded(schema_name, typename, field, val)))

                t1 = time.time()

                self._pendingSubscriptionRecheck = []

            #we need to send everything we know about 'identities', keeping in mind that we have to 
            #check any new identities that get written to in the background to see if they belong
            #in the new set
            identities_left_to_send = set(identities)

            done = False
            messageCount = 0
            while True:
                locktime_start = time.time()

                if self._subscriptionBackgroundThreadCallback:
                    self._subscriptionBackgroundThreadCallback(messageCount)

                with self._lock:
                    messageCount += 1
                    if messageCount == 2:
                        logging.info(
                            "Beginning large subscription for %s/%s/%s", 
                            schema_name, msg.typename, msg.fieldname_and_value
                            )

                    self._sendPartialSubscription(
                        connectedChannel,
                        schema_name, typename, msg.fieldname_and_value,
                        typedef,
                        identities,
                        identities_left_to_send
                        )

                    if not identities_left_to_send:
                        self._markSubscriptionComplete(
                            schema_name, typename, msg.fieldname_and_value,
                            identities,
                            connectedChannel
                            )

                        connectedChannel.channel.write(
                            ServerToClient.SubscriptionComplete(
                                schema=schema_name,
                                typename=msg.typename,
                                fieldname_and_value=msg.fieldname_and_value,
                                tid=self._cur_transaction_num
                                )
                            )

                        break

                #don't hold the lock more than 75% of the time.
                time.sleep( (time.time() - locktime_start) / 3 )

            if self._subscriptionBackgroundThreadCallback:
                self._subscriptionBackgroundThreadCallback("DONE")

            if messageCount > 5:
                logging.info(
                    "Subscription took [%.2f, %.2f] seconds over %s messages and produced %s objects for %s/%s/%s", 
                    t1 - t0,
                    time.time() - t1,
                    messageCount,
                    len(identities),
                    schema_name, msg.typename, msg.fieldname_and_value
                    )
        finally:
            with self._lock:
                self._pendingSubscriptionRecheck = None

    def _markSubscriptionComplete(self, schema, typename, fieldname_and_value, identities, connectedChannel):
        if fieldname_and_value is not None:
            #this is an index subscription
            for ident in identities:
                self._id_to_channel.setdefault(ident, set()).add(connectedChannel)
                connectedChannel.subscribedIds.add(ident)

            if fieldname_and_value[0] != '_identity':
                index_key = keymapping.index_key_from_names_encoded(schema, typename, fieldname_and_value[0], fieldname_and_value[1])

                self._index_to_channel.setdefault(index_key, set()).add(connectedChannel)

                connectedChannel.subscribedIndexKeys.add(index_key)
            else:
                #an object's identity cannot change, so we don't need to track our subscription to it
                pass
        else:
            #this is a type-subscription
            if (schema, typename) not in self._type_to_channel:
                self._type_to_channel[schema, typename] = set()

            self._type_to_channel[schema, typename].add(connectedChannel)
            connectedChannel.subscribedTypes.add((schema, typename))


    def _sendPartialSubscription(self, 
                connectedChannel,
                schema_name, typename, fieldname_and_value,
                typedef,
                identities,
                identities_left_to_send):

        #get some objects to send
        BATCH_SIZE = 100

        kvs = {}
        index_vals = {}

        to_send = []
        for transactionMessage in self._pendingSubscriptionRecheck:
            for key, val in transactionMessage.writes.items():
                #if we write to a key we've already sent, we'll need to resend it
                identity = keymapping.split_data_key(key)[2]
                if identity in identities:
                    identities_left_to_send.add(identity)

            for add_index_key, add_index_identities in transactionMessage.set_adds.items():
                add_schema, add_typename, add_fieldname, add_hashVal = keymapping.split_index_key_full(add_index_key)

                if add_schema == schema_name and add_typename == typename and (
                        fieldname_and_value is None and add_fieldname == " exists" or 
                        fieldname_and_value is not None and tuple(fieldname_and_value) == (add_fieldname, add_hashVal)
                        ):
                    identities_left_to_send.update(add_index_identities)

        self._pendingSubscriptionRecheck = []

        while identities_left_to_send and len(to_send) < BATCH_SIZE:
            to_send.append(identities_left_to_send.pop())
                
        for fieldname in typedef.fields:
            keys = [keymapping.data_key_from_names(schema_name, typename, identity, fieldname)
                            for identity in to_send]

            vals = self._kvstore.getSeveral(keys)

            for i in range(len(keys)):
                kvs[keys[i]] = vals[i]

        for fieldname in typedef.indices:
            keys = [keymapping.data_reverse_index_key(schema_name, typename, identity, fieldname)
                            for identity in to_send]

            vals = self._kvstore.getSeveral(keys)

            for i in range(len(keys)):
                index_vals[keys[i]] = vals[i]

        connectedChannel.channel.write(
            ServerToClient.SubscriptionData(
                schema=schema_name,
                typename=typename,
                fieldname_and_value=fieldname_and_value,
                values=kvs,
                index_values=index_vals,
                identities=None if fieldname_and_value is None else tuple(to_send)
                )
            )


    def onClientToServerMessage(self, connectedChannel, msg):
        assert isinstance(msg, ClientToServer)

        if msg.matches.Heartbeat:
            connectedChannel.heartbeat()
        elif msg.matches.Flush:
            with self._lock:
                connectedChannel.channel.write(ServerToClient.FlushResponse(guid=msg.guid))
        elif msg.matches.DefineSchema:
            assert isinstance(msg.definition, SchemaDefinition)
            connectedChannel.definedSchemas[msg.name] = msg.definition
        elif msg.matches.Subscribe:
            self._subscriptionQueue.put((connectedChannel, msg))
        elif msg.matches.NewTransaction:
            try:
                with self._lock:
                    isOK, badKey = self._handleNewTransaction(
                        connectedChannel,
                        {k: v for k,v in msg.writes.items()},
                        {k: set(a) for k,a in msg.set_adds.items() if a},
                        {k: set(a) for k,a in msg.set_removes.items() if a},
                        msg.key_versions,
                        msg.index_versions,
                        msg.as_of_version
                        )
            except:
                logging.error("Unknown error committing transaction: %s", traceback.format_exc())
                isOK = False
                badKey = "<NONE>"

            connectedChannel.sendTransactionSuccess(msg.transaction_guid, isOK, badKey)

    def indexReverseLookupKvs(self, adds, removes):
        res = {}

        for indexKey, identities in removes.items():
            schemaname, typename, fieldname, valuehash = keymapping.split_index_key_full(indexKey)

            for ident in identities:
                res[keymapping.data_reverse_index_key(schemaname, typename, ident, fieldname)] = None

        for indexKey, identities in adds.items():
            schemaname, typename, fieldname, valuehash = keymapping.split_index_key_full(indexKey)

            for ident in identities:
                res[keymapping.data_reverse_index_key(schemaname, typename, ident, fieldname)] = valuehash

        return res

    def _broadcastSubscriptionIncrease(self, channel, indexKey, newIds):
        newIds = list(newIds)

        schema_name, typename, fieldname, fieldval = keymapping.split_index_key_full(indexKey)

        channel.channel.write(
            ServerToClient.SubscriptionIncrease(
                schema=schema_name,
                typename=typename,
                fieldname_and_value=(fieldname, fieldval),
                identities=newIds
                )
            )

    def _increaseBroadcastTransactionToInclude(self, channel, indexKey, newIds, key_value, set_adds, set_removes):
        #we need to include all the data for the objects in 'newIds' to the transaction
        #that we're broadcasting
        schema_name, typename, fieldname, fieldval = keymapping.split_index_key_full(indexKey)

        typedef = channel.definedSchemas.get(schema_name).get(typename)

        valsToGet = []
        for field_to_pull in typedef.fields:
            for ident in newIds:
                valsToGet.append(keymapping.data_key_from_names(schema_name,typename, ident, field_to_pull))

        results = self._kvstore.getSeveral(valsToGet)
        
        key_value.update({valsToGet[i]: results[i] for i in range(len(valsToGet))})

        reverseKeys = []
        for index_name in typedef.indices:
            for ident in newIds:
                reverseKeys.append(keymapping.data_reverse_index_key(schema_name, typename, ident, index_name))

        reverseVals = self._kvstore.getSeveral(reverseKeys)
        reverseKVMap = {reverseKeys[i]:reverseVals[i] for i in range(len(reverseKeys))}

        for index_name in typedef.indices:
            for ident in newIds:
                fieldval = reverseKVMap.get(keymapping.data_reverse_index_key(schema_name, typename, ident, index_name))

                if fieldval is not None:
                    ik = keymapping.index_key_from_names_encoded(schema_name, typename, index_name, fieldval)
                    set_adds.setdefault(ik, set()).add(ident)

    def _handleNewTransaction(self, 
                sourceChannel,
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
        self._cur_transaction_num += 1
        transaction_id = self._cur_transaction_num
        assert transaction_id > as_of_version

        t0 = time.time()

        set_adds = {k:v for k,v in set_adds.items() if v}
        set_removes = {k:v for k,v in set_removes.items() if v}

        identities_mentioned = set()

        keysWritingTo = set()
        setsWritingTo = set()
        schemaTypePairsWriting = set()

        if sourceChannel:
            #check if we created any new objects to which we are not type-subscribed
            #and if so, ensure we are subscribed
            for add_index, added_identities in set_adds.items():
                schema_name, typename, fieldname, fieldval = keymapping.split_index_key_full(add_index)
                if fieldname == ' exists':
                    if (schema_name, typename) not in sourceChannel.subscribedTypes:
                        sourceChannel.subscribedIds.update(added_identities)
                        for new_id in added_identities:
                            self._id_to_channel.setdefault(new_id, set()).add(sourceChannel)
                        self._broadcastSubscriptionIncrease(sourceChannel, add_index, added_identities)

        for key in key_value:
            keysWritingTo.add(key)

            schema_name, typename, ident = keymapping.split_data_key(key)[:3]
            schemaTypePairsWriting.add((schema_name,typename))

            identities_mentioned.add(ident)

        for subset in [set_adds, set_removes]:
            for k in subset:
                if subset[k]:
                    schema_name, typename = keymapping.split_index_key(k)[:2]
                    
                    schemaTypePairsWriting.add((schema_name,typename))

                    setsWritingTo.add(k)

                    identities_mentioned.update(subset[k])

        #check all version numbers for transaction conflicts.
        for subset in [keys_to_check_versions, indices_to_check_versions]:
            for key in subset:
                last_tid = self._version_numbers.get(key, -1)
                if as_of_version < last_tid:
                    return (False, key)

        for key in keysWritingTo:
            self._version_numbers[key] = transaction_id

        for key in setsWritingTo:
            self._version_numbers[key] = transaction_id

        t1 = time.time()

        #set the json representation in the database
        target_kvs = {k: v for k,v in key_value.items()}
        target_kvs.update(self.indexReverseLookupKvs(set_adds, set_removes))

        new_sets, dropped_sets = self._kvstore.setSeveral(target_kvs, set_adds, set_removes)

        #update the metadata index
        indexSetAdds = {}
        indexSetRemoves = {}
        for s in new_sets:
            index_key, index_val = keymapping.split_index_key(s)
            if index_key not in indexSetAdds:
                indexSetAdds[index_key] = set()
            indexSetAdds[index_key].add(index_val)

        for s in dropped_sets:
            index_key, index_val = keymapping.split_index_key(s)
            if index_key not in indexSetRemoves:
                indexSetRemoves[index_key] = set()
            indexSetRemoves[index_key].add(index_val)

        self._kvstore.setSeveral({}, indexSetAdds,indexSetRemoves)

        t2 = time.time()

        #check any index-level subscriptions that are going to increase as a result of this
        #transaction and add the backing data to the relevant transaction.
        for index_key, adds in list(set_adds.items()):
            if index_key in self._index_to_channel:
                idsToAddToTransaction = set()

                for channel in self._index_to_channel.get(index_key):
                    newIds = adds.difference(channel.subscribedIds)
                    for new_id in newIds:
                        self._id_to_channel.setdefault(new_id, set()).add(channel)
                        channel.subscribedIds.add(new_id)
    
                    self._broadcastSubscriptionIncrease(channel, index_key, newIds)

                    idsToAddToTransaction.update(newIds)

                if idsToAddToTransaction:
                    self._increaseBroadcastTransactionToInclude(
                        channel, #deliberately just using whatever random channel, under
                                 #the assumption they're all the same. it would be better
                                 #to explictly compute the union of the relevant set of defined fields,
                                 #as its possible one channel has more fields for a type than another
                                 #and we'd like to broadcast them all
                        index_key, idsToAddToTransaction, key_value, set_adds, set_removes)

        transaction_message = None
        channelsTriggered = set()

        for schema_type_pair in schemaTypePairsWriting:
            channelsTriggered.update(self._type_to_channel.get(schema_type_pair,()))

        for i in identities_mentioned:
            if i in self._id_to_channel:
                channelsTriggered.update(self._id_to_channel[i])
    
        transaction_message = ServerToClient.Transaction(
            writes={k:v for k,v in key_value.items()},
            set_adds=set_adds,
            set_removes=set_removes,
            transaction_id=transaction_id
            )

        if self._pendingSubscriptionRecheck is not None:
            self._pendingSubscriptionRecheck.append(transaction_message)

        for channel in channelsTriggered:
            channel.sendTransaction(transaction_message)
        
        if self.verbose or time.time() - t0 > self.longTransactionThreshold:
            logging.info("Transaction [%.2f/%.2f/%.2f] with %s writes, %s set ops: %s", 
                t1 - t0, t2 - t1, time.time() - t2,
                len(key_value), len(set_adds) + len(set_removes), sorted(key_value)[:3]
                )

        return (True, None)
