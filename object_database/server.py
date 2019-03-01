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
from object_database.identity import IdentityProducer
from object_database.messages import SchemaDefinition
from object_database.core_schema import core_schema
import object_database.keymapping as keymapping
from object_database.util import Timer
from typed_python import *

import queue
import time
import logging
import threading
import traceback

DEFAULT_GC_INTERVAL = 900.0


class ConnectedChannel:
    def __init__(self, initial_tid, channel, connectionObject, identityRoot):
        super(ConnectedChannel, self).__init__()
        self.channel = channel
        self.initial_tid = initial_tid
        self.connectionObject = connectionObject
        self.missedHeartbeats = 0
        self.definedSchemas = {}
        self.subscribedTypes = {}  # schema, type to the lazy transaction id (or -1 if not lazy)
        self.subscribedIds = set()  # identities
        self.subscribedIndexKeys = {}  # full index keys to lazy transaction id
        self.identityRoot = identityRoot
        self.pendingTransactions = {}
        self._needsAuthentication = True

    @property
    def needsAuthentication(self):
        return self._needsAuthentication

    def authenticate(self):
        self._needsAuthentication = False

    def heartbeat(self):
        self.missedHeartbeats = 0

    def sendTransaction(self, msg):
        # we need to cut the transaction down
        self.channel.write(msg)

    def sendInitializationMessage(self):
        self.channel.write(
            ServerToClient.Initialize(
                transaction_num=self.initial_tid,
                connIdentity=self.connectionObject._identity,
                identity_root=self.identityRoot
            )
        )

    def sendTransactionSuccess(self, guid, success, badKey):
        self.channel.write(
            ServerToClient.TransactionResult(transaction_guid=guid, success=success, badKey=badKey)
        )

    def handleTransactionData(self, msg):
        guid = msg.transaction_guid
        if guid not in self.pendingTransactions:
            self.pendingTransactions[guid] = {
                'writes': {},
                'set_adds': {},
                'set_removes': {},
                'key_versions': set(),
                'index_versions': set()
            }

        self.pendingTransactions[guid]['writes'].update({k: msg.writes[k] for k in msg.writes})
        self.pendingTransactions[guid]['set_adds'].update({k: set(msg.set_adds[k]) for k in msg.set_adds if msg.set_adds[k]})
        self.pendingTransactions[guid]['set_removes'].update({k: set(msg.set_removes[k]) for k in msg.set_removes if msg.set_removes[k]})
        self.pendingTransactions[guid]['key_versions'].update(msg.key_versions)
        self.pendingTransactions[guid]['index_versions'].update(msg.index_versions)

    def extractTransactionData(self, guid):
        return self.pendingTransactions.pop(guid)


class Server:
    def __init__(self, kvstore, auth_token):
        self._kvstore = kvstore
        self._auth_token = auth_token

        self._lock = threading.RLock()

        self.verbose = False

        self._gc_interval = DEFAULT_GC_INTERVAL

        self._removeOldDeadConnections()

        # InMemoryChannel or ServerToClientProtocol -> ConnectedChannel
        self._clientChannels = {}

        # id of the next transaction
        self._cur_transaction_num = 0

        # for each key, the last version number we committed
        self._version_numbers = {}
        self._version_numbers_timestamps = {}

        # (schema,type) to set(subscribed channel)
        self._type_to_channel = {}

        # index-stringname to set(subscribed channel)
        self._index_to_channel = {}

        # for each individually subscribed ID, a set of channels
        self._id_to_channel = {}

        self.longTransactionThreshold = 1.0
        self.logFrequency = 10.0

        self.MAX_NORMAL_TO_SEND_SYNCHRONOUSLY = 1000
        self.MAX_LAZY_TO_SEND_SYNCHRONOUSLY = 10000

        self._transactions = 0
        self._keys_set = 0
        self._index_values_updated = 0
        self._subscriptions_written = 0

        self._subscriptionResponseThread = None

        self._shouldStop = threading.Event()

        # a queue of queue-subscription messages. we have to handle
        # these on another thread because they can be quite large, and we don't want
        # to prevent message processing on the main thread.
        self._subscriptionQueue = queue.Queue()

        # if we're building a subscription up, all the objects that have changed while our
        # lock was released.
        self._pendingSubscriptionRecheck = None

        # fault injector to test this thing
        self._subscriptionBackgroundThreadCallback = None
        self._lazyLoadCallback = None

        self._last_garbage_collect_timestamp = None

        self.identityProducer = IdentityProducer(self.allocateNewIdentityRoot())

        self._logger = logging.getLogger(__name__)

    def start(self):
        self._subscriptionResponseThread = threading.Thread(target=self.serviceSubscriptions)
        self._subscriptionResponseThread.daemon = True
        self._subscriptionResponseThread.start()

    def stop(self):
        self._shouldStop.set()
        self._subscriptionQueue.put((None, None))
        self._subscriptionResponseThread.join()

    def allocateNewIdentityRoot(self):
        with self._lock:
            curIdentityRoot = self._kvstore.get(" identityRoot")
            if curIdentityRoot is None:
                curIdentityRoot = 0
            else:
                curIdentityRoot = deserialize(int, bytes.fromhex(curIdentityRoot))

            result = curIdentityRoot

            self._kvstore.set(" identityRoot", serialize(int, curIdentityRoot+1).hex())

            return result

    def serviceSubscriptions(self):
        while not self._shouldStop.is_set():
            try:
                try:
                    (connectedChannel, msg) = self._subscriptionQueue.get(timeout=1.0)
                    if connectedChannel is not None:
                        self.handleSubscriptionOnBackgroundThread(connectedChannel, msg)
                except queue.Empty:
                    pass
            except Exception:
                self._logger.error("Unexpected error in serviceSubscription thread:\n%s", traceback.format_exc())

    def _removeOldDeadConnections(self):
        connection_index = keymapping.index_key(core_schema.Connection, " exists", True)
        oldIds = self._kvstore.getSetMembers(keymapping.index_key(core_schema.Connection, " exists", True))

        if oldIds:
            self._kvstore.setSeveral(
                {keymapping.data_key(core_schema.Connection, identity, " exists"): None for identity in oldIds},
                {},
                {connection_index: set(oldIds)}
            )

    def checkForDeadConnections(self):
        with self._lock:
            heartbeatCount = {}

            for c in list(self._clientChannels):
                missed = self._clientChannels[c].missedHeartbeats
                self._clientChannels[c].missedHeartbeats += 1

                heartbeatCount[missed] = heartbeatCount.get(missed, 0) + 1

                if missed >= 4:
                    self._logger.info(
                        "Connection %s has not heartbeat in a long time. Killing it.",
                        self._clientChannels[c].connectionObject._identity
                    )

                    c.close()
                    self.dropConnection(c)

            self._logger.debug("Connection heartbeat distribution is %s", heartbeatCount)

    def dropConnection(self, channel):
        with self._lock:
            if channel not in self._clientChannels:
                self._logger.warn('Tried to drop a nonexistant channel')
                return

            connectedChannel = self._clientChannels[channel]

            for schema_name, typename in connectedChannel.subscribedTypes:
                self._type_to_channel[schema_name, typename].discard(connectedChannel)

            for index_key in connectedChannel.subscribedIndexKeys:
                self._index_to_channel[index_key].discard(connectedChannel)
                if not self._index_to_channel[index_key]:
                    del self._index_to_channel[index_key]

            for identity in connectedChannel.subscribedIds:
                if identity in self._id_to_channel:
                    self._id_to_channel[identity].discard(connectedChannel)
                    if not self._id_to_channel[identity]:
                        del self._id_to_channel[identity]

            co = connectedChannel.connectionObject

            self._logger.info("Server dropping connection for connectionObject._identity = %s", co._identity)

            del self._clientChannels[channel]

            self._dropConnectionEntry(co)

    def _createConnectionEntry(self):
        identity = self.identityProducer.createIdentity()
        exists_key = keymapping.data_key(core_schema.Connection, identity, " exists")
        exists_index = keymapping.index_key(core_schema.Connection, " exists", True)
        identityRoot = self.allocateNewIdentityRoot()

        self._handleNewTransaction(
            None,
            {exists_key: serialize(bool, True).hex()},
            {exists_index: set([identity])},
            {},
            [],
            [],
            self._cur_transaction_num
        )

        return core_schema.Connection.fromIdentity(identity), identityRoot

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
                connectionObject, identityRoot = self._createConnectionEntry()

                connectedChannel = ConnectedChannel(
                    self._cur_transaction_num,
                    channel,
                    connectionObject,
                    identityRoot
                )

                self._clientChannels[channel] = connectedChannel

                channel.setClientToServerHandler(
                    lambda msg: self.onClientToServerMessage(connectedChannel, msg)
                )

                connectedChannel.sendInitializationMessage()
        except Exception:
            self._logger.error(
                "Failed during addConnection which should never happen:\n%s",
                traceback.format_exc()
            )

    def _handleSubscriptionInForeground(self, channel, msg):
        # first see if this would be an easy subscription to handle
        with Timer("Handle subscription in foreground: %s/%s/%s/isLazy=%s over %s",
                   msg.schema, msg.typename, msg.fieldname_and_value, msg.isLazy, lambda: len(identities)):
            typedef, identities = self._parseSubscriptionMsg(channel, msg)

            if not (msg.isLazy and len(identities) < self.MAX_LAZY_TO_SEND_SYNCHRONOUSLY or len(identities) < self.MAX_NORMAL_TO_SEND_SYNCHRONOUSLY):
                self._subscriptionQueue.put((channel, msg))
                return

            # handle this directly
            if msg.isLazy:
                self._completeLazySubscription(
                    msg.schema, msg.typename, msg.fieldname_and_value,
                    typedef,
                    identities,
                    channel
                )
                return

            self._sendPartialSubscription(
                channel,
                msg.schema,
                msg.typename,
                msg.fieldname_and_value,
                typedef,
                identities,
                set(identities),
                BATCH_SIZE=None,
                checkPending=False
            )

            self._markSubscriptionComplete(
                msg.schema,
                msg.typename,
                msg.fieldname_and_value,
                identities,
                channel,
                isLazy=False
            )

            channel.channel.write(
                ServerToClient.SubscriptionComplete(
                    schema=msg.schema,
                    typename=msg.typename,
                    fieldname_and_value=msg.fieldname_and_value,
                    tid=self._cur_transaction_num
                )
            )

    def _parseSubscriptionMsg(self, channel, msg):
        schema_name = msg.schema

        definition = channel.definedSchemas.get(schema_name)

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

        return typedef, identities

    def handleSubscriptionOnBackgroundThread(self, connectedChannel, msg):
        with Timer("Subscription requiring %s messages and produced %s objects for %s/%s/%s/isLazy=%s",
                   lambda: messageCount,
                   lambda: len(identities),
                   msg.schema,
                   msg.typename,
                   msg.fieldname_and_value,
                   msg.isLazy
                   ):
            try:
                with self._lock:
                    typedef, identities = self._parseSubscriptionMsg(connectedChannel, msg)

                    if connectedChannel.channel not in self._clientChannels:
                        self._logger.warn("Ignoring subscription from dead channel.")
                        return

                    if msg.isLazy:
                        assert msg.fieldname_and_value is None or msg.fieldname_and_value[0] != '_identity', 'makes no sense to lazily subscribe to specific values!'

                        messageCount = 1

                        self._completeLazySubscription(
                            msg.schema, msg.typename, msg.fieldname_and_value,
                            typedef,
                            identities,
                            connectedChannel
                        )
                        return True

                    self._pendingSubscriptionRecheck = []

                # we need to send everything we know about 'identities', keeping in mind that we have to
                # check any new identities that get written to in the background to see if they belong
                # in the new set
                identities_left_to_send = set(identities)

                messageCount = 0
                while True:
                    locktime_start = time.time()

                    if self._subscriptionBackgroundThreadCallback:
                        self._subscriptionBackgroundThreadCallback(messageCount)

                    with self._lock:
                        messageCount += 1
                        if messageCount == 2:
                            self._logger.info(
                                "Beginning large subscription for %s/%s/%s",
                                msg.schema, msg.typename, msg.fieldname_and_value
                            )

                        self._sendPartialSubscription(
                            connectedChannel,
                            msg.schema,
                            msg.typename,
                            msg.fieldname_and_value,
                            typedef,
                            identities,
                            identities_left_to_send
                        )

                        self._pendingSubscriptionRecheck = []

                        if not identities_left_to_send:
                            self._markSubscriptionComplete(
                                msg.schema,
                                msg.typename,
                                msg.fieldname_and_value,
                                identities,
                                connectedChannel,
                                isLazy=False
                            )

                            connectedChannel.channel.write(
                                ServerToClient.SubscriptionComplete(
                                    schema=msg.schema,
                                    typename=msg.typename,
                                    fieldname_and_value=msg.fieldname_and_value,
                                    tid=self._cur_transaction_num
                                )
                            )

                            break

                    # don't hold the lock more than 75% of the time.
                    time.sleep( (time.time() - locktime_start) / 3 )

                if self._subscriptionBackgroundThreadCallback:
                    self._subscriptionBackgroundThreadCallback("DONE")
            finally:
                with self._lock:
                    self._pendingSubscriptionRecheck = None

    def _completeLazySubscription(self,
                                  schema_name,
                                  typename,
                                  fieldname_and_value,
                                  typedef,
                                  identities,
                                  connectedChannel
                                  ):
        index_vals = self._buildIndexValueMap(typedef, schema_name, typename, identities)

        connectedChannel.channel.write(
            ServerToClient.LazySubscriptionData(
                schema=schema_name,
                typename=typename,
                fieldname_and_value=fieldname_and_value,
                identities=identities,
                index_values=index_vals
            )
        )

        # just send the identities
        self._markSubscriptionComplete(
            schema_name,
            typename,
            fieldname_and_value,
            identities,
            connectedChannel,
            isLazy=True
        )

        connectedChannel.channel.write(
            ServerToClient.SubscriptionComplete(
                schema=schema_name,
                typename=typename,
                fieldname_and_value=fieldname_and_value,
                tid=self._cur_transaction_num
            )
        )

    def _buildIndexValueMap(self, typedef, schema_name, typename, identities):
        # build a map from reverse-index-key to {identity}
        index_vals = {}

        for fieldname in typedef.indices:
            keys = [keymapping.data_reverse_index_key(schema_name, typename, identity, fieldname)
                    for identity in identities]

            vals = self._kvstore.getSeveral(keys)

            for i in range(len(keys)):
                index_vals[keys[i]] = vals[i]

        return index_vals

    def _markSubscriptionComplete(self, schema, typename, fieldname_and_value, identities, connectedChannel, isLazy):
        if fieldname_and_value is not None:
            # this is an index subscription
            for ident in identities:
                self._id_to_channel.setdefault(ident, set()).add(connectedChannel)

                connectedChannel.subscribedIds.add(ident)

            if fieldname_and_value[0] != '_identity':
                index_key = keymapping.index_key_from_names_encoded(schema, typename, fieldname_and_value[0], fieldname_and_value[1])

                self._index_to_channel.setdefault(index_key, set()).add(connectedChannel)

                connectedChannel.subscribedIndexKeys[index_key] = -1 if not isLazy else self._cur_transaction_num
            else:
                # an object's identity cannot change, so we don't need to track our subscription to it
                assert not isLazy
        else:
            # this is a type-subscription
            if (schema, typename) not in self._type_to_channel:
                self._type_to_channel[schema, typename] = set()

            self._type_to_channel[schema, typename].add(connectedChannel)

            connectedChannel.subscribedTypes[(schema, typename)] = -1 if not isLazy else self._cur_transaction_num

    def _sendPartialSubscription(self,
                                 connectedChannel,
                                 schema_name,
                                 typename,
                                 fieldname_and_value,
                                 typedef,
                                 identities,
                                 identities_left_to_send,
                                 BATCH_SIZE=100,
                                 checkPending=True):

        # get some objects to send
        kvs = {}
        index_vals = {}

        to_send = []
        if checkPending:
            for transactionMessage in self._pendingSubscriptionRecheck:
                for key in transactionMessage.writes:
                    transactionMessage.writes[key]

                    # if we write to a key we've already sent, we'll need to resend it
                    identity = keymapping.split_data_key(key)[2]
                    if identity in identities:
                        identities_left_to_send.add(identity)

                for add_index_key in transactionMessage.set_adds:
                    add_index_identities = transactionMessage.set_adds[add_index_key]

                    add_schema, add_typename, add_fieldname, add_hashVal = keymapping.split_index_key_full(add_index_key)

                    if add_schema == schema_name and add_typename == typename and (
                            fieldname_and_value is None and add_fieldname == " exists" or
                            fieldname_and_value is not None and tuple(fieldname_and_value) == (add_fieldname, add_hashVal)
                    ):
                        identities_left_to_send.update(add_index_identities)

        while identities_left_to_send and (BATCH_SIZE is None or len(to_send) < BATCH_SIZE):
            to_send.append(identities_left_to_send.pop())

        for fieldname in typedef.fields:
            keys = [keymapping.data_key_from_names(schema_name, typename, identity, fieldname)
                    for identity in to_send]

            vals = self._kvstore.getSeveral(keys)

            for i in range(len(keys)):
                kvs[keys[i]] = vals[i]

        index_vals = self._buildIndexValueMap(typedef, schema_name, typename, to_send)

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

        # Handle Authentication messages
        if msg.matches.Authenticate:
            if msg.token == self._auth_token:
                connectedChannel.authenticate()
            # else, do we need to do something?
            return

        # Abort if connection is not authenticated
        if connectedChannel.needsAuthentication:
            self._logger.info(
                "Received unexpected client message on unauthenticated channel %s",
                connectedChannel.connectionObject._identity
            )
            return

        # Handle remaining types of messages
        if msg.matches.Heartbeat:
            connectedChannel.heartbeat()
        elif msg.matches.LoadLazyObject:
            with self._lock:
                self._loadLazyObject(connectedChannel, msg)

            if self._lazyLoadCallback:
                self._lazyLoadCallback(msg.identity)

        elif msg.matches.Flush:
            with self._lock:
                connectedChannel.channel.write(ServerToClient.FlushResponse(guid=msg.guid))
        elif msg.matches.DefineSchema:
            assert isinstance(msg.definition, SchemaDefinition)
            connectedChannel.definedSchemas[msg.name] = msg.definition
        elif msg.matches.Subscribe:
            with self._lock:
                self._handleSubscriptionInForeground(connectedChannel, msg)
        elif msg.matches.TransactionData:
            connectedChannel.handleTransactionData(msg)
        elif msg.matches.CompleteTransaction:
            try:
                data = connectedChannel.extractTransactionData(msg.transaction_guid)

                with self._lock:
                    isOK, badKey = self._handleNewTransaction(
                        connectedChannel,
                        data['writes'],
                        data['set_adds'],
                        data['set_removes'],
                        data['key_versions'],
                        data['index_versions'],
                        msg.as_of_version
                    )
            except Exception:
                self._logger.error("Unknown error committing transaction: %s", traceback.format_exc())
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

    def _loadValuesForObject(self, channel, schema_name, typename, identities):
        typedef = channel.definedSchemas.get(schema_name)[typename]

        valsToGet = []
        for field_to_pull in typedef.fields:
            for ident in identities:
                valsToGet.append(keymapping.data_key_from_names(schema_name, typename, ident, field_to_pull))

        results = self._kvstore.getSeveral(valsToGet)

        return {valsToGet[i]: results[i] for i in range(len(valsToGet))}

    def _increaseBroadcastTransactionToInclude(self, channel, indexKey, newIds, key_value, set_adds, set_removes):
        # we need to include all the data for the objects in 'newIds' to the transaction
        # that we're broadcasting
        schema_name, typename, fieldname, fieldval = keymapping.split_index_key_full(indexKey)

        typedef = channel.definedSchemas.get(schema_name)[typename]

        key_value.update(self._loadValuesForObject(channel, schema_name, typename, newIds))

        reverseKeys = []
        for index_name in typedef.indices:
            for ident in newIds:
                reverseKeys.append(keymapping.data_reverse_index_key(schema_name, typename, ident, index_name))

        reverseVals = self._kvstore.getSeveral(reverseKeys)
        reverseKVMap = {reverseKeys[i]: reverseVals[i] for i in range(len(reverseKeys))}

        for index_name in typedef.indices:
            for ident in newIds:
                fieldval = reverseKVMap.get(keymapping.data_reverse_index_key(schema_name, typename, ident, index_name))

                if fieldval is not None:
                    ik = keymapping.index_key_from_names_encoded(schema_name, typename, index_name, fieldval)
                    set_adds.setdefault(ik, set()).add(ident)

    def _loadLazyObject(self, channel, msg):
        channel.channel.write(
            ServerToClient.LazyLoadResponse(
                identity=msg.identity,
                values=self._loadValuesForObject(channel, msg.schema, msg.typename, [msg.identity])
            )
        )

    def _garbage_collect(self, intervalOverride=None):
        """Cleanup anything in '_version_numbers' where we have deleted the entry
        and it's inactive for a long time."""
        interval = intervalOverride or self._gc_interval

        if self._last_garbage_collect_timestamp is None or time.time() - self._last_garbage_collect_timestamp > interval:
            threshold = time.time() - interval

            new_ts = {}
            for key, ts in self._version_numbers_timestamps.items():
                if ts < threshold:
                    if keymapping.isIndexKey(key):
                        if not self._kvstore.getSetMembers(key):
                            del self._version_numbers[key]
                    else:
                        if self._kvstore.get(key) is None:
                            del self._version_numbers[key]
                else:
                    new_ts[key] = ts

            self._version_numbers_timestamps = new_ts

            self._last_garbage_collect_timestamp = time.time()

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

        set_adds = {k: v for k, v in set_adds.items() if v}
        set_removes = {k: v for k, v in set_removes.items() if v}

        identities_mentioned = set()

        keysWritingTo = set()
        setsWritingTo = set()
        schemaTypePairsWriting = set()

        if sourceChannel:
            # check if we created any new objects to which we are not type-subscribed
            # and if so, ensure we are subscribed
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
            schemaTypePairsWriting.add((schema_name, typename))

            identities_mentioned.add(ident)

        for subset in [set_adds, set_removes]:
            for k in subset:
                if subset[k]:
                    schema_name, typename = keymapping.split_index_key(k)[:2]

                    schemaTypePairsWriting.add((schema_name, typename))

                    setsWritingTo.add(k)

                    identities_mentioned.update(subset[k])

        # check all version numbers for transaction conflicts.
        for subset in [keys_to_check_versions, indices_to_check_versions]:
            for key in subset:
                last_tid = self._version_numbers.get(key, -1)
                if as_of_version < last_tid:
                    return (False, key)

        t1 = time.time()

        for key in keysWritingTo:
            self._version_numbers[key] = transaction_id
            self._version_numbers_timestamps[key] = t1

        for key in setsWritingTo:
            self._version_numbers[key] = transaction_id
            self._version_numbers_timestamps[key] = t1

        priorValues = self._kvstore.getSeveralAsDictionary(key_value)

        # set the json representation in the database
        target_kvs = {k: v for k, v in key_value.items()}
        target_kvs.update(self.indexReverseLookupKvs(set_adds, set_removes))

        new_sets, dropped_sets = self._kvstore.setSeveral(target_kvs, set_adds, set_removes)

        # update the metadata index
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

        self._kvstore.setSeveral({}, indexSetAdds, indexSetRemoves)

        t2 = time.time()

        channelsTriggeredForPriors = set()

        # check any index-level subscriptions that are going to increase as a result of this
        # transaction and add the backing data to the relevant transaction.
        for index_key, adds in list(set_adds.items()):
            if index_key in self._index_to_channel:
                idsToAddToTransaction = set()

                for channel in self._index_to_channel.get(index_key):
                    if index_key in channel.subscribedIndexKeys and \
                            channel.subscribedIndexKeys[index_key] >= 0:
                        # this is a lazy subscription. We're not using the transaction ID yet because
                        # we don't store it on a per-object basis here. Instead, we're always sending
                        # everything twice to lazy subscribers.
                        channelsTriggeredForPriors.add(channel)

                    newIds = adds.difference(channel.subscribedIds)
                    for new_id in newIds:
                        self._id_to_channel.setdefault(new_id, set()).add(channel)
                        channel.subscribedIds.add(new_id)

                    self._broadcastSubscriptionIncrease(channel, index_key, newIds)

                    idsToAddToTransaction.update(newIds)

                if idsToAddToTransaction:
                    self._increaseBroadcastTransactionToInclude(
                        channel,  # deliberately just using whatever random channel, under
                                  # the assumption they're all the same. it would be better
                                  # to explictly compute the union of the relevant set of
                                  # defined fields, as its possible one channel has more fields
                                  # for a type than another and we'd like to broadcast them all
                        index_key, idsToAddToTransaction, key_value, set_adds, set_removes)

        transaction_message = None
        channelsTriggered = set()

        for schema_type_pair in schemaTypePairsWriting:
            for channel in self._type_to_channel.get(schema_type_pair, ()):
                if channel.subscribedTypes[schema_type_pair] >= 0:
                    # this is a lazy subscription. We're not using the transaction ID yet because
                    # we don't store it on a per-object basis here. Instead, we're always sending
                    # everything twice to lazy subscribers.
                    channelsTriggeredForPriors.add(channel)
                channelsTriggered.add(channel)

        for i in identities_mentioned:
            if i in self._id_to_channel:
                channelsTriggered.update(self._id_to_channel[i])

        for channel in channelsTriggeredForPriors:
            lazy_message = ServerToClient.LazyTransactionPriors(writes=priorValues)  # noqa

        transaction_message = ServerToClient.Transaction(
            writes={k: v for k, v in key_value.items()},
            set_adds=set_adds,
            set_removes=set_removes,
            transaction_id=transaction_id
        )

        if self._pendingSubscriptionRecheck is not None:
            self._pendingSubscriptionRecheck.append(transaction_message)

        for channel in channelsTriggered:
            channel.sendTransaction(transaction_message)

        if self.verbose or time.time() - t0 > self.longTransactionThreshold:
            self._logger.info(
                "Transaction [%.2f/%.2f/%.2f] with %s writes, %s set ops: %s",
                t1 - t0, t2 - t1, time.time() - t2,
                len(key_value), len(set_adds) + len(set_removes), sorted(key_value)[:3]
            )

        self._garbage_collect()

        return (True, None)
