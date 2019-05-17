#   Coyright 2017-2019 Nativepython Authors
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

from object_database.schema import ObjectFieldId, IndexId, FieldDefinition, indexValueFor
from object_database.messages import ClientToServer, getHeartbeatInterval
from object_database.core_schema import core_schema

from object_database.view import View, Transaction, _cur_view
from object_database.reactor import Reactor
from object_database.identity import IDENTITY_BLOCK_SIZE
from object_database._types import DatabaseConnectionState

from typed_python.SerializationContext import SerializationContext
from typed_python.Codebase import Codebase as TypedPythonCodebase
from typed_python import Alternative, Dict, OneOf

import threading
import logging
import traceback
import time

from object_database.view import DisconnectedException


class Everything:
    """Singleton to mark subscription to everything in a slice."""


TransactionResult = Alternative(
    "TransactionResult",
    Success={},
    RevisionConflict={'key': OneOf(str, ObjectFieldId, IndexId)},
    Disconnected={}
)


class DatabaseConnection:
    def __init__(self, channel, connectionMetadata=None):
        self._channel = channel
        self._transaction_callbacks = {}
        self._connectionMetadata = connectionMetadata or {}

        self._lock = threading.RLock()

        # transaction of what's in the KV store
        self._cur_transaction_num = 0

        self.serializationContext = TypedPythonCodebase.coreSerializationContext().withoutCompression()

        # a datastructure that keeps track of all the different versions of the objects
        # we have mapped in.
        self._connection_state = DatabaseConnectionState()
        self._connection_state.setSerializationContext(self.serializationContext)
        self._connection_state.setTriggerLazyLoad(self.loadLazyObject)

        self._lazy_object_read_blocks = {}

        self.initialized = threading.Event()
        self.disconnected = threading.Event()

        # for each schema name we've sent, an event that's triggered
        # when the server has acknowledged the schema and given us a definition
        self._schema_response_events = {}
        self._fields_to_field_ids = Dict(FieldDefinition, int)()
        self._field_id_to_schema_and_typename = {}
        self._field_id_to_field_def = Dict(int, FieldDefinition)()

        self.connectionObject = None

        # transaction handlers. These must be nonblocking since we call them under lock
        self._onTransactionHandlers = set()

        self._flushEvents = {}

        # set(schema)
        self._schemas = set()

        self._messages_received = 0

        self._pendingSubscriptions = {}

        # from (schema, typename, fieldname_and_val) -> {'values', 'index_values', 'identities'}
        # where (fieldname_and_val) is OneOf(None, (str, IndexValue))
        self._subscription_buildup = {}

        self._channel.setServerToClientHandler(self._onMessage)

        self._flushIx = 0

        self._largeSubscriptionHeartbeatDelay = 0

        self._logger = logging.getLogger(__name__)

        self._auth_token = None

        self._max_tid_by_schema = {}
        self._max_tid_by_schema_and_type = {}

    def getConnectionMetadata(self):
        """Return any data provided to us by the underlying transport.

        Returns:
            A dictionary of extra metadata.

            If we are a TCP-based connection, this will have the members:
                'peername': the remote address to which the socket is connected,
                            result of socket.socket.getpeername() (None on error)
                'socket':   socket.socket instance
                'sockname': the socket's own address, result of socket.socket.getsockname()
        """
        return self._connectionMetadata

    def registerOnTransactionHandler(self, handler):
        with self._lock:
            self._onTransactionHandlers.add(handler)

    def dropTransactionHandler(self, handler):
        with self._lock:
            self._onTransactionHandlers.discard(handler)

    def setSerializationContext(self, context):
        assert isinstance(context, SerializationContext), context
        self.serializationContext = context.withoutCompression()
        self._connection_state.setSerializationContext(self.serializationContext)
        return self

    def serializeFromModule(self, module):
        """Give the project root we want to serialize from."""
        self.setSerializationContext(
            TypedPythonCodebase.FromRootlevelModule(module).serializationContext
        )

    def currentTransactionId(self):
        return self._cur_transaction_num

    def currentTransactionIdForSchema(self, schema):
        return self._max_tid_by_schema.get(schema.name, 0)

    def currentTransactionIdForType(self, dbType):
        return self._max_tid_by_schema_and_type.get((dbType.__schema__.name, dbType.__qualname__), 0)

    def waitForTransactionId(self, tid):
        if tid > self._cur_transaction_num:
            e = threading.Event()

            def handler(*args):
                if args[-1] >= tid:
                    e.set()

            with self._lock:
                # check again, in case the transaction came in while we
                # were asleep
                if tid <= self._cur_transaction_num:
                    return

                self.registerOnTransactionHandler(handler)

            e.wait()

            self.dropTransactionHandler(handler)

    def _stopHeartbeating(self):
        self._channel._stopHeartbeating()

    def disconnect(self, block=False):
        self.disconnected.set()
        self._connection_state.setTriggerLazyLoad(None)
        self._channel.close(block=block)

    def _noViewsOutstanding(self):
        with self._lock:
            return self._connection_state.outstandingViewCount() == 0

    def authenticate(self, token):
        assert self._auth_token is None, "We already authenticated."
        self._auth_token = token

        self._channel.write(
            ClientToServer.Authenticate(token=token)
        )

    def addSchema(self, schema):
        schema.freeze()

        with self._lock:
            if schema not in self._schemas:
                self._schemas.add(schema)

                schemaDesc = schema.toDefinition()

                self._channel.write(
                    ClientToServer.DefineSchema(
                        name=schema.name,
                        definition=schemaDesc
                    )
                )

                self._schema_response_events[schema.name] = threading.Event()

            e = self._schema_response_events[schema.name]

        e.wait()

        if self.disconnected.is_set():
            raise DisconnectedException()

    def flush(self):
        """Make sure we know all transactions that have happened up to this point."""
        with self._lock:
            if self.disconnected.is_set():
                raise DisconnectedException()

            self._flushIx += 1
            ix = self._flushIx
            e = self._flushEvents[ix] = threading.Event()
            self._channel.write(ClientToServer.Flush(guid=ix))

        e.wait()

        if self.disconnected.is_set():
            raise DisconnectedException()

    def subscribeToObject(self, t):
        self.subscribeToObjects([t])

    def subscribeToNone(self, type):
        self.addSchema(type.__schema__)

    def subscribeToObjects(self, objects):
        for t in objects:
            self.addSchema(type(t).__schema__)

        self.subscribeMultiple([
            (type(t).__schema__.name, type(t).__qualname__,
                ("_identity", indexValueFor(type(t), t, self.serializationContext)),
                False)
            for t in objects
        ])

    def _lazinessForType(self, typeObj, desiredLaziness):
        if desiredLaziness is not None:
            return desiredLaziness
        if hasattr(typeObj, '__object_database_lazy_subscription__'):
            return True
        return False

    def subscribeToIndex(self, t, block=True, lazySubscription=None, **kwarg):
        self.addSchema(t.__schema__)

        toSubscribe = []

        for fieldname, fieldvalue in kwarg.items():
            indexVal = indexValueFor(
                t.__schema__.indexType(t.__qualname__, fieldname),
                fieldvalue,
                self.serializationContext
            )

            toSubscribe.append((
                t.__schema__.name,
                t.__qualname__,
                (fieldname, indexVal),
                self._lazinessForType(t, lazySubscription)
            ))

        return self.subscribeMultiple(toSubscribe, block=block)

    def subscribeToType(self, t, block=True, lazySubscription=None):
        self.addSchema(t.__schema__)

        if self._connection_state.typeSubscriptionLowestTransaction(t.__schema__.name, t.__qualname__) is not None:
            return ()

        return self.subscribeMultiple([(t.__schema__.name, t.__qualname__, None, self._lazinessForType(t, lazySubscription))], block)

    def subscribeToSchema(self, *schemas, block=True, lazySubscription=None, excluding=()):
        for s in schemas:
            self.addSchema(s)

        unsubscribedTypes = []
        for schema in schemas:
            for tname, t in schema._types.items():
                if not self.isSubscribedToType(t) and t not in excluding:
                    unsubscribedTypes.append((schema.name, tname, None, self._lazinessForType(t, lazySubscription)))

        if unsubscribedTypes:
            return self.subscribeMultiple(unsubscribedTypes, block=block)

        return ()

    def isSubscribedToSchema(self, schema):
        return all(self.isSubscribedToType(t) for t in schema._types.values())

    def isSubscribedToType(self, t):
        return self._connection_state.typeSubscriptionLowestTransaction(t.__schema__.name, t.__qualname__) is not None

    def subscribeMultiple(self, subscriptionTuples, block=True):
        with self._lock:
            if self.disconnected.is_set():
                raise DisconnectedException()

            events = []

            for tup in subscriptionTuples:
                e = self._pendingSubscriptions.get(tup)

                if not e:
                    e = self._pendingSubscriptions[(tup[0], tup[1], tup[2])] = threading.Event()

                assert tup[0] and tup[1]

                self._channel.write(
                    ClientToServer.Subscribe(schema=tup[0], typename=tup[1], fieldname_and_value=tup[2], isLazy=tup[3])
                )

                events.append(e)

        if not block:
            return tuple(events)

        for e in events:
            e.wait()

        with self._lock:
            if self.disconnected.is_set():
                raise DisconnectedException()

        return ()

    def waitForCondition(self, cond, timeout):
        # eventally we will replace this with something that watches the calculation
        try:
            def checkCondition():
                with self.view():
                    return cond()

            reactor = Reactor(self, checkCondition)

            return reactor.blockUntilTrue(timeout)
        finally:
            reactor.teardown()

    def __str__(self):
        return "DatabaseConnection(%s)" % id(self)

    def __repr__(self):
        return "DatabaseConnection(%s)" % id(self)

    def current_transaction(self):
        if not hasattr(_cur_view, "view"):
            return None
        return _cur_view.view

    def view(self, transaction_id=None):
        with self._lock:
            if self.disconnected.is_set():
                raise DisconnectedException()

            if transaction_id is None:
                transaction_id = self._cur_transaction_num

            assert transaction_id <= self._cur_transaction_num

            return View(self, transaction_id)

    def transaction(self):
        """Only one transaction may be committed on the current transaction number."""
        with self._lock:
            if self.disconnected.is_set():
                raise DisconnectedException()

            return Transaction(self, self._cur_transaction_num)

    def _onMessage(self, msg):
        self._messages_received += 1

        if msg.matches.Disconnected:
            with self._lock:
                self.disconnected.set()
                self.connectionObject = None

                for e in self._lazy_object_read_blocks.values():
                    e.set()

                for e in self._flushEvents.values():
                    e.set()

                for e in self._pendingSubscriptions.values():
                    e.set()

                for e in self._schema_response_events.values():
                    e.set()

                for q in self._transaction_callbacks.values():
                    try:
                        q(TransactionResult.Disconnected())
                    except Exception:
                        self._logger.error(
                            "Transaction commit callback threw an exception:\n%s",
                            traceback.format_exc()
                        )

                self._transaction_callbacks = {}
                self._flushEvents = {}
        elif msg.matches.FlushResponse:
            with self._lock:
                e = self._flushEvents.get(msg.guid)
                if not e:
                    self._logger.error("Got an unrequested flush response: %s", msg.guid)
                else:
                    e.set()
        elif msg.matches.Initialize:
            with self._lock:
                self._cur_transaction_num = msg.transaction_num
                self._connection_state.setIdentityRoot(IDENTITY_BLOCK_SIZE * msg.identity_root)
                self.connectionObject = core_schema.Connection.fromIdentity(msg.connIdentity)
                self.initialized.set()
        elif msg.matches.TransactionResult:
            with self._lock:
                try:
                    self._transaction_callbacks.pop(msg.transaction_guid)(
                        TransactionResult.Success() if msg.success else
                        TransactionResult.RevisionConflict(key=msg.badKey)
                    )
                except Exception:
                    self._logger.error(
                        "Transaction commit callback threw an exception:\n%s",
                        traceback.format_exc()
                    )
        elif msg.matches.Transaction:
            with self._lock:
                self._markSchemaAndTypeMaxTids(set(k.fieldId for k in msg.writes), msg.transaction_id)

                self._connection_state.incomingTransaction(
                    msg.transaction_id,
                    msg.writes,
                    msg.set_adds,
                    msg.set_removes
                )

                self._cur_transaction_num = msg.transaction_id

            for handler in list(self._onTransactionHandlers):
                try:
                    handler(msg.writes, msg.set_adds, msg.set_removes, msg.transaction_id)
                except Exception:
                    self._logger.error(
                        "_onTransaction handler %s threw an exception:\n%s",
                        handler,
                        traceback.format_exc()
                    )

        elif msg.matches.SchemaMapping:
            with self._lock:
                for fieldDef, fieldId in msg.mapping.items():
                    self._field_id_to_field_def[fieldId] = fieldDef
                    self._fields_to_field_ids[fieldDef] = fieldId

                    self._field_id_to_schema_and_typename[fieldId] = (fieldDef.schema, fieldDef.fieldname)

                    self._connection_state.setFieldId(
                        fieldDef.schema,
                        fieldDef.typename,
                        fieldDef.fieldname,
                        fieldId
                    )

                self._schema_response_events[msg.schema].set()

        elif msg.matches.SubscriptionIncrease:
            with self._lock:
                for oid in msg.identities:
                    self._connection_state.markObjectSubscribed(oid, msg.transaction_id)

        elif msg.matches.SubscriptionData:
            with self._lock:
                lookupTuple = (msg.schema, msg.typename, msg.fieldname_and_value)

                if lookupTuple not in self._subscription_buildup:
                    self._subscription_buildup[lookupTuple] = {'values': {}, 'index_values': {}, 'identities': None, 'markedLazy': False}
                else:
                    assert not self._subscription_buildup[lookupTuple]['markedLazy'], 'received non-lazy data for a lazy subscription'

                self._subscription_buildup[lookupTuple]['values'].update({k: msg.values[k] for k in msg.values})
                self._subscription_buildup[lookupTuple]['index_values'].update({k: msg.index_values[k] for k in msg.index_values})

                if msg.identities is not None:
                    if self._subscription_buildup[lookupTuple]['identities'] is None:
                        self._subscription_buildup[lookupTuple]['identities'] = set()
                    self._subscription_buildup[lookupTuple]['identities'].update(msg.identities)
        elif msg.matches.LazyTransactionPriors:
            with self._lock:
                self._connection_state.incomingTransaction( 0, msg.writes, {}, {})

        elif msg.matches.LazyLoadResponse:
            with self._lock:
                self._connection_state.incomingTransaction(0, msg.values, {}, {})

                self._connection_state.markObjectNotLazy(msg.identity)

                e = self._lazy_object_read_blocks.pop(msg.identity, None)

                if e:
                    e.set()

        elif msg.matches.LazySubscriptionData:
            with self._lock:
                lookupTuple = (msg.schema, msg.typename, msg.fieldname_and_value)

                assert lookupTuple not in self._subscription_buildup

                self._subscription_buildup[lookupTuple] = {
                    'values': {},
                    'index_values': msg.index_values,
                    'identities': msg.identities,
                    'markedLazy': True
                }

        elif msg.matches.SubscriptionComplete:
            with self._lock:
                event = self._pendingSubscriptions.get((
                    msg.schema,
                    msg.typename,
                    tuple(msg.fieldname_and_value) if msg.fieldname_and_value is not None else None
                ))

                if not event:
                    self._logger.error(
                        "Received unrequested subscription to schema %s / %s / %s. have %s",
                        msg.schema, msg.typename, msg.fieldname_and_value, self._pendingSubscriptions
                    )
                    return

                lookupTuple = (msg.schema, msg.typename, msg.fieldname_and_value)

                identities = self._subscription_buildup[lookupTuple]['identities']
                values = self._subscription_buildup[lookupTuple]['values']
                index_values = self._subscription_buildup[lookupTuple]['index_values']
                markedLazy = self._subscription_buildup[lookupTuple]['markedLazy']
                del self._subscription_buildup[lookupTuple]

                self._markSchemaAndTypeMaxTids(set(v.fieldId for v in values), msg.tid)

                sets = self._indexValuesToSetAdds(index_values)

                if msg.fieldname_and_value is None:
                    if msg.typename is None:
                        for typename in self._schemaToType[msg.schema]:
                            self._connection_state.markTypeSubscribed(msg.schema, typename, msg.tid)
                    else:
                        self._connection_state.markTypeSubscribed(msg.schema, msg.typename, msg.tid)
                else:
                    assert msg.typename is not None
                    for oid in identities:
                        self._connection_state.markObjectSubscribed(oid, msg.tid)

                heartbeatInterval = getHeartbeatInterval()

                # this is a fault injection to allow us to verify that heartbeating during this
                # function will keep the server connection alive.
                for _ in range(self._largeSubscriptionHeartbeatDelay):
                    self._channel.sendMessage(
                        ClientToServer.Heartbeat()
                    )
                    time.sleep(heartbeatInterval)

                if markedLazy:
                    schema, typename = lookupTuple[:2]

                    for i in identities:
                        self._connection_state.markObjectLazy(schema, typename, i)

                self._connection_state.incomingTransaction(msg.tid, values, sets, {})

                # this should be inline with the stream of messages coming from the server
                assert self._cur_transaction_num <= msg.tid

                self._cur_transaction_num = msg.tid

                event.set()
        else:
            assert False, "unknown message type " + msg._which

    def _markSchemaAndTypeMaxTids(self, fieldIds, tid):
        for fieldId in fieldIds:
            fieldDef = self._field_id_to_field_def.get(fieldId)
            if fieldDef is not None:
                existing = self._max_tid_by_schema.get(fieldDef.schema, 0)
                if existing < tid:
                    self._max_tid_by_schema[fieldDef.schema] = tid

                existing = self._max_tid_by_schema_and_type.get((fieldDef.schema, fieldDef.typename), 0)
                if existing < tid:
                    self._max_tid_by_schema_and_type[fieldDef.schema, fieldDef.typename] = tid

    def _indexValuesToSetAdds(self, indexValues):
        # indexValues contains (schema:typename:identity:fieldname -> indexHashVal) which builds
        # up the indices we need. We need to transpose to a dictionary ordered by the hash values,
        # not the identities

        t0 = time.time()
        heartbeatInterval = getHeartbeatInterval()

        setAdds = {}

        for iv in indexValues:
            val = indexValues[iv]

            if val is not None:
                fieldId = iv.fieldId
                identity = iv.objId

                index_key = IndexId(fieldId=fieldId, indexValue=val)

                setAdds.setdefault(index_key, set()).add(identity)

                # this could take a long time, so we need to keep heartbeating
                if time.time() - t0 > heartbeatInterval:
                    # note that this needs to be 'sendMessage' which sends immediately,
                    # not, 'write' which queues the message after this function finishes!
                    self._channel.sendMessage(
                        ClientToServer.Heartbeat()
                    )
                    t0 = time.time()
        return setAdds

    def requestLazyObjects(self, objects):
        with self._lock:
            for o in objects:
                self._loadLazyObject(o._identity, type(o).__schema__.name, type(o).__qualname__)

    def loadLazyObject(self, identity, schemaName, typeName):
        with self._lock:
            e = self._loadLazyObject(identity, schemaName, typeName)

        e.wait()

    def _loadLazyObject(self, identity, schemaName, typeName):
        e = self._lazy_object_read_blocks.get(identity)

        if e:
            return e

        e = self._lazy_object_read_blocks[identity] = threading.Event()

        self._channel.write(
            ClientToServer.LoadLazyObject(
                identity=identity,
                schema=schemaName,
                typename=typeName
            )
        )

        return e

    def _createTransaction(self,
                           key_value,
                           set_adds,
                           set_removes,
                           keys_to_check_versions,
                           indices_to_check_versions,
                           as_of_version,
                           confirmCallback
                           ):
        assert confirmCallback is not None

        transaction_guid = self._connection_state.allocateIdentity()

        self._transaction_callbacks[transaction_guid] = confirmCallback

        out_writes = {}

        for k, v in key_value.items():
            out_writes[k] = v
            if len(out_writes) > 10000:
                self._channel.write(
                    ClientToServer.TransactionData(
                        writes=out_writes, set_adds={}, set_removes={},
                        key_versions=(), index_versions=(),
                        transaction_guid=transaction_guid
                    )
                )
                self._channel.write(ClientToServer.Heartbeat())
                out_writes = {}

        ct = 0
        out_set_adds = {}
        for k, v in set_adds.items():
            out_set_adds[k] = tuple(v)
            ct += len(v)

            if len(out_set_adds) > 10000 or ct > 100000:
                self._channel.write(
                    ClientToServer.TransactionData(
                        writes={}, set_adds=out_set_adds, set_removes={},
                        key_versions=(), index_versions=(),
                        transaction_guid=transaction_guid
                    )
                )
                self._channel.write(ClientToServer.Heartbeat())
                out_set_adds = {}
                ct = 0

        ct = 0
        out_set_removes = {}
        for k, v in set_removes.items():
            out_set_removes[k] = tuple(v)
            ct += len(v)

            if len(out_set_removes) > 10000 or ct > 100000:
                self._channel.write(
                    ClientToServer.TransactionData(
                        writes={}, set_adds={}, set_removes=out_set_removes,
                        key_versions=(), index_versions=(),
                        transaction_guid=transaction_guid
                    )
                )
                self._channel.write(ClientToServer.Heartbeat())
                out_set_removes = {}
                ct = 0

        keys_to_check_versions = list(keys_to_check_versions)
        while len(keys_to_check_versions) > 10000:
            self._channel.write(
                ClientToServer.TransactionData(
                    writes={}, set_adds={}, set_removes={},
                    key_versions=keys_to_check_versions[:10000],
                    index_versions=(), transaction_guid=transaction_guid
                )
            )
            self._channel.write(ClientToServer.Heartbeat())
            keys_to_check_versions = keys_to_check_versions[10000:]

        indices_to_check_versions = list(indices_to_check_versions)
        while len(indices_to_check_versions) > 10000:
            self._channel.write(
                ClientToServer.TransactionData(
                    writes={}, set_adds={}, set_removes={},
                    key_versions=(), index_versions=indices_to_check_versions[:10000],
                    transaction_guid=transaction_guid)
            )
            indices_to_check_versions = indices_to_check_versions[10000:]

        self._channel.write(
            ClientToServer.TransactionData(
                writes=out_writes,
                set_adds=out_set_adds,
                set_removes=out_set_removes,
                key_versions=keys_to_check_versions,
                index_versions=indices_to_check_versions,
                transaction_guid=transaction_guid
            )
        )

        self._channel.write(
            ClientToServer.CompleteTransaction(
                as_of_version=as_of_version,
                transaction_guid=transaction_guid
            )
        )
