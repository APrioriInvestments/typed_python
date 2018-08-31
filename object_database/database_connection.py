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

from object_database.schema import Schema
from object_database.messages import ClientToServer, ServerToClient, SchemaDefinition, getHeartbeatInterval
from object_database.core_schema import core_schema
from object_database.view import View, JsonWithPyRep, Transaction, _cur_view
import object_database.keymapping as keymapping
import json
from typed_python.hash import sha_hash
from typed_python import *

import queue
import threading
import logging
import uuid
import traceback
import time

from object_database.view import RevisionConflictException, DisconnectedException

class Everything:
    """Singleton to mark subscription to everything in a slice."""

TransactionResult = Alternative(
    "TransactionResult", 
    Success = {},
    RevisionConflict = {'key': str},
    Disconnected = {}
    )

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

class TransactionListener:
    def __init__(self, db, handler):
        self._thread = threading.Thread(target=self._doWork)
        self._thread.daemon = True
        self._shouldStop = False
        self._db = db
        self._db._onTransaction.append(self._onTransaction)
        self._queue = queue.Queue()

        self.handler = handler

    def start(self):
        self._thread.start()
        
    def stop(self):
        self._shouldStop = True
        self._thread.join()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.stop()

    def flush(self):
        while self._queue.qsize():
            time.sleep(0.001)

    def _doWork(self):
        while not self._shouldStop:
            try:
                todo = self._queue.get(timeout=0.1)
            except queue.Empty:
                todo = None

            if todo:
                try:
                    self.handler(todo)
                except:
                    logging.error("Callback threw exception:\n%s", traceback.format_exc())


    def _onTransaction(self, key_value, priors, set_adds, set_removes, tid):
        changed = {}

        for k in key_value:
            o, fieldname = self._db._data_key_to_object(k)

            if o:
                if o not in changed:
                    changed[o] = []

                if fieldname != " exists":
                    changed[o].append((
                        fieldname,
                        View.unwrapJsonWithPyRep(key_value[k], o.__types__[fieldname]),
                        View.unwrapJsonWithPyRep(priors[k], o.__types__[fieldname])
                        ))

        self._queue.put(changed)

class DatabaseConnection:
    def __init__(self, channel):
        self._channel = channel
        self._transaction_callbacks = {}
        
        self._lock = threading.Lock()

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

        self.initialized = threading.Event()
        self.disconnected = threading.Event()

        self.connectionObject = None

        #transaction handlers. These must be nonblocking since we call them under lock
        self._onTransaction = []

        self._flushEvents = {}

        self._schemas = {}

        self._messages_received = 0

        self._pendingSubscriptions = {}

        #if we have object-level subscriptions to a particular type (e.g. not everything)
        #then, this is from (schema, typename) -> {object_id -> transaction_id} so that
        #we can tell when the subscription should become valid. Subscriptions are permanent
        #otherwise, if we're subscribed, it's 'Everything'
        self._schema_and_typename_to_subscription_set = {}

        #from (schema,typename,field_val) -> {'values', 'index_values', 'identities'}
        self._subscription_buildup = {}

        self._channel.setServerToClientHandler(self._onMessage)

        self._flushIx = 0

        self._largeSubscriptionHeartbeatDelay = 0



    def _stopHeartbeating(self):
        self._channel._stopHeartbeating()

    def addSchema(self, schema):
        schema.freeze()

        with self._lock:
            assert schema.name not in self._schemas or self._schemas[schema.name] is schema

            self._schemas[schema.name] = schema

            schemaDesc = schema.toDefinition()

            self._channel.write(
                ClientToServer.DefineSchema(
                    name=schema.name,
                    definition=schemaDesc
                    )
                )

    def flush(self):
        """Make sure we know all transactions that have happened up to this point."""
        with self._lock:
            if self.disconnected.is_set():
                raise DisconnectedException()

            self._flushIx += 1
            ix = str(self._flushIx)
            e = self._flushEvents[ix] = threading.Event()
            self._channel.write(ClientToServer.Flush(guid=ix))

        e.wait()

        if self.disconnected.is_set():
            raise DisconnectedException()

    def subscribeToObject(self, t):
        self.subscribeToObjects([t])

    def subscribeToObjects(self, objects):
        for t in objects:
            self.addSchema(type(t).__schema__)
        self.subscribeMultiple([
            (type(t).__schema__.name, type(t).__qualname__, ("_identity", t._identity))
            ])

    def subscribeToIndex(self, t, **kwarg):
        self.addSchema(t.__schema__)

        toSubscribe = []
        for fieldname,fieldvalue in kwarg.items():
            toSubscribe.append((t.__schema__.name, t.__qualname__, (fieldname, keymapping.index_value_to_hash(fieldvalue))))

        self.subscribeMultiple(toSubscribe)

    def subscribeToType(self, t):
        self.addSchema(t.__schema__)
        if self._isTypeSubscribedAll(t):
            return
        self.subscribeMultiple([(t.__schema__.name, t.__qualname__, None)])

    def subscribeToNone(self, t):
        self.addSchema(t.__schema__)
        with self._lock:
            self._schema_and_typename_to_subscription_set.setdefault((t.__schema__.name, t.__qualname__), set())

    def subscribeToSchema(self, *schemas):
        for s in schemas:
            self.addSchema(s)

        self.subscribeMultiple([(schema.name, tname, None) for schema in schemas for tname in schema._types])

    def _isTypeSubscribed(self, t):
        return (t.__schema__.name, t.__qualname__) in self._schema_and_typename_to_subscription_set

    def _isTypeSubscribedAll(self, t):
        return self._schema_and_typename_to_subscription_set.get((t.__schema__.name, t.__qualname__)) is Everything

    def subscribeMultiple(self, subscriptionTuples):
        with self._lock:
            if self.disconnected.is_set():
                raise DisconnectedException()

            events = []

            for tup in subscriptionTuples:
                e = self._pendingSubscriptions.get(tup)

                if not e:
                    e = self._pendingSubscriptions[tup] = threading.Event()

                assert tup[0] and tup[1]
                self._channel.write(
                    ClientToServer.Subscribe(schema=tup[0], typename=tup[1], fieldname_and_value=tup[2])
                    )

                events.append(e)

        for e in events:
            e.wait()

        with self._lock:
            if self.disconnected.is_set():
                raise DisconnectedException()

    def waitForCondition(self, cond, timeout):
        #eventally we will replace this with something that watches the calculation
        t0 = time.time()
        while time.time() - t0 < timeout:
            with self.view():
                try:
                    if cond():
                        return True
                except:
                    logging.error("Condition callback threw an exception:\n%s", traceback.format_exc())

                time.sleep(timeout / 20)
        return False

    def _data_key_to_object(self, key):
        schema_name, typename, identity, fieldname = keymapping.split_data_key(key)
        
        schema = self._schemas.get(schema_name)
        if not schema:
            return None,None

        cls = schema._types.get(typename)
        
        if cls:
            return cls.fromIdentity(identity), fieldname

        return None,None

    def clearCache(self):
        with self._lock:
            self._versioned_objects = {k:v for k,v in self._versioned_objects.items() if not v.isEmpty()}

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
            assert transaction_id >= self._min_transaction_num, transaction_id

            view = View(self, transaction_id)

            self._incversion(transaction_id)

            return view

    def _incversion(self, transaction_id):
        if transaction_id not in self._version_number_counts:
            self._version_number_counts[transaction_id] = 1
            if self._min_reffed_version_number is None:
                self._min_reffed_version_number = transaction_id
            else:
                self._min_reffed_version_number = min(transaction_id, self._min_reffed_version_number)
        else:
            self._version_number_counts[transaction_id] += 1

    def _decversion(self, transaction_id):
        assert transaction_id in self._version_number_counts

        self._version_number_counts[transaction_id] -= 1

        assert self._version_number_counts[transaction_id] >= 0

        if self._version_number_counts[transaction_id] == 0:
            del self._version_number_counts[transaction_id]

            if transaction_id == self._min_reffed_version_number:
                if not self._version_number_counts:
                    self._min_reffed_version_number = None
                else:
                    self._min_reffed_version_number = min(self._version_number_counts)

    def transaction(self):
        """Only one transaction may be committed on the current transaction number."""
        with self._lock:
            if self.disconnected.is_set():
                raise DisconnectedException()

            view = Transaction(self, self._cur_transaction_num)

            transaction_id = self._cur_transaction_num

            self._incversion(transaction_id)

            return view

    def _releaseView(self, view):
        transaction_id = view._transaction_num

        with self._lock:
            self._decversion(transaction_id)

            self._cleanup()

    def _cleanup(self):
        """Get rid of old objects we don't need to keep around and increase the min_transaction_id"""
        while True:
            if not self._version_numbers:
                #nothing to cleanup because we have no transactions
                return

            #this is the lowest write we have in the in-mem database
            lowest = self._version_numbers[0]

            if self._min_reffed_version_number is not None and self._min_reffed_version_number < lowest:
                #some transactions still refer to values before this version
                return

            self._version_numbers.pop(0)

            keys_touched = self._version_number_objects[lowest]
            del self._version_number_objects[lowest]

            self._min_transaction_num = lowest

            for key in keys_touched:
                self._versioned_objects[key].cleanup(lowest)

    def isSubscribedToObject(self, object):
        return not self._suppressKey(object._identity)

    def _suppressKey(self, k):
        keyname = schema, typename, ident, fieldname = keymapping.split_data_key(k)
        
        subscriptionSet = self._schema_and_typename_to_subscription_set.get((schema,typename))
        
        if subscriptionSet is Everything:
            return False
        if isinstance(subscriptionSet, set) and ident in subscriptionSet:
            return False
        return True

    def _suppressIdentities(self, index_key, identities):
        schema, typename, fieldname, valhash = keymapping.split_index_key_full(index_key)

        subscriptionSet = self._schema_and_typename_to_subscription_set.get((schema,typename))
        
        if subscriptionSet is Everything:
            return identities
        elif subscriptionSet is None:
            return set()
        else:
            return identities.intersection(subscriptionSet)


    def _onMessage(self, msg):
        with self._lock:
            self._messages_received += 1

            if msg.matches.Disconnected:
                self.disconnected.set()
                self.connectionObject = None
                
                for e in self._flushEvents.values():
                    e.set()

                for e in self._pendingSubscriptions.values():
                    e.set()

                for q in self._transaction_callbacks.values():
                    try:
                        q(TransactionResult.Disconnected())
                    except:
                        logging.error(
                            "Transaction commit callback threw an exception:\n%s", 
                            traceback.format_exc()
                            )

                self._transaction_callbacks = {}
                self._flushEvents = {} 
            elif msg.matches.FlushResponse:
                e = self._flushEvents.get(msg.guid)
                if not e:
                    logging.error("Got an unrequested flush response: %s", msg.guid)
                else:
                    e.set()
            elif msg.matches.Initialize:
                self._min_transaction_num = self._cur_transaction_num = msg.transaction_num
                self.connectionObject = core_schema.Connection.fromIdentity(msg.connIdentity)
                self.initialized.set()
            elif msg.matches.TransactionResult:
                try:
                    self._transaction_callbacks.pop(msg.transaction_guid)(
                        TransactionResult.Success() if msg.success 
                            else TransactionResult.RevisionConflict(key=msg.badKey)
                        )
                except:
                    logging.error(
                        "Transaction commit callback threw an exception:\n%s", 
                        traceback.format_exc()
                        )
            elif msg.matches.Transaction:
                key_value = {}
                priors = {}

                writes = dict(msg.writes)
                set_adds = dict(msg.set_adds)
                set_removes = dict(msg.set_removes)

                for k,val_serialized in writes.items():
                    if not self._suppressKey(k):
                        json_val = json.loads(val_serialized) if val_serialized is not None else None

                        key_value[k] = json_val

                        if k not in self._versioned_objects:
                            self._versioned_objects[k] = VersionedValue(JsonWithPyRep(None, None))

                        versioned = self._versioned_objects[k]

                        priors[k] = versioned.newestValue()

                        versioned.setVersionedValue(msg.transaction_id, JsonWithPyRep(json_val, None))

                for k,a in set_adds.items():
                    if k not in self._versioned_objects:
                        self._versioned_objects[k] = VersionedSet(set())
                    a = self._suppressIdentities(k, set(a))
                    if a:
                        self._versioned_objects[k].setVersionedAddsAndRemoves(msg.transaction_id, a, set())

                for k,r in set_removes.items():
                    if k not in self._versioned_objects:
                        self._versioned_objects[k] = VersionedSet(set())
                    r = self._suppressIdentities(k, set(r))
                    if r:
                        self._versioned_objects[k].setVersionedAddsAndRemoves(msg.transaction_id, set(), r)

                self._cur_transaction_num = msg.transaction_id

                for handler in self._onTransaction:
                    try:
                        handler(key_value, priors, set_adds, set_removes, msg.transaction_id)
                    except:
                        logging.error(
                            "_onTransaction callback %s threw an exception:\n%s", 
                            handler, 
                            traceback.format_exc()
                            )

                self._cleanup()
            elif msg.matches.SubscriptionIncrease:
                subscribedIdentities = self._schema_and_typename_to_subscription_set.setdefault((msg.schema, msg.typename), set())
                if subscribedIdentities is not Everything:
                    subscribedIdentities.update(
                        msg.identities
                        )
            elif msg.matches.SubscriptionData:
                lookupTuple = (msg.schema, msg.typename, msg.fieldname_and_value)

                if lookupTuple not in self._subscription_buildup:
                    self._subscription_buildup[lookupTuple] = {'values': {}, 'index_values': {}, 'identities': None}

                self._subscription_buildup[lookupTuple]['values'].update(msg.values)
                self._subscription_buildup[lookupTuple]['index_values'].update(msg.index_values)

                if msg.identities is not None:
                    if self._subscription_buildup[lookupTuple]['identities'] is None:
                        self._subscription_buildup[lookupTuple]['identities'] = set()
                    self._subscription_buildup[lookupTuple]['identities'].update(msg.identities)

            elif msg.matches.SubscriptionComplete:
                event = self._pendingSubscriptions.get((msg.schema, msg.typename, msg.fieldname_and_value))

                if not event:
                    logging.error("Received unrequested subscription to schema %s / %s / %s", msg.schema, msg.typename, msg.fieldname_and_value)
                    return

                lookupTuple = (msg.schema, msg.typename, msg.fieldname_and_value)

                identities = self._subscription_buildup[lookupTuple]['identities']
                values = self._subscription_buildup[lookupTuple]['values']
                index_values = self._subscription_buildup[lookupTuple]['index_values']

                sets = self.indexValuesToSetAdds(index_values)

                del self._subscription_buildup[lookupTuple]

                if msg.fieldname_and_value is None:
                    if msg.typename is None:
                        for tname in self._schemas[msg.schema]._types:
                            self._schema_and_typename_to_subscription_set[msg.schema, tname] = Everything
                    else:
                        self._schema_and_typename_to_subscription_set[msg.schema, msg.typename] = Everything
                else:
                    assert msg.typename is not None
                    subscribedIdentities = self._schema_and_typename_to_subscription_set.setdefault((msg.schema, msg.typename), set())
                    if subscribedIdentities is not Everything:
                        subscribedIdentities.update(
                            identities
                            )

                t0 = time.time()
                heartbeatInterval = getHeartbeatInterval()

                #this is a fault injection to allow us to verify that heartbeating during this
                #function will keep the server connection alive.
                for _ in range(self._largeSubscriptionHeartbeatDelay):
                    #print("Waiting for %s seconds because of _largeSubscriptionHeartbeatDelay" % heartbeatInterval)
                    self._channel.sendMessage(
                        ClientToServer.Heartbeat()
                        )
                    time.sleep(heartbeatInterval)

                for key, val in values.items():
                    if key not in self._versioned_objects:
                        self._versioned_objects[key] = VersionedValue(JsonWithPyRep(None, None))
                        self._versioned_objects[key].setVersionedValue(
                            msg.tid,
                            JsonWithPyRep(
                                json.loads(val) if val is not None else None,
                                None
                                )
                            )
                    #this could take a long time, so we need to keep heartbeating
                    if time.time() - t0 > heartbeatInterval:
                        #note that this needs to be 'sendMessage' which sends immediately,
                        #not, 'write' which queues the message after this function finishes!
                        self._channel.sendMessage(
                            ClientToServer.Heartbeat()
                            )
                        t0 = time.time()

                for key, setval in sets.items():
                    if key not in self._versioned_objects:
                        self._versioned_objects[key] = VersionedSet(set())
                        self._versioned_objects[key].setVersionedAddsAndRemoves(msg.tid, set(setval), set())

                    #this could take a long time, so we need to keep heartbeating
                    if time.time() - t0 > heartbeatInterval:
                        #note that this needs to be 'sendMessage' which sends immediately,
                        #not, 'write' which queues the message after this function finishes!
                        self._channel.sendMessage(
                            ClientToServer.Heartbeat()
                            )
                        t0 = time.time()

                #this should be inline with the stream of messages coming from the server
                assert self._cur_transaction_num <= msg.tid

                self._cur_transaction_num = msg.tid

                event.set()
            else:
                assert False, "unknown message type " + msg._which
    
    def indexValuesToSetAdds(self, indexValues):
        #indexValues contains (schema:typename:identity:fieldname -> indexHashVal) which builds 
        #up the indices we need. We need to transpose to a dictionary ordered by the hash values,
        #not the identities

        t0 = time.time()
        heartbeatInterval = getHeartbeatInterval()

        setAdds = {}

        for iv, val in indexValues.items():
            schema_name, typename, identity, field_name = keymapping.split_data_reverse_index_key(iv)

            index_key = keymapping.index_key_from_names_encoded(schema_name, typename, field_name, val)

            setAdds.setdefault(index_key, set()).add(identity)

            #this could take a long time, so we need to keep heartbeating
            if time.time() - t0 > heartbeatInterval:
                #note that this needs to be 'sendMessage' which sends immediately,
                #not, 'write' which queues the message after this function finishes!
                self._channel.sendMessage(
                    ClientToServer.Heartbeat()
                    )
                t0 = time.time()
        return setAdds

    def _get_versioned_set_data(self, key, transaction_id):
        assert transaction_id >= self._min_transaction_num, (transaction_id, self._min_transaction_num)

        with self._lock:
            if key in self._versioned_objects:
                return self._versioned_objects[key].valueForVersion(transaction_id)
            
            if self.disconnected.is_set():
                raise DisconnectedException()

            return SetWithEdits(set(),set(),set())

    def _get_versioned_object_data(self, key, transaction_id):
        assert transaction_id >= self._min_transaction_num

        with self._lock:
            if key in self._versioned_objects:
                return self._versioned_objects[key].valueForVersion(transaction_id)

            if self.disconnected.is_set():
                raise DisconnectedException()

            return None

    def _set_versioned_object_data(self, 
                key_value, 
                set_adds, 
                set_removes, 
                keys_to_check_versions, 
                indices_to_check_versions, 
                as_of_version,
                confirmCallback
                ):
        assert confirmCallback is not None
        
        transaction_guid = str(uuid.uuid4()).replace("-","")

        self._transaction_callbacks[transaction_guid] = confirmCallback

        self._channel.write(
            ClientToServer.NewTransaction(
                writes={k:json.dumps(v.jsonRep) if v.jsonRep is not None else None for k,v in key_value.items()},
                set_adds=set_adds,
                set_removes=set_removes,
                key_versions=keys_to_check_versions,
                index_versions=indices_to_check_versions,
                as_of_version=as_of_version,
                transaction_guid=transaction_guid
                )
            )
