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

from object_database.object import DatabaseObject, Index, Indexed
from object_database.view import View, JsonWithPyRep, Transaction, _cur_view
from object_database.algebraic_protocol import AlgebraicProtocol
import asyncio
import inspect
import json
from typed_python.hash import sha_hash
from typed_python import *

from types import FunctionType

import queue
import threading
import logging
import uuid
import traceback
import time

from object_database.view import RevisionConflictException, DisconnectedException

TransactionResult = Alternative(
    "TransactionResult", 
    Success = {},
    RevisionConflict = {},
    Disconnected = {}
    )

def revisionConflictRetry(f):
    MAX_TRIES = 100

    def inner(*args, **kwargs):
        tries = 0
        while tries < MAX_TRIES:
            try:
                return f(*args, **kwargs)
            except RevisionConflictException:
                logging.info("Handled a RevisionConflictException")
                tries += 1

        raise RevisionConflictException()

    inner.__name__ = f.__name__
    return inner

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

                if fieldname != ".exists":
                    changed[o].append((
                        fieldname,
                        View.unwrapJsonWithPyRep(key_value[k], o.__types__[fieldname]),
                        View.unwrapJsonWithPyRep(priors[k], o.__types__[fieldname])
                        ))

        self._queue.put(changed)


class Schema:
    """A collection of types that can be used to access data in a database."""
    def __init__(self, name="default"):
        self._name = name
        self._types = {}
        #class -> indexname -> fun(object->value)
        self._indices = {}
        self._indexTypes = {}
        self._frozen = False

    @property
    def name(self):
        return self._name

    def freeze(self):
        if not self._frozen:
            for tname, t in self._types.items():
                if issubclass(t, DatabaseObject) and t.__types__ is None:
                    raise Exception("Database subtype %s is not defined." % tname)

            self._frozen = True

    def __setattr__(self, typename, val):
        if typename[:1] == "_":
            self.__dict__[typename] = val
            return
        
        assert not self._frozen, "Schema is already frozen."

        self._types[typename] = val

    def __getattr__(self, typename):
        assert '.' not in typename

        if typename[:1] == "_":
            return self.__dict__[typename]

        if typename not in self._types:
            if self._frozen:
                raise AttributeError(typename)

            class cls(DatabaseObject):
                pass

            cls.__qualname__ = typename
            cls.__schema__ = self

            self._types[typename] = cls
            self._indices[cls] = {' exists': lambda e: True}
            self._indexTypes[cls] = {' exists': bool}

        return self._types[typename]

    def _addIndex(self, type, prop, fun = None, index_type = None):
        assert issubclass(type, DatabaseObject)

        if type not in self._indices:
            self._indices[type] = {}
            self._indexTypes[type] = {}

        if fun is None:
            fun = lambda o: getattr(o, prop)
            index_type = type.__types__[prop]
        else:
            if index_type is None:
                spec = inspect.getfullargspec(fun)
                index_type = spec.annotations.get('return', None)

        self._indices[type][prop] = fun
        self._indexTypes[type][prop] = index_type

    def define(self, cls):
        assert cls.__name__[:1] != "_", "Illegal to use _ for first character in database classnames."
        assert not self._frozen, "Schema is already frozen"

        t = getattr(self, cls.__name__)
        
        types = {}
        
        for name, val in cls.__dict__.items():
            if name[:2] != '__' and isinstance(val, type):
                types[name] = val
            elif name[:2] != '__' and isinstance(val, Indexed):
                if isinstance(val.obj, type):
                    types[name] = val.obj

        t._define(**types)

        for name, val in cls.__dict__.items():
            if isinstance(val, Index):
                self._addIndex(t, name, val, Tuple(*tuple(types[k] for k in val.names)))

            if name[:2] != '__' and isinstance(val, Indexed):
                if isinstance(val.obj, FunctionType):
                    self._addIndex(t, name, val.obj)
                    setattr(t, name, val.obj)
                else:
                    self._addIndex(t, name)
            elif (not name.startswith("__") or name in ["__str__", "__repr__"]):
                if isinstance(val, FunctionType):
                    setattr(t, name, val)

        return t

core_schema = Schema("core")

@core_schema.define
class Connection:
    pass


class DatabaseCore:
    def __init__(self):
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

        self.stableIdentities = False
        self._identityIx = 0

        #transaction handlers. These must be nonblocking since we call them under lock
        self._onTransaction = []

        self._schemas = {}

    def addSchema(self, schema):
        schema.freeze()

        if schema.name in self._schemas:
            assert schema is self._schemas.get(schema.name), "Schema %s already defined" % schema.name
            return

        self._schemas[schema.name] = schema

    def _data_key_to_object(self, key):
        typename,identity,fieldname = key.split(":")

        schemaname, typename, suffix = typename.split("-")
        
        schema = self._schemas.get(schemaname)
        if not schema:
            return None,None

        cls = schema._types.get(typename)
        
        if cls:
            return cls.fromIdentity(identity), fieldname

        return None,None

    def getStableIdentity(self):
        with self._lock:
            self._identityIx = self._identityIx + 1
            return "ID_" + str(self._identityIx)
    
    def clearCache(self):
        with self._lock:
            self._versioned_objects = {k:v for k,v in self._versioned_objects.items() if not v.isEmpty()}

    def __str__(self):
        return "Database(%s)" % id(self)

    def __repr__(self):
        return "Database(%s)" % id(self)

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

            if self._min_reffed_version_number is None or self._min_reffed_version_number < lowest:
                #some transactions still refer to values before this version
                return

            self._version_numbers.pop(0)

            keys_touched = self._version_number_objects[lowest]
            del self._version_number_objects[lowest]

            self._min_transaction_num = lowest

            for key in keys_touched:
                self._versioned_objects[key].cleanup(lowest)
                

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



class Database(DatabaseCore):
    def __init__(self, kvstore):
        DatabaseCore.__init__(self)
        self._kvstore = kvstore
        self._clientChannels = []
        self._transactionWriteLock = threading.Lock()
        self.initialized.set()


    def clone(self):
        assert self.initialized.isSet()

        with self.transaction():
            conn = Connection()

        return DatabaseWrapper(self, conn)
    
    def dropConnection(self, channel):
        with self._lock:
            self._clientChannels = [x for x in self._clientChannels if x.channel is not channel]

        try:
            with self.transaction():
                channel.connectionObject.delete()
        except:
            logging.error('Error deleting connection objects for channel %s:\n%s', 
                channel.connectionObject._identity, 
                traceback.format_exc()
                )

    def addConnection(self, channel):
        with self.transaction():
            connectionObject = core_schema.Connection()

        try:
            with self._lock:
                connectedChannel = ConnectedChannel(self._cur_transaction_num, channel, connectionObject)

                self._clientChannels.append(connectedChannel)

                channel.setClientToServerHandler(
                    lambda msg: self._onClientToServerMessage(connectedChannel, msg)
                    )

                connectedChannel.sendInitializationMessage()
        except:
            try:
                with self.transaction():
                    connectionObject.delete()
            except:
                logging.error(
                    "Failed to delete a Connection object during connection creation:\n%s", 
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
                isOK = [None]

                def onCommit(successful):
                    isOK[0] = successful.matches.Success

                self._set_versioned_object_data(
                    {k: JsonWithPyRep(json.loads(v),None) for k,v in msg.writes.items()},
                    {k: set(a) for k,a in msg.set_adds.items() if a},
                    {k: set(a) for k,a in msg.set_removes.items() if a},
                    msg.key_versions,
                    msg.index_versions,
                    msg.as_of_version,
                    onCommit
                    )

                self._cleanup()
            except:
                traceback.print_exc()
                logging.error("Unknown error: %s", traceback.format_exc())
                isOK[0] = False

            connectedChannel.sendTransactionSuccess(msg.transaction_guid, isOK[0])


    def _get_versioned_set_data(self, key, transaction_id):
        with self._lock:
            assert transaction_id >= self._min_transaction_num

            if key not in self._versioned_objects:
                members = self._kvstore.setMembers(key)

                self._versioned_objects[key] = VersionedSet(members)

            #get the largest version number less than or equal to transaction_id
            return self._versioned_objects[key].valueForVersion(transaction_id)

    def _get_versioned_object_data(self, key, transaction_id):
        with self._lock:
            assert transaction_id >= self._min_transaction_num

            if key not in self._versioned_objects:
                self._versioned_objects[key] = VersionedValue(
                    JsonWithPyRep(
                        self._kvstore.get(key),
                        None
                        )
                    )

            return self._versioned_objects[key].valueForVersion(transaction_id)

    def _set_versioned_object_data(self, 
                key_value, 
                set_adds, 
                set_removes, 
                keys_to_check_versions, 
                indices_to_check_versions, 
                as_of_version,
                confirmCallback
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

        with self._lock:
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
                        confirmCallback(TransactionResult.RevisionConflict())
                        return

            for key in keysWritingTo:
                obj = self._versioned_objects[key]

                for client in self._clientChannels:
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

            for client in self._clientChannels:
                client.sendTransaction(
                    key_value,
                    set_adds,
                    set_removes,
                    transaction_id
                    )

            confirmCallback(TransactionResult.Success())

            self._transactionWriteLock.acquire()

        try:
            for handler in self._onTransaction:
                try:
                    handler(key_value, priors, set_adds, set_removes, transaction_id)
                except:
                    logging.error(
                        "_onTransaction callback %s threw an exception:\n%s", 
                        handler, 
                        traceback.format_exc()
                        )
        finally:
            self._transactionWriteLock.release()

ClientToServer = Alternative(
    "ClientToServer",
    NewTransaction = {
        "writes": ConstDict(str, str),
        "set_adds": ConstDict(str, TupleOf(str)),
        "set_removes": ConstDict(str, TupleOf(str)),
        "key_versions": TupleOf(str),
        "index_versions": TupleOf(str),
        "as_of_version": int,
        "transaction_guid": str
        },
    SendValues = {
        "keys": TupleOf(str)
        },
    SendSets = {
        "keys": TupleOf(str)
        }
    )

ServerToClient = Alternative(
    "ServerToClient",
    Initialize = {'transaction_num': int, 'connIdentity': str},
    TransactionResult = {'transaction_guid': str, 'success': bool},
    KeyInfo = {'key': str, 'data': str, 'is_set': bool, 'transaction_id': int},
    Disconnected = {},
    Transaction = {
        "writes": ConstDict(str, str),
        "set_adds": ConstDict(str, TupleOf(str)),
        "set_removes": ConstDict(str, TupleOf(str)),
        "transaction_id": int
        }
    )

class InMemoryChannel:
    def __init__(self):
        self._clientCallback = None
        self._serverCallback = None
        self._clientToServerMsgQueue = queue.Queue()
        self._serverToClientMsgQueue = queue.Queue()
        self._shouldStop = True

        self._pumpThreadServer = threading.Thread(target=self.pumpMessagesFromServer)
        self._pumpThreadServer.daemon = True
        
        self._pumpThreadClient = threading.Thread(target=self.pumpMessagesFromClient)
        self._pumpThreadClient.daemon = True
        
    def clone(self):
        res = InMemoryChannel()
        res.setServerToClientHandler(self._clientCallback)
        res.setClientToServerHandler(self._serverCallback)
        res.start()
        return res

    def pumpMessagesFromServer(self):
        while not self._shouldStop:
            try:
                e = self._serverToClientMsgQueue.get(timeout=0.01)
            except queue.Empty:
                e = None

            if e:
                try:
                    self._clientCallback(e)
                except:
                    traceback.print_exc()
                    logging.error("Pump thread failed: %s", traceback.format_exc())
                    return
        
    def pumpMessagesFromClient(self):
        while not self._shouldStop:
            try:
                e = self._clientToServerMsgQueue.get(timeout=0.01)
            except queue.Empty:
                e = None

            if e:
                try:
                    self._serverCallback(e)
                except:
                    traceback.print_exc()
                    logging.error("Pump thread failed: %s", traceback.format_exc())
                    return

    def start(self):
        assert self._shouldStop
        self._shouldStop = False
        self._pumpThreadServer.start()
        self._pumpThreadClient.start()

    def stop(self):
        self._shouldStop = True
        self._pumpThreadServer.join()
        self._pumpThreadClient.join()

    def write(self, msg):
        if isinstance(msg, ClientToServer):
            self._clientToServerMsgQueue.put(msg)
        elif isinstance(msg, ServerToClient):
            self._serverToClientMsgQueue.put(msg)
        else:
            assert False

    def setServerToClientHandler(self, callback):
        self._clientCallback = callback

    def setClientToServerHandler(self, callback):
        self._serverCallback = callback

class DatabaseWrapper:
    def __init__(self, inner_db, connectionObject):
        self._db = inner_db
        self.connectionObject = connectionObject
        self.initialized = inner_db.initialized

        self.disconnected = threading.Event()

    def clone(self):
        assert self.initialized.isSet()

        with self.transaction():
            conn = Connection()

        return DatabaseWrapper(self, conn)
    
    def view(self):
        return self._db.view()

    def transaction(self):
        return self._db.transaction()

    def _get_versioned_set_data(self, key, transaction_id):
        if self.disconnected.is_set():
            raise DisconnectedException()
        return self._db._get_versioned_set_data(key,transaction_id)

    def _get_versioned_object_data(self, key, transaction_id):
        if self.disconnected.is_set():
            raise DisconnectedException()
        return self._db._get_versioned_object_data(key,transaction_id)

    def disconnect(self):
        with self.transaction():
            self.connectionObject.delete()
        self.disconnected.set()

    def _set_versioned_object_data(self, 
                key_value, 
                set_adds, 
                set_removes, 
                keys_to_check_versions, 
                indices_to_check_versions, 
                as_of_version,
                confirmCallback
                ):

        if self.disconnected.is_set():
            raise DisconnectedException()

        return self._db._set_versioned_object_data(
                key_value, 
                set_adds, 
                set_removes, 
                keys_to_check_versions, 
                indices_to_check_versions, 
                as_of_version,
                confirmCallback
                )



class DatabaseConnection(DatabaseCore):
    def __init__(self, channel):
        DatabaseCore.__init__(self)
        self._channel = channel
        self._channel.setServerToClientHandler(self._onMessage)
        self._transaction_callbacks = {}
        self._read_events = {}

    def clone(self):
        assert self.initialized.isSet()

        return DatabaseCore(self._channel.clone())

    def _onMessage(self, msg):
        with self._lock:
            if msg.matches.Disconnected:
                self.disconnected.set()
                self.connectionObject = None
            
                for q in self._transaction_callbacks.values():
                    try:
                        q(TransactionResult.Disconnected())
                    except:
                        logging.error(
                            "Transaction commit callback threw an exception:\n%s", 
                            traceback.format_exc()
                            )

                for i in self._read_events.values():
                    i.set()
                self._transaction_callbacks = {}
                self._read_events = {}

            elif msg.matches.Initialize:
                self._min_transaction_num = self._cur_transaction_num = msg.transaction_num
                self.connectionObject = core_schema.Connection.fromIdentity(msg.connIdentity)
                self.initialized.set()
            elif msg.matches.TransactionResult:
                try:
                    self._transaction_callbacks.pop(msg.transaction_guid)(
                        TransactionResult.Success() if msg.success 
                            else TransactionResult.RevisionConflict()
                        )
                except:
                    logging.error(
                        "Transaction commit callback threw an exception:\n%s", 
                        traceback.format_exc()
                        )
            elif msg.matches.KeyInfo:
                key = msg.key
                if key not in self._versioned_objects:
                    if msg.is_set:                    
                        self._versioned_objects[key] = VersionedSet(set(json.loads(msg.data)))
                    else:
                        self._versioned_objects[key] = VersionedValue(JsonWithPyRep(json.loads(msg.data), None))

                if key in self._read_events:
                    self._read_events.pop(key).set()

            elif msg.matches.Transaction:
                key_value = {}
                priors = {}
                for k,val_serialized in msg.writes.items():
                    json_val = json.loads(val_serialized)

                    key_value[k] = json_val

                    versioned = self._versioned_objects[k]

                    priors[k] = versioned.newestValue()

                    versioned.setVersionedValue(msg.transaction_id, JsonWithPyRep(json_val, None))

                for k,a in msg.set_adds.items():
                    self._versioned_objects[k].setVersionedAddsAndRemoves(msg.transaction_id, set(a), set())

                for k,r in msg.set_removes.items():
                    self._versioned_objects[k].setVersionedAddsAndRemoves(msg.transaction_id, set(), set(r))

                self._cur_transaction_num = msg.transaction_id

                for handler in self._onTransaction:
                    try:
                        handler(key_value, priors, msg.set_adds, msg.set_removes, msg.transaction_id)
                    except:
                        logging.error(
                            "_onTransaction callback %s threw an exception:\n%s", 
                            handler, 
                            traceback.format_exc()
                            )
            
        
    def _get_versioned_set_data(self, key, transaction_id):
        assert transaction_id >= self._min_transaction_num

        with self._lock:
            if key in self._versioned_objects:
                return self._versioned_objects[key].valueForVersion(transaction_id)
            
            if self.disconnected.is_set():
                raise DisconnectedException()

            self._channel.write(ClientToServer.SendSets(keys=(key,)))

            if key not in self._read_events:
                self._read_events[key] = threading.Event()

            e = self._read_events[key]

        e.wait()

        if self.disconnected.is_set():
            raise DisconnectedException()

        with self._lock:
            return self._versioned_objects[key].valueForVersion(transaction_id)

    def _get_versioned_object_data(self, key, transaction_id):
        assert transaction_id >= self._min_transaction_num

        with self._lock:
            if key in self._versioned_objects:
                return self._versioned_objects[key].valueForVersion(transaction_id)

            if self.disconnected.is_set():
                raise DisconnectedException()

            self._channel.write(ClientToServer.SendValues(keys=(key,)))

            if key not in self._read_events:
                self._read_events[key] = threading.Event()

            e = self._read_events[key]

        e.wait()

        if self.disconnected.is_set():
            raise DisconnectedException()

        with self._lock:
            return self._versioned_objects[key].valueForVersion(transaction_id)

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
                writes={k:json.dumps(v.jsonRep) for k,v in key_value.items()},
                set_adds=set_adds,
                set_removes=set_removes,
                key_versions=keys_to_check_versions,
                index_versions=indices_to_check_versions,
                as_of_version=as_of_version,
                transaction_guid=transaction_guid
                )
            )


class ServerToClientProtocol(AlgebraicProtocol):
    def __init__(self, dbserver):
        AlgebraicProtocol.__init__(self, ClientToServer, ServerToClient)
        self.dbserver = dbserver

    def setClientToServerHandler(self, handler):
        self.handler = handler

    def messageReceived(self, msg):
        self.handler(msg)

    def onConnected(self):
        self.dbserver.db.addConnection(self)

    def write(self, msg):
        self.sendMessage(msg)

    def connection_lost(self, e):
        self.dbserver.db.dropConnection(self)

class ClientToServerProtocol(AlgebraicProtocol):
    def __init__(self, host, port):
        AlgebraicProtocol.__init__(self, ServerToClient, ClientToServer)
        self.lock = threading.Lock()
        self.host = host
        self.port = port
        self.handler = None
        self.msgs = []
    
    def clone(self):
        return eventLoop.create_connection(
            lambda: ClientToServerProtocol(self.host, self.port),
            self.host,
            self.port
            )

    def setServerToClientHandler(self, handler):
        with self.lock:
            self.handler = handler
            for m in self.msgs:
                _eventLoop.loop.call_soon_threadsafe(self.handler, m)
            self.msgs = None

    def messageReceived(self, msg):
        with self.lock:
            if not self.handler:
                self.msgs.append(msg)
            else:
                _eventLoop.loop.call_soon_threadsafe(self.handler, msg)
        
    def onConnected(self):
        pass

    def connection_lost(self, e):
        self.messageReceived(ServerToClient.Disconnected())

    def write(self, msg):
        _eventLoop.loop.call_soon_threadsafe(self.sendMessage, msg)

class EventLoopInThread:
    def __init__(self):
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self.runEventLoop)
        self.thread.daemon = True
        self.started = False

    def runEventLoop(self):
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    def start(self):
        if not self.started:
            self.started = True
            self.thread.start()

    def create_connection(self, callback, host, port):
        self.start()

        async def doit():
            return await self.loop.create_connection(callback, host, port)

        return asyncio.run_coroutine_threadsafe(doit(), self.loop).result(10)

    def create_server(self, callback, host, port):
        self.start()

        async def doit():
            return await self.loop.create_server(callback, host, port)

        res = asyncio.run_coroutine_threadsafe(doit(), self.loop)

        return res.result(10)

_eventLoop = EventLoopInThread()

def connect(host, port):
    _, proto = _eventLoop.create_connection(
        lambda: ClientToServerProtocol(host, port),
        host,
        port
        )

    return DatabaseConnection(proto)

class DatabaseServer:
    def __init__(self, db, host, port):
        self.db = db
        self.host = host
        self.port = port
        self.server = None

    def start(self):
        self.server = _eventLoop.create_server(
            lambda: ServerToClientProtocol(self), 
            self.host, 
            self.port
            )
        
    def stop(self):
        if self.server:
            self.server.close()

