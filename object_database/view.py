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

from typed_python.hash import sha_hash

from typed_python import Alternative, OneOf, TupleOf, ConstDict, TypeConvert, Tuple, Kwargs

import object_database.algebraic_to_json as algebraic_to_json

import threading
import queue
import time
import traceback
import uuid

class DisconnectedException(Exception):
    pass

class RevisionConflictException(Exception):
    pass

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

class JsonWithPyRep:
    """A value stored as Json with a python representation."""
    def __init__(self, jsonRep, pyRep):
        self.jsonRep = jsonRep
        self.pyRep = pyRep

def data_key(obj_type, identity, field_name):
    return obj_type.__schema__.name + "-" + obj_type.__qualname__ + "-val:" + identity + ":" + field_name

def index_key(obj_type, field_name, value):
    if isinstance(value, int):
        value_hash = "int_" + str(value)
    else:
        value_hash = sha_hash(value).hexdigest

    return obj_type.__schema__.name + "-" + obj_type.__qualname__ + "-ix:" + field_name + ":" + value_hash

def default_initialize(t):
    if t in (str,bool,bytes,int,float):
        return t()
    if t is type(None):
        return None
    if isinstance(t, Kwargs):
        return Kwargs(**{k:default_initialize(t) for k,t in t.ElementTypes.items()})
    if isinstance(t, (TupleOf, ConstDict)):
        return t()
    if isinstance(t, OneOf):
        if None in t.options:
            return None
        for o in t.options:
            if isinstance(o, (str,bool,bytes,int,float)):
                return o
        for o in sorted(t.options, key=str):
            if hasattr(o, '__default_initializer__'):
                return o.__default_initializer__()

        raise Exception("can't default initialize OneOf(%s)" % t.options)

    if hasattr(t, '__default_initializer__'):
        return t.__default_initializer__()

    raise Exception("Can't default initialize %s" % (t))

_encoder = algebraic_to_json.Encoder()
_encoder.allowExtraFields = True

_cur_view = threading.local()

class View(object):
    _writeable = False

    def __init__(self, db, transaction_id):
        object.__init__(self)
        self._db = db
        self._transaction_num = transaction_id
        self._writes = {}
        self._reads = set()
        self._indexReads = set()
        self._set_adds = {}
        self._set_removes = {}
        self._t0 = None
        self._stack = None
        self._schemas = set()
        self._readWatcher = None
        self._insistReadsConsistent = True
        self._insistWritesConsistent = True
        self._insistIndexReadsConsistent = False
        self._confirmCommitCallback = None

    def _new(self, cls, kwds):
        if not self._writeable:
            raise Exception("Views are static. Please open a transaction.")

        self._schemas.add(cls.__schema__)

        if "_identity" in kwds:
            identity = kwds["_identity"]
            kwds = dict(kwds)
            del kwds["_identity"]
        else:
            identity = sha_hash(str(uuid.uuid4())).hexdigest

        o = cls.fromIdentity(identity)

        writes = {}

        kwds = dict(kwds)
        
        if not hasattr(cls, '__types__') or cls.__types__ is None:
            raise Exception("Please initialize the type object for %s" % str(cls.__qualname__))

        for tname, t in cls.__types__.items():
            if tname not in kwds:
                kwds[tname] = default_initialize(t)

        for kwd, val in kwds.items():
            if kwd not in cls.__types__:
                raise TypeError("Unknown field %s on %s" % (kwd, cls))

            coerced_val = TypeConvert(cls.__types__[kwd], val, allow_construct_new=True)

            writes[data_key(cls, identity, kwd)] = (cls.__types__[kwd], coerced_val)

        writes[data_key(cls, identity, " exists")] = True

        self._writes.update(writes)

        if cls in cls.__schema__._indices:
            for index_name, index_fun in cls.__schema__._indices[cls].items():
                val = index_fun(o)

                if val is not None:
                    ik = index_key(cls, index_name, val)

                    self._add_to_index(ik, identity)

        return o        

    def _get(self, obj, identity, field_name, field_type):
        key = data_key(type(obj), identity, field_name)

        self._reads.add(key)

        if self._readWatcher:
            self._readWatcher("key", key)

        if key in self._writes:
            res = self._writes[key][1]
            if isinstance(res, tuple):
                return res[1]
            return res

        dbValWithPyrep = self._db._get_versioned_object_data(key, self._transaction_num)

        return self.unwrapJsonWithPyRep(dbValWithPyrep, field_type)

    @staticmethod
    def unwrapJsonWithPyRep(dbValWithPyrep, field_type):
        if dbValWithPyrep is None:
            return None

        if dbValWithPyrep.jsonRep is None:
            return None

        if dbValWithPyrep.pyRep is None:
            dbValWithPyrep.pyRep = _encoder.from_json(dbValWithPyrep.jsonRep, field_type)

        return dbValWithPyrep.pyRep

    def _exists(self, obj, identity):
        key = data_key(type(obj), identity, " exists")

        if self._readWatcher:
            self._readWatcher("key", key)

        if key in self._writes:
            return self._writes[key] is not None
        
        val = self._db._get_versioned_object_data(key, self._transaction_num)

        return val.jsonRep is not None

    def _delete(self, obj, identity, field_names):
        existing_index_vals = self._compute_index_vals(obj)

        for name in field_names:
            key = data_key(type(obj), identity, name)
            self._writes[key] = None

        self._writes[data_key(type(obj), identity, " exists")] = None

        self._update_indices(obj, identity, existing_index_vals, {})

    def _set(self, obj, identity, field_name, field_type, val):
        if not self._writeable:
            raise Exception("Views are static. Please open a transaction.")

        key = data_key(type(obj), identity, field_name)

        existing_index_vals = self._compute_index_vals(obj)

        self._writes[key] = (field_type, val)
        
        new_index_vals = self._compute_index_vals(obj)

        self._update_indices(obj, identity, existing_index_vals, new_index_vals)

    def _compute_index_vals(self, obj):
        existing_index_vals = {}

        if type(obj) in obj.__schema__._indices:
            for index_name, index_fun in obj.__schema__._indices[type(obj)].items():
                existing_index_vals[index_name] = index_fun(obj)

        return existing_index_vals

    def _update_indices(self, obj, identity, existing_index_vals, new_index_vals):
        if type(obj) in obj.__schema__._indices:
            for index_name, index_fun in obj.__schema__._indices[type(obj)].items():
                new_index_val = new_index_vals.get(index_name, None)
                cur_index_val = existing_index_vals.get(index_name, None)

                if cur_index_val != new_index_val:
                    if cur_index_val is not None:
                        old_index_name = index_key(type(obj), index_name, cur_index_val)
                        self._remove_from_index(old_index_name, identity)

                    if new_index_val is not None:
                        new_index_name = index_key(type(obj), index_name, new_index_val)
                        self._add_to_index(new_index_name, identity)

    def _add_to_index(self, index_key, identity):
        assert isinstance(identity, str)

        if index_key not in self._set_adds:
            self._set_adds[index_key] = set()
            self._set_removes[index_key] = set()
            
        if identity in self._set_removes[index_key]:
            self._set_removes[index_key].discard(identity)
        else:
            self._set_adds[index_key].add(identity)

    def _remove_from_index(self, index_key, identity):
        assert isinstance(identity, str)

        if index_key not in self._set_adds:
            self._set_adds[index_key] = set()
            self._set_removes[index_key] = set()
        
        if identity in self._set_adds[index_key]:
            self._set_adds[index_key].discard(identity)
        else:
            self._set_removes[index_key].add(identity)

    def indexLookup(self, db_type, **kwargs):
        assert len(kwargs) == 1, "Can only lookup one index at a time."
        tname, value = list(kwargs.items())[0]

        if db_type not in db_type.__schema__._indices or tname not in db_type.__schema__._indices[db_type]:
            raise Exception("No index enabled for %s.%s.%s" % (db_type.__schema__.name, db_type.__qualname__, tname))

        if not hasattr(_cur_view, "view"):
            raise Exception("Please access indices from within a view.")

        indexType = db_type.__schema__._indexTypes[db_type][tname]

        if indexType is not None:
            value = TypeConvert(indexType, value, allow_construct_new=True)

        keyname = index_key(db_type, tname, value)

        self._indexReads.add(keyname)

        identities = self._db._get_versioned_set_data(keyname, self._transaction_num).toSet()
        identities = identities.union(self._set_adds.get(keyname, set()))
        identities = identities.difference(self._set_removes.get(keyname, set()))

        return tuple([db_type.fromIdentity(x) for x in identities])

    def indexLookupAny(self, db_type, **kwargs):
        assert len(kwargs) == 1, "Can only lookup one index at a time."
        tname, value = list(kwargs.items())[0]

        if db_type not in db_type.__schema__._indices or tname not in db_type.__schema__._indices[db_type]:
            raise Exception("No index enabled for %s.%s.%s" % (db_type.__schema__.name, db_type.__qualname__, tname))

        if not hasattr(_cur_view, "view"):
            raise Exception("Please access indices from within a view.")

        indexType = db_type.__schema__._indexTypes[db_type][tname]

        if indexType is not None:
            value = TypeConvert(indexType, value, allow_construct_new=True)

        keyname = index_key(db_type, tname, value)

        added = self._set_adds.get(keyname, set())
        removed = self._set_removes.get(keyname, set())

        if added:
            return db_type.fromIdentity(list(added)[0])

        self._indexReads.add(keyname)

        res = self._db._get_versioned_set_data(keyname, self._transaction_num).pickAny(removed)

        if res:
            return db_type.fromIdentity(res)

        return None

    def commit(self):
        if not self._writeable:
            raise Exception("Views are static. Please open a transaction.")

        if self._writes:
            def encode(val):
                if isinstance(val, tuple) and len(val) == 2 and isinstance(val[0], type):
                    return JsonWithPyRep(_encoder.to_json(val[0], val[1]), val[1])
                else:
                    return JsonWithPyRep(val, val)

            writes = {key: encode(v) for key, v in self._writes.items()}
            tid = self._transaction_num

            if (self._set_adds or self._set_removes) and not self._insistReadsConsistent:
                raise Exception("You can't update an indexed value without read and write consistency.")

            

            if self._confirmCommitCallback is None:
                result_queue = queue.Queue()

                confirmCallback = result_queue.put
            else:
                confirmCallback = self._confirmCommitCallback

            self._db._set_versioned_object_data(
                writes, 
                {k:v for k,v in self._set_adds.items() if v}, 
                {k:v for k,v in self._set_removes.items() if v}, 
                self._reads.union(set(writes)) if self._insistReadsConsistent 
                    else set(writes) if self._insistWritesConsistent 
                    else set(),
                self._indexReads if self._insistIndexReadsConsistent else set(),
                tid,
                confirmCallback
                )

            if not self._confirmCommitCallback:
                #this is the synchronous case - we want to wait for the confirm
                res = result_queue.get()
                if res.matches.Success:
                    return
                if res.matches.Disconnected:
                    raise DisconnectedException()
                if res.matches.RevisionConflict:
                    raise RevisionConflictException()

                assert False, "unknown transaction result: " + str(res)


    def nocommit(self):
        class Scope:
            def __enter__(scope):
                assert not hasattr(_cur_view, 'view')
                _cur_view.view = self

            def __exit__(self, *args):
                del _cur_view.view

        return Scope()

    def __enter__(self):
        self._t0 = time.time()
        
        assert not hasattr(_cur_view, 'view')
        _cur_view.view = self
        return self

    def __exit__(self, type, val, tb):
        if time.time() - self._t0 > 30.0:
            logging.warn("long db transaction: %s elapsed.\n%s", time.time() - self._t0, "".join(traceback.format_stack()))
        del _cur_view.view
        if type is None and self._writes:
            self.commit()

        self._db._releaseView(self)

    def indexLookupOne(self, lookup_type, **kwargs):
        res = self.indexLookup(lookup_type, **kwargs)
        if not res:
            raise Exception("No instances of %s found with %s" % (lookup_type, kwargs))
        if len(res) != 1:
            raise Exception("Multiple instances of %s found with %s" % (lookup_type, kwargs))
        return res[0]

class Transaction(View):
    _writeable = True

    def consistency(self, writes=False, reads=False, full=False, none=False):
        """Set the consistency model for the Transaction.
        
        if 'none', then the transaction always succeeds
        if 'writes', then we insist that any key you write to has not been updated, but
            allow read keys to have been updated.
        if 'reads', then we insist that any key you read or write is not updated, but allow
            index updates
        if 'full' is True, then we insist that any index you read from is not updated.
            (this is very stringent)

        This function modifies the view, and the semantics are only in place at
        commit time.
        """
        assert sum(int(i) for i in (reads,writes,full,none)) == 1, "Please set exactly one option"

        self._insistReadsConsistent = bool(reads or full)
        self._insistWritesConsistent = bool(writes or reads or full)
        self._insistIndexReadsConsistent = bool(full)

        return self

    def onConfirmed(self, callback):
        """Set a callback function to be called on the main event thread with a boolean indicating
        whether the transaction was accepted."""
        self._confirmCommitCallback = callback

        return self

    def noconfirm(self):
        """Indicate that the transaction should return immediately without a round-trip to
        confirm that it was successful."""
        def ignoreConfirmResult(result):
            pass
        self._confirmCommitCallback = ignoreConfirmResult

        return self


