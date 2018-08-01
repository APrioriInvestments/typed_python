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

from typed_python import Alternative, OneOf, TupleOf, ConstDict, TypeConvert

import object_database.algebraic_to_json as algebraic_to_json
from typed_python.hash import sha_hash

import threading
import logging
import uuid
import traceback
import time

_encoder = algebraic_to_json.Encoder()
_encoder.allowExtraFields = True

_cur_view = threading.local()

class RevisionConflictException(Exception):
    pass

#singleton object that clients should never see
_creating_the_null_object = []

class DatabaseObject(object):
    __algebraic__ = True
    __types__ = None
    _database = None
    Null = None

    def __ne__(self, other):
        return not (self==other)
        
    def __eq__(self, other):
        if not isinstance(other, DatabaseObject):
            return False
        if not self._database is other._database:
            return False
        if not type(self) is type(other):
            return False
        return self._identity == other._identity

    def __bool__(self):
        return self is not type(self).Null

    def __nonzero__(self):
        return self is not type(self).Null

    def __hash__(self):
        return hash(self._identity)

    @classmethod
    def __typed_python_try_convert_instance__(cls, value, allow_construct_new):
        if isinstance(value, cls):
            return (value,)
        return None

    def __init__(self, identity):
        object.__init__(self)

        if identity is _creating_the_null_object:
            if type(self).Null is None:
                type(self).Null = self
            identity = "NULL"
        else:
            assert isinstance(identity, str), type(identity)

        self.__dict__['_identity'] = identity

    @classmethod
    def __default_initializer__(cls):
        return cls.Null

    @classmethod
    def New(cls, **kwds):
        if not hasattr(_cur_view, "view"):
            raise Exception("Please create new objects from within a transaction.")

        if _cur_view.view._db is not cls._database:
            raise Exception("Please create new objects from within a transaction created on the same database as the object.")

        return _cur_view.view._new(cls, kwds)

    @classmethod
    def lookupOne(cls, **kwargs):
        return cls._database.current_transaction().indexLookupOne(cls, **kwargs)

    @classmethod
    def lookupAll(cls, **kwargs):
        return cls._database.current_transaction().indexLookup(cls, **kwargs)

    @classmethod
    def lookupAny(cls, **kwargs):
        return cls._database.current_transaction().indexLookupAny(cls, **kwargs)

    def exists(self):
        if not hasattr(_cur_view, "view"):
            raise Exception("Please access properties from within a view or transaction.")

        if _cur_view.view._db is not type(self)._database:
            raise Exception("Please access properties from within a view or transaction created on the same database as the object.")

        return _cur_view.view._exists(self, type(self).__qualname__, self._identity)

    def __getattr__(self, name):
        if name[:1] == "_":
            raise AttributeError(name)

        return self.get_field(name)

    def get_field(self, name):
        if self.__dict__["_identity"] == "NULL":
            raise Exception("Null object of type %s has no fields" % type(self).__qualname__)

        if name not in self.__types__:
            raise AttributeError("Object of type %s has no field %s" % (type(self).__qualname__, name))

        if not hasattr(_cur_view, "view"):
            raise Exception("Please access properties from within a view or transaction.")

        if _cur_view.view._db is not type(self)._database:
            raise Exception("Please access properties from within a view or transaction created on the same database as the object.")
        
        return _cur_view.view._get(type(self).__qualname__, self._identity, name, self.__types__[name])

    def __setattr__(self, name, val):
        if self.__dict__["_identity"] == "NULL":
            raise Exception("Null object is not writeable")

        if name not in self.__types__:
            raise AttributeError("Database object of type %s has no attribute %s" % (type(self).__qualname__, name))

        if not hasattr(_cur_view, "view"):
            raise Exception("Please access properties from within a view or transaction.")

        if _cur_view.view._db is not type(self)._database:
            raise Exception("Please access properties from within a view or transaction created on the same database as the object.")

        coerced_val = TypeConvert(self.__types__[name], val, allow_construct_new=True)

        _cur_view.view._set(self, type(self).__qualname__, self._identity, name, self.__types__[name], coerced_val)

    def delete(self):
        if self.__dict__["_identity"] is None:
            raise Exception("Null object is not writeable")

        _cur_view.view._delete(self, type(self).__qualname__, self._identity, self.__types__.keys())

    @classmethod
    def methods_from(cls, other):
        for method_name, method in other.__dict__.items():
            if not method_name.startswith("__") or method_name in ["__str__", "__repr__"]:
                setattr(cls, method_name, method)

    @classmethod
    def define(cls, **types):
        assert not cls.Null, "already defined"
        assert isinstance(types, dict)
        
        cls.__types__ = types
        cls.Null = cls(_creating_the_null_object)

        return cls

    @classmethod
    def to_json(cls, obj):
        return obj.__dict__['_identity']

    @classmethod
    def from_json(cls, obj):
        if obj == "NULL":
            return cls.Null

        assert isinstance(obj, str), obj

        return cls(obj)

    def __sha_hash__(self):
        return sha_hash(self._identity) + sha_hash(type(self).__qualname__)

def data_key(obj_typename, identity, field_name):
    return obj_typename + "-val:" + identity + ":" + field_name

def index_key(obj_typename, field_name, value):
    if isinstance(value, int):
        value_hash = "int_" + str(value)
    else:
        value_hash = sha_hash(value).hexdigest

    return obj_typename + "-ix:" + field_name + ":" + value_hash

def default_initialize(t):
    if t in (str,bool,bytes,int,float):
        return t()
    if t is type(None):
        return None
    if isinstance(t, (TupleOf, ConstDict)):
        return t()
    if isinstance(t, OneOf):
        if None in t.options:
            return None

    if hasattr(t, '__default_initializer__'):
        return t.__default_initializer__()

    raise Exception("Can't default initialize %s" % (t))

class DatabaseView(object):
    _writeable = False

    def __init__(self, db, transaction_id):
        object.__init__(self)
        self._db = db
        self._transaction_num = transaction_id
        self._writes = {}
        self._t0 = None
        self._stack = None
        self._readWatcher = None

    def _get_dbkey(self, key):
        if key in self._writes:
            return self._writes[key]
        return self._db._get_versioned_object_data(key, self._transaction_num)[0]

    def _new(self, cls, kwds):
        if not self._writeable:
            raise Exception("Views are static. Please open a transaction.")

        if "_identity" in kwds:
            identity = kwds["_identity"]
            kwds = dict(kwds)
            del kwds["_identity"]
        else:
            identity = sha_hash(str(uuid.uuid4())).hexdigest

        o = cls(identity)

        writes = {}

        kwds = dict(kwds)
        for tname, t in cls.__types__.items():
            if tname not in kwds:
                kwds[tname] = default_initialize(t)

        for kwd, val in kwds.items():
            if kwd not in cls.__types__:
                raise TypeError("Unknown field %s on %s" % (kwd, cls))

            coerced_val = TypeConvert(cls.__types__[kwd], val, allow_construct_new=True)

            writes[data_key(cls.__qualname__, identity, kwd)] = (cls.__types__[kwd], coerced_val)

        writes[data_key(cls.__qualname__, identity, ".exists")] = True

        self._writes.update(writes)

        if cls.__qualname__ in self._db._indices:
            for index_name, index_fun in self._db._indices[cls.__qualname__].items():
                val = index_fun(o)

                if val is not None:
                    ik = index_key(cls.__qualname__, index_name, val)

                    if ik in self._writes:
                        self._writes[ik] = self._writes[ik] + (identity,)
                    else:
                        existing = self._get_dbkey(ik)
                        if existing is None:
                            existing = ()
                        else:
                            existing = tuple(existing)

                        self._writes[ik] = existing + (identity,)

        return o        

    def _get(self, obj_typename, identity, field_name, type):
        key = data_key(obj_typename, identity, field_name)

        if self._readWatcher:
            self._readWatcher("key", key)

        if key in self._writes:
            return self._writes[key][1]

        db_val, parsed_val = self._db._get_versioned_object_data(key, self._transaction_num)

        db_val = self._get_dbkey(key)

        if db_val is None:
            return db_val

        if parsed_val is not None:
            return parsed_val

        parsed_val = _encoder.from_json(db_val, type)

        self._db._update_versioned_object_data_cache(key, self._transaction_num, parsed_val)

        return parsed_val

    def _exists(self, obj, obj_typename, identity):
        key = data_key(obj_typename, identity, ".exists")

        if self._readWatcher:
            self._readWatcher("key", key)

        return self._get_dbkey(key) is not None

    def _delete(self, obj, obj_typename, identity, field_names):
        existing_index_vals = self._compute_index_vals(obj, obj_typename)

        for name in field_names:
            key = data_key(obj_typename, identity, name)
            self._writes[key] = None

        self._writes[data_key(obj_typename, identity, ".exists")] = None

        self._update_indices(obj, obj_typename, identity, existing_index_vals, {})

    def _set(self, obj, obj_typename, identity, field_name, type, val):
        if not self._writeable:
            raise Exception("Views are static. Please open a transaction.")

        key = data_key(obj_typename, identity, field_name)

        existing_index_vals = self._compute_index_vals(obj, obj_typename)

        self._writes[key] = (type, val)
        
        new_index_vals = self._compute_index_vals(obj, obj_typename)

        self._update_indices(obj, obj_typename, identity, existing_index_vals, new_index_vals)

    def _compute_index_vals(self, obj, obj_typename):
        existing_index_vals = {}

        if obj_typename in self._db._indices:
            for index_name, index_fun in self._db._indices[obj_typename].items():
                existing_index_vals[index_name] = index_fun(obj)

        return existing_index_vals

    def _update_indices(self, obj, obj_typename, identity, existing_index_vals, new_index_vals):
        if obj_typename in self._db._indices:
            for index_name, index_fun in self._db._indices[obj_typename].items():
                new_index_val = new_index_vals.get(index_name, None)
                cur_index_val = existing_index_vals.get(index_name, None)

                if cur_index_val != new_index_val:
                    if cur_index_val is not None:
                        old_index_name = index_key(obj_typename, index_name, cur_index_val)
                        cur_index_list = tuple(self._get_dbkey(old_index_name) or ())
                        self._writes[old_index_name] = tuple([x for x in cur_index_list if x != identity])

                    if new_index_val is not None:
                        new_index_name = index_key(obj_typename, index_name, new_index_val)
                        new_index_list = tuple(self._get_dbkey(new_index_name) or ())
                        self._writes[new_index_name] = new_index_list + (identity,)

    def indexLookup(self, type, **kwargs):
        assert len(kwargs) == 1, "Can only lookup one index at a time."
        tname, value = list(kwargs.items())[0]

        if type.__qualname__ not in self._db._indices or tname not in self._db._indices[type.__qualname__]:
            raise Exception("No index enabled for %s.%s" % (type.__qualname__, tname))

        if not hasattr(_cur_view, "view"):
            raise Exception("Please access indices from within a view.")

        keyname = index_key(type.__qualname__, tname, value)

        if keyname in self._writes:
            identities = self._writes[keyname]
        else:
            identities = self._db._get_versioned_object_data(keyname, self._transaction_num)[0]
            
        if not identities:
            return ()
        
        return tuple([type(str(x)) for x in identities])

    def indexLookupAny(self, type, **kwargs):
        assert len(kwargs) == 1, "Can only lookup one index at a time."
        tname, value = kwargs.items()[0]

        if type.__qualname__ not in self._db._indices or tname not in self._db._indices[type.__qualname__]:
            raise Exception("No index enabled for %s.%s" % (type.__qualname__, tname))

        if not hasattr(_cur_view, "view"):
            raise Exception("Please access indices from within a view.")

        keyname = index_key(type.__qualname__, tname, value)

        if keyname in self._writes:
            identities = self._writes[keyname]
        else:
            identities = self._db._get_versioned_object_data(keyname, self._transaction_num)[0]
            
        if not identities:
            return None
        
        return type(str(identities[0]))

    def commit(self):
        if not self._writeable:
            raise Exception("Views are static. Please open a transaction.")

        if self._writes:
            def encode(val):
                if isinstance(val, tuple) and len(val) == 2 and isinstance(val[0], type):
                    return (_encoder.to_json(val[0], val[1]), val[1])
                else:
                    return (val,val)

            writes = {key: encode(v) for key, v in self._writes.items()}
            tid = self._transaction_num
            
            self._db._set_versioned_object_data(writes, tid)

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
        self._stack = traceback.format_stack()

        assert not hasattr(_cur_view, 'view')
        _cur_view.view = self
        return self

    def __exit__(self, type, val, tb):
        if time.time() - self._t0 > 30.0:
            logging.warn("long db transaction: %s elapsed.\n%s", time.time() - self._t0, "".join(self._stack))
        del _cur_view.view
        if type is None and self._writes:
            self.commit()

        self._db._releaseView(self)

    def indexLookupOne(self, type, **kwargs):
        res = self.indexLookup(type, **kwargs)
        if not res:
            raise Exception("No instances of %s found with %s" % (type, kwargs))
        if len(res) != 1:
            raise Exception("Multiple instances of %s found with %s" % (type, kwargs))
        return res[0]

class DatabaseTransaction(DatabaseView):
    _writeable = True




class Database:
    def __init__(self, kvstore):
        self._kvstore = kvstore
        self._lock = threading.Lock()

        #transaction of what's in the KV store
        self._cur_transaction_num = 0

        #minimum transaction we can support. This is the implicit transaction
        #for all the 'tail values'
        self._min_transaction_num = 0

        self._types = {}
        #typename -> indexname -> fun(object->value)
        self._indices = {}

        #for each version number in _version_numbers, how many views referring to it
        self._version_number_counts = {}
        self._min_reffed_version_number = None

        #list of outstanding version numbers in increasing order where we have writes
        #_min_transaction_num is the minimum of these and the current transaction
        self._version_numbers = []

        #for each version number, a set of keys that were set
        self._version_number_objects = {}

        #for each key, a sorted list of version numbers outstanding and the relevant objects
        self._key_version_numbers = {}

        #for each (key, version), the object, as (json, actual_object)
        self._key_and_version_to_object = {}

        #for each key with versions, the value replaced by the oldest key. (json, actual_object)
        self._tail_values = {}

        #our parsed representation of each thing in the database
        self._current_database_object_cache = {}

        #cache-name to cache function
        self._calcCacheFunctions = {}

        #cache-key (cachename, (arg1,arg2,...)) -> cache value
        self._calcCacheValues = {}

        #cache-key -> tid before which this cache calc is definitely invalid.
        self._calcCacheMinTransactionId = {}

        #cache-key to keys that it depends on
        self._calcCacheKeysNeeded = {}

        #cache-key to other calcs that it depends on
        self._calcCacheToCalcCacheNeeded = {}

        #key -> set(cache_keys)
        self._keyToCacheKeyDepending = {}

        #cache_key -> set(cache_key), cache-keys depending on this one
        self._calcCacheToCalcCacheDepending = {}

    def clearCache(self):
        self._kvstore.clearCache()
        self._current_database_object_cache = {}

    def __str__(self):
        return "Database(%s)" % id(self)

    def __repr__(self):
        return "Database(%s)" % id(self)

    def current_transaction(self):
        if not hasattr(_cur_view, "view"):
            return None
        return _cur_view.view

    def addCalculationCache(self, name, function=None):
        self._calcCacheFunctions[name] = function or name

    def addIndex(self, type, prop, fun = None):
        if type.__qualname__ not in self._indices:
            self._indices[type.__qualname__] = {}

        if fun is None:
            fun = lambda o: getattr(o, prop)

        self._indices[type.__qualname__][prop] = fun

    def __setattr__(self, typename, val):
        if typename[:1] == "_":
            self.__dict__[typename] = val
            return
        
        self._types[typename] = val

    def __getattr__(self, typename):
        if typename[:1] == "_":
            return self.__dict__[typename]

        if typename not in self._types:
            class cls(DatabaseObject):
                pass

            cls._database = self
            cls.__qualname__ = typename

            self._types[typename] = cls

        return self._types[typename]

    def view(self, transaction_id=None):
        with self._lock:
            if transaction_id is None:
                transaction_id = self._cur_transaction_num

            assert transaction_id <= self._cur_transaction_num
            assert transaction_id >= self._min_transaction_num, transaction_id

            view = DatabaseView(self, transaction_id)

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
            view = DatabaseTransaction(self, self._cur_transaction_num)

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

            if not self._version_numbers:
                #views have caught up with the current transaction
                self._min_transaction_num = self._cur_transaction_num
                self._key_version_numbers = {}
                self._key_and_version_numbers = {}
                self._tail_values = {}
            else:
                self._min_transaction_num = lowest

                for key in keys_touched:
                    assert self._key_version_numbers[key][0] == lowest
                    if len(self._key_version_numbers[key]) == 1:
                        #this key is now current in the database
                        del self._key_version_numbers[key]
                        del self._key_and_version_to_object[key, lowest]

                        #it's OK to keep the key around if it's not None
                        del self._tail_values[key]
                    else:
                        self._key_version_numbers[key].pop(0)
                        self._tail_values[key] = self._key_and_version_to_object[key, lowest]
                        del self._key_and_version_to_object[key, lowest]

    def _update_versioned_object_data_cache(self, key, transaction_id, parsed_val):
        with self._lock:
            if key not in self._key_version_numbers:
                if key in self._tail_values:
                    self._tail_values[key] = (self._tail_values[key][0], parsed_val)
                else:
                    self._current_database_object_cache[key] = parsed_val
            else:
                #get the largest version number less than or equal to transaction_id
                version = self._best_version_for(transaction_id, self._key_version_numbers[key])

                if version is not None:
                    self._key_and_version_to_object[key, version] = (
                        self._key_and_version_to_object[key, version][0],
                        parsed_val
                        )
                else:
                    self._tail_values[key] = (
                        self._tail_values[key][0],
                        parsed_val
                        )


    def _get_versioned_object_data(self, key, transaction_id):
        with self._lock:
            assert transaction_id >= self._min_transaction_num

            if key not in self._key_version_numbers:
                if key in self._tail_values:
                    return self._tail_values[key]

                return (self._kvstore.get(key), self._current_database_object_cache.get(key))

            #get the largest version number less than or equal to transaction_id
            version = self._best_version_for(transaction_id, self._key_version_numbers[key])

            if version is not None:
                return self._key_and_version_to_object[key, version]
            else:
                return self._tail_values[key]

    def _best_version_for(self, transactionId, transactions):
        i = len(transactions) - 1

        while i >= 0:
            if transactions[i] <= transactionId:
                return transactions[i]
            i -= 1

        return None

    def _set_versioned_object_data(self, key_value, transaction_id):
        """Commit a transaction. 

        key_value: a map
            db_key -> (json_representation, database_representation)
        that we want to commit. We cache the normal_representation for later.
        """
        with self._lock:
            if transaction_id != self._cur_transaction_num:
                raise RevisionConflictException()

            self._cur_transaction_num += 1

            #we were viewing objects at the old transaction layer. now we write a new one.
            transaction_id += 1
            
            for key in key_value:
                #if this object is not versioned already, we need to keep the old value around
                if key not in self._key_version_numbers:
                    self._tail_values[key] = (self._kvstore.get(key), self._current_database_object_cache.get(key))

            #set the json representation in the database
            self._kvstore.setSeveral({k: v[0] for k,v in key_value.items()})
            for k,v in key_value.items():
                if v[1] is None:
                    if k in self._current_database_object_cache:
                        del self._current_database_object_cache[k]
                else:
                    self._current_database_object_cache[k] = v[1]

            #record what objects we touched
            self._version_number_objects[transaction_id] = list(key_value.keys())
            self._version_numbers.append(transaction_id)

            for key, value in key_value.items():
                if key not in self._key_version_numbers:
                    self._key_version_numbers[key] = []
                self._key_version_numbers[key].append(transaction_id)

                self._key_and_version_to_object[key,transaction_id] = value

                #invalidate the calculation cache
                if key in self._keyToCacheKeyDepending:
                    for cacheKey in list(self._keyToCacheKeyDepending[key]):
                        self._invalidateCachedCalc(cacheKey)

                    del self._keyToCacheKeyDepending[key]

    def _invalidateCachedCalc(self, cacheKey):
        if cacheKey not in self._calcCacheValues:
            return

        del self._calcCacheValues[cacheKey]
        del self._calcCacheToCalcCacheNeeded[cacheKey]
        del self._calcCacheMinTransactionId[cacheKey]
        keysToCheck = self._calcCacheKeysNeeded[cacheKey]
        del self._calcCacheKeysNeeded[cacheKey]

        calcsToCheck = self._calcCacheToCalcCacheDepending[cacheKey]
        del self._calcCacheToCalcCacheDepending[cacheKey]

        for k in keysToCheck:
            self._keyToCacheKeyDepending[k].discard(cacheKey)

        for k in calcsToCheck:
            self._invalidateCachedCalc(k)

    def lookupCachedCalculation(self, name, args):
        if not hasattr(_cur_view, "view"):
            raise Exception("Please access properties from within a view or transaction.")
        
        view = _cur_view.view
        
        #don't cache values from writeable views - too hard to keep track of 
        #whether we invalidated things mid-transaction.
        if view._writeable:
            return self._calcCacheFunctions[name](*args)

        if view._db is not self:
            raise Exception("Please use a view if you want to use caches")

        origReadWatcher = view._readWatcher
        if origReadWatcher:
            origReadWatcher("cache", (name, args))

        cacheKey = (name,args)
        with self._lock:
            if cacheKey in self._calcCacheMinTransactionId:
                minTID = self._calcCacheMinTransactionId.get(cacheKey)
                if minTID is None or minTID <= view._transaction_num:
                    #this is a cache hit!
                    return self._calcCacheValues[cacheKey]

        readKeys = set()
        readCaches = set()

        def readWatcher(readKind, readKey):
            if readKind == 'cache':
                readCaches.add(readKey)
            elif readKind == "key":
                readKeys.add(readKey)
            else:
                assert False 
        
        try:
            view._readWatcher = readWatcher

            cachedValue = self._calcCacheFunctions[name](*args)
        finally:
            view._readWatcher = origReadWatcher

        self._writeCachedValue((name, args), view._transaction_num, readKeys, readCaches, cachedValue)

        return cachedValue

    def _writeCachedValue(self, cacheKey, asOfTransId, readKeys, readCaches, cachedValue):
        with self._lock:
            minValidTID = None
            for key in readKeys:
                if key not in self._key_version_numbers:
                    #do nothing - there is only one version of this object around!
                    pass
                else:
                    tid = self._key_version_numbers[key][-1]
                    minValidTID = max(minValidTID, tid)
            for key in readCaches:
                if key not in self._calcCacheMinTransactionId:
                    #we read from a now-invalidated cache!
                    return
                else:
                    minValidTID = max(minValidTID, self._calcCacheMinTransactionId[key])

            if minValidTID > asOfTransId:
                return

            self._calcCacheMinTransactionId[cacheKey] = minValidTID
            self._calcCacheValues[cacheKey] = cachedValue
            self._calcCacheKeysNeeded[cacheKey] = readKeys
            for k in readKeys:
                if k not in self._keyToCacheKeyDepending:
                    self._keyToCacheKeyDepending[k] = set()
                self._keyToCacheKeyDepending[k].add(cacheKey)

            self._calcCacheToCalcCacheNeeded[cacheKey] = readCaches
            self._calcCacheToCalcCacheDepending[cacheKey] = set()

            for c in readCaches:
                self._calcCacheToCalcCacheDepending[c].add(cacheKey)


    def _get(self, obj_typename, identity, field_name, type):
        raise Exception("Please open a transaction or a view")

    def _set(self, obj, obj_typename, identity, field_name, type, val):
        raise Exception("Please open a transaction")


