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
from object_database.view import View, Transaction, _cur_view

import inspect
from typed_python.hash import sha_hash
from typed_python import Tuple
from types import FunctionType

import threading
import logging
import uuid
import traceback
import time

class RevisionConflictException(Exception):
    pass

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
        self._indexTypes = {}

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

    def addIndex(self, type, prop, fun = None, index_type = None):
        if type.__qualname__ not in self._indices:
            self._indices[type.__qualname__] = {}
            self._indexTypes[type.__qualname__] = {}

        if fun is None:
            fun = lambda o: getattr(o, prop)
            index_type = type.__types__[prop]
        else:
            spec = inspect.getfullargspec(fun)
            index_type = spec.annotations.get('return', None)

        self._indices[type.__qualname__][prop] = fun
        self._indexTypes[type.__qualname__][prop] = index_type

    def __setattr__(self, typename, val):
        if typename[:1] == "_":
            self.__dict__[typename] = val
            return
        
        self._types[typename] = val

    def define(self, cls):
        t = getattr(self, cls.__name__)
        
        types = {}
        
        for name, val in cls.__dict__.items():
            if name[:2] != '__' and isinstance(val, type):
                types[name] = val
            elif name[:2] != '__' and isinstance(val, Indexed):
                if isinstance(val.obj, type):
                    types[name] = val.obj

        t.define(**types)

        for name, val in cls.__dict__.items():
            if isinstance(val, Index):
                self.addIndex(t, name, val, Tuple(*tuple(types[k] for k in val.names)))

            if name[:2] != '__' and isinstance(val, Indexed):
                if isinstance(val.obj, FunctionType):
                    self.addIndex(t, name, val.obj)
                    setattr(t, name, val.obj)
                else:
                    self.addIndex(t, name)
            elif (not name.startswith("__") or name in ["__str__", "__repr__"]):
                if isinstance(val, FunctionType):
                    setattr(t, name, val)

        return t

    def __getattr__(self, typename):
        assert '.' not in typename

        if typename[:1] == "_":
            return self.__dict__[typename]

        if typename not in self._types:
            class cls(DatabaseObject):
                pass

            cls._database = self
            cls.__qualname__ = typename

            self._types[typename] = cls
            self._indices[cls.__qualname__] = {' exists': lambda e: True}
            self._indexTypes[cls.__qualname__] = {' exists': bool}

        return self._types[typename]

    def view(self, transaction_id=None):
        with self._lock:
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

    def _get_versioned_set_data(self, key, transaction_id):
        with self._lock:
            assert transaction_id >= self._min_transaction_num

            if key not in self._key_version_numbers:
                if key in self._tail_values:
                    return self._tail_values[key][0]

                res = self._kvstore.setMembers(key)

                return res

            #get the largest version number less than or equal to transaction_id
            version = self._best_version_for(transaction_id, self._key_version_numbers[key])

            if version is not None:
                return self._key_and_version_to_object[key, version][0]
            else:
                return self._tail_values[key][0]

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

    def _set_versioned_object_data(self, key_value, set_adds, set_removes, transaction_id):
        """Commit a transaction. 

        key_value: a map
            db_key -> (json_representation, database_representation)
        that we want to commit. We cache the normal_representation for later.

        set_adds: a map:
            db_key -> set of identities added to an index
        set_removes: a map:
            db_key -> set of identities removed fromf an index
        """
        for s in set_adds.values():
            for val in s:
                assert isinstance(val, str)
                assert '"' not in val

        for s in set_removes.values():
            for val in s:
                assert isinstance(val, str)
                assert '"' not in val
        
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

            for key in set(list(set_adds) + list(set_removes)):
                #if this object is not versioned already, we need to keep the old value around
                if key not in self._key_version_numbers:
                    self._current_database_object_cache[key] = (set(self._kvstore.setMembers(key)), None)
                    self._tail_values[key] = (self._current_database_object_cache[key], self._current_database_object_cache[key])

            #set the json representation in the database
            self._kvstore.setSeveral({k: v[0] for k,v in key_value.items()}, set_adds, set_removes)

            for k,v in key_value.items():
                if v[1] is None:
                    if k in self._current_database_object_cache:
                        del self._current_database_object_cache[k]
                else:
                    self._current_database_object_cache[k] = v[1]

            for key, ids_added in set_adds.items():
                if key not in self._current_database_object_cache:
                    self._current_database_object_cache[key] = (set(), None)
                self._current_database_object_cache[key][0].update(ids_added)

            for key, ids_removed in set_removes.items():
                if key not in self._current_database_object_cache:
                    self._current_database_object_cache[key] = (set(), None)
                self._current_database_object_cache[key][0].difference_update(ids_removed)

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

            for key in set(list(set_adds) + list(set_removes)):
                if key not in self._key_version_numbers:
                    self._key_version_numbers[key] = []
                self._key_version_numbers[key].append(transaction_id)

                self._key_and_version_to_object[key,transaction_id] = self._current_database_object_cache[key]

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


