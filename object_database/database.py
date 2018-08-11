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

    def versionFor(self, version):
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

    def pickAny(self):
        removed = set()

        for i in reversed(range(len(self.adds))):
            for a in self.adds[i]:
                if a not in removed:
                    return a
            removed.update(self.removes[i])

        for a in self.s:
            if a not in removed:
                return a


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

        #for each key, a VersionedValue or VersionedSet
        self._versioned_objects = {}

    def clearCache(self):
        self._kvstore.clearCache()
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
        assert cls.__name__[:1] != "_", "Illegal to use _ for first character in databse classnames."

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

            self._min_transaction_num = lowest

            for key in keys_touched:
                self._versioned_objects[key].cleanup(lowest)
                
    def _get_versioned_set_data(self, key, transaction_id):
        with self._lock:
            assert transaction_id >= self._min_transaction_num

            if key not in self._versioned_objects:
                members = self._kvstore.setMembers(key)

                self._versioned_objects[key] = VersionedSet(members)

            #get the largest version number less than or equal to transaction_id
            return self._versioned_objects[key].versionFor(transaction_id)

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
        with self._lock:
            assert as_of_version >= self._min_transaction_num

            self._cur_transaction_num += 1
            transaction_id = self._cur_transaction_num
            assert transaction_id > as_of_version

            for key in key_value:
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
                    if k not in self._versioned_objects:
                        self._versioned_objects[k] = VersionedSet(
                            self._kvstore.setMembers(k)
                            )

            for subset in [keys_to_check_versions, indices_to_check_versions]:
                for key in subset:
                    if not self._versioned_objects[key].validVersionIncoming(as_of_version, transaction_id):
                        raise RevisionConflictException()

            #set the json representation in the database
            self._kvstore.setSeveral({k: v.jsonRep for k,v in key_value.items()}, set_adds, set_removes)

            for k,v in key_value.items():
                self._versioned_objects[k].setVersionedValue(transaction_id, v)

            for k,a in set_adds.items():
                if a:
                    self._versioned_objects[k].setVersionedAddsAndRemoves(transaction_id, a, set())
            for k,r in set_removes.items():
                if r:
                    self._versioned_objects[k].setVersionedAddsAndRemoves(transaction_id, set(), r)

            #record what objects we touched
            self._version_number_objects[transaction_id] = list(key_value.keys())
            self._version_numbers.append(transaction_id)

