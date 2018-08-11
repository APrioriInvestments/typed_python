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
import time
import traceback
import uuid

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
        self._set_adds = {}
        self._set_removes = {}
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
        
        if not hasattr(cls, '__types__') or cls.__types__ is None:
            raise Exception("Please initialize the type object for %s" % str(cls.__qualname__))

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

                    self._add_to_index(ik, identity)

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
                        self._remove_from_index(old_index_name, identity)

                    if new_index_val is not None:
                        new_index_name = index_key(obj_typename, index_name, new_index_val)
                        self._add_to_index(new_index_name, identity)

    def _add_to_index(self, index_key, identity):
        assert isinstance(identity, str)

        if index_key not in self._set_adds:
            self._set_adds[index_key] = set()
            self._set_removes[index_key] = set()
            
        self._set_adds[index_key].add(identity)
        self._set_removes[index_key].discard(identity)

    def _remove_from_index(self, index_key, identity):
        assert isinstance(identity, str)

        if index_key not in self._set_adds:
            self._set_adds[index_key] = set()
            self._set_removes[index_key] = set()
            
        self._set_adds[index_key].discard(identity)
        self._set_removes[index_key].add(identity)

    
    def indexLookup(self, type, **kwargs):
        assert len(kwargs) == 1, "Can only lookup one index at a time."
        tname, value = list(kwargs.items())[0]

        if type.__qualname__ not in self._db._indices or tname not in self._db._indices[type.__qualname__]:
            raise Exception("No index enabled for %s.%s" % (type.__qualname__, tname))

        if not hasattr(_cur_view, "view"):
            raise Exception("Please access indices from within a view.")

        indexType = self._db._indexTypes[type.__qualname__][tname]
        if indexType is not None:
            value = TypeConvert(indexType, value, allow_construct_new=True)

        keyname = index_key(type.__qualname__, tname, value)

        identities = set(self._db._get_versioned_set_data(keyname, self._transaction_num))
        identities = identities.union(self._set_adds.get(keyname, set()))
        identities = identities.difference(self._set_removes.get(keyname, set()))

        for i in identities:
            assert isinstance(i, str)
            assert '"' not in i

        return tuple([type(x) for x in identities])

    def indexLookupAny(self, type, **kwargs):
        assert len(kwargs) == 1, "Can only lookup one index at a time."
        tname, value = list(kwargs.items())[0]

        if type.__qualname__ not in self._db._indices or tname not in self._db._indices[type.__qualname__]:
            raise Exception("No index enabled for %s.%s" % (type.__qualname__, tname))

        if not hasattr(_cur_view, "view"):
            raise Exception("Please access indices from within a view.")

        indexType = self._db._indexTypes[type.__qualname__][tname]

        if indexType is not None:
            value = TypeConvert(indexType, value, allow_construct_new=True)

        keyname = index_key(type.__qualname__, tname, value)

        added = self._set_adds.get(keyname, set())
        removed = self._set_removes.get(keyname, set())

        if added:
            return type(list(added)[0])

        for val in self._db._get_versioned_set_data(keyname, self._transaction_num):
            assert isinstance(val, str)

            if val not in removed:
                return type(val)

        return None

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
            
            self._db._set_versioned_object_data(writes, self._set_adds, self._set_removes, tid)

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

class Transaction(View):
    _writeable = True
