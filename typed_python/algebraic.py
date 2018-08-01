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
from typed_python.types import IsTypeFilter, TypeConvert

import types

def valid_fieldname(name):
    return name and name[0] != "_" and name != 'matches' and name != "define"

def Alternative(name, **kwds):
    class Alternative(object):
        __typed_python_alternative__ = True
        __typed_python_type__ = True
        _frozen = False
        _subtypes = {}
        _common_fields = {}
        
        
        @classmethod
        def add_common_fields(cls, fields):
            assert not cls._frozen, "can't modify an Alternative once it has been frozen"
            
            for k,v in fields.items():
                cls.add_common_field(k,v)

        @classmethod
        def add_common_field(cls, k, v):
            assert not cls._frozen, "can't modify an Alternative once it has been frozen"

            cls._common_fields[k] = v

        @classmethod
        def define(cls, **kwds):
            for k,v in kwds.items():
                cls._define(k,v)

        @classmethod
        def __typed_python_try_convert_instance__(cls, value, allow_construct_new):
            if isinstance(value, cls):
                return (value,)
            return None
            
        @classmethod
        def _define(cls, alt_name, defs):
            assert not cls._frozen, "can't modify an Alternative once it has been frozen"

            assert alt_name not in cls._subtypes, "already have a definition for " + alt_name

            if isinstance(defs, dict):
                assert valid_fieldname(alt_name), "invalid alternative name: " + alt_name
            
                for fname, ftype in defs.items():
                    assert valid_fieldname(fname), "%s is not a valid field name" % fname
                
                for d in defs.values():
                    assert IsTypeFilter(d)

                defs = dict(defs)

                alt = makeAlternativeOption(cls, alt_name, defs)

                cls._subtypes[alt_name] = alt

                setattr(cls,alt_name, alt)
            else:
                setattr(cls,alt_name, defs)

        @classmethod
        def _freeze(cls):
            cls._frozen = True

        def __repr__(self):
            return "%s.%s(%s)" % (self._alternative.__qualname__, self._which, ",".join(["%s=%s" % (k,repr(self._fields[k])) for k in sorted(self._fields)]))

    Alternative.define(**kwds)
    Alternative.__qualname__ = name
    return Alternative

def makeAlternativeOption(Alternative, which, inTypedict):
    class AlternativeOption(Alternative):
        _typedict = inTypedict
        _which = which
        _alternative = Alternative

        def __init__(self, *args, **fields):
            Alternative._freeze()

            #make sure we don't modify caller dict
            fields = dict(fields)

            typedict = dict(AlternativeOption._typedict)
            typedict.update(Alternative._common_fields)

            if len(typedict) == 0 and len(args) == 1 and args[0] is None:
                fields = {}
            elif args:
                if len(typedict) == 1:
                    #if we have exactly one possible type, then don't need a name
                    assert not fields and len(args) == 1, "can't infer a name for more than one argument"
                    fields = {list(typedict.keys())[0]: args[0]}
                else:
                    raise TypeError("constructing %s with an extra unnamed argument" % (alternative.__qualname__ + "." + which))

            for f in fields:
                if f not in typedict:
                    raise TypeError("constructing with unused argument %s: %s vs %s" % (f, fields.keys(), typedict.keys()))

            for k in typedict:
                if k not in fields:
                    raise TypeError("Can't default initialize %s" % k)
                else:
                    instance = TypeConvert(typedict[k], fields[k], allow_construct_new=True)
                
                fields[k] = instance

            for k,v in fields.items():
                self.__dict__[k] = v

            self._fields = fields
            self._hash = None
            self._sha_hash_cache = None

        def _withReplacement(self, **kwargs):
            fields = self._fields
            fields.update(kwargs)

            return AlternativeOption(**fields)

        def __sha_hash__(self):
            if self._sha_hash_cache is None:
                base_hash = sha_hash(Alternative.__qualname__)
                
                for fieldname in sorted(self._fields):
                    val = self._fields[fieldname]
                    base_hash = base_hash + sha_hash(fieldname) + sha_hash(val)

                self._sha_hash_cache = base_hash + sha_hash(self._which)
            return self._sha_hash_cache

        def __hash__(self):
            if self._hash is None:
                self._hash = hash(self.__sha_hash__())
            return self._hash

        @property
        def matches(self):
            return AlternativeInstanceMatches(self)

        def __setattr__(self, attr, val):
            if attr[:1] != "_":
                raise Exception("Field %s is read-only" % attr)
            self.__dict__[attr] = val

        def __str__(self):
            return repr(self)

        def __ne__(self, other):
            return not self.__eq__(other)

        def __eq__(self, other):
            if self is other:
                return True

            if not isinstance(other, AlternativeOption):
                return False

            if self._which != other._which:
                return False
            
            if hash(self) != hash(other):
                return False
            
            for f in sorted(self._fields):
                if getattr(self,f) != getattr(other,f):
                    return False
            
            return True

    AlternativeOption.__qualname__ = Alternative.__qualname__ + "." + which

    return AlternativeOption

class AlternativeInstanceMatches(object):
    def __init__(self, instance):
        object.__init__(self)

        self._instance = instance

    def __getattr__(self, attr):
        if self._instance._which == attr:
            return True
        return False
