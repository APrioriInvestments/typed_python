#   Copyright 2017 Braxton Mckee
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

"""
Basic infrastructure for typed union datastructures in python
"""

def valid_type(t):
    if isinstance(t, Alternative) or t in [str, int, bool, float]:
        return True

    if isinstance(t, tuple) and t:
        for sub_t in t:
            if not valid_type(sub_t):
                return False
        return True
    
    if isinstance(t, List):
        return True

    return False

def coerce_instance(instance, to_type):
    if isinstance(to_type, Alternative):
        if isinstance(instance, AlternativeInstance):
            if instance._alternative is to_type:
                return instance
        try:
            return to_type(instance)
        except TypeError as e:
            return None
    elif isinstance(to_type, tuple):
        if not isinstance(instance, tuple) or len(instance) != len(to_type):
            return None
        res = []
        for i in xrange(len(instance)):
            coerced = coerce_instance(instance[i], to_type[i])
            if coerced is None:
                return None
            res.append(coerced)
        return tuple(res)
    elif isinstance(to_type, List):
        try:
            i = iter(instance)
        except TypeError as e:
            return None

        res = []
        while True:
            try:
                val = coerce_instance(i.next(), to_type.subtype)
                if val is None:
                    return None
                res.append(val)
            except StopIteration as e:
                return tuple(res)
    else:
        if isinstance(instance, to_type):
            return instance

def valid_fieldname(name):
    return name and name[0] != "_" and name != 'matches' and name != "define"

class Discard:
    def x(self):
        pass

boundinstancemethod = type(Discard().x)

class Alternative(object):
    def __init__(self, name, **kwds):
        object.__init__(self)
        self._name = name
        self._types = {}
        self._options = {}
        self._instantiated = False
        self._methods = {}
        
        for k, v in kwds.iteritems():
            self.__setattr__(k,v)

    def define(self, **kwds):
        for k,v in kwds.iteritems():
            self.__setattr__(k,v)

    def __setattr__(self, alt_name, defs):
        if len(alt_name) >= 2 and alt_name[0] == "_" and alt_name[1] != "_":
            self.__dict__[alt_name] = defs
            return

        assert not self._instantiated, "can't modify an Alternative once it has been used"

        assert alt_name not in self._types, "already have a definition for " + alt_name

        if isinstance(defs, dict):
            assert valid_fieldname(alt_name), "invalid alternative name: " + alt_name
        
            for fname, ftype in defs.iteritems():
                assert valid_fieldname(fname), "%s is not a valid field name" % fname
                assert valid_type(ftype), "%s is not a valid type" % ftype

            self._types[alt_name] = dict(defs)
        else:
            self._methods[alt_name] = defs

    def __getattr__(self, attr):
        if attr[0] == "_":
            raise AttributeError(attr)

        if attr not in self._types:
            raise AttributeError(attr + " not a valid Alternative in %s" % sorted(self._types))

        if attr not in self._options:
            self._options[attr] = makeAlternativeOption(self, attr, self._types[attr])

        return self._options[attr]

    def __call__(self, *args, **kwds):
        if len(self._types) == 1:
            #there's only one option - no need for any coersion
            return getattr(self, list(self._types)[0])(*args, **kwds)
        else:
            #only allow possibilities by 'arity' and name matching
            possibility = None

            if len(args) == 1 and args[0] is None:
                args = []

            if len(args) == 1:
                assert(len(kwds) == 0)
                for typename,typedict in self._types.iteritems():
                    if len(typedict) == 1:
                        if possibility is not None:
                            raise TypeError("coersion to %s with one unnamed argument is ambiguous" % self._name)
                        possibility = typename
            else:
                assert(len(args) == 0)

                #multiple options, so it's a little ambiguous
                for typename,typedict in self._types.iteritems():
                    if sorted(typedict) == sorted(kwds):
                        if possibility is not None:
                            raise TypeError("coersion to %s with one unnamed argument is ambiguous" % self._name)
                        possibility = typename

            if possibility is not None:
                return getattr(self,possibility)(*args, **kwds)
            else:
                raise TypeError("coersion to %s with one unnamed argument is ambiguous" % self._name)

    def __str__(self):
        return "algebraic.Alternative(%s)" % self._name

class List(object):
    def __init__(self, subtype):
        self.subtype = subtype
        assert valid_type(subtype)

class AlternativeInstance(object):
    def __init__(self):
        object.__init__(self)

def makeAlternativeOption(alternative, which, typedict):
    class AlternativeOption(AlternativeInstance):
        def __init__(self, *args, **fields):
            AlternativeInstance.__init__(self)

            #make sure we don't modify caller dict
            fields = dict(fields)

            if len(typedict) == 0 and len(args) == 1 and args[0] is None:
                fields = {}
            elif args:
                if len(typedict) == 1:
                    #if we have exactly one possible type, then don't need a name
                    assert not fields and len(args) == 1, "can't infer a name for more than one argument"
                    fields = {list(typedict.keys())[0]: args[0]}
                else:
                    raise TypeError("constructing %s with an extra unnamed argument" % (alternative._name + "." + which))

            for f in fields:
                if f not in typedict:
                    raise TypeError("constructing with unused argument %s" % f)

            for k in typedict:
                if k not in fields:
                    raise TypeError("missing field %s" % k)
                instance = coerce_instance(fields[k], typedict[k])
                if instance is None:
                    raise TypeError("field %s needs a %s, not %s" % (k, typedict[k], fields[k]))
                fields[k] = instance

            self._fields = fields
            self._which = which
            self._alternative = alternative
            self._hash = None

        def __hash__(self):
            if self._hash is None:
                self._hash = hash(tuple(sorted(self._fields.iteritems())))
            return self._hash

        @property
        def matches(self):
            return AlternativeInstanceMatches(self)

        def __getattr__(self, attr):
            if attr in self._fields:
                return self._fields[attr]

            if attr in self._alternative._methods:
                return boundinstancemethod(self._alternative._methods[attr], self)

            raise AttributeError("%s not found amongst %s" % (attr, ",".join(list(self._fields) + list(self._alternative._methods))))

        def __add__(self, other):
            if '__add__' in self._alternative._methods:
                return self._alternative._methods['__add__'](self, other)
            raise TypeError("unsupported operand type(s) for +: '%s' and '%s'" % (type(self),type(other)))

        def __str__(self):
            if '__str__' in self._alternative._methods:
                return self._alternative._methods['__str__'](self)
            return repr(self)

        def __repr__(self):
            if '__repr__' in self._alternative._methods:
                return self._alternative._methods['__repr__'](self)
            return "%s.%s(%s)" % (self._alternative._name, self._which, ",".join(["%s=%s" % (k,repr(self._fields[k])) for k in sorted(self._fields)]))

        def __cmp__(self, other):
            if not isinstance(other, AlternativeOption):
                return cmp(type(self), type(other))
            if self._which != other._which:
                return cmp(self._which, other._which)
            for f in sorted(self._fields):
                c = cmp(getattr(self,f), getattr(other,f))
                if c != 0:
                    return c
            return 0

    AlternativeOption.__name__ = alternative._name + "." + which

    return AlternativeOption

class AlternativeInstanceMatches(object):
    def __init__(self, instance):
        object.__init__(self)

        self._instance = instance

    def __getattr__(self, attr):
        if self._instance._which == attr:
            return True
        if attr not in self._instance._alternative._types:
            raise AttributeError("Matcher %s would never match one of %s's fields %s" % (attr, self._instance._alternative._name, sorted(self._instance._alternative._types)))
        return False

_nullable_cache = {}
def Nullable(alternative):
    if alternative not in _nullable_cache:
        _nullable_cache[alternative] = Alternative("Nullable(" + str(alternative) + ")", Null={}, Value={'val': alternative})

    return _nullable_cache[alternative]
