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

from typed_python import Alternative, TupleOf, ConstDict, OneOf

import logging
import json

class Encoder(object):
    """An algebraic <---> json encoder.

    The encoding is:
        * primitive types (str, int, bool, float) are encoded directly
        * OneOfs are encoded as a typename and the value
        * Alternatives are encoded as objects.
        * Lists and Tuples are encoded as arrays
    """    
    def __init__(self):
        object.__init__(self)
        self.overrides = {}

        #if True, then we ignore extra fields that don't correspond to valid fields
        #thisallows us to 
        self.allowExtraFields=False

    def to_json(self, object_type, value):
        if isinstance(object_type, OneOf):
            if value is None or isinstance(value, (int,float,str,bool)):
                return value
            return { 'type': str(type(value)), 'value': self.to_json(type(value), value) }

        if hasattr(object_type, '__typed_python_alternative__'):
            if not value._fields:
                return value._which
            else:
                fields = {}

                for fieldname, ftype in type(value)._typedict.items():
                    fields[fieldname] = self.to_json(ftype, value._fields[fieldname])

                for fieldname, ftype in object_type._common_fields.items():
                    fields[fieldname] = self.to_json(ftype, value._fields[fieldname])
                
                return {
                    'type': value._which, 
                    'fields': fields
                    }

        elif isinstance(value, TupleOf):
            return tuple(self.to_json(value.ElementType, x) for x in value)

        elif isinstance(value, ConstDict):
            return tuple((self.to_json(value.KeyType, k), self.to_json(value.ValueType, value[k])))

        elif isinstance(value, (int,float,bool,str)) or value is None:
            return value
        elif hasattr(type(value), "to_json"):
            return type(value).to_json(value)
        else:
            assert False, "Can't convert %s to %s" % (value,object_type)

    def from_json(self, value, algebraic_type):
        if algebraic_type in self.overrides:
            return self.overrides[algebraic_type](self, algebraic_type, value)
        
        try:
            if isinstance(algebraic_type, OneOf):
                which_type

            if isinstance(algebraic_type, ConstDict):
                return algebraic_type(
                    {self.from_json(k, algebraic_type.KeyType):self.from_json(v, algebraic_type.ValueType) for k,v in value}
                    )

            if isinstance(algebraic_type, TupleOf):
                return algebraic_type(self.from_json(v, algebraic_type.ElementType) for v in value)

            if algebraic_type in (bool, int, str, float):
                return value

            if hasattr(algebraic_type, '__typed_python_alternative__'):
                if isinstance(value, str):
                    return getattr(algebraic_type, value)()

                alt_type = getattr(algebraic_type, value['type'])

                fields = {}

                for fieldname, ftype in alt_type._typedict.items():
                    fields[fieldname] = self.from_json(value["fields"][fieldname], ftype)

                for fieldname, ftype in alt_type._common_fields.items():
                    fields[fieldname] = self.from_json(value["fields"][fieldname], ftype)

                return alt_type(**fields)
            
            if hasattr(algebraic_type, "from_json"):
                return algebraic_type.from_json(value)

            assert False, "Can't handle type %s as value %s" % (algebraic_type,value)
        except:
            logging.error("Parsing error making %s:\n%s", algebraic_type, json.dumps(value,indent=2))
            raise

