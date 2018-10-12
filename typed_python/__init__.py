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

__version__="0.2"

import typed_python._types

from typed_python.hash import sha_hash
from typed_python._types import TupleOf, Tuple, NamedTuple, OneOf, ConstDict, \
                                Alternative, Value, serialize, deserialize, Int8, \
                                Int16, Int32, UInt8, UInt32, UInt64, NoneType


class ClassMetaNamespace:
    def __init__(self):
        self.ns = {}
        self.order = []

    def __getitem__(self, k):
        return self.ns[k]

    def __setitem__(self, k, v):
        self.ns[k] = v
        self.order.append((k,v))

class ClassMetaclass(type):
    @classmethod
    def __prepare__(cls, *args, **kwargs):
        return ClassMetaNamespace()

    def __new__(cls, name, bases, namespace, **kwds):
        if not bases:
            return type.__new__(cls, name,bases, namespace.ns, **kwds)

        return typed_python._types.Class(name, tuple(namespace.order))

class Class(metaclass=ClassMetaclass):
    """Base class for all typed python Class objects."""
    pass
