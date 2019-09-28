#   Coyright 2017-2019 Nativepython Authors
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

__version__ = "0.2"

from typed_python.internals import Member, Final, Function, UndefinedBehaviorException, makeNamedTuple
from typed_python.type_function import TypeFunction
from typed_python.hash import sha_hash
from typed_python.SerializationContext import SerializationContext
from typed_python.type_filter import TypeFilter
from typed_python._types import (
    Forward, TupleOf, ListOf, Tuple, NamedTuple, OneOf, ConstDict,
    Alternative, Value, serialize, deserialize, serializeStream, deserializeStream,
    PointerTo, Dict, validateSerializedObject, validateSerializedObjectStream, decodeSerializedObject,
    getOrSetTypeResolver, Set, Class, Type, PythonObjectOfType
)
import typed_python._types as _types

# in the c module, these are functions, but because they're not parametrized,
# we want them to be actual values. Otherwise, we'll have 'Float64()'
# where we would have written 'Float64' etc.
Bool = _types.Bool()
Int8 = _types.Int8()
Int16 = _types.Int16()
Int32 = _types.Int32()
Int64 = _types.Int64()
UInt8 = _types.UInt8()
UInt16 = _types.UInt16()
UInt32 = _types.UInt32()
UInt64 = _types.UInt64()
Float32 = _types.Float32()
Float64 = _types.Float64()
NoneType = _types.NoneType()
String = _types.String()
Bytes = _types.Bytes()
EmbeddedMessage = _types.EmbeddedMessage()
