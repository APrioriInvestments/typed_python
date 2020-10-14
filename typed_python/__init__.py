#   Copyright 2017-2019 typed_python Authors
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

__version__ = "0.1.1"

from typed_python.internals import (
    Member, Final, Function, UndefinedBehaviorException,
    makeNamedTuple, DisableCompiledCode, isCompiled, Held,
    typeKnownToCompiler,
    localVariableTypesKnownToCompiler,
)
from typed_python._types import bytecount, refcount
from typed_python.module import Module
from typed_python.type_function import TypeFunction
from typed_python.hash import sha_hash
from typed_python.SerializationContext import SerializationContext
from typed_python.type_filter import TypeFilter
from typed_python._types import (
    Forward, TupleOf, ListOf, Tuple, NamedTuple, OneOf, ConstDict,
    Alternative, Value, serialize, deserialize, serializeStream, deserializeStream,
    PointerTo, RefTo, Dict, validateSerializedObject, validateSerializedObjectStream, decodeSerializedObject,
    getOrSetTypeResolver, Set, Class, Type, BoundMethod,
    TypedCell, pointerTo, refTo, copy, identityHash
)
import typed_python._types as _types

# in the c module, these are functions, but because they're not parametrized,
# we want them to be actual values.
Int8 = _types.Int8()
Int16 = _types.Int16()
Int32 = _types.Int32()
UInt8 = _types.UInt8()
UInt16 = _types.UInt16()
UInt32 = _types.UInt32()
UInt64 = _types.UInt64()
Float32 = _types.Float32()
EmbeddedMessage = _types.EmbeddedMessage()
PyCell = _types.PyCell()

from typed_python.compiler.runtime import Entrypoint, Compiled, NotCompiled, Runtime  # noqa

# this has to come at the end to break import cyclic
from typed_python.lib.map import map  # noqa
from typed_python.lib.pmap import pmap  # noqa
from typed_python.lib.reduce import reduce  # noqa

_types.initializeGlobalStatics()
