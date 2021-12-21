#   Copyright 2017-2021 typed_python Authors
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

__version__ = "0.2.4"

# incremented every time the way we serialize a fully-typed
# TP object changes. Consumers who only serialize fully-typed
# instances, where no fields have type 'object' or 'type', and so
# therefore don't depend in any way on looking up types in the serialized
# stream can depend on this number to indicate whether their binaries
# will be out of date.
__fully_typed_serialization_version__ = 1

# incremented every time we change the way we serialize type objects
# or look them up in a codebase. Clients that serialize 'object' and 'type'
# and who want a way to check that they are reading the correct protocol
# can check this version.
__untyped_serialization_version__ = 4

from typed_python.internals import (
    Member, Final, Function, UndefinedBehaviorException,
    makeNamedTuple, DisableCompiledCode, isCompiled, Held,
    typeKnownToCompiler,
    localVariableTypesKnownToCompiler,
    checkOneOfType,
    checkType
)
from typed_python._types import bytecount, refcount
from typed_python.module import Module
from typed_python.type_function import TypeFunction
from typed_python.hash import sha_hash
from typed_python.SerializationContext import SerializationContext
from typed_python.type_filter import TypeFilter
from typed_python.compiler.typeof import TypeOf
from typed_python._types import (
    Forward, TupleOf, ListOf, Tuple, NamedTuple, OneOf, ConstDict, SubclassOf,
    Alternative, Value, serialize, deserialize, serializeStream, deserializeStream,
    PointerTo, RefTo, Dict, validateSerializedObject, validateSerializedObjectStream, decodeSerializedObject,
    getOrSetTypeResolver, Set, Class, Type, BoundMethod,
    TypedCell, pointerTo, refTo, copy, identityHash,
    deepBytecount, deepcopy, deepcopyContiguous, totalBytesAllocatedInSlabs,
    deepBytecountAndSlabs, Slab,
    totalBytesAllocatedOnFreeStore,
    ModuleRepresentation
)
import typed_python._types as _types
import threading

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

from typed_python.generator import Generator  # noqa

# this has to come at the end to break import cyclic
from typed_python.lib.map import map  # noqa
from typed_python.lib.pmap import pmap  # noqa
from typed_python.lib.reduce import reduce  # noqa

_types.initializeGlobalStatics()

# start a background thread to release the GIL for us. Instead of immediately releasing the GIL,
# we prefer to release it a short time after our C code no longer needs it, in case it
# reacquires it immediately, which is very slow.
# this is needed primarily because we often can't tell whether we're about to enter code
# that acquires and releases the GIL very frequently (say, in a tight loop), which has
# a huge performance penalty (a few thousand context switches per second!)
# instead, we have a thread that checks in the background whether any thread wants us
# to release, and if so, swap it out. This can yield a 10-50x performance improvement
# when we're acquiring and releasing frequently.
gilReleaseThreadLoop = threading.Thread(target=_types.gilReleaseThreadLoop, daemon=True)
gilReleaseThreadLoop.start()
