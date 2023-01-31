#   Copyright 2017-2020 typed_python Authors
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

from typed_python import (
    Entrypoint,
    SerializationContext,
    ListOf,
    Value
)
from typed_python._types import (
    serialize as serializeNative,
    deserialize as deserializeNative
)

import threading
import time


def test_can_serialize_under_entrypoint():
    s = SerializationContext().withoutCompression()

    def serialize(s, inst, T):
        return s.serialize(inst, T)

    serializeCompiled = Entrypoint(serialize)

    def check(inst, T):
        assert serialize(s, inst, T) == serializeCompiled(s, inst, T)

    assert serializeCompiled.resultTypeFor(SerializationContext, str, Value(str)).typeRepresentation is bytes

    check('hi', str)
    check(ListOf(str)(['hi']), ListOf(str))


def test_can_deserialize_under_entrypoint():
    s = SerializationContext().withoutCompression()

    def deserialize(s, bytes, T):
        return s.deserialize(bytes, T)

    deserializeCompiled = Entrypoint(deserialize)

    def check(inst, T):
        bytes = s.serialize(inst, T)
        deserializedNormally = deserialize(s, bytes, T)
        deserializedCompiled = deserializeCompiled(s, bytes, T)
        assert deserializedNormally == deserializedCompiled

    assert deserializeCompiled.resultTypeFor(
        SerializationContext,
        bytes,
        Value(str)
    ).typeRepresentation is str

    check('hi', str)
    check(ListOf(str)(['hi']), ListOf(str))


def test_can_serialize_object_under_entrypoint():
    s = SerializationContext()

    def serialize(o):
        return s.serialize(o)

    def deserialize(data):
        return s.deserialize(data)

    serializeC = Entrypoint(serialize)
    deserializeC = Entrypoint(deserialize)

    assert serialize(None) == serializeC(None)
    assert deserialize(serialize(None)) is None
    assert deserializeC(serialize(None)) is None


def timeInNThreads(f, args, threadCount):
    threads = [threading.Thread(target=f, args=args, daemon=True) for _ in range(threadCount)]

    t0 = time.time()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return time.time() - t0


def test_serialization_perf():
    s = SerializationContext().withoutCompression()

    def countBytes(aStr: str, times):
        res = 0

        for _ in range(times):
            res += len(s.deserialize(s.serialize(aStr, str), str))

        return res

    countBytesCompiled = Entrypoint(countBytes)

    assert countBytes("asdf", 1) == countBytesCompiled("asdf", 1)

    N = 10000
    C_RATIO = 20
    TC = 2

    interpTime = timeInNThreads(countBytes, ("asdf", N), 1)
    compiledTime = timeInNThreads(countBytesCompiled, ("asdf", N * C_RATIO), 1) / C_RATIO

    speedup = interpTime / compiledTime

    print(f"speedup is {speedup} ({compiledTime} compiled vs {interpTime} interpreted)")

    # I'm getting around 2.5
    assert speedup > 1.5

    compiledTime2 = timeInNThreads(countBytesCompiled, ("asdf", N * C_RATIO), TC) / C_RATIO

    interpTime2 = timeInNThreads(countBytes, ("asdf", N), TC)

    speedup2 = interpTime2 / compiledTime2

    print(f"for {TC} threads, speedup is {speedup2} ({compiledTime2} compiled vs {interpTime2} interpreted)")

    assert .8 * TC < (interpTime2 / interpTime) < 1.2 * TC

    compiledThreadScaling = compiledTime2 / compiledTime

    print(f"in the compiler, {TC} threads scaled time by {compiledThreadScaling}")
    assert compiledThreadScaling < 1.5


def test_can_serialize_with_or_without_out_context():
    @Entrypoint
    def roundtripCompiled(T, x, sc):
        assert deserializeNative(T, serializeNative(T, x, sc), sc) == x

    @Entrypoint
    def roundtripCompiledAmbiguous(T, x, sc: object):
        assert deserializeNative(T, serializeNative(T, x, sc), sc) == x

    s = SerializationContext()

    roundtripCompiled(str, "asdf", None)
    roundtripCompiledAmbiguous(str, "asdf", None)

    roundtripCompiled(str, "asdf", s)
    roundtripCompiledAmbiguous(str, "asdf", s)
