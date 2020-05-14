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

from typed_python import (
    Float32, Int8, Int16, Int32, UInt8,
    UInt16, UInt32, UInt64
)

arithmetic_types = (
    Float32, Int8, Int16, Int32, UInt8, int, float, bool,
    UInt16, UInt32, UInt64
)

_intsBySignednessAndBitness = {
    (True, 8): Int8,
    (True, 16): Int16,
    (True, 32): Int32,
    (True, 64): int,
    (False, 8): UInt8,
    (False, 16): UInt16,
    (False, 32): UInt32,
    (False, 64): UInt64
}


def floatness(T):
    if T in (float, Float32):
        return True
    return False


def bitness(T):
    if T in (float, int):
        return 64
    if T is bool:
        return 1
    return T.Bits


def signedness(T):
    if T in (int, Int8, Int16, Int32):
        return True
    if T in (bool, UInt8, UInt16, UInt32, UInt64):
        return False
    assert False


def isSignedInt(T):
    return T in (int, Int8, Int16, Int32)


def isUnsignedInt(T):
    return T in (bool, UInt8, UInt16, UInt32, UInt64)


def isInteger(T):
    return isSignedInt(T) or isUnsignedInt(T)


def computeArithmeticBinaryResultType(T1, T2):
    """Compute the promoted type for a binary operation on two arithmetic types.
    This is the type we use when actually performing the operation (in most cases).
    For arithmetic operations, this is usually the result type. For comparison
    operations, this is the type that's used to produce a boolean value for the
    result.

    The basic rule is to pick the max number of bits between the two types,
    for the result to be signed if either value is signed, and for it to be
    a floating point number if either value is a floating point number.

    Note that we don't follow c++ here, where a comparision between an unsigned
    and a signed integer is implemented as an unsigned comparison. That would
    deviate from Python's semantics where a boolean compares to an integer
    by casting the boolean to 0 or 1.

    Args:
        T1 - an arithmetic type
        T2 - an arithmetic type

    Result:
        The typed_python arithmetic type that results from arithmetic operations
        on these two types.
    """
    assert T1 in arithmetic_types, T1
    assert T2 in arithmetic_types, T2

    if T1 is bool and T2 is bool:
        return bool

    outFloatness = floatness(T1) or floatness(T2)
    outBitness = max(bitness(T1), bitness(T2))

    if outFloatness:
        return Float32 if outBitness == 32 else float

    outSignedness = signedness(T1) or signedness(T2)

    return _intsBySignednessAndBitness[outSignedness, outBitness]
