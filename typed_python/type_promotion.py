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

from typed_python import (
    Bool, Float64, Float32, Int8, Int16, Int32, Int64, UInt8,
    UInt16, UInt32, UInt64
)

arithmetic_types = (
    Bool, Float64, Float32, Int8, Int16, Int32, Int64, UInt8,
    UInt16, UInt32, UInt64
)

_intsBySignednessAndBitness = {
    (True, 8): Int8,
    (True, 16): Int16,
    (True, 32): Int32,
    (True, 64): Int64,
    (False, 8): UInt8,
    (False, 16): UInt16,
    (False, 32): UInt32,
    (False, 64): UInt64
}


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
        T1 - a typed_python arithmetic type. Python types (float, int) don't work here.
        T2 - also a typed_python arithmetic type

    Result:
        The typed_python arithmetic type that results from arithmetic operations
        on these two types.
    """
    assert T1 in arithmetic_types, T1
    assert T2 in arithmetic_types, T2

    if T1 is Bool and T2 is Bool:
        return Bool

    floatness = T1.IsFloat or T2.IsFloat
    bitness = max(T1.Bits, T2.Bits)

    if floatness:
        return Float32 if bitness == 32 else Float64

    signedness = T1.IsSignedInt or T2.IsSignedInt

    return _intsBySignednessAndBitness[signedness, bitness]
