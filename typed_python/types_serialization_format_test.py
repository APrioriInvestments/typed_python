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

import unittest
import struct
import numpy
from typed_python import (
    TupleOf, ListOf, OneOf, Dict,
    ConstDict, Alternative, Forward,
    serialize, deserialize, validateSerializedObject, decodeSerializedObject
)


def unsignedVarint(u):
    """Encode an unsigned integer in 'varint' encoding (see google protobufs)"""
    assert u >= 0

    res = []
    while u >= 128:
        res.append(struct.pack("B", u % 128 + 128))
        u = u // 128

    res.append(struct.pack("B", u))

    return b"".join(res)


def signedVarint(s):
    """Encode a signed integer in 'varint' zig-zag encoding (see google protobufs)"""
    if s >= 0:
        return unsignedVarint(s*2)
    else:
        return unsignedVarint((-s)*2 - 1)


def EMPTY(fieldNumber):
    return unsignedVarint(fieldNumber * 8 + 0)


def VARINT(fieldNumber):
    return unsignedVarint(fieldNumber * 8 + 1)


def BITS_32(fieldNumber):
    return unsignedVarint(fieldNumber * 8 + 2)


def BITS_64(fieldNumber):
    return unsignedVarint(fieldNumber * 8 + 3)


def BYTES(fieldNumber):
    return unsignedVarint(fieldNumber * 8 + 4)


def SINGLE(fieldNumber):
    return unsignedVarint(fieldNumber * 8 + 5)


def BEGIN_COMPOUND(fieldNumber):
    return unsignedVarint(fieldNumber * 8 + 6)


def END_COMPOUND():
    return unsignedVarint(7)


def floatToBits(f):
    return struct.pack("d", f)


class TypesSerializationWireFormatTest(unittest.TestCase):
    """Test that the wire format we produce matches our standard.

    Objects are encoded in a form that's inspired by google protobuf, but not identical,
    mostly because our object models are not identical, and the semantics of repeated
    messages in protobufs doesn't map cleanly into our model.

    All unsigned integers are encoded as 'varints', where the least significant seven
    bits of a number encoded first, and the top bit set to indicate that there
    are additional bytes in the integer to follow. This encodes integers < 128 in a
    single byte, at the expense of encoding very large integers in up to 10 bytes.

    Signed integers are transformed into unsigned integers using 'zigzag' encoding:

        def zigZag(x):
            return x*2 if x > 0 else (s+1) * (-2)

    every value is encoded in a predictable format that lets us understand how to parse the
    message, so that even if we don't have a schema we can make sense of the message.

    Each message begins with a single varint that consists of a wire type and a field number
    encoded as 'wire_type + field_number * 8'. The field number tells us the meaning of the
    field in the surrounding object, and the wire type tells us how to make sense of the object.

    The wire types are

        0: EMPTY. The message is empty
        1: VARINT. The message consists of a single varint
        2: BITS_32. The message consists of a 32 bit value. The meaning of the bytes must
            be determined by the exact type being decoded.
        3: BITS_64. Like BITS_32 but with 64 bits.
        4: BYTES. The message consists of a varint byte length followed by that many bytes.
            The meaning of the bytes is determined by the type being decoded.
        5: SINGLE. The message consists of a single submessage.
        6: BEGIN_COMPOUND. The message consists of a string of messages followed by
            an END_COMPOUND with fieldNumber 0.
        7: END_COMPOUND. Terminates a message.

    Any message can be broken into its component pieces (or validated) without knowing the
    exact layout of the internal types.

    primitive python objects break down as follows:

        int - encoded as VARINT + a single zig-zag encoded varint
        bool - encoded as VARINT + a singled unsigned varint
        float - encoded as a BITS_64
        bytes - encoded as BYTES + an unsigned varint giving a bytecount plus the bytes themselves.
        str - encoded as BYTES in utf8

    None, and constants (e.g. constant strings in OneOfs) are encoded as EMPTY.

    OneOf is encoded as a SINGLE followed by a field giving the fieldNumber in the OneOf.

    TupleOf(T) is encoded as a compound (EMPTY, SINGLE, or a BEGIN_COMPOUND/END_COMPOUND
    pair) followed by a record count, and then individual objects. All field numbers are
    zero.

    Lists are encoded like tuples, but they have a single varint identity value first. If
    the list has been seen before, the data is not repeated.

    ConstDict is encoded similarly as a compound object with a size, and then alternating
    key/value pairs with zero fieldnumber.

    Dict is encoded with a varint identity followed by a size, and then interleaved keys and values,
    if the identity is new in the stream.

    NamedTuple and Tuple are encoded as EMPTY, SINGLE, or BEGIN/END with field numbers
    corresponding to the fieldNumbers given in the type. These are the integers by
    default, but can be overridden so that we can insert new fields without disrupting
    our ability to serialize older formats.

    Class is encoded as a varint identity (0) followed by a HeldClass (1).

    HeldClass is encoded like a named tuple. Uninitialized fields are not serialized.

    Alternatives are encoded as a SINGLE followed by a value with a fieldNumber mapping
    to one of the alternatives.

    Python objects that are bound to a specific codebase are encoded as compound objects.
    The exact meaning of named-type fields requires that an associated Codebase object
    in which we can lookup values is present. Python serialization details can be found
    in the PythonSerializationContext.hpp header, and should not be considered fully
    stable yet.

    The root-level serialization always has an initial fieldNumber of 0.
    """
    def test_single_values(self):
        # each value we produce should consist of a single field number (0)
        # encoding a value

        # None encoded as wire type of 'EMPTY', and no data
        self.assertEqual(serialize(None, None), EMPTY(0))

        # integers encoded as a varint
        self.assertEqual(serialize(int, 0), VARINT(0) + signedVarint(0))
        self.assertEqual(serialize(int, -1), VARINT(0) + signedVarint(-1))
        self.assertEqual(serialize(int, 1), VARINT(0) + signedVarint(1))

        # floats encoded as a BITS_64
        self.assertEqual(serialize(float, 1.5), BITS_64(0) + floatToBits(1.5))

        # bools encoded as an unsigned varint
        self.assertEqual(serialize(bool, False), VARINT(0) + unsignedVarint(0))
        self.assertEqual(serialize(bool, True), VARINT(0) + unsignedVarint(1))

        # bytes encoded directly
        self.assertEqual(serialize(bytes, b"123"), BYTES(0) + unsignedVarint(3) + b"123")

        # strings encoded as utf-8
        self.assertEqual(serialize(str, "123"), BYTES(0) + unsignedVarint(3) + b"123")

    def test_message_validation(self):
        self.assertEqual(validateSerializedObject(EMPTY(0)), None)
        self.assertEqual(validateSerializedObject(EMPTY(100)), None)

        # missing an integer value is not valid
        self.assertNotEqual(validateSerializedObject(VARINT(0)), None)
        self.assertEqual(validateSerializedObject(VARINT(0) + signedVarint(0)), None)

        # BEGIN must be matched by END
        self.assertNotEqual(validateSerializedObject(BEGIN_COMPOUND(0)), None)
        self.assertEqual(validateSerializedObject(BEGIN_COMPOUND(0) + END_COMPOUND()), None)

        # SINGLE must have a value after it
        self.assertNotEqual(validateSerializedObject(SINGLE(0)), None)
        self.assertNotEqual(validateSerializedObject(SINGLE(0) + VARINT(0)), None)
        self.assertEqual(validateSerializedObject(SINGLE(0) + VARINT(0) + signedVarint(0)), None)

        # BITS must have the right number of bits after it
        self.assertNotEqual(validateSerializedObject(BITS_32(0) + b"aaa"), None)
        self.assertEqual(validateSerializedObject(BITS_32(0) + b"aaaa"), None)

        self.assertNotEqual(validateSerializedObject(BITS_64(0) + b"aaaabbb"), None)
        self.assertEqual(validateSerializedObject(BITS_64(0) + b"aaaabbbb"), None)

        # BYTES must have the right number of bytes after it
        self.assertNotEqual(validateSerializedObject(BYTES(0)), None)
        self.assertNotEqual(validateSerializedObject(BYTES(0) + unsignedVarint(1)), None)

        self.assertEqual(validateSerializedObject(BYTES(0) + unsignedVarint(0)), None)
        self.assertEqual(validateSerializedObject(BYTES(0) + unsignedVarint(2) + b"  "), None)
        self.assertEqual(validateSerializedObject(BYTES(0) + unsignedVarint(10) + b" " * 10), None)

    def test_roundtrip_strings(self):
        # strings are encoded as utf8.

        def test(s):
            utfForm = s.encode("utf8")

            self.assertEqual(
                serialize(str, s),
                BYTES(0) + unsignedVarint(len(utfForm)) + utfForm
            )

            self.assertEqual(
                deserialize(str, serialize(str, s)).encode('raw_unicode_escape'),
                s.encode('raw_unicode_escape')
            )

        somePieces = [
            '',
            'hi',
            '\x00',
            '\xFF',
            '\u0001',
            '\u0010',
            '\u0100',
            '\uFFFF',
            '\U00010000',
            '\U0001FFFF',
            '\U00020000',
            '\U0002FFFF',
            '\U00030000',
            '\U0003FFFF',
            '\U00030000',
            '\U0003FFFF',
            '\U00040000',
            '\U000DFFFF',
            '\U000E0000',
            '\U000EFFFF',
            '\U000F0000',
            '\U0010FFFF',
        ]

        for s in somePieces:
            test(s)

            for s2 in somePieces:
                test(s+s2)
                test(s+s2+s)

    def test_message_decoding(self):
        self.assertEqual(decodeSerializedObject(VARINT(0) + signedVarint(100)), 100)
        self.assertEqual(decodeSerializedObject(EMPTY(0)), [])
        self.assertEqual(decodeSerializedObject(BYTES(0) + unsignedVarint(4) + b"asdf"), b"asdf")
        self.assertEqual(decodeSerializedObject(BITS_64(0) + floatToBits(1.5)), 1.5)
        self.assertEqual(decodeSerializedObject(SINGLE(0) + VARINT(100) + signedVarint(-200)), [(100, -200)])
        self.assertEqual(
            decodeSerializedObject(
                BEGIN_COMPOUND(0) + (
                    VARINT(100) + signedVarint(-200) +
                    VARINT(200) + signedVarint(-400)
                ) + END_COMPOUND()
            ),
            [(100, -200), (200, -400)]
        )

    def test_message_validation_fuzzer(self):
        def randomMessage(depth=0, maxDepth=8):
            x = numpy.random.uniform()
            k = int((numpy.random.uniform()) * 100)
            if x < .2 or depth >= maxDepth:
                return VARINT(k) + signedVarint(int((numpy.random.uniform() - .5) * 1000))
            if x < .4:
                return BITS_32(k) + b"aaaa"
            if x < .5:
                return BITS_64(k) + b"aaaabbbb"
            if x < .6:
                length = int(numpy.random.uniform() * 200)
                return BYTES(k) + unsignedVarint(length) + b" " * length
            if x < .7:
                return EMPTY(k)
            if x < .8:
                return SINGLE(k) + randomMessage(depth+1, maxDepth)

            return (
                BEGIN_COMPOUND(k) +
                b"".join([randomMessage(depth+1, maxDepth) for _ in range(numpy.random.choice(5))]) +
                END_COMPOUND()
            )

        goodSubmessageCount = 0
        badSubmessageCount = 0

        for _ in range(100):
            msg = randomMessage()
            self.assertEqual(validateSerializedObject(msg), None)

            if len(msg) > 1:
                for _ in range(10):
                    left = numpy.random.choice(len(msg) - 1)
                    right = left + numpy.random.choice(len(msg) - left)

                    subMsg = msg[left:right]

                    if validateSerializedObject(subMsg) is None:
                        goodSubmessageCount += 1
                    else:
                        badSubmessageCount += 1

        # most random submessage messages should be bad.
        self.assertTrue(badSubmessageCount > goodSubmessageCount * 10)

    def test_tuples(self):
        # tuples are a compound with indices on item numbers
        T = TupleOf(int)

        self.assertEqual(serialize(T, T(())), EMPTY(0))

        # A single element tuple is SINGLE + 0=VARINT + count + 0=VARINT + value
        self.assertEqual(
            serialize(T, T((1,))),
            BEGIN_COMPOUND(0) + VARINT(0) + unsignedVarint(1) + VARINT(0) + signedVarint(1) + END_COMPOUND()
        )

        self.assertEqual(
            serialize(T, T((123, 124))),
            BEGIN_COMPOUND(0) + (
                VARINT(0) + unsignedVarint(2) +
                VARINT(0) + signedVarint(123) +
                VARINT(0) + signedVarint(124)
            ) + END_COMPOUND()
        )

        self.assertEqual(
            serialize(T, T((123, 124, 125))),
            BEGIN_COMPOUND(0) + (
                VARINT(0) + unsignedVarint(3) +
                VARINT(0) + signedVarint(123) +
                VARINT(0) + signedVarint(124) +
                VARINT(0) + signedVarint(125)
            ) + END_COMPOUND()
        )

    def test_oneof(self):
        # tuples are a compound with indices on item numbers
        T = OneOf(None, int, float, "HI", TupleOf(int))

        self.assertEqual(serialize(T, None), SINGLE(0) + EMPTY(0))
        self.assertEqual(serialize(T, 1), SINGLE(0) + VARINT(1) + signedVarint(1))
        self.assertEqual(serialize(T, 1.5), SINGLE(0) + BITS_64(2) + floatToBits(1.5))
        self.assertEqual(serialize(T, "HI"), SINGLE(0) + EMPTY(3))
        self.assertEqual(
            serialize(T, (1, 2, 123)),
            SINGLE(0) + BEGIN_COMPOUND(4) + (
                VARINT(0) + unsignedVarint(3) +
                VARINT(0) + signedVarint(1) +
                VARINT(0) + signedVarint(2) +
                VARINT(0) + signedVarint(123)
            ) + END_COMPOUND()
        )

    def test_alternative(self):
        A = Alternative(
            "A",
            X=dict(a=int),
            Y=dict(b=float, c=float)
        )

        self.assertEqual(
            serialize(A, A.X(a=10)),
            SINGLE(0) + SINGLE(0) + VARINT(0) + signedVarint(10)
        )

        self.assertEqual(
            serialize(A, A.Y(b=10, c=20.2)),
            SINGLE(0) + BEGIN_COMPOUND(1) + BITS_64(0) + floatToBits(10) + BITS_64(1) + floatToBits(20.2) + END_COMPOUND()
        )

    def test_recursive_list(self):
        L = Forward("L")
        L = L.define(ListOf(OneOf(int, L)))

        listInst = L()
        listInst.append(10)
        listInst.append(listInst)

        self.assertEqual(
            serialize(L, listInst),
            BEGIN_COMPOUND(0) + (
                VARINT(0) + unsignedVarint(0) +  # the ID
                VARINT(0) + unsignedVarint(2) +  # the size
                SINGLE(0) + VARINT(0) + signedVarint(10) +
                SINGLE(0) + SINGLE(1) + (  # a compound object encoding the second element of the one-of
                    VARINT(0) + unsignedVarint(0)  # the ID and nothing else
                )
            ) + END_COMPOUND()
        )

    def test_const_dict(self):
        T = ConstDict(int, int)

        self.assertEqual(
            serialize(T, T({1: 2, 33: 44})),
            BEGIN_COMPOUND(0) +
            VARINT(0) + unsignedVarint(2) +  # for the size
            VARINT(0) + signedVarint(1) +
            VARINT(0) + signedVarint(2) +
            VARINT(0) + signedVarint(33) +
            VARINT(0) + signedVarint(44) +
            END_COMPOUND()
        )

    def test_dict(self):
        T = Dict(int, int)

        self.assertEqual(
            serialize(T, T({1: 2, 33: 44})),
            BEGIN_COMPOUND(0) +
            VARINT(0) + unsignedVarint(0) +  # for the id
            VARINT(0) + unsignedVarint(2) +  # for the size
            VARINT(0) + signedVarint(1) +
            VARINT(0) + signedVarint(2) +
            VARINT(0) + signedVarint(33) +
            VARINT(0) + signedVarint(44) +
            END_COMPOUND()
        )
