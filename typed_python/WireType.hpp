/******************************************************************************
   Copyright 2017-2019 typed_python Authors

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
******************************************************************************/

#pragma once

#include <stdexcept>

//every message in our serialization format starts with a 3-bit wire type
//plus a field-number or a count (shifted by 3 bits) encoded as a varint.
class WireType {
public:
    enum {
        EMPTY = 0, //no content
        VARINT = 1, //the next value is a single varint
        BITS_32 = 2, //the next value is a 32-bit value
        BITS_64 = 3, //the next value is a 64-bit value
        BYTES = 4, //the next value is a varint encoding bytecount, followed by that many bytes bytes
        SINGLE = 5, //the next value is a single submessage (basically, COMPOUND but not needing an extra byte encoding the '1')
        BEGIN_COMPOUND = 6, //this value consists of a sequence of messages followed by a matching 'EndCompound' message
        END_COMPOUND = 7 //terminate a 'BEGIN_COMPOUND'
    };
};

inline void assertWireTypesEqual(size_t found, size_t expected) {
    if (found != expected) {
        throw std::runtime_error("Invalid wire type encountered.");
    }
}

inline void assertNonemptyCompoundWireType(size_t found) {
    if (found != WireType::BEGIN_COMPOUND && found != WireType::SINGLE) {
        throw std::runtime_error("Invalid wire type encountered.");
    }
}
