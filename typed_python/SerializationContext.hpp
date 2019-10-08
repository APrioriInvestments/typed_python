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

#include <memory>
#include "WireType.hpp"

class SerializationBuffer;
class DeserializationBuffer;

//models a contiguous range of bytes. Underlying could be python data
//or c++ data.
class ByteBuffer {
public:
    ByteBuffer() {};

    virtual ~ByteBuffer() {} ;

    virtual std::pair<uint8_t*, uint8_t*> range() = 0;

    size_t size() {
        return range().second - range().first;
    }
};

//a contiguous range of bytes that are guaranteed to live for the duration
//of the object, held as a begin/end pointer pair
class RangeByteBuffer : public ByteBuffer {
public:
    RangeByteBuffer(uint8_t* low, uint8_t* high) : m_low(low), m_high(high) {
    }

    virtual ~RangeByteBuffer() { }

    virtual std::pair<uint8_t*, uint8_t*> range() {
        return std::make_pair(m_low, m_high);
    }

private:
    uint8_t* m_low;
    uint8_t* m_high;
};

class SerializationContext {
public:
    virtual ~SerializationContext() {};

    virtual void serializePythonObject(PyObject* o, SerializationBuffer& b, size_t fieldNumber) const = 0;
    virtual PyObject* deserializePythonObject(DeserializationBuffer& b, size_t wireType) const = 0;

    virtual bool isCompressionEnabled() const = 0;
    virtual std::shared_ptr<ByteBuffer> compress(uint8_t* begin, uint8_t* end) const = 0;
    virtual std::shared_ptr<ByteBuffer> decompress(uint8_t* begin, uint8_t* end) const = 0;
};
