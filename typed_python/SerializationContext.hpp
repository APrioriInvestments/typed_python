#pragma once

#include <memory>

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

    virtual void serializePythonObject(PyObject* o, SerializationBuffer& b) const = 0;
    virtual PyObject* deserializePythonObject(DeserializationBuffer& b) const = 0;

    virtual std::shared_ptr<ByteBuffer> compress(uint8_t* begin, uint8_t* end) const = 0;
    virtual std::shared_ptr<ByteBuffer> decompress(uint8_t* begin, uint8_t* end) const = 0;
};
