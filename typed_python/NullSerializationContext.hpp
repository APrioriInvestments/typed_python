#pragma once

#include "Type.hpp"
#include "SerializationContext.hpp"

class NullSerializationContext : public SerializationContext {
public:
    virtual void serializePythonObject(PyObject* o, SerializationBuffer& b) const {
        throw std::runtime_error("No serialization plugin provided, so we can't serialize arbitrary python objects.");
    }
    virtual PyObject* deserializePythonObject(DeserializationBuffer& b) const {
        throw std::runtime_error("No serialization plugin provided, so we can't deserialize arbitrary python objects.");
    }
    virtual std::shared_ptr<ByteBuffer> compress(uint8_t* begin, uint8_t* end) const {
        return std::shared_ptr<ByteBuffer>(new RangeByteBuffer(begin, end));
    }
    virtual std::shared_ptr<ByteBuffer> decompress(uint8_t* begin, uint8_t* end) const {
        return std::shared_ptr<ByteBuffer>(new RangeByteBuffer(begin, end));
    }
};
