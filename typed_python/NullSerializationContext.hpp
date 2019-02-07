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
    virtual std::string compress(std::string bytes) const {
        return bytes;
    }
    virtual std::string decompress(std::string bytes) const {
        return bytes;
    }
};
