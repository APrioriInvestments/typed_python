#pragma once

class SerializationBuffer;
class DeserializationBuffer;

class SerializationContext {
public:
    virtual ~SerializationContext() {};

    virtual void serializePythonObject(PyObject* o, SerializationBuffer& b) const = 0;
    virtual PyObject* deserializePythonObject(DeserializationBuffer& b) const = 0;

    virtual std::string compress(std::string bytes) const = 0;
    virtual std::string decompress(std::string bytes) const = 0;
};
