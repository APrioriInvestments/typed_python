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

#include "Type.hpp"
#include "WireType.hpp"
#include <stdexcept>
#include <stdlib.h>
#include <vector>

class SerializationContext;


class DeserializationBuffer {
public:
    DeserializationBuffer(uint8_t* ptr, size_t sz, const SerializationContext& context) :
            m_context(context),
            m_read_head(nullptr),
            m_read_head_offset(0),
            m_size(0),
            m_compressed_blocks(ptr),
            m_compressed_block_data_remaining(sz),
            m_pos(0)
    {
    }

    ~DeserializationBuffer() {
        PyEnsureGilAcquired acquireTheGil;

        for (auto& typeAndList: m_needs_decref) {
            typeAndList.first->check([&](auto& concreteType) {
                for (auto ptr: typeAndList.second) {
                    concreteType.destroy((instance_ptr)&ptr);
                }
            });
        }

        for (auto p: m_pyobj_needs_decref) {
            decref(p);
        }
    }

    DeserializationBuffer(const DeserializationBuffer&) = delete;
    DeserializationBuffer& operator=(const DeserializationBuffer&) = delete;

    uint64_t readUnsignedVarintObject() {
        auto res = readFieldNumberAndWireType();
        assertWireTypesEqual(res.second, WireType::VARINT);
        return readUnsignedVarint();
    }

    //read a 'varint' (a la google protobuf encoding).
    uint64_t readUnsignedVarint() {
        uint64_t accumulator = 0;

        uint64_t shift = 0;

        while (true) {
            uint64_t value = read<uint8_t>();
            accumulator += (value & 127) << shift;
            shift += 7;

            if (value < 128) {
                return accumulator;
            }
        }
    }

    //write a signed 'varint' using zigzag encoding (a la google protobuf)
    int64_t readSignedVarint() {
        uint64_t val = readUnsignedVarint();

        bool isNegative = val & 1;
        val >>= 1;
        if (isNegative) {
            return -val-1;
        }
        return val;
    }

    std::pair<size_t, size_t> readFieldNumberAndWireType() {
        size_t val = readUnsignedVarint();
        return std::make_pair(val >> 3, val & 7);
    }

    //read a single message and drop it. Returns the wire type of the discarded message.
    size_t readMessageAndDiscard() {
        std::pair<size_t, size_t> fieldAndWire = readFieldNumberAndWireType();
        finishReadingMessageAndDiscard(fieldAndWire.second);
        return fieldAndWire.second;
    }

    //copy a message to an output buffer
    template<class buffer_type>
    void copyMessageToOtherBuffer(size_t wireType, buffer_type& outBuffer) {
        if (wireType == WireType::EMPTY) {
            return;
        }

        if (wireType == WireType::VARINT) {
            //read a VARINT and just discard it
            outBuffer.writeUnsignedVarint(readUnsignedVarint());
            return;
        }

        if (wireType == WireType::BITS_32) {
            outBuffer.write(read<int32_t>());
            return;
        }

        if (wireType == WireType::BITS_64) {
            outBuffer.write(read<int64_t>());
            return;
        }

        if (wireType == WireType::BYTES) {
            size_t bytecount = readUnsignedVarint();
            outBuffer.writeUnsignedVarint(bytecount);

            read_bytes_fun(bytecount, [&](uint8_t* ptr) {
                outBuffer.write_bytes(ptr, bytecount);
            });

            return;
        }

        if (wireType == WireType::SINGLE) {
            auto fieldAndWire = readFieldNumberAndWireType();
            outBuffer.writeUnsignedVarint((fieldAndWire.first << 3) + fieldAndWire.second);
            copyMessageToOtherBuffer(fieldAndWire.second, outBuffer);
            return;
        }

        if (wireType == WireType::BEGIN_COMPOUND) {
            while (true) {
                auto fieldAndWire = readFieldNumberAndWireType();
                outBuffer.writeUnsignedVarint((fieldAndWire.first << 3) + fieldAndWire.second);
                if (fieldAndWire.second == WireType::END_COMPOUND) {
                    break;
                } else {
                    copyMessageToOtherBuffer(fieldAndWire.second, outBuffer);
                }
            }
            return;
        }

        if (wireType == WireType::END_COMPOUND) {
            return;
        }

        throw std::runtime_error("Corrupt message with invalid wire type found.");
    }

    //finish a message whose field and wire type we've read
    void finishReadingMessageAndDiscard(size_t wireType) {
        if (wireType == WireType::EMPTY) {
            return;
        }

        if (wireType == WireType::VARINT) {
            //read a VARINT and just discard it
            readUnsignedVarint();
            return;
        }

        if (wireType == WireType::BITS_32) {
            read<int32_t>();
            return;
        }

        if (wireType == WireType::BITS_64) {
            read<int64_t>();
            return;
        }

        if (wireType == WireType::BYTES) {
            size_t bytecount = readUnsignedVarint();
            read_bytes_fun(bytecount, [&](uint8_t* ptr) {});
            return;
        }

        if (wireType == WireType::SINGLE) {
            readMessageAndDiscard();
            return;
        }

        if (wireType == WireType::BEGIN_COMPOUND) {
            while (readMessageAndDiscard() != WireType::END_COMPOUND) {
                //do nothing
            }
            return;
        }

        if (wireType == WireType::END_COMPOUND) {
            return;
        }

        throw std::runtime_error("Corrupt message with invalid wire type found.");
    }

    void finishCompoundMessage(size_t wireType, bool assertEmpty=true) {
        if (wireType == WireType::EMPTY || wireType == WireType::SINGLE) {
            return;
        }

        while (readMessageAndDiscard() != WireType::END_COMPOUND) {
            //do nothing
            if (assertEmpty) {
                throw std::runtime_error("Still have unprocessed messages.");
            }
        }
    }

    template<class callback_type>
    void consumeCompoundMessage(size_t wireType, const callback_type& callback) {
        if (wireType == WireType::EMPTY) {
            return;
        }

        if (wireType == WireType::SINGLE) {
            auto fieldAndWiretype = readFieldNumberAndWireType();
            callback(fieldAndWiretype.first, fieldAndWiretype.second);
            return;
        }

        if (wireType != WireType::BEGIN_COMPOUND) {
            throw std::runtime_error("Invalid wire type for compound message.");
        }

        while (true) {
            auto fieldAndWiretype = readFieldNumberAndWireType();

            if (fieldAndWiretype.second == WireType::END_COMPOUND) {
                return;
            }

            callback(fieldAndWiretype.first, fieldAndWiretype.second);
        }
    }

    //consume a compound message consisting of all-zero field ids, and replace the
    //zeros with the message number in the stream. We do this for containers classes
    //where the positional information is actually meaningful.
    template<class callback_type>
    int64_t consumeCompoundMessageWithImpliedFieldNumbers(size_t wireType, const callback_type& callback) {
        int64_t messageCount = 0;

        consumeCompoundMessage(wireType, [&](size_t fieldNumber, size_t wireType) {
            if (fieldNumber != 0) {
                throw std::runtime_error("Expected all zero field numbers");
            }

            callback(messageCount++, wireType);
        });

        return messageCount;
    }

    double read_double() {
        return read<double>();
    }

    template<class initfun>
    auto read_bytes_fun(size_t bytecount, const initfun& f) -> decltype(f((uint8_t*)nullptr)) {
        while (m_size < bytecount) {
            if (!decompress()) {
                throw std::runtime_error("out of data");
            }
        }

        m_size -= bytecount;
        m_read_head += bytecount;
        m_read_head_offset += bytecount;
        m_pos += bytecount;

        return f(m_read_head - bytecount);
    }

    void read_bytes(uint8_t* ptr, size_t bytecount) {
        while (m_size < bytecount) {
            if (!decompress()) {
                throw std::runtime_error("out of data");
            }
        }

        memcpy(ptr,m_read_head,bytecount);

        m_size -= bytecount;
        m_read_head += bytecount;
        m_read_head_offset += bytecount;
        m_pos += bytecount;
    }

    std::string readStringObject() {
        int32_t sz = readUnsignedVarint();
        return read_bytes_fun(sz, [&](uint8_t* ptr) {
            return std::string(ptr, ptr + sz);
        });
    }

    bool isDone() {
        return !canConsume(1);
    }

    bool canConsume(size_t ct) {
        while (m_size < ct) {
            if (!decompress()) {
                return false;
            }
        }

        return true;
    }

    size_t pos() const {
        return m_pos;
    }

    const SerializationContext& getContext() const {
        return m_context;
    }

    void* lookupCachedPointer(int32_t which) {
        if (which < 0) {
            throw std::runtime_error("corrupt data: invalid cache lookup");
        }

        if (which >= m_cachedPointers.size()) {
            return nullptr;
        }

        return m_cachedPointers[which];
    }

    template<class T>
    void updateCachedPointer(int32_t which, T* ptr) {
        if (which < 0 || which >= m_cachedPointers.size() || !m_cachedPointers[which]) {
            throw std::runtime_error("Corrupt data: can't replace a cached pointer that's empty.");
        }

        m_cachedPointers[which] = ptr;
    }

    template<class T>
    T* addCachedPointer(int32_t which, T* ptr, Type* decrefType=nullptr) {
        if (!ptr) {
            throw std::runtime_error("Corrupt data: can't write a null cache pointer.");
        }
        if (which < 0) {
            throw std::runtime_error("Invalid memo. Shouldn't be negative.");
        }
        if (which > m_cachedPointers.size() + 1000000) {
            throw std::runtime_error("Suspicious memo is out of bounds.");
        }
        while (which >= m_cachedPointers.size()) {
            m_cachedPointers.push_back(nullptr);
        }
        if (m_cachedPointers[which]) {
            throw std::runtime_error("Corrupt data: tried to write a recursive object multiple times");
        }
        m_cachedPointers[which] = ptr;

        if (decrefType) {
            m_needs_decref[decrefType].push_back(ptr);
        }

        return ptr;
    }

    PyObject* addCachedPyObj(int32_t which, PyObject* ptr) {
        if (!ptr) {
            throw std::runtime_error("Corrupt data: can't write a null cache pointer.");
        }
        if (which < 0) {
            throw std::runtime_error("Invalid memo. Shouldn't be negative.");
        }
        if (which > m_cachedPointers.size() + 1000000) {
            throw std::runtime_error("Suspicious memo is out of bounds.");
        }
        while (which >= m_cachedPointers.size()) {
            m_cachedPointers.push_back(nullptr);
        }
        if (m_cachedPointers[which]) {
            throw std::runtime_error("Corrupt data: tried to write a recursive object multiple times");
        }
        m_cachedPointers[which] = (void*)ptr;

        m_pyobj_needs_decref.push_back(ptr);

        return ptr;
    }

    void skipNextEncodedValue() {
        throw std::runtime_error("not implemented yet");
    }

    void readRegisterType(double* out, size_t wireType) {
        if (wireType != WireType::BITS_64) {
            throw std::runtime_error("Invalid wire format: expected BITS_64");
        }
        *out = read<double>();
    }

    void readRegisterType(float* out, size_t wireType) {
        if (wireType != WireType::BITS_32) {
            throw std::runtime_error("Invalid wire format: expected BITS_32");
        }
        *out = read<float>();
    }

    void readRegisterType(bool* out, size_t wireType) {
        if (wireType != WireType::VARINT) {
            throw std::runtime_error("Invalid wire format: expected VARINT");
        }
        *out = readUnsignedVarint();
    }

    void readRegisterType(uint8_t* out, size_t wireType) {
        if (wireType != WireType::VARINT) {
            throw std::runtime_error("Invalid wire format: expected VARINT");
        }
        *out = readUnsignedVarint();
    }

    void readRegisterType(uint16_t* out, size_t wireType) {
        if (wireType != WireType::VARINT) {
            throw std::runtime_error("Invalid wire format: expected VARINT");
        }
        *out = readUnsignedVarint();
    }

    void readRegisterType(uint32_t* out, size_t wireType) {
        if (wireType != WireType::VARINT) {
            throw std::runtime_error("Invalid wire format: expected VARINT");
        }
        *out = readUnsignedVarint();
    }

    void readRegisterType(uint64_t* out, size_t wireType) {
        if (wireType != WireType::VARINT) {
            throw std::runtime_error("Invalid wire format: expected VARINT");
        }
        *out = readUnsignedVarint();
    }

    void readRegisterType(int8_t* out, size_t wireType) {
        if (wireType != WireType::VARINT) {
            throw std::runtime_error("Invalid wire format: expected VARINT");
        }
        *out = readSignedVarint();
    }

    void readRegisterType(int16_t* out, size_t wireType) {
        if (wireType != WireType::VARINT) {
            throw std::runtime_error("Invalid wire format: expected VARINT");
        }
        *out = readSignedVarint();
    }

    void readRegisterType(int32_t* out, size_t wireType) {
        if (wireType != WireType::VARINT) {
            throw std::runtime_error("Invalid wire format: expected VARINT");
        }
        *out = readSignedVarint();
    }

    void readRegisterType(int64_t* out, size_t wireType) {
        if (wireType != WireType::VARINT) {
            throw std::runtime_error("Invalid wire format: expected VARINT");
        }
        *out = readSignedVarint();
    }

    template<class T>
    T read() {
        while (m_size < sizeof(T)) {
            if (!decompress()) {
                throw std::runtime_error("out of data");
            }
        }

        T* ptr = (T*)m_read_head;

        m_size -= sizeof(T);
        m_read_head += sizeof(T);
        m_read_head_offset += sizeof(T);
        m_pos += sizeof(T);

        return *ptr;
    }

private:
    bool decompress();

    const SerializationContext& m_context;

    std::vector<uint8_t> m_decompressed_buffer;
    uint8_t* m_read_head;
    size_t m_read_head_offset;
    size_t m_size;

    uint8_t* m_compressed_blocks;
    size_t m_compressed_block_data_remaining;

    size_t m_pos;

    // maps indices to the pointers we've cached under that index.
    std::vector<void*> m_cachedPointers;

    //a map from the object type to each object that needs decreffing.
    //it must be the case that a pointer is the _natural_ layout of the
    //object (e.g. PyObject, Dict, etc). We pass a pointer to this
    //as the actual instance to the 'destroy' operation
    std::map<Type*, std::vector<void*> > m_needs_decref;

    std::vector<PyObject*> m_pyobj_needs_decref;
};
