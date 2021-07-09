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
#include <stdlib.h>
#include <map>
#include <set>
#include "Type.hpp"
#include "WireType.hpp"

class Type;
class SerializationContext;
class Bytes;

class SerializationBuffer {
public:
    SerializationBuffer(const SerializationContext& context) :
            m_context(context),
            m_wants_compress(context.isCompressionEnabled()),
            m_buffer(nullptr),
            m_size(0),
            m_reserved(0),
            m_last_compression_point(0)
    {
    }

    ~SerializationBuffer() {
        PyEnsureGilAcquired acquireTheGil;

        if (m_buffer) {
            ::free(m_buffer);
        }

        for (auto& typeAndList: m_pointersNeedingDecref) {
            if (typeAndList.first) {
                typeAndList.first->check([&](auto& concreteType) {
                    for (auto ptr: typeAndList.second) {
                        concreteType.destroy((instance_ptr)&ptr);
                    }
                });
            }
        }

        for (auto p: m_pyObjectsNeedingDecref) {
            decref(p);
        }
    }

    SerializationBuffer(const SerializationBuffer&) = delete;
    SerializationBuffer& operator=(const SerializationBuffer&) = delete;

    static Bytes serializeSingleBoolToBytes(bool value);

    void writeBeginBytes(size_t fieldNumber, size_t bytecount) {
        writeUnsignedVarint((fieldNumber << 3) + WireType::BYTES);
        writeUnsignedVarint(bytecount);
    }

    void writeEmpty(size_t fieldNumber) {
        writeUnsignedVarint((fieldNumber << 3) + WireType::EMPTY);
    };

    void writeBeginSingle(size_t fieldNumber) {
        writeUnsignedVarint((fieldNumber << 3) + WireType::SINGLE);
    };

    void writeBeginCompound(size_t fieldNumber) {
        writeUnsignedVarint((fieldNumber << 3) + WireType::BEGIN_COMPOUND);
    }

    void writeEndCompound() {
        writeUnsignedVarint(WireType::END_COMPOUND);
    }

    void writeUnsignedVarintObject(size_t fieldNumber, uint64_t i) {
        writeUnsignedVarint(WireType::VARINT + (fieldNumber << 3));
        writeUnsignedVarint(i);
    }

    void writeSignedVarintObject(size_t fieldNumber, uint64_t i) {
        writeUnsignedVarint(WireType::VARINT + (fieldNumber << 3));
        writeSignedVarint(i);
    }

    //write a 'varint' (a la google protobuf encoding)
    void writeUnsignedVarint(uint64_t i) {
        while (i >= 128) {
            write<uint8_t>(128 + (i & 127));
            i >>= 7;
        }
        write<uint8_t>(i);
    }

    //write a signed 'varint' using zigzag encoding (a la google protobuf)
    void writeSignedVarint(int64_t i) {
        uint64_t val = i < 0 ? -i - 1 : i;
        val *= 2;
        if (i < 0) {
            val += 1;
        }
        writeUnsignedVarint(val);
    }

    void writeRegisterType(size_t fieldNumber, double v) {
        writeUnsignedVarint(WireType::BITS_64 + (fieldNumber << 3));
        write<double>(v);
    }

    void writeRegisterType(size_t fieldNumber, float v) {
        writeUnsignedVarint(WireType::BITS_32 + (fieldNumber << 3));
        write<float>(v);
    }

    void writeRegisterType(size_t fieldNumber, bool v) {
        writeUnsignedVarint(WireType::VARINT + (fieldNumber << 3));
        writeUnsignedVarint(v);
    }

    void writeRegisterType(size_t fieldNumber, uint8_t v) {
        writeUnsignedVarint(WireType::VARINT + (fieldNumber << 3));
        writeUnsignedVarint(v);
    }

    void writeRegisterType(size_t fieldNumber, uint16_t v) {
        writeUnsignedVarint(WireType::VARINT + (fieldNumber << 3));
        writeUnsignedVarint(v);
    }

    void writeRegisterType(size_t fieldNumber, uint32_t v) {
        writeUnsignedVarint(WireType::VARINT + (fieldNumber << 3));
        writeUnsignedVarint(v);
    }

    void writeRegisterType(size_t fieldNumber, uint64_t v) {
        writeUnsignedVarint(WireType::VARINT + (fieldNumber << 3));
        writeUnsignedVarint(v);
    }

    void writeRegisterType(size_t fieldNumber, int8_t v) {
        writeUnsignedVarint(WireType::VARINT + (fieldNumber << 3));
        writeSignedVarint(v);
    }

    void writeRegisterType(size_t fieldNumber, int16_t v) {
        writeUnsignedVarint(WireType::VARINT + (fieldNumber << 3));
        writeSignedVarint(v);
    }

    void writeRegisterType(size_t fieldNumber, int32_t v) {
        writeUnsignedVarint(WireType::VARINT + (fieldNumber << 3));
        writeSignedVarint(v);
    }

    void writeRegisterType(size_t fieldNumber, int64_t v) {
        writeUnsignedVarint(WireType::VARINT + (fieldNumber << 3));
        writeSignedVarint(v);
    }

    void writeStringObject(size_t fieldNumber, const std::string& s) {
        writeUnsignedVarint(WireType::BYTES + (fieldNumber << 3));
        writeUnsignedVarint(s.size());
        write_bytes((uint8_t*)&s[0], s.size());
    }

    uint8_t* buffer() const {
        return m_buffer;
    }

    size_t size() const {
        return m_size;
    }

    //nakedly write bytes into the stream
    void write_bytes(uint8_t* ptr, size_t bytecount, bool allowCompression = true) {
        ensure(bytecount, allowCompression);
        memcpy(m_buffer+m_size, ptr, bytecount);
        m_size += bytecount;
    }

    uint8_t* prepare_bytes(size_t bytecount, bool allowCompression = true) {
        ensure(bytecount, allowCompression);

        uint8_t* bytes = m_buffer + m_size;

        m_size += bytecount;

        return bytes;
    }

    void write_byte(uint8_t byte) {
        write<uint8_t>(byte);
    }

    void ensure(size_t t, bool allowCompression=true) {
        if (m_size + t > m_reserved) {
            reserve(m_size + t + 1024 * 128);

            //compress every meg or so
            if (m_wants_compress && allowCompression && m_size - m_last_compression_point > 1024 * 1024) {
                compress();
            }
        }
    }

    void reserve(size_t new_reserved) {
        if (new_reserved < m_reserved) {
            throw std::runtime_error("Can't make reserved size smaller");
        }

        m_reserved = new_reserved;
        m_buffer = (uint8_t*)::realloc(m_buffer, m_reserved);
    }

    const SerializationContext& getContext() const {
        return m_context;
    }

    /**
        Keep track of whether we've seen this pointer before.

        @param t - A PyObject pointer to be cached
        @param objType - a pointer to the appropriate type object. if populated,
            and this is the first time we've seen the object, we'll incref it
            before we put it in the cache, and decref it when the buffer
            is destroyed. The type object _must_ be a pointer-layout-style
            object (PyObject, Dict, ConstDict, Class, etc.) since we assume
            the pointer is the actual held representation. If this argument
            is null, then we simply copy the pointer with no incref semantics.
        @return - an std::pair containing its cache ID (the size of the cache)
                and 'false' if the pointer was already in the cache.
    */
    std::pair<uint32_t, bool> cachePointer(void* t, Type* objType) {
        auto it = m_idToPointerCache.find(t);
        if (it == m_idToPointerCache.end()) {
            void* otherPointer;

            if (objType) {
                objType->copy_constructor((instance_ptr)&otherPointer, (instance_ptr)&t);
            } else {
                otherPointer = t;
            }

            if (otherPointer != t) {
                throw std::runtime_error("Pointer-copy stash semantics didn't work with type " + objType->name());
            }

            if (objType) {
                m_pointersNeedingDecref[objType].push_back(otherPointer);
            }

            uint32_t id = m_idToPointerCache.size();
            m_idToPointerCache[t] = id;

            return std::pair<uint32_t, bool>(id, true);
        }

        return std::pair<uint32_t, bool>(it->second, false);
    }

    std::pair<uint32_t, bool> cachePointer(PyObject* t) {
        auto it = m_idToPointerCache.find((void*)t);
        if (it == m_idToPointerCache.end()) {
            m_pyObjectsNeedingDecref.push_back(incref(t));

            uint32_t id = m_idToPointerCache.size();
            m_idToPointerCache[(void*)t] = id;

            return std::pair<uint32_t, bool>(id, true);
        }

        return std::pair<uint32_t, bool>(it->second, false);
    }

    bool isAlreadyCached(void* t) {
        return m_idToPointerCache.find(t) != m_idToPointerCache.end();
    }

    bool isAlreadyCached(PyObject* t) {
        return isAlreadyCached((void*)t);
    }

    int32_t memoFor(PyObject* t) {
        auto it = m_idToPointerCache.find(t);
        if (it != m_idToPointerCache.end()) {
            return it->second;
        }
        return -1;
    }

    void finalize() {
        if (m_wants_compress) {
            compress();
        }
    }

    void compress();

    template< class T>
    void write(T i) {
        ensure(sizeof(i));
        *(T*)(m_buffer+m_size) = i;
        m_size += sizeof(i);
    }

    void startSerializing(Type* nativeType) {
        if (m_types_being_serialized.find(nativeType) != m_types_being_serialized.end()) {
            throw std::runtime_error("Can't serialize recursive unnamed types.");
        }

        m_types_being_serialized.insert(nativeType);
    }

    void stopSerializing(Type* nativeType) {
        m_types_being_serialized.erase(nativeType);
    }

    int getGroupCounter(MutuallyRecursiveTypeGroup* group) {
        auto it = m_group_counter.find(group);

        if (it != m_group_counter.end()) {
            return it->second;
        }

        int ix = m_group_counter.size();

        m_group_counter[group] = ix;

        return ix;
    }

private:
    const SerializationContext& m_context;

    bool m_wants_compress;

    uint8_t* m_buffer;
    size_t m_size;
    size_t m_reserved;
    size_t m_last_compression_point;

    std::map<void*, int32_t> m_idToPointerCache;

    std::map<Type*, std::vector<void*>> m_pointersNeedingDecref;

    std::vector<PyObject*> m_pyObjectsNeedingDecref;

    std::set<Type*> m_types_being_serialized;

    std::unordered_map<MutuallyRecursiveTypeGroup*, int> m_group_counter;
};

class MarkTypeBeingSerialized {
public:
    MarkTypeBeingSerialized(Type* type, SerializationBuffer& buffer) :
            m_type(type),
            m_buffer(buffer)
    {
        m_buffer.startSerializing(type);
    }

    ~MarkTypeBeingSerialized() {
        m_buffer.stopSerializing(m_type);
    }

private:
    Type* m_type;
    SerializationBuffer& m_buffer;
};
