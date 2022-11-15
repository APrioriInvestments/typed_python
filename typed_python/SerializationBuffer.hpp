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
#include <thread>
#include <condition_variable>
#include "Type.hpp"
#include "WireType.hpp"

class Type;
class SerializationContext;
class Bytes;

// represents a single block of data that we can compress after the fact
// in parallel. blocks should be around 1 mb in size.
class SerializationBufferBlock {
public:
    SerializationBufferBlock() :
        m_size(0),
        m_reserved(0),
        m_buffer(nullptr),
        m_compressed(false)
    {
    }

    ~SerializationBufferBlock() {
        if (m_buffer) {
            free(m_buffer);
        }
    }

    uint8_t* buffer() const {
        return m_buffer;
    }

    size_t size() const {
        return m_size;
    }

    bool isCompressed() const {
        return m_compressed;
    }

    void markCompressed() {
        m_compressed = true;
    }

    bool oversized() const {
        return m_size > 1024 * 1024;
    }

    size_t capacity() {
        return m_reserved - m_size;
    }

    template< class T>
    void write(T i) {
        ensure(sizeof(i));
        *(T*)(m_buffer+m_size) = i;
        m_size += sizeof(i);
    }

    template<class callback>
    void initialize_bytes(size_t bytecount, const callback& c) {
        ensure(bytecount);

        uint8_t* bytes = m_buffer + m_size;

        c(bytes);

        m_size += bytecount;
    }

    //nakedly write bytes into the stream
    void write_bytes(uint8_t* ptr, size_t bytecount) {
        ensure(bytecount);
        memcpy(m_buffer + m_size, ptr, bytecount);
        m_size += bytecount;
    }

    void ensure(size_t bytecount) {
        if (m_size + bytecount > m_reserved) {
            reserve((m_size + bytecount + 1024) * 1.5);
        }
    }

    void reserve(size_t new_reserved) {
        if (new_reserved < m_reserved) {
            throw std::runtime_error("Can't make reserved size smaller");
        }

        m_reserved = new_reserved;
        m_buffer = (uint8_t*)::realloc(m_buffer, m_reserved);
    }

    void compress();

private:
    size_t m_size;
    size_t m_reserved;
    uint8_t* m_buffer;
    bool m_compressed;
};

class SerializationBuffer {
public:
    SerializationBuffer(const SerializationContext& context) :
        m_context(context),
        m_wants_compress(context.isCompressionEnabled()),
        m_compress_using_threads(context.compressUsingThreads()),
        m_is_consolidated(false)
    {
        m_top_block = new SerializationBufferBlock();
        m_blocks.push_back(std::shared_ptr<SerializationBufferBlock>(m_top_block));
    }

    ~SerializationBuffer() {
        PyEnsureGilAcquired acquireTheGil;

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

    uint8_t* buffer() {
        if (!m_is_consolidated) {
            consolidate();
        }

        if (m_blocks.size() == 0) {
            return nullptr;
        }

        return m_blocks[0]->buffer();
    }

    void copyInto(uint8_t* ptr) {
        for (auto b: m_blocks) {
            if (b->size()) {
                memcpy(ptr, b->buffer(), b->size());
                ptr += b->size();
            }
        }
    }

    size_t size() {
        size_t res = 0;
        for (auto& bPtr: m_blocks) {
            res += bPtr->size();
        }
        return res;
    }

    //nakedly write bytes into the stream
    void write_bytes(uint8_t* ptr, size_t bytecount) {
        while (bytecount > 1024 * 1024) {
            write_bytes(ptr, 1024 * 1024);

            ptr += 1024 * 1024;
            bytecount -= 1024 * 1024;
        }

        checkTopBlock();

        m_top_block->write_bytes(ptr, bytecount);
    }

    // allocate some memory and call 'c' with a uint8_t* pointing at it
    // to initialize it.
    template<class callback>
    void initialize_bytes(size_t bytecount, const callback& c) {
        checkTopBlock();

        m_top_block->initialize_bytes(bytecount, c);
    }

    void write_byte(uint8_t byte) {
        write<uint8_t>(byte);
    }

    void consolidate();

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
            markForCompression(m_blocks.back());

            for (auto b: m_blocks) {
                waitForCompression(b);
            }
        }
    }

    template< class T>
    void write(T i) {
        checkTopBlock();

        m_top_block->write(i);
    }

    void checkTopBlock() {
        if (m_top_block->oversized()) {
            m_top_block = new SerializationBufferBlock();

            if (m_wants_compress) {
                markForCompression(m_blocks.back());
            }

            m_blocks.push_back(
                std::shared_ptr<SerializationBufferBlock>(
                    m_top_block
                )
            );
        }
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

    bool m_compress_using_threads;

    bool m_is_consolidated;

    size_t m_size;

    // the
    SerializationBufferBlock* m_top_block;

    std::vector<std::shared_ptr<SerializationBufferBlock > > m_blocks;

    std::map<void*, int32_t> m_idToPointerCache;

    std::map<Type*, std::vector<void*>> m_pointersNeedingDecref;

    std::vector<PyObject*> m_pyObjectsNeedingDecref;

    std::set<Type*> m_types_being_serialized;

    std::unordered_map<MutuallyRecursiveTypeGroup*, int> m_group_counter;

    static void compressionThread();
    static std::shared_ptr<SerializationBufferBlock> getNextCompressTask();

    void markForCompression(std::shared_ptr<SerializationBufferBlock> block);
    void waitForCompression(std::shared_ptr<SerializationBufferBlock> block);

    static std::mutex s_compress_thread_mutex;
    static std::condition_variable* s_has_work;
    static std::vector<std::thread*> s_compress_threads;
    static std::unordered_set<std::shared_ptr<SerializationBufferBlock> > s_waiting_compress_blocks;
    static std::unordered_set<std::shared_ptr<SerializationBufferBlock> > s_working_compress_blocks;
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
