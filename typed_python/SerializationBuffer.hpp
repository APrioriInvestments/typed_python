#pragma once

#include <stdexcept>
#include <stdlib.h>
#include <map>
#include <set>
#include "Type.hpp"

class Type;
class SerializationContext;

class SerializationBuffer {
public:
    SerializationBuffer(const SerializationContext& context) :
            m_context(context),
            m_buffer(nullptr),
            m_size(0),
            m_reserved(0),
            m_last_compression_point(0)
    {
    }

    ~SerializationBuffer() {
        PyEnsureGilAcquired acquireTheGil;

        if (m_buffer) {
            free(m_buffer);
        }

        for (auto& typeAndList: m_pointersNeedingDecref) {
            typeAndList.first->check([&](auto& concreteType) {
                for (auto ptr: typeAndList.second) {
                    concreteType.destroy((instance_ptr)&ptr);
                }
            });
        }
    }

    SerializationBuffer(const SerializationBuffer&) = delete;
    SerializationBuffer& operator=(const SerializationBuffer&) = delete;

    void write_uint8(uint8_t i) {
        write<uint8_t>(i);
    }

    void write_uint32(uint32_t i) {
        write<uint32_t>(i);
    }

    void write_uint64(uint64_t i) {
        write<uint64_t>(i);
    }

    void write_int64(int64_t i) {
        write<int64_t>(i);
    }

    void write_double(double i) {
        write<double>(i);
    }

    void write_bytes(uint8_t* ptr, size_t bytecount, bool allowCompression = true) {
        ensure(bytecount, allowCompression);
        memcpy(m_buffer+m_size, ptr, bytecount);
        m_size += bytecount;
    }

    void write_string(const std::string& s) {
        write_uint32(s.size());
        write_bytes((uint8_t*)&s[0], s.size());
    }

    uint8_t* buffer() const {
        return m_buffer;
    }

    size_t size() const {
        return m_size;
    }

    void ensure(size_t t, bool allowCompression=true) {
        if (m_size + t > m_reserved) {
            reserve(m_size + t + 1024 * 128);

            //compress every 10 megs or so
            if (allowCompression && m_size - m_last_compression_point > 10 * 1024 * 1024) {
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
            the pointer is the actual held representation.
        @return - an std::pair containing its cache ID (the size of the cache)
                and 'false' if the pointer was already in the cache.
    */
    std::pair<uint32_t, bool> cachePointer(void* t, Type* objType) {
        auto it = m_idToPointerCache.find(t);
        if (it == m_idToPointerCache.end()) {
            void* otherPointer;
            objType->copy_constructor((instance_ptr)&otherPointer, (instance_ptr)&t);

            if (otherPointer != t) {
                throw std::runtime_error("Pointer-copy stash semantics didn't work with type " + objType->name());
            }

            m_pointersNeedingDecref[objType].push_back(otherPointer);

            uint32_t id = m_idToPointerCache.size();
            m_idToPointerCache[t] = id;

            return std::pair<uint32_t, bool>(id, true);
        }

        return std::pair<uint32_t, bool>(it->second, false);
    }

    std::pair<uint32_t, bool> cachePointerToType(Type* t) {
        auto it = m_idToPointerCache.find(t);
        if (it == m_idToPointerCache.end()) {
            uint32_t id = m_idToPointerCache.size();
            m_idToPointerCache[t] = id;
            return std::pair<uint32_t, bool>(id, true);
        }
        return std::pair<uint32_t, bool>(it->second, false);
    }

    void finalize() {
        compress();
    }

    void compress() {
        if (m_last_compression_point == m_size) {
            return;
        }

        //make sure we're holding the GIL because we use python to do the serialization
        PyEnsureGilAcquired acquireTheGil;

        //replace the data we have here with a block of 4 bytes of size of compressed data and
        //then the data stream
        std::shared_ptr<ByteBuffer> buf = m_context.compress(m_buffer + m_last_compression_point, m_buffer + m_size);
        std::pair<uint8_t*, uint8_t*> range = buf->range();

        if (range.first == m_buffer + m_last_compression_point) {
            //we got back the original range
            ensure(sizeof(uint32_t), false);
            memmove(
                m_buffer + m_last_compression_point + sizeof(uint32_t),
                m_buffer + m_last_compression_point,
                m_size - m_last_compression_point
                );
            *(uint32_t*)(m_buffer + m_last_compression_point) = (range.second - range.first);

            m_size += sizeof(uint32_t);
            m_last_compression_point = m_size;
        } else {
            m_size = m_last_compression_point;

            write_uint32(range.second - range.first);
            write_bytes(range.first, range.second - range.first, false);

            m_last_compression_point = m_size;
        }
    }
    template< class T>
    void write(T i) {
        ensure(sizeof(i));
        *(T*)(m_buffer+m_size) = i;
        m_size += sizeof(i);
    }


private:
    const SerializationContext& m_context;

    uint8_t* m_buffer;
    size_t m_size;
    size_t m_reserved;
    size_t m_last_compression_point;

    std::map<void*, int32_t> m_idToPointerCache;

    std::map<Type*, std::vector<void*>> m_pointersNeedingDecref;
};
