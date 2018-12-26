#pragma once

#include <stdexcept>
#include <stdlib.h>
#include <map>
#include <set>
#include "SerializationContext.hpp"

class Type;

class SerializationBuffer {
public:
    SerializationBuffer(const SerializationContext& context) :
            m_buffer(nullptr),
            m_size(0),
            m_reserved(0),
            m_context(context)
    {
    }

    ~SerializationBuffer() {
        if (m_buffer) {
            free(m_buffer);
        }

        for (auto p: m_pointersNeedingDecref) {
            Py_DECREF(p);
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
    void write_bytes(uint8_t* ptr, size_t bytecount) {
        ensure(bytecount);
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

    void ensure(size_t t) {
        if (m_size + t > m_reserved) {
            m_reserved = m_size + t + 1024 * 128;
            m_buffer = (uint8_t*)::realloc(m_buffer, m_reserved);
        }
    }

    const SerializationContext& getContext() const {
        return m_context;
    }

    /**
        Keep track of whether we've seen this pointer before

        @param t A PyObject pointer to be cached
        @return an std::pair containing its cache ID (the size of the cache)
                and 'false' if the pointer was already in the cache
    */
    std::pair<uint32_t, bool> cachePointer(PyObject* t) {
        Py_INCREF(t);
        m_pointersNeedingDecref.insert(t);

        auto it = m_idToPointerCache.find(t);
        if (it == m_idToPointerCache.end()) {
            m_idToPointerCache[t] = m_idToPointerCache.size();
            return std::pair<uint32_t, bool>(m_idToPointerCache.size() - 1, true);
        }
        return std::pair<uint32_t, bool>(it->second, false);
    }

    std::pair<uint32_t, bool> cachePointer(Type* t) {
        auto it = m_idToPointerCache.find(t);
        if (it == m_idToPointerCache.end()) {
            m_idToPointerCache[t] = m_idToPointerCache.size();
            return std::pair<uint32_t, bool>(m_idToPointerCache.size() - 1, true);
        }
        return std::pair<uint32_t, bool>(it->second, false);
    }

private:
    template< class T>
    void write(T i) {
        ensure(sizeof(i));
        *(T*)(m_buffer+m_size) = i;
        m_size += sizeof(i);
    }

    uint8_t* m_buffer;
    size_t m_size;
    size_t m_reserved;
    const SerializationContext& m_context;

    std::map<void*, int32_t> m_idToPointerCache;
    std::set<void*> m_pointersNeedingDecref;
};
