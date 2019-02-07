#pragma once

#include <stdexcept>
#include <stdlib.h>
#include <map>
#include <set>

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
            reserve(m_size + t + 1024 * 128);

            //compress every 10 megs or so
            if (m_size - m_last_compression_point > 10 * 1024 * 1024) {
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
        Keep track of whether we've seen this pointer before

        @param t A PyObject pointer to be cached
        @return an std::pair containing its cache ID (the size of the cache)
                and 'false' if the pointer was already in the cache
    */
    std::pair<uint32_t, bool> cachePointer(PyObject* t) {
        auto it = m_idToPointerCache.find(t);
        if (it == m_idToPointerCache.end()) {
            Py_INCREF(t);
            m_pointersNeedingDecref.insert(t);
            uint32_t id = m_idToPointerCache.size();
            m_idToPointerCache[t] = id;
            return std::pair<uint32_t, bool>(id, true);
        }
        return std::pair<uint32_t, bool>(it->second, false);
    }

    std::pair<uint32_t, bool> cachePointer(Type* t) {
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

        //replace the data we have here with a block of 4 bytes of size of compressed data and
        //then the data stream
        std::string data(m_buffer + m_last_compression_point, m_buffer + m_size);

        data = m_context.compress(data);

        m_size = m_last_compression_point;
        write_string(data);
        m_last_compression_point = m_size;
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
    std::set<void*> m_pointersNeedingDecref;
};
