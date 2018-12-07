#pragma once

#include <stdexcept>
#include <stdlib.h>
#include "SerializationContext.hpp"

class DeserializationBuffer {
public:
    DeserializationBuffer(uint8_t* ptr, size_t sz, const SerializationContext& context) :
            m_buffer(ptr),
            m_size(sz),
            m_orig_size(sz),
            m_context(context)
    {
    }

    ~DeserializationBuffer() {
        return;
        for (long k = 0; k < m_needsPyDecref.size(); k++) {
            if (m_needsPyDecref[k]) {
                Py_DECREF((PyObject*)m_cachedPointers[k]);
            }
        }
    }

    DeserializationBuffer(const DeserializationBuffer&) = delete;
    DeserializationBuffer& operator=(const DeserializationBuffer&) = delete;

    uint8_t read_uint8() {
        return read<uint8_t>();
    }
    uint32_t read_uint32() {
        return read<uint32_t>();
    }
    uint64_t read_uint64() {
        return read<uint64_t>();
    }
    int64_t read_int64() {
        return read<int64_t>();
    }
    double read_double() {
        return read<double>();
    }

    template<class initfun>
    auto read_bytes_fun(size_t bytecount, const initfun& f) -> decltype(f((uint8_t*)nullptr)) {
        if (m_size < bytecount) {
            throw std::runtime_error("out of data");
        }

        m_size -= bytecount;
        m_buffer += bytecount;

        return f(m_buffer - bytecount);
    }
    void read_bytes(uint8_t* ptr, size_t bytecount) {
        if (m_size < bytecount) {
            throw std::runtime_error("out of data");
        }
        memcpy(ptr,m_buffer,bytecount);

        m_size -= bytecount;
        m_buffer += bytecount;
    }

    std::string readString() {
        int32_t sz = read_uint32();
        return read_bytes_fun(sz, [&](uint8_t* ptr) {
            return std::string(ptr, ptr + sz);
        });
    }

    size_t remaining() const {
        return m_size;
    }

    size_t pos() const {
        return m_orig_size - m_size;
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
    T* addCachedPointer(int32_t which, T* ptr, bool needsPyDecref=false) {
        while (which >= m_cachedPointers.size()) {
            m_cachedPointers.push_back(nullptr);
            m_needsPyDecref.push_back(false);
        }
        if (!ptr) {
            throw std::runtime_error("Corrupt data: can't write a null cache pointer.");
        }
        if (m_cachedPointers[which]) {
            throw std::runtime_error("Corrupt data: tried to write a recursive object multiple times");
        }
        m_cachedPointers[which] = ptr;
        m_needsPyDecref[which] = needsPyDecref;
        return ptr;
    }

private:
    template<class T>
    T read() {
        if (m_size < sizeof(T)) {
            throw std::runtime_error("out of data");
        }
        T* ptr = (T*)m_buffer;

        m_size -= sizeof(T);
        m_buffer += sizeof(T);

        return *ptr;
    }


    uint8_t* m_buffer;
    size_t m_size;
    size_t m_orig_size;
    const SerializationContext& m_context;
    std::vector<void*> m_cachedPointers;
    std::vector<bool> m_needsPyDecref;
};