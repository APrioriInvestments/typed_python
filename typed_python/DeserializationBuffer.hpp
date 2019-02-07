#pragma once

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

    std::string readString() {
        int32_t sz = read_uint32();
        return read_bytes_fun(sz, [&](uint8_t* ptr) {
            return std::string(ptr, ptr + sz);
        });
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
    T* addCachedPointer(int32_t which, T* ptr, bool needsPyDecref=false) {
        if (!ptr) {
            throw std::runtime_error("Corrupt data: can't write a null cache pointer.");
        }
        while (which >= m_cachedPointers.size()) {
            m_cachedPointers.push_back(nullptr);
            m_needsPyDecref.push_back(false);
        }
        if (m_cachedPointers[which]) {
            throw std::runtime_error("Corrupt data: tried to write a recursive object multiple times");
        }
        m_cachedPointers[which] = ptr;
        m_needsPyDecref[which] = needsPyDecref;

        return ptr;
    }

    template<class T>
    void readInto(T* out) {
        *out = read<T>();
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
    bool decompress() {
        if (m_compressed_block_data_remaining < sizeof(uint32_t)) {
            return false;
        }

        uint32_t bytesToDecompress = *((uint32_t*)m_compressed_blocks);

        if (bytesToDecompress + sizeof(uint32_t) > m_compressed_block_data_remaining) {
            throw std::runtime_error("Corrupt data: can't decompress this large of a block");
        }

        m_compressed_blocks += sizeof(uint32_t);
        m_compressed_block_data_remaining -= sizeof(uint32_t);

        std::string toDecompress(m_compressed_blocks, m_compressed_blocks + bytesToDecompress);
        m_compressed_blocks += bytesToDecompress;
        m_compressed_block_data_remaining -= bytesToDecompress;

        std::string decompressed = m_context.decompress(toDecompress);

        pushStringIntoDecompressedBuffer(decompressed);

        return true;
    }

    void pushStringIntoDecompressedBuffer(std::string decompressed) {
        std::vector<uint8_t> newData(m_decompressed_buffer.begin() + m_read_head_offset, m_decompressed_buffer.end());
        m_read_head_offset = 0;

        newData.insert(newData.end(), decompressed.begin(), decompressed.end());
        m_size += decompressed.size();

        std::swap(newData, m_decompressed_buffer);
        m_read_head = &m_decompressed_buffer[0];
    }

    const SerializationContext& m_context;

    std::vector<uint8_t> m_decompressed_buffer;
    uint8_t* m_read_head;
    size_t m_read_head_offset;
    size_t m_size;

    uint8_t* m_compressed_blocks;
    size_t m_compressed_block_data_remaining;

    size_t m_pos;

    // These two vectors implement the pointer-cache datastructure. They map
    // ids (indices) to pointers and to the whether they need decref-ing
    std::vector<void*> m_cachedPointers;
    std::vector<bool> m_needsPyDecref;
};