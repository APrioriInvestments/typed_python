/******************************************************************************
   Copyright 2017-2019 Nativepython Authors

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
    T* addCachedPointer(int32_t which, T* ptr, Type* decrefType=nullptr) {
        if (!ptr) {
            throw std::runtime_error("Corrupt data: can't write a null cache pointer.");
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

        //hold the GIL because we use python to do this
        PyEnsureGilAcquired acquireTheGil;

        uint32_t bytesToDecompress = *((uint32_t*)m_compressed_blocks);

        if (bytesToDecompress + sizeof(uint32_t) > m_compressed_block_data_remaining) {
            throw std::runtime_error("Corrupt data: can't decompress this large of a block");
        }

        m_compressed_blocks += sizeof(uint32_t) + bytesToDecompress;
        m_compressed_block_data_remaining -= sizeof(uint32_t) + bytesToDecompress;

        std::shared_ptr<ByteBuffer> buf = m_context.decompress(m_compressed_blocks - bytesToDecompress, m_compressed_blocks);

        pushDecompressedBuffer(buf);

        return true;
    }

    void pushDecompressedBuffer(std::shared_ptr<ByteBuffer> buf) {
        m_decompressed_buffer.erase(m_decompressed_buffer.begin(), m_decompressed_buffer.begin() + m_read_head_offset);
        m_read_head_offset = 0;

        m_decompressed_buffer.insert(m_decompressed_buffer.end(), buf->range().first, buf->range().second);
        m_size += buf->range().second - buf->range().first;

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

    // maps indices to the pointers we've cached under that index.
    std::vector<void*> m_cachedPointers;

    //a map from the object type to each object that needs decreffing.
    //it must be the case that a pointer is the _natural_ layout of the
    //object (e.g. PyObject, Dict, etc). We pass a pointer to this
    //as the actual instance to the 'destroy' operation
    std::map<Type*, std::vector<void*> > m_needs_decref;
};