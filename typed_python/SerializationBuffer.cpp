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

#include "SerializationBuffer.hpp"
#include "lz4frame.h"

/* static */
Bytes SerializationBuffer::serializeSingleBoolToBytes(bool value) {
    uint8_t existsValue[2] = { WireType::VARINT, value ? 1 : 0 };

    return Bytes((const char*)existsValue, 2);
}

void SerializationBufferBlock::compress() {
    if (m_compressed) {
        return;
    }

    LZ4F_preferences_t lz4Prefs;
    memset(&lz4Prefs, 0, sizeof(lz4Prefs));

    lz4Prefs.frameInfo.contentChecksumFlag = LZ4F_contentChecksumEnabled;
    lz4Prefs.frameInfo.blockChecksumFlag = LZ4F_blockChecksumEnabled;

    //replace the data we have here with a block of 4 bytes of size of compressed data and
    //then the data stream
    size_t bytesRequired = LZ4F_compressFrameBound(m_size, &lz4Prefs);

    void* compressedBytes = malloc(bytesRequired);

    size_t compressedBytecount;

    {
        compressedBytecount = LZ4F_compressFrame(
            compressedBytes,
            bytesRequired,
            m_buffer,
            m_size,
            &lz4Prefs
        );

        if (LZ4F_isError(compressedBytecount)) {
            free(compressedBytes);

            throw std::runtime_error(
              std::string("Error compressing data using LZ4: ")
                + LZ4F_getErrorName(compressedBytecount)
            );
        }
    }

    m_size = 0;

    write<uint32_t>(compressedBytecount);

    write_bytes((uint8_t*)compressedBytes, compressedBytecount);

    free(compressedBytes);

    m_compressed = true;
}

void SerializationBuffer::consolidate() {
    if (m_is_consolidated) {
        return;
    }

    if (m_wants_compress) {
        for (auto blockPtr: m_blocks) {
            waitForCompression(blockPtr);
        }
    }

    m_is_consolidated = true;

    if (m_blocks.size() == 1) {
        return;
    }

    size_t totalSize = 0;

    for (auto blockPtr: m_blocks) {
        totalSize += blockPtr->size();
    }

    SerializationBufferBlock* block = new SerializationBufferBlock();
    block->ensure(totalSize);

    for (auto blockPtr: m_blocks) {
        block->write_bytes(blockPtr->buffer(), blockPtr->size());
    }

    if (m_wants_compress) {
        block->markCompressed();
    }

    m_blocks.clear();
    m_blocks.push_back(
        std::shared_ptr<SerializationBufferBlock>(
            block
        )
    );
}

std::shared_ptr<SerializationBufferBlock> SerializationBuffer::getNextCompressTask() {
    std::unique_lock<std::mutex> lock(s_compress_thread_mutex);

    while (true) {
        if (s_waiting_compress_blocks.size()) {
            std::shared_ptr<SerializationBufferBlock> res =
                *s_waiting_compress_blocks.begin();

            s_working_compress_blocks.insert(res);
            s_waiting_compress_blocks.erase(res);

            return res;
        }

        s_has_work->wait(lock);
    }
}

void SerializationBuffer::compressionThread() {
    while (true) {
        std::shared_ptr<SerializationBufferBlock> task = getNextCompressTask();

        task->compress();

        std::unique_lock<std::mutex> lock(s_compress_thread_mutex);

        s_working_compress_blocks.erase(task);

        s_has_work->notify_all();
    }
}

void SerializationBuffer::waitForCompression(std::shared_ptr<SerializationBufferBlock> block) {
    if (!m_compress_using_threads) {
        PyEnsureGilReleased releaseTheGil;
        block->compress();
        return;
    }

    {
        std::unique_lock<std::mutex> lock(s_compress_thread_mutex);

        while (true) {
            if (
                s_waiting_compress_blocks.find(block) == s_waiting_compress_blocks.end()
                && s_working_compress_blocks.find(block) == s_working_compress_blocks.end()
            ) {
                // we're done
                return;
            }

            s_has_work->wait(lock);
        }
    }
}

void SerializationBuffer::markForCompression(std::shared_ptr<SerializationBufferBlock> block) {
    if (!m_compress_using_threads) {
        return;
    }

    {
        std::unique_lock<std::mutex> lock(s_compress_thread_mutex);

        if (!s_compress_threads.size()) {
            for (long i = 0; i < 4; i++) {
                s_compress_threads.push_back(
                    new std::thread(SerializationBuffer::compressionThread)
                );
            }

            s_has_work = new std::condition_variable();
        }

        s_waiting_compress_blocks.insert(block);
        s_has_work->notify_all();
    }
}

// static
std::mutex SerializationBuffer::s_compress_thread_mutex;

// static
std::condition_variable* SerializationBuffer::s_has_work;

// static
std::vector<std::thread*> SerializationBuffer::s_compress_threads;

// static
std::unordered_set<std::shared_ptr<SerializationBufferBlock> > SerializationBuffer::s_waiting_compress_blocks;

// static
std::unordered_set<std::shared_ptr<SerializationBufferBlock> > SerializationBuffer::s_working_compress_blocks;


