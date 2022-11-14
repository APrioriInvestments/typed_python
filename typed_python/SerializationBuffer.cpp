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
        PyEnsureGilReleased releaseTheGil;

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
    if (m_wants_compress) {
        for (auto blockPtr: m_blocks) {
            blockPtr->compress();
        }
    }

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
