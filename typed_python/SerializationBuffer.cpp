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

void SerializationBuffer::compress() {
    if (m_last_compression_point == m_size) {
        return;
    }

    //replace the data we have here with a block of 4 bytes of size of compressed data and
    //then the data stream
    size_t bytesRequired = LZ4F_compressFrameBound(m_size - m_last_compression_point, nullptr);

    void* compressedBytes = malloc(bytesRequired);

    size_t compressedBytecount;

    {
        PyEnsureGilReleased releaseTheGil;

        compressedBytecount = LZ4F_compressFrame(
            compressedBytes,
            bytesRequired,
            m_buffer + m_last_compression_point,
            m_size - m_last_compression_point,
            nullptr
        );
    }

    m_size = m_last_compression_point;

    write<uint32_t>(compressedBytecount);

    write_bytes((uint8_t*)compressedBytes, compressedBytecount, false);

    free(compressedBytes);

    m_last_compression_point = m_size;
}
