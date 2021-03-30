#include "DeserializationBuffer.hpp"

#include "lz4frame.h"

bool DeserializationBuffer::decompress() {
    if (!m_context.isCompressionEnabled()) {
        if (m_compressed_block_data_remaining == 0) {
            return false;
        }

        //it's all one big uncompressed block
        m_decompressed_buffer.insert(
            m_decompressed_buffer.end(),
            m_compressed_blocks,
            m_compressed_blocks + m_compressed_block_data_remaining
        );
        m_size += m_compressed_block_data_remaining;
        m_read_head = &m_decompressed_buffer[0];
        m_compressed_block_data_remaining = 0;
        m_read_head_offset = 0;

        return true;
    } else {
        PyEnsureGilReleased releaseTheGil;

        if (m_compressed_block_data_remaining < sizeof(uint32_t)) {
            return false;
        }

        uint32_t bytesToDecompress = *((uint32_t*)m_compressed_blocks);

        if (bytesToDecompress + sizeof(uint32_t) > m_compressed_block_data_remaining) {
            throw std::runtime_error("Corrupt data: can't decompress this large of a block");
        }

        m_compressed_blocks += sizeof(uint32_t) + bytesToDecompress;
        m_compressed_block_data_remaining -= sizeof(uint32_t) + bytesToDecompress;

        LZ4F_decompressionContext_t compressionContext;

        if (LZ4F_createDecompressionContext(&compressionContext, LZ4F_VERSION)) {
            throw std::runtime_error("Failed to allocate an lz4 compression context.");
        }

        size_t bytesDecompressed = 0;

        while (bytesDecompressed < bytesToDecompress) {
            unsigned char compressionBuffer[1024*1024];

            size_t bytesWritten = 1024 * 1024;
            size_t bytesRead = bytesToDecompress - bytesDecompressed;

            LZ4F_decompress(
                compressionContext,
                compressionBuffer,
                &bytesWritten,
                m_compressed_blocks - bytesToDecompress + bytesDecompressed,
                &bytesRead,
                nullptr
            );

            if (m_read_head_offset) {
                m_decompressed_buffer.erase(
                    m_decompressed_buffer.begin(),
                    m_decompressed_buffer.begin() + m_read_head_offset
                );

                m_read_head_offset = 0;
            }

            m_decompressed_buffer.insert(
                m_decompressed_buffer.end(),
                compressionBuffer,
                compressionBuffer + bytesWritten
            );
            m_size += bytesWritten;
            m_read_head = &m_decompressed_buffer[0];

            bytesDecompressed += bytesRead;
        }

        LZ4F_freeDecompressionContext(compressionContext);

        return true;
    }
}
