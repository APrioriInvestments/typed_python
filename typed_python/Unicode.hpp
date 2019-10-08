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

#pragma once

inline size_t bytesForUtf8Codepoint(size_t codepoint) {
    if (codepoint < 0x80) {
        return 1;
    }
    if (codepoint < 0x800) {
        return 2;
    }
    if (codepoint < 0x10000) {
        return 3;
    }

    return 4;
}

template<class codepoint_type>
void encodeUtf8(codepoint_type* codepoints, int64_t sz, uint8_t* out) {
    for (long k = 0; k < sz; k++) {
        if (codepoints[k] < 0x80) {
            *(out++) = codepoints[k];
        } else if (codepoints[k] < 0x800) {
            *(out++) = ((codepoints[k] & 0b11111000000) >> 6) + 0b11000000;
            *(out++) = ((codepoints[k] & 0b00000111111)     ) + 0b10000000;
        } else if (codepoints[k] < 0x10000) {
            *(out++) = ((codepoints[k] & 0b1111000000000000) >> 12) + 0b11100000;
            *(out++) = ((codepoints[k] & 0b0000111111000000) >> 6 ) + 0b10000000;
            *(out++) = ((codepoints[k] & 0b0000000000111111)      ) + 0b10000000;
        } else {
            *(out++) = ((codepoints[k] & 0b111000000000000000000) >> 18) + 0b11110000;
            *(out++) = ((codepoints[k] & 0b000111111000000000000) >> 12) + 0b10000000;
            *(out++) = ((codepoints[k] & 0b000000000111111000000) >> 6 ) + 0b10000000;
            *(out++) = ((codepoints[k] & 0b000000000000000111111)      ) + 0b10000000;
        }
    }
}

template<class codepoint_type>
size_t countUtf8BytesRequiredFor(codepoint_type* codepoints, int64_t sz) {
    size_t result = 0;

    for (long k = 0; k < sz; k++) {
        result += bytesForUtf8Codepoint(codepoints[k]);
    }

    return result;
}

