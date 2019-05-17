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

#include <string.h>

/******************
HashValue

Stores a 160-bit hash.
******************/

class HashValue {
public:
    HashValue() {
        for (long i = 0; i < 5; i++) {
            m_digest[i] = 0;
        }
    }

    HashValue(uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t a4) {
        m_digest[0] = a0;
        m_digest[1] = a1;
        m_digest[2] = a2;
        m_digest[3] = a3;
        m_digest[4] = a4;
    }

    HashValue(const char* bytes) {
        strncpy((char*)m_digest, bytes, 20);
    }

    bool operator<(const HashValue& other) const {
        for (long i = 0; i < 5; i++) {
            if (m_digest[i] < other.m_digest[i]) {
                return true;
            }
            if (m_digest[i] > other.m_digest[i]) {
                return false;
            }
        }

        return false;
    }

    bool operator==(const HashValue& other) const {
        for (long i = 0; i < 5; i++) {
            if (m_digest[i] != other.m_digest[i]) {
                return false;
            }
        }

        return true;
    }

    const uint32_t* digest() const {
      return m_digest;
    }
private:
    uint32_t m_digest[5];
};
