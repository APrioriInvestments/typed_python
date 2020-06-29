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

#include "Sha1.hpp"

class ShaHash;
ShaHash operator+(const ShaHash& l, const ShaHash& r);

class ShaHash {
public:
    ShaHash() {
        for (int32_t k = 0; k < 5;k++)
            mData[k] = 0;
    }

    ShaHash(uint64_t t) {
        for (int32_t k = 0; k < 5;k++)
            mData[k] = 0;
        mData[0] = t & 0xFFFFFFFF;
        mData[1] = (t >> 32) & 0xFFFFFFFF;
    }

    ShaHash(uint64_t t1, uint64_t t2) {
        for (int32_t k = 0; k < 5;k++)
            mData[k] = 0;

        mData[0] = t1 & 0xFFFFFFFF;
        mData[1] = (t1 >> 32) & 0xFFFFFFFF;
        mData[2] = t2 & 0xFFFFFFFF;
        mData[3] = (t2 >> 32) & 0xFFFFFFFF;
    }

    ShaHash(const std::string& s) {
        *this = SHA1(s.c_str(), s.size());
    }

    ShaHash(const ShaHash& in) {
        *this = in;
    }

    uint32_t& operator[](int32_t ix) {
        return mData[ix];
    }

    const uint32_t& operator[](int32_t ix) const {
        return mData[ix];
    }

    bool operator==(const ShaHash& in) const {
        return cmp(in) == 0;
    }

    bool operator!=(const ShaHash& in) const {
        return cmp(in) != 0;
    }

    bool operator<(const ShaHash& in) const {
        return cmp(in) == -1;
    }

    bool operator<=(const ShaHash& in) const {
        return cmp(in) <= 0;
    }

    bool operator>(const ShaHash& in) const {
        return cmp(in) == 1;
    }

    bool operator>=(const ShaHash& in) const {
        return cmp(in) >= 0;
    }

    ShaHash& operator+=(const ShaHash& in) {
        *this = *this + in;
        return *this;
    }

    std::string digestAsString() const {
        return std::string((uint8_t*)mData, (uint8_t*)mData + sizeof(ShaHash));
    }

    std::string digestAsHexString() const {
        char output[100];

        uint8_t* datPtr = (uint8_t*)mData;

        int pos = 0;
        for (int i = 0; i < sizeof(*this); i++) {
            pos += sprintf(
                output + pos,
                "%02X",
                datPtr[i]
            );
        }

        return "0x" + std::string(output);
    }

    static ShaHash fromDigest(std::string s) {
        if (s.size() != sizeof(ShaHash)) {
            throw std::runtime_error("Incorrect size for hash-digest");
        }

        ShaHash res;
        memcpy((void*)&res, s.c_str(), sizeof(ShaHash));

        return res;
    }

    // create a 'poison' sha hash that, like a nan in float-land,
    // always produces another poison sha hash when added. We can use this
    // to indicate that a hash is 'bad' in some dimension.
    static ShaHash poison() {
        ShaHash res;
        for (int i = 0; i < 5; i++) {
            res[i] = 0xFFFFFFFF;
        }
        return res;
    }

    bool isPoison() const {
        for (int i = 0; i < 5; i++) {
            if (mData[i] != 0xFFFFFFFF) {
                return false;
            }
        }

        return true;
    }

    int32_t cmp(const ShaHash& in) const {
        for (int32_t k = 0; k < 5;k ++) {
            if (mData[k] < in.mData[k]) {
                return -1;
            } else
            if (mData[k] > in.mData[k]) {
                return 1;
            }
        }

        return 0;
    }

    inline static ShaHash SHA1(std::string s) {
        return SHA1(s.c_str(), s.size());
    }

    static ShaHash SHA1(const void* data, size_t sz) {

        ShaHash tr;

        if (sz) {
            ::mbedtls_sha1_ret((const unsigned char*)data, sz, (unsigned char*)&tr);
        }

        return tr;
    }

private:
    uint32_t mData[5];
};


inline ShaHash operator+(const ShaHash& l, const ShaHash& r) {
    if (l.isPoison() || r.isPoison()) {
        return ShaHash::poison();
    }

    char data[sizeof(ShaHash) * 2];
    ((ShaHash*)data)[0] = l;
    ((ShaHash*)data)[1] = r;

    ShaHash tr = ShaHash::SHA1(data, sizeof(ShaHash)*2);

    return tr;
}
