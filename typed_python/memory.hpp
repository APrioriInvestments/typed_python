/******************************************************************************
   Copyright 2017-2020 typed_python Authors

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

#define TP_TRACK_MEMORY_USAGE 1

//atomic storage for total number of bytes allocated
inline std::atomic<size_t>& tp_total_bytes_allocated() {
    static std::atomic<size_t> s;
    return s;
}


#if TP_TRACK_MEMORY_USAGE

#include <cstddef>

inline void* tp_malloc(size_t s) {
    if (s == 0) {
        return nullptr;
    }

    uint8_t* m = (uint8_t*)malloc(s + sizeof(std::max_align_t));
    ((size_t*)m)[0] = s;
    tp_total_bytes_allocated() += s + sizeof(std::max_align_t);
    return m + sizeof(std::max_align_t);
}

inline void tp_free(void* p) {
    if (!p) {
        return;
    }

    uint8_t* m = (uint8_t*)p - sizeof(std::max_align_t);
    tp_total_bytes_allocated() -= ((size_t*)m)[0] + sizeof(std::max_align_t);
    free(m);
}

inline void* tp_realloc(void* p, size_t newSize) {
    if (!p) {
        return tp_malloc(newSize);
    }

    if (p && !newSize) {
        tp_free(p);
        return nullptr;
    }

    uint8_t* m = (uint8_t*)p - sizeof(std::max_align_t);

    tp_total_bytes_allocated() -= ((size_t*)m)[0];
    tp_total_bytes_allocated() += newSize;

    ((size_t*)m)[0] = newSize;

    return (uint8_t*)realloc(m, newSize + sizeof(std::max_align_t)) + sizeof(std::max_align_t);
}

#else

inline void* tp_malloc(size_t s) {
    return malloc(s);
}

inline void tp_free(void* p) {
    free(p);
}

inline void* tp_realloc(void* p, size_t newSize) {
    return realloc(p, newSize);
}

#endif
