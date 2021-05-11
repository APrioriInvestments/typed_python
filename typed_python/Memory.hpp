/******************************************************************************
   Copyright 2017-2021 typed_python Authors

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

/**************

Memory:

This file defines the typed-python memory allocation model. In order to
track memory usage and to allow memory to be allocated in slabs (and eventually
maybe to allow explicit arena management for large applications), our
memory management routines pack an extra word at the beginning of every
allocation.

The word can either be the bytecount of the allocation with the top bit set,
indicating that this is a direct allocation from malloc, or it can be a pointer
to a Slab object which we decref when the allocation is released.

***************/

#include <cstddef>
#include <atomic>

extern "C" {

void* tp_malloc(size_t bytes);
void* tp_realloc(void* ptr, size_t oldBytes, size_t newBytes);
void tp_free(void* ptr);

}

inline std::atomic<size_t>& tpBytesAllocatedOnFreeStore() {
   static std::atomic<size_t> allocatedCount;
   return allocatedCount;
}

// how many bytes are required to back an allocation of size 's'
// accounts for alignment and extra pointers.
inline size_t bytesRequiredForAllocation(size_t s) {
   if (s == 0) {
      return 0;
   }

   if (s % sizeof(std::max_align_t)) {
      s += sizeof(std::max_align_t) - (s % sizeof(std::max_align_t));
   }

   return s + sizeof(std::max_align_t);
}
