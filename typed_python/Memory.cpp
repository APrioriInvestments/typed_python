#include "Memory.hpp"
#include "Slab.hpp"

#include <cstddef>

static_assert(sizeof(Slab*) <= sizeof(std::max_align_t), "Can't fit a Slab* in the max_align_t?");

void* tp_malloc(size_t s) {
    if (s == 0) {
        return nullptr;
    }

    uint8_t* m = (uint8_t*)malloc(s + sizeof(std::max_align_t));

    ((int64_t*)m)[0] = -(int64_t)s;

    tpBytesAllocatedOnFreeStore() += s + sizeof(std::max_align_t);

    return m + sizeof(std::max_align_t);
}

void tp_free(void* p) {
    if (!p) {
        return;
    }

    uint8_t* m = (uint8_t*)p - sizeof(std::max_align_t);

    int64_t sizeOrSlab = ((int64_t*)m)[0];

    if (sizeOrSlab <= 0) {
        tpBytesAllocatedOnFreeStore() += sizeOrSlab - sizeof(std::max_align_t);
        free(m);
        return;
    }

    Slab* slab = ((Slab**)m)[0];
    slab->free(p);
}

void* tp_realloc(void* p, size_t oldSize, size_t newSize) {
    if (!p) {
        return tp_malloc(newSize);
    }

    if (p && !newSize) {
        tp_free(p);
        return nullptr;
    }

    uint8_t* m = (uint8_t*)p - sizeof(std::max_align_t);

    int64_t sizeOrSlab = ((int64_t*)m)[0];

    if (sizeOrSlab <= 0) {
        tpBytesAllocatedOnFreeStore() += (int64_t)newSize - (int64_t)oldSize;

        uint8_t* res = (uint8_t*)realloc(m, newSize + sizeof(std::max_align_t));

        *(int64_t*)res = -(int64_t)newSize;

        return res + sizeof(std::max_align_t);
    } else {
        Slab* slab = ((Slab**)m)[0];

        void* newData = tp_malloc(newSize);
        memcpy(newData, m + sizeof(std::max_align_t), std::min(newSize, oldSize));
        slab->free(p);

        return newData;
    }
}
