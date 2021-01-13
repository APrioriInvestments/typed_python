#include "Memory.hpp"
#include "Slab.hpp"

#include <cstddef>

static_assert(sizeof(Slab*) <= sizeof(std::max_align_t), "Can't fit a Slab* in the max_align_t?");

void* tp_malloc(size_t s) {
    if (s == 0) {
        return nullptr;
    }

    uint8_t* m = (uint8_t*)malloc(s + sizeof(std::max_align_t));

    ((Slab**)m)[0] = nullptr;

    return m + sizeof(std::max_align_t);
}

void tp_free(void* p) {
    if (!p) {
        return;
    }

    uint8_t* m = (uint8_t*)p - sizeof(std::max_align_t);

    Slab* slab = ((Slab**)m)[0];

    if (!slab) {
        free(m);
    } else {
        slab->free(p);
    }
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

    Slab* slab = ((Slab**)m)[0];

    if (slab) {
        void* newData = tp_malloc(newSize);
        memcpy(newData, m + sizeof(std::max_align_t), std::min(newSize, oldSize));
        slab->free(p);

        return newData;
    } else {
        return (uint8_t*)realloc(m, newSize + sizeof(std::max_align_t)) + sizeof(std::max_align_t);
    }
}
