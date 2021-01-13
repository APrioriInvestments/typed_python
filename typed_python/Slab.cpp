#include "Slab.hpp"
#include "Type.hpp"


void Slab::free(void* data) {
    if (mIsFreeStore) {
        ::tp_free(data);
    } else {

        if (mTrackAllocTypes) {
            std::lock_guard<std::mutex> lock(mAllocMutex);

            if (mAllocOrdering.find(data) == mAllocOrdering.end()) {
                throw std::runtime_error("Free of unknown alloc.");
            }

            if (mAliveAllocs.find(data) == mAliveAllocs.end()) {
                if (mAllocTypes.find(data) == mAllocTypes.end()) {
                    throw std::runtime_error("Double free of alloc of unknown type.");
                } else {
                    throw std::runtime_error("Double free of alloc of type " + mAllocTypes[data]->name());
                }
            }

            mAliveAllocs.erase(data);
        }

        decref();
    }
}
