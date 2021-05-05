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

#include <Python.h>
#include <vector>
#include <atomic>
#include <cstddef>
#include <map>

class Type;

typedef uint8_t* instance_ptr;

// if we end up doing windows, we'll have to just use malloc
#define HAVE_MMAP 1

#ifdef HAVE_MMAP

#include <sys/mman.h>

#endif

/****************

Slab:

Models a contiguous block of memory with an embedded
typed python object graph. Objects within the graph cannot be modified, and are
released when no references are held to the slab.

If you release the reference to the slab and there are still external references to
any of the internal objects, then undefined behavior will result (probably a crash)
so don't do that. We will attempt to warn you by throwing an exception (and leaking the
slab) if we detect this case, but we can't guarantee simply by observing the object
refcounts that no external references exist because we are going to sweep through the
slab looking for incremented refcounts, and if you happen to incref something we already
looked at (and then decref something we're going to look at) you could retain
an undetected reference.

*****************/

class Slab {
public:
    Slab(bool isFreeStoreSlab, size_t slabSize) :
        mSlabBytecount(0),
        mSlabData(nullptr),
        mAllocationPoint(nullptr),
        mIsFreeStore(isFreeStoreSlab),
        mRefcount(1),
        mTrackAllocTypes(false),
        mTag(nullptr)
    {
        if (!mIsFreeStore) {
            if (slabSize > 1024 * 128 && HAVE_MMAP) {
                size_t pageSize = ::getpagesize();
                // round up to a page.
                if (slabSize % pageSize) {
                    slabSize = slabSize + (pageSize - slabSize % pageSize);
                }

                mSlabData = (instance_ptr)::mmap(NULL, slabSize, PROT_READ | PROT_WRITE, MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
            } else {
                mSlabData = (instance_ptr)::malloc(slabSize);
            }

            mAllocationPoint = mSlabData;
            mSlabBytecount = slabSize;
            totalBytesAllocatedInSlabs().fetch_add(slabSize);
        } else {
            if (slabSize > 0) {
                throw std::runtime_error("Allocate the free-store slab with 0 bytecount please.");
            }
        }

        std::lock_guard<std::mutex> guard(aliveSlabsMutex());
        aliveSlabs().insert(this);
    }

    void enableTrackAllocTypes() {
        mTrackAllocTypes = true;
    }

    static Slab* slabForAlloc(void* ptr) {
        if (!ptr) {
            return nullptr;
        }
        unsigned char* p = (unsigned char*)ptr - sizeof(std::max_align_t);

        if (*(int64_t*)p < 0) {
            return nullptr;
        }

        return *(Slab**)(p);
    }

    ~Slab() {
        if (!mIsFreeStore) {
            if (mSlabData) {
                if (mSlabBytecount > 1024 * 128 && HAVE_MMAP) {
                    ::munmap(mSlabData, mSlabBytecount);
                } else {
                    ::free(mSlabData);
                }
            }
            totalBytesAllocatedInSlabs().fetch_sub(mSlabBytecount);
        }

        if (mTag) {
            PyEnsureGilAcquired getTheGil;
            ::decref(mTag);
        }
    }

    void markAllocation(Type* t, void* alloc) {
        if (mTrackAllocTypes) {
            std::lock_guard<std::mutex> lock(mAllocMutex);

            mAllocTypes[alloc] = t;
            mAllocOrdering[alloc] = mAllocTypes.size() - 1;
            mAliveAllocs.insert(alloc);
            mAllocs.push_back(alloc);
        }
    }

    void incref() {
        mRefcount.fetch_add(1);
    }

    void free(void* data);

    void decref() {
        size_t oldRefcount = mRefcount.fetch_sub(1);

        if (oldRefcount == 1) {
            {
                std::lock_guard<std::mutex> guard(aliveSlabsMutex());
                aliveSlabs().erase(this);
            }

            delete this;
        }
    }

    size_t refcount() {
        return mRefcount;
    }

    bool isEmpty() {
        return mSlabData == nullptr;
    }

    size_t getBytecount() {
        return mSlabBytecount;
    }

    size_t getAllocated() {
        return mAllocationPoint - mSlabData;
    }

    static std::atomic<int64_t>& totalBytesAllocatedInSlabs() {
        static std::atomic<int64_t> res;
        return res;
    }

    void* allocate(size_t bytes, Type* t) {
        if (mIsFreeStore) {
            return ::tp_malloc(bytes);
        } else {
            if (bytes == 0) {
                return nullptr;
            }

            if (bytes % sizeof(std::max_align_t)) {
                bytes = bytes + sizeof(std::max_align_t) - (bytes % sizeof(std::max_align_t));
            }

            if (mAllocationPoint + bytes > mSlabData + mSlabBytecount) {
                throw std::runtime_error("Slab ran out of data.");
            }

            incref();

            void* res = mAllocationPoint + sizeof(std::max_align_t);
            ((Slab**)mAllocationPoint)[0] = this;

            mAllocationPoint += bytes + sizeof(std::max_align_t);

            markAllocation(t, res);

            return res;
        }
    }

    // if mTrackAllocTypes is enabled
    size_t allocCount() {
        return mAllocs.size();
    }

    size_t liveAllocCount() {
        return mAliveAllocs.size();
    }

    void* allocPointer(long allocIx) {
        if (allocIx < 0 || allocIx >= mAllocs.size()) {
            throw std::runtime_error("Alloc index out of bounds");
        }

        return mAllocs[allocIx];
    }

    bool allocIsAlive(long allocIx) {
        if (allocIx < 0 || allocIx >= mAllocs.size()) {
            throw std::runtime_error("Alloc index out of bounds");
        }

        return mAliveAllocs.find(mAllocs[allocIx]) != mAliveAllocs.end();
    }

    int64_t allocRefcount(long allocIx) {
        return *(size_t*)allocPointer(allocIx);
    }

    Type* allocType(long allocIx) {
        if (allocIx < 0 || allocIx >= mAllocs.size()) {
            throw std::runtime_error("Alloc index out of bounds");
        }

        return mAllocTypes[mAllocs[allocIx]];
    }

    static std::set<Slab*>& aliveSlabs() {
        static std::set<Slab*> res;
        return res;
    }

    static std::mutex& aliveSlabsMutex() {
        static std::mutex res;
        return res;
    }

    void setTag(PyObject* tag) {
        if (mTag) {
            ::decref(mTag);
        }

        mTag = ::incref(tag);
    }

    PyObject* getTag() {
        return mTag;
    }

private:
    // how many bytes we allocated
    size_t mSlabBytecount;

    std::atomic<int64_t> mRefcount;

    // a malloced block of data for the entire slab.
    instance_ptr mSlabData;

    // the allocation point within the slab
    instance_ptr mAllocationPoint;

    bool mIsFreeStore;

    bool mTrackAllocTypes;

    std::mutex mAllocMutex;
    std::unordered_map<void*, Type*> mAllocTypes;
    std::unordered_map<void*, int> mAllocOrdering;
    std::vector<void*> mAllocs;
    std::unordered_set<void*> mAliveAllocs;

    PyObject* mTag;
};
