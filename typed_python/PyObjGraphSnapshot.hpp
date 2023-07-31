/******************************************************************************
   Copyright 2017-2023 typed_python Authors

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

#include <unordered_set>
#include "PyObjSnapshotGroupSearch.hpp"
#include "ShaHash.hpp"

class PyObjSnapshot;


/*********************************
PyObjGraphSnapshot

Holds a collection of PyObjSnapshot objects. This graph object owns its snapshots
and frees them when it iself is release.

There's a global PyObjGraphSnapshot that holds all the interned objects accessible from

  PyObjGraphSnapshot::internal()

A graph knows how to assign a unique hash to every object it contains.

**********************************/

class PyObjGraphSnapshot {
public:
    PyObjGraphSnapshot() :
        mGroupsConsumed(0),
        mGroupSearch(this)
    {
    }

    ~PyObjGraphSnapshot() {
        for (auto oPtr: mObjects) {
            delete oPtr;
        }
    }

    // get the "internal" graph snapshot, which is responsible for holding all the objects
    // that are actually interned inside the system.
    static PyObjGraphSnapshot& internal() {
        static PyObjGraphSnapshot* graph = new PyObjGraphSnapshot();
        return *graph;
    }

    void registerSnapshot(PyObjSnapshot* obj) {
        if (obj->getGraph() != this) {
            throw std::runtime_error("Can't register a snapshot for a different graph");
        }

        mObjects.insert(obj);
    }

    const std::unordered_set<PyObjSnapshot*>& getObjects() const {
        return mObjects;
    }

    ShaHash hashFor(PyObjSnapshot* snap);

private:
    void installSnapHash(PyObjSnapshot* snap, ShaHash h);

    void computeHashesFor(const std::unordered_set<PyObjSnapshot*>& group);

    PyObjSnapshot* pickRootFor(const std::unordered_set<PyObjSnapshot*>& group);

    // all of our objects
    std::unordered_set<PyObjSnapshot*> mObjects;

    // for each snapshot, a hash
    std::unordered_map<PyObjSnapshot*, ShaHash> mSnapToHash;

    // for each hash, the object that matches it. It's possible to produce objects in a graph
    // that are indistinguishable from each other, in which case we simply pick one to go
    // 'first' and include that extra information in the hash.
    std::unordered_map<ShaHash, PyObjSnapshot*> mHashToSnap;

    // snaps with the same sha hash as each other
    std::unordered_set<PyObjSnapshot*> mRedundantSnaps;

    PyObjSnapshotGroupSearch mGroupSearch;
    long mGroupsConsumed;
};
