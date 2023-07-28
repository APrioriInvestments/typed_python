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


class PyObjSnapshot;


/*********************************
PyObjGraphSnapshot

Holds a collection of PyObjSnapshot objects. When this dies, they'll die.

There's a global PyObjGraphSnapshot that holds all the interned objects.

**********************************/

class PyObjGraphSnapshot {
public:
    PyObjGraphSnapshot() {}

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

private:
    std::unordered_set<PyObjSnapshot*> mObjects;
};
