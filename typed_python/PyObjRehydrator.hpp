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

class PyObjRehydrator {
public:
    void rehydrate(PyObjSnapshot* snapshot);

    Type* typeFor(PyObjSnapshot* snapshot);

    PyObject* pyobjFor(PyObjSnapshot* snapshot);

    void finalize();

    PyObject* getNamedElementPyobj(
        PyObjSnapshot* snapshot,
        std::string name,
        bool allowEmpty=false
    );

    Type* getNamedElementType(
        PyObjSnapshot* snapshot,
        std::string name,
        bool allowEmpty=false
    );

private:
    void rehydrateTpType(PyObjSnapshot* snap);

    void finalizeRehydration(PyObjSnapshot* snap);

    void finalizeRehydration2(PyObjSnapshot* snap);

    std::unordered_set<PyObjSnapshot*> mSnapshots;
};
