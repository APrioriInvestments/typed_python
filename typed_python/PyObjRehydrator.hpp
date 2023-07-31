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
#include "PyObjSnapshot.hpp"

class FunctionOverload;
class FunctionGlobal;

class PyObjRehydrator {
public:
    void start(PyObjSnapshot* snapshot);

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

    Instance getNamedElementInstance(
        PyObjSnapshot* snapshot,
        std::string name,
        bool allowEmpty=false
    );

    void getFrom(
        PyObjSnapshot* snapshot,
        Type*& out
    );

    void getFrom(
        PyObjSnapshot* snapshot,
        PyObject*& out
    );

    void getFrom(
        PyObjSnapshot* snapshot,
        Function*& out
    );

    void getFrom(
        PyObjSnapshot* snapshot,
        HeldClass*& out
    );

    void getFrom(
        PyObjSnapshot* snapshot,
        MemberDefinition& out
    );

    void getFrom(
        PyObjSnapshot* snapshot,
        FunctionOverload& out
    );

    void getFrom(
        PyObjSnapshot* snapshot,
        std::string& out
    );

    void getFrom(
        PyObjSnapshot* snapshot,
        FunctionGlobal& out
    );

    void getFrom(
        PyObjSnapshot* snapshot,
        FunctionArg& out
    );

    void getFrom(
        PyObjSnapshot* snapshot,
        ClosureVariableBinding& out
    );

    void getFrom(
        PyObjSnapshot* snapshot,
        ClosureVariableBindingStep& out
    );

    template<class T>
    void getNamedBundle(
        PyObjSnapshot* snapshot,
        std::string name,
        std::map<std::string, T>& out
    ) {
        auto it = snapshot->mNamedElements.find(name);
        if (it == snapshot->mNamedElements.end()) {
            return;
        }

        if (it->second->mKind != PyObjSnapshot::Kind::InternalBundle) {
            throw std::runtime_error("Corrupt PyObjSnapshot - expected a bundle");
        }

        for (auto nameAndType: it->second->mNamedElements) {
            T item;

            getFrom(nameAndType.second, item);

            out[nameAndType.first] = item;
        }
    }

    template<class T>
    void getNamedBundle(
        PyObjSnapshot* snapshot,
        std::string name,
        std::vector<T>& out
    ) {
        auto it = snapshot->mNamedElements.find(name);
        if (it == snapshot->mNamedElements.end()) {
            return;
        }

        if (it->second->mKind != PyObjSnapshot::Kind::InternalBundle) {
            throw std::runtime_error("Corrupt PyObjSnapshot - expected a bundle");
        }

        for (auto snap: it->second->mElements) {
            T item;

            getFrom(snap, item);

            out.push_back(item);
        }
    }

    template<class T>
    void getNamed(
        PyObjSnapshot* snapshot,
        std::string name,
        T& out
    ) {
        auto it = snapshot->mNamedElements.find(name);
        if (it == snapshot->mNamedElements.end()) {
            return;
        }

        if (it->second->mKind != PyObjSnapshot::Kind::InternalBundle) {
            throw std::runtime_error("Corrupt PyObjSnapshot - expected a bundle");
        }

        getFrom(it->second, out);
    }

private:

    void rehydrate(PyObjSnapshot* snapshot);

    void rehydrateTpType(PyObjSnapshot* snap);

    void finalizeRehydration(PyObjSnapshot* snap);

    std::unordered_set<PyObjSnapshot*> mSnapshots;
};
