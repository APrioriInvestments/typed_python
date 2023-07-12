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

#include <vector>
#include "Type.hpp"
#include "Instance.hpp"


/*********************************
CompilerVisiblePyObject

a representation of a python object that's owned by TypedPython.  We hold these by
pointer and leak them indiscriminately - like Type objects they're considered to be permanent
and singletonish

**********************************/

class CompilerVisiblePyObj {
    enum class Kind {
        Uninitialized = 0,
        Type = 1,
        Instance = 2,
        PyTuple = 3,
    };

    CompilerVisiblePyObj() :
        mKind(Kind::Uninitialized),
        mType(nullptr)
    {
    }

public:
    bool isUninitialized() const {
        return mKind == Kind::Uninitialized;
    }

    bool isType() const {
        return mKind == Kind::Type;
    }

    bool isInstance() const {
        return mKind == Kind::Instance;
    }

    bool isPyTuple() const {
        return mKind == Kind::PyTuple;
    }

    static CompilerVisiblePyObj* Type(Type* t) {
        CompilerVisiblePyObj* res = new CompilerVisiblePyObj();

        res->mKind = Kind::Type;
        res->mType = t;

        return res;
    }

    static CompilerVisiblePyObj* Instance(Instance i) {
        CompilerVisiblePyObj* res = new CompilerVisiblePyObj();

        res->mKind = Kind::Instance;
        res->mInstance = i;

        return res;
    }

    static CompilerVisiblePyObj* PyTuple() {
        CompilerVisiblePyObj* res = new CompilerVisiblePyObj();

        res->mKind = Kind::PyTuple;

        return res;
    }

    void append(CompilerVisiblePyObj* elt) {
        if (mKind != Kind::PyTuple) {
            throw std::runtime_error("Expected a PyTuple");
        }

        mElements.push_back(elt);
    }

    const std::vector<CompilerVisiblePyObj*>& elements() const {
        return mElements;
    }

    ::Type* getType() const {
        return mType;
    }

    const ::Instance& getInstance() const {
        return mInstance;
    }

    template<class visitor_type>
    void _visitReferencedTypes(const visitor_type& v) {
        if (mKind == Kind::Type) {
            v(mType);
            return;
        }

        if (mKind == Kind::Instance) {
            // TODO: what to do here?
        }

        if (mKind == Kind::PyTuple) {
            // TODO: what to do here?
        }
    }

    template<class visitor_type>
    void _visitCompilerVisibleInternals(const visitor_type& visitor) {

    }

    // get the python object representation of this object, which isn't guaranteed
    // to exist and may need to be constructed on demand.
    PyObject* getPyObj() {
        if (mKind == Kind::Type) {
            return (PyObject*)PyInstance::typeObj(mType);
        }

        throw std::runtime_error("Can't make a python object representation for this pyobj");
    }

    std::string toString() {
        if (mKind == Kind::Type) {
            return "CompilerVisiblePyObj.Type(" + mType->name() + ")";
        }

        if (mKind == Kind::Instance) {
            return "CompilerVisiblePyObj.Instance(" + mInstance.type()->name() + ")";
        }

        if (mKind == Kind::PyTuple) {
            return "CompilerVisiblePyObj.PyTuple()";
        }

        throw std::runtime_error("Unknown CompilerVisiblePyObj Kind.");
    }

private:
    Kind mKind;

    ::Type* mType;
    ::Instance mInstance;

    std::vector<CompilerVisiblePyObj*> mElements;
};
