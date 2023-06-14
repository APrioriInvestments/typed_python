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

#include "Type.hpp"
#include "ReprAccumulator.hpp"

PyDoc_STRVAR(Forward_doc,
    "Forward(n) -> new forward type named n\n"
    "\n"
    "Forward types must be resolved before any types that contain them can be used.\n"
);

// forward types must be resolved (removed from the graph) before
// any types that contain them can be used.
class Forward : public Type {
public:
    Forward(std::string name, PyObject* cellOrDict=nullptr) :
        Type(TypeCategory::catForward),
        mTarget(nullptr),
        mCellOrDict(incref(cellOrDict))
    {
        m_name = name;
        m_is_forward_defined = true;
    }

    const char* docConcrete() {
        return Forward_doc;
    }

    std::string moduleNameConcrete() {
        if (mTarget) {
            return mTarget->moduleNameConcrete();
        }

        return "<unresolved>";
    }

    std::string nameWithModuleConcrete() {
        if (mTarget) {
            return mTarget->nameWithModule();
        }

        return m_name;
    }

    static Forward* Make() {
        return Make("unnamed");
    }

    static Forward* Make(std::string name) {
        return new Forward(name);
    }

    static Forward* MakeFromFunction(PyObject* funcObj) {
        if (!PyFunction_Check(funcObj)) {
            throw std::runtime_error(
                "Forwards can only be made from functions which return a single variable lookup."
            );
        }

        PyCodeObject* code = (PyCodeObject*)PyFunction_GetCode(funcObj);
        PyObject* closure = PyFunction_GetClosure(funcObj);

        if (!PyTuple_Check(code->co_names)) {
            throw std::runtime_error("Corrupt Forward lookup function: Code object co_names is not a tuple");
        }

        if (closure) {
            PyObjectStealer coFreevars(PyObject_GetAttrString(PyFunction_GetCode(funcObj), "co_freevars"));

            if (PyTuple_Size(code->co_names)) {
                throw std::runtime_error("Corrupt Forward lookup function: can't have more than one name");
            }

            if (!PyTuple_Check(coFreevars)) {
                throw std::runtime_error("Corrupt Forward lookup function: co_freevars was not a tuple");
            }

            if (PyTuple_Size(coFreevars) != PyTuple_Size(closure)) {
                throw std::runtime_error("Corrupt Forward lookup function: co_freevars not the same size as closure");
            }

            if (PyTuple_Size(coFreevars) != 1) {
                throw std::runtime_error("Corrupt Forward lookup function: can't have more than 1 free variable");
            }

            PyObject* varname = PyTuple_GetItem(coFreevars, 0);
            if (!PyUnicode_Check(varname)) {
                throw std::runtime_error("Corrupt Forward lookup function: closure varnames need to be strings");
            }

            std::string name = PyUnicode_AsUTF8(varname);

            if (!PyCell_Check(PyTuple_GetItem(closure, 0))) {
                throw std::runtime_error("Corrupt Forward lookup function: closure was not a cell");
            }

            return new Forward(name, PyTuple_GetItem(closure, 0));
        }

        if (PyTuple_Size(code->co_names) != 1) {
            throw std::runtime_error("Corrupt Forward lookup function: Code object co_names refers to more than 1 name");
        }

        if (!PyUnicode_Check(PyTuple_GetItem(code->co_names, 0))) {
            throw std::runtime_error("Corrupt Forward lookup function: Code object co_names refers to more than 1 name");
        }

        std::string name = PyUnicode_AsUTF8(PyTuple_GetItem(code->co_names, 0));
        return new Forward(name, PyFunction_GetGlobals(funcObj));
    }

    Type* getTargetTransitive() {
        if (!mTarget) {
            return nullptr;
        }

        if (!mTarget->isForward()) {
            return mTarget;
        }

        std::set<Type*> seen;

        Type* curFwd = this;

        while (curFwd->isForward()) {
            if (seen.find(curFwd) != seen.end()) {
                throw std::runtime_error("Forward cycle detected.");
            }

            seen.insert(curFwd);

            curFwd = ((Forward*)curFwd)->getTarget();

            if (!curFwd) {
                return nullptr;
            }
        }

        return curFwd;
    }

    Type* define(Type* target) {
        if (!target) {
            throw std::runtime_error("Can't resolve a Forward to the nullptr");
        }

        if (mTarget) {
            throw std::runtime_error("Forward is already resolved.");
        }

        mTarget = target;

        return target;
    }

    Type* getTarget() const {
        return mTarget;
    }

    template<class visitor_type>
    void _visitCompilerVisibleInternals(const visitor_type& v) {
        v.visitHash(ShaHash(1, m_typeCategory));
        if (mTarget) {
            v.visitHash(ShaHash(1));
            v.visitTopo(mTarget);
        } else {
            v.visitHash(ShaHash(2));
        }
    }

    template<class visitor_type>
    void _visitReferencedTypes(const visitor_type& v) {
        if (mTarget) {
            v(mTarget);
        }

        if (lambdaDefinitionPopulated()) {
            v(lambdaDefinition());
        }
    }

    bool hasLambdaDefinition() {
        return mCellOrDict != nullptr;
    }

    bool lambdaDefinitionPopulated() {
        if (!mCellOrDict) {
            return false;
        }

        if (PyCell_Check(mCellOrDict)) {
            return PyCell_Get(mCellOrDict) != nullptr;
        }

        if (PyDict_Check(mCellOrDict)) {
            return PyDict_GetItemString(mCellOrDict, m_name.c_str());
        }

        return false;
    }

    Type* lambdaDefinition();

    void installDefinitionIfLambda();

    void constructor(instance_ptr self) {
        throw std::runtime_error("Forward types should never be explicity instantiated.");
    }

    void destroy(instance_ptr self) {
        throw std::runtime_error("Forward types should never be explicity instantiated.");
    }

    void copy_constructor(instance_ptr self, instance_ptr other) {
        throw std::runtime_error("Forward types should never be explicity instantiated.");
    }

    void assign(instance_ptr self, instance_ptr other) {
        throw std::runtime_error("Forward types should never be explicity instantiated.");
    }

    template<class buf_t>
    void deserialize(instance_ptr self, buf_t& buffer, size_t wireType) {
        throw std::runtime_error("Forward types should never be explicity instantiated.");
    }

    template<class buf_t>
    void serialize(instance_ptr self, buf_t& buffer, size_t fieldNumber) {
        throw std::runtime_error("Forward types should never be explicity instantiated.");
    }

private:
    Type* mTarget;

    // if we are a recursive forward defined by a simple lambda function encoded by name,
    // then this will contain either a PyCell from the closure, or a PyDict in which
    // we are supposed to look to find a 'name'.
    PyObject* mCellOrDict;
};
