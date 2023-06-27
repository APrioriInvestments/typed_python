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

#include "CompilerVisiblePyObj.hpp"


class FunctionGlobal {
    enum class GlobalType {
        Unbound = 1, // this global is not bound to anything - its a Name error
        NamedModuleMember = 2, // this global is a member of a particular module
        Constant = 3, // this global is bound to a constant that we resolved to at some point
        ForwardInDict = 4, // this global is an unresolved variable residing in some python dict
        ForwardInCell = 5, // this global is an unresolved variable residing in some PyCell
    };

    FunctionGlobal() :
        mKind(GlobalType::Unbound),
        mModuleDictOrCell(nullptr)
    {
    }

    FunctionGlobal(
        GlobalType inKind,
        PyObject* dictOrCell,
        std::string name,
        CompilerVisiblePyObj* constant
    ) :
        mKind(inKind),
        mModuleDictOrCell(incref(dictOrCell)),
        mName(name),
        mConstant(constant)
    {
    }

public:
    static FunctionGlobal Unbound() {
        return FunctionGlobal();
    }

    static FunctionGlobal Constant(CompilerVisiblePyObj* constant) {
        return FunctionGlobal(
            GlobalType::Constant,
            nullptr,
            "",
            constant
        );
    }

    static FunctionGlobal NamedModuleMember(PyObject* moduleDict, std::string name) {
        if (!PyDict_Check(moduleDict)) {
            throw std::runtime_error("NamedModuleMember requires a Python dict object");
        }

        return FunctionGlobal(
            GlobalType::NamedModuleMember,
            moduleDict,
            name,
            nullptr
        );
    }

    static FunctionGlobal Constant(CompilerVisiblePyObj* constant) {
        return FunctionGlobal(
            GlobalType::Constant,
            nullptr,
            "",
            constant
        );
    }

    static FunctionGlobal ForwardInDict(PyObject* dict, std::string name) {
        if (!PyDict_Check(dict)) {
            throw std::runtime_error("ForwardInDict requires a Python dict object");
        }

        return FunctionGlobal(
            GlobalType::ForwardInDict,
            dict,
            name,
            nullptr
        );
    }

    static FunctionGlobal ForwardInCell(PyObject* cell, std::string name) {
        if (!PyCell_Check(cell)) {
            throw std::runtime_error("ForwardInCell requires a Python cell object");
        }

        return FunctionGlobal(
            GlobalType::ForwardInCell,
            cell,
            name,
            nullptr
        );
    }

    bool isUnbound() const {
        return mKind == GlobalType::Unbound;
    }

    bool isNamedModuleMember() const {
        return mKind == GlobalType::NamedModuleMember;
    }

    bool isConstant() const {
        return mKind == GlobalType::Constant;
    }

    bool isForwardInDict() const {
        return mKind == GlobalType::ForwardInDict;
    }

    bool isForwardInCell() const {
        return mKind == GlobalType::ForwardInCell;
    }

    GlobalType getKind() const {
        return mKind;
    }

    CompilerVisiblePyObj* getConstant() const {
        return mConstant;
    }

    const std::string& getName() const {
        return mName;
    }

    PyObject* getModuleDictOrCell() const {
        return mModuleDictOrCell;
    }

private:
    GlobalType mKind;

    CompilerVisiblePyObj* mConstant;

    std::string mName;
    PyObject* mModuleDictOrCell;
};
