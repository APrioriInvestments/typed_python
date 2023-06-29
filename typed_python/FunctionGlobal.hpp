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
    FunctionGlobal() :
        mKind(GlobalType::Unbound),
        mModuleDictOrCell(nullptr)
    {
    }

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

    // return an approriate FunctionGlobal given that we are indexing into 'funcGlobals'
    // with a dot-variable sequence. Returns a pair of (dotSequence, FunctionGlobal) that
    // we actually want to use
    static std::pair<std::string, FunctionGlobal> DottedGlobalsLookup(
        PyObject* funcGlobals,
        std::string dotSequence
    ) {
        if (!PyDict_Check(funcGlobals)) {
            throw std::runtime_error("Can't handle a non-dict function globals");
        }

        std::string shortGlobalName = dotSequence;

        size_t indexOfDot = shortGlobalName.find('.');
        if (indexOfDot != std::string::npos) {
            shortGlobalName = shortGlobalName.substr(0, indexOfDot);
        }

        PyObject* globalVal = PyDict_GetItemString(funcGlobals, shortGlobalName.c_str());

        if (!globalVal) {
            PyObject* builtins = PyDict_GetItemString(funcGlobals, "__builtins__");
            if (builtins && PyDict_Check(builtins) && PyDict_GetItemString(builtins, shortGlobalName.c_str())) {
                return std::make_pair(
                    shortGlobalName,
                    FunctionGlobal::NamedModuleMember(builtins, shortGlobalName)
                );
            }
        }

        // TODO: this should be more precise - we need to decide what to do if
        // this is a named global in a globally visible module vs a module
        // still being defined, or whatever
        return std::make_pair(
            shortGlobalName,
            FunctionGlobal::ForwardInDict(funcGlobals, shortGlobalName)
        );
    }

    // construct a Global from a cell. If the cell is defined, we can resolve immediately
    static FunctionGlobal FromCell(PyObject* cell) {
        // pass
        throw std::runtime_error("FunctionGlobal::FromCell not implemented yet");
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

    PyObject* getValueAsPyobj() {
        throw std::runtime_error("FunctionGlobal::getValueAsPyobj not implemented");
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

    template<class visitor_type>
    void _visitReferencedTypes(const visitor_type& visitor) {
        if (isUnbound()) {
            return;
        }

        if (isConstant()) {
            mConstant->_visitReferencedTypes(visitor);
            return;
        }

        //TODO: what should we do here?
        return;
    }

    template<class visitor_type>
    void _visitCompilerVisibleInternals(const visitor_type& visitor) {
        if (isUnbound()) {
            return;
        }

        if (isNamedModuleMember()) {
            // TODO: need to know whether we're an identity or compiler visitor?
            return;
        }

        if (isConstant()) {
            mConstant->_visitCompilerVisibleInternals(visitor);
            return;
        }

        if (isForwardInDict()) {
            return;
        }

        if (isForwardInCell()) {
            return;
        }

        // if (!PyCell_Check(nameAndGlobal.second)) {
        //     throw std::runtime_error(
        //         "A global in mFunctionGlobalsInCells is somehow not a cell"
        //     );
        // }

        // PyObject* o = PyCell_Get(nameAndGlobal.second);
        // Type* t = o && PyType_Check(o) ? PyInstance::extractTypeFrom(o) : nullptr;

        // if (t && t->isForwardDefined()) {
        //     if (t->isResolved()) {
        //         visitor.visitNamedTopo(
        //             nameAndGlobal.first,
        //             t->forwardResolvesTo()
        //         );
        //     } else {
        //         // deliberately ignore non-resolved forwards
        //         std::cout << "Deliberately ignoring " << nameAndGlobal.first << " -> " << TypeOrPyobj(t).name() << " since its not reoslved..\n";
        //     }
        // } else if (t) {
        //     visitor.visitNamedTopo(nameAndGlobal.first, t);
        // } else {
        //     visitor.visitNamedTopo(
        //         nameAndGlobal.first,
        //         nameAndGlobal.second
        //     );
        // }



        // or alternatively
        //
        // _visitCompilerVisibleGlobals([&](const std::string& name, PyObject* val) {
        //     Type* t = PyInstance::extractTypeFrom(val);

        //     if (t && t->isForwardDefined()) {
        //         if (t->isResolved()) {
        //             visitor.visitNamedTopo(
        //                 name,
        //                 t->forwardResolvesTo()
        //             );
        //         } else {
        //             // deliberately ignore non-resolved forwards
        //         }
        //     } else if (t) {
        //         visitor.visitNamedTopo(name, t);
        //     } else {
        //         visitor.visitNamedTopo(
        //             name,
        //             val
        //         );
        //     }
        // });

    }

    bool isUnresolved(bool insistForwardsResolved) {
        throw std::runtime_error("FunctionGlobal::isUnresolved not implemented yet");

        // auto it = mFunctionGlobalsInCells.find(name);
        // if (it != mFunctionGlobalsInCells.end()) {
        //     if (PyCell_Check(it->second) && PyCell_GET(it->second)) {
        //         if (insistForwardsResolved) {
        //             PyObject* o = PyCell_Get(it->second);
        //             Type* t = PyInstance::extractTypeFrom(o);
        //             if (t && t->isForwardDefined()) {
        //                 return true;
        //             }
        //         }

        //         return false;
        //     }

        //     return true;
        // }

        // if (mFunctionGlobals && PyDict_Check(mFunctionGlobals)) {
        //     if (PyDict_GetItemString(mFunctionGlobals, name.c_str())) {
        //         if (insistForwardsResolved) {
        //             PyObject* o = PyDict_GetItemString(mFunctionGlobals, name.c_str());
        //             Type* t = PyInstance::extractTypeFrom(o);
        //             if (t && t->isForwardDefined()) {
        //                 return true;
        //             }
        //         }

        //         return false;
        //     }

        //     PyObject* builtins = PyDict_GetItemString(mFunctionGlobals, "__builtins__");
        //     if (builtins && PyDict_GetItemString(builtins, name.c_str())) {
        //         return false;
        //     }
        // }

        // return true;
    }

    void autoresolveGlobal(
        const std::set<Type*>& resolvedForwards
    ) {
        throw std::runtime_error("FunctionGlobal::autoresolveGlobal not implemented yet");

        // if (it != mFunctionGlobalsInCells.end()) {
        //     if (PyCell_Check(it->second) && PyCell_GET(it->second)) {
        //         if (PyType_Check(PyCell_GET(it->second))) {
        //             Type* cellContents = PyInstance::extractTypeFrom(
        //                 (PyTypeObject*)PyCell_Get(it->second)
        //             );

        //             if (resolvedForwards.find(cellContents) != resolvedForwards.end()) {
        //                 PyCell_Set(
        //                     it->second,
        //                     (PyObject*)PyInstance::typeObj(
        //                         cellContents->forwardResolvesTo()
        //                     )
        //                 );
        //             }
        //         }
        //     }

        //     return;
        // }

        // PyObject* dictVal = PyDict_GetItemString(mFunctionGlobals, name.c_str());
        // if (dictVal && PyType_Check(dictVal)) {
        //     Type* cellContents = PyInstance::extractTypeFrom(
        //         (PyTypeObject*)dictVal
        //     );

        //     if (resolvedForwards.find(cellContents) != resolvedForwards.end()) {
        //         PyDict_SetItemString(
        //             mFunctionGlobals,
        //             name.c_str(),
        //             (PyObject*)PyInstance::typeObj(
        //                 cellContents->forwardResolvesTo()
        //             )
        //         );
        //     }
        // }
    }

    template<class serialization_context_t, class buf_t>
    void serialize(serialization_context_t& context, buf_t& buffer, int fieldNumber) const {
        // context.serializePythonObject(nameAndCell.second, buffer, varIx++);
        throw std::runtime_error("FunctionGlobal::serialize not implemented yet");
    }

    template<class serialization_context_t, class buf_t>
    static FunctionGlobal deserialize(serialization_context_t& context, buf_t& buffer, int wireType) {
        throw std::runtime_error("FunctionGlobal::deserialize not implemented yet");
        // functionGlobalsInCells[last].steal(context.deserializePythonObject(buffer, wireType));
        // functionGlobalsInCellsRaw[last] = functionGlobalsInCells[last];
    }

    bool operator<(const FunctionGlobal& g) const {
        if (mKind < g.mKind) {
            return true;
        }
        if (mKind > g.mKind) {
            return false;
        }

        if (mConstant < g.mConstant) {
            return true;
        }
        if (mConstant > g.mConstant) {
            return false;
        }

        if (mName < g.mName) {
            return true;
        }
        if (mName > g.mName) {
            return false;
        }

        if (mModuleDictOrCell < g.mModuleDictOrCell) {
            return true;
        }
        if (mModuleDictOrCell > g.mModuleDictOrCell) {
            return false;
        }

        return false;
    }

private:
    GlobalType mKind;

    CompilerVisiblePyObj* mConstant;

    std::string mName;
    PyObject* mModuleDictOrCell;
};
