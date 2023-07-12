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
        GlobalInDict = 4, // this global is an unresolved variable residing in some python dict
        GlobalInCell = 5, // this global is an unresolved variable residing in some PyCell
    };

    FunctionGlobal(
        GlobalType inKind,
        PyObject* dictOrCell,
        std::string name,
        std::string moduleName,
        CompilerVisiblePyObj* constant
    ) :
        mKind(inKind),
        mModuleDictOrCell(incref(dictOrCell)),
        mName(name),
        mModuleName(name),
        mConstant(constant)
    {
    }

public:
    FunctionGlobal() :
        mKind(GlobalType::Unbound),
        mModuleDictOrCell(nullptr)
    {
    }

    std::string toString() {
        if (isUnbound()) {
            return "FunctionGlobal.Unbound()";
        }

        if (isNamedModuleMember()) {
            return "FunctionGlobal.NamedModuleMember(" + mModuleName + ", " + mName + ")";
        }

        if (isGlobalInCell()) {
            return "FunctionGlobal.GlobalInCell()";
        }

        if (isGlobalInDict()) {
            return "FunctionGlobal.GlobalInDict()";
        }

        if (isConstant()) {
            return "FunctionGlobal.Constant(" + mConstant->toString() + ")";
        }

        throw std::runtime_error("Unknown FunctionGlobal Kind");
    }

    static FunctionGlobal Unbound() {
        return FunctionGlobal();
    }

    static FunctionGlobal Constant(CompilerVisiblePyObj* constant) {
        return FunctionGlobal(
            GlobalType::Constant,
            nullptr,
            "",
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
    );

    static FunctionGlobal NamedModuleMember(PyObject* moduleDict, std::string moduleName, std::string name) {
        if (!PyDict_Check(moduleDict)) {
            throw std::runtime_error("NamedModuleMember requires a Python dict object");
        }

        return FunctionGlobal(
            GlobalType::NamedModuleMember,
            moduleDict,
            name,
            moduleName,
            nullptr
        );
    }

    static FunctionGlobal GlobalInDict(PyObject* dict, std::string name) {
        if (!PyDict_Check(dict)) {
            throw std::runtime_error("GlobalInDict requires a Python dict object");
        }

        return FunctionGlobal(
            GlobalType::GlobalInDict,
            dict,
            name,
            "",
            nullptr
        );
    }

    static FunctionGlobal GlobalInCell(PyObject* cell) {
        if (!PyCell_Check(cell)) {
            throw std::runtime_error("GlobalInCell requires a Python cell object");
        }

        return FunctionGlobal(
            GlobalType::GlobalInCell,
            cell,
            "",
            "",
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

    bool isGlobalInDict() const {
        return mKind == GlobalType::GlobalInDict;
    }

    bool isGlobalInCell() const {
        return mKind == GlobalType::GlobalInCell;
    }

    Type* getValueAsType() {
        PyObject* obj = getValueAsPyobj();
        if (obj) {
            return PyInstance::extractTypeFrom(obj);
        }
        return nullptr;
    }

    PyObject* getValueAsPyobj() {
        if (isGlobalInCell() || isGlobalInCell() || isNamedModuleMember()) {
            return extractGlobalRefFromDictOrCell();
        }

        if (isUnbound()) {
            throw std::runtime_error("Unbound globals don't have python values.");
        }

        if (isConstant()) {
            return mConstant->getPyObj();
        }


        throw std::runtime_error("Unknown global kind.");
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

    const std::string& getModuleName() const {
        return mModuleName;
    }

    PyObject* getModuleDictOrCell() const {
        return mModuleDictOrCell;
    }

    PyObject* extractGlobalRefFromDictOrCell() {
        if (isGlobalInDict()) {
            return PyDict_GetItemString(mModuleDictOrCell, mName.c_str());
        }

        if (isGlobalInCell()) {
            return PyCell_Get(mModuleDictOrCell);
        }

        if (isNamedModuleMember()) {
            return PyDict_GetItemString(mModuleDictOrCell, mName.c_str());
        }

        return nullptr;
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

        if (isGlobalInDict() || isGlobalInCell() || isNamedModuleMember()) {
            PyObject* obj = extractGlobalRefFromDictOrCell();

            if (obj) {
                _visitReferencedTypesInPyobj(obj, visitor);
            }
            return;
        }
    }

    template<class visitor_type>
    void _visitReferencedTypesInPyobj(PyObject* obj, visitor_type vis) {
        if (!obj) {
            return;
        }

        if (!PyType_Check(obj)) {
            return;
        }

        Type* t = PyInstance::extractTypeFrom(obj);

        if (t) {
            vis(t);
        }
    }

    template<class visitor_type>
    void _visitCompilerVisibleInternals(const visitor_type& visitor) {
        if (isUnbound()) {
            return;
        }

        if (isConstant()) {
            mConstant->_visitCompilerVisibleInternals(visitor);
            return;
        }

        if (isGlobalInDict() || isGlobalInCell() || isNamedModuleMember()) {
            PyObject* obj = extractGlobalRefFromDictOrCell();

            if (obj) {
                _visitCompilerVisibleInternalsInPyobj(obj, visitor);
            }
            return;
        }
    }

    template<class visitor_type>
    void _visitCompilerVisibleInternalsInPyobj(PyObject* obj, const visitor_type& visitor) {
        if (!PyType_Check(obj)) {
            return;
        }

        Type* t = PyInstance::extractTypeFrom(obj);

        if (t && t->isForwardDefined()) {
            if (t->isResolved()) {
                visitor.visitTopo(
                    t->forwardResolvesTo()
                );
            } else {
            }
        } else if (t) {
            visitor.visitTopo(t);
        } else {
            visitor.visitTopo(obj);
        }
    }

    FunctionGlobal withUpdatedInternalTypePointers(const std::map<Type*, Type*>& groupMap) {
        Type* t = getValueAsType();

        auto it = groupMap.find(t);

        if (it != groupMap.end()) {
            // we're actually a constant!
            return FunctionGlobal::Constant(
                CompilerVisiblePyObj::Type(it->second)
            );
        }

        return *this;
    }


    bool isUnresolved(bool insistForwardsResolved) {
        if (isConstant()) {
            return false;
        }
        if (isUnbound()) {
            return false;
        }

        if (isGlobalInDict() || isGlobalInCell() || isNamedModuleMember()) {
            PyObject* obj = extractGlobalRefFromDictOrCell();

            if (!obj) {
                return true;
            }

            if (insistForwardsResolved) {
                Type* t = PyInstance::extractTypeFrom(obj);
                if (t && t->isForwardDefined()) {
                    return true;
                }
            }

            return false;
        }

        return true;
    }

    void autoresolveGlobal(
        const std::set<Type*>& resolvedForwards
    ) {
        if (isGlobalInCell()) {
            Type* cellContents = PyInstance::extractTypeFrom(
                (PyTypeObject*)PyCell_Get(mModuleDictOrCell)
            );

            if (cellContents && resolvedForwards.find(cellContents) != resolvedForwards.end()) {
                PyCell_Set(
                    mModuleDictOrCell,
                    (PyObject*)PyInstance::typeObj(
                        cellContents->forwardResolvesTo()
                    )
                );
            }

            return;
        }

        if (isGlobalInDict() || isNamedModuleMember()) {
            PyObject* dictVal = PyDict_GetItemString(mModuleDictOrCell, mName.c_str());

            if (dictVal) {
                Type* cellContents = PyInstance::extractTypeFrom(
                    (PyTypeObject*)dictVal
                );

                if (resolvedForwards.find(cellContents) != resolvedForwards.end()) {
                    PyDict_SetItemString(
                        mModuleDictOrCell,
                        mName.c_str(),
                        (PyObject*)PyInstance::typeObj(
                            cellContents->forwardResolvesTo()
                        )
                    );
                }
            }
        }
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
    std::string mModuleName;
    PyObject* mModuleDictOrCell;
};
