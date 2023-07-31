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


#include "PyObjSnapshot.hpp"


class FunctionGlobal {
    enum class GlobalType {
        Unbound = 1, // this global is not bound to anything - its a Name error
        NamedModuleMember = 2, // this global is a member of a particular module
        Constant = 3, // this global is bound to a constant that we resolved at some point
        GlobalInDict = 4, // this global is an unresolved variable residing in some python dict
        GlobalInCell = 5, // this global is an unresolved variable residing in some PyCell
    };

    FunctionGlobal(
        GlobalType inKind,
        PyObject* dictOrCell,
        std::string name,
        std::string moduleName,
        Type* constant
    ) :
        mKind(inKind),
        mModuleDictOrCell(incref(dictOrCell)),
        mName(name),
        mModuleName(moduleName),
        mConstant(constant)
    {
    }

public:
    FunctionGlobal() :
        mKind(GlobalType::Unbound),
        mModuleDictOrCell(nullptr)
    {
    }

    FunctionGlobal withConstantsInternalized(const std::map<Type*, Type*>& typeMap) {
        if (!(isGlobalInDict() || isGlobalInCell())) {
            return *this;
        }

        Type* t = getValueAsType();

        if (t) {
            auto it = typeMap.find(t);
            if (it != typeMap.end()) {
                return FunctionGlobal::Constant(it->second);
            } else {
                return FunctionGlobal::Constant(t);
            }
        }

        return *this;
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
            return "FunctionGlobal.Constant(type=" + mConstant->name() + ")";
        }

        throw std::runtime_error("Unknown FunctionGlobal Kind");
    }

    static FunctionGlobal Unbound() {
        return FunctionGlobal();
    }

    static FunctionGlobal Constant(Type* constant) {
        if (!constant) {
            throw std::runtime_error("FunctionGlobal::Constant can't be null");
        }

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
        if (isGlobalInDict() || isGlobalInCell() || isNamedModuleMember()) {
            PyObject* ref = extractGlobalRefFromDictOrCell();
            if (!ref) {
                return nullptr;
            }
            return PyInstance::extractTypeFrom(ref);
        }

        if (isUnbound()) {
            return nullptr;
        }

        if (isConstant()) {
            return mConstant;
        }

        throw std::runtime_error("Unknown global kind.");
    }

    PyObject* getValueAsPyobj() {
        if (isGlobalInDict() || isGlobalInCell() || isNamedModuleMember()) {
            return extractGlobalRefFromDictOrCell();
        }

        if (isUnbound()) {
            return nullptr;
        }

        if (isConstant()) {
            return (PyObject*)PyInstance::typeObj(mConstant);
        }

        throw std::runtime_error("Unknown global kind.");
    }

    GlobalType getKind() const {
        return mKind;
    }

    Type* getConstant() const {
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
            visitor(mConstant);
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
            visitor.visitTopo(mConstant);
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
            }
        } else if (t) {
            visitor.visitTopo(t);
        } else {
            visitor.visitTopo(obj);
        }
    }

    FunctionGlobal withUpdatedInternalTypePointers(const std::map<Type*, Type*>& groupMap) {
        Type* t = getValueAsType();

        if (!t) {
            return *this;
        }

        auto it = groupMap.find(t);

        if (it != groupMap.end()) {
            // we're actually a constant!
            return FunctionGlobal::Constant(
                it->second
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
        buffer.writeBeginCompound(fieldNumber);
        buffer.writeRegisterType(0, (uint64_t)mKind);

        if (mKind == GlobalType::NamedModuleMember) {
            context.serializePythonObject(mModuleDictOrCell, buffer, 1);
            buffer.writeStringObject(2, mName);
            buffer.writeStringObject(3, mModuleName);
        } else
        if (mKind == GlobalType::Constant) {
            context.serializeNativeType(mConstant, buffer, 4);
        } else
        if (mKind == GlobalType::GlobalInDict) {
            context.serializePythonObject(mModuleDictOrCell, buffer, 1);
            buffer.writeStringObject(2, mName);
        } else
        if (mKind == GlobalType::GlobalInCell) {
            context.serializePythonObject(mModuleDictOrCell, buffer, 1);
        }

        buffer.writeEndCompound();
    }

    template<class serialization_context_t, class buf_t>
    static FunctionGlobal deserialize(serialization_context_t& context, buf_t& buffer, int wireType) {
        uint64_t kind = 0;
        std::string name, moduleName;
        PyObjectHolder cellOrModule;
        Type* constant;

        buffer.consumeCompoundMessage(wireType, [&](size_t fieldNumber, size_t wireType) {
            if (fieldNumber == 0) {
                buffer.readRegisterType(&kind, wireType);
            } else
            if (fieldNumber == 1) {
                cellOrModule.steal(context.deserializePythonObject(buffer, wireType));
            } else
            if (fieldNumber == 2) {
                name = buffer.readStringObject();
            } else
            if (fieldNumber == 3) {
                moduleName = buffer.readStringObject();
            } else
            if (fieldNumber == 4) {
                constant = context.deserializeNativeType(buffer, wireType);
            }
        });

        if (kind == (int)GlobalType::Unbound) {
            return FunctionGlobal::Unbound();
        }
        if (kind == (int)GlobalType::NamedModuleMember) {
            if (!cellOrModule || !name.size() || !moduleName.size()) {
                throw std::runtime_error("Corrupt FunctionGlobal - invalid NamedModuleMember");
            }
            return FunctionGlobal::NamedModuleMember(cellOrModule, name, moduleName);
        }
        if (kind == (int)GlobalType::Constant) {
            if (!constant) {
                throw std::runtime_error("Corrupt FunctionGlobal - invalid constant");
            }
            return FunctionGlobal::Constant(constant);
        }
        if (kind == (int)GlobalType::GlobalInDict) {
            if (!cellOrModule || !name.size()) {
                throw std::runtime_error("Corrupt FunctionGlobal - invalid GlobalInDict");
            }
            return FunctionGlobal::GlobalInDict(cellOrModule, name);
        }
        if (kind == (int)GlobalType::GlobalInCell) {
            if (!cellOrModule) {
                throw std::runtime_error("Corrupt FunctionGlobal - invalid cell");
            }
            return FunctionGlobal::GlobalInCell((PyObject*)cellOrModule);
        }

        throw std::runtime_error("Corrupt FunctionGlobal - invalid 'kind'");
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

    Type* mConstant;

    std::string mName;
    std::string mModuleName;
    PyObject* mModuleDictOrCell;
};
