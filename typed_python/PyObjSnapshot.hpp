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
#include "PythonTypeInternals.hpp"

class PyObjGraphSnapshot;
class PyObjSnapshot;
class FunctionGlobal;
class FunctionOverload;

/*********************************
PyObjSnapshot

A representation of a python object that's owned by TypedPython. They are intended to be a
'snapshot' of the state of a collection of python objects and contain enough information
for the compiler to use them to build compiled code, and for us to rebuild the object
entirely from the snapshot. This gives us a scaffolding to describe objects and object
graphs independent of the state of the python interpreter.

To the extent that they visit 'mutable' objects like a list (which might be contained within
a default argument), they will encode the state of the object when it was first seen. This
lets us determine if the object was modified (which would break compiler invariants) and also
gives us a self-consistent view of the world to compile against so we don't have state
changing underneath us.
**********************************/


class PyObjSnapshotMaker {
public:
    PyObjSnapshotMaker(
        std::unordered_map<PyObject*, PyObjSnapshot*>& inObjMapCache,
        std::unordered_map<Type*, PyObjSnapshot*>& inTypeMapCache,
        std::unordered_map<InstanceRef, PyObjSnapshot*>& inInstanceCache,
        const std::map<::Type*, ::Type*>& inGroupMap,
        PyObjGraphSnapshot* inGraph,
        bool inLinkBackToOriginalObject
    ) :
        mObjMapCache(inObjMapCache),
        mTypeMapCache(inTypeMapCache),
        mInstanceCache(inInstanceCache),
        mGroupMap(inGroupMap),
        mGraph(inGraph),
        mLinkBackToOriginalObject(inLinkBackToOriginalObject)
    {
    }

    PyObjSnapshot* internalize(const std::string& def);
    PyObjSnapshot* internalize(const MemberDefinition& def);
    PyObjSnapshot* internalize(const FunctionGlobal& def);
    PyObjSnapshot* internalize(const FunctionOverload& def);
    PyObjSnapshot* internalize(const FunctionArg& def);
    PyObjSnapshot* internalize(const ClosureVariableBinding& def);
    PyObjSnapshot* internalize(const ClosureVariableBindingStep& def);
    PyObjSnapshot* internalize(const std::vector<FunctionOverload>& def);
    PyObjSnapshot* internalize(const std::vector<FunctionArg>& inArgs);
    PyObjSnapshot* internalize(const std::vector<std::string>& inArgs);
    PyObjSnapshot* internalize(const std::map<std::string, ClosureVariableBinding>& inBindings);
    PyObjSnapshot* internalize(const std::map<std::string, FunctionGlobal>& inGlobals);
    PyObjSnapshot* internalize(const std::map<std::string, Function*>& inMethods);
    PyObjSnapshot* internalize(const std::map<std::string, PyObject*>& inMethods);
    PyObjSnapshot* internalize(const std::vector<MemberDefinition>& inMethods);
    PyObjSnapshot* internalize(PyObject* val);
    PyObjSnapshot* internalize(Type* val);
    PyObjSnapshot* internalize(const Instance& val) {
        return internalize(val.ref());
    }
    PyObjSnapshot* internalize(InstanceRef val);

    bool linkBackToOriginalObject() const {
        return mLinkBackToOriginalObject;
    }

    PyObjGraphSnapshot* graph() const {
        return mGraph;
    }

    const std::map<::Type*, ::Type*>& getGroupMap() const {
        return mGroupMap;
    }

private:
    std::unordered_map<PyObject*, PyObjSnapshot*>& mObjMapCache;
    std::unordered_map<Type*, PyObjSnapshot*>& mTypeMapCache;
    std::unordered_map<InstanceRef, PyObjSnapshot*>& mInstanceCache;
    const std::map<::Type*, ::Type*>& mGroupMap;
    PyObjGraphSnapshot* mGraph;
    bool mLinkBackToOriginalObject;
};


class PyObjSnapshot {
private:
    friend class PyObjSnapshotMaker;

    enum class Kind {
        // this should never be visible in a running program
        Uninitialized = 0,
        // a string held in mStringObject
        String,
        // we're pointing to a "canonical" python object that's visible from a module
        // and that we don't want to look inside of.  mName will contain the
        // name, and mModuleName will contain the module. We will assume that this object
        // is the same across program invocations (its a C function and we can't
        // look inside of it)
        NamedPyObject,
        // we're pointing back into a typed_python Type held in mType.
        // it must be a 'leaf' type with no internals
        Type,
        // we're pointing into a TP leaf instance (a register type, an int, bytes, etc.)
        Instance,
        // we're a primitive type (like int)
        PrimitiveType,
        // a TP ListOf type. The element type will be in element_type
        ListOfType,
        // a TP TupleOf type. The element type will be in element_type
        TupleOfType,
        // a TP Tuple type. The subtypes will be in mElements
        TupleType,
        // a TP NamedTuple type. The subtypes will be in mElements and the names in mNames
        NamedTupleType,
        // a TP OneOf type. The subtypes will be in mElements
        OneOfType,
        // a TP Value type. The instance will be in value_instance
        ValueType,
        // a TP DictType type. Will have key_type and value_type
        DictType,
        // a TP ConstDictType type. Will have key_type and value_type
        ConstDictType,
        // a TP ConstDictType type. Will have key_type and value_type
        SetType,
        // a TP PointerTo type. The element type will be in element_type
        PointerToType,
        // a TP RefTo type. The element type will be in element_type
        RefToType,
        // a TP Alternative type.
        AlternativeType,
        // a TP ConcreteAlternative type.
        ConcreteAlternativeType,
        // a TP AlternativeMatcher type.
        AlternativeMatcherType,
        // a TP PythonObjectOfType type.
        PythonObjectOfTypeType,
        // a TP SubclassOfType type.
        SubclassOfTypeType,
        // a TP Class type.
        ClassType,
        // a TP HeldClass type.
        HeldClassType,
        // a TP FunctionType type.
        FunctionType,
        // a TP FunctionOverload.
        FunctionOverload,
        // a TP FunctionGlobal.
        FunctionGlobal,
        // a TP FunctionArg.
        FunctionArg,
        // a TP ClosureVariableBinding.
        FunctionClosureVariableBinding,
        // a TP ClosureVariableBinding.Step
        FunctionClosureVariableBindingStep,
        // a TP MemberDefinition in a Class.
        ClassMemberDefinition,
        // a TP BoundMethod type
        BoundMethodType,
        // a TP Forward type
        ForwardType,
        // a TP TypedCellType
        TypedCellType,
        // a python list, with elements in mElements
        PyList,
        // a python Dict, with values in mElements and keys in mKeys
        PyDict,
        // a python Dict, with values in mElements and keys in mKeys
        PySet,
        // a python tuple with elements in mElements
        PyTuple,
        // a vanilla python 'class' which must be constructible by calling 'type'
        PyClass,
        // a vanilla python object whose type is a PyClass
        PyObject,
        // a python function.  mNamedElements will contain code, annotations, ec.
        // mStringObject will contain the name.
        PyFunction,
        // a code object. mName
        PyCodeObject,
        PyCell,
        // a module object that can be looked up by name. We don't walk into this
        // from here to avoid making a snapshot of module objects while they're being imported
        PyModule,
        // a module object's dict.
        PyModuleDict,
        // the dict of a vanilla PyClass
        PyClassDict,
        PyStaticMethod,
        PyClassMethod,
        PyBoundMethod,
        PyProperty,
        // the bailout pathway for cases we don't handle well. We should
        // assume that the compiler will treat this object as a plain
        // PyObject without looking inside of it, and the details of this
        // object are insufficient to differentiate two different Function
        // types that both refer to different ArbitraryPyObject instances.
        ArbitraryPyObject,
        // a bundle of types that doesn't represent a specific object or type
        // but holds collections of types.
        InternalBundle
    };

    PyObjSnapshot(PyObjGraphSnapshot* inGraph=nullptr) :
        mKind(Kind::Uninitialized),
        mType(nullptr),
        mPyObject(nullptr),
        mGraph(inGraph)
    {
    }

public:
    ~PyObjSnapshot() {
        decref(mPyObject);
    }

    static PyObjSnapshot* ForType(Type* t) {
        PyObjSnapshot* res = new PyObjSnapshot();
        res->mKind = Kind::Type;
        res->mType = t;
        return res;
    }

    PyObjGraphSnapshot* getGraph() const {
        return mGraph;
    }

    bool isUninitialized() const {
        return mKind == Kind::Uninitialized;
    }

    bool isType() const {
        return mKind == Kind::Type;
    }

    bool isNamedPyObject() const {
        return mKind == Kind::NamedPyObject;
    }

    bool isString() const {
        return mKind == Kind::String;
    }

    bool isInstance() const {
        return mKind == Kind::Instance;
    }

    bool isPyList() const {
        return mKind == Kind::PyList;
    }

    bool isPyDict() const {
        return mKind == Kind::PyDict;
    }

    bool isPySet() const {
        return mKind == Kind::PySet;
    }

    bool isPyTuple() const {
        return mKind == Kind::PyTuple;
    }

    bool isPyClass() const {
        return mKind == Kind::PyClass;
    }

    bool isPyFunction() const {
        return mKind == Kind::PyFunction;
    }

    bool isPyCell() const {
        return mKind == Kind::PyCell;
    }

    bool isPyObject() const {
        return mKind == Kind::PyObject;
    }

    bool isPyCodeObject() const {
        return mKind == Kind::PyCodeObject;
    }

    bool isPyModule() const {
        return mKind == Kind::PyModule;
    }

    bool isArbitraryPyObject() const {
        return mKind == Kind::ArbitraryPyObject;
    }

    std::string kindAsString() const {
        if (mKind == Kind::Uninitialized) {
            return "Uninitialized";
        }

        if (mKind == Kind::InternalBundle) {
            return "InternalBundle";
        }

        if (mKind == Kind::String) {
            return "String";
        }

        if (mKind == Kind::NamedPyObject) {
            return "NamedPyObject";
        }

        if (mKind == Kind::Type) {
            return "Type";
        }

        if (mKind == Kind::Instance) {
            return "Instance";
        }

        if (mKind == Kind::PrimitiveType) {
            return "PrimitiveType";
        }

        if (mKind == Kind::ListOfType) {
            return "ListOfType";
        }

        if (mKind == Kind::TupleOfType) {
            return "TupleOfType";
        }

        if (mKind == Kind::TupleType) {
            return "TupleType";
        }

        if (mKind == Kind::NamedTupleType) {
            return "NamedTupleType";
        }

        if (mKind == Kind::OneOfType) {
            return "OneOfType";
        }

        if (mKind == Kind::ValueType) {
            return "ValueType";
        }

        if (mKind == Kind::DictType) {
            return "DictType";
        }

        if (mKind == Kind::ConstDictType) {
            return "ConstDictType";
        }

        if (mKind == Kind::SetType) {
            return "SetType";
        }

        if (mKind == Kind::PointerToType) {
            return "PointerToType";
        }

        if (mKind == Kind::RefToType) {
            return "RefToType";
        }

        if (mKind == Kind::AlternativeType) {
            return "AlternativeType";
        }

        if (mKind == Kind::ConcreteAlternativeType) {
            return "ConcreteAlternativeType";
        }

        if (mKind == Kind::AlternativeMatcherType) {
            return "AlternativeMatcherType";
        }

        if (mKind == Kind::PythonObjectOfTypeType) {
            return "PythonObjectOfTypeType";
        }

        if (mKind == Kind::SubclassOfTypeType) {
            return "SubclassOfTypeType";
        }

        if (mKind == Kind::ClassType) {
            return "ClassType";
        }

        if (mKind == Kind::HeldClassType) {
            return "HeldClassType";
        }

        if (mKind == Kind::FunctionType) {
            return "FunctionType";
        }

        if (mKind == Kind::FunctionOverload) {
            return "FunctionOverload";
        }

        if (mKind == Kind::FunctionGlobal) {
            return "FunctionGlobal";
        }

        if (mKind == Kind::BoundMethodType) {
            return "BoundMethodType";
        }

        if (mKind == Kind::ForwardType) {
            return "ForwardType";
        }

        if (mKind == Kind::TypedCellType) {
            return "TypedCellType";
        }

        if (mKind == Kind::PyList) {
            return "PyList";
        }

        if (mKind == Kind::PyDict) {
            return "PyDict";
        }

        if (mKind == Kind::PySet) {
            return "PySet";
        }

        if (mKind == Kind::PyTuple) {
            return "PyTuple";
        }

        if (mKind == Kind::PyClass) {
            return "PyClass";
        }

        if (mKind == Kind::PyFunction) {
            return "PyFunction";
        }

        if (mKind == Kind::PyObject) {
            return "PyObject";
        }

        if (mKind == Kind::PyCell) {
            return "PyCell";
        }

        if (mKind == Kind::PyCodeObject) {
            return "PyCodeObject";
        }

        if (mKind == Kind::PyModule) {
            return "PyModule";
        }

        if (mKind == Kind::PyModuleDict) {
            return "PyModuleDict";
        }

        if (mKind == Kind::PyClassDict) {
            return "PyClassDict";
        }

        if (mKind == Kind::PyStaticMethod) {
            return "PyStaticMethod";
        }

        if (mKind == Kind::PyClassMethod) {
            return "PyClassMethod";
        }

        if (mKind == Kind::PyBoundMethod) {
            return "PyBoundMethod";
        }

        if (mKind == Kind::PyProperty) {
            return "PyProperty";
        }

        if (mKind == Kind::ArbitraryPyObject) {
            return "ArbitraryPyObject";
        }

        throw std::runtime_error("Unknown PyObjSnapshot Kind: " + format((int)mKind));
    }

    static std::string dictGetStringOrEmpty(PyObject* dict, const char* name) {
        if (!dict || !PyDict_Check(dict)) {
            return "";
        }

        PyObject* o = PyDict_GetItemString(dict, name);
        if (!o || !PyUnicode_Check(o)) {
            return "";
        }

        return PyUnicode_AsUTF8(o);
    }

    static std::string stringOrEmpty(PyObject* o) {
        if (!o || !PyUnicode_Check(o)) {
            return "";
        }

        return PyUnicode_AsUTF8(o);
    }

    static PyTypeObject* createAVanillaType() {
        PyObjectStealer emptyBases(PyTuple_New(0));
        PyObjectStealer emptyDict(PyDict_New());

        return (PyTypeObject*)PyObject_CallFunction(
            (PyObject*)&PyType_Type,
            "sOO",
            "AVanillaType",
            (PyObject*)emptyBases,
            (PyObject*)emptyDict
        );
    }

    bool isVanillaClassType(PyTypeObject* o) {
        static PyTypeObject* aVanillaType = createAVanillaType();
        return o->tp_new == aVanillaType->tp_new;
    }

    static bool isPyObjectGloballyIdentifiable(PyObject* h) {
        PyObject* sysModuleModules = staticPythonInstance("sys", "modules");

        if (PyObject_HasAttrString(h, "__module__") && PyObject_HasAttrString(h, "__name__")) {
            PyObjectStealer moduleName(PyObject_GetAttrString(h, "__module__"));
            if (!moduleName) {
                PyErr_Clear();
                return false;
            }

            PyObjectStealer clsName(PyObject_GetAttrString(h, "__name__"));
            if (!clsName) {
                PyErr_Clear();
                return false;
            }

            if (!PyUnicode_Check(moduleName) || !PyUnicode_Check(clsName)) {
                return false;
            }

            PyObjectStealer moduleObject(PyObject_GetItem(sysModuleModules, moduleName));

            if (!moduleObject) {
                PyErr_Clear();
                return false;
            }

            PyObjectStealer obj(PyObject_GetAttr(moduleObject, clsName));

            if (!obj) {
                PyErr_Clear();
                return false;
            }
            if ((PyObject*)obj == h) {
                return true;
            }
        }

        return false;
    }

    void becomeInternalizedOf(
        const std::string& val,
        PyObjSnapshotMaker& maker
    );

    void becomeInternalizedOf(
        const FunctionArg& val,
        PyObjSnapshotMaker& maker
    );

    void becomeInternalizedOf(
        const ClosureVariableBinding& val,
        PyObjSnapshotMaker& maker
    );

    void becomeInternalizedOf(
        const ClosureVariableBindingStep& val,
        PyObjSnapshotMaker& maker
    );

    void becomeInternalizedOf(
        const MemberDefinition& val,
        PyObjSnapshotMaker& maker
    );

    void becomeInternalizedOf(
        const FunctionGlobal& val,
        PyObjSnapshotMaker& maker
    );

    void becomeInternalizedOf(
        const FunctionOverload& val,
        PyObjSnapshotMaker& maker
    );

    void becomeInternalizedOf(
        InstanceRef val,
        PyObjSnapshotMaker& maker
    );

    void becomeInternalizedOf(
        Type* t,
        PyObjSnapshotMaker& maker
    );

    void becomeInternalizedOf(
        PyObject* val,
        PyObjSnapshotMaker& maker
    );

    void append(PyObjSnapshot* elt) {
        if (mKind != Kind::PyTuple) {
            throw std::runtime_error("Expected a PyTuple");
        }

        mElements.push_back(elt);
    }

    const std::vector<PyObjSnapshot*>& elements() const {
        return mElements;
    }

    const std::vector<PyObjSnapshot*>& keys() const {
        return mKeys;
    }

    const std::map<std::string, PyObjSnapshot*>& namedElements() const {
        return mNamedElements;
    }

    const std::map<std::string, int64_t>& namedInts() const {
        return mNamedInts;
    }

    ::Type* getType() const {
        return mType;
    }

    const ::Instance& getInstance() const {
        return mInstance;
    }

    const std::string& getStringValue() const {
        return mStringValue;
    }

    const std::string& getName() const {
        return mName;
    }

    const std::string& getModuleName() const {
        return mModuleName;
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
        if (mKind == Kind::Type) {
            visitor.visitTopo(mType);
        }

        if (mKind == Kind::Instance) {
            // TODO: what to do here?
            visitor.visitInstance(mInstance.type(), mInstance.data());
        }

        if (mKind == Kind::PyTuple) {
            // TODO: what to do here?
            throw std::runtime_error("TODO: PyObjSnapshot::_visitCompilerVisibleInternals PyTuple");
        }

        if (mKind == Kind::ArbitraryPyObject) {
            visitor.visitTopo(mPyObject);
        }
    }

    // get the python object representation of this object, which isn't guaranteed
    // to exist and may need to be constructed on demand.  this will do a pass over
    // all reachable objects, building skeletons, and then performs a second pass where
    // we fill items out once we have the skeletons in place.
    PyObject* getPyObj() {
        if (mPyObject) {
            return mPyObject;
        }

        std::unordered_set<PyObjSnapshot*> needsResolution;

        PyObject* res = getPyObj(needsResolution);

        for (auto n: needsResolution) {
            n->finalizeGetPyObj();
        }

        // do a second pass patching up classes. They don't inherit things
        // from their dict correctly
        for (auto n: needsResolution) {
            n->finalizeGetPyObj2();
        }

        return res;
    }

    void finalizeGetPyObj2() {
        if (mKind == Kind::PyClass) {
            if (mNamedElements.find("cls_dict") == mNamedElements.end()) {
                throw std::runtime_error("Corrupt PyClass - no cls_dict");
            }

            PyObjSnapshot* clsDictPyObj = mNamedElements["cls_dict"];

            for (long k = 0; k < clsDictPyObj->elements().size() && k < clsDictPyObj->keys().size(); k++) {
                if (clsDictPyObj->keys()[k]->isString()) {
                    PyObject_SetAttrString(
                        mPyObject,
                        clsDictPyObj->keys()[k]->getStringValue().c_str(),
                        clsDictPyObj->elements()[k]->getPyObj()
                    );
                }
            }
        }
    }

    // we're a skeleton - finish ourselves out
    void finalizeGetPyObj() {
        if (mKind == Kind::PyDict || mKind == Kind::PyClassDict) {
            for (long k = 0; k < mElements.size() && k < mKeys.size(); k++) {
                PyDict_SetItem(
                    mPyObject,
                    mKeys[k]->getPyObj(),
                    mElements[k]->getPyObj()
                );
            }
        } else if (mKind == Kind::PyCell) {
            auto it = mNamedElements.find("cell_contents");
            if (it != mNamedElements.end()) {
                PyCell_Set(
                    mPyObject,
                    it->second->getPyObj()
                );
            }
        }
    }

    PyObject* getPyObj(std::unordered_set<PyObjSnapshot*>& needsResolution) {
        PyObject* sysModuleModules = staticPythonInstance("sys", "modules");

        if (!mPyObject) {
            if (mKind == Kind::ArbitraryPyObject) {
                throw std::runtime_error("Corrupt PyObjSnapshot.ArbitraryPyObject: missing mPyObject");
            } else if (mKind == Kind::Type) {
                mPyObject = (PyObject*)PyInstance::typeObj(mType);
            } else if (mKind == Kind::String) {
                mPyObject = PyUnicode_FromString(mStringValue.c_str());
            } else if (mKind == Kind::Instance) {
                mPyObject = PyInstance::extractPythonObject(mInstance);
            } else if (mKind == Kind::NamedPyObject) {
                PyObjectStealer nameAsStr(PyUnicode_FromString(mModuleName.c_str()));
                PyObjectStealer moduleObject(
                    PyObject_GetItem(sysModuleModules, nameAsStr)
                );

                if (!moduleObject) {
                    PyErr_Clear();
                    // TODO: should we be importing here? that seems dangerous...
                    throw std::runtime_error("Somehow module " + mModuleName + " is not loaded!");
                }
                PyObjectStealer inst(PyObject_GetAttrString(moduleObject, mName.c_str()));
                if (!inst) {
                    PyErr_Clear();

                    // TODO: should we be importing here? that seems dangerous...
                    throw std::runtime_error("Somehow module " + mModuleName + " is missing member " + mName);
                }

                mPyObject = incref(inst);
            } else if (mKind == Kind::PyDict || mKind == Kind::PyClassDict) {
                mPyObject = PyDict_New();
                needsResolution.insert(this);
            } else if (mKind == Kind::PyList) {
                mPyObject = PyList_New(0);

                for (long k = 0; k < mElements.size(); k++) {
                    PyList_Append(mPyObject, mElements[k]->getPyObj(needsResolution));
                }
            } else if (mKind == Kind::PyTuple) {
                mPyObject = PyTuple_New(mElements.size());

                // first initialize it in case we throw somehow
                for (long k = 0; k < mElements.size(); k++) {
                    PyTuple_SetItem(mPyObject, k, incref(Py_None));
                }

                for (long k = 0; k < mElements.size(); k++) {
                    PyTuple_SetItem(mPyObject, k, incref(mElements[k]->getPyObj(needsResolution)));
                }
            } else if (mKind == Kind::PySet) {
                mPyObject = PySet_New(nullptr);

                for (long k = 0; k < mElements.size(); k++) {
                    PySet_Add(mPyObject, incref(mElements[k]->getPyObj(needsResolution)));
                }
            } else if (mKind == Kind::PyClass) {
                needsResolution.insert(this);

                PyObjectStealer argTup(PyTuple_New(3));
                PyTuple_SetItem(argTup, 0, PyUnicode_FromString(mName.c_str()));
                PyTuple_SetItem(argTup, 1, incref(getNamedElementPyobj("cls_bases", needsResolution)));
                PyTuple_SetItem(argTup, 2, incref(getNamedElementPyobj("cls_dict", needsResolution)));

                mPyObject = PyType_Type.tp_new(&PyType_Type, argTup, nullptr);

                if (!mPyObject) {
                    throw PythonExceptionSet();
                }
            } else if (mKind == Kind::PyModule) {
                PyObjectStealer nameAsStr(PyUnicode_FromString(mName.c_str()));
                PyObjectStealer moduleObject(
                    PyObject_GetItem(sysModuleModules, nameAsStr)
                );

                if (!moduleObject) {
                    PyErr_Clear();
                    // TODO: should we be importing here? that seems dangerous...
                    throw std::runtime_error("Somehow module " + mName + " is not loaded!");
                }

                mPyObject = incref(moduleObject);
            } else if (mKind == Kind::PyModuleDict) {
                mPyObject = PyObject_GenericGetDict(getNamedElementPyobj("module_dict_of", needsResolution), nullptr);
                if (!mPyObject) {
                    throw PythonExceptionSet();
                }
            } else if (mKind == Kind::PyCell) {
                mPyObject = PyCell_New(nullptr);
                needsResolution.insert(this);
            } else if (mKind == Kind::PyObject) {
                PyObject* t = getNamedElementPyobj("inst_type", needsResolution);
                PyObject* d = getNamedElementPyobj("inst_dict", needsResolution);

                static PyObject* emptyTuple = PyTuple_Pack(0);

                mPyObject = ((PyTypeObject*)t)->tp_new(
                    ((PyTypeObject*)t),
                    emptyTuple,
                    nullptr
                );

                if (PyObject_GenericSetDict(mPyObject, d, nullptr)) {
                    throw PythonExceptionSet();
                }
            } else if (mKind == Kind::PyFunction) {
                mPyObject = PyFunction_New(
                    getNamedElementPyobj("func_code", needsResolution),
                    getNamedElementPyobj("func_globals", needsResolution)
                );

                if (mNamedElements.find("func_closure") != mNamedElements.end()) {
                    PyFunction_SetClosure(
                        mPyObject,
                        getNamedElementPyobj("func_closure", needsResolution)
                    );
                }

                if (mNamedElements.find("func_annotations") != mNamedElements.end()) {
                    PyFunction_SetAnnotations(
                        mPyObject,
                        getNamedElementPyobj("func_annotations", needsResolution)
                    );
                }

                if (mNamedElements.find("func_defaults") != mNamedElements.end()) {
                    PyFunction_SetAnnotations(
                        mPyObject,
                        getNamedElementPyobj("func_defaults", needsResolution)
                    );
                }

                if (mNamedElements.find("func_qualname") != mNamedElements.end()) {
                    PyObject_SetAttrString(
                        mPyObject,
                        "__qualname__",
                        getNamedElementPyobj("func_qualname", needsResolution)
                    );
                }

                if (mNamedElements.find("func_kwdefaults") != mNamedElements.end()) {
                    PyObject_SetAttrString(
                        mPyObject,
                        "__kwdefaults__",
                        getNamedElementPyobj("func_kwdefaults", needsResolution)
                    );
                }

                if (mNamedElements.find("func_name") != mNamedElements.end()) {
                    PyObject_SetAttrString(
                        mPyObject,
                        "__name__",
                        getNamedElementPyobj("func_name", needsResolution)
                    );
                }
            } else if (mKind == Kind::PyCodeObject) {
#if PY_MINOR_VERSION < 8
                mPyObject = (PyObject*)PyCode_New(
#else
                mPyObject = (PyObject*)PyCode_NewWithPosOnlyArgs(
#endif
                    mNamedInts["co_argcount"],
#if PY_MINOR_VERSION >= 8
                    mNamedInts["co_posonlyargcount"],
#endif
                    mNamedInts["co_kwonlyargcount"],
                    mNamedInts["co_nlocals"],
                    mNamedInts["co_stacksize"],
                    mNamedInts["co_flags"],
                    getNamedElementPyobj("co_code", needsResolution),
                    getNamedElementPyobj("co_consts", needsResolution),
                    getNamedElementPyobj("co_names", needsResolution),
                    getNamedElementPyobj("co_varnames", needsResolution),
                    getNamedElementPyobj("co_freevars", needsResolution),
                    getNamedElementPyobj("co_cellvars", needsResolution),
                    getNamedElementPyobj("co_filename", needsResolution),
                    getNamedElementPyobj("co_name", needsResolution),
                    mNamedInts["co_firstlineno"],
                    getNamedElementPyobj(
#if PY_MINOR_VERSION >= 10
                        "co_linetable",
#else
                        "co_lnotab",
#endif
                    needsResolution
                    )
                );
            } else if (mKind == Kind::PyStaticMethod) {
                mPyObject = PyStaticMethod_New(Py_None);

                JustLikeAClassOrStaticmethod* method = (JustLikeAClassOrStaticmethod*)mPyObject;
                decref(method->cm_callable);
                method->cm_callable = incref(getNamedElementPyobj("meth_func", needsResolution));
            } else if (mKind == Kind::PyClassMethod) {
                mPyObject = PyClassMethod_New(Py_None);

                JustLikeAClassOrStaticmethod* method = (JustLikeAClassOrStaticmethod*)mPyObject;
                decref(method->cm_callable);
                method->cm_callable = incref(getNamedElementPyobj("meth_func", needsResolution));
            } else if (mKind == Kind::PyProperty) {
                static PyObject* nones = PyTuple_Pack(3, Py_None, Py_None, Py_None);

                mPyObject = PyObject_CallObject((PyObject*)&PyProperty_Type, nones);

                JustLikeAPropertyObject* dest = (JustLikeAPropertyObject*)mPyObject;

                decref(dest->prop_get);
                decref(dest->prop_set);
                decref(dest->prop_del);
                decref(dest->prop_doc);

                dest->prop_get = nullptr;
                dest->prop_set = nullptr;
                dest->prop_del = nullptr;
                dest->prop_doc = nullptr;

                #if PY_MINOR_VERSION >= 10
                decref(dest->prop_name);
                dest->prop_name = nullptr;
                #endif

                dest->prop_get = incref(getNamedElementPyobj("prop_get", needsResolution, true));
                dest->prop_set = incref(getNamedElementPyobj("prop_set", needsResolution, true));
                dest->prop_del = incref(getNamedElementPyobj("prop_del", needsResolution, true));
                dest->prop_doc = incref(getNamedElementPyobj("prop_doc", needsResolution, true));

                #if PY_MINOR_VERSION >= 10
                decref(dest->prop_name);
                dest->prop_name = incref(getNamedElementPyobj("prop_name", needsResolution, true));
                #endif
            } else if (mKind == Kind::PyBoundMethod) {
                mPyObject = PyMethod_New(Py_None, Py_None);
                PyMethodObject* method = (PyMethodObject*)mPyObject;
                decref(method->im_func);
                decref(method->im_self);
                method->im_func = nullptr;
                method->im_self = nullptr;

                method->im_func = incref(getNamedElementPyobj("meth_func", needsResolution));
                method->im_self = incref(getNamedElementPyobj("meth_self", needsResolution));
            } else {
                throw std::runtime_error(
                    "Can't make a python object representation for a CVPO of kind " + kindAsString()
                );
            }
        }

        return mPyObject;
    }

    PyObject* getNamedElementPyobj(
        std::string name,
        std::unordered_set<PyObjSnapshot*>& needsResolution,
        bool allowEmpty=false
    ) {
        auto it = mNamedElements.find(name);
        if (it == mNamedElements.end()) {
            if (allowEmpty) {
                return nullptr;
            }
            throw std::runtime_error(
                "Corrupt PyObjSnapshot." + kindAsString() + ". missing " + name
            );
        }

        return it->second->getPyObj(needsResolution);
    }

    std::string toString() const {
        return "PyObjSnapshot." + kindAsString() + "()";
    }

    template<class serialization_context_t, class buf_t>
    void serialize(serialization_context_t& context, buf_t& buffer, int fieldNumber) const {
        uint32_t id;
        bool isNew;
        std::tie(id, isNew) = buffer.cachePointer((void*)this, nullptr);

        if (!isNew) {
            buffer.writeBeginCompound(fieldNumber);
            buffer.writeUnsignedVarintObject(0, id);
            buffer.writeEndCompound();
            return;
        } else {
            buffer.writeBeginCompound(fieldNumber);
            buffer.writeUnsignedVarintObject(0, id);
            buffer.writeUnsignedVarintObject(1, (int)mKind);

            if (mKind == Kind::Type) {
                context.serializeNativeType(mType, buffer, 2);
            } else
            if (mKind == Kind::Instance) {
                buffer.writeBeginCompound(3);
                context.serializeNativeType(mInstance.type(), buffer, 0);
                mInstance.type()->serialize(mInstance.data(), buffer, 1);
                buffer.writeEndCompound();
            } else
            if (mKind == Kind::PyTuple) {
                buffer.writeBeginCompound(4);
                for (long i = 0; i < mElements.size(); i++) {
                    mElements[i]->serialize(context, buffer, i);
                }
                buffer.writeEndCompound();
            } else
            if (mKind == Kind::ArbitraryPyObject) {
                context.serializePythonObject(mPyObject, buffer, 5);
            }

            buffer.writeEndCompound();
        }
    }

    template<class serialization_context_t, class buf_t>
    static PyObjSnapshot* deserialize(serialization_context_t& context, buf_t& buffer, int wireType) {
        int64_t kind = 0;

        ::Type* type = nullptr;

        std::vector<PyObjSnapshot*> vec;
        ::Instance i;
        uint32_t id = -1;
        PyObjectHolder pyobj;
        PyObjSnapshot* result = nullptr;

        buffer.consumeCompoundMessage(wireType, [&](size_t fieldNumber, size_t wireType) {
            if (fieldNumber == 0) {
                id = buffer.readUnsignedVarint();

                void* ptr = buffer.lookupCachedPointer(id);
                if (ptr) {
                    result = (PyObjSnapshot*)result;
                } else {
                    result = new PyObjSnapshot();
                    buffer.addCachedPointer(id, result, nullptr);
                }
            } else
            if (fieldNumber == 1) {
                kind = buffer.readUnsignedVarint();
            } else
            if (fieldNumber == 2) {
                type = context.deserializeNativeType(buffer, wireType);
            } else
            if (fieldNumber == 3) {
                i = context.deserializeNativeInstance(buffer, wireType);
            } else
            if (fieldNumber == 4) {
                buffer.consumeCompoundMessage(wireType, [&](size_t fieldNumber, size_t wireType) {
                    vec.push_back(PyObjSnapshot::deserialize(context, buffer, wireType));
                });
            } else
            if (fieldNumber == 5) {
                pyobj.steal(context.deserializePythonObject(buffer, wireType));
            }
        });

        if (!result) {
            throw std::runtime_error("corrupt PyObjSnapshot - no memo found");
        }

        if (kind == -1) {
            throw std::runtime_error("corrupt PyObjSnapshot - invalid kind");
        }

        result->mKind = Kind(kind);
        result->mType = type;
        result->mInstance = i;
        result->mPyObject = incref(pyobj);
        result->mElements = vec;

        result->validateAfterDeserialization();

        return result;
    }

    template<class T>
    void becomeBundleOf(const std::map<std::string, T>& namedElements, PyObjSnapshotMaker& maker) {
        mKind = Kind::InternalBundle;

        for (auto& nameAndElt: namedElements) {
            mNames.push_back(nameAndElt.first);
            mElements.push_back(maker.internalize(nameAndElt.second));
        }
    }

    template<class T>
    void becomeBundleOf(const std::vector<T>& elements, PyObjSnapshotMaker& maker) {
        mKind = Kind::InternalBundle;

        for (auto& elt: elements) {
            mElements.push_back(maker.internalize(elt));
        }
    }

private:
    // ensure we won't crash if we interact with this object.
    void validateAfterDeserialization() {
        if (mKind == Kind::Type) {
            if (!mType) {
                throw std::runtime_error("Corrupt CVPO: no Type");
            }
        }

        if (mKind == Kind::ArbitraryPyObject) {
            if (!mPyObject) {
                throw std::runtime_error("Corrupt CVPO: no PyObject");
            }
        }
    }

    Kind mKind;
    PyObjGraphSnapshot* mGraph;

    ::Type* mType;
    ::Instance mInstance;

    // if we are an ArbitraryPythonObject this is always populated
    // otherwise, it will be a cache for a constructed instance of the object
    PyObject* mPyObject;

    std::map<std::string, PyObjSnapshot*> mNamedElements;
    std::map<std::string, int64_t> mNamedInts;

    // if we're a tuple, list, or dict
    std::vector<PyObjSnapshot*> mElements;

    std::vector<std::string> mNames;

    // if we're a PyDict
    std::vector<PyObjSnapshot*> mKeys;

    std::string mStringValue;
    std::string mModuleName;
    std::string mName;
    std::string mQualname;
};
