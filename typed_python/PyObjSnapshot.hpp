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
#include "PyObjSnapshotMaker.hpp"

class PyObjGraphSnapshot;
class PyObjSnapshot;
class FunctionGlobal;
class FunctionOverload;
class PyObjRehydrator;
class PyObjSnapshotMaker;

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

class PyObjSnapshot {
private:
    friend class PyObjSnapshotMaker;
    friend class PyObjRehydrator;

    PyObjSnapshot(PyObjGraphSnapshot* inGraph=nullptr) :
        mKind(Kind::Uninitialized),
        mType(nullptr),
        mPyObject(nullptr),
        mGraph(inGraph)
    {
    }

public:
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
        // a TP Forward type. Contains fwd_target for the target, if set,
        // fwd_cell_or_dict for a forward defined using a lambda, and
        // mName if the forward dict was a global.
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

    ~PyObjSnapshot() {
        decref(mPyObject);
    }

    static PyObjSnapshot* ForType(Type* t) {
        // TODO: this needs to go away. its a shim to handle the current
        // code which is a little confused about these objects.
        PyObjSnapshot* res = new PyObjSnapshot();
        res->mKind = Kind::PrimitiveType;
        res->mType = t;
        return res;
    }

    Kind getKind() const {
        return mKind;
    }

    PyObjGraphSnapshot* getGraph() const {
        return mGraph;
    }

    bool isType() const {
        return mKind == Kind::PrimitiveType;
    }

    bool isString() const {
        return mKind == Kind::String;
    }

    bool isInstance() const {
        return mKind == Kind::Instance;
    }

    std::string kindAsString() const;

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

    bool hasNamedElement(std::string s) const {
        return mNamedElements.find(s) != mNamedElements.end();
    }

    PyObjSnapshot* getNamedElement(std::string s) const {
        auto it = mNamedElements.find(s);

        if (it != mNamedElements.end()) {
            return it->second;
        }

        return nullptr;
    }

    const std::vector<std::string>& names() const {
        return mNames;
    }

    const std::map<std::string, int64_t>& namedInts() const {
        return mNamedInts;
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

    const std::string& getQualname() const {
        return mQualname;
    }

    template<class visitor_type>
    void _visitReferencedTypes(const visitor_type& v) {
        if (mKind == Kind::PrimitiveType) {
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
        if (mKind == Kind::PrimitiveType) {
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

    bool willBeATpType() const {
        return mKind == Kind::PrimitiveType
            || mKind == Kind::ListOfType
            || mKind == Kind::TupleOfType
            || mKind == Kind::TupleType
            || mKind == Kind::NamedTupleType
            || mKind == Kind::OneOfType
            || mKind == Kind::ValueType
            || mKind == Kind::DictType
            || mKind == Kind::ConstDictType
            || mKind == Kind::SetType
            || mKind == Kind::PointerToType
            || mKind == Kind::RefToType
            || mKind == Kind::AlternativeType
            || mKind == Kind::ConcreteAlternativeType
            || mKind == Kind::AlternativeMatcherType
            || mKind == Kind::PythonObjectOfTypeType
            || mKind == Kind::SubclassOfTypeType
            || mKind == Kind::ClassType
            || mKind == Kind::HeldClassType
            || mKind == Kind::FunctionType
            || mKind == Kind::BoundMethodType
            || mKind == Kind::ForwardType
            || mKind == Kind::TypedCellType
        ;
    }


    ::Type* getType() {
        if (mType) {
            return mType;
        }

        if (mKind == Kind::PrimitiveType) {
            return mType;
        }

        if (willBeATpType() && !mType) {
            rehydrate();
            return mType;
        }

        return nullptr;
    }

    const ::Instance& getInstance() {
        return mInstance;
    }

    bool needsHydration() const {
        // currently, instances are 'out of band'. Eventually they'll be just like the
        // other content.
        if (mKind == Kind::Instance) {
            return false;
        }

        if (mKind == Kind::String) {
            return false;
        }

        if (mKind == Kind::Uninitialized || mKind == Kind::InternalBundle) {
            return false;
        }

        if (mPyObject) {
            return false;
        }

        return true;
    }

    // get the python object representation of this object, which isn't guaranteed
    // to exist and may need to be constructed on demand.  this will do a pass over
    // all reachable objects, building skeletons, and then performs a second pass where
    // we fill items out once we have the skeletons in place.
    PyObject* getPyObj() {
        if (mPyObject) {
            return mPyObject;
        }

        if (mType) {
            mPyObject = incref((PyObject*)PyInstance::typeObj(mType));
            return mPyObject;
        }

        if (mKind == Kind::String) {
            mPyObject = PyUnicode_FromString(mStringValue.c_str());
            return mPyObject;
        }

        if (mKind == Kind::Instance) {
            mPyObject = PyInstance::extractPythonObject(mInstance);
            return mPyObject;
        }

        rehydrate();
        return mPyObject;
    }

    void rehydrate();

    std::string toString() const {
        return "PyObjSnapshot." + kindAsString() + "()";
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

    // TODO: fix serialization. This is just not right.
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

            // we only serialize Type, Instance, and PyObj if they are not caches.
            if (mKind == Kind::PrimitiveType) {
                context.serializeNativeType(mType, buffer, 2);
            } else
            if (mKind == Kind::Instance) {
                buffer.writeBeginCompound(3);
                context.serializeNativeType(mInstance.type(), buffer, 0);
                mInstance.type()->serialize(mInstance.data(), buffer, 1);
                buffer.writeEndCompound();
            } else
            if (mKind == Kind::ArbitraryPyObject) {
                context.serializePythonObject(mPyObject, buffer, 5);
            } else
            if (mElements.size()) {
                buffer.writeBeginCompound(4);
                for (long i = 0; i < mElements.size(); i++) {
                    mElements[i]->serialize(context, buffer, i);
                }
                buffer.writeEndCompound();
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

    template<class visitor_type>
    void visitOutbound(const visitor_type& v) {
        for (auto e: mElements) {
            v(e);
        }

        for (auto e: mKeys) {
            v(e);
        }

        for (auto& nameAndE: mNamedElements) {
            v(nameAndE.second);
        }
    }

private:
    // ensure we won't crash if we interact with this object.
    void validateAfterDeserialization() {
        if (mKind == Kind::PrimitiveType) {
            if (!mType) {
                throw std::runtime_error("Corrupt PyObjSnapshot: no Type");
            }
        }

        if (mKind == Kind::ArbitraryPyObject) {
            if (!mPyObject) {
                throw std::runtime_error("Corrupt PyObjSnapshot: no PyObject");
            }
        }
    }

    Kind mKind;
    PyObjGraphSnapshot* mGraph;

    // if we are a PrimitiveType, then our type. If we resolve to a Type, a representation
    // of us as that type. Otherwise nullptr.
    ::Type* mType;

    // if we are a PrimitiveInstance, the instance value. Otherwise a cache of our value
    // as an instance.
    ::Instance mInstance;

    // if we are an ArbitraryPythonObject, then our value. Otherwise, a cache of our value
    // as a PyObject.
    PyObject* mPyObject;

    // these are the actual data that represent us
    std::map<std::string, PyObjSnapshot*> mNamedElements;

    std::map<std::string, int64_t> mNamedInts;

    std::vector<PyObjSnapshot*> mElements;

    std::vector<std::string> mNames;

    std::vector<PyObjSnapshot*> mKeys;

    std::string mStringValue;

    std::string mModuleName;

    std::string mName;

    std::string mQualname;
};
