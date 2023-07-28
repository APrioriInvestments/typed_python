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


/*********************************
CompilerVisiblePyObject

a representation of a python object that's owned by TypedPython.  We hold these by
pointer and leak them indiscriminately - like Type objects they're considered to be permanent
and singletonish.

They are intended to be a 'snapshot' of the state of a collection of python objects and
contain enough information for the compiler to use them to build compiled code.

To the extent that they visit 'mutable' objects like a list (which might be contained within
a default argument), they will encode the state of the object when it was first seen. This
lets us determine if the object was modified (which would break compiler invariants) and also
gives us a self-consistent view of the world to compile against so we don't have state
changing underneath us.
**********************************/

class CompilerVisiblePyObj {
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
        Type,
        // we're pointing into a TP instance that doesn't reach a more complex object.
        // It can have Type leaves in it. It will be held in 'mInstance'
        Instance,
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
        ArbitraryPyObject
    };

    CompilerVisiblePyObj() :
        mKind(Kind::Uninitialized),
        mType(nullptr),
        mPyObject(nullptr)
    {
    }

public:
    static CompilerVisiblePyObj* ForType(Type* t) {
        CompilerVisiblePyObj* res = new CompilerVisiblePyObj();
        res->mKind = Kind::Type;
        res->mType = t;
        return res;
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

        throw std::runtime_error("Unknown CompilerVisiblePyObj Kind");
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

    // return a CVPO for 'val', stashing it in 'constantMapCache'
    // in case we hit a recursion.
    static CompilerVisiblePyObj* internalizePyObj(
        PyObject* val,
        std::unordered_map<PyObject*, CompilerVisiblePyObj*>& constantMapCache,
        const std::map<::Type*, ::Type*>& groupMap,
        bool linkBackToOriginalObject = true
    ) {
        auto it = constantMapCache.find(val);

        if (it != constantMapCache.end()) {
            return it->second;
        }

        constantMapCache[val] = new CompilerVisiblePyObj();

        constantMapCache[val]->becomeInternalizedOf(
            val, constantMapCache, groupMap, linkBackToOriginalObject
        );

        return constantMapCache[val];
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
        PyObject* val,
        std::unordered_map<PyObject*, CompilerVisiblePyObj*>& constantMapCache,
        const std::map<::Type*, ::Type*>& groupMap,
        bool linkBackToOriginalObject
    ) {
        // we're always the internalized version of this object
        if (linkBackToOriginalObject) {
            mPyObject = incref(val);
        }

        auto internalize = [&](PyObject* o) {
            return CompilerVisiblePyObj::internalizePyObj(
                o, constantMapCache, groupMap, linkBackToOriginalObject
            );
        };

        ::Type* t = PyInstance::extractTypeFrom(val, true);

        if (t) {
            if (groupMap.find(t) != groupMap.end()) {
                t = groupMap.find(t)->second;
            } else {
                if (t->isForwardDefined()) {
                    if (t->isResolved()) {
                        t = t->forwardResolvesTo();
                    }
                }
            }

            mKind = Kind::Type;
            mType = t;
            return;
        }

        PyObject* environType = staticPythonInstance("os", "_Environ");

        if (val->ob_type == (PyTypeObject*)environType) {
            mKind = Kind::NamedPyObject;
            mName = "_Environ";
            mModuleName = "os";
            return;
        }


        if (PyUnicode_Check(val)) {
            mKind = Kind::String;
            mStringValue = PyUnicode_AsUTF8(val);
            return;
        }

        if (PyTuple_Check(val)) {
            mKind = Kind::PyTuple;
            for (long i = 0; i < PyTuple_Size(val); i++) {
                mElements.push_back(internalize(PyTuple_GetItem(val, i)));
            }
            return;
        }

        if (PyList_Check(val)) {
            mKind = Kind::PyList;
            for (long i = 0; i < PyList_Size(val); i++) {
                mElements.push_back(internalize(PyList_GetItem(val, i)));
            }
            return;
        }

        if (PySet_Check(val)) {
            mKind = Kind::PySet;
            iterate(val, [&](PyObject* o) {
                mElements.push_back(internalize(o));
            });
            return;
        }

        if (PyDict_Check(val)) {
            // see if this is a moduledict
            PyObject* mname = PyDict_GetItemString(val, "__name__");
            if (mname && PyUnicode_Check(mname)) {
                PyObject* sysModuleModules = staticPythonInstance("sys", "modules");
                PyObjectStealer moduleObj(PyObject_GetItem(sysModuleModules, mname));

                if (moduleObj) {
                    PyObjectStealer moduleObjDict(PyObject_GenericGetDict(moduleObj, nullptr));
                    if (!moduleObjDict) {
                        PyErr_Clear();
                    } else
                    if (moduleObjDict == val) {
                        mKind = Kind::PyModuleDict;
                        mName = PyUnicode_AsUTF8(mname);
                        mNamedElements["module_dict_of"] = internalize(moduleObj);
                        return;
                    }
                } else {
                    PyErr_Clear();
                }
            }

            // see if this is a vanilla class dict
            PyObject* dictAccessor = PyDict_GetItemString(val, "__dict__");
            if (dictAccessor && dictAccessor->ob_type == &PyGetSetDescr_Type) {
                PyTypeObject* clsType = PyDescr_TYPE(dictAccessor);
                if (clsType) {
                    if (clsType->tp_dict == val) {
                        if (isVanillaClassType(clsType)) {
                            mKind = Kind::PyClassDict;

                            mNamedElements["class_dict_of"] = internalize(
                                (PyObject*)clsType
                            );

                            PyObject *key, *value;
                            Py_ssize_t pos = 0;

                            while (val && PyDict_Next(val, &pos, &key, &value)) {
                                if (PyUnicode_Check(key)
                                    && PyUnicode_AsUTF8(key) != std::string("__dict__")
                                    && PyUnicode_AsUTF8(key) != std::string("__weakref__")
                                ) {
                                    mElements.push_back(internalize(value));
                                    mKeys.push_back(internalize(key));
                                }
                            }

                            return;
                        }
                    }
                }
            }

            // this is a vanilla dict
            mKind = Kind::PyDict;

            PyObject *key, *value;
            Py_ssize_t pos = 0;

            while (val && PyDict_Next(val, &pos, &key, &value)) {
                mElements.push_back(internalize(value));
                mKeys.push_back(internalize(key));
            }
            return;
        }

        if (PyCell_Check(val)) {
            mKind = Kind::PyCell;

            if (PyCell_Get(val)) {
                mNamedElements["cell_contents"] = internalize(PyCell_Get(val));
            }

            return;
        }

        if (PyType_Check(val)) {
            PyTypeObject* tp = (PyTypeObject*)val;

            if (isVanillaClassType(tp)) {
                mKind = Kind::PyClass;

                mName = tp->tp_name;
                mModuleName = dictGetStringOrEmpty(tp->tp_dict, "__module__");

                if (tp->tp_dict) {
                    mNamedElements["cls_dict"] = internalize(tp->tp_dict);
                }
                if (tp->tp_bases) {
                    mNamedElements["cls_bases"] = internalize(tp->tp_bases);
                }

                return;
            }
        }

        if (PyFunction_Check(val)) {
            mKind = Kind::PyFunction;

            PyFunctionObject* f = (PyFunctionObject*)val;

            mName = stringOrEmpty(f->func_name);
            mModuleName = stringOrEmpty(f->func_module);

            if (f->func_name) {
                mNamedElements["func_name"] = internalize(f->func_name);
            }
            if (f->func_module) {
                mNamedElements["func_module"] = internalize(f->func_module);
            }
            if (f->func_qualname) {
                mNamedElements["func_qualname"] = internalize(f->func_qualname);
            }
            if (PyFunction_GetClosure(val)) {
                mNamedElements["func_closure"] = internalize(PyFunction_GetClosure(val));
            }
            if (PyFunction_GetCode(val)) {
                mNamedElements["func_code"] = internalize(PyFunction_GetCode(val));
            }
            if (PyFunction_GetModule(val)) {
                mNamedElements["func_module"] = internalize(PyFunction_GetModule(val));
            }
            if (PyFunction_GetAnnotations(val)) {
                mNamedElements["func_annotations"] = internalize(PyFunction_GetAnnotations(val));
            }
            if (PyFunction_GetDefaults(val)) {
                mNamedElements["func_defaults"] = internalize(PyFunction_GetDefaults(val));
            }
            if (PyFunction_GetKwDefaults(val)) {
                mNamedElements["func_kwdefaults"] = internalize(PyFunction_GetKwDefaults(val));
            }
            if (PyFunction_GetGlobals(val)) {
                mNamedElements["func_globals"] = internalize(PyFunction_GetGlobals(val));
            }
            return;
        }

        if (PyCode_Check(val)) {
            mKind = Kind::PyCodeObject;

            PyCodeObject* co = (PyCodeObject*)val;

            mNamedInts["co_argcount"] = co->co_argcount;
            mNamedInts["co_kwonlyargcount"] = co->co_kwonlyargcount;
            mNamedInts["co_nlocals"] = co->co_nlocals;
            mNamedInts["co_stacksize"] = co->co_stacksize;
            mNamedInts["co_firstlineno"] = co->co_firstlineno;
            mNamedInts["co_posonlyargcount"] = co->co_posonlyargcount;
            mNamedInts["co_flags"] = co->co_flags;

            mNamedElements["co_code"] = internalize(co->co_code);
            mNamedElements["co_consts"] = internalize(co->co_consts);
            mNamedElements["co_names"] = internalize(co->co_names);
            mNamedElements["co_varnames"] = internalize(co->co_varnames);
            mNamedElements["co_freevars"] = internalize(co->co_freevars);
            mNamedElements["co_cellvars"] = internalize(co->co_cellvars);
            mNamedElements["co_name"] = internalize(co->co_name);
            mNamedElements["co_filename"] = internalize(co->co_filename);

#           if PY_MINOR_VERSION >= 10
                mNamedElements["co_linetable"] = internalize(co->co_linetable);
#           else
                mNamedElements["co_lnotab"] = internalize(co->co_lnotab);
#           endif

            return;
        }

        ::Type* instanceType = PyInstance::extractTypeFrom(val->ob_type);
        if (instanceType) {
            mKind = Kind::Instance;
            mInstance = ::Instance::create(
                instanceType,
                ((PyInstance*)val)->dataPtr()
            );
            return;
        }

        if (val == Py_None) {
            mKind = Kind::Instance;
            return;
        }

        if (PyBool_Check(val)) {
            mKind = Kind::Instance;
            mInstance = Instance::create(val == Py_True);
            return;
        }

        if (PyLong_Check(val)) {
            mKind = Kind::Instance;

            try {
                mInstance = Instance::create((int64_t)PyLong_AsLongLong(val));
            }
            catch(...) {
                mInstance = Instance::create((uint64_t)PyLong_AsUnsignedLongLong(val));
            }

            return;
        }

        if (PyFloat_Check(val)) {
            mKind = Kind::Instance;
            mInstance = Instance::create(PyFloat_AsDouble(val));
            return;
        }

        if (PyModule_Check(val)) {
            PyObject* sysModuleModules = staticPythonInstance("sys", "modules");

            PyObjectStealer name(PyObject_GetAttrString(val, "__name__"));
            if (name) {
                if (PyUnicode_Check(name)) {
                    PyObjectStealer moduleObject(PyObject_GetItem(sysModuleModules, name));
                    if (moduleObject) {
                        if (moduleObject == val) {
                            mKind = Kind::PyModule;
                            mName = PyUnicode_AsUTF8(name);
                            return;
                        }
                    } else {
                        PyErr_Clear();
                    }
                }
            } else {
                PyErr_Clear();
            }
        }

        if (PyBytes_Check(val)) {
            mKind = Kind::Instance;
            mInstance = Instance::createAndInitialize(
                BytesType::Make(),
                [&](instance_ptr i) {
                    BytesType::Make()->constructor(
                        i,
                        PyBytes_GET_SIZE(val),
                        PyBytes_AsString(val)
                    );
                }
            );
            return;
        }

        if (isVanillaClassType(val->ob_type)) {
            mKind = Kind::PyObject;

            mNamedElements["inst_type"] = internalize((PyObject*)val->ob_type);

            PyObjectStealer dict(PyObject_GenericGetDict(val, nullptr));
            if (dict) {
                mNamedElements["inst_dict"] = internalize(dict);
            }
            return;
        }

        if (val->ob_type == &PyStaticMethod_Type || val->ob_type == &PyClassMethod_Type) {
            if (val->ob_type == &PyStaticMethod_Type) {
                mKind = Kind::PyStaticMethod;
            } else {
                mKind = Kind::PyClassMethod;
            }

            PyObjectStealer funcObj(PyObject_GetAttrString(val, "__func__"));

            mNamedElements["meth_func"] = internalize(funcObj);

            return;
        }

        if (val->ob_type == &PyProperty_Type) {
            mKind = Kind::PyProperty;

            JustLikeAPropertyObject* prop = (JustLikeAPropertyObject*)val;

            if (prop->prop_get) {
                mNamedElements["prop_get"] = internalize(prop->prop_get);
            }
            if (prop->prop_set) {
                mNamedElements["prop_set"] = internalize(prop->prop_set);
            }
            if (prop->prop_del) {
                mNamedElements["prop_del"] = internalize(prop->prop_del);
            }
            if (prop->prop_doc) {
                mNamedElements["prop_doc"] = internalize(prop->prop_doc);
            }

            #if PY_MINOR_VERSION >= 10
            if (prop->prop_name) {
                mNamedElements["prop_name"] = internalize(prop->prop_name);
            }
            #endif

            return;
        }

        if (val->ob_type == &PyMethod_Type) {
            mKind = Kind::PyBoundMethod;

            PyObjectStealer fself(PyObject_GetAttrString(val, "__self__"));
            PyObjectStealer ffunc(PyObject_GetAttrString(val, "__func__"));

            mNamedElements["meth_self"] = internalize(fself);
            mNamedElements["meth_func"] = internalize(ffunc);

            return;
        }

        if (isPyObjectGloballyIdentifiable(val)) {
            mKind = Kind::NamedPyObject;

            // no checks are necessary because isPyObjectGloballyIdentifiable
            // confirms that this is OK.
            PyObjectStealer moduleName(PyObject_GetAttrString(val, "__module__"));
            PyObjectStealer clsName(PyObject_GetAttrString(val, "__name__"));

            mModuleName = PyUnicode_AsUTF8(moduleName);
            mName = PyUnicode_AsUTF8(clsName);
            return;
        }

        mKind = Kind::ArbitraryPyObject;
        mPyObject = incref(val);
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

    const std::vector<CompilerVisiblePyObj*>& keys() const {
        return mKeys;
    }

    const std::map<std::string, CompilerVisiblePyObj*>& namedElements() const {
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
            throw std::runtime_error("TODO: CompilerVisiblePyObj::_visitCompilerVisibleInternals PyTuple");
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

        std::unordered_set<CompilerVisiblePyObj*> needsResolution;

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

            CompilerVisiblePyObj* clsDictPyObj = mNamedElements["cls_dict"];

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

    PyObject* getPyObj(std::unordered_set<CompilerVisiblePyObj*>& needsResolution) {
        PyObject* sysModuleModules = staticPythonInstance("sys", "modules");

        if (!mPyObject) {
            if (mKind == Kind::ArbitraryPyObject) {
                throw std::runtime_error("Corrupt CompilerVisiblePyObj.ArbitraryPyObject: missing mPyObject");
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
        std::unordered_set<CompilerVisiblePyObj*>& needsResolution,
        bool allowEmpty=false
    ) {
        auto it = mNamedElements.find(name);
        if (it == mNamedElements.end()) {
            if (allowEmpty) {
                return nullptr;
            }
            throw std::runtime_error(
                "Corrupt CompilerVisiblePyObj." + kindAsString() + ". missing " + name
            );
        }

        return it->second->getPyObj(needsResolution);
    }

    std::string toString() const {
        return "CompilerVisiblePyObj." + kindAsString() + "()";
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
    static CompilerVisiblePyObj* deserialize(serialization_context_t& context, buf_t& buffer, int wireType) {
        int64_t kind = 0;

        ::Type* type = nullptr;

        std::vector<CompilerVisiblePyObj*> vec;
        ::Instance i;
        uint32_t id = -1;
        PyObjectHolder pyobj;
        CompilerVisiblePyObj* result = nullptr;

        buffer.consumeCompoundMessage(wireType, [&](size_t fieldNumber, size_t wireType) {
            if (fieldNumber == 0) {
                id = buffer.readUnsignedVarint();

                void* ptr = buffer.lookupCachedPointer(id);
                if (ptr) {
                    result = (CompilerVisiblePyObj*)result;
                } else {
                    result = new CompilerVisiblePyObj();
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
                    vec.push_back(CompilerVisiblePyObj::deserialize(context, buffer, wireType));
                });
            } else
            if (fieldNumber == 5) {
                pyobj.steal(context.deserializePythonObject(buffer, wireType));
            }
        });

        if (!result) {
            throw std::runtime_error("corrupt CompilerVisiblePyObj - no memo found");
        }

        if (kind == -1) {
            throw std::runtime_error("corrupt CompilerVisiblePyObj - invalid kind");
        }

        result->mKind = Kind(kind);
        result->mType = type;
        result->mInstance = i;
        result->mPyObject = incref(pyobj);
        result->mElements = vec;

        result->validateAfterDeserialization();

        return result;
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

    ::Type* mType;
    ::Instance mInstance;

    // if we are an ArbitraryPythonObject this is always populated
    // otherwise, it will be a cache for a constructed instance of the object
    PyObject* mPyObject;

    std::map<std::string, CompilerVisiblePyObj*> mNamedElements;
    std::map<std::string, int64_t> mNamedInts;

    // if we're a tuple, list, or dict
    std::vector<CompilerVisiblePyObj*> mElements;

    // if we're a PyDict
    std::vector<CompilerVisiblePyObj*> mKeys;

    std::string mStringValue;
    std::string mModuleName;
    std::string mName;
};
