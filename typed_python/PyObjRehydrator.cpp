#include "PyObjRehydrator.hpp"
#include "PyObjSnapshot.hpp"


Type* PyObjRehydrator::typeFor(PyObjSnapshot* snapshot) {
    if (snapshot->mType) {
        return snapshot->mType;
    }

    if (snapshot->willBeATpType()) {
        rehydrate(snapshot);
        return snapshot->mType;
    }

    return nullptr;
}


PyObject* PyObjRehydrator::pyobjFor(PyObjSnapshot* snapshot) {
    if (snapshot->mPyObject) {
        return snapshot->mPyObject;
    }

    rehydrate(snapshot);

    return snapshot->mPyObject;
}


PyObject* PyObjRehydrator::getNamedElementPyobj(
    PyObjSnapshot* snapshot,
    std::string name,
    bool allowEmpty
) {
    auto it = snapshot->mNamedElements.find(name);
    if (it == snapshot->mNamedElements.end()) {
        if (allowEmpty) {
            return nullptr;
        }
        throw std::runtime_error(
            "Corrupt PyObjSnapshot." + snapshot->kindAsString() + ". missing " + name
        );
    }

    return pyobjFor(it->second);
}

Type* PyObjRehydrator::getNamedElementType(
    PyObjSnapshot* snapshot,
    std::string name,
    bool allowEmpty
) {
    auto it = snapshot->mNamedElements.find(name);
    if (it == snapshot->mNamedElements.end()) {
        if (allowEmpty) {
            return nullptr;
        }
        throw std::runtime_error(
            "Corrupt PyObjSnapshot." + snapshot->kindAsString() + ". missing " + name
        );
    }

    return typeFor(it->second);
}

void PyObjRehydrator::rehydrateTpType(PyObjSnapshot* snap) {
    if (snap->mType) {
        return;
    }

    if (snap->mKind == PyObjSnapshot::Kind::ListOfType) {
        snap->mType = new ListOfType();
        snap->mType->setIsForwardDefined(snap->mNamedInts["type_is_forward"]);
        ((ListOfType*)snap->mType)->initializeDuringDeserialization(getNamedElementType(snap, "element_type"));
        return;
    }

    if (snap->mKind == PyObjSnapshot::Kind::TupleOfType) {
        snap->mType = new TupleOfType();
        snap->mType->setIsForwardDefined(snap->mNamedInts["type_is_forward"]);
        ((TupleOfType*)snap->mType)->initializeDuringDeserialization(getNamedElementType(snap, "element_type"));
        return;
    }

    throw std::runtime_error("Can't rehydrate a PyObjSnapshot of kind " + snap->kindAsString());
}

void PyObjRehydrator::rehydrate(PyObjSnapshot* obj) {
    if (mSnapshots.find(obj) != mSnapshots.end()) {
        throw std::runtime_error("We should already have a rehydration for this object.");
    }

    mSnapshots.insert(obj);

    if (obj->willBeATpType()) {
        rehydrateTpType(obj);
        return;
    }

    static PyObject* sysModuleModules = staticPythonInstance("sys", "modules");

    PyObjSnapshot::Kind kind = obj->mKind;
    PyObject*& pyObject(obj->mPyObject);

    if (kind == PyObjSnapshot::Kind::ArbitraryPyObject) {
        throw std::runtime_error("Corrupt PyObjSnapshot.ArbitraryPyObject: missing pyObject");
    } else if (kind == PyObjSnapshot::Kind::PrimitiveType) {
        pyObject = (PyObject*)PyInstance::typeObj(obj->mType);
    } else if (kind == PyObjSnapshot::Kind::String) {
        pyObject = PyUnicode_FromString(obj->mStringValue.c_str());
    } else if (kind == PyObjSnapshot::Kind::Instance) {
        pyObject = PyInstance::extractPythonObject(obj->mInstance);
    } else if (kind == PyObjSnapshot::Kind::NamedPyObject) {
        PyObjectStealer nameAsStr(PyUnicode_FromString(obj->mModuleName.c_str()));
        PyObjectStealer moduleObject(
            PyObject_GetItem(sysModuleModules, nameAsStr)
        );

        if (!moduleObject) {
            PyErr_Clear();
            // TODO: should we be importing here? that seems dangerous...
            throw std::runtime_error("Somehow module " + obj->mModuleName + " is not loaded!");
        }
        PyObjectStealer inst(PyObject_GetAttrString(moduleObject, obj->mName.c_str()));
        if (!inst) {
            PyErr_Clear();

            // TODO: should we be importing here? that seems dangerous...
            throw std::runtime_error("Somehow module " + obj->mModuleName + " is missing member " + obj->mName);
        }

        pyObject = incref(inst);
    } else if (kind == PyObjSnapshot::Kind::PyDict || kind == PyObjSnapshot::Kind::PyClassDict) {
        pyObject = PyDict_New();
    } else if (kind == PyObjSnapshot::Kind::PyList) {
        pyObject = PyList_New(0);

        for (long k = 0; k < obj->mElements.size(); k++) {
            PyList_Append(pyObject, pyobjFor(obj->mElements[k]));
        }
    } else if (kind == PyObjSnapshot::Kind::PyTuple) {
        pyObject = PyTuple_New(obj->mElements.size());

        // first initialize it in case we throw somehow
        for (long k = 0; k < obj->mElements.size(); k++) {
            PyTuple_SetItem(pyObject, k, incref(Py_None));
        }

        for (long k = 0; k < obj->mElements.size(); k++) {
            PyTuple_SetItem(pyObject, k, incref(pyobjFor(obj->mElements[k])));
        }
    } else if (kind == PyObjSnapshot::Kind::PySet) {
        pyObject = PySet_New(nullptr);

        for (long k = 0; k < obj->mElements.size(); k++) {
            PySet_Add(pyObject, incref(pyobjFor(obj->mElements[k])));
        }
    } else if (kind == PyObjSnapshot::Kind::PyClass) {
        PyObjectStealer argTup(PyTuple_New(3));
        PyTuple_SetItem(argTup, 0, PyUnicode_FromString(obj->mName.c_str()));
        PyTuple_SetItem(argTup, 1, incref(getNamedElementPyobj(obj, "cls_bases")));
        PyTuple_SetItem(argTup, 2, incref(getNamedElementPyobj(obj, "cls_dict")));

        pyObject = PyType_Type.tp_new(&PyType_Type, argTup, nullptr);

        if (!pyObject) {
            throw PythonExceptionSet();
        }
    } else if (kind == PyObjSnapshot::Kind::PyModule) {
        PyObjectStealer nameAsStr(PyUnicode_FromString(obj->mName.c_str()));
        PyObjectStealer moduleObject(
            PyObject_GetItem(sysModuleModules, nameAsStr)
        );

        if (!moduleObject) {
            PyErr_Clear();
            // TODO: should we be importing here? that seems dangerous...
            throw std::runtime_error("Somehow module " + obj->mName + " is not loaded!");
        }

        pyObject = incref(moduleObject);
    } else if (kind == PyObjSnapshot::Kind::PyModuleDict) {
        pyObject = PyObject_GenericGetDict(getNamedElementPyobj(obj, "module_dict_of"), nullptr);
        if (!pyObject) {
            throw PythonExceptionSet();
        }
    } else if (kind == PyObjSnapshot::Kind::PyCell) {
        pyObject = PyCell_New(nullptr);
    } else if (kind == PyObjSnapshot::Kind::PyObject) {
        PyObject* t = getNamedElementPyobj(obj, "inst_type");
        PyObject* d = getNamedElementPyobj(obj, "inst_dict");

        static PyObject* emptyTuple = PyTuple_Pack(0);

        pyObject = ((PyTypeObject*)t)->tp_new(
            ((PyTypeObject*)t),
            emptyTuple,
            nullptr
        );

        if (PyObject_GenericSetDict(pyObject, d, nullptr)) {
            throw PythonExceptionSet();
        }
    } else if (kind == PyObjSnapshot::Kind::PyFunction) {
        pyObject = PyFunction_New(
            getNamedElementPyobj(obj, "func_code"),
            getNamedElementPyobj(obj, "func_globals")
        );

        if (obj->mNamedElements.find("func_closure") != obj->mNamedElements.end()) {
            PyFunction_SetClosure(
                pyObject,
                getNamedElementPyobj(obj, "func_closure")
            );
        }

        if (obj->mNamedElements.find("func_annotations") != obj->mNamedElements.end()) {
            PyFunction_SetAnnotations(
                pyObject,
                getNamedElementPyobj(obj, "func_annotations")
            );
        }

        if (obj->mNamedElements.find("func_defaults") != obj->mNamedElements.end()) {
            PyFunction_SetAnnotations(
                pyObject,
                getNamedElementPyobj(obj, "func_defaults")
            );
        }

        if (obj->mNamedElements.find("func_qualname") != obj->mNamedElements.end()) {
            PyObject_SetAttrString(
                pyObject,
                "__qualname__",
                getNamedElementPyobj(obj, "func_qualname")
            );
        }

        if (obj->mNamedElements.find("func_kwdefaults") != obj->mNamedElements.end()) {
            PyObject_SetAttrString(
                pyObject,
                "__kwdefaults__",
                getNamedElementPyobj(obj, "func_kwdefaults")
            );
        }

        if (obj->mNamedElements.find("func_name") != obj->mNamedElements.end()) {
            PyObject_SetAttrString(
                pyObject,
                "__name__",
                getNamedElementPyobj(obj, "func_name")
            );
        }
    } else if (kind == PyObjSnapshot::Kind::PyCodeObject) {
#if PY_MINOR_VERSION < 8
        pyObject = (PyObject*)PyCode_New(
#else
        pyObject = (PyObject*)PyCode_NewWithPosOnlyArgs(
#endif
            obj->mNamedInts["co_argcount"],
#if PY_MINOR_VERSION >= 8
            obj->mNamedInts["co_posonlyargcount"],
#endif
            obj->mNamedInts["co_kwonlyargcount"],
            obj->mNamedInts["co_nlocals"],
            obj->mNamedInts["co_stacksize"],
            obj->mNamedInts["co_flags"],
            getNamedElementPyobj(obj, "co_code"),
            getNamedElementPyobj(obj, "co_consts"),
            getNamedElementPyobj(obj, "co_names"),
            getNamedElementPyobj(obj, "co_varnames"),
            getNamedElementPyobj(obj, "co_freevars"),
            getNamedElementPyobj(obj, "co_cellvars"),
            getNamedElementPyobj(obj, "co_filename"),
            getNamedElementPyobj(obj, "co_name"),
            obj->mNamedInts["co_firstlineno"],
            getNamedElementPyobj(obj,
#if PY_MINOR_VERSION >= 10
                "co_linetable"
#else
                "co_lnotab"
#endif
            )
        );
    } else if (kind == PyObjSnapshot::Kind::PyStaticMethod) {
        pyObject = PyStaticMethod_New(Py_None);

        JustLikeAClassOrStaticmethod* method = (JustLikeAClassOrStaticmethod*)pyObject;
        decref(method->cm_callable);
        method->cm_callable = incref(getNamedElementPyobj(obj, "meth_func"));
    } else if (kind == PyObjSnapshot::Kind::PyClassMethod) {
        pyObject = PyClassMethod_New(Py_None);

        JustLikeAClassOrStaticmethod* method = (JustLikeAClassOrStaticmethod*)pyObject;
        decref(method->cm_callable);
        method->cm_callable = incref(getNamedElementPyobj(obj, "meth_func"));
    } else if (kind == PyObjSnapshot::Kind::PyProperty) {
        static PyObject* nones = PyTuple_Pack(3, Py_None, Py_None, Py_None);

        pyObject = PyObject_CallObject((PyObject*)&PyProperty_Type, nones);

        JustLikeAPropertyObject* dest = (JustLikeAPropertyObject*)pyObject;

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

        dest->prop_get = incref(getNamedElementPyobj(obj, "prop_get", true));
        dest->prop_set = incref(getNamedElementPyobj(obj, "prop_set", true));
        dest->prop_del = incref(getNamedElementPyobj(obj, "prop_del", true));
        dest->prop_doc = incref(getNamedElementPyobj(obj, "prop_doc", true));

        #if PY_MINOR_VERSION >= 10
        decref(dest->prop_name);
        dest->prop_name = incref(getNamedElementPyobj(obj, "prop_name", true));
        #endif
    } else if (kind == PyObjSnapshot::Kind::PyBoundMethod) {
        pyObject = PyMethod_New(Py_None, Py_None);
        PyMethodObject* method = (PyMethodObject*)pyObject;
        decref(method->im_func);
        decref(method->im_self);
        method->im_func = nullptr;
        method->im_self = nullptr;

        method->im_func = incref(getNamedElementPyobj(obj, "meth_func"));
        method->im_self = incref(getNamedElementPyobj(obj, "meth_self"));
    } else {
        throw std::runtime_error(
            "Can't make a python object representation for a PyObjSnapshot of kind " +
            obj->kindAsString()
        );
    }
}


void PyObjRehydrator::finalize() {
    for (auto n: mSnapshots) {
        finalizeRehydration(n);
    }

    for (auto n: mSnapshots) {
        finalizeRehydration2(n);
    }
}


void PyObjRehydrator::finalizeRehydration2(PyObjSnapshot* obj) {
    if (obj->mKind == PyObjSnapshot::Kind::PyClass) {
        if (obj->mNamedElements.find("cls_dict") == obj->mNamedElements.end()) {
            throw std::runtime_error("Corrupt PyClass - no cls_dict");
        }

        PyObjSnapshot* clsDictPyObj = obj->mNamedElements["cls_dict"];

        for (long k = 0; k < clsDictPyObj->elements().size() && k < clsDictPyObj->keys().size(); k++) {
            if (clsDictPyObj->keys()[k]->isString()) {
                PyObject_SetAttrString(
                    obj->mPyObject,
                    clsDictPyObj->keys()[k]->getStringValue().c_str(),
                    clsDictPyObj->elements()[k]->getPyObj()
                );
            }
        }
    }
}

void PyObjRehydrator::finalizeRehydration(PyObjSnapshot* obj) {
    if (obj->mKind == PyObjSnapshot::Kind::PyDict || obj->mKind == PyObjSnapshot::Kind::PyClassDict) {
        for (long k = 0; k < obj->mElements.size() && k < obj->mKeys.size(); k++) {
            PyDict_SetItem(
                obj->mPyObject,
                obj->mKeys[k]->getPyObj(),
                obj->mElements[k]->getPyObj()
            );
        }
    } else if (obj->mKind == PyObjSnapshot::Kind::PyCell) {
        auto it = obj->mNamedElements.find("cell_contents");
        if (it != obj->mNamedElements.end()) {
            PyCell_Set(
                obj->mPyObject,
                it->second->getPyObj()
            );
        }
    }
}
