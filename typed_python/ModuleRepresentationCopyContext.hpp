/******************************************************************************
   Copyright 2017-2021 typed_python Authors

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

#include "PyInstance.hpp"
#include "FunctionType.hpp"
#include "PyObjectHandle.hpp"
#include "_types.hpp"
#include <set>
#include <unordered_map>


class OidMap {
public:
    bool has(PyObjectHandle o) const {
        return mObjToId.find(o) != mObjToId.end();
    }

    bool has(size_t s) const {
        return mIdToObj.find(s) != mIdToObj.end();
    }

    size_t get(PyObjectHandle o) const {
        auto it = mObjToId.find(o);
        if (it == mObjToId.end()) {
            return 0;
        }
        return it->second;
    }

    PyObjectHandle get(size_t s) const {
        auto it = mIdToObj.find(s);
        if (it == mIdToObj.end()) {
            return PyObjectHandle();
        }
        return it->second;
    }

    void add(PyObjectHandle o, size_t oid) {
        mObjToId[o] = oid;
        mIdToObj[oid] = o;
    }

private:
    std::unordered_map<PyObjectHandle, size_t> mObjToId;
    std::unordered_map<size_t, PyObjectHandle> mIdToObj;
};


class ModuleRepresentationCopyContext {
public:
    ModuleRepresentationCopyContext(
        const OidMap& inSourceOidMap,
        OidMap& inDestOidMap
    ) : mSourceOidMap(inSourceOidMap),
        mDestOidMap(inDestOidMap)
    {
    }

    void setMemo(PyObjectHandle key, Type* valueType) {
        setMemo(key, PyObjectHandle(valueType));
    }

    void setMemo(PyObjectHandle key, PyObject* obj) {
        return setMemo(key, PyObjectHandle(obj));
    }

    void setMemo(PyObjectHandle key, PyObjectHandle obj) {
        if (!mSourceOidMap.has(key)) {
            throw std::runtime_error(
                "mSourceOidMap doesn't have " + key.shortRepr()
            );
        }

        mDestOidMap.add(obj, mSourceOidMap.get(key));
    }

    Type* copyType(Type* t) {
        if (!t) {
            return t;
        }

        PyObjectHandle h(t);

        Type* res = copy(h).forceTypeObj();

        return res;
    }

    PyObject* copyObj(PyObject* o) {
        if (!o) {
            return o;
        }

        return copy(PyObjectHandle(o)).pyobj();
    }

    static bool isPrimitive(PyObjectHandle topo) {
        if (!topo.pyobj()) {
            return false;
        }

        PyObject* o = topo.pyobj();

        if (o == (PyObject*)&PyType_Type) {
            return true;
        }

        if (o == (PyObject*)&PyLong_Type) {
            return true;
        }

        if (o == (PyObject*)&PyFloat_Type) {
            return true;
        }

        if (o == (PyObject*)Py_None->ob_type) {
            return true;
        }

        if (o == (PyObject*)&PyBool_Type) {
            return true;
        }

        if (o == (PyObject*)&PyBytes_Type) {
            return true;
        }

        if (o == (PyObject*)&PyUnicode_Type) {
            return true;
        }

        if (PyUnicode_Check(o) || PyLong_Check(o) || PyBytes_Check(o) || PyFloat_Check(o) || o == Py_None) {
            return true;
        }

        return false;
    }

    static bool isModuleObjectOrModuleDict(PyObjectHandle topo) {
        if (!topo.pyobj()) {
            return false;
        }

        PyObject* o = topo.pyobj();

        if (PyModule_Check(o)) {
            return true;
        }

        if (PyDict_Check(o)) {
            if (PyDict_GetItemString(o, "__spec__") && PyDict_GetItemString(o, "__name__")
                 && PyDict_GetItemString(o, "__loader__")) {
                return true;
            }
        }

        return false;
    }

    // duplicate 'obj', replacing references to other objects in the
    // source module with objects in the dest module, adding the oids
    // if necessary.
    PyObjectHandle copy(PyObjectHandle obj) {
        size_t oid = mSourceOidMap.get(obj);

        // this is external
        if (!oid) {
            return obj;
        }

        // see if we've already copied this
        if (mDestOidMap.has(oid)) {
            return mDestOidMap.get(oid);
        }

        if (isPrimitive(obj)) {
            return obj;
        }

        if (isModuleObjectOrModuleDict(obj)) {
            return obj;
        }

        if (obj.typeObj()) {
            Type* t = obj.typeObj();

            if (t->isFunction()) {
                Forward* forwardF = Forward::Make(t->name());

                // put the forward into the memo
                setMemo(obj, forwardF);

                Function* f = (Function*)t;

                std::vector<Function::Overload> overloads;

                for (auto& o: f->getOverloads()) {
                    std::vector<Function::FunctionArg> args;
                    for (auto& a: o.getArgs()) {
                        args.push_back(
                            Function::FunctionArg(
                                a.getName(),
                                copyType(a.getTypeFilter()),
                                copyObj(a.getDefaultValue()),
                                a.getIsStarArg(),
                                a.getIsKwarg()
                            )
                        );
                    }

                    overloads.push_back(
                        Function::Overload(
                            o.getFunctionCode(),
                            copyObj(o.getFunctionGlobals()),
                            copyObj(o.getFunctionDefaults()),
                            copyObj(o.getFunctionAnnotations()),
                            o.getFunctionGlobalsInCells(),
                            o.getFunctionClosureVarnames(),
                            o.getClosureVariableBindings(),
                            copyType(o.getReturnType()),
                            o.getSignatureFunction(),
                            args,
                            o.getMethodOf()
                        )
                    );
                }

                Function* outF =
                    Function::Make(
                        f->name(),
                        f->qualname(),
                        f->moduleName(),
                        overloads,
                        f->getClosureType(),
                        f->isEntrypoint(),
                        f->isNocompile()
                    );

                forwardF->define(outF);

                // update the memo so its not a forward
                setMemo(obj, outF);

                return PyObjectHandle(outF);
            }

            if (t->isClass()) {
                std::string name = t->name();

                Forward* forwardC = Forward::Make(name);

                // put the forward into the memo
                setMemo(obj, forwardC);

                Class* c = (Class*)t;

                std::vector<Class*> bases;

                std::map<std::string, Function*> memberFunctions;
                std::map<std::string, Function*> staticFunctions;
                std::map<std::string, Function*> propertyFunctions;
                std::map<std::string, PyObject*> classMembers;
                std::map<std::string, Function*> classMethods;

                for (auto& base: c->getBases()) {
                    bases.push_back((Class*)copyType(base->getClassType()));
                }

                for (auto& nameAndF: c->getOwnMemberFunctions()) {
                    memberFunctions[nameAndF.first] = (Function*)copyType(nameAndF.second);
                }
                for (auto& nameAndF: c->getOwnStaticFunctions()) {
                    staticFunctions[nameAndF.first] = (Function*)copyType(nameAndF.second);
                }
                for (auto& nameAndF: c->getOwnClassMethods()) {
                    classMethods[nameAndF.first] = (Function*)copyType(nameAndF.second);
                }
                for (auto& nameAndF: c->getOwnPropertyFunctions()) {
                    propertyFunctions[nameAndF.first] = (Function*)copyType(nameAndF.second);
                }
                for (auto& nameAndObj: c->getOwnClassMembers()) {
                    classMembers[nameAndObj.first] = copyObj(nameAndObj.second);
                }

                Type* outC = Class::Make(
                    name,
                    bases,
                    c->isFinal(),
                    c->getOwnMembers(),
                    memberFunctions,
                    staticFunctions,
                    propertyFunctions,
                    classMembers,
                    classMethods
                );

                forwardC->define(outC);

                // update the memo so its not a forward
                setMemo(obj, outC);

                return PyObjectHandle(outC);
            }

            if (t->isForward()) {
                Forward* newForward = Forward::Make(t->name());

                // put the forward into the memo
                setMemo(obj, newForward);

                Forward* f = (Forward*)t;

                if (f->getTarget()) {
                    newForward->define(copyType(f->getTarget()));
                }

                return PyObjectHandle(newForward);
            }
        }

        if (obj.pyobj()) {
            PyObject* o = obj.pyobj();

            if (PyInstance::isNativeType(o->ob_type)) {
                Type* t = PyInstance::extractTypeFrom(o->ob_type);

                if (t->isFunction() || t->isClass()) {
                    // 'duplicating' types doesn't change their layout, and for the moment
                    // we ignore the possibility that class instances could hold references
                    // to globals through their instance data
                    Type* updatedF = copyType(t);

                    return PyObjectHandle::steal(
                        PyInstance::fromInstance(
                            Instance::create(updatedF, ((PyInstance*)o)->dataPtr())
                        )
                    );
                }

                // otherwise, don't do anything
                return obj;
            }


            // this object is 'internal' and needs to get duplicated.
            if (PyDict_Check(o)) {
                PyObject* res = PyDict_New();

                setMemo(obj, res);

                PyObject *key, *value;
                Py_ssize_t pos = 0;

                while (PyDict_Next(o, &pos, &key, &value)) {
                    PyDict_SetItem(
                        res,
                        copy(key).pyobj(),
                        copy(value).pyobj()
                    );
                }

                return PyObjectHandle::steal(res);
            } else if (PyTuple_Check(o)) {
                // empty tuple is a singleton
                if (!PyTuple_Size(o)) {
                    return obj;
                }

                PyObject* res = PyTuple_New(PyTuple_Size(o));

                setMemo(obj, res);

                for (long k = 0; k < PyTuple_Size(o); k++) {
                    //PyTuple_SET_ITEM steals a reference
                    PyTuple_SET_ITEM(res, k, incref(copy(PyTuple_GetItem(o, k)).pyobj()));
                }

                return PyObjectHandle::steal(res);
            } else if (PyList_Check(o)) {
                PyObject* res = PyList_New(PyList_Size(o));
                setMemo(obj, res);

                for (long k = 0; k < PyList_Size(o); k++) {
                    // steals a reference to 'val', so we have to incref
                    PyList_SetItem(res, k, incref(copy(PyList_GetItem(o, k)).pyobj()));
                }

                return PyObjectHandle::steal(res);
            } else if (PySet_Check(o)) {
                PyObject* res = PySet_New(nullptr);
                setMemo(obj, res);

                iterate(o, [&](PyObject* o2) {
                    PySet_Add(res, copy(o2).pyobj());
                });

                return PyObjectHandle::steal(res);
            } else if (o->ob_type == &PyProperty_Type) {
                static PyObject* nones = PyTuple_Pack(3, Py_None, Py_None, Py_None);

                PyObject* res = PyObject_CallObject((PyObject*)&PyProperty_Type, nones);

                JustLikeAPropertyObject* source = (JustLikeAPropertyObject*)o;
                JustLikeAPropertyObject* dest = (JustLikeAPropertyObject*)res;

                decref(dest->prop_get);
                decref(dest->prop_set);
                decref(dest->prop_del);
                decref(dest->prop_doc);

                #if PY_MINOR_VERSION >= 10
                decref(dest->prop_name);
                #endif

                setMemo(obj, res);

                if (source->prop_get) {
                    dest->prop_get = incref(copy(source->prop_get).pyobj());
                } else {
                    dest->prop_get = nullptr;
                }

                if (source->prop_set) {
                    dest->prop_set = incref(copy(source->prop_set).pyobj());
                } else {
                    dest->prop_set = nullptr;
                }

                if (source->prop_del) {
                    dest->prop_del = incref(copy(source->prop_del).pyobj());
                } else {
                    dest->prop_del = nullptr;
                }

                if (source->prop_doc) {
                    dest->prop_doc = incref(copy(source->prop_doc).pyobj());
                } else {
                    dest->prop_doc = nullptr;
                }

                #if PY_MINOR_VERSION >= 10
                if (source->prop_name) {
                    dest->prop_name = incref(copy(source->prop_name).pyobj());
                } else {
                    dest->prop_name = nullptr;
                }
                #endif

                dest->getter_doc = source->getter_doc;

                return PyObjectHandle::steal(res);
            } else if (o->ob_type == &PyStaticMethod_Type || o->ob_type == &PyClassMethod_Type) {
                static PyObject* nones = PyTuple_Pack(1, Py_None);

                PyObject* res = PyObject_CallObject((PyObject*)o->ob_type, nones);

                JustLikeAClassOrStaticmethod* source = (JustLikeAClassOrStaticmethod*)o;
                JustLikeAClassOrStaticmethod* dest = (JustLikeAClassOrStaticmethod*)res;

                decref(dest->cm_callable);
                decref(dest->cm_dict);

                setMemo(obj, res);

                if (source->cm_callable) {
                    dest->cm_callable = incref(copy(source->cm_callable).pyobj());
                } else {
                    dest->cm_callable = nullptr;
                }

                if (source->cm_dict) {
                    dest->cm_dict = incref(copy(source->cm_dict).pyobj());
                } else {
                    dest->cm_dict = nullptr;
                }

                return PyObjectHandle::steal(res);
            } else if (PyFunction_Check(o)) {
                PyObject* res = PyFunction_New(
                    PyFunction_GetCode(o),
                    copy(PyFunction_GetGlobals(o)).pyobj()
                );

                setMemo(obj, res);

                if (PyFunction_GetClosure(o)) {
                    PyFunction_SetClosure(res, copy(PyFunction_GetClosure(o)).pyobj());
                }

                if (PyFunction_GetDefaults(o)) {
                    PyFunction_SetDefaults(res, copy(PyFunction_GetDefaults(o)).pyobj());
                }

                if (PyFunction_GetAnnotations(o)) {
                    PyFunction_SetAnnotations(res, copy(PyFunction_GetAnnotations(o)).pyobj());
                }

                PyFunctionObject* inF = (PyFunctionObject*)o;
                PyFunctionObject* outF = (PyFunctionObject*)res;

                if (outF->func_module) {
                    decref(outF->func_module);
                    outF->func_module = nullptr;
                }

                if (inF->func_module) {
                    outF->func_module = incref(inF->func_module);
                }

                if (outF->func_qualname) {
                    decref(outF->func_qualname);
                    outF->func_qualname = nullptr;
                }

                if (inF->func_qualname) {
                    outF->func_qualname = incref(inF->func_qualname);
                }

                return PyObjectHandle::steal(res);
            } else if (PyCell_Check(o)) {
                PyObject* res = PyCell_New(nullptr);
                setMemo(obj, res);

                if (PyCell_GET(o)) {
                    PyCell_Set(res, copy(PyCell_GET(o)).pyobj());
                }

                return PyObjectHandle::steal(res);
            } else if (PyType_Check(o)) {
                if (o->ob_type == &PyType_Type) {
                    PyTypeObject* in = (PyTypeObject*)o;

                    PyObjectHandle bases;

                    if (in->tp_bases) {
                        bases = copy(in->tp_bases);
                    }

                    PyObjectStealer newEmptyTypeDict(PyDict_New());

                    static PyObject* emptyTuple = PyTuple_Pack(0);

                    PyObjectStealer tpNewArgs(
                        Py_BuildValue(
                            "(sOO)",
                            in->tp_name,
                            bases.pyobj() ? bases.pyobj() : emptyTuple,
                            (PyObject*)newEmptyTypeDict
                        )
                    );

                    PyObject* res = PyObject_CallObject((PyObject*)&PyType_Type, (PyObject*)tpNewArgs);

                    setMemo(obj, res);

                    if (!res) {
                        throw PythonExceptionSet();
                    }

                    if (in->tp_dict) {
                        PyObject *key, *value;
                        Py_ssize_t pos = 0;

                        while (PyDict_Next(in->tp_dict, &pos, &key, &value)) {
                            if (!PyDict_GetItem(((PyTypeObject*)res)->tp_dict, key)) {
                                PyObject_SetAttr(res, key, copy(value).pyobj());
                            }
                        }
                    }

                    if (PyObject_HasAttrString(o, "__module__")) {
                        PyObject_SetAttrString(res, "__module__", PyObject_GetAttrString(o, "__module__"));
                    }

                    if (PyObject_HasAttrString(o, "__doc__")) {
                        PyObject_SetAttrString(res, "__doc__", PyObject_GetAttrString(o, "__doc__"));
                    }

                    return PyObjectHandle::steal(res);
                }
            } else if (PyObject_HasAttrString(o, "__dict__")) {
                PyObjectStealer dict(PyObject_GetAttrString(o, "__dict__"));

                // duplicate the type object itself
                PyObjectHandle typeObjAsPyObj = copy((PyObject*)o->ob_type);

                if (!PyType_Check(typeObjAsPyObj.pyobj())) {
                    throw std::runtime_error(
                        "Duplicating " + std::string(o->ob_type->tp_name) + " didn't result in a type"
                    );
                }

                PyTypeObject* typeObj = ((PyTypeObject*)typeObjAsPyObj.pyobj());

                static PyObject* emptyTuple = PyTuple_Pack(0);

                PyObject* res(typeObj->tp_new(typeObj, emptyTuple, NULL));

                if (!res) {
                    throw std::runtime_error(
                        "tp_new for " + std::string(typeObj->tp_name) + " threw an exception."
                    );
                }

                setMemo(obj, res);

                PyObjectHandle otherDict = copy((PyObject*)dict);

                if (PyObject_GenericSetDict((PyObject*)res, otherDict.pyobj(), nullptr) == -1) {
                    decref(res);
                    throw PythonExceptionSet();
                }

                return PyObjectHandle::steal(res);
            }
        }


        return obj;
    }

    // put all the objects reachable from 'source' in a single hop into 'reachable'.
    template<class set_type>
    static void computeReachableFrom(PyObjectHandle source, set_type& reachable) {
        if (source.typeObj()) {
            Type* t = source.typeObj();

            if (t->isFunction()) {
                Function* f = (Function*)t;

                for (auto& o: f->getOverloads()) {
                    for (auto& a: o.getArgs()) {
                        if (a.getTypeFilter()) {
                            reachable.insert(a.getTypeFilter());
                        }

                        if (a.getDefaultValue()) {
                            reachable.insert(a.getDefaultValue());
                        }
                    }

                    reachable.insert(PyObjectHandle(o.getFunctionGlobals()));

                    for (auto nameAndObj: o.getFunctionGlobalsInCells()) {
                        if (nameAndObj.second) {
                            reachable.insert(PyObjectHandle(nameAndObj.second));
                        }
                    }

                    if (o.getFunctionAnnotations()) {
                        reachable.insert(o.getFunctionAnnotations());
                    }

                    if (o.getFunctionDefaults()) {
                        reachable.insert(o.getFunctionDefaults());
                    }

                    if (o.getReturnType()) {
                        reachable.insert(PyObjectHandle(o.getReturnType()));
                    }
                }
            }

            if (t->isClass()) {
                Class* c = (Class*)t;

                for (auto& base: c->getBases()) {
                    reachable.insert(PyObjectHandle(base->getClassType()));
                }

                for (auto& nameAndF: c->getOwnMemberFunctions()) {
                    reachable.insert(PyObjectHandle(nameAndF.second));
                }
                for (auto& nameAndF: c->getOwnStaticFunctions()) {
                    reachable.insert(PyObjectHandle(nameAndF.second));
                }
                for (auto& nameAndF: c->getOwnPropertyFunctions()) {
                    reachable.insert(PyObjectHandle(nameAndF.second));
                }
                for (auto& nameAndObj: c->getOwnClassMembers()) {
                    reachable.insert(PyObjectHandle(nameAndObj.second));
                }
            }

            if (t->isForward()) {
                Forward* f = (Forward*)t;

                if (f->getTarget()) {
                    reachable.insert(PyObjectHandle(f->getTarget()));
                }
            }

            return;
        }

        if (source.pyobj()) {
            PyObject* o = source.pyobj();

            if (PyInstance::isNativeType(o->ob_type)) {
                reachable.insert(PyObjectHandle(PyInstance::extractTypeFrom(o->ob_type)));
                return;
            }

            if (PyDict_Check(o)) {
                PyObject *key, *value;
                Py_ssize_t pos = 0;

                while (PyDict_Next(o, &pos, &key, &value)) {
                    if (!isPrimitive(key)) {
                        reachable.insert(key);
                    }

                    if (!isPrimitive(value)) {
                        reachable.insert(value);
                    }
                }
            } else if (PyTuple_Check(o)) {
                for (long k = 0; k < PyTuple_Size(o); k++) {
                    reachable.insert(PyTuple_GetItem(o, k));
                }
            } else if (PyList_Check(o)) {
                for (long k = 0; k < PyList_Size(o); k++) {
                    reachable.insert(PyList_GetItem(o, k));
                }
            } else if (PySet_Check(o)) {
                iterate(o, [&](PyObject* o2) { reachable.insert(o2); });
            } else if (PyFunction_Check(o)) {
                reachable.insert(PyFunction_GetGlobals(o));

                if (PyFunction_GetClosure(o)) {
                    reachable.insert(PyFunction_GetClosure(o));
                }

                if (PyFunction_GetAnnotations(o)) {
                    reachable.insert(PyFunction_GetAnnotations(o));
                }
            } else if (PyCell_Check(o)) {
                if (PyCell_GET(o)) {
                    reachable.insert(PyCell_GET(o));
                }
            } else if (PyType_Check(o)) {
                if (o->ob_type == &PyType_Type) {
                    // this is a user-defined type
                    if (((PyTypeObject*)o)->tp_dict) {
                        PyObject *key, *value;
                        Py_ssize_t pos = 0;

                        while (PyDict_Next(((PyTypeObject*)o)->tp_dict, &pos, &key, &value)) {
                            reachable.insert(value);
                        }
                    }
                    if (((PyTypeObject*)o)->tp_bases) {
                        reachable.insert(((PyTypeObject*)o)->tp_bases);
                    }
                }
            } else if (o->ob_type == &PyProperty_Type) {
                JustLikeAPropertyObject* source = (JustLikeAPropertyObject*)o;

                if (source->prop_get) {
                    reachable.insert(source->prop_get);
                }

                if (source->prop_set) {
                    reachable.insert(source->prop_set);
                }

                if (source->prop_del) {
                    reachable.insert(source->prop_del);
                }

                if (source->prop_doc) {
                    reachable.insert(source->prop_doc);
                }

                #if PY_MINOR_VERSION >= 10
                    if (source->prop_name) {
                        reachable.insert(source->prop_name);
                    }
                #endif


            } else if (o->ob_type == &PyStaticMethod_Type || o->ob_type == &PyClassMethod_Type) {
                JustLikeAClassOrStaticmethod* source = (JustLikeAClassOrStaticmethod*)o;

                if (source->cm_callable) {
                    reachable.insert(source->cm_callable);
                }

                if (source->cm_dict) {
                    reachable.insert(source->cm_dict);
                }

            } else if (PyObject_HasAttrString(o, "__dict__")) {
                PyObjectStealer dict(PyObject_GetAttrString(o, "__dict__"));

                computeReachableFrom(PyObjectHandle(dict), reachable);

                // the type object itself is also reachable
                reachable.insert((PyObject*)o->ob_type);
            }
        }
    }

private:
    const OidMap& mSourceOidMap;
    OidMap& mDestOidMap;
};
