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
#include "TypeOrPyobj.hpp"
#include "_types.hpp"
#include <set>
#include <unordered_map>

class ModuleRepresentation {
public:
    ModuleRepresentation(std::string name) {
        mModuleObject.steal(
            PyModule_New(name.c_str())
        );
    }

    void addExternal(PyObject* name, PyObject* value) {
        if (!PyUnicode_Check(name)) {
            throw std::runtime_error("Module member names are supposed to be strings.");
        }

        std::string nameStr(PyUnicode_AsUTF8(name));

        // reset the current 'extenral objects'
        for (auto obj: mExternalObjects[nameStr]) {
            decExternal(obj);
        }
        mExternalObjects[nameStr].clear();
        mInternalObjects[nameStr].clear();

        markExternal(nameStr, TypeOrPyobj(value));

        mVisible[nameStr];
        mPriorUpdateValues[nameStr] = value;

        PyDict_SetItem(PyModule_GetDict(mModuleObject), name, value);
    }

    // walk over the module definition and do two things:
    //     (1) mark any new 'external' objects (objects that cannot see back to this object)
    //     (2) build a map of which module members can "see" which other module members
    void update() {
        PyObject *key, *value;
        Py_ssize_t pos = 0;

        PyObject* moduleDict = PyModule_GetDict(mModuleObject);

        while (PyDict_Next(moduleDict, &pos, &key, &value)) {
            if (!PyUnicode_Check(key)) {
                throw std::runtime_error("Module member names are supposed to be strings.");
            }

            update(std::string(PyUnicode_AsUTF8(key)), value);
        }
    }

    void update(std::string key, TypeOrPyobj value) {
        if (value == mPriorUpdateValues[key]) {
            return;
        }

        clearName(key);

        mPriorUpdateValues[key] = value;
        PyObject* moduleObjectDict = PyModule_GetDict(mModuleObject);

        // check if its known to be external
        if (mExternalCount.find(value) != mExternalCount.end() 
            || isModuleObjectOrModuleDict(value) 
            || isPrimitive(value)
        ) {
            mExternalObjects[key].insert(value);
            incExternal(value);
            return;
        }

        // walk the object graph starting with 'value'.
        // we stop anytime we hit a module object - if that module is us, then
        // we've found a cycle and those objects are part of our graph. Otherwise
        // we've found an external object, which we mark

        // all objects we've seen.
        std::set<TypeOrPyobj> visited;

        // objects that can reach our module object or its dict
        std::set<TypeOrPyobj> internalObjects;
        
        // the stack above us
        std::vector<TypeOrPyobj> stack;
        std::set<TypeOrPyobj> stackSet;

        // for each thing we're checking, which values below it do we need
        // to look at
        std::map<TypeOrPyobj, std::set<TypeOrPyobj> > toCheck;

        stack.push_back(value);
        stackSet.insert(value);
        visited.insert(value);

        while (stack.size()) {
            // what we have on the stack is new. 
            TypeOrPyobj item = stack.back();

            if (toCheck.find(item) == toCheck.end()) {
                computeReachableFrom(item, toCheck[item]);
            }

            auto toCheckIt = toCheck.find(item);

            if (toCheckIt->second.size() == 0) {
                // we've checked everything below 'item' without consuming it, 
                // so it must not be 'internal'
                stack.pop_back();
                stackSet.erase(item);
                toCheck.erase(item);
                visited.insert(item);
            } else {
                TypeOrPyobj reachable = *toCheckIt->second.begin();
                toCheckIt->second.erase(reachable);

                // this object can reach our module. That means the entire stack above us
                // also reaches the module object
                if (reachable == moduleObjectDict || reachable == mModuleObject.get()) {
                    // mark the entire stack as internal and 'visited'. We'll still unwind
                    // the stack above us, but we won't worry about this otherwise
                    for (long k = 0; k < stack.size(); k++) {
                        internalObjects.insert(stack[k]);
                        visited.insert(stack[k]);
                    }
                } else if (visited.find(reachable) == visited.end()) {
                    // don't reach into module objects or module dicts
                    if (!isModuleObjectOrModuleDict(reachable)) {
                        // we have not visited this object before. Push it on the stack
                        stack.push_back(reachable);
                        stackSet.insert(reachable);
                    }
                }
            }
        }

        for (auto o: visited) {
            if (internalObjects.find(o) == internalObjects.end()) {
                markExternal(key, o);
            } else {
                mInternalObjects[key].insert(o);

                if (o.pyobj() && PyFunction_Check(o.pyobj())) {
                    std::set<std::string> names;
                    
                    Function::Overload::extractGlobalAccessesFromCode(
                        (PyCodeObject*)PyFunction_GetCode(o.pyobj()), 
                        names
                    );

                    for (auto n: names) {
                        mVisible[key].insert(n);
                    }
                }

                if (o.typeOrPyobjAsType() && o.typeOrPyobjAsType()->isFunction()) {
                    Function* f = (Function*)o.typeOrPyobjAsType();

                    std::set<std::string> names;

                    for (auto& o: f->getOverloads()) {
                        Function::Overload::extractGlobalAccessesFromCode(
                            (PyCodeObject*)o.getFunctionCode(), 
                            names
                        );

                        for (auto n: names) {
                            mVisible[key].insert(n);
                        }
                    }
                }
            }
        }
    }

    static bool isModuleObjectOrModuleDict(TypeOrPyobj topo) {
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

    static bool isPrimitive(TypeOrPyobj topo) {
        if (!topo.pyobj()) {
            return false;
        }

        PyObject* o = topo.pyobj();
        
        if (o == (PyObject*)&PyType_Type) {
            return true;
        }

        if (PyUnicode_Check(o) || PyLong_Check(o) || PyBytes_Check(o) || PyFloat_Check(o) || o == Py_None) {
            return true;
        }

        return false;
    }

    // put all the objects reachable from 'source' into 'reachable'
    void computeReachableFrom(TypeOrPyobj source, std::set<TypeOrPyobj>& reachable) {
        if (source.typeOrPyobjAsType()) {
            Type* t = source.typeOrPyobjAsType();

            if (t->isFunction()) {
                Function* f = (Function*)t;

                for (auto& o: f->getOverloads()) {
                    reachable.insert(TypeOrPyobj(o.getFunctionGlobals()));

                    for (auto nameAndObj: o.getFunctionGlobalsInCells()) {
                        if (nameAndObj.second) {
                            reachable.insert(TypeOrPyobj(nameAndObj.second));
                        }
                    }

                    if (o.getFunctionAnnotations()) {
                        reachable.insert(o.getFunctionAnnotations());
                    }

                    if (o.getFunctionDefaults()) {
                        reachable.insert(o.getFunctionDefaults());
                    }
                }
            }

            if (t->isClass()) {
                Class* c = (Class*)t;

                for (auto& base: c->getBases()) {
                    reachable.insert(TypeOrPyobj(base));
                }

                for (auto& nameAndF: c->getOwnMemberFunctions()) {
                    reachable.insert(TypeOrPyobj(nameAndF.second));
                }
                for (auto& nameAndF: c->getOwnStaticFunctions()) {
                    reachable.insert(TypeOrPyobj(nameAndF.second));
                }
                for (auto& nameAndF: c->getOwnPropertyFunctions()) {
                    reachable.insert(TypeOrPyobj(nameAndF.second));
                }
                for (auto& nameAndObj: c->getOwnClassMembers()) {
                    reachable.insert(TypeOrPyobj(nameAndObj.second));
                }
            }
        }

        if (source.pyobj()) {
            PyObject* o = source.pyobj();

            if (PyInstance::isNativeType(o->ob_type)) {
                reachable.insert(TypeOrPyobj(PyInstance::extractTypeFrom(o->ob_type)));
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
                        reachable.insert(((PyTypeObject*)o)->tp_dict);
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

                computeReachableFrom(TypeOrPyobj(dict), reachable);

                // the type object itself is also reachable
                reachable.insert((PyObject*)o->ob_type);
            }
        }
    }

    // copy our objects named by 'names' into 'other', deepcopying the 
    // 'internal' objects and migrating our references to 'external' objects.
    void copyInto(ModuleRepresentation& other, std::set<std::string>& names) {
        // make sure we're up to date
        update();

        std::map<TypeOrPyobj, TypeOrPyobj> objectMemo;

        for (auto name: names) {
            if (mPriorUpdateValues.find(name) != mPriorUpdateValues.end()) {
                other.clearName(name);

                // now copy our externals over
                for (auto o: mExternalObjects[name]) {
                    other.markExternal(name, o);
                }

                TypeOrPyobj updateVal = copyObject(
                    mPriorUpdateValues[name], 
                    objectMemo, 
                    mExternalObjects[name],
                    mModuleObject, 
                    other.mModuleObject
                );

                PyDict_SetItemString(
                    PyModule_GetDict(other.mModuleObject),
                    name.c_str(), 
                    updateVal.typeOrPyobjAsObject()
                );

                other.mPriorUpdateValues[name] = updateVal;

                for (auto o: mInternalObjects[name]) {
                    other.mInternalObjects[name].insert(
                        copyObject(
                            o, 
                            objectMemo,
                            mExternalObjects[name],
                            mModuleObject, 
                            other.mModuleObject
                        )
                    );
                }
            }
        }
    }

    // we have to assign both the Type* and PyObject* form in the memo
    static void setMemo(
        std::map<TypeOrPyobj, TypeOrPyobj>& objectMemo, 
        TypeOrPyobj key, 
        Type* valueType
    ) {
        Type* keyType = key.typeOrPyobjAsType();

        objectMemo[TypeOrPyobj(keyType)] = TypeOrPyobj(valueType);
        objectMemo[TypeOrPyobj((PyObject*)PyInstance::typeObj(keyType))] = TypeOrPyobj(
            (PyObject*)PyInstance::typeObj(valueType)
        );
    }

    // duplicate 'obj', replacing references to 'sourceModule' or its dict with 'destModule' and its dict.
    // objects should be placed into 'objectMemo' and recovered from there as well.
    static TypeOrPyobj copyObject(
        TypeOrPyobj obj, 
        std::map<TypeOrPyobj, TypeOrPyobj>& objectMemo, 
        const std::set<TypeOrPyobj>& externalObjects,
        PyObject* sourceModule, 
        PyObject* destModule
    ) {
        auto inMemo = objectMemo.find(obj);

        if (inMemo != objectMemo.end()) {
            return inMemo->second;
        }

        if (externalObjects.find(obj) != externalObjects.end()) {
            return obj;
        }

        if (isPrimitive(obj)) {
            return obj;
        }

        if (obj.pyobj() && obj.pyobj() == sourceModule) {
            return TypeOrPyobj(destModule);
        }

        if (obj.pyobj() && obj.pyobj() == PyModule_GetDict(sourceModule)) {
            return TypeOrPyobj(PyModule_GetDict(destModule));
        }

        if (isModuleObjectOrModuleDict(obj)) {
            return obj;
        }

        auto copy = [&](TypeOrPyobj o) {
            return copyObject(o, objectMemo, externalObjects, sourceModule, destModule);
        };

        if (obj.typeOrPyobjAsType()) {
            Type* t = obj.typeOrPyobjAsType();

            if (t->isFunction()) {
                Forward* forwardF = Forward::Make(t->name());

                // put the forward into the memo
                setMemo(objectMemo, obj, forwardF);

                Function* f = (Function*)t;

                std::vector<Function::Overload> overloads;

                for (auto& o: f->getOverloads()) {
                    overloads.push_back(
                        Function::Overload(
                            o.getFunctionCode(),
                            o.getFunctionGlobals() ? copy(o.getFunctionGlobals()).pyobj() : nullptr,
                            o.getFunctionDefaults() ? copy(o.getFunctionDefaults()).pyobj() : nullptr,
                            o.getFunctionAnnotations() ? copy(o.getFunctionAnnotations()).pyobj() : nullptr,
                            o.getFunctionGlobalsInCells(),
                            o.getFunctionClosureVarnames(),
                            o.getClosureVariableBindings(),
                            o.getReturnType(),
                            o.getArgs()
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
                setMemo(objectMemo, obj, outF);

                return objectMemo[obj];
            }

            if (t->isClass()) {
                Forward* forwardC = Forward::Make(t->name());

                // put the forward into the memo
                setMemo(objectMemo, obj, forwardC);

                Class* c = (Class*)t;

                std::vector<Class*> bases;

                std::map<std::string, Function*> memberFunctions;
                std::map<std::string, Function*> staticFunctions;
                std::map<std::string, Function*> propertyFunctions;
                std::map<std::string, PyObject*> classMembers;

                for (auto& base: c->getBases()) {
                    bases.push_back((Class*)copy(TypeOrPyobj(base->getClassType())).typeOrPyobjAsType());
                }

                for (auto& nameAndF: c->getOwnMemberFunctions()) {
                    memberFunctions[nameAndF.first] = (Function*)copy(nameAndF.second).typeOrPyobjAsType();
                }
                for (auto& nameAndF: c->getOwnStaticFunctions()) {
                    staticFunctions[nameAndF.first] = (Function*)copy(nameAndF.second).typeOrPyobjAsType();
                }
                for (auto& nameAndF: c->getOwnPropertyFunctions()) {
                    propertyFunctions[nameAndF.first] = (Function*)copy(nameAndF.second).typeOrPyobjAsType();
                }
                for (auto& nameAndObj: c->getOwnClassMembers()) {
                    classMembers[nameAndObj.first] = copy(nameAndObj.second).typeOrPyobjAsObject();
                }

                Type* outC = Class::Make(
                    c->name(),
                    bases,
                    c->isFinal(),
                    c->getOwnMembers(),
                    memberFunctions,
                    staticFunctions,
                    propertyFunctions,
                    classMembers
                );

                forwardC->define(outC);

                // update the memo so its not a forward
                setMemo(objectMemo, obj, outC);

                return objectMemo[obj];
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
                    Type* updatedF = copy(TypeOrPyobj(t)).typeOrPyobjAsType();

                    return TypeOrPyobj::steal(
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

                objectMemo[obj] = res;

                PyObject *key, *value;
                Py_ssize_t pos = 0;

                while (PyDict_Next(o, &pos, &key, &value)) {
                    PyDict_SetItem(
                        res, 
                        copy(key).pyobj(),
                        copy(value).pyobj()
                    );
                }

                return TypeOrPyobj::steal(res);
            } else if (PyTuple_Check(o)) {
                // empty tuple is a singleton
                if (!PyTuple_Size(o)) {
                    return obj;
                }

                PyObject* res = PyTuple_New(PyTuple_Size(o));

                objectMemo[obj] = res;

                for (long k = 0; k < PyTuple_Size(o); k++) {
                    //PyTuple_SET_ITEM steals a reference
                    PyTuple_SET_ITEM(res, k, incref(copy(PyTuple_GetItem(o, k)).pyobj()));
                }

                return TypeOrPyobj::steal(res);
            } else if (PyList_Check(o)) {
                PyObject* res = PyList_New(PyList_Size(o));
                objectMemo[obj] = res;

                for (long k = 0; k < PyList_Size(o); k++) {
                    // steals a reference to 'val', so we have to incref
                    PyList_SetItem(res, k, incref(copy(PyList_GetItem(o, k)).pyobj()));
                }

                return TypeOrPyobj::steal(res);
            } else if (PySet_Check(o)) {
                PyObject* res = PySet_New(nullptr);
                objectMemo[obj] = res;

                iterate(o, [&](PyObject* o2) {
                    PySet_Add(res, copy(o2).pyobj());
                });

                return TypeOrPyobj::steal(res);
            } else if (o->ob_type == &PyProperty_Type) {
                static PyObject* nones = PyTuple_Pack(3, Py_None, Py_None, Py_None);

                PyObject* res = PyObject_CallObject((PyObject*)&PyProperty_Type, nones);

                JustLikeAPropertyObject* source = (JustLikeAPropertyObject*)o;
                JustLikeAPropertyObject* dest = (JustLikeAPropertyObject*)res;

                decref(dest->prop_get);
                decref(dest->prop_set);
                decref(dest->prop_del);
                decref(dest->prop_doc);

                objectMemo[obj] = res;

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

                dest->getter_doc = source->getter_doc;

                return TypeOrPyobj::steal(res);
            } else if (o->ob_type == &PyStaticMethod_Type || o->ob_type == &PyClassMethod_Type) {
                static PyObject* nones = PyTuple_Pack(1, Py_None);

                PyObject* res = PyObject_CallObject((PyObject*)o->ob_type, nones);

                JustLikeAClassOrStaticmethod* source = (JustLikeAClassOrStaticmethod*)o;
                JustLikeAClassOrStaticmethod* dest = (JustLikeAClassOrStaticmethod*)res;

                decref(dest->cm_callable);
                decref(dest->cm_dict);

                objectMemo[obj] = res;

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

                return TypeOrPyobj::steal(res);
            } else if (PyFunction_Check(o)) {
                PyObject* res = PyFunction_New(
                    PyFunction_GetCode(o), 
                    copy(PyFunction_GetGlobals(o)).pyobj()
                );

                objectMemo[obj] = res;

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

                return TypeOrPyobj::steal(res);
            } else if (PyCell_Check(o)) {
                PyObject* res = PyCell_New(copy(PyCell_GET(o)).pyobj());
                objectMemo[obj] = res;
                return TypeOrPyobj::steal(res);
            } else if (PyType_Check(o)) {
                if (o->ob_type == &PyType_Type) {
                    PyTypeObject* in = (PyTypeObject*)o;

                    TypeOrPyobj bases;

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

                    if (!res) {
                        throw PythonExceptionSet();
                    }

                    if (in->tp_dict) {
                        TypeOrPyobj newTypeDict = copy(in->tp_dict);

                        PyObject *key, *value;
                        Py_ssize_t pos = 0;

                        while (PyDict_Next(newTypeDict.pyobj(), &pos, &key, &value)) {
                            if (!PyDict_GetItem(((PyTypeObject*)res)->tp_dict, key)) {
                                PyObject_SetAttr(res, key, value);
                            }
                        }
                    }

                    objectMemo[obj] = res;

                    return TypeOrPyobj::steal(res);
                }
            } else if (PyObject_HasAttrString(o, "__dict__")) {
                PyObjectStealer dict(PyObject_GetAttrString(o, "__dict__"));

                // duplicate the type object itself
                TypeOrPyobj typeObjAsPyObj = copy((PyObject*)o->ob_type);

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
                
                objectMemo[obj] = res;

                TypeOrPyobj otherDict = copy((PyObject*)dict);

                if (PyObject_GenericSetDict((PyObject*)res, otherDict.pyobj(), nullptr) == -1) {
                    decref(res);
                    throw PythonExceptionSet();
                }

                return TypeOrPyobj::steal(res);
            }
        }


        return obj;
    }   

    void clearName(std::string name) {
        mPriorUpdateValues.erase(name);
        mVisible.erase(name);

        for (auto obj: mExternalObjects[name]) {
            decExternal(obj);
        }

        mExternalObjects.erase(name);
        mInternalObjects.erase(name);
    }

    void markExternal(std::string name, TypeOrPyobj o) {
        if (mExternalObjects[name].find(o) != mExternalObjects[name].end()) {
            return;
        }

        mExternalObjects[name].insert(o);
        incExternal(o);
    }

    void incExternal(TypeOrPyobj o) {
        mExternalCount[o] += 1;
    }

    void decExternal(TypeOrPyobj o) {
        mExternalCount[o] -= 1;

        if (!mExternalCount[o]) {
            mExternalCount.erase(o);
        }
    }

    const std::set<TypeOrPyobj>& getInternalReferences(std::string name) {
        static std::set<TypeOrPyobj> empty;

        auto it = mInternalObjects.find(name);
        if (it == mInternalObjects.end()) {
            return empty;
        }

        return it->second;
    }

    const std::set<TypeOrPyobj>& getExternalReferences(std::string name) {
        static std::set<TypeOrPyobj> empty;

        auto it = mExternalObjects.find(name);
        if (it == mExternalObjects.end()) {
            return empty;
        }

        return it->second;
    }

    const std::set<std::string>& getVisibleNames(std::string name) {
        static std::set<std::string> empty;

        auto it = mVisible.find(name);
        if (it == mVisible.end()) {
            return empty;
        }

        return it->second;
    }

    PyObjectHolder mModuleObject;

    // for each module member, which other members are visible via a class or function
    std::map<std::string, std::set<std::string> > mVisible;

    // for each module member, the last value we used to compute the update
    std::map<std::string, TypeOrPyobj> mPriorUpdateValues;

    // for each module member, what is reachable from it that's considered external
    std::map<std::string, std::set<TypeOrPyobj> > mExternalObjects;

    // for each module member, the objects that are reachable from it back to the module
    std::map<std::string, std::set<TypeOrPyobj> > mInternalObjects;

    // for each thing that's considered external, how many things can reach it?
    std::map<TypeOrPyobj, long> mExternalCount;
};

