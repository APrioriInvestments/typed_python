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
#include "ModuleRepresentationCopyContext.hpp"



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
        return ModuleRepresentationCopyContext::isModuleObjectOrModuleDict(topo);
    }

    static bool isPrimitive(TypeOrPyobj topo) {
        return ModuleRepresentationCopyContext::isPrimitive(topo);
    }

    // put all the objects reachable from 'source' into 'reachable'
    void computeReachableFrom(TypeOrPyobj source, std::set<TypeOrPyobj>& reachable) {
        if (source.typeOrPyobjAsType()) {
            Type* t = source.typeOrPyobjAsType();

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

                    if (o.getReturnType()) {
                        reachable.insert(TypeOrPyobj(o.getReturnType()));
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

            if (t->isForward()) {
                Forward* f = (Forward*)t;

                if (f->getTarget()) {
                    reachable.insert(TypeOrPyobj(f->getTarget()));
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

    static TypeOrPyobj copyObject(
            TypeOrPyobj obj,
            std::map<TypeOrPyobj, TypeOrPyobj>& objectMemo,
            const std::set<TypeOrPyobj>& externalObjects,
            PyObject* sourceModule,
            PyObject* destModule
        )
    {
        ModuleRepresentationCopyContext ctx(objectMemo, externalObjects, sourceModule, destModule);

        return ctx.copy(obj);
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
