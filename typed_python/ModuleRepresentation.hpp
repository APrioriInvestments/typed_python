/******************************************************************************
   Copyright 2017-2022 typed_python Authors

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
#include "ModuleRepresentationCopyContext.hpp"



/************************************

ModuleRepresentation models a collection of python values that
can see each other through a module object. Its core purpose is to
let us model the values inside of python modules at intermediate points
of the module's definition. For instance, if you have

    def f():
        return g()

at this point in the module, 'f' will throw a name error if you call it.
If you then 'def g', 'f' will be able to see 'g'. ModuleRepresentation
allows us to model both copies of 'f' - we can call (assuming evaluateInto
has been defined as it is in the module_representation_test)

    m = ModuleRepresentation('module')
    evaluateInto(m, 'def f()\n\treturn g()')

    m2 = ModuleRepresentation('module')
    m.copyInto(m, ['f'])
    evaluateInto(m2, 'def g()\n\treturn 1')

We can then ask 'm' and 'm2' for their dicts and get two different
copies of 'f'. This works by walking the objects visible to the
module and seeing which of them can 'see' the module or its dict. These
objects are considered 'internal' to the module. When we copy them
into another ModuleRepresentation object, we deepcopy them, replacing
their reference to the original module object with the new one. This
allows both copies of 'f' to coexist simultaneously in the two modules.

In order to ensure that all values get correctly updated when we copy them
around, we also allow you to add values from one module to another
in an 'inactive' way.  This allows you to move subsets of values from
one module to another without deepcopying them, but still retaining the
relationship they have to the module.  For instance imagine that in
m1 you define

    class O:
        def f(self):
            return S

then in the m2 you define

    class A(O):
        pass

and finally in m3 you define

    class S(O):
        pass

When we copy 'O' from m in to 'm2' we will want for 'A' to have
the same version of 'O' as m1 since at this point in the module's
life O has not actually changed.  However, when we copy A and O
into m3, we'll need to ensure that 'A' gets the new copy of 'O'
as its base.

To facilitate this behavior, we allow two ways of copying variables:
'active' and 'inactive'. 'active' copying from one module to another
means that we will get a new copy of the variable in the new module -
the entire portion of the object graph that can see the module from
that value will be duplicated in the new module.  'inactive' copying
means that the value's identity is retained but that we acknowledge
that it was produced 'in this module' and any values that are internal
to it are also 'internal' to this module, and are being tracked. This
ensures we don't get duplicate copies of a module value.

Let us be precise now. A ModuleRepresentation is just a module object.
Any object reachable from the module object itself is considered to
be in one of three states:

    external - this value cannot see back into this module at all
    active - this value can see back into this specific module
        object and is defined here.
    inactive - this value came from another copy of this module
        where it could see that copy's module dict, but we don't
        have our own copy of it.

In order for this to work, we require a strict order of operations

(1) the module object is created.
(2) all existing named objects are loaded and are either
        external - from the outside
        active - from another ModuleRepresentation and mutable
        inactive - from another ModuleRepresentation and immutable
(3) the module object is 'setupComplete()' - we do any necessary
    deepcopying all at once at this point.
(4) external code executes within the context of the module
(5) the module is 'updated' - any module entries that are new are
    walked so that we can find any

Basic usage looks something like

    m = ModuleRepresentation('modulename')

    m.addExternal('someName', someValue)
    someOtherModule.copyInto(m, ['name1', ...])
    someOtherModule.copyIntoAsInactive(m, ['name2', ...])

    m.setupComplete()

    ... do something with m.getDict() ...

    m.update()

************************************/

class ModuleRepresentation {
public:
    class Entry {
    public:
        Entry() : mActive(false)
        {}

        Entry(
            PyObjectHandle value,
            std::shared_ptr<ModuleRepresentation> module,
            bool isActive
        ) :
            mValue(value),
            mModule(module),
            mActive(isActive)
        {
            if (!mModule) {
                throw std::runtime_error(
                    "Module can't be null here"
                );
            }
        }

        Entry(
            PyObjectHandle value
        ) :
            mValue(value),
            mActive(false)
        {
        }

        bool isExternal() const {
            return !bool(mModule);
        }

        const PyObjectHandle& value() const {
            return mValue;
        }

        const std::shared_ptr<ModuleRepresentation>& module() const {
            return mModule;
        }

        bool isActive() const {
            return mActive && !isExternal();
        }

        bool isInactive() const {
            return !mActive && !isExternal();
        }

    private:
        PyObjectHandle mValue;
        std::shared_ptr<ModuleRepresentation> mModule;
        bool mActive;
    };

    ModuleRepresentation(std::string name) :
        mHasBeenSetup(false)
    {
        mModuleObject = PyObjectHandle::steal(
            PyModule_New(name.c_str())
        );
        // the module object is always 1 and its dict is always 2
        mOidMap.add(mModuleObject, 1);
        mOidMap.add(PyModule_GetDict(mModuleObject.pyobj()), 2);
    }

    bool isSetupComplete() const {
        return mHasBeenSetup;
    }

    // add a value that's guaranteed to not see this module at all
    void addExternal(std::string nameStr, PyObjectHandle value) {
        if (mHasBeenSetup) {
            throw std::runtime_error("Can't add to a module that's already setUp");
        }

        if (mInitialEntries.find(nameStr) != mInitialEntries.end()) {
            throw std::runtime_error("Can't define " + nameStr + " twice.");
        }

        mInitialEntries[nameStr] = Entry(value);
    }

    // add a value from another module. We must be in setup phase.
    void copyFrom(
        std::shared_ptr<ModuleRepresentation> other,
        std::set<std::string>& names,
        bool active
    ) {
        if (mHasBeenSetup) {
            throw std::runtime_error("Can't add to a module that's already setUp");
        }

        if (!other) {
            throw std::runtime_error("other can't be null");
        }

        for (auto name: names) {
            if (mInitialEntries.find(name) != mInitialEntries.end()) {
                throw std::runtime_error("Can't redefine " + name);
            }

            PyObjectHandle value = other->getValue(name);

            if (!value) {
                throw std::runtime_error("Other module doesn't have " + name);
            }

            mInitialEntries[name] = Entry(
                value,
                other,
                active
            );
        }
    }

    PyObjectHandle getValue(std::string name) {
        if (!mHasBeenSetup) {
            throw std::runtime_error("Can't query a non-setup module");
        }

        return PyDict_GetItemString(PyModule_GetDict(mModuleObject.pyobj()), name.c_str());
    }

    // finish the module - duplicate any objects and update any memos
    void setupComplete() {
        if (mHasBeenSetup) {
            throw std::runtime_error("Already set up");
        }

        mHasBeenSetup = true;

        for (auto& nameAndEntry: mInitialEntries) {
            if (nameAndEntry.second.isExternal()) {
                mExternalObjects.insert(nameAndEntry.second.value());

                PyDict_SetItemString(
                    PyModule_GetDict(mModuleObject.pyobj()),
                    nameAndEntry.first.c_str(),
                    nameAndEntry.second.value().pyobj()
                );
                mPriorUpdateValues[nameAndEntry.first] = (
                    nameAndEntry.second.value()
                );
            } else
            if (nameAndEntry.second.isActive()) {
                PyObjectHandle copied = copyActiveObjectFrom(
                    nameAndEntry.second.value(),
                    nameAndEntry.second.module()
                );

                PyDict_SetItemString(
                    PyModule_GetDict(mModuleObject.pyobj()),
                    nameAndEntry.first.c_str(),
                    copied.pyobj()
                );
                mPriorUpdateValues[nameAndEntry.first] = copied;
            }
        }

        for (auto& nameAndEntry: mInitialEntries) {
            if (nameAndEntry.second.isInactive()) {
                addInactiveFrom(
                    nameAndEntry.second.value(),
                    nameAndEntry.second.module()
                );

                PyDict_SetItemString(
                    PyModule_GetDict(mModuleObject.pyobj()),
                    nameAndEntry.first.c_str(),
                    nameAndEntry.second.value().pyobj()
                );
                mPriorUpdateValues[nameAndEntry.first] = (
                    nameAndEntry.second.value()
                );
            }
        }

        sortModuleObject();

        // allow old modules to get deleted
        mInitialEntries.clear();
    }

    // the module object may have changed underneath us. Any new objects
    // we havn't seen before must be walked and categorized
    void update() {
        PyObject *key, *value;
        Py_ssize_t pos = 0;

        PyObject* moduleDict = PyModule_GetDict(mModuleObject.pyobj());

        while (PyDict_Next(moduleDict, &pos, &key, &value)) {
            if (!PyUnicode_Check(key)) {
                throw std::runtime_error("Module member names are supposed to be strings.");
            }

            update(std::string(PyUnicode_AsUTF8(key)), value);
        }

        sortModuleObject();
    }

    static bool isModuleObjectOrModuleDict(PyObjectHandle obj) {
        return ModuleRepresentationCopyContext::isModuleObjectOrModuleDict(obj);
    }

    static bool isPrimitive(PyObjectHandle obj) {
        return ModuleRepresentationCopyContext::isPrimitive(obj);
    }

    const size_t oidFor(PyObjectHandle h) {
        return mOidMap.get(h);
    }

    PyObjectHandle moduleObject() const {
        return mModuleObject;
    }

private:
    // duplicate this object
    PyObjectHandle copyActiveObjectFrom(
        PyObjectHandle obj,
        std::shared_ptr<ModuleRepresentation> otherModule
    ) {
        ModuleRepresentationCopyContext ctx(
            otherModule.get()->mOidMap,
            mOidMap
        );

        return ctx.copy(obj);
    }

    void addInactiveFrom(
        PyObjectHandle value,
        std::shared_ptr<ModuleRepresentation> module
    ) {
        std::unordered_set<PyObjectHandle> queue;

        queue.insert(value);

        while (queue.size()) {
            PyObjectHandle toCheck = *queue.begin();
            queue.erase(toCheck);

            size_t oid = module->mOidMap.get(toCheck);

            if (!oid) {
                // this is external in the other module
                mExternalObjects.insert(toCheck);
            } else
            if (mOidMap.has(oid)) {
                // we already have this object so we don't
                // need to do anything
            } else {
                // this is new - recurse into it
                mOidMap.add(toCheck, oid);
                ModuleRepresentationCopyContext::computeReachableFrom(toCheck, queue);
            }
        }
    }

    void update(std::string key, PyObjectHandle value) {
        if (value == mPriorUpdateValues[key]) {
            return;
        }

        mPriorUpdateValues[key] = value;

        // check if its known to be external
        if (mExternalObjects.find(value) != mExternalObjects.end()) {
            return;
        }

        if (isPrimitive(value)) {
           mExternalObjects.insert(value);
           return;
        }

        if (isModuleObjectOrModuleDict(value) &&
            value != mModuleObject &&
            value != PyModule_GetDict(mModuleObject.pyobj())
        ) {
            mExternalObjects.insert(value);
            return;
        }

        if (mOidMap.has(value)) {
            return;
        }

        // walk the object graph starting with 'value'.
        // we stop anytime we hit an object that's already considered
        // "internal" (and therefore is in the Oid map)

        // all objects we've seen.
        std::unordered_set<PyObjectHandle> visited;

        // for each object, all the objects that can see it that we visited
        std::unordered_map<PyObjectHandle, std::unordered_set<PyObjectHandle> > incoming;

        // every internal object that was visible from this set
        // these were the terminal points on the walk
        std::unordered_set<PyObjectHandle> internalLeaves;

        // everything we still need to walk
        std::unordered_set<PyObjectHandle> toCheck;
        toCheck.insert(value);

        // loop
        while (toCheck.size()) {
            PyObjectHandle item = *toCheck.begin();
            toCheck.erase(item);
            visited.insert(item);

            if (mOidMap.has(item)) {
                internalLeaves.insert(item);
            } else {
                std::set<PyObjectHandle> reachable;
                ModuleRepresentationCopyContext::computeReachableFrom(item, reachable);

                for (auto r: reachable) {
                    incoming[r].insert(item);

                    if (mOidMap.has(r)) {
                        internalLeaves.insert(r);
                    }

                    // we need to look inside this object if its not visited.
                    // however we exclude modules and module dicts, as well as
                    // primitive (like integers, etc.). we also don't want
                    // to visit anything that's already in our Oid map since
                    // we already know about it
                    if (!isPrimitive(r)
                        && !isModuleObjectOrModuleDict(r)
                        && visited.find(r) == visited.end()
                        && mExternalObjects.find(r) == mExternalObjects.end()
                    ) {
                        toCheck.insert(r);
                    }
                }
            }
        }

        // visited should have everything in it now, so we can walk over it and compute the
        // internal subset, which consists of everything reachable from the module object
        // or the module object dict.
        // objects that can reach our module object or its dict
        std::unordered_set<PyObjectHandle> newInternalObjects;

        for (auto o: internalLeaves) {
            toCheck.insert(o);
        }

        while (toCheck.size()) {
            PyObjectHandle item = *toCheck.begin();
            toCheck.erase(item);

            if (!mOidMap.has(item)) {
                newInternalObjects.insert(item);
            }

            for (auto i: incoming[item]) {
                if (newInternalObjects.find(i) == newInternalObjects.end()) {
                    toCheck.insert(i);
                }
            }
        }

        for (auto o: visited) {
            if (mOidMap.has(o)) {
                // do nothing
            } else
            if (newInternalObjects.find(o) == newInternalObjects.end()) {
                mExternalObjects.insert(o);
            } else {
                mOidMap.add(o, allocateIdentity());
            }
        }
    }

    void sortModuleObject() {
        std::map<std::string, PyObjectHandle> items;

        PyObject* moduleDict = PyModule_GetDict(mModuleObject.pyobj());

        PyObject *key, *value;
        Py_ssize_t pos = 0;

        while (PyDict_Next(moduleDict, &pos, &key, &value)) {
            if (!PyUnicode_Check(key)) {
                throw std::runtime_error("Module member names are supposed to be strings.");
            }

            items[std::string(PyUnicode_AsUTF8(key))] = value;
        }

        PyDict_Clear(moduleDict);

        for (auto kv: items) {
            PyDict_SetItemString(moduleDict, kv.first.c_str(), kv.second.pyobj());
        }
    }

    static size_t allocateIdentity() {
        // implicitly locked by the GIL so we don't need to worry that multiple threads
        // are incrementing the counter at the same time.

        // our first value will be 3, since 1 and 2 are reserved
        static size_t counter = 2;

        counter++;

        return counter;
    }

private:
    PyObjectHandle mModuleObject;

    std::map<std::string, Entry> mInitialEntries;

    // has 'completeSetup' been called, triggering any deepcopying
    bool mHasBeenSetup;

    // for each internal object we know about, an object identity
    OidMap mOidMap;

    // all external objects
    std::set<PyObjectHandle> mExternalObjects;

    // for each module member, the last value we used to compute the update
    std::map<std::string, PyObjectHandle> mPriorUpdateValues;
};
