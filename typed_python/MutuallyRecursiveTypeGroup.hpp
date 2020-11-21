/******************************************************************************
   Copyright 2017-2020 typed_python Authors

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

#include "TypeOrPyobj.hpp"
#include "ShaHash.hpp"
#include <map>
#include <unordered_map>

class MutuallyRecursiveTypeGroup {
public:
    MutuallyRecursiveTypeGroup() : mAnyPyObjectsIncorrectlyOrdered(false)
    {}

    MutuallyRecursiveTypeGroup(ShaHash hash);

    ShaHash hash() {
        if (mHash == ShaHash()) {
            computeHash();
        }

        return mHash;
    }

    void computeHashAndInstall();

    // indicate that when we deserialize 'sourceHash', we are going to get the group
    // mapped to 'destHash'.
    static void installSourceToDestHashLookup(ShaHash sourceHash, ShaHash destHash);

    // if 'sourceHash' was marked to go to 'destHash', return 'destHash'. otherwise
    // return sourceHash.
    static ShaHash sourceToDestHashLookup(ShaHash sourceHash);

    const std::map<int32_t, TypeOrPyobj>& getIndexToObject() const {
        return mIndexToObject;
    }

    void setIndexToObject(int32_t index, TypeOrPyobj obj) {
        auto it = mIndexToObject.find(index);
        if (it != mIndexToObject.end()) {
            it->second = obj;
        } else {
            mIndexToObject.insert({index, obj});
        }
    }

    std::string repr(bool deep=false);

    static ShaHash pyObjectShaHash(PyObject* h, MutuallyRecursiveTypeGroup* groupHead);

    static ShaHash tpInstanceShaHash(Instance h, MutuallyRecursiveTypeGroup* groupHead);

    static ShaHash tpInstanceShaHash(Type* t, uint8_t* data, MutuallyRecursiveTypeGroup* groupHead);

    static void constructRecursiveTypeGroup(TypeOrPyobj root);

    static ShaHash pyCodeObjectShaHash(PyCodeObject* co, MutuallyRecursiveTypeGroup* groupHead);

    static std::string pyObjectSortName(PyObject* o);

    static MutuallyRecursiveTypeGroup* getGroupFromHash(ShaHash h);

    static std::pair<MutuallyRecursiveTypeGroup*, int> pyObjectGroupHeadAndIndex(
        PyObject* o,
        bool constructIfNotInGroup=true
    );

    // is this object in this group? If this is a native type, we'll unpack it and check that.
    // returns -1 if its not.
    int32_t indexOfObjectInThisGroup(PyObject* o);

    int32_t indexOfObjectInThisGroup(TypeOrPyobj o);

    static bool pyObjectGloballyIdentifiable(PyObject* h);

    static ShaHash computePyObjectShaHashConstant(PyObject* h);

    static bool isSimpleConstant(PyObject* h);

    static void visibleFrom(TypeOrPyobj root, std::vector<TypeOrPyobj>& outReachable);

    static void buildCompilerRecursiveGroup(const std::set<TypeOrPyobj>& types);

    static bool objectIsUnassigned(TypeOrPyobj obj);

    // find a type by hash if we have it. return null if we don't.
    static Type* lookupType(const ShaHash& h);

    // find an object by hash if we have it. return null if we don't.
    static PyObject* lookupObject(const ShaHash& h);

    static void installTypeHash(Type* t);
private:
    static ShaHash pyObjectShaHashByVisiting(PyObject* obj, MutuallyRecursiveTypeGroup* groupHead);

    static ShaHash computePyObjectShaHash(PyObject* h, MutuallyRecursiveTypeGroup* groupHead);

    // for each python object that's in a type group, the group and index within the group
    static std::unordered_map<PyObject*, std::pair<MutuallyRecursiveTypeGroup*, int> > mPythonObjectTypeGroups;

    static std::map<ShaHash, MutuallyRecursiveTypeGroup*> mHashToGroup;

    // for each python object where we have a hash
    static std::unordered_map<PyObject*, ShaHash> mPythonObjectShaHashes;

    static std::map<ShaHash, Type*> mHashToType;

    static std::map<ShaHash, ShaHash> mSourceToDestHashLookup;

    static std::map<ShaHash, PyObject*> mHashToObject;

    void computeHash();

    ShaHash mHash;

    std::map<int32_t, TypeOrPyobj> mIndexToObject;

    bool mAnyPyObjectsIncorrectlyOrdered;
};
