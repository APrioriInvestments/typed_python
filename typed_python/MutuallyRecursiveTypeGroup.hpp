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

    ShaHash hash() {
        if (mHash == ShaHash()) {
            computeHash();
        }

        return mHash;
    }

    const std::map<int32_t, TypeOrPyobj>& getIndexToObject() const {
        return mIndexToObject;
    }

    static ShaHash pyObjectShaHash(PyObject* h, MutuallyRecursiveTypeGroup* groupHead);

    static ShaHash tpInstanceShaHash(Instance h, MutuallyRecursiveTypeGroup* groupHead);

    static void constructRecursiveTypeGroup(TypeOrPyobj root);

    static ShaHash pyCodeObjectShaHash(PyCodeObject* co, MutuallyRecursiveTypeGroup* groupHead);

    static void extractNamesFromCode(PyCodeObject* co, std::set<std::string>& outNames);

    static std::string pyObjectSortName(PyObject* o);

    static std::pair<MutuallyRecursiveTypeGroup*, int> pyObjectGroupHeadAndIndex(PyObject* o);

    static bool pyObjectGloballyIdentifiable(PyObject* h);

    static ShaHash computePyObjectShaHashConstant(PyObject* h);

    static bool isSimpleConstant(PyObject* h);

    static void visibleFrom(TypeOrPyobj root, std::vector<TypeOrPyobj>& outReachable);

    static void buildCompilerRecursiveGroup(const std::set<TypeOrPyobj>& types);

    static bool objectIsUnassigned(TypeOrPyobj obj);

    // find a type by hash if we have it. return null if we don't.
    static Type* lookupType(const ShaHash& h);

    static void installTypeHash(Type* t);
private:
    static ShaHash pyObjectShaHashByVisiting(PyObject* obj, MutuallyRecursiveTypeGroup* groupHead);

    static ShaHash computePyObjectShaHash(PyObject* h, MutuallyRecursiveTypeGroup* groupHead);

    // for each python object that's in a type group, the group and index within the group
    static std::unordered_map<PyObject*, std::pair<MutuallyRecursiveTypeGroup*, int> > mPythonObjectTypeGroups;

    // for each python object where we have a hash
    static std::unordered_map<PyObject*, ShaHash> mPythonObjectShaHashes;

    static std::map<ShaHash, Type*> mHashToType;

    // a guard for mHashToType, which can be accessed by multiple threads in the serializer
    static std::recursive_mutex mHashToTypeMutex;

    void computeHash();

    ShaHash mHash;

    std::map<int32_t, TypeOrPyobj> mIndexToObject;

    bool mAnyPyObjectsIncorrectlyOrdered;
};